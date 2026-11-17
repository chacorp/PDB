"""
Inputs for the weight_vis_combined overview (render_weight_vis.py frames 0-199 +
render_dynamic_6faces.py frames 200-1146), all from one checkpoint:
ckpts_CBD/2026-04-27-15-36-49-NGBCv5/model_best.pth.

Source: ict-cap ID_000, exp 1 (20240325_MySlate_924), all 1147 frames.
Targets (render_dynamic_6faces.SEQUENCES order):
  ict-neutral, ict-cap ID_000, biwi ID_012, coma ID_006, coma ID_008, mf_ROM ID_012
Writes vis_CBD/2026-04-27-15-36-49-NGBCv5-best/{sequence}/verts/{frame:06d}.npy, and for
the ict-neutral target also key_weight.npy, key_d/ and faces.npy (render_weight_vis.py).

    cd render && python save_dynamic_6faces_data.py
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).parent.parent.absolute())
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.argv = [sys.argv[0]]

import igl  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dataloader_CBD import EvalDataset  # noqa: E402
from eval_CBD import Trainer  # noqa: E402
from utils.remesh_utils import ICT_face_model  # noqa: E402
from vis_CBD_retarget_fig import Pipeline  # noqa: E402

CKPT = os.path.join(ROOT, 'ckpts_CBD', '2026-04-27-15-36-49-NGBCv5')
OUT_ROOT = os.path.join(ROOT, 'vis_CBD', '2026-04-27-15-36-49-NGBCv5-best')
DEVICE = 'cuda:0'
# (sequence dir, data selection, mesh id) ; 'ict-neutral' = ICT neutral without identity
TARGETS = [
    ('ict-cap-ID_000_test-to-ict-neutral_01-masked',         'ict-neutral', None),
    ('ict-cap-ID_000_test-to-ict-cap-ID_000_test_01-masked', 'ict-cap',     0),
    ('ict-cap-ID_000_test-to-biwi-ID_012_test_01-masked',    'biwi',        12),
    ('ict-cap-ID_000_test-to-coma-ID_006_test_01-masked',    'coma',        6),
    ('ict-cap-ID_000_test-to-coma-ID_008_test_01-masked',    'coma',        8),
    ('ict-cap-ID_000_test-to-mf_ROM-ID_012_test_01-masked',  'mf_ROM',      12),
]


def load_model():
    """same options as save_weight_vis_data.py used for this checkpoint"""
    opts_yaml = yaml.safe_load(open(os.path.join(CKPT, 'train_opts.yml')))
    opts_yaml.update({
        'device': DEVICE, 'ckpt': CKPT, 'version': 5, 'batch_size': 1,
        'continue_ckpt': False, 'seed': 42, 'last_activation': 'relu',
        'no_pou': False, 'align_latent': True, 'is_train': False,
    })
    opts = argparse.Namespace(**opts_yaml)
    trainer = Trainer(opts)          # continue_ckpt False -> model_best.pth
    trainer.model.eval()
    return trainer.model, opts


def th(v, f):
    n = igl.per_vertex_normals(np.asarray(v, np.float64), np.asarray(f))
    return (torch.tensor(np.asarray(v)).float()[None].to(DEVICE),
            torch.tensor(n).float()[None].to(DEVICE))


def main():
    model, opts = load_model()
    pipe = Pipeline(opts)
    src_ds = EvalDataset(data_name='ict-cap', toggle=False, ict_cap_id_num=0, ict_cap_exp_num=1)
    n = len(src_ds)
    item0 = src_ds[0]
    src_tmpl = item0[1].numpy()
    src_faces = item0[4].numpy() if hasattr(item0[4], 'numpy') else np.asarray(item0[4])
    print(f"source frames: {n}", flush=True)
    sv, sn = th(src_tmpl, src_faces)

    for seq, sel, mid in TARGETS:
        out = os.path.join(OUT_ROOT, seq)
        os.makedirs(os.path.join(out, 'verts'), exist_ok=True)
        if sel == 'ict-neutral':
            ict = ICT_face_model()
            tv_np, tf_np = np.asarray(ict.neutral_verts, np.float64).squeeze(), np.asarray(ict.faces)
        else:
            tv_np, tf_np, _ = pipe.get_mesh(sel, None, mid)
            tv_np, tf_np = np.asarray(tv_np, np.float64), np.asarray(tf_np)
        tv, tn = th(tv_np, tf_np)
        with torch.no_grad():
            kw = model.predict_coordinate(tv, tn)
        if sel == 'ict-neutral':
            np.save(os.path.join(out, 'key_weight.npy'), kw[0].cpu().numpy())
            np.save(os.path.join(out, 'faces.npy'), tf_np)
            os.makedirs(os.path.join(out, 'key_d'), exist_ok=True)
        for i in tqdm(range(n), desc=seq[:40]):
            v = src_ds[i][0].numpy()
            ev, en = th(v, src_faces)
            with torch.no_grad():
                pred, key_d = model.retarget_animation(sv, sn, ev, en, kw, tv)
            np.save(os.path.join(out, 'verts', f'{i:06d}.npy'), pred[0].cpu().numpy())
            if sel == 'ict-neutral':
                np.save(os.path.join(out, 'key_d', f'{i:06d}.npy'), key_d[0].cpu().numpy())
        print(f"saved {seq}: {n} frames", flush=True)


if __name__ == '__main__':
    main()
