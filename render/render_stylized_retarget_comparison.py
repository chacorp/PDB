"""
Cross retargeting of the MF test identity and ICT m00 animations onto the
aligned stylized meshes (test-mesh/test-*-aligned.obj), for the NC / NFS / PDB
models. Stage 1 (inference): writes the predicted vertices to
vis_CBD/vis_comparison/stylize_retarget/preds_cache.npz. Rendering and the
gallery HTML are done by render/render_stylized_retarget_mpr.py (stage 2).
Inference is skipped when the cache already exists; delete it to rerun.

Run from repo root:
    python render/render_stylized_retarget_comparison.py
    cd _tmp && python -I ../render/render_stylized_retarget_mpr.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.argv = [sys.argv[0]]

import argparse  # noqa: E402
import yaml  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import igl  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

import dataloader_CBD  # noqa: E402
import utils.nfr_utils as nfr_utils  # noqa: E402
from vis_CBD_retarget_fig import Pipeline  # noqa: E402
from eval_CBD_cyc import Options  # noqa: E402
from eval_CBD import Trainer  # noqa: E402

TARGETS = ['aura', 'bowen', 'jupiter', 'proteus']
MF_TEST_IDX = 12
ICT_ID_M00 = 0
ICT_EXP_924 = 1
MF_FRAMES = [1039, 5044, 5112, 7623, 9013]
ICT_FRAMES = [107, 602, 843, 873, 1117]

# (name, ckpt, version)  version 1 = NC, 0 = NFS, 5 = PDB
MODELS = [
    ('NC', './ckpts_CBD/2025-10-20-00-15-40-CBD', 1),
    ('NFS', './ckpt_stage1/2024-08-18-23-32-29-all', 0),
    ('PDB', './ckpts_CBD8/2026-09-16-02-19-03-NGBCv5', 5),
]

OUT_ROOT = REPO_ROOT / 'vis_CBD' / 'vis_comparison'
IMG_ROOT = OUT_ROOT / 'stylize_retarget'
CACHE_PATH = IMG_ROOT / 'preds_cache.npz'


def build_opts(ckpt, version):
    opts = Options()
    opts.ckpt = ckpt
    opts.version = version
    if version == 0:
        opts_yaml = yaml.load(open('config/train_NFS.yml'), Loader=yaml.FullLoader)
        opts_ = vars(opts)
        opts_yaml.update(opts_)
        opts = argparse.Namespace(**opts_yaml)
        opts.img_feat_dim = 128
        opts.feature_type = 'cents&norms'
        opts.stage1 = True
        opts.scale_exp = 1.0
        opts.ict_face_only = False
        opts.use_NFR = False
        opts.design = 'new2'
        opts.dec_type = 'disp'
    else:
        opts_yaml = yaml.load(open(f'{ckpt}/train_opts.yml'), Loader=yaml.FullLoader)
        opts_ = vars(opts)
        opts_yaml.update(opts_)
        opts = argparse.Namespace(**opts_yaml)
    opts.use_t_mask = True
    opts.continue_ckpt = False
    opts.batch_size = 1
    return opts


def load_targets():
    out = {}
    for n in TARGETS:
        m = trimesh.load(f'test-mesh/test-{n}-aligned.obj', process=False, maintain_order=True)
        out[n] = (np.asarray(m.vertices, float), np.asarray(m.faces, np.int64))
    return out


def load_sources(pipe):
    mf_ds = dataloader_CBD.EvalDataset(data_name='mf_ROM', toggle=False)
    ict_ds = dataloader_CBD.EvalDataset(
        data_name='ict-cap', toggle=False, ict_cap_id_num=ICT_ID_M00, ict_cap_exp_num=ICT_EXP_924,
    )
    srcs = {}
    for key, ds, sel, mid, frames in [
        ('mf', mf_ds, 'mf_ROM', MF_TEST_IDX, MF_FRAMES),
        ('ict', ict_ds, 'ict-cap', ICT_ID_M00, ICT_FRAMES),
    ]:
        src_v, src_f, _ = pipe.get_mesh(sel, None, mid)
        frame_vs = [ds[i][0].numpy() for i in frames]
        srcs[key] = dict(neutral_v=np.asarray(src_v, float), faces=np.asarray(src_f, np.int64),
                         frames=frame_vs, idx=frames)
    return srcs


def run_model(pipe, version, src_v, src_f, frames, tgt_v, tgt_f):
    """Cross-retarget a list of source expression frames onto (tgt_v, tgt_f)."""
    device = pipe.device
    model = pipe.model
    model.eval()
    outs = []
    if version == 1:
        src_v_th = torch.tensor(src_v).float()[None].to(device)
        tgt_v_th = torch.tensor(tgt_v).float()[None].to(device)
        for fv in frames:
            fv_th = torch.tensor(fv).float()[None].to(device)
            with torch.no_grad():
                pred, _ = model.retarget(src_v_th, fv_th, tgt_v_th)
            outs.append(pred[0].detach().cpu().numpy())
    elif version == 0:
        model.eval()
        tgt_m = trimesh.Trimesh(vertices=tgt_v, faces=tgt_f)
        tgt_verts = torch.tensor(tgt_v).float().to(device)
        tgt_faces = torch.from_numpy(tgt_f).to(device)
        tgt_dfn_info = nfr_utils.get_dfn_info(tgt_m, map_location=device)
        tgt_operators = nfr_utils_get_ops(tgt_m)
        tgt_img = model.renderer.render_img(tgt_m).float().to(device)
        tgt_img_feat = model.get_img_feat(tgt_img)
        tgt_vert_feat = model.get_local_feature(tgt_verts[None], tgt_faces, tgt_img_feat).float()
        tgt_tri_feat = model.get_local_feature(tgt_verts[None], tgt_faces, tgt_img_feat, at='faces').float()
        with torch.no_grad():
            pred_seg_coeff = model.encode_seg(tgt_vert_feat, tgt_dfn_info)
            pred_id_coeff = model.encode_id(tgt_vert_feat, tgt_dfn_info)

        src_m = trimesh.Trimesh(vertices=src_v, faces=src_f)
        src_faces_th = torch.from_numpy(src_f).to(device)
        src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
        src_img = model.renderer.render_img(src_m).float().to(device)
        src_img_feat = model.get_img_feat(src_img)
        for fv in frames:
            fv_th = torch.tensor(fv).float()[None].to(device)
            with torch.no_grad():
                vert_feat_exp = model.get_local_feature(fv_th, src_faces_th, src_img_feat).float() * 1.3
                pred_exp_coeff = model.encode_exp(vert_feat_exp, src_dfn_info, batch_process=True, verbose=False)
                inputs = (
                    tgt_tri_feat if pipe.opts.dec_type == 'jacob' else tgt_vert_feat,
                    pred_exp_coeff, pred_id_coeff, pred_seg_coeff,
                    None, tgt_verts[None], tgt_faces, tgt_operators
                )
                decode_out = model.decode(inputs, batch_process=True)
                pred = decode_out[0] if isinstance(decode_out, tuple) else decode_out
            outs.append(pred[0].detach().cpu().numpy())
    else:
        src_n = igl.per_vertex_normals(src_v, src_f)
        tgt_n = igl.per_vertex_normals(tgt_v, tgt_f)
        src_v_th = torch.tensor(src_v).float()[None].to(device)
        src_n_th = torch.tensor(src_n).float()[None].to(device)
        tgt_v_th = torch.tensor(tgt_v).float()[None].to(device)
        tgt_n_th = torch.tensor(tgt_n).float()[None].to(device)
        with torch.no_grad():
            key_weight = model.predict_coordinate(tgt_v_th, tgt_n_th)
        for fv in frames:
            fn = igl.per_vertex_normals(fv, src_f)
            fv_th = torch.tensor(fv).float()[None].to(device)
            fn_th = torch.tensor(fn).float()[None].to(device)
            with torch.no_grad():
                pred, _ = model.retarget_animation(src_v_th, src_n_th, fv_th, fn_th, key_weight, tgt_v_th)
            outs.append(pred[0].detach().cpu().numpy())
    return outs


def nfr_utils_get_ops(mesh):
    from utils.mesh_utils import get_mesh_operators
    return get_mesh_operators(mesh)


def main():
    if CACHE_PATH.exists():
        print(f'cache exists, skip inference (delete to rerun): {CACHE_PATH}')
        return
    IMG_ROOT.mkdir(parents=True, exist_ok=True)
    targets = load_targets()

    # source frames and neutral templates depend only on the dataset, not the model
    base_pipe = Pipeline(build_opts(MODELS[0][1], MODELS[0][2]))
    srcs = load_sources(base_pipe)

    results = {}  # (model, target, src) -> list of predicted verts
    for model_name, ckpt, version in MODELS:
        print(f'===== {model_name} ({ckpt}) =====', flush=True)
        opts = build_opts(ckpt, version)
        trainer = Trainer(opts)
        pipe = Pipeline(opts)
        pipe.model = trainer.model
        for tname, (tv, tf) in targets.items():
            for sname, s in srcs.items():
                outs = run_model(pipe, version, s['neutral_v'], s['faces'], s['frames'], tv, tf)
                results[(model_name, tname, sname)] = outs
                print(f'  {tname} <- {sname}: {len(outs)} frames', flush=True)
        del trainer, pipe
        torch.cuda.empty_cache()

    save_cache(targets, srcs, results)
    print('wrote', CACHE_PATH)


def save_cache(targets, srcs, results):
    """Flat npz: target/source meshes, source frames and per-model predictions."""
    d = {'targets': np.array(list(targets)), 'sources': np.array(list(srcs)),
         'models': np.array([m for m, _, _ in MODELS])}
    for tname, (tv, tf) in targets.items():
        d[f'tgt/{tname}/v'] = tv
        d[f'tgt/{tname}/f'] = tf
    for sname, s in srcs.items():
        d[f'src/{sname}/v'] = s['neutral_v']
        d[f'src/{sname}/f'] = s['faces']
        d[f'src/{sname}/idx'] = np.array(s['idx'])
        d[f'src/{sname}/frames'] = np.stack(s['frames'])
    for (m, tname, sname), outs in results.items():
        d[f'pred/{m}/{tname}/{sname}'] = np.stack(outs)
    np.savez_compressed(CACHE_PATH, **d)


if __name__ == '__main__':
    main()
