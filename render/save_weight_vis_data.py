"""
Pre-compute and save key_weight and key_d for the weight visualization render.

Outputs into the sequence directory:
  key_weight.npy      — (N, M=512) per-vertex barycentric weights, constant
  key_d/000000.npy    — (M, 3) deformed control-point positions, per-frame

Run once before render_weight_vis.py:
    cd render
    python save_weight_vis_data.py
"""

import argparse
import os
import sys
import yaml
from functools import partial
from pathlib import Path

import igl
import numpy as np
import torch
from tqdm import tqdm

ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)

from dataloader_CBD import EvalDataset, CBD_collate_wrapper_eval
from eval_CBD import Trainer
from utils.remesh_utils import ICT_face_model

CKPT    = os.path.join(ROOT, 'ckpts_CBD8', '2026-09-16-02-19-03-NGBCv5')  # PDB
SEQ_DIR = os.path.join(ROOT, 'vis_CBD', '2026-09-16-02-19-03-NGBCv5',
                       'ict-cap-ID_000_test-to-ict-neutral_01-masked')
MAX_FRAMES = 201   # frames 0 … 200 inclusive (render_weight_vis.MAX_DATA = 200)
DEVICE     = 'cuda:0'


def load_model():
    config = os.path.join(CKPT, 'train_opts.yml')
    opts_yaml = yaml.safe_load(open(config))
    opts_yaml.update({
        'device':        DEVICE,
        'ckpt':          CKPT,
        'version':       5,
        'batch_size':    1,
        'continue_ckpt': False,
        'seed':          42,
        'is_train':      False,
    })  # architecture options (activation, pou, align_latent) come from the checkpoint
    opts = argparse.Namespace(**opts_yaml)
    trainer = Trainer(opts)
    trainer.model.eval()
    return trainer.model


def main():
    print("Loading model...")
    model = load_model()

    # ── Target mesh: ict-neutral ─────────────────────────────────────────────
    ict   = ICT_face_model()
    tgt_v = np.asarray(ict.neutral_verts, dtype=np.float64)
    tgt_f = np.asarray(ict.faces)
    tgt_n = igl.per_vertex_normals(tgt_v, tgt_f)
    tgt_v_th = torch.tensor(tgt_v).float().unsqueeze(0).to(DEVICE)
    tgt_n_th = torch.tensor(tgt_n).float().unsqueeze(0).to(DEVICE)
    os.makedirs(SEQ_DIR, exist_ok=True)
    np.save(os.path.join(SEQ_DIR, 'faces.npy'), tgt_f)  # topology for render_weight_vis.py

    # ── key_weight: (N, M), constant for this target ─────────────────────────
    key_weight = model.predict_coordinate(tgt_v_th, tgt_n_th)  # (1, N, M)
    kw_np = key_weight[0].cpu().numpy()
    np.save(os.path.join(SEQ_DIR, 'key_weight.npy'), kw_np)
    print(f"Saved key_weight.npy  shape={kw_np.shape}")

    # ── Source dataloader: ict-cap ID_000, exp sequence 1 ────────────────────
    src_dataset = EvalDataset(
        data_name='ict-cap', toggle=False,
        ict_cap_id_num=0, ict_cap_exp_num=1,
    )
    src_loader = torch.utils.data.DataLoader(
        src_dataset, batch_size=1,
        collate_fn=partial(CBD_collate_wrapper_eval, device=DEVICE),
    )

    # ── key_d: (M, 3) per frame ───────────────────────────────────────────────
    key_d_dir = os.path.join(SEQ_DIR, 'key_d')
    verts_dir = os.path.join(SEQ_DIR, 'verts')   # retargeted target meshes per frame
    os.makedirs(key_d_dir, exist_ok=True)
    os.makedirs(verts_dir, exist_ok=True)

    saved = 0
    for idx, batch in enumerate(tqdm(src_loader, desc='key_d frames')):
        if idx >= MAX_FRAMES:
            break
        pred, key_d = model.retarget_animation(
            batch.template, batch.template_normal,
            batch.vertices, batch.vertices_normal,
            key_weight, tgt_v_th,
        )
        np.save(os.path.join(key_d_dir, f'{idx:06d}.npy'), key_d[0].cpu().numpy())
        np.save(os.path.join(verts_dir, f'{idx:06d}.npy'), pred[0].cpu().numpy())
        saved += 1

    print(f"Saved {saved} key_d frames → {key_d_dir}")


if __name__ == '__main__':
    main()
