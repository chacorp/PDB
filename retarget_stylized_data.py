"""
Retargets the ICT (m00, 922 capture) expression sequence onto stylized target
meshes (test-mesh/*-align.obj etc.), to generate extra face-like mesh shape
variation as training data for the rotation estimator (train_R.py).

Retargeting quality is NOT the goal here: only rotation estimation will
consume this data, and that only needs plausible face-like geometry per
frame, not semantically correct expressions. Saves one .npz per target mesh
(stacked vertices across all frames + faces) under /data/sihun/stylized/,
plus a few sanity-check render PNGs per mesh so the output can be eyeballed.

Usage:
    python retarget_stylized_data.py --ckpt ./ckpts_CBD/2026-04-27-15-36-49-NGBCv5 --version 5 --align_latent
"""
import os
from functools import partial

import numpy as np
import torch
import trimesh
import igl
import yaml
import argparse
from tqdm import tqdm

from vis_CBD_retarget_fig import Pipeline, Options
from dataloader_CBD import EvalDataset, CBD_collate_wrapper_eval
from utils.matplotlib_rnd import plot_image_array

STYLE_MESHES = [
    'mary-align', 'malcolm-align', 'piers-align', 'morphy-bald-align',
    'girl-align', 'FLAME_sample_align', 'bonnie-bald-align',
    'head-reference', 'face-reference',
]

# same nudges vis_style_retarget.py uses -- head/face-reference aren't
# pre-aligned to the model's coordinate frame like the "-align" meshes are.
MESH_TRANSFORM = {
    'head-reference': (0.092, np.array([0, -0.18, 0])),
    'face-reference':  (0.1,   np.array([0, -0.2,  0])),
    'bonnie-bald-align': (1.0, np.array([0, -0.02, 0.25])),
    'morphy-bald-align': (1.0, np.array([0, 0,    -0.25])),
}

OUT_DIR = '/data/sihun/stylized'
N_VIS = 3  # sanity-check renders per mesh


def load_target_mesh(name):
    mesh = trimesh.load(f'test-mesh/{name}.obj', process=False, maintain_order=True)
    v, f = mesh.vertices.copy(), mesh.faces.copy()
    if name in MESH_TRANSFORM:
        s, t = MESH_TRANSFORM[name]
        v = v * s + t
    return v, f


def main():
    opts = Options()
    config = f'{opts.ckpt}/train_opts.yml'
    opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)
    opts_ = vars(opts)
    opts_yaml.update(opts_)
    opts = argparse.Namespace(**opts_yaml)
    opts.use_t_mask = True
    opts.continue_ckpt = False

    from eval_CBD import Trainer
    trainer = Trainer(opts)

    pipeline = Pipeline(opts)
    pipeline.model = trainer.model
    pipeline.model.eval()
    device = pipeline.device

    # ---- source: ICT m00 driven by the 922 capture (ict_cap_exp_num=0) ----
    src_v, src_f, _ = pipeline.get_mesh('ict-cap', None, 0)
    src_n = igl.per_vertex_normals(src_v, src_f)
    src_v_th = torch.tensor(src_v).float()[None].to(device)
    src_n_th = torch.tensor(src_n).float()[None].to(device)

    src_dataset = EvalDataset(data_name='ict-cap', toggle=False, ict_cap_id_num=0, ict_cap_exp_num=0)
    total_frames = len(src_dataset)
    print(f'source (ict m00, 922 capture): {total_frames} frames', flush=True)

    loader = torch.utils.data.DataLoader(
        src_dataset, batch_size=1, collate_fn=partial(CBD_collate_wrapper_eval, device=device)
    )
    frames_data = [(batch.vertices, batch.vertices_normal) for batch in loader]
    print(f'materialized {len(frames_data)} source frames', flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)

    for name in STYLE_MESHES:
        tgt_v, tgt_f = load_target_mesh(name)
        tgt_n = igl.per_vertex_normals(tgt_v, tgt_f)
        tgt_v_th = torch.tensor(tgt_v).float()[None].to(device)
        tgt_n_th = torch.tensor(tgt_n).float()[None].to(device)

        with torch.no_grad():
            key_weight = pipeline.model.predict_coordinate(tgt_v_th, tgt_n_th)

        out_verts = np.zeros((len(frames_data), tgt_v.shape[0], 3), dtype=np.float32)
        for i, (v_th, n_th) in enumerate(tqdm(frames_data, desc=name)):
            with torch.no_grad():
                pred_outputs, _ = pipeline.model.retarget_animation(
                    src_v_th, src_n_th, v_th, n_th, key_weight, tgt_v_th
                )
            out_verts[i] = pred_outputs[0].detach().cpu().numpy()

        out_path = f'{OUT_DIR}/{name}.npz'
        np.savez(out_path, vertices=out_verts, faces=tgt_f.astype(np.int64))
        print(f'saved: {out_path} vertices={out_verts.shape}', flush=True)

        # quick visual sanity check (few frames spread across the sequence)
        vis_idx = np.linspace(0, len(frames_data) - 1, N_VIS).astype(int)
        v_list = [out_verts[i] for i in vis_idx]
        f_list = [tgt_f] * len(v_list)
        plot_image_array(
            v_list, f_list, rot_list=[[0, 0, 0]] * len(v_list),
            size=2, bg_black=False, mode='shade',
            logdir=OUT_DIR, name=f'{name}_sanity', save=True,
        )

    print('done.', flush=True)


if __name__ == '__main__':
    main()
