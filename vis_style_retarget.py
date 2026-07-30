"""
Stylized-mesh retargeting experiment (not part of report/render_retarget_request_2026-07-27.md).

Source: ICT m00 driven by _cap/20240325_MySlate_924_exp_coeffs.npy (ict-cap, id=0, exp_num=1).
Targets: 9 stylized meshes under test-mesh/ (mary-align, malcolm-align, piers-align,
morphy-bald-align, girl-align, FLAME_sample_align, bonnie-bald-align, head-reference,
face-reference).

head-reference and face-reference are not pre-aligned to the model's coordinate frame
(unlike the "-align" meshes). Scale/translation for both, plus small extra nudges for
bonnie-bald-align/morphy-bald-align, come from prior validated values found in
notebook/{NFR_inference,NBC_visualize-Copy1,experiments}.ipynb (not derived here) --
re-verified visually before running the full set (see _tmp/orientation_check/).

Only 10 randomly sampled source frames (fixed seed) are retargeted, to keep this experiment
cheap. Saves per-frame vertex npy + rendered PNG (render_trimesh_fig.py's camera/lighting)
under ./vis_style_retarget/<mesh_name>/.
"""
import os
import sys
import random
from functools import partial

sys.path += ['.', './utils']
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
sys.path.insert(0, 'render')

import numpy as np
import torch
import trimesh
import igl
import cv2
import pickle
import yaml
import argparse
from tqdm import tqdm

import render_trimesh_fig as R
from vis_CBD_retarget_fig import Pipeline, Options
from dataloader_CBD import EvalDataset, CBD_collate_wrapper_eval

STYLE_MESHES = [
    'mary-align', 'malcolm-align', 'piers-align', 'morphy-bald-align',
    'girl-align', 'FLAME_sample_align', 'bonnie-bald-align',
    'head-reference', 'face-reference',
]
N_FRAMES = 10
FRAME_SEED = 42
OUT_BASE = './vis_style_retarget'

H = W = 800
CAMERA_PARAMS = {
    'c': np.array([H // 2, W // 2]),
    'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
    'f': np.array([4754.97941935 / 2, 4754.97941935 / 2]),
}


# scale/translation from notebook/NFR_inference.ipynb, NBC_visualize-Copy1.ipynb,
# and notebook/experiments.ipynb cell 67 (bonnie/morphy nudges) -- validated there,
# re-checked visually here (_tmp/orientation_check/*_notebookprecedent.png).
MESH_TRANSFORM = {
    'head-reference': (0.092, np.array([0, -0.18, 0])),
    'face-reference':  (0.1,   np.array([0, -0.2,  0])),
    'bonnie-bald-align': (1.0, np.array([0, -0.02, 0.25])),
    'morphy-bald-align': (1.0, np.array([0, 0,    -0.25])),
}


def load_target_mesh(name):
    mesh = trimesh.load(f'test-mesh/{name}.obj', process=False, maintain_order=True)
    v, f = mesh.vertices.copy(), mesh.faces.copy()
    if name in MESH_TRANSFORM:
        s, t = MESH_TRANSFORM[name]
        v = v * s + t
    return v, f


def render_frame(v, f, out_path, scale=0.5):
    m = trimesh.Trimesh(vertices=v * scale, faces=f)
    center = np.mean(v * scale, axis=0)
    img = R.render_mesh_helper(m, center, CAMERA_PARAMS, z_offset=1.3, H=H, W=W)
    cv2.imwrite(out_path, img)


def main():
    opts = Options()

    config = f'{opts.ckpt}/train_opts.yml'
    opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)
    opts_ = vars(opts)
    opts_yaml.update(opts_)
    opts = argparse.Namespace(**opts_yaml)
    opts.use_t_mask = True
    opts.continue_ckpt = False

    print('loaded version:', opts.version)
    from eval_CBD import Trainer
    trainer = Trainer(opts)

    pipeline = Pipeline(opts)
    pipeline.model = trainer.model
    pipeline.model.eval()
    device = pipeline.device

    # ---- source: ICT m00 driven by the 924 capture ----
    src_v, src_f, _ = pipeline.get_mesh('ict-cap', None, 0)
    src_n = igl.per_vertex_normals(src_v, src_f)
    src_v_th = torch.tensor(src_v).float()[None].to(device)
    src_n_th = torch.tensor(src_n).float()[None].to(device)

    src_dataset = EvalDataset(data_name='ict-cap', toggle=False, ict_cap_id_num=0, ict_cap_exp_num=1)
    total_frames = len(src_dataset)
    print(f'source dataset (ict m00, 924 capture): {total_frames} frames total')

    random.seed(FRAME_SEED)
    selected = sorted(random.sample(range(total_frames), N_FRAMES))
    print('selected frame indices:', selected)

    subset = torch.utils.data.Subset(src_dataset, selected)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=1, collate_fn=partial(CBD_collate_wrapper_eval, device=device)
    )
    frames_data = []
    for frame_idx, batch in zip(selected, loader):
        frames_data.append((frame_idx, batch.vertices, batch.vertices_normal))
    print(f'materialized {len(frames_data)} source frames')

    # ---- retarget + render for each stylized mesh ----
    for name in STYLE_MESHES:
        tgt_v, tgt_f = load_target_mesh(name)
        tgt_n = igl.per_vertex_normals(tgt_v, tgt_f)
        tgt_v_th = torch.tensor(tgt_v).float()[None].to(device)
        tgt_n_th = torch.tensor(tgt_n).float()[None].to(device)

        out_dir = f'{OUT_BASE}/{name}'
        vert_dir = out_dir + '/verts'
        frame_dir = out_dir + '/frames'
        os.makedirs(vert_dir, exist_ok=True)
        os.makedirs(frame_dir, exist_ok=True)

        with torch.no_grad():
            key_weight = pipeline.model.predict_coordinate(tgt_v_th, tgt_n_th)

        for frame_idx, v_th, n_th in tqdm(frames_data, desc=name):
            with torch.no_grad():
                pred_outputs, _ = pipeline.model.retarget_animation(
                    src_v_th, src_n_th, v_th, n_th, key_weight, tgt_v_th
                )
            out_v = pred_outputs[0].detach().cpu().numpy()
            np.save(f'{vert_dir}/{frame_idx:06d}.npy', out_v)
            render_frame(out_v, tgt_f, f'{frame_dir}/{frame_idx:06d}.png')

        print(f'done: {name}, {len(frames_data)} frames -> {out_dir}')


if __name__ == '__main__':
    main()
