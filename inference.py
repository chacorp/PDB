#!/usr/bin/env python3
"""
Retargeting inference + visualization for NeuralFacialAnimation.

Source/Target can each be either a dataset split or a custom mesh file/folder.

Dataset mode examples:
    # cross-retarget: mf_ROM -> ict ID 2
    python infer_retarget.py --ckpt ckpts_CBD/2026-04-27-15-36-49-NGBCv5 \
        --src mf_ROM --tgt ict --tgt_id 2

    # self-retarget: ict ID 0 -> ict ID 0  (with visualization)
    python infer_retarget.py --ckpt ckpts_CBD/2026-04-27-15-36-49-NGBCv5 \
        --src ict --src_id 0 --tgt ict --tgt_id 0 --vis

Custom mode examples:
    # custom source folder (OBJ sequence) -> dataset target
    python infer_retarget.py --ckpt ckpts_CBD/2026-04-27-15-36-49-NGBCv5 \
        --src_custom /path/to/obj_seq/ --tgt ict --tgt_id 2 --vis

    # dataset source -> custom target mesh
    python infer_retarget.py --ckpt ckpts_CBD/2026-04-27-15-36-49-NGBCv5 \
        --src mf_ROM --tgt_custom /path/to/target.ply --vis

    # both custom (optionally provide explicit neutral for source)
    python infer_retarget.py --ckpt ckpts_CBD/2026-04-27-15-36-49-NGBCv5 \
        --src_custom /path/to/obj_seq/ --src_neutral /path/to/neutral.obj \
        --tgt_custom /path/to/target.obj --vis
"""

import os
import sys
import yaml
import argparse
import pickle
from glob import glob
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import igl
import trimesh
from functools import partial
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

__abs_path__ = str(Path(__file__).parent.absolute())
if __abs_path__ not in sys.path:
    sys.path.insert(0, __abs_path__)

from eval_CBD import Trainer
from dataloader_CBD import EvalDataset, CBD_collate_wrapper_eval

DATA_NAME_LIST = ['voca', 'biwi', 'mf_SEN', 'coma', 'mf_ROM', 'ict', 'ict-cap']


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Retargeting inference + visualization',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint directory')

    # Source: dataset OR custom folder (mutually exclusive)
    src_grp = parser.add_mutually_exclusive_group(required=True)
    src_grp.add_argument('--src',        type=str, choices=DATA_NAME_LIST,
                         help='source dataset name')
    src_grp.add_argument('--src_custom', type=str, metavar='FOLDER',
                         help='source OBJ sequence folder (in-the-wild)')

    # Target: dataset OR custom file (mutually exclusive)
    tgt_grp = parser.add_mutually_exclusive_group(required=True)
    tgt_grp.add_argument('--tgt',        type=str, choices=DATA_NAME_LIST,
                         help='target dataset name')
    tgt_grp.add_argument('--tgt_custom', type=str, metavar='FILE',
                         help='target neutral mesh file (.obj / .ply, in-the-wild)')

    # Dataset-mode IDs
    parser.add_argument('--src_id',  type=int, default=0, help='source mesh ID (dataset mode)')
    parser.add_argument('--tgt_id',  type=int, default=0, help='target mesh ID (dataset mode)')
    parser.add_argument('--exp_num', type=int, default=0, help='expression number (ict-cap only)')

    # Custom-source option
    parser.add_argument('--src_neutral', type=str, default=None, metavar='FILE',
                        help='neutral mesh for custom source (default: first frame in folder)')

    # Runtime
    parser.add_argument('--device',     type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size (dataset mode only)')
    parser.add_argument('--max_frames', type=int, default=-1, help='limit frames (-1: all)')

    # Output
    parser.add_argument('--output_dir', type=str, default='eval_CBD', help='root .npy output dir')
    parser.add_argument('--save_gt',    action='store_true', help='also save GT source frames')

    # Visualization
    parser.add_argument('--vis',      action='store_true', help='save rendered images')
    parser.add_argument('--vis_dir',  type=str, default=None,
                        help='image output dir (default: <output_dir>/vis)')
    parser.add_argument('--size',     type=int, default=4, help='panel size in inches')
    parser.add_argument('--bg_black', action='store_true')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model / opts helpers
# ---------------------------------------------------------------------------

def build_opts(ckpt_path, device):
    config_path = os.path.join(ckpt_path, 'train_opts.yml')
    opts_dict = yaml.safe_load(open(config_path))
    opts = argparse.Namespace(**opts_dict)
    opts.ckpt          = ckpt_path
    opts.device        = device
    opts.continue_ckpt = False
    return opts


# ---------------------------------------------------------------------------
# Dataset-mode mesh helpers
# ---------------------------------------------------------------------------

def get_dataset_mesh(selection, dataset, select_id):
    """Return (vertices, faces) numpy arrays for the neutral mesh of a dataset identity."""
    if selection in ('ict', 'ict-cap'):
        id_vecs  = torch.load(f'{__abs_path__}/ict_face_pt/ict_id_vecs_test.pt').numpy()
        id_disps = dataset.ict_face_model.get_id_disp(id_vecs[select_id])
        mesh_v   = np.array(dataset.ict_face_model.neutral_verts.squeeze()) + np.array(id_disps.squeeze())
        return mesh_v, np.array(dataset.ict_face_model.faces)

    if selection == 'voca':
        mesh = dataset.voca_mesh
    elif selection == 'biwi':
        biwi_tri = trimesh.load(f'{__abs_path__}/test-mesh/BIWI.ply')
        with open(f'{__abs_path__}/test-mesh/biwi_templates.pkl', 'rb') as f:
            mesh = pickle.load(f)
        mesh['face'] = biwi_tri.faces
    elif selection == 'mf_SEN':
        mesh = dataset.mf_SEN_mesh
    elif selection == 'coma':
        mesh = dataset.coma_mesh
    elif selection == 'mf_ROM':
        mesh = dataset.mf_ROM_mesh
    else:
        raise ValueError(f'Unknown dataset: {selection}')

    mesh_list = [k for k in mesh.keys() if k != 'face']
    mesh_v    = np.array(mesh[mesh_list[select_id]])

    if selection == 'biwi':
        m_align = np.load(f'{__abs_path__}/utils/biwi/align.npy')
        mesh_v  = np.concatenate([mesh_v, np.ones((mesh_v.shape[0], 1))], axis=1) @ m_align.T

    return mesh_v, np.array(mesh['face'])


def make_dataset_output_path(output_dir, ckpt_name, src, tgt, src_id, tgt_id, exp_num):
    if src == 'ict-cap':
        fname = f'{src}-ID_{src_id:03d}_test-to-{tgt}-ID_{tgt_id:03d}_test_{exp_num:02d}'
    elif 'ict' in tgt:
        fname = f'{src}_test-to-{tgt}_test-ID_{tgt_id:03d}'
    else:
        fname = f'{src}_test-to-{tgt}_test'
    return os.path.join(output_dir, ckpt_name, fname)


# ---------------------------------------------------------------------------
# Custom-mode helpers
# ---------------------------------------------------------------------------

def load_custom_target(path):
    """Load a single OBJ/PLY file as target neutral mesh. Returns (v, f, n)."""
    mesh = trimesh.load(path, process=False)
    v = np.array(mesh.vertices, dtype=np.float32)
    f = np.array(mesh.faces)
    n = igl.per_vertex_normals(v, f).astype(np.float32)
    return v, f, n


class CustomSourceDataset:
    """Loads a folder of OBJ files as a source animation sequence (batch_size=1)."""

    EXTS = ('*.obj', '*.OBJ')

    def __init__(self, folder, neutral_path=None):
        frame_paths = []
        for ext in self.EXTS:
            frame_paths += glob(os.path.join(folder, ext))
        frame_paths = sorted(frame_paths)

        if not frame_paths:
            raise ValueError(f'No .obj files found in {folder}')

        self.frame_paths = frame_paths
        neutral_path = neutral_path or frame_paths[0]

        neu = trimesh.load(neutral_path, process=False)
        self.neutral_v = np.array(neu.vertices, dtype=np.float32)
        self.neutral_f = np.array(neu.faces)
        self.neutral_n = igl.per_vertex_normals(self.neutral_v, self.neutral_f).astype(np.float32)

    def __len__(self):
        return len(self.frame_paths)

    def iter_batches(self, device):
        """Yield SimpleNamespace batches compatible with the inference loop."""
        neu_v_th = torch.from_numpy(self.neutral_v)[None].to(device)
        neu_n_th = torch.from_numpy(self.neutral_n)[None].to(device)

        for path in self.frame_paths:
            m    = trimesh.load(path, process=False)
            df_v = np.array(m.vertices, dtype=np.float32)
            df_n = igl.per_vertex_normals(df_v, self.neutral_f).astype(np.float32)
            batch = SimpleNamespace(
                template        = neu_v_th,
                template_normal = neu_n_th,
                vertices        = torch.from_numpy(df_v)[None].to(device),
                vertices_normal = torch.from_numpy(df_n)[None].to(device),
                mesh_data       = 0,
            )
            yield batch


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_vis(src_neu, src_exp, src_f, tgt_neu, tgt_pred, tgt_f,
             vis_dir, frame_idx, size, bg_black):
    from matplotrender import plot_mesh_gouraud

    v_list = [src_neu, src_exp, tgt_neu, tgt_pred]
    f_list = [src_f,   src_f,   tgt_f,   tgt_f]

    plot_mesh_gouraud(
        v_list, f_list,
        rot_list=[[0, -10, 0]] * 4,
        size=size,
        mode='shade',
        mesh_scale=0.6,
        bg_black=bg_black,
        logdir=vis_dir,
        name=f'{frame_idx:05d}',
        save=True,
        show=False,
    )
    plt.close('all')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- model --------------------------------------------------------------
    opts    = build_opts(args.ckpt, args.device)
    trainer = Trainer(opts)
    trainer.model.eval()
    device = args.device

    ckpt_name = os.path.basename(args.ckpt.rstrip('/'))

    # --- source setup -------------------------------------------------------
    if args.src_custom:
        src_ds   = CustomSourceDataset(args.src_custom, args.src_neutral)
        src_iter = src_ds.iter_batches(device)
        n_frames = len(src_ds)
        src_v    = src_ds.neutral_v   # for visualization
        src_f    = src_ds.neutral_f
        src_name = Path(args.src_custom).stem
        print(f'Custom source: {args.src_custom} | frames: {n_frames}')
    else:
        src_dataset = EvalDataset(
            data_name=args.src, toggle=False,
            ict_cap_id_num=args.src_id, ict_cap_exp_num=args.exp_num,
        )
        src_loader = torch.utils.data.DataLoader(
            src_dataset, batch_size=args.batch_size,
            collate_fn=partial(CBD_collate_wrapper_eval, device=device),
        )
        src_iter = iter(src_loader)
        n_frames = len(src_dataset)
        src_v, src_f = get_dataset_mesh(args.src, src_dataset, args.src_id)
        src_name = args.src
        print(f'Dataset source: {args.src} | frames: {n_frames}')

    # --- target setup -------------------------------------------------------
    if args.tgt_custom:
        tgt_v, tgt_f, tgt_n = load_custom_target(args.tgt_custom)
        tgt_name = Path(args.tgt_custom).stem
        print(f'Custom target:  {args.tgt_custom} | verts: {tgt_v.shape[0]}')
    else:
        tgt_dataset = EvalDataset(data_name=args.tgt, toggle=False,
                                  ict_cap_id_num=args.tgt_id)
        tgt_v, tgt_f = get_dataset_mesh(args.tgt, tgt_dataset, args.tgt_id)
        tgt_n        = igl.per_vertex_normals(tgt_v, tgt_f)
        tgt_name = args.tgt
        print(f'Dataset target: {args.tgt} | verts: {tgt_v.shape[0]}')

    tgt_v_th = torch.tensor(tgt_v).float()[None].to(device)
    tgt_n_th = torch.tensor(tgt_n).float()[None].to(device)

    # --- output paths -------------------------------------------------------
    custom_mode = bool(args.src_custom or args.tgt_custom)
    if custom_mode:
        out_path = os.path.join(args.output_dir, ckpt_name,
                                f'custom-{src_name}-to-{tgt_name}')
    else:
        out_path = make_dataset_output_path(
            args.output_dir, ckpt_name,
            args.src, args.tgt, args.src_id, args.tgt_id, args.exp_num,
        )
    os.makedirs(out_path, exist_ok=True)
    print(f'Output path: {out_path}')

    vis_dir = args.vis_dir or os.path.join(out_path, 'vis')
    if args.vis:
        os.makedirs(vis_dir, exist_ok=True)
        print(f'Vis dir:     {vis_dir}')

    if args.save_gt:
        gt_out_path = os.path.join(args.output_dir, f'GT-{src_name}_test')
        os.makedirs(gt_out_path, exist_ok=True)

    # --- self/cross retarget ------------------------------------------------
    SELF_RETARGET = (not custom_mode and
                     args.src == args.tgt and
                     args.src_id == args.tgt_id)
    print('Mode: self-retargeting' if SELF_RETARGET else 'Mode: cross-retargeting')

    # pre-compute key_weight for cross-retarget / any custom target
    key_weight    = None
    prev_template = None
    if not SELF_RETARGET:
        with torch.no_grad():
            key_weight = trainer.model.predict_coordinate(tgt_v_th, tgt_n_th)

    # --- inference loop -----------------------------------------------------
    BS      = 1 if args.src_custom else args.batch_size
    total   = (args.max_frames if args.max_frames > 0 else n_frames)
    # ceil-div for dataset loader
    n_iters = (total + BS - 1) // BS if not args.src_custom else total

    pbar = tqdm(enumerate(src_iter), total=n_iters, ncols=100)
    for index, batch in pbar:
        frame_start = index * BS

        if args.max_frames >= 0 and frame_start >= args.max_frames:
            break

        # self-retarget: recompute key_weight when template changes identity
        if SELF_RETARGET:
            if (prev_template is None or
                    (batch.template[0] - prev_template[0]).abs().mean() > 0):
                with torch.no_grad():
                    key_weight = trainer.model.predict_coordinate(
                        batch.template, batch.template_normal,
                    )
                prev_template = batch.template.clone()

        with torch.no_grad():
            pred_outputs, _ = trainer.model.retarget_animation(
                batch.template, batch.template_normal,
                batch.vertices,  batch.vertices_normal,
                key_weight,
            )

        pred_np = pred_outputs.detach().cpu().numpy()
        actual_bs = pred_np.shape[0]

        for b in range(actual_bs):
            fid = frame_start + b
            np.save(os.path.join(out_path, f'{fid:05d}.npy'), pred_np[b])

            if args.vis:
                save_vis(
                    src_neu  = batch.template.detach().cpu().numpy()[b],
                    src_exp  = batch.vertices.detach().cpu().numpy()[b],
                    src_f    = src_f,
                    tgt_neu  = tgt_v,
                    tgt_pred = pred_np[b],
                    tgt_f    = tgt_f,
                    vis_dir  = vis_dir,
                    frame_idx= fid,
                    size     = args.size,
                    bg_black = args.bg_black,
                )

            if args.save_gt:
                gt_np = batch.vertices.detach().cpu().numpy()
                np.save(os.path.join(gt_out_path, f'{fid:05d}.npy'), gt_np[b])

    saved = glob(os.path.join(out_path, '*.npy'))
    print(f'Done. {len(saved)} frames saved to {out_path}')
    if args.vis:
        vis_saved = glob(os.path.join(vis_dir, '*.png'))
        print(f'      {len(vis_saved)} images  saved to {vis_dir}')


if __name__ == '__main__':
    main()
