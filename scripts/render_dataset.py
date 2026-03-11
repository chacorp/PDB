"""
Render real mesh data as video files (mp4) with frame number overlay.
Output: dataset_vis/{dataset}/{split}/{identity}__{sentence}.mp4

Usage:
    python scripts/render_dataset.py --dataset mf_ROM
    python scripts/render_dataset.py --dataset all
"""
import os
import sys
import io
import argparse
import numpy as np
import pickle
import glob
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import cv2

__abs_path__ = str(Path(__file__).parents[1].absolute())
sys.path.insert(0, __abs_path__)

from utils.matplotlib_rnd import (
    translate, yrotate, xrotate, zrotate, ortho,
    _homogeneous, calc_face_norm,
)

# Precompute constant matrices
_MODEL = (yrotate(0) @ xrotate(0) @ zrotate(0))[:3, :3]
_VIEW = translate(0, 0, -5)
_PROJ = ortho(-1, 1, -1, 1, 1, 100)
_MVP = _PROJ @ _VIEW
_LIGHT = np.array([0, 0, 1])


def render_frame_to_array(vertices, faces, frame_num, frame_label='', size=3, dpi=100):
    """Render a single mesh frame and return as numpy array (H, W, 3) uint8."""
    V = vertices.astype(np.float64)
    F = faces

    fig = plt.figure(figsize=(size, size))
    ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)

    V_mu = np.median(V, axis=0)
    V_model = (V - V_mu) @ _MODEL.T + V_mu
    V_proj = _homogeneous(V_model) @ _MVP.T
    V_proj = V_proj[:, :3] / V_proj[:, 3:4]
    VF_tri = V_proj[F]

    T = VF_tri[:, :, :2]
    Z = -VF_tri[:, :, 2].mean(axis=1)
    zmin, zmax = Z.min(), Z.max()
    Z = (Z - zmin) / (zmax - zmin)

    C = calc_face_norm(V, F) @ _MODEL[:3, :3].T
    I = np.argsort(Z)
    T, C = T[I, :], C[I, :]
    NI = np.argwhere(C[:, 2] > 0).squeeze()
    T, C = T[NI, :], C[NI, :]
    C = (C @ _LIGHT)[:, np.newaxis].repeat(3, axis=-1)
    C = np.clip(C, 0, 1) * 0.7 + 0.2

    collection = PolyCollection(T, closed=False, linewidth=0, facecolor=C, edgecolor=C)
    ax.add_collection(collection)

    # Frame number + label overlay
    label = f'{frame_num}'
    if frame_label:
        label = f'{frame_label}  f:{frame_num}'
    ax.text(0.02, 0.96, label, transform=ax.transAxes,
            fontsize=10, color='black', fontweight='bold',
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax.set_xticks([])
    ax.set_yticks([])

    # Render to numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format='raw', dpi=dpi, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    w, h = int(fig.get_figwidth() * dpi), int(fig.get_figheight() * dpi)
    img = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(h, w, 4)
    img = img[:, :, :3]  # drop alpha
    return img


def get_dataset_config(dataset_name, data_basedir='/data/sihun'):
    configs = {
        'mf_SEN': {
            'base_path': f'{data_basedir}/multiface_align/SEN',
            'face_source': 'mf_std',
            'splits': ['train', 'val', 'test'],
        },
        'mf_ROM': {
            'base_path': f'{data_basedir}/multiface_align/ROM',
            'face_source': 'mf_std',
            'splits': ['train', 'val', 'test'],
        },
        'voca': {
            'base_path': f'{data_basedir}/VOCA-COMA/VOCASET',
            'face_source': 'voca_template',
            'splits': ['train', 'val', 'test'],
        },
        'coma': {
            'base_path': f'{data_basedir}/VOCA-COMA/COMA',
            'face_source': 'voca_template',
            'splits': ['train', 'val', 'test'],
        },
        'biwi': {
            'base_path': f'{data_basedir}/BIWI_align_deci',
            'face_source': 'biwi_template',
            'splits': ['train', 'val', 'test'],
        },
    }
    return configs[dataset_name]


def load_faces(face_source, data_basedir='/data/sihun'):
    if face_source == 'mf_std':
        mf_std = np.load(f'{data_basedir}/multiface_align/mf_standardize.npy', allow_pickle=True).item()
        return np.array(mf_std['new_f']).astype(np.int32)
    elif face_source == 'voca_template':
        with open(f'{data_basedir}/VOCA-COMA/voca_templates.pkl', 'rb') as f:
            t = pickle.load(f)
        return np.array(t['face']).astype(np.int32)
    elif face_source == 'biwi_template':
        with open(f'{data_basedir}/BIWI_align_deci/templates_align_deci.pkl', 'rb') as f:
            t = pickle.load(f)
        return np.array(t['face']).astype(np.int32)


def discover_sequences(base_path, split, dataset_name):
    """
    Discover all sequences for a dataset split.
    Returns list of (identity, sentence, frame_files_sorted) tuples.
    """
    verts_dir = os.path.join(base_path, split, 'vertices_npy')
    split_dir = os.path.join(base_path, split)

    results = []

    if dataset_name in ('mf_SEN', 'mf_ROM'):
        if not os.path.exists(verts_dir):
            print(f"  [WARN] {verts_dir} not found, skipping.")
            return []
        identities = sorted([d for d in os.listdir(verts_dir)
                             if os.path.isdir(os.path.join(verts_dir, d))])
        for ident in identities:
            ident_dir = os.path.join(verts_dir, ident)
            sentences = sorted([s for s in os.listdir(ident_dir)
                                if os.path.isdir(os.path.join(ident_dir, s))])
            for sent in sentences:
                sent_dir = os.path.join(ident_dir, sent)
                frames = sorted(glob.glob(os.path.join(sent_dir, '*.npy')))
                if frames:
                    results.append((ident, sent, frames))

    elif dataset_name == 'voca':
        split_dir = os.path.join(base_path, split)
        identities = sorted([d for d in os.listdir(split_dir)
                             if os.path.isdir(os.path.join(split_dir, d))])
        for ident in identities:
            vnpy_dir = os.path.join(split_dir, ident, 'vertices_npy')
            if not os.path.exists(vnpy_dir):
                continue
            sentences = sorted([s for s in os.listdir(vnpy_dir)
                                if os.path.isdir(os.path.join(vnpy_dir, s))])
            for sent in sentences:
                sent_dir = os.path.join(vnpy_dir, sent)
                frames = sorted(glob.glob(os.path.join(sent_dir, '*.npy')))
                if frames:
                    results.append((ident, sent, frames))

    elif dataset_name == 'coma':
        split_dir = os.path.join(base_path, split)
        identities = sorted([d for d in os.listdir(split_dir)
                             if os.path.isdir(os.path.join(split_dir, d))])
        for ident in identities:
            vnpy_dir = os.path.join(split_dir, ident, 'vertices_npy')
            if not os.path.exists(vnpy_dir):
                continue
            expressions = sorted([e for e in os.listdir(vnpy_dir)
                                  if os.path.isdir(os.path.join(vnpy_dir, e))])
            for expr in expressions:
                expr_dir = os.path.join(vnpy_dir, expr)
                frames = sorted(glob.glob(os.path.join(expr_dir, '*.npy')))
                if frames:
                    results.append((ident, expr, frames))

    elif dataset_name == 'biwi':
        if not os.path.exists(verts_dir):
            print(f"  [WARN] {verts_dir} not found, skipping.")
            return []
        sessions = sorted([d for d in os.listdir(verts_dir)
                          if os.path.isdir(os.path.join(verts_dir, d))])
        for sess in sessions:
            sess_dir = os.path.join(verts_dir, sess)
            frames = sorted(glob.glob(os.path.join(sess_dir, '*.npy')))
            if frames:
                results.append((sess, '_', frames))

    return results


def extract_frame_num(fname):
    """Extract frame number from filename."""
    num_str = fname.replace('.npy', '')
    if '-' in num_str:
        num_str = num_str.split('-')[-1]
    elif '.' in num_str:
        num_str = num_str.split('.')[-1]
    return int(num_str)


def render_sequence_to_video(frame_files, faces, video_path, fps=30, size=3, dpi=100, frame_label=''):
    """Render a sequence of mesh frames directly to an mp4 video."""
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Render first frame to get dimensions
    v0 = np.load(frame_files[0])
    img0 = render_frame_to_array(v0, faces, extract_frame_num(os.path.basename(frame_files[0])),
                                  frame_label=frame_label, size=size, dpi=dpi)
    h, w = img0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    # Write first frame
    writer.write(cv2.cvtColor(img0, cv2.COLOR_RGB2BGR))

    # Write remaining frames
    for fpath in frame_files[1:]:
        vertices = np.load(fpath)
        frame_num = extract_frame_num(os.path.basename(fpath))
        img = render_frame_to_array(vertices, faces, frame_num,
                                     frame_label=frame_label, size=size, dpi=dpi)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    writer.release()


def render_dataset(dataset_name, out_base, data_basedir='/data/sihun', fps=30):
    """Render all sequences for a dataset as video files."""
    config = get_dataset_config(dataset_name, data_basedir)
    faces = load_faces(config['face_source'], data_basedir)

    print(f"\n{'='*60}")
    print(f"Rendering: {dataset_name}")
    print(f"{'='*60}")

    total_videos = 0

    for split in config['splits']:
        print(f"\n  [{dataset_name}/{split}]")
        sequences = discover_sequences(config['base_path'], split, dataset_name)
        print(f"    Found {len(sequences)} sequences")

        for ident, sent, frame_files in tqdm(sequences, desc=f'    {split}'):
            # Sanitize names for filename
            ident_short = ident.replace('--', '-')
            sent_clean = sent.strip('_') if sent != '_' else ''
            if sent_clean:
                vid_name = f'{ident_short}__{sent_clean}.mp4'
            else:
                vid_name = f'{ident_short}.mp4'

            video_path = os.path.join(out_base, dataset_name, split, vid_name)

            # Skip if already exists
            if os.path.exists(video_path):
                continue

            frame_label = sent_clean[:30] if sent_clean else ident_short[:30]
            render_sequence_to_video(frame_files, faces, video_path,
                                     fps=fps, frame_label=frame_label)
            total_videos += 1

    print(f"\n  Total new videos: {total_videos}")


def main():
    parser = argparse.ArgumentParser(description='Render dataset meshes as mp4 videos')
    parser.add_argument('--dataset', type=str, default='mf_ROM',
                        choices=['mf_SEN', 'mf_ROM', 'voca', 'coma', 'biwi', 'all'])
    parser.add_argument('--data_basedir', type=str, default='/data/sihun')
    parser.add_argument('--out_dir', type=str, default='dataset_vis')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    out_base = os.path.join(__abs_path__, args.out_dir)
    os.makedirs(out_base, exist_ok=True)

    if args.dataset == 'all':
        datasets = ['mf_ROM', 'mf_SEN', 'voca', 'coma', 'biwi']
    else:
        datasets = [args.dataset]

    for ds in datasets:
        render_dataset(ds, out_base, args.data_basedir, fps=args.fps)

    print("\nDone!")


if __name__ == '__main__':
    main()
