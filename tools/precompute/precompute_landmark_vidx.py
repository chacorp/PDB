"""
precompute_landmark_vidx.py — Convert mean DTU3D landmarks (3D coords) into
mean-mesh vertex indices via argmin Euclidean. Saves [73] long array per
topology and renders a verification PNG (mesh + landmark vertex highlights).

Usage:
    python precompute_landmark_vidx.py --topos ict mf --rig_path maya_rig/hybrid \
        --landmarks_dir maya_rig/hybrid/landmarks \
        --out_dir maya_rig/hybrid
"""
import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))
from run_deepmvlm_on_meshes import (
    _xrot_mat, _yrot_mat, _perspective, _translate, _normalize_unit
)


def _load_mean_mesh(topo, data_basedir='/data/sihun'):
    if topo == 'ict':
        from utils.remesh_utils import ICT_face_model
        m = ICT_face_model()
        return m.neutral_verts.astype(np.float32), m.faces.astype(np.int32)
    if topo == 'mf':
        import pickle
        pkl = os.path.join(data_basedir, 'multiface_align', 'mf_templates.pkl')
        if not os.path.exists(pkl):
            pkl = os.path.join(data_basedir, 'pca', 'multiface_align', 'mf_templates.pkl')
        with open(pkl, 'rb') as f:
            t = pickle.load(f)
        faces = np.array(t['face'], dtype=np.int32)
        keys = [k for k in t if k != 'face']
        verts = np.stack([np.array(t[k], dtype=np.float32) for k in keys]).mean(axis=0)
        return verts.astype(np.float32), faces
    raise ValueError(f'unknown topology: {topo}')


def _argmin_euclid(verts, points):
    """For each point [P, 3], find argmin_v ||V[v] - point||. Returns [P] long."""
    d2 = ((verts[None, :, :] - points[:, None, :]) ** 2).sum(axis=-1)   # [P, V]
    return d2.argmin(axis=1).astype(np.int64)


def render_landmark_vidx(verts, faces, vidx, save_path,
                         xrot=0, yrots=(-30, 0, 30),
                         figsize_per_view=2.5, point_size=8.0,
                         label_idx=False):
    """Render mesh + landmark VERTICES (highlighted in red) with optional text label."""
    V = _normalize_unit(np.asarray(verts, dtype=np.float64))
    L_vert = V[vidx]                                           # [P, 3] in normalized space
    F = np.asarray(faces, dtype=np.int64)

    Vh = np.concatenate([V, np.ones((V.shape[0], 1))], axis=1)
    Lh = np.concatenate([L_vert, np.ones((L_vert.shape[0], 1))], axis=1)
    view = _translate(0, 0, -4.5)
    proj = _perspective(55, 1.0, 1.0, 100.0)
    MV = proj @ view

    n = len(yrots)
    fig = plt.figure(figsize=(figsize_per_view * n, figsize_per_view))
    for j, yrot in enumerate(yrots):
        model = _yrot_mat(yrot) @ _xrot_mat(xrot)
        Vm = (Vh @ model.T) @ MV.T
        Vndc = Vm[:, :3] / Vm[:, 3:4]
        VF = Vndc[F]
        T2 = VF[:, :, :2]
        Z = -VF[:, :, 2].mean(axis=1)
        order = np.argsort(Z)
        T_sorted = T2[order]

        v0 = (Vh[:, :3] @ model[:3, :3].T)[F[:, 0]]
        v1 = (Vh[:, :3] @ model[:3, :3].T)[F[:, 1]]
        v2 = (Vh[:, :3] @ model[:3, :3].T)[F[:, 2]]
        Nf = np.cross(v1 - v0, v2 - v0)
        Nf = Nf / (np.linalg.norm(Nf, axis=1, keepdims=True) + 1e-8)
        Nf = Nf[order]
        front = Nf[:, 2] > 0
        T_sorted = T_sorted[front]
        Nf_f = Nf[front]
        light_dir = np.array([0, 0, 1.0])
        shade = np.clip(Nf_f @ light_dir, 0, 1)[:, None].repeat(3, axis=1) * 0.7 + 0.2
        face_rgba = np.concatenate([shade, np.ones((shade.shape[0], 1))], axis=1)

        Lm = (Lh @ model.T) @ MV.T
        Lndc = Lm[:, :3] / Lm[:, 3:4]

        xy = np.concatenate([Vndc[:, :2], Lndc[:, :2]], axis=0)
        xmin, ymin = xy.min(axis=0); xmax, ymax = xy.max(axis=0)
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        half = 0.55 * max(xmax - xmin, ymax - ymin)

        ax = fig.add_axes([j / n, 0, 1 / n, 1], aspect=1, frameon=False)
        coll = PolyCollection(T_sorted, closed=True, linewidth=0.05,
                              facecolor=face_rgba, edgecolor='black')
        ax.add_collection(coll)
        ax.scatter(Lndc[:, 0], Lndc[:, 1], s=point_size, c='red',
                   edgecolors='black', linewidths=0.3, zorder=10)
        if label_idx:
            for k in range(len(vidx)):
                ax.text(Lndc[k, 0], Lndc[k, 1], str(k),
                        fontsize=3, color='blue', ha='center', va='center',
                        zorder=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topos', nargs='+', default=['ict', 'mf'])
    ap.add_argument('--rig_path', type=str, default='maya_rig/hybrid')
    ap.add_argument('--landmarks_dir', type=str, default='maya_rig/hybrid/landmarks')
    ap.add_argument('--out_dir', type=str, default='maya_rig/hybrid')
    ap.add_argument('--data_basedir', type=str, default='/data/sihun')
    ap.add_argument('--label_idx', action='store_true',
                    help='Annotate landmark indices on visualization')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for topo in args.topos:
        lm_npy = os.path.join(args.landmarks_dir, f'{topo}_mean_landmarks.npy')
        if not os.path.exists(lm_npy):
            print(f'[skip] {topo}: {lm_npy} not found')
            continue
        landmarks = np.load(lm_npy).astype(np.float32)        # [73, 3]
        verts, faces = _load_mean_mesh(topo, args.data_basedir)
        print(f'\n[{topo}] mesh V={verts.shape[0]} F={faces.shape[0]}, '
              f'landmarks={landmarks.shape}')

        vidx = _argmin_euclid(verts, landmarks)               # [73]
        # Distance from landmark to its closest vertex (sanity)
        d = np.linalg.norm(verts[vidx] - landmarks, axis=-1)
        print(f'  vidx range=[{vidx.min()}, {vidx.max()}], unique={len(set(vidx.tolist()))}/73')
        print(f'  landmark→closest-vertex distance: '
              f'min={d.min():.4f}, mean={d.mean():.4f}, max={d.max():.4f}')

        out_npy = os.path.join(args.out_dir, f'landmark_vidx_{topo}.npy')
        np.save(out_npy, vidx)
        print(f'  saved → {out_npy}')

        out_png = os.path.join(args.out_dir, f'landmark_vidx_{topo}_vis.png')
        render_landmark_vidx(verts, faces, vidx, out_png,
                             xrot=0, label_idx=args.label_idx, point_size=10.0)
        print(f'  saved → {out_png}')


if __name__ == '__main__':
    main()
