"""
run_deepmvlm_on_meshes.py — Run Deep-MVLM facial landmark detection on our face meshes.

Handles auto-scaling (Deep-MVLM expects mesh in ±150 units; our meshes are ~±2).
Writes scaled OBJ to temp, runs MVLM, reads landmarks, descales back.

Usage:
    # Single mesh file
    python run_deepmvlm_on_meshes.py --mesh maya_rig/hybrid/anchor_vis_maya/template_ict_000.obj \
        --out maya_rig/hybrid/landmarks_ict_000.npy

    # ICT mean
    python run_deepmvlm_on_meshes.py --topo ict --out_dir maya_rig/hybrid/landmarks/

    # MF mean
    python run_deepmvlm_on_meshes.py --topo mf --data_basedir /data/sihun \
        --out_dir maya_rig/hybrid/landmarks/

    # All ICT + MF + BIWI + COMA mean meshes
    python run_deepmvlm_on_meshes.py --topos ict mf biwi coma --data_basedir /data/sihun \
        --out_dir maya_rig/hybrid/landmarks/
"""
import os
import sys
import argparse
import tempfile
import numpy as np
import trimesh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.dirname(__file__))
# NOTE: Deep-MVLM's sys.path is added LAZILY inside run_deepmvlm() to avoid
# its `utils/` package shadowing our project's `utils/` (e.g., utils.remesh_utils).


# ── Mesh loaders (mean per topology) ─────────────────────────────────────────

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
    if topo == 'biwi':
        import pickle
        with open('utils/templates/biwi_templates.pkl', 'rb') as f:
            t = pickle.load(f)
        faces = np.array(t['face'], dtype=np.int32)
        keys = [k for k in t if k != 'face']
        verts = np.stack([np.array(t[k], dtype=np.float32) for k in keys]).mean(axis=0)
        return verts.astype(np.float32), faces
    if topo == 'coma':
        import pickle
        with open('utils/templates/voca_templates.pkl', 'rb') as f:
            t = pickle.load(f)
        faces = np.array(t['face'], dtype=np.int32)
        keys = [k for k in t if k != 'face']
        verts = np.stack([np.array(t[k], dtype=np.float32) for k in keys]).mean(axis=0)
        return verts.astype(np.float32), faces
    raise ValueError(f'unknown topology: {topo}')


def _load_obj(path):
    s = trimesh.load(path, process=False)
    if hasattr(s, 'geometry') and len(s.geometry) > 0:
        m = list(s.geometry.values())[0]
    else:
        m = s
    return np.array(m.vertices, dtype=np.float32), np.array(m.faces, dtype=np.int32)


# ── Auto-scaling ─────────────────────────────────────────────────────────────

TARGET_RANGE = 150.0   # Deep-MVLM hardcoded ymin/ymax = ±150


def _compute_scale(verts, target_range=TARGET_RANGE):
    """Scale factor s such that scaled mesh max(|axis|) ≈ target_range."""
    half_extent = (verts.max(0) - verts.min(0)).max() / 2
    return target_range / max(half_extent, 1e-6)


def _write_obj(path, verts, faces):
    with open(path, 'w') as f:
        for v in verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for tri in faces:
            f.write(f'f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n')


# ── Visualization ────────────────────────────────────────────────────────────

def _xrot_mat(theta):
    t = np.pi * theta / 180.0
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]], dtype=np.float64)

def _yrot_mat(theta):
    t = np.pi * theta / 180.0
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]], dtype=np.float64)

def _frustum(l, r, b, t, n, f):
    M = np.zeros((4, 4))
    M[0, 0] = 2 * n / (r - l); M[1, 1] = 2 * n / (t - b)
    M[2, 2] = -(f + n) / (f - n); M[0, 2] = (r + l) / (r - l)
    M[2, 1] = (t + b) / (t - b); M[2, 3] = -2 * n * f / (f - n); M[3, 2] = -1
    return M

def _perspective(fovy, aspect, n, f):
    h = np.tan(0.5 * np.radians(fovy)) * n
    w = h * aspect
    return _frustum(-w, w, -h, h, n, f)

def _translate(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=np.float64)

def _normalize_unit(V):
    V = V - V.mean(axis=0)
    return V / np.max(np.linalg.norm(V, axis=1))

def render_landmarks_on_mesh(verts, faces, landmarks, save_path,
                             xrot=0, yrots=(-30, 0, 30),
                             figsize_per_view=2.5, point_size=1.0):
    """Render mesh + landmark overlay to a multi-view PNG."""
    V = _normalize_unit(np.asarray(verts, dtype=np.float64))
    L = (np.asarray(landmarks, dtype=np.float64) - np.asarray(verts, dtype=np.float64).mean(axis=0))
    L = L / np.max(np.linalg.norm(np.asarray(verts, dtype=np.float64) - np.asarray(verts, dtype=np.float64).mean(axis=0), axis=1))
    F = np.asarray(faces, dtype=np.int64)

    Vh = np.concatenate([V, np.ones((V.shape[0], 1))], axis=1)
    Lh = np.concatenate([L, np.ones((L.shape[0], 1))], axis=1)
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

        # face normals after model rotation, back-face cull
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

        # tight bbox: union of mesh + landmark NDC, with small margin → fills axes
        xy = np.concatenate([Vndc[:, :2], Lndc[:, :2]], axis=0)
        xmin, ymin = xy.min(axis=0); xmax, ymax = xy.max(axis=0)
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        half = 0.55 * max(xmax - xmin, ymax - ymin)  # square + 10% margin

        ax = fig.add_axes([j / n, 0, 1 / n, 1], aspect=1, frameon=False)
        coll = PolyCollection(T_sorted, closed=True, linewidth=0.05,
                              facecolor=face_rgba, edgecolor='black')
        ax.add_collection(coll)
        ax.scatter(Lndc[:, 0], Lndc[:, 1], s=point_size, c='red',
                   edgecolors='none', linewidths=0.0, zorder=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# ── Deep-MVLM runner ─────────────────────────────────────────────────────────

def run_deepmvlm(verts, faces, config_path, device='cpu'):
    """
    Auto-scale, run Deep-MVLM (in a subprocess to avoid sys.modules conflicts
    between our utils/ and Deep-MVLM's utils/), descale, return landmarks.

    Args:
        verts: [V, 3] np.float32 (original scale)
        faces: [F, 3] np.int32
        config_path: path to Deep-MVLM JSON config (e.g. configs/DTU3D-geometry.json)

    Returns:
        landmarks: [N, 3] np.float32 at ORIGINAL mesh scale
    """
    import subprocess

    s = _compute_scale(verts)
    verts_scaled = verts * s

    # Write scaled OBJ to temp (absolute paths everywhere — subprocess cwd changes)
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tf:
        tmp_obj = os.path.abspath(tf.name)
    _write_obj(tmp_obj, verts_scaled, faces)

    deepmvlm_dir = os.path.join(os.path.dirname(__file__), 'third_party', 'Deep-MVLM')
    config_abs = os.path.abspath(config_path)

    # Run Deep-MVLM as subprocess in its own directory (saved/ + sys.path isolated)
    env = os.environ.copy()
    if device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    cmd = [sys.executable, 'predict.py', '-c', config_abs, '-n', tmp_obj]
    res = subprocess.run(cmd, cwd=deepmvlm_dir, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    if res.returncode != 0:
        print(res.stdout)
        os.remove(tmp_obj)
        raise RuntimeError(f'Deep-MVLM subprocess failed (rc={res.returncode})')

    # Landmark output written next to OBJ (predict.py convention)
    lm_txt = os.path.splitext(tmp_obj)[0] + '_landmarks.txt'
    lm_vtk = os.path.splitext(tmp_obj)[0] + '_landmarks.vtk'
    if not os.path.exists(lm_txt):
        os.remove(tmp_obj)
        raise RuntimeError(f'landmark file not found: {lm_txt}')
    landmarks_scaled = np.loadtxt(lm_txt).astype(np.float32)
    landmarks = landmarks_scaled / s   # descale

    # Cleanup temps
    for p in (tmp_obj, lm_txt, lm_vtk):
        try:
            os.remove(p)
        except Exception:
            pass

    return landmarks


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--mesh', type=str,
                     help='Path to input OBJ mesh')
    src.add_argument('--topo', type=str, choices=['ict', 'mf', 'biwi', 'coma'],
                     help='Topology mean mesh to process')
    src.add_argument('--topos', nargs='+', choices=['ict', 'mf', 'biwi', 'coma'],
                     help='Multiple topology mean meshes')
    ap.add_argument('--config', type=str,
                    default='third_party/Deep-MVLM/configs/DTU3D-geometry.json',
                    help='Deep-MVLM config path')
    ap.add_argument('--data_basedir', type=str, default='/data/sihun')
    ap.add_argument('--out', type=str, default=None,
                    help='Output .npy path (single mesh)')
    ap.add_argument('--out_dir', type=str, default=None,
                    help='Output directory (for --topo / --topos)')
    ap.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--save_vis', action='store_true', default=True,
                    help='Save PNG visualization of landmarks on mesh (default on)')
    ap.add_argument('--no_save_vis', dest='save_vis', action='store_false')
    ap.add_argument('--vis_xrot', type=float, default=0.0,
                    help='X rotation (deg) applied to mesh before render')
    ap.add_argument('--vis_point_size', type=float, default=1.0,
                    help='Landmark scatter point size')
    args = ap.parse_args()

    # Resolve targets to process
    targets = []     # list of (verts, faces, name)
    if args.mesh:
        v, f = _load_obj(args.mesh)
        name = os.path.splitext(os.path.basename(args.mesh))[0]
        targets.append((v, f, name))
    elif args.topo:
        v, f = _load_mean_mesh(args.topo, args.data_basedir)
        targets.append((v, f, f'{args.topo}_mean'))
    elif args.topos:
        for topo in args.topos:
            v, f = _load_mean_mesh(topo, args.data_basedir)
            targets.append((v, f, f'{topo}_mean'))

    # Output
    out_dir = args.out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for verts, faces, name in targets:
        print(f'\n[{name}] V={verts.shape[0]} F={faces.shape[0]}, '
              f'range=[{verts.min():.3f}, {verts.max():.3f}]')
        s = _compute_scale(verts)
        print(f'  scale factor: ×{s:.2f}  (auto to ±{TARGET_RANGE})')

        landmarks = run_deepmvlm(verts, faces, args.config, device=args.device)
        print(f'  landmarks: {landmarks.shape}, range=[{landmarks.min():.3f}, {landmarks.max():.3f}]')

        # Save
        if args.out:
            out_path = args.out
        elif out_dir:
            out_path = os.path.join(out_dir, f'{name}_landmarks.npy')
        else:
            out_path = f'{name}_landmarks.npy'
        np.save(out_path, landmarks)
        print(f'  saved → {out_path}')

        if args.save_vis:
            vis_path = os.path.splitext(out_path)[0] + '_vis.png'
            render_landmarks_on_mesh(verts, faces, landmarks, vis_path,
                                     xrot=args.vis_xrot,
                                     point_size=args.vis_point_size)
            print(f'  saved → {vis_path}')


if __name__ == '__main__':
    main()
