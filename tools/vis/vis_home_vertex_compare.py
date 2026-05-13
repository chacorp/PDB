"""
vis_home_vertex_compare.py — Compare landmark-based vs maya_bind-based
home_vertex on ICT/MF mean meshes.

Renders mean mesh + two overlay point sets:
  - BLUE  = home_vertex from landmark (Phase A current setup)
  - GREEN = home_vertex from argmin ||V_mean - maya_bind[j]||

Shows which joints' home_vertex differs between the two methods, and
where each lands anatomically. Used to decide whether landmark-based
home_vertex is worth the extra Deep-MVLM dependency.

Usage:
    python vis_home_vertex_compare.py
        --rig_path maya_rig/hybrid
        --topos ict mf
        --out_dir maya_rig/hybrid/home_vertex_compare
"""
import os
import sys
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tools', 'precompute'))  # run_deepmvlm_on_meshes
from utils.rig_loader import load_rig
from run_deepmvlm_on_meshes import (
    _xrot_mat, _yrot_mat, _perspective, _translate, _normalize_unit
)


def _load_mean_verts(topo, data_basedir='/data/sihun'):
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


def _render_two_sets(verts, faces, pts_a, pts_b, save_path,
                     label_a='landmark', label_b='maya_bind',
                     xrot=0, yrots=(-30, 0, 30),
                     figsize_per_view=2.5, point_size=8.0,
                     pts_a_text=None,
                     pts_c=None, label_c='legacy_landmark'):
    V = _normalize_unit(np.asarray(verts, dtype=np.float64))
    _Vc = np.asarray(verts, dtype=np.float64).mean(axis=0)
    _Vs = np.max(np.linalg.norm(np.asarray(verts, dtype=np.float64) - _Vc, axis=1))
    def _norm(p):
        return (np.asarray(p, dtype=np.float64) - _Vc) / _Vs
    Pa = _norm(pts_a)
    Pb = _norm(pts_b)
    Pc = _norm(pts_c) if pts_c is not None else None
    F = np.asarray(faces, dtype=np.int64)

    Vh = np.concatenate([V, np.ones((V.shape[0], 1))], axis=1)
    Pah = np.concatenate([Pa, np.ones((Pa.shape[0], 1))], axis=1)
    Pbh = np.concatenate([Pb, np.ones((Pb.shape[0], 1))], axis=1)
    Pch = np.concatenate([Pc, np.ones((Pc.shape[0], 1))], axis=1) if Pc is not None else None
    view = _translate(0, 0, -4.5)
    proj = _perspective(55, 1.0, 1.0, 100.0)
    MV = proj @ view

    n = len(yrots)
    fig = plt.figure(figsize=(figsize_per_view * n + 0.5, figsize_per_view))
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

        Pa_m = (Pah @ model.T) @ MV.T; Pa_ndc = Pa_m[:, :3] / Pa_m[:, 3:4]
        Pb_m = (Pbh @ model.T) @ MV.T; Pb_ndc = Pb_m[:, :3] / Pb_m[:, 3:4]
        Pc_ndc = None
        if Pch is not None:
            Pc_m = (Pch @ model.T) @ MV.T; Pc_ndc = Pc_m[:, :3] / Pc_m[:, 3:4]

        _xy_list = [Vndc[:, :2], Pa_ndc[:, :2], Pb_ndc[:, :2]]
        if Pc_ndc is not None: _xy_list.append(Pc_ndc[:, :2])
        xy = np.concatenate(_xy_list, axis=0)
        xmin, ymin = xy.min(axis=0); xmax, ymax = xy.max(axis=0)
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        half = 0.55 * max(xmax - xmin, ymax - ymin)

        ax = fig.add_axes([j / n, 0, 1 / n, 1], aspect=1, frameon=False)
        coll = PolyCollection(T_sorted, closed=True, linewidth=0.05,
                              facecolor=face_rgba, edgecolor='black')
        ax.add_collection(coll)
        ax.scatter(Pa_ndc[:, 0], Pa_ndc[:, 1], s=point_size, c='blue',
                   edgecolors='black', linewidths=0.3, zorder=10, label=label_a)
        ax.scatter(Pb_ndc[:, 0], Pb_ndc[:, 1], s=point_size, c='green',
                   edgecolors='black', linewidths=0.3, zorder=11, marker='^', label=label_b)
        if Pc_ndc is not None:
            ax.scatter(Pc_ndc[:, 0], Pc_ndc[:, 1], s=point_size, c='red',
                       edgecolors='black', linewidths=0.3, zorder=12, marker='s', label=label_c)
        if pts_a_text is not None:
            for k, txt in enumerate(pts_a_text):
                if txt is None or txt == '':
                    continue
                ax.text(Pa_ndc[k, 0] + 0.012, Pa_ndc[k, 1] + 0.012,
                        str(txt), fontsize=3.5, color='darkblue',
                        ha='left', va='bottom', zorder=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        if j == 0:
            ax.legend(fontsize=6, loc='lower left')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rig_path', type=str, default='maya_rig/hybrid')
    ap.add_argument('--topos', nargs='+', default=['ict', 'mf'])
    ap.add_argument('--out_dir', type=str, default='maya_rig/hybrid/home_vertex_compare')
    ap.add_argument('--data_basedir', type=str, default='/data/sihun')
    ap.add_argument('--landmark_map', type=str,
                    default='maya_rig/hybrid/joint_landmark_map.json')
    ap.add_argument('--landmark_vidx_dir', type=str, default=None,
                    help='Defaults to --rig_path')
    ap.add_argument('--active_joints_json', type=str,
                    default='maya_rig/hybrid/active_joints_manual.json',
                    help='If set, only compare face joints (cleaner view)')
    ap.add_argument('--per_id', action='store_true',
                    help='Render per-id meshes (not just mean). Uses the SAME home_vertex '
                         'indices computed on mean, but plots V_id[home_vertex] on each id mesh '
                         '— reveals anatomical drift of inherited indices across IDs.')
    ap.add_argument('--joint_home_vidx_dir', type=str, default=None,
                    help='Dir with joint_home_vidx_{topo}.npy [J] precomputed. When provided, '
                         'overrides the landmark-based home_vertex computation for "method A" '
                         '(blue points).')
    args = ap.parse_args()

    lvidx_dir = args.landmark_vidx_dir or args.rig_path
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.landmark_map) as f:
        joint_lm_map = json.load(f)['mapping']

    face_idx = None
    if args.active_joints_json and os.path.exists(args.active_joints_json):
        with open(args.active_joints_json) as f:
            face_idx = json.load(f)['face_joint_idx']

    rig = load_rig(args.rig_path, device='cpu')
    J = len(rig.joint_names)

    for topo in args.topos:
        if topo not in rig.bind_pos_dict:
            print(f'[skip] {topo}: bind_pos_dict not loaded'); continue
        verts, faces = _load_mean_verts(topo, args.data_basedir)
        maya_bind = rig.bind_pos_dict[topo].cpu().numpy()
        V = verts.shape[0]
        print(f'\n[{topo}] V={V} F={faces.shape[0]}')

        lvidx_path = os.path.join(lvidx_dir, f'landmark_vidx_{topo}.npy')
        landmark_vidx = np.load(lvidx_path).astype(np.int64) if os.path.exists(lvidx_path) else None

        # joint_home_vidx override (manual spec)
        joint_home_path = None
        joint_home_vidx = None
        if args.joint_home_vidx_dir:
            _p = os.path.join(args.joint_home_vidx_dir, f'joint_home_vidx_{topo}.npy')
            if os.path.exists(_p):
                joint_home_path = _p
                joint_home_vidx = np.load(_p).astype(np.int64)
                print(f'  using joint_home_vidx override: {_p}')

        # Method A = manual override if available, else landmark-based
        # Method B = maya_bind argmin (always)
        # Method C = legacy landmark-based (shown only when A is the override → 3-way)
        hv_A = np.zeros(J, dtype=np.int64)
        hv_B = np.zeros(J, dtype=np.int64)
        hv_C = np.zeros(J, dtype=np.int64)
        for j in range(J):
            jname = rig.joint_names[j]
            lm = joint_lm_map.get(jname) if joint_lm_map else None
            # B: maya_bind argmin
            d2 = ((verts - maya_bind[j]) ** 2).sum(axis=-1)
            hv_B[j] = int(d2.argmin())
            # C: legacy landmark vidx (or fallback)
            if landmark_vidx is not None and lm is not None:
                hv_C[j] = int(landmark_vidx[lm])
            else:
                hv_C[j] = hv_B[j]
            # A: manual override if loaded, else same as C
            if joint_home_vidx is not None:
                hv_A[j] = int(joint_home_vidx[j])
            else:
                hv_A[j] = hv_C[j]

        # Stats: how many face joints differ between A and B
        F_set = set(face_idx) if face_idx else set(range(J))
        diff_face = [j for j in F_set if hv_A[j] != hv_B[j]]
        same_face = [j for j in F_set if hv_A[j] == hv_B[j]]
        print(f'  face joints: same={len(same_face)}, diff={len(diff_face)}/{len(F_set)}')
        if diff_face:
            print('  diff joints (landmark idx vs maya_bind argmin idx):')
            for j in sorted(diff_face):
                d_pos = np.linalg.norm(verts[hv_A[j]] - verts[hv_B[j]])
                print(f'    [{j:2d}] {rig.joint_names[j]:38s}  '
                      f'A={hv_A[j]:5d}  B={hv_B[j]:5d}  ||V_A - V_B||={d_pos:.4f}')

        # Render — restrict to face joints if available
        joint_subset = sorted(F_set)
        pts_A = verts[hv_A[joint_subset]]
        pts_B = verts[hv_B[joint_subset]]
        pts_C = verts[hv_C[joint_subset]] if joint_home_vidx is not None else None
        # Per-joint DTU3D landmark index label (1-indexed for blue points).
        labels_A = []
        for j in joint_subset:
            lm = joint_lm_map.get(rig.joint_names[j])
            labels_A.append(str(int(lm) + 1) if lm is not None else '')
        label_a = 'manual_override' if joint_home_vidx is not None else 'landmark'
        out_path = os.path.join(args.out_dir, f'{topo}_mean_home_vertex_compare.png')
        _render_two_sets(verts, faces, pts_A, pts_B, out_path,
                         label_a=label_a, pts_a_text=labels_A,
                         pts_c=pts_C, label_c='legacy_landmark')
        print(f'  saved → {out_path}')

        # Per-id rendering (same home_vertex indices, applied to each id's V_id)
        if args.per_id:
            from extract_nfs_feat import load_templates_ict, load_templates_mf
            if topo == 'ict':
                templates = load_templates_ict()
            elif topo == 'mf':
                templates = load_templates_mf(args.data_basedir)
            else:
                templates = []
            per_id_out = os.path.join(args.out_dir, f'{topo}_per_id')
            os.makedirs(per_id_out, exist_ok=True)
            print(f'  rendering {len(templates)} per-id meshes...')
            for V_id, _faces_id, name in templates:
                V_id = V_id.astype(np.float32)
                pts_A_id = V_id[hv_A[joint_subset]]
                pts_B_id = V_id[hv_B[joint_subset]]
                _out = os.path.join(per_id_out, f'{name}.png')
                _render_two_sets(V_id, faces, pts_A_id, pts_B_id, _out)
            print(f'  saved {len(templates)} → {per_id_out}/')


if __name__ == '__main__':
    main()
