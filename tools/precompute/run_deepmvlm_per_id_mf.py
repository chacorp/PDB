"""
run_deepmvlm_per_id_mf.py — Run Deep-MVLM on each MF identity mesh (not mean).

Multiface meshes are scan-registered, so vertex i may not be at exactly the
same anatomical position across IDs. To get accurate per-id bind_pose GT,
we need per-id landmarks (instead of inheriting mean's landmark vertex
indices). This script runs Deep-MVLM on each MF id's neutral template and
saves per-id 73-point landmarks as 3D coords.

Output:
    maya_rig/hybrid/landmarks/per_id_mf/{id_name}_landmarks.npy   shape [73, 3]
    (in original mesh scale; Deep-MVLM auto-rescaling internal to wrapper)

Usage:
    python run_deepmvlm_per_id_mf.py \
        --data_basedir /data/sihun \
        --out_dir maya_rig/hybrid/landmarks/per_id_mf \
        --device cpu
"""
import os
import sys
import argparse
import pickle
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))
from run_deepmvlm_on_meshes import run_deepmvlm, render_landmarks_on_mesh


def _load_mf_templates(data_basedir):
    pkl = os.path.join(data_basedir, 'multiface_align', 'mf_templates.pkl')
    if not os.path.exists(pkl):
        pkl = os.path.join(data_basedir, 'pca', 'multiface_align', 'mf_templates.pkl')
    with open(pkl, 'rb') as f:
        t = pickle.load(f)
    faces = np.array(t['face'], dtype=np.int32)
    id_names = [k for k in t if k != 'face']
    return [(np.array(t[n], dtype=np.float32), faces, n) for n in id_names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_basedir', type=str, default='/data/sihun')
    ap.add_argument('--config', type=str,
                    default='third_party/Deep-MVLM/configs/DTU3D-geometry.json')
    ap.add_argument('--out_dir', type=str,
                    default='maya_rig/hybrid/landmarks/per_id_mf')
    ap.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--ids', nargs='+', default=None,
                    help='Optional subset of MF id names to process')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    templates = _load_mf_templates(args.data_basedir)
    if args.ids:
        templates = [t for t in templates if t[2] in args.ids]
    print(f'MF per-id Deep-MVLM run: {len(templates)} identities')

    for verts, faces, name in templates:
        out_path = os.path.join(args.out_dir, f'{name}_landmarks.npy')
        if os.path.exists(out_path) and not args.overwrite:
            print(f'  [skip] {name}: cache exists ({out_path})')
            continue

        print(f'\n[{name}] V={verts.shape[0]} F={faces.shape[0]}, '
              f'range=[{verts.min():.3f}, {verts.max():.3f}]')
        landmarks = run_deepmvlm(verts, faces, args.config, device=args.device)
        print(f'  landmarks: {landmarks.shape}, '
              f'range=[{landmarks.min():.3f}, {landmarks.max():.3f}]')
        np.save(out_path, landmarks)
        print(f'  saved → {out_path}')

        vis_path = os.path.splitext(out_path)[0] + '_vis.png'
        render_landmarks_on_mesh(verts, faces, landmarks, vis_path,
                                 xrot=0.0, point_size=1.0)
        print(f'  saved → {vis_path}')


if __name__ == '__main__':
    main()
