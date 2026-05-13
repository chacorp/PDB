"""
precompute_per_id_bind_pos_v2.py — Per-id bind_pose GT via joint_home_vidx.

Replaces v1 (landmark_vidx + joint_landmark_map) with a single per-joint home
vertex source: joint_home_vidx_{topo}.npy ([J] long).

Formula (uniform across all joints):
    offset[j]      = bind_pos_mean_GT[j] - V_mean[ home_vidx[j] ]
    bind_pose_id[j] = V_id[ home_vidx[j] ] + offset[j]

bind_pos_mean_GT comes from rig.bind_pos_dict[topo]. For ICT, rig was
mirrored to be perfectly symmetric (mirror_ict_rig_bind_pos.py).

Output: {feat_dir}/{id_name}_bind_pos_landmark.npy   [J, 3]
"""
import os
import sys
import argparse
import pickle
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))
from utils.rig_loader import load_rig
from extract_nfs_feat import load_templates_ict, load_templates_mf


def _load_templates(dataset, data_basedir):
    if dataset == 'ict':
        return load_templates_ict()
    if dataset == 'mf':
        return load_templates_mf(data_basedir)
    raise ValueError(f'unknown dataset: {dataset}')


def _load_mean_verts(topo, data_basedir):
    if topo == 'ict':
        from utils.remesh_utils import ICT_face_model
        return ICT_face_model().neutral_verts.astype(np.float32)
    if topo == 'mf':
        pkl = os.path.join(data_basedir, 'multiface_align', 'mf_templates.pkl')
        if not os.path.exists(pkl):
            pkl = os.path.join(data_basedir, 'pca', 'multiface_align', 'mf_templates.pkl')
        with open(pkl, 'rb') as f:
            t = pickle.load(f)
        keys = [k for k in t if k != 'face']
        return np.stack([np.array(t[k], dtype=np.float32) for k in keys]).mean(axis=0).astype(np.float32)
    raise ValueError(f'unknown topology: {topo}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rig_path',  type=str, default='maya_rig/hybrid')
    ap.add_argument('--home_vidx_dir', type=str, default='maya_rig/hybrid',
                    help='Dir with joint_home_vidx_{topo}.npy [J] from compute_joint_home_vidx_*.py')
    ap.add_argument('--feat_dir',  type=str, default='nfs_features_seg',
                    help='Output dir for {id_name}_bind_pos_landmark.npy')
    ap.add_argument('--datasets',  nargs='+', default=['ict', 'mf'], choices=['ict', 'mf'])
    ap.add_argument('--data_basedir', type=str, default='/data/sihun')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--active_json', type=str,
                    default='maya_rig/hybrid/active_joints_manual.json',
                    help='Used for helper override: helper joints inherit per-id parent bind_pose '
                         '(not a surface vertex). Detected via helper_joint_idx field.')
    args = ap.parse_args()

    rig = load_rig(args.rig_path, device='cpu')
    J = len(rig.joint_names)
    print(f'rig: J={J}')

    # Helper joints: per-id bind_pose inherits per-id PARENT bind_pose
    # (helper = "anchored at parent joint center", not on mesh surface).
    helper_idx = set()
    if args.active_json and os.path.exists(args.active_json):
        import json as _json
        with open(args.active_json) as f:
            _aj = _json.load(f)
        helper_idx = set(_aj.get('helper_joint_idx', []))
    if helper_idx:
        print(f'  helper joints: {len(helper_idx)} (will inherit parent per-id bind_pose)')

    os.makedirs(args.feat_dir, exist_ok=True)
    n_total = 0; n_skipped = 0

    for ds in args.datasets:
        topo = ds
        if topo not in rig.bind_pos_dict:
            print(f'[skip] {topo}: bind_pos_dict not loaded'); continue
        bind_pos_mean = rig.bind_pos_dict[topo].cpu().numpy().astype(np.float32)   # [J, 3]

        hvidx_path = os.path.join(args.home_vidx_dir, f'joint_home_vidx_{topo}.npy')
        if not os.path.exists(hvidx_path):
            print(f'[skip] {topo}: {hvidx_path} not found'); continue
        home_vidx = np.load(hvidx_path).astype(np.int64)                           # [J]

        V_mean = _load_mean_verts(topo, args.data_basedir)
        offsets = bind_pos_mean - V_mean[home_vidx]                                # [J, 3]
        print(f'\n[{topo}] V_mean={V_mean.shape[0]}, home_vidx range=[{home_vidx.min()}, {home_vidx.max()}]')
        print(f'  offset ||δ|| stats: min={np.linalg.norm(offsets, axis=-1).min():.4f}  '
              f'mean={np.linalg.norm(offsets, axis=-1).mean():.4f}  '
              f'max={np.linalg.norm(offsets, axis=-1).max():.4f}')

        templates = _load_templates(ds, args.data_basedir)
        for V_id, _faces, name in templates:
            out_path = os.path.join(args.feat_dir, f'{name}_bind_pos_landmark.npy')
            if os.path.exists(out_path) and not args.overwrite:
                n_skipped += 1; continue

            V_id = V_id.astype(np.float32)
            assert V_id.shape[0] == V_mean.shape[0], \
                f'{name}: V_id={V_id.shape[0]} vs V_mean={V_mean.shape[0]}'

            bind_pos_id = V_id[home_vidx] + offsets                                # [J, 3]
            # Helper override: helper's per-id bind_pose = per-id PARENT bind_pose
            # (process_order traversal so parents are computed before helpers).
            if helper_idx:
                parent_idx_np = rig.parent_idx.cpu().numpy()
                proc_order = rig.process_order.cpu().numpy()
                for j in proc_order:
                    if int(j) in helper_idx:
                        p = int(parent_idx_np[j])
                        if p >= 0:
                            bind_pos_id[j] = bind_pos_id[p]
            np.save(out_path, bind_pos_id.astype(np.float32))
            n_total += 1

        print(f'  {ds}: written {n_total}  (skipped existing: {n_skipped})')

    print(f'\nTotal written: {n_total}, skipped: {n_skipped}')
    print(f'Cache dir: {args.feat_dir}')


if __name__ == '__main__':
    main()
