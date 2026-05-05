"""
precompute_per_id_bind_pos_landmark.py — Per-id bind_pose GT via landmark
vertex-index inheritance + mean offset (Phase B of landmark-based supervision).

Per-topology, on the mean mesh:
    offset[j] = bind_pos_mean_GT[j] - V_mean[ landmark_vidx[ map[j] ] ]
                                                  └─ for joints with landmark mapping
    (unmapped joints: offset[j] = 0, fallback to bind_pos_mean_GT)

Per-id (same topology, same vertex index semantics):
    bind_pose_id[j] = V_id[ landmark_vidx[ map[j] ] ] + offset[j]    (mapped)
    bind_pose_id[j] = bind_pos_mean_GT[j]                            (unmapped)

This deterministically gives per-id bind_pose that adapts to per-id mesh shape
at anatomically meaningful landmarks (lips/eyes/brow/nose/jaw), while keeping
unmapped joints (cheek/maxilla/etc.) anchored to per-topo mean.

Output: {feat_dir}/{id_name}_bind_pos_landmark.npy   shape [J, 3]

Usage:
    python precompute_per_id_bind_pos_landmark.py \
        --rig_path maya_rig/hybrid \
        --landmark_map maya_rig/hybrid/joint_landmark_map.json \
        --feat_dir nfs_features_seg \
        --datasets ict mf
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils.rig_loader import load_rig
from extract_nfs_feat import load_templates_ict, load_templates_mf, load_templates_coma


def _load_templates(dataset, data_basedir):
    if dataset == 'ict':
        return load_templates_ict()
    if dataset == 'mf':
        return load_templates_mf(data_basedir)
    if dataset == 'coma':
        return load_templates_coma()
    raise ValueError(f'unknown dataset: {dataset}')


def _load_mean_verts(topo, data_basedir):
    """Mean vertex positions for a topology (same source used in landmark precompute)."""
    if topo == 'ict':
        from utils.remesh_utils import ICT_face_model
        return ICT_face_model().neutral_verts.astype(np.float32)
    if topo == 'mf':
        import pickle
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
    ap.add_argument('--rig_path',     type=str, required=True)
    ap.add_argument('--landmark_map', type=str, required=True,
                    help='joint_landmark_map.json (joint_name → DTU3D landmark idx, null for fallback)')
    ap.add_argument('--landmark_vidx_dir', type=str, default=None,
                    help='Dir with landmark_vidx_{topo}.npy. Defaults to --rig_path.')
    ap.add_argument('--feat_dir',     type=str, default='nfs_features_seg',
                    help='Output dir for {id_name}_bind_pos_landmark.npy')
    ap.add_argument('--datasets',     nargs='+', default=['ict', 'mf'],
                    choices=['ict', 'mf', 'coma'])
    ap.add_argument('--data_basedir', type=str, default='/data/sihun')
    ap.add_argument('--overwrite',    action='store_true')
    args = ap.parse_args()

    lvidx_dir = args.landmark_vidx_dir or args.rig_path

    # ── Load mapping + rig ──────────────────────────────────────────────
    with open(args.landmark_map) as f:
        joint_lm_map = json.load(f)['mapping']
    rig = load_rig(args.rig_path, device='cpu')
    J = len(rig.joint_names)
    print(f'rig: J={J}, landmark map entries={len(joint_lm_map)}')

    # Map dataset → topology key (for rig.bind_pos_dict + landmark_vidx file)
    DATASET_TO_TOPO = {'ict': 'ict', 'mf': 'mf', 'coma': 'voca'}

    n_total = 0; n_skipped = 0

    for ds in args.datasets:
        topo = DATASET_TO_TOPO[ds]
        print(f'\n── dataset: {ds} (topo={topo}) ──')

        # Per-topo GT mean bind_pos
        if topo not in rig.bind_pos_dict:
            print(f'  [skip] {topo}: bind_pos_dict not loaded'); continue
        bind_pos_mean_GT = rig.bind_pos_dict[topo].cpu().numpy().astype(np.float32)   # [J, 3]

        # landmark_vidx file
        lvidx_path = os.path.join(lvidx_dir, f'landmark_vidx_{topo}.npy')
        if not os.path.exists(lvidx_path):
            print(f'  [skip] {lvidx_path} not found'); continue
        landmark_vidx = np.load(lvidx_path).astype(np.int64)                           # [73]

        # Mean verts
        V_mean = _load_mean_verts(topo, args.data_basedir)                             # [V, 3]
        if V_mean.shape[0] != landmark_vidx.max() + 1 and landmark_vidx.max() >= V_mean.shape[0]:
            raise ValueError(f'{topo}: landmark_vidx max ({landmark_vidx.max()}) ≥ V ({V_mean.shape[0]})')

        # ── Compute offsets per joint ──────────────────────────────────
        # mapped[j] = True iff joint j has landmark mapping
        # offset[j] = bind_pos_mean_GT[j] - V_mean[lvidx[lm_idx]]
        mapped   = np.zeros(J, dtype=bool)
        offsets  = np.zeros((J, 3), dtype=np.float32)
        lvidx_per_joint = np.full(J, -1, dtype=np.int64)
        for j, jname in enumerate(rig.joint_names):
            lm = joint_lm_map.get(jname)
            if lm is not None:
                mapped[j] = True
                lvidx_per_joint[j] = int(landmark_vidx[lm])
                offsets[j] = bind_pos_mean_GT[j] - V_mean[lvidx_per_joint[j]]
        n_mapped = int(mapped.sum())
        print(f'  joints: mapped={n_mapped}/{J}, fallback={J-n_mapped}')
        if n_mapped > 0:
            print(f'  offset ||δ||: min={np.linalg.norm(offsets[mapped], axis=-1).min():.4f}  '
                  f'mean={np.linalg.norm(offsets[mapped], axis=-1).mean():.4f}  '
                  f'max={np.linalg.norm(offsets[mapped], axis=-1).max():.4f}')

        # ── Per-id bind_pose GT ─────────────────────────────────────────
        templates = _load_templates(ds, args.data_basedir)
        for V_id, _faces, name in templates:
            out_path = os.path.join(args.feat_dir, f'{name}_bind_pos_landmark.npy')
            if os.path.exists(out_path) and not args.overwrite:
                n_skipped += 1
                continue

            V_id = V_id.astype(np.float32)
            assert V_id.shape[0] == V_mean.shape[0], \
                f'{name}: V_id={V_id.shape[0]} vs V_mean={V_mean.shape[0]}'

            bind_pos_id = bind_pos_mean_GT.copy()                                       # default
            if n_mapped > 0:
                # bind_pose_id[j] = V_id[lvidx] + offset[j]   for mapped j
                bind_pos_id[mapped] = V_id[lvidx_per_joint[mapped]] + offsets[mapped]

            os.makedirs(args.feat_dir, exist_ok=True)
            np.save(out_path, bind_pos_id)
            n_total += 1

        print(f'  written: {n_total}  (skipped existing: {n_skipped})')

    print(f'\nDone. Total written: {n_total}, skipped: {n_skipped}')
    print(f'Cache dir: {args.feat_dir}  (file pattern: {{id_name}}_bind_pos_landmark.npy)')


if __name__ == '__main__':
    main()
