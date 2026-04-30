"""
precompute_per_id_bind_pos.py — Cache per-identity bind_pos for training efficiency.

For every cached NFS feature in `nfs_features_seg/{id_name}_nfs_feat.npy`,
applies anchor pooling (anchor + offset from precompute_joint_anchors.py) using
the identity's mesh template, and saves `{id_name}_bind_pos.npy [J, 3]`.

Same anchor & offset are used for all IDs and all topologies — they were
derived once on ICT mean and are universal.

Usage:
    python precompute_per_id_bind_pos.py \
        --rig_path maya_rig/hybrid \
        --feat_dir nfs_features_seg \
        --datasets ict mf \
        --data_basedir /data/sihun
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from utils.anchor_pool import compute_bind_pos
from extract_nfs_feat import load_templates_ict, load_templates_mf, load_templates_coma


def _load_templates(dataset, data_basedir):
    if dataset == 'ict':
        return load_templates_ict()
    if dataset == 'mf':
        return load_templates_mf(data_basedir)
    if dataset == 'coma':
        return load_templates_coma()
    raise ValueError(f'unknown dataset: {dataset}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rig_path',  type=str, required=True)
    ap.add_argument('--feat_dir',  type=str, default='nfs_features_seg',
                    help='Directory of NFS feat caches (also where bind_pos cache is written)')
    ap.add_argument('--datasets',  nargs='+', default=['ict', 'mf'],
                    choices=['ict', 'mf', 'coma'])
    ap.add_argument('--data_basedir', type=str, default='/data/sihun')
    ap.add_argument('--anchors_npy', type=str, default=None,
                    help='Path to joint_anchors.npy (default: {rig_path}/joint_anchors.npy)')
    ap.add_argument('--offsets_npy', type=str, default=None,
                    help='Path to joint_offsets.npy (default: {rig_path}/joint_offsets.npy)')
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    device = torch.device(args.device)
    anchors_path = args.anchors_npy or os.path.join(args.rig_path, 'joint_anchors.npy')
    offsets_path = args.offsets_npy or os.path.join(args.rig_path, 'joint_offsets.npy')

    anchor = torch.tensor(np.load(anchors_path), dtype=torch.float32, device=device)  # [J, 256]
    offset = torch.tensor(np.load(offsets_path), dtype=torch.float32, device=device)  # [J, 3]
    J, K = anchor.shape
    print(f'anchor:  {tuple(anchor.shape)}  from {anchors_path}')
    print(f'offset:  {tuple(offset.shape)}  ||δ|| range='
          f'[{offset.norm(dim=-1).min():.4f}, {offset.norm(dim=-1).max():.4f}]')
    print(f'temp:    {args.temperature}')

    n_total = 0
    n_skipped = 0
    n_missing = 0

    for ds in args.datasets:
        print(f'\n── dataset: {ds} ──')
        templates = _load_templates(ds, args.data_basedir)
        # Map id_name -> verts
        verts_by_id = {name: verts for verts, _faces, name in templates}

        # Find feat caches matching this dataset's id_names
        for name, verts_np in verts_by_id.items():
            feat_path = os.path.join(args.feat_dir, f'{name}_nfs_feat.npy')
            out_path  = os.path.join(args.feat_dir, f'{name}_bind_pos.npy')

            if not os.path.exists(feat_path):
                n_missing += 1
                continue
            if os.path.exists(out_path) and not args.overwrite:
                n_skipped += 1
                continue

            feat = torch.tensor(np.load(feat_path), dtype=torch.float32, device=device)  # [V, K]
            verts = torch.tensor(verts_np, dtype=torch.float32, device=device)            # [V, 3]
            assert feat.shape[0] == verts.shape[0], \
                f'{name}: feat V={feat.shape[0]} vs mesh V={verts.shape[0]}'

            with torch.no_grad():
                bind_pos = compute_bind_pos(verts, feat, anchor, offset,
                                             temperature=args.temperature,
                                             offset_frame='world')                       # [J, 3]
            np.save(out_path, bind_pos.cpu().numpy().astype(np.float32))
            n_total += 1

        print(f'  computed: {n_total}  (skipped existing: {n_skipped}, missing feat: {n_missing})')

    print(f'\nDone. Total written: {n_total}, skipped: {n_skipped}, missing: {n_missing}')
    print(f'Cache dir: {args.feat_dir}')


if __name__ == '__main__':
    main()
