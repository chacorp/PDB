"""
setup_helper_joints.py — Apply helper_joints_v1.json to:
  1. rig_info_ict.json / rig_info_mf.json: reparent helpers + reset bind to parent
     (so T_bind_local[helper] = identity; helper starts AT parent, drifts via training)
  2. active_joints_manual.json: add helper_joint_idx + move helpers into face_joint_idx
  3. sigma_targets.npy: set helper σ to default_sigma (looser locality)

Backups (.bak_pre_helpers) written before modification.

Usage:
    python setup_helper_joints.py --config maya_rig/hybrid/helper_joints_v1.json
"""
import os
import sys
import json
import shutil
import argparse
import numpy as np


def _backup(path, suffix='.bak_pre_helpers'):
    bak = path + suffix
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f'  backup → {bak}')


def _update_rig_file(rig_path, helpers, parent_aliases):
    """For each helper, set parent_index + bind_pos = new parent's bind_pos."""
    _backup(rig_path)
    with open(rig_path) as f:
        rig = json.load(f)
    joints = rig['joints']
    name_to_idx = {j['name']: i for i, j in enumerate(joints)}

    n_changed = 0
    for h in helpers:
        jname = h['joint']
        pname = h['parent']
        if jname not in name_to_idx:
            print(f'  [skip] {jname}: not in rig'); continue
        if pname not in [j['name'].replace('hybrid_jnt_', '') for j in joints] and \
           pname not in [j['name'] for j in joints]:
            # try via alias
            pidx = parent_aliases.get(pname)
            if pidx is None:
                print(f'  [skip] {jname}: parent {pname} not found'); continue
        else:
            full_p = pname if pname.startswith('hybrid_jnt_') else f'hybrid_jnt_{pname}'
            pidx = name_to_idx[full_p]

        parent_j = joints[pidx]
        parent_pos = parent_j['bind_world_pos']

        j = joints[name_to_idx[jname]]
        j['parent_index'] = pidx
        j['bind_world_pos'] = list(parent_pos)
        # bind_pre_matrix (column-major): last row = [-px, -py, -pz, 1], rest identity
        j['bind_pre_matrix'] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-parent_pos[0], -parent_pos[1], -parent_pos[2], 1.0],
        ]
        n_changed += 1
        print(f'  {jname:40s} parent → {parent_j["name"]:30s} pos={parent_pos}')

    with open(rig_path, 'w') as f:
        json.dump(rig, f, indent=2)
    print(f'  {n_changed} joints reparented in {rig_path}')


def _update_active_joints(aj_path, helpers, rig_path_for_lookup):
    """Move helper joints from frozen → face_joint_idx + add helper_joint_idx field."""
    _backup(aj_path)
    with open(aj_path) as f:
        aj = json.load(f)
    with open(rig_path_for_lookup) as f:
        rig = json.load(f)
    name_to_idx = {j['name']: i for i, j in enumerate(rig['joints'])}

    helper_idx = sorted(name_to_idx[h['joint']] for h in helpers if h['joint'] in name_to_idx)
    face_idx = sorted(set(aj['face_joint_idx']) | set(helper_idx))
    frozen_idx = sorted(set(aj['frozen_joint_idx']) - set(helper_idx))

    aj['face_joint_idx'] = face_idx
    aj['frozen_joint_idx'] = frozen_idx
    aj['helper_joint_idx'] = helper_idx

    # helper region mapping
    region_to_idx = {}
    for h in helpers:
        if h['joint'] not in name_to_idx: continue
        region_to_idx.setdefault(h['region'], []).append(name_to_idx[h['joint']])
    aj['helper_regions'] = {k: sorted(v) for k, v in region_to_idx.items()}

    aj['counts'] = {
        'active':       len(face_idx),
        'frozen':       len(frozen_idx),
        'helper':       len(helper_idx),
        'helper_per_region': {k: len(v) for k, v in aj['helper_regions'].items()},
    }
    with open(aj_path, 'w') as f:
        json.dump(aj, f, indent=2)
    print(f'  active: {len(face_idx)}, frozen: {len(frozen_idx)}, helper: {len(helper_idx)}')


def _update_sigma_targets(npy_path, helpers, rig_path, sigma_val):
    """Set σ_target for helper indices to sigma_val (looser locality)."""
    _backup(npy_path)
    with open(rig_path) as f:
        rig = json.load(f)
    name_to_idx = {j['name']: i for i, j in enumerate(rig['joints'])}
    sig = np.load(npy_path).astype(np.float32)
    n_set = 0
    for h in helpers:
        if h['joint'] not in name_to_idx: continue
        j = name_to_idx[h['joint']]
        sig[j] = sigma_val
        n_set += 1
    np.save(npy_path, sig)
    print(f'  σ updated for {n_set} helpers → σ={sigma_val}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str,
                    default='maya_rig/hybrid/helper_joints_v1.json')
    ap.add_argument('--rig_ict', type=str,
                    default='maya_rig/hybrid/rig_info_ict.json')
    ap.add_argument('--rig_mf',  type=str,
                    default='maya_rig/hybrid/rig_info_mf.json')
    ap.add_argument('--active_json', type=str,
                    default='maya_rig/hybrid/active_joints_manual.json')
    ap.add_argument('--sigma_npy', type=str,
                    default='maya_rig/hybrid/sigma_targets.npy')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    helpers = cfg['helpers']
    parent_aliases = cfg.get('_parent_aliases', {})
    sigma_val = cfg.get('default_sigma', 0.4)

    print('── Updating rig_info_ict.json ──')
    _update_rig_file(args.rig_ict, helpers, parent_aliases)
    print('\n── Updating rig_info_mf.json ──')
    _update_rig_file(args.rig_mf, helpers, parent_aliases)
    print('\n── Updating active_joints_manual.json ──')
    _update_active_joints(args.active_json, helpers, args.rig_ict)
    print('\n── Updating sigma_targets.npy ──')
    _update_sigma_targets(args.sigma_npy, helpers, args.rig_ict, sigma_val)


if __name__ == '__main__':
    main()
