"""
mirror_ict_rig_bind_pos.py — Make ICT rig bind_pos perfectly symmetric.

ICT mean mesh is x-mirror-symmetric, but the Maya rig's bind_world_pos may
contain tiny asymmetries between L/R joint pairs. This script enforces:

    bind_world_pos[Right_X][0] = -bind_world_pos[Left_X][0]
    bind_world_pos[Right_X][1] =  bind_world_pos[Left_X][1]
    bind_world_pos[Right_X][2] =  bind_world_pos[Left_X][2]

Also updates bind_pre_matrix accordingly (translation part).

Center joints (no L/R) → x forced to exactly 0.

A backup `.bak` is written before modification.
"""
import os
import sys
import json
import shutil
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rig_json', type=str,
                    default='maya_rig/hybrid/rig_info_ict.json')
    args = ap.parse_args()

    bak = args.rig_json + '.bak_premirror'
    if not os.path.exists(bak):
        shutil.copy2(args.rig_json, bak)
        print(f'backup → {bak}')
    else:
        print(f'backup already exists, leaving as-is: {bak}')

    with open(args.rig_json) as f:
        rig = json.load(f)
    joints = rig['joints']
    name_to_idx = {j['name']: i for i, j in enumerate(joints)}

    n_mirrored, n_centered = 0, 0
    for j in joints:
        name = j['name']

        # ── Center joints: x forced to 0 ────────────────────────────────
        if name.startswith('hybrid_jnt_') and ('Left_' not in name and 'Right_' not in name):
            pos = j['bind_world_pos']
            if abs(pos[0]) > 0:
                pos[0] = 0.0
                # bind_pre_matrix last row = [-px, -py, -pz, 1] (column-major stored)
                j['bind_pre_matrix'][3][0] = 0.0
                n_centered += 1
                print(f'  center  {name:42s} x→0')

        # ── L→R mirror ─────────────────────────────────────────────────
        elif 'Right_' in name:
            left_name = name.replace('Right_', 'Left_')
            if left_name not in name_to_idx:
                print(f'  [skip] {name}: no Left counterpart {left_name}')
                continue
            left_j = joints[name_to_idx[left_name]]
            lp = left_j['bind_world_pos']
            j['bind_world_pos'] = [-lp[0], lp[1], lp[2]]
            j['bind_pre_matrix'][3][0] =  lp[0]   # -(-lp[0]) = +lp[0]
            j['bind_pre_matrix'][3][1] = -lp[1]
            j['bind_pre_matrix'][3][2] = -lp[2]
            n_mirrored += 1
            print(f'  mirror  {name:42s} ← {left_name}')

    with open(args.rig_json, 'w') as f:
        json.dump(rig, f, indent=2)
    print(f'\nmirrored: {n_mirrored}, centered: {n_centered}, total {len(joints)} joints')
    print(f'updated → {args.rig_json}')


if __name__ == '__main__':
    main()
