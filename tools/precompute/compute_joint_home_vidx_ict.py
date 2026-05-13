"""
compute_joint_home_vidx_ict.py — Manual ICT home_vertex specification.

User-curated home_vertex per face joint, using a mix of:
  - Single landmark coord     (existing pattern)
  - Average of landmark coords
  - Direct vertex index on ICT mean
  - Mirror x of another joint's home_vertex (ICT mean is symmetric)

Landmark indices below are 1-indexed (matching DTU3D doc + PNG labels);
converted to 0-index in code via -1.

Output:
  maya_rig/hybrid/joint_home_vidx_ict.npy   [J] long  vertex indices
"""
import os
import sys
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))
from utils.rig_loader import load_rig
from utils.remesh_utils import ICT_face_model


# Landmark indices = 1-indexed (DTU3D doc / PNG label convention)
SPECS = {
    # ── Center (no L/R) — x forced to 0 (mid-sagittal plane) ─────────────
    'hybrid_jnt_Top_Lip':              ('lm',  50),
    'hybrid_jnt_Bottom_Lip':           ('lm',  55),
    'hybrid_jnt_Mouth_Center_Top':     ('avg', [50, 40]),
    'hybrid_jnt_Mouth_Center_Bottom':  ('avg', [55, 68]),
    'hybrid_jnt_Nose_Tip':             ('lm',  46),
    'hybrid_jnt_Nose_Base_Rotate':     ('lm',  40),

    # ── Left side ───────────────────────────────────────────────────────
    'hybrid_jnt_Left_Brow_Outer':      ('lm',  9),
    'hybrid_jnt_Left_Eyebrow_Mid':     ('avg', [15, 11]),
    'hybrid_jnt_Left_Eyebrow_Inner':   ('lm',  13),
    'hybrid_jnt_Left_Upper_Eyelid':    ('lm',  31),
    'hybrid_jnt_Left_Lower_Eyelid':    ('lm',  27),
    'hybrid_jnt_Left_Eye_Inner_Corner':('lm',  29),
    'hybrid_jnt_Left_Eye_Outer_Corner':('lm',  25),
    'hybrid_jnt_Left_Chin':            ('lm',  71),
    'hybrid_jnt_Left_Lip_Upper_Side':  ('avg', [51, 52]),
    'hybrid_jnt_Left_Lip_Lower_Side':  ('lm',  54),
    'hybrid_jnt_Left_Nose_Wing':       ('lm',  37),
    'hybrid_jnt_Mouth_Left_Corner':    ('vidx', 6102),     # direct vertex idx

    # ── Right side — all mirror x of corresponding Left ─────────────────
    'hybrid_jnt_Right_Brow_Outer':       ('mirror', 'hybrid_jnt_Left_Brow_Outer'),
    'hybrid_jnt_Right_Eyebrow_Mid':      ('mirror', 'hybrid_jnt_Left_Eyebrow_Mid'),
    'hybrid_jnt_Right_Eyebrow_Inner':    ('mirror', 'hybrid_jnt_Left_Eyebrow_Inner'),
    'hybrid_jnt_Right_Upper_Eyelid':     ('mirror', 'hybrid_jnt_Left_Upper_Eyelid'),
    'hybrid_jnt_Right_Lower_Eyelid':     ('mirror', 'hybrid_jnt_Left_Lower_Eyelid'),
    'hybrid_jnt_Right_Eye_Inner_Corner': ('mirror', 'hybrid_jnt_Left_Eye_Inner_Corner'),
    'hybrid_jnt_Right_Eye_Outer_Corner': ('mirror', 'hybrid_jnt_Left_Eye_Outer_Corner'),
    'hybrid_jnt_Right_Cheek_Bone':       ('mirror', 'hybrid_jnt_Left_Cheek_Bone'),
    'hybrid_jnt_Right_Cheek':            ('mirror', 'hybrid_jnt_Left_Cheek'),
    'hybrid_jnt_Right_Chin':             ('mirror', 'hybrid_jnt_Left_Chin'),
    'hybrid_jnt_Right_Lip_Upper_Side':   ('mirror', 'hybrid_jnt_Left_Lip_Upper_Side'),
    'hybrid_jnt_Right_Lip_Lower_Side':   ('mirror', 'hybrid_jnt_Left_Lip_Lower_Side'),
    'hybrid_jnt_Right_Nose_Wing':        ('mirror', 'hybrid_jnt_Left_Nose_Wing'),
    'hybrid_jnt_Mouth_Right_Corner':     ('mirror', 'hybrid_jnt_Mouth_Left_Corner'),
}

# Center joints — pt is projected to x=0, AND nearest-vertex search is
# restricted to mesh centerline (|x| < CENTERLINE_EPS).
# Includes fallback-only center joints (Maxilla, Jaw_Front, Brow_Center) too.
CENTER_JOINTS = {
    'hybrid_jnt_Top_Lip',
    'hybrid_jnt_Bottom_Lip',
    'hybrid_jnt_Mouth_Center_Top',
    'hybrid_jnt_Mouth_Center_Bottom',
    'hybrid_jnt_Nose_Tip',
    'hybrid_jnt_Nose_Base_Rotate',
    'hybrid_jnt_Jaw_Front',
    'hybrid_jnt_Brow_Center',
}
CENTERLINE_EPS = 0.01     # vertices with |x| < this count as "on the centerline"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rig_path', type=str, default='maya_rig/hybrid')
    ap.add_argument('--landmarks_npy', type=str,
                    default='maya_rig/hybrid/landmarks/ict_mean_landmarks.npy')
    ap.add_argument('--out_path', type=str,
                    default='maya_rig/hybrid/joint_home_vidx_ict.npy')
    args = ap.parse_args()

    m = ICT_face_model()
    V = m.neutral_verts.astype(np.float32)
    rig = load_rig(args.rig_path, device='cpu')
    J = len(rig.joint_names)
    lm = np.load(args.landmarks_npy).astype(np.float32)            # [73, 3]
    print(f'ICT mean: V={V.shape[0]}, landmarks={lm.shape[0]}')

    centerline_mask = np.abs(V[:, 0]) < CENTERLINE_EPS                 # [V] bool
    centerline_idx = np.where(centerline_mask)[0]
    print(f'  centerline vertices (|x|<{CENTERLINE_EPS}): {len(centerline_idx)}')

    def nearest_v(pt, on_centerline=False):
        if on_centerline:
            d2 = ((V[centerline_idx] - pt) ** 2).sum(axis=-1)
            return int(centerline_idx[int(d2.argmin())])
        return int(np.argmin(((V - pt) ** 2).sum(axis=-1)))

    home_pos = {}
    home_vidx = {}

    # Pass 1: non-mirror specs (lm idx converted from 1- to 0-indexed)
    for jname, spec in SPECS.items():
        typ = spec[0]
        if typ == 'lm':
            pt = lm[spec[1] - 1].copy()
        elif typ == 'avg':
            pt = lm[[k - 1 for k in spec[1]]].mean(axis=0)
        elif typ == 'vidx':
            pt = V[spec[1]].copy()
        else:
            continue
        is_center = jname in CENTER_JOINTS
        if is_center:
            pt = pt.copy(); pt[0] = 0.0
        home_pos[jname]  = pt
        home_vidx[jname] = nearest_v(pt, on_centerline=is_center)

    # Pass 2: mirror specs
    for jname, spec in SPECS.items():
        if spec[0] != 'mirror':
            continue
        src = spec[1]
        if src not in home_pos:
            print(f'  [warn] {jname}: mirror src {src} not resolved; skip')
            continue
        pt = home_pos[src].copy()
        pt[0] = -pt[0]
        home_pos[jname]  = pt
        home_vidx[jname] = nearest_v(pt)

    # Build [J] array; fallback to maya_bind argmin for unmapped
    maya_bind = rig.bind_pos_dict['ict'].cpu().numpy().astype(np.float32)
    out = np.zeros(J, dtype=np.int64)
    n_user, n_fallback = 0, 0
    for j, jname in enumerate(rig.joint_names):
        if jname in home_vidx:
            out[j] = home_vidx[jname]
            n_user += 1
        else:
            pt = maya_bind[j].astype(np.float32).copy()
            is_center = jname in CENTER_JOINTS
            if is_center:
                pt[0] = 0.0
            out[j] = nearest_v(pt, on_centerline=is_center)
            n_fallback += 1
    print(f'  user-spec joints: {n_user}, maya_bind fallback: {n_fallback}')
    print(f'  out shape: {out.shape}, range=[{out.min()}, {out.max()}]')

    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
    np.save(args.out_path, out)
    print(f'  saved → {args.out_path}')


if __name__ == '__main__':
    main()
