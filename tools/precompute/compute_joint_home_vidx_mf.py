"""
compute_joint_home_vidx_mf.py — Manual MF home_vertex specification.

MF mesh is scan-registered (NOT parametric morph) and may have slight
left/right asymmetry. So unlike ICT we DON'T mirror by default; each side
gets its own DTU3D landmark.

Exception: Right_Chin = mirror x of Left_Chin's vertex (user explicit).

Output:
  maya_rig/hybrid/joint_home_vidx_mf.npy   [J] long  vertex indices

Landmark idx = 1-indexed (DTU3D doc / PNG label convention).
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


SPECS = {
    # ── Center (no L/R) — x forced to 0 + centerline-restricted ──────────
    'hybrid_jnt_Top_Lip':              ('lm',  50),
    'hybrid_jnt_Bottom_Lip':           ('lm',  55),
    'hybrid_jnt_Mouth_Center_Top':     ('avg', [50, 40]),
    'hybrid_jnt_Mouth_Center_Bottom':  ('avg', [55, 68]),
    'hybrid_jnt_Nose_Tip':             ('lm',  46),
    'hybrid_jnt_Nose_Base_Rotate':     ('lm',  40),

    # ── Left side ────────────────────────────────────────────────────────
    'hybrid_jnt_Left_Brow_Outer':       ('lm',  9),
    'hybrid_jnt_Left_Eyebrow_Mid':      ('avg', [15, 11]),
    'hybrid_jnt_Left_Eyebrow_Inner':    ('lm',  13),
    'hybrid_jnt_Left_Upper_Eyelid':     ('lm',  31),
    'hybrid_jnt_Left_Lower_Eyelid':     ('lm',  27),
    'hybrid_jnt_Left_Eye_Inner_Corner': ('lm',  29),
    'hybrid_jnt_Left_Eye_Outer_Corner': ('lm',  25),
    'hybrid_jnt_Left_Chin':             ('lm',  71),
    'hybrid_jnt_Left_Lip_Upper_Side':   ('lm',  52),
    'hybrid_jnt_Left_Lip_Lower_Side':   ('lm',  54),
    'hybrid_jnt_Left_Nose_Wing':        ('lm',  37),
    'hybrid_jnt_Mouth_Left_Corner':     ('vidx', 2950),

    # ── Right side — direct DTU3D landmarks (no mirror, MF asymmetric) ──
    'hybrid_jnt_Right_Brow_Outer':       ('lm',  1),
    'hybrid_jnt_Right_Eyebrow_Mid':      ('avg', [3, 7]),     # mirror rule from Left, but with right-side landmarks
    'hybrid_jnt_Right_Eyebrow_Inner':    ('lm',  5),
    'hybrid_jnt_Right_Upper_Eyelid':     ('lm',  19),
    'hybrid_jnt_Right_Lower_Eyelid':     ('lm',  23),
    'hybrid_jnt_Right_Eye_Inner_Corner': ('lm',  21),
    'hybrid_jnt_Right_Eye_Outer_Corner': ('lm',  17),
    # 71만 대칭: Right_Chin = mirror x of Left_Chin's home vertex
    'hybrid_jnt_Right_Chin':             ('mirror', 'hybrid_jnt_Left_Chin'),
    'hybrid_jnt_Right_Lip_Upper_Side':   ('lm',  48),
    'hybrid_jnt_Right_Lip_Lower_Side':   ('lm',  56),
    'hybrid_jnt_Right_Nose_Wing':        ('lm',  43),
    'hybrid_jnt_Mouth_Right_Corner':     ('vidx', 1409),
}

# Center joints — pt projected to x=0 + nearest-vertex restricted to centerline.
# Includes fallback center joints (Jaw_Front, Brow_Center).
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
CENTERLINE_EPS = 0.02     # MF mesh has sparser centerline than ICT (V=5223 vs 11248)


def _load_mf_mean(data_basedir):
    pkl = os.path.join(data_basedir, 'multiface_align', 'mf_templates.pkl')
    if not os.path.exists(pkl):
        pkl = os.path.join(data_basedir, 'pca', 'multiface_align', 'mf_templates.pkl')
    with open(pkl, 'rb') as f:
        t = pickle.load(f)
    faces = np.array(t['face'], dtype=np.int32)
    keys = [k for k in t if k != 'face']
    verts = np.stack([np.array(t[k], dtype=np.float32) for k in keys]).mean(axis=0)
    return verts.astype(np.float32), faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rig_path', type=str, default='maya_rig/hybrid')
    ap.add_argument('--data_basedir', type=str, default='/data/sihun')
    ap.add_argument('--landmarks_npy', type=str,
                    default='maya_rig/hybrid/landmarks/mf_mean_landmarks.npy')
    ap.add_argument('--out_path', type=str,
                    default='maya_rig/hybrid/joint_home_vidx_mf.npy')
    args = ap.parse_args()

    V, _faces = _load_mf_mean(args.data_basedir)
    rig = load_rig(args.rig_path, device='cpu')
    J = len(rig.joint_names)
    lm = np.load(args.landmarks_npy).astype(np.float32)
    print(f'MF mean: V={V.shape[0]}, landmarks={lm.shape[0]}')

    centerline_mask = np.abs(V[:, 0]) < CENTERLINE_EPS
    centerline_idx = np.where(centerline_mask)[0]
    print(f'  centerline vertices (|x|<{CENTERLINE_EPS}): {len(centerline_idx)}')

    def nearest_v(pt, on_centerline=False):
        if on_centerline:
            d2 = ((V[centerline_idx] - pt) ** 2).sum(axis=-1)
            return int(centerline_idx[int(d2.argmin())])
        return int(np.argmin(((V - pt) ** 2).sum(axis=-1)))

    home_pos, home_vidx = {}, {}

    # Pass 1: non-mirror specs
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

    # Build [J] array with maya_bind fallback for unmapped joints
    maya_bind = rig.bind_pos_dict['mf'].cpu().numpy().astype(np.float32)
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
