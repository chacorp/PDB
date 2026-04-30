"""
Export MF mean mesh (averaged across MF identities) as OBJ for Maya import.

Usage:
    python export_mf_mean_obj.py --data_basedir /data/sihun --out maya_rig/hybrid/mf_mean.obj
"""
import argparse
import os
import pickle
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_basedir', type=str, default='/data/sihun')
    ap.add_argument('--out', type=str, default='maya_rig/hybrid/mf_mean.obj')
    args = ap.parse_args()

    pkl = os.path.join(args.data_basedir, 'multiface_align', 'mf_templates.pkl')
    if not os.path.exists(pkl):
        pkl = os.path.join(args.data_basedir, 'pca', 'multiface_align', 'mf_templates.pkl')
    with open(pkl, 'rb') as f:
        t = pickle.load(f)

    faces = np.array(t['face'], dtype=np.int32)
    id_keys = [k for k in t if k != 'face']
    verts = np.stack([np.array(t[k], dtype=np.float32) for k in id_keys]).mean(axis=0)
    print(f'MF mean: V={verts.shape[0]}, F={faces.shape[0]}, '
          f'averaged across {len(id_keys)} identities')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        for v in verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for tri in faces:
            f.write(f'f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
