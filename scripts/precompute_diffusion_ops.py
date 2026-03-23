"""
scripts/precompute_diffusion_ops.py
=====================================
Precompute DiffusionNet spectral operators (mass, Laplacian eigenvectors, etc.)
for each mesh topology used in training.

Operators are cached by get_operators() as .npz files keyed by mesh hash,
so this script just warms the cache. Subsequent runs (and training) will
load from cache instead of recomputing.

DiffusionNet shared weights work across topologies because the operators
are topology-specific but the learned weights are topology-agnostic.

Usage:
    python scripts/precompute_diffusion_ops.py

Output:
    utils/diffusion_ops/*.npz   (auto-named by mesh hash via get_operators)
"""

import os
import sys
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add DiffusionNet to path
DIFFNET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'third_party', 'diffusion-net', 'src')
if DIFFNET_PATH not in sys.path:
    sys.path.insert(0, DIFFNET_PATH)

import diffusion_net

# ============================================================
# CONFIG
# ============================================================
OP_CACHE_DIR = 'utils/diffusion_ops'
K_EIG        = 128          # number of Laplacian eigenvectors
DATA_BASEDIR = '/data/sihun'

DATASETS = {
    'mf': {
        'loader': 'obj',
        'obj': 'utils/mf/mf_aligned_mean.obj',
    },
    'biwi': {
        'loader': 'biwi',
        'pkl': os.path.join(DATA_BASEDIR, 'BIWI_align_deci/templates_align_deci.pkl'),
        # biwi pkl: dict with subject keys (F1..M5) for verts, 'face' key for faces
    },
    'voca': {
        'loader': 'voca',
        'pkl': os.path.join(DATA_BASEDIR, 'VOCA-COMA/voca_templates.pkl'),
        'std': 'utils/voca/standardization.npy',  # contains 'new_f' face array
    },
}
# ============================================================


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == 'v':
                verts.append([float(x) for x in tok[1:4]])
            elif tok[0] == 'f':
                idx = [int(t.split('/')[0]) - 1 for t in tok[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return (np.array(verts, dtype=np.float32),
            np.array(faces, dtype=np.int32))


def load_biwi_template(pkl_path):
    """BIWI pkl: dict of subject_id → verts [V,3], plus 'face' key for faces."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    # Use first subject as representative template
    subj_keys = [k for k in data if k != 'face']
    verts = data[subj_keys[0]].astype(np.float32)
    faces = data['face'].astype(np.int32)
    return verts, faces


def load_voca_template(pkl_path, std_path):
    """VOCA pkl: dict of subject_id → verts [V,3].
    Faces are in standardization.npy under 'new_f'."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    subj_keys = list(data.keys())
    verts = np.array(data[subj_keys[0]], dtype=np.float32)
    std = np.load(std_path, allow_pickle=True).item()
    faces = std['new_f'].astype(np.int32)
    return verts, faces


def compute_and_cache(name, verts_np, faces_np, cache_dir, k_eig):
    """Run get_operators to warm the cache for this topology."""
    V = torch.tensor(verts_np, dtype=torch.float32)
    F = torch.tensor(faces_np, dtype=torch.int32)

    print(f"  Computing operators for '{name}': {V.shape[0]} verts, {F.shape[0]} faces ...")
    frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
        V, F, k_eig=k_eig, op_cache_dir=cache_dir
    )
    print(f"  '{name}' done: evals={evals.shape}, evecs={evecs.shape}, mass={mass.shape}")
    return frames, mass, L, evals, evecs, gradX, gradY


def main():
    os.makedirs(OP_CACHE_DIR, exist_ok=True)
    print(f"DiffusionNet operator cache: {OP_CACHE_DIR}  k_eig={K_EIG}\n")

    for name, cfg in DATASETS.items():
        print(f"=== {name.upper()} ===")

        loader = cfg['loader']
        try:
            if loader == 'obj':
                if not os.path.exists(cfg['obj']):
                    print(f"  [SKIP] obj not found: {cfg['obj']}\n"); continue
                verts_np, faces_np = load_obj(cfg['obj'])
            elif loader == 'biwi':
                if not os.path.exists(cfg['pkl']):
                    print(f"  [SKIP] pkl not found: {cfg['pkl']}\n"); continue
                verts_np, faces_np = load_biwi_template(cfg['pkl'])
            elif loader == 'voca':
                if not os.path.exists(cfg['pkl']):
                    print(f"  [SKIP] pkl not found: {cfg['pkl']}\n"); continue
                verts_np, faces_np = load_voca_template(cfg['pkl'], cfg['std'])
            else:
                print(f"  [SKIP] unknown loader '{loader}'\n"); continue
        except Exception as e:
            print(f"  [ERROR] {e}\n"); continue

        compute_and_cache(name, verts_np, faces_np, OP_CACHE_DIR, K_EIG)
        print()

    print("Done. Cache files in:", OP_CACHE_DIR)
    for f in sorted(os.listdir(OP_CACHE_DIR)):
        path = os.path.join(OP_CACHE_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {f}  ({size_kb:.0f} KB)")


if __name__ == '__main__':
    main()
