"""
maya_rig/transfer_weights.py
=============================
Transfer skin weights from the MF template to other mesh topologies
(BIWI, VOCA) via barycentric interpolation.

For each target vertex v_t:
    1. Find best-matching face in MF mesh (kNN on face centers + normal scoring)
    2. Project v_t onto that face plane → barycentric coords (u, v, w)
    3. W_target[v_t] = u*W_mf[f0] + v*W_mf[f1] + w*W_mf[f2]
    4. Renormalize rows to sum = 1

Reuses exp_utils.find_barycentric_with_adjacency which already handles
normal-aware face selection and projection.

Usage:
    python maya_rig/transfer_weights.py

Output:
    maya_rig/skin_weights_biwi.npy  [2560, 84]
    maya_rig/skin_weights_voca.npy  [3525, 84]
"""

import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.exp_utils import find_barycentric_with_adjacency

# ============================================================
# CONFIG
# ============================================================
RIG_DIR      = 'maya_rig'
MF_OBJ_PATH  = 'utils/mf/mf_aligned_mean.obj'
DATA_BASEDIR = '/data/sihun'

TARGET_DATASETS = {
    'biwi': {
        'pkl': os.path.join(DATA_BASEDIR, 'BIWI_align_deci/templates_align_deci.pkl'),
        'out': os.path.join(RIG_DIR, 'skin_weights_biwi.npy'),
    },
    'voca': {
        'pkl': os.path.join(DATA_BASEDIR, 'VOCA-COMA/voca_templates.pkl'),
        'out': os.path.join(RIG_DIR, 'skin_weights_voca.npy'),
    },
}
# ============================================================


def load_obj(path):
    """Minimal OBJ loader → (verts [V,3], faces [F,3])."""
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
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def load_template_from_pkl(pkl_path):
    """
    Extract a representative template mesh (verts, faces) from dataset pkl.
    Tries common key formats used in this codebase.
    Returns (verts [V,3], faces [F,3] or None).
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    verts, faces = None, None

    if isinstance(data, dict):
        # Try vertex arrays
        for k in ('verts', 'vertices', 'template', 'v_template', 'neutral'):
            if k in data:
                v = np.array(data[k])
                if v.ndim == 3:   # [N_ids, V, 3]
                    v = v[0]
                if v.ndim == 2 and v.shape[1] == 3:
                    verts = v.astype(np.float32)
                    break
        # Try face arrays
        for k in ('faces', 'tris', 'triangles', 'f'):
            if k in data:
                faces = np.array(data[k]).astype(np.int32)
                break

    elif isinstance(data, (list, tuple)):
        # Some pkls are lists of per-identity dicts
        item = data[0]
        if isinstance(item, dict):
            for k in ('verts', 'vertices', 'template'):
                if k in item:
                    verts = np.array(item[k], dtype=np.float32)
                    break
        elif isinstance(item, np.ndarray) and item.ndim == 2:
            verts = item.astype(np.float32)

    if verts is None:
        raise RuntimeError(
            f"Cannot parse template vertices from {pkl_path}.\n"
            f"Keys found: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )

    return verts, faces


def transfer_weights(
    src_verts, src_faces, src_W,
    tgt_verts, tgt_faces,
    k=10, normal_weight=0.3,
):
    """
    Transfer skin weights [Vs, J] → [Vt, J] via barycentric interpolation.

    Uses find_barycentric_with_adjacency from exp_utils:
        - kNN on face centers in MF mesh
        - normal-aware scoring to avoid back-face matches
        - project target vertex onto chosen face plane → barycentric (u,v,w)

    Args:
        src_verts  [Vs, 3]
        src_faces  [Fs, 3]
        src_W      [Vs, J]
        tgt_verts  [Vt, 3]
        tgt_faces  [Ft, 3]  (target mesh faces, used for adjacency context)
        k          int       candidate faces per vertex
        normal_weight float  0=distance only, 1=normal only

    Returns:
        tgt_W  [Vt, J]  float32, rows sum to 1
    """
    print(f"  find_barycentric_with_adjacency: {tgt_verts.shape[0]} tgt verts → MF {src_verts.shape[0]} verts ...")

    # find_barycentric_with_adjacency convention:
    #   V_src / F_src = the QUERY mesh (vertices we want to project)
    #   V_tar / F_tar = the TARGET mesh (mesh we project onto)
    # So: V_src=tgt (BIWI/VOCA), V_tar=src (MF)
    tri_idx, bary = find_barycentric_with_adjacency(
        V_src=tgt_verts,
        F_src=tgt_faces if tgt_faces is not None else np.zeros((0, 3), dtype=np.int32),
        V_tar=src_verts,
        F_tar=src_faces,
        normal_weight=normal_weight,
        k=k,
    )

    Vt = tgt_verts.shape[0]
    J  = src_W.shape[1]
    tgt_W = np.zeros((Vt, J), dtype=np.float32)

    for vi in range(Vt):
        fi = tri_idx[vi]
        if fi < 0:
            # fallback: uniform over all joints
            tgt_W[vi] = 1.0 / J
            continue
        u, v, w = bary[vi]
        i0, i1, i2 = src_faces[fi]
        interp = u * src_W[i0] + v * src_W[i1] + w * src_W[i2]
        s = interp.sum()
        tgt_W[vi] = interp / (s + 1e-12)

    return tgt_W


def main():
    # ---- Load MF source ----
    print(f"Loading MF template: {MF_OBJ_PATH}")
    mf_verts, mf_faces = load_obj(MF_OBJ_PATH)
    print(f"  MF: {mf_verts.shape[0]} verts, {mf_faces.shape[0]} faces")

    mf_W_path = os.path.join(RIG_DIR, 'skin_weights.npy')
    mf_W = np.load(mf_W_path).astype(np.float32)   # [5223, 84]
    print(f"  MF weights: {mf_W.shape}")

    # ---- Transfer to each target topology ----
    for name, cfg in TARGET_DATASETS.items():
        pkl_path = cfg['pkl']
        out_path = cfg['out']

        if not os.path.exists(pkl_path):
            print(f"\n[SKIP] {name}: pkl not found at {pkl_path}")
            continue

        print(f"\n--- {name.upper()} ---")
        tgt_verts, tgt_faces = load_template_from_pkl(pkl_path)
        print(f"  Template: {tgt_verts.shape[0]} verts")
        if tgt_faces is not None:
            print(f"  Faces:    {tgt_faces.shape[0]}")

        tgt_W = transfer_weights(mf_verts, mf_faces, mf_W, tgt_verts, tgt_faces)

        np.save(out_path, tgt_W)
        rs = tgt_W.sum(axis=1)
        nnz = (tgt_W > 1e-6).sum(axis=1)
        print(f"  Saved {out_path}")
        print(f"  shape={tgt_W.shape}  row_sum: {rs.min():.4f}~{rs.max():.4f}  "
              f"nnz/row: {nnz.mean():.1f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
