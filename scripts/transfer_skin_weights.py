"""
Transfer skinning weights from ICT mesh to MF mesh via barycentric projection.

For each MF vertex, find the nearest triangle on ICT mesh (with normal consistency),
compute barycentric coordinates, and interpolate ICT skinning weights.

Usage:
    python scripts/transfer_skin_weights.py \
        --rig_dir maya_rig/hybrid \
        --out_name skin_weights_mf.npy
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.exp_utils import find_closest_valid_feature_normal_first
import trimesh


def transfer_weights(V_src, F_src, V_tar, F_tar, W_tar):
    """
    Transfer per-vertex weights from target mesh to source mesh.

    Args:
        V_src: [N_src, 3] source mesh vertices (MF) — we want weights for these
        F_src: [F_src, 3] source mesh faces
        V_tar: [N_tar, 3] target mesh vertices (ICT) — has known weights
        F_tar: [F_tar, 3] target mesh faces
        W_tar: [N_tar, J] target skinning weights

    Returns:
        W_src: [N_src, J] transferred weights for source mesh
    """
    # Find nearest triangle on ICT for each MF vertex
    nearest_tri_idxs, bary_weights = find_closest_valid_feature_normal_first(
        V_src, F_src, V_tar, F_tar
    )

    N_src = V_src.shape[0]
    J = W_tar.shape[1]
    W_src = np.zeros((N_src, J), dtype=np.float64)

    failed = 0
    for i in range(N_src):
        tri_idx = nearest_tri_idxs[i]
        if tri_idx == -1:
            # Fallback: nearest vertex
            dists = np.linalg.norm(V_tar - V_src[i], axis=1)
            nearest_v = np.argmin(dists)
            W_src[i] = W_tar[nearest_v]
            failed += 1
            continue

        # Barycentric interpolation of weights
        v0, v1, v2 = F_tar[tri_idx]
        u, v, w = bary_weights[i]
        W_src[i] = u * W_tar[v0] + v * W_tar[v1] + w * W_tar[v2]

    # Normalize rows to sum to 1
    row_sums = W_src.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    W_src = W_src / row_sums

    if failed > 0:
        print(f"  [warn] {failed}/{N_src} vertices used nearest-vertex fallback")

    return W_src


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig_dir", type=str, default="maya_rig/hybrid")
    parser.add_argument("--ict_obj", type=str, default="utils/ict/ict_aligned_mean.obj")
    parser.add_argument("--mf_obj", type=str, default="utils/mf/mf_aligned_mean.obj")
    parser.add_argument("--out_name", type=str, default="skin_weights_mf.npy")
    args = parser.parse_args()

    # Load meshes
    print(f"Loading ICT mesh: {args.ict_obj}")
    mesh_ict = trimesh.load(args.ict_obj, process=False)
    V_ict, F_ict = np.array(mesh_ict.vertices), np.array(mesh_ict.faces)
    print(f"  vertices: {V_ict.shape[0]}, faces: {F_ict.shape[0]}")

    print(f"Loading MF mesh: {args.mf_obj}")
    mesh_mf = trimesh.load(args.mf_obj, process=False)
    V_mf, F_mf = np.array(mesh_mf.vertices), np.array(mesh_mf.faces)
    print(f"  vertices: {V_mf.shape[0]}, faces: {F_mf.shape[0]}")

    # Load ICT skinning weights
    ict_weight_path = os.path.join(args.rig_dir, "skin_weights_ict.npy")
    print(f"Loading ICT weights: {ict_weight_path}")
    W_ict = np.load(ict_weight_path)
    print(f"  shape: {W_ict.shape}")

    # Transfer
    print("Transferring weights via barycentric projection...")
    W_mf = transfer_weights(V_mf, F_mf, V_ict, F_ict, W_ict)
    print(f"  result shape: {W_mf.shape}")

    # Save
    out_path = os.path.join(args.rig_dir, args.out_name)
    np.save(out_path, W_mf.astype(np.float32))
    print(f"Saved: {out_path}")

    # Quick stats
    print(f"\n  Weight stats:")
    print(f"    min: {W_mf.min():.6f}, max: {W_mf.max():.6f}")
    print(f"    row sums: {W_mf.sum(1).min():.6f} ~ {W_mf.sum(1).max():.6f}")
    nonzero_per_row = (W_mf > 0.01).sum(axis=1)
    print(f"    avg joints per vertex (>0.01): {nonzero_per_row.mean():.1f}")


if __name__ == "__main__":
    main()
