"""
precompute_geo_dist.py — Per-topology geodesic distance from each joint's
home vertex to all mesh vertices, for use in geodesic Gauss prior.

For each topology (ict, mf):
    1. Load mean mesh (ict from ICT_face_model, mf from mf_templates.pkl mean)
    2. For each joint j, find home_vertex_j = argmin_v ||V[v] - maya_bind[j]||
    3. Heat method (potpourri3d) to compute geodesic distance from home_v_j to all vertices
    4. Save:
        maya_rig/hybrid/geo_dist_{topo}.npy   [J, V]   geodesic distances
        maya_rig/hybrid/home_verts_{topo}.npy [J]      home vertex indices

Usage:
    python precompute_geo_dist.py --rig_path maya_rig/hybrid \
        --topologies ict mf --data_basedir /data/sihun
"""
import os
import sys
import argparse
import numpy as np
import potpourri3d as pp3d

sys.path.insert(0, os.path.dirname(__file__))
from utils.rig_loader import load_rig


def _load_mean_mesh(topo, data_basedir):
    if topo == 'ict':
        from utils.remesh_utils import ICT_face_model
        m = ICT_face_model()
        return m.neutral_verts.astype(np.float32), m.faces.astype(np.int32)
    if topo == 'mf':
        import pickle
        pkl = os.path.join(data_basedir, 'multiface_align', 'mf_templates.pkl')
        if not os.path.exists(pkl):
            pkl = os.path.join(data_basedir, 'pca', 'multiface_align', 'mf_templates.pkl')
        with open(pkl, 'rb') as f:
            t = pickle.load(f)
        faces = np.array(t['face'], dtype=np.int32)
        keys = [k for k in t if k != 'face']
        verts = np.stack([np.array(t[k], dtype=np.float32) for k in keys]).mean(axis=0)
        return verts.astype(np.float32), faces
    raise ValueError(f'unknown topology: {topo}')


def _compute_geo_dist(verts, faces, sources):
    """Heat method: geodesic distance from each source vertex to all verts.
    sources: list of vertex indices (length J)
    Returns: [J, V] distances
    """
    solver = pp3d.MeshHeatMethodDistanceSolver(verts, faces)
    V = verts.shape[0]
    out = np.zeros((len(sources), V), dtype=np.float32)
    for j, sv in enumerate(sources):
        d = solver.compute_distance(int(sv))
        out[j] = d.astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rig_path', type=str, required=True)
    ap.add_argument('--topologies', nargs='+', default=['ict', 'mf'])
    ap.add_argument('--data_basedir', type=str, default='/data/sihun')
    ap.add_argument('--out_dir', type=str, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or args.rig_path
    rig = load_rig(args.rig_path, device='cpu')
    J = len(rig.joint_names)

    for topo in args.topologies:
        if topo not in rig.bind_pos_dict:
            print(f'[skip] {topo}: bind_pos_dict not loaded')
            continue

        verts, faces = _load_mean_mesh(topo, args.data_basedir)
        maya_bind = rig.bind_pos_dict[topo].cpu().numpy()
        V = verts.shape[0]
        print(f'\n[{topo}] mesh V={V} F={faces.shape[0]}, J={J}')

        # Home vertex per joint = argmin Euclidean dist to maya_bind
        home_verts = np.zeros(J, dtype=np.int32)
        for j in range(J):
            d = np.linalg.norm(verts - maya_bind[j], axis=-1)
            home_verts[j] = int(d.argmin())
        print(f'  home_verts: range=[{home_verts.min()}, {home_verts.max()}]')

        # Heat method per source
        print('  computing geodesic distances (heat method)...')
        geo_dist = _compute_geo_dist(verts, faces, home_verts.tolist())
        print(f'  geo_dist: shape={geo_dist.shape}, '
              f'range=[{geo_dist.min():.4f}, {geo_dist.max():.4f}]')

        # Save
        gp = os.path.join(out_dir, f'geo_dist_{topo}.npy')
        hp = os.path.join(out_dir, f'home_verts_{topo}.npy')
        np.save(gp, geo_dist)
        np.save(hp, home_verts)
        print(f'  saved → {gp}')
        print(f'  saved → {hp}')

        # Sanity report — top 5 joints by mean geo_dist (= "most spread")
        mean_per_j = geo_dist.mean(axis=-1)
        order = np.argsort(-mean_per_j)
        print(f'  top-5 by mean geo_dist (largest spread joints):')
        for j in order[:5]:
            print(f'    [{j:2d}] {rig.joint_names[j]:42s}  mean={mean_per_j[j]:.4f}')


if __name__ == '__main__':
    main()
