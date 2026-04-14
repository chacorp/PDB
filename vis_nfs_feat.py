"""
vis_nfs_feat.py — Visualize NFS seg encoder features across topologies.

PCA-reduce [V, 256] features to 3D, map to RGB.
All topologies are PCA-fit together so colors are cross-comparable.

Usage:
    python vis_nfs_feat.py --out_dir nfs_feat_vis
"""
import os
import sys
import argparse
import pickle
import numpy as np
import torch
import igl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from utils.matplotlib_rnd import normalize_homogeneous, perspective, translate, yrotate, xrotate, calc_face_norm


# ── Rendering ────────────────────────────────────────────────────────────────

def render_mesh_rgb(verts, faces, vert_colors, save_path, title='',
                    view_yrots=(0, 45, -45), SIZE=4):
    """Render mesh with per-vertex RGB colors."""
    # Per-face color = mean of vertex colors
    Cf = vert_colors[faces].mean(axis=1)  # [F, 3]

    V = normalize_homogeneous(verts)
    view = translate(0, 0, -4.5)
    proj = perspective(55, 1.0, 1.0, 100.0)
    MV = proj @ view
    light_dir = np.array([0, 0, 1])

    num_views = len(view_yrots)
    fig = plt.figure(figsize=(SIZE * num_views, SIZE))

    for vi, add_rot in enumerate(view_yrots):
        model = yrotate(add_rot)
        V_mu = np.median(V, axis=0)
        V_model = (V - V_mu) @ model.T + V_mu

        V_proj = V_model @ MV.T
        V_proj = V_proj[:, :3] / V_proj[:, 3:4]

        VF = V_proj[faces]
        T = VF[:, :, :2]
        Z = -VF[:, :, 2].mean(1)

        C_norm = calc_face_norm(V_model[:, :3], faces)
        front = C_norm[:, 2] > 0
        T = T[front]
        Z = Z[front]
        C_norm = C_norm[front]
        Cf_front = Cf[front]
        order = np.argsort(Z)
        T_sorted = T[order]
        C_norm_sorted = C_norm[order]
        Cf_sorted = Cf_front[order]

        # Shade: blend RGB with lighting
        shade_val = (C_norm_sorted @ light_dir)[:, np.newaxis].repeat(3, axis=-1)
        shade_val = np.clip(shade_val, 0, 1) * 0.4 + 0.6
        C = np.clip(Cf_sorted * shade_val, 0, 1)

        ax = fig.add_axes([vi / num_views, 0, 1 / num_views, 1],
                          xlim=[-1, 1], ylim=[-1, 1], aspect=1, frameon=False)
        coll = PolyCollection(T_sorted, closed=True, linewidth=0.1,
                              facecolor=C, edgecolor=C)
        ax.add_collection(coll)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    if title:
        fig.suptitle(title, fontsize=12)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Template loaders ─────────────────────────────────────────────────────────

def load_template(dataset, identity_idx=0):
    if dataset == 'ict':
        from utils.remesh_utils import ICT_face_model
        ict = ICT_face_model()
        iden_vecs = torch.load('ict_face_pt/ict_id_vecs_test.pt', weights_only=False).numpy()
        id_disps = ict.get_id_disp(iden_vecs[identity_idx]).squeeze()
        verts = (ict.neutral_verts + id_disps).astype(np.float32)
        faces = ict.faces.astype(np.int32)
        return verts, faces, f'ict_{identity_idx:03d}'

    local_pkl_map = {
        'mf': 'utils/templates/mf_templates.pkl',
        'biwi': 'utils/templates/biwi_templates.pkl',
        'coma': 'utils/templates/voca_templates.pkl',
    }
    with open(local_pkl_map[dataset], 'rb') as f:
        templates = pickle.load(f)
    faces = np.array(templates['face'], dtype=np.int32)
    id_names = [k for k in templates if k != 'face']
    id_name = id_names[identity_idx]
    verts = np.array(templates[id_name], dtype=np.float32)
    return verts, faces, id_name


# ── NFS feature loading ─────────────────────────────────────────────────────

def load_nfs_feat(id_name, feat_dir='nfs_features_seg'):
    fp = os.path.join(feat_dir, f'{id_name}_nfs_feat.npy')
    if os.path.exists(fp):
        return np.load(fp)  # [V, 256]
    return None


def extract_nfs_feat_online(verts_np, faces_np, nfs_ckpt='ckpts_comparison/NFS-best', device='cpu'):
    """Extract seg feat online for mesh without cached features."""
    import yaml
    import trimesh
    from models.NFS import NFS
    from utils.nfr_utils import get_dfn_info

    dev = torch.device(device)
    with open(os.path.join(nfs_ckpt, 'train_opts.yml')) as f:
        nfs_opts = yaml.safe_load(f)

    class _Opts:
        pass
    opts = _Opts()
    for k, v in nfs_opts.items():
        setattr(opts, k, v)
    opts.device = str(dev)
    opts.is_train = False

    nfs_model = NFS(opts=opts).to(dev)
    ckpt_path = os.path.join(nfs_ckpt, 'model_best.pth')
    nfs_model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=False), strict=False)
    nfs_model.eval()

    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    dfn_info = get_dfn_info(mesh, map_location=dev)

    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=dev).unsqueeze(0)
    normals = igl.per_vertex_normals(verts_np, faces_np).astype(np.float32)
    norms_t = torch.tensor(normals, dtype=torch.float32, device=dev).unsqueeze(0)

    img = nfs_model.renderer.render_img(mesh).float().to(dev)
    img_feat = nfs_model.get_img_feat(img).squeeze()
    img_feat_exp = img_feat.unsqueeze(0).unsqueeze(0).expand(1, verts_np.shape[0], -1)
    vert_feat = torch.cat([verts_t, norms_t, img_feat_exp], dim=-1)

    encoder = nfs_model.mesh_seg_encoder
    encoder.update_precomputes(dfn_info)
    dfn = encoder.dfn

    L = torch.sparse_coo_tensor(encoder.L_ind, encoder.L_val, encoder.L_size, device=dev)
    batch_mass = encoder.mass.unsqueeze(0)
    batch_evals = encoder.evals.unsqueeze(0)
    batch_evecs = encoder.evecs.unsqueeze(0)
    gradX = [torch.sparse_coo_tensor(encoder.grad_X_ind, encoder.grad_X_val, encoder.grad_X_size, device=dev)]
    gradY = [torch.sparse_coo_tensor(encoder.grad_Y_ind, encoder.grad_Y_val, encoder.grad_Y_size, device=dev)]

    with torch.no_grad():
        x = dfn.first_lin(vert_feat)
        for block in dfn.blocks:
            x = block(x, batch_mass, L=[L], evals=batch_evals, evecs=batch_evecs, gradX=gradX, gradY=gradY)

    return x[0].cpu().numpy()  # [V, 256]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="nfs_feat_vis")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_components", type=int, default=3,
                        help="PCA components (3 for RGB)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Dataset configs: (key, identity_idx, display_name)
    dataset_configs = [
        ('ict',  0,  'ICT_id0'),
        ('ict',  5,  'ICT_id5'),
        ('mf',   0,  'MF_id0'),
        ('mf',   12, 'MF_id12'),
        ('biwi', 1,  'BIWI_F2'),
        ('biwi', 10, 'BIWI_M3'),
        ('coma', 6,  'COMA_id6'),
        ('coma', 8,  'COMA_id8'),
    ]

    # ── Load features for all datasets ───────────────────────────────────
    all_data = []
    for ds_key, id_idx, ds_name in dataset_configs:
        print(f"\n{'='*50}")
        print(f"Loading {ds_name} (id_idx={id_idx})")

        verts, faces, id_name = load_template(ds_key, id_idx)
        print(f"  Template: {id_name}, V={verts.shape[0]}, F={faces.shape[0]}")

        # Load or extract NFS features
        feat = load_nfs_feat(id_name)
        if feat is None:
            print(f"  Cached feature not found, extracting online...")
            feat = extract_nfs_feat_online(verts, faces, device=args.device)
        print(f"  Feature shape: {feat.shape}")

        all_data.append({
            'key': ds_key, 'name': ds_name, 'id_name': id_name,
            'verts': verts, 'faces': faces, 'feat': feat,
        })

    # ── Joint PCA across all topologies ──────────────────────────────────
    print(f"\n{'='*50}")
    print("Fitting joint PCA across all topologies...")
    all_feats = np.concatenate([d['feat'] for d in all_data], axis=0)
    print(f"  Total vertices: {all_feats.shape[0]}, dim: {all_feats.shape[1]}")

    pca = PCA(n_components=args.n_components)
    all_pca = pca.fit_transform(all_feats)  # [total_V, 3]
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")

    # PCA → RGB with per-channel histogram equalization for max contrast
    def hist_equalize(x, bins=256):
        """Per-channel histogram equalization to [0, 1]."""
        out = np.zeros_like(x)
        for c in range(x.shape[1]):
            col = x[:, c]
            hist, bin_edges = np.histogram(col, bins=bins)
            cdf = hist.cumsum().astype(np.float64)
            cdf = cdf / cdf[-1]
            out[:, c] = np.interp(col, bin_edges[:-1], cdf)
        return out

    all_pca_norm = hist_equalize(all_pca)

    # ── Split back and render ────────────────────────────────────────────
    offset = 0
    for d in all_data:
        N = d['feat'].shape[0]
        rgb = all_pca_norm[offset:offset + N]  # [N, 3]
        offset += N

        save_path = os.path.join(args.out_dir, f"nfs_feat_pca_{d['name']}.png")
        render_mesh_rgb(
            d['verts'], d['faces'], rgb, save_path,
            title=f"{d['name']} ({d['id_name']}, V={N})",
        )
        print(f"  Saved: {save_path}")

    # ── Combined comparison image ────────────────────────────────────────
    fig, axes = plt.subplots(1, len(all_data), figsize=(5 * len(all_data), 5))
    for i, d in enumerate(all_data):
        img = plt.imread(os.path.join(args.out_dir, f"nfs_feat_pca_{d['name']}.png"))
        axes[i].imshow(img)
        axes[i].set_title(f"{d['name']} (V={d['feat'].shape[0]})", fontsize=11)
        axes[i].axis('off')
    fig.suptitle("NFS Seg Feature — Joint PCA across topologies", fontsize=14)
    combined_path = os.path.join(args.out_dir, "nfs_feat_pca_comparison.png")
    fig.savefig(combined_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Combined: {combined_path}")
    print(f"\nAll done. Output: {args.out_dir}")


if __name__ == '__main__':
    main()
