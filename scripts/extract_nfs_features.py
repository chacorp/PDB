"""
Extract NFS id_encoder per-vertex features for all training identities.

For each identity: neutral mesh → NFS img_encoder + id_encoder → [V, 256] feature
Saves as {id_name}_nfs_feat.npy for use as skin_weight_net input.

Usage:
    python scripts/extract_nfs_features.py \
        --nfs_ckpt ckpts_comparison/NFS-best \
        --out_dir nfs_features
"""
import os, sys, argparse, pickle
import numpy as np
import torch
import trimesh
import igl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def extract_per_vertex_feature(model, mesh, dfn_info, device):
    """Run NFS id_encoder but return per-vertex feature [V, C_width] before last_lin."""
    # Render image
    img = model.renderer.render_img(mesh).float().to(device)
    img_feat = model.get_img_feat(img)

    # Get local feature (position + normal + img_feat) [1, V, 134]
    verts_t = torch.tensor(mesh.vertices, dtype=torch.float32, device=device).unsqueeze(0)
    faces_t = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    vert_feat = model.get_local_feature(verts_t, faces_t, img_feat, at='verts').float()

    # Forward through id_encoder's DiffusionNet, but stop before last_lin
    encoder = model.mesh_id_encoder
    encoder.update_precomputes(dfn_info)

    # Reconstruct sparse tensors
    L = torch.sparse_coo_tensor(encoder.L_ind, encoder.L_val, encoder.L_size, device=device)
    batch_size = vert_feat.shape[0]
    batch_L = [L for _ in range(batch_size)]
    batch_mass = encoder.mass.unsqueeze(0).expand(batch_size, -1)
    batch_evals = encoder.evals.unsqueeze(0).expand(batch_size, -1)
    batch_evecs = encoder.evecs.unsqueeze(0).expand(batch_size, -1, -1)
    gradX = [torch.sparse_coo_tensor(encoder.grad_X_ind, encoder.grad_X_val,
             encoder.grad_X_size, device=device) for _ in range(batch_size)]
    gradY = [torch.sparse_coo_tensor(encoder.grad_Y_ind, encoder.grad_Y_val,
             encoder.grad_Y_size, device=device) for _ in range(batch_size)]

    # Manual forward: first_lin → blocks → STOP (no last_lin, no pooling)
    dfn = encoder.dfn
    x = dfn.first_lin(vert_feat)
    for block in dfn.blocks:
        x = block(x, batch_mass, L=batch_L, evals=batch_evals,
                  evecs=batch_evecs, gradX=gradX, gradY=gradY)

    # x is [1, V, C_width=256] — per-vertex feature before last_lin
    return x[0].cpu().numpy()  # [V, 256]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfs_ckpt", type=str, default="ckpts_comparison/NFS-best")
    parser.add_argument("--out_dir", type=str, default="nfs_features")
    parser.add_argument("--data_basedir", type=str, default="/data/sihun")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load NFS model ──
    import yaml
    from models.NFS import NFS

    nfs_opts_path = os.path.join(args.nfs_ckpt, "train_opts.yml")
    with open(nfs_opts_path) as f:
        nfs_opts = yaml.safe_load(f)

    # Create opts namespace
    class Opts:
        pass
    opts = Opts()
    for k, v in nfs_opts.items():
        setattr(opts, k, v)
    opts.device = str(device)
    opts.is_train = False

    model = NFS(opts=opts).to(device)
    ckpt_path = os.path.join(args.nfs_ckpt, "model_best.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False), strict=False)
    model.eval()
    print(f"Loaded NFS: {ckpt_path}")

    from utils.nfr_utils import get_dfn_info

    # ── Process MF identities ──
    mf_template_path = f"{args.data_basedir}/multiface_align/mf_templates.pkl"
    if not os.path.exists(mf_template_path):
        mf_template_path = f"{args.data_basedir}/pca/multiface_align/mf_templates.pkl"
    with open(mf_template_path, 'rb') as f:
        mf_templates = pickle.load(f)

    mf_faces = np.array(mf_templates['face'], dtype=np.int32)
    mf_ids = [k for k in mf_templates if k != 'face']

    print(f"\nProcessing {len(mf_ids)} MF identities...")
    for id_name in mf_ids:
        verts = np.array(mf_templates[id_name], dtype=np.float32)
        mesh = trimesh.Trimesh(vertices=verts, faces=mf_faces, process=False)

        with torch.no_grad():
            dfn_info = get_dfn_info(mesh, map_location=device)
            feat = extract_per_vertex_feature(model, mesh, dfn_info, device)

        out_path = os.path.join(args.out_dir, f"{id_name}_nfs_feat.npy")
        np.save(out_path, feat.astype(np.float32))
        print(f"  {id_name}: {feat.shape} → {out_path}")

    # ── Process ICT identities ──
    print(f"\nProcessing ICT identities...")
    from utils.remesh_utils import ICT_face_model
    ict_model = ICT_face_model()

    # Load identity vectors
    ict_id_path = 'data/ICT_live_100/iden_vecs.npy'
    if os.path.exists(ict_id_path):
        iden_vecs = np.load(ict_id_path)
    else:
        iden_vecs = np.zeros((1, 100))

    ict_faces = ict_model.faces.astype(np.int32)

    for i in range(len(iden_vecs)):
        id_coeff = iden_vecs[i]
        id_disps = ict_model.get_id_disp(id_coeff).squeeze()
        verts = (ict_model.neutral_verts + id_disps).astype(np.float32)
        mesh = trimesh.Trimesh(vertices=verts, faces=ict_faces, process=False)

        with torch.no_grad():
            dfn_info = get_dfn_info(mesh, map_location=device)
            feat = extract_per_vertex_feature(model, mesh, dfn_info, device)

        out_path = os.path.join(args.out_dir, f"ict_{i:03d}_nfs_feat.npy")
        np.save(out_path, feat.astype(np.float32))
        print(f"  ict_{i:03d}: {feat.shape} → {out_path}")

    print(f"\nDone. All features saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
