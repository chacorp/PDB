"""
extract_nfs_feat.py — Extract and cache NFS seg encoder features for all datasets.

Requires GPU (pytorch3d rasterizer).

Usage:
    python extract_nfs_feat.py --device cuda:0
"""
import os
import sys
import argparse
import pickle
import numpy as np
import torch
import yaml
import igl
import trimesh

sys.path.insert(0, os.path.dirname(__file__))


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


def extract_and_save(verts_np, faces_np, id_name, nfs_model, device, out_dir):
    """Extract seg feat and save to cache."""
    from utils.nfr_utils import get_dfn_info

    out_path = os.path.join(out_dir, f'{id_name}_nfs_feat.npy')
    if os.path.exists(out_path):
        print(f"  [skip] {out_path} already exists")
        return

    dev = torch.device(device)
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

    feat = x[0].cpu().numpy()  # [V, 256]
    np.save(out_path, feat)
    print(f"  Saved: {out_path} ({feat.shape})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--nfs_ckpt", type=str, default="ckpts_comparison/NFS-best")
    parser.add_argument("--out_dir", type=str, default="nfs_features_seg")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)

    # Load NFS model
    from models.NFS import NFS
    with open(os.path.join(args.nfs_ckpt, 'train_opts.yml')) as f:
        nfs_opts = yaml.safe_load(f)

    class _Opts:
        pass
    opts = _Opts()
    for k, v in nfs_opts.items():
        setattr(opts, k, v)
    opts.device = str(dev)
    opts.is_train = False

    nfs_model = NFS(opts=opts).to(dev)
    ckpt_path = os.path.join(args.nfs_ckpt, 'model_best.pth')
    nfs_model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=False), strict=False)
    nfs_model.eval()
    print(f"Loaded NFS model: {ckpt_path}")

    # Datasets to extract
    datasets = [
        # BIWI: all 14 identities
        *[('biwi', i) for i in range(14)],
        # COMA/VOCA: all 12 identities
        *[('coma', i) for i in range(12)],
    ]

    for ds_key, id_idx in datasets:
        verts, faces, id_name = load_template(ds_key, id_idx)
        print(f"\n{ds_key} [{id_idx}] {id_name}: V={verts.shape[0]}")
        extract_and_save(verts, faces, id_name, nfs_model, args.device, args.out_dir)

    print(f"\nDone. Features saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
