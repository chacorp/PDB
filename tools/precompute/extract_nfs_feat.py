"""
extract_nfs_feat.py — Extract and cache NFS seg/id encoder features for all datasets.

Requires GPU (pytorch3d rasterizer).

Usage:
    # All datasets (default)
    python extract_nfs_feat.py --device cuda:0

    # Specific datasets only
    python extract_nfs_feat.py --datasets biwi coma
    python extract_nfs_feat.py --datasets mf ict

    # Use id encoder instead of seg encoder
    python extract_nfs_feat.py --encoder id
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))


def load_templates_biwi():
    with open('utils/templates/biwi_templates.pkl', 'rb') as f:
        templates = pickle.load(f)
    faces = np.array(templates['face'], dtype=np.int32)
    id_names = [k for k in templates if k != 'face']
    return [(np.array(templates[n], dtype=np.float32), faces, n) for n in id_names]


def load_templates_coma():
    with open('utils/templates/voca_templates.pkl', 'rb') as f:
        templates = pickle.load(f)
    faces = np.array(templates['face'], dtype=np.int32)
    id_names = [k for k in templates if k != 'face']
    return [(np.array(templates[n], dtype=np.float32), faces, n) for n in id_names]


def load_templates_mf(data_basedir):
    mf_path = f"{data_basedir}/multiface_align/mf_templates.pkl"
    if not os.path.exists(mf_path):
        mf_path = f"{data_basedir}/pca/multiface_align/mf_templates.pkl"
    with open(mf_path, 'rb') as f:
        templates = pickle.load(f)
    faces = np.array(templates['face'], dtype=np.int32)
    id_names = [k for k in templates if k != 'face']
    return [(np.array(templates[n], dtype=np.float32), faces, n) for n in id_names]


def load_templates_ict(n_ids=111):
    """ICT templates from ict_face_pt/random_identity_vecs.npy — the training
    identity source (train_hlbs.py train mode + precompute_per_id_bind_pos_v2.py).
    random_identity_vecs[0:100] == data/ICT_live_100/iden_vecs.npy, plus 11 more
    (ids 100-110). `ict_NNN` naming matches the per-id GT cache.
    """
    from utils.remesh_utils import ICT_face_model
    ict = ICT_face_model()
    ict_faces = ict.faces.astype(np.int32)

    ict_id_path = 'ict_face_pt/random_identity_vecs.npy'
    if os.path.exists(ict_id_path):
        iden_vecs = np.load(ict_id_path)[:n_ids]
    else:
        iden_vecs = np.zeros((1, 100))

    results = []
    for i in range(len(iden_vecs)):
        id_disps = ict.get_id_disp(iden_vecs[i]).squeeze()
        verts = (ict.neutral_verts + id_disps).astype(np.float32)
        results.append((verts, ict_faces, f'ict_{i:03d}'))
    return results


def extract_per_vertex_feature(model, mesh, dfn_info, device, encoder_type='seg'):
    """Run NFS encoder and return per-vertex feature [V, C_width] before last_lin."""
    img = model.renderer.render_img(mesh).float().to(device)
    img_feat = model.get_img_feat(img)

    verts_t = torch.tensor(mesh.vertices, dtype=torch.float32, device=device).unsqueeze(0)
    faces_t = torch.tensor(mesh.faces, dtype=torch.long, device=device)
    vert_feat = model.get_local_feature(verts_t, faces_t, img_feat, at='verts').float()

    encoder = model.mesh_seg_encoder if encoder_type == 'seg' else model.mesh_id_encoder
    encoder.update_precomputes(dfn_info)
    dfn = encoder.dfn

    L = torch.sparse_coo_tensor(encoder.L_ind, encoder.L_val, encoder.L_size, device=device)
    batch_mass = encoder.mass.unsqueeze(0)
    batch_evals = encoder.evals.unsqueeze(0)
    batch_evecs = encoder.evecs.unsqueeze(0)
    gradX = [torch.sparse_coo_tensor(encoder.grad_X_ind, encoder.grad_X_val,
             encoder.grad_X_size, device=device)]
    gradY = [torch.sparse_coo_tensor(encoder.grad_Y_ind, encoder.grad_Y_val,
             encoder.grad_Y_size, device=device)]

    x = dfn.first_lin(vert_feat)
    for block in dfn.blocks:
        x = block(x, batch_mass, L=[L], evals=batch_evals,
                  evecs=batch_evecs, gradX=gradX, gradY=gradY)

    return x[0].cpu().numpy()  # [V, 256]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--nfs_ckpt", type=str, default="ckpts_comparison/NFS-best")
    parser.add_argument("--out_dir", type=str, default="nfs_features_seg")
    parser.add_argument("--data_basedir", type=str, default="/data/sihun")
    parser.add_argument("--encoder", type=str, default="seg", choices=["id", "seg"],
                        help="Which NFS encoder to extract features from (default: seg)")
    parser.add_argument("--datasets", nargs='+', default=['biwi', 'coma', 'mf', 'ict'],
                        choices=['biwi', 'coma', 'mf', 'ict'],
                        help="Datasets to process (default: all)")
    parser.add_argument("--n_ict_ids", type=int, default=111,
                        help="ICT ids sliced from random_identity_vecs.npy "
                             "(train_hlbs.py train mode uses [:111]).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load NFS model
    from models.NFS import NFS
    with open(os.path.join(args.nfs_ckpt, 'train_opts.yml')) as f:
        nfs_opts = yaml.safe_load(f)

    class _Opts:
        pass
    opts = _Opts()
    for k, v in nfs_opts.items():
        setattr(opts, k, v)
    opts.device = str(device)
    opts.is_train = False

    model = NFS(opts=opts).to(device)
    ckpt_path = os.path.join(args.nfs_ckpt, 'model_best.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False), strict=False)
    model.eval()
    print(f"Loaded NFS model: {ckpt_path}")
    print(f"Encoder: {args.encoder}, Datasets: {args.datasets}")

    from utils.nfr_utils import get_dfn_info

    loaders = {
        'biwi': load_templates_biwi,
        'coma': load_templates_coma,
        'mf': lambda: load_templates_mf(args.data_basedir),
        'ict': lambda: load_templates_ict(args.n_ict_ids),
    }

    for ds in args.datasets:
        entries = loaders[ds]()
        print(f"\n── {ds.upper()} ({len(entries)} identities) ──")

        for verts, faces, id_name in entries:
            out_path = os.path.join(args.out_dir, f'{id_name}_nfs_feat.npy')
            if os.path.exists(out_path):
                print(f"  [skip] {out_path}")
                continue

            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            with torch.no_grad():
                dfn_info = get_dfn_info(mesh, map_location=device)
                feat = extract_per_vertex_feature(model, mesh, dfn_info, device, encoder_type=args.encoder)

            np.save(out_path, feat.astype(np.float32))
            print(f"  {id_name}: {feat.shape} -> {out_path}")

    print(f"\nDone. Features saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
