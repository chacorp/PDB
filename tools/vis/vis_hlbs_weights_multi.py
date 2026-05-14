"""
vis_hlbs_weights_multi.py — Visualize HLBS skin weights per joint for multiple datasets.

Generates 66 individual joint weight images for ICT, MF, BIWI, COMA.

Usage:
    python vis_hlbs_weights_multi.py \
        --ckpt_dir ckpts_hlbs/2026-04-10-19-15-00-HLBS-FullPred-mf-jTrans-cur50 \
        --epoch 250
"""
import os
import sys
import argparse
import pickle
import numpy as np
import yaml
import torch
import torch.nn.functional as F
import igl
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

from utils.rig_loader import load_rig
from models.hierarchical_lbs import HierarchicalLBS_FullPred
from utils.matplotlib_rnd import normalize_homogeneous, perspective, translate, yrotate, xrotate, calc_face_norm


# ── Rendering ────────────────────────────────────────────────────────────────

def render_joint_weight(verts, faces, W, joint_idx, joint_name, save_path,
                        view_yrots=(180,), view_xrot=180, SIZE=4,
                        cmap='magma', vmin=0, vmax=None):
    """Render a single joint weight heatmap with correct front-face orientation."""
    Wv = W[:, joint_idx]
    Wf = Wv[faces].mean(axis=1)

    if vmax is None:
        vmax = max(float(Wf.max()), 0.01)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_fn = plt.get_cmap(cmap)

    V = normalize_homogeneous(verts)
    view = translate(0, 0, -4.5)
    proj = perspective(55, 1.0, 1.0, 100.0)
    MV = proj @ view

    num_views = len(view_yrots)
    fig = plt.figure(figsize=(SIZE * num_views, SIZE))

    light_dir = np.array([0, 0, 1])

    for vi, add_rot in enumerate(view_yrots):
        # Y rotation for view angle, X rotation 180 to flip upside-down
        model = xrotate(view_xrot) @ yrotate(add_rot)

        V_mu = np.median(V, axis=0)
        V_model = (V - V_mu) @ model.T + V_mu

        V_proj = V_model @ MV.T
        V_proj = V_proj[:, :3] / V_proj[:, 3:4]

        VF = V_proj[faces]
        T = VF[:, :, :2]
        Z = -VF[:, :, 2].mean(1)

        # Back-face culling + shade
        C_norm = calc_face_norm(V_model[:, :3], faces)
        front = C_norm[:, 2] > 0
        T = T[front]
        Z = Z[front]
        C_norm = C_norm[front]
        Wf_front = Wf[front]
        order = np.argsort(Z)
        T_sorted = T[order]
        C_norm = C_norm[order]
        W_sorted = Wf_front[order]

        shade_val = (C_norm @ light_dir)[:, np.newaxis].repeat(3, axis=-1)
        shade_val = np.clip(shade_val, 0, 1) * 0.7 + 0.2

        heatmap_rgb = cmap_fn(norm(W_sorted))[:, :3]
        alpha = np.clip((W_sorted - vmin) / (vmax - vmin + 1e-12), 0, 1)[:, np.newaxis]
        C = shade_val * (1 - alpha) + heatmap_rgb * alpha
        C = np.clip(C, 0, 1)

        ax = fig.add_axes([vi / num_views, 0, 1 / num_views, 1],
                          xlim=[-1, 1], ylim=[-1, 1], aspect=1, frameon=False)
        coll = PolyCollection(T_sorted, closed=True, linewidth=0.1,
                              facecolor=C, edgecolor=C)
        ax.add_collection(coll)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        sm = ScalarMappable(norm=norm, cmap=cmap_fn)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label(f"W[:, {joint_idx}]")

    fig.suptitle(f"j{joint_idx:02d} {joint_name}  (max={Wv.max():.4f})", fontsize=11)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Bind pose viz (predicted joint positions overlaid on mesh) ──────────────

def render_bind_pose(verts, faces, joint_pos, joint_names, save_path,
                     active_idx=None, point_size=18.0):
    """Frontal (XY) + Side (ZY) scatter of predicted joint_pos on mesh wireframe.

    verts      : [N, 3]
    faces      : [F, 3]
    joint_pos  : [J, 3]   predicted bind pose from model
    joint_names: list[str]
    active_idx : optional list of face joint indices (other joints rendered fainter)
    """
    fig = plt.figure(figsize=(8, 4))

    # Mesh wireframe (light gray) + joint scatter
    for ax_i, axis_x, axis_y, xlabel, ylabel, title in [
        (1, 0, 1, 'x', 'y', 'Frontal (XY)'),
        (2, 2, 1, 'z', 'y', 'Side (ZY)'),
    ]:
        ax = fig.add_subplot(1, 2, ax_i)
        polys = verts[faces][:, :, [axis_x, axis_y]]
        pc = PolyCollection(polys, facecolors='#dddddd', edgecolors='#bbbbbb',
                            linewidths=0.08, alpha=0.35, zorder=1)
        ax.add_collection(pc)

        # Active vs non-active joints
        J = joint_pos.shape[0]
        active_mask = np.zeros(J, dtype=bool)
        if active_idx is not None and len(active_idx) > 0:
            active_mask[np.asarray(active_idx, dtype=int)] = True
        else:
            active_mask[:] = True

        non_act = ~active_mask
        ax.scatter(joint_pos[non_act, axis_x], joint_pos[non_act, axis_y],
                   s=point_size * 0.5, c='#9ca3af', alpha=0.5,
                   edgecolors='k', linewidths=0.2, zorder=5)
        ax.scatter(joint_pos[active_mask, axis_x], joint_pos[active_mask, axis_y],
                   s=point_size, c='#e63946', alpha=0.95,
                   edgecolors='k', linewidths=0.3, zorder=10)

        ax.set_aspect('equal'); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ── Template loaders ─────────────────────────────────────────────────────────

def load_template(dataset, identity_idx=0, data_basedir='/data/inyup'):
    """Load neutral template verts, faces, id_name for a given dataset."""
    if dataset == 'ict':
        from utils.remesh_utils import ICT_face_model
        ict_model = ICT_face_model()
        _test_pt = 'ict_face_pt/ict_id_vecs_test.pt'
        iden_vecs = torch.load(_test_pt, weights_only=False).numpy()
        id_coeff = iden_vecs[identity_idx]
        id_disps = ict_model.get_id_disp(id_coeff).squeeze()
        verts = (ict_model.neutral_verts + id_disps).astype(np.float32)
        faces = ict_model.faces.astype(np.int32)
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


# ── NFS feature ──────────────────────────────────────────────────────────────

def load_nfs_feat_cached(id_name, feat_dir='nfs_features_seg', device='cuda:0'):
    """Load pre-computed NFS seg feature from cache."""
    fp = os.path.join(feat_dir, f'{id_name}_nfs_feat.npy')
    if os.path.exists(fp):
        return torch.tensor(np.load(fp), dtype=torch.float32).to(device)
    return None


_nfs_model_cache = None

def _load_nfs_model(nfs_ckpt='ckpts_comparison/NFS-best', device='cpu'):
    """Lazy-load NFS model (same approach as eval_hlbs)."""
    global _nfs_model_cache
    if _nfs_model_cache is not None:
        return _nfs_model_cache

    from models.NFS import NFS

    dev = torch.device(device)
    nfs_opts_path = os.path.join(nfs_ckpt, 'train_opts.yml')
    with open(nfs_opts_path) as f:
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
    nfs_model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=False),
                              strict=False)
    nfs_model.eval()
    print(f"[NFS] Loaded: {ckpt_path}")

    _nfs_model_cache = nfs_model
    return nfs_model


def extract_nfs_feat_online(verts_np, faces_np, nfs_ckpt='ckpts_comparison/NFS-best', device='cpu'):
    """Extract seg feat online for unseen mesh (same as eval_hlbs._extract_seg_feat_online)."""
    from utils.nfr_utils import get_dfn_info

    dev = torch.device(device)
    nfs_model = _load_nfs_model(nfs_ckpt, device)

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

    return x[0]  # [V, 256]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--epoch", type=str, default="250",
                        help="Epoch number or 'best'")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--view_yrots", type=str, default="0,45,-45",
                        help="Comma-separated Y rotation angles")
    parser.add_argument("--view_xrot", type=float, default=0,
                        help="X rotation (0 = normal orientation)")
    parser.add_argument("--datasets", type=str, default='ict:0,mf:12,biwi:1,coma:6',
                        help='Comma-separated dataset:idx pairs '
                             '(e.g. ict:2,mf:12). Overrides default dataset_configs.')
    parser.add_argument("--save_bind_pose", action='store_true',
                        help='Also render predicted bind pose (joint_pos) overlaid on mesh.')
    args = parser.parse_args()

    device = torch.device(args.device)
    view_yrots = tuple(float(x) for x in args.view_yrots.split(','))

    # Load train opts
    opts_path = os.path.join(args.ckpt_dir, 'train_opts.yml')
    with open(opts_path) as f:
        opts = yaml.safe_load(f)
    print(f"Loaded opts from {opts_path}")

    # Load model — replicate training-time options EXACTLY so eval path matches.
    rig = load_rig(opts['rig_path'])

    # Peek at checkpoint state_dict to recover the EXACT face_joint_idx the
    # checkpoint was trained with (avoids size mismatch when active_joints_json
    # has been modified since training).
    _ckpt_path_peek = (os.path.join(args.ckpt_dir, 'model_hlbs_best.pth')
                       if args.epoch == 'best'
                       else os.path.join(args.ckpt_dir, f'model_hlbs_{int(args.epoch):03d}.pth'))
    _sd_peek = torch.load(_ckpt_path_peek, map_location='cpu', weights_only=False)
    _face_idx_ckpt = (_sd_peek['face_joint_idx'].tolist()
                      if 'face_joint_idx' in _sd_peek else None)

    # Resolve face_mask config: prefer ckpt buffer; fall back to active_joints_json.
    _face_idx, _base_idx = None, None
    _aj_path = opts.get('active_joints_json')
    if _aj_path and os.path.exists(_aj_path):
        with open(_aj_path) as f:
            _aj = yaml.safe_load(f) if _aj_path.endswith(('.yml', '.yaml')) else __import__('json').load(f)
        _face_idx = _aj['face_joint_idx']
        _base_idx = _aj.get('base_joint_idx')
    if _face_idx_ckpt is not None:
        _face_idx = _face_idx_ckpt
        print(f"[vis] using face_joint_idx from ckpt ({len(_face_idx)} joints) — "
              f"avoids mismatch with current {_aj_path}")

    # Resolve sigma_targets (per-joint init if set in training)
    _sig = None
    if opts.get('sigma_targets_npy') and os.path.exists(opts['sigma_targets_npy']):
        _sig = np.load(opts['sigma_targets_npy']).astype(np.float32)

    # Resolve anchor_pool assets (if used in training)
    _anc = _off = None
    if opts.get('joint_anchors_npy') and os.path.exists(opts['joint_anchors_npy']):
        _anc = np.load(opts['joint_anchors_npy']).astype(np.float32)
    if opts.get('joint_offsets_npy') and os.path.exists(opts['joint_offsets_npy']):
        _off = np.load(opts['joint_offsets_npy']).astype(np.float32)

    model = HierarchicalLBS_FullPred(
        rig=rig,
        in_dim_exp=12,
        hid_dim=opts.get('hid_dim', 128),
        num_layers=opts.get('num_layers', 4),
        device=str(device),
        use_joint_trans=opts.get('use_joint_trans', False),
        dfn_skin=opts.get('dfn_skin', False),
        dfn_bind=opts.get('dfn_bind', False),
        dfn_exp=opts.get('dfn_exp', False),
        nfs_feat_dim=256 if opts.get('nfs_feat_dir') else 0,
        nfs_concat=opts.get('nfs_concat', False),
        adain_pos_norm=opts.get('adain_pos_norm', False),
        freeze_bind_pose=opts.get('freeze_bind_pose', False),
        use_gmm_hybrid=opts.get('use_gmm_hybrid', False),
        init_log_sigma=opts.get('init_log_sigma', -1.2),
        gmm_mode=opts.get('gmm_mode', 'additive'),
        sigma_targets=_sig,
        face_joint_idx=_face_idx,
        base_joint_idx=_base_idx,
        face_mask_r0=opts.get('face_mask_r0', 1.0),
        face_mask_r1=opts.get('face_mask_r1', 2.25),
        bind_pose_mode=opts.get('bind_pose_mode', 'net'),
        joint_anchors=_anc,
        joint_offsets=_off,
        attn_temperature_init=opts.get('attn_temperature_init', 0.1),
    ).to(device)

    ckpt_path = os.path.join(args.ckpt_dir, f'model_hlbs_{args.epoch}.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  use_gmm_hybrid: {opts.get('use_gmm_hybrid', False)}")
    if opts.get('use_gmm_hybrid', False):
        sigma_vals = torch.exp(model.log_sigma).detach().cpu().numpy()
        print(f"  σ range: [{sigma_vals.min():.4f}, {sigma_vals.max():.4f}], "
              f"mean={sigma_vals.mean():.4f}")

    J = model.num_joints
    joint_names = model.joint_names
    nfs_feat_dir = opts.get('nfs_feat_dir', None)

    # Dataset configs: parsed from --datasets flag (dataset_key:idx[, ...])
    dataset_configs = []
    for pair in args.datasets.split(','):
        pair = pair.strip()
        if not pair: continue
        ds_key, _id_idx = pair.split(':')
        dataset_configs.append((ds_key.strip(), int(_id_idx),
                                ds_key.strip().upper() + f'_id{_id_idx}'))
    print(f"Dataset configs: {dataset_configs}")

    base_out_dir = os.path.join(args.ckpt_dir, f'weight_vis_{args.epoch}_individual')
    os.makedirs(base_out_dir, exist_ok=True)

    for ds_key, id_idx, ds_name in dataset_configs:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (identity idx={id_idx})")
        print(f"{'='*60}")

        # Load template
        verts, faces, id_name = load_template(ds_key, id_idx)
        normals = igl.per_vertex_normals(verts, faces).astype(np.float32)
        print(f"  Template: {id_name}, V={verts.shape[0]}, F={faces.shape[0]}")

        src_v = torch.tensor(verts, dtype=torch.float32, device=device).unsqueeze(0)
        src_n = torch.tensor(normals, dtype=torch.float32, device=device).unsqueeze(0)

        # NFS features
        nfs_feat = None
        if nfs_feat_dir:
            nfs_feat = load_nfs_feat_cached(id_name, nfs_feat_dir, device=str(device))
            if nfs_feat is None:
                print(f"  NFS feat not cached, extracting online for {id_name}...")
                nfs_feat = extract_nfs_feat_online(verts, faces, device=str(device))
            if nfs_feat is not None:
                nfs_feat = nfs_feat.unsqueeze(0)  # [1, V, 256]
                print(f"  NFS feat: {nfs_feat.shape}")
            else:
                print(f"  [WARN] No NFS feat available for {id_name}")

        # Get predicted weights
        with torch.no_grad():
            skin_input, _adain = model._prepare_feat(src_v, src_n, nfs_feat)
            # Bind pose dispatch: freeze / anchor_pool / net (matches model.forward)
            if model.freeze_bind_pose:
                B_size = src_v.shape[0]
                joint_pos = model.bind_pos_target.unsqueeze(0).expand(B_size, -1, -1)
            elif getattr(model, 'bind_pose_mode', 'net') == 'anchor_pool':
                _, joint_pos = model._get_bind_pose_anchor(src_v, nfs_feat)
            else:
                _, joint_pos = model._get_bind_pose(skin_input, adain_input=_adain)
            W_pred, logit_W = model._get_skinning_weights(
                skin_input, adain_input=_adain,
                source_vert=src_v, joint_pos=joint_pos,
            )

        W_np = W_pred[0].cpu().numpy()  # [N, J]
        print(f"  W range: [{W_np.min():.6f}, {W_np.max():.6f}]")

        # Output directory
        out_dir = os.path.join(base_out_dir, ds_name)
        os.makedirs(out_dir, exist_ok=True)

        # Render 66 individual joint images
        for j in range(J):
            name = joint_names[j] if j < len(joint_names) else f'joint_{j}'
            save_path = os.path.join(out_dir, f'W_{j:02d}_{name}.png')
            j_vmax = max(float(W_np[:, j].max()), 0.01)

            render_joint_weight(
                verts, faces, W_np, j, name, save_path,
                view_yrots=view_yrots,
                view_xrot=args.view_xrot,
                vmax=j_vmax,
            )
            if (j + 1) % 10 == 0 or j == J - 1:
                print(f"  [{ds_name}] Rendered {j+1}/{J} joints")

        print(f"  Saved {J} images to: {out_dir}")

    print(f"\nAll done. Output: {base_out_dir}")


if __name__ == '__main__':
    main()
