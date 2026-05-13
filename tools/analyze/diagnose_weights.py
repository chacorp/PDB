"""
diagnose_weights.py — Diagnose why skin weight goes to wrong regions.

For selected (dataset, joint) pairs, renders 4-panel diagnostic:
  [1] joint_pos (μ) marked on mesh
  [2] softmax(logit_net)[:, j]   — network-only prediction
  [3] softmax(logit_gauss)[:, j] — Gaussian prior around μ
  [4] W_final[:, j] = softmax(net+gauss)

Also prints learned log_sigma per joint.

Usage:
    python diagnose_weights.py \
        --ckpt_dir ckpts_hlbs/2026-04-22-10-13-53-HLBS-FullPred-ict-jTrans-nrm0.1-Wsm0.01 \
        --epoch best \
        --joints 12,13,21,22,8         # Upper/Lower eyelid L/R, Brow_Center
"""
import os
import sys
import argparse
import numpy as np
import yaml
import torch
import torch.nn.functional as F
import igl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tools', 'vis'))  # vis_hlbs_weights_multi

from utils.rig_loader import load_rig
from models.hierarchical_lbs import HierarchicalLBS_FullPred
from utils.matplotlib_rnd import normalize_homogeneous, perspective, translate, yrotate, xrotate, calc_face_norm
from vis_hlbs_weights_multi import load_template, load_nfs_feat_cached, extract_nfs_feat_online


def _project(V, model, MV):
    V_mu = np.median(V, axis=0)
    V_model = (V - V_mu) @ model.T + V_mu
    V_proj = V_model @ MV.T
    V_proj = V_proj[:, :3] / V_proj[:, 3:4]
    return V_model, V_proj


def _draw_heatmap_ax(ax, verts, faces, vals, V_model, V_proj, vmin, vmax, cmap='magma'):
    light_dir = np.array([0, 0, 1])
    Wf = vals[faces].mean(axis=1)
    VF = V_proj[faces]
    T  = VF[:, :, :2]
    Z  = -VF[:, :, 2].mean(1)

    C_norm = calc_face_norm(V_model[:, :3], faces)
    front  = C_norm[:, 2] > 0
    T = T[front]; Z = Z[front]; C_norm = C_norm[front]; Wf = Wf[front]
    order = np.argsort(Z)
    T = T[order]; C_norm = C_norm[order]; Wf = Wf[order]

    shade = (C_norm @ light_dir)[:, None].repeat(3, axis=-1)
    shade = np.clip(shade, 0, 1) * 0.7 + 0.2

    norm = Normalize(vmin=vmin, vmax=vmax)
    rgb  = plt.get_cmap(cmap)(norm(Wf))[:, :3]
    alpha = np.clip((Wf - vmin) / (vmax - vmin + 1e-12), 0, 1)[:, None]
    C = np.clip(shade * (1 - alpha) + rgb * alpha, 0, 1)

    coll = PolyCollection(T, closed=True, linewidth=0.1, facecolor=C, edgecolor=C)
    ax.add_collection(coll)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect(1)
    ax.set_xticks([]); ax.set_yticks([])


def _project_points(pts_world, verts_world, model, MV):
    """Project 3D points using the SAME normalization applied to the mesh
    (homogeneous-coord path mirrors _project)."""
    Vmin, Vmax = verts_world.min(0), verts_world.max(0)
    V_center = (Vmax + Vmin) / 2
    V_scale  = (Vmax - Vmin).max() / 2
    pts_norm = (pts_world - V_center) / V_scale                                 # (N, 3)
    pts_4 = np.concatenate([pts_norm, np.ones((pts_norm.shape[0], 1))], -1)     # (N, 4)

    V_hom = normalize_homogeneous(verts_world)                                  # (V, 4)
    V_mu  = np.median(V_hom, axis=0)                                            # (4,)

    pts_model = (pts_4 - V_mu) @ model.T + V_mu                                 # (N, 4)
    pts_proj  = pts_model @ MV.T                                                # (N, 4)
    return pts_proj[:, :2] / pts_proj[:, 3:4]


def render_diagnostic_panel(verts, faces, joint_pos_pred, joint_pos_maya,
                            logit_net, logit_gauss, W_final,
                            joint_idx, joint_name, log_sigma_j, save_path,
                            yrot=0, xrot=0):
    """4-panel diagnostic for one joint on one mesh.
    joint_pos_pred: [1,3] predicted μ (bind_pose_net)
    joint_pos_maya: [1,3] Maya init position (bind_pos_target)
    """
    V = normalize_homogeneous(verts)
    view = translate(0, 0, -4.5)
    proj = perspective(55, 1.0, 1.0, 100.0)
    MV   = proj @ view
    model = xrotate(xrot) @ yrotate(yrot)
    V_model, V_proj = _project(V, model, MV)

    jpos_pred_2d = _project_points(joint_pos_pred, verts, model, MV)[0]
    jpos_maya_2d = _project_points(joint_pos_maya, verts, model, MV)[0]
    shift = float(np.linalg.norm(joint_pos_pred[0] - joint_pos_maya[0]))

    # Normalize per-panel
    W_net   = F.softmax(torch.tensor(logit_net),   dim=-1).numpy()
    W_gauss = F.softmax(torch.tensor(logit_gauss), dim=-1).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    panels = [
        ('μ: Maya(blue) vs pred(red)', None),
        ('softmax(logit_net)',         W_net[:, joint_idx]),
        ('softmax(logit_gauss)',       W_gauss[:, joint_idx]),
        ('W_final',                    W_final[:, joint_idx]),
    ]

    for i, (title, vals) in enumerate(panels):
        ax = axes[i]
        if i == 0:
            dummy = np.zeros(verts.shape[0])
            _draw_heatmap_ax(ax, verts, faces, dummy, V_model, V_proj,
                             vmin=0, vmax=1, cmap='gray')
            # Maya init (blue circle) + predicted (red X) + arrow
            ax.scatter(jpos_maya_2d[0], jpos_maya_2d[1], s=120, facecolors='none',
                       edgecolors='blue', linewidths=2, zorder=4, label='Maya init')
            ax.scatter(jpos_pred_2d[0], jpos_pred_2d[1], s=200, c='red', marker='x',
                       linewidths=3, zorder=5, label='predicted')
            ax.annotate('', xy=jpos_pred_2d, xytext=jpos_maya_2d,
                        arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                        zorder=4)
            ax.text(0.02, 0.02, f'shift={shift:.3f}',
                    transform=ax.transAxes, fontsize=9, color='orange',
                    bbox=dict(facecolor='white', edgecolor='orange', pad=2))
            ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
        else:
            vmax = max(float(vals.max()), 0.01)
            _draw_heatmap_ax(ax, verts, faces, vals, V_model, V_proj,
                             vmin=0, vmax=vmax)
            # Overlay μ markers on weight panels too
            ax.scatter(jpos_maya_2d[0], jpos_maya_2d[1], s=60, facecolors='none',
                       edgecolors='cyan', linewidths=1.5, zorder=5)
            ax.scatter(jpos_pred_2d[0], jpos_pred_2d[1], s=80, c='red', marker='x',
                       linewidths=2, zorder=5)
            sm = ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap='magma')
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        ax.set_title(title, fontsize=10)

    sigma = float(np.exp(log_sigma_j))
    fig.suptitle(
        f"j{joint_idx:02d} {joint_name}  |  log σ={log_sigma_j:.3f}  σ={sigma:.4f}  "
        f"|  μ shift={shift:.3f}  |  W_max={W_final[:, joint_idx].max():.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', type=str, required=True)
    ap.add_argument('--epoch', type=str, default='best')
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--joints', type=str, default='12,13,21,22,8',
                    help='Comma-separated joint indices to diagnose')
    ap.add_argument('--datasets', type=str, default='ict:0,mf:12',
                    help='Comma-separated ds:idx pairs')
    args = ap.parse_args()

    device = torch.device(args.device)

    opts_path = os.path.join(args.ckpt_dir, 'train_opts.yml')
    with open(opts_path) as f:
        opts = yaml.safe_load(f)

    rig = load_rig(opts['rig_path'])
    # Replicate training-time options EXACTLY (same as vis_hlbs_weights_multi.py)
    _face_idx, _base_idx = None, None
    _aj_path = opts.get('active_joints_json')
    if _aj_path and os.path.exists(_aj_path):
        import json as _json
        with open(_aj_path) as f:
            _aj = _json.load(f)
        _face_idx = _aj['face_joint_idx']
        _base_idx = _aj['base_joint_idx']
    _sig = None
    if opts.get('sigma_targets_npy') and os.path.exists(opts['sigma_targets_npy']):
        _sig = np.load(opts['sigma_targets_npy']).astype(np.float32)
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
    print(f'Loaded {ckpt_path}')

    # ── log_sigma summary ────────────────────────────────────────────────
    if opts.get('use_gmm_hybrid', False):
        log_sigma = model.log_sigma.detach().cpu().numpy()      # [J]
        sigma    = np.exp(log_sigma)
        init_sig = float(opts.get('init_log_sigma', -1.2))

        print('\n=== log_sigma summary ===')
        print(f'  init log σ: {init_sig}  (init σ={np.exp(init_sig):.4f})')
        print(f'  learned log σ range: [{log_sigma.min():.3f}, {log_sigma.max():.3f}]')
        print(f'  learned σ range:     [{sigma.min():.4f},   {sigma.max():.4f}]')
        print('\n  per-joint (sorted by σ, top 15):')
        order = np.argsort(-sigma)
        for rank, j in enumerate(order[:15]):
            name = model.joint_names[j]
            print(f'    [{j:2d}] {name:40s}  log σ={log_sigma[j]:+.3f}  σ={sigma[j]:.4f}'
                  + ('  ⚠ grew' if log_sigma[j] > init_sig + 0.3 else ''))
        print('\n  bottom 5 (smallest σ):')
        for j in order[-5:]:
            name = model.joint_names[j]
            print(f'    [{j:2d}] {name:40s}  log σ={log_sigma[j]:+.3f}  σ={sigma[j]:.4f}')
    else:
        print('[WARN] use_gmm_hybrid=False — logit_gauss panel will be zero')
        log_sigma = np.zeros(model.num_joints)
        sigma = np.ones(model.num_joints)

    # ── For each dataset: forward + diagnostic panel per joint ───────────
    joint_indices = [int(x) for x in args.joints.split(',')]
    ds_specs = [(s.split(':')[0], int(s.split(':')[1])) for s in args.datasets.split(',')]
    base_out = os.path.join(args.ckpt_dir, f'diagnose_{args.epoch}')
    os.makedirs(base_out, exist_ok=True)

    for ds_key, id_idx in ds_specs:
        verts, faces, id_name = load_template(ds_key, id_idx)
        normals = igl.per_vertex_normals(verts, faces).astype(np.float32)
        src_v = torch.tensor(verts,   dtype=torch.float32, device=device).unsqueeze(0)
        src_n = torch.tensor(normals, dtype=torch.float32, device=device).unsqueeze(0)

        print(f'\n── {ds_key}:{id_idx} ({id_name}, V={verts.shape[0]}) ──')

        nfs_feat = None
        nfs_feat_dir = opts.get('nfs_feat_dir')
        if nfs_feat_dir:
            nfs_feat = load_nfs_feat_cached(id_name, nfs_feat_dir, device=str(device))
            if nfs_feat is None:
                nfs_feat = extract_nfs_feat_online(verts, faces, device=str(device))
            if nfs_feat is not None:
                nfs_feat = nfs_feat.unsqueeze(0)

        with torch.no_grad():
            skin_input, _adain = model._prepare_feat(src_v, src_n, nfs_feat)
            # Bind pose dispatch: net / anchor_pool / freeze
            if model.freeze_bind_pose:
                joint_pos = model.bind_pos_target.unsqueeze(0).expand(src_v.shape[0], -1, -1)
            elif getattr(model, 'bind_pose_mode', 'net') == 'anchor_pool':
                _, joint_pos = model._get_bind_pose_anchor(src_v, nfs_feat)
            else:
                _, joint_pos = model._get_bind_pose(skin_input, adain_input=_adain)

            # Replicate _get_skinning_weights, keeping separate logits
            if model.dfn_skin:
                logit_net = model.skin_weight_net(skin_input)
            else:
                logit_net = model.skin_weight_net(skin_input, adain_input=_adain)
            logit_net = model._smooth_logit_W(logit_net)

            if model.use_gmm_hybrid:
                diff = src_v.unsqueeze(2) - joint_pos.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1)
                sigma_sq = torch.exp(2 * model.log_sigma).view(1, 1, -1)
                logit_gauss = -dist_sq / (2 * sigma_sq)
            else:
                logit_gauss = torch.zeros_like(logit_net)

            # W_final per gmm_mode
            mode = getattr(model, 'gmm_mode', 'additive')
            if model.use_gmm_hybrid and mode == 'multiplicative':
                W_prior = F.softmax(logit_gauss, dim=-1)
                modul   = torch.sigmoid(logit_net)
                W_raw   = W_prior * modul
                W_final = W_raw / (W_raw.sum(dim=-1, keepdim=True) + 1e-8)
            elif model.use_gmm_hybrid and mode == 'residual':
                s = float(getattr(model, 'residual_scale', 2.0))
                bounded = torch.tanh(logit_net / s) * s
                window  = torch.exp(logit_gauss)
                W_final = F.softmax(logit_gauss + window * bounded, dim=-1)
            else:  # additive (or no GMM)
                W_final = F.softmax(logit_net + logit_gauss, dim=-1)

        jpos_np     = joint_pos[0].cpu().numpy()        # [J, 3]  predicted
        jpos_maya   = model.bind_pos_target.detach().cpu().numpy()  # [J, 3] Maya init
        lnet_np     = logit_net[0].cpu().numpy()        # [N, J]
        lgauss_np   = logit_gauss[0].cpu().numpy()      # [N, J]
        Wfinal_np   = W_final[0].cpu().numpy()          # [N, J]

        out_dir = os.path.join(base_out, f'{ds_key}_{id_idx:03d}')
        for j in joint_indices:
            name = model.joint_names[j] if j < len(model.joint_names) else f'joint_{j}'
            save_path = os.path.join(out_dir, f'diag_j{j:02d}_{name}.png')
            render_diagnostic_panel(
                verts, faces,
                jpos_np[j:j+1], jpos_maya[j:j+1],
                lnet_np, lgauss_np, Wfinal_np,
                j, name, float(log_sigma[j]),
                save_path, yrot=0, xrot=0,
            )
            jp = jpos_np[j]; jm = jpos_maya[j]
            shift = np.linalg.norm(jp - jm)
            d_to_nearest = np.linalg.norm(verts - jp, axis=-1).min()
            print(f'  j{j:02d} {name:38s}  μ_pred=({jp[0]:+.2f},{jp[1]:+.2f},{jp[2]:+.2f})  '
                  f'shift_from_maya={shift:.3f}  d_vert={d_to_nearest:.3f}  '
                  f'σ={sigma[j]:.4f}  W_max={Wfinal_np[:, j].max():.4f}')
        print(f'  saved → {out_dir}')

    print(f'\nDone. Output: {base_out}')


if __name__ == '__main__':
    main()
