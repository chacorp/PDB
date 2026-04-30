"""
precompute_joint_anchors.py — Derive (anchor, offset) for all 66 joints from ICT mean mesh.

Anchor encodes the NFS-feature signature of each joint's region.
Offset is the world-space residual that recovers Maya GT bind_pos exactly on ICT mean.

These two are universal: applied identically to every other mesh (ICT id-N, MF, BIWI, COMA, ...).

Outputs:
    {rig_path}/joint_anchors.npy   [J, 256]
    {rig_path}/joint_offsets.npy   [J, 3]   world-space
    {rig_path}/joint_anchors_report.json   per-joint sanity stats

Usage:
    python precompute_joint_anchors.py \
        --rig_path maya_rig/hybrid \
        --topo_key ict \
        --sigma_targets maya_rig/hybrid/sigma_targets.npy \
        --nfs_ckpt ckpts_comparison/NFS-best
"""
import os
import sys
import json
import argparse
import numpy as np
import yaml
import torch
import igl

sys.path.insert(0, os.path.dirname(__file__))
from utils.rig_loader import load_rig
from utils.anchor_pool import derive_anchor_offset, compute_bind_pos


def _load_mean_mesh(topo_key, data_basedir):
    """Return (verts [V,3], faces [F,3]) mean mesh for a topology."""
    if topo_key == 'ict':
        from utils.remesh_utils import ICT_face_model
        m = ICT_face_model()
        return m.neutral_verts.astype(np.float32), m.faces.astype(np.int32)
    if topo_key == 'mf':
        import pickle
        pkl = os.path.join(data_basedir, 'multiface_align', 'mf_templates.pkl')
        if not os.path.exists(pkl):
            pkl = os.path.join(data_basedir, 'pca', 'multiface_align', 'mf_templates.pkl')
        with open(pkl, 'rb') as f:
            t = pickle.load(f)
        faces = np.array(t['face'], dtype=np.int32)
        keys  = [k for k in t if k != 'face']
        verts = np.stack([np.array(t[k], dtype=np.float32) for k in keys]).mean(axis=0)
        return verts.astype(np.float32), faces
    raise ValueError(f'unknown topo_key: {topo_key}')


def _extract_nfs_feat(verts, faces, nfs_ckpt, device):
    """Run NFS seg encoder on a mesh -> [V, 256] tensor on device."""
    from vis_hlbs_weights_multi import extract_nfs_feat_online
    feat = extract_nfs_feat_online(verts, faces, nfs_ckpt=nfs_ckpt, device=str(device))
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rig_path', type=str, required=True)
    ap.add_argument('--topo_key', type=str, default='ict',
                    help='Primary topology — used to derive INITIAL anchor (spatial-α) and offset')
    ap.add_argument('--aux_topo_keys', nargs='*', default=[],
                    help='Additional topologies to include in joint anchor optimization '
                         '(e.g., --aux_topo_keys mf). Their bind_pos targets are used to '
                         'further constrain anchor for cross-topology generalization.')
    ap.add_argument('--data_basedir', type=str, default='/data/sihun',
                    help='For loading non-ICT topology mean meshes')
    ap.add_argument('--sigma_targets', type=str, default=None,
                    help='Path to sigma_targets.npy (default: {rig_path}/sigma_targets.npy)')
    ap.add_argument('--nfs_ckpt', type=str, default='ckpts_comparison/NFS-best',
                    help='Pretrained NFS encoder checkpoint dir')
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--out_dir', type=str, default=None,
                    help='Output dir (default: {rig_path})')
    # Anchor optimization options
    ap.add_argument('--optimize', dest='optimize', action='store_true',
                    help='After deriving initial (anchor, offset), optimize anchor '
                         'via gradient descent so cosine-attention replays maya_bind on ICT.')
    ap.add_argument('--no_optimize', dest='optimize', action='store_false',
                    help='Skip optimization, use spatial-α-derived anchor only.')
    ap.set_defaults(optimize=True)
    ap.add_argument('--opt_steps', type=int, default=500,
                    help='Gradient steps for anchor optimization')
    ap.add_argument('--opt_lr', type=float, default=1e-2,
                    help='Learning rate for anchor optimization')
    ap.add_argument('--opt_temperature', type=float, default=0.1,
                    help='Temperature used in compute_bind_pos during anchor optimization')
    ap.add_argument('--opt_lambda_reg', type=float, default=1e-3,
                    help='L2 reg pulling optimized anchor toward initial (preserves '
                         'cross-topology generalization)')
    ap.add_argument('--aux_loss_weight', type=float, default=1.0,
                    help='Multiplier on aux topology loss terms (>1 to weight aux higher; '
                         'e.g., 2.0 to prioritize MF over ICT)')
    args = ap.parse_args()

    device = torch.device(args.device)
    sigma_path = args.sigma_targets or os.path.join(args.rig_path, 'sigma_targets.npy')
    out_dir = args.out_dir or args.rig_path

    rig = load_rig(args.rig_path, device='cpu')
    J = len(rig.joint_names)

    # ── Load σ targets ──────────────────────────────────────────────────
    sigma = torch.tensor(np.load(sigma_path), dtype=torch.float32, device=device)
    assert sigma.shape == (J,), f'sigma shape {tuple(sigma.shape)} != ({J},)'
    print(f'sigma:     range=[{sigma.min():.3f}, {sigma.max():.3f}] from {sigma_path}')

    # ── Load primary topology (used for INITIAL anchor + offset) ────────
    primary = args.topo_key
    if primary not in rig.bind_pos_dict:
        print(f'[WARN] no per-topo bind_pos for "{primary}", using rig.bind_pos default')
        maya_bind_primary = rig.bind_pos.float().to(device)
    else:
        maya_bind_primary = rig.bind_pos_dict[primary].float().to(device)

    primary_verts_np, primary_faces_np = _load_mean_mesh(primary, args.data_basedir)
    primary_verts = torch.tensor(primary_verts_np, dtype=torch.float32, device=device)
    print(f'\n[primary={primary}] mean mesh: V={primary_verts.shape[0]}, F={primary_faces_np.shape[0]}')
    print(f'  maya_bind: range=[{maya_bind_primary.min():.3f}, {maya_bind_primary.max():.3f}]')

    print(f'  Running NFS seg encoder on {primary} mean...')
    primary_feat = _extract_nfs_feat(primary_verts_np, primary_faces_np, args.nfs_ckpt, device)
    if primary_feat.dim() == 1:
        primary_feat = primary_feat.unsqueeze(0)
    print(f'  feat: {tuple(primary_feat.shape)}')

    # ── Load auxiliary topologies (used for joint optimization only) ────
    aux = []  # list of (key, verts_t, feat_t, maya_bind_t)
    for topo in args.aux_topo_keys:
        if topo not in rig.bind_pos_dict:
            print(f'[WARN] aux topology "{topo}" missing in rig.bind_pos_dict, skipping')
            continue
        bp = rig.bind_pos_dict[topo].float().to(device)
        v_np, f_np = _load_mean_mesh(topo, args.data_basedir)
        v_t = torch.tensor(v_np, dtype=torch.float32, device=device)
        print(f'\n[aux={topo}] mean mesh: V={v_t.shape[0]}, F={f_np.shape[0]}')
        print(f'  maya_bind: range=[{bp.min():.3f}, {bp.max():.3f}]')
        print(f'  Running NFS seg encoder on {topo} mean...')
        ft = _extract_nfs_feat(v_np, f_np, args.nfs_ckpt, device)
        if ft.dim() == 1:
            ft = ft.unsqueeze(0)
        print(f'  feat: {tuple(ft.shape)}')
        aux.append((topo, v_t, ft, bp))

    # ── Derive anchor + offset on PRIMARY (initial) ─────────────────────
    anchor_init, offset, pool_pos, alpha_max = derive_anchor_offset(
        primary_verts, primary_feat, maya_bind_primary, sigma)
    anchor = anchor_init.clone()

    # ── Initial replay error per topology ──────────────────────────────
    def _err_for(verts, feat, bind):
        with torch.no_grad():
            pred = compute_bind_pos(verts, feat, anchor, offset,
                                     temperature=args.opt_temperature,
                                     offset_frame='world')
        return (pred - bind).norm(dim=-1).cpu().numpy()

    print(f'\n=== Sanity: initial replay (T={args.opt_temperature}) ===')
    err_init_primary = _err_for(primary_verts, primary_feat, maya_bind_primary)
    print(f'  [{primary}]  median={np.median(err_init_primary):.4f}  '
          f'mean={err_init_primary.mean():.4f}  max={err_init_primary.max():.4f}')
    for topo, v, f, bp in aux:
        e = _err_for(v, f, bp)
        print(f'  [{topo:>4s}] median={np.median(e):.4f}  mean={e.mean():.4f}  max={e.max():.4f}')

    # ── Anchor optimization (joint loss across primary + aux) ──────────
    if args.optimize:
        topo_list = [(primary, primary_verts, primary_feat, maya_bind_primary)] + aux
        # Per-topology weights: primary=1.0, aux=aux_loss_weight
        topo_weights = {primary: 1.0}
        for t in aux:
            topo_weights[t[0]] = args.aux_loss_weight
        print(f'\n=== Anchor optimization ({args.opt_steps} steps, lr={args.opt_lr}, '
              f'T={args.opt_temperature}, λ_reg={args.opt_lambda_reg}, '
              f'topo_weights={topo_weights}) ===')
        anchor = anchor.clone().requires_grad_(True)
        opt = torch.optim.Adam([anchor], lr=args.opt_lr)
        offset_fixed = offset.detach()
        anchor_init_fixed = anchor_init.detach()

        log_every = max(1, args.opt_steps // 10)
        for step in range(args.opt_steps):
            l_fit_total = 0.0
            per_topo_fit = {}
            for tname, v, f, bp in topo_list:
                pred = compute_bind_pos(v, f, anchor, offset_fixed,
                                         temperature=args.opt_temperature,
                                         offset_frame='world')
                lf = (pred - bp).pow(2).sum(dim=-1).mean()
                l_fit_total = l_fit_total + topo_weights[tname] * lf
                per_topo_fit[tname] = lf.item()
            l_reg = (anchor - anchor_init_fixed).pow(2).sum(dim=-1).mean()
            loss = l_fit_total + args.opt_lambda_reg * l_reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % log_every == 0 or step == args.opt_steps - 1:
                summary = ' '.join(f'{n}={v:.3e}' for n, v in per_topo_fit.items())
                with torch.no_grad():
                    pe = compute_bind_pos(primary_verts, primary_feat, anchor, offset_fixed,
                                          temperature=args.opt_temperature,
                                          offset_frame='world')
                    err_now = (pe - maya_bind_primary).norm(dim=-1)
                print(f'  step {step:4d}: fit_sum={l_fit_total.item():.3e}  '
                      f'reg={l_reg.item():.2e}  ({summary})  '
                      f'{primary} err median={err_now.median().item():.4f} '
                      f'max={err_now.max().item():.4f}')
        anchor = anchor.detach()

    anchor_np = anchor.cpu().numpy().astype(np.float32)
    offset_np = offset.cpu().numpy().astype(np.float32)
    pool_pos_np = pool_pos.cpu().numpy().astype(np.float32)
    alpha_max_np = alpha_max.cpu().numpy().astype(np.float32)

    # ── Final sanity replay per topology ──────────────────────────────
    print(f'\n=== Sanity: final replay (T={args.opt_temperature}) ===')
    with torch.no_grad():
        bind_replay = compute_bind_pos(primary_verts, primary_feat, anchor, offset,
                                        temperature=args.opt_temperature,
                                        offset_frame='world')
    bind_replay_np = bind_replay.cpu().numpy()
    maya_bind_np = maya_bind_primary.cpu().numpy()
    err = np.linalg.norm(bind_replay_np - maya_bind_np, axis=-1)                            # primary
    err_construct = np.linalg.norm(pool_pos_np + offset_np - maya_bind_np, axis=-1)
    print(f'  [{primary}]  median={np.median(err):.4f}  mean={err.mean():.4f}  max={err.max():.4f}')

    # Aux topologies final err (same anchor & offset, evaluated on each)
    aux_final = {}
    for tname, v, f, bp in aux:
        with torch.no_grad():
            pr = compute_bind_pos(v, f, anchor, offset,
                                   temperature=args.opt_temperature,
                                   offset_frame='world')
        e = (pr - bp).norm(dim=-1).cpu().numpy()
        aux_final[tname] = e
        print(f'  [{tname:>4s}] median={np.median(e):.4f}  mean={e.mean():.4f}  max={e.max():.4f}')

    # ── Save ────────────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, 'joint_anchors.npy'), anchor_np)
    np.save(os.path.join(out_dir, 'joint_offsets.npy'), offset_np)

    print(f'\nSaved:')
    print(f'  joint_anchors.npy   shape {anchor_np.shape}')
    print(f'  joint_offsets.npy   shape {offset_np.shape}')

    # ── Per-joint report ────────────────────────────────────────────────
    print('\n=== Per-joint report (sorted by replay error) ===')
    print(f'  {"idx":>3s}  {"name":42s}  {"||δ||":>8s}  {"σ":>7s}  {"α_max":>8s}  {"err_constr":>11s}  {"err_replay":>11s}')
    order = np.argsort(-err)
    delta_norm = np.linalg.norm(offset_np, axis=-1)
    for j in order:
        flag = ''
        if delta_norm[j] > 0.05:
            flag += ' ⚠δ'
        if err[j] > 0.05:
            flag += ' ⚠replay'
        print(f'  {j:3d}  {rig.joint_names[j]:42s}  {delta_norm[j]:8.4f}  '
              f'{float(sigma[j].cpu()):7.3f}  {alpha_max_np[j]:8.4f}  '
              f'{err_construct[j]:11.6f}  {err[j]:11.6f}{flag}')

    report = {
        'topo_key':       args.topo_key,
        'aux_topo_keys':  list(args.aux_topo_keys),
        'rig_path':       args.rig_path,
        'sigma_path':     sigma_path,
        'nfs_ckpt':       args.nfs_ckpt,
        'V_primary':      int(primary_verts.shape[0]),
        'J':              int(J),
        'aux_final_err':  {k: {'median': float(np.median(e)),
                                'mean':   float(e.mean()),
                                'max':    float(e.max())} for k, e in aux_final.items()},
        'per_joint': [{
            'idx':        int(j),
            'name':       rig.joint_names[j],
            'sigma':      float(sigma[j].cpu()),
            'delta_norm': float(delta_norm[j]),
            'alpha_max':  float(alpha_max_np[j]),
            'err_construct': float(err_construct[j]),
            'err_replay':    float(err[j]),
        } for j in range(J)],
    }
    rep_path = os.path.join(out_dir, 'joint_anchors_report.json')
    with open(rep_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nReport → {rep_path}')

    # Summary
    print(f'\nSummary:')
    print(f'  ||δ||  median={np.median(delta_norm):.4f}  max={delta_norm.max():.4f}')
    print(f'  err_construct (should be ~0): max={err_construct.max():.6f}')
    print(f'  err_replay (anchor pool reconstructs maya_bind): '
          f'median={np.median(err):.4f}  max={err.max():.4f}')


if __name__ == '__main__':
    main()
