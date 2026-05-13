"""
analyze_active_joints.py — Identify which joints actually contribute to motion.

Runs forward on validation data and measures per-joint:
  - motion magnitude  : rotation angle + translation norm
  - W influence       : sum of skinning weights across vertices (B-mean)
  - activity          : motion × influence  (the composite score)

Joints with activity below a threshold are candidates for freezing.

Outputs to {ckpt_dir}/active_joints_{epoch}/:
  - active_joints.json    : face_joint_idx, base_joint_idx, full stats per joint
  - active_joints.png     : bar chart of activity (sorted)

Usage:
    python analyze_active_joints.py \
        --ckpt_dir ckpts_hlbs/2026-04-22-10-13-53-HLBS-FullPred-ict-jTrans-nrm0.1-Wsm0.01 \
        --epoch best \
        --max_batches 100 \
        --threshold_pct 10            # bottom 10% activity → frozen candidates

Notes:
  - "Frozen" joints are those with activity below threshold, i.e., that
    barely move AND have little weight influence.
  - Output face_joint_idx is the COMPLEMENT (joints to KEEP for prediction).
"""
import os
import sys
import json
import argparse
import numpy as np
import yaml
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))
from utils.rig_loader import load_rig
from models.hierarchical_lbs import HierarchicalLBS_FullPred
from dataloader_CBD import CBDDataset, CBDdataSampler, CBD_collate_wrapper


def _rotation_angle(R):
    """
    R: [..., 3, 3]  rotation matrix
    Returns angle in radians: acos((tr(R)-1)/2)  in [0, pi]
    """
    trace = R.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_t = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos_t)


def _load_model(ckpt_dir, epoch, device):
    opts_path = os.path.join(ckpt_dir, 'train_opts.yml')
    with open(opts_path) as f:
        opts = yaml.safe_load(f)

    rig = load_rig(opts['rig_path'])
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
    ).to(device)
    ckpt_path = os.path.join(ckpt_dir, f'model_hlbs_{epoch}.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()
    print(f'Loaded {ckpt_path}')
    return model, opts


class _Opts:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def _load_nfs_cache(opts, device):
    import glob as _glob
    cache = {}
    if opts.get('nfs_feat_dir'):
        for fp in _glob.glob(os.path.join(opts['nfs_feat_dir'], '*_nfs_feat.npy')):
            name = os.path.basename(fp).replace('_nfs_feat.npy', '')
            cache[name] = torch.tensor(np.load(fp), dtype=torch.float32).to(device)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', type=str, required=True)
    ap.add_argument('--epoch', type=str, default='best')
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--max_batches', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--threshold_pct', type=float, default=10.0,
                    help='Joints below this percentile of activity → frozen candidates')
    ap.add_argument('--base_joint_name', type=str, default='hybrid_jnt_Skull_Root',
                    help='Joint that absorbs weight from non-face region (default: Skull_Root)')
    ap.add_argument('--alpha_trans', type=float, default=1.0,
                    help='Weight on translation_norm vs rotation_angle in motion metric')
    args = ap.parse_args()

    device = torch.device(args.device)
    model, opts = _load_model(args.ckpt_dir, args.epoch, device)
    J = model.num_joints
    joint_names = list(model.joint_names)

    # ── Dataset (validation split) ───────────────────────────────────────
    opts_obj = _Opts(opts)
    opts_obj.num_workers = 0
    valid_ds = CBDDataset(opts_obj, is_valid=True,
                          toggle=opts.get('data_toggle', False),
                          data_basedir=opts.get('data_basedir', '/data/sihun'))

    sampler = CBDdataSampler(valid_ds.len_list, args.batch_size,
                             shuffle=False, balance=False, is_train=False, region_min=1)
    loader = torch.utils.data.DataLoader(
        valid_ds, batch_sampler=sampler,
        collate_fn=partial(CBD_collate_wrapper, device='cpu'),
        num_workers=0, persistent_workers=False,
    )
    print(f'valid_ds size: {len(valid_ds)}, batches: {len(loader)} (using up to {args.max_batches})')

    nfs_cache = _load_nfs_cache(opts, device)

    # ── Accumulators ────────────────────────────────────────────────────
    sum_theta   = torch.zeros(J, device=device)   # rotation angle (rad)
    sum_tmag    = torch.zeros(J, device=device)   # translation magnitude
    sum_Wsum    = torch.zeros(J, device=device)   # W summed over vertices
    sum_Wmax    = torch.zeros(J, device=device)   # max W over vertices (per-frame max)
    sum_active  = torch.zeros(J, device=device)   # motion * W_sum
    n_frames    = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, total=min(args.max_batches, len(loader)))):
            if idx >= args.max_batches:
                break

            batch = batch.to(device)
            src_v = batch.template
            src_n = batch.template_normal
            gt_v  = batch.vertices
            gt_n  = batch.vertices_normal

            B, V, _ = src_v.shape

            # NFS feat
            _nfs_feat = None
            if opts.get('nfs_feat_dir') and nfs_cache:
                feats = []
                for b in range(B):
                    id_key = batch.id_name[b]
                    if id_key in nfs_cache:
                        f = nfs_cache[id_key]
                        if f.shape[0] > V:
                            f = f[:V]
                        feats.append(f)
                    else:
                        feats.append(torch.zeros(V, 256, device=device))
                _nfs_feat = torch.stack(feats, dim=0)

            delta     = gt_v - src_v
            src_in    = torch.cat([src_v, src_n], dim=-1)
            deform_in = torch.cat([delta, gt_n, src_in], dim=-1)

            _, extras = model(src_v, deform_in, source_normal=src_n,
                              nfs_feat=_nfs_feat, return_extras=True)

            W       = extras['W']          # [B, V, J]
            local_R = extras['local_R']    # [B, J, 3, 3]
            local_t = extras['local_t']    # [B, J, 3]

            theta = _rotation_angle(local_R)                     # [B, J]
            tmag  = local_t.norm(dim=-1)                          # [B, J]
            Wsum  = W.sum(dim=1)                                  # [B, J]  (sum over V)
            Wmax  = W.amax(dim=1)                                 # [B, J]
            motion = theta + args.alpha_trans * tmag              # [B, J]
            active = motion * Wsum

            sum_theta  += theta.sum(dim=0)
            sum_tmag   += tmag.sum(dim=0)
            sum_Wsum   += Wsum.sum(dim=0)
            sum_Wmax   += Wmax.sum(dim=0)
            sum_active += active.sum(dim=0)
            n_frames   += B

    # ── Average per frame ───────────────────────────────────────────────
    avg = lambda t: (t / max(n_frames, 1)).cpu().numpy()
    theta  = avg(sum_theta)
    tmag   = avg(sum_tmag)
    Wsum   = avg(sum_Wsum)
    Wmax   = avg(sum_Wmax)
    active = avg(sum_active)
    motion = theta + args.alpha_trans * tmag

    # ── Threshold & partition ───────────────────────────────────────────
    threshold = np.percentile(active, args.threshold_pct)
    frozen_idx = [int(j) for j in np.where(active <= threshold)[0]]
    face_idx   = [int(j) for j in np.where(active >  threshold)[0]]

    # Base joint: find by name; fallback to Skull_Root by index
    if args.base_joint_name in joint_names:
        base_idx = joint_names.index(args.base_joint_name)
    else:
        print(f'[WARN] {args.base_joint_name} not found, defaulting base_idx=2')
        base_idx = 2

    # ── Print summary ───────────────────────────────────────────────────
    print(f'\n=== Activity analysis (n_frames={n_frames}, threshold_pct={args.threshold_pct}, threshold_val={threshold:.4e}) ===')
    print(f'{"idx":>3s}  {"name":40s}  {"θ (rad)":>9s}  {"|t|":>8s}  {"W_sum":>9s}  {"W_max":>7s}  {"activity":>10s}  {"status":>8s}')
    order = np.argsort(active)
    for j in order:
        status = 'FREEZE' if j in frozen_idx else 'active'
        print(f'{j:3d}  {joint_names[j]:40s}  {theta[j]:9.4f}  {tmag[j]:8.4f}  '
              f'{Wsum[j]:9.2f}  {Wmax[j]:7.3f}  {active[j]:10.4e}  {status:>8s}')

    # ── Output JSON ─────────────────────────────────────────────────────
    out_dir = os.path.join(args.ckpt_dir, f'active_joints_{args.epoch}')
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, 'active_joints.json')

    report = {
        'ckpt_dir':           args.ckpt_dir,
        'epoch':              args.epoch,
        'n_frames':           n_frames,
        'threshold_pct':      args.threshold_pct,
        'threshold_val':      float(threshold),
        'alpha_trans':        args.alpha_trans,
        'base_joint_name':    joint_names[base_idx],
        'base_joint_idx':     base_idx,
        'face_joint_idx':     face_idx,
        'frozen_joint_idx':   frozen_idx,
        'face_joint_names':   [joint_names[j] for j in face_idx],
        'frozen_joint_names': [joint_names[j] for j in frozen_idx],
        'per_joint_stats': [
            {
                'idx':      int(j),
                'name':     joint_names[j],
                'theta':    float(theta[j]),
                't_mag':    float(tmag[j]),
                'W_sum':    float(Wsum[j]),
                'W_max':    float(Wmax[j]),
                'motion':   float(motion[j]),
                'activity': float(active[j]),
                'frozen':   bool(j in frozen_idx),
            }
            for j in range(J)
        ],
    }
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nJSON → {out_json}')
    print(f'  face_joint_idx  ({len(face_idx)}): {face_idx}')
    print(f'  frozen_joint_idx ({len(frozen_idx)}): {frozen_idx}')
    print(f'  base_joint_idx:  {base_idx} ({joint_names[base_idx]})')

    # ── Plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(max(12, J * 0.2), 14))
    xs = np.arange(J)
    order = np.argsort(-active)
    colors = ['tomato' if j in frozen_idx else 'steelblue' for j in order]
    labels = [joint_names[j].replace('hybrid_jnt_', '') for j in order]

    axes[0].bar(xs, active[order], color=colors)
    axes[0].axhline(threshold, linestyle='--', color='black', linewidth=0.8,
                    label=f'threshold (p{args.threshold_pct:.0f})')
    axes[0].set_ylabel('activity = motion · W_sum')
    axes[0].set_yscale('log'); axes[0].legend()
    axes[0].set_title(f'Joint activity (sorted).  red=frozen ({len(frozen_idx)}), blue=active ({len(face_idx)})')

    axes[1].bar(xs, motion[order], color=colors)
    axes[1].set_ylabel('motion = θ + α·|t|')

    axes[2].bar(xs, Wsum[order], color=colors)
    axes[2].set_ylabel('W_sum (per frame avg)')

    axes[3].bar(xs, Wmax[order], color=colors)
    axes[3].set_ylabel('W_max (per vertex)')
    axes[3].set_xticks(xs)
    axes[3].set_xticklabels(labels, rotation=90, fontsize=7)

    for ax in axes[:3]:
        ax.set_xticks([])

    fig.tight_layout()
    png_path = os.path.join(out_dir, 'active_joints.png')
    fig.savefig(png_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'PNG  → {png_path}')


if __name__ == '__main__':
    main()
