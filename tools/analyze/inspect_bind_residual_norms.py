"""inspect_bind_residual_norms.py
Load a HLBS ckpt, run _get_bind_pose once on the ICT mean template, then
report per-joint L2 norm of the bind-pose residual split by helper vs
non-helper. Verifies whether the L_bind_residual=0.1 / L_helper_residual
=0.01 differential actually produces more freedom for helpers.

Usage:
    python tools/analyze/inspect_bind_residual_norms.py \
        --ckpt ckpts_hlbs/2026-05-26-09-59-07-HLBS-FullPred-ict-jTrans \
        [--epoch 100|best|-1]
"""
import os, sys, json, argparse
import numpy as np
import torch
import yaml
import igl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.hierarchical_lbs import HierarchicalLBS_FullPred
from utils.rig_loader import load_rig
from utils.remesh_utils import ICT_face_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--epoch', default='-1', help='int, or "best"/-1 for newest')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    with open(os.path.join(args.ckpt, 'train_opts.yml')) as f:
        opts = yaml.safe_load(f)
    print(f"[opts] lambda_bind_residual={opts.get('lambda_bind_residual')}  "
          f"lambda_helper_residual={opts.get('lambda_helper_residual')}  "
          f"bind_pose_base_residual={opts.get('bind_pose_base_residual')}")

    with open(opts['active_joints_json']) as f:
        aj = json.load(f)
    face_idx = aj['face_joint_idx']
    base_idx = aj.get('base_joint_idx')   # model wants int
    helper_idx = aj.get('helper_joint_idx', [])

    # resolve ckpt file
    if args.epoch in ('best', '-1', -1):
        cand = os.path.join(args.ckpt, 'model_hlbs_best.pth')
        if not os.path.isfile(cand):
            files = sorted(f for f in os.listdir(args.ckpt)
                           if f.startswith('model_hlbs_') and f.endswith('.pth'))
            if not files:
                raise FileNotFoundError(f'no model_hlbs_*.pth in {args.ckpt}')
            cand = os.path.join(args.ckpt, files[-1])
        ckpt_path = cand
    else:
        ckpt_path = os.path.join(args.ckpt, f'model_hlbs_{int(args.epoch):03d}.pth')
    sd = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    if 'helper_joint_idx_buf' in sd:
        helper_idx = sd['helper_joint_idx_buf'].tolist()
    if 'face_joint_idx' in sd:
        face_idx = sd['face_joint_idx'].tolist()
    print(f"[ckpt] {os.path.basename(ckpt_path)}  "
          f"face={len(face_idx)} helper={len(helper_idx)}")

    rig = load_rig(opts['rig_path'])
    model = HierarchicalLBS_FullPred(
        rig=rig,
        topology=opts.get('topo_key', 'ict'),
        in_dim_exp=12,
        hid_dim=opts.get('hid_dim', 128),
        num_layers=opts.get('num_layers', 4),
        device=args.device,
        use_joint_trans=opts.get('use_joint_trans', True),
        smooth_W=opts.get('smooth_delta_W', 0),
        smooth_W_alpha=opts.get('smooth_delta_W_alpha', 0.5),
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
        residual_scale=opts.get('residual_scale', 2.0),
        sigma_targets=None,
        bind_pose_mode=opts.get('bind_pose_mode', 'net'),
        joint_anchors=None,
        joint_offsets=None,
        attn_temperature_init=opts.get('attn_temperature_init', 0.1),
        face_joint_idx=face_idx,
        base_joint_idx=base_idx,
        face_mask_r0=opts.get('face_mask_r0', 1.0),
        face_mask_r1=opts.get('face_mask_r1', 2.25),
        helper_joint_idx=helper_idx,
        bind_pose_base_residual=bool(opts.get('bind_pose_base_residual', 0)),
    ).to(args.device)
    msg = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")
    if msg.unexpected_keys:
        print(f"  unexpected[:5]: {msg.unexpected_keys[:5]}")
    model.eval()

    # ICT mean template (zero identity coeff)
    ict = ICT_face_model()
    v_num = ict.region[0][0]
    template_np = ict.neutral_verts[:v_num].astype(np.float32)
    normals_np = igl.per_vertex_normals(template_np, ict.faces).astype(np.float32)
    V = torch.from_numpy(template_np).unsqueeze(0).to(args.device)
    N = torch.from_numpy(normals_np).unsqueeze(0).to(args.device)

    with torch.no_grad():
        skin_in, adain = model._prepare_feat(V, N, nfs_feat=None)
        _, joint_pos = model._get_bind_pose(skin_in, adain_input=adain)
    res = model._last_bind_pose_residual
    print(f"[residual] shape={tuple(res.shape)}")
    res_norm = res.squeeze(0).norm(dim=-1).cpu().numpy()  # [J]
    J = len(res_norm)
    helper_set = set(int(x) for x in helper_idx)
    non_helper = [j for j in range(J) if j not in helper_set]

    def stats(name, vals):
        if len(vals) == 0:
            print(f'  {name:>15s}: empty'); return
        v = np.asarray(vals)
        print(f'  {name:>15s}: n={len(v):3d}  mean={v.mean():.4f}  '
              f'median={np.median(v):.4f}  p90={np.percentile(v,90):.4f}  '
              f'max={v.max():.4f}')

    print('\n=== per-joint bind-pose residual L2 norm (ICT mean template) ===')
    stats('all', res_norm)
    if helper_set:
        stats('helper', res_norm[sorted(helper_set)])
    stats('non-helper', res_norm[non_helper])

    if helper_set:
        h_mean = res_norm[sorted(helper_set)].mean()
        nh_mean = res_norm[non_helper].mean()
        ratio = h_mean / max(nh_mean, 1e-12)
        print(f"\n  helper/non-helper mean ratio = {ratio:.2f}x")
        print(f"  expected: > 1 if lambda differential is effective")
        print(f"  (lambda_bind_residual={opts.get('lambda_bind_residual')} on non-helper, "
              f"lambda_helper_residual={opts.get('lambda_helper_residual')} on helper — "
              f"~{(opts.get('lambda_bind_residual', 0.1) / max(opts.get('lambda_helper_residual', 0.01), 1e-12)):.0f}x weaker on helper)")

    # also list top-5 helpers and top-5 non-helpers by residual norm
    print('\n  top-5 helpers by residual norm:')
    for j in sorted(sorted(helper_set), key=lambda j: -res_norm[j])[:5]:
        print(f'    j={j:3d}  norm={res_norm[j]:.4f}')
    print('  top-5 non-helpers by residual norm:')
    for j in sorted(non_helper, key=lambda j: -res_norm[j])[:5]:
        print(f'    j={j:3d}  norm={res_norm[j]:.4f}')


if __name__ == '__main__':
    main()
