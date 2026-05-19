"""
cross_retarget_internals.py — Extract per-target W, per-target bind_pose, and
per-frame joint transforms across 4 retargeting combinations:
   (ict src, ict tgt), (ict src, mf tgt), (mf src, ict tgt), (mf src, mf tgt)

For each (src_topo, src_id, tgt_topo, tgt_id) tuple:
  W           = model skin weight for tgt identity     [V_tgt, J]   (same across frames)
  joint_pos   = model bind_pose_net for tgt identity   [J, 3]
  T_world     = FK chain per source frame              [n_frames, J, 4, 4]
  pred_lbs    = retargeted prediction                  [n_frames, V_tgt, 3]

Source animation: per-id deformed sequence sampled from the same
source identity's expression history (n_frames=90 by default).

Outputs saved under {out_dir}/{src_topo}_id{src_id}__{tgt_topo}_id{tgt_id}/.

Usage:
    python tools/analyze/cross_retarget_internals.py \
        --ckpt_dir ckpts_hlbs/2026-05-14-14-02-03-HLBS-FullPred-ict-jTrans-nrm0.1-Wsm0.01 \
        --epoch 200 \
        --src_ict_id 2 --src_mf_id 12 \
        --tgt_ict_id 2 --tgt_mf_id 12 \
        --n_frames 90
"""
import os
import sys
import argparse
import pickle
import numpy as np
import torch
import igl
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

from utils.rig_loader import load_rig
from models.hierarchical_lbs import HierarchicalLBS_FullPred


# ── Mesh loaders ──────────────────────────────────────────────────────────

def _load_neutral(topo, identity_idx, data_basedir='/data/sihun'):
    if topo == 'ict':
        from utils.remesh_utils import ICT_face_model
        m = ICT_face_model()
        iden_vecs = torch.load('ict_face_pt/ict_id_vecs_test.pt', weights_only=False).numpy()
        id_coeff = iden_vecs[identity_idx]
        id_disps = m.get_id_disp(id_coeff).squeeze()
        V = (m.neutral_verts + id_disps).astype(np.float32)
        F = m.faces.astype(np.int32)
        return V, F, f'ict_{identity_idx:03d}'
    if topo == 'mf':
        pkl = os.path.join(data_basedir, 'multiface_align', 'mf_templates.pkl')
        if not os.path.exists(pkl):
            pkl = os.path.join(data_basedir, 'pca', 'multiface_align', 'mf_templates.pkl')
        with open(pkl, 'rb') as f:
            t = pickle.load(f)
        keys = [k for k in t if k != 'face']
        id_name = keys[identity_idx]
        V = np.array(t[id_name], dtype=np.float32)
        F = np.array(t['face'], dtype=np.int32)
        return V, F, id_name
    raise ValueError(f'unknown topo: {topo}')


def _load_deformed_sequence(topo, identity_idx, n_frames, data_basedir='/data/sihun'):
    """Pick n_frames deformed vertex sequences for the given identity.
    For ICT: synthesize via random expressions (or load ict-cap if available).
    For MF:  load from /data/sihun/multiface_align/{ROM|SEN}/.../vertices_npy/.
    """
    if topo == 'mf':
        # Try MF SEN/ROM sequences
        keys_pkl = os.path.join(data_basedir, 'multiface_align', 'mf_templates.pkl')
        if not os.path.exists(keys_pkl):
            keys_pkl = os.path.join(data_basedir, 'pca', 'multiface_align', 'mf_templates.pkl')
        with open(keys_pkl, 'rb') as f:
            t = pickle.load(f)
        keys = [k for k in t if k != 'face']
        id_name = keys[identity_idx]
        import glob as _glob
        for split in ('ROM', 'SEN'):
            for sub in ('train', 'test'):
                vd = os.path.join(data_basedir, 'multiface_align', split, sub, 'vertices_npy', id_name)
                if not os.path.isdir(vd):
                    continue
                # Frames may live under EXP_xxx/ subdirectories or directly as .npy files.
                paths = sorted(_glob.glob(os.path.join(vd, '**', '*.npy'), recursive=True))
                paths = [p for p in paths if os.path.isfile(p)][:n_frames]
                if paths:
                    return np.stack([np.load(p) for p in paths]).astype(np.float32)
        raise RuntimeError(f'MF deformed sequence not found for {id_name}')
    if topo == 'ict':
        # Synthesize ICT expression sequence via random AU coefficients
        from utils.remesh_utils import ICT_face_model
        m = ICT_face_model()
        iden_vecs = torch.load('ict_face_pt/ict_id_vecs_test.pt', weights_only=False).numpy()
        id_coeff = iden_vecs[identity_idx]
        rng = np.random.RandomState(42)
        seq = []
        for _ in range(n_frames):
            exp_coeff = rng.random(53).astype(np.float32)
            v_def, _, _ = m.apply_coeffs(id_coeff, exp_coeff, return_all=True, region=0)
            seq.append(v_def[0])
        return np.stack(seq).astype(np.float32)


# ── Model construction (mirrors eval_hlbs.py: ckpt peek + helper_joint_idx) ──

def build_model_from_ckpt(ckpt_dir, epoch, device):
    opts_path = os.path.join(ckpt_dir, 'train_opts.yml')
    with open(opts_path) as f:
        opts = yaml.safe_load(f)
    rig = load_rig(opts['rig_path'])

    # Peek state_dict to get exact face_joint_idx and helper_joint_idx
    ckpt_path = (os.path.join(ckpt_dir, 'model_hlbs_best.pth') if str(epoch) == 'best'
                 else os.path.join(ckpt_dir, f'model_hlbs_{int(epoch):03d}.pth'))
    sd_peek = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    face_idx = sd_peek['face_joint_idx'].tolist() if 'face_joint_idx' in sd_peek else None
    helper_idx = (sd_peek['helper_joint_idx_buf'].tolist()
                  if 'helper_joint_idx_buf' in sd_peek else None)

    base_idx = None
    aj_path = opts.get('active_joints_json')
    if aj_path and os.path.exists(aj_path):
        import json
        with open(aj_path) as f:
            aj = json.load(f)
        base_idx = aj.get('base_joint_idx')

    sig = None
    if opts.get('sigma_targets_npy') and os.path.exists(opts['sigma_targets_npy']):
        sig = np.load(opts['sigma_targets_npy']).astype(np.float32)

    model = HierarchicalLBS_FullPred(
        rig=rig, topology=opts.get('topo_key', 'mf'),
        in_dim_exp=12, hid_dim=opts.get('hid_dim', 128), num_layers=opts.get('num_layers', 4),
        device=str(device), use_joint_trans=opts.get('use_joint_trans', False),
        dfn_skin=opts.get('dfn_skin', False), dfn_bind=opts.get('dfn_bind', False),
        dfn_exp=opts.get('dfn_exp', False),
        nfs_feat_dim=256 if opts.get('nfs_feat_dir') else 0,
        nfs_concat=opts.get('nfs_concat', False), adain_pos_norm=opts.get('adain_pos_norm', False),
        freeze_bind_pose=opts.get('freeze_bind_pose', False),
        use_gmm_hybrid=opts.get('use_gmm_hybrid', False),
        init_log_sigma=opts.get('init_log_sigma', -1.2),
        gmm_mode=opts.get('gmm_mode', 'additive'),
        residual_scale=opts.get('residual_scale', 2.0),
        sigma_targets=sig, bind_pose_mode=opts.get('bind_pose_mode', 'net'),
        face_joint_idx=face_idx, base_joint_idx=base_idx,
        face_mask_r0=opts.get('face_mask_r0', 1.0), face_mask_r1=opts.get('face_mask_r1', 2.25),
        helper_joint_idx=helper_idx,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()
    return model, opts


def _load_nfs_feat(id_name, nfs_dir, device):
    p = os.path.join(nfs_dir, f'{id_name}_nfs_feat.npy')
    if not os.path.exists(p):
        return None
    return torch.from_numpy(np.load(p)).float().to(device)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--epoch', default='200')
    ap.add_argument('--src_ict_id', type=int, default=2)
    ap.add_argument('--src_mf_id', type=int, default=12)
    ap.add_argument('--tgt_ict_id', type=int, default=2)
    ap.add_argument('--tgt_mf_id', type=int, default=12)
    ap.add_argument('--n_frames', type=int, default=90)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out_dir', default=None,
                    help='Default: {ckpt_dir}/cross_retarget_internals_{epoch}')
    ap.add_argument('--data_basedir', default='/data/sihun')
    args = ap.parse_args()

    device = torch.device(args.device)
    model, opts = build_model_from_ckpt(args.ckpt_dir, args.epoch, device)
    nfs_dir = opts.get('nfs_feat_dir')

    out_dir = args.out_dir or os.path.join(args.ckpt_dir, f'cross_retarget_internals_{args.epoch}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'out_dir: {out_dir}')

    cases = [
        ('ict', args.src_ict_id, 'ict', args.tgt_ict_id),
        ('ict', args.src_ict_id, 'mf',  args.tgt_mf_id),
        ('mf',  args.src_mf_id,  'ict', args.tgt_ict_id),
        ('mf',  args.src_mf_id,  'mf',  args.tgt_mf_id),
    ]

    for src_topo, src_id, tgt_topo, tgt_id in cases:
        print(f'\n── {src_topo}:{src_id} → {tgt_topo}:{tgt_id} ──')
        src_neu_np, src_F, src_name = _load_neutral(src_topo, src_id, args.data_basedir)
        tgt_neu_np, tgt_F, tgt_name = _load_neutral(tgt_topo, tgt_id, args.data_basedir)
        src_def_seq = _load_deformed_sequence(src_topo, src_id, args.n_frames, args.data_basedir)
        print(f'   src {src_name} V={src_neu_np.shape[0]}  tgt {tgt_name} V={tgt_neu_np.shape[0]}'
              f'  src_def_seq {src_def_seq.shape}')

        src_neu_n = igl.per_vertex_normals(src_neu_np, src_F).astype(np.float32)
        tgt_neu_n = igl.per_vertex_normals(tgt_neu_np, tgt_F).astype(np.float32)

        # Tensors
        src_neu = torch.from_numpy(src_neu_np).unsqueeze(0).to(device)               # [1, V_s, 3]
        src_neu_norm = torch.from_numpy(src_neu_n).unsqueeze(0).to(device)
        tgt_neu = torch.from_numpy(tgt_neu_np).unsqueeze(0).to(device)               # [1, V_t, 3]
        tgt_neu_norm = torch.from_numpy(tgt_neu_n).unsqueeze(0).to(device)

        # NFS feat for target
        tgt_nfs = _load_nfs_feat(tgt_name, nfs_dir, device) if nfs_dir else None
        tgt_nfs = tgt_nfs.unsqueeze(0) if tgt_nfs is not None else None
        if tgt_nfs is not None:
            print(f'   tgt NFS feat: {tgt_nfs.shape}')

        # Identity (target) — once: W and bind_pose
        with torch.no_grad():
            tgt_skin_input, tgt_adain = model._prepare_feat(tgt_neu, tgt_neu_norm, tgt_nfs)
            B_inv_tgt, tgt_joint_pos = model._get_bind_pose(tgt_skin_input, adain_input=tgt_adain)
            W_tgt, _ = model._get_skinning_weights(
                tgt_skin_input, adain_input=tgt_adain,
                source_vert=tgt_neu, joint_pos=tgt_joint_pos)

        # Per-frame joint transform (source-driven) — batched
        n_frames = src_def_seq.shape[0]
        src_def_t = torch.from_numpy(src_def_seq).to(device)                          # [F, V_s, 3]
        # Per-frame normals
        src_def_norm = np.stack([igl.per_vertex_normals(v, src_F).astype(np.float32)
                                  for v in src_def_seq])
        src_def_n_t = torch.from_numpy(src_def_norm).to(device)                       # [F, V_s, 3]

        T_world_list = []
        pred_lbs_list = []
        chunk = 8
        with torch.no_grad():
            for s in range(0, n_frames, chunk):
                e = min(s + chunk, n_frames)
                B = e - s
                _src_neu = src_neu.expand(B, -1, -1)
                _src_neu_n = src_neu_norm.expand(B, -1, -1)
                _src_def = src_def_t[s:e]
                _src_def_n = src_def_n_t[s:e]
                _tgt_neu = tgt_neu.expand(B, -1, -1)
                _tgt_neu_n = tgt_neu_norm.expand(B, -1, -1)
                _tgt_nfs = tgt_nfs.expand(B, -1, -1) if tgt_nfs is not None else None
                # Retarget (returns final pred). For T_world we replicate inline by
                # calling the same code path manually to extract T_world.
                pred = model.retarget(_src_neu, _src_neu_n, _src_def, _src_def_n,
                                       _tgt_neu, _tgt_neu_n, tgt_nfs_feat=_tgt_nfs)
                pred_lbs_list.append(pred.cpu().numpy())

                # Re-derive T_world (lbs_exp_z + lbs_pose) for save
                delta_src = _src_def - _src_neu
                src_in    = torch.cat([_src_neu, _src_neu_n], dim=-1)
                deform_in = torch.cat([delta_src, _src_def_n, src_in], dim=-1)
                z_exp = model.lbs_exp_z_model(deform_in)
                z_exp_flat = z_exp.squeeze(1) if z_exp.dim() == 3 else z_exp
                pose_out = model.lbs_pose_model(z_exp_flat.unsqueeze(1)).squeeze(1)
                J = model.num_joints
                if model.use_joint_trans:
                    rot6d = pose_out[:, :J*6].reshape(B*J, 6)
                    local_t = pose_out[:, J*6:].reshape(B, J, 3, 1)
                else:
                    rot6d = pose_out.reshape(B*J, 6)
                    local_t = torch.zeros(B, J, 3, 1, device=device, dtype=pred.dtype)
                local_R = model._rot6d(rot6d).reshape(B, J, 3, 3)
                top = torch.cat([local_R, local_t], dim=-1)
                bot = torch.cat([torch.zeros(B, J, 1, 3, device=device, dtype=pred.dtype),
                                  torch.ones(B, J, 1, 1, device=device, dtype=pred.dtype)],
                                  dim=-1)
                T_local = torch.cat([top, bot], dim=-2)
                T_world = model._chain_hierarchy(T_local)
                T_world_list.append(T_world.cpu().numpy())

        pred_lbs = np.concatenate(pred_lbs_list, axis=0)
        T_world  = np.concatenate(T_world_list,  axis=0)

        # Save
        case_dir = os.path.join(out_dir, f'{src_topo}_id{src_id}__{tgt_topo}_id{tgt_id}')
        os.makedirs(case_dir, exist_ok=True)
        np.save(os.path.join(case_dir, 'W_tgt.npy'),       W_tgt[0].cpu().numpy())
        np.save(os.path.join(case_dir, 'joint_pos_tgt.npy'), tgt_joint_pos[0].cpu().numpy())
        np.save(os.path.join(case_dir, 'T_world.npy'),     T_world)
        np.save(os.path.join(case_dir, 'pred_lbs.npy'),    pred_lbs)
        # Reference info
        np.save(os.path.join(case_dir, 'tgt_neutral_verts.npy'), tgt_neu_np)
        np.save(os.path.join(case_dir, 'tgt_faces.npy'),         tgt_F)
        np.save(os.path.join(case_dir, 'src_def_seq.npy'),       src_def_seq)
        with open(os.path.join(case_dir, 'meta.txt'), 'w') as f:
            f.write(f'src_topo={src_topo}  src_id={src_id}  src_name={src_name}\n'
                    f'tgt_topo={tgt_topo}  tgt_id={tgt_id}  tgt_name={tgt_name}\n'
                    f'n_frames={n_frames}\n'
                    f'W: {tuple(W_tgt.shape)}\n'
                    f'joint_pos: {tuple(tgt_joint_pos.shape)}\n'
                    f'T_world: {T_world.shape}\n')
        print(f'   W={W_tgt.shape[1:]}  joint_pos={tgt_joint_pos.shape[1:]}  '
              f'T_world={T_world.shape}  pred_lbs={pred_lbs.shape}')
        print(f'   saved → {case_dir}/')

    print(f'\nDone.')


if __name__ == '__main__':
    main()
