"""
eval_hlbs.py — Evaluate a trained HierarchicalLBS checkpoint.

Self-retargeting : reconstruction metrics on EvalDataset (real test frames).
Cross-retargeting: transfer source expression to a different target identity.

Metrics (self-retarget only):
    - MSE (overall / inner / outer face regions)
    - Per-vertex L2 (mean, median, p95, p99, max, max_mean)
    - Laplacian smoothness error  [Neural Cages, Yifan et al. CVPR 2020]
    - Normal consistency          [Occupancy Networks, Mescheder et al. CVPR 2019]
    - Edge length distortion      [ARAP, Sorkine & Alexa SGP 2007]

Usage (self-retarget):
    python eval_hlbs.py \
        --ckpt ckpts_hlbs/2025-...-HLBS-mf-s16 \
        --rig_path utils/mf/rig_info.json \
        --use_data1 --data_toggle --batch_size 1

Usage (cross-retarget):
    python eval_hlbs.py \
        --ckpt ckpts_hlbs/2025-...-HLBS-mf-s16 \
        --rig_path utils/mf/rig_info.json \
        --use_data1 --data_toggle --batch_size 1 \
        --cross_retarget \
        --tgt_vert_path /data/.../tgt_verts.npy \
        --tgt_norm_path /data/.../tgt_normals.npy \
        --tgt_obj_path  /data/.../tgt_mesh.obj
"""
import os
import sys
import json
import argparse
import glob
import random
import numpy as np
import yaml
import cv2
import trimesh

import torch
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
import igl

sys.path.insert(0, os.path.dirname(__file__))
from utils.matplotlib_rnd import plot_image_array
from dataloader_CBD import EvalDataset, CBD_collate_wrapper_eval
from utils.exp_utils import plateau_hat_points
from utils.mesh_utils import calc_norm_torch


# ── CLI ─────────────────────────────────────────────────────────────────────

def Options():
    parser = argparse.ArgumentParser(description='Evaluate HierarchicalLBS')

    # rig / topology (defaults filled from train_opts.yml if available)
    parser.add_argument("--rig_path", type=str, default=None)
    parser.add_argument("--topo_key", type=str, default=None,
                        choices=['mf', 'biwi', 'voca', 'ict'])
    parser.add_argument("--num_identities", type=int, default=None)
    parser.add_argument("--hid_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)

    # ablation flags (must match training)
    parser.add_argument("--freeze_adapt", dest='freeze_adapt', action='store_true')
    parser.set_defaults(freeze_adapt=False)
    parser.add_argument("--use_joint_trans", dest='use_joint_trans', action='store_true')
    parser.set_defaults(use_joint_trans=False)
    parser.add_argument("--full_prediction", dest='full_prediction', action='store_true')
    parser.set_defaults(full_prediction=False)

    # target (for smooth_gt computation)
    parser.add_argument("--target", type=str, default='gt',
                        choices=['gt', 'smooth_gt'])
    parser.add_argument("--smooth_n_iter", type=int, default=16)

    # checkpoint
    parser.add_argument("--ckpt", type=str, required=True,
                        help='Checkpoint directory (must contain model_hlbs_*.pth)')
    parser.add_argument("--start_epoch", type=int, default=-1,
                        help='Epoch to load (-1 = best)')
    parser.add_argument("--best_epoch_tag", type=int, default=None,
                        help='When loading best (--start_epoch -1), tag the output dir '
                             'as e_best{N:03d} so re-eval at a new best epoch does not '
                             'overwrite the previous "best" outputs.')

    # data
    parser.add_argument("--data_selection", type=str, default='mf_ROM',
                        choices=['voca', 'biwi', 'mf_SEN', 'coma', 'mf_ROM', 'ict', 'ict-cap'],
                        help='Dataset to evaluate on')
    parser.add_argument("--src_identity", type=int, default=-1,
                        help='Source identity index (-1 = all identities)')
    parser.add_argument("--exp_num", type=int, default=0,
                        help='Expression sequence number for ict-cap (0 or 1)')
    parser.add_argument("--data_toggle", dest='data_toggle', action='store_true')
    parser.set_defaults(data_toggle=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # mask
    parser.add_argument("--no_t_mask", dest='no_t_mask', action='store_true')
    parser.set_defaults(no_t_mask=False)
    parser.add_argument("--use_t_mask", dest='no_t_mask', action='store_false')

    # cross-retargeting
    parser.add_argument("--cross_retarget", dest='cross_retarget', action='store_true')
    parser.set_defaults(cross_retarget=False)
    parser.add_argument("--tgt_vert_path", type=str, default=None,
                        help='Target neutral vertices .npy [V, 3]')
    parser.add_argument("--tgt_norm_path", type=str, default=None,
                        help='Target neutral normals  .npy [V, 3]')
    parser.add_argument("--tgt_obj_path", type=str, default=None,
                        help='Target neutral mesh     .obj (for face topology)')
    parser.add_argument("--tgt_dataset", type=str, default=None,
                        choices=['mf', 'voca', 'biwi', 'coma', 'ict'],
                        help='Target dataset for cross-retarget (auto-loads from pkl)')
    parser.add_argument("--tgt_identity", type=int, default=0,
                        help='Target identity index within the dataset')

    # output
    parser.add_argument("--log_dir", type=str, default="eval_hlbs",
                        help='Base output directory (default: eval_hlbs)')
    parser.add_argument("--save_vert", dest='save_vert', action='store_true')
    parser.set_defaults(save_vert=False)
    parser.add_argument("--save_gt", dest='save_gt', action='store_true')
    parser.set_defaults(save_gt=False)
    parser.add_argument("--save_obj", dest='save_obj', action='store_true')
    parser.set_defaults(save_obj=False)
    parser.add_argument("--make_video", dest='make_video', action='store_true')
    parser.add_argument("--no_video", dest='make_video', action='store_false')
    parser.set_defaults(make_video=True)
    parser.add_argument("--no_vis", dest='no_vis', action='store_true',
                        help='Skip image rendering (metrics only)')
    parser.set_defaults(no_vis=False)
    parser.add_argument("--vis_every", type=int, default=1,
                        help='Render image every N batches (1 = all frames)')

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_basedir", type=str, default="/data/sihun",
                        help='Base directory for datasets')
    parser.add_argument("--nfs_ckpt", type=str, default="ckpts_comparison/NFS-best",
                        help='NFS checkpoint for online seg feature extraction (unseen mesh)')

    opts = parser.parse_args()

    # ── Auto-load model config from train_opts.yml ──────────────────────
    train_opts_path = os.path.join(opts.ckpt, "train_opts.yml")
    if os.path.exists(train_opts_path):
        with open(train_opts_path) as f:
            train_opts = yaml.safe_load(f)
        # Model args: fill from yml only if not explicitly set via CLI
        model_keys = ['rig_path', 'topo_key', 'num_identities', 'hid_dim',
                      'num_layers', 'freeze_adapt', 'use_joint_trans', 'full_prediction',
                      'nfs_feat_dir', 'nfs_feat_dim', 'nfs_proj_dim', 'init_log_sigma',
                      'bind_pose_base_residual']
        for key in model_keys:
            if key in train_opts and getattr(opts, key, None) is None:
                setattr(opts, key, train_opts[key])
        # Boolean flags: always inherit from yml (CLI store_true can't distinguish default)
        for key in ['freeze_adapt', 'use_joint_trans', 'full_prediction',
                     'nfs_concat', 'adain_pos_norm', 'dfn_skin', 'dfn_bind', 'dfn_exp',
                     'freeze_bind_pose', 'use_gmm_hybrid']:
            if key in train_opts and f'--{key}' not in sys.argv:
                setattr(opts, key, train_opts[key])
        print(f"[eval] Loaded model config from: {train_opts_path}")

    # Final defaults if still None
    if opts.rig_path is None:
        opts.rig_path = 'maya_rig'
    if opts.topo_key is None:
        opts.topo_key = 'mf'
    if opts.hid_dim is None:
        opts.hid_dim = 256
    if opts.num_layers is None:
        opts.num_layers = 4
    if opts.num_identities is None:
        opts.num_identities = 13
    # NFS feat defaults
    if not hasattr(opts, 'nfs_feat_dir') or opts.nfs_feat_dir is None:
        opts.nfs_feat_dir = None
    for bkey in ['nfs_concat', 'adain_pos_norm', 'dfn_skin', 'dfn_bind', 'dfn_exp']:
        if not hasattr(opts, bkey):
            setattr(opts, bkey, False)

    return opts


# ── Utilities ───────────────────────────────────────────────────────────────

def images_to_video(img_dir, out_path, fps=30):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if len(imgs) == 0:
        print(f"[WARN] No images in {img_dir}, skipping video.")
        return
    first = cv2.imread(imgs[0])
    h, w, _ = first.shape
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in imgs:
        writer.write(cv2.imread(p))
    writer.release()
    print(f"Video saved: {out_path}")


def _write_obj(path, verts, faces):
    with open(path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# ── Metric helpers ──────────────────────────────────────────────────────────

def _build_cot_laplacian(verts_np, faces_np):
    """Cotangent Laplacian via libigl [Sorkine et al. SGP 2004]."""
    return igl.cotmatrix(verts_np.astype(np.float64), faces_np.astype(np.int32))


def _laplacian_error(L_sp, pred, gt):
    """||L*pred - L*gt||^2 per vertex, averaged [Neural Cages, CVPR 2020]."""
    B = pred.shape[0]
    total = 0.0
    for b in range(B):
        Lp = L_sp @ pred[b].numpy().astype(np.float64)
        Lg = L_sp @ gt[b].numpy().astype(np.float64)
        total += float(np.mean(np.sum((Lp - Lg) ** 2, axis=-1)))
    return total / B


def _normal_consistency(pred, gt, faces):
    """1 - cos(n_pred, n_gt), averaged [Occupancy Networks, CVPR 2019]."""
    pred_n = calc_norm_torch(pred, faces)
    gt_n   = calc_norm_torch(gt, faces)
    cos_sim = F.cosine_similarity(pred_n, gt_n, dim=-1)
    return (1.0 - cos_sim).mean().item()


def _edge_length_distortion(pred, gt, edges):
    """mean |e_pred - e_gt| / e_gt  [ARAP, Sorkine & Alexa SGP 2007]."""
    e0, e1 = edges[:, 0], edges[:, 1]
    len_pred = torch.sqrt(((pred[:, e0] - pred[:, e1]) ** 2).sum(dim=-1))
    len_gt   = torch.sqrt(((gt[:, e0]   - gt[:, e1])   ** 2).sum(dim=-1))
    return ((len_pred - len_gt).abs() / (len_gt + 1e-8)).mean().item()


def _build_edges(faces_np):
    """Unique undirected edges from face array [F, 3]."""
    e = np.concatenate([faces_np[:, [0,1]], faces_np[:, [1,2]], faces_np[:, [0,2]]], axis=0)
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    return torch.tensor(e, dtype=torch.long)


def _per_segment_mse(pred, gt, seg_labels):
    """Per-segment MSE.
    Args:
        pred, gt: [B, V, 3]
        seg_labels: [V, S] one-hot segment labels
    Returns:
        dict {seg_idx: mse}, overall_mean
    """
    seg_idx = seg_labels.argmax(dim=-1)  # [V]
    S = seg_labels.shape[-1]
    per_seg = {}
    for s in range(S):
        mask = (seg_idx == s)
        if mask.sum() == 0:
            continue
        per_seg[s] = F.mse_loss(pred[:, mask, :], gt[:, mask, :]).item()
    overall = np.mean(list(per_seg.values()))
    return per_seg, overall


def _temporal_smoothness(pred_sequence):
    """Acceleration-based jitter: mean ||v(t+1) - 2*v(t) + v(t-1)||².
    Args:
        pred_sequence: [T, V, 3]
    Returns:
        float (mean jitter)
    """
    if pred_sequence.shape[0] < 3:
        return 0.0
    accel = pred_sequence[2:] - 2 * pred_sequence[1:-1] + pred_sequence[:-2]
    return (accel ** 2).sum(dim=-1).mean().item()


# ── Evaluator ───────────────────────────────────────────────────────────────

class HLBSEvaluator:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device(opts.device if torch.cuda.is_available() else 'cpu')

        torch.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)

        # ── Load model ──────────────────────────────────────────────────
        from utils.rig_loader import load_rig
        from models.hierarchical_lbs import HierarchicalLBS, HierarchicalLBS_FullPred

        rig = load_rig(opts.rig_path)

        # ── Mirror trainer's config preprocessing ────────────────────────
        # (active_joints / sigma_targets / anchor_pool assets) so checkpoint
        # buffers (face_joint_idx, log_sigma_target, joint_anchors, …) line up.
        _face_joint_idx = None; _base_joint_idx = None
        if getattr(opts, 'active_joints_json', None):
            import json as _json
            with open(opts.active_joints_json) as _f:
                _aj = _json.load(_f)
            _face_joint_idx = _aj['face_joint_idx']
            _base_joint_idx = _aj['base_joint_idx']

        # Peek at checkpoint state_dict early to recover the EXACT
        # face_joint_idx and helper_joint_idx the ckpt was trained with —
        # avoids size mismatch when active_joints_json has been modified since
        # training, and ensures helper-aware models get helper buffer registered.
        if str(opts.start_epoch) == 'best':
            _ckpt_peek_path = os.path.join(opts.ckpt, "model_hlbs_best.pth")
        else:
            _ckpt_peek_path = os.path.join(opts.ckpt, f"model_hlbs_{int(opts.start_epoch):03d}.pth")
        _helper_idx_ckpt = None
        if os.path.isfile(_ckpt_peek_path):
            _sd_peek = torch.load(_ckpt_peek_path, map_location='cpu', weights_only=False)
            if 'face_joint_idx' in _sd_peek:
                _face_joint_idx = _sd_peek['face_joint_idx'].tolist()
                print(f"[eval] using face_joint_idx from ckpt ({len(_face_joint_idx)} joints)")
            if 'helper_joint_idx_buf' in _sd_peek:
                _helper_idx_ckpt = _sd_peek['helper_joint_idx_buf'].tolist()
                print(f"[eval] using helper_joint_idx from ckpt "
                      f"({len(_helper_idx_ckpt)} helpers — residual reparameterization active)")

        _sigma_targets = None
        if getattr(opts, 'sigma_targets_npy', None):
            _sigma_targets = np.load(opts.sigma_targets_npy).astype(np.float32)

        _joint_anchors = None; _joint_offsets = None
        if getattr(opts, 'bind_pose_mode', 'net') == 'anchor_pool':
            if getattr(opts, 'joint_anchors_npy', None) and getattr(opts, 'joint_offsets_npy', None):
                _joint_anchors = np.load(opts.joint_anchors_npy).astype(np.float32)
                _joint_offsets = np.load(opts.joint_offsets_npy).astype(np.float32)

        if opts.full_prediction:
            self.model = HierarchicalLBS_FullPred(
                rig=rig,
                topology=getattr(opts, 'topo_key', 'mf'),
                in_dim_exp=12,
                hid_dim=opts.hid_dim,
                num_layers=opts.num_layers,
                device=str(self.device),
                use_joint_trans=opts.use_joint_trans,
                smooth_W=getattr(opts, 'smooth_delta_W', 0),
                smooth_W_alpha=getattr(opts, 'smooth_delta_W_alpha', 0.5),
                dfn_skin=opts.dfn_skin,
                dfn_bind=opts.dfn_bind,
                dfn_exp=opts.dfn_exp,
                nfs_feat_dim=(int(getattr(opts,'nfs_feat_dim',256) or 256) if opts.nfs_feat_dir else 0),
                nfs_proj_dim=int(getattr(opts,'nfs_proj_dim',0) or 0),
                nfs_concat=getattr(opts, 'nfs_concat', False),
                adain_pos_norm=getattr(opts, 'adain_pos_norm', False),
                freeze_bind_pose=getattr(opts, 'freeze_bind_pose', False),
                use_gmm_hybrid=getattr(opts, 'use_gmm_hybrid', False),
                init_log_sigma=getattr(opts, 'init_log_sigma', -1.2),
                gmm_mode=getattr(opts, 'gmm_mode', 'additive'),
                residual_scale=getattr(opts, 'residual_scale', 2.0),
                sigma_targets=_sigma_targets,
                bind_pose_mode=getattr(opts, 'bind_pose_mode', 'net'),
                joint_anchors=_joint_anchors,
                joint_offsets=_joint_offsets,
                attn_temperature_init=getattr(opts, 'attn_temperature_init', 0.1),
                face_joint_idx=_face_joint_idx,
                base_joint_idx=_base_joint_idx,
                face_mask_r0=getattr(opts, 'face_mask_r0', 1.0),
                face_mask_r1=getattr(opts, 'face_mask_r1', 2.25),
                helper_joint_idx=_helper_idx_ckpt,
                bind_pose_base_residual=bool(getattr(opts, 'bind_pose_base_residual', 0)),
            ).to(self.device)
        else:
            self.model = HierarchicalLBS(
                rig=rig,
                topology=opts.topo_key,
                in_dim_exp=12,
                hid_dim=opts.hid_dim,
                num_layers=opts.num_layers,
                device=str(self.device),
                freeze_adapt=opts.freeze_adapt,
                use_joint_trans=opts.use_joint_trans,
            ).to(self.device)

        # resolve checkpoint
        if opts.start_epoch < 0:
            ckpt_path = os.path.join(opts.ckpt, "model_hlbs_best.pth")
        else:
            ckpt_path = os.path.join(opts.ckpt, f"model_hlbs_{opts.start_epoch:03d}.pth")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        _msg = self.model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device), strict=False)
        if _msg.missing_keys or _msg.unexpected_keys:
            print(f"[eval load] missing={len(_msg.missing_keys)} "
                  f"unexpected={len(_msg.unexpected_keys)}")
            if _msg.missing_keys:
                print(f"  missing: {_msg.missing_keys[:5]}{'...' if len(_msg.missing_keys)>5 else ''}")
            if _msg.unexpected_keys:
                print(f"  unexpected: {_msg.unexpected_keys[:5]}{'...' if len(_msg.unexpected_keys)>5 else ''}")
        self.model.eval()
        print(f"Loaded: {ckpt_path}")

        # ── Per-topo geodesic dist tables (mirror training) ──────────────
        self._geo_dist_per_topo = {}
        if getattr(opts, 'use_geodesic_gauss', False):
            _gd_dir = getattr(opts, 'geo_dist_dir', None) or opts.rig_path
            for _topo in ('ict', 'mf', 'biwi', 'coma'):
                _p = os.path.join(_gd_dir, f'geo_dist_{_topo}.npy')
                if os.path.exists(_p):
                    self._geo_dist_per_topo[_topo] = torch.from_numpy(
                        np.load(_p)).to(self.device).float()
            if self._geo_dist_per_topo:
                _msg = ', '.join(f'{k}{tuple(v.shape)}'
                                 for k, v in self._geo_dist_per_topo.items())
                print(f"[geo_dist] eval loaded: {_msg} from {_gd_dir}")

        # ── Per-id bind_pos cache (anchor_pool / freeze_bind_pose) ───────
        self._bind_pos_cache = {}
        _cache_dir = getattr(opts, 'bind_pos_cache_dir', None)
        if _cache_dir is None and getattr(opts, 'bind_pose_mode', 'net') == 'anchor_pool':
            _cache_dir = getattr(opts, 'nfs_feat_dir', None)
        if _cache_dir:
            import glob as _glob
            for fp in _glob.glob(os.path.join(_cache_dir, '*_bind_pos.npy')):
                k = os.path.basename(fp).replace('_bind_pos.npy', '')
                self._bind_pos_cache[k] = torch.tensor(
                    np.load(fp), dtype=torch.float32).to(self.device)
            if self._bind_pos_cache:
                print(f"[bind_pos cache] eval loaded {len(self._bind_pos_cache)} per-id from {_cache_dir}")

        # ── Load NFS features ──────────────────────────────────────────
        self._nfs_feat_cache = {}
        self._nfs_model = None
        if opts.nfs_feat_dir:
            import glob as _glob
            feat_files = _glob.glob(os.path.join(opts.nfs_feat_dir, '*_nfs_feat.npy'))
            for fp in feat_files:
                fname = os.path.basename(fp).replace('_nfs_feat.npy', '')
                self._nfs_feat_cache[fname] = torch.tensor(
                    np.load(fp), dtype=torch.float32).to(self.device)
            print(f"[eval] NFS feat loaded: {len(self._nfs_feat_cache)} identities on GPU")

        # ── Output dir ──────────────────────────────────────────────────
        # eval_hlbs/{ckpt_basename}-eval/e050-self/mf_ROM/
        if opts.start_epoch < 0:
            epoch_tag = (f"best{opts.best_epoch_tag:03d}"
                        if opts.best_epoch_tag is not None else "best")
        else:
            epoch_tag = f"{opts.start_epoch:03d}"
        ckpt_basename = os.path.basename(os.path.normpath(opts.ckpt))
        if opts.cross_retarget:
            tgt_tag = f"{opts.tgt_dataset}_id{opts.tgt_identity}" if opts.tgt_dataset else "custom"
            mode_tag = f"cross-{tgt_tag}"
        else:
            mode_tag = "self"
        data_tag = opts.data_selection
        if opts.src_identity >= 0:
            data_tag += f"_id{opts.src_identity}"
        if opts.data_selection == 'ict-cap':
            data_tag += f"_exp{opts.exp_num}"
        self.out_dir = os.path.join(
            opts.log_dir, f"{ckpt_basename}-eval",
            f"e{epoch_tag}-{mode_tag}", data_tag)
        os.makedirs(self.out_dir, exist_ok=True)

        self.img_dir = os.path.join(self.out_dir, "img")
        os.makedirs(self.img_dir, exist_ok=True)

        if opts.save_vert:
            self.vert_dir = os.path.join(self.out_dir, "pred_vert")
            os.makedirs(self.vert_dir, exist_ok=True)
        if opts.save_gt:
            self.gt_dir = os.path.join(self.out_dir, "gt_vert")
            os.makedirs(self.gt_dir, exist_ok=True)

        # ── Load target mesh for cross-retarget ─────────────────────────
        self.tgt_neu_vert = None
        self.tgt_neu_norm = None
        self.tgt_faces = None
        if opts.cross_retarget:
            if opts.tgt_dataset:
                # Auto-load from dataset pkl
                tgt_verts, tgt_faces, tgt_id_name = self._load_tgt_from_dataset(
                    opts.tgt_dataset, opts.tgt_identity, opts.data_basedir)
                self._tgt_id_name = tgt_id_name
                tgt_normals = igl.per_vertex_normals(tgt_verts, tgt_faces).astype(np.float32)
                self.tgt_neu_vert = torch.from_numpy(tgt_verts).float().unsqueeze(0).to(self.device)
                self.tgt_neu_norm = torch.from_numpy(tgt_normals).float().unsqueeze(0).to(self.device)
                self.tgt_faces = tgt_faces
                self.tgt_faces_torch = torch.from_numpy(
                    tgt_faces).long().unsqueeze(0).to(self.device)
                print(f"Target: {opts.tgt_dataset}[{opts.tgt_identity}] ({tgt_id_name}) "
                      f"— {tgt_verts.shape[0]} verts, {tgt_faces.shape[0]} faces")
            else:
                assert opts.tgt_vert_path and opts.tgt_norm_path and opts.tgt_obj_path, \
                    "Cross-retarget requires --tgt_dataset or --tgt_vert_path/--tgt_norm_path/--tgt_obj_path"
                self.tgt_neu_vert = torch.from_numpy(
                    np.load(opts.tgt_vert_path)).float().unsqueeze(0).to(self.device)
                self.tgt_neu_norm = torch.from_numpy(
                    np.load(opts.tgt_norm_path)).float().unsqueeze(0).to(self.device)
                tgt_mesh = trimesh.load(opts.tgt_obj_path, process=False)
                if isinstance(tgt_mesh, trimesh.Scene):
                    tgt_mesh = trimesh.util.concatenate(tuple(tgt_mesh.geometry.values()))
                self.tgt_faces = tgt_mesh.faces
                self.tgt_faces_torch = torch.from_numpy(
                    self.tgt_faces).long().unsqueeze(0).to(self.device)
                print(f"Target mesh: {self.tgt_neu_vert.shape[1]} verts, "
                      f"{self.tgt_faces.shape[0]} faces")

            # Prepare target NFS feat for cross-retarget
            if opts.nfs_feat_dir:
                _tgt_id_name = getattr(self, '_tgt_id_name', None) or 'tgt_unknown'
                tgt_verts_np = self.tgt_neu_vert[0].cpu().numpy()
                tgt_faces_np = self.tgt_faces
                feat = self._get_nfs_feat(_tgt_id_name, tgt_verts_np, tgt_faces_np)
                if feat is not None:
                    self._tgt_nfs_feat = feat.unsqueeze(0)
                    print(f"[eval] Target NFS feat ready: {_tgt_id_name}")

    # ── Target mesh loader ──────────────────────────────────────────────

    @staticmethod
    def _load_tgt_from_dataset(dataset_name, identity_idx, data_basedir):
        """Load target neutral mesh from dataset pkl.

        Returns: (verts [V,3] float32, faces [F,3] int32, identity_name str)

        Available identities:
            mf:   0-12  (13 ids, 5223 verts)
            voca: 0-11  (12 ids, 3525 verts)
            biwi: 0-13  (14 ids, 2560 verts)
            coma: 0-11  (same as voca)
            ict:  0     (single template, 11248 verts)
        """
        import pickle

        if dataset_name == 'ict':
            from utils.remesh_utils import ICT_face_model
            ict_model = ICT_face_model()
            _test_pt = 'ict_face_pt/ict_id_vecs_test.pt'
            _train_npy = 'data/ICT_live_100/iden_vecs.npy'
            if os.path.exists(_test_pt):
                iden_vecs = torch.load(_test_pt, weights_only=False).numpy()
            elif os.path.exists(_train_npy):
                print(f"[WARN] ict_id_vecs_test.pt not found, falling back to train iden_vecs")
                iden_vecs = np.load(_train_npy)
            else:
                raise FileNotFoundError("No ICT identity vectors found")
            if identity_idx >= len(iden_vecs):
                raise ValueError(f"tgt_identity {identity_idx} out of range for ICT (max: {len(iden_vecs)-1})")
            id_coeff = iden_vecs[identity_idx]
            id_disps = ict_model.get_id_disp(id_coeff).squeeze()
            verts = (ict_model.neutral_verts + id_disps).astype(np.float32)
            faces = ict_model.faces.astype(np.int32)
            return verts, faces, f'ict_{identity_idx:03d}'

        # Repo-local pkl files (fallback when data_basedir doesn't have them)
        local_pkl_map = {
            'mf':   'utils/templates/mf_templates.pkl',
            'voca': 'utils/templates/voca_templates.pkl',
            'biwi': 'utils/templates/biwi_templates.pkl',
            'coma': 'utils/templates/voca_templates.pkl',
        }
        remote_pkl_map = {
            'mf':   f'{data_basedir}/multiface_align/mf_templates.pkl',
            'voca': f'{data_basedir}/VOCA-COMA/voca_templates.pkl',
            'biwi': f'{data_basedir}/BIWI_align_deci/templates_align_deci.pkl',
            'coma': f'{data_basedir}/VOCA-COMA/voca_templates.pkl',
        }
        # Try remote paths first, then pca subdirectory, then repo-local
        pkl_path = remote_pkl_map[dataset_name]
        if not os.path.exists(pkl_path):
            pkl_path = pkl_path.replace(data_basedir, f'{data_basedir}/pca')
        if not os.path.exists(pkl_path):
            pkl_path = local_pkl_map[dataset_name]

        with open(pkl_path, 'rb') as f:
            templates = pickle.load(f)

        faces = np.array(templates['face'], dtype=np.int32)
        id_names = [k for k in templates if k != 'face']

        if identity_idx >= len(id_names):
            raise ValueError(
                f"tgt_identity {identity_idx} out of range for {dataset_name} "
                f"(max: {len(id_names)-1}). Available: {id_names}")

        id_name = id_names[identity_idx]
        verts = np.array(templates[id_name], dtype=np.float32)
        return verts, faces, id_name

    # ── Online NFS feature extraction ──────────────────────────────────

    def _load_nfs_model(self):
        """Lazy-load NFS model for online feature extraction."""
        if self._nfs_model is not None:
            return
        import yaml as _yaml
        from models.NFS import NFS

        nfs_ckpt = self.opts.nfs_ckpt
        with open(os.path.join(nfs_ckpt, 'train_opts.yml')) as f:
            nfs_opts = _yaml.safe_load(f)

        class _Opts: pass
        opts = _Opts()
        for k, v in nfs_opts.items():
            setattr(opts, k, v)
        opts.device = str(self.device)
        opts.is_train = False

        self._nfs_model = NFS(opts=opts).to(self.device)
        ckpt_path = os.path.join(nfs_ckpt, 'model_best.pth')
        self._nfs_model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device, weights_only=False), strict=False)
        self._nfs_model.eval()
        print(f"[eval] NFS model loaded for online extraction: {ckpt_path}")

    @torch.no_grad()
    def _extract_seg_feat_online(self, verts_np, faces_np):
        """Extract seg_encoder per-vertex feature [V, 256] for unseen mesh."""
        self._load_nfs_model()
        from utils.nfr_utils import get_dfn_info

        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        dfn_info = get_dfn_info(mesh, map_location=self.device)

        verts_t = torch.tensor(verts_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        normals = igl.per_vertex_normals(verts_np, faces_np).astype(np.float32)
        norms_t = torch.tensor(normals, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Get img feat
        img = self._nfs_model.renderer.render_img(mesh).float().to(self.device)
        img_feat = self._nfs_model.get_img_feat(img).squeeze()
        img_feat_exp = img_feat.unsqueeze(0).unsqueeze(0).expand(1, verts_np.shape[0], -1)
        vert_feat = torch.cat([verts_t, norms_t, img_feat_exp], dim=-1)

        # Seg encoder forward (before last_lin)
        encoder = self._nfs_model.mesh_seg_encoder
        encoder.update_precomputes(dfn_info)
        dfn = encoder.dfn

        L = torch.sparse_coo_tensor(encoder.L_ind, encoder.L_val, encoder.L_size, device=self.device)
        batch_mass = encoder.mass.unsqueeze(0)
        batch_evals = encoder.evals.unsqueeze(0)
        batch_evecs = encoder.evecs.unsqueeze(0)
        gradX = [torch.sparse_coo_tensor(encoder.grad_X_ind, encoder.grad_X_val, encoder.grad_X_size, device=self.device)]
        gradY = [torch.sparse_coo_tensor(encoder.grad_Y_ind, encoder.grad_Y_val, encoder.grad_Y_size, device=self.device)]

        x = dfn.first_lin(vert_feat)
        for block in dfn.blocks:
            x = block(x, batch_mass, L=[L], evals=batch_evals, evecs=batch_evecs, gradX=gradX, gradY=gradY)

        return x[0]  # [V, 256] on GPU

    def _get_nfs_feat(self, id_name, verts_np=None, faces_np=None):
        """Get NFS seg feature: from cache if available, otherwise extract online."""
        if id_name in self._nfs_feat_cache:
            return self._nfs_feat_cache[id_name]

        # Online extraction for unseen identity
        if verts_np is not None and faces_np is not None and self.opts.nfs_feat_dir:
            print(f"[eval] Online seg feat extraction for: {id_name}")
            feat = self._extract_seg_feat_online(verts_np, faces_np)
            self._nfs_feat_cache[id_name] = feat  # cache for reuse
            return feat

        return None

    # ── Self-retargeting evaluation ─────────────────────────────────────

    def evaluate_self(self):
        opts = self.opts

        dataset = EvalDataset(data_name=opts.data_selection, toggle=opts.data_toggle, data_basedir=opts.data_basedir,
                              ict_cap_id_num=opts.src_identity if opts.src_identity >= 0 else 2,
                              ict_cap_exp_num=opts.exp_num)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opts.batch_size,
            collate_fn=partial(CBD_collate_wrapper_eval, device='cpu'),
            num_workers=0,
        )

        # lazily built from first batch
        L_sp = None
        edges = None

        # accumulators
        total = {
            "mse": 0.0, "mse_in": 0.0, "mse_out": 0.0,
            "l2": 0.0, "lap": 0.0, "norm_cos": 0.0, "edge_dist": 0.0,
            "l2_max_sum": 0.0,
        }
        n_batches = 0
        all_per_vertex_l2 = []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), ncols=120,
                    desc="Eval HLBS [self]")

        with torch.no_grad():
            for idx, batch in pbar:
                batch = batch.to(self.device)
                B = batch.vertices.shape[0]

                src_v = batch.template
                src_n = batch.template_normal
                gt_v  = batch.vertices
                gt_n  = batch.vertices_normal

                # Get NFS features if available
                _nfs_feat = None
                if self.opts.nfs_feat_dir and hasattr(batch, 'id_name'):
                    B_cur = src_v.shape[0]
                    N_cur = src_v.shape[1]
                    feats = []
                    _faces_np = batch.faces[0].cpu().numpy() if batch.faces.dim() == 3 else batch.faces.cpu().numpy()
                    for b in range(B_cur):
                        id_key = batch.id_name[b] if isinstance(batch.id_name, list) else batch.id_name
                        f = self._get_nfs_feat(
                            id_key,
                            verts_np=src_v[b].cpu().numpy(),
                            faces_np=_faces_np)
                        if f is not None:
                            if f.shape[0] > N_cur:
                                f = f[:N_cur]
                            feats.append(f)
                        else:
                            feats.append(torch.zeros(N_cur, 256, device=self.device))
                    _nfs_feat = torch.stack(feats, dim=0)

                # forward
                delta     = gt_v - src_v
                src_in    = torch.cat([src_v, src_n], dim=-1)
                deform_in = torch.cat([delta, gt_n, src_in], dim=-1)

                # Per-id bind_pos cache lookup (parity with training)
                _bind_pos_cache = None
                if self._bind_pos_cache:
                    _items = []; _all_hit = True
                    for b in range(src_v.shape[0]):
                        k = batch.id_name[b] if isinstance(batch.id_name, list) else batch.id_name
                        if k in self._bind_pos_cache:
                            _items.append(self._bind_pos_cache[k])
                        else:
                            _all_hit = False; break
                    if _all_hit:
                        _bind_pos_cache = torch.stack(_items, dim=0)

                # Per-batch geodesic dist² lookup
                _dist_sq_geo = None
                if self._geo_dist_per_topo:
                    _per_b = []; _all_hit = True
                    for b in range(src_v.shape[0]):
                        idn = batch.id_name[b] if isinstance(batch.id_name, list) else batch.id_name
                        topo = ('ict'  if idn.startswith('ict_')  else
                                'mf'   if idn.startswith('m--')   else
                                'biwi' if idn.startswith('biwi_') else
                                'coma' if idn.startswith('coma_') else None)
                        if topo and topo in self._geo_dist_per_topo:
                            _gd = self._geo_dist_per_topo[topo].t()      # [V, J]
                            if _gd.shape[0] != src_v.shape[1]:
                                _gd = _gd[:src_v.shape[1]]
                            _per_b.append(_gd)
                        else:
                            _all_hit = False; break
                    if _all_hit:
                        _dist_sq_geo = (torch.stack(_per_b, dim=0)) ** 2

                pred_lbs  = self.model(src_v, deform_in, source_normal=src_n,
                                       nfs_feat=_nfs_feat,
                                       bind_pos_cache=_bind_pos_cache,
                                       dist_sq_geo=_dist_sq_geo)

                # build mesh operators once
                if L_sp is None:
                    f_np = batch.faces[0].cpu().numpy()
                    v_np = src_v[0].cpu().numpy()
                    L_sp = _build_cot_laplacian(v_np, f_np)
                    edges = _build_edges(f_np).to(self.device)

                # ── compute metrics ─────────────────────────────────────
                mse = F.mse_loss(gt_v, pred_lbs).item()
                pv_l2 = torch.sqrt(((gt_v - pred_lbs) ** 2).sum(dim=-1))  # [B, N]

                total["mse"]  += mse
                total["l2"]   += pv_l2.mean().item()
                total["l2_max_sum"] += pv_l2.max(dim=-1).values.mean().item()
                total["lap"]       += _laplacian_error(L_sp, pred_lbs.cpu(), gt_v.cpu())
                total["norm_cos"]  += _normal_consistency(pred_lbs, gt_v, batch.faces)
                total["edge_dist"] += _edge_length_distortion(pred_lbs, gt_v, edges)
                all_per_vertex_l2.append(pv_l2.cpu())

                if not opts.no_t_mask:
                    t_mask   = plateau_hat_points(src_v)
                    inv_mask = 1.0 - t_mask
                    total["mse_in"]  += F.mse_loss(gt_v * t_mask,   pred_lbs * t_mask).item()
                    total["mse_out"] += F.mse_loss(gt_v * inv_mask,  pred_lbs * inv_mask).item()

                n_batches += 1
                pbar.set_description(
                    f"MSE:{mse:.4e} L2:{pv_l2.mean():.4f} Lap:{total['lap']/n_batches:.4e}")

                # save
                self._save_outputs(idx, batch, pred_lbs, gt_v)

                # visualize
                if not opts.no_vis and idx % opts.vis_every == 0:
                    self._render_self(idx, batch, pred_lbs)

        # ── aggregate ───────────────────────────────────────────────────
        inv = 1.0 / max(n_batches, 1)
        results = {
            "MSE":                 total["mse"] * inv,
            "L2_mean":             total["l2"]  * inv,
            "L2_max_mean":         total["l2_max_sum"] * inv,
            "Laplacian_err":       total["lap"] * inv,
            "Normal_cos_dist":     total["norm_cos"] * inv,
            "Edge_len_distortion": total["edge_dist"] * inv,
        }
        if not opts.no_t_mask:
            results["MSE_inner"] = total["mse_in"]  * inv
            results["MSE_outer"] = total["mse_out"] * inv

        all_pv = torch.cat(all_per_vertex_l2, dim=0).numpy()
        pv_flat = all_pv.flatten()
        results["L2_median"] = float(np.median(pv_flat))
        results["L2_p95"]    = float(np.percentile(pv_flat, 95))
        results["L2_p99"]    = float(np.percentile(pv_flat, 99))
        results["L2_max"]    = float(np.max(pv_flat))
        results["num_frames"] = int(all_pv.shape[0])

        self._print_and_save(results, all_pv)
        return results

    # ── Cross-retargeting evaluation ────────────────────────────────────

    def evaluate_cross(self):
        opts = self.opts

        dataset = EvalDataset(data_name=opts.data_selection, toggle=opts.data_toggle, data_basedir=opts.data_basedir,
                              ict_cap_id_num=opts.src_identity if opts.src_identity >= 0 else 2,
                              ict_cap_exp_num=opts.exp_num)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opts.batch_size,
            collate_fn=partial(CBD_collate_wrapper_eval, device='cpu'),
            num_workers=0,
        )

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), ncols=120,
                    desc="Eval HLBS [cross]")

        with torch.no_grad():
            for idx, batch in pbar:
                batch = batch.to(self.device)
                B = batch.vertices.shape[0]

                # source expression data
                src_v = batch.template
                src_n = batch.template_normal
                gt_v  = batch.vertices
                gt_n  = batch.vertices_normal

                # expand target to batch
                tgt_v = self.tgt_neu_vert.expand(B, -1, -1)
                tgt_n = self.tgt_neu_norm.expand(B, -1, -1)

                # Target NFS feat (same for all frames)
                _tgt_nfs_feat = None
                if hasattr(self, '_tgt_nfs_feat'):
                    _tgt_nfs_feat = self._tgt_nfs_feat.expand(B, -1, -1)

                pred_lbs = self.model.retarget(
                    src_v, src_n, gt_v, gt_n, tgt_v, tgt_n,
                    tgt_nfs_feat=_tgt_nfs_feat)

                # save predicted vertices
                if opts.save_vert:
                    for b in range(B):
                        np.save(
                            os.path.join(self.vert_dir, f"{idx * opts.batch_size + b:06d}.npy"),
                            pred_lbs[b].cpu().numpy())

                # save OBJ
                if opts.save_obj:
                    obj_dir = os.path.join(self.out_dir, "obj")
                    os.makedirs(obj_dir, exist_ok=True)
                    for b in range(B):
                        fid = idx * opts.batch_size + b
                        _write_obj(
                            os.path.join(obj_dir, f"pred_{fid:06d}.obj"),
                            pred_lbs[b].cpu().numpy(), self.tgt_faces)

                # visualize: source GT | target neutral | retargeted
                if not opts.no_vis and idx % opts.vis_every == 0:
                    src_faces_cpu = batch.faces[0].cpu()
                    tgt_faces_cpu = self.tgt_faces_torch[0].cpu()
                    v_list = [
                        gt_v[0].cpu(),           # source expression
                        tgt_v[0].cpu(),           # target neutral
                        pred_lbs[0].cpu(),        # retargeted
                    ]
                    f_list = [src_faces_cpu, tgt_faces_cpu, tgt_faces_cpu]
                    plot_image_array(
                        v_list, f_list,
                        rot_list=[[0, 0, 0]] * len(v_list),
                        size=1, bg_black=False, mode='shade',
                        logdir=self.img_dir,
                        name=f"{idx:06d}", save=True)

        print(f"\nCross-retarget done. Outputs: {self.out_dir}")

        if opts.make_video:
            images_to_video(self.img_dir, os.path.join(self.out_dir, "eval_hlbs_cross.mp4"))

    # ── Shared helpers ──────────────────────────────────────────────────

    def _save_outputs(self, idx, batch, pred_lbs, gt_v):
        opts = self.opts
        B = pred_lbs.shape[0]
        if opts.save_vert:
            for b in range(B):
                np.save(
                    os.path.join(self.vert_dir, f"{idx * opts.batch_size + b:06d}.npy"),
                    pred_lbs[b].cpu().numpy())
        if opts.save_gt:
            for b in range(B):
                np.save(
                    os.path.join(self.gt_dir, f"{idx * opts.batch_size + b:06d}.npy"),
                    gt_v[b].cpu().numpy())
        if opts.save_obj:
            obj_dir = os.path.join(self.out_dir, "obj")
            os.makedirs(obj_dir, exist_ok=True)
            faces_np = batch.faces[0].cpu().numpy()
            for b in range(B):
                fid = idx * opts.batch_size + b
                _write_obj(
                    os.path.join(obj_dir, f"pred_{fid:06d}.obj"),
                    pred_lbs[b].cpu().numpy(), faces_np)

    def _render_self(self, idx, batch, pred_lbs):
        faces_cpu = batch.faces[0].cpu()
        v_list = [
            batch.vertices[0].cpu(),    # GT
            batch.template[0].cpu(),    # template
            pred_lbs[0].cpu(),          # prediction
        ]
        f_list = [faces_cpu] * len(v_list)
        plot_image_array(
            v_list, f_list,
            rot_list=[[0, 0, 0]] * len(v_list),
            size=1, bg_black=False, mode='shade',
            logdir=self.img_dir,
            name=f"{idx:06d}", save=True)

    def _print_and_save(self, results, all_pv):
        print("\n=== HLBS Evaluation Results ===")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k:22s}: {v:.6e}")
            else:
                print(f"  {k:22s}: {v}")
        print("===============================")

        results_path = os.path.join(self.out_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Saved: {results_path}")

        np.save(os.path.join(self.out_dir, "per_vertex_l2.npy"), all_pv)

        if self.opts.make_video:
            images_to_video(self.img_dir, os.path.join(self.out_dir, "eval_hlbs.mp4"))


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    opts = Options()

    # inherit model config from training checkpoint
    train_cfg_path = os.path.join(opts.ckpt, "train_opts.yml")
    if os.path.isfile(train_cfg_path):
        with open(train_cfg_path) as f:
            train_cfg = yaml.safe_load(f)
        # All keys whose value affects model architecture / forward path —
        # must match training to avoid silent shape/buffer mismatch on load.
        _INHERIT_KEYS = [
            # core arch
            'hid_dim', 'num_layers', 'topo_key', 'freeze_adapt',
            'use_joint_trans', 'smooth_n_iter', 'target',
            'full_prediction',
            # input/feature pipeline
            'nfs_feat_dir', 'nfs_concat', 'adain_pos_norm',
            'dfn_skin', 'dfn_bind', 'dfn_exp',
            'smooth_delta_W', 'smooth_delta_W_alpha',
            # bind pose
            'freeze_bind_pose', 'bind_pose_mode',
            'joint_anchors_npy', 'joint_offsets_npy',
            'attn_temperature_init', 'bind_pos_cache_dir',
            # GMM hybrid + sigma
            'use_gmm_hybrid', 'init_log_sigma', 'gmm_mode',
            'residual_scale', 'sigma_targets_npy',
            # face mask (Option A)
            'active_joints_json', 'face_mask_r0', 'face_mask_r1',
            # geodesic
            'use_geodesic_gauss', 'geo_dist_dir',
        ]
        _inherited = []
        for key in _INHERIT_KEYS:
            if key in train_cfg:
                cli_flags = [f'--{key}', f'--{key.replace("_", "-")}']
                if not any(flag in sys.argv for flag in cli_flags):
                    setattr(opts, key, train_cfg[key])
                    _inherited.append(key)
        print(f"Inherited model config from: {train_cfg_path}")
        if _inherited:
            print(f"  inherited keys ({len(_inherited)}): {_inherited}")

    evaluator = HLBSEvaluator(opts)

    if opts.cross_retarget:
        evaluator.evaluate_cross()
    else:
        evaluator.evaluate_self()
