"""
train_hlbs.py — HierarchicalLBS standalone pre-training.

HLBS only: predicts coarse LBS deformation toward smooth_GT.
EDD is not involved. Analogous to Stage-1 in train_stage_disp.py.

Usage:
    python train_hlbs.py --config configs/train.yml \
        --rig_path utils/mf/rig_info.json \
        --topo_key mf \
        --num_identities 13 \
        --smooth_n_iter 16 \
        --use_data1 --data_toggle --batch_size 16 --max_epoch 200 --tb
"""
import os
import sys
import json
import argparse
import glob
import random
import numpy as np
import yaml

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils.matplotlib_rnd import plot_image_array
from dataloader_CBD import CBDDataset, CBDdataSampler, CBD_collate_wrapper
from utils.vis_loader import CheckpointVisLoader


class Logger:
    def __init__(self, file_path):
        self.file_path = file_path
    def write(self, txt):
        with open(self.file_path, 'a') as f:
            f.write(txt)


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yml")

    # rig / topology
    parser.add_argument("--rig_path", type=str, required=True,
                        help='Directory containing rig_info.json and skin_weights*.npy (from maya_rig/export_rig.py)')
    parser.add_argument("--topo_key", type=str, default='mf',
                        choices=['mf', 'biwi', 'voca'],
                        help='Which skin weight topology to use')
    parser.add_argument("--num_identities", type=int, default=13)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)

    # regularization
    parser.add_argument("--lambda_W_reg", type=float, default=1e-4,
                        help='L2 reg on delta_W')
    parser.add_argument("--lambda_t_reg", type=float, default=1e-4,
                        help='L2 reg on delta_t')
    parser.add_argument("--lambda_neu", type=float, default=1.0,
                        help='Neutral reconstruction loss: forward(src, delta=0) == src')
    parser.add_argument("--lambda_W_smooth", type=float, default=0.0,
                        help='Dirichlet smoothness on W: mean ||W_i - W_j||^2 over edges. '
                             '0 = disabled (default)')

    # ablation
    parser.add_argument("--freeze_adapt", dest='freeze_adapt', action='store_true',
                        help='Freeze skin_weight_net & bind_pose_net (delta_W=0, delta_t=0). '
                             'Only joint transforms are learned. Ablation for Maya init quality.')
    parser.set_defaults(freeze_adapt=False)
    parser.add_argument("--use_joint_trans", dest='use_joint_trans', action='store_true',
                        help='Predict per-joint local translation in addition to rotation (6+3=9 DOF per joint)')
    parser.set_defaults(use_joint_trans=False)
    parser.add_argument("--smooth_delta_W", type=int, default=0,
                        help='Laplacian smoothing iterations on delta_W in forward pass (0=off)')
    parser.add_argument("--smooth_delta_W_alpha", type=float, default=0.5,
                        help='Smoothing blend ratio (0=no smooth, 1=full neighbor average)')

    # surface losses
    parser.add_argument("--lambda_normal", type=float, default=0.0,
                        help='Normal consistency loss (1 - cos(n_pred, n_gt))')
    parser.add_argument("--lambda_curvature", type=float, default=0.0,
                        help='Curvature loss (Laplacian difference)')

    # full prediction mode
    parser.add_argument("--full_prediction", dest='full_prediction', action='store_true',
                        help='Use HierarchicalLBS_FullPred (no Maya base dependency)')
    parser.set_defaults(full_prediction=False)
    parser.add_argument("--init_phase_epochs", type=int, default=20,
                        help='Total Phase 1 epochs. Behavior depends on --init_mode.')
    parser.add_argument("--init_hold_epochs", type=int, default=0,
                        help='Hold epochs before annealing (only used in hold_anneal mode)')
    parser.add_argument("--init_mode", type=str, default='anneal',
                        choices=['anneal', 'hold_anneal', 'hold_cutoff'],
                        help='anneal: linear decay 0→K. '
                             'hold_anneal: hold H epochs then anneal to K. '
                             'hold_cutoff: hold H epochs then drop to 0.')
    parser.add_argument("--lambda_init", type=float, default=1.0,
                        help='Init supervision loss weight')

    # NFS encoder
    parser.add_argument("--nfs_ckpt", type=str, default=None,
                        help='Pretrained NFS checkpoint. If set, use NFS expression encoder for z_exp.')

    # target
    parser.add_argument("--target", type=str, default='gt',
                        choices=['gt', 'smooth_gt'],
                        help='"gt": HLBS → GT, EDD learns true LBS residual (default/recommended). '
                             '"smooth_gt": HLBS → smooth_GT, EDD learns wrinkle = GT-smooth_GT.')
    parser.add_argument("--smooth_n_iter", type=int, default=16,
                        help='Taubin smoothing iters (used when --target smooth_gt, default 16)')

    # training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--eval_iter", type=int, default=25,
                        help='Visualize every N epochs via vis_loader')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sc_step", type=int, default=1000000)
    parser.add_argument("--sc_gamma", type=float, default=0.5)
    parser.add_argument("--lambda_vert", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)

    # mask
    parser.add_argument("--no_t_mask", dest='no_t_mask', action='store_true')
    parser.set_defaults(no_t_mask=False)
    parser.add_argument("--use_t_mask", dest='no_t_mask', action='store_false')

    # data
    parser.add_argument("--use_data0", dest='use_data0', action='store_true')
    parser.set_defaults(use_data0=False)
    parser.add_argument("--use_data1", dest='use_data1', action='store_true')
    parser.set_defaults(use_data1=False)
    parser.add_argument("--use_data2", dest='use_data2', action='store_true')
    parser.set_defaults(use_data2=False)
    parser.add_argument("--use_data3", dest='use_data3', action='store_true')
    parser.set_defaults(use_data3=False)
    parser.add_argument("--data_toggle", dest='data_toggle', action='store_true')
    parser.add_argument("--no_fullhead", dest='no_fullhead', action='store_true',
                        help='Exclude ICT fullhead region (11248 verts) to save GPU memory')
    parser.set_defaults(no_fullhead=False)
    parser.set_defaults(data_toggle=False)

    # checkpoint
    parser.add_argument("--ckpt", type=str, default=None,
                        help='Checkpoint dir to resume from')
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--continue_ckpt", dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)

    # data path
    parser.add_argument("--data_basedir", type=str, default="/data/sihun",
                        help='Base directory for datasets')

    # logging
    parser.add_argument("--log_dir", type=str, default="./ckpts_hlbs")
    parser.add_argument("--tb", dest='tb', action='store_true')
    parser.set_defaults(tb=False)
    parser.add_argument("--debug", dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument("--vis_frames", type=str, default="config/vis_frames.yml")

    # dataloader compat fields (required by CBDDataset but unused here)
    parser.add_argument("--version", type=int, default=10)
    parser.add_argument("--use_strain", dest='use_strain', action='store_true')
    parser.set_defaults(use_strain=False)
    parser.add_argument("--strain_dim", type=int, default=1)
    parser.add_argument("--use_lbs", dest='use_lbs', action='store_true')
    parser.set_defaults(use_lbs=False)
    parser.add_argument("--use_lbs_joint_center", dest='use_lbs_joint_center', action='store_true')
    parser.set_defaults(use_lbs_joint_center=False)
    parser.add_argument("--no_use_translation", dest='no_use_translation', action='store_true')
    parser.set_defaults(no_use_translation=False)
    parser.add_argument("--use_laplacian", dest='use_laplacian', action='store_true')
    parser.set_defaults(use_laplacian=False)
    parser.add_argument("--data_rand_trans", dest='data_rand_trans', action='store_true')
    parser.set_defaults(data_rand_trans=False)
    parser.add_argument("--data_rand_scale", dest='data_rand_scale', action='store_true')
    parser.set_defaults(data_rand_scale=False)
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--use_decimate", dest='use_decimate', action='store_true')
    parser.set_defaults(use_decimate=False)
    parser.add_argument("--use_data9", dest='use_data9', action='store_true')
    parser.set_defaults(use_data9=False)

    return parser.parse_args()


class HLBSTrainer:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)

        from utils.rig_loader import load_rig
        from models.hierarchical_lbs import HierarchicalLBS

        if not opts.full_prediction:
            rig = load_rig(opts.rig_path)
            self.model = HierarchicalLBS(
                rig=rig,
                topology=opts.topo_key,
                in_dim_exp=12,
                hid_dim=opts.hid_dim,
                num_layers=opts.num_layers,
                device=str(self.device),
                freeze_adapt=opts.freeze_adapt,
                use_joint_trans=opts.use_joint_trans,
                smooth_delta_W=opts.smooth_delta_W,
                smooth_delta_W_alpha=opts.smooth_delta_W_alpha,
            ).to(self.device)
        else:
            self.model = None  # built in train_full_prediction

        if not opts.full_prediction:
            if opts.freeze_adapt:
                print("[HLBS] freeze_adapt=True: delta_W=0, delta_t=0 (Maya init only, joint transforms learned)")
                opts.lambda_W_reg = 0.0
                opts.lambda_t_reg = 0.0
                opts.lambda_W_smooth = 0.0

            if opts.ckpt and opts.continue_ckpt:
                paths = sorted(glob.glob(os.path.join(opts.ckpt, f"model_hlbs_{opts.start_epoch:03d}.pth")))
                if not paths:
                    paths = sorted(glob.glob(os.path.join(opts.ckpt, "model_hlbs_best.pth")))
                if paths:
                    self.model.load_state_dict(torch.load(paths[0], map_location=self.device))
                    print(f"Resumed HLBS from: {paths[0]}")

        # ── NFS expression encoder (optional) ─────────────────────────────
        self.nfs_encoder = None
        self.nfs_z_adapter = None
        if opts.nfs_ckpt:
            self._load_nfs_encoder(opts)

        self.vis_loader = CheckpointVisLoader(opts, device=self.device)
        self._edges = None   # lazily computed from first batch faces

    def _load_nfs_encoder(self, opts):
        """Load NFS expression encoder standalone (no full NFS model needed)."""
        import yaml, pickle, trimesh
        from models.encoder import BaseDiffusionNetEncoder
        from models.CNN import TextureEncoder
        from utils.nfr_utils import get_dfn_info

        print(f"[HLBS] Loading NFS expression encoder from: {opts.nfs_ckpt}")

        # ── Read NFS config ──
        nfs_dir = os.path.dirname(opts.nfs_ckpt)
        with open(os.path.join(nfs_dir, "train_opts.yml")) as f:
            nfs_cfg = yaml.safe_load(f)
        nfs_rig_dim = nfs_cfg.get('rig_dim', 128)
        nfs_img_feat_dim = nfs_cfg.get('img_feat_dim', 128)

        # ── Build standalone modules ──
        in_shape = 6 + nfs_img_feat_dim   # pos(3) + normal(3) + img_feat(128)
        exp_encoder = BaseDiffusionNetEncoder(
            in_shape=in_shape, pre_computes=None, out_shape=nfs_rig_dim,
        ).to(self.device)
        img_encoder = TextureEncoder().to(self.device)
        img_fc = torch.nn.Linear(128, nfs_img_feat_dim).to(self.device)

        # ── Load weights from NFS checkpoint ──
        ckpt = torch.load(opts.nfs_ckpt, map_location=self.device)
        skip_suffixes = {'mass', 'L_ind', 'L_val', 'evals', 'evecs', 'grad_X', 'grad_Y', 'faces'}

        def _extract(prefix):
            out = {}
            for k, v in ckpt.items():
                if k.startswith(prefix):
                    short = k[len(prefix):]
                    if not any(short.startswith(s) or s in short for s in skip_suffixes):
                        out[short] = v
            return out

        exp_encoder.load_state_dict(_extract('mesh_exp_encoder.'), strict=False)
        img_encoder.load_state_dict(_extract('img_encoder.'), strict=False)
        img_fc.load_state_dict(_extract('img_fc.'), strict=False)
        print(f"  Loaded exp_encoder, img_encoder, img_fc weights")

        # ── Freeze all ──
        for m in [exp_encoder, img_encoder, img_fc]:
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()

        self.nfs_exp_encoder = exp_encoder
        self.nfs_img_encoder = img_encoder
        self.nfs_img_fc = img_fc
        self.nfs_encoder = exp_encoder  # for None-check in train loop

        # ── Precompute DiffusionNet operators for MF template ──
        with open("/data/sihun/pca/multiface_align/mf_templates.pkl", 'rb') as f:
            templates = pickle.load(f)
        faces_np = np.array(templates['face'], dtype=np.int32)
        first_id = [k for k in templates if k != 'face'][0]
        verts_np = templates[first_id].astype(np.float32)
        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        self._nfs_dfn_info = get_dfn_info(mesh, map_location=self.device)
        print(f"  DiffusionNet operators precomputed")

        # ── Precompute image feature from prerendered neutral image ──
        img_npy_path = "data/MF_all_v5/m--20180426--0000--002643814--GHS_neutral_img.npy"
        img_np = np.load(img_npy_path)                                  # [256, 256, 3]
        img_t = torch.tensor(img_np, dtype=torch.float32, device=self.device)
        img_t = img_t.unsqueeze(0).permute(0, 3, 1, 2)                 # [1, 3, 256, 256]
        with torch.no_grad():
            self._nfs_img_feat = img_fc(img_encoder(img_t))             # [1, img_feat_dim]
        print(f"  Image feature precomputed from {img_npy_path}")

        # ── Adapter: NFS z_GE [B, rig_dim] → HLBS z_exp [B, hid_dim] ──
        if nfs_rig_dim != opts.hid_dim:
            self.nfs_z_adapter = torch.nn.Linear(nfs_rig_dim, opts.hid_dim).to(self.device)
            print(f"  z_adapter: {nfs_rig_dim} → {opts.hid_dim}")
        else:
            self.nfs_z_adapter = torch.nn.Identity()

        # ── Freeze HLBS's own expression encoder ──
        for p in self.model.lbs_exp_z_model.parameters():
            p.requires_grad_(False)
        print("[HLBS] NFS encoder ready. HLBS lbs_exp_z_model frozen.")

    @torch.no_grad()
    def _nfs_encode_exp(self, gt_v, gt_n):
        """Encode expression using NFS pretrained encoder.
        Args:
            gt_v: [B, V, 3] expression vertices
            gt_n: [B, V, 3] expression normals
        Returns:
            z_exp: [B, hid_dim]
        """
        B, V, _ = gt_v.shape
        # vertex feature: pos + normal + img_feat (broadcast over V)
        vert_feat = torch.cat([gt_v, gt_n], dim=-1)                    # [B, V, 6]
        img_feat = self._nfs_img_feat.expand(B, -1)                    # [B, 128]
        img_exp = img_feat.unsqueeze(1).expand(-1, V, -1)              # [B, V, 128]
        nfs_input = torch.cat([vert_feat, img_exp], dim=-1)            # [B, V, 134]
        # DiffusionNet encode
        self.nfs_exp_encoder.update_precomputes(self._nfs_dfn_info)
        z_ge = self.nfs_exp_encoder(nfs_input)                         # [B, 128]
        return self.nfs_z_adapter(z_ge)                                # [B, hid_dim]

    def train(self, epochs):
        opts = self.opts
        BS = opts.batch_size

        # Collect trainable params: HLBS model + NFS adapter (if any)
        train_params = list(self.model.parameters())
        if self.nfs_z_adapter is not None and not isinstance(self.nfs_z_adapter, torch.nn.Identity):
            train_params += list(self.nfs_z_adapter.parameters())
        self.optimizer = torch.optim.AdamW(
            train_params, lr=opts.lr, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opts.sc_step, gamma=opts.sc_gamma)

        # When target is GT, smooth data is not needed — override to skip loading
        if opts.target == 'gt':
            opts.smooth_n_iter = 0
        
        train_ds = CBDDataset(opts, is_train=True, toggle=opts.data_toggle, data_basedir=opts.data_basedir)
        valid_ds = CBDDataset(opts, is_valid=True, toggle=opts.data_toggle, data_basedir=opts.data_basedir)
        train_sampler = CBDdataSampler(train_ds.len_list, BS, shuffle=True,  balance=False, is_train=True)
        valid_sampler = CBDdataSampler(valid_ds.len_list, BS, shuffle=True,  balance=False, is_valid=True)
        _nw = opts.num_workers
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_sampler=train_sampler,
            collate_fn=partial(CBD_collate_wrapper, device='cpu'),
            num_workers=_nw, persistent_workers=(_nw > 0))
        valid_loader = torch.utils.data.DataLoader(
            valid_ds, batch_sampler=valid_sampler,
            collate_fn=partial(CBD_collate_wrapper, device='cpu'), num_workers=0)

        # logging
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        resume_mode = opts.ckpt and opts.continue_ckpt
        if resume_mode and os.path.isdir(opts.ckpt):
            if opts.log_dir != './ckpts_hlbs':
                # Explicit --log_dir given → use it as new dir
                pass
            else:
                # No explicit --log_dir → reuse ckpt dir
                opts.log_dir = opts.ckpt
        else:
            freeze_tag = "-frozenAdapt" if opts.freeze_adapt else ""
            trans_tag = "-jTrans" if opts.use_joint_trans else ""
            sdw_tag = f"-sdw{opts.smooth_delta_W}a{opts.smooth_delta_W_alpha}" if opts.smooth_delta_W > 0 else ""
            nfs_tag = "-nfsEnc" if opts.nfs_ckpt else ""
            surf_tag = ""
            if opts.lambda_normal > 0: surf_tag += f"-nrm{opts.lambda_normal}"
            if opts.lambda_curvature > 0: surf_tag += f"-crv{opts.lambda_curvature}"
            tag = f"-HLBS-{opts.topo_key}-s{opts.smooth_n_iter}{freeze_tag}{trans_tag}{sdw_tag}{nfs_tag}{surf_tag}"
            opts.log_dir = os.path.join(opts.log_dir, now + tag)

        os.makedirs(opts.log_dir, exist_ok=True)
        os.makedirs(f"{opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{opts.log_dir}/img/valid/mesh", exist_ok=True)

        with open(os.path.join(opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(opts), f, indent=4)
        with open(os.path.join(opts.log_dir, "train_opts.yml"), 'w') as f:
            yaml.dump(vars(opts), f, sort_keys=False)

        writer_train = writer_valid = None
        if opts.tb:
            writer_train = SummaryWriter(log_dir=os.path.join(opts.log_dir, "train"))
            writer_valid = SummaryWriter(log_dir=os.path.join(opts.log_dir, "valid"))

        logger = Logger(os.path.join(opts.log_dir, "log.txt"))
        print(f'Log: {logger.file_path}')

        if opts.target == 'smooth_gt':
            tgt_desc = f"smooth_GT (Taubin n_iter={opts.smooth_n_iter})"
            smooth_line = f"  smooth_n_iter : {opts.smooth_n_iter}\n"
        else:
            tgt_desc = "GT (full target, no smoothing)"
            smooth_line = ""
        config_text = (
            f"=== HLBS Training ===\n"
            f"  topo_key      : {opts.topo_key}\n"
            f"  num_identities: {opts.num_identities}\n"
            f"  target        : {opts.target}  ({tgt_desc})\n"
            f"{smooth_line}"
            f"  lambda_W_reg  : {opts.lambda_W_reg}\n"
            f"  lambda_t_reg  : {opts.lambda_t_reg}\n"
            f"  lambda_neu    : {opts.lambda_neu}\n"
            f"  lambda_W_smooth: {opts.lambda_W_smooth}\n"
            f"  freeze_adapt  : {opts.freeze_adapt}\n"
            f"  use_joint_trans: {opts.use_joint_trans}\n"
            f"=====================\n"
        )
        print(config_text)
        logger.write(config_text)
        logger.write(train_ds.get_data_config())

        loss_lambda = {
            "recon-lbs":   opts.lambda_vert,
            "recon-neu":   opts.lambda_neu,
            "lbs-W-reg":   opts.lambda_W_reg,
            "lbs-t-reg":   opts.lambda_t_reg,
            "lbs-W-smooth": opts.lambda_W_smooth,
            "recon-normal": opts.lambda_normal,
            "recon-curvature": opts.lambda_curvature,
        }

        BEST_LOSS  = 1e8
        BEST_EPOCH = 0
        len_train  = len(train_loader)
        len_valid  = len(valid_loader)
        interv     = max(1, round(len_train / 10))
        interv_val = max(1, round(len_valid / 3))

        # Precompute mesh edges for delta_W forward smoothing (if enabled)
        if opts.smooth_delta_W > 0 and self.model._mesh_edges is None:
            # Get faces from first batch
            _first = next(iter(train_loader))
            _faces = _first.faces[0] if hasattr(_first.faces, '__getitem__') else _first.faces
            self.model.set_mesh_edges(_faces)
            print(f"[HLBS] delta_W forward smoothing: iters={opts.smooth_delta_W}, alpha={opts.smooth_delta_W_alpha}")

        for epoch in range(opts.start_epoch, epochs + 1):
            # ── Train ────────────────────────────────────────────────────────
            self.model.train()
            running = {"recon-lbs": 0.0, "recon-neu": 0.0, "recon-normal": 0.0, "recon-curvature": 0.0, "lbs-W-reg": 0.0, "lbs-t-reg": 0.0, "lbs-W-smooth": 0.0, "total": 0.0}
            cnt = 0

            pbar = tqdm(enumerate(train_loader), total=len_train, ncols=120,
                        desc=f"[{epoch:03d}] Train HLBS")
            for idx, batch in pbar:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                src_v  = batch.template
                src_n  = batch.template_normal
                gt_v   = batch.vertices
                gt_n   = batch.vertices_normal
                smt_v  = batch.smooth_vertices

                target_v = smt_v if opts.target == 'smooth_gt' else gt_v

                delta     = gt_v - src_v
                src_in    = torch.cat([src_v, src_n], dim=-1)        # [B, N, 6]
                deform_in = torch.cat([delta, gt_n, src_in], dim=-1) # [B, N, 12]

                # z_exp from NFS encoder or HLBS internal encoder
                z_exp_ext = None
                if self.nfs_encoder is not None:
                    z_exp_ext = self._nfs_encode_exp(gt_v, gt_n)      # [B, hid_dim]

                pred_lbs = self.model(src_v, deform_in, z_exp_override=z_exp_ext)  # [B, N, 3]

                if opts.no_t_mask:
                    loss_dict = {"recon-lbs": F.mse_loss(target_v, pred_lbs)}
                else:
                    from utils.exp_utils import plateau_hat_points
                    t_mask   = plateau_hat_points(src_v)
                    inv_mask = 1.0 - t_mask
                    loss_dict = {
                        "recon-lbs": (
                            F.mse_loss(target_v * t_mask,  pred_lbs * t_mask)
                            + F.mse_loss(src_v * inv_mask, pred_lbs * inv_mask)
                        )
                    }

                # ── Neutral reconstruction loss ───────────────────────────
                # When delta=0 (no deformation), pred_lbs should equal src_v
                if opts.lambda_neu > 0:
                    delta_zero   = torch.zeros_like(src_v)
                    neu_deform_in = torch.cat([delta_zero, src_n, src_v, src_n], dim=-1)
                    pred_neutral = self.model(src_v, neu_deform_in)
                    if opts.no_t_mask:
                        loss_dict["recon-neu"] = F.mse_loss(src_v, pred_neutral)
                    else:
                        loss_dict["recon-neu"] = (
                            F.mse_loss(src_v * t_mask,    pred_neutral * t_mask)
                            + F.mse_loss(src_v * inv_mask, pred_neutral * inv_mask)
                        )

                # ── Normal consistency loss ───────────────────────────────
                if opts.lambda_normal > 0:
                    from utils.mesh_utils import calc_norm_torch
                    pred_n = calc_norm_torch(pred_lbs, batch.faces, at='verts')
                    gt_n_recomputed = calc_norm_torch(target_v, batch.faces, at='verts')
                    normal_diff = 1 - F.cosine_similarity(pred_n, gt_n_recomputed, dim=-1)  # [B, V]
                    if not opts.no_t_mask:
                        normal_diff = normal_diff * t_mask.squeeze(-1)
                    loss_dict["recon-normal"] = normal_diff.mean()

                # ── Curvature loss (Laplacian difference) ────────────────────
                if opts.lambda_curvature > 0:
                    from train_edd_real import _uniform_laplacian
                    loss_dict["recon-curvature"] = F.mse_loss(
                        _uniform_laplacian(pred_lbs, batch.faces),
                        _uniform_laplacian(target_v, batch.faces))

                # ── Regularization + smoothness ───────────────────────────
                # Lazily build mesh edges for smoothness loss
                if opts.lambda_W_smooth > 0 and self._edges is None:
                    f_raw = batch.faces.cpu()
                    f_np = f_raw[0].numpy() if f_raw.dim() == 3 else f_raw.numpy()
                    e = np.concatenate([f_np[:, [0,1]], f_np[:, [1,2]], f_np[:, [0,2]]], axis=0)
                    e = np.sort(e, axis=1)
                    e = np.unique(e, axis=0)
                    self._edges = torch.tensor(e, dtype=torch.long, device=self.device)

                edges = self._edges if opts.lambda_W_smooth > 0 else None
                regs = self.model.reg_loss(src_v, edges=edges)
                loss_dict["lbs-W-reg"]    = regs["L_W_reg"]
                loss_dict["lbs-t-reg"]    = regs["L_t_reg"]
                if opts.lambda_W_smooth > 0:
                    loss_dict["lbs-W-smooth"] = regs["L_W_smooth"]

                loss = sum(loss_dict[k] * loss_lambda[k] for k in loss_dict)
                loss.backward()
                self.optimizer.step()

                for k in running:
                    if k != "total" and k in loss_dict:
                        running[k] += (loss_dict[k] * loss_lambda[k]).item()
                running["total"] += loss.item()
                cnt += 1
                pbar.set_description(f"[{epoch:03d}] lbs: {loss_dict['recon-lbs']:.5e}")

                if idx % interv == 1:
                    inv = 1.0 / cnt
                    log_text = f"[{epoch:03d}/{epochs:03d}][{idx:04d}][Train] "
                    log_text += " ".join(f"{k}: {v*inv:.6e}" for k, v in running.items())
                    logger.write(log_text + "\n")

                    HB = BS // 2
                    _s = lambda i: min(i, BS-1)
                    _d = lambda t: t.cpu().detach()
                    faces_cpu = batch.faces.cpu()
                    v_list = [
                        _d(gt_v[0]),        _d(gt_v[_s(1)]),
                        _d(gt_v[_s(HB)]),   _d(gt_v[BS-1]),
                    ]
                    if opts.target == 'smooth_gt':
                        v_list += [
                            _d(smt_v[0]),       _d(smt_v[_s(1)]),
                            _d(smt_v[_s(HB)]),  _d(smt_v[BS-1]),
                        ]
                    v_list += [
                        _d(pred_lbs[0]),    _d(pred_lbs[_s(1)]),
                        _d(pred_lbs[_s(HB)]), _d(pred_lbs[BS-1]),
                    ]
                    f_list = [faces_cpu] * len(v_list)
                    plot_image_array(
                        v_list, f_list, rot_list=[[0,0,0]]*len(v_list),
                        size=1, bg_black=False, mode='shade',
                        logdir=f"{opts.log_dir}/img/train/mesh",
                        name=f"{epoch:03d}_{idx:04d}", save=True)

                if opts.debug:
                    break

            if epoch != 0:
                self.scheduler.step()
            if writer_train:
                for k, v in running.items():
                    writer_train.add_scalar(k, v / cnt, epoch)

            if epoch % opts.save_interval == 0:
                torch.save(self.model.state_dict(),
                           f'{opts.log_dir}/model_hlbs_{epoch:03d}.pth')

            if epoch % opts.eval_iter == 0:
                self.vis_loader.visualize(
                    self.model, None, epoch,
                    save_dir=f'{opts.log_dir}/img/eval',
                    mode='hlbs',
                    smooth_n_iter=opts.smooth_n_iter,
                    no_t_mask=opts.no_t_mask,
                )

            # ── Valid ────────────────────────────────────────────────────────
            self.model.eval()
            running_val = {"recon-lbs": 0.0, "recon-neu": 0.0, "recon-normal": 0.0, "recon-curvature": 0.0, "total": 0.0}
            vcnt = 0

            pbar = tqdm(enumerate(valid_loader), total=len_valid, ncols=120,
                        desc=f"[{epoch:03d}] Valid HLBS")
            for idx, batch in pbar:
                batch = batch.to(self.device)
                vcnt += 1
                with torch.no_grad():
                    src_v  = batch.template
                    src_n  = batch.template_normal
                    gt_v   = batch.vertices
                    gt_n   = batch.vertices_normal
                    smt_v  = batch.smooth_vertices

                    delta     = gt_v - src_v
                    src_in    = torch.cat([src_v, src_n], dim=-1)
                    deform_in = torch.cat([delta, gt_n, src_in], dim=-1)

                    z_exp_ext = None
                    if self.nfs_encoder is not None:
                        z_exp_ext = self._nfs_encode_exp(gt_v, gt_n)

                    pred_lbs  = self.model(src_v, deform_in, z_exp_override=z_exp_ext)

                    target_v_val = smt_v if opts.target == 'smooth_gt' else gt_v
                    val_loss = F.mse_loss(target_v_val, pred_lbs).item() * loss_lambda["recon-lbs"]
                    running_val["recon-lbs"] += val_loss
                    running_val["total"]     += val_loss

                    # Neutral reconstruction loss (val)
                    if opts.lambda_neu > 0:
                        delta_zero    = torch.zeros_like(src_v)
                        neu_deform_in = torch.cat([delta_zero, src_n, src_v, src_n], dim=-1)
                        pred_neutral  = self.model(src_v, neu_deform_in)
                        val_neu = F.mse_loss(src_v, pred_neutral).item() * loss_lambda["recon-neu"]
                        running_val["recon-neu"] += val_neu
                        running_val["total"]     += val_neu

                    # Normal consistency loss (val)
                    if opts.lambda_normal > 0:
                        from utils.mesh_utils import calc_norm_torch
                        pred_n = calc_norm_torch(pred_lbs, batch.faces, at='verts')
                        gt_n_val = calc_norm_torch(target_v_val, batch.faces, at='verts')
                        val_nrm = (1 - F.cosine_similarity(pred_n, gt_n_val, dim=-1)).mean().item() * loss_lambda["recon-normal"]
                        running_val["recon-normal"] += val_nrm
                        running_val["total"]        += val_nrm

                    # Curvature loss (val)
                    if opts.lambda_curvature > 0:
                        from train_edd_real import _uniform_laplacian
                        val_crv = F.mse_loss(
                            _uniform_laplacian(pred_lbs, batch.faces),
                            _uniform_laplacian(target_v_val, batch.faces)).item() * loss_lambda["recon-curvature"]
                        running_val["recon-curvature"] += val_crv
                        running_val["total"]           += val_crv

                pbar.set_description(f"[{epoch:03d}] val lbs: {val_loss:.5e}")

                if idx % interv_val == 0:
                    BS_v = batch.vertices.shape[0]
                    HB_v = BS_v // 2
                    _s = lambda i: min(i, BS_v-1)
                    faces_cpu = batch.faces.cpu()
                    v_list = [
                        gt_v[0].cpu(),        gt_v[_s(1)].cpu(),
                        gt_v[_s(HB_v)].cpu(), gt_v[BS_v-1].cpu(),
                    ]
                    if opts.target == 'smooth_gt':
                        v_list += [
                            smt_v[0].cpu(),        smt_v[_s(1)].cpu(),
                            smt_v[_s(HB_v)].cpu(), smt_v[BS_v-1].cpu(),
                        ]
                    v_list += [
                        pred_lbs[0].cpu(),        pred_lbs[_s(1)].cpu(),
                        pred_lbs[_s(HB_v)].cpu(), pred_lbs[BS_v-1].cpu(),
                    ]
                    f_list = [faces_cpu] * len(v_list)
                    plot_image_array(
                        v_list, f_list, rot_list=[[0,0,0]]*len(v_list),
                        size=1, bg_black=False, mode='shade',
                        logdir=f"{opts.log_dir}/img/valid/mesh",
                        name=f"{epoch:03d}_{idx:04d}", save=True)

                if opts.debug:
                    break

            if writer_valid:
                for k, v in running_val.items():
                    writer_valid.add_scalar(k, v / vcnt, epoch)

            total_val = running_val["total"] / vcnt
            if total_val < BEST_LOSS:
                BEST_LOSS  = total_val
                BEST_EPOCH = epoch
                torch.save(self.model.state_dict(), f'{opts.log_dir}/model_hlbs_best.pth')
                print(f"[{epoch:03d}] Best: {BEST_LOSS:.6e} (epoch {BEST_EPOCH})")
                logger.write(f"[{epoch:03d}] Best Loss: {BEST_LOSS:.6e}\n")
            else:
                print(f"[{epoch:03d}] Val: {total_val:.6e} (Best: {BEST_LOSS:.6e} [{BEST_EPOCH}])")
                logger.write(f"[{epoch:03d}] Val: {total_val:.6e} (Best: {BEST_LOSS:.6e} [{BEST_EPOCH}])\n")


    def train_full_prediction(self, epochs):
        """Train with HierarchicalLBS_FullPred: full W/bind prediction + Phase 1 annealing."""
        from models.hierarchical_lbs import HierarchicalLBS_FullPred
        opts = self.opts
        BS = opts.batch_size

        # ── Build FullPred model ─────────────────────────────────────────
        from utils.rig_loader import load_rig
        rig = load_rig(opts.rig_path)
        self.model = HierarchicalLBS_FullPred(
            rig=rig,
            topology=opts.topo_key,
            in_dim_exp=12,
            hid_dim=opts.hid_dim,
            num_layers=opts.num_layers,
            device=str(self.device),
            use_joint_trans=opts.use_joint_trans,
            smooth_W=opts.smooth_delta_W,
            smooth_W_alpha=opts.smooth_delta_W_alpha,
        ).to(self.device)
        print(f"[HLBS FullPred] {sum(p.numel() for p in self.model.parameters()):,} params")

        # Resume from checkpoint if specified
        if opts.ckpt and opts.continue_ckpt:
            ckpt_path = os.path.join(opts.ckpt, f"model_hlbs_{opts.start_epoch:03d}.pth")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(opts.ckpt, "model_hlbs_best.pth")
            if os.path.exists(ckpt_path):
                self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                print(f"[FullPred] Resumed from: {ckpt_path}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=opts.lr, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opts.sc_step, gamma=opts.sc_gamma)

        if opts.target == 'gt':
            opts.smooth_n_iter = 0

        train_ds = CBDDataset(opts, is_train=True, toggle=opts.data_toggle, data_basedir=opts.data_basedir)
        valid_ds = CBDDataset(opts, is_valid=True, toggle=opts.data_toggle, data_basedir=opts.data_basedir)
        _region_min = 1 if opts.no_fullhead else 0
        train_sampler = CBDdataSampler(train_ds.len_list, BS, shuffle=True,  balance=False, is_train=True, region_min=_region_min)
        valid_sampler = CBDdataSampler(valid_ds.len_list, BS, shuffle=True,  balance=False, is_valid=True, region_min=_region_min)
        _nw = opts.num_workers
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_sampler=train_sampler,
            collate_fn=partial(CBD_collate_wrapper, device='cpu'),
            num_workers=_nw, persistent_workers=(_nw > 0))
        valid_loader = torch.utils.data.DataLoader(
            valid_ds, batch_sampler=valid_sampler,
            collate_fn=partial(CBD_collate_wrapper, device='cpu'), num_workers=0)

        # ── Logging ──────────────────────────────────────────────────────
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        trans_tag = "-jTrans" if opts.use_joint_trans else ""
        sdw_tag = f"-sdw{opts.smooth_delta_W}a{opts.smooth_delta_W_alpha}" if opts.smooth_delta_W > 0 else ""
        fh_tag = "-noFH" if opts.no_fullhead else ""
        tag = f"-HLBS-FullPred-{opts.topo_key}{trans_tag}{sdw_tag}{fh_tag}"
        if opts.ckpt and opts.continue_ckpt:
            opts.log_dir = opts.ckpt  # resume into same dir
        else:
            opts.log_dir = os.path.join(opts.log_dir, now + tag)

        os.makedirs(opts.log_dir, exist_ok=True)
        os.makedirs(f"{opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{opts.log_dir}/img/valid/mesh", exist_ok=True)

        with open(os.path.join(opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(opts), f, indent=4)
        with open(os.path.join(opts.log_dir, "train_opts.yml"), 'w') as f:
            yaml.dump(vars(opts), f, sort_keys=False)

        writer_train = writer_valid = None
        if opts.tb:
            writer_train = SummaryWriter(log_dir=os.path.join(opts.log_dir, "train"))
            writer_valid = SummaryWriter(log_dir=os.path.join(opts.log_dir, "valid"))

        logger = Logger(os.path.join(opts.log_dir, "log.txt"))
        print(f'Log: {logger.file_path}')

        config_text = (
            f"=== HLBS FullPred Training ===\n"
            f"  topo_key       : {opts.topo_key}\n"
            f"  use_joint_trans: {opts.use_joint_trans}\n"
            f"  init_phase     : {opts.init_phase_epochs} epochs\n"
            f"  lambda_init    : {opts.lambda_init}\n"
            f"  lambda_vert    : {opts.lambda_vert}\n"
            f"  lambda_neu     : {opts.lambda_neu}\n"
            f"==============================\n"
        )
        print(config_text)
        logger.write(config_text)
        logger.write(train_ds.get_data_config())

        # ── Precompute mesh edges ────────────────────────────────────────
        if opts.smooth_delta_W > 0 and self.model._mesh_edges is None:
            _first = next(iter(train_loader))
            _faces = _first.faces[0] if hasattr(_first.faces, '__getitem__') else _first.faces
            self.model.set_mesh_edges(_faces)

        BEST_LOSS = 1e8
        BEST_EPOCH = 0
        len_train = len(train_loader)
        len_valid = len(valid_loader)
        interv = max(1, round(len_train / 10))
        interv_val = max(1, round(len_valid / 3))
        K = opts.init_phase_epochs

        for epoch in range(opts.start_epoch, epochs + 1):
            # Phase 1 init loss scheduling
            H = opts.init_hold_epochs
            if opts.init_mode == 'anneal':
                lambda_init = opts.lambda_init * max(0.0, 1.0 - epoch / K) if K > 0 else 0.0
            elif opts.init_mode == 'hold_anneal':
                if epoch < H:
                    lambda_init = opts.lambda_init
                elif epoch < K:
                    lambda_init = opts.lambda_init * (1.0 - (epoch - H) / max(K - H, 1))
                else:
                    lambda_init = 0.0
            elif opts.init_mode == 'hold_cutoff':
                lambda_init = opts.lambda_init if epoch < H else 0.0

            # ── Train ────────────────────────────────────────────────────
            self.model.train()
            running = {"recon-lbs": 0.0, "recon-neu": 0.0, "recon-normal": 0.0, "init-W": 0.0, "init-bind": 0.0, "total": 0.0}
            cnt = 0

            pbar = tqdm(enumerate(train_loader), total=len_train, ncols=120,
                        desc=f"[{epoch:03d}] Train FullPred (phase{'1' if lambda_init > 0 else '2'})")
            for idx, batch in pbar:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                src_v  = batch.template
                src_n  = batch.template_normal
                gt_v   = batch.vertices
                gt_n   = batch.vertices_normal
                target_v = gt_v

                delta     = gt_v - src_v
                src_in    = torch.cat([src_v, src_n], dim=-1)
                deform_in = torch.cat([delta, gt_n, src_in], dim=-1)

                pred_lbs = self.model(src_v, deform_in)

                # ── Recon loss ───────────────────────────────────────────
                if opts.no_t_mask:
                    loss_dict = {"recon-lbs": F.mse_loss(target_v, pred_lbs)}
                else:
                    from utils.exp_utils import plateau_hat_points
                    t_mask   = plateau_hat_points(src_v)
                    inv_mask = 1.0 - t_mask
                    loss_dict = {
                        "recon-lbs": (
                            F.mse_loss(target_v * t_mask,  pred_lbs * t_mask)
                            + F.mse_loss(src_v * inv_mask, pred_lbs * inv_mask)
                        )
                    }

                # ── Neutral recon loss ───────────────────────────────────
                if opts.lambda_neu > 0:
                    delta_zero = torch.zeros_like(src_v)
                    neu_deform_in = torch.cat([delta_zero, src_n, src_v, src_n], dim=-1)
                    pred_neutral = self.model(src_v, neu_deform_in)
                    if opts.no_t_mask:
                        loss_dict["recon-neu"] = F.mse_loss(src_v, pred_neutral)
                    else:
                        loss_dict["recon-neu"] = (
                            F.mse_loss(src_v * t_mask,    pred_neutral * t_mask)
                            + F.mse_loss(src_v * inv_mask, pred_neutral * inv_mask)
                        )

                # ── Normal consistency loss (skip when vertex-permuted) ──────
                if opts.lambda_normal > 0 and not is_permed:
                    from utils.mesh_utils import calc_norm_torch
                    pred_n = calc_norm_torch(pred_lbs, batch.faces, at='verts')
                    gt_n_recomp = calc_norm_torch(target_v, batch.faces, at='verts')
                    normal_diff = 1 - F.cosine_similarity(pred_n, gt_n_recomp, dim=-1)
                    if not opts.no_t_mask:
                        normal_diff = normal_diff * t_mask.squeeze(-1)
                    loss_dict["recon-normal"] = normal_diff.mean()

                # ── Phase 1: init supervision ────────────────────────────
                if lambda_init > 0:
                    _md = batch.mesh_data if hasattr(batch, 'mesh_data') else None
                    init_losses = self.model.init_loss(src_v, mesh_data=_md)
                    for k, v in init_losses.items():
                        loss_dict[k] = v

                # ── Total loss ───────────────────────────────────────────
                loss_lambda = {
                    "recon-lbs": opts.lambda_vert,
                    "recon-neu": opts.lambda_neu,
                    "recon-normal": opts.lambda_normal,
                    "L_W_init": lambda_init,
                    "L_bind_init": lambda_init,
                }
                loss = sum(loss_dict[k] * loss_lambda.get(k, 0.0) for k in loss_dict)
                loss.backward()
                self.optimizer.step()

                for k in running:
                    if k != "total" and k in loss_dict:
                        running[k] += (loss_dict[k] * loss_lambda.get(k, 1.0)).item()
                    elif k == "init-W" and "L_W_init" in loss_dict:
                        running[k] += (loss_dict["L_W_init"] * lambda_init).item()
                    elif k == "init-bind" and "L_bind_init" in loss_dict:
                        running[k] += (loss_dict["L_bind_init"] * lambda_init).item()
                running["total"] += loss.item()
                cnt += 1
                pbar.set_description(
                    f"[{epoch:03d}] lbs:{loss_dict['recon-lbs']:.4e} init:{lambda_init:.2f}")

                if idx % interv == 1:
                    inv = 1.0 / cnt
                    log_text = f"[{epoch:03d}/{epochs:03d}][{idx:04d}][Train] "
                    log_text += " ".join(f"{k}: {v*inv:.6e}" for k, v in running.items())
                    logger.write(log_text + "\n")

                    HB = BS // 2
                    _d = lambda t: t.cpu().detach()
                    _s = lambda i: min(i, BS-1)
                    faces_cpu = batch.faces.cpu()
                    v_list = [
                        _d(gt_v[0]),        _d(gt_v[_s(1)]),
                        _d(gt_v[_s(HB)]),   _d(gt_v[BS-1]),
                        _d(pred_lbs[0]),    _d(pred_lbs[_s(1)]),
                        _d(pred_lbs[_s(HB)]), _d(pred_lbs[BS-1]),
                    ]
                    f_list = [faces_cpu] * len(v_list)
                    plot_image_array(
                        v_list, f_list, rot_list=[[0,0,0]]*len(v_list),
                        size=1, bg_black=False, mode='shade',
                        logdir=f"{opts.log_dir}/img/train/mesh",
                        name=f"{epoch:03d}_{idx:04d}", save=True)

                if opts.debug:
                    break

            if epoch != 0:
                self.scheduler.step()
            if writer_train:
                for k, v in running.items():
                    writer_train.add_scalar(k, v / cnt, epoch)

            if epoch % opts.save_interval == 0:
                torch.save(self.model.state_dict(),
                           f'{opts.log_dir}/model_hlbs_{epoch:03d}.pth')

            if epoch % opts.eval_iter == 0:
                self.vis_loader.visualize(
                    self.model, None, epoch,
                    save_dir=f'{opts.log_dir}/img/eval',
                    mode='hlbs',
                    smooth_n_iter=opts.smooth_n_iter,
                    no_t_mask=opts.no_t_mask,
                )

            # ── Valid ────────────────────────────────────────────────────
            self.model.eval()
            running_val = {"recon-lbs": 0.0, "recon-neu": 0.0, "total": 0.0}
            vcnt = 0

            pbar = tqdm(enumerate(valid_loader), total=len_valid, ncols=120,
                        desc=f"[{epoch:03d}] Valid FullPred")
            for idx, batch in pbar:
                batch = batch.to(self.device)
                vcnt += 1
                with torch.no_grad():
                    src_v  = batch.template
                    src_n  = batch.template_normal
                    gt_v   = batch.vertices
                    gt_n   = batch.vertices_normal

                    delta     = gt_v - src_v
                    src_in    = torch.cat([src_v, src_n], dim=-1)
                    deform_in = torch.cat([delta, gt_n, src_in], dim=-1)
                    pred_lbs  = self.model(src_v, deform_in)

                    target_v_val = gt_v
                    val_loss = F.mse_loss(target_v_val, pred_lbs).item() * opts.lambda_vert
                    running_val["recon-lbs"] += val_loss
                    running_val["total"]     += val_loss

                    if opts.lambda_neu > 0:
                        delta_zero = torch.zeros_like(src_v)
                        neu_deform_in = torch.cat([delta_zero, src_n, src_v, src_n], dim=-1)
                        pred_neutral = self.model(src_v, neu_deform_in)
                        val_neu = F.mse_loss(src_v, pred_neutral).item() * opts.lambda_neu
                        running_val["recon-neu"] += val_neu
                        running_val["total"]     += val_neu

                pbar.set_description(f"[{epoch:03d}] val lbs: {val_loss:.5e}")

                if idx % interv_val == 0:
                    BS_v = batch.vertices.shape[0]
                    HB_v = BS_v // 2
                    _s = lambda i: min(i, BS_v-1)
                    faces_cpu = batch.faces[0].cpu() if batch.faces.dim() == 3 else batch.faces.cpu()
                    v_list = [
                        gt_v[0].cpu(),        gt_v[_s(1)].cpu(),
                        gt_v[_s(HB_v)].cpu(), gt_v[BS_v-1].cpu(),
                        pred_lbs[0].cpu(),        pred_lbs[_s(1)].cpu(),
                        pred_lbs[_s(HB_v)].cpu(), pred_lbs[BS_v-1].cpu(),
                    ]
                    f_list = [faces_cpu] * len(v_list)
                    plot_image_array(
                        v_list, f_list, rot_list=[[0,0,0]]*len(v_list),
                        size=1, bg_black=False, mode='shade',
                        logdir=f"{opts.log_dir}/img/valid/mesh",
                        name=f"{epoch:03d}_{idx:04d}", save=True)

                if opts.debug:
                    break

            if writer_valid:
                for k, v in running_val.items():
                    writer_valid.add_scalar(k, v / vcnt, epoch)

            total_val = running_val["total"] / vcnt
            if total_val < BEST_LOSS:
                BEST_LOSS  = total_val
                BEST_EPOCH = epoch
                torch.save(self.model.state_dict(), f'{opts.log_dir}/model_hlbs_best.pth')
                print(f"[{epoch:03d}] Best: {BEST_LOSS:.6e} (epoch {BEST_EPOCH})")
                logger.write(f"[{epoch:03d}] Best Loss: {BEST_LOSS:.6e}\n")
            else:
                print(f"[{epoch:03d}] Val: {total_val:.6e} (Best: {BEST_LOSS:.6e} [{BEST_EPOCH}])")
                logger.write(f"[{epoch:03d}] Val: {total_val:.6e} (Best: {BEST_LOSS:.6e} [{BEST_EPOCH}])\n")


if __name__ == "__main__":
    opts = Options()

    if os.path.exists(opts.config):
        opts_yaml = yaml.load(open(opts.config), Loader=yaml.FullLoader)
        opts_dict = vars(opts)
        opts_yaml.update(opts_dict)
        opts = argparse.Namespace(**opts_yaml)

    if opts.target == 'smooth_gt':
        assert opts.smooth_n_iter > 0, "target=smooth_gt requires --smooth_n_iter > 0"

    trainer = HLBSTrainer(opts)
    if opts.full_prediction:
        trainer.train_full_prediction(epochs=opts.max_epoch)
    else:
        trainer.train(epochs=opts.max_epoch)
