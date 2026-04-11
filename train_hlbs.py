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

import warnings
warnings.filterwarnings("ignore", message="torch.sparse.SparseTensor.*is deprecated")

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
                        choices=['mf', 'biwi', 'voca', 'ict'],
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

    # regional weight constraint
    parser.add_argument("--lambda_rwc", type=float, default=0.0,
                        help='Regional weight constraint loss weight (0=disabled)')
    parser.add_argument("--rwc_alpha", type=float, default=0.5,
                        help='Min threshold = alpha * mean(Maya weight) per constrained joint')
    parser.add_argument("--rwc_adaptive", dest='rwc_adaptive', action='store_true',
                        help='Adaptive alpha: joints with fewer dominant vertices get stronger constraint')
    parser.set_defaults(rwc_adaptive=False)

    # DiffusionNet options (per-module)
    parser.add_argument("--dfn_skin", dest='dfn_skin', action='store_true',
                        help='Use DiffusionNet for skin_weight_net')
    parser.set_defaults(dfn_skin=False)
    parser.add_argument("--dfn_bind", dest='dfn_bind', action='store_true',
                        help='Use DiffusionNet for bind_pose_net')
    parser.set_defaults(dfn_bind=False)
    parser.add_argument("--dfn_exp", dest='dfn_exp', action='store_true',
                        help='Use DiffusionNet for lbs_exp_z_model')
    parser.set_defaults(dfn_exp=False)

    # NFS pretrained feature
    parser.add_argument("--nfs_feat_dir", type=str, default=None,
                        help='Directory with NFS per-identity features (*_nfs_feat.npy). '
                             'If set, skin_weight_net and bind_pose_net use these as input.')
    parser.add_argument("--nfs_concat", dest='nfs_concat', action='store_true',
                        help='Concat seg feat with pos+norm as input [262], 4 layers.')
    parser.set_defaults(nfs_concat=False)
    parser.add_argument("--adain_pos_norm", dest='adain_pos_norm', action='store_true',
                        help='AdaIN conditioning on pos+norm [6] only. Without: AdaIN on full input.')
    parser.set_defaults(adain_pos_norm=False)

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
    parser.add_argument("--val_every", type=int, default=5,
                        help='Run validation every N epochs (default: 5)')
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
    # curriculum
    parser.add_argument("--curriculum", dest='curriculum', action='store_true',
                        help='Curriculum learning: Phase 1 = ICT single-basis only, '
                             'Phase 2 = full data (ICT + MF)')
    parser.set_defaults(curriculum=False)
    parser.add_argument("--curriculum_epochs", type=int, default=30,
                        help='Number of epochs for curriculum Phase 1 (single-basis ICT)')

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

            # Enable multi-topology delta if rig has 2+ topologies loaded
            if len(rig.W_init) >= 2:
                self.model.enable_multi_topo(rig)
                print(f"[HLBS] Multi-topology delta enabled ({list(rig.W_init.keys())})")
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

    @torch.no_grad()
    def _visualize_curriculum_bases(self, epoch, save_dir):
        """Visualize all 53 ICT expression bases: GT vs pred for each basis."""
        from utils.keys import ICT_KEYS
        from utils.remesh_utils import ICT_face_model
        import igl

        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()

        ict_model = ICT_face_model()
        # Use first identity
        iden_vecs = np.load('ict_face_pt/random_identity_vecs.npy')
        id_coeff = iden_vecs[0]

        v_gt_list, v_pred_list = [], []
        for basis_idx in range(53):
            exp_coeff = np.zeros(53)
            exp_coeff[basis_idx] = 1.0

            v_num, faces_np = ict_model.get_random_v_and_f(select=0)  # fullhead
            deformed, template, _ = ict_model.apply_coeffs(id_coeff, exp_coeff, return_all=True, region=0)
            deformed = deformed[0]
            template = template[0]

            template_t = torch.tensor(template).float().unsqueeze(0).to(self.device)
            deformed_t = torch.tensor(deformed).float().unsqueeze(0).to(self.device)
            template_n = torch.tensor(igl.per_vertex_normals(template, faces_np)).float().unsqueeze(0).to(self.device)
            deformed_n = torch.tensor(igl.per_vertex_normals(deformed, faces_np)).float().unsqueeze(0).to(self.device)

            delta = deformed_t - template_t
            src_in = torch.cat([template_t, template_n], dim=-1)
            deform_in = torch.cat([delta, deformed_n, src_in], dim=-1)

            # Get NFS feat for visualization if available
            _vis_nfs_feat = None
            if hasattr(self, '_nfs_feat_cache') and self._nfs_feat_cache:
                _key = f"ict_{0:03d}"  # first identity
                if _key in self._nfs_feat_cache:
                    _f = self._nfs_feat_cache[_key]
                    N_cur = template_t.shape[1]
                    if _f.shape[0] > N_cur:
                        _f = _f[:N_cur]
                    _vis_nfs_feat = _f.unsqueeze(0)

            pred = self.model(template_t, deform_in, source_normal=template_n, nfs_feat=_vis_nfs_feat)

            faces_cpu = torch.tensor(faces_np).long()
            v_gt_list.append(deformed_t[0].cpu())
            v_pred_list.append(pred[0].cpu())

        # Plot in groups of 8 (GT row + pred row)
        for start in range(0, 53, 8):
            end = min(start + 8, 53)
            v_list = [v_gt_list[i] for i in range(start, end)]
            v_list += [v_pred_list[i] for i in range(start, end)]
            f_list = [faces_cpu] * len(v_list)
            names = [ICT_KEYS[i] for i in range(start, end)]
            plot_image_array(
                v_list, f_list, rot_list=[[0, 0, 0]] * len(v_list),
                size=1, bg_black=False, mode='shade',
                logdir=save_dir,
                name=f"{epoch:03d}_bases_{start:02d}-{end-1:02d}", save=True)

        self.model.train()

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

        # logging
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        resume_mode = opts.ckpt and opts.continue_ckpt
        if resume_mode and os.path.isdir(opts.ckpt):
            opts.log_dir = opts.ckpt
        else:
            freeze_tag = "-frozenAdapt" if opts.freeze_adapt else ""
            trans_tag = "-jTrans" if opts.use_joint_trans else ""
            sdw_tag = f"-sdw{opts.smooth_delta_W}a{opts.smooth_delta_W_alpha}" if opts.smooth_delta_W > 0 else ""
            nfs_tag = "-nfsEnc" if opts.nfs_ckpt else ""
            surf_tag = ""
            if opts.lambda_normal > 0: surf_tag += f"-nrm{opts.lambda_normal}"
            if opts.lambda_curvature > 0: surf_tag += f"-crv{opts.lambda_curvature}"
            wsm_tag = f"-Wsm{opts.lambda_W_smooth}" if opts.lambda_W_smooth > 0 else ""
            cur_tag = f"-cur{opts.curriculum_epochs}" if opts.curriculum else ""
            tag = f"-HLBS-{opts.topo_key}-s{opts.smooth_n_iter}{freeze_tag}{trans_tag}{sdw_tag}{nfs_tag}{surf_tag}{wsm_tag}{cur_tag}"
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
        if opts.smooth_delta_W > 0:
            print(f"[HLBS] delta_W forward smoothing: iters={opts.smooth_delta_W}, alpha={opts.smooth_delta_W_alpha}")

        # ── Curriculum: Phase 1 loader (ICT single-basis only) ──────────
        curriculum_transitioned = False
        if opts.curriculum:
            # Filter len_list to ICT only (mesh_data == 5)
            ict_len_list = [x for x in train_ds.len_list if x[2].item() == 5]
            # Override expression count: 53 bases × identities
            n_ict_ids = ict_len_list[0][3]
            pad = 53 % BS
            ict_len_list[0] = [53 + (BS - pad if pad else 0), ict_len_list[0][1], ict_len_list[0][2], n_ict_ids]
            train_ds.curriculum_single_basis = True
            cur_sampler = CBDdataSampler(ict_len_list, BS, shuffle=True, balance=False, is_train=True, region_min=_region_min)
            cur_loader = torch.utils.data.DataLoader(
                train_ds, batch_sampler=cur_sampler,
                collate_fn=partial(CBD_collate_wrapper, device='cpu'),
                num_workers=_nw, persistent_workers=(_nw > 0))
            print(f"[Curriculum] Phase 1: ICT single-basis for {opts.curriculum_epochs} epochs "
                  f"({len(cur_loader)} batches/epoch, {n_ict_ids} identities × 53 bases)")

        for epoch in range(opts.start_epoch, epochs + 1):
            # ── Curriculum phase transition ──────────────────────────────
            if opts.curriculum and not curriculum_transitioned:
                if epoch < opts.curriculum_epochs:
                    active_loader = cur_loader
                else:
                    # Switch to full data
                    train_ds.curriculum_single_basis = False
                    active_loader = train_loader
                    if epoch == opts.curriculum_epochs:
                        curriculum_transitioned = True
                        print(f"[Curriculum] Phase 2: switching to full data at epoch {epoch}")
            else:
                active_loader = train_loader
            # ── Train ────────────────────────────────────────────────────────
            self.model.train()
            running = {"recon-lbs": 0.0, "recon-neu": 0.0, "recon-normal": 0.0, "recon-curvature": 0.0, "lbs-W-reg": 0.0, "lbs-t-reg": 0.0, "lbs-W-smooth": 0.0, "total": 0.0}
            cnt = 0

            _len_active = len(active_loader)
            pbar = tqdm(enumerate(active_loader), total=_len_active, ncols=120,
                        desc=f"[{epoch:03d}] Train HLBS")
            for idx, batch in pbar:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                src_v  = batch.template
                src_n  = batch.template_normal
                gt_v   = batch.vertices
                gt_n   = batch.vertices_normal
                smt_v  = batch.smooth_vertices

                # Register mesh edges for this topology if not cached
                if opts.smooth_delta_W > 0:
                    N_cur = src_v.shape[1]
                    if self.model._mesh_edges_by_N is None or N_cur not in self.model._mesh_edges_by_N:
                        _faces = batch.faces[0] if batch.faces.dim() == 3 else batch.faces
                        self.model.set_mesh_edges(_faces)

                target_v = smt_v if opts.target == 'smooth_gt' else gt_v

                delta     = gt_v - src_v
                src_in    = torch.cat([src_v, src_n], dim=-1)        # [B, N, 6]
                deform_in = torch.cat([delta, gt_n, src_in], dim=-1) # [B, N, 12]

                # z_exp from NFS encoder or HLBS internal encoder
                z_exp_ext = None
                if self.nfs_encoder is not None:
                    z_exp_ext = self._nfs_encode_exp(gt_v, gt_n)      # [B, hid_dim]

                pred_lbs = self.model(src_v, deform_in, source_normal=src_n, z_exp_override=z_exp_ext)  # [B, N, 3]

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
                    pred_neutral = self.model(src_v, neu_deform_in, source_normal=src_n, nfs_feat=_nfs_feat if hasattr(self, '_nfs_feat_cache') else None)
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
                # Lazily build mesh edges per topology for smoothness loss
                edges = None
                if opts.lambda_W_smooth > 0:
                    N_cur = src_v.shape[1]
                    if not hasattr(self, '_edges_by_N'):
                        self._edges_by_N = {}
                    if N_cur not in self._edges_by_N:
                        f_raw = batch.faces.cpu()
                        f_np = f_raw[0].numpy() if f_raw.dim() == 3 else f_raw.numpy()
                        e = np.concatenate([f_np[:, [0,1]], f_np[:, [1,2]], f_np[:, [0,2]]], axis=0)
                        e = np.sort(e, axis=1)
                        e = np.unique(e, axis=0)
                        self._edges_by_N[N_cur] = torch.tensor(e, dtype=torch.long, device=self.device)
                    edges = self._edges_by_N[N_cur]
                src_feat = torch.cat([src_v, src_n], dim=-1)
                regs = self.model.reg_loss(src_feat, edges=edges)
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

                _interv = max(1, round(_len_active / 10))
                if idx % _interv == 1:
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
                    nfs_feat_cache=self._nfs_feat_cache if hasattr(self, '_nfs_feat_cache') else None,
                )
                if opts.curriculum and epoch > 0:
                    self._visualize_curriculum_bases(
                        epoch, save_dir=f'{opts.log_dir}/img/eval_bases')

            # ── Valid ────────────────────────────────────────────────────────
            if epoch == 0 or epoch % opts.val_every != 0:
                continue

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

                    pred_lbs  = self.model(src_v, deform_in, source_normal=src_n, z_exp_override=z_exp_ext)

                    target_v_val = smt_v if opts.target == 'smooth_gt' else gt_v
                    val_loss = F.mse_loss(target_v_val, pred_lbs).item() * loss_lambda["recon-lbs"]
                    running_val["recon-lbs"] += val_loss
                    running_val["total"]     += val_loss

                    # Neutral reconstruction loss (val)
                    if opts.lambda_neu > 0:
                        delta_zero    = torch.zeros_like(src_v)
                        neu_deform_in = torch.cat([delta_zero, src_n, src_v, src_n], dim=-1)
                        pred_neutral  = self.model(src_v, neu_deform_in, source_normal=src_n)
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
            dfn_skin=opts.dfn_skin,
            dfn_bind=opts.dfn_bind,
            dfn_exp=opts.dfn_exp,
            nfs_feat_dim=256 if opts.nfs_feat_dir else 0,
            nfs_concat=opts.nfs_concat if hasattr(opts, 'nfs_concat') else False,
            adain_pos_norm=opts.adain_pos_norm if hasattr(opts, 'adain_pos_norm') else False,
        ).to(self.device)
        print(f"[HLBS FullPred] {sum(p.numel() for p in self.model.parameters()):,} params")

        # Load NFS pretrained features
        self._nfs_feat_cache = {}
        if opts.nfs_feat_dir:
            import glob as _glob
            feat_files = _glob.glob(os.path.join(opts.nfs_feat_dir, '*_nfs_feat.npy'))
            for fp in feat_files:
                fname = os.path.basename(fp).replace('_nfs_feat.npy', '')
                self._nfs_feat_cache[fname] = torch.tensor(np.load(fp), dtype=torch.float32).to(self.device)  # GPU
            print(f"[NFS feat] Loaded {len(self._nfs_feat_cache)} identity features "
                  f"({sum(v.numel()*4 for v in self._nfs_feat_cache.values())/1e6:.1f} MB on GPU)")

        # Build regional weight constraints
        if opts.lambda_rwc > 0:
            self.model._build_regional_weight_constraints(alpha=opts.rwc_alpha, adaptive=opts.rwc_adaptive)

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
        surf_tag = ""
        if opts.lambda_normal > 0: surf_tag += f"-nrm{opts.lambda_normal}"
        if opts.lambda_curvature > 0: surf_tag += f"-crv{opts.lambda_curvature}"
        wsm_tag = f"-Wsm{opts.lambda_W_smooth}" if opts.lambda_W_smooth > 0 else ""
        cur_tag = f"-cur{opts.curriculum_epochs}" if opts.curriculum else ""
        tag = f"-HLBS-FullPred-{opts.topo_key}{trans_tag}{sdw_tag}{fh_tag}{surf_tag}{wsm_tag}{cur_tag}"
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

        # Determine skin_weight_net config for logging
        try:
            _sw_net = self.model.skin_weight_net
            _sw_in_dim = _sw_net.layer_in.weight.shape[1] if hasattr(_sw_net, 'layer_in') else '?'
            _sw_n_layers = len(_sw_net.layers) if hasattr(_sw_net, 'layers') else '?'
            _adain_layers = [m for m in _sw_net.adain_in.modules() if hasattr(m, 'weight') and m.weight.dim() == 2] if hasattr(_sw_net, 'adain_in') else []
            _adain_dim = _adain_layers[0].weight.shape[1] if _adain_layers else '?'
        except Exception:
            _sw_in_dim = _sw_n_layers = _adain_dim = '?'

        config_text = (
            f"=== HLBS FullPred Training ===\n"
            f"  topo_key       : {opts.topo_key}\n"
            f"  use_joint_trans: {opts.use_joint_trans}\n"
            f"  init_phase     : {opts.init_phase_epochs} epochs ({opts.init_mode})\n"
            f"  lambda_init    : {opts.lambda_init}\n"
            f"  lambda_vert    : {opts.lambda_vert}\n"
            f"  lambda_neu     : {opts.lambda_neu}\n"
            f"  lambda_rwc     : {opts.lambda_rwc} (adaptive={getattr(opts, 'rwc_adaptive', False)})\n"
            f"  nfs_feat_dir   : {opts.nfs_feat_dir}\n"
            f"  nfs_concat     : {getattr(opts, 'nfs_concat', False)}\n"
            f"  adain_pos_norm : {getattr(opts, 'adain_pos_norm', False)}\n"
            f"  skin_weight_net: in={_sw_in_dim}, layers={_sw_n_layers}, adain_in={_adain_dim}\n"
            f"  dfn_skin/bind/exp: {opts.dfn_skin}/{opts.dfn_bind}/{opts.dfn_exp}\n"
            f"==============================\n"
        )
        print(config_text)
        logger.write(config_text)
        logger.write(train_ds.get_data_config())

        # ── Precompute mesh edges ────────────────────────────────────────
        if opts.smooth_delta_W > 0:
            print(f"[FullPred] delta_W forward smoothing: iters={opts.smooth_delta_W}, alpha={opts.smooth_delta_W_alpha}")

        BEST_LOSS = 1e8
        BEST_EPOCH = 0
        len_train = len(train_loader)
        len_valid = len(valid_loader)
        interv = max(1, round(len_train / 10))
        interv_val = max(1, round(len_valid / 3))
        K = opts.init_phase_epochs

        # ── Curriculum: Phase 1 loader (ICT single-basis only) ──────────
        curriculum_transitioned = False
        if opts.curriculum:
            ict_len_list = [x for x in train_ds.len_list if x[2].item() == 5]
            n_ict_ids = ict_len_list[0][3]
            pad = 53 % BS
            ict_len_list[0] = [53 + (BS - pad if pad else 0), ict_len_list[0][1], ict_len_list[0][2], n_ict_ids]
            train_ds.curriculum_single_basis = True
            cur_sampler = CBDdataSampler(ict_len_list, BS, shuffle=True, balance=False, is_train=True, region_min=_region_min)
            cur_loader = torch.utils.data.DataLoader(
                train_ds, batch_sampler=cur_sampler,
                collate_fn=partial(CBD_collate_wrapper, device='cpu'),
                num_workers=_nw, persistent_workers=(_nw > 0))
            print(f"[Curriculum] Phase 1: ICT single-basis for {opts.curriculum_epochs} epochs "
                  f"({len(cur_loader)} batches/epoch, {n_ict_ids} identities × 53 bases)")

        for epoch in range(opts.start_epoch, epochs + 1):
            # ── Curriculum phase transition ──────────────────────────────
            if opts.curriculum and not curriculum_transitioned:
                if epoch < opts.curriculum_epochs:
                    active_loader = cur_loader
                else:
                    train_ds.curriculum_single_basis = False
                    active_loader = train_loader
                    if epoch == opts.curriculum_epochs:
                        curriculum_transitioned = True
                        print(f"[Curriculum] Phase 2: switching to full data at epoch {epoch}")
            else:
                active_loader = train_loader
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

            _len_active = len(active_loader)
            pbar = tqdm(enumerate(active_loader), total=_len_active, ncols=120,
                        desc=f"[{epoch:03d}] Train FullPred (phase{'1' if lambda_init > 0 else '2'})")
            for idx, batch in pbar:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                src_v  = batch.template
                src_n  = batch.template_normal
                gt_v   = batch.vertices
                gt_n   = batch.vertices_normal
                target_v = gt_v

                # Register mesh edges for this topology if not cached
                if opts.smooth_delta_W > 0:
                    N_cur = src_v.shape[1]
                    if self.model._mesh_edges_by_N is None or N_cur not in self.model._mesh_edges_by_N:
                        _faces = batch.faces[0] if batch.faces.dim() == 3 else batch.faces
                        self.model.set_mesh_edges(_faces)

                # Update DiffusionNet precomputes when topology changes
                if opts.dfn_skin or opts.dfn_bind or opts.dfn_exp:
                    N_cur = src_v.shape[1]
                    if not hasattr(self, '_dfn_cached_N') or self._dfn_cached_N != N_cur:
                        import trimesh as _tm
                        from utils.nfr_utils import get_dfn_info
                        _f = batch.faces[0].cpu().numpy() if batch.faces.dim() == 3 else batch.faces.cpu().numpy()
                        _v = src_v[0].cpu().numpy()
                        _mesh = _tm.Trimesh(vertices=_v, faces=_f, process=False)
                        _dfn = get_dfn_info(_mesh, cache_dir='dfn_cache', map_location=self.device)
                        self.model.update_dfn_precomputes(_dfn)
                        self._dfn_cached_N = N_cur

                # Get NFS features if available
                _nfs_feat = None
                if opts.nfs_feat_dir and self._nfs_feat_cache:
                    B_cur = src_v.shape[0]
                    N_cur = src_v.shape[1]
                    feats = []
                    for b in range(B_cur):
                        id_key = batch.id_name[b]
                        if id_key in self._nfs_feat_cache:
                            f = self._nfs_feat_cache[id_key]
                            if f.shape[0] > N_cur:
                                f = f[:N_cur]  # region select slice
                            feats.append(f)
                        else:
                            feats.append(torch.zeros(N_cur, 256, device=self.device))
                    _nfs_feat = torch.stack(feats, dim=0)  # [B, N, 256]

                delta     = gt_v - src_v
                src_in    = torch.cat([src_v, src_n], dim=-1)
                deform_in = torch.cat([delta, gt_n, src_in], dim=-1)

                pred_lbs = self.model(src_v, deform_in, source_normal=src_n, nfs_feat=_nfs_feat)

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
                    pred_neutral = self.model(src_v, neu_deform_in, source_normal=src_n, nfs_feat=_nfs_feat if hasattr(self, '_nfs_feat_cache') else None)
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
                    init_losses = self.model.init_loss(src_v, source_normal=src_n, mesh_data=_md, nfs_feat=_nfs_feat)
                    for k, v in init_losses.items():
                        loss_dict[k] = v

                # ── Regional weight constraint ──────────────────────────
                if opts.lambda_rwc > 0:
                    _md = batch.mesh_data if hasattr(batch, 'mesh_data') else None
                    _perm = getattr(batch, 'perm_idx', None)
                    rwc_losses = self.model.regional_weight_constraint_loss(
                        src_v, source_normal=src_n, mesh_data=_md, perm_idx=_perm, nfs_feat=_nfs_feat)
                    for k, v in rwc_losses.items():
                        loss_dict[k] = v

                # ── Total loss ───────────────────────────────────────────
                loss_lambda = {
                    "recon-lbs": opts.lambda_vert,
                    "recon-neu": opts.lambda_neu,
                    "recon-normal": opts.lambda_normal,
                    "L_W_init": lambda_init,
                    "L_bind_init": lambda_init,
                    "L_rwc_init": opts.lambda_rwc,
                    "L_rwc_min": opts.lambda_rwc,
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

                _interv = max(1, round(_len_active / 10))
                if idx % _interv == 1:
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
                    nfs_feat_cache=self._nfs_feat_cache if hasattr(self, '_nfs_feat_cache') else None,
                )
                if opts.curriculum and epoch > 0:
                    self._visualize_curriculum_bases(
                        epoch, save_dir=f'{opts.log_dir}/img/eval_bases')

            # ── Valid ────────────────────────────────────────────────────
            if epoch == 0 or epoch % opts.val_every != 0:
                continue

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

                    # Get NFS features for valid batch
                    _nfs_feat = None
                    if opts.nfs_feat_dir and self._nfs_feat_cache:
                        B_cur = src_v.shape[0]
                        N_cur = src_v.shape[1]
                        feats = []
                        for b in range(B_cur):
                            id_key = batch.id_name[b]
                            if id_key in self._nfs_feat_cache:
                                f = self._nfs_feat_cache[id_key]
                                if f.shape[0] > N_cur:
                                    f = f[:N_cur]
                                feats.append(f)
                            else:
                                feats.append(torch.zeros(N_cur, 256, device=self.device))
                        _nfs_feat = torch.stack(feats, dim=0)

                    delta     = gt_v - src_v
                    src_in    = torch.cat([src_v, src_n], dim=-1)
                    deform_in = torch.cat([delta, gt_n, src_in], dim=-1)
                    pred_lbs  = self.model(src_v, deform_in, source_normal=src_n, nfs_feat=_nfs_feat)

                    target_v_val = gt_v
                    val_loss = F.mse_loss(target_v_val, pred_lbs).item() * opts.lambda_vert
                    running_val["recon-lbs"] += val_loss
                    running_val["total"]     += val_loss

                    if opts.lambda_neu > 0:
                        delta_zero = torch.zeros_like(src_v)
                        neu_deform_in = torch.cat([delta_zero, src_n, src_v, src_n], dim=-1)
                        pred_neutral = self.model(src_v, neu_deform_in, source_normal=src_n, nfs_feat=_nfs_feat)
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
