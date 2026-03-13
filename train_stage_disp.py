"""
train_stage_disp.py — Stage-based LBS + DispNet training (v11).

Stage 1: LBS only → smooth_GT target. DispNet frozen.
Stage 2: Joint fine-tune (LBS + DispNet). LBS→smooth_GT, DispNet→wrinkle.
         Optionally load pre-trained DispNet from train_disp_only.py via --disp_ckpt.

Usage:
    # Stage 1 only (LBS pre-train, 200 epochs)
    python train_stage_disp.py --config configs/train.yml \
        --version 9 --use_strain --strain_mode norm \
        --smooth_n_iter 16 --lbs_pretrained_epochs 200 --max_epoch 200 \
        --use_t_mask --use_data1 --data_toggle --batch_size 16 --tb

    # Full pipeline (stage 1 → stage 2 with pre-trained DispNet)
    python train_stage_disp.py --config configs/train.yml \
        --version 9 --use_strain --strain_mode principal \
        --smooth_n_iter 16 --lbs_pretrained_epochs 200 --max_epoch 500 \
        --disp_ckpt ./ckpts_CBD/.../model_disp_best.pth \
        --use_t_mask --use_data1 --data_toggle --batch_size 16 --tb

    # Resume from stage 2 (LBS ckpt at epoch 200)
    python train_stage_disp.py --config configs/train.yml \
        --version 9 --use_strain --strain_mode norm \
        --smooth_n_iter 16 --lbs_pretrained_epochs 200 \
        --ckpt ./ckpts_CBD/...-StageDisp-... --start_epoch 200 --continue_ckpt \
        --disp_ckpt ./ckpts_CBD/.../model_disp_best.pth \
        --max_epoch 500 --use_t_mask --use_data1 --data_toggle --batch_size 16 --tb
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

# project imports
sys.path.insert(0, os.path.dirname(__file__))
from utils.matplotlib_rnd import plot_image_array
from utils.mesh_utils import calc_norm_torch, compute_strain_signal, STRAIN_MODE_DIM, taubin_smooth_np
class Logger:
    def __init__(self, file_path):
        self.file_path = file_path
    def write(self, txt):
        with open(self.file_path, 'a') as f:
            f.write(txt)
from models.NGBC import (
    NeuralGeneralizedBarycentricCoordinateLBS,
    NeuralStrainDisplacement,
)
from dataloader_CBD import CBDDataset, CBDdataSampler, CBD_collate_wrapper
from utils.vis_loader import CheckpointVisLoader

# Reuse losses from train_CBD if available
try:
    from train_CBD import (
        laplacian_loss, lbs_laplacian_loss, lbs_ent_loss,
        lbs_usage_balance_loss, rotation_loss, non_ict_loss,
    )
except ImportError:
    laplacian_loss = lbs_laplacian_loss = lbs_ent_loss = None
    lbs_usage_balance_loss = rotation_loss = non_ict_loss = None


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yml")

    # model
    parser.add_argument("--version", type=int, default=9)
    parser.add_argument("--in_type", type=int, default=1)
    parser.add_argument("--out_type", type=int, default=1)
    parser.add_argument("--num_cage_v", type=int, default=1)
    parser.add_argument("--last_activation", choices=["relu", "elu", "softmax", "softplus", "none"],
                        default="relu")
    parser.add_argument("--no_pou", dest='no_pou', action='store_true')
    parser.set_defaults(no_pou=False)
    parser.add_argument("--align_latent", dest='align_latent', action='store_true')
    parser.set_defaults(align_latent=False)

    # LBS
    parser.add_argument("--use_lbs", dest='use_lbs', action='store_true')
    parser.set_defaults(use_lbs=False)
    parser.add_argument("--vis_joint_pos", dest='vis_joint_pos', action='store_true')
    parser.set_defaults(vis_joint_pos=False)
    parser.add_argument("--use_hyb_delta_lbs_input", dest='use_hyb_delta_lbs_input', action='store_true')
    parser.set_defaults(use_hyb_delta_lbs_input=False)
    parser.add_argument("--use_lbs_joint_center", dest='use_lbs_joint_center', action='store_true')
    parser.set_defaults(use_lbs_joint_center=False)
    parser.add_argument("--use_joint_predict", dest='use_joint_predict', action='store_true')
    parser.set_defaults(use_joint_predict=False)
    parser.add_argument("--use_exp_joint_predict", dest='use_exp_joint_predict', action='store_true')
    parser.set_defaults(use_exp_joint_predict=False)
    parser.add_argument("--no_use_translation", dest='no_use_translation', action='store_true')
    parser.set_defaults(no_use_translation=False)
    parser.add_argument("--use_weighted_joint_pos", dest='use_weighted_joint_pos', action='store_true')
    parser.set_defaults(use_weighted_joint_pos=False)
    parser.add_argument("--num_lbs_joints", type=int, default=4)
    parser.add_argument("--lbs_pretrained_epochs", type=int, default=200,
                        help='Epoch at which stage transitions from 1→2')

    # LBS regularizers
    parser.add_argument("--use_lbs_laplacian", dest='use_lbs_laplacian', action='store_true')
    parser.set_defaults(use_lbs_laplacian=False)
    parser.add_argument("--use_lbs_ent", dest='use_lbs_ent', action='store_true')
    parser.set_defaults(use_lbs_ent=False)
    parser.add_argument("--use_lbs_t", dest='use_lbs_t', action='store_true')
    parser.set_defaults(use_lbs_t=False)
    parser.add_argument("--use_lbs_R", dest='use_lbs_R', action='store_true')
    parser.set_defaults(use_lbs_R=False)
    parser.add_argument("--use_lbs_bal", dest='use_lbs_bal', action='store_true')
    parser.set_defaults(use_lbs_bal=False)

    # general losses
    parser.add_argument("--use_laplacian", dest='use_laplacian', action='store_true')
    parser.set_defaults(use_laplacian=False)
    parser.add_argument("--use_normal_loss", dest='use_normal_loss', action='store_true')
    parser.set_defaults(use_normal_loss=False)

    # strain displacement
    parser.add_argument("--use_strain", dest='use_strain', action='store_true')
    parser.set_defaults(use_strain=False)
    parser.add_argument("--strain_dim", type=int, default=1)
    parser.add_argument("--strain_mode", type=str, default='norm',
                        choices=['norm', 'norm_trace', 'full', 'principal', 'local'])
    parser.add_argument("--strain_full_grad", dest='strain_full_grad', action='store_true')
    parser.set_defaults(strain_full_grad=False)
    parser.add_argument("--use_source_template", dest='use_source_template', action='store_true')
    parser.set_defaults(use_source_template=False)
    parser.add_argument("--no_exp_z", dest='no_exp_z', action='store_true',
                        help='DispNet without exp_z encoder (plain pointwise MLP)')
    parser.set_defaults(no_exp_z=False)
    parser.add_argument("--smooth_n_iter", type=int, required=True)
    parser.add_argument("--use_strain_match", dest='use_strain_match', action='store_true',
                        help='Add strain matching loss: strain(smooth_GT,tmpl) vs strain(pred_LBS,tmpl)')
    parser.set_defaults(use_strain_match=False)
    parser.add_argument("--lambda_strain_match", type=float, default=0.01)
    parser.add_argument("--strain_match_loss_type", type=str, default='mse',
                        choices=['mse', 'l1', 'smooth_l1'],
                        help='Loss function for strain matching')
    parser.add_argument("--strain_match_mode", type=str, default=None,
                        choices=['norm', 'norm_trace', 'full', 'principal', 'local'],
                        help='Strain mode for matching loss (default: same as --strain_mode)')
    parser.add_argument("--use_true_edd", dest='use_true_edd', action='store_true',
                        help='true EDD target: (GT-sGT)-(T-sT) instead of GT-sGT')
    parser.set_defaults(use_true_edd=False)

    # training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--eval_iter", type=int, default=25,
                        help='Visualize every N epochs (strain heatmaps + mesh)')
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
    parser.set_defaults(data_toggle=False)

    # checkpoint
    parser.add_argument("--ckpt", type=str, default=None,
                        help='LBS checkpoint dir (for resume or stage 2 init)')
    parser.add_argument("--disp_ckpt", type=str, default=None,
                        help='Pre-trained DispNet checkpoint (from train_disp_only.py)')
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--continue_ckpt", dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)

    # logging
    parser.add_argument("--log_dir", type=str, default="./ckpts_CBD")
    parser.add_argument("--tb", dest='tb', action='store_true')
    parser.set_defaults(tb=False)
    parser.add_argument("--debug", dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument("--vis_frames", type=str, default="config/vis_frames.yml",
                        help='YAML config for eval_iter visualization frames')

    return parser.parse_args()


class StageDispTrainer:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if opts.use_strain:
            opts.strain_dim = STRAIN_MODE_DIM[opts.strain_mode]
        # Default strain_match_mode to strain_mode if not specified
        if opts.strain_match_mode is None:
            opts.strain_match_mode = opts.strain_mode

        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)

        # LBS model
        last_act_list = ["relu", "elu", "softmax", "softplus", "none"]
        last_act_list = [opts.last_activation == l for l in last_act_list]
        self.model = NeuralGeneralizedBarycentricCoordinateLBS(
            opts, num_layers=4,
            num_cage_vertices=opts.num_cage_v,
            use_exp_recon=False, use_shp_recon=False, use_shp=False,
            use_relu=last_act_list[0], use_elu=last_act_list[1],
            use_softmax=last_act_list[2], use_softplus=last_act_list[3],
            no_activation=last_act_list[4],
            is_train=True, use_pou=~opts.no_pou, device=self.device,
            hid_dim=128 if opts.align_latent else 256,
        )

        # DispNet
        strain_dim = opts.strain_dim if opts.use_strain else 0
        self.model_disp = NeuralStrainDisplacement(
            opts, hid_dim=256, num_layers=4,
            strain_dim=strain_dim, device=self.device,
            use_source_template=opts.use_source_template,
            no_exp_z=opts.no_exp_z,
        )

        # Load checkpoints
        if opts.ckpt and opts.continue_ckpt:
            self._load_weight(self.model, "lbs", opts.ckpt, opts.start_epoch)
            # Try loading disp from same ckpt dir (stage 2 resume)
            disp_path = sorted(glob.glob(os.path.join(opts.ckpt, f"model_disp_{opts.start_epoch:03d}.pth")))
            if disp_path:
                self.model_disp.load_state_dict(torch.load(disp_path[0]))
                print(f"Loaded DispNet from ckpt: {disp_path[0]}")

        # Pre-trained DispNet (from train_disp_only.py)
        self._has_pretrained_disp = False
        if opts.disp_ckpt and os.path.exists(opts.disp_ckpt):
            self.model_disp.load_state_dict(torch.load(opts.disp_ckpt))
            self._has_pretrained_disp = True
            print(f"Loaded pre-trained DispNet: {opts.disp_ckpt}")

        self.vis_loader = CheckpointVisLoader(opts, device=self.device)

        # neutral_detail cache for true EDD: {id_name: tensor [V, 3]}
        self._neutral_detail_cache = {}

    def _get_neutral_detail(self, batch):
        """Compute and cache neutral_detail = template - smooth(template) per identity."""
        id_name = batch.id_name
        if id_name not in self._neutral_detail_cache:
            tmpl_np = batch.template[0].cpu().numpy()
            faces_np = batch.faces.cpu().numpy()
            smooth_tmpl_np = taubin_smooth_np(tmpl_np, faces_np, n_iter=self.opts.smooth_n_iter)
            nd = tmpl_np - smooth_tmpl_np  # [V, 3]
            self._neutral_detail_cache[id_name] = torch.tensor(nd).float().to(self.device)
        return self._neutral_detail_cache[id_name]  # [V, 3]

    def _load_weight(self, model, name, ckpt_dir, epoch):
        if epoch > 0:
            paths = sorted(glob.glob(os.path.join(ckpt_dir, f"model_{name}_{epoch:03d}.pth")))
        else:
            paths = sorted(glob.glob(os.path.join(ckpt_dir, f"model_{name}_best.pth")))
        if paths:
            model.load_state_dict(torch.load(paths[0]))
            print(f"Loaded {name}: {paths[0]}")
        else:
            print(f"Warning: no {name} checkpoint found in {ckpt_dir}")

    def _stage_banner(self, stage, logger=None):
        """Print a clear stage banner to stdout and logger."""
        if stage == 1:
            lines = [
                "",
                "=" * 60,
                "  STAGE 1: LBS-only training",
                "    Target: smooth_GT (Taubin n_iter={})".format(self.opts.smooth_n_iter),
                "    DispNet: FROZEN (not updated)",
                "    Trainable params: LBS network only",
                "=" * 60,
                "",
            ]
        elif stage == 2:
            disp_info = "w/ pretrained DispNet" if self._has_pretrained_disp else "w/o pretrained DispNet (random init)"
            lines = [
                "",
                "=" * 60,
                "  STAGE 2: Joint training (LBS + DispNet)",
                "    LBS target: smooth_GT | DispNet target: wrinkle (GT - smooth_GT)",
                "    Strain mode: {} (dim={})".format(self.opts.strain_mode, self.opts.strain_dim),
                "    DispNet: {} ".format(disp_info),
                "    Trainable params: LBS + DispNet",
                "=" * 60,
                "",
            ]
        else:
            lines = [f"  Unknown stage: {stage}"]

        text = "\n".join(lines)
        print(text)
        if logger:
            logger.write(text + "\n")

    def train(self, epochs):
        opts = self.opts
        BS = opts.batch_size

        # Determine initial stage
        lbs_pretrained_epochs = opts.lbs_pretrained_epochs
        if opts.start_epoch >= lbs_pretrained_epochs:
            stage = 2
        else:
            stage = 1

        def set_stage(s):
            if s == 1:
                for p in self.model.parameters():
                    p.requires_grad_(True)
                for p in self.model_disp.parameters():
                    p.requires_grad_(False)
            elif s == 2:
                for p in self.model.parameters():
                    p.requires_grad_(True)
                for p in self.model_disp.parameters():
                    p.requires_grad_(True)

        set_stage(stage)

        def build_optimizer():
            params = [p for p in self.model.parameters() if p.requires_grad]
            params += [p for p in self.model_disp.parameters() if p.requires_grad]
            return torch.optim.AdamW(params, lr=opts.lr, betas=(0.9, 0.999))

        self.optimizer = build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opts.sc_step, gamma=opts.sc_gamma)

        # datasets
        train_ds = CBDDataset(opts, is_train=True, toggle=opts.data_toggle)
        valid_ds = CBDDataset(opts, is_valid=True, toggle=opts.data_toggle)
        train_sampler = CBDdataSampler(train_ds.len_list, BS, shuffle=True, balance=False, is_train=True)
        valid_sampler = CBDdataSampler(valid_ds.len_list, BS, shuffle=True, balance=False, is_valid=True)
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
        if resume_mode:
            opts.log_dir = opts.ckpt
        else:
            sm_tag = f"-sm{opts.strain_match_mode}" if opts.strain_match_mode != opts.strain_mode else ""
            edd_tag = "-trueEDD" if opts.use_true_edd else ""
            tag = f"-StageDisp-{opts.strain_mode}{sm_tag}-s{opts.smooth_n_iter}{edd_tag}"
            opts.log_dir = os.path.join(opts.log_dir, now + tag)

        os.makedirs(opts.log_dir, exist_ok=True)
        os.makedirs(f"{opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{opts.log_dir}/img/valid/mesh", exist_ok=True)

        with open(os.path.join(opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(opts), f, indent=4)

        if opts.tb:
            writer_train = SummaryWriter(log_dir=os.path.join(opts.log_dir, "train"))
            writer_valid = SummaryWriter(log_dir=os.path.join(opts.log_dir, "valid"))
        else:
            writer_train = writer_valid = None

        logger = Logger(os.path.join(opts.log_dir, "log.txt"))
        print(f'Log: {logger.file_path}')

        config_text = (
            f"=== StageDisp Training ===\n"
            f"  strain_mode: {opts.strain_mode} (dim={opts.strain_dim})\n"
            f"  smooth_n_iter: {opts.smooth_n_iter}\n"
            f"  lbs_pretrained_epochs: {lbs_pretrained_epochs}\n"
            f"  disp_ckpt: {opts.disp_ckpt}\n"
            f"==========================\n"
        )
        print(config_text)
        logger.write(config_text)
        logger.write(train_ds.get_data_config())
        logger.write(self.model.get_model_config())
        logger.write(str(self.model_disp) + "\n")

        # Print initial stage banner
        self._stage_banner(stage, logger)

        # loss lambdas
        loss_lambda = {
            "recon-def": opts.lambda_vert,
            "recon-neu": opts.lambda_vert,
            "recon-lbs": opts.lambda_vert,
            "recon-wrinkle": opts.lambda_vert,
            "exp-v": opts.lambda_vert,
            "shape": opts.lambda_vert,
        }
        if opts.use_lbs_laplacian:
            loss_lambda['lbs-lap'] = 1e-2
        if opts.use_lbs_ent:
            loss_lambda['lbs-ent'] = 1e-3
        if opts.use_lbs_t:
            loss_lambda['lbs-t'] = 1e-2
        if opts.use_lbs_R:
            loss_lambda['lbs-R'] = 1e-3
        if opts.use_lbs_bal:
            loss_lambda['lbs-bal'] = 1e-4
        if opts.use_strain_match:
            loss_lambda['strain-match'] = opts.lambda_strain_match

        # training loop
        BEST_LOSS = 1e8
        BEST_EPOCH = 0
        len_train = len(train_loader)
        len_valid = len(valid_loader)
        interv = max(1, round(len_train / 10))
        interv_val = max(1, round(len_valid / 4))

        for epoch in range(opts.start_epoch, epochs + 1):
            # Stage transition
            if stage == 1 and epoch >= lbs_pretrained_epochs:
                stage = 2
                set_stage(2)
                # Load DispNet if provided and not already loaded at init
                if opts.disp_ckpt and os.path.exists(opts.disp_ckpt):
                    self.model_disp.load_state_dict(torch.load(opts.disp_ckpt))
                    self._has_pretrained_disp = True
                self.optimizer = build_optimizer()
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=opts.sc_step, gamma=opts.sc_gamma)
                self._stage_banner(stage, logger)

            stag = f"Stage{stage}"
            if stage == 1:
                pbar_prefix = f"[{epoch:03d}/{epochs:03d}][{stag}] LBS-only"
            else:
                disp_tag = "pretrained" if self._has_pretrained_disp else "scratch"
                pbar_prefix = f"[{epoch:03d}/{epochs:03d}][{stag}] Joint(LBS+Disp:{disp_tag})"

            # --- Train ---
            self.model.train()
            self.model_disp.train() if stage == 2 else self.model_disp.eval()

            running = {"recon-lbs": 0.0, "recon-def": 0.0, "total": 0.0}
            if opts.use_strain_match:
                running["strain-match"] = 0.0
            if stage == 2:
                running["recon-wrinkle"] = 0.0
            cnt = 0

            pbar = tqdm(enumerate(train_loader), total=len_train, ncols=120)
            for idx, batch in pbar:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                tmpl_v = batch.template
                tmpl_n = batch.template_normal
                gt_v = batch.vertices
                gt_n = batch.vertices_normal
                smooth_v = batch.smooth_vertices

                # 1. LBS forward
                pred_lbs, recon_vertices, recon_source, exp_z, pred_source, t_mask, key_d, pred_key_weight, W_lbs, T_lbs = self.model(
                    tmpl_v, gt_v, tmpl_n, gt_n, batch.mesh_data, epoch=epoch)

                if not opts.no_t_mask:
                    inv_t_mask = 1.0 - t_mask
                else:
                    t_mask = 1.0
                    inv_t_mask = 0.0

                loss_dict = {}
                mesh_data_num = batch.mesh_data.cpu().numpy().astype(int)

                # LBS → smooth_GT
                if opts.no_t_mask:
                    loss_dict['recon-lbs'] = F.mse_loss(smooth_v, pred_lbs)
                else:
                    loss_dict['recon-lbs'] = F.mse_loss(smooth_v * t_mask, pred_lbs * t_mask)
                    loss_dict['recon-lbs'] += F.mse_loss(tmpl_v * inv_t_mask, pred_lbs * inv_t_mask)

                # Strain matching: strain(smooth_GT, tmpl) ≈ strain(pred_LBS, tmpl)
                if opts.use_strain_match and opts.use_strain:
                    _nsi = getattr(batch, 'neutral_span_inv', None)
                    gt_strain = compute_strain_signal(
                        smooth_v, tmpl_v, batch.faces,
                        mode=opts.strain_match_mode, neutral_span_inv=_nsi)
                    pred_strain = compute_strain_signal(
                        pred_lbs, tmpl_v, batch.faces,
                        mode=opts.strain_match_mode, neutral_span_inv=_nsi)
                    _sm_fn = {'mse': F.mse_loss, 'l1': F.l1_loss, 'smooth_l1': F.smooth_l1_loss}[opts.strain_match_loss_type]
                    if opts.no_t_mask:
                        loss_dict['strain-match'] = _sm_fn(gt_strain, pred_strain)
                    else:
                        loss_dict['strain-match'] = _sm_fn(gt_strain * t_mask, pred_strain * t_mask)

                if stage == 2:
                    # 2. Strain (from LBS output)
                    strain = None
                    if opts.use_strain:
                        lbs_for_strain = pred_lbs.detach() if not opts.strain_full_grad else pred_lbs
                        _nsi = getattr(batch, 'neutral_span_inv', None)
                        strain = compute_strain_signal(
                            lbs_for_strain, tmpl_v, batch.faces,
                            mode=opts.strain_mode, neutral_span_inv=_nsi)

                    # 3. DispNet
                    lbs_norm = calc_norm_torch(pred_lbs, batch.faces, at='verts')
                    displacement, _ = self.model_disp(
                        pred_lbs, lbs_norm,
                        source_vert=tmpl_v if opts.use_source_template else None,
                        source_norm=tmpl_n if opts.use_source_template else None,
                        strain=strain)
                    if not opts.no_t_mask:
                        displacement = displacement * t_mask
                    pred_vertices = pred_lbs + displacement

                    # Wrinkle loss (true EDD subtracts neutral detail)
                    wrinkle_target = gt_v - smooth_v
                    if opts.use_true_edd:
                        neutral_detail = self._get_neutral_detail(batch)
                        wrinkle_target = wrinkle_target - neutral_detail.unsqueeze(0)
                    if opts.no_t_mask:
                        loss_dict['recon-wrinkle'] = F.mse_loss(wrinkle_target, displacement)
                    else:
                        loss_dict['recon-wrinkle'] = F.mse_loss(wrinkle_target * t_mask, displacement * t_mask)

                    # End-to-end recon
                    if opts.no_t_mask:
                        loss_dict['recon-def'] = F.mse_loss(gt_v, pred_vertices)
                    else:
                        loss_dict['recon-def'] = F.mse_loss(gt_v * t_mask, pred_vertices * t_mask)
                        loss_dict['recon-def'] += F.mse_loss(tmpl_v * inv_t_mask, pred_vertices * inv_t_mask)
                else:
                    # Stage 1: recon-def = LBS vs smooth_GT (same target)
                    loss_dict['recon-def'] = loss_dict['recon-lbs'].clone()

                # Neutral recon
                if self.model.use_full_vertex:
                    if opts.no_t_mask:
                        loss_dict['recon-neu'] = F.mse_loss(tmpl_v, pred_source)
                    else:
                        loss_dict['recon-neu'] = F.mse_loss(tmpl_v * t_mask, pred_source * t_mask)
                        loss_dict['recon-neu'] += F.mse_loss(tmpl_v * inv_t_mask, pred_source * inv_t_mask)

                # LBS regularizers
                if opts.use_lbs_laplacian and lbs_laplacian_loss:
                    loss_dict['lbs-lap'] = lbs_laplacian_loss(
                        batch, W_lbs, train_ds, mesh_data_num, self.device) / BS
                if opts.use_lbs_ent and lbs_ent_loss:
                    loss_dict['lbs-ent'] = lbs_ent_loss(W_lbs)
                if opts.use_lbs_t:
                    loss_dict['lbs-t'] = torch.mean(T_lbs[:, :, :3, 3] ** 2)
                if opts.use_lbs_R and rotation_loss:
                    loss_dict['lbs-R'] = rotation_loss(T_lbs)
                if opts.use_lbs_bal and lbs_usage_balance_loss:
                    loss_dict['lbs-bal'] = lbs_usage_balance_loss(W_lbs)

                # Total
                loss = 0
                for key, value in loss_dict.items():
                    if key in loss_lambda:
                        tmp = value * loss_lambda[key]
                        loss += tmp
                        if key in running:
                            running[key] += tmp.item() if hasattr(tmp, 'item') else tmp

                loss.backward()
                self.optimizer.step()

                running["total"] += loss.item()
                cnt += 1
                pbar.set_description(f"{pbar_prefix} | loss: {loss:.5e}")

                # Vis
                if idx % interv == 1:
                    log_text = f"[{epoch:03d}][{idx:04d}][{stag}] "
                    inv = 1.0 / cnt
                    for k, v in running.items():
                        log_text += f"{k}: {v*inv:.6e} "
                    logger.write(log_text + "\n")

                    HB = BS // 2
                    faces_cpu = batch.faces.cpu()
                    v_list = [
                        gt_v[0].cpu().detach(), gt_v[min(1,BS-1)].cpu().detach(),
                        gt_v[min(HB,BS-1)].cpu().detach(), gt_v[BS-1].cpu().detach(),
                        smooth_v[0].cpu().detach(), smooth_v[min(1,BS-1)].cpu().detach(),
                        smooth_v[min(HB,BS-1)].cpu().detach(), smooth_v[BS-1].cpu().detach(),
                        pred_lbs[0].cpu().detach(), pred_lbs[min(1,BS-1)].cpu().detach(),
                        pred_lbs[min(HB,BS-1)].cpu().detach(), pred_lbs[BS-1].cpu().detach(),
                    ]
                    if stage == 2:
                        v_list += [
                            pred_vertices[0].cpu().detach(), pred_vertices[min(1,BS-1)].cpu().detach(),
                            pred_vertices[min(HB,BS-1)].cpu().detach(), pred_vertices[BS-1].cpu().detach(),
                        ]
                    f_list = [faces_cpu] * len(v_list)
                    plot_image_array(
                        v_list, f_list,
                        rot_list=[[0,0,0]] * len(v_list),
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

            # Save
            if epoch % opts.save_interval == 0:
                torch.save(self.model.state_dict(), f'{opts.log_dir}/model_lbs_{epoch:03d}.pth')
                if stage == 2:
                    torch.save(self.model_disp.state_dict(), f'{opts.log_dir}/model_disp_{epoch:03d}.pth')

            # Eval-iter visualization (real frames via vis_loader)
            if epoch % opts.eval_iter == 0 and opts.use_strain:
                eval_vis_dir = f"{opts.log_dir}/img/eval"
                self.vis_loader.visualize(
                    self.model, self.model_disp, epoch, eval_vis_dir,
                    mode='stage_disp', stage=stage,
                    strain_mode=opts.strain_mode,
                    smooth_n_iter=opts.smooth_n_iter,
                    no_t_mask=opts.no_t_mask,
                    use_source_template=opts.use_source_template,
                )

            # --- Valid ---
            self.model.eval()
            self.model_disp.eval()

            running_val = {"recon-lbs": 0.0, "recon-def": 0.0, "total": 0.0}
            if stage == 2:
                running_val["recon-wrinkle"] = 0.0
            vcnt = 0

            if stage == 1:
                val_prefix = f"[{epoch:03d}][{stag}] Valid LBS-only"
            else:
                disp_tag = "pretrained" if self._has_pretrained_disp else "scratch"
                val_prefix = f"[{epoch:03d}][{stag}] Valid Joint(LBS+Disp:{disp_tag})"

            pbar = tqdm(enumerate(valid_loader), total=len_valid, ncols=120, desc=val_prefix)
            for idx, batch in pbar:
                batch = batch.to(self.device)
                vcnt += 1
                with torch.no_grad():
                    pred_lbs, _, _, _, pred_source, t_mask_v, _, _, W_lbs, T_lbs = self.model(
                        batch.template, batch.vertices,
                        batch.template_normal, batch.vertices_normal,
                        batch.mesh_data, epoch=epoch)

                    loss_dict = {}
                    loss_dict['recon-lbs'] = F.mse_loss(batch.smooth_vertices, pred_lbs)

                    if stage == 2:
                        _nsi = getattr(batch, 'neutral_span_inv', None)
                        strain = compute_strain_signal(
                            pred_lbs, batch.template, batch.faces,
                            mode=opts.strain_mode, neutral_span_inv=_nsi) if opts.use_strain else None
                        lbs_norm = calc_norm_torch(pred_lbs, batch.faces, at='verts')
                        displacement, _ = self.model_disp(
                            pred_lbs, lbs_norm,
                            source_vert=batch.template if opts.use_source_template else None,
                            source_norm=batch.template_normal if opts.use_source_template else None,
                            strain=strain)
                        if not opts.no_t_mask:
                            displacement = displacement * t_mask_v
                        pred_vertices = pred_lbs + displacement
                        wrinkle_target = batch.vertices - batch.smooth_vertices
                        if opts.use_true_edd:
                            neutral_detail = self._get_neutral_detail(batch)
                            wrinkle_target = wrinkle_target - neutral_detail.unsqueeze(0)
                        loss_dict['recon-wrinkle'] = F.mse_loss(wrinkle_target, displacement)
                        loss_dict['recon-def'] = F.mse_loss(batch.vertices, pred_vertices)
                    else:
                        loss_dict['recon-def'] = loss_dict['recon-lbs'].clone()

                    if self.model.use_full_vertex:
                        loss_dict['recon-neu'] = F.mse_loss(batch.template, pred_source)

                    loss = 0
                    for k, v in loss_dict.items():
                        if k in loss_lambda:
                            val = v.item() if hasattr(v, 'item') else v
                            tmp = val * loss_lambda[k]
                            loss += tmp
                            if k in running_val:
                                running_val[k] += tmp

                running_val["total"] += loss
                pbar.set_description(f"{val_prefix} | loss: {loss:.5e}")

                # Per-iter valid mesh visualization (same layout as train/mesh)
                if idx % interv_val == 0:
                    BS = batch.vertices.shape[0]
                    HB = BS // 2
                    faces_cpu = batch.faces.cpu()
                    v_list = [
                        batch.vertices[0].cpu(), batch.vertices[min(1,BS-1)].cpu(),
                        batch.vertices[min(HB,BS-1)].cpu(), batch.vertices[BS-1].cpu(),
                        batch.smooth_vertices[0].cpu(), batch.smooth_vertices[min(1,BS-1)].cpu(),
                        batch.smooth_vertices[min(HB,BS-1)].cpu(), batch.smooth_vertices[BS-1].cpu(),
                        pred_lbs[0].cpu().detach(), pred_lbs[min(1,BS-1)].cpu().detach(),
                        pred_lbs[min(HB,BS-1)].cpu().detach(), pred_lbs[BS-1].cpu().detach(),
                    ]
                    if stage == 2:
                        v_list += [
                            pred_vertices[0].cpu().detach(), pred_vertices[min(1,BS-1)].cpu().detach(),
                            pred_vertices[min(HB,BS-1)].cpu().detach(), pred_vertices[BS-1].cpu().detach(),
                        ]
                    f_list = [faces_cpu] * len(v_list)
                    plot_image_array(
                        v_list, f_list,
                        rot_list=[[0,0,0]] * len(v_list),
                        size=1, bg_black=False, mode='shade',
                        logdir=f"{opts.log_dir}/img/valid/mesh",
                        name=f"{epoch:03d}_{idx:04d}", save=True)

                if opts.debug:
                    break

            if writer_valid:
                for k, v in running_val.items():
                    writer_valid.add_scalar(k, v / vcnt, epoch)

            val_loss = running_val["total"] / vcnt
            if val_loss < BEST_LOSS:
                BEST_LOSS = val_loss
                BEST_EPOCH = epoch
                print(f"[{epoch:03d}][{stag}] Best: {BEST_LOSS:.6e} (epoch {BEST_EPOCH})")
                logger.write(f"[{epoch:03d}] Best Loss: {BEST_LOSS:.6e}\n")
                torch.save(self.model.state_dict(), f'{opts.log_dir}/model_lbs_best.pth')
                if stage == 2:
                    torch.save(self.model_disp.state_dict(), f'{opts.log_dir}/model_disp_best.pth')
            else:
                print(f"[{epoch:03d}][{stag}] Val: {val_loss:.6e} (Best: {BEST_LOSS:.6e} [{BEST_EPOCH}])")
                logger.write(f"[{epoch:03d}] Val: {val_loss:.6e} (Best: {BEST_LOSS:.6e} [{BEST_EPOCH}])\n")


if __name__ == "__main__":
    opts = Options()

    if os.path.exists(opts.config):
        opts_yaml = yaml.load(open(opts.config), Loader=yaml.FullLoader)
        opts_dict = vars(opts)
        opts_yaml.update(opts_dict)
        opts = argparse.Namespace(**opts_yaml)

    assert opts.smooth_n_iter > 0, "train_stage_disp requires --smooth_n_iter > 0"

    trainer = StageDispTrainer(opts)
    trainer.train(epochs=opts.max_epoch)
