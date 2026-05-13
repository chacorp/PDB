"""
train_facial_anim.py — Stage-based HLBS + EDD training.

Stage 1: HLBS only → smooth_GT.  EDD frozen/skipped.
Stage 2: Joint (HLBS + EDD).     HLBS → smooth_GT,  EDD → wrinkle (GT - smooth_GT).
         Optionally load pre-trained EDD from train_edd.py via --edd_ckpt.

Usage:
    # Stage 1 only (HLBS pre-train, 200 epochs)
    python train_facial_anim.py --config configs/train.yml \
        --rig_path utils/mf/rig_info.json --topo_key mf \
        --neutral_obj utils/mf/mf_aligned_mean.obj \
        --num_identities 13 --smooth_n_iter 16 \
        --hlbs_pretrained_epochs 200 --max_epoch 200 \
        --use_data1 --data_toggle --batch_size 8 --tb

    # Full pipeline (stage 1 → stage 2 with pre-trained EDD)
    python train_facial_anim.py --config configs/train.yml \
        --rig_path utils/mf/rig_info.json --topo_key mf \
        --neutral_obj utils/mf/mf_aligned_mean.obj \
        --num_identities 13 --smooth_n_iter 16 \
        --hlbs_pretrained_epochs 200 --max_epoch 500 \
        --edd_ckpt ./ckpts_edd/.../model_edd_best.pth \
        --use_data1 --data_toggle --batch_size 8 --tb

    # Resume from stage 2
    python train_facial_anim.py ... \
        --ckpt ./ckpts_facial/.../  --start_epoch 200 --continue_ckpt \
        --edd_ckpt ./ckpts_edd/.../model_edd_best.pth \
        --hlbs_pretrained_epochs 200 --max_epoch 500 --tb
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
from utils.mesh_utils import compute_jacobian_features
from dataloader_CBD import CBDDataset, CBDdataSampler, CBD_collate_wrapper
from utils.vis_loader import CheckpointVisLoader


class Logger:
    def __init__(self, file_path):
        self.file_path = file_path
    def write(self, txt):
        with open(self.file_path, 'a') as f:
            f.write(txt)


def _load_obj_verts_faces(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == 'v':
                verts.append([float(x) for x in tok[1:4]])
            elif tok[0] == 'f':
                idx = [int(t.split('/')[0]) - 1 for t in tok[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yml")

    # rig / topology
    parser.add_argument("--rig_path", type=str, required=True)
    parser.add_argument("--topo_key", type=str, default='mf',
                        choices=['mf', 'biwi', 'voca'])
    parser.add_argument("--neutral_obj", type=str, required=True,
                        help='Neutral mesh .obj for DiffusionNet operators')
    parser.add_argument("--op_cache_dir", type=str, default='utils/diffusion_ops')
    parser.add_argument("--num_identities", type=int, default=13)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)

    # EDD architecture
    parser.add_argument("--edd_hid_channels", type=int, default=128)
    parser.add_argument("--edd_n_blocks", type=int, default=4)
    parser.add_argument("--edd_k_eig", type=int, default=128)

    # regularization
    parser.add_argument("--lambda_W_reg", type=float, default=1e-4)
    parser.add_argument("--lambda_t_reg", type=float, default=1e-4)

    # target decomposition
    parser.add_argument("--lbs_target", type=str, default='gt',
                        choices=['gt', 'smooth_gt'],
                        help='"gt": HLBS → GT (default). "smooth_gt": HLBS → smooth_GT.')
    parser.add_argument("--edd_target", type=str, default='lbs_residual',
                        choices=['lbs_residual', 'smooth_gt'],
                        help='"lbs_residual": EDD target = GT - pred_LBS (default). '
                             '"smooth_gt": EDD target = GT - smooth_GT.')
    parser.add_argument("--smooth_n_iter", type=int, default=0,
                        help='Taubin smoothing iters (only needed when lbs_target or edd_target = smooth_gt)')

    # stage control
    parser.add_argument("--hlbs_pretrained_epochs", type=int, default=200,
                        help='Epoch at which training transitions from stage 1 to stage 2')

    # training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--eval_iter", type=int, default=25)
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

    # checkpoints
    parser.add_argument("--ckpt", type=str, default=None,
                        help='Checkpoint dir (for resume: both models loaded from here)')
    parser.add_argument("--edd_ckpt", type=str, default=None,
                        help='Pre-trained EDD checkpoint (from train_edd.py)')
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--continue_ckpt", dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)

    # logging
    parser.add_argument("--log_dir", type=str, default="./ckpts_facial")
    parser.add_argument("--tb", dest='tb', action='store_true')
    parser.set_defaults(tb=False)
    parser.add_argument("--debug", dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument("--vis_frames", type=str, default="config/vis_frames.yml")

    # dataloader compat
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

    return parser.parse_args()


class FacialAnimTrainer:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)

        from utils.rig_loader import load_rig
        from models.hierarchical_lbs import HierarchicalLBS
        from models.diffusion_edd import DiffusionNetEDD

        # ── HLBS ──────────────────────────────────────────────────────────
        rig = load_rig(opts.rig_path)
        self.model_hlbs = HierarchicalLBS(
            rig=rig,
            topology=opts.topo_key,
            in_dim_exp=12,
            hid_dim=opts.hid_dim,
            num_layers=opts.num_layers,
            device=str(self.device),
        ).to(self.device)

        # ── EDD ───────────────────────────────────────────────────────────
        z_dim = opts.hid_dim
        self.model_edd = DiffusionNetEDD(
            z_dim=z_dim,
            hid_channels=opts.edd_hid_channels,
            n_blocks=opts.edd_n_blocks,
            k_eig=opts.edd_k_eig,
        ).to(self.device)

        verts_np, faces_np = _load_obj_verts_faces(opts.neutral_obj)
        self.model_edd.register_topology(
            opts.topo_key, verts_np, faces_np,
            op_cache_dir=opts.op_cache_dir,
            cuda_device=str(self.device),
        )
        self.model_edd.set_topology(opts.topo_key, self.device)
        self.faces_t = torch.tensor(faces_np, dtype=torch.long, device=self.device)

        # ── Checkpoint loading ────────────────────────────────────────────
        self._has_pretrained_edd = False
        if opts.ckpt and opts.continue_ckpt:
            self._load_weight(self.model_hlbs, "hlbs", opts.ckpt, opts.start_epoch)
            disp_path = sorted(glob.glob(
                os.path.join(opts.ckpt, f"model_edd_{opts.start_epoch:03d}.pth")))
            if disp_path:
                self.model_edd.load_state_dict(
                    torch.load(disp_path[0], map_location=self.device))
                self._has_pretrained_edd = True
                print(f"Loaded EDD from ckpt: {disp_path[0]}")

        if opts.edd_ckpt and os.path.exists(opts.edd_ckpt):
            self.model_edd.load_state_dict(
                torch.load(opts.edd_ckpt, map_location=self.device))
            self._has_pretrained_edd = True
            print(f"Loaded pre-trained EDD: {opts.edd_ckpt}")

        self.vis_loader = CheckpointVisLoader(opts, device=self.device)

    def _load_weight(self, model, name, ckpt_dir, epoch):
        if epoch > 0:
            paths = sorted(glob.glob(os.path.join(ckpt_dir, f"model_{name}_{epoch:03d}.pth")))
        else:
            paths = sorted(glob.glob(os.path.join(ckpt_dir, f"model_{name}_best.pth")))
        if paths:
            model.load_state_dict(torch.load(paths[0], map_location=self.device))
            print(f"Loaded {name}: {paths[0]}")
        else:
            print(f"Warning: no {name} checkpoint found in {ckpt_dir}")

    def _stage_banner(self, stage, logger=None):
        if stage == 1:
            lines = [
                "", "=" * 60,
                "  STAGE 1: HLBS-only training",
                f"    HLBS target: {'smooth_GT (n_iter=' + str(self.opts.smooth_n_iter) + ')' if self.opts.lbs_target == 'smooth_gt' else 'GT'}",
                "    EDD: FROZEN (not updated)",
                "    Trainable: HLBS",
                "=" * 60, "",
            ]
        else:
            edd_info = "w/ pretrained EDD" if self._has_pretrained_edd else "w/o pretrained EDD (random init)"
            lines = [
                "", "=" * 60,
                "  STAGE 2: Joint training (HLBS + EDD)",
                f"    HLBS target: smooth_GT  |  EDD target: wrinkle (GT - smooth_GT)",
                f"    EDD: {edd_info}",
                "    Trainable: HLBS + EDD",
                "=" * 60, "",
            ]
        text = "\n".join(lines)
        print(text)
        if logger:
            logger.write(text + "\n")

    def _set_stage(self, stage):
        if stage == 1:
            for p in self.model_hlbs.parameters():
                p.requires_grad_(True)
            for p in self.model_edd.parameters():
                p.requires_grad_(False)
        else:
            for p in self.model_hlbs.parameters():
                p.requires_grad_(True)
            for p in self.model_edd.parameters():
                p.requires_grad_(True)

    def _build_optimizer(self):
        params = ([p for p in self.model_hlbs.parameters() if p.requires_grad]
                  + [p for p in self.model_edd.parameters() if p.requires_grad])
        return torch.optim.AdamW(params, lr=self.opts.lr, betas=(0.9, 0.999))

    def train(self, epochs):
        opts = self.opts
        BS = opts.batch_size
        hlbs_pretrained_epochs = opts.hlbs_pretrained_epochs

        stage = 2 if opts.start_epoch >= hlbs_pretrained_epochs else 1
        self._set_stage(stage)
        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opts.sc_step, gamma=opts.sc_gamma)

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
            tgt_tag = (f"-s{opts.smooth_n_iter}"
                   if (opts.lbs_target == 'smooth_gt' or opts.edd_target == 'smooth_gt')
                   else "-lbsres")
        tag = f"-FacialAnim-{opts.topo_key}{tgt_tag}"
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
            f"=== FacialAnim Stage Training ===\n"
            f"  topo_key              : {opts.topo_key}\n"
            f"  hlbs_pretrained_epochs: {hlbs_pretrained_epochs}\n"
            f"  edd_ckpt              : {opts.edd_ckpt}\n"
            f"  lbs_target            : {opts.lbs_target}\n"
            f"  edd_target            : {opts.edd_target}\n"
            f"  smooth_n_iter         : {opts.smooth_n_iter}\n"
            f"  lambda_W_reg          : {opts.lambda_W_reg}\n"
            f"  lambda_t_reg          : {opts.lambda_t_reg}\n"
            f"=================================\n"
        )
        print(config_text)
        logger.write(config_text)
        logger.write(train_ds.get_data_config())
        self._stage_banner(stage, logger)

        loss_lambda = {
            "recon-lbs":    opts.lambda_vert,
            "recon-wrinkle": opts.lambda_vert,
            "recon-def":    opts.lambda_vert,
            "lbs-W-reg":    opts.lambda_W_reg,
            "lbs-t-reg":    opts.lambda_t_reg,
        }

        BEST_LOSS  = 1e8
        BEST_EPOCH = 0
        len_train  = len(train_loader)
        len_valid  = len(valid_loader)
        interv     = max(1, round(len_train / 10))
        interv_val = max(1, round(len_valid / 4))

        for epoch in range(opts.start_epoch, epochs + 1):
            # Stage transition
            if stage == 1 and epoch >= hlbs_pretrained_epochs:
                stage = 2
                self._set_stage(2)
                if opts.edd_ckpt and os.path.exists(opts.edd_ckpt):
                    self.model_edd.load_state_dict(
                        torch.load(opts.edd_ckpt, map_location=self.device))
                    self._has_pretrained_edd = True
                self.optimizer = self._build_optimizer()
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=opts.sc_step, gamma=opts.sc_gamma)
                self._stage_banner(stage, logger)

            stag = f"Stage{stage}"
            if stage == 1:
                pbar_prefix = f"[{epoch:03d}/{epochs:03d}][{stag}] HLBS-only"
            else:
                edd_tag = "pretrained" if self._has_pretrained_edd else "scratch"
                pbar_prefix = f"[{epoch:03d}/{epochs:03d}][{stag}] Joint(HLBS+EDD:{edd_tag})"

            # ── Train ────────────────────────────────────────────────────────
            self.model_hlbs.train()
            self.model_edd.train() if stage == 2 else self.model_edd.eval()

            running = {"recon-lbs": 0.0, "recon-def": 0.0,
                       "lbs-W-reg": 0.0, "lbs-t-reg": 0.0, "total": 0.0}
            if stage == 2:
                running["recon-wrinkle"] = 0.0
            cnt = 0

            pbar = tqdm(enumerate(train_loader), total=len_train, ncols=120)
            for idx, batch in pbar:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                src_v  = batch.template
                src_n  = batch.template_normal
                gt_v   = batch.vertices
                gt_n   = batch.vertices_normal
                smt_v  = batch.smooth_vertices
                delta     = gt_v - src_v
                src_in    = torch.cat([src_v, src_n], dim=-1)
                deform_in = torch.cat([delta, gt_n, src_in], dim=-1)

                pred_lbs, z_exp = self.model_hlbs(
                    src_v, deform_in, return_z_exp=True)

                if opts.no_t_mask:
                    t_mask   = 1.0
                    inv_mask = 0.0
                else:
                    from utils.exp_utils import plateau_hat_points
                    t_mask   = plateau_hat_points(src_v)
                    inv_mask = 1.0 - t_mask

                loss_dict = {}

                # HLBS loss
                lbs_tgt = smt_v if opts.lbs_target == 'smooth_gt' else gt_v
                if opts.no_t_mask:
                    loss_dict["recon-lbs"] = F.mse_loss(lbs_tgt, pred_lbs)
                else:
                    loss_dict["recon-lbs"] = (
                        F.mse_loss(lbs_tgt * t_mask,   pred_lbs * t_mask)
                        + F.mse_loss(src_v * inv_mask, pred_lbs * inv_mask)
                    )

                # HLBS regularization
                regs = self.model_hlbs.reg_loss()
                loss_dict["lbs-W-reg"] = regs["L_W_reg"]
                loss_dict["lbs-t-reg"] = regs["L_t_reg"]

                if stage == 2:
                    jac_feat = compute_jacobian_features(pred_lbs, src_v, self.faces_t)
                    disp     = self.model_edd(jac_feat, z_exp)

                    if not opts.no_t_mask:
                        disp = disp * t_mask

                    pred_vertices  = pred_lbs + disp
                    wrinkle_target = (gt_v - pred_lbs.detach()
                                      if opts.edd_target == 'lbs_residual'
                                      else gt_v - smt_v)

                    if opts.no_t_mask:
                        loss_dict["recon-wrinkle"] = F.mse_loss(wrinkle_target, disp)
                        loss_dict["recon-def"]     = F.mse_loss(gt_v, pred_vertices)
                    else:
                        loss_dict["recon-wrinkle"] = F.mse_loss(
                            wrinkle_target * t_mask, disp * t_mask)
                        loss_dict["recon-def"] = (
                            F.mse_loss(gt_v * t_mask,    pred_vertices * t_mask)
                            + F.mse_loss(src_v * inv_mask, pred_vertices * inv_mask)
                        )
                else:
                    loss_dict["recon-def"] = loss_dict["recon-lbs"].clone()

                loss = 0
                for k, v in loss_dict.items():
                    if k in loss_lambda:
                        tmp = v * loss_lambda[k]
                        loss = loss + tmp
                        if k in running:
                            running[k] += tmp.item() if hasattr(tmp, 'item') else float(tmp)

                loss.backward()
                self.optimizer.step()

                running["total"] += loss.item()
                cnt += 1
                pbar.set_description(f"{pbar_prefix} | loss: {loss:.5e}")

                if idx % interv == 1:
                    inv = 1.0 / cnt
                    log_text = f"[{epoch:03d}][{idx:04d}][{stag}] "
                    log_text += " ".join(f"{k}: {v*inv:.6e}" for k, v in running.items())
                    logger.write(log_text + "\n")

                    HB = BS // 2
                    faces_cpu = batch.faces.cpu()
                    v_list = [
                        gt_v[0].cpu().detach(),            gt_v[min(1,BS-1)].cpu().detach(),
                        gt_v[min(HB,BS-1)].cpu().detach(), gt_v[BS-1].cpu().detach(),
                        smt_v[0].cpu().detach(),            smt_v[min(1,BS-1)].cpu().detach(),
                        smt_v[min(HB,BS-1)].cpu().detach(), smt_v[BS-1].cpu().detach(),
                        pred_lbs[0].cpu().detach(),            pred_lbs[min(1,BS-1)].cpu().detach(),
                        pred_lbs[min(HB,BS-1)].cpu().detach(), pred_lbs[BS-1].cpu().detach(),
                    ]
                    if stage == 2:
                        v_list += [
                            pred_vertices[0].cpu().detach(),            pred_vertices[min(1,BS-1)].cpu().detach(),
                            pred_vertices[min(HB,BS-1)].cpu().detach(), pred_vertices[BS-1].cpu().detach(),
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

            # Save
            if epoch % opts.save_interval == 0:
                torch.save(self.model_hlbs.state_dict(),
                           f'{opts.log_dir}/model_hlbs_{epoch:03d}.pth')
                if stage == 2:
                    torch.save(self.model_edd.state_dict(),
                               f'{opts.log_dir}/model_edd_{epoch:03d}.pth')

            # Eval-iter visualization
            if epoch % opts.eval_iter == 0:
                self.vis_loader.visualize(
                    self.model_hlbs, self.model_edd if stage == 2 else None,
                    epoch, save_dir=f'{opts.log_dir}/img/eval',
                    mode='facial_anim', stage=stage,
                    topo_key=opts.topo_key,
                    smooth_n_iter=opts.smooth_n_iter,
                    no_t_mask=opts.no_t_mask,
                )

            # ── Valid ────────────────────────────────────────────────────────
            self.model_hlbs.eval()
            self.model_edd.eval()

            running_val = {"recon-lbs": 0.0, "recon-def": 0.0, "total": 0.0}
            if stage == 2:
                running_val["recon-wrinkle"] = 0.0
            vcnt = 0

            val_prefix = (f"[{epoch:03d}][{stag}] Valid HLBS-only" if stage == 1
                          else f"[{epoch:03d}][{stag}] Valid Joint")
            pbar = tqdm(enumerate(valid_loader), total=len_valid, ncols=120, desc=val_prefix)
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
                    pred_lbs, z_exp = self.model_hlbs(
                        src_v, deform_in, return_z_exp=True)

                    loss_dict = {}
                    lbs_tgt_val = smt_v if opts.lbs_target == 'smooth_gt' else gt_v
                    loss_dict["recon-lbs"] = F.mse_loss(lbs_tgt_val, pred_lbs)

                    if stage == 2:
                        jac_feat = compute_jacobian_features(pred_lbs, src_v, self.faces_t)
                        disp     = self.model_edd(jac_feat, z_exp)
                        pred_vertices  = pred_lbs + disp
                        wrinkle_target = (gt_v - pred_lbs
                                          if opts.edd_target == 'lbs_residual'
                                          else gt_v - smt_v)

                        if not opts.no_t_mask:
                            from utils.exp_utils import plateau_hat_points
                            t_mask_v = plateau_hat_points(src_v)
                            loss_dict["recon-wrinkle"] = F.mse_loss(
                                wrinkle_target * t_mask_v, disp * t_mask_v)
                            loss_dict["recon-def"] = F.mse_loss(
                                gt_v * t_mask_v, pred_vertices * t_mask_v)
                        else:
                            loss_dict["recon-wrinkle"] = F.mse_loss(wrinkle_target, disp)
                            loss_dict["recon-def"]     = F.mse_loss(gt_v, pred_vertices)
                    else:
                        loss_dict["recon-def"] = loss_dict["recon-lbs"].clone()

                    loss = 0
                    for k, v in loss_dict.items():
                        if k in loss_lambda:
                            val = v.item() if hasattr(v, 'item') else float(v)
                            tmp = val * loss_lambda[k]
                            loss = loss + tmp
                            if k in running_val:
                                running_val[k] += tmp

                running_val["total"] += loss
                pbar.set_description(f"{val_prefix} | loss: {loss:.5e}")

                if idx % interv_val == 0:
                    BS_v = batch.vertices.shape[0]
                    HB_v = BS_v // 2
                    faces_cpu = batch.faces.cpu()
                    v_list = [
                        gt_v[0].cpu(),              gt_v[min(1,BS_v-1)].cpu(),
                        gt_v[min(HB_v,BS_v-1)].cpu(), gt_v[BS_v-1].cpu(),
                        smt_v[0].cpu(),              smt_v[min(1,BS_v-1)].cpu(),
                        smt_v[min(HB_v,BS_v-1)].cpu(), smt_v[BS_v-1].cpu(),
                        pred_lbs[0].cpu(),           pred_lbs[min(1,BS_v-1)].cpu(),
                        pred_lbs[min(HB_v,BS_v-1)].cpu(), pred_lbs[BS_v-1].cpu(),
                    ]
                    if stage == 2:
                        v_list += [
                            pred_vertices[0].cpu(),           pred_vertices[min(1,BS_v-1)].cpu(),
                            pred_vertices[min(HB_v,BS_v-1)].cpu(), pred_vertices[BS_v-1].cpu(),
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
                torch.save(self.model_hlbs.state_dict(),
                           f'{opts.log_dir}/model_hlbs_best.pth')
                if stage == 2:
                    torch.save(self.model_edd.state_dict(),
                               f'{opts.log_dir}/model_edd_best.pth')
                print(f"[{epoch:03d}][{stag}] Best: {BEST_LOSS:.6e} (epoch {BEST_EPOCH})")
                logger.write(f"[{epoch:03d}] Best Loss: {BEST_LOSS:.6e}\n")
            else:
                print(f"[{epoch:03d}][{stag}] Val: {total_val:.6e} (Best: {BEST_LOSS:.6e} [{BEST_EPOCH}])")
                logger.write(f"[{epoch:03d}] Val: {total_val:.6e} (Best: {BEST_LOSS:.6e} [{BEST_EPOCH}])\n")


if __name__ == "__main__":
    opts = Options()

    if os.path.exists(opts.config):
        opts_yaml = yaml.load(open(opts.config), Loader=yaml.FullLoader)
        opts_dict = vars(opts)
        opts_yaml.update(opts_dict)
        opts = argparse.Namespace(**opts_yaml)

    if opts.lbs_target == 'smooth_gt' or opts.edd_target == 'smooth_gt':
        assert opts.smooth_n_iter > 0, "lbs_target/edd_target=smooth_gt requires --smooth_n_iter > 0"

    trainer = FacialAnimTrainer(opts)
    trainer.train(epochs=opts.max_epoch)
