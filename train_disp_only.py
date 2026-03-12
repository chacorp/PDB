"""
train_disp_only.py — DispNet-only pre-training with GT oracle signals (v10).

Usage:
    python train_disp_only.py --config configs/train.yml \
        --version 9 --use_strain --strain_mode norm \
        --smooth_n_iter 16 --use_t_mask --use_data1 --data_toggle \
        --batch_size 16 --max_epoch 300 --tb

Input:  GT strain = strain(GT_deformed, template) — perfect signal
Target: GT wrinkle = GT - smooth_GT
No LBS training. DispNet learns strain→wrinkle mapping independently.
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
from utils.mesh_utils import calc_norm_torch, compute_strain_signal, STRAIN_MODE_DIM
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

    # LBS (needed for model init + t_mask, but weights frozen)
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
    parser.add_argument("--smooth_n_iter", type=int, required=True,
                        help='Taubin smoothing iters (must be >0, e.g. 8/16/32)')

    # training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epoch", type=int, default=300)
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
                        help='LBS checkpoint dir (for t_mask). Frozen, not trained.')
    parser.add_argument("--disp_ckpt", type=str, default=None,
                        help='Pre-trained DispNet to resume from')
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


class DispOnlyTrainer:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Auto-set strain_dim from strain_mode
        if opts.use_strain:
            opts.strain_dim = STRAIN_MODE_DIM[opts.strain_mode]

        # seed
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)

        # LBS model (frozen, used only for t_mask)
        last_act_list = ["relu", "elu", "softmax", "softplus", "none"]
        last_act_list = [opts.last_activation == l for l in last_act_list]
        self.model_lbs = NeuralGeneralizedBarycentricCoordinateLBS(
            opts, num_layers=4,
            num_cage_vertices=opts.num_cage_v,
            use_exp_recon=False, use_shp_recon=False, use_shp=False,
            use_relu=last_act_list[0], use_elu=last_act_list[1],
            use_softmax=last_act_list[2], use_softplus=last_act_list[3],
            no_activation=last_act_list[4],
            is_train=False, use_pou=~opts.no_pou, device=self.device,
            hid_dim=128 if opts.align_latent else 256,
        )
        # Load LBS weights if provided
        if opts.ckpt:
            lbs_path = sorted(glob.glob(os.path.join(opts.ckpt, "model_lbs_best.pth")))
            if not lbs_path:
                lbs_path = sorted(glob.glob(os.path.join(opts.ckpt, "model_lbs_*.pth")))
            if lbs_path:
                self.model_lbs.load_state_dict(torch.load(lbs_path[-1]))
                print(f"Loaded LBS (frozen): {lbs_path[-1]}")
        for p in self.model_lbs.parameters():
            p.requires_grad_(False)
        self.model_lbs.eval()

        # DispNet (trainable)
        strain_dim = opts.strain_dim if opts.use_strain else 0
        self.model_disp = NeuralStrainDisplacement(
            opts, hid_dim=256, num_layers=4,
            strain_dim=strain_dim, device=self.device,
            use_source_template=opts.use_source_template,
        )
        if opts.disp_ckpt and os.path.exists(opts.disp_ckpt):
            self.model_disp.load_state_dict(torch.load(opts.disp_ckpt))
            print(f"Loaded DispNet: {opts.disp_ckpt}")

        self.vis_loader = CheckpointVisLoader(opts, device=self.device)

    def train(self, epochs):
        opts = self.opts
        BS = opts.batch_size

        # optimizer (DispNet only)
        self.optimizer = torch.optim.AdamW(
            self.model_disp.parameters(), lr=opts.lr, betas=(0.9, 0.999))
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

        resume_mode = opts.disp_ckpt and opts.continue_ckpt
        if resume_mode and opts.log_dir and os.path.isdir(opts.log_dir):
            pass  # keep existing log_dir
        else:
            tag = f"-DispOnly-{opts.strain_mode}-s{opts.smooth_n_iter}"
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
            f"=== DispOnly Training ===\n"
            f"  strain_mode : {opts.strain_mode} (dim={opts.strain_dim})\n"
            f"  smooth_n_iter: {opts.smooth_n_iter}\n"
            f"  use_source_template: {opts.use_source_template}\n"
            f"  LBS ckpt (frozen): {opts.ckpt}\n"
            f"  Input: GT strain({opts.strain_mode}) from GT deformed vs template\n"
            f"  Target: wrinkle = GT - smooth_GT\n"
            f"=========================\n"
        )
        print(config_text)
        logger.write(config_text)
        logger.write(train_ds.get_data_config())
        logger.write(str(self.model_disp) + "\n")

        # training loop
        BEST_LOSS = 1e8
        BEST_EPOCH = 0
        loss_lambda = {"recon-wrinkle": opts.lambda_vert}

        len_train = len(train_loader)
        len_valid = len(valid_loader)
        interv = max(1, round(len_train / 10))

        for epoch in range(opts.start_epoch, epochs + 1):
            # --- Train ---
            self.model_lbs.eval()
            self.model_disp.train()
            running = {"recon-wrinkle": 0.0, "total": 0.0}
            cnt = 0

            pbar = tqdm(enumerate(train_loader), total=len_train, ncols=100,
                        desc=f"[{epoch:03d}] Train DispOnly")
            for idx, batch in pbar:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                tmpl_v = batch.template
                tmpl_n = batch.template_normal
                gt_v = batch.vertices
                smooth_v = batch.smooth_vertices

                wrinkle_target = gt_v - smooth_v

                # GT strain: from GT deformed vs template
                _nsi = getattr(batch, 'neutral_span_inv', None)
                strain = compute_strain_signal(
                    gt_v, tmpl_v, batch.faces,
                    mode=opts.strain_mode, neutral_span_inv=_nsi)

                # DispNet input: use smooth_GT as proxy for "LBS output"
                smooth_norm = calc_norm_torch(smooth_v, batch.faces, at='verts')
                displacement, _ = self.model_disp(
                    smooth_v, smooth_norm,
                    source_vert=tmpl_v if opts.use_source_template else None,
                    source_norm=tmpl_n if opts.use_source_template else None,
                    strain=strain)

                # t_mask from LBS (no grad)
                with torch.no_grad():
                    _, _, _, _, _, t_mask, _, _, _, _ = self.model_lbs(
                        tmpl_v, gt_v, tmpl_n, batch.vertices_normal,
                        batch.mesh_data, epoch=epoch)

                if not opts.no_t_mask:
                    displacement = displacement * t_mask

                # loss
                if opts.no_t_mask:
                    loss = F.mse_loss(wrinkle_target, displacement)
                else:
                    loss = F.mse_loss(wrinkle_target * t_mask, displacement * t_mask)

                weighted = loss * loss_lambda["recon-wrinkle"]
                weighted.backward()
                self.optimizer.step()

                running["recon-wrinkle"] += weighted.item()
                running["total"] += weighted.item()
                cnt += 1
                pbar.set_description(f"[{epoch:03d}] wrinkle: {loss:.5e}")

                # vis
                if idx % interv == 1:
                    log_text = f"[{epoch:03d}/{epochs:03d}][{idx:04d}][Train] wrinkle: {running['recon-wrinkle']/cnt:.6e}"
                    logger.write(log_text + "\n")

                    HB = BS // 2
                    pred_full = smooth_v + displacement
                    v_list = [
                        gt_v[0].cpu().detach(), gt_v[min(1,BS-1)].cpu().detach(),
                        gt_v[min(HB,BS-1)].cpu().detach(), gt_v[BS-1].cpu().detach(),
                        smooth_v[0].cpu().detach(), smooth_v[min(1,BS-1)].cpu().detach(),
                        smooth_v[min(HB,BS-1)].cpu().detach(), smooth_v[BS-1].cpu().detach(),
                        pred_full[0].cpu().detach(), pred_full[min(1,BS-1)].cpu().detach(),
                        pred_full[min(HB,BS-1)].cpu().detach(), pred_full[BS-1].cpu().detach(),
                    ]
                    f_list = [batch.faces.cpu()] * len(v_list)
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

            if epoch % opts.save_interval == 0:
                torch.save(self.model_disp.state_dict(),
                           f'{opts.log_dir}/model_disp_{epoch:03d}.pth')

            # Eval-iter visualization (real frames via vis_loader)
            if epoch % opts.eval_iter == 0 and opts.use_strain:
                eval_vis_dir = f"{opts.log_dir}/img/eval"
                self.vis_loader.visualize(
                    self.model_lbs, self.model_disp, epoch, eval_vis_dir,
                    mode='disp_only',
                    strain_mode=opts.strain_mode,
                    smooth_n_iter=opts.smooth_n_iter,
                    no_t_mask=opts.no_t_mask,
                    use_source_template=opts.use_source_template,
                )

            # --- Valid ---
            self.model_disp.eval()
            running_val = {"recon-wrinkle": 0.0, "total": 0.0}
            vcnt = 0

            pbar = tqdm(enumerate(valid_loader), total=len_valid, ncols=100,
                        desc=f"[{epoch:03d}] Valid DispOnly")
            for idx, batch in pbar:
                batch = batch.to(self.device)
                vcnt += 1
                with torch.no_grad():
                    smooth_v = batch.smooth_vertices
                    wrinkle_target = batch.vertices - smooth_v
                    _nsi = getattr(batch, 'neutral_span_inv', None)
                    strain = compute_strain_signal(
                        batch.vertices, batch.template, batch.faces,
                        mode=opts.strain_mode, neutral_span_inv=_nsi)
                    smooth_norm = calc_norm_torch(smooth_v, batch.faces, at='verts')
                    displacement, _ = self.model_disp(
                        smooth_v, smooth_norm,
                        source_vert=batch.template if opts.use_source_template else None,
                        source_norm=batch.template_normal if opts.use_source_template else None,
                        strain=strain)
                    _, _, _, _, _, t_mask_v, _, _, _, _ = self.model_lbs(
                        batch.template, batch.vertices,
                        batch.template_normal, batch.vertices_normal,
                        batch.mesh_data, epoch=epoch)
                    if not opts.no_t_mask:
                        displacement = displacement * t_mask_v
                    loss = F.mse_loss(wrinkle_target * t_mask_v, displacement * t_mask_v) if not opts.no_t_mask \
                        else F.mse_loss(wrinkle_target, displacement)
                    weighted = loss.item() * loss_lambda["recon-wrinkle"]
                    running_val["recon-wrinkle"] += weighted
                    running_val["total"] += weighted
                pbar.set_description(f"[{epoch:03d}] val wrinkle: {loss:.5e}")
                if opts.debug:
                    break

            if writer_valid:
                for k, v in running_val.items():
                    writer_valid.add_scalar(k, v / vcnt, epoch)

            val_loss = running_val["total"] / vcnt
            if val_loss < BEST_LOSS:
                BEST_LOSS = val_loss
                BEST_EPOCH = epoch
                print(f"[{epoch:03d}] Best: {BEST_LOSS:.6e} (epoch {BEST_EPOCH})")
                logger.write(f"[{epoch:03d}] Best Loss: {BEST_LOSS:.6e}\n")
                torch.save(self.model_disp.state_dict(), f'{opts.log_dir}/model_disp_best.pth')
            else:
                print(f"[{epoch:03d}] Val: {val_loss:.6e} (Best: {BEST_LOSS:.6e} [{BEST_EPOCH}])")
                logger.write(f"[{epoch:03d}] Val: {val_loss:.6e} (Best: {BEST_LOSS:.6e} [{BEST_EPOCH}])\n")


if __name__ == "__main__":
    opts = Options()

    # merge yaml config
    if os.path.exists(opts.config):
        opts_yaml = yaml.load(open(opts.config), Loader=yaml.FullLoader)
        opts_dict = vars(opts)
        opts_yaml.update(opts_dict)
        opts = argparse.Namespace(**opts_yaml)

    assert opts.smooth_n_iter > 0, "train_disp_only requires --smooth_n_iter > 0"
    assert opts.use_strain, "train_disp_only requires --use_strain"

    trainer = DispOnlyTrainer(opts)
    trainer.train(epochs=opts.max_epoch)
