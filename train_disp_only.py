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
from utils.mesh_utils import calc_norm_torch, compute_strain_signal, STRAIN_MODE_DIM, taubin_smooth_np
from utils.exp_utils import plateau_hat_points
class Logger:
    def __init__(self, file_path):
        self.file_path = file_path
    def write(self, txt):
        with open(self.file_path, 'a') as f:
            f.write(txt)
from models.NGBC import NeuralStrainDisplacement
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
    parser.add_argument("--use_laplacian", dest='use_laplacian', action='store_true')
    parser.set_defaults(use_laplacian=False)

    # LBS args (kept for dataloader compatibility)
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
    parser.add_argument("--no_exp_z", dest='no_exp_z', action='store_true',
                        help='DispNet without exp_z encoder (plain pointwise MLP)')
    parser.set_defaults(no_exp_z=False)
    parser.add_argument("--smooth_n_iter", type=int, required=True,
                        help='Taubin smoothing iters (must be >0, e.g. 8/16/32)')
    parser.add_argument("--disp_loss_type", type=str, default='mse',
                        choices=['mse', 'l1', 'smooth_l1'],
                        help='Loss function for displacement/wrinkle target')
    parser.add_argument("--use_true_edd", dest='use_true_edd', action='store_true',
                        help='true EDD target: (GT-sGT)-(T-sT) instead of GT-sGT')
    parser.set_defaults(use_true_edd=False)
    parser.add_argument("--norm_stats_file", type=str, default=None,
                        help='Path to .npz with strain/disp z-score stats (from precompute_norm_stats.py)')

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
                        help='(unused, kept for compatibility)')
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

        # DispNet (trainable)
        strain_dim = opts.strain_dim if opts.use_strain else 0
        self.model_disp = NeuralStrainDisplacement(
            opts, hid_dim=256, num_layers=4,
            strain_dim=strain_dim, device=self.device,
            use_source_template=opts.use_source_template,
            no_exp_z=opts.no_exp_z,
        )
        if opts.disp_ckpt and os.path.exists(opts.disp_ckpt):
            self.model_disp.load_state_dict(torch.load(opts.disp_ckpt))
            print(f"Loaded DispNet: {opts.disp_ckpt}")

        self.vis_loader = CheckpointVisLoader(opts, device=self.device)

        # Z-score normalization stats
        self.norm_stats = None
        if opts.norm_stats_file and os.path.exists(opts.norm_stats_file):
            data = np.load(opts.norm_stats_file)
            self.norm_stats = {
                'strain_mean': torch.tensor(data['strain_mean']).float().to(self.device),
                'strain_std':  torch.tensor(data['strain_std']).float().to(self.device),
                'disp_mean':   torch.tensor(data['disp_mean']).float().to(self.device),
                'disp_std':    torch.tensor(data['disp_std']).float().to(self.device),
            }
            print(f"[NormStats] Loaded: {opts.norm_stats_file}")
            print(f"  strain μ={self.norm_stats['strain_mean'].cpu().numpy()}, σ={self.norm_stats['strain_std'].cpu().numpy()}")
            print(f"  disp   μ={self.norm_stats['disp_mean'].cpu().numpy()}, σ={self.norm_stats['disp_std'].cpu().numpy()}")

    def _normalize_strain(self, strain):
        """Z-score normalize strain: (strain - μ) / σ"""
        if self.norm_stats is None:
            return strain
        return (strain - self.norm_stats['strain_mean']) / self.norm_stats['strain_std']

    def _normalize_disp_target(self, disp):
        """Z-score normalize displacement target: (disp - μ) / σ"""
        if self.norm_stats is None:
            return disp
        return (disp - self.norm_stats['disp_mean']) / self.norm_stats['disp_std']

    def _denormalize_disp(self, disp_pred):
        """Reverse z-score on DispNet output: pred * σ + μ"""
        if self.norm_stats is None:
            return disp_pred
        return disp_pred * self.norm_stats['disp_std'] + self.norm_stats['disp_mean']

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
            _loss_tag = f"-{opts.disp_loss_type}" if opts.disp_loss_type != 'mse' else ""
            tag = f"-DispOnly-{opts.strain_mode}-s{opts.smooth_n_iter}{_loss_tag}"
            opts.log_dir = os.path.join(opts.log_dir, now + tag)

        os.makedirs(opts.log_dir, exist_ok=True)
        os.makedirs(f"{opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{opts.log_dir}/img/valid/mesh", exist_ok=True)

        with open(os.path.join(opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(opts), f, indent=4)
        with open(os.path.join(opts.log_dir, "train_opts.yml"), 'w') as f:
            yaml.dump(vars(opts), f, sort_keys=False)

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
            f"  no_exp_z: {opts.no_exp_z}\n"
            f"  disp_loss_type: {opts.disp_loss_type}\n"
            f"  use_true_edd: {opts.use_true_edd}\n"
            f"  norm_stats_file: {opts.norm_stats_file}\n"
            f"  Input: GT strain({opts.strain_mode}) from GT deformed vs template\n"
            f"  Target: {'true EDD = (GT-sGT)-(T-sT)' if opts.use_true_edd else 'wrinkle = GT-sGT'}\n"
            f"=========================\n"
        )
        print(config_text)
        logger.write(config_text)
        logger.write(train_ds.get_data_config())
        logger.write(str(self.model_disp) + "\n")

        # neutral_detail cache for true EDD: {id_name: tensor [V, 3]}
        self._neutral_detail_cache = {}

        # training loop
        BEST_LOSS = 1e8
        BEST_EPOCH = 0
        loss_lambda = {"recon-wrinkle": opts.lambda_vert}
        _disp_loss_fn = {'mse': F.mse_loss, 'l1': F.l1_loss, 'smooth_l1': F.smooth_l1_loss}[opts.disp_loss_type]

        len_train = len(train_loader)
        len_valid = len(valid_loader)
        interv = max(1, round(len_train / 10))
        interv_val = max(1, round(len_valid / 3))

        for epoch in range(opts.start_epoch, epochs + 1):
            # --- Train ---
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
                if opts.use_true_edd:
                    neutral_detail = self._get_neutral_detail(batch)  # [V, 3]
                    wrinkle_target = wrinkle_target - neutral_detail.unsqueeze(0)

                # GT strain: from GT deformed vs template
                _nsi = getattr(batch, 'neutral_span_inv', None)
                strain = compute_strain_signal(
                    gt_v, tmpl_v, batch.faces,
                    mode=opts.strain_mode, neutral_span_inv=_nsi)
                strain = self._normalize_strain(strain)

                # DispNet input: use smooth_GT as proxy for "LBS output"
                smooth_norm = calc_norm_torch(smooth_v, batch.faces, at='verts')
                displacement, _ = self.model_disp(
                    smooth_v, smooth_norm,
                    source_vert=tmpl_v if opts.use_source_template else None,
                    source_norm=tmpl_n if opts.use_source_template else None,
                    strain=strain)

                # t_mask from geometry (plateau hat on template)
                t_mask = plateau_hat_points(tmpl_v)

                if not opts.no_t_mask:
                    displacement = displacement * t_mask

                # Normalize wrinkle target for loss (z-score)
                wrinkle_target_norm = self._normalize_disp_target(wrinkle_target)

                # loss (in normalized space)
                if opts.no_t_mask:
                    loss = _disp_loss_fn(wrinkle_target_norm, displacement)
                else:
                    loss = _disp_loss_fn(wrinkle_target_norm * t_mask, displacement * t_mask)

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
                    pred_full = smooth_v + self._denormalize_disp(displacement)
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
                    None, self.model_disp, epoch, eval_vis_dir,
                    mode='disp_only',
                    strain_mode=opts.strain_mode,
                    smooth_n_iter=opts.smooth_n_iter,
                    no_t_mask=opts.no_t_mask,
                    use_source_template=opts.use_source_template,
                    norm_stats=self.norm_stats,
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
                    if opts.use_true_edd:
                        neutral_detail = self._get_neutral_detail(batch)
                        wrinkle_target = wrinkle_target - neutral_detail.unsqueeze(0)
                    _nsi = getattr(batch, 'neutral_span_inv', None)
                    strain = compute_strain_signal(
                        batch.vertices, batch.template, batch.faces,
                        mode=opts.strain_mode, neutral_span_inv=_nsi)
                    strain = self._normalize_strain(strain)
                    smooth_norm = calc_norm_torch(smooth_v, batch.faces, at='verts')
                    displacement, _ = self.model_disp(
                        smooth_v, smooth_norm,
                        source_vert=batch.template if opts.use_source_template else None,
                        source_norm=batch.template_normal if opts.use_source_template else None,
                        strain=strain)
                    t_mask_v = plateau_hat_points(batch.template)
                    if not opts.no_t_mask:
                        displacement = displacement * t_mask_v
                    wrinkle_target_norm = self._normalize_disp_target(wrinkle_target)
                    loss = _disp_loss_fn(wrinkle_target_norm * t_mask_v, displacement * t_mask_v) if not opts.no_t_mask \
                        else _disp_loss_fn(wrinkle_target_norm, displacement)
                    weighted = loss.item() * loss_lambda["recon-wrinkle"]
                    running_val["recon-wrinkle"] += weighted
                    running_val["total"] += weighted
                pbar.set_description(f"[{epoch:03d}] val wrinkle: {loss:.5e}")

                # Per-iter val mesh visualization
                if idx % interv_val == 0:
                    with torch.no_grad():
                        BS_v = batch.vertices.shape[0]
                        HB_v = BS_v // 2
                        pred_full = smooth_v + self._denormalize_disp(displacement)
                        v_list = [
                            batch.vertices[0].cpu(), batch.vertices[min(1,BS_v-1)].cpu(),
                            batch.vertices[min(HB_v,BS_v-1)].cpu(), batch.vertices[BS_v-1].cpu(),
                            smooth_v[0].cpu(), smooth_v[min(1,BS_v-1)].cpu(),
                            smooth_v[min(HB_v,BS_v-1)].cpu(), smooth_v[BS_v-1].cpu(),
                            pred_full[0].cpu(), pred_full[min(1,BS_v-1)].cpu(),
                            pred_full[min(HB_v,BS_v-1)].cpu(), pred_full[BS_v-1].cpu(),
                        ]
                        f_list = [batch.faces.cpu()] * len(v_list)
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
