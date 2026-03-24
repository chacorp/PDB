"""
train_edd.py — DiffusionNetEDD standalone training with frozen HLBS.

HLBS is loaded from a pre-trained checkpoint and kept frozen.
EDD learns to predict per-vertex displacement from Jacobian features
and the HLBS expression latent z_exp.

Target: wrinkle = GT - smooth_GT   (high-frequency residual)

Usage:
    python train_edd.py --config configs/train.yml \
        --rig_path utils/mf/rig_info.json \
        --topo_key mf \
        --neutral_obj utils/mf/mf_aligned_mean.obj \
        --hlbs_ckpt ./ckpts_hlbs/.../model_hlbs_best.pth \
        --num_identities 13 \
        --smooth_n_iter 16 \
        --use_data1 --data_toggle --batch_size 8 --max_epoch 300 --tb
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
    parser.add_argument("--rig_path", type=str, default=None,
                        help='Path to rig dir (required when --hlbs_ckpt is set)')
    parser.add_argument("--topo_key", type=str, default='mf',
                        choices=['mf', 'biwi', 'voca'])
    parser.add_argument("--neutral_obj", type=str, required=True,
                        help='Neutral mesh .obj for DiffusionNet operator precomputation')
    parser.add_argument("--op_cache_dir", type=str, default='utils/diffusion_ops',
                        help='DiffusionNet operator cache directory')
    parser.add_argument("--hid_dim", type=int, default=256,
                        help='HLBS/z_exp dim (must match hlbs_ckpt when provided)')
    parser.add_argument("--num_layers", type=int, default=4)

    # EDD architecture
    parser.add_argument("--edd_hid_channels", type=int, default=128)
    parser.add_argument("--edd_n_blocks", type=int, default=4)
    parser.add_argument("--edd_k_eig", type=int, default=128)

    # target decomposition
    parser.add_argument("--edd_target", type=str, default='lbs_residual',
                        choices=['lbs_residual', 'smooth_gt'],
                        help='"lbs_residual": EDD target = GT - pred_LBS (true error-driven, default). '
                             '"smooth_gt": EDD target = GT - smooth_GT (wrinkle only).')
    parser.add_argument("--smooth_n_iter", type=int, default=0,
                        help='Taubin smoothing iters (only needed when --edd_target smooth_gt)')

    # training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epoch", type=int, default=300)
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
    parser.add_argument("--hlbs_ckpt", type=str, default=None,
                        help='Pre-trained HLBS checkpoint .pth. '
                             'If omitted, runs in standalone mode (smooth_GT proxy, z_exp=zeros).')
    parser.add_argument("--edd_ckpt", type=str, default=None,
                        help='EDD checkpoint to resume from')
    parser.add_argument("--ckpt", type=str, default=None,
                        help='Log dir to resume from (sets log_dir)')
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--continue_ckpt", dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)

    # logging
    parser.add_argument("--log_dir", type=str, default="./ckpts_edd")
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


class EDDTrainer:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)

        from models.diffusion_edd import DiffusionNetEDD

        # ── HLBS (frozen, optional) ────────────────────────────────────────
        self.standalone = opts.hlbs_ckpt is None
        self.model_hlbs = None
        if not self.standalone:
            from utils.rig_loader import load_rig
            from models.hierarchical_lbs import HierarchicalLBS
            assert opts.rig_path is not None, "--rig_path required when --hlbs_ckpt is set"
            rig = load_rig(opts.rig_path)
            self.model_hlbs = HierarchicalLBS(
                rig=rig,
                topology=opts.topo_key,
                in_dim_exp=12,
                hid_dim=opts.hid_dim,
                num_layers=opts.num_layers,
                device=str(self.device),
            ).to(self.device)
            self.model_hlbs.load_state_dict(
                torch.load(opts.hlbs_ckpt, map_location=self.device))
            print(f"Loaded HLBS: {opts.hlbs_ckpt}")
            for p in self.model_hlbs.parameters():
                p.requires_grad_(False)
        else:
            print("[EDDTrainer] Standalone mode: smooth_GT as pred_lbs proxy, z_exp=zeros")

        # ── EDD ───────────────────────────────────────────────────────────
        z_dim = opts.hid_dim   # HLBS expression latent dim
        self.model_edd = DiffusionNetEDD(
            z_dim=z_dim,
            hid_channels=opts.edd_hid_channels,
            n_blocks=opts.edd_n_blocks,
            k_eig=opts.edd_k_eig,
        ).to(self.device)

        # Register topology (precomputes DiffusionNet operators + Poisson solve)
        verts_np, faces_np = _load_obj_verts_faces(opts.neutral_obj)
        dev_str = str(self.device)
        if dev_str == 'cuda':
            dev_str = 'cuda:0'
        self.model_edd.register_topology(
            opts.topo_key, verts_np, faces_np,
            op_cache_dir=opts.op_cache_dir,
            cuda_device=dev_str,
        )
        self.model_edd.set_topology(opts.topo_key, self.device)
        self.faces_np = faces_np  # keep for jacobian feature computation

        # faces tensor for compute_jacobian_features
        self.faces_t = torch.tensor(faces_np, dtype=torch.long, device=self.device)

        if opts.edd_ckpt and os.path.exists(opts.edd_ckpt):
            self.model_edd.load_state_dict(
                torch.load(opts.edd_ckpt, map_location=self.device))
            print(f"Loaded EDD: {opts.edd_ckpt}")

        self.vis_loader = CheckpointVisLoader(opts, device=self.device)

    def train(self, epochs):
        opts = self.opts
        BS = opts.batch_size

        self.optimizer = torch.optim.AdamW(
            self.model_edd.parameters(), lr=opts.lr, betas=(0.9, 0.999))
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
        if resume_mode and opts.ckpt and os.path.isdir(opts.ckpt):
            opts.log_dir = opts.ckpt
        else:
            tgt_tag = f"-s{opts.smooth_n_iter}" if opts.edd_target == 'smooth_gt' else "-lbsres"
            tag = f"-EDD-{opts.topo_key}{tgt_tag}"
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
            f"=== EDD Training (HLBS frozen) ===\n"
            f"  topo_key      : {opts.topo_key}\n"
            f"  hlbs_ckpt     : {opts.hlbs_ckpt}\n"
            f"  edd_target    : {opts.edd_target}\n"
            f"  smooth_n_iter : {opts.smooth_n_iter}\n"
            f"  edd_hid_channels: {opts.edd_hid_channels}\n"
            f"  edd_n_blocks  : {opts.edd_n_blocks}\n"
            f"  Target: {'GT - pred_LBS (LBS residual)' if opts.edd_target == 'lbs_residual' else 'GT - smooth_GT (wrinkle)'}\n"
            f"==================================\n"
        )
        print(config_text)
        logger.write(config_text)
        logger.write(train_ds.get_data_config())

        loss_lambda = {"recon-wrinkle": opts.lambda_vert}

        BEST_LOSS  = 1e8
        BEST_EPOCH = 0
        len_train  = len(train_loader)
        len_valid  = len(valid_loader)
        interv     = max(1, round(len_train / 10))
        interv_val = max(1, round(len_valid / 3))

        for epoch in range(opts.start_epoch, epochs + 1):
            # ── Train ────────────────────────────────────────────────────────
            if self.model_hlbs is not None:
                self.model_hlbs.eval()
            self.model_edd.train()
            running = {"recon-wrinkle": 0.0, "total": 0.0}
            cnt = 0

            pbar = tqdm(enumerate(train_loader), total=len_train, ncols=120,
                        desc=f"[{epoch:03d}] Train EDD")
            for idx, batch in pbar:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                src_v  = batch.template
                src_n  = batch.template_normal
                gt_v   = batch.vertices
                gt_n   = batch.vertices_normal
                smt_v  = batch.smooth_vertices

                # HLBS forward or standalone proxy
                if self.standalone:
                    pred_lbs = smt_v
                    z_exp    = torch.zeros(
                        gt_v.shape[0], opts.hid_dim, device=self.device)
                else:
                    with torch.no_grad():
                        delta     = gt_v - src_v
                        src_in    = torch.cat([src_v, src_n], dim=-1)
                        deform_in = torch.cat([delta, gt_n, src_in], dim=-1)
                        pred_lbs, z_exp = self.model_hlbs(
                            src_v, deform_in, return_z_exp=True)

                # Jacobian features of LBS output relative to neutral
                jac_feat = compute_jacobian_features(pred_lbs, src_v, self.faces_t)

                # EDD forward
                disp = self.model_edd(jac_feat, z_exp)   # [B, N, 3]

                if opts.edd_target == 'lbs_residual':
                    wrinkle_target = gt_v - pred_lbs
                else:
                    wrinkle_target = gt_v - smt_v

                if opts.no_t_mask:
                    loss = F.mse_loss(wrinkle_target, disp)
                else:
                    from utils.exp_utils import plateau_hat_points
                    t_mask = plateau_hat_points(src_v)
                    loss   = F.mse_loss(wrinkle_target * t_mask, disp * t_mask)

                weighted = loss * loss_lambda["recon-wrinkle"]
                weighted.backward()
                self.optimizer.step()

                running["recon-wrinkle"] += weighted.item()
                running["total"]         += weighted.item()
                cnt += 1
                pbar.set_description(f"[{epoch:03d}] wrinkle: {loss:.5e}")

                if idx % interv == 1:
                    inv = 1.0 / cnt
                    log_text = f"[{epoch:03d}/{epochs:03d}][{idx:04d}][Train] "
                    log_text += " ".join(f"{k}: {v*inv:.6e}" for k, v in running.items())
                    logger.write(log_text + "\n")

                    HB = BS // 2
                    pred_full = pred_lbs + disp
                    faces_cpu = batch.faces.cpu()
                    v_list = [
                        gt_v[0].cpu().detach(),             gt_v[min(1,BS-1)].cpu().detach(),
                        gt_v[min(HB,BS-1)].cpu().detach(),  gt_v[BS-1].cpu().detach(),
                        smt_v[0].cpu().detach(),             smt_v[min(1,BS-1)].cpu().detach(),
                        smt_v[min(HB,BS-1)].cpu().detach(),  smt_v[BS-1].cpu().detach(),
                        pred_lbs[0].cpu().detach(),          pred_lbs[min(1,BS-1)].cpu().detach(),
                        pred_lbs[min(HB,BS-1)].cpu().detach(), pred_lbs[BS-1].cpu().detach(),
                        pred_full[0].cpu().detach(),         pred_full[min(1,BS-1)].cpu().detach(),
                        pred_full[min(HB,BS-1)].cpu().detach(), pred_full[BS-1].cpu().detach(),
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
                torch.save(self.model_edd.state_dict(),
                           f'{opts.log_dir}/model_edd_{epoch:03d}.pth')

            if epoch % opts.eval_iter == 0:
                vis_mode = 'disp_only' if self.standalone else 'stage_disp'
                self.vis_loader.visualize(
                    self.model_hlbs, self.model_edd, epoch,
                    save_dir=f'{opts.log_dir}/img/eval',
                    mode=vis_mode,
                    stage=2,
                    smooth_n_iter=opts.smooth_n_iter,
                    no_t_mask=opts.no_t_mask,
                )

            # ── Valid ────────────────────────────────────────────────────────
            self.model_edd.eval()
            running_val = {"recon-wrinkle": 0.0, "total": 0.0}
            vcnt = 0

            pbar = tqdm(enumerate(valid_loader), total=len_valid, ncols=120,
                        desc=f"[{epoch:03d}] Valid EDD")
            for idx, batch in pbar:
                batch = batch.to(self.device)
                vcnt += 1
                with torch.no_grad():
                    src_v  = batch.template
                    src_n  = batch.template_normal
                    gt_v   = batch.vertices
                    gt_n   = batch.vertices_normal
                    smt_v  = batch.smooth_vertices

                    if self.standalone:
                        pred_lbs = smt_v
                        z_exp    = torch.zeros(
                            gt_v.shape[0], opts.hid_dim, device=self.device)
                    else:
                        delta     = gt_v - src_v
                        src_in    = torch.cat([src_v, src_n], dim=-1)
                        deform_in = torch.cat([delta, gt_n, src_in], dim=-1)
                        pred_lbs, z_exp = self.model_hlbs(
                            src_v, deform_in, return_z_exp=True)

                    jac_feat  = compute_jacobian_features(pred_lbs, src_v, self.faces_t)
                    disp      = self.model_edd(jac_feat, z_exp)
                    pred_full = pred_lbs + disp

                    wrinkle_target = (gt_v - pred_lbs) if opts.edd_target == 'lbs_residual' else (gt_v - smt_v)
                    val_loss = F.mse_loss(wrinkle_target, disp).item() * loss_lambda["recon-wrinkle"]
                    running_val["recon-wrinkle"] += val_loss
                    running_val["total"]         += val_loss

                pbar.set_description(f"[{epoch:03d}] val wrinkle: {val_loss:.5e}")

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
                        pred_full[0].cpu(),          pred_full[min(1,BS_v-1)].cpu(),
                        pred_full[min(HB_v,BS_v-1)].cpu(), pred_full[BS_v-1].cpu(),
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
                torch.save(self.model_edd.state_dict(), f'{opts.log_dir}/model_edd_best.pth')
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

    if opts.edd_target == 'smooth_gt':
        assert opts.smooth_n_iter > 0, "edd_target=smooth_gt requires --smooth_n_iter > 0"
    if opts.hlbs_ckpt is None:
        assert opts.edd_target == 'smooth_gt', \
            "Standalone mode (no --hlbs_ckpt) requires --edd_target smooth_gt --smooth_n_iter N"

    trainer = EDDTrainer(opts)
    trainer.train(epochs=opts.max_epoch)
