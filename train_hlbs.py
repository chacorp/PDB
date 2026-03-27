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

        rig = load_rig(opts.rig_path)
        self.model = HierarchicalLBS(
            rig=rig,
            topology=opts.topo_key,
            in_dim_exp=12,          # delta(3) + deform_norm(3) + src_v(3) + src_n(3)
            hid_dim=opts.hid_dim,
            num_layers=opts.num_layers,
            device=str(self.device),
            freeze_adapt=opts.freeze_adapt,
            use_joint_trans=opts.use_joint_trans,
        ).to(self.device)

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

        self.vis_loader = CheckpointVisLoader(opts, device=self.device)
        self._edges = None   # lazily computed from first batch faces

    def train(self, epochs):
        opts = self.opts
        BS = opts.batch_size

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=opts.lr, betas=(0.9, 0.999))
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
            tag = f"-HLBS-{opts.topo_key}-s{opts.smooth_n_iter}{freeze_tag}{trans_tag}"
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
        }

        BEST_LOSS  = 1e8
        BEST_EPOCH = 0
        len_train  = len(train_loader)
        len_valid  = len(valid_loader)
        interv     = max(1, round(len_train / 10))
        interv_val = max(1, round(len_valid / 3))

        for epoch in range(opts.start_epoch, epochs + 1):
            # ── Train ────────────────────────────────────────────────────────
            self.model.train()
            running = {"recon-lbs": 0.0, "recon-neu": 0.0, "lbs-W-reg": 0.0, "lbs-t-reg": 0.0, "lbs-W-smooth": 0.0, "total": 0.0}
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

                pred_lbs = self.model(src_v, deform_in)              # [B, N, 3]

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
                    faces_cpu = batch.faces.cpu()
                    v_list = [
                        gt_v[0].cpu().detach(),            gt_v[min(1,BS-1)].cpu().detach(),
                        gt_v[min(HB,BS-1)].cpu().detach(), gt_v[BS-1].cpu().detach(),
                        smt_v[0].cpu().detach(),            smt_v[min(1,BS-1)].cpu().detach(),
                        smt_v[min(HB,BS-1)].cpu().detach(), smt_v[BS-1].cpu().detach(),
                        pred_lbs[0].cpu().detach(),            pred_lbs[min(1,BS-1)].cpu().detach(),
                        pred_lbs[min(HB,BS-1)].cpu().detach(), pred_lbs[BS-1].cpu().detach(),
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
            running_val = {"recon-lbs": 0.0, "recon-neu": 0.0, "total": 0.0}
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
                    pred_lbs  = self.model(src_v, deform_in)

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

                pbar.set_description(f"[{epoch:03d}] val lbs: {val_loss:.5e}")

                if idx % interv_val == 0:
                    BS_v = batch.vertices.shape[0]
                    HB_v = BS_v // 2
                    faces_cpu = batch.faces.cpu()
                    v_list = [
                        gt_v[0].cpu(),             gt_v[min(1,BS_v-1)].cpu(),
                        gt_v[min(HB_v,BS_v-1)].cpu(), gt_v[BS_v-1].cpu(),
                        smt_v[0].cpu(),             smt_v[min(1,BS_v-1)].cpu(),
                        smt_v[min(HB_v,BS_v-1)].cpu(), smt_v[BS_v-1].cpu(),
                        pred_lbs[0].cpu(),             pred_lbs[min(1,BS_v-1)].cpu(),
                        pred_lbs[min(HB_v,BS_v-1)].cpu(), pred_lbs[BS_v-1].cpu(),
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
    trainer.train(epochs=opts.max_epoch)
