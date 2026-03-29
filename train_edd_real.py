"""
train_edd_real.py — EDD training with real npy data.

Coarse output source:
  - 'smooth_gt': Taubin-smoothed GT (pipeline test only, not for real experiments)
  - 'nfs_precomputed': precomputed NFS output npy (from precompute_nfs_coarse.py)
  - 'hlbs': frozen HLBS checkpoint (use train_edd.py instead)

EDD target: GT_real - coarse_output

Usage (pipeline test with smooth_gt):
    python train_edd_real.py \
        --neutral_obj utils/mf/mf_aligned_mean.obj \
        --datasets mf_ROM \
        --data_basedir /data/sihun \
        --coarse_mode smooth_gt --smooth_n_iter 16 \
        --batch_size 8 --max_epoch 300 --tb

Usage (real experiment with NFS precomputed):
    python train_edd_real.py \
        --neutral_obj utils/mf/mf_aligned_mean.obj \
        --datasets mf_ROM \
        --data_basedir /data/sihun \
        --coarse_mode nfs_precomputed --coarse_dir ./precomputed_nfs/ \
        --batch_size 8 --max_epoch 300 --tb --lambda_normal 0.1
"""
import os
import sys
import json
import argparse
import random
import numpy as np
import yaml

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils.matplotlib_rnd import plot_image_array
from utils.mesh_utils import compute_jacobian_features, calc_norm_torch, taubin_smooth_np


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

    # topology
    parser.add_argument("--topo_key", type=str, default='mf',
                        choices=['mf', 'biwi', 'voca'])
    parser.add_argument("--neutral_obj", type=str, required=True)
    parser.add_argument("--op_cache_dir", type=str, default='utils/diffusion_ops')

    # EDD architecture
    parser.add_argument("--edd_z_dim", type=int, default=0,
                        help='z_exp dim (0=no z_exp, jac_feat only)')
    parser.add_argument("--edd_hid_channels", type=int, default=128)
    parser.add_argument("--edd_n_blocks", type=int, default=4)
    parser.add_argument("--edd_k_eig", type=int, default=128)

    # coarse output source
    parser.add_argument("--coarse_mode", type=str, default='smooth_gt',
                        choices=['smooth_gt', 'nfs_precomputed', 'hlbs'])
    parser.add_argument("--smooth_n_iter", type=int, default=16,
                        help='Taubin smoothing iters (coarse_mode=smooth_gt)')
    parser.add_argument("--coarse_dir", type=str, default=None,
                        help='Precomputed coarse npy dir (coarse_mode=nfs_precomputed)')

    # data
    parser.add_argument("--datasets", type=str, nargs='+', default=['mf_ROM'])
    parser.add_argument("--data_basedir", type=str, default='/data/sihun')
    parser.add_argument("--data_toggle", action='store_true')

    # training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--eval_iter", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sc_step", type=int, default=1000000)
    parser.add_argument("--sc_gamma", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)

    # loss
    parser.add_argument("--lambda_vert", type=float, default=1.0)
    parser.add_argument("--lambda_normal", type=float, default=0.0,
                        help='Normal consistency loss (1 - cos(n_pred, n_gt))')
    parser.add_argument("--lambda_curvature", type=float, default=0.0,
                        help='Curvature loss (Laplacian difference)')

    # mask
    parser.add_argument("--no_t_mask", dest='no_t_mask', action='store_true')
    parser.set_defaults(no_t_mask=False)

    # checkpoints
    parser.add_argument("--edd_ckpt", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)

    # logging
    parser.add_argument("--log_dir", type=str, default="./ckpts_edd")
    parser.add_argument("--tb", dest='tb', action='store_true')
    parser.set_defaults(tb=False)
    parser.add_argument("--debug", dest='debug', action='store_true')
    parser.set_defaults(debug=False)

    return parser.parse_args()


def _uniform_laplacian(verts, faces):
    """L @ v = mean(neighbors) - v, approximate curvature."""
    B, V, _ = verts.shape
    device = verts.device
    f = faces if faces.dim() == 2 else faces[0]

    e0 = torch.cat([f[:, 0], f[:, 1], f[:, 0]])
    e1 = torch.cat([f[:, 1], f[:, 2], f[:, 2]])
    src = torch.cat([e0, e1])
    tgt = torch.cat([e1, e0])

    neighbor_sum = torch.zeros(B, V, 3, device=device)
    neighbor_cnt = torch.zeros(B, V, 1, device=device)
    tgt_exp = tgt.unsqueeze(0).unsqueeze(-1).expand(B, -1, 3)
    neighbor_sum.scatter_add_(1, tgt_exp, verts[:, src, :])
    neighbor_cnt.scatter_add_(1, tgt.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1),
                              torch.ones(B, src.shape[0], 1, device=device))
    return neighbor_sum / neighbor_cnt.clamp(min=1) - verts


class EDDRealTrainer:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)

        from models.diffusion_edd import DiffusionNetEDD

        self.model_edd = DiffusionNetEDD(
            z_dim=opts.edd_z_dim,
            hid_channels=opts.edd_hid_channels,
            n_blocks=opts.edd_n_blocks,
            k_eig=opts.edd_k_eig,
        ).to(self.device)

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
        self.faces_np = faces_np
        self.faces_t = torch.tensor(faces_np, dtype=torch.long, device=self.device)

        if opts.edd_ckpt and os.path.exists(opts.edd_ckpt):
            self.model_edd.load_state_dict(
                torch.load(opts.edd_ckpt, map_location=self.device))
            print(f"Loaded EDD: {opts.edd_ckpt}")

    def _get_coarse(self, gt_v, src_v, faces):
        """Get coarse deformed mesh. Returns [B, V, 3] on same device."""
        opts = self.opts
        if opts.coarse_mode == 'smooth_gt':
            B = gt_v.shape[0]
            faces_np = faces.cpu().numpy() if faces.dim() == 2 else faces[0].cpu().numpy()
            smoothed = []
            for b in range(B):
                s = taubin_smooth_np(gt_v[b].cpu().numpy(), faces_np,
                                     n_iter=opts.smooth_n_iter)
                smoothed.append(torch.tensor(s, dtype=torch.float32))
            return torch.stack(smoothed).to(gt_v.device)
        elif opts.coarse_mode == 'nfs_precomputed':
            # TODO: load precomputed npy aligned with real npy indices
            raise NotImplementedError(
                "nfs_precomputed: run precompute_nfs_coarse.py first, "
                "then pass --coarse_dir")
        elif opts.coarse_mode == 'hlbs':
            raise NotImplementedError("hlbs: use train_edd.py instead")

    def train(self, epochs):
        opts = self.opts
        BS = opts.batch_size

        self.optimizer = torch.optim.AdamW(
            self.model_edd.parameters(), lr=opts.lr, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opts.sc_step, gamma=opts.sc_gamma)

        from dataloader_real import RealNpyDataset, real_collate_fn

        train_ds = RealNpyDataset(
            data_basedir=opts.data_basedir, datasets=opts.datasets,
            mode='train', toggle=opts.data_toggle)
        valid_ds = RealNpyDataset(
            data_basedir=opts.data_basedir, datasets=opts.datasets,
            mode='val', toggle=opts.data_toggle)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=BS, shuffle=True,
            collate_fn=real_collate_fn, num_workers=opts.num_workers, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_ds, batch_size=BS, shuffle=False,
            collate_fn=real_collate_fn, num_workers=0, drop_last=False)

        # Logging
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        coarse_tag = f"-{opts.coarse_mode}"
        if opts.coarse_mode == 'smooth_gt':
            coarse_tag += f"-s{opts.smooth_n_iter}"
        zdim_tag = f"-z{opts.edd_z_dim}" if opts.edd_z_dim > 0 else "-noZ"
        loss_tag = ""
        if opts.lambda_normal > 0:
            loss_tag += f"-nrm{opts.lambda_normal}"
        if opts.lambda_curvature > 0:
            loss_tag += f"-crv{opts.lambda_curvature}"
        tag = f"-EDD-real-{opts.topo_key}{coarse_tag}{zdim_tag}{loss_tag}"
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
            f"=== EDD Real Training ===\n"
            f"  topo_key       : {opts.topo_key}\n"
            f"  datasets       : {opts.datasets}\n"
            f"  coarse_mode    : {opts.coarse_mode}\n"
            f"  smooth_n_iter  : {opts.smooth_n_iter}\n"
            f"  edd_z_dim      : {opts.edd_z_dim}\n"
            f"  lambda_vert    : {opts.lambda_vert}\n"
            f"  lambda_normal  : {opts.lambda_normal}\n"
            f"  lambda_curvature: {opts.lambda_curvature}\n"
            f"  train frames   : {len(train_ds)}\n"
            f"  val frames     : {len(valid_ds)}\n"
            f"=============================\n"
        )
        print(config_text)
        logger.write(config_text)

        BEST_LOSS = 1e8
        BEST_EPOCH = 0
        len_train = len(train_loader)
        len_valid = len(valid_loader)
        interv = max(1, round(len_train / 10))
        interv_val = max(1, round(len_valid / 3))

        loss_keys = ["recon-vert"]
        if opts.lambda_normal > 0:
            loss_keys.append("recon-normal")
        if opts.lambda_curvature > 0:
            loss_keys.append("recon-curvature")
        loss_keys.append("total")

        for epoch in range(opts.start_epoch, epochs + 1):
            # ── Train ─────────────────────────────────────────────────
            self.model_edd.train()
            running = {k: 0.0 for k in loss_keys}
            cnt = 0

            pbar = tqdm(enumerate(train_loader), total=len_train, ncols=120,
                        desc=f"[{epoch:03d}] Train EDD")
            for idx, batch in pbar:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                src_v = batch.template
                gt_v  = batch.vertices
                gt_n  = batch.vertices_normal

                with torch.no_grad():
                    coarse = self._get_coarse(gt_v, src_v, batch.faces)

                jac_feat = compute_jacobian_features(coarse, src_v, self.faces_t)
                disp = self.model_edd(jac_feat)
                pred_full = coarse + disp

                # Vertex loss
                target = gt_v - coarse
                if opts.no_t_mask:
                    loss_vert = F.mse_loss(target, disp)
                else:
                    from utils.exp_utils import plateau_hat_points
                    t_mask = plateau_hat_points(src_v)
                    loss_vert = F.mse_loss(target * t_mask, disp * t_mask)

                total_loss = loss_vert * opts.lambda_vert
                running["recon-vert"] += (loss_vert * opts.lambda_vert).item()

                # Normal loss
                if opts.lambda_normal > 0:
                    pred_n = calc_norm_torch(pred_full, batch.faces, at='verts')
                    loss_normal = (1 - F.cosine_similarity(pred_n, gt_n, dim=-1)).mean()
                    total_loss = total_loss + loss_normal * opts.lambda_normal
                    running["recon-normal"] += (loss_normal * opts.lambda_normal).item()

                # Curvature loss
                if opts.lambda_curvature > 0:
                    loss_curv = F.mse_loss(
                        _uniform_laplacian(pred_full, batch.faces),
                        _uniform_laplacian(gt_v, batch.faces))
                    total_loss = total_loss + loss_curv * opts.lambda_curvature
                    running["recon-curvature"] += (loss_curv * opts.lambda_curvature).item()

                total_loss.backward()
                self.optimizer.step()

                running["total"] += total_loss.item()
                cnt += 1
                pbar.set_description(f"[{epoch:03d}] vert: {loss_vert:.5e}")

                # Vis
                if idx % interv == 1:
                    inv = 1.0 / cnt
                    log_text = f"[{epoch:03d}/{epochs:03d}][{idx:04d}][Train] "
                    log_text += " ".join(f"{k}: {v*inv:.6e}" for k, v in running.items())
                    logger.write(log_text + "\n")

                    HB = BS // 2
                    _d = lambda t: t.cpu().detach()
                    _s = lambda i: min(i, BS - 1)
                    faces_cpu = batch.faces[0].cpu() if batch.faces.dim() == 3 else batch.faces.cpu()
                    v_list = [
                        _d(gt_v[0]),      _d(gt_v[_s(1)]),
                        _d(gt_v[_s(HB)]), _d(gt_v[BS-1]),
                        _d(coarse[0]),      _d(coarse[_s(1)]),
                        _d(coarse[_s(HB)]), _d(coarse[BS-1]),
                        _d(pred_full[0]),      _d(pred_full[_s(1)]),
                        _d(pred_full[_s(HB)]), _d(pred_full[BS-1]),
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

            # ── Valid ─────────────────────────────────────────────────
            self.model_edd.eval()
            running_val = {k: 0.0 for k in loss_keys}
            vcnt = 0

            pbar = tqdm(enumerate(valid_loader), total=len_valid, ncols=120,
                        desc=f"[{epoch:03d}] Valid EDD")
            for idx, batch in pbar:
                batch = batch.to(self.device)
                vcnt += 1
                with torch.no_grad():
                    src_v = batch.template
                    gt_v  = batch.vertices
                    gt_n  = batch.vertices_normal

                    coarse = self._get_coarse(gt_v, src_v, batch.faces)
                    jac_feat = compute_jacobian_features(coarse, src_v, self.faces_t)
                    disp = self.model_edd(jac_feat)
                    pred_full = coarse + disp

                    target = gt_v - coarse
                    vl = F.mse_loss(target, disp).item() * opts.lambda_vert
                    running_val["recon-vert"] += vl
                    vt = vl

                    if opts.lambda_normal > 0:
                        pred_n = calc_norm_torch(pred_full, batch.faces, at='verts')
                        nl = (1 - F.cosine_similarity(pred_n, gt_n, dim=-1)).mean().item() * opts.lambda_normal
                        running_val["recon-normal"] += nl
                        vt += nl

                    if opts.lambda_curvature > 0:
                        cl = F.mse_loss(
                            _uniform_laplacian(pred_full, batch.faces),
                            _uniform_laplacian(gt_v, batch.faces)).item() * opts.lambda_curvature
                        running_val["recon-curvature"] += cl
                        vt += cl

                    running_val["total"] += vt

                pbar.set_description(f"[{epoch:03d}] val: {vl:.5e}")

                if idx % interv_val == 0:
                    BS_v = gt_v.shape[0]
                    HB_v = BS_v // 2
                    _s = lambda i: min(i, BS_v - 1)
                    faces_cpu = batch.faces[0].cpu() if batch.faces.dim() == 3 else batch.faces.cpu()
                    v_list = [
                        gt_v[0].cpu(),      gt_v[_s(1)].cpu(),
                        gt_v[_s(HB_v)].cpu(), gt_v[BS_v-1].cpu(),
                        coarse[0].cpu(),      coarse[_s(1)].cpu(),
                        coarse[_s(HB_v)].cpu(), coarse[BS_v-1].cpu(),
                        pred_full[0].cpu(),      pred_full[_s(1)].cpu(),
                        pred_full[_s(HB_v)].cpu(), pred_full[BS_v-1].cpu(),
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
                BEST_LOSS = total_val
                BEST_EPOCH = epoch
                torch.save(self.model_edd.state_dict(), f'{opts.log_dir}/model_edd_best.pth')
                print(f"[{epoch:03d}] Best: {BEST_LOSS:.6e} (epoch {BEST_EPOCH})")
                logger.write(f"[{epoch:03d}] Best Loss: {BEST_LOSS:.6e}\n")
            else:
                print(f"[{epoch:03d}] Val: {total_val:.6e} (Best: {BEST_LOSS:.6e} [{BEST_EPOCH}])")
                logger.write(f"[{epoch:03d}] Val: {total_val:.6e} (Best: {BEST_LOSS:.6e} [{BEST_EPOCH}])\n")


if __name__ == "__main__":
    opts = Options()
    trainer = EDDRealTrainer(opts)
    trainer.train(epochs=opts.max_epoch)
