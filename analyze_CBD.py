#!/usr/bin/env python3
"""
analyze_CBD.py

Analyze a trained LBS+CBD joint training model (version 8):
  (1) LBS joint weights  & joint positions visualization
  (2) CBD cage vertex weights visualization  (top-K by avg key_d magnitude)
  (3) Residual over template: ||GT - pred|| per vertex as heatmap
  (4) MSE sanity check: raw MSE vs global-translation-corrected MSE

Usage:
    python analyze_CBD.py \
        --version 8 \
        --ckpt ./ckpts_CBD/2026-02-13-07-54-53-NGBC++v8 \
        --data_selection 4 \
        --realtest \
        --batch_size 1 \
        --num_top_cage 32 \
        --log_dir ./analysis_CBD
"""

import os
import glob
import json
import yaml
import random
import csv
import numpy as np
import argparse
from tqdm import tqdm
from functools import partial

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.collections import PolyCollection

import torch
import torch.nn.functional as F
import cv2

from dataloader_CBD import (
    CBDdataSampler,
    CBDDataset,
    CBD_collate_wrapper,
    EvalDataset,
    CBDDataBatch_eval,
    CBD_collate_wrapper_eval,
)
from utils.matplotlib_rnd import (
    vis_mesh_key_weight,
    vis_mesh_all_cage_weights,
    plot_image_array,
    normalize_homogeneous,
    translate,
    perspective,
    yrotate,
)
from models.NGBC import (
    NeuralGeneralizedBarycentricCoordinateLBS,
    NeuralGeneralizedBarycentricCoordinateCBD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def images_to_video_cv(img_dir, out_path, fps=30, pattern="*.png"):
    imgs = sorted(glob.glob(os.path.join(img_dir, pattern)))
    if not imgs:
        print(f"[video] No images found in {img_dir} matching {pattern}")
        return
    first = cv2.imread(imgs[0])
    h, w, _ = first.shape
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for p in imgs:
        writer.write(cv2.imread(p))
    writer.release()
    print(f"[video] Saved: {out_path}")


def make_grid_from_images(img_paths, save_path, ncols=8, thumb_hw=(150, 450)):
    """Stack individual images into a grid for overview."""
    loaded = []
    for p in sorted(img_paths):
        img = cv2.imread(p)
        if img is not None:
            loaded.append(cv2.resize(img, (thumb_hw[1], thumb_hw[0])))
    if not loaded:
        return

    nrows = (len(loaded) + ncols - 1) // ncols
    blank = np.zeros((thumb_hw[0], thumb_hw[1], 3), dtype=np.uint8)
    while len(loaded) < nrows * ncols:
        loaded.append(blank)

    rows = []
    for r in range(nrows):
        rows.append(np.hstack(loaded[r * ncols: (r + 1) * ncols]))
    grid = np.vstack(rows)
    cv2.imwrite(save_path, grid)
    print(f"[grid] Saved: {save_path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def Options():
    parser = argparse.ArgumentParser(
        description="LBS+CBD Analysis: weights, residuals, MSE sanity check"
    )
    parser.add_argument("--config",       default="config/train_CBD.yml")
    parser.add_argument("--device",       type=str, default="cuda:0")
    parser.add_argument("--log_dir",      type=str, default="analysis_CBD")
    parser.add_argument("--version",      type=int, default=8)
    parser.add_argument("--data_selection", type=int, default=4,
                        help="-1=all, 0=voca, 1=biwi, 2=mf_SEN, 3=coma, 4=mf_ROM, 5=ict")
    parser.add_argument("--last_activation", default="relu",
                        choices=["relu", "elu", "softmax", "softplus", "none", "sqrelu"])
    parser.add_argument("--no_pou",       dest="no_pou", action="store_true")
    parser.set_defaults(no_pou=False)
    parser.add_argument("--start_epoch",  type=int, default=0)
    parser.add_argument("--batch_size",   type=int, default=1)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--ckpt",         type=str, default=None)
    parser.add_argument("--continue_ckpt", dest="continue_ckpt", action="store_true")
    parser.set_defaults(continue_ckpt=False)

    # LBS/CBD architecture flags (overridden by train_opts.yml)
    parser.add_argument("--num_lbs_joints",            type=int,   default=64)
    parser.add_argument("--use_lbs",                   dest="use_lbs",                    action="store_true"); parser.set_defaults(use_lbs=False)
    parser.add_argument("--vis_joint_pos",             dest="vis_joint_pos",              action="store_true"); parser.set_defaults(vis_joint_pos=False)
    parser.add_argument("--use_lbs_joint_center",      dest="use_lbs_joint_center",       action="store_true"); parser.set_defaults(use_lbs_joint_center=False)
    parser.add_argument("--use_exp_joint_predict",     dest="use_exp_joint_predict",      action="store_true"); parser.set_defaults(use_exp_joint_predict=False)
    parser.add_argument("--use_joint_predict",         dest="use_joint_predict",          action="store_true"); parser.set_defaults(use_joint_predict=False)
    parser.add_argument("--use_weighted_joint_pos",    dest="use_weighted_joint_pos",     action="store_true"); parser.set_defaults(use_weighted_joint_pos=False)
    parser.add_argument("--no_use_translation",        dest="no_use_translation",         action="store_true"); parser.set_defaults(no_use_translation=False)
    parser.add_argument("--use_lbs_laplacian",         dest="use_lbs_laplacian",          action="store_true"); parser.set_defaults(use_lbs_laplacian=False)
    parser.add_argument("--use_lbs_ent",               dest="use_lbs_ent",                action="store_true"); parser.set_defaults(use_lbs_ent=False)
    parser.add_argument("--use_lbs_t",                 dest="use_lbs_t",                  action="store_true"); parser.set_defaults(use_lbs_t=False)
    parser.add_argument("--use_lbs_R",                 dest="use_lbs_R",                  action="store_true"); parser.set_defaults(use_lbs_R=False)
    parser.add_argument("--use_lbs_bal",               dest="use_lbs_bal",                action="store_true"); parser.set_defaults(use_lbs_bal=False)
    parser.add_argument("--use_hyb_delta_lbs_input",   dest="use_hyb_delta_lbs_input",    action="store_true"); parser.set_defaults(use_hyb_delta_lbs_input=False)
    parser.add_argument("--use_hyb_concat_lbs",        dest="use_hyb_concat_lbs",         action="store_true"); parser.set_defaults(use_hyb_concat_lbs=False)
    parser.add_argument("--use_finetune_lbs",          dest="use_finetune_lbs",           action="store_true"); parser.set_defaults(use_finetune_lbs=False)
    parser.add_argument("--use_hyb_joint_train",       dest="use_hyb_joint_train",        action="store_true"); parser.set_defaults(use_hyb_joint_train=False)

    # Data toggles
    parser.add_argument("--use_data0", dest="use_data0", action="store_true"); parser.set_defaults(use_data0=False)
    parser.add_argument("--use_data1", dest="use_data1", action="store_true"); parser.set_defaults(use_data1=False)
    parser.add_argument("--use_data2", dest="use_data2", action="store_true"); parser.set_defaults(use_data2=False)
    parser.add_argument("--use_data3", dest="use_data3", action="store_true"); parser.set_defaults(use_data3=False)
    parser.add_argument("--use_data9", dest="use_data9", action="store_true"); parser.set_defaults(use_data9=False)

    parser.add_argument("--realtest",     dest="realtest",    action="store_true"); parser.set_defaults(realtest=False)
    parser.add_argument("--align_latent", dest="align_latent", action="store_true"); parser.set_defaults(align_latent=False)

    # Analysis-specific
    parser.add_argument("--num_top_cage", type=int, default=32,
                        help="Top-K cage vertices to visualize (sorted by avg ||key_d||)")
    parser.add_argument("--max_frames",   type=int, default=-1,
                        help="Max frames to process for residuals (-1 = all)")
    parser.add_argument("--weight_vis_only", dest="weight_vis_only", action="store_true",
                        help="Only visualize weights from first frame, skip residual loop")
    parser.set_defaults(weight_vis_only=False)
    parser.add_argument("--vis_frame", type=int, default=0,
                        help="Frame index to use for weight visualization (default: 0)")
    parser.add_argument("--run_tag", type=str, default="",
                        help="Sub-directory tag appended to base_dir (e.g. 'eBest', 'e500')")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, opts):
        self.opts = opts
        self._set_seed(opts)
        self.device = opts.device

        assert opts.version in (6, 8), "analyze_CBD.py supports version 6 (LBS-only) and 8 (LBS+CBD)"

        last_act_list = ["relu", "elu", "softmax", "softplus", "none", "sqrelu"]
        la = [opts.last_activation == l for l in last_act_list]

        model_kwargs = dict(
            num_layers=4,
            num_cage_vertices=opts.num_cage_v,
            use_exp_recon=False,
            use_shp_recon=False,
            use_shp=False,
            use_relu=la[0], use_elu=la[1], use_softmax=la[2],
            use_softplus=la[3], no_activation=la[4],
            is_train=True,
            use_pou=not opts.no_pou,
            device=self.device,
            hid_dim=128 if opts.align_latent else 256,
        )

        self.model = NeuralGeneralizedBarycentricCoordinateLBS(opts, **model_kwargs).to(self.device)

        if opts.version == 8:
            self.model_CBD = NeuralGeneralizedBarycentricCoordinateCBD(opts, **model_kwargs).to(self.device)
            self._load_weight(self.model,     name="lbs", ckpt_dir=opts.ckpt, epoch=opts.start_epoch)
            self._load_weight(self.model_CBD, name="cbd", ckpt_dir=opts.ckpt, epoch=opts.start_epoch)
            self.model_CBD.eval()
        elif opts.version == 6:
            self.model_CBD = None
            self._load_weight_v6(self.model, ckpt_dir=opts.ckpt, epoch=opts.start_epoch)

        self.model.eval()

    # ------------------------------------------------------------------

    def _load_weight(self, model, name, ckpt_dir, epoch):
        """v8: model_lbs_best.pth / model_cbd_best.pth style."""
        if not ckpt_dir:
            print(f"[warn] No ckpt_dir for {name}")
            return
        if self.opts.continue_ckpt:
            ckpt = glob.glob(os.path.join(ckpt_dir, f"*_{name}_{epoch:03d}.pth"))[0]
        else:
            ckpt = glob.glob(os.path.join(ckpt_dir, f"*_{name}_best.pth"))[0]
        ckpt_dict = torch.load(ckpt, map_location=self.device)
        model.load_state_dict(ckpt_dict)
        print(f"[load] {name}: {os.path.basename(ckpt)}")

    def _load_weight_v6(self, model, ckpt_dir, epoch):
        """v6 (LBS-only): single model_300.pth style."""
        if not ckpt_dir:
            print("[warn] No ckpt_dir for v6")
            return
        if self.opts.continue_ckpt:
            ckpt = glob.glob(os.path.join(ckpt_dir, f"*_{epoch:03d}.pth"))[0]
        else:
            ckpt = glob.glob(os.path.join(ckpt_dir, "*_best.pth"))[0]
        ckpt_dict = torch.load(ckpt, map_location=self.device)
        model.load_state_dict(ckpt_dict)
        print(f"[load] v6: {os.path.basename(ckpt)}")

    @staticmethod
    def _set_seed(opts):
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opts.seed)
        random.seed(opts.seed)

    # ------------------------------------------------------------------
    # Main analysis entry point
    # ------------------------------------------------------------------

    def analyze(self):
        # ---- Dataset ----
        data_name_list = ["voca", "biwi", "mf_SEN", "coma", "mf_ROM", "ict"]
        assert self.opts.data_selection != -1, "--data_selection must be a single dataset (not -1)"
        selection = data_name_list[self.opts.data_selection]

        if self.opts.realtest:
            dataset = EvalDataset(data_name=selection, toggle=False)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.opts.batch_size,
                collate_fn=partial(CBD_collate_wrapper_eval, device=self.device),
            )
        else:
            flags = [False] * 5
            flags[self.opts.data_selection] = True
            dataset = CBDDataset(
                self.opts, n_components=1_000,
                is_train=False, is_valid=False,
                use_voca=flags[0], use_biwi=flags[1],
                use_mf_SEN=flags[2], use_coma=flags[3], use_mf_ROM=flags[4],
            )
            sampler = CBDdataSampler(
                dataset.len_list, self.opts.batch_size,
                shuffle=False, balance=False, is_train=False, is_valid=False,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_sampler=sampler,
                collate_fn=partial(CBD_collate_wrapper, device=self.opts.device),
                num_workers=0,
            )

        # ---- Output dirs ----
        ckpt_name = self.opts.ckpt.rstrip("/").split("/")[-1]
        base_dir = os.path.join(self.opts.log_dir, ckpt_name, selection,
                                self.opts.run_tag) if self.opts.run_tag else \
                   os.path.join(self.opts.log_dir, ckpt_name, selection)
        lbs_w_dir = os.path.join(base_dir, "lbs_weights")
        cbd_w_dir = os.path.join(base_dir, "cbd_weights")
        res_dir   = os.path.join(base_dir, "residual")
        mse_dir   = os.path.join(base_dir, "mse_analysis")
        for d in [lbs_w_dir, cbd_w_dir, res_dir, mse_dir]:
            os.makedirs(d, exist_ok=True)
        print(f"[analyze] Saving to: {base_dir}")

        # ---- Accumulators ----
        first_frame_done  = False
        first_template_np = None
        first_faces_np    = None
        first_W_lbs_np    = None
        first_C_lbs_np    = None
        first_kw_np       = None

        key_d_norms_list  = []   # (M,) per frame, for top-K selection
        mse_records       = []

        len_data = len(dataloader)
        pbar = tqdm(enumerate(dataloader), total=len_data, ncols=100)

        for index, batch in pbar:
            if self.opts.max_frames > 0 and index >= self.opts.max_frames:
                break

            with torch.no_grad():
                # -- LBS forward --
                # Returns 10 values (standard, vis_joint_pos=False):
                # pred_lbs, recon, recon_src, exp_z, pred_src, t_mask,
                # key_d_zeros, kw_zeros, W_lbs, T_lbs
                (pred_lbs, _recon_lbs, _recon_src_lbs, _exp_z_lbs, _pred_src_lbs,
                 _t_mask_lbs, _kd_zeros, _kw_zeros,
                 W_lbs, T_lbs) = self.model(
                    batch.template, batch.vertices,
                    batch.template_normal, batch.vertices_normal,
                    batch.mesh_data, epoch=0,
                )

                # -- CBD forward (v8 only) --
                if self.model_CBD is not None:
                    # Returns 6 values: pred_cbd, recon, recon_src, exp_z, key_d, key_weight
                    if self.opts.use_hyb_delta_lbs_input:
                        (pred_cbd, _r, _rs, _ez, key_d, key_weight) = self.model_CBD(
                            batch.template, batch.vertices,
                            batch.template_normal, batch.vertices_normal,
                            batch.mesh_data, epoch=0, lbs_output=pred_lbs, out_kw=True,
                        )
                    elif self.opts.use_hyb_concat_lbs:
                        (pred_cbd, _r, _rs, _ez, key_d, key_weight) = self.model_CBD(
                            batch.template, batch.vertices,
                            batch.template_normal, batch.vertices_normal,
                            batch.mesh_data, epoch=0,
                            lbs_output=pred_lbs, lbs_source=_pred_src_lbs, out_kw=True,
                        )
                    else:  # default: independent CBD branches
                        (pred_cbd, _r, _rs, _ez, key_d, key_weight) = self.model_CBD(
                            batch.template, batch.vertices,
                            batch.template_normal, batch.vertices_normal,
                            batch.mesh_data, epoch=0, out_kw=True,
                        )
                    pred_final = pred_lbs + pred_cbd
                else:
                    # v6: LBS only
                    pred_cbd = torch.zeros_like(pred_lbs)
                    key_d = None
                    key_weight = None
                    pred_final = pred_lbs

            # ---- numpy conversion (batch_size=1) ----
            template_np   = batch.template[0].cpu().numpy()      # (N, 3)
            gt_np         = batch.vertices[0].cpu().numpy()       # (N, 3)
            pred_lbs_np   = pred_lbs[0].cpu().numpy()             # (N, 3)
            pred_cbd_np   = pred_cbd[0].cpu().numpy()             # (N, 3)
            pred_final_np = pred_final[0].cpu().numpy()           # (N, 3)
            W_lbs_np      = W_lbs[0].cpu().numpy()                # (N, J)
            key_weight_np = key_weight[0].cpu().numpy() if key_weight is not None else None

            # Joint centers = T_lbs last column (translation t).
            # T_lbs is (B, J, 3, 4) = [R | t], so last column is joint center.
            C_lbs_np = T_lbs[0, :, :, 3].cpu().numpy()            # (J, 3)
            key_d_np = key_d[0].cpu().numpy() if key_d is not None else None

            # batch.faces is (B, F, 3) — always take [0] to get (F, 3)
            faces_np = batch.faces[0].cpu().numpy()

            # ---- Store vis_frame data for weight visualization ----
            if not first_frame_done and index == self.opts.vis_frame:
                first_template_np = template_np.copy()
                first_faces_np    = faces_np.copy()
                first_W_lbs_np    = W_lbs_np.copy()
                first_C_lbs_np    = C_lbs_np.copy()
                first_kw_np       = key_weight_np.copy() if key_weight_np is not None else None
                first_frame_done  = True

            # Accumulate key_d norm for top-K cage selection (v8 only)
            if key_d_np is not None:
                key_d_norms_list.append(np.linalg.norm(key_d_np, axis=-1))  # (M,)

            # ---- MSE analysis ----
            mse_full = float(np.mean((gt_np - pred_final_np) ** 2))
            mse_lbs  = float(np.mean((gt_np - pred_lbs_np) ** 2))

            # Global translation correction: remove per-frame mean offset
            global_t      = (gt_np - pred_final_np).mean(axis=0, keepdims=True)  # (1, 3)
            global_t_lbs  = (gt_np - pred_lbs_np).mean(axis=0, keepdims=True)

            mse_full_c = float(np.mean((gt_np - (pred_final_np + global_t)) ** 2))
            mse_lbs_c  = float(np.mean((gt_np - (pred_lbs_np  + global_t_lbs)) ** 2))
            global_t_mag = float(np.linalg.norm(global_t))

            mse_records.append({
                "frame_idx":                    index,
                "mse_full":                     mse_full,
                "mse_lbs":                      mse_lbs,
                "mse_full_centered":            mse_full_c,
                "mse_lbs_centered":             mse_lbs_c,
                "global_translation_magnitude": global_t_mag,
            })

            # ---- Residual heatmaps ----
            if not self.opts.weight_vis_only:
                res_lbs  = np.linalg.norm(gt_np - pred_lbs_np,   axis=-1)  # (N,)
                res_full = np.linalg.norm(gt_np - pred_final_np,  axis=-1)  # (N,)
                cbd_cont = np.linalg.norm(pred_cbd_np,            axis=-1)  # (N,)

                # shared vmax across LBS and Full for fair comparison
                vmax_res = max(res_lbs.max(), res_full.max())

                vis_mesh_key_weight(
                    verts=template_np, faces=faces_np,
                    key_weight=res_lbs.reshape(-1, 1), cage_idx=0,
                    cmap="hot", vmin=0.0, vmax=vmax_res,
                    view_yrots=(0, 90, 180, 270),
                    save_path=os.path.join(res_dir, f"frame_{index:04d}_lbs.png"),
                )
                vis_mesh_key_weight(
                    verts=template_np, faces=faces_np,
                    key_weight=res_full.reshape(-1, 1), cage_idx=0,
                    cmap="hot", vmin=0.0, vmax=vmax_res,
                    view_yrots=(0, 90, 180, 270),
                    save_path=os.path.join(res_dir, f"frame_{index:04d}_full.png"),
                )
                vis_mesh_key_weight(
                    verts=template_np, faces=faces_np,
                    key_weight=cbd_cont.reshape(-1, 1), cage_idx=0,
                    cmap="Blues", vmin=0.0, vmax=cbd_cont.max(),
                    view_yrots=(0, 90, 180, 270),
                    save_path=os.path.join(res_dir, f"frame_{index:04d}_cbd_contrib.png"),
                )

        # ==================================================================
        # Post-loop: weight visualizations, MSE CSV, videos
        # ==================================================================

        # ---- (1) LBS joint weight heatmaps ----
        print("\n[analyze] Generating LBS joint weight visualizations...")
        J = first_W_lbs_np.shape[1]

        # Global vmax across all J joints for consistent brightness mapping
        global_vmax_lbs = float(first_W_lbs_np.max())

        for j in tqdm(range(J), desc="LBS joints", ncols=80):
            vis_mesh_key_weight(
                verts=first_template_np, faces=first_faces_np,
                key_weight=first_W_lbs_np, cage_idx=j,
                cmap="viridis", vmin=0.0, vmax=global_vmax_lbs,
                view_yrots=(0, 90, 180, 270),
                save_path=os.path.join(lbs_w_dir, f"joint_{j:03d}.png"),
                title=f"Joint {j} weight",
                overlay_pos_3d=first_C_lbs_np[j],
            )

        # Grid summary
        joint_imgs = sorted(glob.glob(os.path.join(lbs_w_dir, "joint_*.png")))
        make_grid_from_images(joint_imgs, os.path.join(lbs_w_dir, "lbs_joint_weight_grid.png"), ncols=8)

        # Argmax overview: which joint dominates each face region (mirrors cbd_cage_argmax_topk)
        vis_mesh_all_cage_weights(
            verts=first_template_np, faces=first_faces_np,
            key_weight=first_W_lbs_np,
            cage_indices=list(range(J)),
            mode="argmax",
            view_yrots=(0, 90, 180, 270),
        )
        plt.savefig(os.path.join(lbs_w_dir, "lbs_joint_argmax.png"), dpi=150, bbox_inches="tight")
        plt.close("all")

        # Joint positions overlay
        self._vis_joint_positions(
            verts=first_template_np, faces=first_faces_np,
            C_lbs=first_C_lbs_np,
            save_path=os.path.join(lbs_w_dir, "lbs_joint_positions.png"),
        )
        print(f"[analyze] LBS weights → {lbs_w_dir}")

        # ---- (2) CBD cage vertex weight heatmaps (v8 only) ----
        if first_kw_np is not None and len(key_d_norms_list) > 0:
            print("\n[analyze] Generating CBD cage weight visualizations...")
            key_d_mean_norms = np.stack(key_d_norms_list, axis=0).mean(axis=0)  # (M,)
            top_k   = min(self.opts.num_top_cage, first_kw_np.shape[1])
            top_idx = np.argsort(key_d_mean_norms)[::-1][:top_k]

            np.save(os.path.join(cbd_w_dir, "top_k_cage_indices.npy"), top_idx)
            np.save(os.path.join(cbd_w_dir, "key_d_mean_norms.npy"),   key_d_mean_norms)

            # Global vmax across all M cage vertices for consistent brightness mapping
            global_vmax_cbd = float(first_kw_np.max())

            # Precompute implied position for every cage vertex (weighted centroid of template)
            w_kw = first_kw_np  # (N, M)
            w_sum = np.maximum(w_kw.sum(0), 1e-8)  # (M,)
            cage_implied_pos = (w_kw.T @ first_template_np) / w_sum[:, None]  # (M, 3)

            for rank, m in enumerate(tqdm(top_idx, desc="CBD cages", ncols=80)):
                vis_mesh_key_weight(
                    verts=first_template_np, faces=first_faces_np,
                    key_weight=first_kw_np, cage_idx=int(m),
                    cmap="magma", vmin=0.0, vmax=global_vmax_cbd,
                    view_yrots=(0, 90, 180, 270),
                    save_path=os.path.join(cbd_w_dir, f"cage_{rank:03d}_m{m:03d}.png"),
                    title=f"Cage vertex {rank} (m{m}) weight",
                    overlay_pos_3d=cage_implied_pos[m],
                )

            # Argmax overview
            vis_mesh_all_cage_weights(
                verts=first_template_np, faces=first_faces_np,
                key_weight=first_kw_np,
                cage_indices=top_idx.tolist(),
                mode="argmax",
                view_yrots=(0, 90, 180, 270),
            )
            plt.savefig(os.path.join(cbd_w_dir, "cbd_cage_argmax_topk.png"), dpi=150, bbox_inches="tight")
            plt.close("all")

            # Grid summary
            cage_imgs = sorted(glob.glob(os.path.join(cbd_w_dir, "cage_*.png")))
            make_grid_from_images(cage_imgs, os.path.join(cbd_w_dir, "cbd_cage_weight_grid.png"), ncols=8)
            print(f"[analyze] CBD weights → {cbd_w_dir}")
        else:
            print("\n[analyze] Skipping CBD weight visualizations (v6: LBS only)")

        # ---- (3) Residual videos ----
        if not self.opts.weight_vis_only:
            for suffix in ["lbs", "full", "cbd_contrib"]:
                images_to_video_cv(
                    img_dir=res_dir,
                    out_path=os.path.join(res_dir, f"video_residual_{suffix}.mp4"),
                    fps=30,
                    pattern=f"frame_*_{suffix}.png",
                )

        # ---- (4) MSE CSV & plot ----
        print("\n[analyze] Saving MSE analysis...")
        fieldnames = [
            "frame_idx", "mse_full", "mse_lbs",
            "mse_full_centered", "mse_lbs_centered", "global_translation_magnitude",
        ]
        csv_path = os.path.join(mse_dir, "mse_per_frame.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mse_records)

        mse_full_arr  = np.array([r["mse_full"]                     for r in mse_records])
        mse_lbs_arr   = np.array([r["mse_lbs"]                      for r in mse_records])
        mse_full_c    = np.array([r["mse_full_centered"]             for r in mse_records])
        mse_lbs_c     = np.array([r["mse_lbs_centered"]              for r in mse_records])
        gt_arr        = np.array([r["global_translation_magnitude"]  for r in mse_records])
        frames        = [r["frame_idx"] for r in mse_records]

        print(f"\n{'='*60}")
        print(f"MSE Summary  (N={len(mse_records)} frames)")
        print(f"  MSE Full (LBS+CBD):     {mse_full_arr.mean():.6e}  ±  {mse_full_arr.std():.6e}")
        print(f"  MSE LBS-only:           {mse_lbs_arr.mean():.6e}  ±  {mse_lbs_arr.std():.6e}")
        print(f"  MSE Full  (centered):   {mse_full_c.mean():.6e}  ±  {mse_full_c.std():.6e}")
        print(f"  MSE LBS   (centered):   {mse_lbs_c.mean():.6e}  ±  {mse_lbs_c.std():.6e}")
        print(f"  Global trans mag:       {gt_arr.mean():.6e}  ±  {gt_arr.std():.6e}")
        print(f"{'='*60}\n")

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        axes[0].plot(frames, mse_full_arr,  label="Full LBS+CBD",           color="royalblue",   lw=1.5)
        axes[0].plot(frames, mse_lbs_arr,   label="LBS-only",               color="tomato",      lw=1.5)
        axes[0].plot(frames, mse_full_c,    label="Full LBS+CBD (centered)", color="royalblue",   lw=1, linestyle="--", alpha=0.7)
        axes[0].plot(frames, mse_lbs_c,     label="LBS-only (centered)",     color="tomato",      lw=1, linestyle="--", alpha=0.7)
        axes[0].set_ylabel("MSE")
        axes[0].set_title("MSE per frame  (dashed = global-translation-corrected)")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(frames, gt_arr, color="seagreen", lw=1.5)
        axes[1].set_xlabel("Frame index")
        axes[1].set_ylabel("||mean(GT − pred)||")
        axes[1].set_title("Global drift magnitude (if large → systematic global shift in prediction)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(mse_dir, "mse_global_translation.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[analyze] MSE analysis → {mse_dir}")
        print("[analyze] Done!")

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _vis_joint_positions(self, verts, faces, C_lbs, save_path):
        """Render template mesh + joint positions as colored scatter (3 views)."""
        V4 = normalize_homogeneous(verts)
        view_ = translate(0, 0, -4.5)
        proj_ = perspective(55, 1.0, 1.0, 100.0)
        MV = proj_ @ view_

        J4 = normalize_homogeneous(C_lbs)  # (J, 4)

        view_yrots = [0, 90, 180, 270]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        cmap_j = cm.get_cmap("rainbow", len(C_lbs))

        for ax, yrot_deg in zip(axes, view_yrots):
            rot = yrotate(yrot_deg)
            V_mu = np.median(V4, axis=0)

            # Rotate mesh
            V_rot = (V4 - V_mu) @ rot.T + V_mu
            V_proj = V_rot @ MV.T
            V_proj = V_proj[:, :3] / V_proj[:, 3:4]

            VF = V_proj[faces]
            T  = VF[:, :, :2]
            Z  = -VF[:, :, 2].mean(1)
            order = np.argsort(Z)

            # Transparent mesh: face fill very low alpha so interior joints are visible,
            # edges at moderate alpha to preserve silhouette.
            coll = PolyCollection(
                T[order], closed=True, linewidth=0.3,
                facecolors=(0.75, 0.75, 0.75, 0.10),
                edgecolors=(0.40, 0.40, 0.40, 0.35),
            )
            ax.add_collection(coll)

            # Rotate joint positions
            J_rot  = (J4 - V_mu) @ rot.T + V_mu
            J_proj = J_rot @ MV.T
            J_proj = J_proj[:, :3] / J_proj[:, 3:4]

            for j_idx in range(len(C_lbs)):
                ax.scatter(
                    J_proj[j_idx, 0], J_proj[j_idx, 1],
                    c=[cmap_j(j_idx)], s=25, zorder=10, marker="o",
                    linewidths=0,
                )

            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"y={yrot_deg}°")

        plt.suptitle(f"LBS Joint Positions  ({len(C_lbs)} joints)", fontsize=12)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[analyze] Joint positions → {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    opts = Options()
    assert opts.ckpt is not None, "--ckpt is required"

    # Load train_opts.yml from checkpoint (same logic as eval_CBD.py main)
    config_path = os.path.join(opts.ckpt, "train_opts.yml")
    opts_yaml = yaml.load(open(config_path), Loader=yaml.FullLoader)
    opts_yaml_orig = dict(opts_yaml)   # save before override

    # argparse values override yaml (so CLI flags take precedence)
    opts_yaml.update(vars(opts))
    opts = argparse.Namespace(**opts_yaml)

    # Restore model-architecture opts from YAML.
    # store_true argparse flags default to False and would silently override
    # YAML's true values — restore them so the model forward pass matches training.
    _arch_keys = [
        "use_lbs_joint_center", "use_joint_predict", "use_weighted_joint_pos",
        "no_use_translation", "use_hyb_delta_lbs_input", "use_hyb_concat_lbs",
        "use_exp_joint_predict", "use_lbs_t", "use_lbs_R", "use_lbs_bal",
        "use_lbs_ent", "use_lbs_laplacian", "use_hyb_joint_train",
    ]
    for key in _arch_keys:
        if key in opts_yaml_orig:
            setattr(opts, key, opts_yaml_orig[key])

    print(f"[analyze] version={opts.version}  ckpt={opts.ckpt}")
    print(f"[analyze] data_selection={opts.data_selection}  realtest={opts.realtest}")

    trainer = Trainer(opts)
    trainer.analyze()
