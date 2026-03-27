"""
eval_hlbs.py — Evaluate a trained HierarchicalLBS checkpoint.

Self-retargeting : reconstruction metrics on EvalDataset (real test frames).
Cross-retargeting: transfer source expression to a different target identity.

Metrics (self-retarget only):
    - MSE (overall / inner / outer face regions)
    - Per-vertex L2 (mean, median, p95, p99, max, max_mean)
    - Laplacian smoothness error  [Neural Cages, Yifan et al. CVPR 2020]
    - Normal consistency          [Occupancy Networks, Mescheder et al. CVPR 2019]
    - Edge length distortion      [ARAP, Sorkine & Alexa SGP 2007]

Usage (self-retarget):
    python eval_hlbs.py \
        --ckpt ckpts_hlbs/2025-...-HLBS-mf-s16 \
        --rig_path utils/mf/rig_info.json \
        --use_data1 --data_toggle --batch_size 1

Usage (cross-retarget):
    python eval_hlbs.py \
        --ckpt ckpts_hlbs/2025-...-HLBS-mf-s16 \
        --rig_path utils/mf/rig_info.json \
        --use_data1 --data_toggle --batch_size 1 \
        --cross_retarget \
        --tgt_vert_path /data/.../tgt_verts.npy \
        --tgt_norm_path /data/.../tgt_normals.npy \
        --tgt_obj_path  /data/.../tgt_mesh.obj
"""
import os
import sys
import json
import argparse
import glob
import random
import numpy as np
import yaml
import cv2
import trimesh

import torch
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
import igl

sys.path.insert(0, os.path.dirname(__file__))
from utils.matplotlib_rnd import plot_image_array
from dataloader_CBD import EvalDataset, CBD_collate_wrapper_eval
from utils.exp_utils import plateau_hat_points
from utils.mesh_utils import calc_norm_torch


# ── CLI ─────────────────────────────────────────────────────────────────────

def Options():
    parser = argparse.ArgumentParser(description='Evaluate HierarchicalLBS')

    # rig / topology
    parser.add_argument("--rig_path", type=str, required=True)
    parser.add_argument("--topo_key", type=str, default='mf',
                        choices=['mf', 'biwi', 'voca'])
    parser.add_argument("--num_identities", type=int, default=13)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)

    # ablation flags (must match training)
    parser.add_argument("--freeze_adapt", dest='freeze_adapt', action='store_true')
    parser.set_defaults(freeze_adapt=False)
    parser.add_argument("--use_joint_trans", dest='use_joint_trans', action='store_true')
    parser.set_defaults(use_joint_trans=False)

    # target (for smooth_gt computation)
    parser.add_argument("--target", type=str, default='gt',
                        choices=['gt', 'smooth_gt'])
    parser.add_argument("--smooth_n_iter", type=int, default=16)

    # checkpoint
    parser.add_argument("--ckpt", type=str, required=True,
                        help='Checkpoint directory (must contain model_hlbs_*.pth)')
    parser.add_argument("--start_epoch", type=int, default=-1,
                        help='Epoch to load (-1 = best)')

    # data
    parser.add_argument("--data_selection", type=str, default='mf_ROM',
                        choices=['voca', 'biwi', 'mf_SEN', 'coma', 'mf_ROM', 'ict', 'ict-cap'],
                        help='Dataset to evaluate on')
    parser.add_argument("--data_toggle", dest='data_toggle', action='store_true')
    parser.set_defaults(data_toggle=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # mask
    parser.add_argument("--no_t_mask", dest='no_t_mask', action='store_true')
    parser.set_defaults(no_t_mask=False)
    parser.add_argument("--use_t_mask", dest='no_t_mask', action='store_false')

    # cross-retargeting
    parser.add_argument("--cross_retarget", dest='cross_retarget', action='store_true')
    parser.set_defaults(cross_retarget=False)
    parser.add_argument("--tgt_vert_path", type=str, default=None,
                        help='Target neutral vertices .npy [V, 3]')
    parser.add_argument("--tgt_norm_path", type=str, default=None,
                        help='Target neutral normals  .npy [V, 3]')
    parser.add_argument("--tgt_obj_path", type=str, default=None,
                        help='Target neutral mesh     .obj (for face topology)')

    # output
    parser.add_argument("--log_dir", type=str, default="eval_hlbs",
                        help='Base output directory (default: eval_hlbs)')
    parser.add_argument("--save_vert", dest='save_vert', action='store_true')
    parser.set_defaults(save_vert=False)
    parser.add_argument("--save_gt", dest='save_gt', action='store_true')
    parser.set_defaults(save_gt=False)
    parser.add_argument("--save_obj", dest='save_obj', action='store_true')
    parser.set_defaults(save_obj=False)
    parser.add_argument("--make_video", dest='make_video', action='store_true')
    parser.add_argument("--no_video", dest='make_video', action='store_false')
    parser.set_defaults(make_video=True)
    parser.add_argument("--no_vis", dest='no_vis', action='store_true',
                        help='Skip image rendering (metrics only)')
    parser.set_defaults(no_vis=False)
    parser.add_argument("--vis_every", type=int, default=1,
                        help='Render image every N batches (1 = all frames)')

    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


# ── Utilities ───────────────────────────────────────────────────────────────

def images_to_video(img_dir, out_path, fps=30):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if len(imgs) == 0:
        print(f"[WARN] No images in {img_dir}, skipping video.")
        return
    first = cv2.imread(imgs[0])
    h, w, _ = first.shape
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in imgs:
        writer.write(cv2.imread(p))
    writer.release()
    print(f"Video saved: {out_path}")


def _write_obj(path, verts, faces):
    with open(path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# ── Metric helpers ──────────────────────────────────────────────────────────

def _build_cot_laplacian(verts_np, faces_np):
    """Cotangent Laplacian via libigl [Sorkine et al. SGP 2004]."""
    return igl.cotmatrix(verts_np.astype(np.float64), faces_np.astype(np.int32))


def _laplacian_error(L_sp, pred, gt):
    """||L*pred - L*gt||^2 per vertex, averaged [Neural Cages, CVPR 2020]."""
    B = pred.shape[0]
    total = 0.0
    for b in range(B):
        Lp = L_sp @ pred[b].numpy().astype(np.float64)
        Lg = L_sp @ gt[b].numpy().astype(np.float64)
        total += float(np.mean(np.sum((Lp - Lg) ** 2, axis=-1)))
    return total / B


def _normal_consistency(pred, gt, faces):
    """1 - cos(n_pred, n_gt), averaged [Occupancy Networks, CVPR 2019]."""
    pred_n = calc_norm_torch(pred, faces)
    gt_n   = calc_norm_torch(gt, faces)
    cos_sim = F.cosine_similarity(pred_n, gt_n, dim=-1)
    return (1.0 - cos_sim).mean().item()


def _edge_length_distortion(pred, gt, edges):
    """mean |e_pred - e_gt| / e_gt  [ARAP, Sorkine & Alexa SGP 2007]."""
    e0, e1 = edges[:, 0], edges[:, 1]
    len_pred = torch.sqrt(((pred[:, e0] - pred[:, e1]) ** 2).sum(dim=-1))
    len_gt   = torch.sqrt(((gt[:, e0]   - gt[:, e1])   ** 2).sum(dim=-1))
    return ((len_pred - len_gt).abs() / (len_gt + 1e-8)).mean().item()


def _build_edges(faces_np):
    """Unique undirected edges from face array [F, 3]."""
    e = np.concatenate([faces_np[:, [0,1]], faces_np[:, [1,2]], faces_np[:, [0,2]]], axis=0)
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    return torch.tensor(e, dtype=torch.long)


# ── Evaluator ───────────────────────────────────────────────────────────────

class HLBSEvaluator:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device(opts.device if torch.cuda.is_available() else 'cpu')

        torch.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)

        # ── Load model ──────────────────────────────────────────────────
        from utils.rig_loader import load_rig
        from models.hierarchical_lbs import HierarchicalLBS

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
        ).to(self.device)

        # resolve checkpoint
        if opts.start_epoch < 0:
            ckpt_path = os.path.join(opts.ckpt, "model_hlbs_best.pth")
        else:
            ckpt_path = os.path.join(opts.ckpt, f"model_hlbs_{opts.start_epoch:03d}.pth")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()
        print(f"Loaded: {ckpt_path}")

        # ── Output dir ──────────────────────────────────────────────────
        # eval_hlbs/{ckpt_basename}-eval/e050-self/mf_ROM/
        epoch_tag = "best" if opts.start_epoch < 0 else f"{opts.start_epoch:03d}"
        ckpt_basename = os.path.basename(os.path.normpath(opts.ckpt))
        mode_tag = "cross" if opts.cross_retarget else "self"
        self.out_dir = os.path.join(
            opts.log_dir, f"{ckpt_basename}-eval",
            f"e{epoch_tag}-{mode_tag}", opts.data_selection)
        os.makedirs(self.out_dir, exist_ok=True)

        self.img_dir = os.path.join(self.out_dir, "img")
        os.makedirs(self.img_dir, exist_ok=True)

        if opts.save_vert:
            self.vert_dir = os.path.join(self.out_dir, "pred_vert")
            os.makedirs(self.vert_dir, exist_ok=True)
        if opts.save_gt:
            self.gt_dir = os.path.join(self.out_dir, "gt_vert")
            os.makedirs(self.gt_dir, exist_ok=True)

        # ── Load target mesh for cross-retarget ─────────────────────────
        self.tgt_neu_vert = None
        self.tgt_neu_norm = None
        self.tgt_faces = None
        if opts.cross_retarget:
            assert opts.tgt_vert_path and opts.tgt_norm_path and opts.tgt_obj_path, \
                "Cross-retarget requires --tgt_vert_path, --tgt_norm_path, --tgt_obj_path"
            self.tgt_neu_vert = torch.from_numpy(
                np.load(opts.tgt_vert_path)).float().unsqueeze(0).to(self.device)
            self.tgt_neu_norm = torch.from_numpy(
                np.load(opts.tgt_norm_path)).float().unsqueeze(0).to(self.device)
            tgt_mesh = trimesh.load(opts.tgt_obj_path, process=False)
            if isinstance(tgt_mesh, trimesh.Scene):
                tgt_mesh = trimesh.util.concatenate(tuple(tgt_mesh.geometry.values()))
            self.tgt_faces = tgt_mesh.faces
            self.tgt_faces_torch = torch.from_numpy(
                self.tgt_faces).long().unsqueeze(0).to(self.device)
            print(f"Target mesh: {self.tgt_neu_vert.shape[1]} verts, "
                  f"{self.tgt_faces.shape[0]} faces")

    # ── Self-retargeting evaluation ─────────────────────────────────────

    def evaluate_self(self):
        opts = self.opts

        dataset = EvalDataset(data_name=opts.data_selection, toggle=opts.data_toggle)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opts.batch_size,
            collate_fn=partial(CBD_collate_wrapper_eval, device='cpu'),
            num_workers=0,
        )

        # lazily built from first batch
        L_sp = None
        edges = None

        # accumulators
        total = {
            "mse": 0.0, "mse_in": 0.0, "mse_out": 0.0,
            "l2": 0.0, "lap": 0.0, "norm_cos": 0.0, "edge_dist": 0.0,
            "l2_max_sum": 0.0,
        }
        n_batches = 0
        all_per_vertex_l2 = []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), ncols=120,
                    desc="Eval HLBS [self]")

        with torch.no_grad():
            for idx, batch in pbar:
                batch = batch.to(self.device)
                B = batch.vertices.shape[0]

                src_v = batch.template
                src_n = batch.template_normal
                gt_v  = batch.vertices
                gt_n  = batch.vertices_normal

                # forward
                delta     = gt_v - src_v
                src_in    = torch.cat([src_v, src_n], dim=-1)
                deform_in = torch.cat([delta, gt_n, src_in], dim=-1)
                pred_lbs  = self.model(src_v, deform_in)

                # build mesh operators once
                if L_sp is None:
                    f_np = batch.faces[0].cpu().numpy()
                    v_np = src_v[0].cpu().numpy()
                    L_sp = _build_cot_laplacian(v_np, f_np)
                    edges = _build_edges(f_np).to(self.device)

                # ── compute metrics ─────────────────────────────────────
                mse = F.mse_loss(gt_v, pred_lbs).item()
                pv_l2 = torch.sqrt(((gt_v - pred_lbs) ** 2).sum(dim=-1))  # [B, N]

                total["mse"]  += mse
                total["l2"]   += pv_l2.mean().item()
                total["l2_max_sum"] += pv_l2.max(dim=-1).values.mean().item()
                total["lap"]       += _laplacian_error(L_sp, pred_lbs.cpu(), gt_v.cpu())
                total["norm_cos"]  += _normal_consistency(pred_lbs, gt_v, batch.faces)
                total["edge_dist"] += _edge_length_distortion(pred_lbs, gt_v, edges)
                all_per_vertex_l2.append(pv_l2.cpu())

                if not opts.no_t_mask:
                    t_mask   = plateau_hat_points(src_v)
                    inv_mask = 1.0 - t_mask
                    total["mse_in"]  += F.mse_loss(gt_v * t_mask,   pred_lbs * t_mask).item()
                    total["mse_out"] += F.mse_loss(gt_v * inv_mask,  pred_lbs * inv_mask).item()

                n_batches += 1
                pbar.set_description(
                    f"MSE:{mse:.4e} L2:{pv_l2.mean():.4f} Lap:{total['lap']/n_batches:.4e}")

                # save
                self._save_outputs(idx, batch, pred_lbs, gt_v)

                # visualize
                if not opts.no_vis and idx % opts.vis_every == 0:
                    self._render_self(idx, batch, pred_lbs)

        # ── aggregate ───────────────────────────────────────────────────
        inv = 1.0 / max(n_batches, 1)
        results = {
            "MSE":                 total["mse"] * inv,
            "L2_mean":             total["l2"]  * inv,
            "L2_max_mean":         total["l2_max_sum"] * inv,
            "Laplacian_err":       total["lap"] * inv,
            "Normal_cos_dist":     total["norm_cos"] * inv,
            "Edge_len_distortion": total["edge_dist"] * inv,
        }
        if not opts.no_t_mask:
            results["MSE_inner"] = total["mse_in"]  * inv
            results["MSE_outer"] = total["mse_out"] * inv

        all_pv = torch.cat(all_per_vertex_l2, dim=0).numpy()
        pv_flat = all_pv.flatten()
        results["L2_median"] = float(np.median(pv_flat))
        results["L2_p95"]    = float(np.percentile(pv_flat, 95))
        results["L2_p99"]    = float(np.percentile(pv_flat, 99))
        results["L2_max"]    = float(np.max(pv_flat))
        results["num_frames"] = int(all_pv.shape[0])

        self._print_and_save(results, all_pv)
        return results

    # ── Cross-retargeting evaluation ────────────────────────────────────

    def evaluate_cross(self):
        opts = self.opts

        dataset = EvalDataset(data_name=opts.data_selection, toggle=opts.data_toggle)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opts.batch_size,
            collate_fn=partial(CBD_collate_wrapper_eval, device='cpu'),
            num_workers=0,
        )

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), ncols=120,
                    desc="Eval HLBS [cross]")

        with torch.no_grad():
            for idx, batch in pbar:
                batch = batch.to(self.device)
                B = batch.vertices.shape[0]

                # source expression data
                src_v = batch.template
                src_n = batch.template_normal
                gt_v  = batch.vertices
                gt_n  = batch.vertices_normal

                # expand target to batch
                tgt_v = self.tgt_neu_vert.expand(B, -1, -1)
                tgt_n = self.tgt_neu_norm.expand(B, -1, -1)

                pred_lbs = self.model.retarget(
                    src_v, src_n, gt_v, gt_n, tgt_v, tgt_n)

                # save predicted vertices
                if opts.save_vert:
                    for b in range(B):
                        np.save(
                            os.path.join(self.vert_dir, f"{idx * opts.batch_size + b:06d}.npy"),
                            pred_lbs[b].cpu().numpy())

                # save OBJ
                if opts.save_obj:
                    obj_dir = os.path.join(self.out_dir, "obj")
                    os.makedirs(obj_dir, exist_ok=True)
                    for b in range(B):
                        fid = idx * opts.batch_size + b
                        _write_obj(
                            os.path.join(obj_dir, f"pred_{fid:06d}.obj"),
                            pred_lbs[b].cpu().numpy(), self.tgt_faces)

                # visualize: source GT | target neutral | retargeted
                if not opts.no_vis and idx % opts.vis_every == 0:
                    src_faces_cpu = batch.faces[0].cpu()
                    tgt_faces_cpu = self.tgt_faces_torch[0].cpu()
                    v_list = [
                        gt_v[0].cpu(),           # source expression
                        tgt_v[0].cpu(),           # target neutral
                        pred_lbs[0].cpu(),        # retargeted
                    ]
                    f_list = [src_faces_cpu, tgt_faces_cpu, tgt_faces_cpu]
                    plot_image_array(
                        v_list, f_list,
                        rot_list=[[0, 0, 0]] * len(v_list),
                        size=1, bg_black=False, mode='shade',
                        logdir=self.img_dir,
                        name=f"{idx:06d}", save=True)

        print(f"\nCross-retarget done. Outputs: {self.out_dir}")

        if opts.make_video:
            images_to_video(self.img_dir, os.path.join(self.out_dir, "eval_hlbs_cross.mp4"))

    # ── Shared helpers ──────────────────────────────────────────────────

    def _save_outputs(self, idx, batch, pred_lbs, gt_v):
        opts = self.opts
        B = pred_lbs.shape[0]
        if opts.save_vert:
            for b in range(B):
                np.save(
                    os.path.join(self.vert_dir, f"{idx * opts.batch_size + b:06d}.npy"),
                    pred_lbs[b].cpu().numpy())
        if opts.save_gt:
            for b in range(B):
                np.save(
                    os.path.join(self.gt_dir, f"{idx * opts.batch_size + b:06d}.npy"),
                    gt_v[b].cpu().numpy())
        if opts.save_obj:
            obj_dir = os.path.join(self.out_dir, "obj")
            os.makedirs(obj_dir, exist_ok=True)
            faces_np = batch.faces[0].cpu().numpy()
            for b in range(B):
                fid = idx * opts.batch_size + b
                _write_obj(
                    os.path.join(obj_dir, f"pred_{fid:06d}.obj"),
                    pred_lbs[b].cpu().numpy(), faces_np)

    def _render_self(self, idx, batch, pred_lbs):
        faces_cpu = batch.faces[0].cpu()
        v_list = [
            batch.vertices[0].cpu(),    # GT
            batch.template[0].cpu(),    # template
            pred_lbs[0].cpu(),          # prediction
        ]
        f_list = [faces_cpu] * len(v_list)
        plot_image_array(
            v_list, f_list,
            rot_list=[[0, 0, 0]] * len(v_list),
            size=1, bg_black=False, mode='shade',
            logdir=self.img_dir,
            name=f"{idx:06d}", save=True)

    def _print_and_save(self, results, all_pv):
        print("\n=== HLBS Evaluation Results ===")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k:22s}: {v:.6e}")
            else:
                print(f"  {k:22s}: {v}")
        print("===============================")

        results_path = os.path.join(self.out_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Saved: {results_path}")

        np.save(os.path.join(self.out_dir, "per_vertex_l2.npy"), all_pv)

        if self.opts.make_video:
            images_to_video(self.img_dir, os.path.join(self.out_dir, "eval_hlbs.mp4"))


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    opts = Options()

    # inherit model config from training checkpoint
    train_cfg_path = os.path.join(opts.ckpt, "train_opts.yml")
    if os.path.isfile(train_cfg_path):
        with open(train_cfg_path) as f:
            train_cfg = yaml.safe_load(f)
        for key in ['hid_dim', 'num_layers', 'topo_key', 'freeze_adapt',
                     'use_joint_trans', 'smooth_n_iter', 'target']:
            if key in train_cfg:
                cli_flags = [f'--{key}', f'--{key.replace("_", "-")}']
                if not any(flag in sys.argv for flag in cli_flags):
                    setattr(opts, key, train_cfg[key])
        print(f"Inherited model config from: {train_cfg_path}")

    evaluator = HLBSEvaluator(opts)

    if opts.cross_retarget:
        evaluator.evaluate_cross()
    else:
        evaluator.evaluate_self()
