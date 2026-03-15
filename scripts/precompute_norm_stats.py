"""
precompute_norm_stats.py — Compute z-score statistics (μ, σ) for strain and displacement
over the training set, for use in normalized DispNet training.

Outputs:
    norm_stats_{strain_mode}_s{smooth_n_iter}.npz containing:
        strain_mean:  [D_strain]  per-channel mean
        strain_std:   [D_strain]  per-channel std
        disp_mean:    [3]         per-channel mean (xyz displacement)
        disp_std:     [3]         per-channel std

Usage:
    python scripts/precompute_norm_stats.py \
        --config config/train.yml \
        --strain_mode principal --smooth_n_iter 16 \
        --use_true_edd --use_data1 \
        --out_dir ./norm_stats
"""
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.mesh_utils import compute_strain_signal, STRAIN_MODE_DIM, taubin_smooth_np
from utils.exp_utils import plateau_hat_points
from dataloader_CBD import CBDDataset, CBDdataSampler, CBD_collate_wrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train.yml")
    parser.add_argument("--strain_mode", type=str, default='principal',
                        choices=['norm', 'norm_trace', 'full', 'principal', 'local'])
    parser.add_argument("--smooth_n_iter", type=int, required=True)
    parser.add_argument("--use_true_edd", action='store_true')
    parser.add_argument("--use_t_mask", action='store_true')
    parser.add_argument("--out_dir", type=str, default="./norm_stats")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)

    # data flags (same as train scripts)
    parser.add_argument("--use_data0", action='store_true')
    parser.add_argument("--use_data1", action='store_true')
    parser.add_argument("--use_data2", action='store_true')
    parser.add_argument("--use_data3", action='store_true')
    parser.add_argument("--data_toggle", action='store_true')

    # compatibility args (not used but needed by CBDDataset)
    parser.add_argument("--version", type=int, default=9)
    parser.add_argument("--use_strain", action='store_true', default=True)
    parser.add_argument("--use_shp_recon", action='store_true', default=False)
    parser.add_argument("--data_rand_trans", action='store_true', default=False)
    parser.add_argument("--data_rand_scale", action='store_true', default=False)
    parser.add_argument("--use_laplacian", action='store_true', default=False)
    parser.add_argument("--use_decimate", action='store_true', default=False)
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--use_data9", action='store_true', default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    strain_dim = STRAIN_MODE_DIM[args.strain_mode]

    # Dataset (training set only — stats must come from train split)
    train_ds = CBDDataset(args, is_train=True, toggle=args.data_toggle)
    train_sampler = CBDdataSampler(train_ds.len_list, args.batch_size,
                                   shuffle=False, balance=False, is_train=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_sampler=train_sampler,
        collate_fn=partial(CBD_collate_wrapper, device='cpu'),
        num_workers=args.num_workers)

    # Running stats (Welford's online algorithm for numerical stability)
    n_strain = 0
    strain_sum = np.zeros(strain_dim, dtype=np.float64)
    strain_sq_sum = np.zeros(strain_dim, dtype=np.float64)

    n_disp = 0
    disp_sum = np.zeros(3, dtype=np.float64)
    disp_sq_sum = np.zeros(3, dtype=np.float64)

    # Cache for neutral detail (true EDD)
    neutral_detail_cache = {}

    print(f"Computing stats: strain_mode={args.strain_mode}, smooth_n_iter={args.smooth_n_iter}")
    print(f"  use_true_edd={args.use_true_edd}, use_t_mask={args.use_t_mask}")
    print(f"  Training samples: {len(train_loader)} batches")

    for batch in tqdm(train_loader, desc="Computing stats"):
        batch = batch.to(device)
        B = batch.vertices.shape[0]

        tmpl_v = batch.template         # [B, V, 3]
        gt_v = batch.vertices           # [B, V, 3]
        smooth_v = batch.smooth_vertices  # [B, V, 3]

        # --- Strain ---
        _nsi = getattr(batch, 'neutral_span_inv', None)
        strain = compute_strain_signal(
            gt_v, tmpl_v, batch.faces,
            mode=args.strain_mode, neutral_span_inv=_nsi)  # [B, V, D]

        # --- Displacement (EDD) ---
        wrinkle = gt_v - smooth_v  # [B, V, 3]
        if args.use_true_edd:
            id_name = batch.id_name
            if id_name not in neutral_detail_cache:
                tmpl_np = tmpl_v[0].cpu().numpy()
                faces_np = batch.faces.cpu().numpy()
                smooth_tmpl_np = taubin_smooth_np(tmpl_np, faces_np, n_iter=args.smooth_n_iter)
                nd = tmpl_np - smooth_tmpl_np
                neutral_detail_cache[id_name] = torch.tensor(nd).float().to(device)
            wrinkle = wrinkle - neutral_detail_cache[id_name].unsqueeze(0)

        # --- Mask (optional) ---
        if args.use_t_mask:
            t_mask = plateau_hat_points(tmpl_v)  # [1, V, 1]
            mask_flat = t_mask[0, :, 0].cpu().numpy()  # [V]
            valid_mask = mask_flat > 1e-6  # only count face-region vertices
        else:
            valid_mask = None

        # --- Accumulate per-vertex stats ---
        strain_np = strain.cpu().numpy()  # [B, V, D]
        wrinkle_np = wrinkle.cpu().numpy()  # [B, V, 3]

        for b in range(B):
            if valid_mask is not None:
                s_vals = strain_np[b, valid_mask, :]  # [V', D]
                d_vals = wrinkle_np[b, valid_mask, :]  # [V', 3]
            else:
                s_vals = strain_np[b]  # [V, D]
                d_vals = wrinkle_np[b]  # [V, 3]

            n_s = s_vals.shape[0]
            strain_sum += s_vals.sum(axis=0)
            strain_sq_sum += (s_vals ** 2).sum(axis=0)
            n_strain += n_s

            n_d = d_vals.shape[0]
            disp_sum += d_vals.sum(axis=0)
            disp_sq_sum += (d_vals ** 2).sum(axis=0)
            n_disp += n_d

    # Compute μ, σ
    strain_mean = (strain_sum / n_strain).astype(np.float32)
    strain_std = np.sqrt(strain_sq_sum / n_strain - strain_mean.astype(np.float64) ** 2).astype(np.float32)
    strain_std = np.maximum(strain_std, 1e-8)  # avoid div-by-zero

    disp_mean = (disp_sum / n_disp).astype(np.float32)
    disp_std = np.sqrt(disp_sq_sum / n_disp - disp_mean.astype(np.float64) ** 2).astype(np.float32)
    disp_std = np.maximum(disp_std, 1e-8)

    print(f"\nStrain stats (per-channel):")
    print(f"  mean: {strain_mean}")
    print(f"  std:  {strain_std}")
    print(f"\nDisplacement stats (per-channel):")
    print(f"  mean: {disp_mean}")
    print(f"  std:  {disp_std}")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    edd_tag = "_trueedd" if args.use_true_edd else ""
    mask_tag = "_masked" if args.use_t_mask else ""
    fname = f"norm_stats_{args.strain_mode}_s{args.smooth_n_iter}{edd_tag}{mask_tag}.npz"
    out_path = os.path.join(args.out_dir, fname)
    np.savez(out_path,
             strain_mean=strain_mean, strain_std=strain_std,
             disp_mean=disp_mean, disp_std=disp_std,
             strain_mode=args.strain_mode,
             smooth_n_iter=args.smooth_n_iter,
             use_true_edd=args.use_true_edd,
             use_t_mask=args.use_t_mask)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
