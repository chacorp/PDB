"""
Smoothing Analysis Script for v10 Pipeline.
Determines optimal Taubin smoothing iterations by analyzing
displacement (GT - smooth_GT) convergence across datasets.

Usage:
    python scripts/smooth_analysis.py --n_samples 300 --data_basedir /data/sihun
"""
import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.mesh_utils import taubin_smooth_np, laplacian_smooth_np
from utils.exp_utils import PCA_holder


def load_dataset_info(data_basedir):
    """Load template meshes, faces, and PCA holders for each dataset."""
    datasets = {}

    # VOCA
    voca_tmpl_path = f"{data_basedir}/VOCA-COMA/voca_templates.pkl"
    voca_std_path = "utils/voca/standardization.npy"
    if os.path.exists(voca_tmpl_path):
        with open(voca_tmpl_path, 'rb') as f:
            voca_mesh = pickle.load(f)
        voca_std = np.load(voca_std_path, allow_pickle=True).item()
        faces = np.array(voca_std['new_f']).astype(np.int32)
        pca_dir = f"{data_basedir}/VOCA-COMA/VOCASET/train"
        pca_files = sorted([f for f in os.listdir(pca_dir) if f.endswith('_pca.npz')])
        holders = []
        templates = []
        id_names = []
        for pf in pca_files:
            id_name = pf.replace('_pca.npz', '')
            if id_name in voca_mesh:
                holders.append(PCA_holder(os.path.join(pca_dir, pf)))
                templates.append(voca_mesh[id_name])
                id_names.append(id_name)
        if holders:
            datasets['VOCA'] = {
                'holders': holders, 'templates': templates,
                'faces': faces, 'id_names': id_names,
            }
            print(f"  VOCA: {len(holders)} identities, V={templates[0].shape[0]}, F={faces.shape[0]}")

    # BIWI
    biwi_tmpl_path = f"{data_basedir}/BIWI_align_deci/templates_align_deci.pkl"
    if os.path.exists(biwi_tmpl_path):
        with open(biwi_tmpl_path, 'rb') as f:
            biwi_mesh = pickle.load(f)
        faces_biwi = np.array(biwi_mesh['face']).astype(np.int32)
        pca_dir = f"{data_basedir}/BIWI_align_deci/train/vertices_npy"
        if not os.path.exists(pca_dir):
            pca_dir = f"{data_basedir}/BIWI_align_deci/train"
        pca_files = sorted([f for f in os.listdir(pca_dir) if f.endswith('_pca.npz')])
        holders = []
        templates = []
        id_names = []
        for pf in pca_files:
            id_name = pf.replace('_pca.npz', '')
            if id_name in biwi_mesh:
                holders.append(PCA_holder(os.path.join(pca_dir, pf)))
                templates.append(biwi_mesh[id_name])
                id_names.append(id_name)
        if holders:
            datasets['BIWI'] = {
                'holders': holders, 'templates': templates,
                'faces': faces_biwi, 'id_names': id_names,
            }
            print(f"  BIWI: {len(holders)} identities, V={templates[0].shape[0]}, F={faces_biwi.shape[0]}")

    # Multiface SEN
    mf_tmpl_path = f"{data_basedir}/multiface_align/mf_templates.pkl"
    mf_std_path = "utils/mf/standardization.npy"
    if os.path.exists(mf_tmpl_path) and os.path.exists(mf_std_path):
        with open(mf_tmpl_path, 'rb') as f:
            mf_mesh = pickle.load(f)
        mf_std = np.load(mf_std_path, allow_pickle=True).item()
        faces_mf = np.array(mf_std['new_f']).astype(np.int32)
        for subset, label in [('SEN', 'MF_SEN'), ('ROM', 'MF_ROM')]:
            pca_dir = f"{data_basedir}/multiface_align/{subset}/train/vertices_npy"
            if not os.path.exists(pca_dir):
                pca_dir = f"{data_basedir}/multiface_align/{subset}/train"
            if not os.path.exists(pca_dir):
                continue
            pca_files = sorted([f for f in os.listdir(pca_dir) if f.endswith('_pca.npz')])
            holders = []
            templates = []
            id_names = []
            for pf in pca_files:
                id_name = pf.replace('_pca.npz', '')
                if id_name in mf_mesh:
                    holders.append(PCA_holder(os.path.join(pca_dir, pf)))
                    templates.append(mf_mesh[id_name])
                    id_names.append(id_name)
            if holders:
                datasets[label] = {
                    'holders': holders, 'templates': templates,
                    'faces': faces_mf, 'id_names': id_names,
                }
                print(f"  {label}: {len(holders)} identities, V={templates[0].shape[0]}, F={faces_mf.shape[0]}")

    # COMA
    coma_std_path = "utils/voca/standardization.npy"
    coma_tmpl_path = f"{data_basedir}/VOCA-COMA/voca_templates.pkl"
    pca_dir_coma = f"{data_basedir}/VOCA-COMA/COMA/train"
    if os.path.exists(pca_dir_coma) and os.path.exists(coma_tmpl_path):
        with open(coma_tmpl_path, 'rb') as f:
            coma_mesh = pickle.load(f)
        coma_std = np.load(coma_std_path, allow_pickle=True).item()
        faces_coma = np.array(coma_std['new_f']).astype(np.int32)
        pca_files = sorted([f for f in os.listdir(pca_dir_coma) if f.endswith('_pca.npz')])
        holders = []
        templates = []
        id_names = []
        for pf in pca_files:
            id_name = pf.replace('_pca.npz', '')
            if id_name in coma_mesh:
                holders.append(PCA_holder(os.path.join(pca_dir_coma, pf)))
                templates.append(coma_mesh[id_name])
                id_names.append(id_name)
        if holders:
            datasets['COMA'] = {
                'holders': holders, 'templates': templates,
                'faces': faces_coma, 'id_names': id_names,
            }
            print(f"  COMA: {len(holders)} identities, V={templates[0].shape[0]}, F={faces_coma.shape[0]}")

    return datasets


def analyze_smoothing(datasets, n_samples=300, iter_list=None, out_dir='smooth_analysis_output'):
    """
    For each dataset and each n_iter, compute displacement statistics.
    """
    if iter_list is None:
        iter_list = [1, 2, 4, 8, 16, 32, 64]

    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    for ds_name, ds in datasets.items():
        print(f"\n=== Analyzing {ds_name} ===")
        faces = ds['faces']
        results = {n: {'max_disps': [], 'mean_disps': []} for n in iter_list}
        # Store per-sample data for top-K visualization
        sample_data = []  # (disp_magnitude, gt_verts, smooth_verts, template, id_name)

        n_ids = len(ds['holders'])
        samples_per_id = max(1, n_samples // n_ids)

        for i, (holder, template, id_name) in enumerate(zip(ds['holders'], ds['templates'], ds['id_names'])):
            print(f"  [{i+1}/{n_ids}] {id_name}: {samples_per_id} samples", end='')

            for s in range(samples_per_id):
                # Sample from PCA (half at scale 1.0, half at 1.5)
                scale = 1.0 if s < samples_per_id // 2 else 1.5
                gt = holder.sample_from_pca(scale=scale)  # (V, 3)

                for n_iter in iter_list:
                    smooth = taubin_smooth_np(gt, faces, n_iter=n_iter)
                    disp = np.linalg.norm(gt - smooth, axis=-1)  # (V,)
                    results[n_iter]['max_disps'].append(disp.max())
                    results[n_iter]['mean_disps'].append(disp.mean())

                # For top-K: use vis_iter for reference displacement
                vis_iter = 16 if 16 in iter_list else min(iter_list, key=lambda x: abs(x - 16))
                smooth_ref = taubin_smooth_np(gt, faces, n_iter=vis_iter)
                disp_ref = np.linalg.norm(gt - smooth_ref, axis=-1).mean()
                sample_data.append((disp_ref, gt.copy(), smooth_ref.copy(), template, id_name))

            print(f" done")

        all_results[ds_name] = results

        # --- Plot convergence curve ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        means = [np.mean(results[n]['mean_disps']) for n in iter_list]
        maxs = [np.mean(results[n]['max_disps']) for n in iter_list]

        ax1.plot(iter_list, means, 'b-o', label='mean(mean_disp)')
        ax1.plot(iter_list, maxs, 'r-o', label='mean(max_disp)')
        ax1.set_xlabel('Taubin iterations')
        ax1.set_ylabel('Displacement magnitude')
        ax1.set_title(f'{ds_name}: Displacement vs smoothing iterations')
        ax1.legend()
        ax1.grid(True)

        # Rate of change
        if len(iter_list) > 1:
            delta_means = np.diff(means) / np.diff(iter_list)
            end_iters = iter_list[1:]  # rate at each iter (vs previous)
            ax2.plot(end_iters, delta_means, 'g-o')
            ax2.set_xlabel('Taubin iterations')
            ax2.set_ylabel('Δ(mean_disp) / Δ(iter)')
            ax2.set_title(f'{ds_name}: Rate of displacement change')
            ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'convergence_{ds_name}.png'), dpi=150)
        plt.close()
        print(f"  Saved convergence plot: convergence_{ds_name}.png")

        # --- Print summary table ---
        print(f"\n  {'n_iter':>6} | {'mean(mean_disp)':>15} | {'mean(max_disp)':>14} | {'Δ_rate':>10}")
        print(f"  {'-'*6} | {'-'*15} | {'-'*14} | {'-'*10}")
        for j, n in enumerate(iter_list):
            m = np.mean(results[n]['mean_disps'])
            mx = np.mean(results[n]['max_disps'])
            rate = f"{(means[j]-means[j-1])/(iter_list[j]-iter_list[j-1]):.2e}" if j > 0 else "N/A"
            print(f"  {n:>6} | {m:>15.6e} | {mx:>14.6e} | {rate:>10}")

        # --- Top-8 visualization: GT / Taubin n=16 / Laplacian n=1,2,4 (5-row) ---
        sample_data.sort(key=lambda x: -x[0])
        top_k = sample_data[:8]

        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from utils.matplotlib_rnd import plot_image_array
            from PIL import Image, ImageDraw

            gt_verts = [s[1] for s in top_k]
            taubin_verts = [s[2] for s in top_k]  # already computed with n_iter=16
            lap1_verts = [laplacian_smooth_np(g, faces, n_iter=1) for g in gt_verts]
            lap2_verts = [laplacian_smooth_np(g, faces, n_iter=2) for g in gt_verts]
            lap4_verts = [laplacian_smooth_np(g, faces, n_iter=4) for g in gt_verts]
            faces_list = [faces] * len(top_k)

            rows = [
                ('_tmp_gt', gt_verts, "GT"),
                ('_tmp_taubin', taubin_verts, "Taubin\nn=16"),
                ('_tmp_lap1', lap1_verts, "Laplacian\nn=1"),
                ('_tmp_lap2', lap2_verts, "Laplacian\nn=2"),
                ('_tmp_lap4', lap4_verts, "Laplacian\nn=4"),
            ]

            # Render each row
            for prefix, verts, _ in rows:
                plot_image_array(
                    verts, faces_list,
                    size=4, mode='shade', bg_black=False,
                    logdir=out_dir, name=f'{prefix}_{ds_name}', save=True,
                )

            # Combine into 5-row image with labels
            imgs = [Image.open(os.path.join(out_dir, f'{prefix}_{ds_name}.png'))
                    for prefix, _, _ in rows]

            label_w = 100
            w = max(im.width for im in imgs) + label_w
            h = sum(im.height for im in imgs)
            combined = Image.new('RGB', (w, h), (255, 255, 255))

            y_offset = 0
            draw = ImageDraw.Draw(combined)
            for img, (_, _, label) in zip(imgs, rows):
                combined.paste(img, (label_w, y_offset))
                draw.text((5, y_offset + img.height // 2 - 10), label, fill=(0, 0, 0))
                y_offset += img.height

            out_path = os.path.join(out_dir, f'top8_compare_{ds_name}.png')
            combined.save(out_path, dpi=(150, 150))
            print(f"  Saved 5-row comparison: top8_compare_{ds_name}.png")

            # Clean up temp files
            for prefix, _, _ in rows:
                os.remove(os.path.join(out_dir, f'{prefix}_{ds_name}.png'))
        except Exception as e:
            import traceback
            print(f"  Warning: Could not render top-8 visualization: {e}")
            traceback.print_exc()

    return all_results


def compute_mesh_volume(verts, faces):
    """Signed volume of a triangle mesh via divergence theorem."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return np.abs(np.sum(v0 * np.cross(v1, v2)) / 6.0)


def analyze_shrinkage(datasets, iter_list=None, n_samples=30, out_dir='smooth_analysis_output'):
    """Compare volume preservation: Taubin vs uniform Laplacian across iterations."""
    if iter_list is None:
        iter_list = [1, 2, 4, 8, 16, 32]

    os.makedirs(out_dir, exist_ok=True)

    for ds_name, ds in datasets.items():
        print(f"\n=== Shrinkage analysis: {ds_name} ===")
        faces = ds['faces']

        n_ids = len(ds['holders'])
        samples_per_id = max(1, n_samples // n_ids)

        taubin_ratios = {n: [] for n in iter_list}
        lap_ratios = {n: [] for n in iter_list}

        for i, (holder, template, id_name) in enumerate(zip(ds['holders'], ds['templates'], ds['id_names'])):
            for s in range(samples_per_id):
                scale = 1.0 if s < samples_per_id // 2 else 1.5
                gt = holder.sample_from_pca(scale=scale)
                vol_gt = compute_mesh_volume(gt.astype(np.float64), faces)
                if vol_gt < 1e-12:
                    continue

                for n_iter in iter_list:
                    # Taubin
                    sm_taubin = taubin_smooth_np(gt, faces, n_iter=n_iter)
                    vol_taubin = compute_mesh_volume(sm_taubin.astype(np.float64), faces)
                    taubin_ratios[n_iter].append(vol_taubin / vol_gt)

                    # Uniform Laplacian
                    sm_lap = laplacian_smooth_np(gt, faces, n_iter=n_iter)
                    vol_lap = compute_mesh_volume(sm_lap.astype(np.float64), faces)
                    lap_ratios[n_iter].append(vol_lap / vol_gt)

            print(f"  [{i+1}/{n_ids}] {id_name} done")

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        taubin_means = [np.mean(taubin_ratios[n]) for n in iter_list]
        lap_means = [np.mean(lap_ratios[n]) for n in iter_list]
        taubin_stds = [np.std(taubin_ratios[n]) for n in iter_list]
        lap_stds = [np.std(lap_ratios[n]) for n in iter_list]

        ax.errorbar(iter_list, taubin_means, yerr=taubin_stds, fmt='b-o', capsize=4, label='Taubin')
        ax.errorbar(iter_list, lap_means, yerr=lap_stds, fmt='r-o', capsize=4, label='Uniform Laplacian')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Original volume')
        ax.set_xlabel('Smoothing iterations')
        ax.set_ylabel('Volume ratio (smoothed / original)')
        ax.set_title(f'{ds_name}: Volume preservation — Taubin vs Laplacian')
        ax.legend()
        ax.grid(True)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'shrinkage_{ds_name}.png'), dpi=150)
        plt.close()

        # Print table
        print(f"\n  {'n_iter':>6} | {'Taubin vol_ratio':>16} | {'Laplacian vol_ratio':>19}")
        print(f"  {'-'*6} | {'-'*16} | {'-'*19}")
        for j, n in enumerate(iter_list):
            print(f"  {n:>6} | {taubin_means[j]:>16.6f} | {lap_means[j]:>19.6f}")

        print(f"  Saved: shrinkage_{ds_name}.png")


def main():
    parser = argparse.ArgumentParser(description='Smoothing analysis for v10 pipeline')
    parser.add_argument('--data_basedir', type=str, default='/data/sihun')
    parser.add_argument('--n_samples', type=int, default=300,
                        help='Number of PCA samples per dataset (split across identities)')
    parser.add_argument('--out_dir', type=str, default='smooth_analysis_output')
    parser.add_argument('--iters', type=str, default='1,2,4,8,16,32,64',
                        help='Comma-separated list of smoothing iterations to test')
    parser.add_argument('--shrinkage_only', action='store_true',
                        help='Only run shrinkage analysis (skip convergence)')
    args = parser.parse_args()

    iter_list = [int(x) for x in args.iters.split(',')]

    print("Loading datasets...")
    datasets = load_dataset_info(args.data_basedir)

    if not datasets:
        print("ERROR: No datasets found. Check --data_basedir path.")
        return

    if args.shrinkage_only:
        shrink_iters = [n for n in iter_list if n <= 32]  # skip 64 (unstable)
        analyze_shrinkage(datasets, iter_list=shrink_iters, n_samples=30, out_dir=args.out_dir)
    else:
        print(f"\nRunning analysis with {args.n_samples} samples per dataset, iters={iter_list}")
        analyze_smoothing(datasets, n_samples=args.n_samples, iter_list=iter_list, out_dir=args.out_dir)

    print("\nDone! Check output in:", args.out_dir)


if __name__ == '__main__':
    main()
