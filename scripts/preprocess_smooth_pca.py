"""
Preprocess smooth PCA basis for v10 pipeline.
For each identity's PCA npz, applies the Taubin smoothing operator S
to mean_ and components_ and saves as {id}_smooth_pca.npz.

Usage:
    python scripts/preprocess_smooth_pca.py --n_iter 16 --data_basedir /data/sihun
"""
import os
import sys
import argparse
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.mesh_utils import build_smooth_operator


def process_pca_dataset(name, pca_dir, template_dict, faces, n_iter,
                        lambda_factor, mu_factor):
    """Process all PCA npz files in a directory."""
    pca_files = sorted([f for f in os.listdir(pca_dir) if f.endswith('_pca.npz')])
    if not pca_files:
        print(f"  [{name}] No PCA files found in {pca_dir}")
        return 0

    count = 0
    for pf in pca_files:
        id_name = pf.replace('_pca.npz', '')
        if id_name not in template_dict:
            print(f"  [{name}] Skipping {id_name} — no template found")
            continue

        template_verts = np.array(template_dict[id_name]).astype(np.float64)
        V = template_verts.shape[0]

        # Build smoothing operator from template mesh
        S = build_smooth_operator(template_verts, faces, n_iter=n_iter,
                                  lambda_factor=lambda_factor, mu_factor=mu_factor)

        # Load original PCA
        npz_path = os.path.join(pca_dir, pf)
        data = np.load(npz_path)
        mean_ = data['mean_']              # (V*3,)
        components_ = data['components_']  # (n_comp, V*3)
        explained_variance_ = data['explained_variance_']

        # Apply S to mean: reshape (V*3,) → (V, 3), apply S per-column, flatten back
        mean_3d = mean_.reshape(V, 3)
        smooth_mean_3d = (S @ mean_3d).astype(np.float32)  # S is (V,V), mean is (V,3)
        smooth_mean = smooth_mean_3d.reshape(-1)

        # Apply S to each component
        n_comp = components_.shape[0]
        smooth_components = np.zeros_like(components_)
        for c in range(n_comp):
            comp_3d = components_[c].reshape(V, 3)
            smooth_comp_3d = (S @ comp_3d).astype(np.float32)
            smooth_components[c] = smooth_comp_3d.reshape(-1)

        # Save smooth PCA npz alongside original (include n_iter in filename)
        out_path = os.path.join(pca_dir, f'{id_name}_smooth{n_iter}_pca.npz')
        np.savez(out_path,
                 mean_=smooth_mean,
                 components_=smooth_components,
                 explained_variance_=explained_variance_)
        count += 1

    print(f"  [{name}] Processed {count} identities → saved *_smooth{n_iter}_pca.npz in {pca_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description='Preprocess smooth PCA basis for v10')
    parser.add_argument('--n_iter', type=int, required=True,
                        help='Number of Taubin smoothing iterations (determined from smooth_analysis.py)')
    parser.add_argument('--data_basedir', type=str, default='/data/sihun')
    parser.add_argument('--lambda_factor', type=float, default=0.5)
    parser.add_argument('--mu_factor', type=float, default=-0.53)
    args = parser.parse_args()

    print(f"Preprocessing smooth PCA basis with n_iter={args.n_iter}, "
          f"lambda={args.lambda_factor}, mu={args.mu_factor}")
    print(f"Data basedir: {args.data_basedir}\n")

    total = 0

    # --- VOCA ---
    voca_tmpl_path = f"{args.data_basedir}/VOCA-COMA/voca_templates.pkl"
    voca_std_path = "utils/voca/standardization.npy"
    if os.path.exists(voca_tmpl_path):
        with open(voca_tmpl_path, 'rb') as f:
            voca_mesh = pickle.load(f)
        voca_std = np.load(voca_std_path, allow_pickle=True).item()
        faces = np.array(voca_std['new_f']).astype(np.int32)
        for mode in ['train', 'val', 'test']:
            pca_dir = f"{args.data_basedir}/VOCA-COMA/VOCASET/{mode}"
            if os.path.exists(pca_dir):
                total += process_pca_dataset(f'VOCA-{mode}', pca_dir, voca_mesh, faces,
                                             args.n_iter, args.lambda_factor, args.mu_factor)

    # --- COMA ---
    for mode in ['train', 'val', 'test']:
        pca_dir = f"{args.data_basedir}/VOCA-COMA/COMA/{mode}"
        if os.path.exists(pca_dir) and os.path.exists(voca_tmpl_path):
            total += process_pca_dataset(f'COMA-{mode}', pca_dir, voca_mesh, faces,
                                         args.n_iter, args.lambda_factor, args.mu_factor)

    # --- BIWI ---
    biwi_tmpl_path = f"{args.data_basedir}/BIWI_align_deci/templates_align_deci.pkl"
    if os.path.exists(biwi_tmpl_path):
        with open(biwi_tmpl_path, 'rb') as f:
            biwi_mesh = pickle.load(f)
        faces_biwi = np.array(biwi_mesh['face']).astype(np.int32)
        for mode in ['train', 'val', 'test']:
            for subdir in [f'{mode}/vertices_npy', mode]:
                pca_dir = f"{args.data_basedir}/BIWI_align_deci/{subdir}"
                if os.path.exists(pca_dir):
                    total += process_pca_dataset(f'BIWI-{mode}', pca_dir, biwi_mesh, faces_biwi,
                                                 args.n_iter, args.lambda_factor, args.mu_factor)
                    break

    # --- Multiface SEN + ROM ---
    mf_tmpl_path = f"{args.data_basedir}/multiface_align/mf_templates.pkl"
    mf_std_path = "utils/mf/standardization.npy"
    if os.path.exists(mf_tmpl_path) and os.path.exists(mf_std_path):
        with open(mf_tmpl_path, 'rb') as f:
            mf_mesh = pickle.load(f)
        mf_std = np.load(mf_std_path, allow_pickle=True).item()
        faces_mf = np.array(mf_std['new_f']).astype(np.int32)
        for subset in ['SEN', 'ROM']:
            for mode in ['train', 'val', 'test']:
                for subdir in [f'{mode}/vertices_npy', mode]:
                    pca_dir = f"{args.data_basedir}/multiface_align/{subset}/{subdir}"
                    if os.path.exists(pca_dir):
                        total += process_pca_dataset(f'MF_{subset}-{mode}', pca_dir, mf_mesh, faces_mf,
                                                     args.n_iter, args.lambda_factor, args.mu_factor)
                        break

    print(f"\nTotal: {total} smooth PCA basis files created.")
    print("Done!")


if __name__ == '__main__':
    main()
