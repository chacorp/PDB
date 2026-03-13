"""
Visualize all strain modes with true EDD (expression-dependent detail).
Layout: GT | smoothGT | naive_EDD | true_EDD | masked_trueEDD | sGT+strain_ch0 | ...

true_EDD = (GT - smooth(GT)) - (template - smooth(template))
         = (I-S)(GT - template)   i.e. high-pass of expression displacement
masked_trueEDD = true_EDD * t_mask   (face region only)

Usage:
    python scripts/vis_strain_modes.py --smooth_n_iter 16
"""
import os
import sys
import argparse
import numpy as np
import torch
import yaml
import pickle
import shutil
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.mesh_utils import compute_strain_signal, taubin_smooth_np
from utils.matplotlib_rnd import vis_mesh_key_weight
from utils.exp_utils import plateau_hat_points

STRAIN_MODES = ['norm', 'norm_trace', 'full', 'principal', 'local']

MODE_CHANNELS = {
    'norm':       ['||E||_F'],
    'norm_trace': ['||E||_F', 'trace(E)'],
    'full':       ['E_xx', 'E_yy', 'E_zz', 'E_xy', 'E_xz', 'E_yz'],
    'principal':  ['λ1(max)', 'λ2(mid)', 'λ3(min)'],
    'local':      ['self_norm', 'neigh_mean', 'neigh_std'],
}


def load_vis_frames(config_path, data_basedir='/data/sihun'):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    with open(f"{data_basedir}/multiface_align/mf_templates.pkl", 'rb') as f:
        mf_mesh = pickle.load(f)
    mf_std = np.load("utils/mf/standardization.npy", allow_pickle=True).item()
    mf_faces = np.array(mf_std['new_f']).astype(np.int32)

    entries = cfg.get('frames', [])
    frames = []
    for entry in entries:
        if any('FILL' in str(v) for v in entry.values()):
            continue
        dataset = entry['dataset']
        split = entry['split']
        id_name = entry['id_name']
        seq = entry['sequence']
        frame_idx = entry['frame']

        ds_dir = dataset.replace('mf_', '')
        npy_dir = f"{data_basedir}/multiface_align/{ds_dir}/{split}/vertices_npy/{id_name}/{seq}"
        npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])
        if frame_idx >= len(npy_files):
            continue
        verts = np.load(os.path.join(npy_dir, npy_files[frame_idx])).astype(np.float32)
        template = np.array(mf_mesh[id_name]).astype(np.float32)

        frames.append({
            'verts': verts,
            'template': template,
            'faces': mf_faces,
            'label': f"{seq}_f{frame_idx}",
        })
    return frames


def render_panel(verts_np, faces_np, vals, cmap, title, save_path):
    """Single panel render using vis_mesh_key_weight."""
    has_neg = vals.min() < -1e-8
    if has_neg:
        vmax = max(abs(vals.max()), abs(vals.min()), 1e-6)
        vmin = -vmax
    else:
        vmax = max(float(np.percentile(vals, 95)), 1e-6)
        vmin = 0

    vis_mesh_key_weight(
        verts_np, faces_np, vals[:, None], cage_idx=0,
        cmap=cmap, vmin=vmin, vmax=vmax,
        view_yrots=(0,),
        save_path=save_path, close=True, title=title,
        shade=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smooth_n_iter', type=int, default=16)
    parser.add_argument('--config', type=str, default='config/vis_frames.yml')
    parser.add_argument('--out_dir', type=str, default='strain_option_vis/true_EDD_vis')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading vis frames from {args.config}...")
    frames = load_vis_frames(args.config)
    print(f"Loaded {len(frames)} frames (all).")

    # Precompute smooth templates (per identity, only once)
    smooth_template_cache = {}
    # Precompute t_mask per identity (from template positions)
    tmask_cache = {}

    for mode in STRAIN_MODES:
        mode_dir = os.path.join(args.out_dir, mode)
        tmp_dir = os.path.join(mode_dir, '_tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        channels = MODE_CHANNELS[mode]
        print(f"\n=== {mode} ({len(channels)}ch: {channels}) ===")

        for fi, frame in enumerate(frames):
            verts_np = frame['verts']
            template_np = frame['template']
            faces_np = frame['faces']
            label = frame['label']

            # Smooth GT
            smooth_np = taubin_smooth_np(verts_np, faces_np, n_iter=args.smooth_n_iter)

            # Smooth template (cached per identity via template pointer)
            tmpl_key = id(frame['template'])
            if tmpl_key not in smooth_template_cache:
                smooth_template_cache[tmpl_key] = taubin_smooth_np(
                    template_np, faces_np, n_iter=args.smooth_n_iter)
            smooth_template_np = smooth_template_cache[tmpl_key]

            # t_mask (cached per identity)
            if tmpl_key not in tmask_cache:
                tmask_t = plateau_hat_points(
                    torch.tensor(template_np).float().to(device))  # [V, 1]
                tmask_cache[tmpl_key] = tmask_t.squeeze(-1).cpu().numpy()  # [V]
            t_mask = tmask_cache[tmpl_key]

            # Naive EDD: GT - smooth(GT)
            naive_edd = verts_np - smooth_np                          # [V, 3]
            naive_edd_mag = np.linalg.norm(naive_edd, axis=-1)        # [V]

            # Neutral detail: template - smooth(template)
            neutral_detail = template_np - smooth_template_np         # [V, 3]

            # True EDD: naive_edd - neutral_detail = (I-S)(GT - T)
            true_edd = naive_edd - neutral_detail                     # [V, 3]
            true_edd_mag = np.linalg.norm(true_edd, axis=-1)          # [V]

            # Masked true EDD: face region only
            masked_true_edd_mag = true_edd_mag * t_mask               # [V]

            # Compute strain on smooth GT
            smooth_t = torch.tensor(smooth_np).float().unsqueeze(0).to(device)
            template_t = torch.tensor(template_np).float().unsqueeze(0).to(device)
            faces_t = torch.tensor(faces_np).long().to(device)
            sm_strain = compute_strain_signal(smooth_t, template_t, faces_t, mode=mode)
            sm_np = sm_strain[0].cpu().numpy()  # [V, D]

            # === Panels ===
            panels = []

            # 1) GT mesh (plain)
            p = os.path.join(tmp_dir, f'{label}_gt.png')
            render_panel(verts_np, faces_np, np.zeros(verts_np.shape[0]), 'YlOrRd', 'GT', p)
            panels.append(Image.open(p))

            # 2) smooth GT mesh (plain)
            p = os.path.join(tmp_dir, f'{label}_sgt.png')
            render_panel(smooth_np, faces_np, np.zeros(smooth_np.shape[0]), 'YlOrRd', 'smoothGT', p)
            panels.append(Image.open(p))

            # 3) naive EDD: |GT - smoothGT|
            p = os.path.join(tmp_dir, f'{label}_naive_edd.png')
            render_panel(verts_np, faces_np, naive_edd_mag, 'YlOrRd', 'naive|GT-sGT|', p)
            panels.append(Image.open(p))

            # 4) true EDD: |(GT-sGT) - (T-sT)|
            p = os.path.join(tmp_dir, f'{label}_true_edd.png')
            render_panel(verts_np, faces_np, true_edd_mag, 'YlOrRd', 'trueEDD', p)
            panels.append(Image.open(p))

            # 5) masked true EDD: face region only (t_mask applied)
            p = os.path.join(tmp_dir, f'{label}_masked_edd.png')
            render_panel(verts_np, faces_np, masked_true_edd_mag, 'YlOrRd', 'maskedEDD', p)
            panels.append(Image.open(p))

            # 6+) smooth GT with strain channels
            for ci, ch_name in enumerate(channels):
                sm_vals = sm_np[:, ci]
                has_neg = sm_vals.min() < -1e-8
                cmap = 'coolwarm' if has_neg else 'YlOrRd'

                p = os.path.join(tmp_dir, f'{label}_strain_{ci}.png')
                render_panel(smooth_np, faces_np, sm_vals, cmap, f'sGT:{ch_name}', p)
                panels.append(Image.open(p))

            # Stitch
            total_w = sum(img.width for img in panels)
            max_h = max(img.height for img in panels)
            stitched = Image.new('RGB', (total_w, max_h), (255, 255, 255))
            x = 0
            for img in panels:
                stitched.paste(img, (x, 0))
                x += img.width

            out_path = os.path.join(mode_dir, f'{label}.png')
            stitched.save(out_path)
            print(f"  [{fi+1}/{len(frames)}] {label} → {out_path}")

        # Cleanup tmp
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    print(f"\nDone! Results in {args.out_dir}/")


if __name__ == '__main__':
    main()
