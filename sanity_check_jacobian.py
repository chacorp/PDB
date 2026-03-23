"""
sanity_check_jacobian.py
========================
vis_frames.yml에 정의된 fixed evaluation frames에 대해
strain (norm_trace, principal) vs Jacobian det(F)-1 를 나란히 시각화.

Output: jacobian_strain_vis/<sequence>_f<frame>.png
        각 이미지 = [GT mesh | ||E||_F | trace(E) | lam1 | lam2 | lam3 | det(F)-1]

Usage:
    python sanity_check_jacobian.py
    python sanity_check_jacobian.py --vis_frames config/vis_frames.yml --out_dir jacobian_strain_vis
"""
import os
import sys
import argparse
import tempfile
import types
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from utils.vis_loader import CheckpointVisLoader
from utils.mesh_utils import compute_strain_signal, compute_jacobian_det, taubin_smooth_np
from utils.matplotlib_rnd import vis_mesh_key_weight
from utils.exp_utils import plateau_hat_points


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vis_frames', default='config/vis_frames.yml')
    p.add_argument('--data_basedir', default='/data/sihun')
    p.add_argument('--out_dir', default='jacobian_strain_vis')
    return p.parse_args()


def render_panel(verts_np, faces_np, weights_1d, cmap, vmin, vmax, title, tmp_dir, **kwargs):
    path = os.path.join(tmp_dir, f'{title.replace(" ","_").replace("/","_").replace(":","_")}.png')
    vis_mesh_key_weight(
        verts_np, faces_np, weights_1d.reshape(-1, 1), cage_idx=0,
        cmap=cmap, vmin=vmin, vmax=vmax,
        view_yrots=(0,),
        title=title, save_path=path, close=True, shade=True,
    )
    img = Image.open(path).copy()
    os.remove(path)
    return img


def stitch_horizontal(imgs):
    total_w = sum(i.width for i in imgs)
    max_h   = max(i.height for i in imgs)
    out = Image.new('RGB', (total_w, max_h), (255, 255, 255))
    x = 0
    for img in imgs:
        out.paste(img, (x, 0))
        x += img.width
    return out


def sym_vmax(*arrs, pct=95):
    """Robust symmetric vmax using percentile to avoid outlier blow-up."""
    v = max(float(np.percentile(np.abs(a), pct)) for a in arrs)
    return max(v, 1e-6)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # CheckpointVisLoader 를 최소 opts 로 초기화
    opts = types.SimpleNamespace(
        vis_frames=args.vis_frames,
        data_basedir=args.data_basedir,
    )
    loader = CheckpointVisLoader(opts, device='cpu')

    if not loader._enabled:
        print('No frames loaded. Check vis_frames.yml path.')
        return

    print(f'{len(loader.frames)} frames to process.')

    with tempfile.TemporaryDirectory() as tmp_dir:
        for entry in loader.frames:
            label = entry['label']
            print(f'  processing: {label}')

            try:
                sample = loader._load_frame(entry)
            except Exception as e:
                print(f'  [skip] {label}: {e}')
                continue

            template_v = sample['template'].unsqueeze(0)   # [1,V,3]
            vertices_v = sample['vertices'].unsqueeze(0)   # [1,V,3]
            faces_t    = sample['faces']                   # [F,3]
            faces_np   = sample['faces_np']
            template_np = sample['template'].numpy()
            vertices_np = sample['vertices'].numpy()

            # ── smooth GT (Taubin, same n_iter as training default)
            smooth_np = taubin_smooth_np(vertices_np, faces_np, n_iter=16)
            smooth_v  = torch.tensor(smooth_np).float().unsqueeze(0)

            # ── t_mask (plateau hat: 1 on face, 0 on neck/boundary)
            t_mask = plateau_hat_points(template_v)  # [1, V, 1]

            # ── masked GT: smooth_GT + (GT - smooth_GT) * t_mask
            edd_disp     = vertices_v - smooth_v            # [1, V, 3]
            masked_gt_v  = smooth_v + edd_disp * t_mask    # [1, V, 3]
            masked_gt_np = masked_gt_v[0].numpy()

            # ── strain: norm_trace (template → GT, conditioning signal)
            nt      = compute_strain_signal(vertices_v, template_v, faces_t, mode='norm_trace')
            norm_v  = nt[0, :, 0].numpy()   # unsigned ||E||_F
            trace_v = nt[0, :, 1].numpy()   # signed trace(E)

            # ── strain: principal (template → GT)
            pr   = compute_strain_signal(vertices_v, template_v, faces_t, mode='principal')
            lam1 = pr[0, :, 0].numpy()
            lam2 = pr[0, :, 1].numpy()
            lam3 = pr[0, :, 2].numpy()

            # ── Jacobian det panels ──────────────────────────────────────
            # naiveEDD: template → GT  (full deformation, conditioning signal view)
            jdet_naive   = compute_jacobian_det(vertices_v, template_v, faces_t)[0, :, 0].numpy()
            # trueEDD:  smooth_GT → GT  (pure EDD, unmasked output target)
            jdet_true    = compute_jacobian_det(vertices_v, smooth_v,   faces_t)[0, :, 0].numpy()
            # maskedEDD: smooth_GT → masked_GT  (actual training output target)
            jdet_masked  = compute_jacobian_det(masked_gt_v, smooth_v,  faces_t)[0, :, 0].numpy()

            # ── panel specs ─────────────────────────────────────────────
            lam_vmax = sym_vmax(lam1, lam3)
            panels_spec = [
                # (verts_for_render, weights, cmap, vmin, vmax, title)
                (vertices_np,  np.zeros(len(vertices_np)), 'Greys',    0,                    1,                            'GT'),
                (smooth_np,    np.zeros(len(smooth_np)),   'Greys',    0,                    1,                            'smooth_GT'),
                (vertices_np,  norm_v,   'YlOrRd',   0,                    max(float(norm_v.max()), 1e-6), '||E||_F'),
                (vertices_np,  trace_v,  'coolwarm', -sym_vmax(trace_v),   sym_vmax(trace_v),              'trace(E)'),
                (vertices_np,  lam1,     'coolwarm', -lam_vmax,            lam_vmax,                       'lam1(max)'),
                (vertices_np,  lam3,     'coolwarm', -lam_vmax,            lam_vmax,                       'lam3(min)'),
                (vertices_np,  jdet_naive,  'coolwarm', -sym_vmax(jdet_naive),  sym_vmax(jdet_naive),  'naiveEDD:det(F)-1'),
                (vertices_np,  jdet_true,   'coolwarm', -sym_vmax(jdet_true),   sym_vmax(jdet_true),   'trueEDD:det(F)-1'),
                (masked_gt_np, jdet_masked, 'coolwarm', -sym_vmax(jdet_masked), sym_vmax(jdet_masked), 'maskedEDD:det(F)-1'),
            ]

            imgs = [
                render_panel(verts, faces_np, w, cm, vmin, vmax, title, tmp_dir)
                for verts, w, cm, vmin, vmax, title in panels_spec
            ]

            stitched = stitch_horizontal(imgs)
            safe_label = label.replace('/', '_')
            save_path = os.path.join(args.out_dir, f'{safe_label}.png')
            stitched.save(save_path)

    print(f'\nDone. {len(loader.frames)} images saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
