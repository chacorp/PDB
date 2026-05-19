"""
precompute_caricaturized_aug.py — Build augmented per-id mesh + bind pose cache
via the computational caricaturization executable [Sela et al. CVIU 2015].

For each ICT / MF identity:
  1. Write the neutral mesh as a temp .obj.
  2. Run noref caricaturization with the chosen gamma → augmented .obj.
  3. Read the augmented vertex coords V_id_aug.
  4. Re-compute per-id bind pose using the SAME home_vidx + offset rule as
     precompute_per_id_bind_pos_v2.py, but on V_id_aug instead of V_id.
       offset[j]      = bind_pos_mean[j] - V_mean[home_vidx[j]]
       bind_pos_aug[j] = V_id_aug[home_vidx[j]] + offset[j]
  5. Helper override (when active_joints.helper_joint_idx present): helper
     joints inherit the augmented parent's per-id bind pose.

Outputs (default --out_dir nfs_features_seg_aug):
  {id_name}_aug.npy                    : [N, 3]  augmented vertex coords
  {id_name}_aug_bind_pos_landmark.npy  : [J, 3]  per-id bind pose for augmented

Usage:
    # First, build the caricaturization executable on this server:
    #   cd third_party/coupe.computational-caricaturization
    #   git submodule update --init --recursive
    #   mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make
    # Then:
    python tools/precompute/precompute_caricaturized_aug.py \
        --gamma 0.15 \
        --rig_path maya_rig/hybrid \
        --home_vidx_dir maya_rig/hybrid \
        --feat_dir nfs_features_seg \
        --out_dir nfs_features_seg_aug \
        --datasets ict mf
"""
import os
import sys
import subprocess
import tempfile
import pickle
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

from utils.rig_loader import load_rig
from extract_nfs_feat import load_templates_ict, load_templates_mf


def _write_obj(path, V, F):
    with open(path, 'w') as f:
        for v in V:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for tri in F:
            f.write(f'f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n')


def _read_obj_verts(path, expected_N=None):
    V = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                _, x, y, z = line.split()
                V.append([float(x), float(y), float(z)])
    V = np.array(V, dtype=np.float32)
    if expected_N is not None and V.shape[0] != expected_N:
        raise RuntimeError(f'{path}: expected N={expected_N}, got {V.shape[0]}')
    return V


def _load_mean_verts(topo, data_basedir):
    if topo == 'ict':
        from utils.remesh_utils import ICT_face_model
        return ICT_face_model().neutral_verts.astype(np.float32)
    if topo == 'mf':
        pkl = os.path.join(data_basedir, 'multiface_align', 'mf_templates.pkl')
        if not os.path.exists(pkl):
            pkl = os.path.join(data_basedir, 'pca', 'multiface_align', 'mf_templates.pkl')
        with open(pkl, 'rb') as f:
            t = pickle.load(f)
        keys = [k for k in t if k != 'face']
        return (np.stack([np.array(t[k], dtype=np.float32) for k in keys])
                .mean(axis=0).astype(np.float32))
    raise ValueError(f'unknown topology: {topo}')


def _load_templates(dataset, data_basedir):
    if dataset == 'ict':
        return load_templates_ict()
    if dataset == 'mf':
        return load_templates_mf(data_basedir)
    raise ValueError(f'unknown dataset: {dataset}')


# DTU3D landmark groupings (1-indexed) — must match train_hlbs.py.
_DTU3D_BROW_R = (1, 3, 5, 7); _DTU3D_BROW_L = (9, 11, 13, 15)
_DTU3D_EYE_R  = (17, 19, 21, 23); _DTU3D_EYE_L = (25, 27, 29, 31)
_DTU3D_NOSE   = (36, 37, 43, 44)
_DTU3D_MOUTH  = (40, 47, 48, 50, 52, 53, 54, 55, 56)


def _align_similarity(V_aug, V_orig):
    """Translation + uniform scale alignment.
    Match V_aug's centroid + RMS-from-centroid to V_orig's, so the augmented
    mesh sits in the original training-scale bbox.
    """
    c_orig = V_orig.mean(axis=0)
    c_aug  = V_aug.mean(axis=0)
    rms_orig = float(np.sqrt(((V_orig - c_orig) ** 2).sum(axis=-1).mean()))
    rms_aug  = float(np.sqrt(((V_aug  - c_aug ) ** 2).sum(axis=-1).mean()))
    s = rms_orig / max(rms_aug, 1e-8)
    return ((V_aug - c_aug) * s + c_orig).astype(np.float32)


def compute_blend_mask(V_orig, landmark_vidx, mode='landmark',
                       w_brow=1.0, w_eye=1.0, w_nose=0.3, w_mouth=0.3,
                       r0=0.05, r1=0.25):
    """Per-vertex blend weight in [0, 1] for caricaturization blending.

    V_final[v] = (1 - mask[v]) · V_orig[v] + mask[v] · V_carica[v]

    mode='face'     : single plateau_hat from face center (default centers).
    mode='landmark' : per-region plateau_hat with per-region weights.
                      Lower weights on mouth/nose reduce lip self-penetration
                      and nose interior poke-through.
    """
    import torch
    from utils.exp_utils import plateau_hat_points

    V_t = torch.from_numpy(V_orig).float()                                  # [N, 3]

    if mode == 'face':
        return plateau_hat_points(V_t).squeeze(-1).numpy().astype(np.float32)

    # landmark mode: combine per-region bumps with weights.
    def _region_bump(lm_ids):
        valid = [k - 1 for k in lm_ids if 0 <= k - 1 < len(landmark_vidx)]
        if not valid:
            return torch.zeros(V_t.shape[0])
        vidx = torch.from_numpy(landmark_vidx[valid].astype(np.int64))
        vidx = vidx[vidx < V_t.shape[0]]
        if len(vidx) == 0:
            return torch.zeros(V_t.shape[0])
        C = V_t[vidx]                                                       # [K, 3]
        return plateau_hat_points(V_t, C, r0=r0, r1=r1).max(dim=-1).values  # [N]

    bump_brow  = _region_bump(_DTU3D_BROW_R + _DTU3D_BROW_L)
    bump_eye   = _region_bump(_DTU3D_EYE_R  + _DTU3D_EYE_L)
    bump_nose  = _region_bump(_DTU3D_NOSE)
    bump_mouth = _region_bump(_DTU3D_MOUTH)

    mask = torch.zeros(V_t.shape[0])
    mask = torch.maximum(mask, w_brow  * bump_brow)
    mask = torch.maximum(mask, w_eye   * bump_eye)
    mask = torch.maximum(mask, w_nose  * bump_nose)
    mask = torch.maximum(mask, w_mouth * bump_mouth)
    return mask.clamp(0.0, 1.0).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gamma', type=float, default=0.15,
                    help='Caricaturization exaggeration factor (0.05~0.15 plausible).')
    ap.add_argument('--cari_exec', type=str,
                    default='third_party/coupe.computational-caricaturization/build/main')
    ap.add_argument('--rig_path', type=str, default='maya_rig/hybrid')
    ap.add_argument('--home_vidx_dir', type=str, default='maya_rig/hybrid',
                    help='Dir with joint_home_vidx_{topo}.npy [J] (Phase B v3).')
    ap.add_argument('--feat_dir', type=str, default='nfs_features_seg',
                    help='(Unused now — kept for parity with v2 script.)')
    ap.add_argument('--out_dir', type=str, default='nfs_features_seg_aug')
    ap.add_argument('--datasets', nargs='+', default=['ict', 'mf'], choices=['ict', 'mf'])
    ap.add_argument('--data_basedir', type=str, default='/data/sihun')
    ap.add_argument('--active_json', type=str,
                    default='maya_rig/hybrid/active_joints_manual.json',
                    help='For helper override: helpers inherit parent per-id bind pose.')
    ap.add_argument('--overwrite', action='store_true')
    # Region-selective blending (mitigates lip self-penetration / nose poke-through)
    ap.add_argument('--blend_mask', type=str, default='none',
                    choices=['none', 'face', 'landmark'],
                    help='Per-vertex blend mask. none=full caricaturization (default). '
                         'face=plateau_hat from face center. '
                         'landmark=per-region weights (low on mouth/nose to avoid self-penetration).')
    ap.add_argument('--mask_w_brow',  type=float, default=1.0)
    ap.add_argument('--mask_w_eye',   type=float, default=1.0)
    ap.add_argument('--mask_w_nose',  type=float, default=0.3)
    ap.add_argument('--mask_w_mouth', type=float, default=0.3)
    ap.add_argument('--mask_r0',      type=float, default=0.05)
    ap.add_argument('--mask_r1',      type=float, default=0.25)
    ap.add_argument('--align', type=str, default='similarity',
                    choices=['none', 'similarity'],
                    help='Post-caricaturization alignment to original training scale. '
                         'similarity = translate + uniform scale so centroid + RMS match V_orig.')
    args = ap.parse_args()

    assert os.path.exists(args.cari_exec), (
        f'caricaturization executable not found: {args.cari_exec}\n'
        f'Build it first:\n'
        f'  cd third_party/coupe.computational-caricaturization\n'
        f'  git submodule update --init --recursive\n'
        f'  mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make')
    args.cari_exec = os.path.abspath(args.cari_exec)

    rig = load_rig(args.rig_path, device='cpu')
    J = len(rig.joint_names)
    print(f'rig: J={J}')

    helper_idx = set()
    if args.active_json and os.path.exists(args.active_json):
        import json as _json
        with open(args.active_json) as f:
            _aj = _json.load(f)
        helper_idx = set(_aj.get('helper_joint_idx', []))
    if helper_idx:
        print(f'  helper joints: {len(helper_idx)} (will inherit augmented parent bind pose)')

    os.makedirs(args.out_dir, exist_ok=True)
    n_total = 0
    n_skipped = 0

    for ds in args.datasets:
        topo = ds
        if topo not in rig.bind_pos_dict:
            print(f'[skip] {topo}: bind_pos_dict not loaded'); continue
        bind_pos_mean = rig.bind_pos_dict[topo].cpu().numpy().astype(np.float32)

        hvidx_path = os.path.join(args.home_vidx_dir, f'joint_home_vidx_{topo}.npy')
        if not os.path.exists(hvidx_path):
            print(f'[skip] {topo}: {hvidx_path} not found'); continue
        home_vidx = np.load(hvidx_path).astype(np.int64)

        V_mean = _load_mean_verts(topo, args.data_basedir)
        offsets = bind_pos_mean - V_mean[home_vidx]
        print(f'\n[{topo}] V_mean={V_mean.shape[0]}, home_vidx OK, offsets ‖δ‖ mean='
              f'{np.linalg.norm(offsets, axis=-1).mean():.4f}')

        # Optional landmark_vidx for blend mask
        lm_vidx_for_mask = None
        if args.blend_mask in ('face', 'landmark'):
            _lv_path = os.path.join(args.home_vidx_dir, f'landmark_vidx_{topo}.npy')
            if os.path.exists(_lv_path):
                lm_vidx_for_mask = np.load(_lv_path).astype(np.int64)
                print(f'  blend_mask={args.blend_mask}, landmark_vidx loaded ({len(lm_vidx_for_mask)})')
            elif args.blend_mask == 'landmark':
                print(f'  [warn] blend_mask=landmark needs {_lv_path}; falling back to no mask')

        templates = _load_templates(ds, args.data_basedir)
        for V_id, F, name in templates:
            out_aug  = os.path.join(args.out_dir, f'{name}_aug.npy')
            out_bind = os.path.join(args.out_dir, f'{name}_aug_bind_pos_landmark.npy')
            if os.path.exists(out_aug) and os.path.exists(out_bind) and not args.overwrite:
                n_skipped += 1; continue

            V_id = V_id.astype(np.float32)
            assert V_id.shape[0] == V_mean.shape[0], \
                f'{name}: V_id={V_id.shape[0]} vs V_mean={V_mean.shape[0]}'

            # Run caricaturization via temp .obj files
            tmp_dir = tempfile.mkdtemp(prefix='carica_')
            tmp_in  = os.path.join(tmp_dir, f'{name}_in.obj')
            tmp_out = os.path.join(tmp_dir, f'{name}_out.obj')
            try:
                _write_obj(tmp_in, V_id, F)
                cmd = [args.cari_exec, 'noref', tmp_in, str(args.gamma), tmp_out]
                proc = subprocess.run(cmd, capture_output=True, check=False)
                if proc.returncode != 0 or not os.path.exists(tmp_out):
                    print(f'[fail] {name}: cari exec returned {proc.returncode}\n'
                          f'  stderr={proc.stderr.decode()[:300]}')
                    continue
                V_aug_raw = _read_obj_verts(tmp_out, expected_N=V_id.shape[0])
                # Align to training scale (translation + uniform scale to V_id's centroid+RMS).
                if args.align == 'similarity':
                    V_aug_raw = _align_similarity(V_aug_raw, V_id)
                # Optional region-selective blending
                if args.blend_mask != 'none' and lm_vidx_for_mask is not None:
                    mask = compute_blend_mask(
                        V_id, lm_vidx_for_mask, mode=args.blend_mask,
                        w_brow=args.mask_w_brow, w_eye=args.mask_w_eye,
                        w_nose=args.mask_w_nose, w_mouth=args.mask_w_mouth,
                        r0=args.mask_r0, r1=args.mask_r1)
                    V_aug = V_id * (1 - mask[:, None]) + V_aug_raw * mask[:, None]
                    V_aug = V_aug.astype(np.float32)
                else:
                    V_aug = V_aug_raw
            finally:
                for p in (tmp_in, tmp_out):
                    if os.path.exists(p): os.remove(p)
                if os.path.exists(tmp_dir): os.rmdir(tmp_dir)

            # Per-id bind pose for augmented mesh
            aug_bind = V_aug[home_vidx] + offsets                          # [J, 3]
            if helper_idx:
                parent_idx_np = rig.parent_idx.cpu().numpy()
                proc_order = rig.process_order.cpu().numpy()
                for j in proc_order:
                    if int(j) in helper_idx:
                        p = int(parent_idx_np[j])
                        if p >= 0:
                            aug_bind[j] = aug_bind[p]

            np.save(out_aug,  V_aug.astype(np.float32))
            np.save(out_bind, aug_bind.astype(np.float32))
            n_total += 1
            if n_total % 10 == 0:
                print(f'  [{topo}] processed {n_total} ids ...')

        print(f'  {topo}: written {n_total} (skipped existing: {n_skipped})')

    print(f'\nDone. Output dir: {args.out_dir}')
    print(f'Total written: {n_total}, skipped existing: {n_skipped}')


if __name__ == '__main__':
    main()
