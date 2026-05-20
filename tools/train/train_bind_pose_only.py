"""
train_bind_pose_only.py — Stage-1 bind_pose_net 단독 학습.

bind_pose_net의 함수 자체는 identity-only (input = neutral mesh feat) 라
deformed/expression 데이터 없이 per-id GT cache만으로 충분히 학습 가능.

For each step:
  - Sample batch_size identities (from 113 total = 100 ICT + 13 MF)
  - Apply data augmentation (random trans + scale + vertex subsample)
  - Forward only bind_pose_net (LBS / skin_weight_net 모두 skip)
  - L_bind_reg = MSE(pred_joint_pos, per_id_GT)   (helpers excluded)
  - Helpers excluded entirely (no L_bind_reg / L_helper_residual / L_mirror)
  - Backward → only bind_pose_net (+ adapter for helper output) updated

Outputs:
  - {out_dir}/bind_only_{epoch:03d}.pth      (bind_pose_net state_dict only)
  - {out_dir}/log.txt
  - {out_dir}/train_opts.yml

Usage:
    python tools/train/train_bind_pose_only.py \
        --base_ckpt_dir ckpts_hlbs/2026-05-14-14-02-03-HLBS-FullPred-ict-jTrans-nrm0.1-Wsm0.01 \
        --base_epoch 200 \
        --out_dir   ckpts_bind_only/$(date +%Y-%m-%d-%H-%M-%S)-bind_only \
        --max_epoch 300 --batch_size 8 --lr 1e-4 \
        --aug_trans --aug_scale --aug_subsample mix4
"""
import os
import sys
import argparse
import pickle
import time
import yaml
import json
import numpy as np
import torch
import torch.nn.functional as F
import igl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

from utils.rig_loader import load_rig
from models.hierarchical_lbs import HierarchicalLBS_FullPred
from utils.exp_utils import plateau_hat_points, plateau_hat_r


# ── DTU3D landmark groups (mirror train_hlbs.py) ─────────────────────────
_DTU3D_BROW_R = (1, 3, 5, 7); _DTU3D_BROW_L = (9, 11, 13, 15)
_DTU3D_EYE_R  = (17, 19, 21, 23); _DTU3D_EYE_L = (25, 27, 29, 31)
_DTU3D_NOSE   = (36, 37, 43, 44)
_DTU3D_MOUTH  = (40, 47, 48, 50, 52, 53, 54, 55, 56)
_DTU3D_SEGMENTS = [
    (1,3),(3,5),(5,7),(9,11),(11,13),(13,15),
    (17,19),(19,21),(21,23),(23,17),(25,27),(27,29),(29,31),(31,25),
    (36,37),(43,44),
    (47,48),(48,50),(50,52),(52,53),(53,54),(54,55),(55,56),(56,47),(40,50),
]


def _load_ict_identities(data_basedir, n_ids):
    """Return list of (V, faces, id_name) for n_ids ICT identities.
    Loads `ict_face_pt/random_identity_vecs.npy` — the SAME identity source as
    precompute_per_id_bind_pos_v2.py (default) and train_hlbs.py train mode —
    so `ict_NNN` naming matches the regenerated per-id GT cache (111 entries).
    Note: random_identity_vecs[0:100] == data/ICT_live_100/iden_vecs.npy.
    """
    from utils.remesh_utils import ICT_face_model
    m = ICT_face_model()
    iden_vecs = np.load('ict_face_pt/random_identity_vecs.npy')[:n_ids]
    faces = m.faces.astype(np.int32)
    out = []
    for i in range(min(n_ids, iden_vecs.shape[0])):
        id_disps = m.get_id_disp(iden_vecs[i]).squeeze()
        V = (m.neutral_verts + id_disps).astype(np.float32)
        out.append((V, faces, f'ict_{i:03d}'))
    return out


def _load_mf_identities(data_basedir):
    pkl = os.path.join(data_basedir, 'multiface_align', 'mf_templates.pkl')
    if not os.path.exists(pkl):
        pkl = os.path.join(data_basedir, 'pca', 'multiface_align', 'mf_templates.pkl')
    with open(pkl, 'rb') as f:
        t = pickle.load(f)
    faces = np.array(t['face'], dtype=np.int32)
    keys = [k for k in t if k != 'face']
    return [(np.array(t[k], dtype=np.float32), faces, k) for k in keys]


def _load_per_id_gt(id_name, gt_dir):
    p = os.path.join(gt_dir, f'{id_name}_bind_pos_landmark.npy')
    return np.load(p).astype(np.float32) if os.path.exists(p) else None


def _load_nfs_feat(id_name, nfs_dir):
    if not nfs_dir: return None
    p = os.path.join(nfs_dir, f'{id_name}_nfs_feat.npy')
    return np.load(p).astype(np.float32) if os.path.exists(p) else None


def _load_unseen_identities(nfs_feat_dir, coma_pkl, biwi_pkl):
    """COMA + BIWI identities for pred-only vis (no GT — landmark-less topos).

    Returns list of dicts {V, F, normals, GT(None), nfs, name, topo}.
    bind_pose_net inference needs only mesh+normal+NFS, so these run fine; they
    are NEVER merged into train/valid pools (vis-only).
    """
    out = []
    for topo, pkl in (('coma', coma_pkl), ('biwi', biwi_pkl)):
        if not pkl or not os.path.exists(pkl):
            print(f'  [unseen] {topo}: template pkl not found ({pkl}) — skip')
            continue
        try:
            with open(pkl, 'rb') as f:
                t = pickle.load(f)
        except Exception as e:
            print(f'  [unseen] {topo}: failed to load {pkl}: {e}'); continue
        faces = np.array(t['face'], dtype=np.int32)
        names = [k for k in t if k != 'face']
        n_ok = 0
        for name in names:
            nfs = _load_nfs_feat(name, nfs_feat_dir)
            if nfs_feat_dir and nfs is None:
                print(f'  [unseen][skip] {name}: no NFS feat'); continue
            V = np.array(t[name], dtype=np.float32)
            normals = igl.per_vertex_normals(V, faces).astype(np.float32)
            out.append({'V': V, 'F': faces, 'normals': normals,
                        'GT': None, 'nfs': nfs, 'name': name, 'topo': topo})
            n_ok += 1
        print(f'  [unseen] {topo}: {n_ok}/{len(names)} identities loaded')
    return out


# ── Augmentation ────────────────────────────────────────────────────────

def _augment(V, normals, GT, nfs_feat, opts, rng):
    """Apply random scale + translate + vertex subsample.
    All three apply to V/normals/(nfs_feat) consistently; GT is transformed
    by the same affine (scale + trans).
    """
    trans = np.zeros(3, dtype=np.float32)
    scale = 1.0
    if opts.aug_trans:
        t_range = 0.1
        trans = (rng.random(3).astype(np.float32) - 0.5) * t_range
    if opts.aug_scale:
        scale = float(rng.uniform(0.8, 1.2))
    V_aug  = V * scale + trans                           # [N, 3]
    GT_aug = GT * scale + trans                          # [J, 3]
    n_aug  = normals                                     # normals are scale/trans invariant

    perm = None
    if opts.aug_subsample != 'none':
        N = V_aug.shape[0]
        ratio = float(rng.uniform(opts.aug_ratio_min, opts.aug_ratio_max))
        K = int(round(N * ratio))
        if K < N and K > 0:
            # mix-style: pick mode per call
            mode = opts.aug_subsample
            if mode == 'mix4':
                # 1/4 each of: full, random, fps, importance
                roll = int(rng.integers(0, 4))
                if roll == 0: K = N  # full
                elif roll == 1: mode = 'random'
                elif roll == 2: mode = 'fps'
                else:           mode = 'importance'
            elif mode == 'mix3_halffull':
                # 1/2 full mesh + 1/2 (random / fps / importance) 균등 1/6 each
                if float(rng.random()) < 0.5:
                    K = N  # full mesh
                else:
                    mode = ['random', 'fps', 'importance'][int(rng.integers(0, 3))]
            if K < N:
                if mode == 'random':
                    perm = rng.permutation(N)[:K]
                elif mode == 'fps':
                    perm = _fps_numpy(V_aug, K, rng)
                elif mode == 'importance':
                    # plateau_hat_points needs torch
                    V_t = torch.from_numpy(V_aug)
                    bump = plateau_hat_points(V_t).squeeze(-1)
                    w = 1.0 + 4.0 * bump
                    perm = torch.multinomial(w, K, replacement=False).numpy()
                else:
                    perm = rng.permutation(N)[:K]
    if perm is not None:
        V_aug  = V_aug[perm]
        n_aug  = n_aug[perm]
        if nfs_feat is not None:
            nfs_feat = nfs_feat[perm]
    return V_aug, n_aug, GT_aug, nfs_feat


def _render_bind_grid(items, save_path=None, title='', return_fig=False):
    """Tile multiple meshes into ONE figure — one COLUMN per identity, 2 rows
    (XY frontal on top, ZY side on bottom). Identities laid out left-to-right.
    Each cell: mesh wireframe + pred joints (red) and, if present, GT joints
    (blue) + GT↔pred connecting lines.

    items : list of dict {V[N,3], F[Fc,3], pred[J,3], gt[J,3] or None, label}.
    save_path / return_fig : as in the per-id renderer.
    """
    n = len(items)
    if n == 0:
        return None
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 5.6), squeeze=False)
    for c, it in enumerate(items):
        V, faces_arr = it['V'], it['F']
        pred_jp, gt_jp, label = it['pred'], it['gt'], it['label']
        for r, (axx, axy, vname) in enumerate([(0, 1, 'XY'), (2, 1, 'ZY')]):
            ax = axes[r][c]
            polys = V[faces_arr][:, :, [axx, axy]]
            pc = PolyCollection(polys, facecolors='#dddddd', edgecolors='#bbbbbb',
                                 linewidths=0.06, alpha=0.4, zorder=1)
            ax.add_collection(pc)
            if gt_jp is not None:
                ax.scatter(gt_jp[:, axx], gt_jp[:, axy], s=10, c='#1d4ed8',
                           alpha=0.9, edgecolors='white', linewidths=0.25,
                           zorder=5, label='GT')
            ax.scatter(pred_jp[:, axx], pred_jp[:, axy], s=10, c='#e63946',
                       alpha=0.9, edgecolors='white', linewidths=0.25,
                       zorder=6, label='pred')
            if gt_jp is not None:
                for j in range(pred_jp.shape[0]):
                    ax.plot([gt_jp[j, axx], pred_jp[j, axx]],
                             [gt_jp[j, axy], pred_jp[j, axy]],
                             color='#aaaaaa', linewidth=0.3, zorder=4)
            ax.set_aspect('equal'); ax.tick_params(labelsize=5)
            if r == 0:
                ax.set_title(label, fontsize=7)
            if c == 0:
                ax.set_ylabel(vname, fontsize=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=6, loc='lower right')

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
    if return_fig:
        return fig
    plt.close(fig)
    return None


def _fig_to_tensor(fig):
    """matplotlib Figure → [3, H, W] uint8 tensor for tb.add_image. Closes fig."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())          # [H, W, 4]
    img = torch.from_numpy(buf[:, :, :3].copy()).permute(2, 0, 1).contiguous()
    plt.close(fig)
    return img                                           # uint8 [3, H, W]


def _fps_numpy(V, K, rng):
    N = V.shape[0]
    indices = np.zeros(K, dtype=np.int64)
    dists = np.full(N, np.inf, dtype=np.float32)
    farthest = int(rng.integers(0, N))
    for i in range(K):
        indices[i] = farthest
        diff = V - V[farthest]
        d = (diff * diff).sum(axis=-1)
        dists = np.minimum(dists, d)
        farthest = int(np.argmax(dists))
    return indices


# ── Visualization logging ───────────────────────────────────────────────

def _predict_bind(model, d, device):
    """Clean forward of bind_pose_net on the ORIGINAL (non-augmented) mesh.
    Returns predicted joint positions [J, 3] as numpy. Caller in no_grad."""
    V_t = torch.from_numpy(d['V']).float().unsqueeze(0).to(device)
    N_t = torch.from_numpy(d['normals']).float().unsqueeze(0).to(device)
    nfs_t = (torch.from_numpy(d['nfs']).float().unsqueeze(0).to(device)
             if d['nfs'] is not None else None)
    skin_input, _adain = model._prepare_feat(V_t, N_t, nfs_t)
    _, pred = model._get_bind_pose(skin_input, adain_input=_adain)
    return pred[0].cpu().numpy()                                   # [J, 3]


def _render_topo_grids(model, ids, vis_joint_sel, device, save_dir,
                        tag_prefix, tb, tb_step, title_prefix=''):
    """Forward each id (clean, ORIGINAL mesh), group by dataset/topo, render ONE
    tiled grid PNG + tb.add_image per topo. Caller must be inside no_grad."""
    by_topo = {}
    for d in ids:
        pred = _predict_bind(model, d, device)
        gt = d['GT']
        by_topo.setdefault(d['topo'], []).append({
            'V': d['V'], 'F': d['F'],
            'pred': pred[vis_joint_sel],
            'gt': gt[vis_joint_sel] if gt is not None else None,
            'label': d['name'],
        })
    for topo, items in by_topo.items():
        png = os.path.join(save_dir, f'{topo}.png')
        fig = _render_bind_grid(items, save_path=png, return_fig=True,
                                title=f'{title_prefix}{topo} ({len(items)})')
        tb.add_image(f'{tag_prefix}/{topo}', _fig_to_tensor(fig), tb_step)


def _log_train_batch_vis(model, batch_ids, vis_joint_sel, epoch, step,
                          global_step, device, out_dir, tb):
    """Tiled per-dataset grid of the current training batch: original mesh +
    GT(blue) + pred(red) joints. Clean forward on ORIGINAL mesh (augmented/
    subsampled frame avoided)."""
    was_training = model.training
    model.eval()
    save_dir = os.path.join(out_dir, 'train_vis', f'{epoch:03d}',
                            f'step{step:04d}')
    with torch.no_grad():
        _render_topo_grids(model, batch_ids, vis_joint_sel, device, save_dir,
                           'train_vis', tb, global_step,
                           title_prefix=f'[{epoch:03d}] step{step:04d} ')
    if was_training:
        model.train()


def _log_unseen_vis(model, unseen_ids, vis_joint_sel, epoch, device, out_dir, tb):
    """COMA/BIWI pred-only tiled grid (no GT). Caller must already be inside
    model.eval() + torch.no_grad()."""
    save_dir = os.path.join(out_dir, 'unseen_vis', f'{epoch:03d}')
    _render_topo_grids(model, unseen_ids, vis_joint_sel, device, save_dir,
                       'unseen_vis', tb, epoch,
                       title_prefix=f'[{epoch:03d}] pred-only ')


# ── Main training loop ──────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt_dir', required=True,
                    help='Existing trained ckpt dir (uses its train_opts.yml + model arch).')
    ap.add_argument('--base_epoch', default='200',
                    help='Which epoch state_dict to start bind_pose_net from.')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--per_id_gt_dir', default='nfs_features_seg')
    ap.add_argument('--n_ict_ids', type=int, default=111,
                    help='ICT ids sliced from random_identity_vecs.npy. '
                         'train_hlbs.py train mode uses [:111].')
    ap.add_argument('--data_basedir', default='/data/sihun')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max_epoch', type=int, default=300)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--save_interval', type=int, default=50)
    ap.add_argument('--log_interval', type=int, default=20)
    ap.add_argument('--val_interval', type=int, default=10,
                    help='Run validation + save bind_pose vis every N epochs.')
    ap.add_argument('--n_valid_ict', type=int, default=5,
                    help='Hold out last N ICT ids for validation (not used in training).')
    ap.add_argument('--n_valid_mf', type=int, default=3,
                    help='Hold out last N MF ids for validation.')
    # Aug
    ap.add_argument('--aug_trans', action='store_true')
    ap.add_argument('--aug_scale', action='store_true')
    ap.add_argument('--aug_subsample', default='none',
                    choices=['none', 'random', 'fps', 'importance', 'mix4', 'mix3_halffull'],
                    help='mix4 = 1/4 each of (full, random, fps, importance). '
                         'mix3_halffull = 1/2 full + 1/2 random pick of (random, fps, importance).')
    ap.add_argument('--aug_ratio_min', type=float, default=0.3)
    ap.add_argument('--aug_ratio_max', type=float, default=0.8)
    # Visualization
    ap.add_argument('--vis_train', type=int, default=1,
                    help='Per-batch GT-vs-pred bind pose vis at ~steps/10 '
                         'cadence (matches train_hlbs.py). 0=off.')
    ap.add_argument('--vis_unseen', type=int, default=1,
                    help='COMA/BIWI pred-only bind pose vis in the validation '
                         'block (every --val_interval). 0=off.')
    ap.add_argument('--vis_all_joints', action='store_true',
                    help='Render all joints (default: non-helper joints only).')
    ap.add_argument('--coma_template', default='utils/templates/voca_templates.pkl')
    ap.add_argument('--biwi_template', default='utils/templates/biwi_templates.pkl')
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'train_opts.yml'), 'w') as f:
        yaml.safe_dump(vars(args), f)

    # ── 1. Load model + restore base weights ─────────────────────────
    opts_path = os.path.join(args.base_ckpt_dir, 'train_opts.yml')
    with open(opts_path) as f:
        base_opts = yaml.safe_load(f)
    rig = load_rig(base_opts['rig_path'])

    base_ckpt = (os.path.join(args.base_ckpt_dir, 'model_hlbs_best.pth')
                 if str(args.base_epoch) == 'best'
                 else os.path.join(args.base_ckpt_dir, f'model_hlbs_{int(args.base_epoch):03d}.pth'))
    sd_peek = torch.load(base_ckpt, map_location='cpu', weights_only=False)
    face_idx = sd_peek['face_joint_idx'].tolist() if 'face_joint_idx' in sd_peek else None
    helper_idx_ckpt = (sd_peek['helper_joint_idx_buf'].tolist()
                       if 'helper_joint_idx_buf' in sd_peek else None)

    base_idx = None
    aj_path = base_opts.get('active_joints_json')
    if aj_path and os.path.exists(aj_path):
        with open(aj_path) as f:
            aj = json.load(f)
        base_idx = aj.get('base_joint_idx')

    sig = None
    if base_opts.get('sigma_targets_npy') and os.path.exists(base_opts['sigma_targets_npy']):
        sig = np.load(base_opts['sigma_targets_npy']).astype(np.float32)

    model = HierarchicalLBS_FullPred(
        rig=rig, topology=base_opts.get('topo_key', 'mf'),
        in_dim_exp=12, hid_dim=base_opts.get('hid_dim', 128),
        num_layers=base_opts.get('num_layers', 4),
        device=str(device), use_joint_trans=base_opts.get('use_joint_trans', False),
        dfn_skin=base_opts.get('dfn_skin', False), dfn_bind=base_opts.get('dfn_bind', False),
        dfn_exp=base_opts.get('dfn_exp', False),
        nfs_feat_dim=256 if base_opts.get('nfs_feat_dir') else 0,
        nfs_concat=base_opts.get('nfs_concat', False),
        adain_pos_norm=base_opts.get('adain_pos_norm', False),
        freeze_bind_pose=False,
        use_gmm_hybrid=base_opts.get('use_gmm_hybrid', False),
        init_log_sigma=base_opts.get('init_log_sigma', -1.2),
        gmm_mode=base_opts.get('gmm_mode', 'additive'),
        residual_scale=base_opts.get('residual_scale', 2.0),
        sigma_targets=sig, bind_pose_mode='net',
        face_joint_idx=face_idx, base_joint_idx=base_idx,
        face_mask_r0=base_opts.get('face_mask_r0', 1.0),
        face_mask_r1=base_opts.get('face_mask_r1', 2.25),
        helper_joint_idx=helper_idx_ckpt,
    ).to(device)
    # bind_pose_net을 처음부터 학습 — base ckpt에서 weight load 안 함.
    # 다른 module들은 frozen이고 forward path에 안 쓰임 (bind_pose_net만 forward).
    print(f'bind_pose_net random-init from scratch (no base ckpt weights loaded)')

    # ── 2. Freeze all params except bind_pose_net ───────────────────
    for p in model.parameters(): p.requires_grad_(False)
    train_params = []
    for p in model.bind_pose_net.parameters():
        p.requires_grad_(True)
        train_params.append(p)
    # AdaIN within bind_pose_net is part of it; covered.
    print(f'Trainable params: {sum(p.numel() for p in train_params):,}')
    optim = torch.optim.Adam(train_params, lr=args.lr)

    # ── 3. Load all identities + their per-id GT + NFS feat ─────────
    ict_ids = _load_ict_identities(args.data_basedir, args.n_ict_ids)
    mf_ids  = _load_mf_identities(args.data_basedir)
    all_ids = []
    for V, faces_arr, name in (ict_ids + mf_ids):
        GT = _load_per_id_gt(name, args.per_id_gt_dir)
        nfs = _load_nfs_feat(name, base_opts.get('nfs_feat_dir'))
        if GT is None:
            print(f'  [skip] {name}: no per-id GT'); continue
        if base_opts.get('nfs_feat_dir') and nfs is None:
            print(f'  [skip] {name}: no NFS feat'); continue
        normals = igl.per_vertex_normals(V, faces_arr).astype(np.float32)
        all_ids.append({'V': V, 'F': faces_arr, 'normals': normals,
                        'GT': GT, 'nfs': nfs, 'name': name,
                        'topo': 'ict' if name.startswith('ict_') else 'mf'})
    print(f'Loaded {len(all_ids)} identities total')

    # ── Train/valid split ────────────────────────────────────────────
    # Hold out last n_valid_ict ICT + last n_valid_mf MF ids for validation.
    ict_pool = [d for d in all_ids if d['topo'] == 'ict']
    mf_pool  = [d for d in all_ids if d['topo'] == 'mf']
    valid_ids = ict_pool[-args.n_valid_ict:] + mf_pool[-args.n_valid_mf:]
    train_ids = ict_pool[:-args.n_valid_ict] + mf_pool[:-args.n_valid_mf]
    print(f'Train: {len(train_ids)}  ({len(ict_pool) - args.n_valid_ict} ICT + '
          f'{len(mf_pool) - args.n_valid_mf} MF)')
    print(f'Valid: {len(valid_ids)}  ({args.n_valid_ict} ICT + {args.n_valid_mf} MF) — '
          f'{", ".join(d["name"] for d in valid_ids)}')
    all_ids = train_ids

    # ── 4. Non-helper joint indices (only supervise these) ───────────
    helper_set = set(helper_idx_ckpt) if helper_idx_ckpt else set()
    non_helper_idx = torch.tensor(
        [j for j in range(len(rig.joint_names)) if j not in helper_set],
        dtype=torch.long, device=device,
    )
    print(f'Supervising {len(non_helper_idx)} non-helper joints '
          f'({len(helper_set)} helpers excluded — no L_bind_reg / L_mirror / L_residual)')

    # Joint selector for visualization (non-helper only unless --vis_all_joints).
    if args.vis_all_joints:
        vis_joint_sel = np.arange(len(rig.joint_names), dtype=np.int64)
    else:
        vis_joint_sel = non_helper_idx.cpu().numpy()

    # ── 4b. COMA/BIWI identities for pred-only vis (vis-only, never trained) ──
    unseen_ids = []
    if args.vis_unseen:
        unseen_ids = _load_unseen_identities(
            base_opts.get('nfs_feat_dir'), args.coma_template, args.biwi_template)
        print(f'Unseen (vis-only): {len(unseen_ids)} COMA/BIWI identities')

    # ── 5. Training loop ─────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)
    log_file = open(os.path.join(args.out_dir, 'log.txt'), 'w')
    tb = SummaryWriter(log_dir=os.path.join(args.out_dir, 'tb'))
    print(f'TensorBoard: tensorboard --logdir {args.out_dir}/tb --port 6006')
    steps_per_epoch = max(1, len(all_ids) // args.batch_size)
    global_step = 0

    for epoch in range(1, args.max_epoch + 1):
        running = {'L_bind_reg': 0.0, 'total': 0.0}
        cnt = 0
        t0 = time.time()
        order = rng.permutation(len(all_ids))
        pbar = tqdm(range(steps_per_epoch),
                    desc=f'[{epoch:03d}/{args.max_epoch}]',
                    leave=False, ncols=100)
        for step in pbar:
            batch_ids = [all_ids[i] for i in order[step * args.batch_size:
                                                    (step + 1) * args.batch_size]]
            if not batch_ids: continue

            # Build batch tensors with augmentation
            Vs, Ns, GTs, Fs_nfs = [], [], [], []
            for d in batch_ids:
                V_aug, n_aug, GT_aug, nfs_aug = _augment(
                    d['V'], d['normals'], d['GT'], d['nfs'], args, rng)
                Vs.append(torch.from_numpy(V_aug).float())
                Ns.append(torch.from_numpy(n_aug).float())
                GTs.append(torch.from_numpy(GT_aug).float())
                if nfs_aug is not None:
                    Fs_nfs.append(torch.from_numpy(nfs_aug).float())
            # Pad to same N within batch (subsample → variable K)
            # Simpler: enforce same K per batch by picking smallest K
            K_min = min(v.shape[0] for v in Vs)
            Vs = torch.stack([v[:K_min] for v in Vs]).to(device)
            Ns = torch.stack([v[:K_min] for v in Ns]).to(device)
            nfs_feat = (torch.stack([v[:K_min] for v in Fs_nfs]).to(device)
                         if Fs_nfs else None)
            GT = torch.stack(GTs).to(device)                              # [B, J, 3]

            # Forward bind_pose_net only
            skin_input, _adain = model._prepare_feat(Vs, Ns, nfs_feat)
            _, pred = model._get_bind_pose(skin_input, adain_input=_adain)   # [B, J, 3]

            # Loss — non-helper joints only (no L_helper_residual / L_mirror)
            L_bind = F.mse_loss(pred.index_select(1, non_helper_idx),
                                 GT.index_select(1, non_helper_idx).detach())
            loss = L_bind

            optim.zero_grad()
            loss.backward()
            optim.step()

            running['L_bind_reg']   += L_bind.item()
            running['total']        += loss.item()
            cnt += 1
            global_step += 1
            pbar.set_postfix({'L_bind': f'{L_bind.item():.3e}'})
            tb.add_scalar('train/L_bind_step', L_bind.item(), global_step)

            # Per-batch bind pose vis — ~steps/10 cadence (matches train_hlbs.py).
            # `1 % _vis_interv` generalizes the train_hlbs.py `== 1` check so it
            # still fires when steps_per_epoch is small (_vis_interv == 1).
            if args.vis_train:
                _vis_interv = max(1, round(steps_per_epoch / 10))
                if step % _vis_interv == (1 % _vis_interv):
                    _log_train_batch_vis(model, batch_ids, vis_joint_sel,
                                         epoch, step, global_step, device,
                                         args.out_dir, tb)

        dt = time.time() - t0
        inv = 1.0 / max(cnt, 1)
        tb.add_scalar('train/L_bind_epoch', running['L_bind_reg'] * inv, epoch)
        tb.add_scalar('train/epoch_time_sec', dt, epoch)
        if epoch % args.log_interval == 0 or epoch == 1:
            msg = (f'[{epoch:03d}/{args.max_epoch}] '
                    f'L_bind={running["L_bind_reg"]*inv:.5e}  '
                    f'total={running["total"]*inv:.5e}  ({dt:.1f}s)')
            print(msg); log_file.write(msg + '\n'); log_file.flush()

        # ── Validation ──────────────────────────────────────────────
        if (epoch % args.val_interval == 0 or epoch == 1) and len(valid_ids) > 0:
            model.eval()
            val_dir = os.path.join(args.out_dir, 'valid_vis', f'{epoch:03d}')
            os.makedirs(val_dir, exist_ok=True)
            val_loss_sum = 0.0
            val_per_id = {}
            with torch.no_grad():
                for d in valid_ids:
                    V_t = torch.from_numpy(d['V']).float().unsqueeze(0).to(device)
                    N_t = torch.from_numpy(d['normals']).float().unsqueeze(0).to(device)
                    nfs_t = (torch.from_numpy(d['nfs']).float().unsqueeze(0).to(device)
                             if d['nfs'] is not None else None)
                    skin_input, _adain = model._prepare_feat(V_t, N_t, nfs_t)
                    _, pred = model._get_bind_pose(skin_input, adain_input=_adain)
                    GT_t = torch.from_numpy(d['GT']).float().unsqueeze(0).to(device)
                    val_loss = F.mse_loss(pred.index_select(1, non_helper_idx),
                                           GT_t.index_select(1, non_helper_idx)).item()
                    val_loss_sum += val_loss
                    val_per_id[d['name']] = val_loss

                # Vis — per-dataset tiled grid (non-helper joints unless
                # --vis_all_joints). GT(blue) + pred(red) overlay.
                _render_topo_grids(model, valid_ids, vis_joint_sel, device,
                                   val_dir, 'valid_vis', tb, epoch,
                                   title_prefix=f'[{epoch:03d}] valid ')

                # COMA/BIWI pred-only grid (still inside eval + no_grad)
                if args.vis_unseen and len(unseen_ids) > 0:
                    _log_unseen_vis(model, unseen_ids, vis_joint_sel,
                                    epoch, device, args.out_dir, tb)
            val_loss_avg = val_loss_sum / len(valid_ids)
            v_msg = f'[{epoch:03d}] VAL L_bind={val_loss_avg:.5e}  vis → {val_dir}/'
            print(v_msg); log_file.write(v_msg + '\n'); log_file.flush()
            tb.add_scalar('valid/L_bind_avg', val_loss_avg, epoch)
            for name, v in val_per_id.items():
                tb.add_scalar(f'valid_per_id/{name}', v, epoch)
            model.train()

        if epoch % args.save_interval == 0 or epoch == args.max_epoch:
            sd = {k: v for k, v in model.state_dict().items()
                   if k.startswith('bind_pose_net.') or k == 'helper_joint_idx_buf'}
            out_path = os.path.join(args.out_dir, f'bind_only_{epoch:03d}.pth')
            torch.save(sd, out_path)
            print(f'  saved → {out_path}')

    log_file.close()
    tb.close()
    print(f'Done. {args.out_dir}')


if __name__ == '__main__':
    main()
