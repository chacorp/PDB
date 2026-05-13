"""
vis_subsample.py — Visualize vertex subsample selection for ICT/MF mean meshes.

Replicates train_hlbs.py:_build_subsample_perm logic and renders:
  - mesh wireframe (light gray)
  - selected vertices (red)
  - landmark anchor vertices (blue, larger)

For each topology (ict, mf) and each mode (random, importance), saves a PNG
grid showing front view at the given ratio.

Usage:
    python tools/vis/vis_subsample.py \
        --out_dir maya_rig/hybrid/subsample_vis \
        --ratio 0.5 \
        --data_basedir /data/sihun
"""
import os
import sys
import argparse
import pickle
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

from utils.exp_utils import plateau_hat_points, plateau_hat_r


# ── Mesh loaders ──────────────────────────────────────────────────────────

def _load_ict_mean():
    from utils.remesh_utils import ICT_face_model
    m = ICT_face_model()
    return m.neutral_verts.astype(np.float32), m.faces.astype(np.int32)


def _load_mf_mean(data_basedir='/data/sihun'):
    pkl = os.path.join(data_basedir, 'multiface_align', 'mf_templates.pkl')
    if not os.path.exists(pkl):
        pkl = os.path.join(data_basedir, 'pca', 'multiface_align', 'mf_templates.pkl')
    with open(pkl, 'rb') as f:
        t = pickle.load(f)
    keys = [k for k in t if k != 'face']
    V = np.stack([np.array(t[k], dtype=np.float32) for k in keys]).mean(axis=0)
    F = np.array(t['face'], dtype=np.int32)
    return V.astype(np.float32), F


# ── DTU3D 73-point landmark groupings (1-indexed) + segment connectivity ──
# These define the brow/eye/nose/mouth contours for importance sampling.
# Each "segment" is a pair of landmarks that approximate a small piece of curve.
# Vertex-to-segment distance (point-to-line-segment) → plateau_hat → bump.

_DTU3D_BROW_R = (1, 3, 5, 7)            # right brow, outer→inner
_DTU3D_BROW_L = (9, 11, 13, 15)
_DTU3D_EYE_R  = (17, 19, 21, 23)
_DTU3D_EYE_L  = (25, 27, 29, 31)
_DTU3D_NOSE   = (36, 37, 43, 44)        # nose wing perimeter (L: 36-37, R: 43-44)
_DTU3D_MOUTH  = (40, 47, 48, 50, 52, 53, 54, 55, 56)
# 40    : Nose_Base_Rotate (above upper lip)
# 47,53 : mouth corners (R, L)
# 50,55 : top, bottom lip center
# 48,52 : upper lip side (R, L)
# 56,54 : lower lip side (R, L)

# Segments connecting consecutive landmarks along each contour.
# Closed loops for eye / mouth ring (last connects back to first).
_DTU3D_SEGMENTS = [
    # Brow (open curves, outer→inner)
    (1, 3), (3, 5), (5, 7),
    (9, 11), (11, 13), (13, 15),
    # Eye (closed loops around lid)
    (17, 19), (19, 21), (21, 23), (23, 17),
    (25, 27), (27, 29), (29, 31), (31, 25),
    # Nose wings (two short separate curves)
    (36, 37),
    (43, 44),
    # Mouth ring (closed loop) + 40-50 nose-base→top-lip connector
    (47, 48), (48, 50), (50, 52), (52, 53),
    (53, 54), (54, 55), (55, 56), (56, 47),
    (40, 50),
]


def _all_landmark_ids():
    """All 1-indexed DTU3D landmark IDs used in importance sampling."""
    return tuple(sorted(set(_DTU3D_BROW_R + _DTU3D_BROW_L + _DTU3D_EYE_R + _DTU3D_EYE_L
                            + _DTU3D_NOSE + _DTU3D_MOUTH)))


def _point_to_segments_min_dist(V, seg_starts, seg_ends):
    """Distance from each vertex to the closest line segment.

    V          : [N, 3]   query points
    seg_starts : [S, 3]   start of each segment
    seg_ends   : [S, 3]   end   of each segment

    Returns: [N] min distance to any of the S segments.
    """
    seg     = seg_ends - seg_starts                              # [S, 3]
    seg_len = (seg ** 2).sum(-1).clamp_min(1e-12)                # [S]
    diff    = V.unsqueeze(1) - seg_starts.unsqueeze(0)           # [N, S, 3]
    t       = (diff * seg.unsqueeze(0)).sum(-1) / seg_len.unsqueeze(0)   # [N, S]
    t       = t.clamp(0.0, 1.0)                                  # project to segment
    foot    = seg_starts.unsqueeze(0) + t.unsqueeze(-1) * seg.unsqueeze(0)   # [N, S, 3]
    return (V.unsqueeze(1) - foot).norm(dim=-1).min(dim=-1).values    # [N]


# Region groupings for centroid+radius ball plateau (region mode).
_REGION_GROUPS = [
    ('brow_R', _DTU3D_BROW_R),
    ('brow_L', _DTU3D_BROW_L),
    ('eye_R',  _DTU3D_EYE_R),
    ('eye_L',  _DTU3D_EYE_L),
    ('nose',   _DTU3D_NOSE),
    ('mouth',  _DTU3D_MOUTH),
]


def _region_bump(V, lm_to_vidx, region_lm_ids, falloff):
    """Centroid+radius ball plateau for one region.

    centroid = mean(region landmark positions)
    R        = max ||lm - centroid||   (the furthest landmark from centroid)
    For each vertex v:
        d = ||v - centroid||
        d ≤ R          → bump = 1   (plateau: all landmarks covered + interior)
        R < d < R+fall → smooth falloff (quintic smoothstep)
        d ≥ R+fall     → bump = 0
    Returns: [N] or None if not enough landmarks.
    """
    valid_vidx = [lm_to_vidx[lm] for lm in region_lm_ids if lm in lm_to_vidx]
    if len(valid_vidx) < 2:
        return None
    pts = V[torch.tensor(valid_vidx, dtype=torch.long)]              # [K, 3]
    centroid = pts.mean(dim=0)                                       # [3]
    R = (pts - centroid).norm(dim=-1).max().item()                   # scalar
    dist = (V - centroid).norm(dim=-1)                               # [N]
    return plateau_hat_r(dist, r0=R, r1=R + falloff)                 # [N] ∈ [0, 1]


def build_subsample_perm(V, ratio, mode, landmark_vidx=None,
                         r0=0.1, r1=0.3, alpha=4.0, kind='segment',
                         region_falloff=0.1, seed=42):
    """Replicates train_hlbs.py:_build_subsample_perm.

    - 'random'     : uniform, no anchor pinning.
    - 'importance' : oversample landmark-defined regions.
        * kind='point'   : ball at each landmark, max over centers (union).
        * kind='segment' : tube around segments connecting consecutive landmarks.
        * kind='region'  : per-region plateau — centroid+max-radius ball with
                           plateau interior, smooth falloff over `region_falloff`.
                           Filled blob per anatomical region (brow_L/R, eye_L/R,
                           nose, mouth).
        Falls back to random when landmark_vidx is None.
    """
    rng = torch.Generator().manual_seed(seed)
    src_v = torch.from_numpy(V).unsqueeze(0)               # [1, N, 3]
    N = V.shape[0]
    K = int(round(N * ratio))
    if K >= N or K <= 0:
        return None, None

    contour_used = np.array([], dtype=np.int64)

    if mode == 'importance':
        if landmark_vidx is None or len(landmark_vidx) == 0:
            print('   [importance] no landmark_vidx — falling back to random')
            perm = torch.randperm(N, generator=rng)[:K]
            return perm.numpy(), contour_used

        # Resolve all landmark IDs → mesh vertex indices (1-indexed → 0-indexed)
        all_lm_ids = _all_landmark_ids()
        lm_to_vidx = {}
        for lm in all_lm_ids:
            idx_in_array = lm - 1
            if 0 <= idx_in_array < len(landmark_vidx):
                v_idx = int(landmark_vidx[idx_in_array])
                if 0 <= v_idx < N:
                    lm_to_vidx[lm] = v_idx
        contour_used = np.array(sorted(lm_to_vidx.values()), dtype=np.int64)
        if not lm_to_vidx:
            print('   [importance] no valid landmark vidx — falling back to random')
            perm = torch.randperm(N, generator=rng)[:K]
            return perm.numpy(), contour_used
        V_t = src_v.squeeze(0)                              # [N, 3]

        if kind == 'point':
            # Ball query: bump per landmark center, max over centers.
            C = torch.stack([V_t[v] for v in lm_to_vidx.values()], dim=0)   # [K_lm, 3]
            bump = plateau_hat_points(V_t, C, r0=r0, r1=r1).max(dim=-1).values
        elif kind == 'region':
            # Per-region plateau ball (centroid + max landmark radius, then falloff).
            region_bumps = []
            for _name, lm_ids in _REGION_GROUPS:
                b_r = _region_bump(V_t, lm_to_vidx, lm_ids, region_falloff)
                if b_r is not None:
                    region_bumps.append(b_r)
            if not region_bumps:
                print('   [importance/region] no valid regions — falling back to random')
                perm = torch.randperm(N, generator=rng)[:K]
                return perm.numpy(), contour_used
            bump = torch.stack(region_bumps, dim=-1).max(dim=-1).values
        else:  # segment
            seg_pairs = [(a, b) for (a, b) in _DTU3D_SEGMENTS
                         if a in lm_to_vidx and b in lm_to_vidx]
            if not seg_pairs:
                print('   [importance/segment] no valid segments — falling back to random')
                perm = torch.randperm(N, generator=rng)[:K]
                return perm.numpy(), contour_used
            seg_starts = torch.stack([V_t[lm_to_vidx[a]] for a, _ in seg_pairs], dim=0)
            seg_ends   = torch.stack([V_t[lm_to_vidx[b]] for _, b in seg_pairs], dim=0)
            dist = _point_to_segments_min_dist(V_t, seg_starts, seg_ends)
            bump = plateau_hat_r(dist, r0=r0, r1=r1)        # [N] ∈ [0, 1]

        w = 1.0 + alpha * bump                              # contour ≈ (1+α)×
        perm = torch.multinomial(w, K, replacement=False, generator=rng)
    else:  # random
        perm = torch.randperm(N, generator=rng)[:K]

    return perm.numpy(), contour_used


# ── Rendering ────────────────────────────────────────────────────────────

def _project_xy(V):
    """Front view: XY plane, +x = right, +y = up. ICT/MF both face +z."""
    return V[:, 0], V[:, 1]


def render_ax(ax, V, F, selected_vidx, contour_vidx, title,
              point_size=4.0, label_landmarks=None, landmark_vidx_arr=None):
    """If label_landmarks is given (1-indexed iterable), annotate those landmark
    indices on the rendered mesh (uses landmark_vidx_arr for vertex lookup)."""
    x, y = _project_xy(V)

    # Mesh wireframe as filled polygons with very low alpha
    polys = V[F][:, :, :2]                                  # [n_faces, 3, 2]
    pc = PolyCollection(polys, facecolors='#dddddd', edgecolors='#bbbbbb',
                        linewidths=0.1, alpha=0.4, zorder=1)
    ax.add_collection(pc)

    # Unselected vertices (very faint, for context)
    selected_mask = np.zeros(V.shape[0], dtype=bool)
    selected_mask[selected_vidx] = True
    unsel = ~selected_mask
    ax.scatter(x[unsel], y[unsel], s=0.6, c='#999999', alpha=0.35, zorder=2,
               linewidths=0)

    # Selected (red, prominent)
    ax.scatter(x[selected_vidx], y[selected_vidx], s=point_size, c='#e63946',
               alpha=0.85, zorder=3, linewidths=0)

    # Contour-landmark centers (blue, larger — visible when importance mode used)
    if len(contour_vidx) > 0:
        ax.scatter(x[contour_vidx], y[contour_vidx], s=point_size * 4.0, c='#1d4ed8',
                   alpha=1.0, zorder=4, edgecolors='white', linewidths=0.4)

    # Optional: annotate DTU3D landmark indices for verification
    if label_landmarks is not None and landmark_vidx_arr is not None:
        for lm in label_landmarks:
            idx0 = lm - 1
            if 0 <= idx0 < len(landmark_vidx_arr):
                v = int(landmark_vidx_arr[idx0])
                if 0 <= v < V.shape[0]:
                    ax.annotate(str(lm), (x[v], y[v]),
                                fontsize=5.5, color='#0c4a6e', zorder=5,
                                ha='center', va='center',
                                bbox=dict(boxstyle='circle,pad=0.1',
                                          fc='#ffffffcc', ec='none'))

    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    n_sel = len(selected_vidx); n_c = len(contour_vidx)
    extra = f' · contour-lm={n_c}' if n_c else ''
    ax.set_title(f'{title}\nselected={n_sel} ({n_sel/V.shape[0]*100:.0f}%){extra}',
                 fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='maya_rig/hybrid/subsample_vis')
    ap.add_argument('--ratio', type=float, default=0.5)
    ap.add_argument('--landmark_vidx_dir', default='maya_rig/hybrid')
    ap.add_argument('--data_basedir', default='/data/sihun')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--r0', type=float, default=0.1,
                    help='inner (full-weight) radius of contour bump')
    ap.add_argument('--r1', type=float, default=0.3,
                    help='outer (zero-weight) radius of contour bump')
    ap.add_argument('--alpha', type=float, default=4.0,
                    help='contour oversample factor (weight = 1 + α·bump)')
    ap.add_argument('--region_falloff', type=float, default=0.1,
                    help='[region mode] absolute falloff distance beyond region radius')
    ap.add_argument('--label_landmarks', action='store_true',
                    help='Annotate DTU3D landmark indices used in importance mode')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    topos = []
    print('[load] ICT mean mesh ...')
    V_ict, F_ict = _load_ict_mean()
    print(f'   ICT V={V_ict.shape[0]}, F={F_ict.shape[0]}')
    topos.append(('ict', V_ict, F_ict))

    print('[load] MF mean mesh ...')
    try:
        V_mf, F_mf = _load_mf_mean(args.data_basedir)
        print(f'   MF V={V_mf.shape[0]}, F={F_mf.shape[0]}')
        topos.append(('mf', V_mf, F_mf))
    except Exception as e:
        print(f'   [skip MF] {e}')

    # 4 columns: random | importance (point/ball) | importance (segment/curve) | importance (region/plateau)
    cols = [('random', None), ('importance', 'point'),
            ('importance', 'segment'), ('importance', 'region')]
    fig, axes = plt.subplots(len(topos), len(cols),
                             figsize=(4.0 * len(cols), 4.5 * len(topos)))
    if len(topos) == 1:
        axes = axes[None, :]

    for r, (topo, V, F) in enumerate(topos):
        lv_path = os.path.join(args.landmark_vidx_dir, f'landmark_vidx_{topo}.npy')
        landmark_vidx = None
        if os.path.exists(lv_path):
            landmark_vidx = np.load(lv_path).astype(np.int64)
            print(f'   [{topo}] landmark_vidx loaded: {len(landmark_vidx)} from {lv_path}')
        else:
            print(f'   [{topo}] WARNING: no {lv_path}, importance falls back to random')

        for c, (mode, kind) in enumerate(cols):
            perm, contour_used = build_subsample_perm(
                V, args.ratio, mode, landmark_vidx=landmark_vidx,
                r0=args.r0, r1=args.r1, alpha=args.alpha,
                kind=(kind if kind else 'segment'),
                region_falloff=args.region_falloff,
                seed=args.seed)
            if perm is None:
                axes[r, c].text(0.5, 0.5, f'K>=N (ratio={args.ratio})',
                                ha='center', va='center')
                continue
            _labels = list(_all_landmark_ids()) if (args.label_landmarks and mode == 'importance') else None
            kind_tag = f' [{kind}]' if kind else ''
            render_ax(axes[r, c], V, F, perm, contour_used,
                      title=f'{topo.upper()}  |  {mode}{kind_tag}',
                      label_landmarks=_labels, landmark_vidx_arr=landmark_vidx)

    fig.suptitle(
        f'Vertex subsample (ratio={args.ratio}, r0={args.r0}, r1={args.r1}, '
        f'α={args.alpha}, seed={args.seed})\n'
        f'red=selected · blue=landmark centers (point: union of balls / segment: curves) · gray=unselected',
        fontsize=10, y=1.00)
    fig.tight_layout()
    out_path = os.path.join(
        args.out_dir,
        f'subsample_r{args.ratio:.2f}_r0{args.r0}_r1{args.r1}_a{args.alpha}_s{args.seed}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nsaved → {out_path}')


if __name__ == '__main__':
    main()
