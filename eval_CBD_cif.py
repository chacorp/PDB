import os
import glob
import json
import yaml
import random
import itertools
import re

import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
import igl

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataloader_CBD import (
    EvalDataset,
    CBD_collate_wrapper_eval,
)

from utils.matplotlib_rnd import plot_image_array
from utils.ckpt_utils import *
from utils.remesh_utils import calc_norm_torch

from models.NGBC import NeuralGeneralizedBarycentricCoordinate

# Reuse dataset machinery already built and validated by the sibling eval
# scripts instead of re-deriving per-dataset file paths here:
#   - NeutralEvalDataset (eval_CBD_lp.py): neutral-template pool for any of
#     voca/biwi/mf_SEN/coma/mf_ROM/ict, used for Part I/II/III's identity pool.
#   - GCEvalDataset / GC_collate_wrapper / vertex_area_weights (eval_CBD_gc.py):
#     per-(ICT identity, one-hot blendshape) items with the identity-independent
#     analytic exp_disp, used for Part IV's paired semantic validation, and the
#     surface-area vertex weighting used throughout Part I.
from eval_CBD_lp import NeutralEvalDataset
from eval_CBD_gc import GCEvalDataset, GC_collate_wrapper, vertex_area_weights, basis_group

import torch.multiprocessing as mp

from pathlib import Path
__abs_path__ = str(Path(__file__).parents[0].absolute())


# ---------------------------------------------------------------------------
# Small containers
# ---------------------------------------------------------------------------

class _Identity:
    """One evaluated identity: neutral mesh + cached per-identity quantities."""
    __slots__ = ['name', 'dataset', 'neu_vert', 'neu_norm', 'faces', 'n_vert']

    def __init__(self, name, dataset, neu_vert, neu_norm, faces):
        self.name = name
        self.dataset = dataset          # 'ict' | 'voca' | 'biwi' | 'mf_SEN' | 'coma' | 'mf_ROM'
        self.neu_vert = neu_vert        # [1, N, 3] (device)
        self.neu_norm = neu_norm        # [1, N, 3] (device)
        self.faces = faces              # [F, 3] long (device)
        self.n_vert = neu_vert.shape[1]


def direction_tag(a, b):
    """Dataset-direction bucket without relabeling every non-ICT set as Multiface."""
    def tag(name):
        if name == 'ict':
            return 'ICT'
        if name.startswith('mf_'):
            return 'MF'
        return name.upper()
    da = tag(a.dataset)
    db = tag(b.dataset)
    return f'{da}->{db}'


# ---------------------------------------------------------------------------
# Geometry / distance helpers (Part I)
# ---------------------------------------------------------------------------

def normalize_coords(vert):
    """
    Rigid (centroid) alignment + scale normalization for a single mesh
    (plan sec. 4): centers at the centroid and divides by the RMS distance
    to it. This assumes -- as the plan explicitly requires (sec. 4, not
    non-rigid registration) -- that the datasets already share a common
    global orientation convention (true here: all identities are loaded
    through this codebase's existing aligned template pickles / ICT rig,
    the same assumption every other eval_CBD_*.py cross-identity script
    already relies on).

    Args:
        vert (torch.tensor): [N, 3]
    Returns:
        [N, 3] centered + scale-normalized vertex positions
    """
    center = vert.mean(dim=0, keepdim=True)
    centered = vert - center
    scale = centered.pow(2).sum(-1).mean().sqrt().clamp_min(1e-8)
    return centered / scale


def sliced_w1_cost_matrix(pos_a, w_a, pos_b, w_b, directions, l_chunk=32):
    """
    Cross-mesh column cost matrix via Sliced-Wasserstein distance (plan sec. 5).

    Exploits that, for a fixed random projection direction, the *order* of a
    mesh's vertices along that direction does not depend on which column k is
    being queried -- only the per-vertex weight assigned to each column does.
    So vertices are sorted once per projection, and the weighted 1D CDF (a
    step function with those fixed breakpoints) is evaluated for every column
    at once, instead of re-sorting per column. This keeps cost O(n_proj *
    (Na+Nb) * Ka * Kb) instead of O(n_proj * (Na+Nb) * Ka * Kb * log(...))
    with a much larger constant, and lets the K-column loop be replaced by
    plain batched tensor ops (chunked over the candidate/target column axis
    `l_chunk` to bound peak memory).

    Args:
        pos_a: [Na, 3] rigid+scale-normalized vertex positions (mesh a)
        w_a:   [Na, Ka] per-column weights (need not sum to 1; renormalized
               internally per column so this also works on a vertex subsample)
        pos_b: [Nb, 3]
        w_b:   [Nb, Kb]
        directions: [P, 3] shared unit projection directions. The same set
            must be used for every identity pair so pair scores are comparable.
        l_chunk (int): candidate-column chunk size, bounds peak memory
    Returns:
        [Ka, Kb] cost matrix, C[k, l] = mean_proj W1(rho_a_k, rho_b_l)
    """
    device = pos_a.device
    eps = 1e-12
    Ka = w_a.shape[1]
    Kb = w_b.shape[1]

    w_a = w_a / (w_a.sum(0, keepdim=True) + eps)
    w_b = w_b / (w_b.sum(0, keepdim=True) + eps)

    directions = directions.to(device)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    n_proj = directions.shape[0]

    proj_a_all = pos_a @ directions.T  # [Na, n_proj]
    proj_b_all = pos_b @ directions.T  # [Nb, n_proj]

    cost = torch.zeros(Ka, Kb, device=device)
    for p in range(n_proj):
        sa, idx_a = torch.sort(proj_a_all[:, p])
        sb, idx_b = torch.sort(proj_b_all[:, p])
        cdf_a = torch.cumsum(w_a[idx_a], dim=0)  # [Na, Ka]
        cdf_b = torch.cumsum(w_b[idx_b], dim=0)  # [Nb, Kb]

        merged, _ = torch.sort(torch.cat([sa, sb]))
        widths = torch.diff(merged, append=merged[-1:])  # last width is 0

        idx_in_a = torch.searchsorted(sa, merged, right=True) - 1
        idx_in_b = torch.searchsorted(sb, merged, right=True) - 1
        valid_a = idx_in_a >= 0
        valid_b = idx_in_b >= 0

        F_a = torch.zeros(merged.shape[0], Ka, device=device)
        F_a[valid_a] = cdf_a[idx_in_a[valid_a]]
        F_b_full = torch.zeros(merged.shape[0], Kb, device=device)
        F_b_full[valid_b] = cdf_b[idx_in_b[valid_b]]

        for l0 in range(0, Kb, l_chunk):
            l1 = min(l0 + l_chunk, Kb)
            diff = (F_a.unsqueeze(-1) - F_b_full[:, l0:l1].unsqueeze(1)).abs()  # [n_merge, Ka, chunk]
            cost[:, l0:l1] += (diff * widths.view(-1, 1, 1)).sum(0)

    cost /= n_proj
    return cost


def save_heatmap(mat_np, path, title, xlabel='column (mesh b)', ylabel='column (mesh a)'):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat_np, cmap='viridis', aspect='auto')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def Options():
    parser = argparse.ArgumentParser(
        description='Cross-Identity Factor (CIF) consistency evaluation for NGBC '
                     '(see eval_CBD_cp/cross_id_factor_consist.md, Parts I-IV)'
    )
    parser.add_argument('-c', '--config', default='config/train_CBD.yml', help='config file path')
    parser.add_argument("--device",       type=str,   default="cuda:0")

    parser.add_argument("--log_dir",      type=str,   default="eval_CBD_cif")

    parser.add_argument("--version",      type=int,   default=5,
                        help='model version (only 5: NGBC is supported -- see eval_CBD_cp.py '
                             'for why: this needs the identity-coordinate / expression-code '
                             'split exposed by predict_coordinate() / retarget_animation())')

    parser.add_argument("--last_activation", default=None,
                        choices=["relu", "elu", "softmax", "softplus", "none", "sqrelu"])
    parser.add_argument("--no_pou", dest='no_pou', action='store_true')
    parser.set_defaults(no_pou=None)

    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--ckpt",         type=str,   default=None)
    parser.add_argument("--continue_ckpt", dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)
    parser.add_argument("--start_epoch",  type=int,   default=0)
    parser.add_argument("--align_latent", dest='align_latent', action='store_true')
    parser.set_defaults(align_latent=None)
    parser.add_argument("--allow_partial_load", action='store_true',
                        help='allow checkpoint keys required by the evaluation model to be missing; '
                             'disabled by default because partial loading invalidates the metrics')
    parser.set_defaults(allow_partial_load=False)

    # ---- identity pool (Part I / II / III) --------------------------------
    parser.add_argument("--datasets", type=str, nargs='+', default=['ict', 'mf_SEN'],
                        help='datasets contributing unseen target identities to the pool '
                             '(any of voca, biwi, mf_SEN, coma, mf_ROM, ict)')
    parser.add_argument("--n_ict_identity", type=int, default=5,
                        help='number of ICT identities in the pool (also used for Part IV)')
    parser.add_argument("--n_id_per_dataset", type=int, default=5,
                        help='number of identities per non-ICT dataset in --datasets')

    # ---- source expression sequence (Part II / III route) -----------------
    parser.add_argument("--src_name", type=str, default='mf_SEN',
                        help='dataset supplying the real driving expression sequence')
    parser.add_argument("--n_src_frames", type=int, default=30,
                        help='number of source frames randomly sampled (fixed seed) for '
                             'Part II (CP coherence) and Part III (route consistency)')

    # ---- Part I: weight-column spatial coherence ---------------------------
    parser.add_argument("--n_columns_sample", type=int, default=64,
                        help='number of control-point columns randomly sampled (fixed seed, '
                             'shared across all identities) for the column cost matrix / '
                             'retrieval metrics; -1 uses all --num_cage_v columns (slow)')
    parser.add_argument("--sw_n_proj", type=int, default=16,
                        help='number of random projection directions for the Sliced-Wasserstein '
                             'column cost matrix')
    parser.add_argument("--sw_n_vert_sample", type=int, default=1500,
                        help='per-identity vertex subsample size used for the SW cost matrix '
                             '(Monte-Carlo approximation of the continuous surface distribution; '
                             '-1 uses all vertices, slow for high-res meshes)')
    parser.add_argument("--sw_chunk", type=int, default=32,
                        help='candidate-column chunk size for the SW cost matrix (memory bound)')
    parser.add_argument("--active_mass_ratio", type=float, default=0.01,
                        help='a column is "active" for an identity if its area-weighted mass '
                             'exceeds this fraction of that identity\'s mean column mass')

    # ---- Part II: control-point index coherence ----------------------------
    parser.add_argument("--n_pert_columns", type=int, default=24,
                        help='number of control-point columns (sampled from --n_columns_sample\'s '
                             'index set) perturbed for the finite-difference Jacobian probe')
    parser.add_argument("--pert_eta", type=float, default=-1.0,
                        help='perturbation magnitude eta for the control-point Jacobian probe, in '
                             'raw cage (key_d) units; <=0 (default) auto-derives eta from the RMS '
                             'per-element magnitude of real (v_s - v_s^0) differences seen in Part '
                             'II\'s probe frames, scaled by --pert_eta_scale. A hand-picked constant '
                             'eta is not meaningful here: a single control point\'s generalized- '
                             'barycentric weight column spreads its mass over the whole mesh, so a '
                             '"small"-looking eta in cage units can correspond to a vertex-space '
                             'nudge far below what the encoder was ever trained to respond to')
    parser.add_argument("--pert_eta_scale", type=float, default=1.0,
                        help='multiplier applied to the auto-derived eta (only used when --pert_eta <= 0)')

    # ---- misc / scope control ----------------------------------------------
    parser.add_argument("--skip_part1", action='store_true', help='skip weight-column spatial coherence')
    parser.add_argument("--skip_part2", action='store_true', help='skip control-point index coherence')
    parser.add_argument("--skip_part3", action='store_true', help='skip joint factor compatibility')
    parser.add_argument("--skip_part4", action='store_true', help='skip ICT paired semantic validation')
    parser.add_argument("--no_vis", action='store_true', help='skip example visualizations')
    parser.add_argument("--quick", action='store_true',
                        help='override sample sizes with small values for a fast smoke test')
    parser.add_argument("--run_name", type=str, default=None,
                        help='optional tag appended to the result directory')

    parser.add_argument("--tb", action='store_true')
    parser.set_defaults(is_train=True)

    args = parser.parse_args()
    if args.n_ict_identity < 1 or args.n_id_per_dataset < 1:
        parser.error('identity counts must be positive')
    if args.n_columns_sample == 0 or args.n_pert_columns < 1:
        parser.error('--n_columns_sample must be -1 or positive, and --n_pert_columns must be positive')
    if args.sw_n_proj < 1 or args.sw_chunk < 1:
        parser.error('--sw_n_proj and --sw_chunk must be positive')
    if args.active_mass_ratio < 0:
        parser.error('--active_mass_ratio must be non-negative')
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('--datasets must not contain duplicates')
    return args


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer():
    def __init__(self, opts):
        self.opts = opts
        self.set_seed(self.opts)
        self.device = opts.device
        self.rng = np.random.RandomState(opts.seed)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(opts.seed)

        if opts.version != 5:
            raise NotImplementedError(
                'eval_CBD_cif.py only supports version=5 (NGBC): cross-identity factor '
                'consistency relies on NeuralGeneralizedBarycentricCoordinate.predict_coordinate() '
                '/ .retarget_animation() exposing the identity-coordinate / expression-code split.'
            )
        if self.opts.out_type not in (0, 1):
            raise ValueError(
                'CIF evaluation currently supports out_type 0 (delta) and 1 '
                '(absolute control positions) only; transform-control modes require '
                'a group-wise perturbation definition.'
            )
        if self.opts.last_activation == 'sqrelu':
            raise ValueError(
                'NGBC version 5 does not pass a sqrelu option to its coordinate encoder; '
                'evaluating it as sqrelu would silently instantiate a different activation.'
            )

        if opts.quick:
            opts.n_ict_identity = min(opts.n_ict_identity, 2)
            opts.n_id_per_dataset = min(opts.n_id_per_dataset, 2)
            opts.n_columns_sample = min(opts.n_columns_sample, 16) if opts.n_columns_sample > 0 else 16
            opts.sw_n_proj = min(opts.sw_n_proj, 4)
            opts.sw_n_vert_sample = min(opts.sw_n_vert_sample, 500) if opts.sw_n_vert_sample > 0 else 500
            opts.n_src_frames = min(opts.n_src_frames, 5)
            opts.n_pert_columns = min(opts.n_pert_columns, 8)

        last_act_list = ["relu", "elu", "softmax", "softplus", "none", "sqrelu"]
        last_act_list = [self.opts.last_activation == l_act for l_act in last_act_list]

        self.model = NeuralGeneralizedBarycentricCoordinate(
            opts, num_layers=4,
            num_cage_vertices=self.opts.num_cage_v,
            use_exp_recon=False,
            use_shp_recon=False,
            use_shp=False,
            use_relu=last_act_list[0],
            use_elu=last_act_list[1],
            use_softmax=last_act_list[2],
            use_softplus=last_act_list[3],
            no_activation=last_act_list[4],
            use_least_N_on_V=False,
            is_train=True,
            use_pou=not self.opts.no_pou,
            device=self.device,
            hid_dim=128 if self.opts.align_latent else 256,
        )
        self.load_weight()

        self.K = self.opts.num_cage_v
        self._kw_cache = {}    # identity name -> key_weight [1, N, K]
        self._mass_cache = {}  # identity name -> (active_mask [K] bool, col_mass [K])

        n_sample = self.opts.n_columns_sample if self.opts.n_columns_sample > 0 else self.K
        n_sample = min(n_sample, self.K)
        self.sampled_columns = torch.from_numpy(
            self.rng.choice(self.K, size=n_sample, replace=False)
        ).long().to(self.device)
        # self.pert_columns (Part II Jacobian probe) is selected later, inside
        # part2_cp_index_coherence, once the identity pool is loaded -- it must
        # be restricted to columns active for every probed target (see there
        # for why perturbing an inactive column is a meaningless, trivially-zero
        # probe rather than a real disentanglement signal).

    def load_weight(self):
        if self.opts.ckpt:
            print(self.opts.ckpt)
            if self.opts.continue_ckpt:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, f"*_{self.opts.start_epoch:03d}.pth"))[0]
            else:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
            print(f"Loading... {ckpt}")
            ckpt_dict = torch.load(ckpt, map_location=self.device)
            incompatible = self.model.load_state_dict(ckpt_dict, strict=False)
            if incompatible.unexpected_keys:
                print(
                    'WARNING: checkpoint keys unused by the evaluation model:\n  ' +
                    '\n  '.join(incompatible.unexpected_keys)
                )
            if incompatible.missing_keys:
                message = (
                    'Checkpoint is missing parameters required by the evaluation model:\n  ' +
                    '\n  '.join(incompatible.missing_keys)
                )
                if self.opts.allow_partial_load:
                    print('WARNING: ' + message)
                else:
                    raise RuntimeError(
                        message + '\nUse --allow_partial_load only if this is intentional.'
                    )
        else:
            print('no ckpt found, training from scratch!')

    # ------------------------------------------------------------------
    # shared helpers
    # ------------------------------------------------------------------
    def _apply_key_d(self, key_weight, key_d, tgt_neu_vert):
        """
        Applies an already-encoded control-point displacement `key_d` through
        a (possibly different identity's) `key_weight`, mirroring exactly what
        `retarget_animation` does internally after encoding (models/NGBC.py
        lines ~510-517) -- needed here because several Part II/III/IV
        quantities require applying a *cached* v to a target without
        re-deriving v from a real mesh pair (route consistency's via-b leg,
        the perturbation Jacobian probe, and ICT cross-identity GT reconstruction).
        """
        tgt_def_v = torch.einsum('bnc,bci->bni', key_weight, key_d)
        if self.model.use_full_vertex:
            return tgt_def_v
        return tgt_def_v + tgt_neu_vert

    @torch.no_grad()
    def _get_key_weight(self, ident):
        if ident.name not in self._kw_cache:
            kw = self.model.predict_coordinate(ident.neu_vert, ident.neu_norm)  # [1, N, K]
            self._kw_cache[ident.name] = kw

            area_np = vertex_area_weights(
                ident.neu_vert[0].cpu().numpy(), ident.faces.cpu().numpy()
            ).astype(np.float32)
            area = torch.from_numpy(area_np).to(self.device)  # [N]
            col_mass = (area[:, None] * kw[0]).sum(0)  # [K]
            active = col_mass > (self.opts.active_mass_ratio * col_mass.mean())
            self._mass_cache[ident.name] = (active, col_mass, area)
        return self._kw_cache[ident.name]

    def _get_mass_info(self, ident):
        self._get_key_weight(ident)
        return self._mass_cache[ident.name]

    # ------------------------------------------------------------------
    # identity pool
    # ------------------------------------------------------------------
    def _load_identity_pool(self):
        pool = []
        for name in self.opts.datasets:
            n_cap = self.opts.n_ict_identity if name == 'ict' else self.opts.n_id_per_dataset
            ds = NeutralEvalDataset(data_name=name, toggle=False)
            items = ds.items[:n_cap]
            for template_np, faces_np, id_name in items:
                normal_np = igl.per_vertex_normals(template_np, faces_np)
                pool.append(_Identity(
                    name=f'{name}_{id_name}',
                    dataset=name,
                    neu_vert=torch.tensor(template_np).float()[None].to(self.device),
                    neu_norm=torch.tensor(normal_np).float()[None].to(self.device),
                    faces=torch.tensor(faces_np).long().to(self.device),
                ))
        return pool

    def _sample_src_probe_frames(self):
        """A fixed, reproducible subset of --src_name frames, batch_size=1 each,
        shared between Part II (CP coherence) and Part III (route consistency)."""
        src_dataset = EvalDataset(data_name=self.opts.src_name, toggle=False)
        n = min(self.opts.n_src_frames, len(src_dataset))
        if n < 1:
            raise RuntimeError(f'No source probe frame is available for dataset {self.opts.src_name}')
        indices = sorted(self.rng.choice(len(src_dataset), size=n, replace=False).tolist())
        subset = torch.utils.data.Subset(src_dataset, indices)
        loader = torch.utils.data.DataLoader(
            subset, batch_size=1, collate_fn=partial(CBD_collate_wrapper_eval, device=self.device)
        )
        return src_dataset, list(loader)

    # ==================================================================
    # Part I -- Weight-column spatial coherence
    # ==================================================================
    @torch.no_grad()
    def part1_weight_column_coherence(self, pool):
        print('\n[Part I] Weight-column spatial coherence...')
        n_sample = self.sampled_columns.shape[0]
        # One fixed projection set is shared across every identity pair.
        # Otherwise pair-to-pair differences include Monte-Carlo direction
        # noise and C(a,b) need not equal C(b,a)^T.
        directions = torch.randn(
            self.opts.sw_n_proj, 3, generator=self.torch_gen, device=self.device
        )
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # per-identity normalized positions / subsampled column weights ------
        cache = {}
        for ident in pool:
            kw = self._get_key_weight(ident)[0, :, self.sampled_columns]  # [N, n_sample]
            active, col_mass, area = self._get_mass_info(ident)
            pos = normalize_coords(ident.neu_vert[0])  # [N, 3]

            N = pos.shape[0]
            n_v = self.opts.sw_n_vert_sample if self.opts.sw_n_vert_sample > 0 else N
            n_v = min(n_v, N)
            v_idx = torch.from_numpy(self.rng.choice(N, size=n_v, replace=False)).long().to(self.device)

            pos_sub = pos[v_idx]                       # [n_v, 3]
            w_sub = area[v_idx, None] * kw[v_idx]       # [n_v, n_sample], unnormalized mass
            sampled_mass_valid = w_sub.sum(0) > 1e-12

            # Gram consistency (encoder-free, full-K, plan sec. 9)
            kw_full = self._get_key_weight(ident)[0]    # [N, K]
            min_weight = kw_full.min().item()
            negative_ratio = (kw_full < -1e-8).float().mean().item()
            pou_max_error = (kw_full.sum(-1) - 1.0).abs().max().item()
            if min_weight < -1e-8:
                raise ValueError(
                    f'Part I requires non-negative column measures, but {ident.name} has '
                    f'min(W)={min_weight:.4e} and negative ratio={negative_ratio:.4e}. '
                    'A probability-measure Wasserstein distance is not defined for signed weights.'
                )
            G = kw_full.T @ (area[:, None] * kw_full)   # [K, K]
            D = torch.clamp(torch.diagonal(G), min=1e-12)
            Dinv = D.rsqrt()
            G_bar = G * Dinv[:, None] * Dinv[None, :]

            cache[ident.name] = dict(
                pos_sub=pos_sub, w_sub=w_sub,
                active_sampled=active[self.sampled_columns] & sampled_mass_valid,
                active_full=active,
                sampled_mass_valid=sampled_mass_valid,
                G_bar=G_bar,
                diagnostics=dict(
                    min_weight=min_weight,
                    negative_weight_ratio=negative_ratio,
                    pou_max_error=pou_max_error,
                    sampled_column_mass_valid_ratio=sampled_mass_valid.float().mean().item(),
                ),
            )

        pair_records = []
        cost_matrices = {}
        for a, b in itertools.combinations(pool, 2):
            ca, cb = cache[a.name], cache[b.name]
            C = sliced_w1_cost_matrix(
                ca['pos_sub'], ca['w_sub'], cb['pos_sub'], cb['w_sub'],
                directions=directions, l_chunk=self.opts.sw_chunk,
            )  # [n_sample, n_sample]
            cost_matrices[(a.name, b.name)] = C
            cost_matrices[(b.name, a.name)] = C.T

            def append_direction(src, tgt, C_dir, c_src, c_tgt):
                # Columns with (near-)zero sampled mass get divided by the
                # +eps guard inside sliced_w1_cost_matrix and renormalize
                # into an arbitrary, degenerate point-mass distribution --
                # comparing distances to/from such a column is meaningless
                # and can spuriously produce a near-zero cost against another
                # equally degenerate column, which then wins argmin/blows up
                # the margin denominator. Candidates that fail the mass
                # check are therefore never retrievable here, and queries
                # that fail it are dropped from every "all-columns" stat
                # below (their own true location isn't well-defined either).
                valid_tgt = c_tgt['sampled_mass_valid']
                valid_src = c_src['sampled_mass_valid']
                C_dir = C_dir.masked_fill(~valid_tgt[None, :], float('inf'))

                diag = torch.diagonal(C_dir)
                rank_all = (C_dir < diag[:, None]).sum(-1) + 1
                index = torch.arange(n_sample, device=self.device)
                top1_all_vec = C_dir.argmin(-1) == index
                top5_all_vec = rank_all <= min(5, int(valid_tgt.sum().item()))
                off_diag = C_dir.masked_fill(
                    torch.eye(n_sample, device=self.device, dtype=torch.bool),
                    float('inf'),
                ).min(-1).values
                margin_vec = (off_diag - diag) / off_diag.clamp_min(1e-12)

                valid_query = valid_src & valid_tgt & torch.isfinite(diag)
                if valid_query.any():
                    top1_all = top1_all_vec[valid_query].float().mean().item()
                    top5_all = top5_all_vec[valid_query].float().mean().item()
                    mrr_all = (1.0 / rank_all[valid_query].float()).mean().item()
                    margin_all = margin_vec[valid_query].mean().item()
                else:
                    top1_all = top5_all = mrr_all = margin_all = float('nan')

                both_active = c_src['active_sampled'] & c_tgt['active_sampled']
                if both_active.any():
                    # Active retrieval excludes inactive target columns from
                    # the candidate set instead of only filtering queries.
                    C_active = C_dir.masked_fill(
                        ~c_tgt['active_sampled'][None, :], float('inf')
                    )
                    rank_active = (C_active < diag[:, None]).sum(-1) + 1
                    top1_active_vec = C_active.argmin(-1) == index
                    top5_active_vec = rank_active <= min(
                        5, int(c_tgt['active_sampled'].sum().item())
                    )
                    top1_active = top1_active_vec[both_active].float().mean().item()
                    top5_active = top5_active_vec[both_active].float().mean().item()
                    mrr_active = (1.0 / rank_active[both_active].float()).mean().item()
                else:
                    top1_active = top5_active = mrr_active = float('nan')

                common_full = c_src['active_full'] & c_tgt['active_full']
                gram_active = (
                    F.mse_loss(
                        c_src['G_bar'][common_full][:, common_full],
                        c_tgt['G_bar'][common_full][:, common_full],
                    ).item()
                    if common_full.any() else float('nan')
                )
                pair_records.append(dict(
                    a=src.name, b=tgt.name, direction=direction_tag(src, tgt),
                    top1=top1_all,
                    top5=top5_all,
                    mrr=mrr_all,
                    margin=margin_all,
                    top1_active=top1_active,
                    top5_active=top5_active,
                    mrr_active=mrr_active,
                    active_both_ratio=both_active.float().mean().item(),
                    gram_err=F.mse_loss(c_src['G_bar'], c_tgt['G_bar']).item(),
                    gram_err_active=gram_active,
                ))

            append_direction(a, b, C, ca, cb)
            append_direction(b, a, C.T, cb, ca)

        active_ratios = {name: cache[name]['active_sampled'].float().mean().item() for name in cache}
        weight_diagnostics = {name: cache[name]['diagnostics'] for name in cache}

        if not self.opts.no_vis and len(pair_records) > 0:
            (a0, b0) = (pool[0].name, pool[1].name) if len(pool) > 1 else (pool[0].name, pool[0].name)
            save_heatmap(
                cost_matrices[(a0, b0)].cpu().numpy(),
                os.path.join(self.opts.log_dir, 'img', 'part1_cost_matrix_example.png'),
                f'Cross-mesh column cost matrix C[k,l]  ({a0} -> {b0})',
            )

        summary = self._aggregate(pair_records, ['top1', 'top5', 'mrr', 'margin',
                                                  'top1_active', 'top5_active', 'mrr_active',
                                                  'active_both_ratio', 'gram_err',
                                                  'gram_err_active'])
        summary['active_column_ratio_per_identity'] = active_ratios
        summary['active_column_ratio_mean'] = float(np.mean(list(active_ratios.values())))
        summary['weight_diagnostics_per_identity'] = weight_diagnostics
        return {'pairwise': pair_records, 'summary': summary}

    # ==================================================================
    # Part II -- Control-point index coherence
    # ==================================================================
    @torch.no_grad()
    def part2_cp_index_coherence(self, pool, probe_frames):
        print('\n[Part II] Control-point index coherence...')
        eps = 1e-12
        records = []
        ncp_denoms = []  # per-frame MSE(v_s, v_s^0) -- also used to auto-scale the Jacobian eta below

        for batch in tqdm(probe_frames, ncols=100, desc='CP coherence'):
            key_ds_by_target = {}
            v_s = None
            for t in pool:
                kw_t = self._get_key_weight(t)
                pred_t, v_s = self.model.retarget_animation(
                    batch.template, batch.template_normal, batch.vertices, batch.vertices_normal,
                    kw_t, t.neu_vert,
                )
                pred_t_normal = calc_norm_torch(pred_t, t.faces, at='vert')
                _, key_d_out_t = self.model.retarget_animation(
                    t.neu_vert, t.neu_norm, pred_t, pred_t_normal, kw_t, t.neu_vert,
                )
                key_ds_by_target[t.name] = key_d_out_t

            # neutral baseline v_s^0 = phi(src_neu, src_neu) -- identical regardless of target
            kw_first = self._get_key_weight(pool[0])
            _, v_s0 = self.model.retarget_animation(
                batch.template, batch.template_normal, batch.template, batch.template_normal,
                kw_first, pool[0].neu_vert,
            )
            denom_ncp = F.mse_loss(v_s, v_s0).item()
            ncp_denoms.append(denom_ncp)

            r_bar = torch.stack(list(key_ds_by_target.values()), dim=0).mean(0)  # [1, K, 3]

            for t in pool:
                key_d_out_t = key_ds_by_target[t.name]
                cp_mse = F.mse_loss(v_s, key_d_out_t).item()
                records.append(dict(
                    target=t.name, dataset=t.dataset,
                    cp_mse=cp_mse,
                    target_var=F.mse_loss(key_d_out_t, r_bar).item(),
                    normalized_cp_error=cp_mse / (denom_ncp + eps),
                ))

        cp_summary = self._aggregate(records, ['cp_mse', 'target_var', 'normalized_cp_error'])
        cp_summary['per_target'] = {}
        for t in pool:
            rows = [r for r in records if r['target'] == t.name]
            cp_summary['per_target'][t.name] = self._aggregate(rows, ['cp_mse', 'target_var', 'normalized_cp_error'])

        # ---- perturbation-response Jacobian (plan sec. 13) -----------------
        # Restrict the perturbed/read-out columns to ones active for EVERY
        # probed target: perturbing a column that is inactive (near-zero mass,
        # see _get_key_weight) for a given target's W_t moves that mesh by
        # ~nothing regardless of eta, so its Jacobian block is trivially ~0 --
        # that reflects the column being unused there, not a genuine
        # index-mixing / disentanglement failure, and would otherwise bias
        # diag_recovery_error upward for reasons unrelated to the plan's intent.
        active_all = None
        for t in pool:
            active_t, _, _ = self._get_mass_info(t)
            active_sub = active_t[self.sampled_columns]
            active_all = active_sub if active_all is None else (active_all & active_sub)
        if active_all is None or not active_all.any():
            reason = (
                'No sampled control-point column is active for every target identity; '
                'the Jacobian probe is not evaluable without changing its scope.'
            )
            print('  Jacobian skipped: ' + reason)
            jac_summary = dict(
                valid=False, reason=reason,
                diag_recovery_error=float('nan'),
                off_diag_leakage=float('nan'),
                cross_target_jac_var=float('nan'),
                eta=float('nan'),
                diag_recovery_error_at_half_eta=float('nan'),
                eta_sensitivity_rel_change=float('nan'),
                n_pert_columns=0,
            )
            return {'cp': {'records': records, 'summary': cp_summary}, 'jacobian': jac_summary}

        candidate_cols = self.sampled_columns[active_all]
        n_pert = min(self.opts.n_pert_columns, candidate_cols.shape[0])
        pert_local_idx = self.rng.choice(candidate_cols.shape[0], size=n_pert, replace=False)
        self.pert_columns = candidate_cols[pert_local_idx]

        if self.opts.pert_eta > 0:
            eta = self.opts.pert_eta
        else:
            auto_eta = float(np.sqrt(max(np.mean(ncp_denoms), eps))) if ncp_denoms else 0.1
            eta = self.opts.pert_eta_scale * auto_eta
            print(f'  auto eta = {eta:.4e} (from RMS of real (v_s - v_s^0), scale={self.opts.pert_eta_scale})')
        eye3 = torch.eye(3, device=self.device)

        jac_by_target = {}
        for t in pool:
            kw_t = self._get_key_weight(t)
            B_pert = n_pert * 3 + 1
            _, v_t0 = self.model.retarget_animation(
                t.neu_vert, t.neu_norm, t.neu_vert, t.neu_norm,
                kw_t, t.neu_vert,
            )
            v_batch = v_t0.expand(B_pert, -1, -1).clone()
            for j, k in enumerate(self.pert_columns.tolist()):
                for d in range(3):
                    v_batch[1 + j * 3 + d, k, d] += eta

            kw_b = kw_t.expand(B_pert, -1, -1)
            tgt_neu_b = t.neu_vert.expand(B_pert, -1, -1)
            tgt_norm_b = t.neu_norm.expand(B_pert, -1, -1)
            mesh_b = self._apply_key_d(kw_b, v_batch, tgt_neu_b)
            mesh_norm_b = calc_norm_torch(mesh_b, t.faces, at='vert')

            _, R_batch = self.model.retarget_animation(
                tgt_neu_b, tgt_norm_b, mesh_b, mesh_norm_b, kw_b, tgt_neu_b,
            )  # [B_pert, K, 3]

            R_base = R_batch[0]  # [K, 3]
            R_pert = R_batch[1:].reshape(n_pert, 3, self.K, 3)  # [k, d, l, xyz]
            R_sub = R_pert[:, :, self.pert_columns, :]         # [k, d, l(sampled), xyz]
            J = (R_sub - R_base[self.pert_columns][None, None]) / eta  # [k, d, l, xyz]
            J = J.permute(0, 2, 3, 1)  # [k, l, xyz(out), d(in)]
            jac_by_target[t.name] = J

        diag_errs, off_errs = [], []
        for name, J in jac_by_target.items():
            diag_block = torch.stack([J[j, j] for j in range(n_pert)], dim=0)  # [n_pert, 3, 3]
            diag_errs.append((diag_block - eye3[None]).pow(2).sum((-1, -2)).mean().item())
            mask = ~torch.eye(n_pert, device=self.device, dtype=torch.bool)
            off_errs.append(J.pow(2).sum((-1, -2))[mask].mean().item())

        jac_pair_var = []
        names = list(jac_by_target.keys())
        for na, nb in itertools.combinations(names, 2):
            jac_pair_var.append(
                (jac_by_target[na] - jac_by_target[nb]).pow(2).sum().item() / (9.0 * n_pert * n_pert)
            )

        # eta/2 stability check (plan sec. 13.3 / 26): re-run one target only
        t0 = pool[0]
        kw_t0 = self._get_key_weight(t0)
        eta2 = eta * 0.5
        B_pert = n_pert * 3 + 1
        _, v_t0 = self.model.retarget_animation(
            t0.neu_vert, t0.neu_norm, t0.neu_vert, t0.neu_norm,
            kw_t0, t0.neu_vert,
        )
        v_batch2 = v_t0.expand(B_pert, -1, -1).clone()
        for j, k in enumerate(self.pert_columns.tolist()):
            for d in range(3):
                v_batch2[1 + j * 3 + d, k, d] += eta2
        kw_b = kw_t0.expand(B_pert, -1, -1)
        tgt_neu_b = t0.neu_vert.expand(B_pert, -1, -1)
        tgt_norm_b = t0.neu_norm.expand(B_pert, -1, -1)
        mesh_b = self._apply_key_d(kw_b, v_batch2, tgt_neu_b)
        mesh_norm_b = calc_norm_torch(mesh_b, t0.faces, at='vert')
        _, R_batch2 = self.model.retarget_animation(tgt_neu_b, tgt_norm_b, mesh_b, mesh_norm_b, kw_b, tgt_neu_b)
        R_base2 = R_batch2[0]
        R_pert2 = R_batch2[1:].reshape(n_pert, 3, self.K, 3)[:, :, self.pert_columns, :]
        J2 = ((R_pert2 - R_base2[self.pert_columns][None, None]) / eta2).permute(0, 2, 3, 1)
        diag_block2 = torch.stack([J2[j, j] for j in range(n_pert)], dim=0)
        diag_err_eta2 = (diag_block2 - eye3[None]).pow(2).sum((-1, -2)).mean().item()

        jac_summary = dict(
            valid=True,
            diag_recovery_error=float(np.mean(diag_errs)),
            off_diag_leakage=float(np.mean(off_errs)),
            cross_target_jac_var=float(np.mean(jac_pair_var)) if jac_pair_var else float('nan'),
            per_target_diag_error={n: e for n, e in zip(names, diag_errs)},
            per_target_off_diag_leakage={n: e for n, e in zip(names, off_errs)},
            eta=eta, diag_recovery_error_at_half_eta=diag_err_eta2,
            eta_sensitivity_rel_change=abs(diag_err_eta2 - diag_errs[0]) / (diag_errs[0] + eps),
            n_pert_columns=n_pert,
        )

        if not self.opts.no_vis:
            block_norm = jac_by_target[pool[0].name].pow(2).sum((-1, -2)).sqrt().cpu().numpy()
            save_heatmap(
                block_norm, os.path.join(self.opts.log_dir, 'img', 'part2_jacobian_example.png'),
                f'Perturbation-response Jacobian ||J_lk||_F  (target={pool[0].name})',
                xlabel='input column k (sampled)', ylabel='output column l (sampled)',
            )

        return {'cp': {'records': records, 'summary': cp_summary}, 'jacobian': jac_summary}

    # ==================================================================
    # Part III -- Joint factor compatibility
    # ==================================================================
    @torch.no_grad()
    def part3_joint_factor_compatibility(self, pool, _src_dataset, probe_frames):
        print('\n[Part III] Joint factor compatibility...')

        # ---- 15. cross-identity neutral factor swap ------------------------
        swap_records = []
        for a, b in itertools.permutations(pool, 2):
            kw_b = self._get_key_weight(b)
            pred, _ = self.model.retarget_animation(
                a.neu_vert, a.neu_norm, a.neu_vert, a.neu_norm, kw_b, b.neu_vert,
            )
            mse = F.mse_loss(pred, b.neu_vert).item()
            swap_records.append(dict(a=a.name, b=b.name, direction=direction_tag(a, b), mse=mse))

        if not self.opts.no_vis and len(swap_records) > 0:
            a0, b0 = pool[0], pool[1] if len(pool) > 1 else pool[0]
            kw_b0 = self._get_key_weight(b0)
            pred0, _ = self.model.retarget_animation(
                a0.neu_vert, a0.neu_norm, a0.neu_vert, a0.neu_norm, kw_b0, b0.neu_vert,
            )
            v_list = [b0.neu_vert[0].cpu(), pred0[0].detach().cpu()]
            plot_image_array(
                v_list, [b0.faces.cpu()] * 2, rot_list=[[0, 0, 0]] * 2, size=1, bg_black=False,
                mode='shade', logdir=os.path.join(self.opts.log_dir, 'img'),
                name=f'part3_neutral_swap_{a0.name}_to_{b0.name}', save=True,
            )

        # ---- 16. third-target route consistency -----------------------------
        route_records = []
        target_pairs = []
        if len(pool) >= 2:
            all_pairs = list(itertools.permutations(range(len(pool)), 2))
            n_pairs = min(10, len(all_pairs))
            chosen = self.rng.choice(len(all_pairs), size=n_pairs, replace=False)
            target_pairs = [(pool[all_pairs[i][0]], pool[all_pairs[i][1]]) for i in chosen]
        else:
            print('  route consistency skipped: needs >=2 identities in the pool')

        first_vis_done = False
        for batch in tqdm(probe_frames, ncols=100, desc='Route consistency'):
            for b, c in target_pairs:
                kw_b, kw_c = self._get_key_weight(b), self._get_key_weight(c)

                pred_c_direct, v_s = self.model.retarget_animation(
                    batch.template, batch.template_normal, batch.vertices, batch.vertices_normal,
                    kw_c, c.neu_vert,
                )
                pred_c_neutral, _ = self.model.retarget_animation(
                    batch.template, batch.template_normal, batch.template, batch.template_normal,
                    kw_c, c.neu_vert,
                )
                pred_b = self._apply_key_d(kw_b, v_s, b.neu_vert)
                pred_b_normal = calc_norm_torch(pred_b, b.faces, at='vert')
                pred_c_via_b, _ = self.model.retarget_animation(
                    b.neu_vert, b.neu_norm, pred_b, pred_b_normal, kw_c, c.neu_vert,
                )

                mse = F.mse_loss(pred_c_direct, pred_c_via_b).item()
                signal_mse = F.mse_loss(pred_c_direct, pred_c_neutral).item()
                route_records.append(dict(
                    via=b.name, final=c.name, direction=direction_tag(b, c),
                    mse=mse,
                    signal_mse=signal_mse,
                    normalized_mse=mse / (signal_mse + 1e-12),
                    normalized_rmse=np.sqrt(mse / (signal_mse + 1e-12)),
                ))

                if not self.opts.no_vis and not first_vis_done:
                    v_list = [pred_c_direct[0].detach().cpu(), pred_c_via_b[0].detach().cpu()]
                    plot_image_array(
                        v_list, [c.faces.cpu()] * 2, rot_list=[[0, 0, 0]] * 2, size=1, bg_black=False,
                        mode='shade', logdir=os.path.join(self.opts.log_dir, 'img'),
                        name=f'part3_route_direct_vs_via_{b.name}_to_{c.name}', save=True,
                    )
                    first_vis_done = True

        return {
            'neutral_swap': {'records': swap_records, 'summary': self._aggregate(swap_records, ['mse'])},
            'route_consistency': {
                'records': route_records,
                'summary': self._aggregate(
                    route_records, ['mse', 'signal_mse', 'normalized_mse', 'normalized_rmse']
                ),
            },
        }

    # ==================================================================
    # Part IV -- ICT paired semantic validation
    # ==================================================================
    @torch.no_grad()
    def part4_ict_paired_validation(self):
        print('\n[Part IV] ICT paired semantic validation...')
        if self.opts.n_ict_identity < 2:
            print('  skipped: --n_ict_identity must be >= 2 for cross-identity comparison')
            return None

        gc_dataset = GCEvalDataset(n_identity=self.opts.n_ict_identity)
        gc_loader = torch.utils.data.DataLoader(
            gc_dataset, batch_size=gc_dataset.n_basis,
            collate_fn=partial(GC_collate_wrapper, device=self.device),
        )

        neu_vert, kw, v0, v_as, delta_v, exp_disp_shared = {}, {}, {}, {}, {}, None
        basis_names = None
        for batch in tqdm(gc_loader, ncols=100, desc='ICT identity encode'):
            name = batch.identity_name
            t_neu = batch.template[0:1]
            t_norm = batch.template_normal[0:1]
            kw_a = self.model.predict_coordinate(t_neu, t_norm)  # [1, V, K]

            _, v_a0 = self.model.retarget_animation(t_neu, t_norm, t_neu, t_norm, kw_a, t_neu)  # [1, K, 3]

            B = batch.vertices.shape[0]
            kw_a_b = kw_a.expand(B, -1, -1)
            t_neu_b = t_neu.expand(B, -1, -1)
            t_norm_b = t_norm.expand(B, -1, -1)
            _, v_a_s = self.model.retarget_animation(
                t_neu_b, t_norm_b, batch.vertices, batch.vertices_normal, kw_a_b, t_neu_b,
            )  # [53, K, 3]

            neu_vert[name] = t_neu
            kw[name] = kw_a
            v0[name] = v_a0
            v_as[name] = v_a_s
            delta_v[name] = v_a_s - v_a0
            if exp_disp_shared is None:
                exp_disp_shared = batch.exp_disp  # [53, V, 3], identity-independent by construction
                basis_names = list(batch.basis_names)

        names = list(neu_vert.keys())
        paired_records, target_gt_records, target_motion_records = [], [], []

        # Delta-control coherence is symmetric, so each unordered identity
        # pair is counted once. Store one record per blendshape rather than
        # collapsing all 53 bases into a single number.
        for a_name, b_name in itertools.combinations(names, 2):
            cp_err = ((delta_v[a_name] - delta_v[b_name]) ** 2).mean(dim=(1, 2))
            cp_signal = 0.5 * (
                (delta_v[a_name] ** 2).mean(dim=(1, 2)) +
                (delta_v[b_name] ** 2).mean(dim=(1, 2))
            )
            for s, basis_name in enumerate(basis_names):
                err = cp_err[s].item()
                signal = cp_signal[s].item()
                paired_records.append(dict(
                    a=a_name, b=b_name, pair=f'{a_name}--{b_name}',
                    basis=basis_name, group=basis_group(basis_name),
                    paired_cp_mse=err,
                    cp_signal_mse=signal,
                    paired_cp_nrmse=np.sqrt(err / (signal + 1e-12)),
                ))

        # Target reconstruction is directional. Report both the actual raw
        # retargeting result and the isolated expression motion W_b DeltaC_a.
        # The latter lives entirely on target b's vertices and therefore does
        # not require source/target vertex correspondence.
        signal_per_basis = (exp_disp_shared ** 2).mean(dim=(1, 2))
        for a_name, b_name in itertools.permutations(names, 2):
            B = v_as[a_name].shape[0]
            kw_b_b = kw[b_name].expand(B, -1, -1)
            neu_b_b = neu_vert[b_name].expand(B, -1, -1)
            pred_b = self._apply_key_d(kw_b_b, v_as[a_name], neu_b_b)  # [53, V, 3]
            gt_b = neu_b_b + exp_disp_shared
            raw_err = ((pred_b - gt_b) ** 2).mean(dim=(1, 2))

            pred_motion = torch.einsum('bnc,bci->bni', kw_b_b, delta_v[a_name])
            motion_err = ((pred_motion - exp_disp_shared) ** 2).mean(dim=(1, 2))

            for s, basis_name in enumerate(basis_names):
                signal = signal_per_basis[s].item()
                raw = raw_err[s].item()
                motion = motion_err[s].item()
                common = dict(
                    a=a_name, b=b_name, pair=f'{a_name}->{b_name}',
                    basis=basis_name, group=basis_group(basis_name),
                )
                target_gt_records.append(dict(
                    **common,
                    target_gt_mse=raw,
                    signal_mse=signal,
                    target_gt_nrmse=np.sqrt(raw / (signal + 1e-12)),
                ))
                target_motion_records.append(dict(
                    **common,
                    target_motion_mse=motion,
                    signal_mse=signal,
                    target_motion_nrmse=np.sqrt(motion / (signal + 1e-12)),
                ))

        if not self.opts.no_vis and len(names) > 1:
            a0, b0 = names[0], names[1]
            B = v_as[a0].shape[0]
            kw_b0 = kw[b0].expand(B, -1, -1)
            neu_b0 = neu_vert[b0].expand(B, -1, -1)
            pred0 = self._apply_key_d(kw_b0, v_as[a0], neu_b0)[0].detach().cpu()
            motion0 = torch.einsum(
                'bnc,bci->bni', kw_b0, delta_v[a0]
            )
            pred_motion0 = (neu_b0 + motion0)[0].detach().cpu()
            gt0 = (neu_b0 + exp_disp_shared)[0].cpu()
            faces_np = gc_dataset.faces_np
            plot_image_array(
                [gt0, pred0, pred_motion0], [torch.tensor(faces_np).long()] * 3,
                rot_list=[[0, 0, 0]] * 3, size=1, bg_black=False, mode='shade',
                logdir=os.path.join(self.opts.log_dir, 'img'),
                name=f'part4_target_gt_{a0}_to_{b0}', save=True,
            )

        def semantic_summary(records, keys):
            summary = self._aggregate(records, keys)
            for field in ('group', 'basis', 'pair', 'a', 'b'):
                summary[f'by_{field}'] = {}
                for value in sorted(set(r[field] for r in records)):
                    rows = [r for r in records if r[field] == value]
                    summary[f'by_{field}'][value] = self._aggregate(rows, keys)

            # Pair-level bootstrap: first macro-average the 53 bases within
            # each identity pair, then resample pairs. This avoids treating
            # every vertex/basis observation as an independent replicate.
            pair_names = sorted(set(r['pair'] for r in records))
            pair_rng = np.random.RandomState(self.opts.seed + 1701)
            summary['pair_bootstrap_95ci'] = {}
            for key in keys:
                pair_means = np.array([
                    np.mean([float(r[key]) for r in records if r['pair'] == pair])
                    for pair in pair_names
                ], dtype=np.float64)
                if pair_means.size >= 2:
                    boot = np.empty(2000, dtype=np.float64)
                    for i in range(boot.size):
                        sample_idx = pair_rng.randint(0, pair_means.size, size=pair_means.size)
                        boot[i] = pair_means[sample_idx].mean()
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    summary['pair_bootstrap_95ci'][key] = dict(
                        low=float(lo), high=float(hi), n_pairs=int(pair_means.size)
                    )
                else:
                    summary['pair_bootstrap_95ci'][key] = dict(
                        low=float('nan'), high=float('nan'), n_pairs=int(pair_means.size)
                    )
            return summary

        return {
            'paired_cp': {
                'records': paired_records,
                'summary': semantic_summary(
                    paired_records, ['paired_cp_mse', 'cp_signal_mse', 'paired_cp_nrmse']
                ),
            },
            'target_gt': {
                'records': target_gt_records,
                'summary': semantic_summary(
                    target_gt_records, ['target_gt_mse', 'signal_mse', 'target_gt_nrmse']
                ),
            },
            'target_motion': {
                'records': target_motion_records,
                'summary': semantic_summary(
                    target_motion_records,
                    ['target_motion_mse', 'signal_mse', 'target_motion_nrmse'],
                ),
            },
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate(records, keys):
        def stat(rows, key):
            vals = np.array([
                float(r[key]) for r in rows
                if r.get(key) is not None and np.isfinite(float(r[key]))
            ], dtype=np.float64)
            if vals.size == 0:
                return dict(mean=float('nan'), median=float('nan'), std=float('nan'), n=0)
            return dict(
                mean=float(vals.mean()), median=float(np.median(vals)),
                std=float(vals.std()), n=int(vals.size),
            )

        out = {key: stat(records, key) for key in keys}
        directions = sorted(set(r['direction'] for r in records if 'direction' in r))
        if directions:
            out['by_direction'] = {}
            for d in directions:
                rows = [r for r in records if r.get('direction') == d]
                out['by_direction'][d] = {key: stat(rows, key) for key in keys}
        return out

    # ==================================================================
    # Orchestration
    # ==================================================================
    def evaluate_cif(self):
        """
        Cross-Identity Factor (CIF) consistency evaluation, implementing
        eval_CBD_cp/cross_id_factor_consist.md Parts I-IV:

          I.   Weight-column spatial coherence (Sliced-Wasserstein column cost
               matrix, Top-1/Top-5/MRR/diagonal margin, active-column ratio,
               normalized Gram consistency).
          II.  Control-point index coherence (multi-target CP re-encoding MSE,
               target-conditioned variance, normalized CP error, and a
               perturbation Jacobian evaluated at each target's encoded
               neutral control factor rather than at the zero vector).
          III. Joint factor compatibility (cross-identity neutral factor swap
               against real target-neutral GT; third-target route consistency).
          IV.  ICT paired semantic validation, recorded per identity pair and
               blendshape: delta-control CP-MSE, raw cross-identity target-GT
               error, and correspondence-free motion-only error
               MSE(W_b (C_a^s-C_a^0), d^s).

        As the plan repeatedly stresses (sec. 14, 19, 28), no single number
        here proves full disentanglement -- Parts I-IV probe different,
        complementary failure modes and must be read together.
        """
        os.makedirs(self.opts.log_dir, exist_ok=True)
        ckpt_path = Path(self.opts.ckpt).name if self.opts.ckpt else 'no_ckpt'
        datasets_tag = '-'.join(self.opts.datasets)
        run_tag = ''
        if self.opts.run_name:
            run_tag = '-' + re.sub(r'[^A-Za-z0-9_.-]+', '_', self.opts.run_name)
        setting_tag = (
            f'-cif-{datasets_tag}-K{self.opts.n_columns_sample}-'
            f'P{self.opts.sw_n_proj}-V{self.opts.sw_n_vert_sample}-'
            f'I{self.opts.n_ict_identity}x{self.opts.n_id_per_dataset}-'
            f'S{self.opts.src_name}{self.opts.n_src_frames}-'
            f'A{self.opts.active_mass_ratio:g}{run_tag}'
        )
        self.opts.log_dir = os.path.join(self.opts.log_dir, ckpt_path + setting_tag)
        os.makedirs(self.opts.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.opts.log_dir, 'img'), exist_ok=True)

        with open(os.path.join(self.opts.log_dir, 'opts.json'), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
        self.dump_yaml(os.path.join(self.opts.log_dir, 'train_opts.yml'), self.opts)
        self.logger = open(os.path.join(self.opts.log_dir, 'log.txt'), 'w')
        print(f'Saving log at: {self.opts.log_dir}')

        self.model.eval()

        pool = self._load_identity_pool()
        pool_desc = ', '.join(f'{i.name}({i.dataset})' for i in pool)
        print(f'[Identity pool] {len(pool)} identities: {pool_desc}')
        self.logger.write(f'[Identity pool] {len(pool)} identities: {pool_desc}\n')

        if ((not self.opts.skip_part1) or (not self.opts.skip_part2) or
                (not self.opts.skip_part3)) and len(pool) < 2:
            raise RuntimeError(
                'Parts I-III require at least two target identities, but the '
                f'configured pool contains {len(pool)}.'
            )

        results = {}

        if not self.opts.skip_part1:
            results['part1'] = self.part1_weight_column_coherence(pool)

        need_probe = (not self.opts.skip_part2) or (not self.opts.skip_part3)
        if need_probe:
            src_dataset, probe_frames = self._sample_src_probe_frames()
            print(f'[Source probe] {self.opts.src_name}: {len(probe_frames)} frames sampled')

        if not self.opts.skip_part2:
            results['part2'] = self.part2_cp_index_coherence(pool, probe_frames)

        if not self.opts.skip_part3:
            results['part3'] = self.part3_joint_factor_compatibility(pool, src_dataset, probe_frames)

        if not self.opts.skip_part4:
            results['part4'] = self.part4_ict_paired_validation()

        self._write_report(results)
        self.logger.close()
        print('done!')

    def _write_report(self, results):
        lines = []
        lines.append('=' * 70)
        lines.append('[CIF Eval] Recommended table (plan sec. 23)')
        lines.append('=' * 70)
        lines.append('Values are mean / median / std (n).')

        def fmt(d, key):
            if d is None:
                return 'n/a'
            item = d.get(key, {})
            mean = item.get('mean', float('nan'))
            median = item.get('median', float('nan'))
            std = item.get('std', float('nan'))
            n = item.get('n', 0)
            return f'{mean:.4e} / {median:.4e} / {std:.4e} ({n})' if np.isfinite(mean) else 'n/a'

        def fmt_ci(d, key):
            item = d.get('pair_bootstrap_95ci', {}).get(key, {})
            lo, hi = item.get('low', float('nan')), item.get('high', float('nan'))
            return f'[{lo:.4e}, {hi:.4e}]' if np.isfinite(lo) else 'n/a'

        if 'part1' in results:
            s1 = results['part1']['summary']
            wdiag = list(s1['weight_diagnostics_per_identity'].values())
            lines.append(f"Active-column ratio (mean):       {s1['active_column_ratio_mean']:.4f}")
            lines.append(f"Column retrieval Top-1 (all/act): {fmt(s1,'top1')} / {fmt(s1,'top1_active')}")
            lines.append(f"Column retrieval Top-5 (all/act): {fmt(s1,'top5')} / {fmt(s1,'top5_active')}")
            lines.append(f"Column MRR (all/active):          {fmt(s1,'mrr')} / {fmt(s1,'mrr_active')}")
            lines.append(f"Diagonal margin:                  {fmt(s1,'margin')}")
            lines.append(f"Gram consistency (lower better):  {fmt(s1,'gram_err')}")
            lines.append(f"Gram consistency (common active): {fmt(s1,'gram_err_active')}")
            lines.append(
                'Weight diagnostics: '
                f"max negative ratio={max(d['negative_weight_ratio'] for d in wdiag):.3e}, "
                f"max POU error={max(d['pou_max_error'] for d in wdiag):.3e}, "
                f"min sampled-mass coverage={min(d['sampled_column_mass_valid_ratio'] for d in wdiag):.3f}"
            )
        if 'part2' in results:
            s2c = results['part2']['cp']['summary']
            s2j = results['part2']['jacobian']
            lines.append(f"CP re-encoding MSE:                {fmt(s2c,'cp_mse')}")
            lines.append(f"Target-conditioned variance:       {fmt(s2c,'target_var')}")
            lines.append(f"Normalized CP error:                {fmt(s2c,'normalized_cp_error')}")
            if s2j.get('valid', True):
                lines.append(f"Jacobian diagonal recovery error:  {s2j['diag_recovery_error']:.4e}")
                lines.append(f"Jacobian off-diagonal leakage:      {s2j['off_diag_leakage']:.4e}")
                lines.append(f"Cross-target Jacobian variance:     {s2j['cross_target_jac_var']:.4e}")
                lines.append(f"  (eta sensitivity, rel. change vs eta/2: {s2j['eta_sensitivity_rel_change']:.4f})")
            else:
                lines.append(f"Jacobian: n/a ({s2j.get('reason', 'invalid probe')})")
        if 'part3' in results:
            s3n = results['part3']['neutral_swap']['summary']
            s3r = results['part3']['route_consistency']['summary']
            lines.append(f"Neutral factor-swap MSE:           {fmt(s3n,'mse')}")
            lines.append(f"Route-consistency MSE:              {fmt(s3r,'mse')}")
            lines.append(f"Route-consistency normalized RMSE: {fmt(s3r,'normalized_rmse')}")
        if 'part4' in results and results['part4'] is not None:
            s4p = results['part4']['paired_cp']['summary']
            s4g = results['part4']['target_gt']['summary']
            s4m = results['part4']['target_motion']['summary']
            lines.append('[ICT semantic evaluation]')
            lines.append(f"Paired blendshape CP-MSE:           {fmt(s4p,'paired_cp_mse')}")
            lines.append(f"Paired blendshape CP-NRMSE:         {fmt(s4p,'paired_cp_nrmse')}")
            lines.append(f"Cross-identity target GT MSE:       {fmt(s4g,'target_gt_mse')}")
            lines.append(f"Cross-identity target GT NRMSE:     {fmt(s4g,'target_gt_nrmse')}")
            lines.append(f"Motion-only W_b DeltaC_a MSE:       {fmt(s4m,'target_motion_mse')}")
            lines.append(f"Motion-only W_b DeltaC_a NRMSE:     {fmt(s4m,'target_motion_nrmse')}")
            lines.append(f"  pair-bootstrap 95% CI (NRMSE):    {fmt_ci(s4m,'target_motion_nrmse')}")

        report_text = '\n'.join(lines)
        print(report_text)
        self.logger.write(report_text + '\n')

        def to_jsonable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                value = obj.item()
                return None if isinstance(value, float) and not np.isfinite(value) else value
            if isinstance(obj, dict):
                return {k: to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_jsonable(v) for v in obj]
            if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
                return None
            return obj

        # cost matrices / per-pair raw tensors are large -- drop them from the
        # JSON dump, keep only the scalar summaries + per-pair scalar records
        dumpable = {}
        for part_name, part_val in results.items():
            dumpable[part_name] = to_jsonable(part_val)

        with open(os.path.join(self.opts.log_dir, 'metrics.json'), 'w') as f:
            json.dump(dumpable, f, indent=2)

    # ------------------------------------------------------------------
    @staticmethod
    def set_seed(opts):
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opts.seed)
        random.seed(opts.seed)

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def dump_yaml(yaml_file_path, opts):
        with open(yaml_file_path, 'w') as f:
            yaml.dump(vars(opts), f, sort_keys=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Cross-Identity Factor (CIF) consistency evaluation (NGBC / version=5 only).
    Implements eval_CBD_cp/cross_id_factor_consist.md Parts I-IV; see
    `Trainer.evaluate_cif` for a summary of each part.

    Examples:
        # full run against an ICT + Multiface pool
        python eval_CBD_cif.py --ckpt ./ckpts_CBD/2026-06-28-19-16-49-NGBCv5 \
            --align_latent --last_activation relu \
            --datasets ict mf_SEN --n_ict_identity 5 --n_id_per_dataset 5 \
            --src_name mf_SEN --n_src_frames 30

        # fast smoke test
        python eval_CBD_cif.py --ckpt <ckpt> --align_latent --last_activation relu --quick

        # only the minimum-recommended subset (plan sec. 29): Part II + III (+ IV)
        python eval_CBD_cif.py --ckpt <ckpt> --align_latent --last_activation relu --skip_part1
    """
    mp.set_start_method('spawn', force=True)

    opts = Options()

    config = f'{opts.ckpt}/train_opts.yml'
    opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)

    # Model-architecture values come from the checkpoint configuration unless
    # the user explicitly supplied the corresponding CLI flag. Defaults of
    # None prevent an evaluation default from silently changing the model.
    opts_cli = vars(opts)
    for key, value in opts_cli.items():
        if key in {'last_activation', 'no_pou', 'align_latent'} and value is None:
            continue
        opts_yaml[key] = value
    architecture_defaults = dict(
        last_activation='relu', no_pou=False, align_latent=False, out_type=1,
    )
    for key, default in architecture_defaults.items():
        if opts_yaml.get(key) is None:
            opts_yaml[key] = default
    opts = argparse.Namespace(**opts_yaml)

    print('loaded version:', opts.version)

    trainer = Trainer(opts)
    trainer.evaluate_cif()
