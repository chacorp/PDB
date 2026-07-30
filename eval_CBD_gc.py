import os
import glob
import json
import yaml
import random
import re

import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
import trimesh
import igl
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from utils.matplotlib_rnd import plot_image_array_col
from utils.ckpt_utils import *
from utils.remesh_utils import ICT_face_model
from utils.keys import ICT_KEYS, ict_data_split

from models.baseline import CageNet
from models.NGBC import NeuralGeneralizedBarycentricCoordinate
from models.NGBCv2 import NeuralBarycentricCoordinatev2, NeuralBarycentricCoordinatev3

import torch.multiprocessing as mp

from pathlib import Path
__abs_path__ = str(Path(__file__).parents[0].absolute())


# ---------------------------------------------------------------------------
# Basis grouping (for eye / brow / mouth / ... breakdowns, plan sec. 8)
# ---------------------------------------------------------------------------

_GROUP_PREFIXES = ['Brow', 'Cheek', 'Eye', 'Jaw', 'Mouth', 'Nose']


def basis_group(basis_name):
    for prefix in _GROUP_PREFIXES:
        if basis_name.startswith(prefix):
            return prefix
    return 'Other'


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GCEvalDataset(data.Dataset):
    """
    Global-coupling (non-local motion leakage) evaluation dataset.

    Same construction as `BasisEvalDataset` in eval_CBD_bs.py: for one or
    more ICT neutral identities, builds one item per (identity,
    expression-basis axis) pair -- a one-hot activation of that basis on
    that identity.  `exp_disp` (= vertices - template) is the analytic
    per-vertex displacement caused solely by that basis; since ICT
    expressions are additive blendshapes on top of the identity
    (id_exp_verts = neutral + id_disp(id) + exp_disp(exp)), `exp_disp`
    does not depend on which identity is chosen, so it doubles as the
    ground-truth deformation used both to build the active-region mask
    and as the deformation target itself (plan sec. 2, 5).

    Items are ordered identity-major, basis-minor (all 53 bases for
    identity 0, then all 53 for identity 1, ...), so consecutive items
    share the same identity -- callers can cheaply detect an identity
    switch by comparing `identity_name` between items.

    `ict_split_set=True` switches from synthetic identities (coefficient
    vectors in `data/ICT_live_100/iden_vecs.npy`) to the 10 real captured
    ICT-FaceKit identities in `ict_data_split['test']` (utils/keys.py --
    train/val/test all name the same 10 identities, so there is no
    train/val/test choice to make here). Their neutral templates come
    from `ict_real_templates.pkl`, which shares vertex ordering/topology
    with `ICT_face_model`'s default region (verified: identical face
    array, region 0). Since ICT expression bases are additive
    displacements independent of identity, `exp_disp` is synthesized the
    same way as for synthetic identities (`ICT_face_model.get_exp_disp`);
    only the neutral template differs (real scan instead of a
    synthesized one). When set, `--n_identity`/`--identity_indices` are
    ignored.

    Output per item:
        (vertices, template, vertices_normal, template_normal, faces,
         basis_name, mesh_data, exp_disp, identity_name)
    """

    ICT_SPLIT_SET_PKL = '/data/sihun/ICT-audio2face/split_set/ict_real_templates.pkl'

    def __init__(self, n_identity=1, identity_indices=None, ict_split_set=False):
        super().__init__()
        self.ict = ICT_face_model()
        self.exp_names = ICT_KEYS
        self.n_basis = len(ICT_KEYS)  # 53
        self.mesh_data = torch.tensor([5])  # 'ict'

        self.faces_np = self.ict.faces
        self.items = []  # (vertices_np, template_np, exp_disp_np, basis_name, identity_name)
        eye = np.eye(self.n_basis, dtype=np.float32)

        if ict_split_set:
            self.identity_names = list(ict_data_split['test'])
            self.identity_indices = None
            self.n_identity = len(self.identity_names)

            with open(self.ICT_SPLIT_SET_PKL, 'rb') as f:
                real_templates = pickle.load(f)

            for identity_name in self.identity_names:
                template = real_templates[identity_name].astype(np.float32)
                for k in range(self.n_basis):
                    exp_coeff = eye[k]
                    exp_disp = self.ict.get_exp_disp(exp_coeff)[0]
                    vertices = template + exp_disp
                    self.items.append(
                        (vertices, template, exp_disp, self.exp_names[k], identity_name)
                    )
        else:
            id_vecs = np.load(f'{__abs_path__}/data/ICT_live_100/iden_vecs.npy')
            if identity_indices is None:
                if n_identity < 1 or n_identity > len(id_vecs):
                    raise ValueError(
                        f'--n_identity must be in [1, {len(id_vecs)}], got {n_identity}'
                    )
                self.identity_indices = list(range(n_identity))
            else:
                self.identity_indices = list(identity_indices)
                if not self.identity_indices:
                    raise ValueError('--identity_indices must contain at least one index')
                invalid = [i for i in self.identity_indices if i < 0 or i >= len(id_vecs)]
                if invalid:
                    raise ValueError(
                        f'identity indices out of range [0, {len(id_vecs) - 1}]: {invalid}'
                    )
                if len(set(self.identity_indices)) != len(self.identity_indices):
                    raise ValueError('--identity_indices must not contain duplicates')

            self.identity_names = [f'{i:03d}' for i in self.identity_indices]
            self.n_identity = len(self.identity_indices)

            for identity_idx, identity_name in zip(self.identity_indices, self.identity_names):
                id_coeff = id_vecs[identity_idx].astype(np.float32)
                for k in range(self.n_basis):
                    exp_coeff = eye[k]
                    vertices, template, exp_disp = self.ict.apply_coeffs(
                        id_coeff, exp_coeff, return_all=True
                    )
                    self.items.append(
                        (vertices[0], template[0], exp_disp[0], self.exp_names[k], identity_name)
                    )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        vertices_np, template_np, exp_disp_np, basis_name, identity_name = self.items[index]

        vertices_normal_np = igl.per_vertex_normals(vertices_np, self.faces_np)
        template_normal_np = igl.per_vertex_normals(template_np, self.faces_np)

        vertices = torch.tensor(vertices_np).float()
        template = torch.tensor(template_np).float()
        exp_disp = torch.tensor(exp_disp_np).float()
        vertices_normal = torch.tensor(vertices_normal_np).float()
        template_normal = torch.tensor(template_normal_np).float()
        faces = torch.tensor(self.faces_np).long()

        return (
            vertices, template, vertices_normal, template_normal,
            faces, basis_name, self.mesh_data, exp_disp, identity_name,
        )

    def get_data_config(self):
        text  = "===========[GCEvalDataset]===========\n"
        text += f"[Identities]: {', '.join(self.identity_names)}\n"
        text += f"[Total basis axes]: {self.n_basis}\n"
        text += f"[Total items]: {len(self.items)}\n"
        text += "==========================================\n"
        return text


class GCDataBatch:
    def __init__(self, data):
        transposed = list(zip(*data))
        self.vertices = torch.stack(transposed[0], 0)          # [B, V, 3]
        self.template = torch.stack(transposed[1], 0)          # [B, V, 3]
        self.vertices_normal = torch.stack(transposed[2], 0)   # [B, V, 3]
        self.template_normal = torch.stack(transposed[3], 0)   # [B, V, 3]
        self.faces = torch.stack(transposed[4], 0)              # [B, F, 3]
        self.basis_names = list(transposed[5])                  # basis name per item, len B
        self.mesh_data = transposed[6][0]
        self.exp_disp = torch.stack(transposed[7], 0)           # [B, V, 3] == vertices - template
        self.identity_name = transposed[8][0]                   # identity name (string);
        # all items in a batch share one identity -- see evaluate_gc's fixed
        # batch_size == n_basis, which keeps every batch aligned to exactly
        # one identity

    def to(self, device):
        self.vertices = self.vertices.to(device)
        self.template = self.template.to(device)
        self.vertices_normal = self.vertices_normal.to(device)
        self.template_normal = self.template_normal.to(device)
        self.faces = self.faces.to(device)
        self.exp_disp = self.exp_disp.to(device)
        return self


def GC_collate_wrapper(batch, device="cpu"):
    return GCDataBatch(batch).to(device)


# ---------------------------------------------------------------------------
# Mask / geometry helpers (plan sec. 4.4, 5, 6)
# ---------------------------------------------------------------------------

def locality_mask(disp, tau_abs, alpha):
    """
    Combined absolute + relative threshold active-region mask (plan sec. 5):

        M_i^s = 1[ ||d_i^s|| > max(tau_abs, alpha * max_j ||d_j^s||) ]

    Args:
        disp: [B, V, 3] per-basis GT displacement (B blendshapes batched together)
        tau_abs (float): absolute displacement threshold
        alpha (float): relative threshold, as a fraction of this blendshape's
            own peak displacement
    Returns:
        [B, V] float mask (1.0 = active region for that row's blendshape)
    """
    norm = torch.norm(disp, dim=-1)                     # [B, V]
    max_norm = norm.amax(dim=-1, keepdim=True)           # [B, 1]
    thresh = torch.clamp(alpha * max_norm, min=tau_abs)  # [B, 1]
    return (norm > thresh).float()


def build_mesh_graph(vertices_np, faces_np):
    """
    Undirected, edge-length-weighted mesh graph used as a graph-geodesic
    approximation for dilating the active-region mask (plan sec. 6). Built
    once per identity since topology + vertex positions (the neutral
    template) are fixed for every basis of that identity.
    """
    edges = np.vstack([faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    vi, vj = edges[:, 0], edges[:, 1]
    lengths = np.linalg.norm(vertices_np[vi] - vertices_np[vj], axis=1)
    n = vertices_np.shape[0]
    w = sparse.coo_matrix((lengths, (vi, vj)), shape=(n, n))
    return (w + w.T).tocsr()


def dilate_mask_geodesic(active_mask_np, graph, dilation_dist):
    """
    Args:
        active_mask_np: [V] bool array, this blendshape's active region
        graph: [V, V] sparse edge-length graph, see `build_mesh_graph`
        dilation_dist (float): graph-geodesic distance to dilate by (the
            "transition band" width of plan sec. 6)
    Returns:
        [V] bool array: active region unioned with the dilation band.
        Always a superset of `active_mask_np`.
    """
    active_idx = np.nonzero(active_mask_np)[0]
    if active_idx.size == 0:
        return active_mask_np.copy()
    dist = dijkstra(graph, directed=False, indices=active_idx, min_only=True)
    return dist <= dilation_dist


def vertex_area_weights(vertices_np, faces_np):
    """
    Per-vertex barycentric surface-area weight A_i (one third of each
    incident triangle area), normalized
    to mean 1 so that turning area-weighting on/off does not change the
    overall scale of E_in / E_out (only how mass is redistributed across
    vertices).
    """
    # Barycentric mass is non-negative even on obtuse/non-Delaunay facial
    # triangles, unlike a signed circumcentric/Voronoi construction.
    m = igl.massmatrix(vertices_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    area = np.asarray(m.diagonal())
    area = np.clip(area, a_min=1e-12, a_max=None)
    return area / area.mean()


def weighted_quantile(values, weights, q):
    """Area-weighted quantile for 1-D tensors; returns NaN for an empty region."""
    valid = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0)
    if not torch.any(valid):
        return torch.tensor(float('nan'), device=values.device, dtype=values.dtype)
    values = values[valid]
    weights = weights[valid]
    order = torch.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = torch.cumsum(weights, dim=0)
    cutoff = q * cumulative[-1]
    idx = torch.searchsorted(cumulative, cutoff).clamp_max(values.numel() - 1)
    return values[idx]


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def region_mask_to_vertex_color(active_mask, dilated_mask,
                                 active_color=(0.90, 0.15, 0.15),
                                 transition_color=(0.95, 0.65, 0.10),
                                 far_color=(0.75, 0.75, 0.75)):
    """
    3-way region coloring for the GT side of the visualization: active
    region (red), transition/dilation band (orange), far-field inactive
    region (gray). `dilated_mask` must be a superset of `active_mask`.
    """
    active_mask = active_mask.detach().cpu().numpy() if torch.is_tensor(active_mask) else active_mask
    dilated_mask = dilated_mask.detach().cpu().numpy() if torch.is_tensor(dilated_mask) else dilated_mask
    active_mask = active_mask.astype(bool)
    dilated_mask = dilated_mask.astype(bool)
    vc = np.tile(np.array(far_color, dtype=np.float32), (active_mask.shape[0], 1))
    vc[dilated_mask] = np.array(transition_color, dtype=np.float32)
    vc[active_mask] = np.array(active_color, dtype=np.float32)
    return vc


def leakage_to_vertex_color(residual_norm, tau_abs,
                             leak_color=(0.85, 0.10, 0.65), base_color=(0.75, 0.75, 0.75)):
    """
    Prediction-side coloring: highlights vertices where the prediction
    deviates from ground truth (||d_hat - d|| > tau_abs), regardless of
    region -- red spilling outside the GT active region is exactly the
    non-local motion leakage this experiment targets.
    """
    residual_norm = residual_norm.detach().cpu().numpy() if torch.is_tensor(residual_norm) else residual_norm
    leak = residual_norm > tau_abs
    vc = np.tile(np.array(base_color, dtype=np.float32), (residual_norm.shape[0], 1))
    vc[leak] = np.array(leak_color, dtype=np.float32)
    return vc


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def Options():
    parser = argparse.ArgumentParser(
        description='Global-coupling / non-local motion leakage evaluation for face retargeting '
                     '(see eval_CBD_bs/global_coupling_eval_plan.md)'
    )
    parser.add_argument('-c', '--config', default='config/train_CBD.yml', help='config file path')
    parser.add_argument("--device",       type=str,   default="cuda:0")

    parser.add_argument("--log_dir",      type=str,   default="eval_CBD_gc")

    parser.add_argument("--version",      type=int,   default=1,
                        help='model version (0: NFS/NFR, 1: NC(baseline), 5/21/22: ours)')

    parser.add_argument("--last_activation", default=None,
                        choices=["relu", "elu", "softmax", "softplus", "none", "sqrelu"],
                        help="Last-layer activation for NGBC.key_weight_model()")

    parser.add_argument("--no_pou", dest='no_pou', action='store_true')
    parser.set_defaults(no_pou=None)

    parser.add_argument("--start_epoch",  type=int,   default=0)
    parser.add_argument("--lr",           type=float, default=0.0002)
    parser.add_argument("--batch_size",   type=int,   default=1)

    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--ckpt",         type=str,   default=None)
    parser.add_argument("--continue_ckpt", dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)

    parser.add_argument("--tb", action='store_true')
    parser.set_defaults(is_train=True)

    parser.add_argument("--save_vert", dest='save_vert', action='store_true')
    parser.set_defaults(save_vert=False)

    parser.add_argument("--no_vis", dest='no_vis', action='store_true',
                        help='skip per-basis GT/leakage visualization (saves time)')
    parser.set_defaults(no_vis=False)

    parser.add_argument("--use_NFR", dest='use_NFR', action='store_true')
    parser.set_defaults(use_NFR=False)

    parser.add_argument("--NFR", dest='NFR', action='store_true')
    parser.set_defaults(NFR=False)

    parser.add_argument("--optim_cage", dest='optim_cage', action='store_true')
    parser.set_defaults(optim_cage=None)

    parser.add_argument("--align_latent", dest='align_latent', action='store_true')
    parser.set_defaults(align_latent=None)

    parser.add_argument("--allow_partial_load", action='store_true',
                        help='allow checkpoint missing/unexpected keys after printing them; '
                             'by default, a partial load aborts evaluation')
    parser.set_defaults(allow_partial_load=False)

    parser.add_argument("--mask_tau_abs", type=float, default=1e-3,
                        help='absolute displacement threshold tau_abs used in the active-region '
                             'mask M_i^s = 1[||d_i|| > max(tau_abs, alpha * max_j ||d_j||)] (plan sec. 5)')
    parser.add_argument("--mask_alpha", type=float, default=0.1,
                        help='relative displacement threshold alpha (fraction of this basis\'s own '
                             'peak displacement), used together with --mask_tau_abs (plan sec. 5)')

    parser.add_argument("--dilation_dist", type=float, default=0.2,
                        help='graph-geodesic distance (in mesh coordinate units) used to dilate the '
                             'active-region mask into a transition band; E_out is only measured beyond '
                             'this band (far-field), so normal deformation falloff at the mask boundary '
                             'is not penalized as leakage (plan sec. 6)')

    parser.add_argument("--no_area_weight", dest='no_area_weight', action='store_true',
                        help='disable per-vertex surface-area weighting (plan sec. 4.4) and fall back '
                             'to a plain unweighted per-vertex average')
    parser.set_defaults(no_area_weight=False)

    parser.add_argument("--n_identity", type=int, default=1,
                        help='number of ICT identities ("000", "001", ... from '
                             'data/ICT_live_100/iden_vecs.npy) to evaluate over; '
                             'metrics are averaged over all (identity, basis) pairs')

    parser.add_argument("--identity_indices", type=int, nargs='+', default=None,
                        help='explicit ICT identity indices to evaluate (for example: '
                             '--identity_indices 0 7 19); overrides --n_identity')

    parser.add_argument("--ict_split_set", dest='ict_split_set', action='store_true',
                        help='evaluate on the 10 real captured ICT-FaceKit identities in '
                             "ict_data_split['test'] (utils/keys.py) instead of synthetic "
                             'iden_vecs.npy identities; neutral templates are loaded from '
                             f'{GCEvalDataset.ICT_SPLIT_SET_PKL}. Overrides --n_identity/'
                             '--identity_indices.')
    parser.set_defaults(ict_split_set=False)

    parser.add_argument("--min_region_area_frac", type=float, default=1e-4,
                        help='minimum surface-area fraction required for both the active and '
                             'far-field regions; invalid basis cases are reported and excluded '
                             'from metric aggregation instead of being assigned a zero error')

    parser.add_argument("--run_name", type=str, default=None,
                        help='optional tag appended to the output directory')

    args = parser.parse_args()
    if args.mask_tau_abs < 0:
        parser.error('--mask_tau_abs must be non-negative')
    if not 0.0 <= args.mask_alpha <= 1.0:
        parser.error('--mask_alpha must be in [0, 1]')
    if args.dilation_dist < 0:
        parser.error('--dilation_dist must be non-negative')
    if not 0.0 < args.min_region_area_frac < 0.5:
        parser.error('--min_region_area_frac must be in (0, 0.5)')
    return args


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer():
    def __init__(self, opts):
        self.opts = opts
        self.set_seed(self.opts)
        self.device = opts.device

        last_act_list = ["relu", "elu", "softmax", "softplus", "none", "sqrelu"]
        last_act_list = [self.opts.last_activation == l_act for l_act in last_act_list]

        if opts.version == 0:
            from utils.nfr_utils import get_dfn_info
            from utils.mesh_utils import get_mesh_operators
            self.get_dfn_info = get_dfn_info
            self.get_mesh_operators = get_mesh_operators

            opts.is_train = False  # prevents NFS from calling set_neutral_ict in __init__
            from evaluation import Trainer as EvalTrainer
            eval_trainer = EvalTrainer(opts)
            self.model = eval_trainer.model
            return  # evaluation.Trainer handles weight loading

        elif opts.version == 1:
            self.model = CageNet(
                device=self.device,
                optim_cage=self.opts.optim_cage,
            )

        elif opts.version == 5:
            if self.opts.last_activation == 'sqrelu':
                raise ValueError('version 5 does not expose a sqrelu output activation')
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

        elif opts.version == 21:
            self.model = NeuralBarycentricCoordinatev2(
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
                use_sqrelu=last_act_list[5],
                use_least_N_on_V=False,
                is_train=True,
                use_pou=not self.opts.no_pou,
                device=self.device,
                hid_dim=128 if self.opts.align_latent else 256,
            )

        elif opts.version == 22:
            self.model = NeuralBarycentricCoordinatev3(
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
                use_sqrelu=last_act_list[5],
                use_least_N_on_V=False,
                is_train=True,
                use_pou=not self.opts.no_pou,
                device=self.device,
                hid_dim=128 if self.opts.align_latent else 256,
            )

        else:
            raise NotImplementedError('No matching model version')

        self.load_weight()

    def load_weight(self):
        if self.opts.ckpt:
            print(self.opts.ckpt)
            if self.opts.continue_ckpt:
                ckpt = glob.glob(
                    os.path.join(self.opts.ckpt, f"*_{self.opts.start_epoch:03d}.pth")
                )[0]
            else:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
            print(f"Loading... {ckpt}")
            ckpt_dict = torch.load(ckpt, map_location=self.device)
            if isinstance(ckpt_dict, dict) and 'model' in ckpt_dict:
                ckpt_dict = ckpt_dict['model']
            incompatible = self.model.load_state_dict(ckpt_dict, strict=False)
            if incompatible.unexpected_keys:
                # Evaluation can intentionally omit training-only heads, so
                # extra checkpoint tensors are reported but are not fatal.
                print(
                    'WARNING: checkpoint contains keys unused by the evaluation model:\n  ' +
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
                    raise RuntimeError(message + '\nUse --allow_partial_load only if this is intentional.')
        else:
            print('no ckpt found, training from scratch!')

    # ------------------------------------------------------------------
    def _forward_pass(self, template, vertices, template_normal, vertices_normal, faces, mesh_data):
        """
        Runs one self-retargeting forward pass (template = source & target
        neutral identity, vertices = driving expression mesh on that same
        identity) and returns pred_vertices [B, V, 3].  For version==0
        (NFS/NFR), relies on the per-identity features cached on `self`
        by `evaluate_gc` (identity is fixed for the whole run).
        """
        device = self.device

        if self.opts.version == 0:
            if self.opts.NFR == False:
                # ── NFS ───────────────────────────────────────────────
                vert_feat_exp = self.model.get_local_feature(
                    vertices, faces, self._img_feat
                ).float()
                pred_exp_coeff = self.model.encode_exp(
                    vert_feat_exp, self._src_dfn_info, batch_process=True, verbose=False
                )
                inputs = (
                    self._tri_feat if self.opts.dec_type == 'jacob' else self._vert_feat,
                    pred_exp_coeff, self._pred_id_coeff, self._pred_seg_coeff,
                    None, template[0][None], faces, self._src_operators,
                )
                decode_out = self.model.decode(inputs, tgt_mesh=self._src_mesh, batch_process=True)
                pred_vertices = decode_out[0] if isinstance(decode_out, tuple) else decode_out
            else:
                # ── NFR ───────────────────────────────────────────────
                inputs_v = self.model.get_inputs(vertices, faces)
                pred_exp = self.model.model.encode(
                    inputs_v, self._src_img.to(device), N_F=faces.shape[0]
                )
                pred_vertices, _, _ = self.model.calc_new_mesh(
                    template[0], faces, pred_exp,
                    self._src_operators, self._src_dfn_info, self._src_img,
                )
                # calc_new_mesh's Poisson solve only recovers the mesh up to an
                # arbitrary rigid translation (integrating a Jacobian field has no
                # translation term), so it re-centers its output to zero mean
                # internally. `vertices` (GT) is in ICT's own non-zero-mean frame,
                # so comparing the two directly would count that translation gap
                # as error on top of the real reconstruction error. Re-align pred
                # to GT's mean here -- matches the same fix in eval_CBD_lp.py.
                dims = tuple(range(pred_vertices.dim() - 1))
                pred_vertices = (
                    pred_vertices
                    - pred_vertices.mean(dim=dims, keepdim=True)
                    + vertices.mean(dim=dims, keepdim=True)
                )

        elif self.opts.version == 1:
            # ── NC (Neural Cage) ────────────────────────────────────
            pred_vertices, _ = self.model.retarget(template, vertices, template)

        elif self.opts.version == 21:
            pred_vertices, recon_vertices, recon_source, exp_z, \
            pred_source, _, _, _, _, _, _ = self.model(
                template, vertices, template_normal, vertices_normal, mesh_data, epoch=0,
            )
            pred_vertices = pred_vertices - pred_source + template

        elif self.opts.version == 22:
            (
                pred_vertices, _, pred_source, _,
                src_exp_z, _, _, _, _, _, _, _, _, _
            ) = self.model(
                template, vertices, template_normal, vertices_normal, mesh_data, epoch=0,
            )

        else:
            # ── ours (NGBC) ──────────────────────────────────────────
            pred_vertices, _, _, _, \
            _, _, _, _, _ = self.model(
                template, vertices, template_normal, vertices_normal,
                mesh_data=mesh_data, epoch=0,
            )

        return pred_vertices

    def _load_identity_context(self, identity_name, template_np, faces_np):
        """
        (Re)builds all per-identity cached state: NFS/NFR dfn_info /
        operators / rendered image (unchanged from eval_CBD_bs.py), plus
        the geometry needed by the global-coupling metrics themselves --
        the per-vertex area weights (plan sec. 4.4) and the mesh graph
        used to dilate each basis's active-region mask into a transition
        band (plan sec. 6). Both depend only on the neutral template
        topology/geometry, so they are computed once per identity and
        reused across all 53 bases.
        """
        device = self.device
        template0 = torch.tensor(template_np).float()[None].to(device)
        faces0 = torch.tensor(faces_np).long().to(device)
        template_normal0 = torch.tensor(
            igl.per_vertex_normals(template_np, faces_np)
        ).float()[None].to(device)

        if self.opts.version == 0:
            self._src_mesh = trimesh.Trimesh(vertices=template_np, faces=faces_np)

            # identities "000".."099" (synthetic) have a precomputed dfn_info /
            # operators / rendered image cache under precompute-synth-fullhead
            # (matches eval_CBD.py / eval_CBD_lp.py's ict precompute handling);
            # the real captured identities used by --ict_split_set (m00, w00,
            # ...) have the same three files under precompute-real-fullhead.
            # Anything with neither falls back to computing (and rendering)
            # on the fly.
            dfn_path = None
            for ict_precompute_path in (
                '/data/sihun/ICT-audio2face/precompute-synth-fullhead',
                '/data/sihun/ICT-audio2face/precompute-real-fullhead',
            ):
                cand_dfn_path = os.path.join(ict_precompute_path, f"{identity_name}_dfn_info.pkl")
                if os.path.exists(cand_dfn_path):
                    dfn_path = cand_dfn_path
                    break
            if dfn_path is not None:
                self._src_dfn_info = pickle.load(open(dfn_path, 'rb'))
                self._src_operators = pickle.load(open(
                    os.path.join(ict_precompute_path, f"{identity_name}_operators.pkl"), 'rb'
                ))
                src_img_np = np.load(
                    os.path.join(ict_precompute_path, f"{identity_name}_img.npy")
                )
                # keep the batch dim ([1, 256, 256, 3]) -- matches what
                # render_img() itself returns; NFR's encode() requires a
                # 4D image (indexing [0] here breaks NFR, though NFS's
                # get_img_feat happens to re-add a missing batch dim)
                self._src_img = torch.from_numpy(src_img_np).float().to(self.device)
            else:
                self._src_dfn_info = self.get_dfn_info(self._src_mesh, map_location=self.device)
                self._src_operators = self.get_mesh_operators(self._src_mesh)
                self._src_img = self.model.renderer.render_img(self._src_mesh).float().to(self.device)

            if self.opts.NFR == False:
                self._img_feat = self.model.get_img_feat(self._src_img)
                self._vert_feat = self.model.get_local_feature(
                    template0, faces0, self._img_feat, at='verts'
                ).float()
                self._tri_feat = self.model.get_local_feature(
                    template0, faces0, self._img_feat, at='faces'
                ).float() if self.opts.dec_type == 'jacob' else None
                self._pred_id_coeff = self.model.encode_id(self._vert_feat, self._src_dfn_info)
                self._pred_seg_coeff = (
                    self.model.encode_seg(self._vert_feat, self._src_dfn_info)
                    if self.opts.design == 'new2' else None
                )
            else:
                self.model.model.update_precomputes(self._src_dfn_info)

        # ── global-coupling metric geometry (plan sec. 4.4, 6) ──────────
        area_np = (
            np.ones(template_np.shape[0], dtype=np.float32)
            if self.opts.no_area_weight else
            vertex_area_weights(template_np, faces_np).astype(np.float32)
        )
        self._area_weights = torch.from_numpy(area_np).to(device)  # [V]
        self._mesh_graph = build_mesh_graph(template_np, faces_np)

    def evaluate_gc(self):
        """
        Global-coupling / non-local motion leakage evaluation
        (see eval_CBD_bs/global_coupling_eval_plan.md).

        For each of the 53 ICT expression-basis one-hot activations on
        each of `--n_identity` ICT identities, self-retargets the
        activated expression back onto that identity's own neutral
        template and measures, per basis s:

            d^s      = X^s - X^0            (GT displacement, = exp_disp)
            d_hat^s  = X_hat^s - X^0        (predicted displacement)
            M^s      = active-region mask, from d^s via an abs+relative
                       threshold (locality_mask, plan sec. 5)
            dilate(M^s) = M^s geodesically dilated by --dilation_dist
                          (transition band, plan sec. 6)

            E_in^s     = area-weighted coordinate MSE over M_i^s
            E_out^s    = area-weighted coordinate MSE over
                         the far-field (outside dilate(M^s))
            E_signal^s = area-weighted coordinate signal MSE over M_i^s
            LSR^s      = 100 * sqrt(E_out^s / (E_signal^s + eps))   [%]

        Two metric sets are reported. "Raw end-to-end" uses
        X_hat^s-X^0 and includes neutral/identity reconstruction drift.
        "Motion-only" uses X_hat^s-X_hat^0, where X_hat^0 is obtained
        from a neutral-to-neutral pass on the same identity. The latter
        isolates expression-induced coupling, while E_neutral separately
        reports the removed baseline drift. LER measures total far-field
        residual energy relative to active signal energy, and P95_out
        reports the area-weighted 95th-percentile far-field RMSE.

        Unlike eval_CBD_bs.py's MSE-in/MSE-out (which compare the
        prediction against the model's own neutral->neutral
        self-reconstruction), E_in/E_out here compare directly against
        the ground-truth blendshape displacement, as specified by the
        eval plan: E_in measures whether the *correct* deformation is
        reproduced in the active region (not just "some" deformation --
        a model outputting garbage motion of the right magnitude would
        still be penalized), and E_out/LSR measure unintended motion in
        the far field, normalized by the basis's own signal magnitude so
        blendshapes with naturally large motion (e.g. jaw) don't dominate
        the average over blendshapes with naturally small motion (e.g.
        eyes). A trivial no-deformation model scores well on E_out/LSR
        but badly on E_in, so the two must be read together (plan sec. 3, 10).

        This script only implements the self-retargeting diagnostic (plan
        sec. 2); the cross-identity extension of plan sec. 9 (retargeting
        identity a's blendshape onto identity b and comparing against b's
        own ground-truth blendshape) is intentionally out of scope here,
        since it needs separate source/target template plumbing that
        isn't uniformly exposed across all model versions in
        `_forward_pass`.
        """
        eps = 1e-12

        # ── dataset ────────────────────────────────────────────────────
        self.dataset = GCEvalDataset(
            n_identity=self.opts.n_identity,
            identity_indices=self.opts.identity_indices,
            ict_split_set=self.opts.ict_split_set,
        )
        # Batch size is forced to exactly one identity's worth of items
        # (n_basis == 53), not taken from --batch_size: dataset items are
        # ordered identity-major/basis-minor, all 53 bases for one identity
        # share the same template/area-weights/mesh-graph, and the
        # per-identity code below (context caching, per-basis metric
        # bookkeeping) assumes each batch is exactly one identity.
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.dataset.n_basis,
            collate_fn=partial(GC_collate_wrapper, device=self.device),
        )

        # ── logging ────────────────────────────────────────────────────
        os.makedirs(self.opts.log_dir, exist_ok=True)

        ckpt_path = Path(self.opts.ckpt).name if self.opts.ckpt else 'no_ckpt'
        identity_tag = '_'.join(self.dataset.identity_names)
        area_tag = 'vertex' if self.opts.no_area_weight else 'area'
        metric_tag = (
            f'tau{self.opts.mask_tau_abs:g}-alpha{self.opts.mask_alpha:g}-'
            f'dil{self.opts.dilation_dist:g}-{area_tag}'
        )
        run_tag = ''
        if self.opts.run_name:
            run_tag = '-' + re.sub(r'[^A-Za-z0-9_.-]+', '_', self.opts.run_name)
        suffix = f'-gc-id{identity_tag}-{metric_tag}{run_tag}'
        self.opts.log_dir = os.path.join(self.opts.log_dir, ckpt_path + suffix)

        os.makedirs(self.opts.log_dir, exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)

        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), self.opts)

        self.logger = open(os.path.join(self.opts.log_dir, "log.txt"), 'w')
        print(f'Saving log at: {self.opts.log_dir}')

        print(self.dataset.get_data_config())
        self.logger.write(self.dataset.get_data_config())

        if self.opts.save_vert:
            save_vert_logdir = f"{self.opts.log_dir}/verts"
            os.makedirs(save_vert_logdir, exist_ok=True)

        if self.opts.NFR:
            self.model.model.eval()
        else:
            self.model.eval()

        # raw per-(identity, basis) records, kept for macro-average / median /
        # std / per-group / per-identity aggregation and dumped to
        # metrics.json for supplementary-material-style analysis (plan sec. 8)
        records = []

        # ── eval loop ──────────────────────────────────────────────────
        len_data = len(self.dataloader)
        current_identity = None

        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)
        for index, batch in pbar:
            with torch.no_grad():
                if batch.identity_name != current_identity:
                    template_np = batch.template[0].cpu().numpy()
                    faces_np = batch.faces[0].cpu().numpy()
                    self._load_identity_context(batch.identity_name, template_np, faces_np)
                    current_identity = batch.identity_name
                    if self.opts.save_vert:
                        os.makedirs(f"{save_vert_logdir}/{current_identity}", exist_ok=True)
                    os.makedirs(f"{self.opts.log_dir}/img/{current_identity}", exist_ok=True)

                    # Identity-specific neutral baseline. Subtracting this
                    # output from an expressive prediction isolates the motion
                    # induced by the expression and prevents neutral/identity
                    # reconstruction drift from being misread as coupling.
                    neutral_pred = self._forward_pass(
                        batch.template[:1], batch.template[:1],
                        batch.template_normal[:1], batch.template_normal[:1],
                        batch.faces[0], batch.mesh_data,
                    )
                    neutral_offset = neutral_pred - batch.template[:1]
                    neutral_sq = (neutral_offset ** 2).mean(-1)
                    neutral_area = self._area_weights[None, :]
                    E_neutral = (
                        (neutral_area * neutral_sq).sum(-1) /
                        neutral_area.sum(-1).clamp_min(eps)
                    )[0]

                pred_vertices = self._forward_pass(
                    batch.template, batch.vertices,
                    batch.template_normal, batch.vertices_normal,
                    batch.faces[0], batch.mesh_data,
                )

                # ── ground-truth / predicted displacement ───────────────
                gt_disp = batch.exp_disp                         # [B, V, 3]
                pred_raw_disp = pred_vertices - batch.template   # end-to-end output motion
                pred_motion_disp = pred_vertices - neutral_pred  # expression-induced motion

                # Per-coordinate MSE (mean over xyz), consistent with
                # torch.nn.functional.mse_loss rather than squared L2.
                residual_raw_sq = ((pred_raw_disp - gt_disp) ** 2).mean(-1)
                residual_motion_sq = ((pred_motion_disp - gt_disp) ** 2).mean(-1)
                gt_sq = (gt_disp ** 2).mean(-1)

                # ── active / transition / far-field masks (plan sec. 5, 6) ──
                mask_active = locality_mask(
                    gt_disp, self.opts.mask_tau_abs, self.opts.mask_alpha
                )  # [B, V] float

                B = mask_active.shape[0]
                mask_dilated = torch.zeros_like(mask_active)
                for b_idx in range(B):
                    active_np = mask_active[b_idx].cpu().numpy().astype(bool)
                    dilated_np = dilate_mask_geodesic(
                        active_np, self._mesh_graph, self.opts.dilation_dist
                    )
                    mask_dilated[b_idx] = torch.from_numpy(dilated_np.astype(np.float32)).to(self.device)
                mask_transition = (mask_dilated - mask_active).clamp(0.0, 1.0)
                mask_far = 1.0 - mask_dilated  # far-field: outside the transition band

                area = self._area_weights[None, :]  # [1, V]
                w_in = mask_active * area            # [B, V]
                w_far = mask_far * area              # [B, V]

                w_in_sum = w_in.sum(-1)
                w_far_sum = w_far.sum(-1)
                total_area = area.sum(-1).expand_as(w_in_sum)
                active_area_frac = w_in_sum / total_area.clamp_min(eps)
                transition_area_frac = (mask_transition * area).sum(-1) / total_area.clamp_min(eps)
                far_area_frac = w_far_sum / total_area.clamp_min(eps)

                signal_energy = (w_in * gt_sq).sum(-1)
                valid = (
                    (active_area_frac >= self.opts.min_region_area_frac) &
                    (far_area_frac >= self.opts.min_region_area_frac) &
                    (signal_energy > eps)
                )

                nan = torch.full_like(w_in_sum, float('nan'))
                E_signal = torch.where(valid, signal_energy / w_in_sum.clamp_min(eps), nan)

                def metric_set(residual_sq):
                    in_energy = (w_in * residual_sq).sum(-1)
                    far_energy = (w_far * residual_sq).sum(-1)
                    e_in = torch.where(valid, in_energy / w_in_sum.clamp_min(eps), nan)
                    e_out = torch.where(valid, far_energy / w_far_sum.clamp_min(eps), nan)
                    lsr = torch.where(
                        valid, 100.0 * torch.sqrt(e_out / E_signal.clamp_min(eps)), nan
                    )
                    # LER retains total far-field energy, so a large inactive
                    # surface cannot be hidden by a small per-vertex average.
                    ler = torch.where(valid, 100.0 * far_energy / signal_energy.clamp_min(eps), nan)
                    p95 = []
                    for row in range(B):
                        if valid[row]:
                            q95 = weighted_quantile(residual_sq[row], w_far[row], 0.95)
                            p95.append(torch.sqrt(q95.clamp_min(0.0)))
                        else:
                            p95.append(torch.tensor(float('nan'), device=self.device))
                    return e_in, e_out, lsr, ler, torch.stack(p95)

                E_in, E_out, LSR, LER, P95_out = metric_set(residual_raw_sq)
                E_in_motion, E_out_motion, LSR_motion, LER_motion, P95_out_motion = metric_set(
                    residual_motion_sq
                )

                def scalar_or_none(value):
                    value = float(value.detach().cpu())
                    return value if np.isfinite(value) else None

                for b_idx in range(B):
                    basis_name = batch.basis_names[b_idx]
                    records.append({
                        'identity': batch.identity_name,
                        'basis': basis_name,
                        'group': basis_group(basis_name),
                        'valid': bool(valid[b_idx].item()),
                        'active_area_frac': active_area_frac[b_idx].item(),
                        'transition_area_frac': transition_area_frac[b_idx].item(),
                        'far_area_frac': far_area_frac[b_idx].item(),
                        'E_neutral': E_neutral.item(),
                        'E_in': scalar_or_none(E_in[b_idx]),
                        'E_out': scalar_or_none(E_out[b_idx]),
                        'E_signal': scalar_or_none(E_signal[b_idx]),
                        'LSR': scalar_or_none(LSR[b_idx]),
                        'LER': scalar_or_none(LER[b_idx]),
                        'P95_out': scalar_or_none(P95_out[b_idx]),
                        'E_in_motion': scalar_or_none(E_in_motion[b_idx]),
                        'E_out_motion': scalar_or_none(E_out_motion[b_idx]),
                        'LSR_motion': scalar_or_none(LSR_motion[b_idx]),
                        'LER_motion': scalar_or_none(LER_motion[b_idx]),
                        'P95_out_motion': scalar_or_none(P95_out_motion[b_idx]),
                    })

                valid_count = int(valid.sum().item())
                if valid_count:
                    pbar.set_description(
                        f'[{batch.identity_name}] valid={valid_count}/{B} '
                        f'E_out(raw/motion): {E_out[valid].mean().item():.3e}/'
                        f'{E_out_motion[valid].mean().item():.3e}'
                    )
                else:
                    pbar.set_description(f'[{batch.identity_name}] valid=0/{B}')

            # ── save vertices ─────────────────────────────────────────
            curr_batch = pred_vertices.shape[0]
            if self.opts.save_vert:
                for b_idx in range(curr_batch):
                    save_vert_name = f"{save_vert_logdir}/{current_identity}/{index*curr_batch + b_idx:06d}.npy"
                    np.save(save_vert_name, pred_vertices[b_idx].detach().cpu().numpy())

            # ── GT region vs. predicted-leakage visualization ──────────
            # one image per basis in the batch (model forward is batched,
            # but rendering is still per-item -- matplotlib has no batched API)
            if not self.opts.no_vis:
                faces_np = batch.faces[0].cpu().numpy()
                # Visualize motion-only residual; raw neutral drift remains
                # available numerically as E_neutral and the raw metric set.
                residual_norm = torch.norm(pred_motion_disp - gt_disp, dim=-1).detach()

                for b_idx in range(curr_batch):
                    gt_v = batch.vertices[b_idx].cpu().numpy()
                    pred_v = pred_vertices[b_idx].detach().cpu().numpy()
                    region_color = region_mask_to_vertex_color(
                        mask_active[b_idx], mask_dilated[b_idx]
                    )
                    leak_color = leakage_to_vertex_color(
                        residual_norm[b_idx], self.opts.mask_tau_abs
                    )

                    plot_image_array_col(
                        [gt_v, pred_v], [faces_np, faces_np], [region_color, leak_color],
                        rot_list=[[0, 0, 0]] * 2,
                        size=3, bg_black=False,
                        logdir=f"{self.opts.log_dir}/img/{current_identity}",
                        name=f"{b_idx:03d}_{batch.basis_names[b_idx]}", save=True,
                    )

        # ── aggregate & write log (plan sec. 8) ──────────────────────────
        self._write_report(records)
        print('done!')

    # ------------------------------------------------------------------
    def _write_report(self, records):
        def stats(rows, key):
            vals = np.array(
                [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])],
                dtype=np.float64,
            )
            if vals.size == 0:
                return {'mean': None, 'median': None, 'std': None, 'n': 0}
            return {
                'mean': float(vals.mean()),
                'median': float(np.median(vals)),
                'std': float(vals.std()),
                'n': int(vals.size),
            }

        def fmt(s, digits=6):
            if s['mean'] is None:
                return 'N/A'
            spec = f'.{digits}e'
            return f"{s['mean']:{spec}}/{s['median']:{spec}}/{s['std']:{spec}}"

        def emit(title, rows):
            valid_rows = [r for r in rows if r['valid']]
            result = {
                'n_total': len(rows),
                'n_valid': len(valid_rows),
                'E_neutral': stats(rows, 'E_neutral'),
                'active_area_frac': stats(rows, 'active_area_frac'),
                'transition_area_frac': stats(rows, 'transition_area_frac'),
                'far_area_frac': stats(rows, 'far_area_frac'),
            }
            for key in (
                'E_in', 'E_out', 'E_signal', 'LSR', 'LER', 'P95_out',
                'E_in_motion', 'E_out_motion', 'LSR_motion', 'LER_motion',
                'P95_out_motion',
            ):
                result[key] = stats(valid_rows, key)

            lines = [
                f"[{title}] n={len(rows)}, valid={len(valid_rows)}, invalid={len(rows)-len(valid_rows)}",
                f"  Raw end-to-end | E_in: {fmt(result['E_in'])}  "
                f"E_out: {fmt(result['E_out'])}  LSR%: {fmt(result['LSR'], 3)}  "
                f"LER%: {fmt(result['LER'], 3)}  P95_out(RMSE): {fmt(result['P95_out'])}",
                f"  Motion-only    | E_in: {fmt(result['E_in_motion'])}  "
                f"E_out: {fmt(result['E_out_motion'])}  LSR%: {fmt(result['LSR_motion'], 3)}  "
                f"LER%: {fmt(result['LER_motion'], 3)}  "
                f"P95_out(RMSE): {fmt(result['P95_out_motion'])}",
                f"  Diagnostic     | E_neutral: {fmt(result['E_neutral'])}  "
                f"area fractions(active/transition/far): "
                f"{fmt(result['active_area_frac'], 3)} / "
                f"{fmt(result['transition_area_frac'], 3)} / "
                f"{fmt(result['far_area_frac'], 3)}",
            ]
            for line in lines:
                print(line)
                self.logger.write(line + "\n")
            return result

        summary = {}

        # ── overall (macro-average over every (identity, basis) pair) ──
        summary['overall'] = emit('Overall', records)

        # ── per-identity ─────────────────────────────────────────────
        if self.dataset.n_identity > 1:
            self.logger.write("[Per-identity]\n")
            print("[Per-identity]")
            summary['per_identity'] = {}
            for name in self.dataset.identity_names:
                rows = [r for r in records if r['identity'] == name]
                summary['per_identity'][name] = emit(f'  identity={name}', rows)

        # ── per basis-group (eye/brow/mouth/...) ────────────────────────
        self.logger.write("[Per-group]\n")
        print("[Per-group]")
        summary['per_group'] = {}
        groups_present = sorted(set(r['group'] for r in records))
        for group in groups_present:
            rows = [r for r in records if r['group'] == group]
            summary['per_group'][group] = emit(f'  group={group}', rows)

        self.logger.close()

        with open(os.path.join(self.opts.log_dir, "metrics.json"), 'w') as f:
            json.dump({'summary': summary, 'per_basis': records}, f, indent=2)

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
    Global-coupling / non-local motion leakage evaluation
    (see eval_CBD_bs/global_coupling_eval_plan.md).

    For one or more ICT neutral identities, activates each of the 53
    expression-basis axes one at a time (one-hot), self-retargets it, and
    reports, per basis and macro-averaged over all bases:
        E_in     -- active-region coordinate MSE vs. ground truth
        E_out    -- far-field (post-dilation) coordinate MSE vs. ground truth
        LSR      -- far-field RMSE normalized by active signal RMS, in %
        LER      -- total far-field error energy / active signal energy, in %
        P95_out  -- area-weighted 95th-percentile far-field RMSE

    Each is reported both as a raw end-to-end result and after subtracting
    the identity's neutral-to-neutral prediction (motion-only). E_neutral
    reports the neutral reconstruction drift separately.

    Examples:
        # NC
        python eval_CBD_gc.py --version 1 --ckpt ./ckpts_CBD/<ckpt-NC>

        # NFS
        python eval_CBD_gc.py --version 0 --ckpt ./ckpt_stage1/<ckpt-NFS>

        # NFR
        python eval_CBD_gc.py --version 0 --NFR --ckpt NFR-pretrained

        # ours (NGBC)
        python eval_CBD_gc.py --version 5 --ckpt ./ckpts_CBD/<ckpt-NGBCv5> --last_activation relu
    """
    mp.set_start_method('spawn', force=True)

    opts = Options()

    if opts.version == 0:
        opts.config = 'config/train_NFS.yml'
        opts_yaml = yaml.load(open(opts.config), Loader=yaml.FullLoader)
    else:
        config = f'{opts.ckpt}/train_opts.yml'
        opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)

    # Preserve checkpoint architecture settings unless the corresponding
    # architecture flag was explicitly supplied for evaluation. The parser
    # uses None for those flags, so ordinary evaluation defaults no longer
    # silently overwrite the training configuration.
    opts_cli = vars(opts)
    for key, value in opts_cli.items():
        if key in {'last_activation', 'no_pou', 'optim_cage', 'align_latent'} and value is None:
            continue
        opts_yaml[key] = value
    opts_yaml.setdefault('last_activation', 'relu')
    opts_yaml.setdefault('no_pou', False)
    opts_yaml.setdefault('optim_cage', False)
    opts_yaml.setdefault('align_latent', False)
    opts = argparse.Namespace(**opts_yaml)

    if opts.version == 0:
        if opts.NFR == False:
            opts.img_feat_dim = 128
            opts.feature_type = "cents&norms"
            opts.stage1 = True
            opts.scale_exp = 1.0
            opts.ict_face_only = False

            if opts.use_NFR:
                opts.design = "nfr"
                opts.dec_type = "jacob"
            else:
                opts.design = "new2"
                opts.dec_type = "disp"
        else:
            opts.ckpt = '/NFR'
            opts.img_feat_dim = 128
            opts.feature_type = "cents&norms"
            opts.stage1 = True
            opts.scale_exp = 1.0
            opts.ict_face_only = False
            opts.design = "nfr"
            opts.dec_type = "jacob"

    print('loaded version:', opts.version)

    trainer = Trainer(opts)
    trainer.evaluate_gc()
