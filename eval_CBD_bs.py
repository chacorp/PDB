import os
import glob
import json
import yaml
import random

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

from utils.matplotlib_rnd import plot_image_array_col
from utils.ckpt_utils import *
from utils.remesh_utils import ICT_face_model
from utils.keys import ICT_KEYS

from models.baseline import CageNet
from models.NGBC import NeuralGeneralizedBarycentricCoordinate
from models.NGBCv2 import NeuralBarycentricCoordinatev2, NeuralBarycentricCoordinatev3

import torch.multiprocessing as mp

from pathlib import Path
__abs_path__ = str(Path(__file__).parents[0].absolute())


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BasisEvalDataset(data.Dataset):
    """
    Global-coupling (expression-basis activation) evaluation dataset.

    For one or more ICT neutral identities, builds one item per
    (identity, expression-basis axis) pair: a one-hot activation of that
    basis on that identity.  `exp_disp` (= vertices - template) is the
    analytic per-vertex displacement caused solely by that basis, used to
    build the ground-truth locality mask (the region that basis is
    supposed to cover).  Since ICT expressions are additive blendshapes on
    top of the identity (id_exp_verts = neutral + id_disp(id) +
    exp_disp(exp)), `exp_disp` -- and therefore the GT mask -- does not
    depend on which identity is chosen.

    Items are ordered identity-major, basis-minor (all 53 bases for
    identity 0, then all 53 for identity 1, ...), so consecutive items
    share the same identity -- callers can cheaply detect an identity
    switch by comparing `identity_name` between items.

    By default this uses a single identity, "000" from
    `data/ICT_live_100/iden_vecs.npy` (equivalently `ict_face_pt/
    random_identity_vecs.npy[0]`) -- this is the same ICT identity pool
    used by `EvalDataset`/`NeutralEvalDataset` elsewhere in this repo for
    'ict' evaluation, and "000".."009" all have a precomputed dfn_info/
    operators/rendered-image cache on disk (see `evaluate_bs`'s precompute
    step) -- this lets NFS/NFR reuse that cache instead of rendering a
    fresh identity on the fly (which requires a CUDA-enabled pytorch3d
    rasterizer build not always available).

    Output per item:
        (vertices, template, vertices_normal, template_normal, faces,
         basis_name, mesh_data, exp_disp, identity_name)
    """

    def __init__(self, n_identity=1):
        super().__init__()
        self.ict = ICT_face_model()
        self.exp_names = ICT_KEYS
        self.n_basis = len(ICT_KEYS)  # 53
        self.mesh_data = torch.tensor([5])  # 'ict'

        id_vecs = np.load(f'{__abs_path__}/data/ICT_live_100/iden_vecs.npy')
        self.identity_names = [f'{i:03d}' for i in range(n_identity)]
        self.n_identity = n_identity

        self.faces_np = self.ict.faces

        self.items = []  # (vertices_np, template_np, exp_disp_np, basis_name, identity_name)
        for i, identity_name in enumerate(self.identity_names):
            id_coeff = id_vecs[i].astype(np.float32)
            for k in range(self.n_basis):
                exp_coeff = np.eye(self.n_basis, dtype=np.float32)[k]
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
        text  = "===========[BasisEvalDataset]===========\n"
        text += f"[Identities]: {', '.join(self.identity_names)}\n"
        text += f"[Total basis axes]: {self.n_basis}\n"
        text += f"[Total items]: {len(self.items)}\n"
        text += "==========================================\n"
        return text


class BSDataBatch:
    def __init__(self, data):
        transposed = list(zip(*data))
        self.vertices = torch.stack(transposed[0], 0)          # [B, V, 3]
        self.template = torch.stack(transposed[1], 0)          # [B, V, 3]
        self.vertices_normal = torch.stack(transposed[2], 0)   # [B, V, 3]
        self.template_normal = torch.stack(transposed[3], 0)   # [B, V, 3]
        self.faces = torch.stack(transposed[4], 0)              # [B, F, 3]
        self.basis_names = list(transposed[5])                  # basis name per item, len B
        self.mesh_data = transposed[6][0]
        self.exp_disp = torch.stack(transposed[7], 0)           # [B, V, 3]
        self.identity_name = transposed[8][0]                   # identity name (string);
        # all items in a batch share one identity -- see evaluate_bs's fixed
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


def BS_collate_wrapper(batch, device="cpu"):
    return BSDataBatch(batch).to(device)


# ---------------------------------------------------------------------------
# Mask / visualization helpers
# ---------------------------------------------------------------------------

def displacement_mask(disp, eps):
    """
    Args:
        disp: [..., V, 3] tensor of per-vertex displacement
        eps (float): hard threshold on displacement magnitude
    Returns:
        [..., V] float mask (1.0 where ||disp|| > eps, else 0.0)
    """
    return (torch.norm(disp, dim=-1) > eps).float()


def mask_to_vertex_color(mask_in, in_color=(0.90, 0.15, 0.15), out_color=(0.75, 0.75, 0.75)):
    """
    Args:
        mask_in: [V] tensor/array, 1.0 for vertices inside the basis region
    Returns:
        [V, 3] numpy vertex-color array
    """
    mask_in = mask_in.detach().cpu().numpy() if torch.is_tensor(mask_in) else mask_in
    mask_in = mask_in.astype(bool)
    vc = np.tile(np.array(out_color, dtype=np.float32), (mask_in.shape[0], 1))
    vc[mask_in] = np.array(in_color, dtype=np.float32)
    return vc


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def Options():
    parser = argparse.ArgumentParser(
        description='Global-coupling (expression-basis activation) evaluation for face retargeting'
    )
    parser.add_argument('-c', '--config', default='config/train_CBD.yml', help='config file path')
    parser.add_argument("--device",       type=str,   default="cuda:0")

    parser.add_argument("--log_dir",      type=str,   default="eval_CBD_bs")

    parser.add_argument("--version",      type=int,   default=1,
                        help='model version (0: NFS/NFR, 1: NC(baseline), 5/21/22: ours)')

    parser.add_argument("--last_activation", default="relu",
                        choices=["relu", "elu", "softmax", "softplus", "none", "sqrelu"],
                        help="Last-layer activation for NGBC.key_weight_model()")

    parser.add_argument("--no_pou", dest='no_pou', action='store_true')
    parser.set_defaults(no_pou=False)

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

    parser.add_argument("--use_NFR", dest='use_NFR', action='store_true')
    parser.set_defaults(use_NFR=False)

    parser.add_argument("--NFR", dest='NFR', action='store_true')
    parser.set_defaults(NFR=False)

    parser.add_argument("--optim_cage", dest='optim_cage', action='store_true')
    parser.set_defaults(optim_cage=False)

    parser.add_argument("--align_latent", dest='align_latent', action='store_true')
    parser.set_defaults(align_latent=False)

    parser.add_argument("--mask_eps", type=float, default=1e-3,
                        help='hard threshold on per-vertex displacement magnitude used to '
                             'build the GT / predicted basis-locality mask')

    parser.add_argument("--n_identity", type=int, default=1,
                        help='number of ICT identities ("000", "001", ... from '
                             'data/ICT_live_100/iden_vecs.npy) to evaluate over; '
                             'metrics are averaged over all (identity, basis) pairs')

    args = parser.parse_args()
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
                use_pou=~self.opts.no_pou,
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
                use_pou=~self.opts.no_pou,
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
                use_pou=~self.opts.no_pou,
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
            self.model.load_state_dict(ckpt_dict, strict=False)
        else:
            print('no ckpt found, training from scratch!')

    # ------------------------------------------------------------------
    def _forward_pass(self, template, vertices, template_normal, vertices_normal, faces, mesh_data):
        """
        Runs one self-retargeting forward pass (template = source & target
        neutral identity, vertices = driving expression mesh on that same
        identity) and returns pred_vertices [B, V, 3].  For version==0
        (NFS/NFR), relies on the per-identity features cached on `self`
        by `evaluate_bs` (identity is fixed for the whole run).
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
        (Re)builds all per-identity cached state -- NFS/NFR dfn_info /
        operators / rendered image, and the model's own neutral -> neutral
        self-reconstruction baseline (`self.pred_neutral`) -- for
        `identity_name`. Called once per identity (identities with a
        precomputed cache on disk are cheap to switch between; the network
        forward pass for `pred_neutral` is the only non-trivial cost).
        """
        device = self.device
        template0 = torch.tensor(template_np).float()[None].to(device)
        faces0 = torch.tensor(faces_np).long().to(device)
        template_normal0 = torch.tensor(
            igl.per_vertex_normals(template_np, faces_np)
        ).float()[None].to(device)

        if self.opts.version == 0:
            self._src_mesh = trimesh.Trimesh(vertices=template_np, faces=faces_np)

            # identities "000".."099" have a precomputed dfn_info / operators
            # / rendered image cache on disk (matches eval_CBD.py /
            # eval_CBD_lp.py's ict precompute handling); anything else falls
            # back to computing on the fly
            ict_precompute_path = '/data/sihun/ICT-audio2face/precompute-synth-fullhead'
            dfn_path = os.path.join(ict_precompute_path, f"{identity_name}_dfn_info.pkl")
            if os.path.exists(dfn_path):
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

        # ── baseline: model's own neutral -> neutral self-reconstruction
        # Used to isolate the deformation actually caused by each basis
        # from the model's baseline reconstruction noise (see docstring).
        self.pred_neutral = self._forward_pass(
            template0, template0, template_normal0, template_normal0,
            faces0, self.dataset.mesh_data,
        ).detach()

    def evaluate_bs(self):
        """
        Global-coupling (basis-activation) evaluation.

        For each of the 53 ICT expression-basis one-hot activations on
        each of `--n_identity` ICT identities, self-retargets the
        activated expression back onto that identity's own neutral
        template, then measures the prediction's deviation from the
        model's OWN neutral->neutral self-reconstruction (`pred_neutral`)
        -- not the raw GT template -- (a) inside and (b) outside the GT
        locality mask for that basis (built by thresholding the basis's
        own analytic displacement, which is identity-independent since
        ICT expressions are additive). Comparing against `pred_neutral`
        rather than the GT template isolates the deformation actually
        caused by this basis from the model's baseline neutral-
        reconstruction noise, which is otherwise large enough to swamp any
        locality signal (see the neutral->neutral sanity check this
        script's design was based on). A model free of global coupling
        should show a large MSE-in and a small MSE-out. Metrics are
        averaged over all (identity, basis) pairs.
        """
        # ── dataset ────────────────────────────────────────────────────
        self.dataset = BasisEvalDataset(n_identity=self.opts.n_identity)
        # Batch size is forced to exactly one identity's worth of items
        # (n_basis == 53), not taken from --batch_size: dataset items are
        # ordered identity-major/basis-minor, all 53 bases for one identity
        # share the same template/pred_neutral, and the per-identity code
        # below (context caching, per-identity metric bookkeeping) assumes
        # each batch is exactly one identity. This lets every basis for an
        # identity run through the model as a single batched forward pass
        # instead of 53 sequential single-item calls.
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.dataset.n_basis,
            collate_fn=partial(BS_collate_wrapper, device=self.device),
        )

        # ── logging ────────────────────────────────────────────────────
        os.makedirs(self.opts.log_dir, exist_ok=True)

        ckpt_path = self.opts.ckpt.split('/')[-1] if self.opts.ckpt else 'no_ckpt'
        suffix = '-bs' if self.opts.n_identity == 1 else f'-bs-{self.opts.n_identity}id'
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

        # per-identity metric bookkeeping, reported alongside the overall average
        per_identity = {name: {"MSE-in": 0.0, "MSE-out": 0.0} for name in self.dataset.identity_names}

        # ── eval loop ──────────────────────────────────────────────────
        len_data = len(self.dataloader)
        denom = 1 / len_data

        losses_val = {"MSE-in": 0.0, "MSE-out": 0.0}
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

                pred_vertices = self._forward_pass(
                    batch.template, batch.vertices,
                    batch.template_normal, batch.vertices_normal,
                    batch.faces[0], batch.mesh_data,
                )

                # ── metrics ───────────────────────────────────────────
                # Compared against `self.pred_neutral` (the model's own
                # neutral self-reconstruction), not the raw GT template --
                # see evaluate_bs docstring.
                mask_in = displacement_mask(batch.exp_disp, self.opts.mask_eps)   # [B, V]
                mask_out = 1.0 - mask_in

                # NOTE: F.mse_loss's default 'mean' reduction already averages
                # over every element of the batched tensor (all n_basis items
                # x V x 3), so MSE_in/MSE_out below are already the correct
                # per-identity average over all n_basis bases in one value --
                # do not divide by n_basis again when aggregating.
                MSE_in = F.mse_loss(
                    pred_vertices * mask_in.unsqueeze(-1),
                    self.pred_neutral * mask_in.unsqueeze(-1),
                ).item()
                MSE_out = F.mse_loss(
                    pred_vertices * mask_out.unsqueeze(-1),
                    self.pred_neutral * mask_out.unsqueeze(-1),
                ).item()
                losses_val['MSE-in'] += MSE_in
                losses_val['MSE-out'] += MSE_out
                # exactly one batch per identity (batch_size == n_basis), so
                # this is a plain assignment, not an accumulation
                per_identity[batch.identity_name]['MSE-in'] = MSE_in
                per_identity[batch.identity_name]['MSE-out'] = MSE_out

                pbar.set_description(
                    f'[{batch.identity_name}] MSE-in: {MSE_in:.5e} MSE-out: {MSE_out:.5e}'
                )

            # ── save vertices ─────────────────────────────────────────
            curr_batch = pred_vertices.shape[0]
            if self.opts.save_vert:
                for b_idx in range(curr_batch):
                    save_vert_name = f"{save_vert_logdir}/{current_identity}/{index*curr_batch + b_idx:06d}.npy"
                    np.save(save_vert_name, pred_vertices[b_idx].detach().cpu().numpy())

            # ── GT-mask vs pred-mask side-by-side visualization ────────
            # one image per basis in the batch (model forward is batched,
            # but rendering is still per-item -- matplotlib has no batched API)
            faces_np = batch.faces[0].cpu().numpy()
            pred_disp = (pred_vertices - self.pred_neutral).detach()
            pred_mask_in = displacement_mask(pred_disp, self.opts.mask_eps)

            for b_idx in range(curr_batch):
                gt_v = batch.vertices[b_idx].cpu().numpy()
                pred_v = pred_vertices[b_idx].detach().cpu().numpy()
                gt_color = mask_to_vertex_color(mask_in[b_idx])
                pred_color = mask_to_vertex_color(pred_mask_in[b_idx])

                plot_image_array_col(
                    [gt_v, pred_v], [faces_np, faces_np], [gt_color, pred_color],
                    rot_list=[[0, 0, 0]] * 2,
                    size=3, bg_black=False,
                    logdir=f"{self.opts.log_dir}/img/{current_identity}",
                    name=f"{b_idx:03d}_{batch.basis_names[b_idx]}", save=True,
                )

        # ── write log ─────────────────────────────────────────────────
        losses_val = {k: v * denom for k, v in losses_val.items()}
        losses_val['Ratio(out/in)'] = losses_val['MSE-out'] / max(losses_val['MSE-in'], 1e-12)

        log_text = "[BS Eval] "
        for key, value in losses_val.items():
            txt = f"{key}: {value:.6e} "
            print(txt)
            log_text += txt
        self.logger.write(log_text + "\n")

        if self.opts.n_identity > 1:
            # per_identity[name] already holds the average over that
            # identity's n_basis items (see the mse_loss note above) --
            # no further division needed here
            self.logger.write("[Per-identity]\n")
            for name, vals in per_identity.items():
                mse_in = vals['MSE-in']
                mse_out = vals['MSE-out']
                ratio = mse_out / max(mse_in, 1e-12)
                line = f"  [{name}] MSE-in: {mse_in:.6e} MSE-out: {mse_out:.6e} Ratio(out/in): {ratio:.6e}"
                print(line)
                self.logger.write(line + "\n")

        print('done!')

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
    Global-coupling (expression-basis activation) evaluation.

    For a single canonical ICT neutral identity, activates each of the 53
    expression-basis axes one at a time (one-hot), self-retargets it, and
    reports MSE-in (inside the GT basis-coverage mask -- should be large)
    vs. MSE-out (outside the mask -- should be small; large values indicate
    global coupling / leakage).

    Examples:
        # NC
        python eval_CBD_bs.py --version 1 --ckpt ./ckpts_CBD/<ckpt-NC>

        # NFS
        python eval_CBD_bs.py --version 0 --ckpt ./ckpt_stage1/<ckpt-NFS>

        # NFR
        python eval_CBD_bs.py --version 0 --NFR --ckpt NFR-pretrained

        # ours (NGBC)
        python eval_CBD_bs.py --version 5 --ckpt ./ckpts_CBD/<ckpt-NGBCv5> --last_activation relu
    """
    mp.set_start_method('spawn', force=True)

    opts = Options()

    if opts.version == 0:
        opts.config = 'config/train_NFS.yml'
        opts_yaml = yaml.load(open(opts.config), Loader=yaml.FullLoader)
    else:
        config = f'{opts.ckpt}/train_opts.yml'
        opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)

    opts_ = vars(opts)
    opts_yaml.update(opts_)
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
    trainer.evaluate_bs()
