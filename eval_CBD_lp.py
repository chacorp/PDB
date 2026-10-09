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
import datetime
from easydict import EasyDict
from utils.mesh_utils import calc_norm_torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data

from dataloader_CBD import (
    CBDdataSampler,
    CBDDataset,
    CBD_collate_wrapper,
    EvalDataset,
    CBDDataBatch_eval,
    CBD_collate_wrapper_eval,
)

from utils.matplotlib_rnd import plot_image_array, plot_image_array_seg, vis_rig
from utils.ckpt_utils import *
from utils.exp_utils import plateau_hat_points
from utils.keys import get_data_splits
from utils.remesh_utils import ICT_face_model

from models.baseline import CageNet
from models.NGBC import NeuralGeneralizedBarycentricCoordinate
from models.NGBCv2 import NeuralBarycentricCoordinatev2, NeuralBarycentricCoordinatev3

import torch.multiprocessing as mp

import sys
from pathlib import Path
__abs_path__ = str(Path(__file__).parents[0].absolute())


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NeutralEvalDataset(data.Dataset):
    """
    Linear-Preservation evaluation dataset.

    Returns the neutral template face as BOTH template and vertices for
    each unique identity in the test split.  Feeding template == vertices
    to the retargeting model exercises the zero-expression (neutral) case;
    the expected prediction should be the same neutral face.

    Output per item (matches CBDDataBatch_eval format):
        (template, template, template_normal, template_normal, faces, id_name, mesh_data)
    """

    def __init__(self, data_name, data_basedir='/data/sihun', toggle=False):
        super().__init__()
        self.data_name = data_name

        data_name_list = [
            'voca', 'biwi', 'mf_SEN', 'coma', 'mf_ROM',
            'ict', 'ict-cap', 'ict_face_only',
        ]
        if data_name not in data_name_list:
            raise ValueError(f'Unknown data_name [{data_name}]. Choose from {data_name_list}')
        d_mask = [dn == data_name for dn in data_name_list]
        self.mesh_data = torch.arange(len(data_name_list))[d_mask]

        _, voca_split, biwi_split, mf_split, _ = get_data_splits()
        template_basedir = data_basedir if toggle else data_basedir + '/pca'

        self.items = []  # list of (template_np [V,3], faces_np [F,3], id_name str)

        if data_name == 'voca':
            voca_std = np.load(
                f"{__abs_path__}/utils/voca/standardization.npy", allow_pickle=True
            ).item()
            with open(f"{template_basedir}/VOCA-COMA/voca_templates.pkl", 'rb') as f:
                voca_mesh = pickle.load(f)
            faces_np = voca_std['new_f']  # numpy int64
            for id_name in voca_split['test']:
                if id_name in voca_mesh:
                    self.items.append((voca_mesh[id_name], faces_np, id_name))

        elif data_name == 'biwi':
            with open(
                f"{template_basedir}/BIWI_align_deci/templates_align_deci.pkl", 'rb'
            ) as f:
                biwi_mesh = pickle.load(f)
            faces_np = biwi_mesh['face']  # numpy int64
            for id_name in biwi_split['test']:
                if id_name in biwi_mesh:
                    self.items.append((biwi_mesh[id_name], faces_np, id_name))

        elif data_name == 'coma':
            coma_std = np.load(
                f"{__abs_path__}/utils/voca/standardization.npy", allow_pickle=True
            ).item()
            with open(f"{template_basedir}/VOCA-COMA/voca_templates.pkl", 'rb') as f:
                coma_mesh = pickle.load(f)
            faces_np = coma_std['new_f']  # numpy int64
            for id_name in voca_split['test']:
                if id_name in coma_mesh:
                    self.items.append((coma_mesh[id_name], faces_np, id_name))

        elif data_name in ('mf_SEN', 'mf_ROM'):
            mf_std = np.load(
                f"{__abs_path__}/utils/mf/standardization.npy", allow_pickle=True
            ).item()
            with open(f"{template_basedir}/multiface_align/mf_templates.pkl", 'rb') as f:
                mf_mesh = pickle.load(f)
            faces_np = mf_std['new_f'].numpy()  # torch.Tensor → numpy
            for id_name in mf_split['test']:
                if id_name in mf_mesh:
                    self.items.append((mf_mesh[id_name], faces_np, id_name))

        elif data_name == 'ict':
            iden_vecs = np.load(f'{__abs_path__}/data/ICT_live_100/iden_vecs.npy')
            ict_face = ICT_face_model()
            faces_np = ict_face.faces  # numpy int64
            for i, id_coeff in enumerate(iden_vecs):
                # exp_coeffs=None → exp_disp=0 → neutral (identity-only) face
                _, template_np, _ = ict_face.apply_coeffs(id_coeff, None, return_all=True)
                self.items.append((template_np[0], faces_np, f'{i:03d}'))

        else:
            raise NotImplementedError(f'data_name [{data_name}] not supported for LP eval')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        template_np, faces_np, id_name = self.items[index]

        template_normal_np = igl.per_vertex_normals(template_np, faces_np)

        template = torch.tensor(template_np).float()
        template_normal = torch.tensor(template_normal_np).float()
        faces = torch.tensor(faces_np).long()

        # vertices == template: neutral face with zero expression displacement
        return template, template, template_normal, template_normal, faces, id_name, self.mesh_data

    def get_data_config(self):
        text  = "===========[NeutralEvalDataset]===========\n"
        text += f"[Dataset]: {self.data_name}\n"
        text += f"[Mode]: LP (Linear Preservation — neutral→neutral)\n"
        text += f"[Total identities]: {len(self.items)}\n"
        text += "==========================================\n"
        return text


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def Options():
    parser = argparse.ArgumentParser(
        description='Linear-preservation evaluation for face retargeting'
    )
    parser.add_argument('-c', '--config', default='config/train_CBD.yml', help='config file path')
    parser.add_argument("--device",       type=str,   default="cuda:0")

    parser.add_argument("--log_dir",      type=str,   default="eval_CBD_lp")

    parser.add_argument("--version",      type=int,   default=1,
                        help='model version (1: baseline, 5/21/22: ours)')

    parser.add_argument("--data_selection", type=int, default=-1,
                        help='select dataset (0:voca, 1:biwi, 2:mf_SEN, 3:coma, 4:mf_ROM, 5:ict)')

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

    parser.add_argument("--use_data0", dest='use_data0', action='store_true')
    parser.set_defaults(use_data0=False)
    parser.add_argument("--use_data1", dest='use_data1', action='store_true')
    parser.set_defaults(use_data1=False)
    parser.add_argument("--use_data2", dest='use_data2', action='store_true')
    parser.set_defaults(use_data2=False)
    parser.add_argument("--use_data3", dest='use_data3', action='store_true')
    parser.set_defaults(use_data3=False)
    parser.add_argument("--use_data8", dest='use_data8', action='store_true')
    parser.set_defaults(use_data8=False)
    parser.add_argument("--use_data9", dest='use_data9', action='store_true')
    parser.set_defaults(use_data9=False)

    parser.add_argument("--tb", action='store_true')
    parser.set_defaults(is_train=True)

    parser.add_argument("--use_t_mask", dest='use_t_mask', action='store_true')
    parser.set_defaults(use_t_mask=False)

    parser.add_argument("--laplacian", dest='laplacian', action='store_true')
    parser.set_defaults(laplacian=False)

    parser.add_argument("--save_vert", dest='save_vert', action='store_true')
    parser.set_defaults(save_vert=False)
    parser.add_argument("--save_gt", dest='save_gt', action='store_true')
    parser.set_defaults(save_gt=False)

    parser.add_argument("--use_NFR", dest='use_NFR', action='store_true')
    parser.set_defaults(use_NFR=False)

    parser.add_argument("--NFR", dest='NFR', action='store_true')
    parser.set_defaults(NFR=False)

    parser.add_argument("--optim_cage", dest='optim_cage', action='store_true')
    parser.set_defaults(optim_cage=False)

    parser.add_argument("--align_latent", dest='align_latent', action='store_true')
    parser.set_defaults(align_latent=False)

    parser.add_argument("--cross", dest='cross', action='store_true',
                        help='cross-identity LP eval: encode expression from --src_name, '
                             'decode onto --tgt_name template (GT = tgt template)')
    parser.set_defaults(cross=False)
    parser.add_argument("--src_name", type=str, default=None,
                        help='source dataset name for --cross (e.g. mf_SEN, ict)')
    parser.add_argument("--tgt_name", type=str, default=None,
                        help='target dataset name for --cross (e.g. mf_SEN, ict)')

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Helpers (kept from eval_CBD.py for NFS compatibility)
# ---------------------------------------------------------------------------

def gaussian_kernel1d(kernel_size=5, sigma=1.0):
    x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel

def apply_gaussian_filter(tensor, kernel_size=5, sigma=1.0):
    kernel_size = int(kernel_size)
    kernel = gaussian_kernel1d(kernel_size, sigma)
    kernel = kernel.reshape(1, 1, -1).to(tensor.device)
    kernel = kernel.repeat(tensor.size(1), 1, 1)
    tensor = tensor.transpose(0, 1).unsqueeze(0)
    filtered_tensor = F.conv1d(tensor, kernel, padding=(kernel_size // 2), groups=tensor.size(1))
    filtered_tensor = filtered_tensor.squeeze(0).transpose(0, 1)
    return filtered_tensor


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

        elif opts.version == 5 or opts.version == 6:
            model_cls = NeuralGeneralizedBarycentricCoordinate
            if opts.version == 6:
                from models.PDBplus import PDBplus
                model_cls = PDBplus
            self.model = model_cls(
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
            if isinstance(ckpt_dict, dict) and 'model' in ckpt_dict:
                ckpt_dict = ckpt_dict['model']
            self.model.load_state_dict(ckpt_dict, strict=False)
        else:
            print('no ckpt found, training from scratch!')

    # ------------------------------------------------------------------
    def evaluate_lp(self):
        """
        Linear-Preservation evaluation.

        Feeds the neutral template face as BOTH source mesh and source
        expression to the retargeting model.  GT = template (neutral).
        A well-behaved model should reproduce the neutral face exactly,
        so low MSE indicates good linear preservation.
        """
        device = self.device

        # ── dataset selection ──────────────────────────────────────────
        if self.opts.data_selection == -1:
            raise NotImplementedError('LP eval only works for individual datasets')

        data_name_list = ['voca', 'biwi', 'mf_SEN', 'coma', 'mf_ROM', 'ict']
        selection = data_name_list[self.opts.data_selection]

        self.mf_precompute_path = '/data/sihun/multiface_align/precomputes'
        self.ict_precompute_path = '/data/sihun/ICT-audio2face/precompute-synth-fullhead'

        self.dataset = NeutralEvalDataset(data_name=selection, toggle=False)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.opts.batch_size,
            collate_fn=partial(CBD_collate_wrapper_eval, device=self.device),
            num_workers=8,
        )

        # ── logging ────────────────────────────────────────────────────
        os.makedirs(self.opts.log_dir, exist_ok=True)

        ckpt_path = self.opts.ckpt.split('/')[-1]
        self.opts.log_dir = os.path.join(self.opts.log_dir, ckpt_path + '-lp', selection)

        if self.opts.use_t_mask:
            self.opts.log_dir = self.opts.log_dir + '-masked'
        if self.opts.laplacian:
            self.opts.log_dir = self.opts.log_dir + '-laplacian'

        os.makedirs(self.opts.log_dir, exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)

        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)

        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), opts)

        self.logger = open(os.path.join(self.opts.log_dir, "log.txt"), 'w')
        print(f'Saving log at: {self.opts.log_dir}')

        print(self.dataset.get_data_config())
        self.logger.write(self.dataset.get_data_config())

        # ── eval loop ──────────────────────────────────────────────────
        len_data = len(self.dataloader)
        denom = 1 / len_data
        interv_val = max(1, round(len_data / 10))

        if self.opts.NFR:
            self.model.model.eval()
        else:
            self.model.eval()

        losses_val = {"MSE": 0.0}
        if self.opts.use_t_mask:
            losses_val["MSE-in"] = 0.0
            losses_val["MSE-out"] = 0.0
        if self.opts.laplacian:
            losses_val["Lap"] = 0.0

        if self.opts.save_gt:
            save_gt_logdir = f"{self.opts.log_dir}/../../GT_{selection}_lp"
            os.makedirs(save_gt_logdir, exist_ok=True)
        if self.opts.save_vert:
            save_vert_logdir = f"{self.opts.log_dir}/verts"
            os.makedirs(save_vert_logdir, exist_ok=True)

        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)
        for index, batch in pbar:
            # NeutralEvalDataset already sets batch.vertices == batch.template.
            # The model receives the neutral face as both source and expression.
            with torch.no_grad():

                if self.opts.version == 0:
                    # ── NFS ───────────────────────────────────────────
                    if self.opts.NFR == False:
                        # In LP eval each batch is a unique identity — always recompute
                        src_mesh = trimesh.Trimesh(
                            vertices=batch.template[0].cpu().numpy(),
                            faces=batch.faces[0].cpu().numpy(),
                        )
                        if self.opts.laplacian:
                            tmp_L = igl.cotmatrix(src_mesh.vertices, src_mesh.faces)
                            src_L = torch.sparse_csc_tensor(
                                torch.LongTensor(tmp_L.indptr).to(device),
                                torch.LongTensor(tmp_L.indices).to(device),
                                torch.FloatTensor(tmp_L.data).to(device),
                                tmp_L.shape,
                            )

                        # mf/ict identities have precomputed dfn_info/operators/img on disk;
                        # everything else is computed on-the-fly (matches eval_CBD.py)
                        if batch.mesh_data in [2, 4, 5]:
                            if batch.mesh_data in [2, 4]:
                                precompute_path = self.mf_precompute_path
                            else:
                                precompute_path = self.ict_precompute_path
                            src_mesh_id = batch.id_name
                            src_dfn_info = pickle.load(open(
                                os.path.join(precompute_path, f"{src_mesh_id}_dfn_info.pkl"), 'rb'
                            ))
                            src_operators = pickle.load(open(
                                os.path.join(precompute_path, f"{src_mesh_id}_operators.pkl"), mode='rb'
                            ))
                            src_img = np.load(os.path.join(precompute_path, f"{src_mesh_id}_img.npy"))
                            # keep the batch dim ([1, 256, 256, 3]) -- matches what
                            # render_img() itself returns; NFR's encode() requires a
                            # 4D image (indexing [0] here breaks NFR, though NFS's
                            # get_img_feat happens to re-add a missing batch dim)
                            src_img = torch.from_numpy(src_img).float().to(self.device)
                        else:
                            src_dfn_info = self.get_dfn_info(src_mesh, map_location=self.device)
                            src_operators = self.get_mesh_operators(src_mesh)
                            src_img = self.model.renderer.render_img(src_mesh).float().to(self.device)

                        img_feat = self.model.get_img_feat(src_img)
                        vert_feat = self.model.get_local_feature(
                            batch.template[0][None], batch.faces[0], img_feat, at='verts'
                        ).float()
                        if self.opts.dec_type == 'jacob':
                            tri_feat = self.model.get_local_feature(
                                batch.template[0][None], batch.faces[0], img_feat, at='faces'
                            ).float()
                        else:
                            tri_feat = None
                        pred_id_coeff = self.model.encode_id(vert_feat, src_dfn_info)
                        pred_seg_coeff = (
                            self.model.encode_seg(vert_feat, src_dfn_info)
                            if self.opts.design == 'new2' else None
                        )
                        # batch.vertices == batch.template → encodes neutral expression
                        vert_feat_exp = self.model.get_local_feature(
                            batch.vertices, batch.faces[0], img_feat
                        ).float()
                        pred_exp_coeff = self.model.encode_exp(
                            vert_feat_exp, src_dfn_info, batch_process=True, verbose=False
                        )
                        inputs = (
                            tri_feat if self.opts.dec_type == 'jacob' else vert_feat,
                            pred_exp_coeff, pred_id_coeff, pred_seg_coeff,
                            None, batch.template[0][None], batch.faces[0], src_operators,
                        )
                        decode_out = self.model.decode(inputs, tgt_mesh=src_mesh, batch_process=True)
                        pred_vertices = decode_out[0] if isinstance(decode_out, tuple) else decode_out

                    else:
                        # NFR branch — each LP batch is a unique identity, always recompute
                        src_verts = batch.template[0]
                        src_faces = batch.faces[0]
                        src_m = trimesh.Trimesh(
                            vertices=src_verts.cpu().numpy(),
                            faces=src_faces.cpu().numpy(),
                        )
                        if self.opts.laplacian:
                            tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                            src_L = torch.sparse_csc_tensor(
                                torch.LongTensor(tmp_L.indptr).to(device),
                                torch.LongTensor(tmp_L.indices).to(device),
                                torch.FloatTensor(tmp_L.data).to(device),
                                tmp_L.shape,
                            )

                        if batch.mesh_data in [2, 4, 5]:
                            if batch.mesh_data in [2, 4]:
                                precompute_path = self.mf_precompute_path
                            else:
                                precompute_path = self.ict_precompute_path
                            src_mesh_id = batch.id_name
                            src_dfn_info = pickle.load(open(
                                os.path.join(precompute_path, f"{src_mesh_id}_dfn_info.pkl"), 'rb'
                            ))
                            src_operators = pickle.load(open(
                                os.path.join(precompute_path, f"{src_mesh_id}_operators.pkl"), mode='rb'
                            ))
                            src_img = np.load(os.path.join(precompute_path, f"{src_mesh_id}_img.npy"))
                            # keep the batch dim ([1, 256, 256, 3]) -- NFR's encode()
                            # requires a 4D image (indexing [0] here breaks NFR)
                            src_img = torch.from_numpy(src_img).float().to(self.device)
                        else:
                            src_dfn_info = self.get_dfn_info(src_m, map_location=self.device)
                            src_operators = self.get_mesh_operators(src_m)
                            src_img = self.model.renderer.render_img(src_m).float().to(device)

                        inputs_v = self.model.get_inputs(batch.vertices, batch.faces[0])
                        self.model.model.update_precomputes(src_dfn_info)
                        pred_exp = self.model.model.encode(
                            inputs_v, src_img.to(device), N_F=src_m.faces.shape[0]
                        )
                        pred_vertices, _, _ = self.model.calc_new_mesh(
                            src_verts, src_faces, pred_exp, src_operators, src_dfn_info, src_img
                        )
                        # calc_new_mesh's Poisson solve only recovers the mesh up to an
                        # arbitrary rigid translation (integrating a Jacobian field has no
                        # translation term), so it re-centers its output to zero mean
                        # internally (see NFR_helper.calc_new_mesh). batch.vertices is in
                        # ICT's own (non-zero-mean) coordinate frame, so comparing the two
                        # directly would count that translation gap as error on top of the
                        # real reconstruction error. Re-align pred to GT's mean here so the
                        # downstream MSE measures shape accuracy only -- matches how NFS/NGBC
                        # already sit in GT's frame with no re-centering needed.
                        dims = tuple(range(pred_vertices.dim() - 1))
                        pred_vertices = (
                            pred_vertices
                            - pred_vertices.mean(dim=dims, keepdim=True)
                            + batch.vertices.mean(dim=dims, keepdim=True)
                        )

                else:
                    # ── NGBC / CageNet ────────────────────────────────
                    if index == 0:
                        src_verts = batch.template[0]
                        src_faces = batch.faces[0]
                        src_m = trimesh.Trimesh(
                            vertices=src_verts.cpu().numpy(),
                            faces=src_faces.cpu().numpy(),
                        )
                        if self.opts.laplacian:
                            tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                            src_L = torch.sparse_csc_tensor(
                                torch.LongTensor(tmp_L.indptr).to(device),
                                torch.LongTensor(tmp_L.indices).to(device),
                                torch.FloatTensor(tmp_L.data).to(device),
                                tmp_L.shape,
                            )
                    else:
                        if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:
                            src_verts = batch.template[0]
                            src_faces = batch.faces[0]
                            src_m = trimesh.Trimesh(
                                vertices=src_verts.cpu().numpy(),
                                faces=src_faces.cpu().numpy(),
                            )
                            if self.opts.laplacian:
                                tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                                src_L = torch.sparse_csc_tensor(
                                    torch.LongTensor(tmp_L.indptr).to(device),
                                    torch.LongTensor(tmp_L.indices).to(device),
                                    torch.FloatTensor(tmp_L.data).to(device),
                                    tmp_L.shape,
                                )

                    if self.opts.version == 1:
                        pred_vertices, _ = self.model.retarget(
                            batch.template, batch.vertices, batch.template
                        )
                    elif self.opts.version == 21:
                        pred_vertices, recon_vertices, recon_source, exp_z, \
                        pred_source, _, _, _, _, _, _ = self.model(
                            batch.template, batch.vertices,
                            batch.template_normal, batch.vertices_normal,
                            batch.mesh_data, epoch=0,
                        )
                        pred_vertices = pred_vertices - pred_source + batch.template
                    elif self.opts.version == 22:
                        (
                            pred_vertices, _, pred_source, _,
                            src_exp_z, _, _, _, _, _, _, _, _, _
                        ) = self.model(
                            batch.template, batch.vertices,
                            batch.template_normal, batch.vertices_normal,
                            batch.mesh_data, epoch=0,
                        )
                    else:
                        pred_vertices, _, _, _, \
                        _, _, _, _, _ = self.model(
                            batch.template, batch.vertices,
                            batch.template_normal, batch.vertices_normal,
                            mesh_data=batch.mesh_data, epoch=0,
                        )

                # ── metrics ───────────────────────────────────────────
                if self.opts.use_t_mask:
                    inner_mask = plateau_hat_points(batch.template, r0=1.0, r1=2.25)
                    outter_mask = 1 - inner_mask

                    MSE_in = F.mse_loss(
                        batch.vertices * inner_mask,
                        pred_vertices * inner_mask,
                    ).item()
                    losses_val['MSE-in'] += MSE_in

                    MSE_out = F.mse_loss(
                        batch.template * outter_mask,
                        pred_vertices * outter_mask,
                    ).item()
                    losses_val['MSE-out'] += MSE_out

                    if self.opts.laplacian:
                        MSE_lap = (
                            tmp_L @ pred_vertices.squeeze().detach().cpu().numpy()
                        ) * inner_mask.cpu().numpy()
                        MSE_lap = MSE_lap.mean()
                        losses_val["Lap"] += MSE_lap
                else:
                    if self.opts.laplacian:
                        MSE_lap = tmp_L @ pred_vertices.squeeze().detach().cpu().numpy()
                        MSE_lap = MSE_lap.mean()
                        losses_val["Lap"] += MSE_lap

                MSE = F.mse_loss(
                    batch.vertices,
                    pred_vertices,
                ).item()
                losses_val['MSE'] += MSE

                pbar_txt = f'MSE: {MSE:.5e}'
                if self.opts.use_t_mask:
                    pbar_txt = f'MSE-in: {MSE_in:.5e}'
                if self.opts.laplacian:
                    pbar_txt += f'\tLap: {MSE_lap:.5e}'
                pbar.set_description(pbar_txt)

            # ── save vertices / GT ────────────────────────────────────
            if self.opts.save_gt:
                curr_batch = batch.vertices.shape[0]
                for b_idx in range(curr_batch):
                    save_gt_name = f"{save_gt_logdir}/{index*curr_batch + b_idx:06d}.npy"
                    np.save(save_gt_name, batch.vertices[b_idx].cpu().numpy())

            if self.opts.save_vert:
                curr_batch = pred_vertices.shape[0]
                for b_idx in range(curr_batch):
                    save_vert_name = f"{save_vert_logdir}/{index*curr_batch + b_idx:06d}.npy"
                    np.save(save_vert_name, pred_vertices[b_idx].detach().cpu().numpy())

            # ── visualization ─────────────────────────────────────────
            if index % interv_val == 0:
                vertices = batch.vertices.cpu()
                faces = batch.faces.cpu()
                pred_vertices_ = pred_vertices.detach().cpu()

                v_list = [vertices[0], pred_vertices_[0]]
                f_list = [faces[0]] * len(v_list)
                plot_image_array(
                    v_list, f_list,
                    rot_list=[[0, 0, 0]] * len(v_list),
                    size=1, bg_black=False, mode='shade',
                    logdir=f"{self.opts.log_dir}/img",
                    name=f"{index:04d}", save=True,
                )

        # ── write log ─────────────────────────────────────────────────
        log_text = "[LP Eval] "
        for key, value in losses_val.items():
            value = value * denom
            txt = f"{key}: {value:.6e} "
            print(txt)
            log_text += txt
        self.logger.write(log_text + "\n")
        print('done!')

    # ------------------------------------------------------------------
    def get_lp_precomputes(self, id_name, verts_np, faces_np, mesh_data_idx, need_operators=True):
        """
        mf/ict identities use precomputed dfn_info/operators/img (matches eval_CBD.py);
        everything else is computed on-the-fly.
        """
        m = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        if mesh_data_idx in [2, 4, 5]:
            precompute_path = self.mf_precompute_path if mesh_data_idx in [2, 4] else self.ict_precompute_path
            dfn_info = pickle.load(open(
                os.path.join(precompute_path, f"{id_name}_dfn_info.pkl"), 'rb'
            ))
            operators = None
            if need_operators:
                operators = pickle.load(open(
                    os.path.join(precompute_path, f"{id_name}_operators.pkl"), mode='rb'
                ))
            img = np.load(os.path.join(precompute_path, f"{id_name}_img.npy"))
            # keep the batch dim ([1, 256, 256, 3]) -- NFR's encode() requires a
            # 4D image (indexing [0] here breaks NFR)
            img = torch.from_numpy(img).float().to(self.device)
        else:
            dfn_info = self.get_dfn_info(m, map_location=self.device)
            operators = self.get_mesh_operators(m) if need_operators else None
            img = self.model.renderer.render_img(m).float().to(self.device)
        return m, dfn_info, operators, img

    # ------------------------------------------------------------------
    def evaluate_lp_cross(self, src_name, tgt_name):
        """
        Cross-identity Linear-Preservation evaluation.

        Encodes the (zero) expression from a SOURCE identity's own neutral
        face, then decodes it onto a DIFFERENT TARGET identity's template.
        Since the source expression is neutral, GT = the target's own
        neutral template — a well-behaved model should reproduce the
        target's neutral face regardless of which identity supplied the
        "expression".

        Implemented for:
          - NFS path (version=0, --NFR off): encode_id/encode_exp/decode
            interface shared by design='nfr'|'new2'.
          - NGBC path (version=5): uses NeuralGeneralizedBarycentricCoordinate's
            own `retarget()` method (geometry-only cage model, no img
            features / precompute needed).
        """
        if self.opts.version == 5 or self.opts.version == 6:
            return self._evaluate_lp_cross_ngbc(src_name, tgt_name)
        if not (self.opts.version == 0 and self.opts.NFR == False):
            raise NotImplementedError(
                'cross LP eval only implemented for version=0 with --NFR off '
                '(the NFS encode_id/encode_exp/decode interface), or version=5/6 (NGBC/PDBplus)'
            )

        device = self.device
        self.mf_precompute_path = '/data/sihun/multiface_align/precomputes'
        self.ict_precompute_path = '/data/sihun/ICT-audio2face/precompute-synth-fullhead'

        name_to_idx = {
            'voca': 0, 'biwi': 1, 'mf_SEN': 2, 'coma': 3, 'mf_ROM': 4,
            'ict': 5, 'ict-cap': 6, 'ict_face_only': 7,
        }
        src_mesh_data = name_to_idx[src_name]
        tgt_mesh_data = name_to_idx[tgt_name]

        src_dataset = NeutralEvalDataset(data_name=src_name, toggle=False)
        tgt_dataset = NeutralEvalDataset(data_name=tgt_name, toggle=False)

        # ── logging ────────────────────────────────────────────────────
        os.makedirs(self.opts.log_dir, exist_ok=True)
        ckpt_path = self.opts.ckpt.split('/')[-1]
        self.opts.log_dir = os.path.join(
            self.opts.log_dir, ckpt_path + '-lp-cross', f'{src_name}-to-{tgt_name}'
        )
        if self.opts.use_t_mask:
            self.opts.log_dir = self.opts.log_dir + '-masked'
        os.makedirs(self.opts.log_dir, exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)

        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), self.opts)

        self.logger = open(os.path.join(self.opts.log_dir, "log.txt"), 'w')
        print(f'Saving log at: {self.opts.log_dir}')

        print(src_dataset.get_data_config())
        print(tgt_dataset.get_data_config())
        self.logger.write(src_dataset.get_data_config())
        self.logger.write(tgt_dataset.get_data_config())

        # ── eval loop ──────────────────────────────────────────────────
        self.model.eval()

        pairs = [(s, t) for s in src_dataset.items for t in tgt_dataset.items]
        len_data = len(pairs)
        denom = 1 / len_data
        interv_val = max(1, round(len_data / 10))

        losses_val = {"MSE": 0.0}
        if self.opts.use_t_mask:
            losses_val["MSE-in"] = 0.0
            losses_val["MSE-out"] = 0.0

        pbar = tqdm(enumerate(pairs), total=len_data, ncols=100)
        for index, ((src_v_np, src_f_np, src_id), (tgt_v_np, tgt_f_np, tgt_id)) in pbar:
            with torch.no_grad():
                tgt_m, tgt_dfn_info, tgt_operators, tgt_img = self.get_lp_precomputes(
                    tgt_id, tgt_v_np, tgt_f_np, tgt_mesh_data, need_operators=True
                )
                _, src_dfn_info, _, src_img = self.get_lp_precomputes(
                    src_id, src_v_np, src_f_np, src_mesh_data, need_operators=False
                )

                tgt_verts = torch.tensor(tgt_v_np).float().to(device)
                tgt_faces = torch.tensor(tgt_f_np).long().to(device)
                src_verts = torch.tensor(src_v_np).float().to(device)
                src_faces = torch.tensor(src_f_np).long().to(device)

                tgt_img_feat = self.model.get_img_feat(tgt_img)
                tgt_vert_feat = self.model.get_local_feature(
                    tgt_verts[None], tgt_faces, tgt_img_feat, at='verts'
                ).float()
                tgt_tri_feat = None
                if self.opts.dec_type == 'jacob':
                    tgt_tri_feat = self.model.get_local_feature(
                        tgt_verts[None], tgt_faces, tgt_img_feat, at='faces'
                    ).float()

                pred_id_coeff = self.model.encode_id(tgt_vert_feat, tgt_dfn_info)
                pred_seg_coeff = (
                    self.model.encode_seg(tgt_vert_feat, tgt_dfn_info)
                    if self.opts.design == 'new2' else None
                )

                src_img_feat = self.model.get_img_feat(src_img)
                src_vert_feat_exp = self.model.get_local_feature(
                    src_verts[None], src_faces, src_img_feat
                ).float()
                pred_exp_coeff = self.model.encode_exp(
                    src_vert_feat_exp, src_dfn_info, batch_process=True, verbose=False
                )

                inputs = (
                    tgt_tri_feat if self.opts.dec_type == 'jacob' else tgt_vert_feat,
                    pred_exp_coeff, pred_id_coeff, pred_seg_coeff,
                    None, tgt_verts[None], tgt_faces, tgt_operators,
                )
                decode_out = self.model.decode(inputs, tgt_mesh=tgt_m, batch_process=True)
                pred_vertices = decode_out[0] if isinstance(decode_out, tuple) else decode_out

                # zero-expression from src → GT is tgt's own neutral face
                gt_vertices = tgt_verts[None]

                if self.opts.use_t_mask:
                    inner_mask = plateau_hat_points(gt_vertices, r0=1.0, r1=2.25)
                    outter_mask = 1 - inner_mask

                    MSE_in = F.mse_loss(
                        gt_vertices * inner_mask, pred_vertices * inner_mask
                    ).item()
                    losses_val['MSE-in'] += MSE_in

                    MSE_out = F.mse_loss(
                        gt_vertices * outter_mask, pred_vertices * outter_mask
                    ).item()
                    losses_val['MSE-out'] += MSE_out

                MSE = F.mse_loss(gt_vertices, pred_vertices).item()
                losses_val['MSE'] += MSE

                pbar_txt = f'MSE: {MSE:.5e}'
                if self.opts.use_t_mask:
                    pbar_txt = f'MSE-in: {MSE_in:.5e}'
                pbar.set_description(pbar_txt)

            if index % interv_val == 0:
                v_list = [tgt_verts.cpu(), pred_vertices[0].detach().cpu()]
                f_list = [tgt_faces.cpu()] * len(v_list)
                plot_image_array(
                    v_list, f_list,
                    rot_list=[[0, 0, 0]] * len(v_list),
                    size=1, bg_black=False, mode='shade',
                    logdir=f"{self.opts.log_dir}/img",
                    name=f"{index:04d}", save=True,
                )

        # ── write log ─────────────────────────────────────────────────
        log_text = "[LP Cross Eval] "
        for key, value in losses_val.items():
            value = value * denom
            txt = f"{key}: {value:.6e} "
            print(txt)
            log_text += txt
        self.logger.write(log_text + "\n")
        print('done!')

    def _evaluate_lp_cross_ngbc(self, src_name, tgt_name):
        """
        Cross-identity Linear-Preservation evaluation for NGBC (version=5).

        Uses NeuralGeneralizedBarycentricCoordinate.retarget(), which takes
        source-neutral/source-deformed/target-neutral vertex+normal triples
        directly — no img features or dfn_info/operators precompute needed
        (geometry-only cage model). Since the "expression" is neutral
        (src_def_vert == src_neu_vert), GT = the target's own neutral face.
        """
        device = self.device

        src_dataset = NeutralEvalDataset(data_name=src_name, toggle=False)
        tgt_dataset = NeutralEvalDataset(data_name=tgt_name, toggle=False)

        # ── logging ────────────────────────────────────────────────────
        os.makedirs(self.opts.log_dir, exist_ok=True)
        ckpt_path = self.opts.ckpt.split('/')[-1]
        self.opts.log_dir = os.path.join(
            self.opts.log_dir, ckpt_path + '-lp-cross', f'{src_name}-to-{tgt_name}'
        )
        if self.opts.use_t_mask:
            self.opts.log_dir = self.opts.log_dir + '-masked'
        os.makedirs(self.opts.log_dir, exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)

        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), self.opts)

        self.logger = open(os.path.join(self.opts.log_dir, "log.txt"), 'w')
        print(f'Saving log at: {self.opts.log_dir}')

        print(src_dataset.get_data_config())
        print(tgt_dataset.get_data_config())
        self.logger.write(src_dataset.get_data_config())
        self.logger.write(tgt_dataset.get_data_config())

        # ── eval loop ──────────────────────────────────────────────────
        self.model.eval()

        pairs = [(s, t) for s in src_dataset.items for t in tgt_dataset.items]
        len_data = len(pairs)
        denom = 1 / len_data
        interv_val = max(1, round(len_data / 10))

        losses_val = {"MSE": 0.0}
        if self.opts.use_t_mask:
            losses_val["MSE-in"] = 0.0
            losses_val["MSE-out"] = 0.0

        pbar = tqdm(enumerate(pairs), total=len_data, ncols=100)
        for index, ((src_v_np, src_f_np, src_id), (tgt_v_np, tgt_f_np, tgt_id)) in pbar:
            with torch.no_grad():
                src_n_np = igl.per_vertex_normals(src_v_np, src_f_np)
                tgt_n_np = igl.per_vertex_normals(tgt_v_np, tgt_f_np)

                src_verts = torch.tensor(src_v_np).float().to(device)[None]
                src_norms = torch.tensor(src_n_np).float().to(device)[None]
                tgt_verts = torch.tensor(tgt_v_np).float().to(device)[None]
                tgt_norms = torch.tensor(tgt_n_np).float().to(device)[None]
                tgt_faces = torch.tensor(tgt_f_np).long().to(device)

                pred_vertices, _ = self.model.retarget(
                    src_verts, src_norms, src_verts, src_norms,
                    tgt_verts, tgt_norms,
                )

                # zero-expression from src → GT is tgt's own neutral face
                gt_vertices = tgt_verts

                if self.opts.use_t_mask:
                    inner_mask = plateau_hat_points(gt_vertices, r0=1.0, r1=2.25)
                    outter_mask = 1 - inner_mask

                    MSE_in = F.mse_loss(
                        gt_vertices * inner_mask, pred_vertices * inner_mask
                    ).item()
                    losses_val['MSE-in'] += MSE_in

                    MSE_out = F.mse_loss(
                        gt_vertices * outter_mask, pred_vertices * outter_mask
                    ).item()
                    losses_val['MSE-out'] += MSE_out

                MSE = F.mse_loss(gt_vertices, pred_vertices).item()
                losses_val['MSE'] += MSE

                pbar_txt = f'MSE: {MSE:.5e}'
                if self.opts.use_t_mask:
                    pbar_txt = f'MSE-in: {MSE_in:.5e}'
                pbar.set_description(pbar_txt)

            if index % interv_val == 0:
                v_list = [tgt_verts[0].cpu(), pred_vertices[0].detach().cpu()]
                f_list = [tgt_faces.cpu()] * len(v_list)
                plot_image_array(
                    v_list, f_list,
                    rot_list=[[0, 0, 0]] * len(v_list),
                    size=1, bg_black=False, mode='shade',
                    logdir=f"{self.opts.log_dir}/img",
                    name=f"{index:04d}", save=True,
                )

        # ── write log ─────────────────────────────────────────────────
        log_text = "[LP Cross Eval] "
        for key, value in losses_val.items():
            value = value * denom
            txt = f"{key}: {value:.6e} "
            print(txt)
            log_text += txt
        self.logger.write(log_text + "\n")
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
    Linear-Preservation evaluation: both source and target are neutral faces.

    Examples:
        python eval_CBD_lp.py --version 21 --ckpt ./ckpts_CBD3/2025-11-26-14-46-32-NGBCv1 --data_selection 0 --batch_size 1
        python eval_CBD_lp.py --version 21 --ckpt ./ckpts_CBD3/2025-11-26-14-46-32-NGBCv1 --data_selection 2 --batch_size 1 --use_t_mask
        python eval_CBD_lp.py --version 22 --ckpt ./ckpts_CBD3/<ckpt>   --data_selection 0 --batch_size 1
        python eval_CBD_lp.py --version 5  --ckpt ./ckpts_CBD/<ckpt>    --data_selection 0 --last_activation relu
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
    if opts.cross:
        trainer.evaluate_lp_cross(opts.src_name, opts.tgt_name)
    else:
        trainer.evaluate_lp()
