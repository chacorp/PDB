"""
eval_comp.py — Evaluate NFS / NFR baselines with the same interface as eval_hlbs.py.

Self-retargeting : reconstruction metrics on EvalDataset (real test frames).
Cross-retargeting: transfer source expression to a different target identity.

Usage (NFS self-retarget):
    python eval_comp.py --model nfs \
        --ckpt ckpts_comparison/NFS-best \
        --data_selection mf_ROM --data_toggle

Usage (NFS cross-retarget):
    python eval_comp.py --model nfs \
        --ckpt ckpts_comparison/NFS-best \
        --data_selection mf_ROM --data_toggle \
        --cross_retarget --tgt_dataset biwi --tgt_identity 3

Usage (NFR cross-retarget):
    python eval_comp.py --model nfr \
        --data_selection mf_ROM --data_toggle \
        --cross_retarget --tgt_dataset ict --tgt_identity 0
"""
import os
import sys
import json
import argparse
import glob
import random
import pickle
import numpy as np
import yaml
import cv2
import trimesh

import torch
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
import igl

sys.path.insert(0, os.path.dirname(__file__))
from utils.matplotlib_rnd import plot_image_array
from dataloader_CBD import EvalDataset, CBD_collate_wrapper_eval
from utils.exp_utils import plateau_hat_points
from utils.mesh_utils import calc_norm_torch
import utils.nfr_utils as nfr_utils
from utils.mesh_utils import get_mesh_operators


# ── Precompute paths ────────────────────────────────────────────────────────

PRECOMPUTE_PATHS = {
    'mf':   '/data/sihun/multiface_align/precomputes',
    'voca': '/data/sihun/VOCA-COMA/precomputes',
    'coma': '/data/sihun/VOCA-COMA/precomputes',
    'biwi': '/data/sihun/BIWI_align_deci/precomputes',
    'ict':  '/data/sihun/ICT-audio2face/ICT/precompute-synth-fullhead',
}


def load_precompute(dataset_name, id_name, device='cuda:0'):
    """Load precomputed dfn_info, img, operators for a given identity.
    Returns: (dfn_info, img_tensor, operators)
    """
    base = PRECOMPUTE_PATHS.get(dataset_name)
    if base is None:
        raise ValueError(f"No precompute path for dataset '{dataset_name}'")

    # ICT uses numeric index naming
    prefix = os.path.join(base, id_name)

    dfn_path = f"{prefix}_dfn_info.pkl"
    img_path = f"{prefix}_img.npy"
    ops_path = f"{prefix}_operators.pkl"

    if not os.path.exists(dfn_path):
        raise FileNotFoundError(f"Precompute not found: {dfn_path}")

    with open(dfn_path, 'rb') as f:
        dfn_info = pickle.load(f)
    dfn_info = [x.to(device).float() if isinstance(x, torch.Tensor) else x for x in dfn_info]

    img = torch.tensor(np.load(img_path)).float().to(device)

    operators = None
    if os.path.exists(ops_path):
        with open(ops_path, 'rb') as f:
            operators = pickle.load(f)

    return dfn_info, img, operators


# ── CLI ─────────────────────────────────────────────────────────────────────

def Options():
    parser = argparse.ArgumentParser(description='Evaluate NFS / NFR baselines')

    # model
    parser.add_argument("--model", type=str, required=True,
                        choices=['nfs', 'nfr'],
                        help='Model to evaluate')
    parser.add_argument("--ckpt", type=str, default=None,
                        help='Checkpoint dir for NFS (ignored for NFR)')

    # NFS config
    parser.add_argument("--config", type=str, default='config/train.yml')
    parser.add_argument("--design", type=str, default='new2')
    parser.add_argument("--dec_type", type=str, default='disp')

    # data
    parser.add_argument("--data_selection", type=str, default='mf_ROM',
                        choices=['voca', 'biwi', 'mf_SEN', 'coma', 'mf_ROM', 'ict', 'ict-cap'],
                        help='Source dataset')
    parser.add_argument("--src_identity", type=int, default=-1,
                        help='Source identity index (-1 = all identities)')
    parser.add_argument("--data_toggle", dest='data_toggle', action='store_true')
    parser.set_defaults(data_toggle=False)
    parser.add_argument("--data_basedir", type=str, default='/data/sihun')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # mask
    parser.add_argument("--no_t_mask", dest='no_t_mask', action='store_true')
    parser.set_defaults(no_t_mask=False)
    parser.add_argument("--use_t_mask", dest='no_t_mask', action='store_false')

    # cross-retargeting
    parser.add_argument("--cross_retarget", dest='cross_retarget', action='store_true')
    parser.set_defaults(cross_retarget=False)
    parser.add_argument("--tgt_dataset", type=str, default=None,
                        choices=['mf', 'voca', 'biwi', 'coma', 'ict'],
                        help='Target dataset for cross-retarget')
    parser.add_argument("--tgt_identity", type=int, default=0,
                        help='Target identity index within the dataset')
    parser.add_argument("--tgt_obj_path", type=str, default=None,
                        help='Custom target mesh .obj (overrides --tgt_dataset)')

    # output
    parser.add_argument("--log_dir", type=str, default="eval_comp")
    parser.add_argument("--save_vert", dest='save_vert', action='store_true')
    parser.set_defaults(save_vert=False)
    parser.add_argument("--save_gt", dest='save_gt', action='store_true')
    parser.set_defaults(save_gt=False)
    parser.add_argument("--save_obj", dest='save_obj', action='store_true')
    parser.set_defaults(save_obj=False)
    parser.add_argument("--make_video", dest='make_video', action='store_true')
    parser.add_argument("--no_video", dest='make_video', action='store_false')
    parser.set_defaults(make_video=True)
    parser.add_argument("--no_vis", dest='no_vis', action='store_true')
    parser.set_defaults(no_vis=False)

    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


# ── Utilities ───────────────────────────────────────────────────────────────

def images_to_video(img_dir, out_path, fps=30):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if len(imgs) == 0:
        print(f"[WARN] No images in {img_dir}, skipping video.")
        return
    first = cv2.imread(imgs[0])
    h, w, _ = first.shape
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in imgs:
        writer.write(cv2.imread(p))
    writer.release()
    print(f"Video saved: {out_path}")


def _write_obj(path, verts, faces):
    with open(path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# ── Metric helpers (same as eval_hlbs.py) ──────────────────────────────────

def _build_cot_laplacian(verts_np, faces_np):
    return igl.cotmatrix(verts_np.astype(np.float64), faces_np.astype(np.int32))


def _laplacian_error(L_sp, pred, gt):
    B = pred.shape[0]
    total = 0.0
    for b in range(B):
        Lp = L_sp @ pred[b].numpy().astype(np.float64)
        Lg = L_sp @ gt[b].numpy().astype(np.float64)
        total += float(np.mean(np.sum((Lp - Lg) ** 2, axis=-1)))
    return total / B


def _normal_consistency(pred, gt, faces):
    pred_n = calc_norm_torch(pred, faces)
    gt_n   = calc_norm_torch(gt, faces)
    cos_sim = F.cosine_similarity(pred_n, gt_n, dim=-1)
    return (1.0 - cos_sim).mean().item()


def _edge_length_distortion(pred, gt, edges):
    e0, e1 = edges[:, 0], edges[:, 1]
    len_pred = torch.sqrt(((pred[:, e0] - pred[:, e1]) ** 2).sum(dim=-1))
    len_gt   = torch.sqrt(((gt[:, e0]   - gt[:, e1])   ** 2).sum(dim=-1))
    return ((len_pred - len_gt).abs() / (len_gt + 1e-8)).mean().item()


def _build_edges(faces_np):
    e = np.concatenate([faces_np[:, [0,1]], faces_np[:, [1,2]], faces_np[:, [0,2]]], axis=0)
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    return torch.tensor(e, dtype=torch.long)


# ── Target loading (same as eval_hlbs.py) ──────────────────────────────────

def _load_tgt_from_dataset(dataset_name, identity_idx, data_basedir='/data/sihun'):
    local_pkl_map = {
        'mf':   'utils/templates/mf_templates.pkl',
        'voca': 'utils/templates/voca_templates.pkl',
        'biwi': 'utils/templates/biwi_templates.pkl',
        'coma': 'utils/templates/voca_templates.pkl',
    }
    remote_pkl_map = {
        'mf':   f'{data_basedir}/multiface_align/mf_templates.pkl',
        'voca': f'{data_basedir}/VOCA-COMA/voca_templates.pkl',
        'biwi': f'{data_basedir}/BIWI_align_deci/templates_align_deci.pkl',
        'coma': f'{data_basedir}/VOCA-COMA/voca_templates.pkl',
    }

    if dataset_name == 'ict':
        mesh = trimesh.load('utils/ict/ict_aligned_mean.obj', process=False)
        verts = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        return verts, faces, 'ict_mean'

    pkl_path = remote_pkl_map[dataset_name]
    if not os.path.exists(pkl_path):
        pkl_path = pkl_path.replace(data_basedir, f'{data_basedir}/pca')
    if not os.path.exists(pkl_path):
        pkl_path = local_pkl_map[dataset_name]

    with open(pkl_path, 'rb') as f:
        templates = pickle.load(f)

    faces = np.array(templates['face'], dtype=np.int32)
    id_names = [k for k in templates if k != 'face']

    if identity_idx >= len(id_names):
        raise ValueError(
            f"tgt_identity {identity_idx} out of range for {dataset_name} "
            f"(max: {len(id_names)-1}). Available: {id_names}")

    id_name = id_names[identity_idx]
    verts = np.array(templates[id_name], dtype=np.float32)
    return verts, faces, id_name


# ── Model loading ──────────────────────────────────────────────────────────

def load_nfs_model(opts):
    """Load NFS model using the Trainer class from evaluation.py."""
    config = opts.config
    opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)
    nfs_opts = argparse.Namespace(**opts_yaml)

    nfs_opts.NFR = False
    nfs_opts.dec_type = opts.dec_type
    nfs_opts.design = opts.design
    nfs_opts.ckpt = opts.ckpt
    nfs_opts.device = opts.device
    nfs_opts.data_rand_trans = False
    nfs_opts.data_rand_scale = False
    nfs_opts.learn_rig_emb = False
    nfs_opts.use_decimate = False
    nfs_opts.stage1 = True
    nfs_opts.scale_exp = 1.0
    nfs_opts.ict_face_only = False

    from evaluation import Trainer
    trainer = Trainer(nfs_opts)
    trainer.model.eval()
    return trainer.model


def load_nfr_model(opts):
    """Load NFR model using the NFR_helper from evaluation.py."""
    config = opts.config
    opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)
    nfr_opts = argparse.Namespace(**opts_yaml)

    nfr_opts.NFR = True
    nfr_opts.dec_type = 'jacob'
    nfr_opts.design = 'nfr'
    nfr_opts.ckpt = ''  # dummy, NFR_helper loads its own
    nfr_opts.device = opts.device
    nfr_opts.data_rand_trans = False
    nfr_opts.data_rand_scale = False
    nfr_opts.learn_rig_emb = False
    nfr_opts.use_decimate = False
    nfr_opts.stage1 = True
    nfr_opts.scale_exp = 1.0
    nfr_opts.ict_face_only = False

    from evaluation import NFR_helper
    model = NFR_helper(nfr_opts, opts.device)
    return model


# ── Evaluator ──────────────────────────────────────────────────────────────

class CompEvaluator:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device(opts.device)

        torch.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)

        # Load model
        print(f"Loading {opts.model.upper()} model...")
        if opts.model == 'nfs':
            self.model = load_nfs_model(opts)
        else:
            self.model = load_nfr_model(opts)

        # ── Output dir ──────────────────────────────────────────────
        if opts.model == 'nfs':
            ckpt_basename = os.path.basename(os.path.normpath(opts.ckpt))
            model_tag = f"nfs-{ckpt_basename}"
        else:
            model_tag = "nfr-pretrained"

        src_tag = f"{opts.data_selection}"
        if opts.src_identity >= 0:
            src_tag += f"_id{opts.src_identity}"

        if opts.cross_retarget:
            tgt_tag = f"{opts.tgt_dataset}_id{opts.tgt_identity}" if opts.tgt_dataset else "custom"
            mode_tag = f"cross-{src_tag}_to_{tgt_tag}"
        else:
            mode_tag = f"self-{src_tag}"

        self.out_dir = os.path.join(opts.log_dir, model_tag, mode_tag)
        os.makedirs(self.out_dir, exist_ok=True)

        self.img_dir = os.path.join(self.out_dir, "img")
        os.makedirs(self.img_dir, exist_ok=True)

        if opts.save_vert:
            self.vert_dir = os.path.join(self.out_dir, "pred_vert")
            os.makedirs(self.vert_dir, exist_ok=True)
        if opts.save_gt:
            self.gt_dir = os.path.join(self.out_dir, "gt_vert")
            os.makedirs(self.gt_dir, exist_ok=True)

        # ── Load target for cross-retarget ──────────────────────────
        self.tgt_verts = None
        self.tgt_faces = None
        self.tgt_mesh = None
        if opts.cross_retarget:
            if opts.tgt_obj_path:
                mesh = trimesh.load(opts.tgt_obj_path, process=False)
                self.tgt_verts = np.array(mesh.vertices, dtype=np.float32)
                self.tgt_faces = np.array(mesh.faces, dtype=np.int32)
                self.tgt_mesh = mesh
                print(f"Target: custom mesh {opts.tgt_obj_path} ({self.tgt_verts.shape[0]} verts)")
            elif opts.tgt_dataset:
                tgt_verts, tgt_faces, tgt_id_name = _load_tgt_from_dataset(
                    opts.tgt_dataset, opts.tgt_identity, opts.data_basedir)
                self.tgt_verts = tgt_verts
                self.tgt_faces = tgt_faces
                self.tgt_mesh = trimesh.Trimesh(vertices=tgt_verts, faces=tgt_faces, process=False)
                print(f"Target: {opts.tgt_dataset}[{opts.tgt_identity}] ({tgt_id_name}) "
                      f"({tgt_verts.shape[0]} verts)")
            else:
                raise ValueError("Cross-retarget requires --tgt_dataset or --tgt_obj_path")

        # ── Load target precomputes ─────────────────────────────────
        self._precompute_cache = {}
        self.tgt_dfn_info = None
        self.tgt_img = None
        self.tgt_operators = None
        self.tgt_id_name = None

        if opts.cross_retarget and opts.tgt_dataset:
            tgt_id_name = self._get_tgt_id_name()
            self.tgt_id_name = tgt_id_name
            self.tgt_dfn_info, self.tgt_img, self.tgt_operators = load_precompute(
                opts.tgt_dataset, tgt_id_name, device=str(self.device))
            print(f"Loaded target precomputes: {opts.tgt_dataset}/{tgt_id_name}")

        print(f"Output: {self.out_dir}")

    def _get_tgt_id_name(self):
        """Resolve target identity name for precompute lookup."""
        opts = self.opts
        if opts.tgt_dataset == 'ict':
            return f"{opts.tgt_identity:03d}"
        # Load template pkl to get id name list
        _, _, tgt_id_name = _load_tgt_from_dataset(
            opts.tgt_dataset, opts.tgt_identity, opts.data_basedir)
        return tgt_id_name

    def _get_src_id_name(self, dataset, batch_idx):
        """Extract source identity name from dataset file paths."""
        opts = self.opts
        ds = opts.data_selection

        if ds in ('mf_ROM', 'mf_SEN'):
            datalist = getattr(dataset, f'{ds}_datalist', None)
            if datalist and batch_idx < len(datalist):
                # e.g. /data/.../vertices_npy/m--2019...-GHS/sentence01/000001.npy
                parts = datalist[batch_idx].split('/')
                for p in parts:
                    if p.startswith('m--'):
                        return p
        elif ds == 'coma':
            datalist = getattr(dataset, 'coma_datalist', None)
            if datalist and batch_idx < len(datalist):
                parts = datalist[batch_idx].split('/')
                for p in parts:
                    if p.startswith('FaceTalk'):
                        return p
        elif ds == 'voca':
            datalist = getattr(dataset, 'voca_datalist', None)
            if datalist and batch_idx < len(datalist):
                parts = datalist[batch_idx].split('/')
                for p in parts:
                    if p.startswith('FaceTalk'):
                        return p
        elif ds == 'biwi':
            datalist = getattr(dataset, 'biwi_datalist', None)
            if datalist and batch_idx < len(datalist):
                parts = datalist[batch_idx].split('/')
                for p in parts:
                    if p in ['F1','F2','F3','F4','F5','F6','F7','F8','M1','M2','M3','M4','M5','M6']:
                        return p
        elif ds in ('ict',):
            return f"{batch_idx // getattr(dataset, 'ict_exp_len', 1):03d}"

        return None

    def _build_src_mesh(self, template_np, faces_np):
        """Build trimesh from template vertices and faces."""
        return trimesh.Trimesh(vertices=template_np, faces=faces_np, process=False)

    def _resolve_id_name(self, dataset_name, batch):
        """Extract identity name from batch for precompute lookup."""
        if hasattr(batch, 'id_name'):
            return batch.id_name
        # Fallback: use data_selection mapping
        return None

    def _load_precomputes_for_dataset(self, dataset_name, id_name):
        """Load precomputes, caching by (dataset, id_name)."""
        key = (dataset_name, id_name)
        if key not in self._precompute_cache:
            self._precompute_cache[key] = load_precompute(
                dataset_name, id_name, device=str(self.device))
        return self._precompute_cache[key]

    @torch.no_grad()
    def _forward_nfs_precompute(self, gt_vertices, src_template, src_faces,
                                 src_dfn_info, src_img, src_operators,
                                 tgt_template, tgt_faces,
                                 tgt_dfn_info, tgt_img, tgt_operators):
        """NFS forward using precomputed data.
        Args:
            gt_vertices: [B, V, 3] source animation vertices
            src_template: [V, 3] source neutral
            src_faces: [F, 3] tensor
            src_dfn_info, src_img, src_operators: source precomputes
            tgt_*: target precomputes (same as src for self-retarget)
        Returns:
            pred: [B, V_tgt, 3]
        """
        model = self.model
        device = self.device

        # Target: encode identity + segmentation
        tgt_img_feat = model.get_img_feat(tgt_img)  # [1, 1, 128]
        tgt_verts_t = torch.tensor(tgt_template).float().unsqueeze(0).to(device)
        tgt_faces_t = torch.tensor(tgt_faces).long().to(device)
        tgt_vert_feat = model.get_local_feature(tgt_verts_t, tgt_faces_t, tgt_img_feat).float()

        model.mesh_id_encoder.update_precomputes(tgt_dfn_info)
        pred_id_coeff = model.encode_id(tgt_vert_feat, tgt_dfn_info)
        pred_seg_coeff = model.encode_seg(tgt_vert_feat, tgt_dfn_info)

        # Source: encode expression per frame
        src_img_feat = model.get_img_feat(src_img)  # [1, 1, 128]
        src_faces_t = torch.tensor(src_faces).long().to(device)
        gt_v = gt_vertices.to(device).float()

        vert_feat_exp_list = []
        for t in range(gt_v.shape[0]):
            vf = model.get_local_feature(gt_v[t:t+1], src_faces_t, src_img_feat).float()
            vert_feat_exp_list.append(vf)
        vert_feat_exp = torch.cat(vert_feat_exp_list, dim=0)  # [T, V, 134]

        pred_exp_coeff = model.encode_exp(vert_feat_exp, src_dfn_info, batch_process=True)

        # Decode on target
        local_feat = tgt_vert_feat  # [1, V, 134]
        style_emb = None
        inputs = (local_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff,
                  style_emb, tgt_verts_t[0], tgt_faces_t, tgt_operators)

        pred_outputs, _ = model.decode(inputs, batch_process=True)
        return pred_outputs

    @torch.no_grad()
    def _forward_nfr_precompute(self, gt_vertices, src_template, src_faces,
                                 src_dfn_info, src_img,
                                 tgt_template, tgt_faces,
                                 tgt_dfn_info, tgt_img, tgt_operators):
        """NFR forward using precomputed data."""
        model = self.model
        device = self.device

        src_faces_t = torch.tensor(src_faces).long().to(device)
        tgt_verts_t = torch.tensor(tgt_template).float().to(device)
        tgt_faces_t = torch.tensor(tgt_faces).long().to(device)
        gt_v = gt_vertices.to(device).float()

        pred_outputs = []
        for src_v in gt_v:
            inputs_v = model.get_inputs(src_v[None], src_faces_t)
            model.model.update_precomputes(src_dfn_info)
            pred_exp = model.model.encode(inputs_v, src_img.to(device),
                                          N_F=src_faces.shape[0])
            tmp, _, _ = model.calc_new_mesh(
                tgt_verts_t, tgt_faces_t, pred_exp,
                tgt_operators, tgt_dfn_info, tgt_img)
            pred_outputs.append(tmp)
        return torch.cat(pred_outputs)

    def evaluate(self):
        opts = self.opts

        dataset = EvalDataset(
            data_name=opts.data_selection,
            toggle=opts.data_toggle,
            data_basedir=opts.data_basedir,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=opts.batch_size,
            collate_fn=partial(CBD_collate_wrapper_eval, device='cpu'),
            num_workers=0,
        )

        L_sp = None
        edges = None

        total = {
            "mse": 0.0, "mse_in": 0.0, "mse_out": 0.0,
            "l2": 0.0, "lap": 0.0, "norm_cos": 0.0, "edge_dist": 0.0,
            "l2_max_sum": 0.0,
        }
        all_pv = []
        n_batches = 0
        frame_idx = 0

        is_cross = opts.cross_retarget and self.tgt_mesh is not None

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), ncols=120,
                    desc=f"Eval {opts.model.upper()} {'cross' if is_cross else 'self'}")

        for idx, batch in pbar:
            batch = batch.to('cpu')
            gt_v = batch.vertices       # [B, V, 3]
            src_v = batch.template      # [B, V, 3]
            faces = batch.faces         # [F, 3] or [B, F, 3]

            B = gt_v.shape[0]
            faces_np = faces[0].numpy() if faces.dim() == 3 else faces.numpy()

            # Resolve source identity name from datalist
            src_id_name = self._get_src_id_name(dataset, idx)

            # Build source mesh
            src_mesh = self._build_src_mesh(src_v[0].numpy(), faces_np)

            # Determine source dataset key for precomputes
            src_ds_key = opts.data_selection.replace('_ROM', '').replace('_SEN', '')  # mf_ROM -> mf

            # Load source precomputes
            if src_id_name:
                src_dfn_info, src_img, src_operators = self._load_precomputes_for_dataset(
                    src_ds_key, src_id_name)
            else:
                # Fallback: compute on-the-fly
                src_dfn_info = nfr_utils.get_dfn_info(src_mesh, map_location=str(self.device))
                src_img = None
                src_operators = None

            if is_cross:
                tgt_faces_np = self.tgt_faces
                tgt_template = self.tgt_verts
                tgt_dfn_info = self.tgt_dfn_info
                tgt_img = self.tgt_img
                tgt_operators = self.tgt_operators
            else:
                tgt_faces_np = faces_np
                tgt_template = src_v[0].numpy()
                tgt_dfn_info = src_dfn_info
                tgt_img = src_img
                tgt_operators = src_operators

            # Forward
            if opts.model == 'nfs':
                pred = self._forward_nfs_precompute(
                    gt_v, src_v[0].numpy(), faces_np,
                    src_dfn_info, src_img, src_operators,
                    tgt_template, tgt_faces_np,
                    tgt_dfn_info, tgt_img, tgt_operators)
            else:
                pred = self._forward_nfr_precompute(
                    gt_v, src_v[0].numpy(), faces_np,
                    src_dfn_info, src_img,
                    tgt_template, tgt_faces_np,
                    tgt_dfn_info, tgt_img, tgt_operators)

            pred = pred.cpu()

            # For self-retarget, compute metrics against GT
            if not is_cross:
                # Build operators lazily
                if L_sp is None:
                    L_sp = _build_cot_laplacian(src_v[0].numpy(), faces_np)
                    edges = _build_edges(faces_np)

                faces_t = torch.tensor(faces_np, dtype=torch.long)

                mse_val = F.mse_loss(gt_v, pred).item()
                total["mse"] += mse_val

                if not opts.no_t_mask:
                    t_mask = plateau_hat_points(src_v)
                    inv_mask = 1.0 - t_mask
                    total["mse_in"]  += F.mse_loss(gt_v * t_mask, pred * t_mask).item()
                    total["mse_out"] += F.mse_loss(gt_v * inv_mask, pred * inv_mask).item()

                pv_l2 = torch.sqrt(((gt_v - pred) ** 2).sum(dim=-1))  # [B, V]
                total["l2"] += pv_l2.mean().item()
                total["l2_max_sum"] += pv_l2.max(dim=-1).values.mean().item()
                all_pv.append(pv_l2.numpy())

                total["lap"] += _laplacian_error(L_sp, pred, gt_v)
                total["norm_cos"] += _normal_consistency(pred, gt_v, faces_t)
                total["edge_dist"] += _edge_length_distortion(pred, gt_v, edges)

            n_batches += 1

            # Visualization
            if not opts.no_vis:
                for b in range(B):
                    _d = lambda t: t[b].cpu()
                    f_cpu = torch.tensor(tgt_faces_np, dtype=torch.long)
                    if is_cross:
                        v_list = [_d(gt_v), _d(pred)]
                        f_list = [torch.tensor(faces_np, dtype=torch.long), f_cpu]
                    else:
                        v_list = [_d(gt_v), _d(src_v), _d(pred)]
                        f_list = [torch.tensor(faces_np, dtype=torch.long)] * 3
                    plot_image_array(
                        v_list, f_list,
                        rot_list=[[0, 0, 0]] * len(v_list),
                        size=1, bg_black=False, mode='shade',
                        logdir=self.img_dir,
                        name=f"{frame_idx:06d}", save=True)

                    if opts.save_vert:
                        np.save(os.path.join(self.vert_dir, f"{frame_idx:06d}.npy"),
                                pred[b].numpy())
                    if opts.save_gt and not is_cross:
                        np.save(os.path.join(self.gt_dir, f"{frame_idx:06d}.npy"),
                                gt_v[b].numpy())
                    if opts.save_obj:
                        obj_dir = os.path.join(self.out_dir, "obj")
                        os.makedirs(obj_dir, exist_ok=True)
                        _write_obj(os.path.join(obj_dir, f"pred_{frame_idx:06d}.obj"),
                                   pred[b].numpy(), tgt_faces_np)

                    frame_idx += 1

            pbar.set_description(
                f"Eval {opts.model.upper()} | mse: {total['mse']/max(n_batches,1):.5e}")

        # ── Aggregate results ───────────────────────────────────────
        if not is_cross and n_batches > 0:
            inv = 1.0 / n_batches
            all_pv = np.concatenate(all_pv, axis=0).flatten()

            results = {
                "model": opts.model,
                "data_selection": opts.data_selection,
                "src_identity": opts.src_identity,
                "MSE": total["mse"] * inv,
                "MSE_inner": total["mse_in"] * inv,
                "MSE_outer": total["mse_out"] * inv,
                "L2_mean": total["l2"] * inv,
                "L2_max_mean": total["l2_max_sum"] * inv,
                "L2_median": float(np.median(all_pv)),
                "L2_p95": float(np.percentile(all_pv, 95)),
                "L2_p99": float(np.percentile(all_pv, 99)),
                "L2_max": float(np.max(all_pv)),
                "Laplacian_err": total["lap"] * inv,
                "Normal_cos_dist": total["norm_cos"] * inv,
                "Edge_len_distortion": total["edge_dist"] * inv,
                "num_frames": n_batches,
            }

            results_path = os.path.join(self.out_dir, "results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\nResults saved: {results_path}")
            for k, v in results.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.6e}")

            np.save(os.path.join(self.out_dir, "per_vertex_l2.npy"), all_pv)
        else:
            print(f"\nCross-retarget done ({frame_idx} frames). Outputs: {self.out_dir}")

        # Video
        if opts.make_video and not opts.no_vis:
            video_name = f"eval_{opts.model}_{'cross' if is_cross else 'self'}.mp4"
            images_to_video(self.img_dir, os.path.join(self.out_dir, video_name))


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    opts = Options()

    # Validate
    if opts.model == 'nfs' and opts.ckpt is None:
        raise ValueError("NFS requires --ckpt")

    evaluator = CompEvaluator(opts)
    evaluator.evaluate()
