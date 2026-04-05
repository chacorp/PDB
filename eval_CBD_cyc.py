import os
import glob
import json
import yaml
import random
import itertools

import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
import trimesh
import igl
import pickle

import torch
import torch.nn.functional as F

from dataloader_CBD import (
    EvalDataset,
    CBD_collate_wrapper_eval,
)

from utils.ckpt_utils import *
from models.baseline import CageNet
from models.NGBC import NeuralGeneralizedBarycentricCoordinate
from models.NGBCv2 import NeuralBarycentricCoordinatev2, NeuralBarycentricCoordinatev3

import torch.multiprocessing as mp


def Options():
    parser = argparse.ArgumentParser(description='cyclic consistency evaluation for cross-identity FA retargeting')
    parser.add_argument('-c', '--config', default='config/train_CBD.yml', help='config file path')
    parser.add_argument("--device",        type=str,   default="cuda:0")

    parser.add_argument("--log_dir",       type=str,   default="eval_CBD")

    parser.add_argument("--version",       type=int,   default=5,
                        help='model version (0: NFS/NFR, 1: CageNet, 5: NGBC, 21: NGBCv2, 22: NGBCv3)')

    parser.add_argument("--last_activation", default="relu",
                        choices=["relu", "elu", "softmax", "softplus", "none", "sqrelu"],
                        help="last layer activation for key_weight_model()")

    parser.add_argument("--no_pou",        dest='no_pou', action='store_true')
    parser.set_defaults(no_pou=False)

    parser.add_argument("--batch_size",    type=int,   default=1,      help='batch size')
    parser.add_argument("--seed",          type=int,   default=42,     help='random seed')
    parser.add_argument("--ckpt",          type=str,   default=None)

    parser.add_argument("--start_epoch",   type=int,   default=0)
    parser.add_argument("--continue_ckpt", dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)

    parser.add_argument("--laplacian",     dest='laplacian', action='store_true')
    parser.set_defaults(laplacian=False)

    parser.add_argument("--save_vert",     dest='save_vert', action='store_true')
    parser.set_defaults(save_vert=False)

    parser.add_argument("--save_gt",       dest='save_gt', action='store_true')
    parser.set_defaults(save_gt=False)

    parser.add_argument("--align_latent",  dest='align_latent', action='store_true')
    parser.set_defaults(align_latent=False)

    parser.add_argument("--NFR",           dest='NFR', action='store_true',
                        help='use public NFR checkpoint (only for version 0)')
    parser.set_defaults(NFR=False)

    # cyclic-eval specific flags
    parser.add_argument("--src_data",      type=str,   default="ict",
                        help='source dataset — provides expressions (e.g. ict, voca, biwi)')
    parser.add_argument("--tgt_data",      type=str,   default="mf_SEN",
                        help='target dataset — provides neutral meshes (e.g. mf_SEN, mf_ROM)')
    parser.add_argument("--n_iter",        type=int,   default=-1,
                        help='max iterations (-1 = min of both dataloader lengths)')
    parser.add_argument("--symmetric",     dest='symmetric', action='store_true',
                        help='also run tgt→src→tgt direction after the main loop')
    parser.set_defaults(symmetric=False)

    args = parser.parse_args()
    return args


def aggregate_results(results):
    keys = ["cycle_full", "cycle_lap"]
    out = {}
    for k in keys:
        vals = [r[k] for r in results if k in r]
        if vals:
            out[k + "_mean"] = float(np.mean(vals))
            out[k + "_std"]  = float(np.std(vals))
    return out


class Trainer():
    def __init__(self, opts):
        self.opts = opts
        self.set_seed(self.opts)
        self.device = opts.device

        last_act_list = ["relu", "elu", "softmax", "softplus", "none", "sqrelu"]
        last_act_list = [self.opts.last_activation == l_act for l_act in last_act_list]

        if opts.version == 0:
            if opts.NFR:
                from evaluation import Trainer as NFRTrainer
                nfr_trainer = NFRTrainer(opts)
                self.model = nfr_trainer.model
            else:
                from models.NFS import NFS
                self.model = NFS(self.opts, None, print_param=True).to(self.device)
                self.load_weight()
        elif opts.version == 1:
            self.model = CageNet(
                device=self.device,
                optim_cage=False,
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
            raise NotImplementedError(f'No matching model for version={opts.version}')

        self.load_weight()

    def load_weight(self):
        if self.opts.ckpt:
            print(self.opts.ckpt)
            if self.opts.continue_ckpt:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, f"*_{self.opts.start_epoch:03d}.pth"))[0]
            else:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
            print(f"Loading... {ckpt}")
            ckpt_dict = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt_dict, strict=False)
        else:
            print('no ckpt found, running with random weights!')

    def _run_cyclic_loop(self, src_data_name, tgt_data_name, log_dir):
        """
        Core cyclic retargeting loop.
        Pass 1: src → tgt  (A_n, A_e → B_n → pred_B)
        Pass 2: tgt → src  (B_n, pred_B → A_n → cycle_A)
        Measures MSE(cycle_A, A_e).
        """
        opts = self.opts
        device = self.device

        # --- Dataloaders ---
        src_dataset = EvalDataset(data_name=src_data_name, toggle=False)
        tgt_dataset = EvalDataset(data_name=tgt_data_name, toggle=False)

        src_dataloader = torch.utils.data.DataLoader(
            src_dataset, batch_size=1, shuffle=False,
            collate_fn=partial(CBD_collate_wrapper_eval, device=device),
        )
        tgt_dataloader = torch.utils.data.DataLoader(
            tgt_dataset, batch_size=1, shuffle=False,
            collate_fn=partial(CBD_collate_wrapper_eval, device=device),
        )

        if len(src_dataset) >= len(tgt_dataset):
            iter_tgt = itertools.cycle(tgt_dataloader)
            iter_src = iter(src_dataloader)
            total = len(src_dataloader)
        else:
            iter_src = itertools.cycle(src_dataloader)
            iter_tgt = iter(tgt_dataloader)
            total = len(tgt_dataloader)

        if opts.n_iter > 0:
            total = min(total, opts.n_iter)

        # --- Save dirs ---
        if opts.save_vert:
            save_dir_pred_B  = os.path.join(log_dir, "verts_pred_B")
            save_dir_cycle_A = os.path.join(log_dir, "verts_cycle_A")
            os.makedirs(save_dir_pred_B,  exist_ok=True)
            os.makedirs(save_dir_cycle_A, exist_ok=True)
        if opts.save_gt:
            save_dir_gt = os.path.join(log_dir, "verts_gt")
            os.makedirs(save_dir_gt, exist_ok=True)

        # --- Cache ---
        prev_src_template = None
        prev_tgt_template = None
        key_weight_src    = None   # barycentric coords for src topology (pass 2)
        key_weight_tgt    = None   # barycentric coords for tgt topology (pass 1)
        L_src             = None   # cotangent Laplacian for src mesh
        src_trimesh       = None   # trimesh for version 0 (NFS/NFR)
        tgt_trimesh       = None   # trimesh for version 0 (NFS/NFR)

        results    = []
        losses_cyc = {"cycle_full": 0.0}
        if opts.laplacian:
            losses_cyc["cycle_lap"] = 0.0
        denom = 1.0 / total

        if opts.version == 0 and opts.NFR:
            self.model.model.eval()
        else:
            self.model.eval()
        pbar = tqdm(range(total), ncols=100)

        for index in pbar:
            batch_src = next(iter_src)
            batch_tgt = next(iter_tgt)

            with torch.no_grad():

                # ---- Recompute key_weight_tgt when tgt neutral changes ----
                if (prev_tgt_template is None or
                        (batch_tgt.template[0] - prev_tgt_template[0]).mean() != 0):
                    if opts.version in [5, 8]:
                        key_weight_tgt = self.model.predict_coordinate(
                            batch_tgt.template, batch_tgt.template_normal
                        )
                    elif opts.version == 0:
                        tgt_trimesh = trimesh.Trimesh(
                            vertices=batch_tgt.template[0].cpu().numpy(),
                            faces=batch_tgt.faces[0].cpu().numpy()
                        )
                    prev_tgt_template = batch_tgt.template

                # ---- Recompute key_weight_src + Laplacian when src neutral changes ----
                if (prev_src_template is None or
                        (batch_src.template[0] - prev_src_template[0]).mean() != 0):
                    if opts.version in [5, 8]:
                        key_weight_src = self.model.predict_coordinate(
                            batch_src.template, batch_src.template_normal
                        )
                    elif opts.version == 0:
                        src_trimesh = trimesh.Trimesh(
                            vertices=batch_src.template[0].cpu().numpy(),
                            faces=batch_src.faces[0].cpu().numpy()
                        )
                    if opts.laplacian:
                        src_m = trimesh.Trimesh(
                            vertices=batch_src.template[0].cpu().numpy(),
                            faces=batch_src.faces[0].cpu().numpy()
                        )
                        tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                        L_src = torch.sparse_csc_tensor(
                            torch.LongTensor(tmp_L.indptr).to(device),
                            torch.LongTensor(tmp_L.indices).to(device),
                            torch.FloatTensor(tmp_L.data).to(device),
                            tmp_L.shape
                        )
                    prev_src_template = batch_src.template

                # ================================================================
                # Pass 1: src → tgt  (A_n, A_e → B_n → pred_B)
                # ================================================================
                if opts.version in [5, 8]:
                    pred_B, _ = self.model.retarget_animation(
                        batch_src.template,        batch_src.template_normal,
                        batch_src.vertices,        batch_src.vertices_normal,
                        key_weight_tgt,
                        batch_tgt.template,
                    )
                elif opts.version == 1:
                    pred_B, _ = self.model.retarget(
                        batch_src.template,
                        batch_src.vertices,
                        batch_tgt.template,
                    )
                elif opts.version in [21, 22]:
                    pred_B, pred_src, *_ = self.model.retarget(
                        batch_src.template,        batch_src.template_normal,
                        batch_src.vertices,        batch_src.vertices_normal,
                        batch_tgt.template,        batch_tgt.template_normal,
                    )
                    if opts.version == 21:
                        pred_B = pred_B - pred_src + batch_tgt.template
                elif opts.version == 0:
                    pred_B = self.model.inference(
                        batch_src.vertices, src_trimesh, tgt_trimesh
                    )  # [T, V_tgt, 3]
                    pred_B = pred_B[0:1]  # [1, V_tgt, 3]

                # Recompute per-vertex normals for pred_B (on tgt topology)
                pred_B_n_np = igl.per_vertex_normals(
                    pred_B[0].detach().cpu().numpy(),
                    batch_tgt.faces[0].cpu().numpy()
                )
                pred_B_n = torch.tensor(pred_B_n_np, dtype=torch.float32).unsqueeze(0).to(device)

                # ================================================================
                # Pass 2: tgt → src  (B_n, pred_B → A_n → cycle_A)
                # ================================================================
                if opts.version in [5, 8]:
                    cycle_A, _ = self.model.retarget_animation(
                        batch_tgt.template,   batch_tgt.template_normal,
                        pred_B,               pred_B_n,
                        key_weight_src,
                        batch_src.template,
                    )
                elif opts.version == 1:
                    cycle_A, _ = self.model.retarget(
                        batch_tgt.template,
                        pred_B,
                        batch_src.template,
                    )
                elif opts.version in [21, 22]:
                    cycle_A, pred_src2, *_ = self.model.retarget(
                        batch_tgt.template,   batch_tgt.template_normal,
                        pred_B,               pred_B_n,
                        batch_src.template,   batch_src.template_normal,
                    )
                    if opts.version == 21:
                        cycle_A = cycle_A - pred_src2 + batch_src.template
                elif opts.version == 0:
                    cycle_A = self.model.inference(
                        pred_B, tgt_trimesh, src_trimesh
                    )  # [T, V_src, 3]
                    cycle_A = cycle_A[0:1]  # [1, V_src, 3]

                # ================================================================
                # Metrics
                # ================================================================
                cyc = cycle_A.squeeze(0)
                gt  = batch_src.vertices.squeeze(0)

                loss_full = F.mse_loss(cyc, gt).item()
                losses_cyc["cycle_full"] += loss_full * denom
                m = {"cycle_full": loss_full}

                if opts.laplacian:
                    lap_loss = F.mse_loss(L_src @ cyc, L_src @ gt).item()
                    losses_cyc["cycle_lap"] += lap_loss * denom
                    m["cycle_lap"] = lap_loss

                results.append(m)
                pbar.set_description(f'cyc_full: {loss_full:.4e}')

                # ================================================================
                # Save vertices
                # ================================================================
                if opts.save_vert:
                    np.save(
                        os.path.join(save_dir_pred_B,  f"{index:06d}.npy"),
                        pred_B[0].detach().cpu().numpy()
                    )
                    np.save(
                        os.path.join(save_dir_cycle_A, f"{index:06d}.npy"),
                        cyc.detach().cpu().numpy()
                    )
                if opts.save_gt:
                    np.save(
                        os.path.join(save_dir_gt, f"{index:06d}.npy"),
                        gt.cpu().numpy()
                    )

        # --- Aggregate ---
        agg = aggregate_results(results)

        return results, losses_cyc, agg

    def evaluate_cyclic(self):
        opts = self.opts
        device = self.device

        # --- Logging setup ---
        ckpt_path = opts.ckpt.split('/')[-1]
        opts.log_dir = os.path.join(
            opts.log_dir,
            f"{ckpt_path}-cyclic-eval",
            f"{opts.src_data}-to-{opts.tgt_data}" + ("-laplacian" if opts.laplacian else "")
        )
        os.makedirs(opts.log_dir, exist_ok=True)

        with open(os.path.join(opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(opts), f, indent=4)
        self.dump_yaml(os.path.join(opts.log_dir, "train_opts.yml"), opts)
        self.logger = open(os.path.join(opts.log_dir, "log.txt"), 'w')
        print(f'Saving log at: {opts.log_dir}')

        # --- Main direction: src → tgt → src ---
        print(f"\n[Cyclic Eval] {opts.src_data} → {opts.tgt_data} → {opts.src_data}")
        results, losses_cyc, agg = self._run_cyclic_loop(
            src_data_name=opts.src_data,
            tgt_data_name=opts.tgt_data,
            log_dir=opts.log_dir,
        )

        log_text = f"[Cyclic Eval] {opts.src_data}→{opts.tgt_data}→{opts.src_data}  "
        for key, value in {**losses_cyc, **agg}.items():
            txt = f"{key}: {value:.6e} "
            print(txt)
            log_text += txt
        self.logger.write(log_text + "\n")

        output = {"direction": f"{opts.src_data}->{opts.tgt_data}->{opts.src_data}",
                  "results": results, "aggregate": agg}

        # --- Optional symmetric direction: tgt → src → tgt ---
        if opts.symmetric:
            sym_log_dir = os.path.join(
                os.path.dirname(opts.log_dir),
                f"{opts.tgt_data}-to-{opts.src_data}" + ("-laplacian" if opts.laplacian else "")
            )
            os.makedirs(sym_log_dir, exist_ok=True)

            print(f"\n[Cyclic Eval] {opts.tgt_data} → {opts.src_data} → {opts.tgt_data}")
            results_sym, losses_cyc_sym, agg_sym = self._run_cyclic_loop(
                src_data_name=opts.tgt_data,
                tgt_data_name=opts.src_data,
                log_dir=sym_log_dir,
            )

            log_text_sym = f"[Cyclic Eval] {opts.tgt_data}→{opts.src_data}→{opts.tgt_data}  "
            for key, value in {**losses_cyc_sym, **agg_sym}.items():
                txt = f"{key}: {value:.6e} "
                print(txt)
                log_text_sym += txt
            self.logger.write(log_text_sym + "\n")

            output["symmetric"] = {
                "direction": f"{opts.tgt_data}->{opts.src_data}->{opts.tgt_data}",
                "results": results_sym, "aggregate": agg_sym,
            }

        # --- Dump results ---
        json.dump(output, open(os.path.join(opts.log_dir, "results.json"), 'w'), indent=2)
        self.logger.close()
        print('done!')

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


if __name__ == "__main__":
    """
    ## Example usage:

    # v5  ict → mf_SEN cyclic
    python eval_CBD_cyc.py --version 5  --ckpt ./ckpts_CBD/...   --src_data ict --tgt_data mf_SEN

    # v21 ict → mf_SEN cyclic with laplacian + save verts
    python eval_CBD_cyc.py --version 21 --ckpt ./ckpts_CBD3/...  --src_data ict --tgt_data mf_SEN --laplacian --save_vert

    # v22 symmetric (ict ↔ mf_ROM both directions)
    python eval_CBD_cyc.py --version 22 --ckpt ./ckpts_CBD3/...  --src_data ict --tgt_data mf_ROM --symmetric

    # limit to 100 iterations
    python eval_CBD_cyc.py --version 5  --ckpt ./ckpts_CBD/...   --src_data ict --tgt_data mf_SEN --n_iter 100
    """
    mp.set_start_method('spawn', force=True)
    opts = Options()

    if opts.version == 0:
        opts.config = 'config/train.yml'
        opts_yaml = yaml.load(open(opts.config), Loader=yaml.FullLoader)
    else:
        config = f'{opts.ckpt}/train_opts.yml'
        opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)

    opts_ = vars(opts)
    opts_yaml.update(opts_)
    opts = argparse.Namespace(**opts_yaml)

    if opts.version == 0:
        if opts.NFR:
            opts.design = "nfr"
            opts.dec_type = "jacob"
        else:
            opts.img_feat_dim = 128
            opts.feature_type = "cents&norms"
            opts.stage1 = True
            opts.scale_exp = 1.0
            opts.ict_face_only = False
            opts.design = "new2"
            opts.dec_type = "disp"

    print('loaded version:', opts.version)

    trainer = Trainer(opts)
    trainer.evaluate_cyclic()
