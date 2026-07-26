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

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader_CBD import (
    EvalDataset,
    CBDDataBatch_eval,
    CBD_collate_wrapper_eval,
)

from utils.matplotlib_rnd import plot_image_array
from utils.ckpt_utils import *
from utils.remesh_utils import calc_norm_torch

from models.NGBC import NeuralGeneralizedBarycentricCoordinate

import torch.multiprocessing as mp


def Options():
    parser = argparse.ArgumentParser(
        description='control-point (cage) consistency evaluation for cross-retargeting (NGBC)'
    )
    parser.add_argument('-c', '--config', default='config/train_CBD.yml', help='config file path')
    parser.add_argument("--device",       type=str,   default="cuda:0")

    parser.add_argument("--log_dir",      type=str,   default="eval_CBD_cp")

    parser.add_argument("--version",      type=int,   default=5,
                        help='model version (only 5: NGBC is supported)')

    #### Choose a last layer activation for key_weight_model()
    parser.add_argument("--last_activation", default="relu", choices=["relu", "elu", "softmax", "softplus", "none", "sqrelu"],
        help="Choose a last layer activation for NGBC.key_weight_model()"
    )

    parser.add_argument("--no_pou",dest='no_pou', action='store_true')
    parser.set_defaults(no_pou=False)

    parser.add_argument("--start_epoch",  type=int,   default=0,      help='number of epochs')
    parser.add_argument("--batch_size",   type=int,   default=1,      help='batch size')

    parser.add_argument("--seed",         type=int,   default=42,     help='random seed')
    parser.add_argument("--ckpt",         type=str,   default=None)
    parser.add_argument("--continue_ckpt",dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)

    parser.add_argument("--tb",           action='store_true')
    parser.set_defaults(is_train=True)

    parser.add_argument("--align_latent",dest='align_latent', action='store_true')
    parser.set_defaults(align_latent=False)

    #### cross-retargeting identity selection
    parser.add_argument("--src_name",     type=str,   default=None,
        help='source dataset name providing the real expression sequence (voca, biwi, mf_SEN, coma, mf_ROM, ict)')
    parser.add_argument("--tgt_name",     type=str,   default=None,
        help='target dataset name providing a different identity to retarget onto')
    parser.add_argument("--tgt_id_idx",   type=int,   default=0,
        help='index within --tgt_name to pick as the target identity (its neutral is used as target template)')

    parser.add_argument("--save_vert",dest='save_vert', action='store_true')
    parser.set_defaults(save_vert=False)
    parser.add_argument("--save_cp",dest='save_cp', action='store_true')
    parser.set_defaults(save_cp=False)

    args = parser.parse_args()
    return args


class Trainer():
    def __init__(self, opts):
        self.opts = opts
        self.set_seed(self.opts)
        self.device = opts.device

        if opts.version != 5:
            raise NotImplementedError(
                'eval_CBD_cp.py only supports version=5 (NGBC): control-point consistency relies on '
                'NeuralGeneralizedBarycentricCoordinate.retarget_animation() / .predict_coordinate()'
            )

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
            use_pou=~self.opts.no_pou,
            device=self.device,
            hid_dim=128 if self.opts.align_latent else 256,
        )

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
            if isinstance(ckpt_dict, dict) and 'model' in ckpt_dict:
                ckpt_dict = ckpt_dict['model']
            self.model.load_state_dict(ckpt_dict, strict=False)
        else:
            print('no ckpt found, training from scratch!')

    def evaluate_cp(self):
        """
        Control-point (cage) consistency for cross-retargeting.

        For each frame of the source identity's real expression sequence, retargeted onto a
        different, fixed target identity:
          - key_d_src: control points encoded directly from the source's own
            (neutral, expression) pair -- this is what drives the retargeted output.
          - key_d_out: control points re-encoded from (target neutral, retargeted result),
            i.e. feeding the cross-retargeted mesh back through the same expression encoder,
            treating the target identity as if it were the source of that same expression.
            One-way only -- this is not a cycle back to the source identity.
        A control-point-consistent model should produce key_d_src == key_d_out, since the
        encoded expression shouldn't depend on which identity's geometry carried it.
        MSE is computed directly on the (M, 3) cage tensors -- no vertex mask is applicable
        since control points are abstract cage vertices, not mesh vertices.
        """
        if not self.opts.src_name or not self.opts.tgt_name:
            raise ValueError('--src_name and --tgt_name are required for cross-retargeting eval')

        device = self.device

        ##########################################################################################################
        # source: real per-frame expression sequence -------------------------------------------------------------
        src_dataset = EvalDataset(data_name=self.opts.src_name, toggle=False)
        src_dataloader = torch.utils.data.DataLoader(
            src_dataset,
            batch_size=self.opts.batch_size,
            collate_fn=partial(CBD_collate_wrapper_eval, device=self.device),
            num_workers=8,
        )

        # target: a single, fixed, different identity's neutral template ------------------------------------------
        tgt_dataset = EvalDataset(data_name=self.opts.tgt_name, toggle=False)
        tgt_vertices, tgt_template, tgt_vertices_normal, tgt_template_normal, tgt_faces, tgt_id_name, tgt_mesh_data = \
            tgt_dataset[self.opts.tgt_id_idx]

        tgt_neu_vert = tgt_template[None].to(device)
        tgt_neu_norm = tgt_template_normal[None].to(device)
        tgt_faces = tgt_faces.to(device)
        ##########################################################################################################

        ###### Logging ###########################################################################################
        os.makedirs(self.opts.log_dir, exist_ok=True)
        ckpt_path = self.opts.ckpt.split('/')[-1]
        self.opts.log_dir = os.path.join(
            self.opts.log_dir, ckpt_path + '-cp', f'{self.opts.src_name}-to-{self.opts.tgt_name}'
        )
        os.makedirs(self.opts.log_dir, exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)

        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), self.opts)

        self.logger = open(os.path.join(self.opts.log_dir, "log.txt"), 'w')
        print(f'Saving log at: {self.opts.log_dir}')

        print(src_dataset.get_data_config())
        self.logger.write(src_dataset.get_data_config())
        tgt_info = f'[Target identity]: {self.opts.tgt_name} (idx={self.opts.tgt_id_idx}, id_name={tgt_id_name})\n'
        print(tgt_info)
        self.logger.write(tgt_info)
        #---------------------------------------------------------------------------------------------------------
        ##########################################################################################################

        self.model.eval()

        with torch.no_grad():
            # target's own cage weights -- fixed for the whole run (target identity doesn't change)
            key_weight_tgt = self.model.predict_coordinate(tgt_neu_vert, tgt_neu_norm)  # (1, N_tgt, M)

        len_data = len(src_dataloader)
        denom = 1 / len_data
        interv_val = max(1, round(len_data / 10))

        losses_val = {"CP-MSE": 0.0}

        if self.opts.save_vert:
            save_vert_logdir = f"{self.opts.log_dir}/verts"
            os.makedirs(save_vert_logdir, exist_ok=True)
        if self.opts.save_cp:
            save_cp_logdir = f"{self.opts.log_dir}/cp"
            os.makedirs(save_cp_logdir, exist_ok=True)

        pbar = tqdm(enumerate(src_dataloader), total=len_data, ncols=100)
        for index, batch in pbar:
            with torch.no_grad():
                B = batch.template.shape[0]
                key_weight_tgt_b = key_weight_tgt.expand(B, -1, -1)
                tgt_neu_vert_b = tgt_neu_vert.expand(B, -1, -1)
                tgt_neu_norm_b = tgt_neu_norm.expand(B, -1, -1)

                # control points estimated from the source expression, and the resulting
                # cross-retargeted mesh on the target identity's topology
                pred_deformed, key_d_src = self.model.retarget_animation(
                    batch.template, batch.template_normal,
                    batch.vertices, batch.vertices_normal,
                    key_weight_tgt_b,
                    tgt_neu_vert_b,
                )

                # control points re-extracted from the retargeted result (one-way, not cyclic):
                # feed (target neutral, retargeted result) through the same expression encoder
                pred_deformed_normal = calc_norm_torch(pred_deformed, tgt_faces, at='vert')
                _, key_d_out = self.model.retarget_animation(
                    tgt_neu_vert_b, tgt_neu_norm_b,
                    pred_deformed, pred_deformed_normal,
                    key_weight_tgt_b,
                    tgt_neu_vert_b,
                )

                CP_MSE = F.mse_loss(key_d_src, key_d_out).item()
                losses_val['CP-MSE'] += CP_MSE * denom

                pbar.set_description(f'CP-MSE: {CP_MSE:.5e}')

            # ------------------------------------------------------------------------------------------------
            if self.opts.save_vert:
                curr_batch = pred_deformed.shape[0]
                for b_idx in range(curr_batch):
                    save_vert_name = f"{save_vert_logdir}/{index*curr_batch + b_idx:06d}.npy"
                    np.save(save_vert_name, pred_deformed[b_idx].detach().cpu().numpy())

            if self.opts.save_cp:
                curr_batch = key_d_src.shape[0]
                for b_idx in range(curr_batch):
                    save_cp_name = f"{save_cp_logdir}/{index*curr_batch + b_idx:06d}.npz"
                    np.savez(
                        save_cp_name,
                        key_d_src=key_d_src[b_idx].detach().cpu().numpy(),
                        key_d_out=key_d_out[b_idx].detach().cpu().numpy(),
                    )

            # ------------------------------------------------------------------------------------------------
            if index % interv_val == 0:
                v_list = [tgt_neu_vert[0].cpu(), pred_deformed[0].detach().cpu()]
                f_list = [tgt_faces.cpu()] * len(v_list)
                plot_image_array(
                    v_list, f_list,
                    rot_list=[[0, 0, 0]] * len(v_list),
                    size=1, bg_black=False, mode='shade',
                    logdir=f"{self.opts.log_dir}/img",
                    name=f"{index:04d}", save=True,
                )
        ##########################################################################################################

        log_text = "[CP Eval] "
        for key, value in losses_val.items():
            txt = f"{key}: {value:.6e} "
            print(txt)
            log_text += txt
        self.logger.write(log_text + "\n")
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
    Control-point (cage) consistency evaluation for cross-retargeting (NGBC / version=5 only).

    Compares the cage displacement (`key_d`, shape [M, 3]) encoded directly from a source
    identity's own real expression sequence against the cage displacement re-encoded from the
    cross-retargeted result on the target identity's topology. No vertex mask (--use_t_mask) is
    used -- control points are abstract cage vertices (M of them), not mesh vertices, so
    inner/outer masking doesn't apply the same way it does for vertex-space metrics.

    Examples:
        python eval_CBD_cp.py --version 5 --ckpt ./ckpts_CBD/2026-07-07-23-25-37-NGBCv5 \
            --align_latent --last_activation relu --src_name mf_SEN --tgt_name ict --batch_size 8

        python eval_CBD_cp.py --version 5 --ckpt ./ckpts_CBD/2026-04-12-04-07-19-NGBCv5-dist \
            --last_activation relu --src_name ict --tgt_name mf_SEN --tgt_id_idx 0 --batch_size 8
    """
    mp.set_start_method('spawn', force=True)

    # argparse configs
    opts = Options()

    # base configs (yaml) -- loaded from the checkpoint's own saved train_opts.yml
    config = f'{opts.ckpt}/train_opts.yml'
    opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)

    # update with argparse configs
    opts_ = vars(opts)
    opts_yaml.update(opts_)
    opts = argparse.Namespace(**opts_yaml)

    print('loaded version:', opts.version)

    ## load model
    trainer = Trainer(opts)
    trainer.evaluate_cp()
