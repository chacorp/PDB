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

import sys
from pathlib import Path
__abs_path__ = str(Path(__file__).parents[1].absolute())

if not __abs_path__ in sys.path:
    sys.path+=[__abs_path__]


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataloader_CBD import (
    CBDdataSampler,
    CBDDataset,
    CBD_collate_wrapper,
    # CBDDataset2,
    # CBD_collate_wrapper2,
)

# from utils.mesh_utils import Renderer #, calc_cent
from utils.matplotlib_rnd import plot_image_array, plot_image_array_seg, vis_rig
from utils.ckpt_utils import *
from utils.remesh_utils import build_padded_neighbors, pca_normal_axis_vectorized
# from utils.exp_utils import Model_mk1, Model_mk3_1
# from utils.remesh_utils import compute_MVC_vertexwise, apply_MVC_weights_batch, build_padded_neighbors, pca_normal_axis_vectorized

from models.baseline import CageNet
from models.NGBC import (
    NeuralGeneralizedBarycentricCoordinate, 
    NeuralGeneralizedBarycentricCoordinate5,
    NeuralGeneralizedBarycentricCoordinate8,
)

sys.path = list(set(sys.path))

def Options():
    parser = argparse.ArgumentParser(description='neural generalized barycentric coordinate for FA retargeting')
    parser.add_argument('-c', '--config', default='config/train_CBD.yml', help='config file path')
    parser.add_argument("--device",       type=str,   default="cuda:0")
    
    parser.add_argument("--log_dir",      type=str,   default="eval_CBD")

    parser.add_argument("--version",      type=int,   default=1,      help='train method (1: baseline, 2: ours)')
    parser.add_argument("--num_cage_v",   type=int,   default=1024,   help='number of cage vertices')

    parser.add_argument("--data_selection",      type=int,   default=-1,
                        help='select dataset (-1: all, 0: voca, 1:biwi, 2: mf_SEN, 3: coma, 4: mf_ROM, 5: mf all)')
    
    parser.add_argument("--in_type",      type=int,   default=0,      
                        help='input type (0: position, 1: position + normal')
    parser.add_argument("--out_type",      type=int,   default=1,      
                        help='output type (0: cage v, 1: cage delta_v, 2: cage delta_T mat, 3: vertex T mat')
    
    parser.add_argument("--start_epoch",  type=int,   default=0,      help='number of epochs')
    parser.add_argument("--lr",           type=float, default=0.0002, help='learning rate')
    
    parser.add_argument("--batch_size",   type=int,   default=8,      help='batch size')

    parser.add_argument("--seed",         type=int,   default=42,     help='random seed')
    parser.add_argument("--ckpt",         type=str,   default=None)    
    parser.add_argument("--continue_ckpt",dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)
    
    parser.add_argument("--n_sampling",   dest='n_sampling', action='store_true')
    parser.set_defaults(n_sampling=False)
    
    parser.add_argument("--use_decimate", dest='use_decimate', action='store_true')
    parser.set_defaults(use_decimate=False)
    
    parser.add_argument("--use_scheduler",dest='use_scheduler', action='store_true')
    parser.set_defaults(use_scheduler=False)

    parser.add_argument("--tb",           action='store_true')
    parser.set_defaults(is_train=True)
    
    
    parser.add_argument("--optim_cage",dest='optim_cage', action='store_true')
    parser.set_defaults(optim_cage=False)
    
    
    args = parser.parse_args()
    return args

# --- Loss Functions ---

def mvc_loss(mvc_weights):
    """ penalize MVC with negative values """
    neg_loss = torch.nn.functional.relu(-mvc_weights) ** 2
    return torch.mean(neg_loss)

def p2f_loss(before_v, after_v, normals_before, normals_after):
    """ Point-to-Surface Loss
    Args:
        before_v: (B, N, 3) vertices source mesh
        after_v: (B, N, 3) vertices deformed mesh
        normals_before: (B, N, 3) normals from pca plane in source mesh
        normals_after: (B, N, 3) normals from pca plane in deformed mesh
    Returns
        loss (float)
    """
    
    def distance(verts, norms):
        dists = torch.abs(torch.sum(verts * norms, dim=-1))
        return dists

    before_dist = distance(before_v, normals_before)
    after_dist = distance(after_v, normals_after)
    return F.mse_loss(before_dist, after_dist)

def pca_normal_axis(verts, neighbors_map):
    """
    Args:
        verts (torch.tensor): (B, N, 3) vertices
        neighbors_map (list(int):
    Returns:
        normal_axis: (B, N, 3)
    """
    B, V, _ = verts.shape
    
    normal_axis = torch.zeros_like(verts)
    for i, neighbors_idx in enumerate(neighbors_map):
        if len(neighbors_idx) < 2: continue

        neighborhood = verts[:, neighbors_idx, :]
        centroid = torch.mean(neighborhood, dim=1)
        _, _, V_svd = torch.linalg.svd(neighborhood - centroid.unsqueeze(1))
        normal_axis[:, i, :] = V_svd[:, -1, :]
    return normal_axis

def norm_loss(normals_before, normals_after):
    """ PCA Normal Loss 
    Args:
        before_v: (B, N, 3) vertices source mesh
        after_v: (B, N, 3) vertices deformed mesh
        normals_before: (B, N, 3) normals from pca plane in source mesh
        normals_after: (B, N, 3) normals from pca plane in deformed mesh
    Returns
        loss (float)
    """
    return torch.mean(1.0 - F.cosine_similarity(normals_before, normals_after, dim=-1))

class Trainer():
    def __init__(self, opts):
        # set opts
        self.opts = opts
        self.set_seed(self.opts)
        self.device = opts.device

        if opts.version==1:
            self.model = CageNet(device=self.device, optim_cage=self.opts.optim_cage)
        elif opts.version==2:
            self.model = NeuralGeneralizedBarycentricCoordinate(
                opts, 
                hid_dim=256,
                num_cage_vertices=self.opts.num_cage_v,
                num_layers=4,
                use_relu=True,
                is_train=True, 
                device=self.device,
            )
        elif opts.version==5:
            self.model = NeuralGeneralizedBarycentricCoordinate5(
                opts, num_layers=4,
                use_exp_recon=False, # not used yet
                use_shp_recon=False, # not used yet
                use_shp=False,
                use_elu=False,
                use_relu=True,
                use_least_N_on_V=False,
                is_train=True,
                device=self.device,
            )
        elif opts.version==8:
            self.model = NeuralGeneralizedBarycentricCoordinate8(
                opts, num_layers=4,
                use_exp_recon=False, # not used yet
                use_shp_recon=False, # not used yet
                use_shp=False,
                use_elu=False,
                use_relu=True,
                use_least_N_on_V=False,
                is_train=True,
                device=self.device,
            )
        else:
            raise NotImplementedError('No matching model version')
        
        # load weight
        self.load_weight()
    
    def load_weight(self):
        if self.opts.ckpt:
            if self.opts.continue_ckpt:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, f"*_{self.opts.start_epoch:03d}.pth"))[0]
            else:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
            print(f"Loading... {ckpt}")
            
            ckpt_dict = torch.load(ckpt, map_location=self.device)            
            self.model.load_state_dict(ckpt_dict,strict=False)
        else:
            print('no ckpt found, training from scratch!')

    def evaluate(self):
        
            
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        BS = self.opts.batch_size
        
        selection=[
            # voca | biwi | mf_SEN | coma | mf_ROM
            [True,  False, False, False, False], # 0
            [False,  True, False, False, False], # 1
            [False, False,  True, False, False], # 2
            [False, False, False,  True, False], # 3
            [False, False, False, False,  True], # 4
            [False, False,  True, False,  True], # 5
            [ True, False, False,  True, False], # 6
            [ True,  True,  True,  True,  True], # -1
        ]
        selection = selection[self.opts.data_selection]
        
        
        self.dataset = CBDDataset(
            self.opts, n_components=1_000, is_train=False, is_valid=False,
            use_voca=selection[0],
            use_biwi=selection[1],
            use_mf_SEN=selection[2],
            use_coma=selection[3],
            use_mf_ROM=selection[4],
        )
        
        # self.neighbor_maps = {
        #     i: igl.adjacency_list(mesh_info['face'])
        #     for i, mesh_info in enumerate([
        #         self.dataset.voca_mesh,
        #         self.dataset.biwi_mesh,
        #         self.dataset.mf_SEN_mesh,
        #     ])
        # }
        # self.neighbor_pad_mask = {}
        # for i in self.neighbor_maps.keys():
        #     # (idx_pad, mask)
        #     self.neighbor_pad_mask[i] = build_padded_neighbors(self.neighbor_maps[i], device=self.device)

        data_sampler = CBDdataSampler(
            self.dataset.len_list, 
            self.opts.batch_size,
            shuffle=False,
            balance=False,
            is_train=False,
            is_valid=False,
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_sampler=data_sampler,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0,
        )
        
        ##########################################################################################################
        
        
        ###### Logging ###########################################################################################
        # make logdir --------------------------------------------------------------------------------------------
        os.makedirs(self.opts.log_dir, exist_ok=True)
        
        import datetime
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        
        tag = "-NGBC"
        if self.opts.optim_cage:
            tag += "-optim_cage"
            
        ckpt_path = self.opts.ckpt.split('ckpts_CBD')[-1][1:]
        self.opts.log_dir = os.path.join(self.opts.log_dir, ckpt_path+'-eval-'+now)
        
        os.makedirs(self.opts.log_dir, exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        
        # save options as json -----------------------------------------------------------------------------------
        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
            
        # save train option as yml
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), opts)
        
        # self logger
        self.logger = open(os.path.join(self.opts.log_dir, "log.txt"), 'w')
        print(f'Saving log at: {self.opts.log_dir}')
        
        print(self.dataset.get_data_config())
        print(data_sampler.get_sampler_config())

        self.logger.write(self.dataset.get_data_config())
        self.logger.write(data_sampler.get_sampler_config())
        #---------------------------------------------------------------------------------------------------------
        ##########################################################################################################
        
        
        
        # eval loop ##############################################################################################
        global_step = 0
        BEST_LOSS = 100_000_000
        
        check_usage = False
        
        len_data = len(self.dataloader)
        denom = 1 / len_data
        
        self.model.eval()
        
        losses_val = {
            "MSE": 0.0,
        }
        
        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)
        for index, batch in pbar:
            
            # model forward ----------------------------------------------------------------------------------
            with torch.no_grad():
                pred_vertices, recon_vertices, recon_source, exp_z, pred_source = self.model(
                    batch.template, batch.vertices, 
                    batch.template_normal, batch.vertices_normal,
                    batch.mesh_data, epoch=0
                )
            # ------------------------------------------------------------------------------------------------
            
            
            # Metric -----------------------------------------------------------------------------------------
            with torch.no_grad():
                mesh_data_num = batch.mesh_data.cpu().numpy()
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf'])[mesh_data_num]
                
                HB = batch.vertices.shape[0] // 2
                
                losses_val['MSE'] += F.mse_loss(batch.vertices, pred_vertices).item() * denom # for NGBC model
            # ------------------------------------------------------------------------------------------------
        
            
            # ------------------------------------------------------------------------------------------------
            interv_val = round(len_data / 5)
            if index % interv_val == 0:
                # for visualization
                vertices = batch.vertices.cpu()
                faces = batch.faces.cpu()
                                
                frame = HB
                v_list = [
                    vertices[0].cpu().detach(),
                    # vertices[1].cpu().detach(),
                    # vertices[HB].cpu().detach(),
                    # vertices[BS-1].cpu().detach(),
                    pred_vertices[0].cpu().detach(),
                    # pred_vertices[1].cpu().detach(),
                    # pred_vertices[HB].cpu().detach(),
                    # pred_vertices[BS-1].cpu().detach(),
                ]
                len_v = len(v_list)
                f_list=[faces] * len_v
                save_logdir = f"{self.opts.log_dir}/img"
                save_img_name = f"{index:04d}"
                
                plot_image_array(
                    v_list, f_list, 
                    rot_list=[[0,0,0]]*len_v,
                    size=1, bg_black=False, mode='shade', 
                    logdir=save_logdir, 
                    name=save_img_name, save=True
                )
        ##########################################################################################################
        
        # write log
        log_text = f"[Eval] "
        for key, value in losses_val.items():
            txt = f"{key}: {value:.6e} "
            print(txt)
            log_text += txt
        self.logger.write(log_text+"\n")                
        print('done!')

    @staticmethod
    def set_seed(opts):
        # set seed
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
    #### voca | biwi | mf_SEN | coma | mf_ROM | mf_all |
    ####   0  |   1  |    2   |   3  |    4   |    5   |
    
        python eval_CBD.py --version 2 --ckpt ./ckpts_CBD/2025-09-08-19-15-47-NGBC --in_type 1 --num_cage_v 768 --data_selection -1
        python eval_CBD.py --version 2 --ckpt ./ckpts_CBD/2025-09-08-13-21-05-NGBC --in_type 0 --num_cage_v 1024 --data_selection 5 --start_epoch 400
        
        python eval_CBD.py --version 2 --ckpt ./ckpts_CBD/2025-09-11-15-26-03-NGBC --in_type 0 --out_type 0 --num_cage_v 640 --data_selection 0
        python eval_CBD.py --version 2 --ckpt ./ckpts_CBD/2025-09-11-17-21-05-NGBC --in_type 0 --out_type 2 --num_cage_v 640 --data_selection 0 --batch_size 1 --device 'cpu'

        python eval_CBD.py --version 2 --ckpt ./ckpts_CBD/2025-09-11-17-53-39-NGBC --in_type 0 --out_type 2 --num_cage_v 640 --data_selection 0

        python eval_CBD.py --version 8 --ckpt ./ckpts_CBD/2025-09-25-18-27-55-NGBCv8 --in_type 1 --out_type 1 --num_cage_v 640 --data_selection 0
        
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-26-10-16-32-NGBCv5 --in_type 1 --out_type 1 --num_cage_v 640 --data_selection 0
    """
    # argparse configs
    opts = Options()
    
    # base configs (yaml)
    opts_yaml = yaml.load(open(opts.config), Loader=yaml.FullLoader)
        
    # update with argparse configs
    opts_ = vars(opts)
    opts_yaml.update(opts_)
    opts = argparse.Namespace(**opts_yaml)
    
    trainer = Trainer(opts)
    trainer.evaluate()

