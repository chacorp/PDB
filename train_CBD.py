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

# import sys
# from pathlib import Path
# __abs_path__ = str(Path(__file__).parents[1].absolute())
# __deep_cage_path__ = f'{__abs_path__}/third_party/deep_cage'

# for __util_path__ in [__abs_path__, __deep_cage_path__]:
#     if not __util_path__ in sys.path:
#         sys.path+=[__util_path__]

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
from utils.matplotlib_rnd import plot_image_array, plot_image_array_seg, vis_rig, plot_image_array_points
from utils.ckpt_utils import *
from utils.remesh_utils import build_padded_neighbors, pca_normal_axis_vectorized
from utils.exp_utils import plateau_hat_points
from utils.mesh_utils import calc_norm_torch

# from utils.exp_utils import Model_mk1, Model_mk3_1
# from utils.remesh_utils import compute_MVC_vertexwise, apply_MVC_weights_batch, build_padded_neighbors, pca_normal_axis_vectorized

from models.baseline import CageNet
from models.NGBC import (
    NeuralGeneralizedBarycentricCoordinate, 
    NeuralGeneralizedBarycentricCoordinateLBS, 
    NeuralGeneralizedBarycentricCoordinateCBD, 
    # NeuralGeneralizedBarycentricCoordinate5,
    # NeuralGeneralizedBarycentricCoordinate8,
    # NeuralGeneralizedBarycentricCoordinate55,
)



# sys.path = list(set(sys.path))
def Options():
    parser = argparse.ArgumentParser(description='neural generalized barycentric coordinate for FA retargeting')
    parser.add_argument('-c', '--config', default='config/train_CBD.yml', help='config file path')
    parser.add_argument("--device",       type=str,   default="cuda:0")
    
    parser.add_argument("--log_dir",      type=str,   default="ckpts_CBD")

    parser.add_argument("--version",      type=int,   default=5,      help='train method (1: baseline, 2: ours)')
    parser.add_argument("--num_cage_v",   type=int,   default=1024,   help='number of cage vertices')
    
    parser.add_argument("--in_type",      type=int,   default=1,
                        help='input type (0: position, 1: position + normal')
    parser.add_argument("--out_type",      type=int,   default=1,      
                        help='output type (0: cage v, 1: cage delta_v, 2: cage delta_T mat, 3: vertex T mat')
    
    parser.add_argument("--save_interval",type=int,   default=50,     help='save interval epoch')
    parser.add_argument("--max_epoch",    type=int,   default=500,    help='number of epochs')
    parser.add_argument("--start_epoch",  type=int,   default=0,      help='number of epochs')
    parser.add_argument("--lr",           type=float, default=0.0002, help='learning rate')
    parser.add_argument("--sc_step",      type=int,   default=10,     help='scheduler step')
    
    parser.add_argument("--batch_size",   type=int,   default=8,      help='batch size')
    parser.add_argument("--seed",         type=int,   default=42,     help='random seed')
    parser.add_argument("--ckpt",         type=str,   default=None)    
    parser.add_argument("--continue_ckpt",dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)
    
    parser.add_argument("--n_sampling",   dest='n_sampling', action='store_true')
    parser.set_defaults(n_sampling=False)
    
    parser.add_argument("--use_decimate", dest='use_decimate', action='store_true')
    parser.set_defaults(use_decimate=False)

    #### Choose a last layer activation for key_weight_model()
    parser.add_argument("--last_activation", choices=["relu", "elu", "softmax", "softplus", "none"],
        help="Choose a last layer activation for NGBC.key_weight_model()"
    )
    
    parser.add_argument("--no_pou",dest='no_pou', action='store_true')
    parser.set_defaults(no_pou=False)
    
    parser.add_argument("--pou_loss",dest='pou_loss', action='store_true')
    parser.set_defaults(pou_loss=False)
    
    parser.add_argument("--use_scheduler",dest='use_scheduler', action='store_true')
    parser.set_defaults(use_scheduler=False)
    
    parser.add_argument("--use_segment_weight",dest='use_segment_weight', action='store_true')
    parser.set_defaults(use_segment_weight=False)
    parser.add_argument("--use_laplacian",dest='use_laplacian', action='store_true')
    parser.set_defaults(use_laplacian=False)
    parser.add_argument("--use_normal_loss",dest='use_normal_loss', action='store_true')
    parser.set_defaults(use_normal_loss=False)
    
    parser.add_argument("--use_perm",dest='use_perm', action='store_true')
    parser.set_defaults(use_perm=False)
    
    ## lbs related options ---
    parser.add_argument("--start_stage", type=int, default=1, choices=[1,2], help='start training from stage 1 (LBS pretrain) or stage 2 (CBD delta)')
    parser.add_argument("--num_lbs_joints", type=int, default=4, help='number of joints for LBS')
    parser.add_argument("--lbs_pretrained_epochs", type=int, default=50, help='number of epochs to pretrain LBS')
    parser.add_argument("--use_lbs", dest='use_lbs',  action='store_true')
    parser.set_defaults(use_lbs=False)
    parser.add_argument("--vis_joint_pos",dest='vis_joint_pos', action='store_true')
    parser.set_defaults(vis_joint_pos=False)
    parser.add_argument("--use_lbs_joint_center", dest='use_lbs_joint_center',  action='store_true')
    parser.set_defaults(use_lbs_joint_center=False)     
    parser.add_argument("--use_exp_joint_predict", dest='use_exp_joint_predict',  action='store_true')
    parser.set_defaults(use_exp_joint_predict=False)
    parser.add_argument("--use_joint_predict", dest='use_joint_predict',  action='store_true')
    parser.set_defaults(use_joint_predict=False)
    parser.add_argument("--use_weighted_joint_pos", dest='use_weighted_joint_pos',  action='store_true')
    parser.set_defaults(use_weighted_joint_pos=False)
    parser.add_argument("--no_use_translation", dest='no_use_translation', action='store_true')
    parser.set_defaults(no_use_translation=False)
    parser.add_argument("--use_lbs_laplacian",dest='use_lbs_laplacian', action='store_true')
    parser.set_defaults(use_lbs_laplacian=False)
    parser.add_argument("--use_lbs_ent",dest='use_lbs_ent', action='store_true')
    parser.set_defaults(use_lbs_ent=False)
    parser.add_argument("--use_lbs_t",dest='use_lbs_t', action='store_true')
    parser.set_defaults(use_lbs_t=False)
    parser.add_argument("--use_lbs_R",dest='use_lbs_R', action='store_true')
    parser.set_defaults(use_lbs_R=False)
    parser.add_argument("--use_lbs_bal",dest='use_lbs_bal', action='store_true')
    parser.set_defaults(use_lbs_bal=False)
    parser.add_argument("--use_hyb_delta_lbs_input",dest='use_hyb_delta_lbs_input', action='store_true')
    parser.set_defaults(use_hyb_delta_lbs_input=False)
    parser.add_argument("--use_hyb_concat_lbs",dest='use_hyb_concat_lbs', action='store_true')
    parser.set_defaults(use_hyb_concat_lbs=False)
    parser.add_argument("--use_hyb_joint_train",dest='use_hyb_joint_train', action='store_true')
    parser.set_defaults(use_hyb_joint_train=False)
    
    parser.add_argument("--debug_stage",dest='debug_stage', action='store_true')
    parser.set_defaults(debug_stage=False)
    ## ----------------------
    
    parser.add_argument("--no_t_mask",dest='no_t_mask', action='store_true')
    parser.set_defaults(no_t_mask=False)
    
    parser.add_argument("--use_data0",dest='use_data0', action='store_true')
    parser.set_defaults(use_data0=False)
    parser.add_argument("--use_data1",dest='use_data1', action='store_true')
    parser.set_defaults(use_data1=False)
    parser.add_argument("--use_data2",dest='use_data2', action='store_true')
    parser.set_defaults(use_data2=False)
    parser.add_argument("--use_data3",dest='use_data3', action='store_true')
    parser.set_defaults(use_data3=False)
    parser.add_argument("--use_data9",dest='use_data9', action='store_true')
    parser.set_defaults(use_data9=False)
    parser.add_argument("--data_toggle",dest='data_toggle', action='store_true')
    parser.set_defaults(data_toggle=False)

    parser.add_argument("--tb",           action='store_true')
    parser.set_defaults(is_train=True)
    
    parser.add_argument("--optim_cage",dest='optim_cage', action='store_true')
    parser.set_defaults(optim_cage=False)
    
    parser.add_argument("--align_latent",dest='align_latent', action='store_true')
    parser.set_defaults(align_latent=False)
    
    args = parser.parse_args()
    return args

# --- Loss Functions ---

## ------------ Trainer.train_v1() ------------
def mvc_loss(mvc_weights):
    """ penalize MVC with negative values """
    # mvc_weights.shape : (number of target verteices, number of cage vertices=K)
    # cage vertices should enclose target vertices, so mvc_weights should be all positive values
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
## --------------------------------------------
def non_ict_loss(pred):
    """Reference from Neural Face Rigging for Animating and Retargeting Facial Meshes in the Wild [Qin et al. 2023], Eq.(3)
    L_FACS = {   
         -x, x < 0 
          0, 0 <= x < 1
        x-1, x > 1
    }
    Args:
        pred (torch.tensor): predicted expression code
    
    Returns:
        loss
    """
    
    loss = torch.where(pred < 0, -pred, torch.where(pred > 1, pred - 1, torch.zeros_like(pred))).mean()
    return loss

def laplacian_loss(batch, pred_key_weight, dataset, mesh_data_num, device):
    """
    Args:
        batch: data class
        pred_key_weight: coordinate prediction (B,V,C)
        dataset: train dataset
        device: cpu, cuda
    Returns:
        laplacian smoothing loss
    """
    ## operator cotangent L 구하기 
        ## 각 vertex의 값이 이웃들과 얼마나 다른지를 측정하는 차분 연산자(cotangent laplacian의 연산식 형태로 만들어서 계산해주는 행렬)
        ## 그 중에서도 cotangent Laplacian (cotangent weights) 사용, mesh의 local geometry를 반영 (이웃한 삼각형의 각도·면적을 고려)
        ## VOCA / BIWI / MultiFace 같이 raw-scan 기반 >> identity마다 connectivity가 다를 수 있어서, id별로 달리 cotangent L 구해놓은 것
    if mesh_data_num == 0:
        L = dataset.voca_cotmatrix[batch.id_name].to(device) # (V,V)
    elif mesh_data_num == 1:
        L = dataset.biwi_cotmatrix[batch.id_name].to(device) 
    elif mesh_data_num == 2:
        L = dataset.mf_SEN_cotmatrix[batch.id_name].to(device)
    elif mesh_data_num == 3:
        L = dataset.coma_cotmatrix[batch.id_name].to(device)
    elif mesh_data_num == 4:
        L = dataset.mf_ROM_cotmatrix[batch.id_name].to(device)
    elif mesh_data_num == 5:
        L = dataset.ict_cotmatrix[batch.id_name].to(device)
    else:
        raise ValueError(f'no data for {mesh_data_num}')
        
    loss = 0
    for pred_key_w in pred_key_weight:
        ## vertex i 에서 cage k의 weight가 이웃들과 얼마나 다르냐
        pred_key_w_lap = L @ pred_key_w # (V,V) @ (V,C) = (V,C), cage별 weight field의 local variation(gradient-like quantity) 
        ## 지금은 global laplacian
            # 모든 vertex의 local Laplacian 응답을 한 번에 더함
        loss += pred_key_w_lap.sum(0).pow(2).mean() # (V,C) -> (1,C) -> (1,C) -> (1,)
        
        ## (maybe..?) option 2: local laplacian
        # loss += pred_key_w_lap.sum(0).pow(2).mean()
        
    return loss

def lbs_laplacian_loss(batch, W_lbs, dataset, mesh_data_num, device):
    ## for local smoothing of LBS weights
    ## L: (V, V)
    if mesh_data_num == 0:
        L = dataset.voca_cotmatrix[batch.id_name].to(device) # (V,V)
    elif mesh_data_num == 1:
        L = dataset.biwi_cotmatrix[batch.id_name].to(device) 
    elif mesh_data_num == 2:
        L = dataset.mf_SEN_cotmatrix[batch.id_name].to(device)
    elif mesh_data_num == 3:
        L = dataset.coma_cotmatrix[batch.id_name].to(device)
    elif mesh_data_num == 4:
        L = dataset.mf_ROM_cotmatrix[batch.id_name].to(device)
    elif mesh_data_num == 5:
        L = dataset.ict_cotmatrix[batch.id_name].to(device)
    else:
        raise ValueError(f'no data for {mesh_data_num}')

    # W_lbs: (B, V, J) -> (V, V) @ (B, V, J) = (B, V, J)
    LW = torch.matmul(L, W_lbs) # torch supprots (V,V) @ (B,V,J)     
    return LW.pow(2).mean()

def lbs_entropy_loss(W_lbs, eps=1e-8):
    ## for locality/sparsity of LBS weights
    ## (B, V, J)
    ent = -(W_lbs * (W_lbs + eps).log()).sum(-1) # (B, V)
    return ent.mean() # scalar

def lbs_transform_reg(T_lbs):
    ## trasform matrix magniture regularization for LBS
    R = T_lbs[..., :3] # (B, J, 3, 3)
    t = T_lbs[..., 3]  # (B, J, 3)
    I = torch.eye(3, device=R.device, dtype=R.dtype)[None, None] # [None, None]을 붙이면 shape이 (1,1,3,3)이 되어 (B,J,3,3)인 R에 broadcast 가능
    loss_t = (t.pow(2)).mean() 
    loss_R = ((R-I).pow(2)).mean() # 회전이 얼마나 identity(=회전 없음)에서 벗어났나를 측정
    return loss_t, loss_R

def lbs_usage_balance_loss(W_lbs):
    ## joint usage balancing loss for LBS weights
    ## 어떤 joint가 거의 0으로 죽거나, 어떤 joint가 전체를 독점하는 degenerate 해를 줄임   
    usage = W_lbs.mean(dim=1) # (B, J)
    target = torch.full_like(usage, 1.0 / usage.shape[-1]) # (B, J), 1/J: 모든 joint가 평균적으로 비슷하게 쓰이길이라는 아주 약한 prior
    return F.mse_loss(usage, target)


class Logger():
    def __init__(self, file_path):
        self.file_path = file_path
        
    def write(self, txt):
        with open(self.file_path, 'a') as f:
            f.write(txt)
        f.close()

class Trainer():
    def __init__(self, opts):
        # set opts
        self.opts = opts
        self.set_seed(self.opts)
        self.device = opts.device

        last_act_list = ["relu", "elu", "softmax", "softplus", "none"]
        last_act_list = [self.opts.last_activation==l_act for l_act in last_act_list]
        if opts.version==1:
            # from deep_cage import NetworkFull
            
            # self.model = NetworkFull(device=self.device, optim_cage=self.opts.optim_cage).to(self.device)
            self.model = CageNet(device=self.device, optim_cage=self.opts.optim_cage)
        elif opts.version==5:
            # self.model = NeuralGeneralizedBarycentricCoordinate5(
            self.model = NeuralGeneralizedBarycentricCoordinate(
                opts, num_layers=4,
                num_cage_vertices=self.opts.num_cage_v,
                use_exp_recon=False, # not used yet
                use_shp_recon=False, # not used yet
                use_shp=False,
                use_relu=last_act_list[0],
                use_elu=last_act_list[1],
                use_softmax=last_act_list[2],
                use_softplus=last_act_list[3],
                no_activation=last_act_list[4],
                is_train=True,
                use_pou = ~self.opts.no_pou,
                device=self.device,
                #hid_dim=128 if self.opts.use_data2 or self.opts.use_data3 else 256,
                hid_dim=128 if self.opts.align_latent else 256,
            )
        elif opts.version==6:
            # self.model = NeuralGeneralizedBarycentricCoordinate5(
            self.model = NeuralGeneralizedBarycentricCoordinateLBS(
                opts, num_layers=4,
                num_cage_vertices=self.opts.num_cage_v,
                use_exp_recon=False, # not used yet
                use_shp_recon=False, # not used yet
                use_shp=False,
                use_relu=last_act_list[0],
                use_elu=last_act_list[1],
                use_softmax=last_act_list[2],
                use_softplus=last_act_list[3],
                no_activation=last_act_list[4],
                is_train=True,
                use_pou = ~self.opts.no_pou,
                device=self.device,
                #hid_dim=128 if self.opts.use_data2 or self.opts.use_data3 else 256,
                hid_dim=128 if self.opts.align_latent else 256,
            )
        elif opts.version==7 or opts.version==8:
            # self.model = NeuralGeneralizedBarycentricCoordinate5(
            self.model = NeuralGeneralizedBarycentricCoordinateLBS(
                opts, num_layers=4,
                num_cage_vertices=self.opts.num_cage_v,
                use_exp_recon=False, # not used yet
                use_shp_recon=False, # not used yet
                use_shp=False,
                use_relu=last_act_list[0],
                use_elu=last_act_list[1],
                use_softmax=last_act_list[2],
                use_softplus=last_act_list[3],
                no_activation=last_act_list[4],
                is_train=True,
                use_pou = ~self.opts.no_pou,
                device=self.device,
                #hid_dim=128 if self.opts.use_data2 or self.opts.use_data3 else 256,
                hid_dim=128 if self.opts.align_latent else 256,
            )
            self.model_CBD = NeuralGeneralizedBarycentricCoordinateCBD(
                opts, num_layers=4,
                num_cage_vertices=self.opts.num_cage_v, #32
                use_exp_recon=False, # not used yet
                use_shp_recon=False, # not used yet
                use_shp=False,
                use_relu=last_act_list[0],
                use_elu=last_act_list[1],
                use_softmax=last_act_list[2],
                use_softplus=last_act_list[3],
                no_activation=last_act_list[4],
                is_train=True,
                use_pou = ~self.opts.no_pou,
                device=self.device,
                #hid_dim=128 if self.opts.use_data2 or self.opts.use_data3 else 256,
                hid_dim=128 if self.opts.align_latent else 256,
            )
        else:
            raise NotImplementedError('No matching model version')
        
        ckpt_has_cbd = False        
        if self.opts.ckpt is not None:
            if self.opts.continue_ckpt:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, f"*_{self.opts.start_epoch:03d}.pth"))[0]
            else:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
                
            ckpt_dict = torch.load(ckpt, map_location="cpu")
            ckpt_has_cbd = any(k.startswith("cbd_") or k.startswith("key_d_model") for k in ckpt_dict.keys())

        if self.opts.use_lbs and self.opts.start_stage == 2 and ckpt_has_cbd:
            # stage2 ckpt에는 CBD branch 키가 포함되어 있으므로
            # load_state_dict 전에 반드시 branch를 만들어야 함
            self.model.load_CBD_brach(self.opts.use_lbs)
            self.cbd_loaded = True  
        
        
        self.load_weight()
        # load weight
    
    def load_weight(self):
        if self.opts.ckpt:
            print(f"Loading... {self.opts.ckpt}")
            if self.opts.continue_ckpt:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, f"*_{self.opts.start_epoch:03d}.pth"))[0]
            else:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
            ckpt_dict = torch.load(ckpt)            
            self.model.load_state_dict(ckpt_dict)
            print(f"Loaded! {ckpt}")
        else:
            print('no ckpt found, training from scratch!')

    def train_v1(self, epochs):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.opts.lr, betas=(0.9, 0.999))
        
        if self.opts.optim_cage:
            self.model.cage_v = nn.Parameter(self.model.cage_v)
            self.optimizer_cage = torch.optim.AdamW([self.model.cage_v], lr=0.0002, betas=(0.9, 0.999))
        # if self.opts.use_scheduler:
        #     self.scheduler = torch.optim.lr_scheduler.StepLR(
        #         self.optimizer, 
        #         step_size=self.opts.sc_step, 
        #         gamma=self.opts.sc_gamma
        #     )
            
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        BS = self.opts.batch_size
        
        self.train_dataset = CBDDataset(
            self.opts, is_train=True, toggle=self.opts.data_toggle
        )
        self.valid_dataset = CBDDataset(
            self.opts, is_train=True, toggle=self.opts.data_toggle
        )
        if self.opts.use_data2:
            self.neighbor_pad_mask = {
                2: build_padded_neighbors(
                    igl.adjacency_list(
                        self.train_dataset.mf_SEN_mesh['face']
                    ), device=self.device
                ),
                4: build_padded_neighbors(
                    igl.adjacency_list(
                        self.train_dataset.mf_SEN_mesh['face']
                    ), device=self.device
                ),
                5: build_padded_neighbors(
                    igl.adjacency_list(
                        self.train_dataset.ict_face_model.faces
                    ), device=self.device
                ),
                6: build_padded_neighbors(
                    igl.adjacency_list(
                        self.train_dataset.ict_face_model.faces
                    ), device=self.device
                ),
            }
        else:
            self.neighbor_maps = {
                i: igl.adjacency_list(mesh_info['face'])
                for i, mesh_info in enumerate([
                    self.train_dataset.voca_mesh,
                    self.train_dataset.biwi_mesh,
                    self.train_dataset.mf_SEN_mesh,
                ])
            }
            self.neighbor_pad_mask = {}
            for i in self.neighbor_maps.keys():
                # (idx_pad, mask)
                self.neighbor_pad_mask[i] = build_padded_neighbors(self.neighbor_maps[i], device=self.device) 
        
        train_sampler = CBDdataSampler(
            self.train_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_train=True
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_sampler=train_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0,
        )
        
        valid_sampler = CBDdataSampler(
            self.valid_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_train=False,
            is_valid=True,
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset, 
            batch_sampler=valid_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0
        )
        ##########################################################################################################
        
        
        ###### Logging ###########################################################################################
        # make logdir --------------------------------------------------------------------------------------------
        os.makedirs(self.opts.log_dir, exist_ok=True)
        import datetime
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        
        tag = "-CBD"
        if self.opts.optim_cage:
            tag += "-optim_cage"
        self.opts.log_dir = os.path.join(self.opts.log_dir, now+tag)
        os.makedirs(self.opts.log_dir, exist_ok=True)

        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/valid/mesh", exist_ok=True)
        
        # save options as json -----------------------------------------------------------------------------------
        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
            
        # save train option as yml
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), opts)
        
        if self.opts.tb:
            train_ = os.path.join(self.opts.log_dir, "train")
            valid_ = os.path.join(self.opts.log_dir, "valid")
            self.writer_train = SummaryWriter(log_dir=train_)
            self.writer_valid = SummaryWriter(log_dir=valid_)
        
        # self logger
        self.logger = Logger(os.path.join(self.opts.log_dir, "log.txt"))
        print(f'Saving log at: {self.logger.file_path}')
        
        print(self.train_dataset.get_data_config())
        print(train_sampler.get_sampler_config())
        print(self.valid_dataset.get_data_config())
        print(valid_sampler.get_sampler_config())
        
        self.logger.write(self.train_dataset.get_data_config())
        self.logger.write(train_sampler.get_sampler_config())  
        self.logger.write(self.valid_dataset.get_data_config())      
        self.logger.write(valid_sampler.get_sampler_config())
        #---------------------------------------------------------------------------------------------------------
        ##########################################################################################################
        
        
        
        # training loop ##########################################################################################
        global_step = 0
        BEST_LOSS = 100_000_000
        BEST_EPOCH = 0
        if self.opts.start_epoch != 0:
            start_epoch = self.opts.start_epoch + 1
        else:
            start_epoch = self.opts.start_epoch
                
        # define loss lamdba 
        self.loss_lambda = {
            "mvc": self.opts.lambda_mvc,
            "align": self.opts.lambda_align,
            "p2f": self.opts.lambda_p2f,
            "norm": self.opts.lambda_norm,
            # symm 
        }
        
        check_usage = False
        
        len_train_data = len(self.train_dataloader)
        len_valid_data = len(self.valid_dataloader)
        interv_train = round(len_train_data / 10)
        for epoch in range(start_epoch, epochs+1):
            print(f"[{epoch:03d}/{epochs:03d}][Train]")
            
            ## for logging loss!
            running_losses = {
                "mvc": 0.0,
                "align": 0.0,
                "p2f": 0.0,
                "norm": 0.0,
                "total": 0.0
            }
            
            self.model.train()
            train_counter = 0
            
            pbar = tqdm(enumerate(self.train_dataloader), total=len_train_data, position=0, ncols=100)
                                             
            for index, batch in pbar:
                
                self.optimizer.zero_grad()
                if self.opts.optim_cage:
                    self.optimizer_cage.zero_grad()

                
                # model prediction -------------------------------------------------------------------------------
                pred_vertices, pred_template, mvc_weights, source_cage_v, deform_cage_v = self.model(
                    # batch_template_v, batch_vertices_v, epoch=epoch, return_cage=True
                    batch.template, batch.vertices, epoch=epoch, return_cage=True
                    # batch.template[0,None], batch.vertices, epoch=epoch, return_cage=True
                )
                # ------------------------------------------------------------------------------------------------
                
                ##################################################################################################
                # ------------------------------------------------------------------------------------------------                
                mesh_data_num = batch.mesh_data.cpu().numpy()
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf', 'ict'])[mesh_data_num]
                                
                
                idx_pad, mask = self.neighbor_pad_mask[batch.mesh_data.item()]
                normals_before = pca_normal_axis_vectorized(batch.template, idx_pad, mask)
                normals_after = pca_normal_axis_vectorized(pred_vertices, idx_pad, mask)
                                
                loss_dict = {} # make it as a dictionary
                                
                loss_dict['mvc'] = mvc_loss(mvc_weights)
                loss_dict['align'] = F.mse_loss(
                    batch.vertices,
                    pred_vertices,
                ) + F.mse_loss(
                    batch.template,
                    pred_template,
                )
                loss_dict['p2f']   = p2f_loss(batch.template, pred_vertices, normals_before, normals_after)                
                loss_dict['norm']  = norm_loss(normals_before, normals_after)
                # ------------------------------------------------------------------------------------------------
                ##################################################################################################
                               
                
                # get total loss (lambda weights are multiplied here!)
                loss = 0
                for key, value in loss_dict.items():
                    key_ = key.split("_")[0]
                    tmp = value*self.loss_lambda[key_]
                    loss += tmp
                    running_losses[key] += tmp
                loss_dict["total"] = loss 

                # running loss
                running_losses["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5e}")
                # ------------------------------------------------------------------------------------------------
                # backward
                loss.backward()

                if True:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                if self.opts.optim_cage:
                    self.optimizer_cage.step()                
                # ------------------------------------------------------------------------------------------------
                
                global_step += 1
                train_counter += 1
                                
                interv_train = round(len_train_data / 10)
                if train_counter % interv_train == 1:
                    
                    # with torch.no_grad():
                    #     pred_vertices, pred_template, mvc_weights, source_cage_v, deform_cage_v = self.model(
                    #         batch.template, batch.vertices, epoch=epoch, return_cage=True
                    #     )
                        
                    # for visualization
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                    
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Train] "
                    for key, value in running_losses.items():
                        log_text += f"{key}: {value:.6f} "
                    self.logger.write(log_text+"\n")
                    
                    frame = BS//2
                    v_list = [ v for v in vertices[frame:frame+2] ] + \
                        [ v for v in pred_vertices[frame:frame+2].cpu().detach() ]
                                            
                    #v_list = [v for v in vertices[frame:frame+2]]+[batch.template.cpu()[0]]*2
                    len_v = len(v_list)
                    f_list = [faces] * len_v

                    # import pdb;pdb.set_trace()
                    cv_list = [ v for v in source_cage_v[frame:frame+2].cpu().detach() ] + \
                        [ v for v in deform_cage_v[frame:frame+2].cpu().detach() ] + \
                        [ self.model.cage_v.cpu().detach() ]
                    len_cv = len(cv_list)
                    cf_list = [self.model.cage_f.cpu().detach()] * len_cv
                    v_list += cv_list
                    f_list += cf_list
                    
                    save_logdir = f"{self.opts.log_dir}/img/train/mesh"
                    save_img_name = f"{epoch:03d}_{index:04d}"

                    plot_image_array(
                        v_list, f_list, 
                        rot_list=[[0,0,0]] * (len_v + len_cv),
                        size=1, bg_black=False, mode='shade',
                        logdir=save_logdir,
                        name=save_img_name, save=True
                    )
                    
                
                if self.opts.debug:
                    break
                # ------------------------------------------------------------------------------------------------
            
            ### scheduler (not used - for now...)
            # if self.opts.use_scheduler:
            #     self.scheduler.step()
            
            # log
            if self.opts.tb:
                self.log_loss(self.writer_train, running_losses, epoch, train_counter)

            # save model
            if epoch % 100 == 0:
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_{epoch:03d}.pth')
            
            
            
            # validation -----------------------------------------------------------------------------------------
            self.model.eval()
            print(f"[{epoch:03d}/{epochs:03d}][Valid]")
            running_losses_val = {
                "mvc": 0.0,
                "align": 0.0,
                "p2f": 0.0,
                "norm": 0.0,
                "total": 0.0
            }
            
            counter = 0
            pbar = tqdm(enumerate(self.valid_dataloader), total=len_valid_data, ncols=100)
            for index, batch in pbar:
                counter += 1
                
                # model validation -------------------------------------------------------------------------------
                with torch.no_grad():
                    pred_vertices, mvc_weights = self.model(batch.template, batch.vertices, epoch=epoch)
                # ------------------------------------------------------------------------------------------------
                
                
                ##################################################################################################
                # ------------------------------------------------------------------------------------------------ 
                with torch.no_grad():
                    mesh_data_num = batch.mesh_data.cpu().numpy()
                    mesh_data = np.array(['voca', 'biwi', 'mf','voca','mf','ict'])[mesh_data_num]
                    template_expanded = batch.template#.expand_as(pred_vertices)
                    
                    
                    idx_pad, mask = self.neighbor_pad_mask[batch.mesh_data.item()]
                    normals_before = pca_normal_axis_vectorized(template_expanded, idx_pad, mask)
                    normals_after = pca_normal_axis_vectorized(pred_vertices, idx_pad, mask)

                    loss_dict = {} # make it as a dictionary
                    
                    loss_dict['mvc'] = mvc_loss(mvc_weights)
                    loss_dict['align'] = F.mse_loss(batch.vertices, pred_vertices)
                    loss_dict['p2f']   = p2f_loss(template_expanded, pred_vertices, normals_before, normals_after)
                    loss_dict['norm']  = norm_loss(normals_before, normals_after)
                # ------------------------------------------------------------------------------------------------
                ##################################################################################################
                
                # get total loss
                loss = 0
                for key, value in loss_dict.items():
                    key_ = key.split("_")[0]
                    tmp = value.item()*self.loss_lambda[key_]
                    loss += tmp
                    running_losses_val[key] += tmp
                loss_dict["total"] = loss 

                # running loss
                running_losses_val["total"] += loss_dict["total"]
            
                pbar.set_description(f"total loss: {loss:.5e}")
                
                # ------------------------------------------------------------------------------------------------
                interv_val = round(len_valid_data / 5)
                if index % interv_val == 0:
                    # for visualization
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Valid] "
                    for key, value in running_losses_val.items():
                        log_text += f"{key}: {value:.6f} "
                    self.logger.write(log_text+"\n")

                    frame = BS//2
                    v_list = [ v for v in vertices[frame:frame+2] ] + \
                        [ v for v in pred_vertices[frame:frame+2].cpu().detach() ]
                    len_v = len(v_list)
                    f_list=[faces] * len_v
                    save_logdir = f"{self.opts.log_dir}/img/valid/mesh"
                    save_img_name = f"{epoch:03d}_{index:04d}"
                    
                    plot_image_array(
                        v_list, f_list, 
                        rot_list=[[0,0,0]]*len_v,
                        size=1, bg_black=False, mode='shade', 
                        logdir=save_logdir, 
                        name=save_img_name, save=True
                    )
                    
                # ------------------------------------------------------------------------------------------------
                if self.opts.debug:
                    break
            # log
            if self.opts.tb:
                self.log_loss(self.writer_valid, running_losses_val, epoch, counter)
            
            # best loss
            val_loss = running_losses_val["total"]/counter
            if val_loss < BEST_LOSS:
                BEST_LOSS = val_loss
                BEST_EPOCH = epoch
                print(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6f} - Best epoch: {BEST_EPOCH:03d}\n")
                self.logger.write(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6f}\n")
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_best.pth')
            else:
                log_txt_val = f"[{epoch:03d}/{epochs:03d}] Curr Loss: {val_loss:.6f} (Best Loss: {BEST_LOSS:.6f} - Best epoch: {BEST_EPOCH:03d})\n"
                self.logger.write(log_txt_val)
                print(log_txt_val)
    
    def train_v2(self, epochs):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.opts.lr,
            betas=(0.9, 0.999)
        )
        
        #if self.opts.use_scheduler:
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.opts.sc_step, 
            gamma=self.opts.sc_gamma
        )
            
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        BS = self.opts.batch_size
        self.train_dataset = CBDDataset(self.opts, is_train=True)
        
        self.neighbor_maps = {
            i: igl.adjacency_list(mesh_info['face'])
            for i, mesh_info in enumerate([
                self.train_dataset.voca_mesh,
                self.train_dataset.biwi_mesh,
                self.train_dataset.mf_SEN_mesh,
            ])
        }
        self.neighbor_pad_mask = {}
        for i in self.neighbor_maps.keys():
            # (idx_pad, mask)
            self.neighbor_pad_mask[i] = build_padded_neighbors(self.neighbor_maps[i], device=self.device)
        
        train_sampler = CBDdataSampler(
            self.train_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_train=True
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_sampler=train_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0,
        )
        
        
        self.valid_dataset = CBDDataset(self.opts, is_valid=True)        
        valid_sampler = CBDdataSampler(
            self.valid_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_valid=True
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset, 
            batch_sampler=valid_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0
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
        self.opts.log_dir = os.path.join(self.opts.log_dir, now+tag)
        os.makedirs(self.opts.log_dir, exist_ok=True)

        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/valid/mesh", exist_ok=True)
        
        # save options as json -----------------------------------------------------------------------------------
        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
            
        # save train option as yml
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), opts)
        
        if self.opts.tb:
            train_ = os.path.join(self.opts.log_dir, "train")
            valid_ = os.path.join(self.opts.log_dir, "valid")
            self.writer_train = SummaryWriter(log_dir=train_)
            self.writer_valid = SummaryWriter(log_dir=valid_)
        
        # self logger
        self.logger = Logger(os.path.join(self.opts.log_dir, "log.txt"))
        print(f'Saving log at: {self.logger.file_path}')
        
        print(self.train_dataset.get_data_config())
        print(train_sampler.get_sampler_config())
        print(self.valid_dataset.get_data_config())
        print(valid_sampler.get_sampler_config())
        
        self.logger.write(self.train_dataset.get_data_config())
        self.logger.write(train_sampler.get_sampler_config())  
        self.logger.write(self.valid_dataset.get_data_config())      
        self.logger.write(valid_sampler.get_sampler_config())
        #---------------------------------------------------------------------------------------------------------
        ##########################################################################################################
        
        
        
        # training loop ##########################################################################################
        global_step = 0
        BEST_LOSS = 100_000_000
        BEST_EPOCH = 0
        start_epoch = self.opts.start_epoch
                        
        # define loss lamdba 
        self.loss_lambda = {
            "recon": self.opts.lambda_vert,
            "exp-z": self.opts.lambda_vert * 0.5,
            "exp-v": self.opts.lambda_vert,
            "shape": self.opts.lambda_vert,
            # symm 
        }
        
        check_usage = False
        
        len_train_data = len(self.train_dataloader)
        len_valid_data = len(self.valid_dataloader)
        interv_train = round(len_train_data / 10)
        for epoch in range(start_epoch, epochs+1):
            print(f"[{epoch:03d}/{epochs:03d}][Train]")
            
            ## for logging loss!
            running_losses = {
                "recon": 0.0,
                "exp-z": 0.0,
                "exp-v": 0.0,
                "shape": 0.0,
                "total": 0.0
            }
            
            self.model.train()
            train_counter = 0
            
            pbar = tqdm(enumerate(self.train_dataloader), total=len_train_data, position=0, ncols=100)
            for index, batch in pbar:
                self.optimizer.zero_grad()
                
                # model prediction -------------------------------------------------------------------------------
                
                pred_vertices, recon_vertices, recon_source, exp_z, pred_source = self.model(
                    batch.template, batch.vertices, 
                    batch.template_normal, batch.vertices_normal,
                    batch.mesh_data, epoch=epoch
                )
                # ------------------------------------------------------------------------------------------------
                
                # loss -------------------------------------------------------------------------------------------
                mesh_data_num = batch.mesh_data.cpu().numpy()
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf','ict'])[mesh_data_num]
                
                loss_dict = {} # make it as a dictionary
                HB = batch.vertices.shape[0] // 2
                loss_dict['recon'] = F.mse_loss(batch.vertices, pred_vertices) # for NGBC model
                if self.model.use_shp_recon:
                    loss_dict['shape'] = F.mse_loss(batch.template, recon_source) # for shape AE
                if self.model.use_exp_recon:
                    loss_dict['exp-v'] = F.mse_loss(batch.vertices, recon_vertices) # for expression AE
                if self.model.use_full_vertex:
                    loss_dict['exp-v'] = F.mse_loss(batch.template, pred_source) # for expression AE

                # loss_dict['exp-z'] = F.mse_loss(exp_z[:HB], exp_z[HB:])

                
                # get total loss (lambda weights are multiplied here!)
                loss = 0
                for key, value in loss_dict.items():
                    key_ = key.split("_")[0]
                    tmp = value*self.loss_lambda[key_]
                    loss += tmp
                    running_losses[key] += tmp
                loss_dict["total"] = loss 

                # running loss for logging
                running_losses["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5e}, mesh data: {mesh_data_num}")
                # ------------------------------------------------------------------------------------------------
                
                # backward ---------------------------------------------------------------------------------------
                loss.backward()
                self.optimizer.step()                
                # ------------------------------------------------------------------------------------------------
                
                global_step += 1
                train_counter += 1
                                
                interv_train = round(len_train_data / 10)
                if train_counter % interv_train == 1:
                    # for visualization
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                    
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Train] "
                    __idx__ = 1/train_counter
                    for key, value in running_losses.items():
                        log_text += f"{key}: {value*__idx__:.6e} "
                    self.logger.write(log_text+"\n")
                    
                    frame = HB
                    v_list = [
                        vertices[0].cpu().detach(),
                        vertices[1].cpu().detach(),
                        vertices[HB].cpu().detach(),
                        vertices[BS-1].cpu().detach(),
                        pred_vertices[0].cpu().detach(),
                        pred_vertices[1].cpu().detach(),
                        pred_vertices[HB].cpu().detach(),
                        pred_vertices[BS-1].cpu().detach(),
                    ]
                    # v_list = [ v for v in vertices[frame:frame+2] ] + \
                    #     [ v for v in pred_vertices[frame:frame+2].cpu().detach() ]
                    
                    #v_list = [v for v in vertices[frame:frame+2]]+[batch.template.cpu()[0]]*2
                    len_v = len(v_list)
                    f_list = [faces] * len_v
                    save_logdir = f"{self.opts.log_dir}/img/train/mesh"
                    save_img_name = f"{epoch:03d}_{index:04d}"

                    plot_image_array(
                        v_list, f_list, 
                        rot_list=[[0,0,0]] * len_v, 
                        size=1, bg_black=False, mode='shade',
                        logdir=save_logdir,
                        name=save_img_name, save=True
                    )
                
                if self.opts.debug:
                    break
                # ------------------------------------------------------------------------------------------------
            
            ### scheduler
            self.scheduler.step()
            
            # log
            if self.opts.tb:
                self.log_loss(self.writer_train, running_losses, epoch, train_counter)

            # save model
            if epoch % self.opts.save_interval == 0:
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_{epoch:03d}.pth')
            
            
            
            # validation -----------------------------------------------------------------------------------------
            self.model.eval()
            print(f"[{epoch:03d}/{epochs:03d}][Valid]")
            running_losses_val = {
                "recon": 0.0,
                "exp-z": 0.0,
                "exp-v": 0.0,
                "shape": 0.0,
                "total": 0.0
            }
            
            counter = 0
            pbar = tqdm(enumerate(self.valid_dataloader), total=len_valid_data, ncols=100)
            for index, batch in pbar:
                counter += 1
                
                # model validation -------------------------------------------------------------------------------
                with torch.no_grad():
                    pred_vertices, recon_vertices, recon_source, exp_z, pred_source = self.model(
                        batch.template, batch.vertices, 
                        batch.template_normal, batch.vertices_normal,
                        batch.mesh_data, epoch=epoch
                    )
                # ------------------------------------------------------------------------------------------------
                
                
                # loss ------------------------------------------------------------------------------------------- 
                with torch.no_grad():
                    mesh_data_num = batch.mesh_data.cpu().numpy()
                    mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf'])[mesh_data_num]
                
                    loss_dict = {} # make it as a dictionary
                    HB = batch.vertices.shape[0] // 2
                    loss_dict['recon'] = F.mse_loss(batch.vertices, pred_vertices) # for NGBC model
                    if self.model.use_shp_recon:
                        loss_dict['shape'] = F.mse_loss(batch.template, recon_source) # for shape AE
                    if self.model.use_exp_recon:
                        loss_dict['exp-v'] = F.mse_loss(batch.vertices, recon_vertices) # for expression AE
                    # loss_dict['exp-z'] = F.mse_loss(exp_z[:HB], exp_z[HB:])
                    
                    # get total loss
                    loss = 0
                    for key, value in loss_dict.items():
                        key_ = key.split("_")[0]
                        tmp = value.item()*self.loss_lambda[key_]
                        loss += tmp
                        running_losses_val[key] += tmp
                    loss_dict["total"] = loss 

                # running loss for logging
                running_losses_val["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5e}, mesh data: {mesh_data_num}")
                # ------------------------------------------------------------------------------------------------
            
                
                # ------------------------------------------------------------------------------------------------
                interv_val = round(len_valid_data / 5)
                if index % interv_val == 0:
                    # for visualization
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Valid] "
                    __jdx__ = 1/counter
                    for key, value in running_losses_val.items():
                        log_text += f"{key}: {value*__jdx__:.6e} "
                    self.logger.write(log_text+"\n")
                    
                    frame = HB
                    v_list = [
                        vertices[0].cpu().detach(),
                        vertices[1].cpu().detach(),
                        vertices[HB].cpu().detach(),
                        vertices[BS-1].cpu().detach(),
                        pred_vertices[0].cpu().detach(),
                        pred_vertices[1].cpu().detach(),
                        pred_vertices[HB].cpu().detach(),
                        pred_vertices[BS-1].cpu().detach(),
                    ]
                    len_v = len(v_list)
                    f_list=[faces] * len_v
                    save_logdir = f"{self.opts.log_dir}/img/valid/mesh"
                    save_img_name = f"{epoch:03d}_{counter:04d}"
                    
                    plot_image_array(
                        v_list, f_list, 
                        rot_list=[[0,0,0]]*len_v,
                        size=1, bg_black=False, mode='shade', 
                        logdir=save_logdir, 
                        name=save_img_name, save=True
                    )
                    
                # ------------------------------------------------------------------------------------------------
                if self.opts.debug:
                    break
            # log
            if self.opts.tb:
                self.log_loss(self.writer_valid, running_losses_val, epoch, counter)
            
            # best loss
            val_loss = running_losses_val["total"]/counter
            if val_loss < BEST_LOSS:
                BEST_LOSS = val_loss
                BEST_EPOCH = epoch
                print(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6e} - Best epoch: {BEST_EPOCH:03d}\n")
                self.logger.write(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6e}\n")
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_best.pth')
            else:
                self.logger.write(f"[{epoch:03d}/{epochs:03d}] Curr Loss: {val_loss:.6e} (Best Loss: {BEST_LOSS:.6e} [{BEST_EPOCH:03d}]\n")
                print(f"[{epoch:03d}/{epochs:03d}] Curr Loss: {val_loss:.6e} (Best Loss: {BEST_LOSS:.6e} [{BEST_EPOCH:03d}]\n")

    def train_v5(self, epochs):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.opts.lr,
            betas=(0.9, 0.999)
        )
        
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, 
        #     milestones=[i for i in range(0, epochs-1, self.opts.sc_step)], 
        #     gamma=self.opts.sc_gamma
        # )
        # dont use (make the stepe size 100000)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.opts.sc_step,
            gamma=self.opts.sc_gamma
        )
            
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        BS = self.opts.batch_size
        BS_denom = 1 / BS
        
        self.train_dataset = CBDDataset(
            self.opts, is_train=True, toggle=self.opts.data_toggle
        )
        self.valid_dataset = CBDDataset(
            self.opts, is_valid=True, toggle=self.opts.data_toggle
        )
        
        train_sampler = CBDdataSampler(
            self.train_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_train=True
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_sampler=train_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0,
        )        
            
        valid_sampler = CBDdataSampler(
            self.valid_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_valid=True
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset, 
            batch_sampler=valid_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0
        )
        ##########################################################################################################
        
        
        ###### Logging ###########################################################################################
        # make logdir --------------------------------------------------------------------------------------------
        os.makedirs(self.opts.log_dir, exist_ok=True)
        import datetime
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        
        tag = f"-NGBCv{self.opts.version}"
        if self.opts.optim_cage:
            tag += "-optim_cage"
        self.opts.log_dir = os.path.join(self.opts.log_dir, now+tag)
        os.makedirs(self.opts.log_dir, exist_ok=True)

        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/valid/mesh", exist_ok=True)
        
        # save options as json -----------------------------------------------------------------------------------
        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
            
        # save train option as yml
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), opts)
        
        if self.opts.tb:
            train_ = os.path.join(self.opts.log_dir, "train")
            valid_ = os.path.join(self.opts.log_dir, "valid")
            self.writer_train = SummaryWriter(log_dir=train_)
            self.writer_valid = SummaryWriter(log_dir=valid_)
        
        # self logger
        self.logger = Logger(os.path.join(self.opts.log_dir, "log.txt"))
        print(f'Saving log at: {self.logger.file_path}')
        
        print(self.train_dataset.get_data_config())
        print(train_sampler.get_sampler_config())
        print(self.valid_dataset.get_data_config())
        print(valid_sampler.get_sampler_config())
        
        self.logger.write(self.train_dataset.get_data_config())
        self.logger.write(train_sampler.get_sampler_config())  
        self.logger.write(self.valid_dataset.get_data_config())      
        self.logger.write(valid_sampler.get_sampler_config())
        #---------------------------------------------------------------------------------------------------------
        ##########################################################################################################
        
        
        
        # training loop ##########################################################################################
        global_step = 0
        BEST_LOSS = 100_000_000
        BEST_EPOCH = 0
        start_epoch = self.opts.start_epoch
                        
        # define loss lamdba 
        self.loss_lambda = {
            "recon-def": self.opts.lambda_vert,
            "recon-neu": self.opts.lambda_vert,
            "exp-z": self.opts.lambda_vert * 0.5,
            "exp-v": self.opts.lambda_vert,
            "shape": self.opts.lambda_vert,            
            # "pou": self.opts.lambda_vert,
            # symm 
        }
        if self.opts.pou_loss:
            self.loss_lambda['pou'] = 1.0
        if self.opts.use_laplacian:
            self.loss_lambda['lap'] = 1.0
        if self.opts.use_normal_loss:
            self.loss_lambda['norm-def']=0.1 # from NFR value (empirically set) - vtx and normal discrepency를 줄일려고 학습되지 않게, 노말은 좀 적게 람다값준것
            self.loss_lambda['norm-neu']=0.1
                
        len_train_data = len(self.train_dataloader)
        len_valid_data = len(self.valid_dataloader)
        interv_train = round(len_train_data / 10)
        
        for epoch in range(start_epoch, epochs+1):
            print(f"[{epoch:03d}/{epochs:03d}][Train]")
            
            ## for logging loss!
            running_losses = {
                "recon-def": 0.0,
                "recon-neu": 0.0,
                "exp-z": 0.0,
                "exp-v": 0.0,
                "shape": 0.0,
                "total": 0.0
            }
            if self.opts.pou_loss:
                running_losses['pou']=0.0
            if self.opts.use_laplacian:
                running_losses['lap']=0.0
            if self.opts.use_normal_loss:
                running_losses['norm-def']=0.0
                running_losses['norm-neu']=0.0
            
            self.model.train()
            train_counter = 0
            
            is_stepped = False
            is_stts_added = False
            
            pbar = tqdm(enumerate(self.train_dataloader), total=len_train_data, position=0, ncols=100)
            for index, batch in pbar:
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                    ## sampling points with probability
                    # margin = 0.8
                    # _p = (plateau_hat_points(batch.template[0]).squeeze() + margin) / (1 + margin)
                        
                    ## random sampling and random permutation
                    N = batch.template.shape[1]
                    if self.opts.use_perm:
                        use_perm = torch.rand(1) > 0.7
                    else:
                        use_perm= False
                        
                    if use_perm:
                        N_range = N-torch.randint(100, N//6, (1,)).item()
                        randperm_idx = torch.randperm(N)[:N_range]
                    else:
                        randperm_idx = torch.arange(N)
                        # randperm_idx = torch.multinomial(_p, 2048)
                    rearange_idx = torch.argsort(randperm_idx)
                    
                    batch_template_v = batch.template[:, randperm_idx]
                    batch_template_n = batch.template_normal[:, randperm_idx]
                    batch_vertices_v = batch.vertices[:, randperm_idx]
                    batch_vertices_n = batch.vertices_normal[:, randperm_idx]

                    ## masking face region using hat function (min x1 ~ max x2)
                    # t_mask = plateau_hat_points(batch_template_v) + 1.0
                    
                # model prediction -------------------------------------------------------------------------------
                ## B: number of batch, Nv : number of vertices, Nc: number of control vertices
                ## weight prediction: (B, Nv, Nc)
                ## key_d prediction:  (B, Nc, 3+3) [deformed cage]
                pred_vertices, recon_vertices, recon_source, exp_z, pred_source, t_mask, pred_key_weight = self.model(
                    batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                    batch.mesh_data, epoch=epoch
                )
                # ------------------------------------------------------------------------------------------------

                # import pdb;pdb.set_trace()
                # vis_mask_plot(batch_template_v[0].detach().cpu(), t_mask[0].detach().cpu(), logdir='./', name='test')
                # vis_mask_plot(batch_template_v[0].detach().cpu(), inv_t_mask[0].detach().cpu(), logdir='./', name='test')
                if self.opts.no_t_mask:
                    t_mask = 1.0
                    inv_t_mask = 0.0
                else:
                    inv_t_mask = 1.0 - t_mask

                
                # use segmentation for loss weight ------- ( not used ) ------------------------------------------
                ## -> re-weighting based on facial region area
                if self.opts.use_segment_weight:
                    with torch.no_grad():
                        # batch.segmentation # (B, Nv, 24)
                        segment_weight = batch.segmentation.sum(1) / N # batch.segmentation.sum(1).sum(1) # (B, 24)
                        batch_segment_weight = (batch.segmentation * segment_weight[:, None])[:, randperm_idx]
                        t_mask = t_mask * batch_segment_weight
                # ------------------------------------------------------------------------------------------------
                
                # loss -------------------------------------------------------------------------------------------
                mesh_data_num = batch.mesh_data.cpu().numpy()
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf', 'ict'])[mesh_data_num]
                
                loss_dict = {} # make it as a dictionary
                HB = batch.vertices.shape[0] // 2
                
                if self.opts.no_t_mask:
                    loss_dict['recon-def'] = F.mse_loss(
                        batch_vertices_v, pred_vertices
                    )
                else:
                    # focus deformation on face
                    loss_dict['recon-def'] = F.mse_loss(
                        batch_vertices_v*t_mask, pred_vertices*t_mask
                    )
                    # should be static on elsewhere
                    loss_dict['recon-def'] += F.mse_loss(
                        batch_template_v*inv_t_mask, pred_vertices*inv_t_mask
                    )

                # directly hanging mesh vertex position ----------------------------------------------------------
                if self.model.use_full_vertex:
                    if self.opts.no_t_mask:
                        loss_dict['recon-neu'] = F.mse_loss(
                            batch_template_v, pred_source
                        )
                    else:
                        loss_dict['recon-neu'] = F.mse_loss(
                            batch_template_v*t_mask, pred_source*t_mask
                        )
                        loss_dict['recon-neu'] += F.mse_loss(
                            batch_template_v*inv_t_mask, pred_source*inv_t_mask
                        )
                #-------------------------------------------------------------------------------------------------

                #-------( not used )------------------------------------------------------------------------------
                if self.model.use_shp_recon:
                    loss_dict['shape'] = F.mse_loss(
                        batch_template_v[:,rearange_idx]*t_mask,
                        recon_source[:,randperm_idx[rearange_idx]]*t_mask
                    ) # for shape AE
                if self.model.use_exp_recon:
                    loss_dict['exp-v'] = F.mse_loss(
                        batch_vertices_v[:,rearange_idx]*t_mask,
                        recon_vertices[:,randperm_idx[rearange_idx]]*t_mask
                    ) # for expression AE
                #-------------------------------------------------------------------------------------------------

                # soft POU ---------------------------------------------------------------------------------------
                if self.opts.pou_loss:
                    pred_key_weight_sum = pred_key_weight.sum(-1)
                    loss_dict['pou'] = F.mse_loss(
                        torch.ones_like(pred_key_weight_sum).to(self.device),
                        pred_key_weight_sum, 
                    )
                #-------------------------------------------------------------------------------------------------
                
                # Laplacian smoothing ----------------------------------------------------------------------------
                if not use_perm and self.opts.use_laplacian:
                    loss_dict['lap'] = laplacian_loss(
                        batch, pred_key_weight, self.train_dataset, mesh_data_num, self.device
                    ) * BS_denom
                    
                    # with torch.no_grad():
                    #     batch_vertices_lap = cotmatrix @ batch_vertices_v
                    #     batch_template_lap = cotmatrix @ batch_template_v
                    # pred_vertices_lap = cotmatrix @ pred_vertices
                    # pred_source_lap = cotmatrix @ pred_source
                    
                    # loss_dict['pois'] += F.mse_loss(batch_vertices_lap, pred_vertices_lap)
                    # loss_dict['pois'] += F.mse_loss(batch_vertices_lap, pred_vertices_lap)
                #-------------------------------------------------------------------------------------------------

                
                # vertex normal loss -----------------------------------------------------------------------------
                if not use_perm and self.opts.use_normal_loss:
                    pred_vertices_norm = calc_norm_torch(pred_vertices, batch.faces, at='verts') # [1, V, 3]
                    
                    if self.opts.no_t_mask:
                        loss_dict['norm-def'] = F.mse_loss(
                                batch_vertices_n, pred_vertices_norm
                            )
                    else:
                        loss_dict['norm-def'] = (
                            F.mse_loss(
                                batch_vertices_n*t_mask, pred_vertices_norm*t_mask
                            ) + F.mse_loss(
                                batch_template_n*inv_t_mask, pred_vertices_norm*inv_t_mask
                            )
                        )
                    
                    if self.model.use_full_vertex:
                        pred_template_norm = calc_norm_torch(pred_source, batch.faces, at='verts')   # [1, V, 3]
                        
                        loss_dict['norm-neu'] = F.mse_loss(
                            batch_template_n, pred_template_norm
                        )
                #-------------------------------------------------------------------------------------------------
                
                # latent alignment loss --------------------------------------------------------------------------
                #if (self.opts.use_data2 or self.opts.use_data3):
                if self.opts.align_latent:
                    if mesh_data=='ict':
                        loss_dict['exp-z'] = F.mse_loss(
                            batch.exp_coeff.unsqueeze(1), exp_z
                        )
                    else:
                        exp_z_facs, exp_z_ext = exp_z[...,:53], exp_z[...,53:]
                        loss_dict['exp-z'] = non_ict_loss(exp_z_facs)
                        loss_dict['exp-z'] += F.mse_loss(
                            torch.zeros_like(exp_z_ext).to(self.device),
                            exp_z_ext
                        )
                # ------------------------------------------------------------------------------------------------

                
                # get total loss (lambda weights are multiplied here!) -------------------------------------------
                loss = 0
                for key, value in loss_dict.items():
                    key_ = key.split("_")[0]
                    tmp = value*self.loss_lambda[key_]
                    loss += tmp
                    running_losses[key] += tmp # for logging
                loss_dict["total"] = loss
                # ------------------------------------------------------------------------------------------------

                
                # backward ---------------------------------------------------------------------------------------
                loss.backward()
                # try:
                #     loss.backward()
                # except:
                #     import pdb;pdb.set_trace()
                self.optimizer.step()
                # ------------------------------------------------------------------------------------------------

                
                # running loss for logging -----------------------------------------------------------------------
                running_losses["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5e}, mesh data: {mesh_data_num}")
                # ------------------------------------------------------------------------------------------------

                
                global_step += 1
                train_counter += 1
                                
                
                if index % interv_train == 1:

                    IDX = torch.tensor([0, 1, HB, BS-1])
                    with torch.no_grad():
                        #pred_vertices, recon_vertices, recon_source, exp_z, pred_source = self.model(
                        pred_vertices, _, _, exp_z, key_d, key_weight = self.model(
                            batch.template[IDX], batch.vertices[IDX],
                            batch.template_normal[IDX], batch.vertices_normal[IDX],
                            batch.mesh_data, epoch=epoch, out_kw=True
                        )
                    # for visualization
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                    
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Train] "
                    __idx__ = 1/train_counter
                    for key, value in running_losses.items():
                        log_text += f"{key}: {value*__idx__:.6e} "

                    if not is_stts_added:
                        log_text+='\n>>> Sum across vertex weights on each cage: '
                        log_text+=f'(max: {key_weight[0].sum(0).max().item():.5e}, min: {key_weight[0].sum(0).min().item():.5e})\n'
                        log_text+=f'>>> Num actually used cage vertex: {torch.count_nonzero(key_weight[0].sum(0))} / {key_weight.shape[-1]}'
                        is_stts_added=True
                    self.logger.write(log_text+"\n")
                    
                    frame = HB
                    v_list = [
                        vertices[0].cpu().detach(),
                        vertices[1].cpu().detach(),
                        vertices[HB].cpu().detach(),
                        vertices[BS-1].cpu().detach(),
                        pred_vertices[0].cpu().detach(),
                        pred_vertices[1].cpu().detach(),
                        pred_vertices[2].cpu().detach(),
                        pred_vertices[3].cpu().detach(),
                    ]
                    
                    len_v = len(v_list)
                    f_list = [faces] * len_v
                    save_logdir = f"{self.opts.log_dir}/img/train/mesh"
                    save_img_name = f"{epoch:03d}_{index:04d}"

                    plot_image_array(
                        v_list, f_list, 
                        rot_list=[[0,0,0]] * len_v, 
                        size=1, bg_black=False, mode='shade',
                        logdir=save_logdir,
                        name=save_img_name, save=True
                    )
                
                if self.opts.debug:
                    break
                # ------------------------------------------------------------------------------------------------
            
            ### scheduler ----------------------------------------------------------------------------------------
            if epoch != 0:
                self.scheduler.step()
                
                curr_lr = self.optimizer.param_groups[0]["lr"]
                if epoch % self.opts.sc_step==0 and not is_stepped:
                    log_notice = f'[{epoch:03d}/{epochs:03d}][{index:04d}][Train] scheduler stepped: {curr_lr:.6e}'
                    self.logger.write(log_notice+"\n")
                    is_stepped=True
            # ----------------------------------------------------------------------------------------------------
                
            # log
            if self.opts.tb:
                self.log_loss(self.writer_train, running_losses, epoch, train_counter)

            # save model
            if epoch % self.opts.save_interval == 0:
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_{epoch:03d}.pth')
            
            
            # validation -----------------------------------------------------------------------------------------
            self.model.eval()
            print(f"[{epoch:03d}/{epochs:03d}][Valid]")
            running_losses_val = {
                "recon-def": 0.0,
                "recon-neu": 0.0,
                "exp-z": 0.0,
                "exp-v": 0.0,
                "shape": 0.0,
                "total": 0.0
            }
            
            counter = 0
            pbar = tqdm(enumerate(self.valid_dataloader), total=len_valid_data, ncols=100)
            for index, batch in pbar:
                counter += 1
                
                # model validation -------------------------------------------------------------------------------
                with torch.no_grad():
                    pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _, _ = self.model(
                        batch.template, batch.vertices, 
                        batch.template_normal, batch.vertices_normal,
                        batch.mesh_data, epoch=epoch
                    )
                # ------------------------------------------------------------------------------------------------
                
                
                # loss ------------------------------------------------------------------------------------------- 
                with torch.no_grad():
                    mesh_data_num = batch.mesh_data.cpu().numpy()
                    mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf','ict'])[mesh_data_num]
                
                    loss_dict = {} # make it as a dictionary
                    HB = batch.vertices.shape[0] // 2
                    
                    loss_dict['recon-def'] = F.mse_loss(batch.vertices, pred_vertices) # for NGBC model
                    if self.model.use_full_vertex:
                        loss_dict['recon-neu'] = F.mse_loss(batch.template, pred_source)
                        
                    if self.model.use_shp_recon:
                        loss_dict['shape'] = F.mse_loss(batch.template, recon_source) # for shape AE
                    if self.model.use_exp_recon:
                        loss_dict['exp-v'] = F.mse_loss(batch.vertices, recon_vertices) # for expression AE
                    # loss_dict['exp-z'] = F.mse_loss(exp_z[:HB], exp_z[HB:])
                    
                    # get total loss
                    loss = 0
                    for key, value in loss_dict.items():
                        key_ = key.split("_")[0]
                        tmp = value.item()*self.loss_lambda[key_]
                        loss += tmp
                        running_losses_val[key] += tmp
                    loss_dict["total"] = loss 

                # running loss for logging
                running_losses_val["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5e}, mesh data: {mesh_data_num}")
                # ------------------------------------------------------------------------------------------------
            
                
                # ------------------------------------------------------------------------------------------------
                interv_val = round(len_valid_data / 5)
                if index % interv_val == 0:
                    # for visualization
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Valid] "
                    __jdx__ = 1/counter
                    for key, value in running_losses_val.items():
                        log_text += f"{key}: {value*__jdx__:.6e} "
                    self.logger.write(log_text+"\n")
                    
                    frame = HB
                    v_list = [
                        vertices[0].cpu().detach(),
                        vertices[1].cpu().detach(),
                        vertices[HB].cpu().detach(),
                        vertices[BS-1].cpu().detach(),
                        pred_vertices[0].cpu().detach(),
                        pred_vertices[1].cpu().detach(),
                        pred_vertices[HB].cpu().detach(),
                        pred_vertices[BS-1].cpu().detach(),
                    ]
                    len_v = len(v_list)
                    f_list=[faces] * len_v
                    save_logdir = f"{self.opts.log_dir}/img/valid/mesh"
                    save_img_name = f"{epoch:03d}_{counter:04d}"
                    
                    plot_image_array(
                        v_list, f_list, 
                        rot_list=[[0,0,0]]*len_v,
                        size=1, bg_black=False, mode='shade', 
                        logdir=save_logdir, 
                        name=save_img_name, save=True
                    )
                    
                # ------------------------------------------------------------------------------------------------
                if self.opts.debug:
                    break
            # log
            if self.opts.tb:
                self.log_loss(self.writer_valid, running_losses_val, epoch, counter)
            
            # best loss
            val_loss = running_losses_val["total"]/counter
            if val_loss < BEST_LOSS:
                BEST_LOSS = val_loss
                BEST_EPOCH = epoch
                print(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6e} - Best epoch: {BEST_EPOCH:03d}\n")
                self.logger.write(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6e}\n")
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_best.pth')
            else:
                self.logger.write(f"[{epoch:03d}/{epochs:03d}] Curr Loss: {val_loss:.6e} (Best Loss: {BEST_LOSS:.6e} [{BEST_EPOCH:03d}])\n")
                print(f"[{epoch:03d}/{epochs:03d}] Curr Loss: {val_loss:.6e} (Best Loss: {BEST_LOSS:.6e} [{BEST_EPOCH:03d}])\n")

    ## (NGBC LBS CBD 하이브리드 메인 학습 루프)
    def train_vLBSHybrid(self, epochs):
        
        def get_lbs_config(opts):
            text = "===========[LBS config]===========\n"
            text+= f"[    use_lbs_joint_center   ]: {opts.use_lbs_joint_center}\n"
            text+= f"   [      use_joint_predict    ]: {opts.use_joint_predict}\n"
            text+= f"       [   use_exp_joint_predict   ]: {opts.use_exp_joint_predict}\n"
            text+= f"   [     no_use_translation    ]: {opts.no_use_translation}\n"
            text+= f"   [   use_weighted_joint_pos  ]: {opts.use_weighted_joint_pos}\n"
            text+= "========== Regularizers ==========\n"
            text+= f"[         use_lbs_ent       ]: {opts.use_lbs_ent}\n"
            text+= f"[      use_lbs_laplacian    ]: {opts.use_lbs_laplacian}\n"
            text+= f"[          use_lbs_R        ]: {opts.use_lbs_R}\n"
            text+= f"[          use_lbs_t        ]: {opts.use_lbs_t}\n"
            text+= f"[         use_lbs_bal       ]: {opts.use_lbs_bal}\n"
            text+= "===============================+++\n"
            return text
        
        def set_stage(stage: int):
            """
            stage=1: train LBS(+exp_z) only, freeze CBD (key_weight/key_d)
            stage=2: train all (LBS + CBD) 
            """
            ## 1) default: freeze all
            for p in self.model.parameters():
                p.requires_grad_(False)
            
            # ## 1-1) if you use shape conditioning in exp_z model
            # if getattr(self.model, "use_shp", False):
            #     print("Successfully loaded shp models \n")        
            #     for p in self.model.shape_model.parameters():
            #         p.requires_grad_(True)

            ## 2) train LBS branch only at stage 1 
            if stage == 1:
                for m in [self.model.lbs_exp_z_model, self.model.lbs_weight_model, self.model.lbs_pose_model]:
                    for p in m.parameters():
                        p.requires_grad_(True)
                print("====================================\n")
                print("== Successfully loaded LBS branch ==\n")  
                print("====================================\n")

            ## 3) stage 2: unfreeze CBD branches too
            elif stage == 2:
                print("===================================================================\n")
                print("== Successfully loaded CBD branch and stop optimizing LBS branch ==\n")  
                print("===================================================================\n")
                for m in [self.model.cbd_exp_z_model, self.model.key_weight_model, self.model.key_d_model]: 
                    for p in m.parameters():
                        p.requires_grad_(True)
                
                # for m in [self.model.lbs_exp_z_model, self.model.lbs_weight_model, self.model.lbs_pose_model]:
                #     for p in m.parameters():
                #         p.requires_grad_(False)

        def rebuild_optimizer():
            params = []
            for p in self.model.parameters():
                if p.requires_grad:
                    params.append(p)
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.opts.lr,
                betas=(0.9, 0.999)
            )
        
        lbs_pretrained_epochs = self.opts.lbs_pretrained_epochs
        # decide inital stage robustly
        if self.opts.start_stage == 2 or self.opts.start_epoch >= lbs_pretrained_epochs:
            stage = 2
            self.model.load_CBD_brach(self.opts.use_lbs) # CBD branch init
            set_stage(stage=2)
            self.cbd_loaded = True
        else:
            stage = 1
            set_stage(stage=1)
            
        rebuild_optimizer()
        #########################################################################################
        
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, 
        #     milestones=[i for i in range(0, epochs-1, self.opts.sc_step)], 
        #     gamma=self.opts.sc_gamma
        # )

        self.opts.sc_step = 1000000 # turning this off
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.opts.sc_step,    
            gamma=self.opts.sc_gamma 
        )
            
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        BS = self.opts.batch_size
        BS_denom = 1 / BS
        
        self.train_dataset = CBDDataset(
            self.opts, 
            is_train=True, 
            toggle=self.opts.data_toggle
        )
        self.valid_dataset = CBDDataset(
            self.opts, 
            is_valid=True, 
            toggle=self.opts.data_toggle
        )
        
        train_sampler = CBDdataSampler(
            self.train_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_train=True
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_sampler=train_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0,
        )        
            
        valid_sampler = CBDdataSampler(
            self.valid_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_valid=True
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset, 
            batch_sampler=valid_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0
        )
        ##########################################################################################################
        
        ###### Logging ###########################################################################################
        
        
        # make logdir --------------------------------------------------------------------------------------------
        import datetime        
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        
        stage2_mode = (self.opts.start_stage == 2) or (self.opts.start_epoch >= self.opts.lbs_pretrained_epochs)
        resume_mode = (self.opts.ckpt is not None) and self.opts.continue_ckpt and (not stage2_mode)
        if resume_mode:
            os.makedirs(self.opts.log_dir, exist_ok=True)
            # if self.opts.ckpt and self.opts.log_dir:
            #     pass
            # else:
            #     # os.makedirs(self.opts.log_dir, exist_ok=True)                
            #     tag = f"-NGBC++v{self.opts.version}" # 1: baseline / 2: ours
            #     if self.opts.optim_cage:
            #         tag += "-optim_cage"
            #     self.opts.log_dir = os.path.join(self.opts.log_dir, now+tag)
            #     os.makedirs(self.opts.log_dir, exist_ok=True)
        else:
            # os.makedirs(self.opts.log_dir, exist_ok=True)
            tag = f"-NGBC++v{self.opts.version}" # 1: baseline / 2: ours
            if self.opts.optim_cage:
                tag += "-optim_cage"
            if stage2_mode:
                tag += "-stage2"
                if self.opts.ckpt:
                    tag += f"-from_lbs_ckpt_{self.opts.start_epoch}"
            self.opts.log_dir = os.path.join(self.opts.log_dir, now+tag)
            os.makedirs(self.opts.log_dir, exist_ok=True)

        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/valid/mesh", exist_ok=True)
            
        # save options as json -----------------------------------------------------------------------------------
        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
            
        # save train option as yml
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), opts)
        
        if self.opts.tb:
            train_ = os.path.join(self.opts.log_dir, "train")
            valid_ = os.path.join(self.opts.log_dir, "valid")
            self.writer_train = SummaryWriter(log_dir=train_)
            self.writer_valid = SummaryWriter(log_dir=valid_)
        
        # self logger
        self.logger = Logger(os.path.join(self.opts.log_dir, "log.txt"))
        print(f'Saving log at: {self.logger.file_path}')
        
        print(self.train_dataset.get_data_config())
        print(train_sampler.get_sampler_config())
        print(self.valid_dataset.get_data_config())
        print(valid_sampler.get_sampler_config())
        
        self.logger.write(self.train_dataset.get_data_config())
        self.logger.write(train_sampler.get_sampler_config())  
        self.logger.write(self.valid_dataset.get_data_config())      
        self.logger.write(valid_sampler.get_sampler_config())
        #---------------------------------------------------------------------------------------------------------
        
        ##############
        ## LBS logging
        ##############
        text = get_lbs_config(self.opts)
        print(text)
        self.logger.write(text)
        
        ##########################################################################################################
        # training loop -----------------------------------------------------------------------------------------
        global_step = 0
        BEST_LOSS = 100_000_000
        BEST_EPOCH = 0
        start_epoch = self.opts.start_epoch
        
        #########################
        ###### hard coded for now
        # best_epoch = 30
        # start_epoch = best_epoch + 1

        lbs_pretrained_epochs = self.opts.lbs_pretrained_epochs
                        
        # define loss lamdba 
        self.loss_lambda = {
            "recon-def": self.opts.lambda_vert,
            "recon-neu": self.opts.lambda_vert,
            "exp-z": self.opts.lambda_vert * 0.5,
            "exp-v": self.opts.lambda_vert,
            "shape": self.opts.lambda_vert,            
            # "pou": self.opts.lambda_vert,
            # symm 
        }
        if self.opts.pou_loss:
            self.loss_lambda['pou'] = 1.0
        if self.opts.use_laplacian:
            self.loss_lambda['lap'] = 1.0
        if self.opts.use_normal_loss:
            self.loss_lambda['norm-def']=0.1
            self.loss_lambda['norm-neu']=0.1
        
        ## lbs hybrid related flags 
        if self.opts.use_lbs_laplacian:
            self.loss_lambda['lbs-lap'] = 1e-2
        if self.opts.use_lbs_ent:
            self.loss_lambda['lbs-ent'] = 1e-3
        if self.opts.use_lbs_t:
            self.loss_lambda['lbs-t'] = 1e-2
        if self.opts.use_lbs_R:
            self.loss_lambda['lbs-R'] = 1e-3
        if self.opts.use_lbs_bal:
            self.loss_lambda['lbs-bal'] = 1e-4
        ##--------------------------------
        
        len_train_data = len(self.train_dataloader)
        len_valid_data = len(self.valid_dataloader)
        interv_train = round(len_train_data / 10)
        
        self.cbd_loaded = False   # train 시작 전에
        
        for epoch in range(start_epoch, epochs+1):
            print(f"[{epoch:03d}/{epochs:03d}][Train]")
            ## for logging loss!
            running_losses = {
                "recon-def": 0.0,
                "recon-neu": 0.0,
                "exp-z": 0.0,
                "exp-v": 0.0,
                "shape": 0.0,
                "total": 0.0
            }
            # import pdb;pdb.set_trace()
            if self.opts.debug_stage:
                if epoch < lbs_pretrained_epochs:
                    continue
                # import pdb;pdb.set_trace()

            ## LBS Hybrid 전용 stage 설정
            if stage == 1 and epoch == lbs_pretrained_epochs and (not self.cbd_loaded):
                stage = 2
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.model.load_CBD_brach(self.opts.use_lbs) # init models for cbd branch
                set_stage(stage=2) # load off lbs branch & load cbd branch
                rebuild_optimizer() # re-setup optimizer 
                self.cbd_loaded = True
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                ## check if using CBD losses (losses that are not applied to baseline)
                if self.opts.pou_loss:
                    running_losses['pou']=0.0
                if self.opts.use_laplacian:
                    running_losses['lap']=0.0
                if self.opts.use_normal_loss:
                    running_losses['norm-def']=0.0
                    running_losses['norm-neu']=0.0
                print(f"=========================================================================================\n")
                print(f"== LBS Pretraining done ({lbs_pretrained_epochs} epochs). Start to training CBD branch ==\n")
                print(f"=========================================================================================\n")

            if stage == 1:
                ## lbs losses initialization (to be applied later)
                if self.opts.use_lbs_laplacian:
                    running_losses['lbs-lap']=0.0
                if self.opts.use_lbs_ent:
                    running_losses['lbs-ent']=0.0
                if self.opts.use_lbs_t:
                    running_losses['lbs-t']=0.0
                if self.opts.use_lbs_R:
                    running_losses['lbs-R']=0.0
                if self.opts.use_lbs_bal:
                    running_losses['lbs-bal']=0.0  
            
            self.model.train()
            train_counter = 0
            
            is_stepped = False
            is_stts_added = False
            
            pbar = tqdm(enumerate(self.train_dataloader), total=len_train_data, position=0, ncols=100)
            for index, batch in pbar:
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                    ## sampling points with probability
                    # margin = 0.8
                    # _p = (plateau_hat_points(batch.template[0]).squeeze() + margin) / (1 + margin)
                        
                    ## random sampling and random permutation
                    N = batch.template.shape[1]
                    
                    if self.opts.use_perm:
                        use_perm = torch.rand(1) > 0.7
                    else:
                        use_perm= False
                    
                    if use_perm:
                        N_range = N-torch.randint(100, N//6, (1,)).item()
                        randperm_idx = torch.randperm(N)[:N_range]
                    else:
                        randperm_idx = torch.arange(N)
                        # randperm_idx = torch.multinomial(_p, 2048)
                    rearange_idx = torch.argsort(randperm_idx)
                    
                    batch_template_v = batch.template[:, randperm_idx]
                    batch_template_n = batch.template_normal[:, randperm_idx]
                    batch_vertices_v = batch.vertices[:, randperm_idx]
                    batch_vertices_n = batch.vertices_normal[:, randperm_idx]

                    ## masking face region using hat function (min x1 ~ max x2)
                    # t_mask = plateau_hat_points(batch_template_v) + 1.0
                    
                ## model prediction -------------------------------------------------------------------------------
                    ## B: number of batch, Nv : number of vertices, Nc: number of control vertices
                    ## weight prediction: (B, Nv, Nc)
                    ## key_d prediction:  (B, Nc, 3+3) [deformed cage]
                ## vLBS hybrid model prediction   ----------------------------------------------------------------
                
                if self.opts.vis_joint_pos:
                    pred_vertices, recon_vertices, recon_source, exp_z, pred_source, t_mask, key_d, pred_key_weight, W_lbs, T_lbs, _ = self.model(
                        batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                        batch.mesh_data, epoch=epoch, stage=stage
                    )
                else:
                    pred_vertices, recon_vertices, recon_source, exp_z, pred_source, t_mask, key_d, pred_key_weight, W_lbs, T_lbs = self.model(
                    batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                    batch.mesh_data, epoch=epoch, stage=stage
                )
                
                # ------------------------------------------------------------------------------------------------
                # vis_mask_plot(batch_template_v[0].detach().cpu(), t_mask[0].detach().cpu(), logdir='./', name='test')
                # vis_mask_plot(batch_template_v[0].detach().cpu(), inv_t_mask[0].detach().cpu(), logdir='./', name='test')
                if self.opts.no_t_mask:
                    t_mask = 1.0
                    inv_t_mask = 0.0
                else:
                    inv_t_mask = 1.0 - t_mask
                
                ## use segmentation for loss weight ------- ( not used ) ------------------------------------------
                ## -> re-weighting based on facial region area
                # if self.opts.use_segment_weight:
                #     with torch.no_grad():
                #         # batch.segmentation # (B, Nv, 24)
                #         segment_weight = batch.segmentation.sum(1) / N # batch.segmentation.sum(1).sum(1) # (B, 24)
                #         batch_segment_weight = (batch.segmentation * segment_weight[:, None])[:, randperm_idx]
                #         t_mask = t_mask * batch_segment_weight
                # ------------------------------------------------------------------------------------------------
                
                # loss -------------------------------------------------------------------------------------------
                mesh_data_num = batch.mesh_data.cpu().numpy()
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf', 'ict'])[mesh_data_num]
                
                loss_dict = {} # make it as a dictionary
                HB = batch.vertices.shape[0] // 2
                
                ## stage-agnostic loss terms
                if self.opts.no_t_mask:
                    loss_dict['recon-def'] = F.mse_loss(batch_vertices_v, pred_vertices)
                else:
                    loss_dict['recon-def'] = F.mse_loss(batch_vertices_v*t_mask, pred_vertices*t_mask) # focus deformation on face

                    loss_dict['recon-def'] += F.mse_loss(batch_template_v*inv_t_mask, pred_vertices*inv_t_mask) # should be static on elsewhere
                if self.model.use_full_vertex: # directly hanging mesh vertex position 
                    if self.opts.no_t_mask:
                        loss_dict['recon-neu'] = F.mse_loss(batch_template_v, pred_source)
                    else:
                        loss_dict['recon-neu'] = F.mse_loss(batch_template_v*t_mask, pred_source*t_mask)
                        loss_dict['recon-neu'] += F.mse_loss(batch_template_v*inv_t_mask, pred_source*inv_t_mask)

                ## lbs hybrid stage-specific loss terms
                if stage == 1: # stage 1 LBS pretraining
                    if self.opts.use_lbs_laplacian:
                        loss_dict['lbs-lap'] = lbs_laplacian_loss(batch, W_lbs, self.train_dataset, mesh_data_num, self.device) * BS_denom # following CBD laplacian loss
                    if self.opts.use_lbs_ent:
                        loss_dict['lbs-ent'] = lbs_entropy_loss(W_lbs) 
                    if self.opts.use_lbs_t or self.opts.use_lbs_R:
                        loss_dict['lbs-t'], loss_dict['lbs-R'] = lbs_transform_reg(T_lbs)
                    if self.opts.use_lbs_bal:
                        loss_dict['lbs-bal'] = lbs_usage_balance_loss(W_lbs)
                elif stage == 2: # stage 2 delta CBD training
                    #-------( not used )------------------------------------------------------------------------------
                    if self.model.use_shp_recon:
                        loss_dict['shape'] = F.mse_loss(
                            batch_template_v[:,rearange_idx]*t_mask,
                            recon_source[:,randperm_idx[rearange_idx]]*t_mask
                        ) # for shape AE
                    if self.model.use_exp_recon:
                        loss_dict['exp-v'] = F.mse_loss(
                            batch_vertices_v[:,rearange_idx]*t_mask,
                            recon_vertices[:,randperm_idx[rearange_idx]]*t_mask
                        ) # for expression AE
                    #-------------------------------------------------------------------------------------------------

                    # soft POU ---------------------------------------------------------------------------------------
                    if self.opts.pou_loss:
                        pred_key_weight_sum = pred_key_weight.sum(-1)
                        loss_dict['pou'] = F.mse_loss(
                            torch.ones_like(pred_key_weight_sum).to(self.device),
                            pred_key_weight_sum, 
                        )
                    #-------------------------------------------------------------------------------------------------
                    
                    # Laplacian smoothing ----------------------------------------------------------------------------
                    if not use_perm and self.opts.use_laplacian:
                        loss_dict['lap'] = laplacian_loss(
                            batch, pred_key_weight, self.train_dataset, mesh_data_num, self.device
                        ) * BS_denom
                        
                        # with torch.no_grad():
                        #     batch_vertices_lap = cotmatrix @ batch_vertices_v
                        #     batch_template_lap = cotmatrix @ batch_template_v
                        # pred_vertices_lap = cotmatrix @ pred_vertices
                        # pred_source_lap = cotmatrix @ pred_source
                        
                        # loss_dict['pois'] += F.mse_loss(batch_vertices_lap, pred_vertices_lap)
                        # loss_dict['pois'] += F.mse_loss(batch_vertices_lap, pred_vertices_lap)
                    #-------------------------------------------------------------------------------------------------

                    # vertex normal loss -----------------------------------------------------------------------------
                    if not use_perm and self.opts.use_normal_loss:
                        pred_vertices_norm = calc_norm_torch(pred_vertices, batch.faces, at='verts') # [1, V, 3]
                        
                        if self.opts.no_t_mask:
                            loss_dict['norm-def'] = F.mse_loss(
                                    batch_vertices_n, pred_vertices_norm
                                )
                        else:
                            loss_dict['norm-def'] = (
                                F.mse_loss(
                                    batch_vertices_n*t_mask, pred_vertices_norm*t_mask
                                ) + F.mse_loss(
                                    batch_template_n*inv_t_mask, pred_vertices_norm*inv_t_mask
                                )
                            )
                        
                        if self.model.use_full_vertex:
                            pred_template_norm = calc_norm_torch(pred_source, batch.faces, at='verts')   # [1, V, 3]
                            
                            loss_dict['norm-neu'] = F.mse_loss(
                                batch_template_n, pred_template_norm
                            )
                    #-------------------------------------------------------------------------------------------------
                    
                    # latent alignment loss --------------------------------------------------------------------------
                    #if (self.opts.use_data2 or self.opts.use_data3):
                    if self.opts.align_latent:
                        if mesh_data=='ict':
                            loss_dict['exp-z'] = F.mse_loss(
                                batch.exp_coeff.unsqueeze(1), exp_z
                            )
                        else:
                            exp_z_facs, exp_z_ext = exp_z[...,:53], exp_z[...,53:]
                            loss_dict['exp-z'] = non_ict_loss(exp_z_facs)
                            loss_dict['exp-z'] += F.mse_loss(
                                torch.zeros_like(exp_z_ext).to(self.device),
                                exp_z_ext
                            )
                    # ------------------------------------------------------------------------------------------------

                # get total loss (lambda weights are multiplied here!) -------------------------------------------
                loss = 0
                for key, value in loss_dict.items():
                    # key_ = key.split("_")[0]
                    tmp = value*self.loss_lambda[key]
                    loss += tmp
                    running_losses[key] += tmp # for logging
                loss_dict["total"] = loss
                # ------------------------------------------------------------------------------------------------

                # backward ---------------------------------------------------------------------------------------
                loss.backward()
                # try:
                #     loss.backward()
                # except:
                #     import pdb;pdb.set_trace()
                self.optimizer.step()
                # ------------------------------------------------------------------------------------------------

                # running loss for logging -----------------------------------------------------------------------
                running_losses["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5e}, mesh data: {mesh_data_num}")
                # ------------------------------------------------------------------------------------------------

                global_step += 1
                train_counter += 1
                                
                if index % interv_train == 1:

                    IDX = torch.tensor([0, 1, HB, BS-1])
                    with torch.no_grad():
                        if self.opts.vis_joint_pos:
                            #pred_vertices, recon_vertices, recon_source, exp_z, pred_source = self.model(
                            pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _, _, key_weight, W_lbs, T_lbs, C_lbs = self.model(
                                batch.template[IDX], batch.vertices[IDX],
                                batch.template_normal[IDX], batch.vertices_normal[IDX],
                                batch.mesh_data, epoch=epoch, out_kw=True, stage=stage
                            )
                        else: # default
                            pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _, _, key_weight, W_lbs, T_lbs = self.model(
                            batch.template[IDX], batch.vertices[IDX],
                            batch.template_normal[IDX], batch.vertices_normal[IDX],
                            batch.mesh_data, epoch=epoch, out_kw=True, stage=stage
                        )
                    
                    # for visualization
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                    
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Train] "
                    __idx__ = 1/train_counter
                    for key, value in running_losses.items():
                        log_text += f"{key}: {value*__idx__:.6e} "

                    if not is_stts_added:
                        log_text+='\n>>> Sum across vertex weights on each cage: '
                        log_text+=f'(max: {key_weight[0].sum(0).max().item():.5e}, min: {key_weight[0].sum(0).min().item():.5e})\n'
                        log_text+=f'>>> Num actually used cage vertex: {torch.count_nonzero(key_weight[0].sum(0))} / {key_weight.shape[-1]}'
                        is_stts_added=True
                    self.logger.write(log_text+"\n")
                    
                    frame = HB
                    v_list = [
                        vertices[0].cpu().detach(),
                        vertices[1].cpu().detach(),
                        vertices[HB].cpu().detach(),
                        vertices[BS-1].cpu().detach(),
                        pred_vertices[0].cpu().detach(),
                        pred_vertices[1].cpu().detach(),
                        pred_vertices[2].cpu().detach(),
                        pred_vertices[3].cpu().detach(),
                    ]
                    
                    len_v = len(v_list)
                    f_list = [faces] * len_v
                    save_logdir = f"{self.opts.log_dir}/img/train/mesh"
                    save_img_name = f"{epoch:03d}_{index:04d}"

                    if self.opts.vis_joint_pos:
                        C_lbs = C_lbs.detach().cpu() # keep torch ok (plot 내부에서 numpy로 바꿔도 됨)
                        C_list = [
                            None, None, None, None, # GT 에는 안찍기
                            C_lbs[0], C_lbs[1], C_lbs[2], C_lbs[3],
                        ]
                        plot_image_array_points(
                            v_list, 
                            f_list,
                            rot_list=[[0,0,0]] * len_v,
                            size=1, bg_black=False, mode='shade',
                            logdir=save_logdir,
                            name=save_img_name, save=True,
                            points_list=C_list,  
                            points_color='r',
                            points_size=4,
                        )
                    else:
                        plot_image_array(
                            v_list, f_list, 
                            rot_list=[[0,0,0]] * len_v, 
                            size=1, bg_black=False, mode='shade',
                            logdir=save_logdir,
                            name=save_img_name, save=True
                        )
                
                if self.opts.debug:
                    break
                # ------------------------------------------------------------------------------------------------
            
            ### scheduler ----------------------------------------------------------------------------------------
            if epoch != 0:
                self.scheduler.step()
                curr_lr = self.optimizer.param_groups[0]["lr"]
                if epoch % self.opts.sc_step==0 and not is_stepped:
                    log_notice = f'[{epoch:03d}/{epochs:03d}][{index:04d}][Train] scheduler stepped: {curr_lr:.6e}'
                    self.logger.write(log_notice+"\n")
                    is_stepped=True
            # ----------------------------------------------------------------------------------------------------
                
            # log
            if self.opts.tb:
                self.log_loss(self.writer_train, running_losses, epoch, train_counter)

            # save model
            if epoch % self.opts.save_interval == 0:
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_{epoch:03d}.pth')
            
            ######################################################################################################
            # validation -----------------------------------------------------------------------------------------
            self.model.eval()
            print(f"[{epoch:03d}/{epochs:03d}][Valid]")
            running_losses_val = {
                "recon-def": 0.0,
                "recon-neu": 0.0,
                "exp-z": 0.0,
                "exp-v": 0.0,
                "shape": 0.0,
                "total": 0.0
            }
            
            if self.opts.use_lbs_laplacian:
                running_losses_val['lbs-lap']=0.0
            if self.opts.use_lbs_ent:
                running_losses_val['lbs-ent']=0.0
            if self.opts.use_lbs_t:
                running_losses_val['lbs-t']=0.0
            if self.opts.use_lbs_R:
                running_losses_val['lbs-R']=0.0
            if self.opts.use_lbs_bal:
                running_losses_val['lbs-bal']=0.0
            
            counter = 0
            pbar = tqdm(enumerate(self.valid_dataloader), total=len_valid_data, ncols=100)
            for index, batch in pbar:
                counter += 1
                
                # model validation -------------------------------------------------------------------------------
                with torch.no_grad():
                    if self.opts.vis_joint_pos:
                        pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _, _, pred_key_weight, W_lbs, T_lbs, C_lbs = self.model(
                            batch.template, batch.vertices, 
                            batch.template_normal, batch.vertices_normal,
                            batch.mesh_data, epoch=epoch, stage=stage
                        )
                    else: # default
                        pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _, _, pred_key_weight, W_lbs, T_lbs = self.model(
                            batch.template, batch.vertices, 
                            batch.template_normal, batch.vertices_normal,
                            batch.mesh_data, epoch=epoch, stage=stage
                        )
                # ------------------------------------------------------------------------------------------------
                
                # loss ------------------------------------------------------------------------------------------- 
                with torch.no_grad():
                    mesh_data_num = batch.mesh_data.cpu().numpy()
                    mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf','ict'])[mesh_data_num]
                
                    loss_dict = {} # make it as a dictionary
                    HB = batch.vertices.shape[0] // 2
                    
                    loss_dict['recon-def'] = F.mse_loss(batch.vertices, pred_vertices) # for NGBC model
                    if self.model.use_full_vertex:
                        loss_dict['recon-neu'] = F.mse_loss(batch.template, pred_source) 
                        
                    if self.model.use_shp_recon:
                        loss_dict['shape'] = F.mse_loss(batch.template, recon_source) # for shape AE
                    if self.model.use_exp_recon:
                        loss_dict['exp-v'] = F.mse_loss(batch.vertices, recon_vertices) # for expression AE
                    # loss_dict['exp-z'] = F.mse_loss(exp_z[:HB], exp_z[HB:])
                    
                    if self.opts.use_lbs_laplacian:
                        loss_dict['lbs-lap'] = 0.0
                    if self.opts.use_lbs_ent:
                        loss_dict['lbs-ent'] = 0.0
                    if self.opts.use_lbs_t:
                        loss_dict['lbs-t'] = 0.0
                    if self.opts.use_lbs_R:
                        loss_dict['lbs-R'] = 0.0
                    if self.opts.use_lbs_bal:
                        loss_dict['lbs-bal'] = 0.0  
                    
                    # get total loss
                    loss = 0
                    for key, value in loss_dict.items():
                        key_ = key.split("_")[0]
                        tmp = value.item()*self.loss_lambda[key_]
                        loss += tmp
                        running_losses_val[key] += tmp
                    loss_dict["total"] = loss 

                # running loss for logging
                running_losses_val["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5e}, mesh data: {mesh_data_num}")
                # ------------------------------------------------------------------------------------------------
            
                # ------------------------------------------------------------------------------------------------
                interv_val = round(len_valid_data / 5)
                if index % interv_val == 0:
                    # for visualization
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Valid] "
                    __jdx__ = 1/counter
                    for key, value in running_losses_val.items():
                        log_text += f"{key}: {value*__jdx__:.6e} "
                    self.logger.write(log_text+"\n")
                    
                    frame = HB
                    v_list = [
                        vertices[0].cpu().detach(),
                        vertices[1].cpu().detach(),
                        vertices[HB].cpu().detach(),
                        vertices[BS-1].cpu().detach(),
                        pred_vertices[0].cpu().detach(),
                        pred_vertices[1].cpu().detach(),
                        pred_vertices[HB].cpu().detach(),
                        pred_vertices[BS-1].cpu().detach(),
                    ]
                    len_v = len(v_list)
                    f_list=[faces] * len_v
                    save_logdir = f"{self.opts.log_dir}/img/valid/mesh"
                    save_img_name = f"{epoch:03d}_{counter:04d}"
                    
                    if self.opts.vis_joint_pos:
                        C_lbs = C_lbs.detach().cpu() # keep torch ok (plot 내부에서 numpy로 바꿔도 됨)
                        C_list = [
                            None, None, None, None, # GT 에는 안찍기
                            C_lbs[0], C_lbs[1], C_lbs[HB], C_lbs[BS-1],
                        ]
                        plot_image_array_points(
                            v_list, 
                            f_list,
                            rot_list=[[0,0,0]] * len_v,
                            size=1, bg_black=False, mode='shade',
                            logdir=save_logdir,
                            name=save_img_name, save=True,
                            points_list=C_list,  
                            points_color='r',
                            points_size=4,
                        )
                    else:
                        plot_image_array(
                            v_list, f_list, 
                            rot_list=[[0,0,0]] * len_v, 
                            size=1, bg_black=False, mode='shade',
                            logdir=save_logdir,
                            name=save_img_name, save=True
                        )
                    
                # ------------------------------------------------------------------------------------------------
                if self.opts.debug:
                    break
            # log
            if self.opts.tb:
                self.log_loss(self.writer_valid, running_losses_val, epoch, counter)
            
            # best loss
            val_loss = running_losses_val["total"]/counter
            if val_loss < BEST_LOSS:
                BEST_LOSS = val_loss
                BEST_EPOCH = epoch
                print(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6e} - Best epoch: {BEST_EPOCH:03d}\n")
                self.logger.write(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6e}\n")
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_best.pth')
            else:
                self.logger.write(f"[{epoch:03d}/{epochs:03d}] Curr Loss: {val_loss:.6e} (Best Loss: {BEST_LOSS:.6e} [{BEST_EPOCH:03d}])\n")
                print(f"[{epoch:03d}/{epochs:03d}] Curr Loss: {val_loss:.6e} (Best Loss: {BEST_LOSS:.6e} [{BEST_EPOCH:03d}])\n")
    
    
    def train_vLBS(self, epochs):
        """ 
        train vLBS model 
        version should be 6
        """
        def get_lbs_config(opts):
            text = "===========[LBS config]===========\n"
            text+= f"[    use_lbs_joint_center   ]: {opts.use_lbs_joint_center}\n"
            text+= f"   [      use_joint_predict    ]: {opts.use_joint_predict}\n"
            text+= f"       [   use_exp_joint_predict   ]: {opts.use_exp_joint_predict}\n"
            text+= f"   [     no_use_translation    ]: {opts.no_use_translation}\n"
            text+= f"   [   use_weighted_joint_pos  ]: {opts.use_weighted_joint_pos}\n"
            text+= "========== Regularizers ==========\n"
            text+= f"[         use_lbs_ent       ]: {opts.use_lbs_ent}\n"
            text+= f"[      use_lbs_laplacian    ]: {opts.use_lbs_laplacian}\n"
            text+= f"[          use_lbs_R        ]: {opts.use_lbs_R}\n"
            text+= f"[          use_lbs_t        ]: {opts.use_lbs_t}\n"
            text+= f"[         use_lbs_bal       ]: {opts.use_lbs_bal}\n"
            text+= "===============================+++\n"
            return text
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.opts.lr,
            betas=(0.9, 0.999)
        )
        
        #########################################################################################
        
        self.opts.sc_step = 1000000 # turning this off
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.opts.sc_step,    
            gamma=self.opts.sc_gamma 
        )
            
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        BS = self.opts.batch_size
        BS_denom = 1 / BS
        
        self.train_dataset = CBDDataset(
            self.opts, 
            is_train=True, 
            toggle=self.opts.data_toggle
        )
        self.valid_dataset = CBDDataset(
            self.opts, 
            is_valid=True, 
            toggle=self.opts.data_toggle
        )
        
        train_sampler = CBDdataSampler(
            self.train_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_train=True
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_sampler=train_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0,
        )        
            
        valid_sampler = CBDdataSampler(
            self.valid_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_valid=True
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset, 
            batch_sampler=valid_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0
        )
        ##########################################################################################################
        
        ###### Logging ###########################################################################################
        
        
        # make logdir --------------------------------------------------------------------------------------------
        import datetime        
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        
        # os.makedirs(self.opts.log_dir, exist_ok=True)
        tag = f"-NGBC++v{self.opts.version}" # 1: baseline / 2: ours
        
        if self.opts.ckpt:
            tag += f"-from_lbs_ckpt_{self.opts.start_epoch}"
        
        self.opts.log_dir = os.path.join(self.opts.log_dir, now+tag)
        os.makedirs(self.opts.log_dir, exist_ok=True)

        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/valid/mesh", exist_ok=True)
            
        # save options as json -----------------------------------------------------------------------------------
        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
            
        # save train option as yml
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), opts)
        
        if self.opts.tb:
            train_ = os.path.join(self.opts.log_dir, "train")
            valid_ = os.path.join(self.opts.log_dir, "valid")
            self.writer_train = SummaryWriter(log_dir=train_)
            self.writer_valid = SummaryWriter(log_dir=valid_)
        
        # self logger
        self.logger = Logger(os.path.join(self.opts.log_dir, "log.txt"))
        print(f'Saving log at: {self.logger.file_path}')
        
        print(self.train_dataset.get_data_config())
        print(train_sampler.get_sampler_config())
        print(self.valid_dataset.get_data_config())
        print(valid_sampler.get_sampler_config())
        
        self.logger.write(self.train_dataset.get_data_config())
        self.logger.write(train_sampler.get_sampler_config())  
        self.logger.write(self.valid_dataset.get_data_config())      
        self.logger.write(valid_sampler.get_sampler_config())
        #---------------------------------------------------------------------------------------------------------
        
        ##############
        ## LBS logging
        ##############
        text = get_lbs_config(self.opts)
        print(text)
        self.logger.write(text)
        
        ##########################################################################################################
        # training loop -----------------------------------------------------------------------------------------
        global_step = 0
        BEST_LOSS = 100_000_000
        BEST_EPOCH = 0
        start_epoch = self.opts.start_epoch
        
        #########################
        ###### hard coded for now
        # best_epoch = 30
        # start_epoch = best_epoch + 1

        lbs_pretrained_epochs = self.opts.lbs_pretrained_epochs
                        
        # define loss lamdba 
        self.loss_lambda = {
            "recon-def": self.opts.lambda_vert,
            "recon-neu": self.opts.lambda_vert,
            "exp-z": self.opts.lambda_vert * 0.5,
            "exp-v": self.opts.lambda_vert,
            "shape": self.opts.lambda_vert,            
            # "pou": self.opts.lambda_vert,
            # symm 
        }
        if self.opts.pou_loss:
            self.loss_lambda['pou'] = 1.0
        if self.opts.use_laplacian:
            self.loss_lambda['lap'] = 1.0
        if self.opts.use_normal_loss:
            self.loss_lambda['norm-def']=0.1
            self.loss_lambda['norm-neu']=0.1
        
        ## lbs hybrid related flags 
        if self.opts.use_lbs_laplacian:
            self.loss_lambda['lbs-lap'] = 1e-2
        if self.opts.use_lbs_ent:
            self.loss_lambda['lbs-ent'] = 1e-3
        if self.opts.use_lbs_t:
            self.loss_lambda['lbs-t'] = 1e-2
        if self.opts.use_lbs_R:
            self.loss_lambda['lbs-R'] = 1e-3
        if self.opts.use_lbs_bal:
            self.loss_lambda['lbs-bal'] = 1e-4
        ##--------------------------------
        
        len_train_data = len(self.train_dataloader)
        len_valid_data = len(self.valid_dataloader)
        interv_train = round(len_train_data / 10)
        
        self.cbd_loaded = False   # train 시작 전에
        
        for epoch in range(start_epoch, epochs+1):
            print(f"[{epoch:03d}/{epochs:03d}][Train]")
            ## for logging loss!
            running_losses = {
                "recon-def": 0.0,
                "recon-neu": 0.0,
                "exp-z": 0.0,
                "exp-v": 0.0,
                "shape": 0.0,
                "total": 0.0
            }
            # import pdb;pdb.set_trace()
            if self.opts.debug_stage:
                if epoch < lbs_pretrained_epochs:
                    continue
                # import pdb;pdb.set_trace()

            ## lbs losses initialization (to be applied later)
            if self.opts.use_lbs_laplacian:
                running_losses['lbs-lap']=0.0
            if self.opts.use_lbs_ent:
                running_losses['lbs-ent']=0.0
            if self.opts.use_lbs_t:
                running_losses['lbs-t']=0.0
            if self.opts.use_lbs_R:
                running_losses['lbs-R']=0.0
            if self.opts.use_lbs_bal:
                running_losses['lbs-bal']=0.0  
        
            self.model.train()
            train_counter = 0
            
            is_stts_added = False
            
            pbar = tqdm(enumerate(self.train_dataloader), total=len_train_data, position=0, ncols=100)
            for index, batch in pbar:
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                    ## sampling points with probability
                    # margin = 0.8
                    # _p = (plateau_hat_points(batch.template[0]).squeeze() + margin) / (1 + margin)
                        
                    ## random sampling and random permutation
                    N = batch.template.shape[1]
                    
                    if self.opts.use_perm:
                        use_perm = torch.rand(1) > 0.7
                    else:
                        use_perm= False
                    
                    if use_perm:
                        N_range = N-torch.randint(100, N//6, (1,)).item()
                        randperm_idx = torch.randperm(N)[:N_range]
                    else:
                        randperm_idx = torch.arange(N)
                        # randperm_idx = torch.multinomial(_p, 2048)
                    rearange_idx = torch.argsort(randperm_idx)
                    
                    batch_template_v = batch.template[:, randperm_idx]
                    batch_template_n = batch.template_normal[:, randperm_idx]
                    batch_vertices_v = batch.vertices[:, randperm_idx]
                    batch_vertices_n = batch.vertices_normal[:, randperm_idx]

                    ## masking face region using hat function (min x1 ~ max x2)
                    # t_mask = plateau_hat_points(batch_template_v) + 1.0
                    
                ## model prediction -------------------------------------------------------------------------------
                    ## B: number of batch, Nv : number of vertices, Nc: number of control vertices
                    ## weight prediction: (B, Nv, Nc)
                    ## key_d prediction:  (B, Nc, 3+3) [deformed cage]
                ## vLBS hybrid model prediction   ----------------------------------------------------------------
                
                if self.opts.vis_joint_pos:
                    pred_vertices, recon_vertices, recon_source, exp_z, pred_source, t_mask, key_d, pred_key_weight, W_lbs, T_lbs, _ = self.model(
                        batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                        batch.mesh_data, epoch=epoch
                    )
                else:
                    pred_vertices, recon_vertices, recon_source, exp_z, pred_source, t_mask, key_d, pred_key_weight, W_lbs, T_lbs = self.model(
                    batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                    batch.mesh_data, epoch=epoch
                )
                
                # ------------------------------------------------------------------------------------------------
                # vis_mask_plot(batch_template_v[0].detach().cpu(), t_mask[0].detach().cpu(), logdir='./', name='test')
                # vis_mask_plot(batch_template_v[0].detach().cpu(), inv_t_mask[0].detach().cpu(), logdir='./', name='test')
                if self.opts.no_t_mask:
                    t_mask = 1.0
                    inv_t_mask = 0.0
                else:
                    inv_t_mask = 1.0 - t_mask
                
                ## use segmentation for loss weight ------- ( not used ) ------------------------------------------
                ## -> re-weighting based on facial region area
                # if self.opts.use_segment_weight:
                #     with torch.no_grad():
                #         # batch.segmentation # (B, Nv, 24)
                #         segment_weight = batch.segmentation.sum(1) / N # batch.segmentation.sum(1).sum(1) # (B, 24)
                #         batch_segment_weight = (batch.segmentation * segment_weight[:, None])[:, randperm_idx]
                #         t_mask = t_mask * batch_segment_weight
                # ------------------------------------------------------------------------------------------------
                
                # loss -------------------------------------------------------------------------------------------
                mesh_data_num = batch.mesh_data.cpu().numpy()
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf', 'ict'])[mesh_data_num]
                
                loss_dict = {} # make it as a dictionary
                HB = batch.vertices.shape[0] // 2
                
                ## stage-agnostic loss terms
                if self.opts.no_t_mask:
                    loss_dict['recon-def'] = F.mse_loss(batch_vertices_v, pred_vertices)
                else:
                    loss_dict['recon-def'] = F.mse_loss(batch_vertices_v*t_mask, pred_vertices*t_mask) # focus deformation on face

                    loss_dict['recon-def'] += F.mse_loss(batch_template_v*inv_t_mask, pred_vertices*inv_t_mask) # should be static on elsewhere
                if self.model.use_full_vertex: # directly hanging mesh vertex position 
                    if self.opts.no_t_mask:
                        loss_dict['recon-neu'] = F.mse_loss(batch_template_v, pred_source)
                    else:
                        loss_dict['recon-neu'] = F.mse_loss(batch_template_v*t_mask, pred_source*t_mask)
                        loss_dict['recon-neu'] += F.mse_loss(batch_template_v*inv_t_mask, pred_source*inv_t_mask)

                ## lbs hybrid stage-specific loss terms
                if self.opts.use_lbs_laplacian:
                    loss_dict['lbs-lap'] = lbs_laplacian_loss(batch, W_lbs, self.train_dataset, mesh_data_num, self.device) * BS_denom # following CBD laplacian loss
                if self.opts.use_lbs_ent:
                    loss_dict['lbs-ent'] = lbs_entropy_loss(W_lbs) 
                if self.opts.use_lbs_t or self.opts.use_lbs_R:
                    loss_dict['lbs-t'], loss_dict['lbs-R'] = lbs_transform_reg(T_lbs)
                if self.opts.use_lbs_bal:
                    loss_dict['lbs-bal'] = lbs_usage_balance_loss(W_lbs)
                    
                # elif stage == 2: # stage 2 delta CBD training
                #     #-------( not used )------------------------------------------------------------------------------
                #     if self.model.use_shp_recon:
                #         loss_dict['shape'] = F.mse_loss(
                #             batch_template_v[:,rearange_idx]*t_mask,
                #             recon_source[:,randperm_idx[rearange_idx]]*t_mask
                #         ) # for shape AE
                #     if self.model.use_exp_recon:
                #         loss_dict['exp-v'] = F.mse_loss(
                #             batch_vertices_v[:,rearange_idx]*t_mask,
                #             recon_vertices[:,randperm_idx[rearange_idx]]*t_mask
                #         ) # for expression AE
                #     #-------------------------------------------------------------------------------------------------

                #     # soft POU ---------------------------------------------------------------------------------------
                #     if self.opts.pou_loss:
                #         pred_key_weight_sum = pred_key_weight.sum(-1)
                #         loss_dict['pou'] = F.mse_loss(
                #             torch.ones_like(pred_key_weight_sum).to(self.device),
                #             pred_key_weight_sum, 
                #         )
                #     #-------------------------------------------------------------------------------------------------
                    
                #     # Laplacian smoothing ----------------------------------------------------------------------------
                #     if not use_perm and self.opts.use_laplacian:
                #         loss_dict['lap'] = laplacian_loss(
                #             batch, pred_key_weight, self.train_dataset, mesh_data_num, self.device
                #         ) * BS_denom
                        
                #         # with torch.no_grad():
                #         #     batch_vertices_lap = cotmatrix @ batch_vertices_v
                #         #     batch_template_lap = cotmatrix @ batch_template_v
                #         # pred_vertices_lap = cotmatrix @ pred_vertices
                #         # pred_source_lap = cotmatrix @ pred_source
                        
                #         # loss_dict['pois'] += F.mse_loss(batch_vertices_lap, pred_vertices_lap)
                #         # loss_dict['pois'] += F.mse_loss(batch_vertices_lap, pred_vertices_lap)
                #     #-------------------------------------------------------------------------------------------------

                # vertex normal loss -----------------------------------------------------------------------------
                if not use_perm and self.opts.use_normal_loss:
                    pred_vertices_norm = calc_norm_torch(pred_vertices, batch.faces, at='verts') # [1, V, 3]
                    
                    if self.opts.no_t_mask:
                        loss_dict['norm-def'] = F.mse_loss(
                                batch_vertices_n, pred_vertices_norm
                            )
                    else:
                        loss_dict['norm-def'] = (
                            F.mse_loss(
                                batch_vertices_n*t_mask, pred_vertices_norm*t_mask
                            ) + F.mse_loss(
                                batch_template_n*inv_t_mask, pred_vertices_norm*inv_t_mask
                            )
                        )
                    
                    if self.model.use_full_vertex:
                        pred_template_norm = calc_norm_torch(pred_source, batch.faces, at='verts')   # [1, V, 3]
                        
                        loss_dict['norm-neu'] = F.mse_loss(
                            batch_template_n, pred_template_norm
                        )
                #-------------------------------------------------------------------------------------------------
                
                # latent alignment loss --------------------------------------------------------------------------
                #if (self.opts.use_data2 or self.opts.use_data3):
                if self.opts.align_latent:
                    if mesh_data=='ict':
                        loss_dict['exp-z'] = F.mse_loss(
                            batch.exp_coeff.unsqueeze(1), exp_z
                        )
                    else:
                        exp_z_facs, exp_z_ext = exp_z[...,:53], exp_z[...,53:]
                        loss_dict['exp-z'] = non_ict_loss(exp_z_facs)
                        loss_dict['exp-z'] += F.mse_loss(
                            torch.zeros_like(exp_z_ext).to(self.device),
                            exp_z_ext
                        )
                    # ------------------------------------------------------------------------------------------------

                # get total loss (lambda weights are multiplied here!) -------------------------------------------
                loss = 0
                for key, value in loss_dict.items():
                    # key_ = key.split("_")[0]
                    tmp = value*self.loss_lambda[key]
                    loss += tmp
                    running_losses[key] += tmp # for logging
                loss_dict["total"] = loss
                # ------------------------------------------------------------------------------------------------

                # backward ---------------------------------------------------------------------------------------
                loss.backward()
                # try:
                #     loss.backward()
                # except:
                #     import pdb;pdb.set_trace()
                self.optimizer.step()
                # ------------------------------------------------------------------------------------------------

                # running loss for logging -----------------------------------------------------------------------
                running_losses["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5e}, mesh data: {mesh_data_num}")
                # ------------------------------------------------------------------------------------------------

                global_step += 1
                train_counter += 1
                                
                if index % interv_train == 1:

                    IDX = torch.tensor([0, 1, HB, BS-1])
                    with torch.no_grad():
                        if self.opts.vis_joint_pos:
                            #pred_vertices, recon_vertices, recon_source, exp_z, pred_source = self.model(
                            pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _, _, key_weight, W_lbs, T_lbs, C_lbs = self.model(
                                batch.template[IDX], batch.vertices[IDX],
                                batch.template_normal[IDX], batch.vertices_normal[IDX],
                                batch.mesh_data, epoch=epoch, out_kw=True
                            )
                        else: # default
                            pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _, _, key_weight, W_lbs, T_lbs = self.model(
                            batch.template[IDX], batch.vertices[IDX],
                            batch.template_normal[IDX], batch.vertices_normal[IDX],
                            batch.mesh_data, epoch=epoch, out_kw=True
                        )
                    
                    # for visualization
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                    
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Train] "
                    __idx__ = 1/train_counter
                    for key, value in running_losses.items():
                        log_text += f"{key}: {value*__idx__:.6e} "

                    if not is_stts_added:
                        log_text+='\n>>> Sum across vertex weights on each cage: '
                        log_text+=f'(max: {key_weight[0].sum(0).max().item():.5e}, min: {key_weight[0].sum(0).min().item():.5e})\n'
                        log_text+=f'>>> Num actually used cage vertex: {torch.count_nonzero(key_weight[0].sum(0))} / {key_weight.shape[-1]}'
                        is_stts_added=True
                    self.logger.write(log_text+"\n")
                    
                    frame = HB
                    v_list = [
                        vertices[0].cpu().detach(),
                        vertices[1].cpu().detach(),
                        vertices[HB].cpu().detach(),
                        vertices[BS-1].cpu().detach(),
                        pred_vertices[0].cpu().detach(),
                        pred_vertices[1].cpu().detach(),
                        pred_vertices[2].cpu().detach(),
                        pred_vertices[3].cpu().detach(),
                    ]
                    
                    len_v = len(v_list)
                    f_list = [faces] * len_v
                    save_logdir = f"{self.opts.log_dir}/img/train/mesh"
                    save_img_name = f"{epoch:03d}_{index:04d}"

                    if self.opts.vis_joint_pos:
                        C_lbs = C_lbs.detach().cpu() # keep torch ok (plot 내부에서 numpy로 바꿔도 됨)
                        C_list = [
                            None, None, None, None, # GT 에는 안찍기
                            C_lbs[0], C_lbs[1], C_lbs[2], C_lbs[3],
                        ]
                        plot_image_array_points(
                            v_list, 
                            f_list,
                            rot_list=[[0,0,0]] * len_v,
                            size=1, bg_black=False, mode='shade',
                            logdir=save_logdir,
                            name=save_img_name, save=True,
                            points_list=C_list,  
                            points_color='r',
                            points_size=4,
                        )
                    else:
                        plot_image_array(
                            v_list, f_list, 
                            rot_list=[[0,0,0]] * len_v, 
                            size=1, bg_black=False, mode='shade',
                            logdir=save_logdir,
                            name=save_img_name, save=True
                        )
                
                if self.opts.debug:
                    break
                # ------------------------------------------------------------------------------------------------
        
        
    def train_vLBSHybrid2(self, epochs):
        
        def get_lbs_config(opts):
            text = "===========[LBS config]===========\n"
            text+= f"[    use_lbs_joint_center   ]: {opts.use_lbs_joint_center}\n"
            text+= f"   [      use_joint_predict    ]: {opts.use_joint_predict}\n"
            text+= f"       [   use_exp_joint_predict   ]: {opts.use_exp_joint_predict}\n"
            text+= f"   [     no_use_translation    ]: {opts.no_use_translation}\n"
            text+= f"   [   use_weighted_joint_pos  ]: {opts.use_weighted_joint_pos}\n"
            text+= f"   [   use_hyb_joint_train  ]: {opts.use_hyb_joint_train}\n"
            text+= f"   [   use_hyb_concat_lbs  ]: {opts.use_hyb_concat_lbs}\n"
            text+= f"   [   use_hyb_delta_lbs_input  ]: {opts.use_hyb_delta_lbs_input}\n"
            text+= "========== Regularizers ==========\n"
            text+= f"[         use_lbs_ent       ]: {opts.use_lbs_ent}\n"
            text+= f"[      use_lbs_laplacian    ]: {opts.use_lbs_laplacian}\n"
            text+= f"[          use_lbs_R        ]: {opts.use_lbs_R}\n"
            text+= f"[          use_lbs_t        ]: {opts.use_lbs_t}\n"
            text+= f"[         use_lbs_bal       ]: {opts.use_lbs_bal}\n"
            text+= "===============================+++\n"
            return text

        def rebuild_optimizer(model):
            params = []
            for p in model.parameters():
                if p.requires_grad:
                    params.append(p)
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.opts.lr,
                betas=(0.9, 0.999)
            )
        
        lbs_pretrained_epochs = self.opts.lbs_pretrained_epochs
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        for p in self.model.parameters():
            p.requires_grad_(False)

        rebuild_optimizer(self.model_CBD)
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        #########################################################################################

        self.opts.sc_step = 1000000 # turning this off
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.opts.sc_step,    
            gamma=self.opts.sc_gamma 
        )
            
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        BS = self.opts.batch_size
        BS_denom = 1 / BS
        
        self.train_dataset = CBDDataset(
            self.opts, 
            is_train=True, 
            toggle=self.opts.data_toggle
        )
        self.valid_dataset = CBDDataset(
            self.opts, 
            is_valid=True, 
            toggle=self.opts.data_toggle
        )
        
        train_sampler = CBDdataSampler(
            self.train_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_train=True
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_sampler=train_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0,
        )        
            
        valid_sampler = CBDdataSampler(
            self.valid_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_valid=True
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset, 
            batch_sampler=valid_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0
        )
        ##########################################################################################################
        
        ###### Logging ###########################################################################################
        
        
        # make logdir --------------------------------------------------------------------------------------------
        import datetime        
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        
        stage2_mode = (self.opts.start_stage == 2) or (self.opts.start_epoch >= self.opts.lbs_pretrained_epochs)
        resume_mode = (self.opts.ckpt is not None) and self.opts.continue_ckpt and (not stage2_mode)
        if resume_mode:
            os.makedirs(self.opts.log_dir, exist_ok=True)
            # if self.opts.ckpt and self.opts.log_dir:
            #     pass
            # else:
            #     # os.makedirs(self.opts.log_dir, exist_ok=True)                
            #     tag = f"-NGBC++v{self.opts.version}" # 1: baseline / 2: ours
            #     if self.opts.optim_cage:
            #         tag += "-optim_cage"
            #     self.opts.log_dir = os.path.join(self.opts.log_dir, now+tag)
            #     os.makedirs(self.opts.log_dir, exist_ok=True)
        else:
            # os.makedirs(self.opts.log_dir, exist_ok=True)
            tag = f"-NGBC++v{self.opts.version}" # 1: baseline / 2: ours
            if self.opts.optim_cage:
                tag += "-optim_cage"
            if stage2_mode:
                tag += "-stage2"
                if self.opts.ckpt:
                    tag += f"-from_lbs_ckpt_{self.opts.start_epoch}"
            self.opts.log_dir = os.path.join(self.opts.log_dir, now+tag)
            os.makedirs(self.opts.log_dir, exist_ok=True)

        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/valid/mesh", exist_ok=True)
            
        # save options as json -----------------------------------------------------------------------------------
        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
            
        # save train option as yml
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), opts)
        
        if self.opts.tb:
            train_ = os.path.join(self.opts.log_dir, "train")
            valid_ = os.path.join(self.opts.log_dir, "valid")
            self.writer_train = SummaryWriter(log_dir=train_)
            self.writer_valid = SummaryWriter(log_dir=valid_)
        
        # self logger
        self.logger = Logger(os.path.join(self.opts.log_dir, "log.txt"))
        print(f'Saving log at: {self.logger.file_path}')
        
        print(self.train_dataset.get_data_config())
        print(train_sampler.get_sampler_config())
        print(self.valid_dataset.get_data_config())
        print(valid_sampler.get_sampler_config())
        
        print("LBS model:\n", self.model.get_model_config())
        self.logger.write(self.model.get_model_config())
        print("CBD model:\n", self.model_CBD.get_model_config())
        self.logger.write(self.model_CBD.get_model_config())
        
        self.logger.write(self.train_dataset.get_data_config())
        self.logger.write(train_sampler.get_sampler_config())  
        self.logger.write(self.valid_dataset.get_data_config())      
        self.logger.write(valid_sampler.get_sampler_config())
        #---------------------------------------------------------------------------------------------------------
        
        ##############
        ## LBS logging
        ##############
        text = get_lbs_config(self.opts)
        print(text)
        self.logger.write(text)
        
        ##########################################################################################################
        # training loop -----------------------------------------------------------------------------------------
        global_step = 0
        BEST_LOSS = 100_000_000
        BEST_EPOCH = 0
        start_epoch = self.opts.start_epoch
        
        #########################
        ###### hard coded for now
        # best_epoch = 30
        # start_epoch = best_epoch + 1

        lbs_pretrained_epochs = self.opts.lbs_pretrained_epochs
                        
        # define loss lamdba 
        self.loss_lambda = {
            "recon-def": self.opts.lambda_vert,
            "recon-neu": self.opts.lambda_vert,
            "exp-z": self.opts.lambda_vert * 0.5,
            "exp-v": self.opts.lambda_vert,
            "shape": self.opts.lambda_vert,            
            # "pou": self.opts.lambda_vert,
            # symm 
        }
        if self.opts.pou_loss:
            self.loss_lambda['pou'] = 1.0
        if self.opts.use_laplacian:
            self.loss_lambda['lap'] = 1.0
        if self.opts.use_normal_loss:
            self.loss_lambda['norm-def']=0.1
            self.loss_lambda['norm-neu']=0.1
        
        ## lbs hybrid related flags 
        if self.opts.use_lbs_laplacian:
            self.loss_lambda['lbs-lap'] = 1e-2
        if self.opts.use_lbs_ent:
            self.loss_lambda['lbs-ent'] = 1e-3
        if self.opts.use_lbs_t:
            self.loss_lambda['lbs-t'] = 1e-2
        if self.opts.use_lbs_R:
            self.loss_lambda['lbs-R'] = 1e-3
        if self.opts.use_lbs_bal:
            self.loss_lambda['lbs-bal'] = 1e-4
        ##--------------------------------
        
        len_train_data = len(self.train_dataloader)
        len_valid_data = len(self.valid_dataloader)
        interv_train = round(len_train_data / 10)
                
        for epoch in range(start_epoch, epochs+1):
            print(f"[{epoch:03d}/{epochs:03d}][Train]")
            ## for logging loss!
            running_losses = {
                "recon-def": 0.0,
                "recon-neu": 0.0,
                "exp-z": 0.0,
                "exp-v": 0.0,
                "shape": 0.0,
                "total": 0.0
            }
            # import pdb;pdb.set_trace()
            if self.opts.debug_stage:
                if epoch < lbs_pretrained_epochs:
                    continue

            ## check if using CBD losses (losses that are not applied to baseline)
            if self.opts.pou_loss:
                running_losses['pou']=0.0
            if self.opts.use_laplacian:
                running_losses['lap']=0.0
            if self.opts.use_normal_loss:
                running_losses['norm-def']=0.0
                running_losses['norm-neu']=0.0

            self.model.eval()
            self.model_CBD.train()
            train_counter = 0
            
            is_stepped = False
            is_stts_added = False
            
            pbar = tqdm(enumerate(self.train_dataloader), total=len_train_data, position=0, ncols=100)
            for index, batch in pbar:
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                    ## sampling points with probability
                    # margin = 0.8
                    # _p = (plateau_hat_points(batch.template[0]).squeeze() + margin) / (1 + margin)
                        
                    ## random sampling and random permutation
                    N = batch.template.shape[1]
                    
                    if self.opts.use_perm:
                        use_perm = torch.rand(1) > 0.7
                    else:
                        use_perm= False
                    
                    if use_perm:
                        N_range = N-torch.randint(100, N//6, (1,)).item()
                        randperm_idx = torch.randperm(N)[:N_range]
                    else:
                        randperm_idx = torch.arange(N)
                        # randperm_idx = torch.multinomial(_p, 2048)
                    rearange_idx = torch.argsort(randperm_idx)
                    
                    batch_template_v = batch.template[:, randperm_idx]
                    batch_template_n = batch.template_normal[:, randperm_idx]
                    batch_vertices_v = batch.vertices[:, randperm_idx]
                    batch_vertices_n = batch.vertices_normal[:, randperm_idx]

                    ## masking face region using hat function (min x1 ~ max x2)
                    # t_mask = plateau_hat_points(batch_template_v) + 1.0
                    
                ## model prediction -------------------------------------------------------------------------------
                    ## B: number of batch, Nv : number of vertices, Nc: number of control vertices
                    ## weight prediction: (B, Nv, Nc)
                    ## key_d prediction:  (B, Nc, 3+3) [deformed cage]
                ## vLBS hybrid model prediction   ----------------------------------------------------------------
                
                if self.opts.vis_joint_pos:
                    pred_vertices, recon_vertices, recon_source, exp_z, pred_source, t_mask, key_d, pred_key_weight, W_lbs, T_lbs, _ = self.model(
                        batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                        batch.mesh_data, epoch=epoch
                    )
                else:
                    pred_vertices, recon_vertices, recon_source, exp_z, pred_source, t_mask, key_d, pred_key_weight, W_lbs, T_lbs = self.model(
                        batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                        batch.mesh_data, epoch=epoch                   
                    )
                    
                    if self.opts.use_hyb_delta_lbs_input:
                        pred_vertices_CBD, recon_vertices_CBD, recon_source_CBD, exp_z_CBD, pred_source_CBD, t_mask_CBD, pred_key_weight_CBD = self.model_CBD(
                            batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                            batch.mesh_data, epoch=epoch, lbs_output = pred_vertices,
                        )
                    
                    elif self.opts.use_hyb_concat_lbs:
                        ## add pred_vertices to batch_template_v
                        pred_vertices_CBD, recon_vertices_CBD, recon_source_CBD, exp_z_CBD, pred_source_CBD, t_mask_CBD, pred_key_weight_CBD = self.model_CBD(
                            batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                            batch.mesh_data, epoch=epoch, lbs_output = pred_vertices, lbs_source = pred_source
                        )
                    
                    else: # default
                        pred_vertices_CBD, recon_vertices_CBD, recon_source_CBD, exp_z_CBD, pred_source_CBD, t_mask_CBD, pred_key_weight_CBD = self.model_CBD(
                            batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                            batch.mesh_data, epoch=epoch, 
                        )
                    pred_vertices = pred_vertices + pred_vertices_CBD # expressed face
                    pred_source = pred_source + pred_source_CBD # neutral face
                
                # ------------------------------------------------------------------------------------------------
                # vis_mask_plot(batch_template_v[0].detach().cpu(), t_mask[0].detach().cpu(), logdir='./', name='test')
                # vis_mask_plot(batch_template_v[0].detach().cpu(), inv_t_mask[0].detach().cpu(), logdir='./', name='test')
                if self.opts.no_t_mask:
                    t_mask = 1.0
                    inv_t_mask = 0.0
                else:
                    inv_t_mask = 1.0 - t_mask
                
                ## use segmentation for loss weight ------- ( not used ) ------------------------------------------
                ## -> re-weighting based on facial region area
                # if self.opts.use_segment_weight:
                #     with torch.no_grad():
                #         # batch.segmentation # (B, Nv, 24)
                #         segment_weight = batch.segmentation.sum(1) / N # batch.segmentation.sum(1).sum(1) # (B, 24)
                #         batch_segment_weight = (batch.segmentation * segment_weight[:, None])[:, randperm_idx]
                #         t_mask = t_mask * batch_segment_weight
                # ------------------------------------------------------------------------------------------------
                
                # loss -------------------------------------------------------------------------------------------
                mesh_data_num = batch.mesh_data.cpu().numpy()
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf', 'ict'])[mesh_data_num]
                
                loss_dict = {} # make it as a dictionary
                HB = batch.vertices.shape[0] // 2
                
                ## stage-agnostic loss terms
                if self.opts.no_t_mask:
                    loss_dict['recon-def'] = F.mse_loss(batch_vertices_v, pred_vertices)
                else:
                    loss_dict['recon-def'] = F.mse_loss(batch_vertices_v*t_mask, pred_vertices*t_mask) # focus deformation on face

                    loss_dict['recon-def'] += F.mse_loss(batch_template_v*inv_t_mask, pred_vertices*inv_t_mask) # should be static on elsewhere
                if self.model.use_full_vertex: # directly hanging mesh vertex position 
                    if self.opts.no_t_mask:
                        loss_dict['recon-neu'] = F.mse_loss(batch_template_v, pred_source)
                    else:
                        loss_dict['recon-neu'] = F.mse_loss(batch_template_v*t_mask, pred_source*t_mask)
                        loss_dict['recon-neu'] += F.mse_loss(batch_template_v*inv_t_mask, pred_source*inv_t_mask)

                
                
                #-------( not used )------------------------------------------------------------------------------
                if self.model.use_shp_recon:
                    loss_dict['shape'] = F.mse_loss(
                        batch_template_v[:,rearange_idx]*t_mask,
                        recon_source[:,randperm_idx[rearange_idx]]*t_mask
                    ) # for shape AE
                if self.model.use_exp_recon:
                    loss_dict['exp-v'] = F.mse_loss(
                        batch_vertices_v[:,rearange_idx]*t_mask,
                        recon_vertices[:,randperm_idx[rearange_idx]]*t_mask
                    ) # for expression AE
                #-------------------------------------------------------------------------------------------------

                # soft POU ---------------------------------------------------------------------------------------
                if self.opts.pou_loss:
                    pred_key_weight_sum = pred_key_weight.sum(-1)
                    loss_dict['pou'] = F.mse_loss(
                        torch.ones_like(pred_key_weight_sum).to(self.device),
                        pred_key_weight_sum, 
                    )
                #-------------------------------------------------------------------------------------------------
                
                # Laplacian smoothing ----------------------------------------------------------------------------
                if not use_perm and self.opts.use_laplacian:
                    loss_dict['lap'] = laplacian_loss(
                        batch, pred_key_weight, self.train_dataset, mesh_data_num, self.device
                    ) * BS_denom
                    
                    # with torch.no_grad():
                    #     batch_vertices_lap = cotmatrix @ batch_vertices_v
                    #     batch_template_lap = cotmatrix @ batch_template_v
                    # pred_vertices_lap = cotmatrix @ pred_vertices
                    # pred_source_lap = cotmatrix @ pred_source
                    
                    # loss_dict['pois'] += F.mse_loss(batch_vertices_lap, pred_vertices_lap)
                    # loss_dict['pois'] += F.mse_loss(batch_vertices_lap, pred_vertices_lap)
                #-------------------------------------------------------------------------------------------------

                # vertex normal loss -----------------------------------------------------------------------------
                if not use_perm and self.opts.use_normal_loss:
                    pred_vertices_norm = calc_norm_torch(pred_vertices, batch.faces, at='verts') # [1, V, 3]
                    
                    if self.opts.no_t_mask:
                        loss_dict['norm-def'] = F.mse_loss(
                                batch_vertices_n, pred_vertices_norm
                            )
                    else:
                        loss_dict['norm-def'] = (
                            F.mse_loss(
                                batch_vertices_n*t_mask, pred_vertices_norm*t_mask
                            ) + F.mse_loss(
                                batch_template_n*inv_t_mask, pred_vertices_norm*inv_t_mask
                            )
                        )
                    
                    if self.model.use_full_vertex:
                        pred_template_norm = calc_norm_torch(pred_source, batch.faces, at='verts')   # [1, V, 3]
                        
                        loss_dict['norm-neu'] = F.mse_loss(
                            batch_template_n, pred_template_norm
                        )
                #-------------------------------------------------------------------------------------------------
                
                # latent alignment loss --------------------------------------------------------------------------
                #if (self.opts.use_data2 or self.opts.use_data3):
                if self.opts.align_latent:
                    if mesh_data=='ict':
                        loss_dict['exp-z'] = F.mse_loss(
                            batch.exp_coeff.unsqueeze(1), exp_z
                        )
                    else:
                        exp_z_facs, exp_z_ext = exp_z[...,:53], exp_z[...,53:]
                        loss_dict['exp-z'] = non_ict_loss(exp_z_facs)
                        loss_dict['exp-z'] += F.mse_loss(
                            torch.zeros_like(exp_z_ext).to(self.device),
                            exp_z_ext
                        )
                    # ------------------------------------------------------------------------------------------------

                # get total loss (lambda weights are multiplied here!) -------------------------------------------
                loss = 0
                for key, value in loss_dict.items():
                    # key_ = key.split("_")[0]
                    tmp = value*self.loss_lambda[key]
                    loss += tmp
                    running_losses[key] += tmp # for logging
                loss_dict["total"] = loss
                # ------------------------------------------------------------------------------------------------

                # backward ---------------------------------------------------------------------------------------
                loss.backward()
                # try:
                #     loss.backward()
                # except:
                #     import pdb;pdb.set_trace()
                self.optimizer.step()
                # ------------------------------------------------------------------------------------------------

                # running loss for logging -----------------------------------------------------------------------
                running_losses["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5e}, mesh data: {mesh_data_num}")
                # ------------------------------------------------------------------------------------------------

                global_step += 1
                train_counter += 1
                                
                if index % interv_train == 1:

                    IDX = torch.tensor([0, 1, HB, BS-1])
                    with torch.no_grad():
                        if self.opts.vis_joint_pos:
                            #pred_vertices, recon_vertices, recon_source, exp_z, pred_source = self.model(
                            pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _, _, key_weight, W_lbs, T_lbs, C_lbs = self.model(
                                batch.template[IDX], batch.vertices[IDX],
                                batch.template_normal[IDX], batch.vertices_normal[IDX],
                                batch.mesh_data, epoch=epoch, out_kw=True,
                            )
                        else: # default
                            pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _, _, key_weight, W_lbs, T_lbs = self.model(
                            batch.template[IDX], batch.vertices[IDX],
                            batch.template_normal[IDX], batch.vertices_normal[IDX],
                            batch.mesh_data, epoch=epoch, out_kw=True,
                        )
                    
                    # for visualization
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                    
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Train] "
                    __idx__ = 1/train_counter
                    for key, value in running_losses.items():
                        log_text += f"{key}: {value*__idx__:.6e} "

                    if not is_stts_added:
                        log_text+='\n>>> Sum across vertex weights on each cage: '
                        log_text+=f'(max: {key_weight[0].sum(0).max().item():.5e}, min: {key_weight[0].sum(0).min().item():.5e})\n'
                        log_text+=f'>>> Num actually used cage vertex: {torch.count_nonzero(key_weight[0].sum(0))} / {key_weight.shape[-1]}'
                        is_stts_added=True
                    self.logger.write(log_text+"\n")
                    
                    frame = HB
                    v_list = [
                        vertices[0].cpu().detach(),
                        vertices[1].cpu().detach(),
                        vertices[HB].cpu().detach(),
                        vertices[BS-1].cpu().detach(),
                        pred_vertices[0].cpu().detach(),
                        pred_vertices[1].cpu().detach(),
                        pred_vertices[2].cpu().detach(),
                        pred_vertices[3].cpu().detach(),
                    ]
                    
                    len_v = len(v_list)
                    f_list = [faces] * len_v
                    save_logdir = f"{self.opts.log_dir}/img/train/mesh"
                    save_img_name = f"{epoch:03d}_{index:04d}"

                    if self.opts.vis_joint_pos:
                        C_lbs = C_lbs.detach().cpu() # keep torch ok (plot 내부에서 numpy로 바꿔도 됨)
                        C_list = [
                            None, None, None, None, # GT 에는 안찍기
                            C_lbs[0], C_lbs[1], C_lbs[2], C_lbs[3],
                        ]
                        plot_image_array_points(
                            v_list, 
                            f_list,
                            rot_list=[[0,0,0]] * len_v,
                            size=1, bg_black=False, mode='shade',
                            logdir=save_logdir,
                            name=save_img_name, save=True,
                            points_list=C_list,  
                            points_color='r',
                            points_size=4,
                        )
                    else:
                        plot_image_array(
                            v_list, f_list, 
                            rot_list=[[0,0,0]] * len_v, 
                            size=1, bg_black=False, mode='shade',
                            logdir=save_logdir,
                            name=save_img_name, save=True
                        )
                
                if self.opts.debug:
                    break
                # ------------------------------------------------------------------------------------------------
            
            ### scheduler ----------------------------------------------------------------------------------------
            if epoch != 0:
                self.scheduler.step()
                curr_lr = self.optimizer.param_groups[0]["lr"]
                if epoch % self.opts.sc_step==0 and not is_stepped:
                    log_notice = f'[{epoch:03d}/{epochs:03d}][{index:04d}][Train] scheduler stepped: {curr_lr:.6e}'
                    self.logger.write(log_notice+"\n")
                    is_stepped=True
            # ----------------------------------------------------------------------------------------------------
                
            # log
            if self.opts.tb:
                self.log_loss(self.writer_train, running_losses, epoch, train_counter)

            # save model
            if epoch % self.opts.save_interval == 0:
                # torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_{epoch:03d}.pth') # save LBS
                torch.save(self.model_CBD.state_dict(), f'{self.opts.log_dir}/model_{epoch:03d}.pth') # save CBD
                
            
            ######################################################################################################
            # validation -----------------------------------------------------------------------------------------
            self.model.eval()
            print(f"[{epoch:03d}/{epochs:03d}][Valid]")
            running_losses_val = {
                "recon-def": 0.0,
                "recon-neu": 0.0,
                "exp-z": 0.0,
                "exp-v": 0.0,
                "shape": 0.0,
                "total": 0.0
            }
            
            if self.opts.use_lbs_laplacian:
                running_losses_val['lbs-lap']=0.0
            if self.opts.use_lbs_ent:
                running_losses_val['lbs-ent']=0.0
            if self.opts.use_lbs_t:
                running_losses_val['lbs-t']=0.0
            if self.opts.use_lbs_R:
                running_losses_val['lbs-R']=0.0
            if self.opts.use_lbs_bal:
                running_losses_val['lbs-bal']=0.0
            
            counter = 0
            pbar = tqdm(enumerate(self.valid_dataloader), total=len_valid_data, ncols=100)
            for index, batch in pbar:
                counter += 1
                
                # model validation -------------------------------------------------------------------------------
                with torch.no_grad():
                    if self.opts.vis_joint_pos:
                        pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _, _, pred_key_weight, W_lbs, T_lbs, C_lbs = self.model(
                            batch.template, batch.vertices, 
                            batch.template_normal, batch.vertices_normal,
                            batch.mesh_data, epoch=epoch
                        )
                    else: # default
                        pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _, _, pred_key_weight, W_lbs, T_lbs = self.model(
                            batch.template, batch.vertices, 
                            batch.template_normal, batch.vertices_normal,
                            batch.mesh_data, epoch=epoch
                        )
                # ------------------------------------------------------------------------------------------------
                
                # loss ------------------------------------------------------------------------------------------- 
                with torch.no_grad():
                    mesh_data_num = batch.mesh_data.cpu().numpy()
                    mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf','ict'])[mesh_data_num]
                
                    loss_dict = {} # make it as a dictionary
                    HB = batch.vertices.shape[0] // 2
                    
                    loss_dict['recon-def'] = F.mse_loss(batch.vertices, pred_vertices) # for NGBC model
                    if self.model.use_full_vertex:
                        loss_dict['recon-neu'] = F.mse_loss(batch.template, pred_source) 
                        
                    if self.model.use_shp_recon:
                        loss_dict['shape'] = F.mse_loss(batch.template, recon_source) # for shape AE
                    if self.model.use_exp_recon:
                        loss_dict['exp-v'] = F.mse_loss(batch.vertices, recon_vertices) # for expression AE
                    # loss_dict['exp-z'] = F.mse_loss(exp_z[:HB], exp_z[HB:])
                    
                    if self.opts.use_lbs_laplacian:
                        loss_dict['lbs-lap'] = 0.0
                    if self.opts.use_lbs_ent:
                        loss_dict['lbs-ent'] = 0.0
                    if self.opts.use_lbs_t:
                        loss_dict['lbs-t'] = 0.0
                    if self.opts.use_lbs_R:
                        loss_dict['lbs-R'] = 0.0
                    if self.opts.use_lbs_bal:
                        loss_dict['lbs-bal'] = 0.0  
                    
                    # get total loss
                    loss = 0
                    for key, value in loss_dict.items():
                        key_ = key.split("_")[0]
                        tmp = value.item()*self.loss_lambda[key_]
                        loss += tmp
                        running_losses_val[key] += tmp
                    loss_dict["total"] = loss 

                # running loss for logging
                running_losses_val["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5e}, mesh data: {mesh_data_num}")
                # ------------------------------------------------------------------------------------------------
            
                # ------------------------------------------------------------------------------------------------
                interv_val = round(len_valid_data / 5)
                if index % interv_val == 0:
                    # for visualization
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Valid] "
                    __jdx__ = 1/counter
                    for key, value in running_losses_val.items():
                        log_text += f"{key}: {value*__jdx__:.6e} "
                    self.logger.write(log_text+"\n")
                    
                    frame = HB
                    v_list = [
                        vertices[0].cpu().detach(),
                        vertices[1].cpu().detach(),
                        vertices[HB].cpu().detach(),
                        vertices[BS-1].cpu().detach(),
                        pred_vertices[0].cpu().detach(),
                        pred_vertices[1].cpu().detach(),
                        pred_vertices[HB].cpu().detach(),
                        pred_vertices[BS-1].cpu().detach(),
                    ]
                    len_v = len(v_list)
                    f_list=[faces] * len_v
                    save_logdir = f"{self.opts.log_dir}/img/valid/mesh"
                    save_img_name = f"{epoch:03d}_{counter:04d}"
                    
                    if self.opts.vis_joint_pos:
                        C_lbs = C_lbs.detach().cpu() # keep torch ok (plot 내부에서 numpy로 바꿔도 됨)
                        C_list = [
                            None, None, None, None, # GT 에는 안찍기
                            C_lbs[0], C_lbs[1], C_lbs[HB], C_lbs[BS-1],
                        ]
                        plot_image_array_points(
                            v_list, 
                            f_list,
                            rot_list=[[0,0,0]] * len_v,
                            size=1, bg_black=False, mode='shade',
                            logdir=save_logdir,
                            name=save_img_name, save=True,
                            points_list=C_list,  
                            points_color='r',
                            points_size=4,
                        )
                    else:
                        plot_image_array(
                            v_list, f_list, 
                            rot_list=[[0,0,0]] * len_v, 
                            size=1, bg_black=False, mode='shade',
                            logdir=save_logdir,
                            name=save_img_name, save=True
                        )
                    
                # ------------------------------------------------------------------------------------------------
                if self.opts.debug:
                    break
            # log
            if self.opts.tb:
                self.log_loss(self.writer_valid, running_losses_val, epoch, counter)
            
            # best loss
            val_loss = running_losses_val["total"]/counter
            if val_loss < BEST_LOSS:
                BEST_LOSS = val_loss
                BEST_EPOCH = epoch
                print(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6e} - Best epoch: {BEST_EPOCH:03d}\n")
                self.logger.write(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6e}\n")
                torch.save(self.model_CBD.state_dict(), f'{self.opts.log_dir}/model_best.pth') # save model CBD 
            else:
                self.logger.write(f"[{epoch:03d}/{epochs:03d}] Curr Loss: {val_loss:.6e} (Best Loss: {BEST_LOSS:.6e} [{BEST_EPOCH:03d}])\n")
                print(f"[{epoch:03d}/{epochs:03d}] Curr Loss: {val_loss:.6e} (Best Loss: {BEST_LOSS:.6e} [{BEST_EPOCH:03d}])\n")
    
    
    def train_vLBSHybrid3(self, epochs):

        """
        
        joint training of LBS + CBD
        
        """
        
        def get_lbs_config(opts):
            text = "===========[LBS config]===========\n"
            text+= f"[    use_lbs_joint_center   ]: {opts.use_lbs_joint_center}\n"
            text+= f"   [      use_joint_predict    ]: {opts.use_joint_predict}\n"
            text+= f"       [   use_exp_joint_predict   ]: {opts.use_exp_joint_predict}\n"
            text+= f"   [     no_use_translation    ]: {opts.no_use_translation}\n"
            text+= f"   [   use_weighted_joint_pos  ]: {opts.use_weighted_joint_pos}\n"
            text+= f"   [   use_hyb_joint_train  ]: {opts.use_hyb_joint_train}\n"
            text+= f"   [   use_hyb_concat_lbs  ]: {opts.use_hyb_concat_lbs}\n"
            text+= f"   [   use_hyb_delta_lbs_input  ]: {opts.use_hyb_delta_lbs_input}\n"
            text+= "========== Regularizers ==========\n"
            text+= f"[         use_lbs_ent       ]: {opts.use_lbs_ent}\n"
            text+= f"[      use_lbs_laplacian    ]: {opts.use_lbs_laplacian}\n"
            text+= f"[          use_lbs_R        ]: {opts.use_lbs_R}\n"
            text+= f"[          use_lbs_t        ]: {opts.use_lbs_t}\n"
            text+= f"[         use_lbs_bal       ]: {opts.use_lbs_bal}\n"
            text+= "===============================+++\n"
            return text

        def rebuild_optimizer_model_list(models : list):
            params = []
            for model in models:
                for p in model.parameters():
                    if p.requires_grad:
                        params.append(p)
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.opts.lr,
                betas=(0.9, 0.999)
            )
        
        lbs_pretrained_epochs = self.opts.lbs_pretrained_epochs
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # for p in self.model.parameters():
        #     p.requires_grad_(False)
        ## joint training: enable all parameters
        # rebuild_optimizer(self.model)
        # rebuild_optimizer(self.model_CBD)
        rebuild_optimizer_model_list(models=[self.model, self.model_CBD])
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        #########################################################################################

        self.opts.sc_step = 1000000 # turning this off
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.opts.sc_step,    
            gamma=self.opts.sc_gamma 
        )
            
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        BS = self.opts.batch_size
        BS_denom = 1 / BS
        
        self.train_dataset = CBDDataset(
            self.opts, 
            is_train=True, 
            toggle=self.opts.data_toggle
        )
        self.valid_dataset = CBDDataset(
            self.opts, 
            is_valid=True, 
            toggle=self.opts.data_toggle
        )
        
        train_sampler = CBDdataSampler(
            self.train_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_train=True
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_sampler=train_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0,
        )        
            
        valid_sampler = CBDdataSampler(
            self.valid_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_valid=True
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset, 
            batch_sampler=valid_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0
        )
        ##########################################################################################################
        
        ###### Logging ###########################################################################################
        ## make logdir --------------------------------------------------------------------------------------------
        import datetime        
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        
        ## no stage
        # stage2_mode = (self.opts.start_stage == 2) or (self.opts.start_epoch >= self.opts.lbs_pretrained_epochs)
        resume_mode = (self.opts.ckpt is not None) and ((self.opts.log_dir is not None))
        if resume_mode:
            os.makedirs(self.opts.log_dir, exist_ok=True)
            # if self.opts.ckpt and self.opts.log_dir:
            #     pass
            # else:
            #     # os.makedirs(self.opts.log_dir, exist_ok=True)                
            #     tag = f"-NGBC++v{self.opts.version}" # 1: baseline / 2: ours
            #     if self.opts.optim_cage:
            #         tag += "-optim_cage"
            #     self.opts.log_dir = os.path.join(self.opts.log_dir, now+tag)
            #     os.makedirs(self.opts.log_dir, exist_ok=True)
        else:
            # os.makedirs(self.opts.log_dir, exist_ok=True)
            tag = f"-NGBC++v{self.opts.version}" # 1: baseline / 2: ours
            self.opts.log_dir = os.path.join(self.opts.log_dir, now+tag)
            os.makedirs(self.opts.log_dir, exist_ok=True)

        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/valid/mesh", exist_ok=True)
            
        # save options as json -----------------------------------------------------------------------------------
        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
            
        # save train option as yml
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), opts)
        
        if self.opts.tb:
            train_ = os.path.join(self.opts.log_dir, "train")
            valid_ = os.path.join(self.opts.log_dir, "valid")
            self.writer_train = SummaryWriter(log_dir=train_)
            self.writer_valid = SummaryWriter(log_dir=valid_)
        
        # self logger
        self.logger = Logger(os.path.join(self.opts.log_dir, "log.txt"))
        print(f'Saving log at: {self.logger.file_path}')
        
        print(self.train_dataset.get_data_config())
        print(train_sampler.get_sampler_config())
        print(self.valid_dataset.get_data_config())
        print(valid_sampler.get_sampler_config())
        
        print("LBS model:\n", self.model.get_model_config())
        self.logger.write(self.model.get_model_config())
        print("CBD model:\n", self.model_CBD.get_model_config())
        self.logger.write(self.model_CBD.get_model_config())
        
        self.logger.write(self.train_dataset.get_data_config())
        self.logger.write(train_sampler.get_sampler_config())  
        self.logger.write(self.valid_dataset.get_data_config())      
        self.logger.write(valid_sampler.get_sampler_config())
        #---------------------------------------------------------------------------------------------------------
        
        ##############
        ## LBS logging
        ##############
        text = get_lbs_config(self.opts)
        print(text)
        self.logger.write(text)
        
        ##########################################################################################################
        # training loop -----------------------------------------------------------------------------------------
        global_step = 0
        BEST_LOSS = 100_000_000
        BEST_EPOCH = 0
        start_epoch = self.opts.start_epoch
        
        #########################
        ###### hard coded for now
        # best_epoch = 30
        # start_epoch = best_epoch + 1
                        
        # define loss lamdba 
        self.loss_lambda = {
            "recon-def": self.opts.lambda_vert,
            "recon-neu": self.opts.lambda_vert,
            "exp-z": self.opts.lambda_vert * 0.5,
            "exp-v": self.opts.lambda_vert,
            "shape": self.opts.lambda_vert,            
            # "pou": self.opts.lambda_vert,
            # symm 
        }
        if self.opts.pou_loss:
            self.loss_lambda['pou'] = 1.0
        if self.opts.use_laplacian:
            self.loss_lambda['lap'] = 1.0
        if self.opts.use_normal_loss:
            self.loss_lambda['norm-def']=0.1
            self.loss_lambda['norm-neu']=0.1
        
        ## lbs hybrid related flags 
        if self.opts.use_lbs_laplacian:
            self.loss_lambda['lbs-lap'] = 1e-2
        if self.opts.use_lbs_ent:
            self.loss_lambda['lbs-ent'] = 1e-3
        if self.opts.use_lbs_t:
            self.loss_lambda['lbs-t'] = 1e-2
        if self.opts.use_lbs_R:
            self.loss_lambda['lbs-R'] = 1e-3
        if self.opts.use_lbs_bal:
            self.loss_lambda['lbs-bal'] = 1e-4
        ##--------------------------------
        
        len_train_data = len(self.train_dataloader)
        len_valid_data = len(self.valid_dataloader)
        interv_train = round(len_train_data / 10)
                
        for epoch in range(start_epoch, epochs+1):
            print(f"[{epoch:03d}/{epochs:03d}][Train]")
            ## for logging loss!
            running_losses = {
                "recon-def": 0.0,
                "recon-neu": 0.0,
                "exp-z": 0.0,
                "exp-v": 0.0,
                "shape": 0.0,
                "total": 0.0
            }
            # import pdb;pdb.set_trace()
            if self.opts.debug_stage:
                if epoch < lbs_pretrained_epochs:
                    continue

            ## check if using CBD losses (losses that are not applied to baseline)
            if self.opts.pou_loss:
                running_losses['pou']=0.0
            if self.opts.use_laplacian:
                running_losses['lap']=0.0
            if self.opts.use_normal_loss:
                running_losses['norm-def']=0.0
                running_losses['norm-neu']=0.0

            self.model.train()
            self.model_CBD.train()
            train_counter = 0
            
            is_stepped = False
            is_stts_added = False
            
            pbar = tqdm(enumerate(self.train_dataloader), total=len_train_data, position=0, ncols=100)
            for index, batch in pbar:
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                    ## sampling points with probability
                    # margin = 0.8
                    # _p = (plateau_hat_points(batch.template[0]).squeeze() + margin) / (1 + margin)
                        
                    ## random sampling and random permutation
                    N = batch.template.shape[1]
                    
                    if self.opts.use_perm:
                        use_perm = torch.rand(1) > 0.7
                    else:
                        use_perm= False
                    
                    if use_perm:
                        N_range = N-torch.randint(100, N//6, (1,)).item()
                        randperm_idx = torch.randperm(N)[:N_range]
                    else:
                        randperm_idx = torch.arange(N)
                        # randperm_idx = torch.multinomial(_p, 2048)
                    rearange_idx = torch.argsort(randperm_idx)
                    
                    batch_template_v = batch.template[:, randperm_idx]
                    batch_template_n = batch.template_normal[:, randperm_idx]
                    batch_vertices_v = batch.vertices[:, randperm_idx]
                    batch_vertices_n = batch.vertices_normal[:, randperm_idx]

                    ## masking face region using hat function (min x1 ~ max x2)
                    # t_mask = plateau_hat_points(batch_template_v) + 1.0
                    
                ## model prediction -------------------------------------------------------------------------------
                    ## B: number of batch, Nv : number of vertices, Nc: number of control vertices
                    ## weight prediction: (B, Nv, Nc)
                    ## key_d prediction:  (B, Nc, 3+3) [deformed cage]
                ## vLBS hybrid model prediction   ----------------------------------------------------------------
                
                if self.opts.vis_joint_pos:
                    pred_vertices, recon_vertices, recon_source, exp_z, pred_source, t_mask, key_d, pred_key_weight, W_lbs, T_lbs, _ = self.model(
                        batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                        batch.mesh_data, epoch=epoch
                    )
                else:
                    pred_vertices, recon_vertices, recon_source, exp_z, pred_source, t_mask, key_d, pred_key_weight, W_lbs, T_lbs = self.model(
                        batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                        batch.mesh_data, epoch=epoch                   
                    )
                    
                    if self.opts.use_hyb_delta_lbs_input:
                        pred_vertices_CBD, recon_vertices_CBD, recon_source_CBD, exp_z_CBD, pred_source_CBD, t_mask_CBD, pred_key_weight_CBD = self.model_CBD(
                            batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                            batch.mesh_data, epoch=epoch, lbs_output = pred_vertices,
                        )
                    
                    elif self.opts.use_hyb_concat_lbs:
                        ## add pred_vertices to batch_template_v
                        pred_vertices_CBD, recon_vertices_CBD, recon_source_CBD, exp_z_CBD, pred_source_CBD, t_mask_CBD, pred_key_weight_CBD = self.model_CBD(
                            batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                            batch.mesh_data, epoch=epoch, lbs_output = pred_vertices, lbs_source = pred_source
                        )
                    
                    else: # default
                        pred_vertices_CBD, recon_vertices_CBD, recon_source_CBD, exp_z_CBD, pred_source_CBD, t_mask_CBD, pred_key_weight_CBD = self.model_CBD(
                            batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                            batch.mesh_data, epoch=epoch, 
                        )
                    pred_vertices = pred_vertices + pred_vertices_CBD # expressed face
                    pred_source = pred_source + pred_source_CBD # neutral face
                
                # ------------------------------------------------------------------------------------------------
                # vis_mask_plot(batch_template_v[0].detach().cpu(), t_mask[0].detach().cpu(), logdir='./', name='test')
                # vis_mask_plot(batch_template_v[0].detach().cpu(), inv_t_mask[0].detach().cpu(), logdir='./', name='test')
                if self.opts.no_t_mask:
                    t_mask = 1.0
                    inv_t_mask = 0.0
                else:
                    inv_t_mask = 1.0 - t_mask
                
                ## use segmentation for loss weight ------- ( not used ) ------------------------------------------
                ## -> re-weighting based on facial region area
                # if self.opts.use_segment_weight:
                #     with torch.no_grad():
                #         # batch.segmentation # (B, Nv, 24)
                #         segment_weight = batch.segmentation.sum(1) / N # batch.segmentation.sum(1).sum(1) # (B, 24)
                #         batch_segment_weight = (batch.segmentation * segment_weight[:, None])[:, randperm_idx]
                #         t_mask = t_mask * batch_segment_weight
                # ------------------------------------------------------------------------------------------------
                
                # loss -------------------------------------------------------------------------------------------
                mesh_data_num = batch.mesh_data.cpu().numpy()
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf', 'ict'])[mesh_data_num]
                
                loss_dict = {} # make it as a dictionary
                HB = batch.vertices.shape[0] // 2
                
                ## stage-agnostic loss terms
                if self.opts.no_t_mask:
                    loss_dict['recon-def'] = F.mse_loss(batch_vertices_v, pred_vertices)
                else:
                    loss_dict['recon-def'] = F.mse_loss(batch_vertices_v*t_mask, pred_vertices*t_mask) # focus deformation on face

                    loss_dict['recon-def'] += F.mse_loss(batch_template_v*inv_t_mask, pred_vertices*inv_t_mask) # should be static on elsewhere
                if self.model.use_full_vertex: # directly hanging mesh vertex position 
                    if self.opts.no_t_mask:
                        loss_dict['recon-neu'] = F.mse_loss(batch_template_v, pred_source)
                    else:
                        loss_dict['recon-neu'] = F.mse_loss(batch_template_v*t_mask, pred_source*t_mask)
                        loss_dict['recon-neu'] += F.mse_loss(batch_template_v*inv_t_mask, pred_source*inv_t_mask)

                
                
                #-------( not used )------------------------------------------------------------------------------
                if self.model.use_shp_recon:
                    loss_dict['shape'] = F.mse_loss(
                        batch_template_v[:,rearange_idx]*t_mask,
                        recon_source[:,randperm_idx[rearange_idx]]*t_mask
                    ) # for shape AE
                if self.model.use_exp_recon:
                    loss_dict['exp-v'] = F.mse_loss(
                        batch_vertices_v[:,rearange_idx]*t_mask,
                        recon_vertices[:,randperm_idx[rearange_idx]]*t_mask
                    ) # for expression AE
                #-------------------------------------------------------------------------------------------------

                # soft POU ---------------------------------------------------------------------------------------
                if self.opts.pou_loss:
                    pred_key_weight_sum = pred_key_weight.sum(-1)
                    loss_dict['pou'] = F.mse_loss(
                        torch.ones_like(pred_key_weight_sum).to(self.device),
                        pred_key_weight_sum, 
                    )
                #-------------------------------------------------------------------------------------------------
                
                # Laplacian smoothing ----------------------------------------------------------------------------
                if not use_perm and self.opts.use_laplacian:
                    loss_dict['lap'] = laplacian_loss(
                        batch, pred_key_weight, self.train_dataset, mesh_data_num, self.device
                    ) * BS_denom
                    
                    # with torch.no_grad():
                    #     batch_vertices_lap = cotmatrix @ batch_vertices_v
                    #     batch_template_lap = cotmatrix @ batch_template_v
                    # pred_vertices_lap = cotmatrix @ pred_vertices
                    # pred_source_lap = cotmatrix @ pred_source
                    
                    # loss_dict['pois'] += F.mse_loss(batch_vertices_lap, pred_vertices_lap)
                    # loss_dict['pois'] += F.mse_loss(batch_vertices_lap, pred_vertices_lap)
                #-------------------------------------------------------------------------------------------------

                # vertex normal loss -----------------------------------------------------------------------------
                if not use_perm and self.opts.use_normal_loss:
                    pred_vertices_norm = calc_norm_torch(pred_vertices, batch.faces, at='verts') # [1, V, 3]
                    
                    if self.opts.no_t_mask:
                        loss_dict['norm-def'] = F.mse_loss(
                                batch_vertices_n, pred_vertices_norm
                            )
                    else:
                        loss_dict['norm-def'] = (
                            F.mse_loss(
                                batch_vertices_n*t_mask, pred_vertices_norm*t_mask
                            ) + F.mse_loss(
                                batch_template_n*inv_t_mask, pred_vertices_norm*inv_t_mask
                            )
                        )
                    
                    if self.model.use_full_vertex:
                        pred_template_norm = calc_norm_torch(pred_source, batch.faces, at='verts')   # [1, V, 3]
                        
                        loss_dict['norm-neu'] = F.mse_loss(
                            batch_template_n, pred_template_norm
                        )
                #-------------------------------------------------------------------------------------------------
                
                # latent alignment loss --------------------------------------------------------------------------
                #if (self.opts.use_data2 or self.opts.use_data3):
                if self.opts.align_latent:
                    if mesh_data=='ict':
                        loss_dict['exp-z'] = F.mse_loss(
                            batch.exp_coeff.unsqueeze(1), exp_z
                        )
                    else:
                        exp_z_facs, exp_z_ext = exp_z[...,:53], exp_z[...,53:]
                        loss_dict['exp-z'] = non_ict_loss(exp_z_facs)
                        loss_dict['exp-z'] += F.mse_loss(
                            torch.zeros_like(exp_z_ext).to(self.device),
                            exp_z_ext
                        )
                    # ------------------------------------------------------------------------------------------------

                # get total loss (lambda weights are multiplied here!) -------------------------------------------
                loss = 0
                for key, value in loss_dict.items():
                    # key_ = key.split("_")[0]
                    tmp = value*self.loss_lambda[key]
                    loss += tmp
                    running_losses[key] += tmp # for logging
                loss_dict["total"] = loss
                # ------------------------------------------------------------------------------------------------

                # backward ---------------------------------------------------------------------------------------
                loss.backward()
                # try:
                #     loss.backward()
                # except:
                #     import pdb;pdb.set_trace()
                self.optimizer.step()
                # ------------------------------------------------------------------------------------------------

                # running loss for logging -----------------------------------------------------------------------
                running_losses["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5e}, mesh data: {mesh_data_num}")
                # ------------------------------------------------------------------------------------------------

                global_step += 1
                train_counter += 1
                                
                if index % interv_train == 1:

                    IDX = torch.tensor([0, 1, HB, BS-1])
                    with torch.no_grad():
                        if self.opts.vis_joint_pos:
                            #pred_vertices, recon_vertices, recon_source, exp_z, pred_source = self.model(
                            pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _, _, key_weight, W_lbs, T_lbs, C_lbs = self.model(
                                batch.template[IDX], batch.vertices[IDX],
                                batch.template_normal[IDX], batch.vertices_normal[IDX],
                                batch.mesh_data, epoch=epoch, out_kw=True,
                            )
                        else: # default
                            pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _, _, key_weight, W_lbs, T_lbs = self.model(
                            batch.template[IDX], batch.vertices[IDX],
                            batch.template_normal[IDX], batch.vertices_normal[IDX],
                            batch.mesh_data, epoch=epoch, out_kw=True,
                        )
                    
                    # for visualization
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                    
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Train] "
                    __idx__ = 1/train_counter
                    for key, value in running_losses.items():
                        log_text += f"{key}: {value*__idx__:.6e} "

                    if not is_stts_added:
                        log_text+='\n>>> Sum across vertex weights on each cage: '
                        log_text+=f'(max: {key_weight[0].sum(0).max().item():.5e}, min: {key_weight[0].sum(0).min().item():.5e})\n'
                        log_text+=f'>>> Num actually used cage vertex: {torch.count_nonzero(key_weight[0].sum(0))} / {key_weight.shape[-1]}'
                        is_stts_added=True
                    self.logger.write(log_text+"\n")
                    
                    frame = HB
                    v_list = [
                        vertices[0].cpu().detach(),
                        vertices[1].cpu().detach(),
                        vertices[HB].cpu().detach(),
                        vertices[BS-1].cpu().detach(),
                        pred_vertices[0].cpu().detach(),
                        pred_vertices[1].cpu().detach(),
                        pred_vertices[2].cpu().detach(),
                        pred_vertices[3].cpu().detach(),
                    ]
                    
                    len_v = len(v_list)
                    f_list = [faces] * len_v
                    save_logdir = f"{self.opts.log_dir}/img/train/mesh"
                    save_img_name = f"{epoch:03d}_{index:04d}"

                    if self.opts.vis_joint_pos:
                        C_lbs = C_lbs.detach().cpu() # keep torch ok (plot 내부에서 numpy로 바꿔도 됨)
                        C_list = [
                            None, None, None, None, # GT 에는 안찍기
                            C_lbs[0], C_lbs[1], C_lbs[2], C_lbs[3],
                        ]
                        plot_image_array_points(
                            v_list, 
                            f_list,
                            rot_list=[[0,0,0]] * len_v,
                            size=1, bg_black=False, mode='shade',
                            logdir=save_logdir,
                            name=save_img_name, save=True,
                            points_list=C_list,  
                            points_color='r',
                            points_size=4,
                        )
                    else:
                        plot_image_array(
                            v_list, f_list, 
                            rot_list=[[0,0,0]] * len_v, 
                            size=1, bg_black=False, mode='shade',
                            logdir=save_logdir,
                            name=save_img_name, save=True
                        )
                
                if self.opts.debug:
                    break
                # ------------------------------------------------------------------------------------------------
            
            ### scheduler ----------------------------------------------------------------------------------------
            if epoch != 0:
                self.scheduler.step()
                curr_lr = self.optimizer.param_groups[0]["lr"]
                if epoch % self.opts.sc_step==0 and not is_stepped:
                    log_notice = f'[{epoch:03d}/{epochs:03d}][{index:04d}][Train] scheduler stepped: {curr_lr:.6e}'
                    self.logger.write(log_notice+"\n")
                    is_stepped=True
            # ----------------------------------------------------------------------------------------------------
                
            # log
            if self.opts.tb:
                self.log_loss(self.writer_train, running_losses, epoch, train_counter)

            # save model
            if epoch % self.opts.save_interval == 0:
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_lbs_{epoch:03d}.pth') # save LBS
                torch.save(self.model_CBD.state_dict(), f'{self.opts.log_dir}/model_cbd_{epoch:03d}.pth') # save CBD
                
            
            ######################################################################################################
            # validation -----------------------------------------------------------------------------------------
            self.model.eval()
            self.model_CBD.eval()
            
            print(f"[{epoch:03d}/{epochs:03d}][Valid]")
            running_losses_val = {
                "recon-def": 0.0,
                "recon-neu": 0.0,
                "exp-z": 0.0,
                "exp-v": 0.0,
                "shape": 0.0,
                "total": 0.0
            }
            
            if self.opts.use_lbs_laplacian:
                running_losses_val['lbs-lap']=0.0
            if self.opts.use_lbs_ent:
                running_losses_val['lbs-ent']=0.0
            if self.opts.use_lbs_t:
                running_losses_val['lbs-t']=0.0
            if self.opts.use_lbs_R:
                running_losses_val['lbs-R']=0.0
            if self.opts.use_lbs_bal:
                running_losses_val['lbs-bal']=0.0
            
            counter = 0
            pbar = tqdm(enumerate(self.valid_dataloader), total=len_valid_data, ncols=100)
            for index, batch in pbar:
                counter += 1
                
                # model validation -------------------------------------------------------------------------------
                with torch.no_grad():
                    if self.opts.vis_joint_pos:
                        pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _, _, pred_key_weight, W_lbs, T_lbs, C_lbs = self.model(
                            batch.template, batch.vertices, 
                            batch.template_normal, batch.vertices_normal,
                            batch.mesh_data, epoch=epoch
                        )
                    else: # default
                        pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _, _, pred_key_weight, W_lbs, T_lbs = self.model(
                            batch.template, batch.vertices, 
                            batch.template_normal, batch.vertices_normal,
                            batch.mesh_data, epoch=epoch
                        )
                # ------------------------------------------------------------------------------------------------
                
                # loss ------------------------------------------------------------------------------------------- 
                with torch.no_grad():
                    mesh_data_num = batch.mesh_data.cpu().numpy()
                    mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf','ict'])[mesh_data_num]
                
                    loss_dict = {} # make it as a dictionary
                    HB = batch.vertices.shape[0] // 2
                    
                    loss_dict['recon-def'] = F.mse_loss(batch.vertices, pred_vertices) # for NGBC model
                    if self.model.use_full_vertex:
                        loss_dict['recon-neu'] = F.mse_loss(batch.template, pred_source) 
                        
                    if self.model.use_shp_recon:
                        loss_dict['shape'] = F.mse_loss(batch.template, recon_source) # for shape AE
                    if self.model.use_exp_recon:
                        loss_dict['exp-v'] = F.mse_loss(batch.vertices, recon_vertices) # for expression AE
                    # loss_dict['exp-z'] = F.mse_loss(exp_z[:HB], exp_z[HB:])
                    
                    if self.opts.use_lbs_laplacian:
                        loss_dict['lbs-lap'] = 0.0
                    if self.opts.use_lbs_ent:
                        loss_dict['lbs-ent'] = 0.0
                    if self.opts.use_lbs_t:
                        loss_dict['lbs-t'] = 0.0
                    if self.opts.use_lbs_R:
                        loss_dict['lbs-R'] = 0.0
                    if self.opts.use_lbs_bal:
                        loss_dict['lbs-bal'] = 0.0  
                    
                    # get total loss
                    loss = 0
                    for key, value in loss_dict.items():
                        key_ = key.split("_")[0]
                        tmp = value.item()*self.loss_lambda[key_]
                        loss += tmp
                        running_losses_val[key] += tmp
                    loss_dict["total"] = loss 

                # running loss for logging
                running_losses_val["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5e}, mesh data: {mesh_data_num}")
                # ------------------------------------------------------------------------------------------------
            
                # ------------------------------------------------------------------------------------------------
                interv_val = round(len_valid_data / 5)
                if index % interv_val == 0:
                    # for visualization
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Valid] "
                    __jdx__ = 1/counter
                    for key, value in running_losses_val.items():
                        log_text += f"{key}: {value*__jdx__:.6e} "
                    self.logger.write(log_text+"\n")
                    
                    frame = HB
                    v_list = [
                        vertices[0].cpu().detach(),
                        vertices[1].cpu().detach(),
                        vertices[HB].cpu().detach(),
                        vertices[BS-1].cpu().detach(),
                        pred_vertices[0].cpu().detach(),
                        pred_vertices[1].cpu().detach(),
                        pred_vertices[HB].cpu().detach(),
                        pred_vertices[BS-1].cpu().detach(),
                    ]
                    len_v = len(v_list)
                    f_list=[faces] * len_v
                    save_logdir = f"{self.opts.log_dir}/img/valid/mesh"
                    save_img_name = f"{epoch:03d}_{counter:04d}"
                    
                    if self.opts.vis_joint_pos:
                        C_lbs = C_lbs.detach().cpu() # keep torch ok (plot 내부에서 numpy로 바꿔도 됨)
                        C_list = [
                            None, None, None, None, # GT 에는 안찍기
                            C_lbs[0], C_lbs[1], C_lbs[HB], C_lbs[BS-1],
                        ]
                        plot_image_array_points(
                            v_list, 
                            f_list,
                            rot_list=[[0,0,0]] * len_v,
                            size=1, bg_black=False, mode='shade',
                            logdir=save_logdir,
                            name=save_img_name, save=True,
                            points_list=C_list,  
                            points_color='r',
                            points_size=4,
                        )
                    else:
                        plot_image_array(
                            v_list, f_list, 
                            rot_list=[[0,0,0]] * len_v, 
                            size=1, bg_black=False, mode='shade',
                            logdir=save_logdir,
                            name=save_img_name, save=True
                        )
                    
                # ------------------------------------------------------------------------------------------------
                if self.opts.debug:
                    break
            # log
            if self.opts.tb:
                self.log_loss(self.writer_valid, running_losses_val, epoch, counter)
            
            # best loss
            val_loss = running_losses_val["total"]/counter
            if val_loss < BEST_LOSS:
                BEST_LOSS = val_loss
                BEST_EPOCH = epoch
                print(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6e} - Best epoch: {BEST_EPOCH:03d}\n")
                self.logger.write(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6e}\n")
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_lbs_best.pth')
                torch.save(self.model_CBD.state_dict(), f'{self.opts.log_dir}/model_cbd_best.pth')
            else:
                self.logger.write(f"[{epoch:03d}/{epochs:03d}] Curr Loss: {val_loss:.6e} (Best Loss: {BEST_LOSS:.6e} [{BEST_EPOCH:03d}])\n")
                print(f"[{epoch:03d}/{epochs:03d}] Curr Loss: {val_loss:.6e} (Best Loss: {BEST_LOSS:.6e} [{BEST_EPOCH:03d}])\n")
    
    
    
    @staticmethod
    def log_loss(writer, loss_dict, step, counter=None):
        if counter:
            N = counter
        else:
            N = 1
        for key, value in loss_dict.items():
            writer.add_scalar(f'{key}', value/N, step)

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

def vis_mask_plot(XX, WW, logdir='./', name='test'):
    #import torch
    #import numpy as np
    import matplotlib.pyplot as plt
    #from utils.remesh_utils import ICT_face_model
    
    fig1 = plt.figure(figsize=(14, 8))
    ax3 = fig1.add_axes([0.3, 0.0, 0.5, 0.5], projection='3d') ## left bottom W H
    
    # Use default colormap; do not specify colors explicitly
    p = ax3.scatter(
        XX[:, 2].numpy(),
        XX[:, 0].numpy(),
        XX[:, 1].numpy(),
        c=WW.numpy(),
        s=5,
        depthshade=False)
    
    ax3.set_title("mask", y=-0.21, fontsize=22)
    cb = fig1.colorbar(p, ax=ax3, shrink=0.75)
    
    # Equal aspect
    try:
        ax3.set_box_aspect([1, 1, 1])
    except Exception:
        # Fallback for older matplotlib: approximate equal aspect
        xyzlim = np.array([ax3.get_xlim3d(), ax3.get_ylim3d(), ax3.get_zlim3d()])
        xyzmin = xyzlim[:, 0].min()
        xyzmax = xyzlim[:, 1].max()
        ax3.set_xlim3d([xyzmin, xyzmax])
        ax3.set_ylim3d([xyzmin, xyzmax])
        ax3.set_zlim3d([xyzmin, xyzmax])
    plt.savefig('{}/{}.png'.format(logdir, name), bbox_inches = 'tight')
    # plt.show()
    plt.close(plt.gcf())
    
if __name__ == "__main__":
    # argparse configs
    opts = Options()
    
    # base configs (yaml)
    opts_yaml = yaml.load(open(opts.config), Loader=yaml.FullLoader)
        
    # update with argparse configs
    opts_ = vars(opts)
    opts_yaml.update(opts_)
    opts = argparse.Namespace(**opts_yaml)
    
    trainer = Trainer(opts)
    # import pdb;pdb.set_trace()
    # -----------------------------
    # [added] LBS hybrid route
    # -----------------------------
    # import pdb;pdb.set_trace()
    if getattr(opts, "use_lbs", False):
        trainer.train_vLBSHybrid(epochs=opts.max_epoch)
    elif opts.version==1:
        trainer.train_v1(epochs=opts.max_epoch)
    elif opts.version==2:
        trainer.train_v2(epochs=opts.max_epoch)
    elif opts.version==5 or opts.version==55:
        trainer.train_v5(epochs=opts.max_epoch)
    elif opts.version==6: # train LBS separately
        trainer.train_vLBS(epochs=opts.max_epoch)
    elif opts.version==7: # train LBS hybrid separately
        trainer.train_vLBSHybrid2(epochs=opts.max_epoch)
    elif opts.version==8: # train LBS + CBD jointly
        trainer.train_vLBSHybrid3(epochs=opts.max_epoch)
    else:
        raise NotImplementedError('no matching version!') 
    

