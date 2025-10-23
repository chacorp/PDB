import os
import glob
import random

import numpy as np
import trimesh

import sys
import pickle

from pathlib import Path
__abs_path__ = str(Path(__file__).parents[1].absolute())
__mesh_util_path__ = f'{__abs_path__}/mesh_utils'

for __util_path__ in [__abs_path__, __mesh_util_path__]:
    if not __util_path__ in sys.path:
        sys.path+=[__util_path__]

import torch
import torch.nn as nn
from utils.remesh_utils import (
    compute_MVC_vertexwise,
    apply_MVC_weights_batch,
    build_padded_neighbors,
    pca_normal_axis_vectorized,
    mvc_weights_torch,
)
from utils.cages import mean_value_coordinates_3D
from utils.exp_utils import Model_mk1, Model_mk3_1
from utils.exp_utils import plateau_hat_points
from models.encoder import PointNet_small, PointNet_large, MLP
from torch.utils.checkpoint import checkpoint

class CageNet(nn.Module):
    """
        Simple implementation of 'Neural Cages for Detail-preserving 3d Deformations'
        for Facial Animation Retargeting
    """
    def __init__(self, 
                 in_dim=3,
                 out_dim=3, 
                 hid_dim=512,
                 device='cpu',
                 optim_cage=False,
                ):
        super(CageNet, self).__init__()
        
        # pointnet encoder
        self.device=device
        self.optim_cage = optim_cage
        
        
        # self.cage_v = nn.Parameter(torch.rand(128, 3))
        test_cage = trimesh.load(f'{__abs_path__}/test_cage.obj') # 512 vertices 988 faces
        ## may need a better mesh!
        self.C = test_cage.vertices.shape[0]        
        
        self.cage_v = torch.tensor(test_cage.vertices+np.array([0,0.1,0])).float().to(device)#*1.2
        self.cage_f = torch.tensor(test_cage.faces).long().to(device)
        # if self.optim_cage:
        #     self.cage_v = nn.Parameter(self.cage_v)
        
        ## poinnet encoder
        # self.encoder = Model_mk3_1(in_dim, hid_dim).to(device)
        self.encoder = PointNet_small(
            in_dim, hid_dim, out_type='global',
            no_norm_layer=True
        ).to(device)
        # self.encoder = PointNet_large(in_dim, hid_dim, out_type='global').to(device)
        
        ## atlasnet decoder => MLP
        self.nc_decoder = MLP(
            # [in_dim+hid_dim]+[hid_dim]*3+[out_dim], 
            [hid_dim]+[hid_dim]*2+[out_dim*self.C], 
            act='lrelu', nrm='none', #dropout=True, p=.2
        ).to(device)
        self.nd_decoder = MLP(
            # [in_dim+hid_dim+hid_dim]+[hid_dim]*3+[out_dim], 
            [hid_dim+hid_dim]+[hid_dim]*2+[out_dim*self.C], 
            act='lrelu', nrm='none', #dropout=True, p=.2
        ).to(device)
        
    def forward(self, source_mesh, deform_mesh, epoch=0, return_cage=False):
        """
        Args:
            source_mesh (torch.tensor) [B, N, 3]: input source mesh
            deform_mesh (torch.tensor) [B, N, 3]: input deformed mesh
        Return:
            predicted deformed mesh
        """
        _, N, _ = source_mesh.shape
        B, N, _ = deform_mesh.shape

        
        # with torch.no_grad():
        #     ## sampling points with probability
        #     margin = 0.8
        #     _p = (plateau_hat_points(source_mesh[0]).squeeze() + margin) / (1 + margin)
                
        #     ## random sampling and random permutation
        #     randperm_idx = torch.multinomial(_p, 1024)
        #     rearange_idx = torch.argsort(randperm_idx)
            
        #     source_sampled_v = source_mesh[:, randperm_idx]
        #     deform_sampled_v = deform_mesh[:, randperm_idx]
        
        ## shares same encoder!
        # x = torch.cat([deform_sampled_v, source_sampled_v], dim=0) # [B+1, N, 3]
        x = torch.cat([deform_mesh, source_mesh], dim=0) # [B+1, N, 3]
        out, _ = self.encoder(x, return_all=True) # [B+1, 512]
        # out, _ = self.encoder(x) # [B+1, 512]
        # out = out.unsqueeze(1) # [B+1, 1, 512]
        t_code, s_code = out[:B], out[B:] # [B, 1, 512] & [1, 1, 512]
        
        ## feed seperately
        # t_code, _ = self.encoder(deform_mesh, return_all=True) 
        # s_code, _ = self.encoder(source_mesh, return_all=True) 
        
        cage_v = self.cage_v.view(1, -1, 3) # [1, 512, 3]
        # cage_v = self.cage_v.permute(1,0).view(1, 3, -1) # [1, 3, 512]
        cage_v = cage_v.repeat(B, 1, 1)  # [B, 3, 512]

        ###### works if the model outputs single code and reshape to cage  
        # x_nc = torch.cat([
        #     cage_v, s_code.expand(-1, self.C, -1), 
        # ],dim=-1) # [1, C, 512+3]
        # import pdb;pdb.set_trace()
        
        # x_nc = cage_v + s_code # [B, 3, 512]
        # source_cage_v = self.nc_decoder(x_nc) + cage_v# [B, C, 3]
        ## x_nd = torch.cat([s_code, cage_v],dim=-2) # [B, 4, 512]
        source_cage_v = self.nc_decoder(s_code).reshape(
            B, self.C, 3
        ) + cage_v # [B, C, 3]
        
        # x_nd = source_cage_v + 
        #     source_cage_v,
        #     s_code.expand(B, self.C, -1),
        #     t_code.expand(-1, self.C, -1),
        # ],dim=-1) # [B, C, 512+512+3]
        # deform_cage_v = self.nd_decoder(x_nd) + source_cage_v 
        x_nd = torch.cat([s_code, t_code],dim=-1) # [B, 1, 512+512]
        deform_cage_v = self.nd_decoder(x_nd).reshape(
            B, self.C, 3
        ) + source_cage_v # [B, C, 3]
        
        # mvc = mvc_weights_torch(
        #     # source_mesh.squeeze(0), 
        #     # source_cage_v.squeeze(0), 
        #     source_mesh[0], 
        #     source_cage_v[0], 
        #     self.cage_f
        # ) # [N, C]
        
        # mvc, mvc_unnormed = mean_value_coordinates_3D(source_mesh[0,None], source_cage_v[0,None], self.cage_f[None], verbose=True)
        # mvc = mean_value_coordinates_3D(
        #     source_mesh[0][None], source_cage_v[0][None], self.cage_f[None]
        # )
        ############### use gradient checkpointing if COO happens..! ###############
        q_chunk = 1024
        q_slices = [(s, min(N, s + q_chunk)) for s in range(0, N, q_chunk)]
        
        # import pdb;pdb.set_trace()
        mvc = torch.empty((1, N, self.C), dtype=s_code.dtype, device=s_code.device)
        for s, e in q_slices:
            q_chunk_t = source_mesh[:1, s:e, :].to(s_code.device, non_blocking=True)
            mvc[:, s:e] = checkpoint(
                mean_value_coordinates_3D, 
                q_chunk_t, source_cage_v[0][None], self.cage_f[None],
                use_reentrant=False,
            )
        ###########################################################################
        
        predicted_src_mesh = mvc @ source_cage_v ## [N, C] @ [B, C, 3] -> [B, N, 3] #### not needed?
        predicted_def_mesh = mvc @ deform_cage_v ## [N, C] @ [B, C, 3] -> [B, N, 3]
        
        if return_cage:
            return predicted_def_mesh, predicted_src_mesh, mvc, source_cage_v, deform_cage_v            
        return predicted_def_mesh, mvc

    @torch.no_grad()
    def retarget(self, 
                 src_neu_vert, 
                 src_def_vert, 
                 tgt_neu_vert,
                 out_kw=False,
                 ):
        """
        Args:
            source_mesh (torch.tensor) [B, N, 3]: input source mesh
            deform_mesh (torch.tensor) [B, N, 3]: input deformed mesh
        Return:
            predicted deformed mesh
        """
        _, N, _ = src_neu_vert.shape
        B, N, _ = src_def_vert.shape
        _, M, _ = tgt_neu_vert.shape
        
        ## feed seperately
        src_neu_code, _ = self.encoder(src_def_vert, return_all=True) 
        src_def_code, _ = self.encoder(src_neu_vert, return_all=True) 
        tgt_neu_code, _ = self.encoder(tgt_neu_vert, return_all=True) 
        
        cage_v = self.cage_v.view(1, -1, 3) # [1, 512, 3]
        cage_v = cage_v.repeat(B, 1, 1)  # [B, 3, 512]

        ###### works if the model outputs single code and reshape to cage
        source_cage_v = self.nc_decoder(src_neu_code).reshape(
            B, self.C, 3
        ) + cage_v # [B, C, 3]
        
        target_cage_v = self.nc_decoder(tgt_neu_code).reshape(
            B, self.C, 3
        ) + cage_v # [B, C, 3]
        
        x_nd = torch.cat([src_neu_code, src_def_code],dim=-1) # [B, 1, 512+512]
        deform_cage_v = self.nd_decoder(x_nd).reshape(
            B, self.C, 3
        ) # [B, C, 3]
        
        mvc = mean_value_coordinates_3D(tgt_neu_vert, target_cage_v[0][None], self.cage_f[None])
        
        pred_source = mvc @ target_cage_v ## [N, C] @ [B, C, 3] -> [B, N, 3] #### not needed?
        pred_deformed = mvc @ (target_cage_v+deform_cage_v) ## [N, C] @ [B, C, 3] -> [B, N, 3]
        
        if out_kw:
            return pred_deformed, pred_source, deform_cage_v, source_cage_v, mvc

        return pred_deformed, pred_source