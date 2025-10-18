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
from utils.exp_utils import Model_mk1, Model_mk3_1
from utils.remesh_utils import (
    compute_MVC_vertexwise,
    apply_MVC_weights_batch,
    build_padded_neighbors,
    pca_normal_axis_vectorized,
    mvc_weights_torch,
    # MVC_vertexwise_batched,
)
from models import LinearEncoder, PointNet_small, MLP


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
        
        self.cage_v = torch.tensor(test_cage.vertices).float().to(device)*1.2
        self.cage_f = torch.tensor(test_cage.faces).long().to(device)
        # if self.optim_cage:
        #     self.cage_v = nn.Parameter(self.cage_v)
        
        ## poinnet encoder
        # self.encoder = Model_mk3_1(in_dim, hid_dim).to(device)
        self.encoder = PointNet_small(in_dim, hid_dim, out_type='global').to(device)
        
        ## atlasnet decoder => MLP
        self.nc_decoder = MLP(
            # [in_dim+hid_dim]+[hid_dim]*3+[out_dim], 
            # [hid_dim]+[hid_dim]*3+[out_dim], 
            # [in_dim]+[hid_dim]*3+[out_dim], 
            [hid_dim]+[hid_dim]*3+[out_dim*self.C], 
            act='lrelu', nrm='none', #dropout=True, p=.2
        ).to(device)
        self.nd_decoder = MLP(
            # [in_dim+hid_dim+hid_dim]+[hid_dim]*3+[out_dim], 
            # [hid_dim+hid_dim]+[hid_dim]*3+[out_dim], 
            [hid_dim+hid_dim]+[hid_dim]*3+[out_dim*self.C], 
            act='lrelu', nrm='none', #dropout=True, p=.2
        ).to(device)
        # self.nc_decoder = LinearEncoder(
        #     # in_dim+hid_dim,out_dim*self.C,
        #     hid_dim, out_dim*self.C,
        #     act='relu', nrm='batch',
        # ).to(device)
        # self.nd_decoder = LinearEncoder(
        #     # in_dim+hid_dim+hid_dim, out_dim*self.C,
        #     hid_dim+hid_dim, out_dim*self.C,
        #     act='relu', nrm='batch',
        # ).to(device)
        
    def forward(self, source_mesh, deform_mesh, epoch=0, return_cage=False):
        """
        Args:
            source_mesh (torch.tensor) [1, N, 3]: input source mesh
            deform_mesh (torch.tensor) [B, N, 3]: input deformed mesh
        Return:
            predicted deformed mesh
        """
        _, N, _ = source_mesh.shape
        B, N, _ = deform_mesh.shape
        
        ## shares same encoder!
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
        
        mvc = mvc_weights_torch(
            # source_mesh.squeeze(0), 
            # source_cage_v.squeeze(0), 
            source_mesh[0], 
            source_cage_v[0], 
            self.cage_f
        ) # [N, C]
        
        predicted_src_mesh = mvc @ source_cage_v ## [N, C] @ [B, C, 3] -> [B, N, 3]
        predicted_def_mesh = mvc @ deform_cage_v ## [N, C] @ [B, C, 3] -> [B, N, 3]
        
        if return_cage:
            return predicted_def_mesh, predicted_src_mesh, mvc, source_cage_v, deform_cage_v            
        return predicted_def_mesh, mvc