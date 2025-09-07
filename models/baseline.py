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
from utils.remesh_utils import compute_MVC_vertexwise, apply_MVC_weights_batch, build_padded_neighbors, pca_normal_axis_vectorized


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
        
        self.cage_v = torch.tensor(test_cage.vertices).float().to(device)
        if self.optim_cage:
            self.cage_v = nn.Parameter(self.cage_v)
        self.cage_f = torch.tensor(test_cage.faces).long().to(device)
        
        # poinnet encoder
        self.encoder = Model_mk3_1(in_dim, hid_dim).to(device)
        
        # atlasnet decoder
        self.nc_decoder = Model_mk1(in_dim+hid_dim, out_dim).to(device)
        self.nd_decoder = Model_mk1(in_dim+hid_dim+hid_dim, out_dim).to(device)
        
        
    def forward(self, source_mesh, deform_mesh, epoch=0):
        """
        Args:
            source_mesh (torch.tensor) [1, N, 3]: input source mesh
            deform_mesh (torch.tensor) [B, N, 3]: input deformed mesh
        Return:
            predicted deformed mesh
        """
        _, V, _ = source_mesh.shape
        B, V, _ = deform_mesh.shape
        
        #shares same encoder!
        x = torch.cat([deform_mesh, source_mesh], dim=0) # [B+1, N, 3]
        out, _ = self.encoder(x) # [B+1, 512]
        out = out.unsqueeze(1) # [B+1, 1, 512]
        
        #t_code, s_code = out.unsqueeze(1).chunk(2)
        t_code, s_code = out[:B], out[B:] # [B, 1, 512] & [1, 1, 512]
        
        cage_v = self.cage_v.view(1, -1, 3) # [1, C, 3]
        
        x_nc = torch.cat([
            s_code.expand(-1, self.C, -1), 
            cage_v
        ],dim=-1) # [1, C, 512+3]
        source_cage_v = self.nc_decoder(x_nc) + cage_v # [1, C, 3]
        source_cage_v_expand = source_cage_v.expand(B, -1, -1) # [B, C, 3]
        
        
        x_nd = torch.cat([
            s_code.expand(B, self.C, -1),
            t_code.expand(-1, self.C, -1),
            source_cage_v_expand
        ],dim=-1) # [B, C, 512+512+3]
        deform_cage_v = self.nd_decoder(x_nd) + source_cage_v # [B, C, 3]
        
        mvc = compute_MVC_vertexwise(
            source_mesh.squeeze(0), 
            source_cage_v.squeeze(0), 
            self.cage_f
        ) # [N, C]
        
        predicted_mesh = mvc @ deform_cage_v ## [N, C] @ [B, C, 3] -> [B, N, 3]
        
        return predicted_mesh, mvc