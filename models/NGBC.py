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
from utils.exp_utils import Model, Model_mk2_1 #Model_mk1, Model_mk3_1


class NeuralGeneralizedBarycentricCoordinate(nn.Module):
    """
        Neural Generalized Barycentric Coordinate 
        (model specialized for facial animation)
    """
    def __init__(self, 
                 opts=None,
                 in_dim=3,
                 out_dim=3, 
                 hid_dim=256,
                 num_cage_vertices=512,
                 num_layers=4,
                 N_list=[3525, 2560, 5223], # voca biwi mf
                 use_softmax=False,
                 use_relu=True,
                 use_least_N=False,
                 least_number_of_zeros=256, # for sparsity (not used)
                 is_train=False,
                 tau=0.05,
                 device='cpu',
                 use_exp_recon=False, # was not necessary
                 use_shp_recon=True,
                ):
        super().__init__()
        self.opts = opts
        self.is_train = is_train
        
        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_exp_recon = use_exp_recon
        
        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros

        # coordinate predictor
        self.key_weight_model = Model(
            in_dim=in_dim, out_dim=M, 
            use_softmax=use_softmax,
            use_relu=use_relu, # default setting
            use_least_N=use_least_N,
        ).to(device)

        # expression encoder
        self.exp_z_model = Model_mk2_1(
            in_dim=in_dim, style_dim=L, out_dim=L,
            num_layers=self.num_layers, 
            use_style=True, out_type='global'
        ).to(device)

        # shape encoder
        self.shape_model = Model(in_dim=in_dim, out_dim=L, out_type='global').to(device)

        # cage displacement predictor
        self.key_d_model = Model_mk2_1(
            in_dim=L, style_dim=L, out_dim=M*out_dim, 
            num_layers=self.num_layers, 
            use_style=True, out_type='global'
        ).to(device)

        if self.is_train:
            if self.use_exp_recon:
                self.recon_exp_model = nn.ModuleList([
                    Model(in_dim=L, out_dim=N*out_dim, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
            
            if self.use_shp_recon:
                self.recon_shp_model = nn.ModuleList([
                    Model(in_dim=L, out_dim=N*out_dim, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
        
        # self.zero_w = torch.zeros(N,M).to(device)
        
    def forward(self, source_vert, deform_vert, source_norm, deform_norm, mesh_data, epoch=0, out_kw=False):
        """
        Args:
            source_vert (torch.tensor): [B, N, 3] source mesh vertices
            deform_vert (torch.tensor): [B, N, 3] deformed mesh vertices
            source_norm (torch.tensor): [B, N, 3] source mesh vertex normals
            deform_norm (torch.tensor): [B, N, 3] deformed mesh vertex normals
            mesh_data (int): indicator for data (0: voca, 1: biwi, 2: multiface)
            epoch (int): train epoch (epoch != iteration)
        Returns:
            (pred_deformed, recon_deformed, recon_source):
            predicted deformed mesh using weight and displacement,
            reconstructed deformed mesh
            reconstructed source mesh
        """
        B, N, _ = deform_vert.shape
        
        if self.in_dim == 6:
            source_in = torch.cat([source_vert, source_norm], dim=-1)
            deform_in = torch.cat([deform_vert, deform_norm], dim=-1)
            
        z_ID_B = self.shape_model(source_in) # (B, 1, L)
                
        key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
        
        exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
        # exp_z_v = self.exp_z_model(source_in, z_ID_B) # (B, 1, L)
        
        key_d = self.key_d_model(exp_z, z_ID_B).reshape(B, self.num_cage_vertices, 3)
        # key_v = self.key_d_model(exp_z_v, z_ID_B).reshape(B, M, 3)
        
        
        if self.use_shp_recon:
            recon_source = self.recon_shp_model[mesh_data](z_ID_B)
            recon_source = recon_source.reshape(B, -1, 3)

        if self.use_exp_recon:
            recon_delta_v = self.recon_exp_model[mesh_data](exp_z)
            recon_delta_v = recon_delta_v.reshape(B, -1, 3)
            recon_deformed = recon_delta_v + source_vert
                
        
        delta_v = torch.einsum('bnc,bci->bni',key_weight,key_d)
        pred_deformed = delta_v + source_vert
        
        #verts_v_th = torch.einsum('bnc,bci->bni',key_weight,key_v)
        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight
            
        if self.use_exp_recon and self.use_shp_recon:
            return pred_deformed, recon_deformed, recon_source, exp_z
        elif self.use_exp_recon and not self.use_shp_recon:
            return pred_deformed, recon_deformed, 0, exp_z
        elif not self.use_exp_recon and self.use_shp_recon:
            return pred_deformed, 0, recon_source, exp_z
        else:
            return pred_deformed, 0, 0, exp_z

