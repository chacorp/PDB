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
                 use_shp_recon=True, # necessary for training, but not needed for inference
                 use_full_vertex=False, # default: false (= delta form)
                ):
        super().__init__()
        self.opts = opts
        
        if self.opts is not None:
            if self.opts.in_type == 0:
                in_dim = 3
            elif self.opts.in_type == 1:
                in_dim = 6
            else:
                raise NotImplementedError('in_type not implemented')
                
            if self.opts.out_type == 0:
                self.use_full_vertex = False
                out_dim = 3
            elif self.opts.out_type == 1:
                self.use_full_vertex = True
                out_dim = 3
            elif self.opts.out_type == 2:
                self.use_full_vertex = False
                out_dim = 9 # (6D + translation 3) will be reshaped into 3x4 matrix
                M_ = num_cage_vertices
                num_cage_vertices = num_cage_vertices * 4
                
                from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
                self._6D_to_rot_ = _6D_to_rot_
            else:
                raise NotImplementedError('out_type not implemented')
                
        self.is_train = is_train
        
        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_shp_recon = use_shp_recon
        self.use_exp_recon = use_exp_recon        
        self.use_full_vertex = use_full_vertex

        
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
            in_dim=L,
            style_dim=L,
            out_dim= M_*out_dim if self.opts.out_type == 2 else M*out_dim,
            num_layers=self.num_layers, 
            use_style=True, out_type='global'
        ).to(device)

        if self.is_train:
            if self.use_exp_recon:
                self.recon_exp_model = nn.ModuleList([
                    Model(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
            
            if self.use_shp_recon:
                self.recon_shp_model = nn.ModuleList([
                    Model(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
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
        
        source_in = source_vert
        deform_in = deform_vert
        if self.in_dim == 6:
            source_in = torch.cat([source_vert, source_norm], dim=-1)
            deform_in = torch.cat([deform_vert, deform_norm], dim=-1)
            
        z_ID_B = self.shape_model(source_in) # (B, 1, L)
                
        key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
        # --> (B, N, 4M) if self.opts.out_type == 2
                
        exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
        # exp_z_v = self.exp_z_model(source_in, z_ID_B) # (B, 1, L)
        
        key_d = self.key_d_model(exp_z, z_ID_B)
        
        if self.opts.out_type == 2:
            # cage transform matrix
            M_ = self.num_cage_vertices // 4
            key_d = key_d.reshape(B, M_, 9)
            tmp_R, tmp_t = key_d[...,:6], key_d[...,6:]
            
            tmp_R = self._6D_to_rot_(tmp_R).reshape(B, -1, 3, 3)
            key_d = torch.cat([tmp_R, tmp_t[..., None]], dim=-1) # (B, M, 3, 4)
            key_d = key_d.permute(0,1,3,2).reshape(B, -1, 3) # (B, M4, 3)
            #key_d = key_d.permute(0,3,1,2).reshape(B, -1, 3) # (B, 4M, 3)
        else:
            key_d = key_d.reshape(B, self.num_cage_vertices, 3)
        # key_v = self.key_d_model(exp_z_v, z_ID_B).reshape(B, M, 3)
        
        ## necessary
        if self.use_shp_recon:
            recon_source = self.recon_shp_model[mesh_data](z_ID_B)
            recon_source = recon_source.reshape(B, -1, 3)
        else:
            recon_source = 0
        
        ## unnecessary
        if self.use_exp_recon:
            recon_delta_v = self.recon_exp_model[mesh_data](exp_z)
            recon_delta_v = recon_delta_v.reshape(B, -1, 3)
            recon_deformed = recon_delta_v + source_vert
        else:
            recon_deformed = 0

        ## optional
        if self.use_full_vertex:
            exp_z_s = self.exp_z_model(source_in, z_ID_B) # (B, 1, L)    
            pred_source = self.key_d_model(exp_z_s, z_ID_B).reshape(B, self.num_cage_vertices, 3)
        else:
            pred_source = 0
            
        delta_v = torch.einsum('bnc,bci->bni',key_weight,key_d)
    
        if self.use_full_vertex:
            pred_deformed = delta_v
        else:
            pred_deformed = delta_v + source_vert
        
        #verts_v_th = torch.einsum('bnc,bci->bni',key_weight,key_v)
        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight
            
        # if self.use_exp_recon and self.use_shp_recon:
        #     return pred_deformed, recon_deformed, recon_source, exp_z
        # elif self.use_exp_recon and not self.use_shp_recon:
        #     return pred_deformed, recon_deformed, 0, exp_z
        # elif not self.use_exp_recon and self.use_shp_recon:
        #     return pred_deformed, 0, recon_source, exp_z
        # else:
        #     return pred_deformed, 0, 0, exp_z
        return pred_deformed, recon_deformed, recon_source, exp_z, pred_source
