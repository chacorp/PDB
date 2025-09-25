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
        Neural ~Generalized~ Barycentric Coordinate (cannot generalize other than face)
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
        
        self.in_type = 0
        self.out_type = 1
        
        if self.opts is not None:
            self.in_type = self.opts.in_type
            self.out_type = self.opts.out_type
            
        if self.in_type == 0:
            in_dim = 3
        elif self.in_type == 1:
            in_dim = 6
        else:
            raise NotImplementedError('in_type not implemented')
            
        if self.out_type == 0:
            self.use_full_vertex = False
            out_dim = 3
        elif self.out_type == 1:
            self.use_full_vertex = True
            out_dim = 3
        elif self.out_type == 2:
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
            out_dim= M_*out_dim if self.out_type == 2 else M*out_dim,
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

    def reshape_key_d(self, key_d, B):
        if self.out_type == 2:
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
        return key_d
        
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
        # --> (B, N, 4M) if self.out_type == 2
                
        exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
        # exp_z_v = self.exp_z_model(source_in, z_ID_B) # (B, 1, L)
        
        key_d = self.key_d_model(exp_z, z_ID_B)

        key_d = self.reshape_key_d(key_d, B)
        
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
        
        return pred_deformed, recon_deformed, recon_source, exp_z, pred_source
        
class NeuralGeneralizedBarycentricCoordinate1(nn.Module):
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
                 use_least_N_on_V=False,
                 least_number_of_zeros=256, # for sparsity (not used)
                 is_train=False,
                 tau=0.05,
                 device='cpu',
                 use_exp_recon=False, # was not necessary
                 use_shp_recon=True, # necessary for training, but not needed for inference
                 use_shp=True,
                 use_full_vertex=False, # default: false (= delta form)
                ):
        super().__init__()
        self.opts = opts
                
        self.is_train = is_train
        
        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_shp_recon = use_shp_recon
        self.use_exp_recon = use_exp_recon        
        self.use_full_vertex = use_full_vertex
        self.use_shp = use_shp

        ###### NN input type settings
        # 0: (src_p),        (def_p, src_p)
        # 1: (src_p, src_n), (def_p, def_n, src_p, src_n)        
        self.in_type = 1
        self.out_type = 1 # vertex
        ######
        
        if self.opts is not None:
            self.in_type = self.opts.in_type
            self.out_type = self.opts.out_type
        
        if self.in_type == 0:
            self.in_dim = 3
            in_dim_exp = self.in_dim+3
        elif self.in_type == 1:
            self.in_dim = 6
            in_dim_exp = self.in_dim+3
        else:
            raise NotImplementedError('in_type not implemented')
            
        if self.out_type == 0: # delta form
            self.use_full_vertex = False
            self.out_dim = 3
        elif self.out_type == 1: # vertex (linear precision)
            self.use_full_vertex = True
            self.out_dim = 3
        elif self.out_type == 2: # transform matrix
            self.use_full_vertex = False
            self.out_dim = 9 # (6D + translation 3) will be reshaped into 3x4 matrix
            M_ = num_cage_vertices
            num_cage_vertices = num_cage_vertices * 4
            
            from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
            self._6D_to_rot_ = _6D_to_rot_
        else:
            raise NotImplementedError('out_type not implemented')
            
        
        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros

        # coordinate predictor
        self.key_weight_model = Model(
            in_dim=self.in_dim, out_dim=M, 
            use_softmax=use_softmax,
            use_relu=use_relu, # default setting
            use_least_N=use_least_N,
            use_least_N_on_V=use_least_N_on_V,
        ).to(device)

        # expression encoder
        if self.use_shp:
            self.exp_z_model = Model_mk2_1(
                in_dim=in_dim_exp,
                style_dim=L,
                out_dim=L,
                num_layers=self.num_layers, 
                use_style=True, out_type='global'
            ).to(device)
        else:
            self.exp_z_model = Model(
                in_dim=in_dim_exp,
                out_dim=L,
                num_layers=self.num_layers, 
                out_type='global'
            ).to(device)

        # shape encoder
        if self.use_shp:
            self.shape_model = Model(
                in_dim=self.in_dim,
                out_dim=L,
                out_type='global'
            ).to(device)

        # cage displacement predictor
        if self.use_shp:
            self.key_d_model = Model_mk2_1(
                in_dim=L,
                style_dim=L,
                out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
                num_layers=self.num_layers, 
                use_style=True, out_type='global'
            ).to(device)
        else:
            self.key_d_model = Model(
                in_dim=L,
                out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
                num_layers=self.num_layers, 
                out_type='global'
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
            
        delta_vert = deform_vert-source_vert
        deform_in = torch.cat([deform_in, delta_vert], dim=-1)

            
        key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
        # --> (B, N, 4M) if self.opts.out_type == 2
        
        if self.use_shp:
            z_ID_B = self.shape_model(source_in) # (B, 1, L)
            
            exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
            key_d = self.key_d_model(exp_z, z_ID_B)
        else:
            exp_z = self.exp_z_model(deform_in) # (B, 1, L)
            key_d = self.key_d_model(exp_z)
            
            
        if self.out_type == 2:
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
        

        delta_v = torch.einsum('bnc,bci->bni',key_weight,key_d)
    
        if self.use_full_vertex:
            pred_deformed = delta_v
        else:
            pred_deformed = delta_v + source_vert

        
        ## necessary ...?
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
            delta_vert_s = source_vert-source_vert
            source_in = torch.cat([source_in, delta_vert_s], dim=-1)
            
            if self.use_shp:
                exp_z_s = self.exp_z_model(source_in, z_ID_B) # (B, 1, L)    
                pred_source = self.key_d_model(exp_z_s, z_ID_B).reshape(
                    B, self.num_cage_vertices, 3
                )
            else:
                exp_z_s = self.exp_z_model(source_in) # (B, 1, L)    
                pred_source = self.key_d_model(exp_z_s).reshape(
                    B, self.num_cage_vertices, 3
                )
        else:
            pred_source = 0
            
        
        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight
            
        return pred_deformed, recon_deformed, recon_source, exp_z, pred_source

class NeuralGeneralizedBarycentricCoordinate2(nn.Module):
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
                 use_least_N_on_V=False,
                 least_number_of_zeros=256, # for sparsity (not used)
                 is_train=False,
                 tau=0.05,
                 device='cpu',
                 use_exp_recon=False, # was not necessary
                 use_shp_recon=True, # necessary for training, but not needed for inference
                 use_shp=True,
                 use_full_vertex=False, # default: false (= delta form)
                ):
        super().__init__()
        self.opts = opts
                
        self.is_train = is_train
        
        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_shp_recon = use_shp_recon        
        self.use_shp = use_shp
        if not self.use_shp:
            self.use_shp_recon = False
            print('shape model not used!, use_shp_recon set to False')
        self.use_exp_recon = use_exp_recon        
        self.use_full_vertex = use_full_vertex
        
        ###### NN input type settings
        ## key_weight model | key_d_model 
        # 0: (src_p),        (def_p, src_p)
        # 1: (src_p, src_n), (def_p, def_n, src_p, src_n)
        
        self.in_type = 1
        self.out_type = 1 # vertex
        ######
        
        if self.opts is not None:
            self.in_type = self.opts.in_type
            self.out_type = self.opts.out_type
        
        if self.in_type == 0:
            self.in_dim = 3
            in_dim_exp = self.in_dim*2
        elif self.in_type == 1:
            self.in_dim = 6
            in_dim_exp = self.in_dim*2
        else:
            raise NotImplementedError('in_type not implemented')
            
        if self.out_type == 0: # delta form
            self.use_full_vertex = False
            self.out_dim = 3
        elif self.out_type == 1: # vertex (linear precision)
            self.use_full_vertex = True
            self.out_dim = 3
        elif self.out_type == 2: # transform matrix
            self.use_full_vertex = False
            self.out_dim = 9 # (6D + translation 3) will be reshaped into 3x4 matrix
            M_ = num_cage_vertices
            num_cage_vertices = num_cage_vertices * 4
            
            from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
            self._6D_to_rot_ = _6D_to_rot_
        else:
            raise NotImplementedError('out_type not implemented')
            
        
        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros

        # coordinate predictor
        self.key_weight_model = Model(
            in_dim=self.in_dim, out_dim=M, 
            use_softmax=use_softmax,
            use_relu=use_relu, # default setting
            use_least_N=use_least_N,
            use_least_N_on_V=use_least_N_on_V,
        ).to(device)

        # expression encoder
        if self.use_shp:
            self.exp_z_model = Model_mk2_1(
                in_dim=in_dim_exp,
                style_dim=L,
                out_dim=L,
                num_layers=self.num_layers, 
                use_style=True, out_type='global'
            ).to(device)
        else:
            self.exp_z_model = Model(
                in_dim=in_dim_exp,
                out_dim=L,
                num_layers=self.num_layers, 
                out_type='global'
            ).to(device)

        # shape encoder
        if self.use_shp:
            self.shape_model = Model(
                in_dim=self.in_dim,
                out_dim=L,
                out_type='global'
            ).to(device)

        # cage displacement predictor
        if self.use_shp:
            self.key_d_model = Model_mk2_1(
                in_dim=L,
                style_dim=L,
                out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
                num_layers=self.num_layers, 
                use_style=True, out_type='global'
            ).to(device)
        else:
            self.key_d_model = Model(
                in_dim=L,
                out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
                num_layers=self.num_layers, 
                out_type='global'
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
        
    def reshape_key_d(self, key_d, B):
        if self.out_type == 2:
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
        return key_d
    
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
        deform_in = deform_vert-source_vert # as a delta
        # deform_in = deform_vert # as a vertex
        
        if self.in_dim == 6:
            source_in = torch.cat([source_in, source_norm], dim=-1)
            deform_in = torch.cat([deform_in, deform_norm], dim=-1)
            
        deform_in = torch.cat([deform_in, source_in], dim=-1)

        
        key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
        # --> (B, N, 4M) if self.opts.out_type == 2
        
        if self.use_shp:
            z_ID_B = self.shape_model(source_in) # (B, 1, L)
            
            exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
            key_d = self.key_d_model(exp_z, z_ID_B)
        else:
            exp_z = self.exp_z_model(deform_in) # (B, 1, L)
            key_d = self.key_d_model(exp_z)
        
        key_d = self.reshape_key_d(key_d, B)
            
        delta_v = torch.einsum('bnc,bci->bni',key_weight,key_d)

        
        if self.use_full_vertex:
            pred_deformed = delta_v
        else:
            pred_deformed = delta_v + source_vert
        
        
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
            source_in_s = source_vert
            deform_in_s = source_vert-source_vert # as a delta
            # deform_in_s = source_vert # as a vertex
            
            if self.in_dim == 6:
                source_in_s = torch.cat([source_in_s, source_norm], dim=-1)
                deform_in_s = torch.cat([deform_in_s, deform_norm], dim=-1)
                
            deform_in_s = torch.cat([deform_in_s, source_in_s], dim=-1)
            
            if self.use_shp:
                exp_z_s = self.exp_z_model(deform_in_s, z_ID_B) # (B, 1, L)    
                key_s = self.key_d_model(exp_z_s, z_ID_B).reshape(
                    B, self.num_cage_vertices, 3
                )
            else:
                exp_z_s = self.exp_z_model(deform_in_s) # (B, 1, L)    
                key_s = self.key_d_model(exp_z_s).reshape(
                    B, self.num_cage_vertices, 3
                )
            pred_source = torch.einsum('bnc,bci->bni',key_weight,key_s)
        else:
            pred_source = 0
        
        #verts_v_th = torch.einsum('bnc,bci->bni',key_weight,key_v)
        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight
            
        return pred_deformed, recon_deformed, recon_source, exp_z, pred_source


# class NeuralGeneralizedBarycentricCoordinate3(nn.Module):
#     """
#         Neural Generalized Barycentric Coordinate 
#         (model specialized for facial animation)
#     """
#     def __init__(self, 
#                  opts=None,
#                  in_dim=3,
#                  out_dim=3, 
#                  hid_dim=256,
#                  num_cage_vertices=512,
#                  num_layers=4,
#                  N_list=[3525, 2560, 5223], # voca biwi mf
#                  use_softmax=False,
#                  use_relu=True,
#                  use_least_N=False,
#                  use_least_N_on_V=False,
#                  least_number_of_zeros=256, # for sparsity (not used)
#                  is_train=False,
#                  tau=0.05,
#                  device='cpu',
#                  use_exp_recon=False, # was not necessary ...
#                  use_shp_recon=True, # may be..? depend on training setting... 
#                  use_shp=True, # may be..? depend on training setting... !!
#                  use_full_vertex=False, # default: false (= delta form)
#                  use_gate_layer=True,
#                 ):
#         super().__init__()
#         self.opts = opts
                
#         self.is_train = is_train
        
#         self.num_layers = num_layers
#         self.num_cage_vertices = num_cage_vertices
#         self.NZ = least_number_of_zeros
#         self.in_dim = in_dim
#         self.out_dim = out_dim

#         self.use_shp = use_shp
#         self.use_shp_recon = use_shp_recon
#         if not use_shp:
#             self.use_shp_recon = False
#         self.use_exp_recon = use_exp_recon        
#         self.use_full_vertex = use_full_vertex
        
#         ###### NN input type settings
#         # 0: (src_p),        (def_p, src_p)
#         # 1: (src_p, src_n), (def_p, def_n, src_p, src_n)        
#         self.in_type = 1
#         self.out_type = 1 # vertex
#         ######
        
#         if self.opts is not None:
#             self.in_type = self.opts.in_type
#             self.out_type = self.opts.out_type
        
#         if self.in_type == 0:
#             self.in_dim = 3
#             in_dim_exp = self.in_dim*2
#         elif self.in_type == 1:
#             self.in_dim = 6
#             in_dim_exp = self.in_dim*2
#         else:
#             raise NotImplementedError('in_type not implemented')
            
#         if self.out_type == 0: # delta form
#             self.use_full_vertex = False
#             self.out_dim = 3
#         elif self.out_type == 1: # vertex (linear precision)
#             self.use_full_vertex = True
#             self.out_dim = 3
#         elif self.out_type == 2: # transform matrix
#             self.use_full_vertex = False
#             self.out_dim = 9 # (6D + translation 3) will be reshaped into 3x4 matrix
#             M_ = num_cage_vertices
#             num_cage_vertices = num_cage_vertices * 4
            
#             from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
#             self._6D_to_rot_ = _6D_to_rot_
#         else:
#             raise NotImplementedError('out_type not implemented')
            
        
#         M = num_cage_vertices
#         L = hid_dim
#         NZ= least_number_of_zeros

#         # coordinate predictor
#         self.key_weight_model = Model(
#             in_dim=self.in_dim, out_dim=M, 
#             use_softmax=use_softmax,
#             use_relu=use_relu, # default setting
#             use_least_N=use_least_N,
#             use_least_N_on_V=use_least_N_on_V,
#             use_gate_layer=use_gate_layer,
#         ).to(device)

#         # expression encoder
#         if self.use_shp:
#             self.exp_z_model = Model_mk2_1(
#                 in_dim=in_dim_exp,
#                 style_dim=L,
#                 out_dim=L,
#                 num_layers=self.num_layers, 
#                 use_style=True, out_type='global'
#             ).to(device)
#         else:
#             self.exp_z_model = Model(
#                 in_dim=in_dim_exp,
#                 out_dim=L,
#                 num_layers=self.num_layers, 
#                 out_type='global'
#             ).to(device)

#         # shape encoder
#         if self.use_shp:
#             self.shape_model = Model(
#                 in_dim=self.in_dim,
#                 out_dim=L,
#                 out_type='global'
#             ).to(device)

#         # cage displacement predictor
#         if self.use_shp:
#             self.key_d_model = Model_mk2_1(
#                 in_dim=L,
#                 style_dim=L,
#                 out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
#                 num_layers=self.num_layers, 
#                 use_style=True, out_type='global'
#             ).to(device)
#         else:
#             self.key_d_model = Model(
#                 in_dim=L,
#                 out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
#                 num_layers=self.num_layers, 
#                 out_type='global'
#             ).to(device)
            
#         if self.is_train:
#             if self.use_exp_recon:
#                 self.recon_exp_model = nn.ModuleList([
#                     Model(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
#                     for N in N_list
#                 ]).to(device)
            
#             if self.use_shp_recon:
#                 self.recon_shp_model = nn.ModuleList([
#                     Model(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
#                     for N in N_list
#                 ]).to(device)
        
#     def reshape_key_d(self, key_d, B):
#         if self.out_type == 2:
#             # cage transform matrix
#             M_ = self.num_cage_vertices // 4
#             key_d = key_d.reshape(B, M_, 9)
#             tmp_R, tmp_t = key_d[...,:6], key_d[...,6:]
            
#             tmp_R = self._6D_to_rot_(tmp_R).reshape(B, -1, 3, 3)
#             key_d = torch.cat([tmp_R, tmp_t[..., None]], dim=-1) # (B, M, 3, 4)
#             key_d = key_d.permute(0,1,3,2).reshape(B, -1, 3) # (B, M4, 3)
#             #key_d = key_d.permute(0,3,1,2).reshape(B, -1, 3) # (B, 4M, 3)
#         else:
#             key_d = key_d.reshape(B, self.num_cage_vertices, 3)
#             # key_v = self.key_d_model(exp_z_v, z_ID_B).reshape(B, M, 3)
#         return key_d
    
#     def forward(self, source_vert, deform_vert, source_norm, deform_norm, mesh_data, epoch=0, out_kw=False):
#         """
#         Args:
#             source_vert (torch.tensor): [B, N, 3] source mesh vertices
#             deform_vert (torch.tensor): [B, N, 3] deformed mesh vertices
#             source_norm (torch.tensor): [B, N, 3] source mesh vertex normals
#             deform_norm (torch.tensor): [B, N, 3] deformed mesh vertex normals
#             mesh_data (int): indicator for data (0: voca, 1: biwi, 2: multiface)
#             epoch (int): train epoch (epoch != iteration)
#         Returns:
#             (pred_deformed, recon_deformed, recon_source):
#             predicted deformed mesh using weight and displacement,
#             reconstructed deformed mesh
#             reconstructed source mesh
#         """
#         B, N, _ = deform_vert.shape
        
#         source_in = source_vert
#         deform_in = deform_vert-source_vert # as a delta
#         # deform_in = deform_vert # as a vertex
        
#         if self.in_dim == 6:
#             source_in = torch.cat([source_in, source_norm], dim=-1)
#             deform_in = torch.cat([deform_in, deform_norm], dim=-1)
            
#         deform_in = torch.cat([deform_in, source_in], dim=-1)

        
#         key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
#         # --> (B, N, 4M) if self.opts.out_type == 2
        
#         if self.use_shp:
#             z_ID_B = self.shape_model(source_in) # (B, 1, L)
            
#             exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
#             key_d = self.key_d_model(exp_z, z_ID_B)
#         else:
#             exp_z = self.exp_z_model(deform_in) # (B, 1, L)
#             key_d = self.key_d_model(exp_z)
        
#         key_d = self.reshape_key_d(key_d, B)
            
#         delta_v = torch.einsum('bnc,bci->bni',key_weight,key_d)

        
#         if self.use_full_vertex:
#             pred_deformed = delta_v
#         else:
#             pred_deformed = delta_v + source_vert
        
        
#         ## necessary
#         if self.use_shp_recon:
#             recon_source = self.recon_shp_model[mesh_data](z_ID_B)
#             recon_source = recon_source.reshape(B, -1, 3)
#         else:
#             recon_source = 0
        
#         ## unnecessary
#         if self.use_exp_recon:
#             recon_delta_v = self.recon_exp_model[mesh_data](exp_z)
#             recon_delta_v = recon_delta_v.reshape(B, -1, 3)
#             recon_deformed = recon_delta_v + source_vert
#         else:
#             recon_deformed = 0
        
#         ## optional
#         if self.use_full_vertex:
#             source_in_s = source_vert
#             deform_in_s = source_vert-source_vert # as a delta
#             # deform_in_s = source_vert # as a vertex
            
#             if self.in_dim == 6:
#                 source_in_s = torch.cat([source_in_s, source_norm], dim=-1)
#                 deform_in_s = torch.cat([deform_in_s, deform_norm], dim=-1)
                
#             deform_in_s = torch.cat([deform_in_s, source_in_s], dim=-1)
            
#             if self.use_shp:
#                 exp_z_s = self.exp_z_model(deform_in_s, z_ID_B) # (B, 1, L)    
#                 key_s = self.key_d_model(exp_z_s, z_ID_B).reshape(
#                     B, self.num_cage_vertices, 3
#                 )
#             else:
#                 exp_z_s = self.exp_z_model(deform_in_s) # (B, 1, L)    
#                 key_s = self.key_d_model(exp_z_s).reshape(
#                     B, self.num_cage_vertices, 3
#                 )
#             pred_source = torch.einsum('bnc,bci->bni',key_weight,key_s)
#         else:
#             pred_source = 0
        
#         #verts_v_th = torch.einsum('bnc,bci->bni',key_weight,key_v)
#         if out_kw:
#             return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight
            
#         return pred_deformed, recon_deformed, recon_source, exp_z, pred_source


class NeuralGeneralizedBarycentricCoordinate4(nn.Module):
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
                 use_elu=False,
                 use_least_N=False,
                 use_least_N_on_V=False,
                 least_number_of_zeros=256, # for sparsity (not used)
                 is_train=False,
                 tau=0.05,
                 device='cpu',
                 use_exp_recon=False, # was not necessary
                 use_shp_recon=True, # necessary for training, but not needed for inference
                 use_shp=True,
                 use_full_vertex=False, # default: false (= delta form)
                ):
        super().__init__()
        self.opts = opts
                
        self.is_train = is_train
        
        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_shp_recon = use_shp_recon
        self.use_shp = use_shp
        if not self.use_shp:
            self.use_shp_recon = False
            print('shape model not used!, use_shp_recon set to False')
        self.use_exp_recon = use_exp_recon        
        self.use_full_vertex = use_full_vertex
        
        ###### NN input type settings
        # 0: (src_p),        (def_p, src_p)
        # 1: (src_p, src_n), (def_p, def_n, src_p, src_n)        
        self.in_type = 1
        self.out_type = 1 # vertex
        ######
        
        if self.opts is not None:
            self.in_type = self.opts.in_type
            self.out_type = self.opts.out_type
        
        if self.in_type == 0:
            self.in_dim = 3
            in_dim_exp = self.in_dim*2
        elif self.in_type == 1:
            self.in_dim = 6
            in_dim_exp = self.in_dim*2
        else:
            raise NotImplementedError('in_type not implemented')
            
        if self.out_type == 0: # delta form
            self.use_full_vertex = False
            self.out_dim = 3
        elif self.out_type == 1: # vertex (linear precision)
            self.use_full_vertex = True
            self.out_dim = 3
        elif self.out_type == 2: # transform matrix
            self.use_full_vertex = False
            self.out_dim = 9 # (6D + translation 3) will be reshaped into 3x4 matrix
            M_ = num_cage_vertices
            num_cage_vertices = num_cage_vertices * 4
            
            from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
            self._6D_to_rot_ = _6D_to_rot_
        else:
            raise NotImplementedError('out_type not implemented')

        
        from models.encoder import PointNet_small, PointNet_large
        from pointnet_utils import feature_transform_reguliarzer #, PointNetEncoder
        self.ft_reg = feature_transform_reguliarzer
        
        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros
                
        # coordinate predictor
        self.key_weight_model = PointNet_small(
            in_dim=self.in_dim, out_dim=M, 
            use_softmax=use_softmax,
            use_relu=use_relu, # default setting
            use_elu=use_elu,
            use_least_N=use_least_N,
            use_least_N_on_V=use_least_N_on_V,
            out_type='vertices',
        ).to(device)

        # expression encoder
        if self.use_shp:
            self.exp_z_model = Model_mk2_1(
                in_dim=in_dim_exp,
                style_dim=L,
                out_dim=L,
                num_layers=self.num_layers, 
                use_style=True, out_type='global'
            ).to(device)
        else:
            self.exp_z_model = PointNet_small(
                in_dim=in_dim_exp,
                out_dim=L,
                out_type='global'
            ).to(device)

        # shape encoder
        if self.use_shp:
            self.shape_model = PointNet_small(
                in_dim=self.in_dim,
                out_dim=L,
                out_type='global'
            ).to(device)

        # cage displacement predictor
        if self.use_shp:
            self.key_d_model = Model_mk2_1(
                in_dim=L,
                style_dim=L,
                out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
                num_layers=self.num_layers, 
                use_style=True, out_type='global'
            ).to(device)
        else:
            self.key_d_model = PointNet_large(
                in_dim=L,
                out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
                out_type='global'
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
        
    def reshape_key_d(self, key_d, B):
        if self.out_type == 2:
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
        return key_d
    
    def forward(self, source_vert, deform_vert, source_norm, deform_norm, mesh_data, epoch=0, out_kw=False, recon_out=True):
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
        deform_in = deform_vert-source_vert # as a delta
        # deform_in = deform_vert # as a vertex
        
        if self.in_dim == 6:
            source_in = torch.cat([source_in, source_norm], dim=-1)
            deform_in = torch.cat([deform_in, deform_norm], dim=-1)
            
        deform_in = torch.cat([deform_in, source_in], dim=-1)

        
        key_weight, trans_feat = self.key_weight_model(source_in, N=self.NZ, return_all=True) # (B, N, M)
        trans_feat_loss = self.ft_reg(trans_feat)
        # --> (B, N, 4M) if self.opts.out_type == 2
        
        if self.use_shp:
            z_ID_B, z_trans_fest = self.shape_model(source_in, return_all=True) # (B, 1, L)
            trans_feat_loss += self.ft_reg(z_trans_fest)
            
            exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
            key_d = self.key_d_model(exp_z, z_ID_B)
        else:
            exp_z, e_trans_feat = self.exp_z_model(deform_in, return_all=True) # (B, 1, L)
            key_d, d_trans_feat = self.key_d_model(exp_z, return_all=True)

            trans_feat_loss += self.ft_reg(e_trans_feat)
            trans_feat_loss += self.ft_reg(d_trans_feat)
        
        key_d = self.reshape_key_d(key_d, B)
            
        delta_v = torch.einsum('bnc,bci->bni',key_weight,key_d)

        
        if self.use_full_vertex:
            pred_deformed = delta_v
        else:
            pred_deformed = delta_v + source_vert
        
        
        ## necessary ?
        if self.use_shp_recon and recon_out:
            recon_source = self.recon_shp_model[mesh_data](z_ID_B)
            recon_source = recon_source.reshape(B, -1, 3)
        else:
            recon_source = 0
        
        ## unnecessary
        if self.use_exp_recon and recon_out:
            recon_delta_v = self.recon_exp_model[mesh_data](exp_z)
            recon_delta_v = recon_delta_v.reshape(B, -1, 3)
            recon_deformed = recon_delta_v + source_vert
        else:
            recon_deformed = 0
        
        
        #verts_v_th = torch.einsum('bnc,bci->bni',key_weight,key_v)
        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight
            
        return pred_deformed, recon_deformed, recon_source, exp_z, trans_feat_loss



class NeuralGeneralizedBarycentricCoordinate5(nn.Module):
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
                 use_relu=False,
                 use_elu=True,
                 use_least_N=False,
                 use_least_N_on_V=False,
                 least_number_of_zeros=256, # for sparsity (not used)
                 is_train=False,
                 tau=0.05,
                 device='cpu',
                 use_exp_recon=False, # was not necessary
                 use_shp_recon=True, # necessary for training, but not needed for inference
                 use_shp=True,
                 use_full_vertex=False, # default: false (= delta form)
                ):
        super().__init__()
        self.opts = opts
                
        self.is_train = is_train
        
        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_shp_recon = use_shp_recon        
        self.use_shp = use_shp
        if not self.use_shp:
            self.use_shp_recon = False
            print('shape model not used!, use_shp_recon set to False')
        self.use_exp_recon = use_exp_recon        
        self.use_full_vertex = use_full_vertex
        
        ###### NN input type settings
        ## key_weight model | key_d_model 
        # 0: (src_p),        (def_p, src_p)
        # 1: (src_p, src_n), (def_p, def_n, src_p, src_n)
        
        self.in_type = 1
        self.out_type = 1 # vertex
        ######
        
        if self.opts is not None:
            self.in_type = self.opts.in_type
            self.out_type = self.opts.out_type
        
        if self.in_type == 0:
            self.in_dim = 3
            in_dim_exp = self.in_dim*2
        elif self.in_type == 1:
            self.in_dim = 6
            in_dim_exp = self.in_dim*2
        else:
            raise NotImplementedError('in_type not implemented')
            
        if self.out_type == 0: # delta form
            self.use_full_vertex = False
            self.out_dim = 3
        elif self.out_type == 1: # vertex (linear precision)
            self.use_full_vertex = True
            self.out_dim = 3
        elif self.out_type == 2: # transform matrix
            self.use_full_vertex = False
            self.out_dim = 9 # (6D + translation 3) will be reshaped into 3x4 matrix
            M_ = num_cage_vertices
            num_cage_vertices = num_cage_vertices * 4
            
            from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
            self._6D_to_rot_ = _6D_to_rot_
        else:
            raise NotImplementedError('out_type not implemented')
            
        from models.encoder import LinearEncoder
        
        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros

        # coordinate predictor
        self.key_weight_model = LinearEncoder(
            in_dim=self.in_dim, out_dim=M, 
            use_softmax=use_softmax,
            use_relu=use_relu, # default setting
            use_elu=use_elu,
            use_least_N=use_least_N,
            use_least_N_on_V=use_least_N_on_V,
        ).to(device)

        # expression encoder
        if self.use_shp:
            self.exp_z_model = Model_mk2_1(
                in_dim=in_dim_exp,
                style_dim=L,
                out_dim=L,
                num_layers=self.num_layers, 
                use_style=True, 
                out_type='global',
            ).to(device)
        else:
            self.exp_z_model = LinearEncoder(
                in_dim=in_dim_exp,
                out_dim=L,
                num_layers=self.num_layers, 
                out_type='global',
            ).to(device)

        # shape encoder
        if self.use_shp:
            self.shape_model = LinearEncoder(
                in_dim=self.in_dim,
                out_dim=L,
                out_type='global'
            ).to(device)

        # cage displacement predictor
        if self.use_shp:
            self.key_d_model = Model_mk2_1(
                in_dim=L,
                style_dim=L,
                out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
                num_layers=self.num_layers, 
                use_style=True, out_type='global'
            ).to(device)
        else:
            self.key_d_model = LinearEncoder(
                in_dim=L,
                out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
                num_layers=self.num_layers, 
                out_type='global'
            ).to(device)

        if self.is_train:
            if self.use_exp_recon:
                self.recon_exp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
            
            if self.use_shp_recon:
                self.recon_shp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
        
    def reshape_key_d(self, key_d, B):
        if self.out_type == 2:
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
        return key_d
    
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
        deform_in = deform_vert-source_vert # as a delta
        # deform_in = deform_vert # as a vertex
        
        if self.in_dim == 6:
            source_in = torch.cat([source_in, source_norm], dim=-1)
            deform_in = torch.cat([deform_in, deform_norm], dim=-1)
            
        deform_in = torch.cat([deform_in, source_in], dim=-1)

        
        key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
        # --> (B, N, 4M) if self.opts.out_type == 2
        
        if self.use_shp:
            z_ID_B = self.shape_model(source_in) # (B, 1, L)
            
            exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
            key_d = self.key_d_model(exp_z, z_ID_B)
        else:
            exp_z = self.exp_z_model(deform_in) # (B, 1, L)
            key_d = self.key_d_model(exp_z)
        
        key_d = self.reshape_key_d(key_d, B)
            
        delta_v = torch.einsum('bnc,bci->bni',key_weight,key_d)

        
        if self.use_full_vertex:
            pred_deformed = delta_v
        else:
            pred_deformed = delta_v + source_vert
        
        
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
            source_in_s = source_vert
            deform_in_s = source_vert-source_vert # as a delta
            # deform_in_s = source_vert # as a vertex
            
            if self.in_dim == 6:
                source_in_s = torch.cat([source_in_s, source_norm], dim=-1)
                deform_in_s = torch.cat([deform_in_s, deform_norm], dim=-1)
                
            deform_in_s = torch.cat([deform_in_s, source_in_s], dim=-1)
            
            if self.use_shp:
                exp_z_s = self.exp_z_model(deform_in_s, z_ID_B) # (B, 1, L)    
                key_s = self.key_d_model(exp_z_s, z_ID_B)            
            else:
                exp_z_s = self.exp_z_model(deform_in_s) # (B, 1, L)    
                key_s = self.key_d_model(exp_z_s)
            key_s = self.reshape_key_d(key_s, B) # (B, M, 3)
                
            pred_source = torch.einsum('bnc,bci->bni',key_weight,key_s)
        else:
            pred_source = 0
        
        #verts_v_th = torch.einsum('bnc,bci->bni',key_weight,key_v)
        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight
            
        return pred_deformed, recon_deformed, recon_source, exp_z, pred_source

    #@torch.no_grad()
    def retarget(self, 
                 src_neu_vert, src_neu_norm, src_def_vert, src_def_norm, tgt_neu_vert, tgt_neu_norm,
                 mesh_data=0, out_kw=False, recon_out=True):
        """
        Args:
            src_neu_vert (torch.tensor): [B, N, 3] source neutral mesh vertex positions
            src_neu_norm (torch.tensor): [B, N, 3] source neutral mesh vertex normals
            
            src_def_vert (torch.tensor): [B, N, 3] source deformed mesh vertex positions
            src_def_norm (torch.tensor): [B, N, 3] source deformed mesh vertex normals
            
            tgt_neu_vert (torch.tensor): [B, M, 3] target neutral mesh vertex positions
            tgt_neu_norm (torch.tensor): [B, M, 3] target neutral mesh vertex normals
            
            mesh_data (int): indicator for data (0: voca, 1: biwi, 2: multiface) -- not used!
            
        Returns:
            (pred_deformed, pred_source):
            predicted target deformation and neutral mesh using weight and cage prediction
        """
        B, N, _ = src_def_vert.shape
        
        tgt_in = tgt_neu_vert        
        src_in = src_neu_vert
        
        deform_in_d = src_def_vert-src_neu_vert # as a delta
        deform_in_s = src_neu_vert-src_neu_vert # as a delta
                
        if self.in_dim == 6:
            tgt_in = torch.cat([tgt_in, tgt_neu_norm], dim=-1)
            src_in = torch.cat([src_in, src_neu_norm], dim=-1)
            deform_in_d = torch.cat([deform_in_d, src_def_norm], dim=-1)
            deform_in_s = torch.cat([deform_in_s, src_def_norm], dim=-1)
            
        deform_in_d = torch.cat([deform_in_d, src_in], dim=-1)
        deform_in_s = torch.cat([deform_in_s, src_in], dim=-1)

        with torch.no_grad():
            key_weight = self.key_weight_model(tgt_in, N=self.NZ) # (B, N, M)
            # --> (B, N, 4M) if self.opts.out_type == 2
            
            if self.use_shp:
                z_ID_B = self.shape_model(src_in) # (B, 1, L)
                
                exp_z_d = self.exp_z_model(deform_in_d, z_ID_B) # (B, 1, L)
                key_d = self.key_d_model(exp_z_d, z_ID_B)
                
                exp_z_s = self.exp_z_model(deform_in_s, z_ID_B) # (B, 1, L)
                key_s = self.key_d_model(exp_z_s, z_ID_B)
            else:
                exp_z_d = self.exp_z_model(deform_in_d) # (B, 1, L)
                key_d = self.key_d_model(exp_z_d)
                
                exp_z_s = self.exp_z_model(deform_in_s) # (B, 1, L)
                key_s = self.key_d_model(exp_z_s)
            
            key_d = self.reshape_key_d(key_d, B)
            key_s = self.reshape_key_d(key_s, B)
                
            delta_dv = torch.einsum('bnc,bci->bni',key_weight,key_d)
            delta_sv = torch.einsum('bnc,bci->bni',key_weight,key_s)
        
        
        if self.use_full_vertex:
            pred_deformed = delta_dv
            pred_source = delta_sv
        else:
            pred_deformed = delta_dv + source_vert
            pred_source = delta_sv + source_vert
        
        # supple networks -------------------------------------
        # ## necessary -- not really...
        # if self.use_shp_recon and recon_out:
        #     recon_source = self.recon_shp_model[mesh_data](z_ID_B)
        #     recon_source = recon_source.reshape(B, -1, 3)
        # else:
        #     recon_source = 0
        
        # ## unnecessary
        # if self.use_exp_recon and recon_out:
        #     recon_delta_v = self.recon_exp_model[mesh_data](exp_z_d)
        #     recon_delta_v = recon_delta_v.reshape(B, -1, 3)
        #     recon_deformed = recon_delta_v + source_vert
        # else:
        #     recon_deformed = 0
        # -----------------------------------------------------        
        
        if out_kw:
            return pred_deformed, pred_source, exp_z_d, key_d, exp_z_s, key_s, key_weight
        
        torch.cuda.empty.cache()
        return pred_deformed, pred_source

class NeuralGeneralizedBarycentricCoordinate51(nn.Module):
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
                 use_relu=False,
                 use_elu=True,
                 use_least_N=False,
                 use_least_N_on_V=False,
                 least_number_of_zeros=256, # for sparsity (not used)
                 is_train=False,
                 tau=0.05,
                 device='cpu',
                 use_exp_recon=False, # was not necessary
                 use_shp_recon=True, # necessary for training, but not needed for inference
                 use_shp=True,
                 use_full_vertex=False, # default: false (= delta form)
                ):
        super().__init__()
        self.opts = opts
                
        self.is_train = is_train
        
        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim

        # self.use_shp_recon = use_shp_recon        
        self.use_shp = True
        print('shape model used! use_shp_recon also set to True')
        self.use_shp_recon = True        
        
        # if not self.use_shp:
        #     self.use_shp_recon = False
        #     print('shape model not used!, use_shp_recon set to False')
        self.use_exp_recon = use_exp_recon        
        self.use_full_vertex = use_full_vertex
        
        ###### NN input type settings
        ## key_weight model | key_d_model 
        # 0: (src_p),        (def_p, src_p)
        # 1: (src_p, src_n), (def_p, def_n, src_p, src_n)
        
        self.in_type = 1
        self.out_type = 1 # vertex
        ######
        
        if self.opts is not None:
            self.in_type = self.opts.in_type
            self.out_type = self.opts.out_type
        
        if self.in_type == 0:
            self.in_dim = 3
            in_dim_exp = self.in_dim*2
        elif self.in_type == 1:
            self.in_dim = 6
            in_dim_exp = self.in_dim*2
        else:
            raise NotImplementedError('in_type not implemented')
            
        if self.out_type == 0: # delta form
            self.use_full_vertex = False
            self.out_dim = 3
        elif self.out_type == 1: # vertex (linear precision)
            self.use_full_vertex = True
            self.out_dim = 3
        elif self.out_type == 2: # transform matrix
            self.use_full_vertex = False
            self.out_dim = 9 # (6D + translation 3) will be reshaped into 3x4 matrix
            M_ = num_cage_vertices
            num_cage_vertices = num_cage_vertices * 4
            
            from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
            self._6D_to_rot_ = _6D_to_rot_
        else:
            raise NotImplementedError('out_type not implemented')
            
        from models.encoder import LinearEncoder, LinearEncoder2
        
        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros

        # coordinate predictor
        self.key_weight_model = LinearEncoder2(
            in_dim=self.in_dim,
            style_dim=L,
            out_dim=M, 
            use_softmax=use_softmax,
            use_relu=use_relu, # default setting
            use_elu=use_elu,
            use_least_N=use_least_N,
            use_least_N_on_V=use_least_N_on_V,
        ).to(device)

        # expression encoder
        # if self.use_shp:
        #     self.exp_z_model = Model_mk2_1(
        #         in_dim=in_dim_exp,
        #         style_dim=L,
        #         out_dim=L,
        #         num_layers=self.num_layers, 
        #         use_style=True, 
        #         out_type='global',
        #     ).to(device)
        # else:
        #     self.exp_z_model = LinearEncoder(
        #         in_dim=in_dim_exp,
        #         out_dim=L,
        #         num_layers=self.num_layers, 
        #         out_type='global',
        #     ).to(device)
        self.exp_z_model = LinearEncoder(
            in_dim=in_dim_exp,
            out_dim=L,
            num_layers=self.num_layers, 
            out_type='global',
        ).to(device)

        # cage displacement predictor
        # if self.use_shp:
        #     self.key_d_model = Model_mk2_1(
        #         in_dim=L,
        #         style_dim=L,
        #         out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
        #         num_layers=self.num_layers, 
        #         use_style=True, out_type='global'
        #     ).to(device)
        # else:
        #     self.key_d_model = LinearEncoder(
        #         in_dim=L,
        #         out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
        #         num_layers=self.num_layers, 
        #         out_type='global'
        #     ).to(device)
        self.key_d_model = LinearEncoder(
            in_dim=L,
            out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
            num_layers=self.num_layers, 
            out_type='global'
        ).to(device)
        
        # shape encoder
        if self.use_shp:
            self.shape_model = LinearEncoder(
                in_dim=self.in_dim,
                out_dim=L,
                out_type='global'
            ).to(device)
            
        if self.is_train:
            if self.use_exp_recon:
                self.recon_exp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
            
            if self.use_shp_recon:
                self.recon_shp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
        
    def reshape_key_d(self, key_d, B):
        if self.out_type == 2:
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
        return key_d
    
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
        deform_in = deform_vert-source_vert # as a delta
        # deform_in = deform_vert # as a vertex
        
        if self.in_dim == 6:
            source_in = torch.cat([source_in, source_norm], dim=-1)
            deform_in = torch.cat([deform_in, deform_norm], dim=-1)
            
        deform_in = torch.cat([deform_in, source_in], dim=-1)
        
        # if self.use_shp:
        #     z_ID_B = self.shape_model(source_in) # (B, 1, L)
        #     exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
        #     key_d = self.key_d_model(exp_z, z_ID_B)
        # else:
        #     exp_z = self.exp_z_model(deform_in) # (B, 1, L)
        #     key_d = self.key_d_model(exp_z)
        
        exp_z = self.exp_z_model(deform_in) # (B, 1, L)
        key_d = self.key_d_model(exp_z)
    
        key_d = self.reshape_key_d(key_d, B)
            
        
        z_ID_B = self.shape_model(source_in) # (B, 1, L)
        key_weight = self.key_weight_model(source_in, z_ID_B, N=self.NZ) # (B, N, M)
        # --> (B, N, 4M) if self.opts.out_type == 2
        
        delta_v = torch.einsum('bnc,bci->bni',key_weight,key_d)

        
        if self.use_full_vertex:
            pred_deformed = delta_v
        else:
            pred_deformed = delta_v + source_vert
        
        
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
            source_in_s = source_vert
            deform_in_s = source_vert-source_vert # as a delta
            # deform_in_s = source_vert # as a vertex
            
            if self.in_dim == 6:
                source_in_s = torch.cat([source_in_s, source_norm], dim=-1)
                deform_in_s = torch.cat([deform_in_s, deform_norm], dim=-1)
                
            deform_in_s = torch.cat([deform_in_s, source_in_s], dim=-1)
            
            # if self.use_shp:
            #     exp_z_s = self.exp_z_model(deform_in_s, z_ID_B) # (B, 1, L)    
            #     key_s = self.key_d_model(exp_z_s, z_ID_B).reshape(
            #         B, self.num_cage_vertices, 3
            #     )
            # else:
            #     exp_z_s = self.exp_z_model(deform_in_s) # (B, 1, L)    
            #     key_s = self.key_d_model(exp_z_s).reshape(
            #         B, self.num_cage_vertices, 3
            #     )
            exp_z_s = self.exp_z_model(deform_in_s) # (B, 1, L)    
            key_s = self.key_d_model(exp_z_s).reshape(
                B, self.num_cage_vertices, 3
            )
            pred_source = torch.einsum('bnc,bci->bni',key_weight,key_s)
        else:
            pred_source = 0
        
        #verts_v_th = torch.einsum('bnc,bci->bni',key_weight,key_v)
        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight
            
        return pred_deformed, recon_deformed, recon_source, exp_z, pred_source

    #@torch.no_grad()
    def retarget(self, 
                 src_neu_vert, src_neu_norm, src_def_vert, src_def_norm, tgt_neu_vert, tgt_neu_norm,
                 mesh_data=0, out_kw=False, recon_out=True):
        """
        Args:
            src_neu_vert (torch.tensor): [B, N, 3] source neutral mesh vertex positions
            src_neu_norm (torch.tensor): [B, N, 3] source neutral mesh vertex normals
            
            src_def_vert (torch.tensor): [B, N, 3] source deformed mesh vertex positions
            src_def_norm (torch.tensor): [B, N, 3] source deformed mesh vertex normals
            
            tgt_neu_vert (torch.tensor): [B, M, 3] target neutral mesh vertex positions
            tgt_neu_norm (torch.tensor): [B, M, 3] target neutral mesh vertex normals
            
            mesh_data (int): indicator for data (0: voca, 1: biwi, 2: multiface) -- not used!
            
        Returns:
            (pred_deformed, pred_source):
            predicted target deformation and neutral mesh using weight and cage prediction
        """
        B, N, _ = src_def_vert.shape
        
        tgt_in = tgt_neu_vert        
        src_in = src_neu_vert
        
        deform_in_d = src_def_vert-src_neu_vert # as a delta
        deform_in_s = src_neu_vert-src_neu_vert # as a delta
                
        if self.in_dim == 6:
            tgt_in = torch.cat([tgt_in, tgt_neu_norm], dim=-1)
            src_in = torch.cat([src_in, src_neu_norm], dim=-1)
            deform_in_d = torch.cat([deform_in_d, src_def_norm], dim=-1)
            deform_in_s = torch.cat([deform_in_s, src_def_norm], dim=-1)
            
        deform_in_d = torch.cat([deform_in_d, src_in], dim=-1)
        deform_in_s = torch.cat([deform_in_s, src_in], dim=-1)

        with torch.no_grad():            
            z_ID_B = self.shape_model(tgt_in) # (B, 1, L)
            key_weight = self.key_weight_model(tgt_in, z_ID_B, N=self.NZ) # (B, N, M)
            # --> (B, N, 4M) if self.opts.out_type == 2
            
            exp_z_d = self.exp_z_model(deform_in_d) # (B, 1, L)
            key_d = self.key_d_model(exp_z_d)
            
            exp_z_s = self.exp_z_model(deform_in_s) # (B, 1, L)
            key_s = self.key_d_model(exp_z_s)
        
            key_d = self.reshape_key_d(key_d, B)
            key_s = self.reshape_key_d(key_s, B)
                
            delta_dv = torch.einsum('bnc,bci->bni',key_weight,key_d)
            delta_sv = torch.einsum('bnc,bci->bni',key_weight,key_s)
            
        if self.use_full_vertex:
            pred_deformed = delta_dv
            pred_source = delta_sv
        else:
            pred_deformed = delta_dv + source_vert
            pred_source = delta_sv + source_vert
        
        # supple networks -------------------------------------
        # ## necessary -- not really...
        # if self.use_shp_recon and recon_out:
        #     recon_source = self.recon_shp_model[mesh_data](z_ID_B)
        #     recon_source = recon_source.reshape(B, -1, 3)
        # else:
        #     recon_source = 0
        
        # ## unnecessary
        # if self.use_exp_recon and recon_out:
        #     recon_delta_v = self.recon_exp_model[mesh_data](exp_z_d)
        #     recon_delta_v = recon_delta_v.reshape(B, -1, 3)
        #     recon_deformed = recon_delta_v + source_vert
        # else:
        #     recon_deformed = 0
        # -----------------------------------------------------        
        
        if out_kw:
            return pred_deformed, pred_source, exp_z_d, key_d, exp_z_s, key_s, key_weight

        torch.cuda.empty.cache()
        return pred_deformed, pred_source

class NeuralGeneralizedBarycentricCoordinate6(nn.Module):
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
                 use_relu=False,
                 use_elu=True,
                 use_least_N=False,
                 use_least_N_on_V=False,
                 least_number_of_zeros=256, # for sparsity (not used)
                 is_train=False,
                 tau=0.05,
                 device='cpu',
                 use_exp_recon=False, # was not necessary
                 use_shp_recon=True, # necessary for training, but not needed for inference
                 use_shp=True,
                 use_full_vertex=False, # default: false (= delta form)
                ):
        super().__init__()
        self.opts = opts
                
        self.is_train = is_train
        
        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_shp_recon = use_shp_recon        
        self.use_shp = use_shp
        if not self.use_shp:
            self.use_shp_recon = False
            print('shape model not used!, use_shp_recon set to False')
        self.use_exp_recon = use_exp_recon        
        self.use_full_vertex = use_full_vertex
        
        ###### NN input type settings
        ## key_weight model | key_d_model 
        # 0: (src_p),        (def_p, src_p)
        # 1: (src_p, src_n), (def_p, def_n, src_p, src_n)
        
        self.in_type = 1
        self.out_type = 1 # vertex
        ######
        
        if self.opts is not None:
            self.in_type = self.opts.in_type
            self.out_type = self.opts.out_type
        
        if self.in_type == 0:
            self.in_dim = 3
            in_dim_exp = self.in_dim*2
        elif self.in_type == 1:
            self.in_dim = 6
            in_dim_exp = self.in_dim*2
        else:
            raise NotImplementedError('in_type not implemented')
            
        if self.out_type == 0: # delta form
            self.use_full_vertex = False
            self.out_dim = 3
        elif self.out_type == 1: # vertex (linear precision)
            self.use_full_vertex = True
            self.out_dim = 3
        elif self.out_type == 2: # transform matrix
            self.use_full_vertex = False
            self.out_dim = 9 # (6D + translation 3) will be reshaped into 3x4 matrix
            M_ = num_cage_vertices
            num_cage_vertices = num_cage_vertices * 4
            
            from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
            self._6D_to_rot_ = _6D_to_rot_
        else:
            raise NotImplementedError('out_type not implemented')
            
        from models.encoder import LinearEncoder
        
        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros

        # coordinate predictor
        self.key_weight_model = LinearEncoder(
            in_dim=self.in_dim, out_dim=M, 
            use_softmax=use_softmax,
            use_relu=use_relu, # default setting
            use_elu=use_elu,
            use_least_N=use_least_N,
            use_least_N_on_V=use_least_N_on_V,
        ).to(device)

        # expression encoder
        if self.use_shp:
            self.exp_z_model = Model_mk2_1(
                in_dim=in_dim_exp,
                style_dim=L,
                out_dim=L,
                num_layers=self.num_layers, 
                use_style=True, 
                out_type='global',
            ).to(device)
        else:
            self.exp_z_model = LinearEncoder(
                in_dim=in_dim_exp,
                out_dim=L,
                num_layers=self.num_layers, 
                out_type='global',
            ).to(device)

        # shape encoder
        if self.use_shp:
            self.shape_model = LinearEncoder(
                in_dim=self.in_dim,
                out_dim=L,
                out_type='global'
            ).to(device)

        # cage displacement predictor
        if self.use_shp:
            self.key_d_model = Model_mk2_1(
                in_dim=L,
                style_dim=L,
                out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
                num_layers=self.num_layers, 
                use_style=True, out_type='global'
            ).to(device)
        else:
            self.key_d_model = LinearEncoder(
                in_dim=L,
                out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
                num_layers=self.num_layers, 
                out_type='global'
            ).to(device)

        if self.is_train:
            if self.use_exp_recon:
                self.recon_exp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
            
            if self.use_shp_recon:
                self.recon_shp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
        
    def reshape_key_d(self, key_d, B):
        if self.out_type == 2:
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
        return key_d
    
    def forward(self, source_vert, deform_vert, source_norm, deform_norm, mesh_data, epoch=0, out_kw=False, recon_out=True):
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
        deform_in = deform_vert-source_vert # as a delta
        # deform_in = deform_vert # as a vertex
        
        if self.in_dim == 6:
            source_in = torch.cat([source_in, source_norm], dim=-1)
            deform_in = torch.cat([deform_in, deform_norm], dim=-1)
            
        deform_in = torch.cat([deform_in, source_in], dim=-1)

        
        key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
        # --> (B, N, 4M) if self.opts.out_type == 2
        
        if self.use_shp:
            z_ID_B = self.shape_model(source_in) # (B, 1, L)
            
            exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
            key_d = self.key_d_model(exp_z, z_ID_B)
        else:
            exp_z = self.exp_z_model(deform_in) # (B, 1, L)
            key_d = self.key_d_model(exp_z)
        
        key_d = self.reshape_key_d(key_d, B) # (B, M, 3)
            
        delta_v = torch.einsum('bnc,bci->bni',key_weight,key_d) # (B, N, 3)

        
        if self.use_full_vertex:
            pred_deformed = delta_v
        else:
            pred_deformed = delta_v + source_vert
        
        # supple networks -------------------------------------
        ## necessary
        if self.use_shp_recon and recon_out:
            recon_source = self.recon_shp_model[mesh_data](z_ID_B)
            recon_source = recon_source.reshape(B, -1, 3)
        else:
            recon_source = 0
        
        ## unnecessary
        if self.use_exp_recon and recon_out:
            recon_delta_v = self.recon_exp_model[mesh_data](exp_z)
            recon_delta_v = recon_delta_v.reshape(B, -1, 3)
            recon_deformed = recon_delta_v + source_vert
        else:
            recon_deformed = 0
        # -----------------------------------------------------        
        
        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight
            
        return pred_deformed, recon_deformed, recon_source, exp_z, key_d

class NeuralGeneralizedBarycentricCoordinate7(nn.Module):
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
                 use_relu=False,
                 use_elu=True,
                 use_least_N=False,
                 use_least_N_on_V=False,
                 least_number_of_zeros=256, # for sparsity (not used)
                 is_train=False,
                 tau=0.05,
                 device='cpu',
                 use_exp_recon=False, # was not necessary
                 use_shp_recon=True, # necessary for training, but not needed for inference
                 use_shp=True,
                 use_full_vertex=False, # default: false (= delta form)
                ):
        super().__init__()
        self.opts = opts
                
        self.is_train = is_train
        
        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_shp_recon = use_shp_recon        
        self.use_shp = use_shp
        if not self.use_shp:
            self.use_shp_recon = False
            print('shape model not used!, use_shp_recon set to False')
        self.use_exp_recon = use_exp_recon        
        self.use_full_vertex = use_full_vertex
        
        ###### NN input type settings
        ## key_weight model | key_d_model 
        # 0: (src_p),        (def_p, src_p)
        # 1: (src_p, src_n), (def_p, def_n, src_p, src_n)
        
        self.in_type = 1
        self.out_type = 1 # vertex
        ######
        
        if self.opts is not None:
            self.in_type = self.opts.in_type
            self.out_type = self.opts.out_type
        
        if self.in_type == 0:
            self.in_dim = 3
            in_dim_exp = self.in_dim*2
        elif self.in_type == 1:
            self.in_dim = 6
            in_dim_exp = self.in_dim*2
        else:
            raise NotImplementedError('in_type not implemented')
            
        if self.out_type == 0: # delta form
            self.use_full_vertex = False
            self.out_dim = 3
        elif self.out_type == 1: # vertex (linear precision)
            self.use_full_vertex = True
            self.out_dim = 3
        elif self.out_type == 2: # transform matrix
            self.use_full_vertex = False
            self.out_dim = 9 # (6D + translation 3) will be reshaped into 3x4 matrix
            M_ = num_cage_vertices
            num_cage_vertices = num_cage_vertices * 4
            
            from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
            self._6D_to_rot_ = _6D_to_rot_
        else:
            raise NotImplementedError('out_type not implemented')
            
        from models.encoder import LinearEncoder
        
        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros

        # coordinate predictor
        self.key_weight_model = LinearEncoder(
            # in_dim=self.in_dim,
            in_dim=in_dim_exp,
            out_dim=M, 
            use_softmax=use_softmax,
            use_relu=use_relu, # default setting
            use_elu=use_elu,
            use_least_N=use_least_N,
            use_least_N_on_V=use_least_N_on_V,
        ).to(device)

        # expression encoder
        if self.use_shp:
            self.exp_z_model = Model_mk2_1(
                in_dim=in_dim_exp,
                style_dim=L,
                out_dim=L,
                num_layers=self.num_layers, 
                use_style=True, 
                out_type='global',
            ).to(device)
        else:
            self.exp_z_model = LinearEncoder(
                in_dim=in_dim_exp,
                out_dim=L,
                num_layers=self.num_layers, 
                out_type='global',
            ).to(device)

        # shape encoder
        if self.use_shp:
            self.shape_model = LinearEncoder(
                in_dim=self.in_dim,
                out_dim=L,
                out_type='global'
            ).to(device)

        # cage displacement predictor (predict both original cage and deformed cage)
        if self.use_shp:
            self.key_d_model = Model_mk2_1(
                in_dim=L,
                style_dim=L,
                out_dim= M_*self.out_dim*2 if self.out_type == 2 else M*self.out_dim*2,
                num_layers=self.num_layers, 
                use_style=True, out_type='global'
            ).to(device)
        else:
            self.key_d_model = LinearEncoder(
                in_dim=L,
                out_dim= M_*self.out_dim*2 if self.out_type == 2 else M*self.out_dim*2,
                num_layers=self.num_layers, 
                out_type='global'
            ).to(device)

        if self.is_train:
            if self.use_exp_recon:
                self.recon_exp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
            
            if self.use_shp_recon:
                self.recon_shp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
        
    def reshape_key_d(self, key_d, B):
        if self.out_type == 2:
            # cage transform matrix
            M_ = self.num_cage_vertices*2 // 4
            key_d = key_d.reshape(B, M_, 9)
            tmp_R, tmp_t = key_d[...,:6], key_d[...,6:]
            
            tmp_R = self._6D_to_rot_(tmp_R).reshape(B, -1, 3, 3)
            key_d = torch.cat([tmp_R, tmp_t[..., None]], dim=-1) # (B, M, 3, 4)
            key_d = key_d.permute(0,1,3,2).reshape(B, -1, 3) # (B, M4, 3)
            #key_d = key_d.permute(0,3,1,2).reshape(B, -1, 3) # (B, 4M, 3)
        else:
            key_d = key_d.reshape(B, self.num_cage_vertices*2, 3)
            # key_v = self.key_d_model(exp_z_v, z_ID_B).reshape(B, M, 3)
        return key_d
    
    def forward(self, source_vert, deform_vert, source_norm, deform_norm, mesh_data, epoch=0, out_kw=False, recon_out=True):
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
        deform_in = deform_vert-source_vert # as a delta
        # deform_in = deform_vert # as a vertex
        
        if self.in_dim == 6:
            source_in = torch.cat([source_in, source_norm], dim=-1)
            deform_in = torch.cat([deform_in, deform_norm], dim=-1)
            
        deform_in = torch.cat([deform_in, source_in], dim=-1)

        ### weight should be different for each shape
        # key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
        key_weight = self.key_weight_model(deform_in, N=self.NZ) # (B, N, M)
        # --> (B, N, 4M) if self.opts.out_type == 2
        
        if self.use_shp:
            z_ID_B = self.shape_model(source_in) # (B, 1, L)
            
            exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
            key_s_key_d = self.key_d_model(exp_z, z_ID_B)
        else:
            exp_z = self.exp_z_model(deform_in) # (B, 1, L)
            key_s_key_d = self.key_d_model(exp_z)
        
        key_s_key_d = self.reshape_key_d(key_s_key_d, B) # (B, 2M, 3)
        
        M = self.num_cage_vertices
        key_s, key_d = key_s_key_d[:,:M], key_s_key_d[:,M:]
        
        #print(key_d.shape)
        def_v = torch.einsum('bnc,bci->bni',key_weight,key_d) # (B, N, 3)
        src_v = torch.einsum('bnc,bci->bni',key_weight,key_s) # (B, N, 3)
        
        if self.use_full_vertex:
            pred_deformed = def_v
            pred_source = src_v
        else:
            pred_deformed = def_v + source_vert
            pred_source = src_v + source_vert
        
        # supple networks -------------------------------------
        ## necessary -- not really...
        if self.use_shp_recon and recon_out:
            recon_source = self.recon_shp_model[mesh_data](z_ID_B)
            recon_source = recon_source.reshape(B, -1, 3)
        else:
            recon_source = 0
        
        ## unnecessary
        if self.use_exp_recon and recon_out:
            recon_delta_v = self.recon_exp_model[mesh_data](exp_z)
            recon_delta_v = recon_delta_v.reshape(B, -1, 3)
            recon_deformed = recon_delta_v + source_vert
        else:
            recon_deformed = 0
        # -----------------------------------------------------        
        
        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight
            
        return pred_deformed, recon_deformed, recon_source, exp_z, pred_source

    # def retarget(self, src_neu_vert, src_neu_norm, src_def_vert, src_def_norm, tgt_neu_vert, tgt_neu_norm,
    #              source_vert, deform_vert, source_norm, deform_norm, mesh_data, epoch=0, out_kw=False, recon_out=True):
    #     """
    #     Args:
    #         source_vert (torch.tensor): [B, N, 3] source mesh vertices
    #         deform_vert (torch.tensor): [B, N, 3] deformed mesh vertices
    #         source_norm (torch.tensor): [B, N, 3] source mesh vertex normals
    #         deform_norm (torch.tensor): [B, N, 3] deformed mesh vertex normals
    #         mesh_data (int): indicator for data (0: voca, 1: biwi, 2: multiface)
    #         epoch (int): train epoch (epoch != iteration)
    #     Returns:
    #         (pred_deformed, recon_deformed, recon_source):
    #         predicted deformed mesh using weight and displacement,
    #         reconstructed deformed mesh
    #         reconstructed source mesh
    #     """
    #     B, N, _ = deform_vert.shape
        
    #     source_in = src_neu_vert
    #     deform_in = src_def_vert-src_neu_vert # as a delta
    #     # deform_in = deform_vert # as a vertex
        
    #     if self.in_dim == 6:
    #         source_in = torch.cat([source_in, src_neu_norm], dim=-1)
    #         deform_in = torch.cat([deform_in, src_def_norm], dim=-1)
            
    #     deform_in = torch.cat([deform_in, source_in], dim=-1)

    #     ### weight should be different for each shape
    #     # key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
    #     key_weight = self.key_weight_model(deform_in, N=self.NZ) # (B, N, M)
    #     # --> (B, N, 4M) if self.opts.out_type == 2
        
    #     if self.use_shp:
    #         z_ID_B = self.shape_model(source_in) # (B, 1, L)
            
    #         exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
    #         key_s_key_d = self.key_d_model(exp_z, z_ID_B)
    #     else:
    #         exp_z = self.exp_z_model(deform_in) # (B, 1, L)
    #         key_s_key_d = self.key_d_model(exp_z)
        
    #     key_s_key_d = self.reshape_key_d(key_s_key_d, B) # (B, 2M, 3)
        
    #     M = self.num_cage_vertices
    #     key_s, key_d = key_s_key_d[:,:M], key_s_key_d[:,M:]
        
    #     #print(key_d.shape)
    #     def_v = torch.einsum('bnc,bci->bni',key_weight,key_d) # (B, N, 3)
    #     src_v = torch.einsum('bnc,bci->bni',key_weight,key_s) # (B, N, 3)
        
    #     if self.use_full_vertex:
    #         pred_deformed = def_v
    #         pred_source = src_v
    #     else:
    #         pred_deformed = def_v + source_vert
    #         pred_source = src_v + source_vert
        
    #     # supple networks -------------------------------------
    #     ## necessary -- not really...
    #     if self.use_shp_recon and recon_out:
    #         recon_source = self.recon_shp_model[mesh_data](z_ID_B)
    #         recon_source = recon_source.reshape(B, -1, 3)
    #     else:
    #         recon_source = 0
        
    #     ## unnecessary
    #     if self.use_exp_recon and recon_out:
    #         recon_delta_v = self.recon_exp_model[mesh_data](exp_z)
    #         recon_delta_v = recon_delta_v.reshape(B, -1, 3)
    #         recon_deformed = recon_delta_v + source_vert
    #     else:
    #         recon_deformed = 0
    #     # -----------------------------------------------------        
        
    #     if out_kw:
    #         return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight
            
    #     return pred_deformed, recon_deformed, recon_source, exp_z, pred_source

class NeuralGeneralizedBarycentricCoordinate8(nn.Module):
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
                 use_relu=False,
                 use_elu=True,
                 use_least_N=False,
                 use_least_N_on_V=False,
                 least_number_of_zeros=256, # for sparsity (not used)
                 is_train=False,
                 tau=0.05,
                 device='cpu',
                 use_exp_recon=False, # was not necessary
                 use_shp_recon=True, # necessary for training, but not needed for inference
                 use_shp=True,
                 use_full_vertex=False, # default: false (= delta form)
                ):
        super().__init__()
        self.opts = opts
                
        self.is_train = is_train
        
        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_shp_recon = use_shp_recon        
        self.use_shp = use_shp
        if not self.use_shp:
            self.use_shp_recon = False
            print('shape model not used!, use_shp_recon set to False')
        self.use_exp_recon = use_exp_recon        
        self.use_full_vertex = use_full_vertex
        
        ###### NN input type settings
        ## key_weight model | key_d_model 
        # 0: (src_p),        (def_p, src_p)
        # 1: (src_p, src_n), (def_p, def_n, src_p, src_n)
        
        self.in_type = 1
        self.out_type = 1 # vertex
        ######
        
        if self.opts is not None:
            self.in_type = self.opts.in_type
            self.out_type = self.opts.out_type
        
        if self.in_type == 0:
            self.in_dim = 3
            in_dim_exp = self.in_dim*2
        elif self.in_type == 1:
            self.in_dim = 6
            in_dim_exp = self.in_dim*2
        else:
            raise NotImplementedError('in_type not implemented')
            
        if self.out_type == 0: # delta form
            self.use_full_vertex = False
            self.out_dim = 3
        elif self.out_type == 1: # vertex (linear precision)
            self.use_full_vertex = True
            self.out_dim = 3
        elif self.out_type == 2: # transform matrix
            self.use_full_vertex = False
            self.out_dim = 9 # (6D + translation 3) will be reshaped into 3x4 matrix
            M_ = num_cage_vertices
            num_cage_vertices = num_cage_vertices * 4
            
            from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
            self._6D_to_rot_ = _6D_to_rot_
        else:
            raise NotImplementedError('out_type not implemented')
            
        from models.encoder import LinearEncoder,LinearEncoder2
        
        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros

        # coordinate predictor
        self.key_weight_model = LinearEncoder2(
            in_dim=self.in_dim,
            # in_dim=in_dim_exp,
            # in_dim=self.in_dim+L, # [vertex feat + exp_z]
            style_dim=L,            
            out_dim=M, 
            use_softmax=use_softmax,
            use_relu=use_relu, # default setting
            use_elu=use_elu,
            use_least_N=use_least_N,
            use_least_N_on_V=use_least_N_on_V,
        ).to(device)

        # expression encoder
        if self.use_shp:
            self.exp_z_model = LinearEncoder2(
                in_dim=in_dim_exp,
                style_dim=L,
                out_dim=L,
                num_layers=self.num_layers, 
                use_style=True, 
                out_type='global',
            ).to(device)
        else:
            self.exp_z_model = LinearEncoder(
                in_dim=in_dim_exp,
                out_dim=L,
                num_layers=self.num_layers, 
                out_type='global',
            ).to(device)

        # shape encoder
        if self.use_shp:
            self.shape_model = LinearEncoder(
                in_dim=self.in_dim,
                out_dim=L,
                out_type='global'
            ).to(device)

        # cage displacement predictor (predict both original cage and deformed cage)
        if self.use_shp:
            self.key_d_model = LinearEncoder2(
                in_dim=L,
                style_dim=L,
                out_dim= M_*self.out_dim*2 if self.out_type == 2 else M*self.out_dim*2,
                num_layers=self.num_layers, 
                use_style=True, out_type='global'
            ).to(device)
        else:
            self.key_d_model = LinearEncoder(
                in_dim=L,
                out_dim= M_*self.out_dim*2 if self.out_type == 2 else M*self.out_dim*2,
                num_layers=self.num_layers, 
                out_type='global'
            ).to(device)

        if self.is_train:
            if self.use_exp_recon:
                self.recon_exp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
            
            if self.use_shp_recon:
                self.recon_shp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
        
    def reshape_key_d(self, key_d, B):
        if self.out_type == 2:
            # cage transform matrix
            M_ = self.num_cage_vertices*2 // 4
            key_d = key_d.reshape(B, M_, 9)
            tmp_R, tmp_t = key_d[...,:6], key_d[...,6:]
            
            tmp_R = self._6D_to_rot_(tmp_R).reshape(B, -1, 3, 3)
            key_d = torch.cat([tmp_R, tmp_t[..., None]], dim=-1) # (B, M, 3, 4)
            key_d = key_d.permute(0,1,3,2).reshape(B, -1, 3) # (B, M4, 3)
            #key_d = key_d.permute(0,3,1,2).reshape(B, -1, 3) # (B, 4M, 3)
        else:
            key_d = key_d.reshape(B, self.num_cage_vertices*2, 3)
            # key_v = self.key_d_model(exp_z_v, z_ID_B).reshape(B, M, 3)
        return key_d
    
    def forward(self, source_vert, deform_vert, source_norm, deform_norm, mesh_data=0, epoch=0, out_kw=False, recon_out=True):
        """
        Trained with self-retargeting setting
        
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

        ### permutation ---------------------------------------------
        if not out_kw:
            N_range = torch.randint(100, N//6, (1,)).item()
            N_ = N-N_range
            randperm_idx = torch.randperm(N)[:N_]
            rearange_idx = torch.argsort(randperm_idx)
            
            source_vert_ = source_vert[:, randperm_idx]
            deform_vert_ = deform_vert[:, randperm_idx]
            source_norm_ = source_norm[:, randperm_idx]
            deform_norm_ = deform_norm[:, randperm_idx]
        else:
            source_vert_ = source_vert
            deform_vert_ = deform_vert
            source_norm_ = source_norm
            deform_norm_ = deform_norm
        ### ---------------------------------------------------------
        
        
        source_in = source_vert_
        deform_in = deform_vert_-source_vert_ # as a delta
        # deform_in = deform_vert # as a vertex
        
        if self.in_dim == 6:
            source_in = torch.cat([source_in, source_norm_], dim=-1)
            deform_in = torch.cat([deform_in, deform_norm_], dim=-1)            
        deform_in = torch.cat([deform_in, source_in], dim=-1)

        
        ### cage prediction -----------------------------------------
        if self.use_shp:
            z_ID_B = self.shape_model(source_in) # (B, 1, L)
            
            exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
            key_s_key_d = self.key_d_model(exp_z, z_ID_B)
        else:
            exp_z = self.exp_z_model(deform_in) # (B, 1, L)
            key_s_key_d = self.key_d_model(exp_z)
        
        key_s_key_d = self.reshape_key_d(key_s_key_d, B) # (B, 2M, 3)
        
        M = self.num_cage_vertices
        key_s, key_d = key_s_key_d[:,:M], key_s_key_d[:,M:]
        ### ---------------------------------------------------------

        
        ### weight prediction ---------------------------------------
        # source_w_in = torch.cat([source_in, exp_z.repeat(1,source_in.shape[1],1)], dim=-1)
        
        ### weight should be different for each shape
        # key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
        key_weight = self.key_weight_model(source_in, exp_z, N=self.NZ) # (B, N, M)
        # key_weight = self.key_weight_model(source_w_in, N=self.NZ) # (B, N, M)
        # key_weight = self.key_weight_model(deform_in, N=self.NZ) # (B, N, M)
        # --> (B, N, 4M) if self.opts.out_type == 2
        ### ---------------------------------------------------------
        
        
        def_v = torch.einsum('bnc,bci->bni',key_weight,key_d) # (B, N, 3)
        src_v = torch.einsum('bnc,bci->bni',key_weight,key_s) # (B, N, 3)
        
        if self.use_full_vertex:
            pred_deformed = def_v
            pred_source = src_v
        else:
            pred_deformed = def_v + source_vert_
            pred_source = src_v + source_vert_

        
        # supple networks -------------------------------------------
        ## necessary -- not really...
        if self.use_shp_recon and recon_out:
            recon_source = self.recon_shp_model[mesh_data](z_ID_B)
            recon_source = recon_source.reshape(B, -1, 3)
        else:
            recon_source = 0
        
        ## unnecessary
        if self.use_exp_recon and recon_out:
            recon_delta_v = self.recon_exp_model[mesh_data](exp_z)
            recon_delta_v = recon_delta_v.reshape(B, -1, 3)
            recon_deformed = recon_delta_v + source_vert
        else:
            recon_deformed = 0
        ### ---------------------------------------------------------  
        
        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight
            
        return pred_deformed, recon_deformed, recon_source, exp_z, pred_source, (randperm_idx, rearange_idx)



class NeuralGeneralizedBarycentricCoordinate81(nn.Module):
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
                 use_relu=False,
                 use_elu=True,
                 use_least_N=False,
                 use_least_N_on_V=False,
                 least_number_of_zeros=256, # for sparsity (not used)
                 is_train=False,
                 tau=0.05,
                 device='cpu',
                 use_exp_recon=False, # was not necessary
                 use_shp_recon=True, # necessary for training, but not needed for inference
                 use_shp=True,
                 use_full_vertex=False, # default: false (= delta form)
                ):
        super().__init__()
        self.opts = opts
                
        self.is_train = is_train
        
        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_shp_recon = use_shp_recon        
        self.use_shp = use_shp
        if not self.use_shp:
            self.use_shp_recon = False
            print('shape model not used!, use_shp_recon set to False')
        self.use_exp_recon = use_exp_recon        
        self.use_full_vertex = use_full_vertex
        
        ###### NN input type settings
        ## key_weight model | key_d_model 
        # 0: (src_p),        (def_p, src_p)
        # 1: (src_p, src_n), (def_p, def_n, src_p, src_n)
        
        self.in_type = 1
        self.out_type = 1 # vertex
        ######
        
        if self.opts is not None:
            self.in_type = self.opts.in_type
            self.out_type = self.opts.out_type
        
        if self.in_type == 0:
            self.in_dim = 3
            in_dim_exp = self.in_dim*2
        elif self.in_type == 1:
            self.in_dim = 6
            in_dim_exp = self.in_dim*2
        else:
            raise NotImplementedError('in_type not implemented')
            
        if self.out_type == 0: # delta form
            self.use_full_vertex = False
            self.out_dim = 3
        elif self.out_type == 1: # vertex (linear precision)
            self.use_full_vertex = True
            self.out_dim = 3
        elif self.out_type == 2: # transform matrix
            self.use_full_vertex = False
            self.out_dim = 9 # (6D + translation 3) will be reshaped into 3x4 matrix
            M_ = num_cage_vertices
            num_cage_vertices = num_cage_vertices * 4
            
            from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
            self._6D_to_rot_ = _6D_to_rot_
        else:
            raise NotImplementedError('out_type not implemented')
            
        from models.encoder import LinearEncoder,LinearEncoder2
        
        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros

        # coordinate predictor
        self.key_weight_model = LinearEncoder2(
            in_dim=self.in_dim,
            # in_dim=in_dim_exp,
            # in_dim=self.in_dim+L, # [vertex feat + exp_z]
            style_dim=L,            
            out_dim=M, 
            use_softmax=use_softmax,
            use_relu=use_relu, # default setting
            use_elu=use_elu,
            use_least_N=use_least_N,
            use_least_N_on_V=use_least_N_on_V,
        ).to(device)

        # expression encoder
        if self.use_shp:
            self.exp_z_model = LinearEncoder2(
                in_dim=in_dim_exp,
                style_dim=L,
                out_dim=L,
                num_layers=self.num_layers, 
                use_style=True, 
                out_type='global',
            ).to(device)
        else:
            self.exp_z_model = LinearEncoder(
                in_dim=in_dim_exp,
                out_dim=L,
                num_layers=self.num_layers, 
                out_type='global',
            ).to(device)

        # shape encoder
        if self.use_shp:
            self.shape_model = LinearEncoder(
                in_dim=self.in_dim,
                out_dim=L,
                out_type='global'
            ).to(device)

        # cage displacement predictor (predict both original cage and deformed cage)
        if self.use_shp:
            self.key_d_model = LinearEncoder2(
                in_dim=L,
                style_dim=L,
                out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
                num_layers=self.num_layers, 
                use_style=True, out_type='global'
            ).to(device)
        else:
            self.key_d_model = LinearEncoder(
                in_dim=L,
                out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
                num_layers=self.num_layers, 
                out_type='global'
            ).to(device)

        if self.is_train:
            if self.use_exp_recon:
                self.recon_exp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
            
            if self.use_shp_recon:
                self.recon_shp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
        
    def reshape_key_d(self, key_d, B):
        if self.out_type == 2:
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
        return key_d
    
    def forward(self, source_vert, deform_vert, source_norm, deform_norm, mesh_data=0, epoch=0, out_kw=False, recon_out=True):
        """
        Trained with self-retargeting setting
        
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

        ### permutation ---------------------------------------------
        if not out_kw:
            N_range = torch.randint(100, N//6, (1,)).item()
            N_ = N-N_range
            randperm_idx = torch.randperm(N)[:N_]
            rearange_idx = torch.argsort(randperm_idx)
            
            source_vert_ = source_vert[:, randperm_idx]
            deform_vert_ = deform_vert[:, randperm_idx]
            source_norm_ = source_norm[:, randperm_idx]
            deform_norm_ = deform_norm[:, randperm_idx]
        else:
            source_vert_ = source_vert
            deform_vert_ = deform_vert
            source_norm_ = source_norm
            deform_norm_ = deform_norm
        ### ---------------------------------------------------------
        
        
        source_in = source_vert_
        deform_in = deform_vert_-source_vert_ # as a delta (source deformed)
        deform_in_s = source_vert_-source_vert_ # as a delta (source neutral)
        # deform_in = deform_vert # as a vertex
        
        if self.in_dim == 6:
            source_in = torch.cat([source_in, source_norm_], dim=-1)
            deform_in = torch.cat([deform_in, deform_norm_], dim=-1)
            deform_in_s = torch.cat([deform_in_s, deform_norm_], dim=-1)
            
        deform_in = torch.cat([deform_in, source_in], dim=-1)
        deform_in_s = torch.cat([deform_in_s, source_in], dim=-1)

        
        ### cage prediction -----------------------------------------
        if self.use_shp:
            z_ID_B = self.shape_model(source_in) # (B, 1, L)
            
            exp_z_d = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
            key_d = self.key_d_model(exp_z_d, z_ID_B)
            
            exp_z_s = self.exp_z_model(deform_in_s, z_ID_B) # (B, 1, L)
            key_s = self.key_d_model(exp_z_s, z_ID_B)
        else:
            exp_z_d = self.exp_z_model(deform_in) # (B, 1, L)
            key_d = self.key_d_model(exp_z_d)
            
            exp_z_s = self.exp_z_model(deform_in_s) # (B, 1, L)
            key_s = self.key_d_model(exp_z_s)
        
        key_d = self.reshape_key_d(key_d, B) # (B, 2M, 3)
        key_s = self.reshape_key_d(key_s, B) # (B, 2M, 3)
        
        M = self.num_cage_vertices
        ### ---------------------------------------------------------

        
        ### weight prediction ---------------------------------------
        
        ### weight should be different for each shape
        key_weight_s = self.key_weight_model(source_in, exp_z_s, N=self.NZ) # (B, N, M)
        key_weight_d = self.key_weight_model(source_in, exp_z_d, N=self.NZ) # (B, N, M)
        ### ---------------------------------------------------------
        
        
        def_v = torch.einsum('bnc,bci->bni',key_weight_d,key_d) # (B, N, 3)
        src_v = torch.einsum('bnc,bci->bni',key_weight_s,key_s) # (B, N, 3)
        
        if self.use_full_vertex:
            pred_deformed = def_v
            pred_source = src_v
        else:
            pred_deformed = def_v + source_vert_
            pred_source = src_v + source_vert_

        
        # supple networks -------------------------------------------
        ## necessary -- not really...
        if self.use_shp_recon and recon_out:
            recon_source = self.recon_shp_model[mesh_data](z_ID_B)
            recon_source = recon_source.reshape(B, -1, 3)
        else:
            recon_source = 0
        
        ## unnecessary
        if self.use_exp_recon and recon_out:
            recon_delta_v = self.recon_exp_model[mesh_data](exp_z_d)
            recon_delta_v = recon_delta_v.reshape(B, -1, 3)
            recon_deformed = recon_delta_v + source_vert
        else:
            recon_deformed = 0
        ### ---------------------------------------------------------  
        
        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z_d, key_d, key_weight_d
            
        return pred_deformed, recon_deformed, recon_source, exp_z_d, pred_source, (randperm_idx, rearange_idx)
    
    def retarget(self, 
                 src_neu_vert, src_neu_norm, src_def_vert, src_def_norm, tgt_neu_vert, tgt_neu_norm,
                 mesh_data=0, out_kw=False, recon_out=True):
        """
        Args:
            src_neu_vert (torch.tensor): [B, N, 3] source neutral mesh vertex positions
            src_neu_norm (torch.tensor): [B, N, 3] source neutral mesh vertex normals
            
            src_def_vert (torch.tensor): [B, N, 3] source deformed mesh vertex positions
            src_def_norm (torch.tensor): [B, N, 3] source deformed mesh vertex normals
            
            tgt_neu_vert (torch.tensor): [B, M, 3] target neutral mesh vertex positions
            tgt_neu_norm (torch.tensor): [B, M, 3] target neutral mesh vertex normals
            
            mesh_data (int): indicator for data (0: voca, 1: biwi, 2: multiface) -- not used!
            
        Returns:
            (pred_deformed, pred_source):
            predicted target deformation and neutral mesh using weight and cage prediction
        """
        B, N, _ = src_neu_vert.shape
        
        tgt_in = tgt_neu_vert        
        src_in = src_neu_vert
        
        deform_in_d = src_def_vert-src_neu_vert # as a delta
        deform_in_s = src_neu_vert-src_neu_vert # as a delta
                
        if self.in_dim == 6:
            tgt_in = torch.cat([tgt_in, tgt_neu_norm], dim=-1)
            src_in = torch.cat([src_in, src_neu_norm], dim=-1)
            deform_in_d = torch.cat([deform_in_d, src_def_norm], dim=-1)
            deform_in_s = torch.cat([deform_in_s, src_def_norm], dim=-1)
            
        deform_in_d = torch.cat([deform_in_d, src_in], dim=-1)
        deform_in_s = torch.cat([deform_in_s, src_in], dim=-1)
        
        with torch.no_grad():
            if self.use_shp:
                z_ID_B = self.shape_model(src_in) # (B, 1, L)
                
                exp_z_d = self.exp_z_model(deform_in_d, z_ID_B) # (B, 1, L)
                key_d = self.key_d_model(exp_z_d, z_ID_B)
                
                exp_z_s = self.exp_z_model(deform_in_s, z_ID_B) # (B, 1, L)
                key_s = self.key_d_model(exp_z_s, z_ID_B)
            else:
                exp_z_d = self.exp_z_model(deform_in_d) # (B, 1, L)
                key_d = self.key_d_model(exp_z_d)
                
                exp_z_s = self.exp_z_model(deform_in_s) # (B, 1, L)
                key_s = self.key_d_model(exp_z_s)
            
            key_d = self.reshape_key_d(key_d, B) # (B, 2M, 3)
            key_s = self.reshape_key_d(key_s, B) # (B, 2M, 3)
            
            ### weight should be different for each shape
            key_weight_d = self.key_weight_model(tgt_in, exp_z_d, N=self.NZ) # (B, N, M)
            key_weight_s = self.key_weight_model(tgt_in, exp_z_s, N=self.NZ) # (B, N, M)
            # --> (B, N, 4M) if self.opts.out_type == 2
        
            M = self.num_cage_vertices
            
            #print(key_d.shape)
            def_v = torch.einsum('bnc,bci->bni',key_weight_d,key_d) # (B, N, 3)
            src_v = torch.einsum('bnc,bci->bni',key_weight_s,key_s) # (B, N, 3)
        
        if self.use_full_vertex:
            pred_deformed = def_v
            pred_source = src_v
        else:
            pred_deformed = def_v + source_vert
            pred_source = src_v + source_vert
        
        if out_kw:
            return pred_deformed, pred_source, exp_z_d, key_d, exp_z_s, key_s, key_weight_d
        
        return pred_deformed, pred_source

class NeuralGeneralizedBarycentricCoordinate85(nn.Module):
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
                 use_relu=False,
                 use_elu=True,
                 use_least_N=False,
                 use_least_N_on_V=False,
                 least_number_of_zeros=256, # for sparsity (not used)
                 is_train=False,
                 tau=0.05,
                 device='cpu',
                 use_exp_recon=False, # was not necessary
                 use_shp_recon=True, # necessary for training, but not needed for inference
                 use_shp=True,
                 use_full_vertex=False, # default: false (= delta form)
                ):
        super().__init__()
        self.opts = opts
                
        self.is_train = is_train
        
        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_shp_recon = use_shp_recon        
        self.use_shp = use_shp
        if not self.use_shp:
            self.use_shp_recon = False
            print('shape model not used!, use_shp_recon set to False')
        self.use_exp_recon = use_exp_recon        
        self.use_full_vertex = use_full_vertex
        
        ###### NN input type settings
        ## key_weight model | key_d_model 
        # 0: (src_p),        (def_p, src_p)
        # 1: (src_p, src_n), (def_p, def_n, src_p, src_n)
        
        self.in_type = 1
        self.out_type = 1 # vertex
        ######
        
        if self.opts is not None:
            self.in_type = self.opts.in_type
            self.out_type = self.opts.out_type
        
        if self.in_type == 0:
            self.in_dim = 3
            in_dim_exp = self.in_dim*2
        elif self.in_type == 1:
            self.in_dim = 6
            in_dim_exp = self.in_dim*2
        else:
            raise NotImplementedError('in_type not implemented')
            
        if self.out_type == 0: # delta form
            self.use_full_vertex = False
            self.out_dim = 3
        elif self.out_type == 1: # vertex (linear precision)
            self.use_full_vertex = True
            self.out_dim = 3
        elif self.out_type == 2: # transform matrix
            self.use_full_vertex = False
            self.out_dim = 9 # (6D + translation 3) will be reshaped into 3x4 matrix
            M_ = num_cage_vertices
            num_cage_vertices = num_cage_vertices * 4
            
            from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
            self._6D_to_rot_ = _6D_to_rot_
        else:
            raise NotImplementedError('out_type not implemented')
            
        from models.encoder import LinearEncoder,LinearEncoder2
        
        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros

        # coordinate predictor
        self.key_weight_model = LinearEncoder2(
            in_dim=self.in_dim,
            # in_dim=in_dim_exp,
            # in_dim=self.in_dim+L, # [vertex feat + exp_z]
            style_dim=L * 2 if self.use_shp else L,            
            out_dim=M, 
            use_softmax=use_softmax,
            use_relu=use_relu, # default setting
            use_elu=use_elu,
            use_least_N=use_least_N,
            use_least_N_on_V=use_least_N_on_V,
        ).to(device)

        # expression encoder
        if self.use_shp:
            self.exp_z_model = LinearEncoder2(
                in_dim=in_dim_exp,
                style_dim=L,
                out_dim=L,
                num_layers=self.num_layers, 
                use_style=True, 
                out_type='global',
            ).to(device)
        else:
            self.exp_z_model = LinearEncoder(
                in_dim=in_dim_exp,
                out_dim=L,
                num_layers=self.num_layers, 
                out_type='global',
            ).to(device)

        # shape encoder
        if self.use_shp:
            self.shape_model = LinearEncoder(
                in_dim=self.in_dim,
                out_dim=L,
                out_type='global'
            ).to(device)

        # cage displacement predictor (predict both original cage and deformed cage)
        if self.use_shp:
            self.key_d_model = LinearEncoder2(
                in_dim=L,
                style_dim=L,
                out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
                num_layers=self.num_layers, 
                use_style=True, out_type='global'
            ).to(device)
        else:
            self.key_d_model = LinearEncoder(
                in_dim=L,
                out_dim= M_*self.out_dim if self.out_type == 2 else M*self.out_dim,
                num_layers=self.num_layers, 
                out_type='global'
            ).to(device)

        if self.is_train:
            if self.use_exp_recon:
                self.recon_exp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
            
            if self.use_shp_recon:
                self.recon_shp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
        
    def reshape_key_d(self, key_d, B):
        if self.out_type == 2:
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
        return key_d
    
    def forward(self, source_vert, deform_vert, source_norm, deform_norm, mesh_data=0, epoch=0, out_kw=False, recon_out=True):
        """
        Trained with self-retargeting setting
        
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

        ### permutation ---------------------------------------------
        if not out_kw:
            N_range = torch.randint(100, N//6, (1,)).item()
            N_ = N-N_range
            randperm_idx = torch.randperm(N)[:N_]
            rearange_idx = torch.argsort(randperm_idx)
            
            source_vert_ = source_vert[:, randperm_idx]
            deform_vert_ = deform_vert[:, randperm_idx]
            source_norm_ = source_norm[:, randperm_idx]
            deform_norm_ = deform_norm[:, randperm_idx]
        else:
            source_vert_ = source_vert
            deform_vert_ = deform_vert
            source_norm_ = source_norm
            deform_norm_ = deform_norm
        ### ---------------------------------------------------------
        
        
        source_in = source_vert_
        deform_in = deform_vert_-source_vert_ # as a delta
        deform_in_s = source_vert_-source_vert_ # as a delta
        
        if self.in_dim == 6:
            source_in = torch.cat([source_in, source_norm_], dim=-1)
            deform_in = torch.cat([deform_in, deform_norm_], dim=-1)
            deform_in_s = torch.cat([deform_in_s, deform_norm_], dim=-1)
            
        deform_in = torch.cat([deform_in, source_in], dim=-1)
        deform_in_s = torch.cat([deform_in_s, source_in], dim=-1)
        
        ### cage prediction -----------------------------------------
        if self.use_shp:
            z_ID_B = self.shape_model(source_in) # (B, 1, L)
            
            exp_z_d = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
            key_d = self.key_d_model(exp_z_d, z_ID_B)
            
            exp_z_s = self.exp_z_model(deform_in_s, z_ID_B) # (B, 1, L)
            key_s = self.key_d_model(exp_z_s, z_ID_B)
        else:
            exp_z_d = self.exp_z_model(deform_in) # (B, 1, L)
            key_d = self.key_d_model(exp_z_d)
            
            exp_z_s = self.exp_z_model(deform_in_s) # (B, 1, L)
            key_s = self.key_d_model(exp_z_s)
        
        key_d = self.reshape_key_d(key_d, B) # (B, M, 3)
        key_s = self.reshape_key_d(key_s, B) # (B, M, 3)
        ### ---------------------------------------------------------

        
        ### weight prediction ---------------------------------------
        ### weight should be different for each shape
        ## NEW !!!
        if self.use_shp:
            exp_z_d_ID = torch.cat([exp_z_d, z_ID_B], dim=-1)
            exp_z_s_ID = torch.cat([exp_z_s, z_ID_B], dim=-1)
            key_weight_d = self.key_weight_model(source_in, exp_z_d_ID, N=self.NZ) # (B, N, M)
            key_weight_s = self.key_weight_model(source_in, exp_z_s_ID, N=self.NZ) # (B, N, M)
        else:
            key_weight_d = self.key_weight_model(source_in, exp_z_d, N=self.NZ) # (B, N, M)
            key_weight_s = self.key_weight_model(source_in, exp_z_s, N=self.NZ) # (B, N, M)
        ### ---------------------------------------------------------
        
        
        def_v = torch.einsum('bnc,bci->bni',key_weight_d,key_d) # (B, N, 3)
        src_v = torch.einsum('bnc,bci->bni',key_weight_s,key_s) # (B, N, 3)
        
        if self.use_full_vertex:
            pred_deformed = def_v
            pred_source = src_v
        else:
            pred_deformed = def_v + source_vert_
            pred_source = src_v + source_vert_

        
        # supple networks -------------------------------------------
        ## necessary -- not really...
        if self.use_shp_recon and recon_out:
            recon_source = self.recon_shp_model[mesh_data](z_ID_B)
            recon_source = recon_source.reshape(B, -1, 3)
        else:
            recon_source = 0
        
        ## unnecessary
        if self.use_exp_recon and recon_out:
            recon_delta_v = self.recon_exp_model[mesh_data](exp_z_d)
            recon_delta_v = recon_delta_v.reshape(B, -1, 3)
            recon_deformed = recon_delta_v + source_vert
        else:
            recon_deformed = 0
        ### ---------------------------------------------------------  
        
        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z_d, key_d, key_weight_d
            
        return pred_deformed, recon_deformed, recon_source, exp_z_d, pred_source, (randperm_idx, rearange_idx)
    
    def retarget(self, 
                 src_neu_vert, src_neu_norm, src_def_vert, src_def_norm, tgt_neu_vert, tgt_neu_norm,
                 mesh_data=0, out_kw=False, recon_out=True):
        """
        Args:
            src_neu_vert (torch.tensor): [B, N, 3] source neutral mesh vertex positions
            src_neu_norm (torch.tensor): [B, N, 3] source neutral mesh vertex normals
            
            src_def_vert (torch.tensor): [B, N, 3] source deformed mesh vertex positions
            src_def_norm (torch.tensor): [B, N, 3] source deformed mesh vertex normals
            
            tgt_neu_vert (torch.tensor): [B, M, 3] target neutral mesh vertex positions
            tgt_neu_norm (torch.tensor): [B, M, 3] target neutral mesh vertex normals
            
            mesh_data (int): indicator for data (0: voca, 1: biwi, 2: multiface) -- not used!
            
        Returns:
            (pred_deformed, pred_source):
            predicted target deformation and neutral mesh using weight and cage prediction
        """
        B, N, _ = src_neu_vert.shape
        
        tgt_in = tgt_neu_vert        
        src_in = src_neu_vert
        
        deform_in_d = src_def_vert-src_neu_vert # as a delta
        deform_in_s = src_neu_vert-src_neu_vert # as a delta
                
        if self.in_dim == 6:
            tgt_in = torch.cat([tgt_in, tgt_neu_norm], dim=-1)
            src_in = torch.cat([src_in, src_neu_norm], dim=-1)
            deform_in_d = torch.cat([deform_in_d, src_def_norm], dim=-1)
            deform_in_s = torch.cat([deform_in_s, src_def_norm], dim=-1)
            
        deform_in_d = torch.cat([deform_in_d, src_in], dim=-1)
        deform_in_s = torch.cat([deform_in_s, src_in], dim=-1)
        
        with torch.no_grad():
            if self.use_shp:
                z_ID_B = self.shape_model(src_in) # (B, 1, L)
                
                exp_z_d = self.exp_z_model(deform_in_d, z_ID_B) # (B, 1, L)
                key_d = self.key_d_model(exp_z_d, z_ID_B)
                
                exp_z_s = self.exp_z_model(deform_in_s, z_ID_B) # (B, 1, L)
                key_s = self.key_d_model(exp_z_s, z_ID_B)
            else:
                exp_z_d = self.exp_z_model(deform_in_d) # (B, 1, L)
                key_d = self.key_d_model(exp_z_d)
                
                exp_z_s = self.exp_z_model(deform_in_s) # (B, 1, L)
                key_s = self.key_d_model(exp_z_s)
            
            key_d = self.reshape_key_d(key_d, B)
            key_s = self.reshape_key_d(key_s, B) # (B, M, 3)
            
            ### weight should be different for each shape
            if self.use_shp:
                exp_z_d_ID = torch.cat([exp_z_d, z_ID_B], dim=-1)
                exp_z_s_ID = torch.cat([exp_z_s, z_ID_B], dim=-1)
                key_weight_d = self.key_weight_model(tgt_in, exp_z_d_ID, N=self.NZ) # (B, N, M)
                key_weight_s = self.key_weight_model(tgt_in, exp_z_s_ID, N=self.NZ) # (B, N, M)
            else:
                key_weight_d = self.key_weight_model(tgt_in, exp_z_d, N=self.NZ) # (B, N, M)
                key_weight_s = self.key_weight_model(tgt_in, exp_z_s, N=self.NZ) # (B, N, M)
            
            #print(key_d.shape)
            def_v = torch.einsum('bnc,bci->bni',key_weight_d,key_d) # (B, N, 3)
            src_v = torch.einsum('bnc,bci->bni',key_weight_s,key_s) # (B, N, 3)
        
        if self.use_full_vertex:
            pred_deformed = def_v
            pred_source = src_v
        else:
            pred_deformed = def_v + source_vert
            pred_source = src_v + source_vert
        
        if out_kw:
            return pred_deformed, pred_source, exp_z_d, key_d, exp_z_s, key_s, key_weight_d
        
        return pred_deformed, pred_source


class NeuralGeneralizedBarycentricCoordinate9(nn.Module):
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
                 use_relu=False,
                 use_elu=True,
                 use_least_N=False,
                 use_least_N_on_V=False,
                 least_number_of_zeros=256, # for sparsity (not used)
                 is_train=False,
                 tau=0.05,
                 device='cpu',
                 use_exp_recon=False, # was not necessary
                 use_shp_recon=True, # necessary for training, but not needed for inference
                 use_shp=True,
                 use_full_vertex=False, # default: false (= delta form)
                ):
        super().__init__()
        self.opts = opts
                
        self.is_train = is_train
        
        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_shp_recon = use_shp_recon        
        self.use_shp = use_shp
        if not self.use_shp:
            self.use_shp_recon = False
            print('shape model not used!, use_shp_recon set to False')
        self.use_exp_recon = use_exp_recon        
        self.use_full_vertex = use_full_vertex
        
        ###### NN input type settings
        ## key_weight model | key_d_model 
        # 0: (src_p),        (def_p, src_p)
        # 1: (src_p, src_n), (def_p, def_n, src_p, src_n)
        
        self.in_type = 1
        self.out_type = 1 # vertex
        ######
        
        if self.opts is not None:
            self.in_type = self.opts.in_type
            self.out_type = self.opts.out_type
        
        if self.in_type == 0:
            self.in_dim = 3
            in_dim_exp = self.in_dim*2
        elif self.in_type == 1:
            self.in_dim = 6
            in_dim_exp = self.in_dim*2
        else:
            raise NotImplementedError('in_type not implemented')
            
        if self.out_type == 0: # delta form
            self.use_full_vertex = False
            self.out_dim = 3
        elif self.out_type == 1: # vertex (linear precision)
            self.use_full_vertex = True
            self.out_dim = 3
        elif self.out_type == 2: # transform matrix
            self.use_full_vertex = False
            self.out_dim = 9 # (6D + translation 3) will be reshaped into 3x4 matrix
            M_ = num_cage_vertices
            num_cage_vertices = num_cage_vertices * 4
            
            from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
            self._6D_to_rot_ = _6D_to_rot_
        else:
            raise NotImplementedError('out_type not implemented')
            
        from models.encoder import LinearEncoder, LinearEncoder2
        
        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros

        # coordinate predictor
        self.key_weight_model = LinearEncoder2(
            # in_dim=self.in_dim,
            # in_dim=in_dim_exp,
            in_dim=self.in_dim, # [vertex feat + exp_z]
            style_dim=L,
            out_dim=M, 
            use_softmax=use_softmax,
            use_relu=use_relu, # default setting
            use_elu=use_elu,
            use_least_N=use_least_N,
            use_least_N_on_V=use_least_N_on_V,
        ).to(device)
        
        self.key_weight_model_Teach = LinearEncoder(
            # in_dim=self.in_dim,
            in_dim=in_dim_exp,
            out_dim=M, 
            use_softmax=use_softmax,
            use_relu=use_relu, # default setting
            use_elu=use_elu,
            use_least_N=use_least_N,
            use_least_N_on_V=use_least_N_on_V,
        ).to(device)

        # expression encoder
        if self.use_shp:
            self.exp_z_model = LinearEncoder2(
                in_dim=in_dim_exp,
                style_dim=L,
                out_dim=L,
                num_layers=self.num_layers, 
                use_style=True, 
                out_type='global',
            ).to(device)
        else:
            self.exp_z_model = LinearEncoder(
                in_dim=in_dim_exp,
                out_dim=L,
                num_layers=self.num_layers, 
                out_type='global',
            ).to(device)

        # shape encoder
        if self.use_shp:
            self.shape_model = LinearEncoder(
                in_dim=self.in_dim,
                out_dim=L,
                out_type='global'
            ).to(device)

        # cage displacement predictor (predict both original cage and deformed cage)
        if self.use_shp:
            self.key_d_model = LinearEncoder2(
                in_dim=L,
                style_dim=L,
                out_dim= M_*self.out_dim*2 if self.out_type == 2 else M*self.out_dim*2,
                num_layers=self.num_layers, 
                use_style=True, out_type='global'
            ).to(device)
        else:
            self.key_d_model = LinearEncoder(
                in_dim=L,
                out_dim= M_*self.out_dim*2 if self.out_type == 2 else M*self.out_dim*2,
                num_layers=self.num_layers, 
                out_type='global'
            ).to(device)

        if self.is_train:
            if self.use_exp_recon:
                self.recon_exp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
            
            if self.use_shp_recon:
                self.recon_shp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)
        
    def reshape_key_d(self, key_d, B):
        if self.out_type == 2:
            # cage transform matrix
            M_ = self.num_cage_vertices*2 // 4
            key_d = key_d.reshape(B, M_, 9)
            tmp_R, tmp_t = key_d[...,:6], key_d[...,6:]
            
            tmp_R = self._6D_to_rot_(tmp_R).reshape(B, -1, 3, 3)
            key_d = torch.cat([tmp_R, tmp_t[..., None]], dim=-1) # (B, M, 3, 4)
            key_d = key_d.permute(0,1,3,2).reshape(B, -1, 3) # (B, M4, 3)
            #key_d = key_d.permute(0,3,1,2).reshape(B, -1, 3) # (B, 4M, 3)
        else:
            key_d = key_d.reshape(B, self.num_cage_vertices*2, 3)
            # key_v = self.key_d_model(exp_z_v, z_ID_B).reshape(B, M, 3)
        return key_d
    
    def forward(self, source_vert, deform_vert, source_norm, deform_norm, mesh_data, epoch=0, out_kw=False, recon_out=True):
        """
        Trained with self-retargeting setting
        
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

        ### permute and sample points -------------------------------
        if not out_kw:
            N_range = torch.randint(100, N//6, (1,)).item()
            N_ = N - N_range
            randperm_idx = torch.randperm(N)[:N_]
            rearange_idx = torch.argsort(randperm_idx)
            
            source_vert_ = source_vert[:, randperm_idx]
            deform_vert_ = deform_vert[:, randperm_idx]
            source_norm_ = source_norm[:, randperm_idx]
            deform_norm_ = deform_norm[:, randperm_idx]
        else:
            source_vert_ = source_vert
            deform_vert_ = deform_vert
            source_norm_ = source_norm
            deform_norm_ = deform_norm
        ### ---------------------------------------------------------
        
        
        source_in = source_vert_
        deform_in = deform_vert_-source_vert_ # as a delta
        # deform_in = deform_vert # as a vertex
        
        if self.in_dim == 6:
            source_in = torch.cat([source_in, source_norm_], dim=-1)
            deform_in = torch.cat([deform_in, deform_norm_], dim=-1)            
        deform_in = torch.cat([deform_in, source_in], dim=-1)

        
        ### cage prediction -----------------------------------------
        if self.use_shp:
            z_ID_B = self.shape_model(source_in) # (B, 1, L)
            
            exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
            key_s_key_d = self.key_d_model(exp_z, z_ID_B)
        else:
            exp_z = self.exp_z_model(deform_in) # (B, 1, L)
            key_s_key_d = self.key_d_model(exp_z)
        
        key_s_key_d = self.reshape_key_d(key_s_key_d, B) # (B, 2M, 3)
        
        M = self.num_cage_vertices
        key_s, key_d = key_s_key_d[:,:M], key_s_key_d[:,M:]
        ### ---------------------------------------------------------


        ### weight prediction ---------------------------------------
        #source_w_in = torch.cat([source_in, exp_z.repeat(1,source_in.shape[1],1)], dim=-1)
        
        ### weight should be different for each shape
        # key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
        key_weight = self.key_weight_model(source_in, exp_z, N=self.NZ) # (B, N, M)
        # key_weight = self.key_weight_model(source_w_in, N=self.NZ) # (B, N, M)
        # --> (B, N, 4M) if self.opts.out_type == 2
        ### ---------------------------------------------------------
                  
        if out_kw:
            def_v = torch.einsum('bnc,bci->bni',key_weight,key_d) # (B, N, 3)
            src_v = torch.einsum('bnc,bci->bni',key_weight,key_s) # (B, N, 3)
            
        else:
            key_weight_Teach = self.key_weight_model_Teach(deform_in, N=self.NZ) # (B, N, M)
            def_v = torch.einsum('bnc,bci->bni',key_weight_Teach,key_d) # (B, N, 3)
            src_v = torch.einsum('bnc,bci->bni',key_weight_Teach,key_s) # (B, N, 3)

            
        if self.use_full_vertex:
            pred_deformed = def_v
            pred_source = src_v
        else:
            pred_deformed = def_v + source_vert_
            pred_source = src_v + source_vert_

        
        # supple networks -------------------------------------------
        ## necessary -- not really...
        if self.use_shp_recon and recon_out:
            recon_source = self.recon_shp_model[mesh_data](z_ID_B)
            recon_source = recon_source.reshape(B, -1, 3)
        else:
            recon_source = 0
        
        ## unnecessary
        if self.use_exp_recon and recon_out:
            recon_delta_v = self.recon_exp_model[mesh_data](exp_z)
            recon_delta_v = recon_delta_v.reshape(B, -1, 3)
            recon_deformed = recon_delta_v + source_vert
        else:
            recon_deformed = 0
        ### ---------------------------------------------------------  
        
        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight
            
        return (pred_deformed, key_weight, key_weight_Teach), recon_deformed, recon_source, exp_z, pred_source, (randperm_idx, rearange_idx)