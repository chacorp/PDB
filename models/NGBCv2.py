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
import torch.nn.functional as F
from utils.exp_utils import plateau_hat_points


class NeuralBarycentricCoordinatev2(nn.Module):
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
                 use_softplus=False,
                 use_least_N=False,
                 use_least_N_on_V=False,
                 no_activation=False,
                 least_number_of_zeros=256, # for sparsity (not used)
                 is_train=False,
                 tau=0.05,
                 device='cpu',
                 use_exp_recon=False, # was not necessary
                 use_shp_recon=True, # necessary for training, but not needed for inference
                 use_shp=True,
                 use_pou=True,
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
        self.no_activation = no_activation
        self.use_pou = use_pou
        
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
        elif self.in_type == 2:
            self.in_dim = 6+1
            in_dim_exp = 6+6+1
        else:
            raise NotImplementedError('in_type not implemented')
            
        if self.out_type == 0: # delta form
            self.use_full_vertex = False
            self.out_dim = 3 * 2
        elif self.out_type == 1: # vertex (linear precision)
            self.use_full_vertex = True
            self.out_dim = 3 * 2
        elif self.out_type == 2: # transform matrix
            self.use_full_vertex = False
            self.out_dim = 9 * 2 # (6D + translation 3) will be reshaped into 3x4 matrix
            M_ = num_cage_vertices
            num_cage_vertices = num_cage_vertices * 4
            
            from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
            self._6D_to_rot_ = _6D_to_rot_
            
        elif self.out_type == 3: # vertex + normal + displacement
            self.use_full_vertex = True
            self.out_dim = 3 * 3
        else:
            raise NotImplementedError('out_type not implemented')
            
        from models.encoder import LinearEncoder,LinearEncoder2
        
        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros

        # coordinate predictor
        self.key_weight_model = LinearEncoder(
            in_dim=self.in_dim, out_dim=M, 
            use_softmax=use_softmax,
            use_relu=use_relu, # default setting
            use_elu=use_elu,
            use_softplus=use_softplus,
            use_least_N=use_least_N,
            use_least_N_on_V=use_least_N_on_V,
            no_activation=no_activation,
            use_pou=use_pou,
        ).to(device)

        # expression encoder
        self.exp_z_model = LinearEncoder(
            in_dim=in_dim_exp,
            out_dim=L,
            num_layers=self.num_layers, 
            out_type='global',
        ).to(device)
        
        # cage displacement predictor (predict both original cage and deformed cage)
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
            key_d = key_d.reshape(B, self.num_cage_vertices*2, 9)
            tmp_R, tmp_t = key_d[...,:6], key_d[...,6:]
            
            tmp_R = self._6D_to_rot_(tmp_R).reshape(B, -1, 3, 3)
            key_d = torch.cat([tmp_R, tmp_t[..., None]], dim=-1) # (B, M, 3, 4)
            key_d = key_d.permute(0,1,3,2).reshape(B, -1, 3) # (B, M4, 3)
            #key_d = key_d.permute(0,3,1,2).reshape(B, -1, 3) # (B, 4M, 3)
        
        elif self.out_type == 22:
            # cage transform matrix
            key_d = key_d.reshape(B,self.num_cage_vertices*2, 6)
            key_d = self._6D_to_rot_(key_d)# (B, M, 3, 4)
            key_d = key_d.permute(0,1,3,2).reshape(B, -1, 3) # (B, M4, 3)
            #key_d = key_d.permute(0,3,1,2).reshape(B, -1, 3) # (B, 4M, 3)
        
        else:
            if self.out_type ==3:
                key_d = key_d.reshape(B, self.num_cage_vertices*3, 3)
                
                M = key_d.shape[1] // 3
                key_s, key_n, key_d = key_d[:,:M], key_d[:,M:2*M], key_d[:,2*M:]
                ### normalize !!!!!!!!! ######################################################
                key_n = F.normalize(key_n, dim=-1)
                ##############################################################################
            else:
                key_n = None
                key_d = key_d.reshape(B, self.num_cage_vertices*2, 3)
                
                M = key_d.shape[1] // 2
                key_s, key_d = key_d[:,:M], key_d[:,M:]+key_d[:,:M]
        
        return key_s, key_n, key_d
    
    def forward(self, source_vert, deform_vert, source_norm, deform_norm, hat_mask=None, mesh_data=0, epoch=0, out_kw=False, recon_out=True):
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
        
        source_in = source_vert
        deform_in = deform_vert-source_vert # as a delta
        # deform_in = deform_vert # as a vertex
        
        hat_mask = plateau_hat_points(source_vert)
        
        if self.in_type > 0:
            source_in = torch.cat([source_in, source_norm], dim=-1)
            deform_in = torch.cat([deform_in, deform_norm], dim=-1)
        deform_in = torch.cat([deform_in, source_in], dim=-1)
        
        if self.in_type == 2:            
            source_in = torch.cat([source_in, hat_mask], dim=-1)
            deform_in = torch.cat([deform_in, hat_mask], dim=-1)

        
        ### cage prediction -----------------------------------------
        exp_z = self.exp_z_model(deform_in) # (B, 1, L)
        key_s_key_d = self.key_d_model(exp_z)
        
        key_s, key_n, key_d = self.reshape_key_d(key_s_key_d, B) # (B, 2M, 3)
        # key_s_key_d = self.reshape_key_d(key_s_key_d, B) # (B, 2M, 3)
        
        # # M = self.num_cage_vertices
        # M = key_s_key_d.shape[1]//2
        # # import pdb;pdb.set_trace()
        # key_s, key_d = key_s_key_d[:,:M], key_s_key_d[:,M:]+key_s_key_d[:,:M]
        ### ---------------------------------------------------------

        
        ### weight prediction ---------------------------------------
        key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
        ### ---------------------------------------------------------
                
        def_v = torch.einsum('bnc,bci->bni',key_weight,key_d) # (B, N, 3)
        src_v = torch.einsum('bnc,bci->bni',key_weight,key_s) # (B, N, 3)

        if self.out_type == 3:
            pred_deformed = def_v #+ source_vert
            pred_source = src_v

            cage_in = torch.cat([key_s, key_n], dim=-1)
            cage_w = self.key_weight_model(cage_in, N=self.NZ) # for Lagrange property
            
        else:
            cage_w=None
            if self.use_full_vertex:
                pred_deformed = def_v
                pred_source = src_v
            else:
                pred_deformed = def_v + source_vert
                pred_source = src_v + source_vert

        
        # supple networks -------------------------------------------
        recon_source = 0
        recon_deformed = 0
        # ## necessary -- not really...
        # if self.use_shp_recon and recon_out:
        #     recon_source = self.recon_shp_model[mesh_data](z_ID_B)
        #     recon_source = recon_source.reshape(B, -1, 3)
        # else:
        #     recon_source = 0
        
        # ## unnecessary
        # if self.use_exp_recon and recon_out:
        #     recon_delta_v = self.recon_exp_model[mesh_data](exp_z)
        #     recon_delta_v = recon_delta_v.reshape(B, -1, 3)
        #     recon_deformed = recon_delta_v + source_vert
        # else:
        #     recon_deformed = 0
        ### ---------------------------------------------------------  
        
        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight, cage_w, key_s
            
        return pred_deformed, recon_deformed, recon_source, exp_z, pred_source, hat_mask, key_weight, cage_w, key_s #, (randperm_idx, rearange_idx)
    
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
        #deform_in_s = src_neu_vert-src_neu_vert # as a delta
                
        if self.in_type > 0:
            tgt_in = torch.cat([tgt_in, tgt_neu_norm], dim=-1)
            src_in = torch.cat([src_in, src_neu_norm], dim=-1)
            deform_in_d = torch.cat([deform_in_d, src_def_norm], dim=-1)
            # deform_in_s = torch.cat([deform_in_s, src_def_norm], dim=-1)
            
        deform_in_d = torch.cat([deform_in_d, src_in], dim=-1)
        #deform_in_s = torch.cat([deform_in_s, src_in], dim=-1)
                
        if self.in_type == 2:
            src_hat_mask = plateau_hat_points(src_neu_vert)
            src_in = torch.cat([src_in, src_hat_mask], dim=-1)
            deform_in_d = torch.cat([deform_in_d, src_hat_mask], dim=-1)
            
            tgt_hat_mask = plateau_hat_points(tgt_neu_vert)
            tgt_in = torch.cat([tgt_in, tgt_hat_mask], dim=-1)

        
        with torch.no_grad():
            if self.use_shp:
                z_ID_B = self.shape_model(src_in) # (B, 1, L)
                
                exp_z = self.exp_z_model(deform_in_d, z_ID_B) # (B, 1, L)
                key_s_key_d = self.key_d_model(exp_z, z_ID_B)
            else:
                exp_z = self.exp_z_model(deform_in_d) # (B, 1, L)
                key_s_key_d = self.key_d_model(exp_z)
            
            #key_s_key_d = self.reshape_key_d(key_s_key_d, B) # (B, 2M, 3)
            key_s, key_n, key_d = self.reshape_key_d(key_s_key_d, B) # (B, 2M, 3)
            
            ### weight should be different for each shape
            key_weight = self.key_weight_model(tgt_in, N=self.NZ) # (B, N, M)
            # --> (B, N, 4M) if self.opts.out_type == 2
        
            # M = self.num_cage_vertices
            # M = key_s_key_d.shape[1]//2
            # key_s, key_d = key_s_key_d[:,:M], key_s_key_d[:,M:]+key_s_key_d[:,:M]
            
            #print(key_d.shape)
            def_v = torch.einsum('bnc,bci->bni',key_weight,key_d) # (B, N, 3)
            src_v = torch.einsum('bnc,bci->bni',key_weight,key_s) # (B, N, 3)
        
        if self.use_full_vertex:
            pred_deformed = def_v
            pred_source = src_v
        else:
            pred_deformed = def_v + tgt_neu_vert
            pred_source = src_v + tgt_neu_vert
        
        if out_kw:
            return pred_deformed, pred_source, exp_z, key_d, exp_z, key_s, key_weight
        
        return pred_deformed, pred_source

