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
from utils.exp_utils import plateau_hat_points

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
                 use_elu=False,
                 use_softplus=False,
                 use_least_N=False,
                 use_least_N_on_V=False,
                 least_number_of_zeros=256, # for sparsity (not used)
                 is_train=False,
                 tau=0.05,
                 device='cpu',
                 use_exp_recon=False, # was not necessary
                 use_shp_recon=True, # necessary for training, but not needed for inference
                 use_shp=True,
                 use_pou=True,
                 use_full_vertex=False, # default: false (= delta form)
                 no_activation=False,
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
        self.no_activation = no_activation
        self.use_pou = use_pou
        
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
        elif self.out_type == 3: # transform matrix (compskin setting)
            self.use_full_vertex = False
            self.out_dim = 6 # (6D + translation 3) will be reshaped into 3x4 matrix
            M_ = num_cage_vertices
            num_cage_vertices = num_cage_vertices * 4
            
            from utils.exp_utils import create_BN
            self._6D_to_rot_ = create_BN
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
            use_softplus=use_softplus,
            use_least_N=use_least_N,
            use_least_N_on_V=use_least_N_on_V,
            no_activation=no_activation,
            use_pou=use_pou,
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
                out_dim= M_*self.out_dim if (self.out_type == 2) or (self.out_type == 3) else M*self.out_dim,
                num_layers=self.num_layers, 
                use_style=True, out_type='global'
            ).to(device)
        else:
            self.key_d_model = LinearEncoder(
                in_dim=L,
                out_dim= M_*self.out_dim if (self.out_type == 2) or (self.out_type == 3) else M*self.out_dim,
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
            key_d = key_d.reshape(B,self.num_cage_vertices, 9)
            tmp_R, tmp_t = key_d[...,:6], key_d[...,6:]
            
            tmp_R = self._6D_to_rot_(tmp_R).reshape(B, -1, 3, 3)
            key_d = torch.cat([tmp_R, tmp_t[..., None]], dim=-1) # (B, M, 3, 4)
            key_d = key_d.permute(0,1,3,2).reshape(B, -1, 3) # (B, M4, 3)
            #key_d = key_d.permute(0,3,1,2).reshape(B, -1, 3) # (B, 4M, 3)
        
        elif self.out_type == 3:
            # cage transform matrix
            key_d = key_d.reshape(B,self.num_cage_vertices, 6)
            key_d = self._6D_to_rot_(key_d)# (B, M, 3, 4)
            key_d = key_d.permute(0,1,3,2).reshape(B, -1, 3) # (B, M4, 3)
            #key_d = key_d.permute(0,3,1,2).reshape(B, -1, 3) # (B, 4M, 3)
        
        else:
            key_d = key_d.reshape(B, self.num_cage_vertices, 3)
            # key_v = self.key_d_model(exp_z_v, z_ID_B).reshape(B, M, 3)
        
        return key_d
        
        
    def forward(self, source_vert, deform_vert, source_norm, deform_norm, mesh_data, hat_mask=None, epoch=0, out_kw=False):
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
        
        hat_mask = plateau_hat_points(source_vert)
            
        if self.in_type > 0:
            source_in = torch.cat([source_in, source_norm], dim=-1)
            deform_in = torch.cat([deform_in, deform_norm], dim=-1)
            
        deform_in = torch.cat([deform_in, source_in], dim=-1)
        
        if self.in_type==2:
            source_in = torch.cat([source_in, hat_mask], dim=-1)
            deform_in = torch.cat([deform_in, hat_mask], dim=-1)
        
        
        if self.use_shp:
            z_ID_B = self.shape_model(source_in) # (B, 1, L)
            
            exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
            key_d = self.key_d_model(exp_z, z_ID_B)
        else:
            exp_z = self.exp_z_model(deform_in) # (B, 1, L)
            key_d = self.key_d_model(exp_z)
        
        key_d = self.reshape_key_d(key_d, B)
            
        key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
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
            
            if self.in_type > 0:
                source_in_s = torch.cat([source_in_s, source_norm], dim=-1)
                deform_in_s = torch.cat([deform_in_s, deform_norm], dim=-1)
                
            deform_in_s = torch.cat([deform_in_s, source_in_s], dim=-1)
            
            if self.in_type == 2:
                source_in_s = torch.cat([source_in_s, hat_mask], dim=-1)
                deform_in_s = torch.cat([deform_in_s, hat_mask], dim=-1)
            
            if self.use_shp:
                exp_z_s = self.exp_z_model(deform_in_s, z_ID_B) # (B, 1, L)    
                key_s = self.key_d_model(exp_z_s, z_ID_B)
            else:
                exp_z_s = self.exp_z_model(deform_in_s) # (B, 1, L)    
                key_s = self.key_d_model(exp_z_s)
            key_s = self.reshape_key_d(key_s, B)
            
            pred_source = torch.einsum('bnc,bci->bni',key_weight,key_s)
        else:
            pred_source = 0
        
        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight
            
        return pred_deformed, recon_deformed, recon_source, exp_z, pred_source, hat_mask, key_weight

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
                
        if self.in_type > 0:
            tgt_in = torch.cat([tgt_in, tgt_neu_norm], dim=-1)
            src_in = torch.cat([src_in, src_neu_norm], dim=-1)
            deform_in_d = torch.cat([deform_in_d, src_def_norm], dim=-1)
            deform_in_s = torch.cat([deform_in_s, src_def_norm], dim=-1)
            
        deform_in_d = torch.cat([deform_in_d, src_in], dim=-1)
        deform_in_s = torch.cat([deform_in_s, src_in], dim=-1)

        if self.in_type == 2:
            src_hat_mask = plateau_hat_points(src_neu_vert)
            src_in = torch.cat([src_in, src_hat_mask], dim=-1)
            deform_in_s = torch.cat([deform_in_s, src_hat_mask], dim=-1)
            deform_in_d = torch.cat([deform_in_d, src_hat_mask], dim=-1)
            
            tgt_hat_mask = plateau_hat_points(tgt_neu_vert)
            tgt_in = torch.cat([tgt_in, tgt_hat_mask], dim=-1)
            
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
            key_s = self.reshape_key_d(key_s, B)
                
            key_weight = self.key_weight_model(tgt_in, N=self.NZ) # (B, N, K)
            # --> (B, N, 4K) if self.opts.out_type == 2
            
            delta_dv = torch.einsum('bnc,bci->bni',key_weight,key_d)
            delta_sv = torch.einsum('bnc,bci->bni',key_weight,key_s)
        
        
        if self.use_full_vertex:
            pred_deformed = delta_dv
            pred_source = delta_sv
        else:
            pred_deformed = delta_dv + tgt_neu_vert
            pred_source = delta_sv + tgt_neu_vert
        
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

        return pred_deformed, pred_source
                
    @torch.no_grad()
    def predict_coordinate(self, tgt_neu_vert, tgt_neu_norm):
        """
        Args:
            tgt_neu_vert (torch.tensor): [B, N, 3] target neutral mesh vertex positions
            tgt_neu_norm (torch.tensor): [B, N, 3] target neutral mesh vertex normals            
        Returns:
            key_weight, cooridnate w.r.t the cage vertices  [B, N, M]
            
        """
        B, N, _ = tgt_neu_vert.shape
        tgt_in = tgt_neu_vert
                
        if self.in_type > 0:
            tgt_in = torch.cat([tgt_in, tgt_neu_norm], dim=-1)

        if self.in_type == 2:
            tgt_hat_mask = plateau_hat_points(tgt_neu_vert)
            tgt_in = torch.cat([tgt_in, tgt_hat_mask], dim=-1)
        
        key_weight = self.key_weight_model(
            source_in,
            N=self.NZ # (not used!)
        ) # (B, N, M)
            
        return key_weight

    @torch.no_grad()
    def retarget_animation(
        self, 
        src_neu_vert,
        src_neu_norm, 
        src_def_vert,
        src_def_norm,
        key_weight,
        tgt_neu_vert=None, # required if the model is delta prediction!
        return_source=False,
    ):
        """
        Args:
            src_neu_vert (torch.tensor): [B, N, 3] source neutral mesh vertex positions
            src_neu_norm (torch.tensor): [B, N, 3] source neutral mesh vertex normals
            
            src_def_vert (torch.tensor): [B, N, 3] source deformed mesh vertex positions
            src_def_norm (torch.tensor): [B, N, 3] source deformed mesh vertex normals
            
            key_weight (torch.tensor):   [B, M, K] target neutral mesh vertex normals
            tgt_neu_vert (torch.tensor): [B, M, 3] target neutral mesh vertex positions

            return_source (bool): if True, returns reconstructed source mesh
        Returns:
            (pred_deformed): [B, M, 3] predicted target deformed mesh vertices
        """
        B, N, _ = src_neu_vert.shape
        src_neu_in = src_neu_vert        
        src_def_in = src_def_vert - src_neu_vert # as a delta (optional)
        
        
        if self.in_type > 0:
            src_neu_in = torch.cat([src_neu_in, src_neu_norm], dim=-1)
            src_def_in = torch.cat([src_def_in, src_def_norm], dim=-1)
            
        src_def_in = torch.cat([src_def_in, src_neu_in], dim=-1)

        if self.in_type == 2:
            src_hat_mask = plateau_hat_points(src_neu_vert)
            src_neu_in = torch.cat([src_neu_in, src_hat_mask], dim=-1)
            src_def_in = torch.cat([src_def_in, src_hat_mask], dim=-1)
        
        if self.use_shp:
            z_ID_B = self.shape_model(src_neu_in) # (B, 1, L)
            
            src_exp_z = self.exp_z_model(src_def_in, z_ID_B) # (B, 1, L)
            src_key_d = self.key_d_model(src_exp_z, z_ID_B) # (B, 3K)
        else:
            src_exp_z = self.exp_z_model(src_def_in) # (B, 1, L)
            src_key_d = self.key_d_model(src_exp_z) # (B, 3K)
        
        src_key_d = self.reshape_key_d(src_key_d, B) # (B, K, 3)
        
        tgt_def_v = torch.einsum('bnc,bci->bni', key_weight, src_key_d)
        
        if self.use_full_vertex:
            # absolute position
            pred_deformed = tgt_def_v
        else:
            # displacement
            pred_deformed = tgt_def_v + tgt_neu_vert

        # if return_source:
        #     exp_zs = self.exp_z_model(src_neu_in) # (B, 1, L)
        #     key_s = self.key_d_model(exp_zs) # (B, 3K)
        #     key_s = self.reshape_key_d(key_s, B)
        #     pred_source_v = torch.einsum('bnc,bci->bni', key_weight, key_s)
        #     return pred_deformed, src_key_d, pred_source_v
            
        return pred_deformed, src_key_d

    @torch.no_grad()
    def blendshape(
        self, 
        exp_z,
        key_weight,
        tgt_neu_vert=None, # required if the model is delta prediction!
        src_neu_vert=None,
        src_neu_norm=None,
    ):
        """
        Animate target mesh using blendshape coefficient
        *available if the model is trained with `align_latent==True`
        
        Args:
            exp_z (torch.tensor): [B, 1, 128] blendshape coefficient for the expression
            
            key_weight (torch.tensor):   [B, M, K] target neutral mesh vertex normals
            tgt_neu_vert (torch.tensor): [B, M, 3] target neutral mesh vertex positions

            src_neu_vert (torch.tensor): [B, N, 3] source neutral mesh vertex positions # iff self.use_shp==True
            src_neu_norm (torch.tensor): [B, N, 3] source neutral mesh vertex normals # iff self.use_shp==True
        Returns:
            (pred_deformed): [B, M, 3] predicted target deformed mesh vertices
        """
        B = exp_z.shape[0]
        
        if self.use_shp:
            src_in = src_neu_vert            
            if self.in_type > 0:
                src_in = torch.cat([src_in, src_neu_norm], dim=-1)            
            if self.in_type == 2:
                src_hat_mask = plateau_hat_points(src_neu_vert)
                src_in = torch.cat([src_in, src_hat_mask], dim=-1)
            z_ID_B = self.shape_model(src_in) # (B, 1, L)
            
            key_d = self.key_d_model(exp_z, z_ID_B) # (B, 3K)
        else:
            key_d = self.key_d_model(exp_z) # (B, 3K)
        
        key_d = self.reshape_key_d(key_d, B) # (B, K, 3)
        
        cage_v = torch.einsum('bnc,bci->bni', key_weight, key_d)
        
        if self.use_full_vertex:
            pred_deformed = cage_v
        else:
            pred_deformed = cage_v + tgt_neu_vert
        
        return pred_deformed, key_d
