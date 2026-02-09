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
from models.encoder import LinearEncoder
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
                #  use_lbs_joint_center=False, # default > apply_lbs_no_center()
                #  use_exp_joint_predict=False,
                #  use_lbs=False,
                ):
        super().__init__()
        self.opts = opts
        
        self.device = device
        self.is_train = is_train
        
        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim

        self.use_softmax=use_softmax
        self.use_relu=use_relu
        self.use_elu=use_elu
        self.use_softplus=use_softplus
        self.use_least_N=use_least_N
        self.use_least_N_on_V=use_least_N_on_V

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
        self.in_dim_exp = in_dim_exp

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
            self.M_ = M_
            num_cage_vertices = num_cage_vertices * 4
            
            from utils.exp_utils import create_BN
            self._6D_to_rot_ = create_BN
        else:
            raise NotImplementedError('out_type not implemented')
                    
        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros
        
        self.M = M
        self.L = L
        self.NZ = NZ

        # -------------------------------------
        # [added] LBS Hybrid related configurations
        # -----------------------------------
        self.use_lbs_joint_center = self.opts.use_lbs_joint_center
        self.use_exp_joint_predict = self.opts.use_exp_joint_predict
        self.use_lbs = self.opts.use_lbs 
        self.num_lbs_joints = self.opts.num_lbs_joints
        self.use_joint_predict = self.opts.use_joint_predict
        self.use_weighted_joint_pos = self.opts.use_weighted_joint_pos
        self.no_use_translation = self.opts.no_use_translation
        
        # expression encoder for lbs
        self.lbs_exp_z_model = LinearEncoder(
            in_dim=in_dim_exp,
            out_dim=L,
            num_layers=self.num_layers, 
            out_type='global',
        ).to(device)

        # -------------------------------------
        # [added] LBS Hybrid related options
        # -------------------------------------
        self.lbs_weight_model = LinearEncoder(
            in_dim=self.in_dim,
            use_softmax=self.use_softmax,
            use_relu=self.use_relu, # default setting
            use_elu=self.use_elu,
            use_softplus=self.use_softplus,
            use_least_N=self.use_least_N,
            use_least_N_on_V=self.use_least_N_on_V,
            no_activation=self.no_activation,
            out_dim=self.num_lbs_joints,
            use_pou=True, # LBS weight는 반드시 sum=1
        ).to(device)
        
        # -------------------------------------
        # [added] LBS pose / transform predictor (global)
        # 6D rotation + translation (3) = 9 per joint
        # -------------------------------------
        if self.no_use_translation: 
            self.lbs_pose_model = LinearEncoder(
                in_dim=L,
                out_dim=self.num_lbs_joints*6,
                out_type='global',
            ).to(device)
        else:
            self.lbs_pose_model = LinearEncoder(
                in_dim=L,
                out_dim=self.num_lbs_joints*9, # 6D
                out_type='global',
            ).to(device)
        
        # import pdb;pdb.set_trace()
        ## 지난번 말해본 케이지 버텍스 투 트랜스폼 매트릭스 맵핑 네트워크 형태로 다시 자보기
        ## vertex norm = 0 & var(V) = var(P) 를 로스로 동시에 주면 메쉬 내부에 어느정도 위치시킬 수 있지않을까 (해보자) 
        print(f"LBS encoders loaded! number of joints are set to {self.num_lbs_joints}")
        if self.use_joint_predict:
            if self.use_exp_joint_predict: # don't use it, or bone(len between joints) will be non-rigid
                self.lbs_joint_center_model = LinearEncoder(
                in_dim=L, # consider expression for center estimation
                out_dim=self.num_lbs_joints*3, # joint position
                out_type='global',
                ).to(device)
                print(f"Expression is considered for center prediction")
            else: 
                self.lbs_joint_center_model = LinearEncoder(
                in_dim=self.in_dim, # don't consider expression for center estimation (cetner doesn't change per deformation)
                out_dim=self.num_lbs_joints*3, # joint position
                out_type='global',
                ).to(device)
                print(f"Only shape is considered for center prediction")
        else:
            print(f"no joint position network prediction")

        if not self.use_lbs:
            self.load_CBD_brach(self.use_lbs)

        from utils.exp_utils import from_6D_to_rotation_matrix_torch
        ## 3x3 회전 행렬에서 두 개의 열 벡터(unconstrained 3D vectors)만을 사용하여 회전을 인코딩
            ## 실제 회전 행렬은 이 두 벡터에 그램-슈미트(Gram-Schmidt) 직교화 과정을 적용하여 세 번째 열 벡터 \(c_{3}\)를 계산함으로써 완전한 3x3 회전 행렬을 재구성
        self._6D_to_rot_lbs = from_6D_to_rotation_matrix_torch

    def load_CBD_brach(self, use_lbs=False):
        # expression encoder for cbd
        self.cbd_exp_z_model = LinearEncoder(
            in_dim=self.in_dim_exp,
            out_dim=self.L,
            num_layers=self.num_layers, 
            out_type='global',
        ).to(self.device)

        # shape encoder
        # if self.use_shp:
        #     self.shape_model = LinearEncoder(
        #         in_dim=self.in_dim,
        #         out_dim=L,
        #         out_type='global'
        #     ).to(device)

        # coordinate predictor
        self.key_weight_model = LinearEncoder(
            in_dim=self.in_dim, out_dim=self.M, 
            use_softmax=self.use_softmax,
            use_relu=self.use_relu, # default setting
            use_elu=self.use_elu,
            use_softplus=self.use_softplus,
            use_least_N=self.use_least_N,
            use_least_N_on_V=self.use_least_N_on_V,
            no_activation=self.no_activation,
            use_pou=self.use_pou,
        ).to(self.device)

        # cage displacement predictor
        self.key_d_model = LinearEncoder(
            in_dim=self.L,
            out_dim= self.M_*self.out_dim if (self.out_type == 2) or (self.out_type == 3) else self.M*self.out_dim,
            num_layers=self.num_layers, 
            out_type='global'
        ).to(self.device)

        if use_lbs:
            print("## Stage 2 started !! CBD branch models successfully initialized !!!!!! ##\n")
        else:
            print("## LBS not used !!!!! CBD branch models successfully initialized !!!!!! ##\n")

    # -------------------------------------
    # [added] LBS formulation
    # -------------------------------------
    def apply_lbs_no_center(self,source_vert, W_lbs, T_lbs):
        """
        LBS-ish formulation w/o joint position in consideration
        Args:
            source_vert (torch.tensor): [B, N, 3] source mesh vertices
            W_lbs (torch.tensor): [B, N, J] LBS weights
            T_lbs (torch.tensor): [B, J, 3, 4] LBS transforms
        Returns:
            pred_deformed: [B, N, 3] predicted deformed mesh vertices
        """
        B, N, _ = source_vert.shape
        J = W_lbs.shape[-1]
        
        if self.no_use_translation:
            R = T_lbs # (B, J, 3, 3)
            # v = source_vert[:,:,None,:] # (B, N, 1, 3)
            v = torch.einsum('bjik,bnk->bnji', R, source_vert) # rotate, (B, N, J, 3)
        else:
            R = T_lbs[..., :3] # (B, J, 3, 3)
            t = T_lbs[..., 3] # (B, J, 3)
            # v = source_vert[:,:,None,:] # (B, N, 1, 3)
            v = torch.einsum('bjik,bnk->bnji', R, source_vert) # rotate, (B, N, J, 3)
            v = v + t[:,None,:,:] # translate, (B, N, J, 3)
        
        v_lbs = torch.einsum('bnj,bnjc->bnc', W_lbs, v)
        return v_lbs
    
    def apply_lbs(self, source_vert, W_lbs, T_lbs, C_lbs):
        """
        standard LBS formulation only using R in (R|t) as T (t is set as C_lbs)
        Args:
            source_vert (torch.Tensor): [B, N, 3] source mesh vertices
            W_lbs       (torch.Tensor): [B, N, J] LBS weights (ideally sum_j = 1)
            T_lbs       (torch.Tensor): [B, J, 3, 3] LBS transforms (R)
            C_lbs       (torch.Tensor): [B, J, 3] joint centers (t: pivot points)

        Returns:
            v_lbs (torch.Tensor): [B, N, 3] LBS-deformed vertices
        """
        # shapes
        B, N, _ = source_vert.shape
        assert W_lbs.dim() == 3 and W_lbs.shape[0] == B and W_lbs.shape[1] == N
        J = W_lbs.shape[-1]

        # assert T_lbs.shape == (B, J, 3, 4), f"T_lbs must be (B,J,3,4), got {T_lbs.shape}"
        if self.no_use_translation:
            assert T_lbs.shape == (B, J, 3, 3), f"T_lbs must be (B,J,3,3), got {T_lbs.shape}"
            assert C_lbs.shape == (B, J, 3),     f"C_lbs must be (B,J,3), got {C_lbs.shape}"
            R = T_lbs
        else:
            assert T_lbs.shape == (B, J, 3, 4), f"T_lbs must be (B,J,3,3), got {T_lbs.shape}"
            assert C_lbs.shape == (B, J, 3),     f"C_lbs must be (B,J,3), got {C_lbs.shape}"
            # split transform
            R = T_lbs[..., :3]   # (B, J, 3, 3)
            # t = T_lbs[..., 3]    # (B, J, 3)

        ## 즉, 센터를 빼고 회전 → 다시 센터를 더하는 건 좌표계 변환의 필수 절차다.
        # move to joint-local coordinates: v - c_j 
        v_local = source_vert[:, :, None, :] - C_lbs[:, None, :, :]  # (B, N, J, 3)
        # v_local = source_vert[:, :, None, :]# (B, N, J, 3)

        # rotate: R_j @ (v - c_j)
        v_rot = torch.einsum('bjik,bnjk->bnji', R, v_local)          # (B, N, J, 3)

        # move back: + c_j + t_j
        # v_joint = v_rot + C_lbs[:, None, :, :] + t[:, None, :, :]    # (B, N, J, 3)
        v_joint = v_rot + C_lbs[:, None, :, :]     # (B, N, J, 3)
        # v_joint = v_rot + t[:, None, :, :]    # (B, N, J, 3)

        # blend across joints
        v_lbs = torch.einsum('bnj,bnjc->bnc', W_lbs, v_joint)         # (B, N, 3)
        return v_lbs
    
    def joint_position_from_weights(self, verts, W, eps=1e-8):
        """
        verts: (B, N, 3)  source vertices (bar V)
        W:     (B, N, J)  skinning weights (w_{i,k})
        return:
        C_bar: (B, J, 3)
        """
        # numerator: sum_i w_{i,k} * v_i
        num = torch.einsum('bnj,bnc->bjc', W, verts)          # (B,J,3)
        den = W.sum(dim=1, keepdim=False).unsqueeze(-1)      # (B,J,1)
        C_bar = num / (den + eps)
        return C_bar
    
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
    
    def forward(self, 
                source_vert, 
                deform_vert, 
                source_norm,
                deform_norm, 
                mesh_data, 
                hat_mask=None, 
                epoch=0, 
                out_kw=False, # False if baseline
                stage=None,
                ):
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
        device = deform_vert.device
        dtype = deform_vert.dtype
        
        M = self.num_cage_vertices
        J = self.num_lbs_joints
        
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
        
        ## ===============================
        ## [added] LBS branch
        ## ===============================
        W_lbs = torch.zeros((B, N, J), device=device, dtype=dtype) 
        T_lbs = torch.zeros((B, N, 3, 4), device=device, dtype=dtype)
        rigid_v = torch.zeros((B, N, 3), device=device, dtype=dtype)
        delta_v = torch.zeros((B, N, 3), device=device, dtype=dtype)
        key_d = torch.zeros((B,M,3), device=device,dtype=dtype)
        key_weight = torch.zeros((B,N,M), device=device,dtype=dtype)
        
        if self.use_lbs and stage != None:  ## NBC++, lbs 와 함께 쓰는 경우
            if self.use_shp:
                z_ID_B = self.shape_model(source_in) # (B, 1, L)
                if stage == 1:
                    exp_z_lbs = self.lbs_exp_z_model(deform_in, z_ID_B) # (B, 1, L)
                elif stage == 2:
                    with torch.no_grad():
                        exp_z_lbs = self.lbs_exp_z_model(deform_in, z_ID_B)
                    if not self.opts.use_hyb_delta_lbs_input: # only get inside if not use delta_lbs_input for hybrid training (2nd stage)    
                        exp_z_cbd = self.cbd_exp_z_model(deform_in, z_ID_B) # (B, 1, L)
                        key_d = self.key_d_model(exp_z_cbd, z_ID_B) # cage displacement 
            else:
                if stage == 1:
                    exp_z_lbs = self.lbs_exp_z_model(deform_in) # (B, 1, L)
                elif stage == 2:
                    with torch.no_grad():
                        exp_z_lbs = self.lbs_exp_z_model(deform_in)
                    if not self.opts.use_hyb_delta_lbs_input:
                        exp_z_cbd = self.cbd_exp_z_model(deform_in) # (B, 1, L)
                        key_d = self.key_d_model(exp_z_cbd)

            if stage == 2 and self.opts.use_hyb_delta_lbs_input == False:
                ## getting delta_v
                key_d = self.reshape_key_d(key_d, B)
                key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
                # --> (B, N, 4M) if self.opts.out_type == 2
                delta_v = torch.einsum('bnc,bci->bni',key_weight,key_d)
            
            if stage == 1:
                ## LBS weights
                W_lbs = self.lbs_weight_model(source_in) # (B, N, J)
                ## LBS transforms
                if self.no_use_translation:
                    T = self.lbs_pose_model(exp_z_lbs).reshape(B, self.num_lbs_joints, 6)  # (B, J, 6)
                else:
                    T = self.lbs_pose_model(exp_z_lbs).reshape(B, self.num_lbs_joints, 9)  # (B, J, 9)
            else: 
                with torch.no_grad(): # preventing OOM (out of memory)
                    ## LBS weights
                    W_lbs = self.lbs_weight_model(source_in) # (B, N, J)
                    ## LBS transforms
                    if self.no_use_translation:
                        T = self.lbs_pose_model(exp_z_lbs).reshape(B, self.num_lbs_joints, 6)  # (B, J, 6)
                    else:
                        T = self.lbs_pose_model(exp_z_lbs).reshape(B, self.num_lbs_joints, 9)  # (B, J, 9)
            
            # with torch.no_grad():
            #     s = W_lbs.sum(dim=-1)          # (B, N)
            #     print("W_sum: mean/min/max", s.mean().item(), s.min().item(), s.max().item())
            #     print("W_min/max", W_lbs.min().item(), W_lbs.max().item())
                        
            if self.no_use_translation:
                R6 = T # (B, J, 6)
            else:
                R6, t = T[..., :6], T[..., 6:] # (B, J, 6), (B, J, 3)
            
            R = self._6D_to_rot_lbs(R6).reshape(B, self.num_lbs_joints, 3, 3) # (B, J, 3, 3)
            
            if self.no_use_translation:
                T_lbs = R # (B, J, 3, 3)
            else:
                T_lbs = torch.cat([R, t[..., None]], dim=-1) # (B, J, 3, 4)
            
            if not self.use_lbs_joint_center:
                rigid_v = self.apply_lbs_no_center(source_vert, W_lbs, T_lbs) # (B, N, 3)
            else:
                if self.use_joint_predict: # using network to predict joint, loss calculation should be needed in the training loop
                    if self.use_exp_joint_predict:
                        C_lbs = self.lbs_joint_center_model(exp_z_lbs).reshape(B, self.num_lbs_joints, 3) # (B, J, 3)
                    else: # default is this
                        C_lbs = self.lbs_joint_center_model(source_in).reshape(B, self.num_lbs_joints, 3) # (B, J, 3)
                    rigid_v = self.apply_lbs(source_vert, W_lbs, T_lbs, C_lbs)
                else:
                    # C_bar = self.joint_position_from_weights(source_vert, W_lbs) # (B,J,3)
                    # rigid_v = self.apply_lbs(source_vert, W_lbs, T_lbs, C_bar)
                    if self.use_weighted_joint_pos:
                        C_bar = self.joint_position_from_weights(source_vert, W_lbs) # (B,J,3)
                    else: # just use t as C_bar, and always dimension should be 9 for pose_model
                        C_bar = t 
                    rigid_v = self.apply_lbs(source_vert, W_lbs, T_lbs, C_bar)
                    
            if self.opts.use_hyb_delta_lbs_input == True:
                deform_in = deform_vert - rigid_v # as a delta over lbs output
                deform_in = torch.cat([deform_in, deform_norm], dim=-1)
                deform_in = torch.cat([deform_in, source_in], dim=-1)
                exp_z_cbd = self.cbd_exp_z_model(deform_in) # (B, 1, L)
                key_d = self.key_d_model(exp_z_cbd)
                key_d = self.reshape_key_d(key_d, B)
                key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
                # --> (B, N, 4M) if self.opts.out_type == 2
                delta_v = torch.einsum('bnc,bci->bni',key_weight,key_d)

        else:   ## training baseline (NBC)
            if self.use_shp:
                z_ID_B = self.shape_model(source_in) # (B, 1, L)
                exp_z_cbd = self.cbd_exp_z_model(deform_in, z_ID_B) # (B, 1, L)
                key_d = self.key_d_model(exp_z_cbd, z_ID_B) # cage displacement 
            else:
                exp_z_cbd = self.cbd_exp_z_model(deform_in) # (B, 1, L)
                key_d = self.key_d_model(exp_z_cbd)
            rigid_v = source_vert 
            ## getting delta_v
            key_d = self.reshape_key_d(key_d, B)
            key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
            # --> (B, N, 4M) if self.opts.out_type == 2
            delta_v = torch.einsum('bnc,bci->bni',key_weight,key_d)
        
        ## ===============================
        ## [added] final composition
        ## ===============================    
        if self.use_lbs: # training lbs hybrid
            pred_deformed = rigid_v + delta_v
            # if self.use_full_vertex: # full vertex prediction from LBS
            #     pred_deformed = rigid_v + delta_v
            # else: # delta prediction from LBS
            #     rigid_delta = rigid_v - source_vert
            #     pred_deformed = rigid_delta + delta_v + source_vert
            
        else: # trainig baseline (original as it is)
            if self.use_full_vertex: # full vertex prediction from CBD
                pred_deformed = delta_v
            else:
                pred_deformed = delta_v + source_vert
        # if self.use_lbs:  # LBS hybrid
        #     # rigid_v: already in world coords (after LBS)
        #     # delta_v: residual displacement in world coords (CBD)
        #     pred_deformed = rigid_v + delta_v
        # else:  # baseline
        #     pred_deformed = delta_v + source_vert
        
        ## ===============================
        ## [old] final composition
        ## ===============================
        recon_source = 0
        recon_deformed = 0
        pred_source = 0
        
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
            deform_in_s = source_vert - source_vert # as a delta    
            if self.use_lbs:
                if self.in_type > 0:
                    source_in_s = torch.cat([source_in_s, source_norm], dim=-1)
                    # why deform norm? 
                    deform_in_s = torch.cat([deform_in_s, deform_norm], dim=-1)
                deform_in_s = torch.cat([deform_in_s, source_in_s], dim=-1)
                if self.in_type == 2:
                    source_in_s = torch.cat([source_in_s, hat_mask], dim=-1)
                    deform_in_s = torch.cat([deform_in_s, hat_mask], dim=-1)
                
                if stage == 1: # from LBS
                    # W_lbs = torch.zeros((B, N, J), device=device, dtype=dtype) 
                    # T_lbs = torch.zeros((B, N, 3, 4), device=device, dtype=dtype)
                    # rigid_v = torch.zeros((B, N, 3), device=device, dtype=dtype)

                    if self.use_shp:
                        exp_z_s = self.lbs_exp_z_model(deform_in_s, z_ID_B) # (B, 1, L)    
                        key_s = self.lbs_pose_model(exp_z_s, z_ID_B)
                    else:
                        exp_z_s = self.lbs_exp_z_model(deform_in_s) # (B, 1, L)    
                        key_s = self.lbs_pose_model(exp_z_s)

                    ## LBS weights
                    W_lbs = self.lbs_weight_model(source_in_s) # (B, N, J)
                    ## LBS transforms
                    if self.no_use_translation:
                        T = key_s.reshape(B, self.num_lbs_joints, 6)  # (B, J, 6)
                        R6 = T # (B, J, 6), (B, J, 3)
                        R = self._6D_to_rot_lbs(R6).reshape(B, self.num_lbs_joints, 3, 3) # (B, J, 3, 3)
                        T_lbs = R # (B, J, 3, 3)
                    else:
                        T = key_s.reshape(B, self.num_lbs_joints, 9)  # (B, J, 9)
                        R6, t = T[..., :6], T[..., 6:] # (B, J, 6), (B, J, 3)
                        R = self._6D_to_rot_lbs(R6).reshape(B, self.num_lbs_joints, 3, 3) # (B, J, 3, 3)
                        T_lbs = torch.cat([R, t[..., None]], dim=-1) # (B, J, 3, 4)
                        
                    if not self.use_lbs_joint_center:
                        pred_source = self.apply_lbs_no_center(source_vert, W_lbs, T_lbs) # (B, N, 3)
                    else:
                        if self.use_joint_predict: # using network to predict joint, loss calculation should be needed in the training loop
                            if self.use_exp_joint_predict:
                                C_lbs = self.lbs_joint_center_model(exp_z).reshape(B, self.num_lbs_joints, 3) # (B, J, 3)
                            else: # default is this
                                C_lbs = self.lbs_joint_center_model(source_in).reshape(B, self.num_lbs_joints, 3) # (B, J, 3)
                            pred_source = self.apply_lbs(source_vert, W_lbs, T_lbs, C_lbs)
                        else:
                            if self.use_weighted_joint_pos:
                                C_bar = self.joint_position_from_weights(source_vert, W_lbs) # (B,J,3)
                            else: # just use t as C_bar
                                C_bar = t 
                            pred_source = self.apply_lbs(source_vert, W_lbs, T_lbs, C_bar)

                else: # from CBD
                    if self.use_shp:
                        exp_z_s = self.cbd_exp_z_model(deform_in_s, z_ID_B) # (B, 1, L)    
                        key_s = self.key_d_model(exp_z_s, z_ID_B)
                    else:
                        exp_z_s = self.cbd_exp_z_model(deform_in_s) # (B, 1, L)    
                        key_s = self.key_d_model(exp_z_s)
                    key_s = self.reshape_key_d(key_s, B)
                    pred_source = torch.einsum('bnc,bci->bni',key_weight,key_s)
            else:
                # deform_in_s = source_vert # as a vertex
                if self.in_type > 0:
                    source_in_s = torch.cat([source_in_s, source_norm], dim=-1)
                    # why deform norm? 
                    deform_in_s = torch.cat([deform_in_s, deform_norm], dim=-1)
                deform_in_s = torch.cat([deform_in_s, source_in_s], dim=-1)
                if self.in_type == 2:
                    source_in_s = torch.cat([source_in_s, hat_mask], dim=-1)
                    deform_in_s = torch.cat([deform_in_s, hat_mask], dim=-1)
                if self.use_shp:
                    exp_z_s = self.cbd_exp_z_model(deform_in_s, z_ID_B) # (B, 1, L)    
                    key_s = self.key_d_model(exp_z_s, z_ID_B)
                else:
                    exp_z_s = self.cbd_exp_z_model(deform_in_s) # (B, 1, L)    
                    key_s = self.key_d_model(exp_z_s)
                key_s = self.reshape_key_d(key_s, B)
                pred_source = torch.einsum('bnc,bci->bni',key_weight,key_s)
        else:
            pred_source = 0
        
        if self.use_lbs:
            out_kw = True
            if stage == 1:
                exp_z = exp_z_lbs
            elif stage == 2:
                exp_z = exp_z_cbd

            if out_kw:
                if self.opts.vis_joint_pos:
                    if self.use_joint_predict:
                        C_lbs = C_lbs
                    else: 
                        C_lbs = C_bar
                    
                    return pred_deformed, recon_deformed, recon_source, exp_z, pred_source, hat_mask, key_d, key_weight, W_lbs, T_lbs, C_lbs
                else:
                    ## added W_lbs, T_lbs
                    return pred_deformed, recon_deformed, recon_source, exp_z, pred_source, hat_mask, key_d, key_weight, W_lbs, T_lbs
                
            return pred_deformed, recon_deformed, recon_source, exp_z, pred_source, hat_mask, key_weight
        else:
            # out_kw = False
            if out_kw:
                return pred_deformed, recon_deformed, recon_source, exp_z_cbd, key_d, key_weight
            return pred_deformed, recon_deformed, recon_source, exp_z_cbd, pred_source, hat_mask, key_weight
            
        
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
    
     
class NeuralGeneralizedBarycentricCoordinateLBS(nn.Module):
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
                #  use_lbs_joint_center=False, # default > apply_lbs_no_center()
                #  use_exp_joint_predict=False,
                #  use_lbs=False,
                ):
        super().__init__()
        self.opts = opts
        
        self.device = device
        self.is_train = is_train
        
        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim

        self.use_softmax=use_softmax
        self.use_relu=use_relu
        self.use_elu=use_elu
        self.use_softplus=use_softplus
        self.use_least_N=use_least_N
        self.use_least_N_on_V=use_least_N_on_V

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
        
        # if self.in_type == 0:
        #     self.in_dim = 3
        #     in_dim_exp = self.in_dim*2
        # elif self.in_type == 1:
        #     self.in_dim = 6
        #     in_dim_exp = self.in_dim*2
        # elif self.in_type == 2:
        #     self.in_dim = 6+1
        #     in_dim_exp = 6+6+1
        # else:
        #     raise NotImplementedError('in_type not implemented')
        self.in_dim = 6
        in_dim_exp = self.in_dim*2
        self.in_dim_exp = in_dim_exp

        self.use_full_vertex = True
        self.out_dim = 3
        # if self.out_type == 0: # delta form
        #     self.use_full_vertex = False
        #     self.out_dim = 3
        # elif self.out_type == 1: # vertex (linear precision)
        #     self.use_full_vertex = True
        #     self.out_dim = 3
        # elif self.out_type == 2: # transform matrix
        #     self.use_full_vertex = False
        #     self.out_dim = 9 # (6D + translation 3) will be reshaped into 3x4 matrix
        #     M_ = num_cage_vertices
        #     num_cage_vertices = num_cage_vertices * 4
            
        #     from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
        #     self._6D_to_rot_ = _6D_to_rot_
        # elif self.out_type == 3: # transform matrix (compskin setting)
        #     self.use_full_vertex = False
        #     self.out_dim = 6 # (6D + translation 3) will be reshaped into 3x4 matrix
        #     M_ = num_cage_vertices
        #     self.M_ = M_
        #     num_cage_vertices = num_cage_vertices * 4
            
        #     from utils.exp_utils import create_BN
        #     self._6D_to_rot_ = create_BN
        # else:
        #     raise NotImplementedError('out_type not implemented')
                    
        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros
        
        self.M = M
        self.L = L
        self.NZ = NZ

        # -------------------------------------
        # [added] LBS Hybrid related configurations
        # -----------------------------------
        self.use_lbs_joint_center = self.opts.use_lbs_joint_center
        self.use_exp_joint_predict = self.opts.use_exp_joint_predict
        self.use_lbs = self.opts.use_lbs 
        self.num_lbs_joints = self.opts.num_lbs_joints
        self.use_joint_predict = self.opts.use_joint_predict
        self.use_weighted_joint_pos = self.opts.use_weighted_joint_pos
        self.no_use_translation = self.opts.no_use_translation
        

        # -------------------------------------
        # [added] LBS Hybrid related options
        # -------------------------------------
        self.lbs_weight_model = LinearEncoder(
            in_dim=self.in_dim,
            use_softmax=self.use_softmax,
            use_relu=self.use_relu, # default setting
            use_elu=self.use_elu,
            use_softplus=self.use_softplus,
            use_least_N=self.use_least_N,
            use_least_N_on_V=self.use_least_N_on_V,
            no_activation=self.no_activation,
            out_dim=self.num_lbs_joints,
            use_pou=True, # LBS weight는 반드시 sum=1
        ).to(device)
        
        # -------------------------------------
        # [added] LBS pose / transform predictor (global)
        # 6D rotation + translation (3) = 9 per joint
        # -------------------------------------
        
        # expression encoder for lbs
        self.lbs_exp_z_model = LinearEncoder(
            in_dim=in_dim_exp,
            out_dim=L,
            num_layers=self.num_layers, 
            out_type='global',
        ).to(device)
        
        if self.no_use_translation: 
            self.lbs_pose_model = LinearEncoder(
                in_dim=L,
                out_dim=self.num_lbs_joints*6,
                out_type='global',
            ).to(device)
        else:
            self.lbs_pose_model = LinearEncoder(
                in_dim=L,
                out_dim=self.num_lbs_joints*9, # 6D
                out_type='global',
            ).to(device)

        from utils.exp_utils import from_6D_to_rotation_matrix_torch
        ## 3x3 회전 행렬에서 두 개의 열 벡터(unconstrained 3D vectors)만을 사용하여 회전을 인코딩
            ## 실제 회전 행렬은 이 두 벡터에 그램-슈미트(Gram-Schmidt) 직교화 과정을 적용하여 세 번째 열 벡터 \(c_{3}\)를 계산함으로써 완전한 3x3 회전 행렬을 재구성
        self._6D_to_rot_lbs = from_6D_to_rotation_matrix_torch

    def get_model_config(self):
        text = "===========[model CBD config]===========\n"
        text+= f"[         num lbs joints        ]: {self.num_lbs_joints}\n"
        text+= "===============================+++\n"
        return text
    # -------------------------------------
    # [added] LBS formulation
    # -------------------------------------
    def apply_lbs_no_center(self,source_vert, W_lbs, T_lbs):
        """
        LBS-ish formulation w/o joint position in consideration
        Args:
            source_vert (torch.tensor): [B, N, 3] source mesh vertices
            W_lbs (torch.tensor): [B, N, J] LBS weights
            T_lbs (torch.tensor): [B, J, 3, 4] LBS transforms
        Returns:
            pred_deformed: [B, N, 3] predicted deformed mesh vertices
        """
        B, N, _ = source_vert.shape
        J = W_lbs.shape[-1]
        
        if self.no_use_translation:
            R = T_lbs # (B, J, 3, 3)
            # v = source_vert[:,:,None,:] # (B, N, 1, 3)
            v = torch.einsum('bjik,bnk->bnji', R, source_vert) # rotate, (B, N, J, 3)
        else:
            R = T_lbs[..., :3] # (B, J, 3, 3)
            t = T_lbs[..., 3] # (B, J, 3)
            # v = source_vert[:,:,None,:] # (B, N, 1, 3)
            v = torch.einsum('bjik,bnk->bnji', R, source_vert) # rotate, (B, N, J, 3)
            v = v + t[:,None,:,:] # translate, (B, N, J, 3)
        
        v_lbs = torch.einsum('bnj,bnjc->bnc', W_lbs, v)
        return v_lbs
    
    def apply_lbs(self, source_vert, W_lbs, T_lbs, C_lbs):
        """
        standard LBS formulation only using R in (R|t) as T (t is set as C_lbs)
        Args:
            source_vert (torch.Tensor): [B, N, 3] source mesh vertices
            W_lbs       (torch.Tensor): [B, N, J] LBS weights (ideally sum_j = 1)
            T_lbs       (torch.Tensor): [B, J, 3, 3] LBS transforms (R)
            C_lbs       (torch.Tensor): [B, J, 3] joint centers (t: pivot points)

        Returns:
            v_lbs (torch.Tensor): [B, N, 3] LBS-deformed vertices
        """
        # shapes
        B, N, _ = source_vert.shape
        assert W_lbs.dim() == 3 and W_lbs.shape[0] == B and W_lbs.shape[1] == N
        J = W_lbs.shape[-1]

        # assert T_lbs.shape == (B, J, 3, 4), f"T_lbs must be (B,J,3,4), got {T_lbs.shape}"
        if self.no_use_translation:
            assert T_lbs.shape == (B, J, 3, 3), f"T_lbs must be (B,J,3,3), got {T_lbs.shape}"
            assert C_lbs.shape == (B, J, 3),     f"C_lbs must be (B,J,3), got {C_lbs.shape}"
            R = T_lbs
        else:
            assert T_lbs.shape == (B, J, 3, 4), f"T_lbs must be (B,J,3,3), got {T_lbs.shape}"
            assert C_lbs.shape == (B, J, 3),     f"C_lbs must be (B,J,3), got {C_lbs.shape}"
            # split transform
            R = T_lbs[..., :3]   # (B, J, 3, 3)
            # t = T_lbs[..., 3]    # (B, J, 3)

        ## 즉, 센터를 빼고 회전 → 다시 센터를 더하는 건 좌표계 변환의 필수 절차다.
        # move to joint-local coordinates: v - c_j 
        v_local = source_vert[:, :, None, :] - C_lbs[:, None, :, :]  # (B, N, J, 3)
        # v_local = source_vert[:, :, None, :]# (B, N, J, 3)

        # rotate: R_j @ (v - c_j)
        v_rot = torch.einsum('bjik,bnjk->bnji', R, v_local)          # (B, N, J, 3)

        # move back: + c_j + t_j
        # v_joint = v_rot + C_lbs[:, None, :, :] + t[:, None, :, :]    # (B, N, J, 3)
        v_joint = v_rot + C_lbs[:, None, :, :]     # (B, N, J, 3)
        # v_joint = v_rot + t[:, None, :, :]    # (B, N, J, 3)

        # blend across joints
        v_lbs = torch.einsum('bnj,bnjc->bnc', W_lbs, v_joint)         # (B, N, 3)
        return v_lbs
    
    def joint_position_from_weights(self, verts, W, eps=1e-8):
        """
        verts: (B, N, 3)  source vertices (bar V)
        W:     (B, N, J)  skinning weights (w_{i,k})
        return:
        C_bar: (B, J, 3)
        """
        # numerator: sum_i w_{i,k} * v_i
        num = torch.einsum('bnj,bnc->bjc', W, verts)          # (B,J,3)
        den = W.sum(dim=1, keepdim=False).unsqueeze(-1)      # (B,J,1)
        C_bar = num / (den + eps)
        return C_bar
    
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
    
    def forward(self, 
                source_vert, 
                deform_vert, 
                source_norm,
                deform_norm, 
                mesh_data, 
                hat_mask=None, 
                epoch=0, 
                out_kw=False, # False if baseline
                stage=None,
                ):
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
        device = deform_vert.device
        dtype = deform_vert.dtype
        
        M = self.num_cage_vertices
        J = self.num_lbs_joints
        
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
        
        ## ===============================
        ## [added] LBS branch
        ## ===============================
        W_lbs = torch.zeros((B, N, J), device=device, dtype=dtype) 
        T_lbs = torch.zeros((B, N, 3, 4), device=device, dtype=dtype)
        rigid_v = torch.zeros((B, N, 3), device=device, dtype=dtype)
        delta_v = torch.zeros((B, N, 3), device=device, dtype=dtype)
        key_d = torch.zeros((B,M,3), device=device,dtype=dtype)
        key_weight = torch.zeros((B,N,M), device=device,dtype=dtype)
        
        exp_z_lbs = self.lbs_exp_z_model(deform_in) # (B, 1, L)

        W_lbs = self.lbs_weight_model(source_in) # (B, N, J)
        ## LBS transforms
        if self.no_use_translation:
            T = self.lbs_pose_model(exp_z_lbs).reshape(B, self.num_lbs_joints, 6)  # (B, J, 6)
        else:
            T = self.lbs_pose_model(exp_z_lbs).reshape(B, self.num_lbs_joints, 9)  # (B, J, 9)
            
        if self.no_use_translation:
            R6 = T # (B, J, 6)
        else:
            R6, t = T[..., :6], T[..., 6:] # (B, J, 6), (B, J, 3)
        
        R = self._6D_to_rot_lbs(R6).reshape(B, self.num_lbs_joints, 3, 3) # (B, J, 3, 3)
        
        if self.no_use_translation:
            T_lbs = R # (B, J, 3, 3)
        else:
            T_lbs = torch.cat([R, t[..., None]], dim=-1) # (B, J, 3, 4)
        
        if not self.use_lbs_joint_center:
            rigid_v = self.apply_lbs_no_center(source_vert, W_lbs, T_lbs) # (B, N, 3)
        else:
            if self.use_joint_predict: # using network to predict joint, loss calculation should be needed in the training loop
                if self.use_exp_joint_predict:
                    C_lbs = self.lbs_joint_center_model(exp_z_lbs).reshape(B, self.num_lbs_joints, 3) # (B, J, 3)
                else: # default is this
                    C_lbs = self.lbs_joint_center_model(source_in).reshape(B, self.num_lbs_joints, 3) # (B, J, 3)
                rigid_v = self.apply_lbs(source_vert, W_lbs, T_lbs, C_lbs)
            else:
                # C_bar = self.joint_position_from_weights(source_vert, W_lbs) # (B,J,3)
                # rigid_v = self.apply_lbs(source_vert, W_lbs, T_lbs, C_bar)
                if self.use_weighted_joint_pos:
                    C_bar = self.joint_position_from_weights(source_vert, W_lbs) # (B,J,3)
                else: # just use t as C_bar, and always dimension should be 9 for pose_model
                    C_bar = t 
                rigid_v = self.apply_lbs(source_vert, W_lbs, T_lbs, C_bar)
                
        ## ===============================
        ## [added] final composition
        ## ===============================    
        pred_deformed = rigid_v + delta_v
        
        ## ===============================
        ## [old] final composition
        ## ===============================
        recon_source = 0
        recon_deformed = 0
        pred_source = 0
        
        ## optional
        source_in_s = source_vert
        deform_in_s = source_vert - source_vert # as a delta    
        
        if self.in_type > 0:
            source_in_s = torch.cat([source_in_s, source_norm], dim=-1)
            # why deform norm? 
            deform_in_s = torch.cat([deform_in_s, deform_norm], dim=-1)
        deform_in_s = torch.cat([deform_in_s, source_in_s], dim=-1)
        if self.in_type == 2:
            source_in_s = torch.cat([source_in_s, hat_mask], dim=-1)
            deform_in_s = torch.cat([deform_in_s, hat_mask], dim=-1)
    
        exp_z_s = self.lbs_exp_z_model(deform_in_s) # (B, 1, L)    
        key_s = self.lbs_pose_model(exp_z_s)

        ## LBS weights
        W_lbs = self.lbs_weight_model(source_in_s) # (B, N, J)
        ## LBS transforms
        if self.no_use_translation:
            T = key_s.reshape(B, self.num_lbs_joints, 6)  # (B, J, 6)
            R6 = T # (B, J, 6), (B, J, 3)
            R = self._6D_to_rot_lbs(R6).reshape(B, self.num_lbs_joints, 3, 3) # (B, J, 3, 3)
            T_lbs = R # (B, J, 3, 3)
        else:
            T = key_s.reshape(B, self.num_lbs_joints, 9)  # (B, J, 9)
            R6, t = T[..., :6], T[..., 6:] # (B, J, 6), (B, J, 3)
            R = self._6D_to_rot_lbs(R6).reshape(B, self.num_lbs_joints, 3, 3) # (B, J, 3, 3)
            T_lbs = torch.cat([R, t[..., None]], dim=-1) # (B, J, 3, 4)
            
        if not self.use_lbs_joint_center:
            pred_source = self.apply_lbs_no_center(source_vert, W_lbs, T_lbs) # (B, N, 3)
        else:
            if self.use_joint_predict: # using network to predict joint, loss calculation should be needed in the training loop
                if self.use_exp_joint_predict:
                    C_lbs = self.lbs_joint_center_model(exp_z).reshape(B, self.num_lbs_joints, 3) # (B, J, 3)
                else: # default is this
                    C_lbs = self.lbs_joint_center_model(source_in).reshape(B, self.num_lbs_joints, 3) # (B, J, 3)
                pred_source = self.apply_lbs(source_vert, W_lbs, T_lbs, C_lbs)
            else:
                if self.use_weighted_joint_pos:
                    C_bar = self.joint_position_from_weights(source_vert, W_lbs) # (B,J,3)
                else: # just use t as C_bar
                    C_bar = t 
                pred_source = self.apply_lbs(source_vert, W_lbs, T_lbs, C_bar)
        
        exp_z = exp_z_lbs

        if self.opts.vis_joint_pos:
            if self.use_joint_predict:
                C_lbs = C_lbs
            else: 
                C_lbs = C_bar
            
            return pred_deformed, recon_deformed, recon_source, exp_z, pred_source, hat_mask, key_d, key_weight, W_lbs, T_lbs, C_lbs
        else:
            ## added W_lbs, T_lbs
            return pred_deformed, recon_deformed, recon_source, exp_z, pred_source, hat_mask, key_d, key_weight, W_lbs, T_lbs
            
            
class NeuralGeneralizedBarycentricCoordinateCBD(nn.Module):
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

        # shape encoder
        if self.use_shp:
            self.shape_model = LinearEncoder(
                in_dim=self.in_dim,
                out_dim=L,
                out_type='global'
            ).to(device)

        # cage displacement predictor
        self.key_d_model = LinearEncoder(
            in_dim=L,
            out_dim= M_*self.out_dim if (self.out_type == 2) or (self.out_type == 3) else M*self.out_dim,
            num_layers=self.num_layers, 
            out_type='global'
        ).to(device)

    def get_model_config(self):
        text = "===========[model CBD config]===========\n"
        text+= f"[         num cage v        ]: {self.num_cage_vertices}\n"
        text+= "===============================+++\n"
        return text
    
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
        
    @torch.no_grad()
    def get_coordinate(self, source_vert, source_norm, out_kw=False):
        """
        Args:
            source_vert (torch.tensor): [B, N, 3] source mesh vertice
            source_norm (torch.tensor): [B, N, 3] source mesh vertex normals
            mesh_data (int): indicator for data (0: voca, 1: biwi, 2: multiface)
            epoch (int): train epoch (epoch != iteration)
        Returns:
            coordinates [B, N, C]
        """
        B, N, _ = source_vert.shape
        
        source_in = source_vert
            
        if self.in_type > 0:
            source_in = torch.cat([source_in, source_norm], dim=-1)
        if self.in_type==2:
            hat_mask = plateau_hat_points(source_vert)
            source_in = torch.cat([source_in, hat_mask], dim=-1)
        
        key_weight = self.key_weight_model(
            source_in,
            N=self.NZ # (not used!)
        ) # (B, N, M)
        
        return key_weight
        
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
            tgt_neu_vert (torch.tensor): [B, M, 3] target neutral mesh vertex positions
            tgt_neu_norm (torch.tensor): [B, M, 3] target neutral mesh vertex normals            
        Returns:
            key_weight, cooridnate w.r.t the cage vertices
            
        """
        B, N, _ = tgt_neu_vert.shape
        tgt_in = tgt_neu_vert
                
        if self.in_type > 0:
            tgt_in = torch.cat([tgt_in, tgt_neu_norm], dim=-1)

        if self.in_type == 2:
            tgt_hat_mask = plateau_hat_points(tgt_neu_vert)
            tgt_in = torch.cat([tgt_in, tgt_hat_mask], dim=-1)
        
        key_weight = self.key_weight_model(tgt_in, N=self.NZ) # (B, M, K)
            
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
