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
import pickle


import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataloader_CBD import (
    CBDdataSampler,
    CBDDataset,
    CBD_collate_wrapper,
    EvalDataset,
    CBDDataBatch_eval,
    CBD_collate_wrapper_eval,
    # CBDDataset2,
    # CBD_collate_wrapper2,
)

# from utils.mesh_utils import Renderer #, calc_cent
from utils.matplotlib_rnd import plot_image_array, plot_image_array_seg, vis_rig
from utils.ckpt_utils import *
from utils.exp_utils import plateau_hat_points
from utils.remesh_utils import build_padded_neighbors, pca_normal_axis_vectorized
from utils.remesh_utils import calc_norm_torch
# from utils.exp_utils import Model_mk1, Model_mk3_1
# from utils.remesh_utils import compute_MVC_vertexwise, apply_MVC_weights_batch, build_padded_neighbors, pca_normal_axis_vectorized

from models.baseline import CageNet
from models.NGBC import (
    NeuralGeneralizedBarycentricCoordinate, 
    NeuralGeneralizedBarycentricCoordinate5,
    NeuralGeneralizedBarycentricCoordinate8,
    NeuralGeneralizedBarycentricCoordinate55
)
from utils.remesh_utils import ICT_face_model
import torch.multiprocessing as mp

# sys.path = list(set(sys.path))

def Options():
    parser = argparse.ArgumentParser(description='neural generalized barycentric coordinate for FA retargeting')
    parser.add_argument('-c', '--config', default='config/train_CBD.yml', help='config file path')
    parser.add_argument("--device",       type=str,   default="cuda:0")
    
    parser.add_argument("--log_dir",      type=str,   default="eval_CBD")

    parser.add_argument("--version",      type=int,   default=1,      help='train method (1: baseline, 2: ours)')
    #parser.add_argument("--num_cage_v",   type=int,   default=1024,   help='number of cage vertices')

    parser.add_argument("--data_selection",      type=int,   default=-1,
                        help='select dataset (-1: all, 0: voca, 1:biwi, 2: mf_SEN, 3: coma, 4: mf_ROM, 5: mf all)')
    
#     parser.add_argument("--in_type",      type=int,   default=0,      
#                         help='input type (0: position, 1: position + normal')
#     parser.add_argument("--out_type",      type=int,   default=1,      
#                         help='output type (0: cage v, 1: cage delta_v, 2: cage delta_T mat, 3: vertex T mat')
    #### Choose a last layer activation for key_weight_model()
#     parser.add_argument("--last_activation", choices=["relu", "elu", "softmax", "softplus", "none"],
#         help="Choose a last layer activation for NGBC.key_weight_model()"
#     )
    
#     parser.add_argument("--no_pou",dest='no_pou', action='store_true')
#     parser.set_defaults(no_pou=False)
    
#     parser.add_argument("--lr",           type=float, default=0.0002, help='learning rate')
    
    parser.add_argument("--batch_size",   type=int,   default=1,      help='batch size')

    parser.add_argument("--seed",         type=int,   default=42,     help='random seed')
    parser.add_argument("--ckpt",         type=str,   default=None)
    
    parser.add_argument("--start_epoch",  type=int,   default=0,      help='number of epochs')
    parser.add_argument("--continue_ckpt",dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)
    
#     parser.add_argument("--n_sampling",   dest='n_sampling', action='store_true')
#     parser.set_defaults(n_sampling=False)
    
#     parser.add_argument("--use_decimate", dest='use_decimate', action='store_true')
#     parser.set_defaults(use_decimate=False)
    
#     parser.add_argument("--use_scheduler",dest='use_scheduler', action='store_true')
#     parser.set_defaults(use_scheduler=False)
    
#     parser.add_argument("--use_data2",dest='use_data2', action='store_true')
#     parser.set_defaults(use_data2=False)
#     parser.add_argument("--use_data3",dest='use_data3', action='store_true')
#     parser.set_defaults(use_data3=False)

    parser.add_argument("--use_eval_data2",dest='use_eval_data2', action='store_true')
    parser.set_defaults(use_eval_data2=False)
    
    parser.add_argument("--realtest",dest='realtest', action='store_true')
    parser.set_defaults(realtest=False)

    parser.add_argument("--tb",           action='store_true')
    parser.set_defaults(is_train=True)
    
    parser.add_argument("--use_t_mask",dest='use_t_mask', action='store_true')
    parser.set_defaults(use_t_mask=False)
    
    parser.add_argument("--save_vert",dest='save_vert', action='store_true')
    parser.set_defaults(save_vert=False)
    parser.add_argument("--save_gt",dest='save_gt', action='store_true')
    parser.set_defaults(save_gt=False)
    
    
    parser.add_argument("--use_NFR",dest='use_NFR', action='store_true')
    parser.set_defaults(use_NFR=False)
    
    parser.add_argument("--NFR",dest='NFR', action='store_true')
    parser.set_defaults(NFR=False)
    
    
    parser.add_argument("--optim_cage",dest='optim_cage', action='store_true')
    parser.set_defaults(optim_cage=False)
    
    
    parser.add_argument("--align_latent",dest='align_latent', action='store_true')
    parser.set_defaults(align_latent=False)
    
    args = parser.parse_args()
    return args

class Pipeline():
    def __init__(self, opts):
        # set opts
        self.opts = opts
        self.set_seed(self.opts)
        self.device = opts.device
        
#         if opts.version>=5:
#             last_act_list = ["relu", "elu", "softmax", "softplus", "none"]
#             last_act_list = [self.opts.last_activation==l_act for l_act in last_act_list]
            
#         if opts.version==0:
#             from models.NFS import NFS
#             from utils.nfr_utils import get_dfn_info
#             self.get_dfn_info = get_dfn_info
#             self.model = NFS(self.opts, None, print_param=True).to(self.device)
            
#         elif opts.version==1:
#             self.model = CageNet(
#                 device=self.device,
#                 optim_cage=self.opts.optim_cage
#             )
#         elif opts.version==2:
#             self.model = NeuralGeneralizedBarycentricCoordinate(
#                 self.opts, 
#                 hid_dim=256,
#                 num_cage_vertices=self.opts.num_cage_v,
#                 num_layers=4,
#                 use_relu=True,
#                 is_train=True, 
#                 device=self.device,
#             )
#         elif opts.version==5:
#             self.model = NeuralGeneralizedBarycentricCoordinate5(
#                 opts, num_layers=4,
#                 num_cage_vertices=self.opts.num_cage_v,
#                 use_exp_recon=False, # not used yet
#                 use_shp_recon=False, # not used yet
#                 use_shp=False,
#                 use_relu=last_act_list[0],
#                 use_elu=last_act_list[1],
#                 use_softmax=last_act_list[2],
#                 use_softplus=last_act_list[3],
#                 no_activation=last_act_list[4],
#                 use_least_N_on_V=False,
#                 is_train=True,
#                 use_pou = ~self.opts.no_pou,
#                 device=self.device,
#                 #hid_dim=128 if self.opts.use_data2 or self.opts.use_data3 else 256,
#                 hid_dim=128 if self.opts.align_latent else 256,
#             )
#         elif opts.version==8:
#             self.model = NeuralGeneralizedBarycentricCoordinate8(
#                 opts, num_layers=4,
#                 num_cage_vertices=self.opts.num_cage_v,
#                 use_exp_recon=False, # not used yet
#                 use_shp_recon=False, # not used yet
#                 use_shp=False,
#                 use_relu=last_act_list[0],
#                 use_elu=last_act_list[1],
#                 use_softmax=last_act_list[2],
#                 use_softplus=last_act_list[3],
#                 no_activation=last_act_list[4],
#                 use_least_N_on_V=False,
#                 is_train=True,
#                 use_pou = ~self.opts.no_pou,
#                 device=self.device,
#                 #hid_dim=128 if self.opts.use_data2 or self.opts.use_data3 else 256,
#                 hid_dim=128 if self.opts.align_latent else 256,
#             )
#         elif opts.version==55:
#             self.model = NeuralGeneralizedBarycentricCoordinate55(
#                 opts, num_layers=4,
#                 num_cage_vertices=self.opts.num_cage_v,
#                 use_exp_recon=False, # not used yet
#                 use_shp_recon=False, # not used yet
#                 use_shp=False,
#                 use_relu=last_act_list[0],
#                 use_elu=last_act_list[1],
#                 use_softmax=last_act_list[2],
#                 use_softplus=last_act_list[3],
#                 no_activation=last_act_list[4],
#                 use_least_N_on_V=False,
#                 is_train=True,
#                 use_pou = ~self.opts.no_pou,
#                 device=self.device,
#                 #hid_dim=128 if self.opts.use_data2 or self.opts.use_data3 else 256,
#                 hid_dim=128 if self.opts.align_latent else 256,
#             )
#         else:
#             raise NotImplementedError('No matching model version')
        
#         # load weight
#         self.load_weight()
    
#     def load_weight(self):
#         if self.opts.ckpt:
#             if self.opts.continue_ckpt:
#                 ckpt = glob.glob(os.path.join(self.opts.ckpt, f"*_{self.opts.start_epoch:03d}.pth"))[0]
#             else:
#                 ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
#             print(f"Loading... {ckpt}")
            
#             ckpt_dict = torch.load(ckpt, map_location=self.device)            
#             self.model.load_state_dict(ckpt_dict,strict=False)
#         else:
#             print('no ckpt found, training from scratch!')
        
    def get_mesh(self, selection, dataset, SELECT_MESH):
        if selection=='ict'or selection=='ict-cap':
            ict_face = ICT_face_model()
            id_vecs = torch.load(f'./ict_face_pt/ict_id_vecs_test.pt').numpy()
            id_disps = ict_face.get_id_disp(id_vecs[SELECT_MESH])
            mesh_v = ict_face.neutral_verts.squeeze() + id_disps.squeeze()

            return mesh_v, ict_face.faces, f'm{SELECT_MESH:02d}'
        else:
            if selection=='voca' or selection=='coma':
                with open('/data/sihun/pca/VOCA-COMA/voca_templates.pkl', 'rb') as f:
                    mesh = pickle.load(f)
                    
            elif selection=='biwi':
                biwi_trimesh = trimesh.load(f'./test-mesh/BIWI.ply')
                with open(f'./test-mesh/biwi_templates.pkl', 'rb') as f:
                    mesh = pickle.load(f)
                mesh['face'] = biwi_trimesh.faces
                
            elif selection=='mf_SEN' or selection=='mf_ROM':
                with open('/data/sihun/pca/multiface_align/mf_templates.pkl', 'rb') as f:
                    mesh = pickle.load(f)

            mesh_list = [idname for idname in mesh.keys() if idname!='face']
            mesh_id = mesh_list[SELECT_MESH]
            mesh_v = mesh[mesh_id]

            if selection=='biwi':
                m_align = np.load(f'./utils/biwi/align.npy')
                mesh_v = np.concatenate((mesh_v,np.ones((mesh_v.shape[0],1))), axis=1) @ m_align.T
            return mesh_v, mesh['face'], mesh_id
        
    def save_test_frames(self):
        src_tgt_set_list = [
            [4, 12, 4, 12, 0],
            [4, 12, 4, 0, 0],
            [4, 12, 6, 0, 0],
            [6, 0, 6, 0, 1],
            [6, 0, 6, 3, 1],
            [6, 0, 6, 9, 1],
            [6, 2, 6, 2, 0],
            [6, 2, 6, 0, 0],
            [6, 2, 6, 5, 0],
            [6, 0, 4, 12, 1],
            [6, 2, 4, 12, 0],
        ]
        for src_tgt_set in src_tgt_set_list:
            print('selection: ',*src_tgt_set)
            self.save_vis_data(*src_tgt_set)
        
    def save_vis_data(self, SRC_SELECT_DATA, SRC_SELECT_MESH, TGT_SELECT_DATA, TGT_SELECT_mesh, EXP_NUM):
        """
            self-retargeting task
        """
        if self.opts.version==0:
            import utils.nfr_utils as nfr_utils
            from easydict import EasyDict
            from utils.mesh_utils import get_mesh_operators
            
        self.mf_precompute_path = os.path.join('/data/sihun/multiface_align', 'precomputes')
        self.ict_precompute_path = '/data/sihun/ICT-audio2face/precompute-real-fullhead'
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        BS = 1
        device=self.device
        
        ##########################################################################################################
        
        
        data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict','ict-cap'] # 0 1 2 3 4 5 6
        
        src_selection = data_name_list[SRC_SELECT_DATA]
        
        src_dataset = EvalDataset(
            data_name=src_selection, 
            toggle=False, 
            ict_cap_id_num  = SRC_SELECT_MESH,
            ict_cap_exp_num = EXP_NUM,
        ) # if eve-s01
        # self.dataset = EvalDataset(data_name=selection, toggle=True) # if char-s02
        
        src_dataloader = torch.utils.data.DataLoader(
            src_dataset,
            batch_size=BS,
            collate_fn=partial(CBD_collate_wrapper_eval, device=self.device),
            #num_workers=8,
        )
        src_v, src_f, src_mesh_id = self.get_mesh(src_selection, src_dataset, SRC_SELECT_MESH)
        # if 'mf' in src_selection:
        #     src_dfn_info  = pickle.load(open(os.path.join(
        #         self.mf_precompute_path, f"{src_mesh_id}_dfn_info.pkl"
        #     ), 'rb'))
            
        #     # tmp=EasyDict({'vertices':src_v.squeeze(), 'faces':src_f.squeeze()})
        #     # src_operators = get_mesh_operators(tmp)
        #     src_operators = pickle.load(open(os.path.join(
        #         self.mf_precompute_path, f"{src_mesh_id}_operators.pkl"
        #     ), mode='rb'))
        #     src_img = np.load(os.path.join(self.mf_precompute_path, f"{src_mesh_id}_img.npy"))
        #     src_img = torch.from_numpy(src_img)[0]
            
        # if 'ict'in src_selection:
        #     src_dfn_info = pickle.load(open(os.path.join(
        #         self.ict_precompute_path, f"{src_mesh_id}_dfn_info.pkl"
        #     ), 'rb'))
        #     src_operators = pickle.load(open(os.path.join(
        #         self.ict_precompute_path, f"{src_mesh_id}_operators.pkl"
        #     ), mode='rb'))
        #     src_img = np.load(os.path.join(self.ict_precompute_path, f"{src_mesh_id}_img.npy"))
        #     src_img = torch.from_numpy(src_img)[0]
            
        src_n = igl.per_vertex_normals(src_v, src_f)
        
        ##########################################################################################################
        tgt_selection = data_name_list[TGT_SELECT_DATA]

        tgt_dataset = EvalDataset(
            data_name=tgt_selection,
            toggle=False,
            ict_cap_id_num=TGT_SELECT_mesh
        ) # if eve-s01

        tgt_v, tgt_f, tgt_mesh_id = self.get_mesh(tgt_selection, tgt_dataset, TGT_SELECT_mesh)
        # if 'mf' in tgt_selection:
        #     tgt_dfn_info = pickle.load(open(os.path.join(
        #         self.mf_precompute_path, f"{tgt_mesh_id}_dfn_info.pkl"
        #     ), 'rb'))
        #     # tmp=EasyDict({'vertices':tgt_v.squeeze(), 'faces':tgt_f.squeeze()})
        #     # tgt_operators = get_mesh_operators(tmp)
        #     tgt_operators = pickle.load(open(os.path.join(
        #         self.mf_precompute_path, f"{tgt_mesh_id}_operators.pkl"
        #     ), mode='rb'))
        #     tgt_img = np.load(os.path.join(self.mf_precompute_path, f"{tgt_mesh_id}_img.npy"))
        #     tgt_img = torch.from_numpy(tgt_img)[0]
            
        # if 'ict'in tgt_selection:
        #     tgt_dfn_info = pickle.load(open(os.path.join(
        #         self.ict_precompute_path, f"{tgt_mesh_id}_dfn_info.pkl"
        #     ), 'rb'))
        #     # tmp=EasyDict({'vertices':tgt_v.squeeze(), 'faces':tgt_f.squeeze()})
        #     # tgt_operators = get_mesh_operators(tmp)
        #     tgt_operators = pickle.load(open(os.path.join(
        #         self.ict_precompute_path, f"{tgt_mesh_id}_operators.pkl"
        #     ), mode='rb'))
        #     tgt_img = np.load(os.path.join(self.ict_precompute_path, f"{tgt_mesh_id}_img.npy"))
        #     tgt_img = torch.from_numpy(tgt_img)[0]
            
        tgt_n = igl.per_vertex_normals(tgt_v, tgt_f)
        tgt_v_th = torch.tensor(tgt_v).float()[None].to(device)
        tgt_n_th = torch.tensor(tgt_n).float()[None].to(device)
        ##########################################################################################################
        
        
        ###### Logging ###########################################################################################
        # make logdir --------------------------------------------------------------------------------------------
        
        
        ckpt_path = self.opts.ckpt.split('/')[-1]
        if src_selection == 'ict-cap':
            file_name = f'{src_selection}-ID_{SRC_SELECT_MESH:03d}_test-to-{tgt_selection}-ID_{TGT_SELECT_mesh:03d}_test_{EXP_NUM:02d}'
        else:
            if 'ict' in  tgt_selection:
                file_name = f'{src_selection}_test-to-{tgt_selection}_test-ID_{TGT_SELECT_mesh:03d}'
            else:
                if 'mf' in tgt_selection:
                    if TGT_SELECT_mesh==12:
                        file_name = f'{src_selection}_test-to-{tgt_selection}_test'
                    elif TGT_SELECT_mesh==11:
                        file_name = f'{src_selection}_test-to-{tgt_selection}_val'
                    else:
                        file_name = f'{src_selection}_test-to-{tgt_selection}_train'
                else:
                    file_name = f'{src_selection}_test-to-{tgt_selection}_test'
                # file_name = f'{src_selection}_test-to-{tgt_selection}_test'
                # file_name = f'{src_selection}_test-to-{tgt_selection}_val'
                # file_name = f'{src_selection}_test-to-{tgt_selection}_train'
                
        self.opts.log_dir = './vis_CBD/' + opts.ckpt.split('/')[-1]+'/'+file_name
        if self.opts.use_t_mask:
            self.opts.log_dir = self.opts.log_dir + '-masked'
        os.makedirs(self.opts.log_dir, exist_ok=True)
        
        self.opts.log_dir_vert = self.opts.log_dir + '/verts'
        os.makedirs(self.opts.log_dir_vert, exist_ok=True)        
        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        
        SELF_RETARGET = SRC_SELECT_MESH==TGT_SELECT_mesh
        if SELF_RETARGET and self.opts.save_gt:
            GT_file_name = f'GT-{src_selection}_test-to-{tgt_selection}_test_{EXP_NUM:02d}'
            GT_log_dir = './vis_CBD/'+ GT_file_name
            os.makedirs(GT_log_dir, exist_ok=True)
        
        # self logger
        self.logger = open(os.path.join(self.opts.log_dir, "log.txt"), 'w')
        print(f'Saving log at: {self.opts.log_dir}')
        
        print(src_dataset.get_data_config())
        self.logger.write(src_dataset.get_data_config())
        #---------------------------------------------------------------------------------------------------------
        ##########################################################################################################
        
        
        
        # eval loop ##############################################################################################        
        global_step = 0
        BEST_LOSS = 100_000_000
        
        check_usage = False
        
        len_data = len(src_dataset)
        len_dataloader = len(src_dataloader)
        denom = 1 / len_data
        
        
        if os.path.exists(self.opts.log_dir_vert):
            if len(glob.glob(self.opts.log_dir_vert+'/*')) >= len_data:
                print('frames already exists!')
                return
        
        def stack_mse(batch, pred_vertices, losses_val, denom):
            if self.opts.use_t_mask:
                inner_mask = plateau_hat_points(batch.template)
                outter_mask = 1 - inner_mask

                losses_val['MSE-in'] += F.mse_loss(
                    batch.vertices*inner_mask, pred_vertices*inner_mask
                ).item() * denom # for NGBC model

                losses_val['MSE-out'] += F.mse_loss(
                    batch.template*outter_mask, pred_vertices*outter_mask
                ).item() * denom # for NGBC model

            losses_val['MSE'] += F.mse_loss(
                batch.vertices,  pred_vertices
            ).item() * denom # for NGBC model
            return losses_val
        
        if self.opts.NFR:
            self.model.model.eval()
        else:
            self.model.eval()
        
        losses_val = {
            "MSE": 0.0
        }
        
        if self.opts.use_t_mask:
            losses_val["MSE-in"] = 0.0
            losses_val["MSE-out"] = 0.0
            
        #mesh_data = src_dataset.data_name
        SELF_RETARGET = SRC_SELECT_MESH==TGT_SELECT_mesh
        
        if self.opts.version==0:
            if opts.NFR==False:
                if SELF_RETARGET:
                    print('self-retargeting! (src == tgt)')

                    pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
                    for index, batch in pbar:
                        if index==0:
                            src_verts = batch.template[0].cpu().numpy()
                            src_faces = batch.faces[0].cpu().numpy()
                            src_m = trimesh.Trimesh(vertices=src_verts, faces=src_faces)
                            # src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
                            # src_operators = get_mesh_operators(src_m)
                            # src_img = self.model.renderer.render_img(src_m).float().to(device)
                            src_img_feat = self.model.get_img_feat(src_img.float().to(device))
                            src_vert_feat = self.model.get_local_feature(
                                batch.template[0][None], batch.faces[0], src_img_feat, at='verts'
                            ).float()
                            src_tri_feat = self.model.get_local_feature(
                                batch.template[0][None], batch.faces[0], src_img_feat, at='faces'
                            ).float()

                            pred_seg_coeff = self.model.encode_seg(src_vert_feat, src_dfn_info)# [1, V, Seg]
                            pred_id_coeff  = self.model.encode_id(src_vert_feat, src_dfn_info)
                        else:
                            if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:
                                pbar.set_description('src chng!?')

                                src_verts = batch.template[0].cpu().numpy()
                                src_faces = batch.faces[0].cpu().numpy()
                                src_m = trimesh.Trimesh(vertices=src_verts, faces=src_faces)
                                # src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
                                # src_operators = get_mesh_operators(src_m)
                                # src_img = self.model.renderer.render_img(src_m).float().to(device)
                                src_img_feat = self.model.get_img_feat(src_img)
                                src_vert_feat = self.model.get_local_feature(
                                    batch.template[0][None], batch.faces[0], src_img_feat, at='verts'
                                ).float()
                                src_tri_feat = self.model.get_local_feature(
                                    batch.template[0][None], batch.faces[0], src_img_feat, at='faces'
                                ).float()

                                pred_seg_coeff = self.model.encode_seg(src_vert_feat, src_dfn_info)# [1, V, Seg]
                                pred_id_coeff  = self.model.encode_id(src_vert_feat, src_dfn_info)

                        with torch.no_grad():
                            vert_feat_exp = []
                            for b_v in batch.vertices:
                                _tmp_ = self.model.get_local_feature(b_v[None], batch.faces[0], src_img_feat).float()
                                vert_feat_exp.append(_tmp_)
                            vert_feat_exp = torch.vstack(vert_feat_exp) * 1.3
                            pred_exp_coeff = self.model.encode_exp(
                                vert_feat_exp, src_dfn_info, batch_process=True, verbose=False
                            )

                            inputs = (
                                src_tri_feat if self.opts.dec_type=='jacob' else src_vert_feat,
                                pred_exp_coeff, pred_id_coeff, pred_seg_coeff,
                                None, batch.template[0][None], batch.faces[0], src_operators
                            )
                            pred_outputs, _ = self.model.decode(inputs, batch_process=True)

                            losses_val = stack_mse(batch, pred_outputs, losses_val, denom)

                            pred_outputs_np = pred_outputs.detach().cpu().numpy()

                            CurrBS=pred_outputs_np.shape[0]
                            for b_idx in range(CurrBS):
                                save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
                                np.save(save_out_name, pred_outputs_np[b_idx])

                                if self.opts.save_gt:
                                    save_gt_name = GT_log_dir+f'/{index*CurrBS+b_idx:06d}.npy'
                                    np.save(save_gt_name, batch.vertices[b_idx].cpu().numpy())
                else:
                    print('cross-retargeting! (src != tgt)')
                    tgt_m = trimesh.Trimesh(vertices=tgt_v, faces=tgt_f)
                    tgt_verts = tgt_v_th[0]
                    tgt_faces = torch.from_numpy(tgt_m.faces).to(device)
                    tgt_dfn_info = nfr_utils.get_dfn_info(tgt_m, map_location=device)
                    tgt_operators = get_mesh_operators(tgt_m)
                    tgt_img = self.model.renderer.render_img(tgt_m).float().to(device)
                    tgt_img_feat = self.model.get_img_feat(tgt_img.float().to(device))
                    tgt_vert_feat = self.model.get_local_feature(tgt_verts[None], tgt_faces, tgt_img_feat).float()
                    tgt_tri_feat = self.model.get_local_feature(
                            tgt_verts[None], tgt_faces, tgt_img_feat, at='faces'
                        ).float()

                    with torch.no_grad():
                        pred_seg_coeff = self.model.encode_seg(tgt_vert_feat, tgt_dfn_info)# [1, V, Seg]
                        pred_id_coeff  = self.model.encode_id(tgt_vert_feat, tgt_dfn_info)

                    pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
                    for index, batch in pbar:
                        if index==0:
                            src_m = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())
                            src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
                            src_img = self.model.renderer.render_img(src_m).float().to(device)
                            src_img_feat = self.model.get_img_feat(src_img)
                        else:
                            if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:
                                pbar.set_description('src chng!?')
                                src_m = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())
                                src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
                                src_img = self.model.renderer.render_img(src_m).float().to(device)
                                src_img_feat = self.model.get_img_feat(src_img)

                        with torch.no_grad():
                            vert_feat_exp = []
                            for b_v in batch.vertices:
                                _tmp_ = self.model.get_local_feature(b_v[None], batch.faces[0], src_img_feat).float()
                                vert_feat_exp.append(_tmp_)
                            vert_feat_exp = torch.vstack(vert_feat_exp)
                            pred_exp_coeff = self.model.encode_exp(
                                vert_feat_exp, src_dfn_info, batch_process=True, verbose=False
                            )# [W, Rig]

                            inputs = (
                                tgt_tri_feat if self.opts.dec_type=='jacob' else tgt_vert_feat,
                                pred_exp_coeff, pred_id_coeff, pred_seg_coeff,
                                None, tgt_verts[None], tgt_faces, tgt_operators
                            )
                            pred_outputs, _ = self.model.decode(inputs, batch_process=True)

                            pred_outputs_np = pred_outputs.detach().cpu().numpy()

                            CurrBS=pred_outputs_np.shape[0]
                            for b_idx in range(CurrBS):
                                save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
                                np.save(save_out_name, pred_outputs_np[b_idx])
            else:
                ## from NFR checkpoint
                if SELF_RETARGET:
                    print('self-retargeting! (src == tgt)')

                    pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
                    for index, batch in pbar:
                        if index==0:
                            src_verts = batch.template[0]
                            src_faces = batch.faces[0]
                            src_m = trimesh.Trimesh(
                                vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                            )

                            src_img = self.model.renderer.render_img(src_m).float().to(device)
                            src_img_feat = self.model.get_img_feat(src_img)[None]
                            src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
                            src_operators = self.model.get_mesh_operators(src_m)
                        else:
                            if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:            
                                src_verts = batch.template[0]
                                src_faces = batch.faces[0]
                                src_m = trimesh.Trimesh(
                                    vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                                )

                                src_img = self.model.renderer.render_img(src_m).float().to(device)
                                src_img_feat = self.model.get_img_feat(src_img)[None]
                                src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
                                src_operators = self.model.get_mesh_operators(src_m)

                        with torch.no_grad():
                            inputs_v = self.model.get_inputs(batch.vertices, batch.faces[0])# [B, V, 3+3]

                            ## get expression
                            self.model.model.update_precomputes(src_dfn_info)
                            pred_exp = self.model.model.encode(inputs_v, src_img.to(device), N_F=src_m.faces.shape[0])
                            pred_outputs, _, _ = self.model.calc_new_mesh(
                                src_verts, src_faces, pred_exp, src_operators, src_dfn_info, src_img
                            )
                            pred_outputs_np = pred_outputs.detach().cpu().numpy()

                            CurrBS=pred_outputs_np.shape[0]
                            for b_idx in range(CurrBS):
                                save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
                                np.save(save_out_name, pred_outputs_np[b_idx])
                else:
                    print('cross-retargeting!')
                    tgt_m = trimesh.Trimesh(vertices=tgt_v, faces=tgt_f)
                    tgt_dfn_info = nfr_utils.get_dfn_info(tgt_m, map_location=device)
                    tgt_verts = tgt_v_th[0]
                    tgt_faces = torch.from_numpy(tgt_m.faces).to(device)
                    tgt_img = trainer.model.renderer.render_img(tgt_m).float().to(device)
                    tgt_operators = trainer.model.get_mesh_operators(tgt_m)
                    
                    pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
                    for index, batch in pbar:
                        if index==0:
                            src_m = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())

                            src_img = trainer.model.renderer.render_img(src_m).float().to(device)
                            src_img_feat = trainer.model.get_img_feat(src_img)[None]
                            src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
                        else:
                            if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:
                                src_m = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())

                                src_img = trainer.model.renderer.render_img(src_m).float().to(device)
                                src_img_feat = trainer.model.get_img_feat(src_img)[None]
                                src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)

                        with torch.no_grad():
                            inputs_v = trainer.model.get_inputs(batch.vertices, batch.faces[0])# [B, V, 3+3]

                            ## get expression
                            trainer.model.model.update_precomputes(src_dfn_info)
                            pred_exp = trainer.model.model.encode(inputs_v, src_img.to(device), N_F=src_m.faces.shape[0])
                            pred_outputs, _, _ = trainer.model.calc_new_mesh(
                                tgt_verts, tgt_faces, pred_exp, tgt_operators, tgt_dfn_info, tgt_img
                            )
                            pred_outputs_np = pred_outputs.detach().cpu().numpy()

                            CurrBS=pred_outputs_np.shape[0]
                            for b_idx in range(CurrBS):
                                save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
                                np.save(save_out_name, pred_outputs_np[b_idx])
        elif self.opts.version == 1:
            if SELF_RETARGET:
                print('self-retargeting! (src == tgt)')
                pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
                for index, batch in pbar:
                    pred_outputs, _ = self.model.retarget(
                        batch.template, batch.vertices, batch.template
                    )
                    losses_val = stack_mse(batch, pred_outputs, losses_val, denom)

                    pred_outputs_np = pred_outputs.detach().cpu().numpy()

                    CurrBS=pred_outputs_np.shape[0]
                    for b_idx in range(CurrBS):
                        save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
                        np.save(save_out_name, pred_outputs_np[b_idx])

                        if self.opts.save_gt:
                            save_gt_name = GT_log_dir+f'/{index*CurrBS+b_idx:06d}.npy'
                            np.save(save_gt_name, batch.vertices[b_idx].cpu().numpy())
            else:
                print('cross-retargeting! (src != tgt)')
                pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
                for index, batch in pbar:
                    pred_outputs, _ = self.model.retarget(
                        batch.template, batch.vertices, tgt_v_th
                    )
                    
                    pred_outputs_np = pred_outputs.detach().cpu().numpy()

                    CurrBS=pred_outputs_np.shape[0]
                    for b_idx in range(CurrBS):
                        save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
                        np.save(save_out_name, pred_outputs_np[b_idx])

                        if self.opts.save_gt:
                            save_gt_name = GT_log_dir+f'/{index*CurrBS+b_idx:06d}.npy'
                            np.save(save_gt_name, batch.vertices[b_idx].cpu().numpy())
                        
        elif self.opts.version==5 or self.opts.version==8 or self.opts.version==55:            
            if SELF_RETARGET:
                print('self-retargeting! (src == tgt)')
                pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
                for index, batch in pbar:
                    
                    if index==0:
                        src_template = batch.template
                        src_template_normal = batch.template_normal
                        with torch.no_grad():
                            key_weight = self.model.predict_coordinate(src_template, src_template_normal)
                    else:
                        if (batch.template[0] - src_template[0]).mean() != 0:
                            src_template = batch.template
                            src_template_normal = batch.template_normal
                            with torch.no_grad():
                                key_weight = self.model.predict_coordinate(src_template, src_template_normal)

                    with torch.no_grad():
                        pred_outputs, key_d = self.model.retarget_animation(
                            src_template, src_template_normal,
                            batch.vertices, batch.vertices_normal, 
                            key_weight,
                            src_template
                        )
                    
                    losses_val = stack_mse(batch, pred_outputs, losses_val, denom)
                    
                    pred_outputs_np = pred_outputs.detach().cpu().numpy()
                    
                    CurrBS=pred_outputs_np.shape[0]
                    for b_idx in range(CurrBS):
                        save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
                        np.save(save_out_name, pred_outputs_np[b_idx])
                        
                        if self.opts.save_gt:
                            save_gt_name = GT_log_dir+f'/{index*CurrBS+b_idx:06d}.npy'
                            np.save(save_gt_name, batch.vertices[b_idx].cpu().numpy())
            else:
                print('cross-retargeting! (src != tgt)')
                key_weight = self.model.predict_coordinate(tgt_v_th, tgt_n_th)

                pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
                for index, batch in pbar:
                    with torch.no_grad():            
                        pred_outputs, key_d = self.model.retarget_animation(
                            batch.template, batch.template_normal, 
                            batch.vertices, batch.vertices_normal, 
                            key_weight,
                            tgt_v_th # -> (optional) only needed for diplacement prediction
                        )
                    
                    pred_outputs_np = pred_outputs.detach().cpu().numpy()
                    
                    CurrBS=pred_outputs_np.shape[0]
                    for b_idx in range(CurrBS):
                        save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
                        np.save(save_out_name, pred_outputs_np[b_idx])
                        
        else:
            pass
                
            
        ##########################################################################################################
        
        # write log
        if SELF_RETARGET:
            log_text = f"[Eval] "
            for key, value in losses_val.items():
                txt = f"{key}: {value:.6e} "
                print(txt)
                log_text += txt
            self.logger.write(log_text+"\n")
        print('done!')
    
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


if __name__ == "__main__":
    """
    #### voca | biwi | mf_SEN | coma | mf_ROM | mf_all |
    ####   0  |   1  |    2   |   3  |    4   |    5   |
    
        python vis_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-10-22-16-26-26-NGBCv5
        
        ## NFS
        python vis_CBD.py --version 0 --ckpt ./ckpt_stage1/2024-07-08-06-27-12-all #
        python vis_CBD.py --version 0 --ckpt ./ckpt_stage1/2024-08-18-23-32-29-all
        
        ## NFR
        python vis_CBD.py --version 0 --ckpt ./ckpt_stage1/exp_019_ICT_MF-jacob_NFR --NFR
        
    """
    mp.set_start_method('spawn', force=True)
    
    # argparse configs
    opts = Options()
    
    # load training configs from checkpoints (yaml)
    if opts.version==0:
        opts.config='config/train.yml'
        opts_yaml = yaml.load(open(opts.config), Loader=yaml.FullLoader)
    else:
        config = f'{opts.ckpt}/train_opts.yml'
        opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)
    
    # update training configs with argparse configs
    opts_ = vars(opts)
    opts_yaml.update(opts_)
    opts = argparse.Namespace(**opts_yaml)
    opts.use_t_mask = True
    opts.continue_ckpt=False
    
    ## helper for loading NFR ans NFS
    if opts.NFR:
        opts.ckpt='/NFR'
        opts.img_feat_dim=128
        opts.feature_type="cents&norms"
        opts.stage1 = True
        opts.scale_exp=1.0
        opts.ict_face_only=False
        opts.design="nfr"
        opts.dec_type="jacob"
    if opts.version==0 and opts.NFR==False:
        opts.img_feat_dim=128        
        opts.feature_type="cents&norms"
        opts.stage1 = True
        opts.scale_exp=1.0
        opts.ict_face_only=False
        
        if opts.use_NFR:
            opts.design="nfr"
            opts.dec_type="jacob"
        else:
            opts.design="new2"
            opts.dec_type="disp"
    print('loaded version:', opts.version)
    
    ## load model
    if opts.version==0:
        from evaluation import Trainer
    else:
        from eval_CBD import Trainer
    trainer = Trainer(opts)
        
    pipeline = Pipeline(opts)
    pipeline.model = trainer.model
    pipeline.save_test_frames() ## pca test data

