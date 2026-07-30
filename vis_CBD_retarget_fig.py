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
    CBD_collate_wrapper,
    EvalDataset,
    CBD_collate_wrapper_eval,
)

# from utils.mesh_utils import Renderer #, calc_cent
from utils.matplotlib_rnd import plot_image_array, plot_image_array_seg, vis_rig
from utils.ckpt_utils import *
from utils.exp_utils import plateau_hat_points
from utils.remesh_utils import build_padded_neighbors, pca_normal_axis_vectorized
from utils.remesh_utils import calc_norm_torch
# from utils.exp_utils import Model_mk1, Model_mk3_1
# from utils.remesh_utils import compute_MVC_vertexwise, apply_MVC_weights_batch, build_padded_neighbors, pca_normal_axis_vectorized

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
        
    def get_mesh(self, selection, dataset, SELECT_MESH):
        if selection=='ict' or selection=='ict-cap':
            ict_face = ICT_face_model()
            id_vecs = torch.load(f'./ict_face_pt/ict_id_vecs_test.pt').numpy()
            id_disps = ict_face.get_id_disp(id_vecs[SELECT_MESH])
            mesh_v = ict_face.neutral_verts.squeeze() + id_disps.squeeze()

            return mesh_v, ict_face.faces, f'm{SELECT_MESH:02d}'
        elif selection=='ict_face_only':
            ict_face = ICT_face_model(face_only=True)
            id_vecs = torch.load(f'./ict_face_pt/ict_id_vecs_test.pt').numpy()
            id_disps = ict_face.get_id_disp(id_vecs[SELECT_MESH], region=1)
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
#             [4, 12, 6, 1, 0], # mf_ROM -> ict-cap
#             [4, 12, 7, 1, 0], # mf_ROM -> ict_face_only
#             [4, 12, 3, 6, 0], # mf_ROM -> coma
            [4, 12, 1, 12, 0], # mf_ROM -> biwi
#             [4, 12, 4, 12, 0],
#             [4, 12, 4, 0, 0],
#             [4, 12, 6, 0, 0],
#             [6, 0, 6, 0, 1],
#             [6, 0, 6, 3, 1],
#             [6, 0, 6, 9, 1],
#             [6, 0, 3, 6, 1],
#             [6, 0, 3, 8, 1],
#             [6, 0, 2, 12, 1],
#             [6, 2, 6, 2, 0],
#             [6, 2, 6, 0, 0],
#             [6, 2, 6, 5, 0],
#             [6, 0, 4, 12, 1],
#             [6, 2, 4, 12, 0],
        ]
        for src_tgt_set in src_tgt_set_list:
            print('selection: ',*src_tgt_set)
            self.save_vis_data(*src_tgt_set)
        
    def save_vis_data(self, SRC_SELECT_DATA, SRC_SELECT_MESH, TGT_SELECT_DATA, TGT_SELECT_mesh, EXP_NUM,
                       src_frame_slice=None, out_name_override=None):
        """
            self-retargeting task

            src_frame_slice: optional (start, end) to restrict the source dataset's
                frame list to a contiguous sub-range (e.g. a single mf_ROM motion
                sequence instead of the full concatenated test set).
            out_name_override: optional explicit output folder name (under
                ./vis_CBD/<ckpt>/) instead of the auto-derived file_name.
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


        data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict','ict-cap','ict_face_only'] # 0 1 2 3 4 5 6 7

        src_selection = data_name_list[SRC_SELECT_DATA]

        src_dataset = EvalDataset(
            data_name=src_selection,
            toggle=False,
            ict_cap_id_num  = SRC_SELECT_MESH,
            ict_cap_exp_num = EXP_NUM,
        ) # if eve-s01
        # self.dataset = EvalDataset(data_name=selection, toggle=True) # if char-s02

        if src_frame_slice is not None:
            if src_selection != 'mf_ROM':
                raise NotImplementedError('src_frame_slice is only supported for mf_ROM sources')
            s, e = src_frame_slice
            src_dataset.mf_ROM_datalist = src_dataset.mf_ROM_datalist[s:e]
            src_dataset.len = len(src_dataset.mf_ROM_datalist)

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
        if out_name_override is not None:
            file_name = out_name_override
        elif src_selection == 'ict-cap':
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
        
        SELF_RETARGET = (SRC_SELECT_DATA == TGT_SELECT_DATA) and (SRC_SELECT_MESH == TGT_SELECT_mesh)
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
        if self.opts.version==0:
            if opts.NFR==False:
                if SELF_RETARGET:
                    print('self-retargeting! (src == tgt)')

                    pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
                    _prev_src_template = None
                    for index, batch in pbar:
                        if index==0:
                            _prev_src_template = batch.template[0]
                            src_verts = batch.template[0].cpu().numpy()
                            src_faces = batch.faces[0].cpu().numpy()
                            src_m = trimesh.Trimesh(vertices=src_verts, faces=src_faces)
                            src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
                            src_operators = get_mesh_operators(src_m)
                            src_img = self.model.renderer.render_img(src_m).float().to(device)
                            src_img_feat = self.model.get_img_feat(src_img)
                            src_vert_feat = self.model.get_local_feature(
                                batch.template[0][None], batch.faces[0], src_img_feat, at='verts'
                            ).float()
                            src_tri_feat = self.model.get_local_feature(
                                batch.template[0][None], batch.faces[0], src_img_feat, at='faces'
                            ).float()

                            pred_seg_coeff = self.model.encode_seg(src_vert_feat, src_dfn_info)# [1, V, Seg]
                            pred_id_coeff  = self.model.encode_id(src_vert_feat, src_dfn_info)
                        else:
                            if not torch.equal(batch.template[0], _prev_src_template):
                                _prev_src_template = batch.template[0]
                                pbar.set_description('src chng!?')

                                src_verts = batch.template[0].cpu().numpy()
                                src_faces = batch.faces[0].cpu().numpy()
                                src_m = trimesh.Trimesh(vertices=src_verts, faces=src_faces)
                                src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
                                src_operators = get_mesh_operators(src_m)
                                src_img = self.model.renderer.render_img(src_m).float().to(device)
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
                            vert_feat_exp = self.model.get_local_feature(batch.vertices, batch.faces[0], src_img_feat).float() * 1.3
                            pred_exp_coeff = self.model.encode_exp(
                                vert_feat_exp, src_dfn_info, batch_process=True, verbose=False
                            )

                            inputs = (
                                src_tri_feat if self.opts.dec_type=='jacob' else src_vert_feat,
                                pred_exp_coeff, pred_id_coeff, pred_seg_coeff,
                                None, batch.template[0][None], batch.faces[0], src_operators
                            )
                            decode_out = self.model.decode(inputs, batch_process=True)
                            pred_outputs = decode_out[0] if isinstance(decode_out, tuple) else decode_out

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
                    _prev_src_template = None
                    for index, batch in pbar:
                        if index==0:
                            _prev_src_template = batch.template[0]
                            src_m = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())
                            src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
                            src_img = self.model.renderer.render_img(src_m).float().to(device)
                            src_img_feat = self.model.get_img_feat(src_img)
                        else:
                            if not torch.equal(batch.template[0], _prev_src_template):
                                _prev_src_template = batch.template[0]
                                pbar.set_description('src chng!?')
                                src_m = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())
                                src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
                                src_img = self.model.renderer.render_img(src_m).float().to(device)
                                src_img_feat = self.model.get_img_feat(src_img)

                        with torch.no_grad():
                            vert_feat_exp = self.model.get_local_feature(batch.vertices, batch.faces[0], src_img_feat).float() * 1.3
                            pred_exp_coeff = self.model.encode_exp(
                                vert_feat_exp, src_dfn_info, batch_process=True, verbose=False
                            )# [W, Rig]

                            inputs = (
                                tgt_tri_feat if self.opts.dec_type=='jacob' else tgt_vert_feat,
                                pred_exp_coeff, pred_id_coeff, pred_seg_coeff,
                                None, tgt_verts[None], tgt_faces, tgt_operators
                            )
                            decode_out = self.model.decode(inputs, batch_process=True)
                            pred_outputs = decode_out[0] if isinstance(decode_out, tuple) else decode_out
                            
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
                    _prev_src_template = None
                    for index, batch in pbar:
                        if index==0:
                            _prev_src_template = batch.template[0]
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
                            if not torch.equal(batch.template[0], _prev_src_template):
                                _prev_src_template = batch.template[0]
                                pbar.set_description('src chng!?')
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
                    tgt_img = self.model.renderer.render_img(tgt_m).float().to(device)
                    tgt_operators = self.model.get_mesh_operators(tgt_m)

                    pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
                    _prev_src_template = None
                    for index, batch in pbar:
                        if index==0:
                            _prev_src_template = batch.template[0]
                            src_m = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())

                            src_img = self.model.renderer.render_img(src_m).float().to(device)
                            src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
                        else:
                            if not torch.equal(batch.template[0], _prev_src_template):
                                _prev_src_template = batch.template[0]
                                pbar.set_description('src chng!?')
                                src_m = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())
                                src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
                                src_img = self.model.renderer.render_img(src_m).float().to(device)

                        with torch.no_grad():
                            inputs_v = self.model.get_inputs(batch.vertices, batch.faces[0])# [B, V, 3+3]

                            ## encode (source identity's dfn_info active)
                            self.model.model.update_precomputes(src_dfn_info)
                            pred_exp = self.model.model.encode(inputs_v, src_img.to(device), N_F=src_m.faces.shape[0])

                            ## apply (switch stateful precomputes to target identity)
                            self.model.model.update_precomputes(tgt_dfn_info)
                            pred_outputs, _, _ = self.model.calc_new_mesh(
                                tgt_verts, tgt_faces, pred_exp, tgt_operators, tgt_dfn_info, tgt_img
                            )
                            pred_outputs_np = pred_outputs.detach().cpu().numpy()

                            CurrBS=pred_outputs_np.shape[0]
                            for b_idx in range(CurrBS):
                                save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
                                np.save(save_out_name, pred_outputs_np[b_idx])
            # else:
            #     ## from NFR checkpoint
            #     if SELF_RETARGET:
            #         print('self-retargeting! (src == tgt)')

            #         pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
            #         for index, batch in pbar:
            #             if index==0:
            #                 src_verts = batch.template[0]
            #                 src_faces = batch.faces[0]
            #                 src_m = trimesh.Trimesh(
            #                     vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
            #                 )

            #                 src_img = self.model.renderer.render_img(src_m).float().to(device)
            #                 src_img_feat = self.model.get_img_feat(src_img)[None]
            #                 src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
            #                 src_operators = self.model.get_mesh_operators(src_m)
            #             else:
            #                 if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:            
            #                     src_verts = batch.template[0]
            #                     src_faces = batch.faces[0]
            #                     src_m = trimesh.Trimesh(
            #                         vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
            #                     )

            #                     src_img = self.model.renderer.render_img(src_m).float().to(device)
            #                     src_img_feat = self.model.get_img_feat(src_img)[None]
            #                     src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
            #                     src_operators = self.model.get_mesh_operators(src_m)

            #             with torch.no_grad():
            #                 inputs_v = self.model.get_inputs(batch.vertices, batch.faces[0])# [B, V, 3+3]

            #                 ## get expression
            #                 self.model.model.update_precomputes(src_dfn_info)
            #                 pred_exp = self.model.model.encode(inputs_v, src_img.to(device), N_F=src_m.faces.shape[0])
            #                 pred_outputs, _, _ = self.model.calc_new_mesh(
            #                     src_verts, src_faces, pred_exp, src_operators, src_dfn_info, src_img
            #                 )
            #                 pred_outputs_np = pred_outputs.detach().cpu().numpy()

            #                 CurrBS=pred_outputs_np.shape[0]
            #                 for b_idx in range(CurrBS):
            #                     save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
            #                     np.save(save_out_name, pred_outputs_np[b_idx])
            #     else:
            #         print('cross-retargeting!')
            #         tgt_m = trimesh.Trimesh(vertices=tgt_v, faces=tgt_f)
            #         tgt_dfn_info = nfr_utils.get_dfn_info(tgt_m, map_location=device)
            #         tgt_verts = tgt_v_th[0]
            #         tgt_faces = torch.from_numpy(tgt_m.faces).to(device)
            #         tgt_img = trainer.model.renderer.render_img(tgt_m).float().to(device)
            #         tgt_operators = trainer.model.get_mesh_operators(tgt_m)
                    
            #         pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
            #         for index, batch in pbar:
            #             if index==0:
            #                 src_m = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())

            #                 src_img = trainer.model.renderer.render_img(src_m).float().to(device)
            #                 src_img_feat = trainer.model.get_img_feat(src_img)[None]
            #                 src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
            #             else:
            #                 if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:
            #                     src_m = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())

            #                     src_img = trainer.model.renderer.render_img(src_m).float().to(device)
            #                     src_img_feat = trainer.model.get_img_feat(src_img)[None]
            #                     src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)

            #             with torch.no_grad():
            #                 inputs_v = trainer.model.get_inputs(batch.vertices, batch.faces[0])# [B, V, 3+3]

            #                 ## get expression
            #                 trainer.model.model.update_precomputes(src_dfn_info)
            #                 pred_exp = trainer.model.model.encode(inputs_v, src_img.to(device), N_F=src_m.faces.shape[0])
            #                 pred_outputs, _, _ = trainer.model.calc_new_mesh(
            #                     tgt_verts, tgt_faces, pred_exp, tgt_operators, tgt_dfn_info, tgt_img
            #                 )
            #                 pred_outputs_np = pred_outputs.detach().cpu().numpy()

            #                 CurrBS=pred_outputs_np.shape[0]
            #                 for b_idx in range(CurrBS):
            #                     save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
            #                     np.save(save_out_name, pred_outputs_np[b_idx])
        elif self.opts.version == 1:
            if SELF_RETARGET:
                print('self-retargeting! (src == tgt)')
                pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
                for index, batch in pbar:
                    with torch.no_grad():
                        pred_outputs, _ = self.model.retarget(
                            batch.template, batch.vertices, batch.template
                        )
                    losses_val = stack_mse(batch, pred_outputs, losses_val, denom)

                    pred_outputs_np = pred_outputs.detach().cpu().numpy()
                    gt_np = batch.vertices.cpu().numpy() if self.opts.save_gt else None

                    CurrBS=pred_outputs_np.shape[0]
                    for b_idx in range(CurrBS):
                        save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
                        np.save(save_out_name, pred_outputs_np[b_idx])

                        if self.opts.save_gt:
                            save_gt_name = GT_log_dir+f'/{index*CurrBS+b_idx:06d}.npy'
                            np.save(save_gt_name, gt_np[b_idx])
            else:
                print('cross-retargeting! (src != tgt)')
                pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
                for index, batch in pbar:
                    with torch.no_grad():
                        pred_outputs, _ = self.model.retarget(
                            batch.template, batch.vertices, tgt_v_th
                        )

                    pred_outputs_np = pred_outputs.detach().cpu().numpy()
                    gt_np = batch.vertices.cpu().numpy() if self.opts.save_gt else None

                    CurrBS=pred_outputs_np.shape[0]
                    for b_idx in range(CurrBS):
                        save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
                        np.save(save_out_name, pred_outputs_np[b_idx])
                        if self.opts.save_gt:
                            save_gt_name = GT_log_dir+f'/{index*CurrBS+b_idx:06d}.npy'
                            np.save(save_gt_name, gt_np[b_idx])
                        
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
                        if not torch.equal(batch.template[0], src_template[0]):
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

    def save_raw_source(self, SELECT_DATA, SELECT_MESH, EXP_NUM, out_name_override, src_frame_slice=None):
        """
            Dump a dataset's raw (untouched, non-retargeted) per-frame vertices.
            Used for the "original" cases: no model inference involved.
        """
        data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict','ict-cap','ict_face_only']
        selection = data_name_list[SELECT_DATA]
        device = self.device

        dataset = EvalDataset(
            data_name=selection,
            toggle=False,
            ict_cap_id_num=SELECT_MESH,
            ict_cap_exp_num=EXP_NUM,
        )

        if src_frame_slice is not None:
            if selection != 'mf_ROM':
                raise NotImplementedError('src_frame_slice is only supported for mf_ROM sources')
            s, e = src_frame_slice
            dataset.mf_ROM_datalist = dataset.mf_ROM_datalist[s:e]
            dataset.len = len(dataset.mf_ROM_datalist)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            collate_fn=partial(CBD_collate_wrapper_eval, device=device),
        )

        out_dir = f'./vis_CBD/{self.opts.ckpt.split("/")[-1]}/{out_name_override}'
        vert_dir = out_dir + '/verts'
        os.makedirs(vert_dir, exist_ok=True)
        os.makedirs(out_dir + '/img', exist_ok=True)

        for index, batch in enumerate(tqdm(dataloader)):
            v = batch.vertices[0].detach().cpu().numpy()
            np.save(f'{vert_dir}/{index:06d}.npy', v)
        print(f'done (raw source): {out_name_override}, {len(dataset)} frames')

    def retarget_from_verts_folder(self, src_verts_folder, src_selection, src_mesh_idx,
                                    tgt_selection, tgt_mesh_idx, out_name_override):
        """
            Cyclic retargeting: treat a folder of already-generated per-frame
            vertices (the output of a previous retargeting pass, i.e. identity
            `src_selection`/`src_mesh_idx` performing the source motion) as the
            new source motion, and retarget it to `tgt_selection`/`tgt_mesh_idx`.

            Mirrors save_vis_data's cross-retargeting branch for whichever
            architecture is selected by self.opts.version/self.opts.NFR (NGBC,
            NC, NFS, or NFR), but sources per-frame data from disk instead of
            an EvalDataset.
        """
        device = self.device

        src_v, src_f, _ = self.get_mesh(src_selection, None, src_mesh_idx)
        tgt_v, tgt_f, _ = self.get_mesh(tgt_selection, None, tgt_mesh_idx)

        out_dir = f'./vis_CBD/{self.opts.ckpt.split("/")[-1]}/{out_name_override}'
        vert_dir = out_dir + '/verts'
        os.makedirs(vert_dir, exist_ok=True)
        os.makedirs(out_dir + '/img', exist_ok=True)

        frame_files = sorted(glob.glob(src_verts_folder + '/*.npy'))
        if len(frame_files) == 0:
            raise FileNotFoundError(f'no frames found in {src_verts_folder}')

        if self.opts.NFR:
            self.model.model.eval()
        else:
            self.model.eval()

        if self.opts.version == 1:
            # ── NC (CageNet): fused retarget(src_neutral, src_deformed, tgt_neutral) ──
            src_v_th = torch.tensor(src_v).float()[None].to(device)
            tgt_v_th = torch.tensor(tgt_v).float()[None].to(device)

            for i, fp in enumerate(tqdm(frame_files)):
                frame_v = np.load(fp)
                frame_v_th = torch.tensor(frame_v).float()[None].to(device)

                with torch.no_grad():
                    pred_outputs, _ = self.model.retarget(src_v_th, frame_v_th, tgt_v_th)
                np.save(f'{vert_dir}/{i:06d}.npy', pred_outputs[0].detach().cpu().numpy())

        elif self.opts.version == 0 and self.opts.NFR == False:
            # ── NFS: encode_id/encode_seg(target) once, encode_exp(source frame) + decode(target) per frame ──
            import utils.nfr_utils as nfr_utils
            from utils.mesh_utils import get_mesh_operators

            tgt_m = trimesh.Trimesh(vertices=tgt_v, faces=tgt_f)
            tgt_verts = torch.tensor(tgt_v).float().to(device)
            tgt_faces = torch.from_numpy(tgt_f).to(device)
            tgt_dfn_info = nfr_utils.get_dfn_info(tgt_m, map_location=device)
            tgt_operators = get_mesh_operators(tgt_m)
            tgt_img = self.model.renderer.render_img(tgt_m).float().to(device)
            tgt_img_feat = self.model.get_img_feat(tgt_img)
            tgt_vert_feat = self.model.get_local_feature(tgt_verts[None], tgt_faces, tgt_img_feat).float()
            tgt_tri_feat = self.model.get_local_feature(
                tgt_verts[None], tgt_faces, tgt_img_feat, at='faces'
            ).float()

            with torch.no_grad():
                pred_seg_coeff = self.model.encode_seg(tgt_vert_feat, tgt_dfn_info)
                pred_id_coeff = self.model.encode_id(tgt_vert_feat, tgt_dfn_info)

            src_m = trimesh.Trimesh(vertices=src_v, faces=src_f)
            src_faces_th = torch.from_numpy(src_f).to(device)
            src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
            src_img = self.model.renderer.render_img(src_m).float().to(device)
            src_img_feat = self.model.get_img_feat(src_img)

            for i, fp in enumerate(tqdm(frame_files)):
                frame_v = np.load(fp)
                frame_v_th = torch.tensor(frame_v).float()[None].to(device)

                with torch.no_grad():
                    vert_feat_exp = self.model.get_local_feature(
                        frame_v_th, src_faces_th, src_img_feat
                    ).float() * 1.3
                    pred_exp_coeff = self.model.encode_exp(
                        vert_feat_exp, src_dfn_info, batch_process=True, verbose=False
                    )
                    inputs = (
                        tgt_tri_feat if self.opts.dec_type == 'jacob' else tgt_vert_feat,
                        pred_exp_coeff, pred_id_coeff, pred_seg_coeff,
                        None, tgt_verts[None], tgt_faces, tgt_operators
                    )
                    decode_out = self.model.decode(inputs, batch_process=True)
                    pred_outputs = decode_out[0] if isinstance(decode_out, tuple) else decode_out
                np.save(f'{vert_dir}/{i:06d}.npy', pred_outputs[0].detach().cpu().numpy())

        elif self.opts.version == 0 and self.opts.NFR == True:
            # ── NFR: encode(source, stateful dfn_info) then switch to target dfn_info + calc_new_mesh ──
            import utils.nfr_utils as nfr_utils

            tgt_m = trimesh.Trimesh(vertices=tgt_v, faces=tgt_f)
            tgt_dfn_info = nfr_utils.get_dfn_info(tgt_m, map_location=device)
            tgt_verts = torch.tensor(tgt_v).float().to(device)
            tgt_faces = torch.from_numpy(tgt_f).to(device)
            tgt_img = self.model.renderer.render_img(tgt_m).float().to(device)
            tgt_operators = self.model.get_mesh_operators(tgt_m)

            src_m = trimesh.Trimesh(vertices=src_v, faces=src_f)
            src_faces_th = torch.from_numpy(src_f).to(device)
            src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
            src_img = self.model.renderer.render_img(src_m).float().to(device)

            for i, fp in enumerate(tqdm(frame_files)):
                frame_v = np.load(fp)
                frame_v_th = torch.tensor(frame_v).float()[None].to(device)

                with torch.no_grad():
                    inputs_v = self.model.get_inputs(frame_v_th, src_faces_th)

                    self.model.model.update_precomputes(src_dfn_info)
                    pred_exp = self.model.model.encode(inputs_v, src_img.to(device), N_F=src_f.shape[0])

                    self.model.model.update_precomputes(tgt_dfn_info)
                    pred_outputs, _, _ = self.model.calc_new_mesh(
                        tgt_verts, tgt_faces, pred_exp, tgt_operators, tgt_dfn_info, tgt_img
                    )
                np.save(f'{vert_dir}/{i:06d}.npy', pred_outputs[0].detach().cpu().numpy())

        else:
            # ── ours (NGBC): predict_coordinate(target) once, retarget_animation per frame ──
            src_n = igl.per_vertex_normals(src_v, src_f)
            tgt_n = igl.per_vertex_normals(tgt_v, tgt_f)

            src_v_th = torch.tensor(src_v).float()[None].to(device)
            src_n_th = torch.tensor(src_n).float()[None].to(device)
            tgt_v_th = torch.tensor(tgt_v).float()[None].to(device)
            tgt_n_th = torch.tensor(tgt_n).float()[None].to(device)

            with torch.no_grad():
                key_weight = self.model.predict_coordinate(tgt_v_th, tgt_n_th)

            for i, fp in enumerate(tqdm(frame_files)):
                frame_v = np.load(fp)
                frame_n = igl.per_vertex_normals(frame_v, src_f)
                frame_v_th = torch.tensor(frame_v).float()[None].to(device)
                frame_n_th = torch.tensor(frame_n).float()[None].to(device)

                with torch.no_grad():
                    pred_outputs, _ = self.model.retarget_animation(
                        src_v_th, src_n_th, frame_v_th, frame_n_th, key_weight, tgt_v_th
                    )
                np.save(f'{vert_dir}/{i:06d}.npy', pred_outputs[0].detach().cpu().numpy())

        print(f'done (cyclic): {out_name_override}, {len(frame_files)} frames')

    def save_render_retarget_request(self):
        """
            Driver for report/render_retarget_request_2026-07-27.md's 12 cases.
            Exp1.1 (raw mf test original) is intentionally NOT produced here --
            it's rendered directly from the untouched source npy folder by
            render/render_trimesh_fig.py, no inference needed.
        """
        EXP_FREE_FACE_SLICE = (3773, 4964)  # mf_ROM test id, EXP_free_face motion only
        MF_TEST_IDX, MF_TRAIN_IDX = 12, 0
        ICT_M00, ICT_M02, ICT_M05 = 0, 2, 5
        ICT_CAP_EXP_924 = 1  # non-zero -> _cap/20240325_MySlate_924_exp_coeffs.npy

        ckpt_dir = self.opts.ckpt.split('/')[-1]
        base = f'./vis_CBD/{ckpt_dir}'

        # ---- Exp1: mf test (EXP_free_face) -> {ict m00, ict m02, mf train} ----
        self.save_vis_data(4, MF_TEST_IDX, 6, ICT_M00, 0,
                            src_frame_slice=EXP_FREE_FACE_SLICE, out_name_override='exp1_2_mf_to_ict_m00')
        self.save_vis_data(4, MF_TEST_IDX, 6, ICT_M02, 0,
                            src_frame_slice=EXP_FREE_FACE_SLICE, out_name_override='exp1_3_mf_to_ict_m02')
        self.save_vis_data(4, MF_TEST_IDX, 4, MF_TRAIN_IDX, 0,
                            src_frame_slice=EXP_FREE_FACE_SLICE, out_name_override='exp1_4_mf_to_mf_train')

        # ---- Exp1 cyclic: retarget the above back to mf test id ----
        # note: save_vis_data appends '-masked' to its output folder name (opts.use_t_mask is
        # forced True in __main__), so the predecessor folders read here carry that suffix.
        self.retarget_from_verts_folder(f'{base}/exp1_2_mf_to_ict_m00-masked/verts', 'ict-cap', ICT_M00,
                                         'mf_ROM', MF_TEST_IDX, 'exp1_5_cyclic_ict_m00_to_mf')
        self.retarget_from_verts_folder(f'{base}/exp1_3_mf_to_ict_m02-masked/verts', 'ict-cap', ICT_M02,
                                         'mf_ROM', MF_TEST_IDX, 'exp1_6_cyclic_ict_m02_to_mf')
        self.retarget_from_verts_folder(f'{base}/exp1_4_mf_to_mf_train-masked/verts', 'mf_ROM', MF_TRAIN_IDX,
                                         'mf_ROM', MF_TEST_IDX, 'exp1_7_cyclic_mf_train_to_mf')

        # ---- Exp2: ict m02 (924 capture) original + -> {ict m05, mf test} ----
        self.save_raw_source(6, ICT_M02, ICT_CAP_EXP_924, out_name_override='exp2_1_ict_m02_original')
        self.save_vis_data(6, ICT_M02, 6, ICT_M05, ICT_CAP_EXP_924,
                            out_name_override='exp2_2_ict_m02_to_ict_m05')
        self.save_vis_data(6, ICT_M02, 4, MF_TEST_IDX, ICT_CAP_EXP_924,
                            out_name_override='exp2_3_ict_m02_to_mf')

        # ---- Exp2 cyclic: retarget the above back to ict m02 ----
        self.retarget_from_verts_folder(f'{base}/exp2_2_ict_m02_to_ict_m05-masked/verts', 'ict-cap', ICT_M05,
                                         'ict-cap', ICT_M02, 'exp2_4_cyclic_ict_m05_to_ict_m02')
        self.retarget_from_verts_folder(f'{base}/exp2_3_ict_m02_to_mf-masked/verts', 'mf_ROM', MF_TEST_IDX,
                                         'ict-cap', ICT_M02, 'exp2_5_cyclic_mf_to_ict_m02')

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
        opts.config='config/train_NFS.yml'
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
    pipeline.save_render_retarget_request() ## report/render_retarget_request_2026-07-27.md, 12-case cyclic retargeting set

