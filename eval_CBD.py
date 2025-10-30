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

# import sys
# from pathlib import Path
# __abs_path__ = str(Path(__file__).parents[1].absolute())

# if not __abs_path__ in sys.path:
#     sys.path+=[__abs_path__]
import pickle

import torch
import torch.nn as nn
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

from models.baseline import CageNet
from models.NGBC import (
    NeuralGeneralizedBarycentricCoordinate, 
    NeuralGeneralizedBarycentricCoordinate5,
    NeuralGeneralizedBarycentricCoordinate8,
    NeuralGeneralizedBarycentricCoordinate55
)

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
    
    parser.add_argument("--in_type",      type=int,   default=1,
                        help='input type (0: position, 1: position + normal')
    parser.add_argument("--out_type",      type=int,   default=1,      
                        help='output type (0: cage v, 1: cage delta_v, 2: cage delta_T mat, 3: vertex T mat')
    
    #### Choose a last layer activation for key_weight_model()
    parser.add_argument("--last_activation", default="relu", choices=["relu", "elu", "softmax", "softplus", "none"],
        help="Choose a last layer activation for NGBC.key_weight_model()", 
    )
    
    parser.add_argument("--no_pou",dest='no_pou', action='store_true')
    parser.set_defaults(no_pou=False)
    
    parser.add_argument("--start_epoch",  type=int,   default=0,      help='number of epochs')
    parser.add_argument("--lr",           type=float, default=0.0002, help='learning rate')
    
    parser.add_argument("--batch_size",   type=int,   default=1,      help='batch size')

    parser.add_argument("--seed",         type=int,   default=42,     help='random seed')
    parser.add_argument("--ckpt",         type=str,   default=None)    
    parser.add_argument("--continue_ckpt",dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)
    
    parser.add_argument("--n_sampling",   dest='n_sampling', action='store_true')
    parser.set_defaults(n_sampling=False)
    
    parser.add_argument("--use_decimate", dest='use_decimate', action='store_true')
    parser.set_defaults(use_decimate=False)
    
    parser.add_argument("--use_scheduler",dest='use_scheduler', action='store_true')
    parser.set_defaults(use_scheduler=False)
    
    parser.add_argument("--use_data2",dest='use_data2', action='store_true')
    parser.set_defaults(use_data2=False)
    parser.add_argument("--use_data3",dest='use_data3', action='store_true')
    parser.set_defaults(use_data3=False)

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
    
    
    parser.add_argument("--optim_cage",dest='optim_cage', action='store_true')
    parser.set_defaults(optim_cage=False)
    
    
    parser.add_argument("--align_latent",dest='align_latent', action='store_true')
    parser.set_defaults(align_latent=False)
    
    args = parser.parse_args()
    return args

# --- Loss Functions ---

class Trainer():
    def __init__(self, opts):
        # set opts
        self.opts = opts
        self.set_seed(self.opts)
        self.device = opts.device

        last_act_list = ["relu", "elu", "softmax", "softplus", "none"]
        last_act_list = [self.opts.last_activation==l_act for l_act in last_act_list]
        if opts.version==0:
            #from models import NFS
            #from utils.nfr_utils import get_dfn_info
            #self.get_dfn_info = get_dfn_info
            
            #self.model = NFS(self.opts, None, print_param=True).to(self.device)
            from evaluation import Trainer
            trainer = Trainer(opts)
            self.model = trainer.model
            
        elif opts.version==1:
            self.model = CageNet(
                device=self.device,
                optim_cage=self.opts.optim_cage
            )
        elif opts.version==2:
            self.model = NeuralGeneralizedBarycentricCoordinate(
                self.opts, 
                hid_dim=256,
                num_cage_vertices=self.opts.num_cage_v,
                num_layers=4,
                use_relu=True,
                is_train=True, 
                device=self.device,
            )
        elif opts.version==5:
            self.model = NeuralGeneralizedBarycentricCoordinate5(
                opts, num_layers=4,
                num_cage_vertices=self.opts.num_cage_v,
                use_exp_recon=False, # not used yet
                use_shp_recon=False, # not used yet
                use_shp=False,
                use_relu=last_act_list[0],
                use_elu=last_act_list[1],
                use_softmax=last_act_list[2],
                use_softplus=last_act_list[3],
                no_activation=last_act_list[4],
                use_least_N_on_V=False,
                is_train=True,
                use_pou = ~self.opts.no_pou,
                device=self.device,
                #hid_dim=128 if self.opts.use_data2 or self.opts.use_data3 else 256,
                hid_dim=128 if self.opts.align_latent else 256,
            )
        elif opts.version==8:
            self.model = NeuralGeneralizedBarycentricCoordinate8(
                opts, num_layers=4,
                num_cage_vertices=self.opts.num_cage_v,
                use_exp_recon=False, # not used yet
                use_shp_recon=False, # not used yet
                use_shp=False,
                use_relu=last_act_list[0],
                use_elu=last_act_list[1],
                use_softmax=last_act_list[2],
                use_softplus=last_act_list[3],
                no_activation=last_act_list[4],
                use_least_N_on_V=False,
                is_train=True,
                use_pou = ~self.opts.no_pou,
                device=self.device,
                #hid_dim=128 if self.opts.use_data2 or self.opts.use_data3 else 256,
                hid_dim=128 if self.opts.align_latent else 256,
            )
        elif opts.version==55:
            self.model = NeuralGeneralizedBarycentricCoordinate55(
                opts, num_layers=4,
                num_cage_vertices=self.opts.num_cage_v,
                use_exp_recon=False, # not used yet
                use_shp_recon=False, # not used yet
                use_shp=False,
                use_relu=last_act_list[0],
                use_elu=last_act_list[1],
                use_softmax=last_act_list[2],
                use_softplus=last_act_list[3],
                no_activation=last_act_list[4],
                use_least_N_on_V=False,
                is_train=True,
                use_pou = ~self.opts.no_pou,
                device=self.device,
                #hid_dim=128 if self.opts.use_data2 or self.opts.use_data3 else 256,
                hid_dim=128 if self.opts.align_latent else 256,
            )
        else:
            raise NotImplementedError('No matching model version')
        
        # load weight
        self.load_weight()
    
    def load_weight(self):
        if self.opts.ckpt:
            if self.opts.continue_ckpt:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, f"*_{self.opts.start_epoch:03d}.pth"))[0]
            else:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
            print(f"Loading... {ckpt}")
            
            ckpt_dict = torch.load(ckpt, map_location=self.device)            
            self.model.load_state_dict(ckpt_dict,strict=False)
        else:
            print('no ckpt found, training from scratch!')

    def evaluate(self):
        
            
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        BS = self.opts.batch_size
        
        selection=[
            # voca | biwi | mf_SEN | coma | mf_ROM
            [True,  False, False, False, False], # 0
            [False,  True, False, False, False], # 1
            [False, False,  True, False, False], # 2
            [False, False, False,  True, False], # 3
            [False, False, False, False,  True], # 4
            [False, False,  True, False,  True], # 5
            [ True, False, False,  True, False], # 6
            [ True,  True,  True,  True,  True], # -1
        ]
        selection = selection[self.opts.data_selection]
        
        
        self.dataset = CBDDataset(
            self.opts, n_components=1_000, is_train=False, is_valid=False,
            use_voca=selection[0],
            use_biwi=selection[1],
            use_mf_SEN=selection[2],
            use_coma=selection[3],
            use_mf_ROM=selection[4],
        )
        
        # self.neighbor_maps = {
        #     i: igl.adjacency_list(mesh_info['face'])
        #     for i, mesh_info in enumerate([
        #         self.dataset.voca_mesh,
        #         self.dataset.biwi_mesh,
        #         self.dataset.mf_SEN_mesh,
        #     ])
        # }
        # self.neighbor_pad_mask = {}
        # for i in self.neighbor_maps.keys():
        #     # (idx_pad, mask)
        #     self.neighbor_pad_mask[i] = build_padded_neighbors(self.neighbor_maps[i], device=self.device)

        data_sampler = CBDdataSampler(
            self.dataset.len_list, 
            self.opts.batch_size,
            shuffle=False,
            balance=False,
            is_train=False,
            is_valid=False,
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_sampler=data_sampler,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0,
        )
        ##########################################################################################################
        
        
        ###### Logging ###########################################################################################
        # make logdir --------------------------------------------------------------------------------------------
        os.makedirs(self.opts.log_dir, exist_ok=True)
        
        import datetime
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
                    
        # ckpt_path = self.opts.ckpt.split('ckpts_CBD')[-1][1:]
        # self.opts.log_dir = os.path.join(self.opts.log_dir, ckpt_path+'-eval-'+now)
        ckpt_path = self.opts.ckpt.split('/')[-1]
        self.opts.log_dir = os.path.join(self.opts.log_dir, ckpt_path+'-eval',selection+'-pca_data')
        
        os.makedirs(self.opts.log_dir, exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        
        # save options as json -----------------------------------------------------------------------------------
        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
            
        # save train option as yml
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), opts)
        
        # self logger
        self.logger = open(os.path.join(self.opts.log_dir, "log.txt"), 'w')
        print(f'Saving log at: {self.opts.log_dir}')
        
        print(self.dataset.get_data_config())
        print(data_sampler.get_sampler_config())

        self.logger.write(self.dataset.get_data_config())
        self.logger.write(data_sampler.get_sampler_config())
        #---------------------------------------------------------------------------------------------------------
        ##########################################################################################################
        
        
        
        # eval loop ##############################################################################################
        global_step = 0
        BEST_LOSS = 100_000_000
        
        check_usage = False
        
        len_data = len(self.dataloader)
        denom = 1 / len_data
        
        self.model.eval()
        
        losses_val = {
            "MSE": 0.0
        }
        if self.opts.use_t_mask:
            losses_val["MSE-in"] = 0.0
            losses_val["MSE-out"] = 0.0
        
        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)
        for index, batch in pbar:
            
            # model forward ----------------------------------------------------------------------------------
            with torch.no_grad():
                pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _ = self.model(
                    batch.template, batch.vertices, 
                    batch.template_normal, batch.vertices_normal,
                    batch.mesh_data, epoch=0
                )
            # ------------------------------------------------------------------------------------------------
            
            
            # Metric -----------------------------------------------------------------------------------------
            with torch.no_grad():
                mesh_data_num = batch.mesh_data.cpu().numpy()
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf', 'ict'])[mesh_data_num]
                
                HB = batch.vertices.shape[0] // 2
                
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
                batch.vertices, pred_vertices
            ).item() * denom # for NGBC model
            # ------------------------------------------------------------------------------------------------
        
            
            # ------------------------------------------------------------------------------------------------
            interv_val = round(len_data / 5)
            if index % interv_val == 0:
                # for visualization
                vertices = batch.vertices.cpu()
                faces = batch.faces.cpu()
                                
                frame = HB
                v_list = [
                    vertices[0].cpu().detach(),
                    # vertices[1].cpu().detach(),
                    # vertices[HB].cpu().detach(),
                    # vertices[BS-1].cpu().detach(),
                    pred_vertices[0].cpu().detach(),
                    # pred_vertices[1].cpu().detach(),
                    # pred_vertices[HB].cpu().detach(),
                    # pred_vertices[BS-1].cpu().detach(),
                ]
                len_v = len(v_list)
                f_list=[faces] * len_v
                save_logdir = f"{self.opts.log_dir}/img"
                save_img_name = f"{index:04d}"
                
                plot_image_array(
                    v_list, f_list, 
                    rot_list=[[0,0,0]]*len_v,
                    size=1, bg_black=False, mode='shade', 
                    logdir=save_logdir, 
                    name=save_img_name, save=True
                )
        ##########################################################################################################
        
        # write log
        log_text = f"[Eval] "
        for key, value in losses_val.items():
            txt = f"{key}: {value:.6e} "
            print(txt)
            log_text += txt
        self.logger.write(log_text+"\n")                
        print('done!')

    
    def evaluate2(self):
        """
            self-retargeting task
        """
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        BS = self.opts.batch_size
        HB = BS // 2
        device=self.device
        
        if self.opts.data_selection == -1:
            raise NotImplementedError('only works for individual data')
        data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict']
        selection = data_name_list[self.opts.data_selection]
        # if 'mf' in selection:
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
            
        self.dataset = EvalDataset(data_name=selection, toggle=False) # if eve-s01
        # self.dataset = EvalDataset(data_name=selection, toggle=True) # if char-s02
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.opts.batch_size,
            collate_fn=partial(CBD_collate_wrapper_eval, device=self.device),
            #num_workers=8,
        )
        ##########################################################################################################
        
        
        ###### Logging ###########################################################################################
        # make logdir --------------------------------------------------------------------------------------------
        os.makedirs(self.opts.log_dir, exist_ok=True)
                            
        ckpt_path = self.opts.ckpt.split('/')[-1]
        self.opts.log_dir = os.path.join(self.opts.log_dir, ckpt_path+'-eval', selection)
        
        if self.opts.use_t_mask:
            self.opts.log_dir = self.opts.log_dir + '-masked'
            
        os.makedirs(self.opts.log_dir, exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        
        # save options as json -----------------------------------------------------------------------------------
        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
            
        # save train option as yml
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), opts)
        
        # self logger
        self.logger = open(os.path.join(self.opts.log_dir, "log.txt"), 'w')
        print(f'Saving log at: {self.opts.log_dir}')
        
        print(self.dataset.get_data_config())
        self.logger.write(self.dataset.get_data_config())
        #---------------------------------------------------------------------------------------------------------
        ##########################################################################################################
        
        
        
        # eval loop ##############################################################################################        
        global_step = 0
        BEST_LOSS = 100_000_000
        
        check_usage = False
        
        len_data = len(self.dataloader)
        denom = 1 / len_data
        
        self.model.eval()
        
        losses_val = {
            "MSE": 0.0
        }
        
        if self.opts.use_t_mask:
            losses_val["MSE-in"] = 0.0
            losses_val["MSE-out"] = 0.0
            
        mesh_data = self.dataset.data_name
                
        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)
        for index, batch in pbar:
            
            # model forward ----------------------------------------------------------------------------------
            with torch.no_grad():
                
                ### only for NFS #############################################################
                if self.opts.version==0:
                    ## only onces!!!
                    if index==0:
                        src_mesh = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())
                        
                        ## common routine
                        dfn_info = self.get_dfn_info(src_mesh, map_location=self.device)
                        img = self.model.renderer.render_img(src_mesh).float().to(self.device)
                        img_feat = self.model.get_img_feat(img)
                        vert_feat = self.model.get_local_feature(batch.template[0][None], batch.faces[0], img_feat).float()
                            
                        pred_id_coeff  = self.model.encode_id(vert_feat, dfn_info)
                        pred_seg_coeff = self.model.encode_seg(vert_feat, dfn_info)
                    else:
                        if (batch.template[0].cpu().numpy() - src_mesh.vertices).mean() != 0:
                            src_mesh = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())
                    
                            ## common routine
                            dfn_info = self.get_dfn_info(src_mesh, map_location=self.device)
                            
                            img = self.model.renderer.render_img(src_mesh).float().to(self.device)
                            img_feat = self.model.get_img_feat(img)
                            vert_feat = self.model.get_local_feature(batch.template[0][None], batch.faces[0], img_feat).float()
                            
                            pred_id_coeff  = self.model.encode_id(vert_feat, dfn_info)
                            pred_seg_coeff = self.model.encode_seg(vert_feat, dfn_info)
                            
                    # vert_feat_exp = []
                    # for gt_v in batch.vertices:
                    #     _tmp_ = self.model.get_local_feature(gt_v[None], batch.faces[0], img_feat).float()
                    #     vert_feat_exp.append(_tmp_)
                    # vert_feat_exp = torch.vstack(vert_feat_exp)
                    vert_feat_exp =  self.model.get_local_feature(batch.vertices, batch.faces[0], img_feat).float()

                    with torch.no_grad():
                        pred_exp_coeff = self.model.encode_exp(vert_feat_exp, dfn_info, batch_process=True, verbose=False)# [W, Rig]
                    
                        inputs = (
                            vert_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff,
                            None, batch.template[0][None], batch.faces[0], None
                        )
                        pred_vertices, _ = self.model.decode(inputs, batch_process=True)
                ##############################################################################
                
                else:
                    #pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _ = self.model(
                    if self.opts.version==1:
                        pred_vertices, _ = trainer.model.retarget(
                            batch.template, batch.vertices, batch.template
                        )
                    else:
                        pred_vertices, _, _, _, _, _, _ = self.model(
                            batch.template, batch.vertices, 
                            batch.template_normal, batch.vertices_normal,
                            mesh_data=batch.mesh_data, epoch=0
                        )
                    
                # Metric
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
            # ------------------------------------------------------------------------------------------------
            if self.opts.save_gt:
                save_gt_logdir = f"{self.opts.log_dir}/../../GT_{selection}"
                os.makedirs(save_gt_logdir, exist_ok=True)
                curr_batch = batch.vertices.shape[0]
                
                for b_idx in range(curr_batch):
                    save_gt_name = f"{save_gt_logdir}/{index*curr_batch + b_idx:06d}.npy"
                    np.save(save_gt_name, batch.vertices[b_idx].cpu().numpy())
                
            if self.opts.save_vert:
                save_vert_logdir = f"{self.opts.log_dir}/verts"
                # for pred_vert in pred_vertices:
                os.makedirs(save_vert_logdir, exist_ok=True)
                curr_batch = pred_vertices.shape[0]
                
                for b_idx in range(curr_batch):
                    save_vert_name = f"{save_vert_logdir}/{index*curr_batch + b_idx:06d}.npy"
                    np.save(save_vert_name, pred_vertices[b_idx].detach().cpu().numpy())
            
            # ------------------------------------------------------------------------------------------------
            # visualization for debugging
            if self.opts.batch_size > 1:
                interv_val = round(len_data / 10)
                if index % interv_val == 0:
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                    pred_vertices_ = pred_vertices.detach().cpu()
                    
                    v_list = [
                        vertices[0],
                        vertices[1],
                        vertices[HB],
                        vertices[-1],
                        pred_vertices_[0],
                        pred_vertices_[1],
                        pred_vertices_[HB],
                        pred_vertices_[-1],
                    ]
                    len_v = len(v_list)
                    f_list=[faces[0]] * len_v
                    save_logdir = f"{self.opts.log_dir}/img"
                    save_img_name = f"{index:04d}"
                    
                    plot_image_array(
                        v_list, f_list, 
                        rot_list=[[0,0,0]]*len_v,
                        size=1, bg_black=False, mode='shade', 
                        logdir=save_logdir, 
                        name=save_img_name, save=True
                    )
                    # 11649/(11649+6309) + 6309/(11649+6309)
        ##########################################################################################################
        
        # write log
        log_text = f"[Eval] "
        for key, value in losses_val.items():
            txt = f"{key}: {value:.6e} "
            print(txt)
            log_text += txt
        self.logger.write(log_text+"\n")
        print('done!')
    
    def evaluate3(self):
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        BS = self.opts.batch_size
        HB = BS // 2
        device=self.device
        
        if self.opts.data_selection != 5:
            raise NotImplementedError('only works for individual data')
        data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict']
        selection = data_name_list[self.opts.data_selection]
        
        
        # self.dataset = EvalDataset(data_name=selection, toggle=False) # if eve-s01
        # # self.dataset = EvalDataset(data_name=selection, toggle=True) # if char-s02
        
        # self.dataloader = torch.utils.data.DataLoader(
        #     self.dataset,
        #     batch_size=self.opts.batch_size,
        #     collate_fn=partial(CBD_collate_wrapper_eval, device=self.device),
        #     #num_workers=8,
        # )
        from dataloader_mesh import (
            NFSDataset,
            # InvRigDataset,
        )
        self.opts.window_size=8
        self.opts.ict_face_only=False
        self.opts.selection=2 ## ICT-all (ICT-capture + ICT-synthetic)
        # self.opts.selection=1 ## ICT-all (ICT-synthetic)
        # self.opts.selection=0 ## ICT-all (ICT-capture)
        self.opts.seg_dim=20
        self.dataset = NFSDataset(self.opts, is_train=False, is_valid=False, return_audio_dir=True)
       
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)
        ##########################################################################################################
        
        
        ###### Logging ###########################################################################################
        # make logdir --------------------------------------------------------------------------------------------
        os.makedirs(self.opts.log_dir, exist_ok=True)
                            
        ckpt_path = self.opts.ckpt.split('/')[-1]
        self.opts.log_dir = os.path.join(self.opts.log_dir, ckpt_path+'-eval',selection)
        
        os.makedirs(self.opts.log_dir, exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        
        # save options as json -----------------------------------------------------------------------------------
        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
            
        # save train option as yml
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), opts)
        
        # self logger
        self.logger = open(os.path.join(self.opts.log_dir, "log.txt"), 'w')
        print(f'Saving log at: {self.opts.log_dir}')
        
        print(self.dataset.get_data_config())
        self.logger.write(self.dataset.get_data_config())
        #---------------------------------------------------------------------------------------------------------
        ##########################################################################################################
        
        
        
        # eval loop ##############################################################################################
        
        global_step = 0
        BEST_LOSS = 100_000_000
        
        check_usage = False
        
        len_data = len(self.dataloader)
        denom = 1 / len_data
        
        self.model.eval()
        
        losses_val = {
            "MSE": 0.0
        }
        
        if self.opts.use_t_mask:
            losses_val["MSE-in"] = 0.0
            losses_val["MSE-out"] = 0.0
            
        mesh_data = 'ict'
        from easydict import EasyDict
        from utils.mesh_utils import calc_norm_torch
        
        # import pdb;pdb.set_trace()
        recon_vDec = []
        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)
        for index, data in pbar:
            batch=EasyDict()
            # model forward ----------------------------------------------------------------------------------
            with torch.no_grad():
                # dummy, id_coeff, exp_coeff, template, dfn_info, operators, vertices, v_normal, faces, img, mesh_data, corr_feat=data
                _, batch.id_coeff, batch.exp_coeff, batch.template, batch.dfn_info, batch.operators, batch.vertices, batch.vertices_normal, batch.faces, batch.img, batch.mesh_data, batch.corr_feat=data
                
                batch.vertices = batch.vertices.squeeze(0).to(self.device)
                batch.vertices_normal = batch.vertices_normal.squeeze(0).to(self.device)
                # batch.template = batch.template.squeeze(0)
                batch.faces = batch.faces.squeeze(0)
                batch.template_normal = calc_norm_torch(batch.template, batch.faces, 'vert').repeat(self.dataset.WS,1,1).to(self.device)
                batch.template = batch.template.repeat(self.dataset.WS,1,1).to(self.device)
                batch.faces = batch.faces.to(self.device)
                
                ### only for NFS #############################################################
                if self.opts.version==0:
                    if index==0:
                        src_mesh = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())
                        
                        ## common routine
                        dfn_info = self.get_dfn_info(src_mesh, map_location=self.device)
                        img = self.model.renderer.render_img(src_mesh).float().to(self.device)
                        img_feat = self.model.get_img_feat(img)
                        vert_feat = self.model.get_local_feature(batch.template[0][None], batch.faces[0], img_feat).float()
                            
                        with torch.no_grad():
                            pred_id_coeff  = self.model.encode_id(vert_feat, dfn_info)
                            pred_seg_coeff = self.model.encode_seg(vert_feat, dfn_info)# [1, V, Seg]
                    else:
                        if (batch.template[0].cpu().numpy() - src_mesh.vertices).mean() != 0:
                            src_mesh = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())
                    
                            ## common routine
                            dfn_info = self.get_dfn_info(src_mesh, map_location=self.device)
                            
                            img = self.model.renderer.render_img(src_mesh).float().to(self.device)
                            img_feat = self.model.get_img_feat(img)
                            vert_feat = self.model.get_local_feature(batch.template[0][None], batch.faces[0], img_feat).float()
                            
                            with torch.no_grad():
                                pred_id_coeff  = self.model.encode_id(vert_feat, dfn_info)
                                pred_seg_coeff = self.model.encode_seg(vert_feat, dfn_info)# [1, V, Seg]

                    vert_feat_exp = []
                    for gt_v in batch.vertices:
                        _tmp_ = self.model.get_local_feature(gt_v[None], batch.faces[0], img_feat).float()
                        vert_feat_exp.append(_tmp_)
                    vert_feat_exp = torch.vstack(vert_feat_exp)

                    with torch.no_grad():
                        pred_exp_coeff = self.model.encode_exp(vert_feat_exp, dfn_info, batch_process=True, verbose=False)# [W, Rig]
                        
                        inputs = (
                            vert_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff,
                            None, batch.template[0][None], batch.faces[0], None
                        )
                        pred_vertices, _ = self.model.decode(inputs, batch_process=True)

                    # pred_vertices = self.model.inference(
                    #     gt_vertices=batch.vertices, 
                    #     src_mesh=src_mesh, 
                    #     tgt_mesh=src_mesh,
                    #     batch_process=True
                    # )
                    
                    # losses_val, pred_vertices, _, pred_exp_coeff, pred_id_coeff, pred_seg = self.model.evaluate(
                    #     batch, \
                    #     batch_process=False, \
                    #     return_all=True, \
                    #     stage=1, \
                    #     epoch=500
                    # )
                ##############################################################################
                
                else:
                    with torch.no_grad():
                        if self.opts.version == 1:
                            pred_vertices, _ = trainer.model.retarget(
                                batch.template, batch.vertices, batch.template
                            )
                        else:
                            #pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _ = self.model(
                            pred_vertices, _, _, _, _, _, _ = self.model(
                                batch.template, batch.vertices, 
                                batch.template_normal, batch.vertices_normal,
                                mesh_data=batch.mesh_data, epoch=0
                            )
                # Metric
                if self.opts.use_t_mask:
                    inner_mask = plateau_hat_points(batch.template)
                    outter_mask = 1 - inner_mask
                    
                    losses_val['MSE-in'] += F.mse_loss(
                        batch.vertices*inner_mask, pred_vertices*inner_mask
                    ).item() * denom # for NGBC model
                    
                    losses_val['MSE-out'] += F.mse_loss(
                        batch.template*outter_mask, pred_vertices*outter_mask
                    ).item() * denom # for NGBC model
                    
                loss_ = F.mse_loss(
                    batch.vertices.detach(), pred_vertices.detach()
                ).item() * denom # for NGBC model
                pbar.set_description(f'loss: {loss_:.5e}')
                losses_val['MSE'] += loss_

                recon_vDec.append(loss_)
                
                if self.opts.save_vert:
                    save_vert_logdir = f"{self.opts.log_dir}/verts"
                    # for pred_vert in pred_vertices:
                    os.makedirs(save_vert_logdir, exist_ok=True)
                    curr_batch = pred_vertices.shape[0]
                    
                    for b_idx in range(curr_batch):
                        save_vert_name = f"{save_vert_logdir}/{index*curr_batch + b_idx:06d}.npy"
                        np.save(save_vert_name, pred_vertices[b_idx].detach().cpu().numpy())
            # ------------------------------------------------------------------------------------------------
        
            
            # ------------------------------------------------------------------------------------------------
            interv_val = round(len_data / 10)
            if index % interv_val == 0:
                # for visualization
                vertices = batch.vertices.cpu().detach()
                faces = batch.faces.cpu().detach()
                pred_vertices_ = pred_vertices.cpu().detach()
                # import pdb;pdb.set_trace()
                
                v_list = [
                    vertices[0],
                    vertices[1],
                    vertices[HB],
                    vertices[-1],
                    pred_vertices_[0],
                    pred_vertices_[1],
                    pred_vertices_[HB],
                    pred_vertices_[-1],
                ]
                len_v = len(v_list)
                f_list=[faces] * len_v
                save_logdir = f"{self.opts.log_dir}/img"
                save_img_name = f"{index:04d}"
                
                plot_image_array(
                    v_list, f_list, 
                    rot_list=[[0,0,0]]*len_v,
                    size=1, bg_black=False, mode='shade', 
                    logdir=save_logdir, 
                    name=save_img_name, save=True
                )
                # 11649/(11649+6309) + 6309/(11649+6309)
        ##########################################################################################################
        
        recon_vDec = np.array(recon_vDec)        
        recon_vDec_mu = np.mean(recon_vDec)
        recon_vDec_std = np.std(recon_vDec)
        
        tmp_log = f"recon_vDec_mu: {recon_vDec_mu}"
        print(tmp_log)
        self.logger.write(tmp_log+'\n')
        tmp_log = f"recon_vDec_std: {recon_vDec_std}"
        print(tmp_log)
        self.logger.write(tmp_log+'\n')
        
        # write log
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
    
        python eval_CBD.py --version 2 --ckpt ./ckpts_CBD/2025-09-08-19-15-47-NGBC --in_type 1 --num_cage_v 768 --data_selection -1
        python eval_CBD.py --version 2 --ckpt ./ckpts_CBD/2025-09-08-13-21-05-NGBC --in_type 0 --num_cage_v 1024 --data_selection 5 --start_epoch 400
        
        python eval_CBD.py --version 2 --ckpt ./ckpts_CBD/2025-09-11-15-26-03-NGBC --in_type 0 --out_type 0 --num_cage_v 640 --data_selection 0
        python eval_CBD.py --version 2 --ckpt ./ckpts_CBD/2025-09-11-17-21-05-NGBC --in_type 0 --out_type 2 --num_cage_v 640 --data_selection 0 --batch_size 1 --device 'cpu'

        python eval_CBD.py --version 2 --ckpt ./ckpts_CBD/2025-09-11-17-53-39-NGBC --in_type 0 --out_type 2 --num_cage_v 640 --data_selection 0

        ######
        # model arch testing 
        python eval_CBD.py --version 8 --ckpt ./ckpts_CBD/2025-09-25-18-27-55-NGBCv8 --in_type 1 --out_type 1 --data_selection 0 --last_activation relu
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-26-10-16-32-NGBCv5 --in_type 1 --out_type 1 --data_selection 0 --last_activation relu
        
        ## experiment w/ pca test
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-27-12-04-29-NGBCv5 --in_type 1 --out_type 0 --data_selection 0 --last_activation relu
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-28-15-27-33-NGBCv5 --in_type 2 --out_type 0 --data_selection 0 --last_activation relu
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-27-07-44-28-NGBCv5 --in_type 2 --out_type 1 --data_selection 0 --last_activation relu        
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-27-08-05-11-NGBCv5 --in_type 1 --out_type 1 --data_selection 0 --last_activation relu
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-28-14-50-34-NGBCv5 --in_type 1 --out_type 1 --data_selection 0 --last_activation softplus            
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-29-00-17-41-NGBCv5 --in_type 2 --out_type 0 --data_selection 0 --last_activation none        
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-30-16-28-51-NGBCv5 --in_type 1 --out_type 1 --data_selection 0 --last_activation none

        ## experiment w/ real test
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-27-12-04-29-NGBCv5 --in_type 1 --out_type 0 --data_selection 0 --last_activation relu --realtest
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-28-15-27-33-NGBCv5 --in_type 2 --out_type 0 --data_selection 0 --last_activation relu --realtest
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-27-07-44-28-NGBCv5 --in_type 2 --out_type 1 --data_selection 0 --last_activation relu --realtest        
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-27-08-05-11-NGBCv5 --in_type 1 --out_type 1 --data_selection 0 --last_activation relu --realtest
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-28-14-50-34-NGBCv5 --in_type 1 --out_type 1 --data_selection 0 --last_activation softplus --realtest        
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-29-00-17-41-NGBCv5 --in_type 2 --out_type 0 --data_selection 0 --last_activation none --realtest
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-09-30-16-28-51-NGBCv5 --in_type 1 --out_type 1 --data_selection 0 --last_activation none --realtest
        python eval_CBD.py --version 8 --ckpt ./ckpts_CBD/2025-09-29-14-42-44-NGBCv8 --in_type 1 --out_type 1 --data_selection 0 --last_activation none --realtest

        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-10-02-00-06-28-NGBCv5 --in_type 1 --out_type 2 --data_selection 0 --last_activation 'relu' --realtest
        
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-10-03-15-28-36-NGBCv5 --in_type 1 --out_type 1 --data_selection 0 --last_activation 'relu' --realtest --no_pou
        
        python eval_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-10-03-22-40-03-NGBCv5 --in_type 1 --out_type 1 --data_selection 0 --last_activation 'relu' --realtest --no_pou

        python eval_CBD.py --version 8 --ckpt ./ckpts_CBD/2025-09-29-14-48-17-NGBCv8 --in_type 1 --out_type 1 --data_selection 0 --last_activation 'relu' --realtest
        python eval_CBD.py --version 8 --ckpt ./ckpts_CBD/2025-09-29-14-53-53-NGBCv8 --in_type 1 --out_type 1 --data_selection 0 --last_activation 'softplus' --realtest
        
        
        ## NFS
        python eval_CBD.py --version 0 --ckpt ./ckpt_stage1/2024-06-09-10-57-34-all --data_selection 0 --realtest
        python eval_CBD.py --version 0 --ckpt ./ckpt_stage1/2024-07-08-06-27-12-all --data_selection 0 --realtest
        
    """
    mp.set_start_method('spawn', force=True)
    
    # argparse configs
    opts = Options()
    
    # base configs (yaml)
    if opts.version==0:
        opts.config='config/train.yml'
        opts_yaml = yaml.load(open(opts.config), Loader=yaml.FullLoader)
    else:
        config = f'{opts.ckpt}/train_opts.yml'
        opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)
        
    # update with argparse configs
    opts_ = vars(opts)
    opts_yaml.update(opts_)
    opts = argparse.Namespace(**opts_yaml)
        
    if opts.version==0:
        opts.img_feat_dim=128        
        opts.feature_type="cents&norms"
        opts.stage1 = True
        opts.scale_exp=1.0
        opts.ict_face_only=False
        
        opts.NFR=False ## willbe using reimplemented model
    
        if opts.use_NFR:
            opts.design="nfr"
            opts.dec_type="jacob"
        else:
            opts.design="new2"
            opts.dec_type="disp"
            
    print('loaded version:', opts.version)
    trainer = Trainer(opts)
    if opts.realtest:
        if opts.use_eval_data2:
            trainer.evaluate3() ## nfs test dataloader
        else:
            trainer.evaluate2() ## real test frames
    else:
        trainer.evaluate() ## pca test data

