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
# __deep_cage_path__ = f'{__abs_path__}/third_party/deep_cage'

# for __util_path__ in [__abs_path__, __deep_cage_path__]:
#     if not __util_path__ in sys.path:
#         sys.path+=[__util_path__]


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataloader_CBD import (
    CBDdataSampler,
    CBDDataset,
    CBD_collate_wrapper,
    # CBDDataset2,
    # CBD_collate_wrapper2,
)

from utils.matplotlib_rnd import plot_image_array
from utils.ckpt_utils import *
from utils.remesh_utils import build_padded_neighbors, pca_normal_axis_vectorized
from utils.exp_utils import plateau_hat_points
from utils.mesh_utils import calc_norm_torch

from models.NGBCv2 import NeuralBarycentricCoordinatev2
from utils.loss_utils import *

# sys.path = list(set(sys.path))
def Options():
    parser = argparse.ArgumentParser(description='neural generalized barycentric coordinate for FA retargeting')
    parser.add_argument('-c', '--config', default='config/train_CBD.yml', help='config file path')
    parser.add_argument("--device",       type=str,   default="cuda:0")
    
    parser.add_argument("--log_dir",      type=str,   default="ckpts_CBD")

    parser.add_argument("--version",      type=int,   default=1,      help='train method (1: baseline, 2: ours)')
    parser.add_argument("--num_cage_v",   type=int,   default=1024,   help='number of cage vertices')
    
    parser.add_argument("--in_type",      type=int,   default=1,
                        help='input type (0: position, 1: position + normal')
    parser.add_argument("--out_type",      type=int,   default=1,      
                        help='output type (0: cage v, 1: cage delta_v, 2: cage delta_T mat, 3: vertex T mat')
    
    parser.add_argument("--save_interval",type=int,   default=25,     help='save interval epoch')
    parser.add_argument("--max_epoch",    type=int,   default=500,    help='number of epochs')
    parser.add_argument("--start_epoch",  type=int,   default=0,      help='number of epochs')
    parser.add_argument("--lr",           type=float, default=0.0001, help='learning rate')
    parser.add_argument("--sc_step",      type=int,   default=100,    help='scheduler step')
    
    parser.add_argument("--batch_size",   type=int,   default=8,      help='batch size')
    parser.add_argument("--seed",         type=int,   default=42,     help='random seed')
    parser.add_argument("--ckpt",         type=str,   default=None)    
    parser.add_argument("--continue_ckpt",dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)
    
    parser.add_argument("--n_sampling",   dest='n_sampling', action='store_true')
    parser.set_defaults(n_sampling=False)
    
    parser.add_argument("--use_decimate", dest='use_decimate', action='store_true')
    parser.set_defaults(use_decimate=False)

    #### Choose a last layer activation for key_weight_model()
    parser.add_argument("--last_activation", choices=["relu", "elu", "softmax", "softplus", "none", "sqrelu"],
        help="Choose a last layer activation for NGBC.key_weight_model()"
    )
    
    parser.add_argument("--no_pou",dest='no_pou', action='store_true')
    parser.set_defaults(no_pou=False)
    
    parser.add_argument("--pou_loss",dest='pou_loss', action='store_true')
    parser.set_defaults(pou_loss=False)
    
    parser.add_argument("--use_scheduler",dest='use_scheduler', action='store_true')
    parser.set_defaults(use_scheduler=False)
    
    parser.add_argument("--use_segment_weight",dest='use_segment_weight', action='store_true')
    parser.set_defaults(use_segment_weight=False)
    parser.add_argument("--use_laplacian",dest='use_laplacian', action='store_true')
    parser.set_defaults(use_laplacian=False)
    parser.add_argument("--use_normal_loss",dest='use_normal_loss', action='store_true')
    parser.set_defaults(use_normal_loss=False)

    
    parser.add_argument("--no_t_mask",dest='no_t_mask', action='store_true')
    parser.set_defaults(no_t_mask=False)
    
    parser.add_argument("--use_data0",dest='use_data0', action='store_true')
    parser.set_defaults(use_data0=False)
    parser.add_argument("--use_data1",dest='use_data1', action='store_true')
    parser.set_defaults(use_data1=False)
    parser.add_argument("--use_data2",dest='use_data2', action='store_true')
    parser.set_defaults(use_data2=False)
    parser.add_argument("--use_data3",dest='use_data3', action='store_true')
    parser.set_defaults(use_data3=False)
    parser.add_argument("--use_data9",dest='use_data9', action='store_true')
    parser.set_defaults(use_data9=False)
    parser.add_argument("--data_toggle",dest='data_toggle', action='store_true')
    parser.set_defaults(data_toggle=False)

    parser.add_argument("--tb",           action='store_true')
    parser.set_defaults(is_train=True)
    
    parser.add_argument("--optim_cage",dest='optim_cage', action='store_true')
    parser.set_defaults(optim_cage=False)
    
    parser.add_argument("--align_latent",dest='align_latent', action='store_true')
    parser.set_defaults(align_latent=False)
    
    args = parser.parse_args()
    return args



class Logger():
    def __init__(self, file_path):
        self.file_path = file_path
        
    def write(self, txt):
        with open(self.file_path, 'a') as f:
            f.write(txt)
        f.close()

class Trainer():
    def __init__(self, opts):
        # set opts
        self.opts = opts
        self.set_seed(self.opts)
        self.device = opts.device

        last_act_list = ["relu", "elu", "softmax", "softplus", "none", "sqrelu"]
        last_act_list = [self.opts.last_activation==l_act for l_act in last_act_list]
        
        self.model = NeuralBarycentricCoordinatev2(
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
            use_sqrelu=last_act_list[5],
            use_least_N_on_V=False,
            is_train=True,
            use_pou = ~self.opts.no_pou,
            device=self.device,
            hid_dim=128 if self.opts.align_latent else 256,
        )
            
        # load weight
        self.load_weight()
    
    def load_weight(self):
        if self.opts.ckpt:
            print(f"Loading... {self.opts.ckpt}")
            if self.opts.continue_ckpt:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, f"*_{self.opts.start_epoch:03d}.pth"))[0]
            else:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
            ckpt_dict = torch.load(ckpt)            
            self.model.load_state_dict(ckpt_dict)
            print(f"Loaded! {ckpt}")
        else:
            print('no ckpt found, training from scratch!')
            
    def train_v5(self, epochs):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.opts.lr,
            betas=(0.9, 0.999)
        )
        
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, 
        #     milestones=[i for i in range(0, epochs-1, self.opts.sc_step)], 
        #     gamma=self.opts.sc_gamma
        # )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.opts.sc_step,
            gamma=self.opts.sc_gamma
        )
            
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        BS = self.opts.batch_size
        BS_denom = 1 / BS
        
        self.train_dataset = CBDDataset(
            self.opts, is_train=True, toggle=self.opts.data_toggle
        )
        self.valid_dataset = CBDDataset(
            self.opts, is_valid=True, toggle=self.opts.data_toggle
        )
        
        train_sampler = CBDdataSampler(
            self.train_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_train=True
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_sampler=train_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0,
        )        
            
        valid_sampler = CBDdataSampler(
            self.valid_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            is_valid=True
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset, 
            batch_sampler=valid_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0
        )
        ##########################################################################################################
        
        
        ###### Logging ###########################################################################################
        # make logdir --------------------------------------------------------------------------------------------
        os.makedirs(self.opts.log_dir, exist_ok=True)
        import datetime
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        
        tag = f"-NGBCv{self.opts.version}"
        if self.opts.optim_cage:
            tag += "-optim_cage"
        self.opts.log_dir = os.path.join(self.opts.log_dir, now+tag)
        os.makedirs(self.opts.log_dir, exist_ok=True)

        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/valid/mesh", exist_ok=True)
        
        # save options as json -----------------------------------------------------------------------------------
        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
            
        # save train option as yml
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), opts)
        
        if self.opts.tb:
            train_ = os.path.join(self.opts.log_dir, "train")
            valid_ = os.path.join(self.opts.log_dir, "valid")
            self.writer_train = SummaryWriter(log_dir=train_)
            self.writer_valid = SummaryWriter(log_dir=valid_)
        
        # self logger
        self.logger = Logger(os.path.join(self.opts.log_dir, "log.txt"))
        print(f'Saving log at: {self.logger.file_path}')
        
        print(self.train_dataset.get_data_config())
        print(train_sampler.get_sampler_config())
        print(self.valid_dataset.get_data_config())
        print(valid_sampler.get_sampler_config())
        
        self.logger.write(self.train_dataset.get_data_config())
        self.logger.write(train_sampler.get_sampler_config())  
        self.logger.write(self.valid_dataset.get_data_config())      
        self.logger.write(valid_sampler.get_sampler_config())
        #---------------------------------------------------------------------------------------------------------
        ##########################################################################################################
        
        
        
        # training loop ##########################################################################################
        global_step = 0
        BEST_LOSS = 100_000_000
        BEST_EPOCH = 0
        start_epoch = self.opts.start_epoch
                        
        # define loss lamdba 
        self.loss_lambda = {
            "recon-def": self.opts.lambda_vert,
            "recon-neu": self.opts.lambda_vert,
            "exp-z": self.opts.lambda_vert * 0.5,
            "exp-v": self.opts.lambda_vert,
            "shape": self.opts.lambda_vert,
            "lag": self.opts.lambda_vert,
            "dist": self.opts.lambda_vert,
            # "pou": self.opts.lambda_vert,
            # symm 
        }
        if self.opts.pou_loss:
            self.loss_lambda['pou'] = 1.0
        if self.opts.use_laplacian:
            self.loss_lambda['lap'] = 1.0
        if self.opts.use_normal_loss:
            self.loss_lambda['norm-def']=0.1
            self.loss_lambda['norm-neu']=0.1
        
        check_usage = False
        
        len_train_data = len(self.train_dataloader)
        len_valid_data = len(self.valid_dataloader)
        interv_train = round(len_train_data / 10)
        
        for epoch in range(start_epoch, epochs+1):
            print(f"[{epoch:03d}/{epochs:03d}][Train]")
            
            ## for logging loss!
            running_losses = {
                "recon-def": 0.0,
                "recon-neu": 0.0,
                "exp-z": 0.0,
                "exp-v": 0.0,
                "shape": 0.0,
                "lag": 0.0,
                "dist": 0.0,
                "total": 0.0
            }
            if self.opts.pou_loss:
                running_losses['pou']=0.0
            if self.opts.use_laplacian:
                running_losses['lap']=0.0
            if self.opts.use_normal_loss:
                running_losses['norm-def']=0.0
                running_losses['norm-neu']=0.0
            
            self.model.train()
            train_counter = 0
            
            is_stepped = False
            is_stts_added = False
            
            pbar = tqdm(enumerate(self.train_dataloader), total=len_train_data, position=0, ncols=100)
            for index, batch in pbar:
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                    ## sampling points with probability
                    # margin = 0.8
                    # _p = (plateau_hat_points(batch.template[0]).squeeze() + margin) / (1 + margin)
                        
                    ## random sampling and random permutation
                    N = batch.template.shape[1]
                    use_perm = torch.rand(1) > 0.3
                    # use_perm= False
                    if use_perm:
                        N_range = N-torch.randint(100, N//6, (1,)).item()
                        randperm_idx = torch.randperm(N)[:N_range]
                    else:
                        randperm_idx = torch.arange(N)
                        # randperm_idx = torch.multinomial(_p, 2048)
                    rearange_idx = torch.argsort(randperm_idx)
                    
                    batch_template_v = batch.template[:, randperm_idx]
                    batch_template_n = batch.template_normal[:, randperm_idx]
                    batch_vertices_v = batch.vertices[:, randperm_idx]
                    batch_vertices_n = batch.vertices_normal[:, randperm_idx]

                    ## masking face region using hat function (min x1 ~ max x2)
                    # t_mask = plateau_hat_points(batch_template_v) + 1.0
                    
                # model prediction -------------------------------------------------------------------------------
                ## B: number of batch, Nv : number of vertices, Nc: number of control vertices
                ## weight prediction: (B, Nv, Nc)
                ## key_d prediction:  (B, Nc, 3+3) [deformed cage]
                pred_vertices, recon_vertices, recon_source, exp_z, \
                pred_source, t_mask, pred_key_weight, pred_cage_w, pred_cage_s = self.model(
                    batch_template_v, batch_vertices_v, batch_template_n, batch_vertices_n,
                    batch.mesh_data, epoch=epoch
                )

                # import pdb;pdb.set_trace()
                # vis_mask_plot(batch_template_v[0].detach().cpu(), t_mask[0].detach().cpu(), logdir='./', name='test')
                # vis_mask_plot(batch_template_v[0].detach().cpu(), inv_t_mask[0].detach().cpu(), logdir='./', name='test')
                if self.opts.no_t_mask:
                    t_mask = 1.0
                    inv_t_mask = 0.0
                else:
                    inv_t_mask = 1.0 - t_mask
                
                ## use segmentation for loss weight
                ## -> re-weighting based on facial region area
                if self.opts.use_segment_weight:
                    with torch.no_grad():
                        # batch.segmentation # (B, Nv, 24)
                        segment_weight = batch.segmentation.sum(1) / N # batch.segmentation.sum(1).sum(1) # (B, 24)
                        batch_segment_weight = (batch.segmentation * segment_weight[:, None])[:, randperm_idx]
                        t_mask = t_mask * batch_segment_weight
                # ------------------------------------------------------------------------------------------------
                
                # loss -------------------------------------------------------------------------------------------
                mesh_data_num = batch.mesh_data.cpu().numpy()
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf', 'ict'])[mesh_data_num]
                
                loss_dict = {} # make it as a dictionary
                HB = batch.vertices.shape[0] // 2
                
                if self.opts.no_t_mask:
                    loss_dict['recon-def'] = F.mse_loss(batch_vertices_v, pred_vertices) ## focus on face
                else:
                    loss_dict['recon-def'] = F.mse_loss(batch_vertices_v*t_mask, pred_vertices*t_mask) ## focus on face
                    loss_dict['recon-def'] += F.mse_loss(batch_template_v*inv_t_mask, pred_vertices*inv_t_mask) # static on elsewhere

                ###### Lagrange property ####################################
                if self.opts.out_type==3:
                    loss_dict['lag'] = F.mse_loss(
                        torch.eye(pred_cage_w.shape[-1]).to(self.device)[None].repeat(BS, 1, 1),
                        pred_cage_w, 
                    )
                #############################################################

                ###### distance loss ########################################
                loss_dict['dist'] = distance_loss(
                    batch_template_v, pred_cage_s, pred_key_weight
                )
                #############################################################
                
                if self.model.use_full_vertex:
                    if self.opts.no_t_mask:
                        loss_dict['recon-neu'] = F.mse_loss(batch_template_v, pred_source)
                    else:
                        loss_dict['recon-neu'] = F.mse_loss(batch_template_v*t_mask, pred_source*t_mask)
                        loss_dict['recon-neu'] += F.mse_loss(batch_template_v*inv_t_mask, pred_source*inv_t_mask)
                
                if self.model.use_shp_recon:
                    loss_dict['shape'] = F.mse_loss(
                        batch_template_v[:,rearange_idx]*t_mask,
                        recon_source[:,randperm_idx[rearange_idx]]*t_mask
                    ) # for shape AE
                if self.model.use_exp_recon:
                    loss_dict['exp-v'] = F.mse_loss(
                        batch_vertices_v[:,rearange_idx]*t_mask,
                        recon_vertices[:,randperm_idx[rearange_idx]]*t_mask
                    ) # for expression AE
                
                if self.opts.pou_loss:
                    pred_key_weight_sum = pred_key_weight.sum(-1)
                    loss_dict['pou'] = F.mse_loss(
                        torch.ones_like(pred_key_weight_sum).to(self.device),
                        pred_key_weight_sum, 
                    )
                    
                if not use_perm and self.opts.use_laplacian:
                    loss_dict['lap'] = laplacian_loss(
                        batch, pred_key_weight, self.train_dataset, mesh_data_num, self.device
                    ) * BS_denom
                    
                    # with torch.no_grad():
                    #     batch_vertices_lap = cotmatrix @ batch_vertices_v
                    #     batch_template_lap = cotmatrix @ batch_template_v
                    # pred_vertices_lap = cotmatrix @ pred_vertices
                    # pred_source_lap = cotmatrix @ pred_source
                    
                    # loss_dict['pois'] += F.mse_loss(batch_vertices_lap, pred_vertices_lap)
                    # loss_dict['pois'] += F.mse_loss(batch_vertices_lap, pred_vertices_lap)
                
                if not use_perm and self.opts.use_normal_loss:
                    pred_vertices_norm = calc_norm_torch(pred_vertices, batch.faces, at='verts') # [1, V, 3]
                    
                    loss_dict['norm-def'] = (
                        F.mse_loss(
                            batch_vertices_n*t_mask, pred_vertices_norm*t_mask
                        ) + F.mse_loss(
                            batch_template_n*inv_t_mask, pred_vertices_norm*inv_t_mask
                        )
                    )
                    
                    if self.model.use_full_vertex:
                        pred_template_norm = calc_norm_torch(pred_source, batch.faces, at='verts')   # [1, V, 3]
                        loss_dict['norm-neu'] = F.mse_loss(
                            batch_template_n, pred_template_norm
                        )
                
                #if (self.opts.use_data2 or self.opts.use_data3):
                if self.opts.align_latent:
                    if mesh_data=='ict':
                        loss_dict['exp-z'] = F.mse_loss(
                            batch.exp_coeff.unsqueeze(1), exp_z
                        )
                    else:
                        exp_z_facs, exp_z_ext = exp_z[...,:53], exp_z[...,53:]
                        loss_dict['exp-z'] = non_ict_loss(exp_z_facs)
                        loss_dict['exp-z'] += F.mse_loss(
                            torch.zeros_like(exp_z_ext).to(self.device),
                            exp_z_ext
                        )
                        # pass

                
                # get total loss (lambda weights are multiplied here!)
                loss = 0
                for key, value in loss_dict.items():
                    key_ = key.split("_")[0]
                    tmp = value*self.loss_lambda[key_]
                    loss += tmp
                    running_losses[key] += tmp # for logging
                loss_dict["total"] = loss
                # ------------------------------------------------------------------------------------------------

                
                # backward ---------------------------------------------------------------------------------------
                loss.backward()
                # try:
                #     loss.backward()
                # except:
                #     import pdb;pdb.set_trace()
                self.optimizer.step()
                # ------------------------------------------------------------------------------------------------

                
                # running loss for logging -----------------------------------------------------------------------
                running_losses["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5e}, mesh data: {mesh_data_num}")
                # ------------------------------------------------------------------------------------------------

                
                global_step += 1
                train_counter += 1
                                
                
                if index % interv_train == 1:

                    IDX = torch.tensor([0, 1, HB, BS-1])
                    with torch.no_grad():
                        #pred_vertices, recon_vertices, recon_source, exp_z, pred_source = self.model(
                        pred_vertices, _, _, exp_z, key_d, key_weight, _, _ = self.model(
                            batch.template[IDX], batch.vertices[IDX],
                            batch.template_normal[IDX], batch.vertices_normal[IDX],
                            batch.mesh_data, epoch=epoch, out_kw=True
                        )
                    # for visualization
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                    
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Train] "
                    __idx__ = 1/train_counter
                    for key, value in running_losses.items():
                        log_text += f"{key}: {value*__idx__:.6e} "

                    if not is_stts_added:
                        log_text+='\n>>> Sum across vertex weights on each cage: '
                        log_text+=f'(max: {key_weight[0].sum(0).max().item():.5e}, min: {key_weight[0].sum(0).min().item():.5e})\n'
                        log_text+=f'>>> Num actually used cage vertex: {torch.count_nonzero(key_weight[0].sum(0))} / {key_weight.shape[-1]}'
                        is_stts_added=True
                    self.logger.write(log_text+"\n")
                    
                    frame = HB
                    v_list = [
                        vertices[0].cpu().detach(),
                        vertices[1].cpu().detach(),
                        vertices[HB].cpu().detach(),
                        vertices[BS-1].cpu().detach(),
                        pred_vertices[0].cpu().detach(),
                        pred_vertices[1].cpu().detach(),
                        pred_vertices[2].cpu().detach(),
                        pred_vertices[3].cpu().detach(),
                    ]
                    
                    len_v = len(v_list)
                    f_list = [faces] * len_v
                    save_logdir = f"{self.opts.log_dir}/img/train/mesh"
                    save_img_name = f"{epoch:03d}_{index:04d}"

                    plot_image_array(
                        v_list, f_list, 
                        rot_list=[[0,0,0]] * len_v, 
                        size=1, bg_black=False, mode='shade',
                        logdir=save_logdir,
                        name=save_img_name, save=True
                    )
                
                if self.opts.debug:
                    break
                # ------------------------------------------------------------------------------------------------
            
            ### scheduler ----------------------------------------------------------------------------------------
            if epoch != 0:
                self.scheduler.step()
                
                curr_lr = self.optimizer.param_groups[0]["lr"]
                if epoch % self.opts.sc_step==0 and not is_stepped:
                    log_notice = f'[{epoch:03d}/{epochs:03d}][{index:04d}][Train] scheduler stepped: {curr_lr:.6e}'
                    self.logger.write(log_notice+"\n")
                    is_stepped=True
            # ----------------------------------------------------------------------------------------------------
                
            # log
            if self.opts.tb:
                self.log_loss(self.writer_train, running_losses, epoch, train_counter)

            # save model
            if epoch % self.opts.save_interval == 0:
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_{epoch:03d}.pth')
            
            
            # validation -----------------------------------------------------------------------------------------
            self.model.eval()
            print(f"[{epoch:03d}/{epochs:03d}][Valid]")
            running_losses_val = {
                "recon-def": 0.0,
                "recon-neu": 0.0,
                "exp-z": 0.0,
                "exp-v": 0.0,
                "shape": 0.0,
                "total": 0.0
            }
            
            counter = 0
            pbar = tqdm(enumerate(self.valid_dataloader), total=len_valid_data, ncols=100)
            for index, batch in pbar:
                counter += 1
                
                # model validation -------------------------------------------------------------------------------
                with torch.no_grad():
                    pred_vertices, recon_vertices, recon_source, exp_z, \
                    pred_source, _, _, _, _ = self.model(
                        batch.template, batch.vertices, 
                        batch.template_normal, batch.vertices_normal,
                        batch.mesh_data, epoch=epoch
                    )
                # ------------------------------------------------------------------------------------------------
                
                
                # loss ------------------------------------------------------------------------------------------- 
                with torch.no_grad():
                    mesh_data_num = batch.mesh_data.cpu().numpy()
                    mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf','ict'])[mesh_data_num]
                
                    loss_dict = {} # make it as a dictionary
                    HB = batch.vertices.shape[0] // 2
                    
                    loss_dict['recon-def'] = F.mse_loss(batch.vertices, pred_vertices) # for NGBC model
                    if self.model.use_full_vertex:
                        loss_dict['recon-neu'] = F.mse_loss(batch.template, pred_source)
                        
                    if self.model.use_shp_recon:
                        loss_dict['shape'] = F.mse_loss(batch.template, recon_source) # for shape AE
                    if self.model.use_exp_recon:
                        loss_dict['exp-v'] = F.mse_loss(batch.vertices, recon_vertices) # for expression AE
                    # loss_dict['exp-z'] = F.mse_loss(exp_z[:HB], exp_z[HB:])
                    
                    # get total loss
                    loss = 0
                    for key, value in loss_dict.items():
                        key_ = key.split("_")[0]
                        tmp = value.item()*self.loss_lambda[key_]
                        loss += tmp
                        running_losses_val[key] += tmp
                    loss_dict["total"] = loss 

                # running loss for logging
                running_losses_val["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5e}, mesh data: {mesh_data_num}")
                # ------------------------------------------------------------------------------------------------
            
                
                # ------------------------------------------------------------------------------------------------
                interv_val = round(len_valid_data / 5)
                if index % interv_val == 0:
                    # for visualization
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Valid] "
                    __jdx__ = 1/counter
                    for key, value in running_losses_val.items():
                        log_text += f"{key}: {value*__jdx__:.6e} "
                    self.logger.write(log_text+"\n")
                    
                    frame = HB
                    v_list = [
                        vertices[0].cpu().detach(),
                        vertices[1].cpu().detach(),
                        vertices[HB].cpu().detach(),
                        vertices[BS-1].cpu().detach(),
                        pred_vertices[0].cpu().detach(),
                        pred_vertices[1].cpu().detach(),
                        pred_vertices[HB].cpu().detach(),
                        pred_vertices[BS-1].cpu().detach(),
                    ]
                    len_v = len(v_list)
                    f_list=[faces] * len_v
                    save_logdir = f"{self.opts.log_dir}/img/valid/mesh"
                    save_img_name = f"{epoch:03d}_{counter:04d}"
                    
                    plot_image_array(
                        v_list, f_list, 
                        rot_list=[[0,0,0]]*len_v,
                        size=1, bg_black=False, mode='shade', 
                        logdir=save_logdir, 
                        name=save_img_name, save=True
                    )
                    
                # ------------------------------------------------------------------------------------------------
                if self.opts.debug:
                    break
            # log
            if self.opts.tb:
                self.log_loss(self.writer_valid, running_losses_val, epoch, counter)
            
            # best loss
            val_loss = running_losses_val["total"]/counter
            if val_loss < BEST_LOSS:
                BEST_LOSS = val_loss
                BEST_EPOCH = epoch
                print(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6e} - Best epoch: {BEST_EPOCH:03d}\n")
                self.logger.write(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6e}\n")
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_best.pth')
            else:
                self.logger.write(f"[{epoch:03d}/{epochs:03d}] Curr Loss: {val_loss:.6e} (Best Loss: {BEST_LOSS:.6e} [{BEST_EPOCH:03d}])\n")
                print(f"[{epoch:03d}/{epochs:03d}] Curr Loss: {val_loss:.6e} (Best Loss: {BEST_LOSS:.6e} [{BEST_EPOCH:03d}])\n")
    
    
    
    @staticmethod
    def log_loss(writer, loss_dict, step, counter=None):
        if counter:
            N = counter
        else:
            N = 1
        for key, value in loss_dict.items():
            writer.add_scalar(f'{key}', value/N, step)

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


def vis_mask_plot(XX, WW, logdir='./', name='test'):
    #import torch
    #import numpy as np
    import matplotlib.pyplot as plt
    #from utils.remesh_utils import ICT_face_model
    
    fig1 = plt.figure(figsize=(14, 8))
    ax3 = fig1.add_axes([0.3, 0.0, 0.5, 0.5], projection='3d') ## left bottom W H
    
    # Use default colormap; do not specify colors explicitly
    p = ax3.scatter(
        XX[:, 2].numpy(),
        XX[:, 0].numpy(),
        XX[:, 1].numpy(),
        c=WW.numpy(),
        s=5,
        depthshade=False)
    
    ax3.set_title("mask", y=-0.21, fontsize=22)
    cb = fig1.colorbar(p, ax=ax3, shrink=0.75)
    
    # Equal aspect
    try:
        ax3.set_box_aspect([1, 1, 1])
    except Exception:
        # Fallback for older matplotlib: approximate equal aspect
        xyzlim = np.array([ax3.get_xlim3d(), ax3.get_ylim3d(), ax3.get_zlim3d()])
        xyzmin = xyzlim[:, 0].min()
        xyzmax = xyzlim[:, 1].max()
        ax3.set_xlim3d([xyzmin, xyzmax])
        ax3.set_ylim3d([xyzmin, xyzmax])
        ax3.set_zlim3d([xyzmin, xyzmax])
    plt.savefig('{}/{}.png'.format(logdir, name), bbox_inches = 'tight')
    # plt.show()
    plt.close(plt.gcf())
    
if __name__ == "__main__":
    # argparse configs
    opts = Options()
    
    # base configs (yaml)
    opts_yaml = yaml.load(open(opts.config), Loader=yaml.FullLoader)
        
    # update with argparse configs
    opts_ = vars(opts)
    opts_yaml.update(opts_)
    opts = argparse.Namespace(**opts_yaml)
    
    trainer = Trainer(opts)
    
    trainer.train_v5(epochs=opts.max_epoch)
    

