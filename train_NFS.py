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

# tensorboard --logdir ./ckpt_nfs --port 6789
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

from utils.matplotlib_rnd import plot_image_array, plot_image_array_seg, vis_rig
from utils.ckpt_utils import *
from utils.remesh_utils import build_padded_neighbors, pca_normal_axis_vectorized
from utils.exp_utils import plateau_hat_points
from utils.mesh_utils import calc_norm_torch

from models.baseline import CageNet
from models.NGBC import NeuralGeneralizedBarycentricCoordinate
from utils.loss_utils import *


# sys.path = list(set(sys.path))
def Options():
    parser = argparse.ArgumentParser(description='neural generalized barycentric coordinate for FA retargeting')
    parser.add_argument('-c', '--config', default='config/train_NFS.yml', help='config file path')
    parser.add_argument("--device",       type=str,   default="cuda:0")
    
    parser.add_argument("--log_dir",      type=str,   default="ckpts_CBD")

    parser.add_argument("--version",      type=int,   default=1,      help='train method (1: baseline, 2: ours)')
    parser.add_argument("--num_cage_v",   type=int,   default=1024,   help='number of cage vertices')
    
    parser.add_argument("--in_type",      type=int,   default=1,
                        help='input type (0: position, 1: position + normal')
    parser.add_argument("--out_type",      type=int,   default=1,      
                        help='output type (0: cage v, 1: cage delta_v, 2: cage delta_T mat, 3: vertex T mat')
    
    parser.add_argument("--save_interval",type=int,   default=10,     help='save interval epoch')
    parser.add_argument("--max_epoch",    type=int,   default=500,    help='number of epochs')
    parser.add_argument("--start_epoch",  type=int,   default=0,      help='number of epochs')
    parser.add_argument("--lr",           type=float, default=0.0002, help='learning rate')
    parser.add_argument("--sc_step",      type=int,   default=10,     help='scheduler step')
    
    parser.add_argument("--batch_size",   type=int,   default=8,      help='batch size')
    parser.add_argument("--seed",         type=int,   default=42,     help='random seed')
    parser.add_argument("--ckpt",         type=str,   default=None)

    parser.add_argument("--dec_type",     type=str,   default="disp", help="vert, disp, jacob")
    parser.add_argument("--design",       type=str,   default="new2", help="nfr, new2")
    
    parser.add_argument("--selection",    type=int, default=20, help='dataset selection')
    parser.add_argument("--warmup",       dest='warmup',       action='store_true')
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
    
    parser.add_argument("--use_dist_loss",dest='use_dist_loss', action='store_true')
    parser.set_defaults(use_dist_loss=False)
    parser.add_argument("--use_segment_weight",dest='use_segment_weight', action='store_true')
    parser.set_defaults(use_segment_weight=False)
    parser.add_argument("--use_laplacian",dest='use_laplacian', action='store_true')
    parser.set_defaults(use_laplacian=False)
    parser.add_argument("--use_normal_loss",dest='use_normal_loss', action='store_true')
    parser.set_defaults(use_normal_loss=False)

    parser.add_argument("--ict_face_only",action='store_true', help="if True, use face region only")
    
    ## training stages (dummy)
    parser.add_argument("--mesh_d",       action='store_true', help="train mesh decoder")
    parser.add_argument("--stage1",       action='store_true', help="train stage1")
    parser.add_argument("--stage11",      action='store_true', help="train stage11")

    
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
    
    parser.set_defaults(is_train=True)
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
        
        if opts.version==1:
            self.model = CageNet(device=self.device, optim_cage=self.opts.optim_cage)
            
        elif opts.version==22:
            from models.NFS import NFS
            self.model = NFS(self.opts, None).to(self.device)
            
        elif opts.version==5:
            self.model = NeuralGeneralizedBarycentricCoordinate(
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
        if opts.version==22:
            self.load_weight_NFS()
        else:
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

    def load_weight_NFS(self):
        if self.opts.ckpt:
            if self.opts.continue_ckpt:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, f"*_{self.opts.start_epoch:03d}.pth"))[0]
            else:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
            print(f"Loading... {ckpt}")
            ckpt_dict = torch.load(ckpt)
            
            ## remove remaining precomputes in DiffusionNet
            del_key_list=[
                'mass', 'L_ind', 'L_val', 'evals', 'evecs',
                'grad_X', 'grad_Y', 'faces', 'audio_encoder'
            ]
            ckpt_dict = del_key(ckpt_dict, del_key_list)
            self.model.load_state_dict(ckpt_dict,strict=False)
        
    def train_v6(self, epochs):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.opts.lr,
            betas=(0.9, 0.999)
        )
            
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        BS = self.opts.batch_size
        BS_denom = 1 / BS
        
        self.train_dataset = CBDDataset(
            self.opts, is_train=True
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

        os.makedirs(f"{self.opts.log_dir}/img/train/rig", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/valid/rig", exist_ok=True)
        
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
        
        
        ## model info
        self.logger.write(self.model.log_parameter_num())
        
        ## dataset info
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
            "recon": self.opts.lambda_vert,
            "vert": self.opts.lambda_loss,
            "norm":  self.opts.lambda_normal,
            "jacob": self.opts.lambda_jacob,
            "temp":  self.opts.lambda_temp,
            "nll":   self.opts.lambda_seg,
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
        interv_val = round(len_valid_data / 10)

        th_eye = torch.eye(self.opts.seg_dim, self.opts.seg_dim)
        gt_seg = th_eye[self.model.ict_vert_segment.cpu().detach()]
        
        for epoch in range(start_epoch, epochs+1):
            print(f"[{epoch:03d}/{epochs:03d}][Train]")
            
            ## for logging loss!
            running_losses = {
                "recon_vDec": 0,
                "vert_rEEnc": 0,
                "vert_rIEnc": 0,
                "vert_vICT":0,
                "vert_rot":0,
                "norm_vDec":0,
                "jacob_vDec":0,
                "nll_vSeg":0,
                "total": 0
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
                                
                mesh_data_num = batch.mesh_data.cpu().numpy()
                B=batch.vertices.shape[0]
                #     0       1      2     3       4     5
                # ['voca', 'biwi', 'mf', 'voca', 'mf', 'ict']
                if mesh_data_num == 2 or mesh_data_num == 4:
                    batch.mesh_data = 3 # remap
                    precompute_path = self.train_dataset.mf_precompute_path
                elif mesh_data_num == 5:
                    batch.mesh_data = 0 # remap
                    
                    if np.random.random(1) > 0.4:
                        if np.random.random(1) > 0.5:
                            r_num = 1
                        else:
                            r_num = 2
                    else:
                        r_num = 0
                        
                    v_idx, quad_f_idx = self.model.ict_face_model.region[r_num]
                    batch.faces = self.model.ict_face_model.quad_Faces[:quad_f_idx, [[0, 1, 2],[0, 2, 3]] ].permute(
                        1, 0, 2
                    ).reshape(-1, 3).to(self.device)
                    
                    if r_num == 0:
                        precompute_path = self.train_dataset.ict_synth_precompute_fh
                    elif r_num == 1:
                        precompute_path = self.train_dataset.ict_synth_precompute_fo
                    else:
                        precompute_path = self.train_dataset.ict_synth_precompute_nf
                    # import pdb;pdb.set_trace()
                    
                    batch.template = batch.template[:, :v_idx]
                    batch.vertices = batch.vertices[:, :v_idx]
                    batch.vertices_normal = batch.vertices_normal[:, :v_idx]
                    
                batch.gt_rig_params = batch.exp_coeff
                batch.normals = batch.vertices_normal
                batch.dfn_info = os.path.join(
                    precompute_path, f"{batch.id_name}_dfn_info.pkl"
                )
                batch.img = torch.from_numpy(np.load(os.path.join(
                    precompute_path, f"{batch.id_name}_img.npy"
                ))).repeat(B,1,1,1).to(self.device)
                batch.operators=''
                
                
                # model prediction -------------------------------------------------------------------------------
                loss_dict, pred_vertices, _, pred_exp, pred_id, pred_seg = self.model(
                    batch, return_all=True, stage=1, epoch=epoch
                )
                # ------------------------------------------------------------------------------------------------
                
                # loss -------------------------------------------------------------------------------------------
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
                self.optimizer.step()
                # ------------------------------------------------------------------------------------------------

                
                # running loss for logging -----------------------------------------------------------------------
                running_losses["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5e}, mesh data: {mesh_data_num}")
                # ------------------------------------------------------------------------------------------------

                
                global_step += 1
                train_counter += 1
                                
                
                if index % interv_train == 1:
                    mesh_data = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data]
                    
                    gt_rig = batch.gt_rig_params.cpu()
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                    
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Train] "
                    #for key, value in running_losses.items():
                    for key, value in loss_dict.items():
                        log_text += f"{key}: {value:.6f} "
                    self.logger.write(log_text+"\n")
                    
                    frame = B//2
                    v_list = [ v for v in vertices[frame:frame+2] ] + \
                        [ v for v in pred_vertices[frame:frame+2].cpu().detach() ]
                    len_v = len(v_list)
                    f_list = [faces] * len_v
                    
                    save_logdir = f"{self.opts.log_dir}/img/train/mesh"
                    save_img_name = f"{epoch:03d}_{index:04d}"
                    if pred_seg is not None:
                        pred_seg=pred_seg.squeeze(0).cpu().detach()
                        
                        if mesh_data == 'ict':
                            c_list=[gt_seg]*2+[seg for seg in pred_seg[frame:frame+2]]
                        else:
                            c_list=[seg for seg in pred_seg[frame:frame+2]] * 2
                        
                        plot_image_array_seg(
                            v_list, f_list, c_list, 
                            rot_list=[[0,0,0]] * len_v, 
                            size=1, bg_black=False, mode='shade', 
                            logdir=save_logdir, 
                            name=save_img_name, save=True
                        )
                    else:
                        plot_image_array(
                            v_list, f_list, 
                            rot_list=[[0,0,0]] * len_v, 
                            size=1, bg_black=False, mode='shade',
                            logdir=save_logdir,
                            name=save_img_name, save=True
                        )
                    if mesh_data == 'ict':
                        if pred_exp is not None:
                            vis_rig(
                                torch.cat([pred_exp.cpu().detach()[None], gt_rig.squeeze()[None]], dim=0), 
                                f"{self.opts.log_dir}/img/train/rig/{epoch:03d}_{index:04d}.jpg",
                                normalize=True
                            )
                            
                if self.opts.debug:
                    break
                # ------------------------------------------------------------------------------------------------
                            
            # log
            if self.opts.tb:
                self.log_loss(self.writer_train, running_losses, epoch, train_counter)

            # save model
            if epoch % self.opts.save_interval == 0:
                self.model.mesh_id_encoder.empty_precomputes()
                self.model.mesh_exp_encoder.empty_precomputes()
                self.model.mesh_seg_encoder.empty_precomputes()
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_{epoch:03d}.pth')
            
            
            # validation -----------------------------------------------------------------------------------------
            self.model.eval()
            print(f"[{epoch:03d}/{epochs:03d}][Valid]")
            running_losses_val = {
                "recon_vDec": 0,
                "vert_rEEnc": 0,
                "vert_rIEnc": 0,
                "vert_vICT": 0,
                "vert_rot": 0,
                "norm_vDec": 0,
                "jacob_vDec": 0,
                "nll_vSeg": 0,
                "total": 0
            }
            
            counter = 0
            pbar = tqdm(enumerate(self.valid_dataloader), total=len_valid_data, ncols=100)
            for index, batch in pbar:
                counter += 1
                
                # model validation -------------------------------------------------------------------------------
                with torch.no_grad():
                    mesh_data_num = batch.mesh_data.cpu().numpy()
                    B=batch.vertices.shape[0]
                    if mesh_data_num == 2 or mesh_data_num == 4:
                        batch.mesh_data = 3 # remap  
                        precompute_path = self.train_dataset.mf_precompute_path
                    elif mesh_data_num == 5:
                        batch.mesh_data = 0 # remap
                        if batch.template.shape[1]==6706:
                            precompute_path = self.train_dataset.ict_synth_precompute_nf
                        else:
                            precompute_path = self.train_dataset.ict_synth_precompute_fh
                    batch.gt_rig_params = batch.exp_coeff
                    batch.normals = batch.vertices_normal                
                    batch.dfn_info = os.path.join(
                        precompute_path, f"{batch.id_name}_dfn_info.pkl"
                    )
                    batch.img = torch.from_numpy(np.load(os.path.join(
                        precompute_path, f"{batch.id_name}_img.npy"
                    ))).repeat(B,1,1,1).to(self.device)
                    batch.operators=''
                    
                    # model prediction ---------------------------------------------------------------------------
                    loss_dict, pred_vertices, _, pred_exp, pred_id, pred_seg = self.model(
                        batch, return_all=True, stage=1, epoch=epoch
                    )
                    # --------------------------------------------------------------------------------------------
                
                
                # loss -------------------------------------------------------------------------------------------
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
                
                if index % interv_val == 0:
                    mesh_data = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data]
                    
                    gt_rig = batch.gt_rig_params.cpu()
                    vertices = batch.vertices.cpu()
                    faces = batch.faces.cpu()
                    
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Valid] "
                    #for key, value in running_losses.items():
                    for key, value in loss_dict.items():
                        log_text += f"{key}: {value:.6f} "
                    self.logger.write(log_text+"\n")

                    frame = BS//2
                    v_list = [ v for v in vertices[frame:frame+2] ] + \
                        [ v for v in pred_vertices[frame:frame+2].cpu().detach() ]
                    len_v = len(v_list)
                    f_list=[faces] * len_v
                    save_logdir = f"{self.opts.log_dir}/img/valid/mesh"
                    save_img_name = f"{epoch:03d}_{index:04d}"
                    
                    if pred_seg is not None:
                        pred_seg=pred_seg.squeeze(0).cpu().detach()
                        
                        if mesh_data == 'ict':
                            c_list=[gt_seg]*2+[seg for seg in pred_seg[frame:frame+2]]
                        else:
                            c_list=[seg for seg in pred_seg[frame:frame+2]]*2
                        
                        plot_image_array_seg(
                            v_list, f_list, c_list, 
                            rot_list=[[0,0,0]]*len_v,
                            size=1, bg_black=False, mode='shade', 
                            logdir=save_logdir, 
                            name=save_img_name, save=True
                        )
                    else:
                        plot_image_array(
                            v_list, f_list, 
                            rot_list=[[0,0,0]]*len_v,
                            size=1, bg_black=False, mode='shade', 
                            logdir=save_logdir, 
                            name=save_img_name, save=True
                        )
                    if mesh_data == 'ict':
                        if pred_exp is not None:
                            vis_rig(
                                torch.cat([pred_exp.cpu().detach()[None], gt_rig.squeeze()[None]], dim=0), 
                                f"{self.opts.log_dir}/img/valid/rig/{epoch:03d}_{index:04d}.jpg",
                                normalize=True
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
                
                self.model.mesh_id_encoder.empty_precomputes()
                self.model.mesh_exp_encoder.empty_precomputes()
                self.model.mesh_seg_encoder.empty_precomputes()
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
    trainer.train_v6(epochs=opts.max_epoch)
    

