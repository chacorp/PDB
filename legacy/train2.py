import os
import glob
import json
import yaml
import random
import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
import torch

import sys
from pathlib import Path
abs_path = str(Path(__file__).parents[0].absolute())
sys.path+=[abs_path]
from torch.utils.tensorboard import SummaryWriter

from dataloader_mesh import (
    NFSDataset,
    MeshDataset,
    MeshSampler,
    collate_wrapper,
    IDMeshDataset,
    IDMeshSampler,
    collate_wrapper_ID
)

from utils.mesh_utils import Renderer
from utils.matplotlib_rnd import plot_image_array, plot_image_array_seg, vis_rig
from utils.ckpt_utils import *


def Options():
    parser = argparse.ArgumentParser(description='NFS train')
    parser.add_argument('-c', '--config', default='config/train2.yml', help='config file path')
    # parser.add_argument("--feat_dim",     type=int,   default=128,    help='64 for vocaset; 128 for BIWI')
    # parser.add_argument("--rig_dim",      type=int,   default=128,    help='rig dim')
    # parser.add_argument("--seg_dim",      type=int,   default=20,     help='rig dim')
    parser.add_argument("--device",       type=str,   default="cuda:0")
    parser.add_argument("--fps",          type=int,   default=30)
    
    # parser.add_argument("--audio_type",   type=str,   default="wav2vec2", help="wav2vec2 or hubert")
    # parser.add_argument("--feat_level",   type=str,   default="05",   help="wav2vec or hubert")
    parser.add_argument("--input_dim",    type=int,   default=768,    help='1024 for hubert; 768 for wav2vec2; 21 for logits features')

    parser.add_argument("--tb",           action='store_true')
    parser.add_argument("--log_dir",      type=str,   default="ckpts_new")
    parser.add_argument("--max_epoch",    type=int,   default=500,    help='number of epochs')
    parser.add_argument("--start_epoch",  type=int,   default=0,      help='number of epochs')
    # parser.add_argument("--lambda_recon", type=float, default=1.0,    help='recon lambda for encoder')
    # parser.add_argument("--lambda_temp",  type=float, default=0.0,    help='temp lambda for encoder')
    parser.add_argument("--lr",           type=float, default=0.0001, help='learning rate')
    
    parser.add_argument("--window_size",  type=int,   default=8,      help='window size')
    parser.add_argument("--batch_size",   type=int,   default=1,      help='batch size')

    parser.add_argument("--data_dir",     type=str,   default="/data/ICT-audio2face/data_30fps")
    parser.add_argument("--seed",         type=int,   default=1234,   help='random seed')
    parser.add_argument("--ckpt",         type=str,   default=None)
    
    parser.add_argument("--dec_type",     type=str,   default="disp", help="vert, disp, jacob")
    parser.add_argument("--design",       type=str,   default="new4", help="nfr, new2")
    
    parser.add_argument("--ict_face_only",action='store_true', help="if True, use face region only")

    ## training stages
    parser.add_argument("--mesh_d",       action='store_true', help="train mesh decoder")
    parser.add_argument("--stage1",       action='store_true', help="train stage1")
    parser.add_argument("--stage11",      action='store_true', help="train stage11")

    parser.add_argument("--debug",        action='store_true')
    
    parser.add_argument("--selection",    type=int, default=20, help='dataset selection')
    parser.add_argument("--warmup",       dest='warmup',       action='store_true')
    parser.add_argument("--continue_ckpt",dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)
    
    parser.add_argument("--n_sampling",   dest='n_sampling', action='store_true')
    parser.set_defaults(n_sampling=False)
    
    parser.add_argument("--use_decimate", dest='use_decimate', action='store_true')
    parser.set_defaults(use_decimate=False)
    
    parser.add_argument("--use_scheduler",dest='use_scheduler', action='store_true')
    parser.set_defaults(use_scheduler=False)

    parser.set_defaults(is_train=True)
    
    
    parser.add_argument("--use_canon",dest='use_canon', action='store_true')
    parser.set_defaults(use_canon=False)
    
    
    # Ablation Study (w/o Decoder loss, w/o NLL loss)
    parser.add_argument("--no_BR",    dest='no_BR', action='store_true')
    parser.set_defaults(no_BR=False)
    parser.add_argument("--no_BP",    dest='no_BP', action='store_true')
    parser.set_defaults(no_BP=False)
    
    parser.add_argument("--scale_exp", type=float, default=1.0, help='scale expression code (not useed for training)')
    
    args = parser.parse_args()
    return args


class Trainer():
    def __init__(self, opts):
        # set opts
        self.opts = opts
        self.set_seed(self.opts)
        self.device = opts.device

        self.opts.warmup = True
        
        # if self.opts.warmup:
        #     print("parser argument --warmup is no longer used!, train it seperately")
                
        if 'exp' in self.opts.design:
            from models.NFS import Exp
            self.model = Exp(self.opts, None).to(self.device)
        else:
            from models.NFS2 import NFS
            self.model = NFS(self.opts, None).to(self.device)
        
        # load weight
        self.load_weight()
    
    def load_weight(self):
        if self.opts.ckpt:
            if self.opts.continue_ckpt:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, f"*_{self.opts.start_epoch:03d}.pth"))[0]
            else:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
            print(f"Loading... {ckpt}")
            ckpt_dict = torch.load(ckpt)
            
            ## remove remaining precomputes in DiffusionNet
            del_key_list=['mass', 'L_ind', 'L_val', 'evals', 'evecs', 'grad_X', 'grad_Y', 'faces']
            if not self.opts.continue_ckpt:
                del_key_list.append('audio_encoder')
            ckpt_dict = del_key(ckpt_dict, del_key_list)
            self.model.load_state_dict(ckpt_dict,strict=False)

    def load_nfr_enc_weights(self):
        nfr_ckpt = torch.load(
            f'{abs_path}/experiments/ICT_augment_cnn_ext_dfn4_grad/ICT_augment_cnn_ext_dfn4_grad_0.pth', 
            map_location='cuda:0'
        )['model']
        
        del_key_list = ['linears', 'linear_out', 'fc_mu', 'fc_var', 'gns']
        enc_ckpt = del_key(nfr_ckpt, del_key_list)
        
        key_pair = {
            'module.img_encoder':'img_encoder',
            'module.img_fc':'img_fc',
            'module.global_pn':'mesh_id_encoder',
            'module.encoder':'mesh_exp_encoder',
        }
        enc_ckpt = replace_key(enc_ckpt, key_pair)
        self.model.load_state_dict(enc_ckpt, strict=False)
    
    def train_stage1(self, epochs,stage=1):
        # define loss lamdba -------------------------------------------------------------------------------------
        self.loss_lambda = {
            "l": 1.0, # dummy
            "recon": self.opts.lambda_vert,
            "vert": 1.0,
            "norm":  self.opts.lambda_normal,
            "jacob": self.opts.lambda_jacob,
            "temp":  self.opts.lambda_temp,
            "nll":   self.opts.lambda_seg,
        }
        
        # define optimizer
        if self.opts.design == 'new':
            self.optimizer = torch.optim.Adam(self.model.mesh_decoder.parameters(), lr=self.opts.lr, betas=(0.5, 0.999))
            updated_optim=False
        else:
            self.optimizer = torch.optim.AdamW(self.model.get_mesh_autoencoder_parameters(), lr=self.opts.lr, betas=(0.5, 0.999))
            # self.optimizer = torch.optim.AdamW(self.model.get_mesh_autoencoder_parameters(), lr=self.opts.lr, betas=(0.9, 0.999))
        
        if self.opts.design == 'nfr' or self.opts.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.opts.sc_step, 
                gamma=self.opts.sc_gamma
            ) # following NFR (not used in our setting)
        
        # define dataset -----------------------------------------------------------------------------------------
        BS = self.opts.batch_size
        self.train_dataset = MeshDataset(self.opts, is_train=True)
        train_sampler = MeshSampler(
            self.train_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            n_sampling=self.opts.n_sampling,
            n_=8
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_sampler=train_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(collate_wrapper, device=opts.device), 
            num_workers=0
        )
        
        self.valid_dataset = MeshDataset(self.opts, is_valid=True)
        valid_sampler = MeshSampler(
            self.valid_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            n_sampling=self.opts.n_sampling,
            n_=64
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset, 
            batch_sampler=valid_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(collate_wrapper, device=opts.device), 
            num_workers=0
        )
        
        ## pass 
        #self.model.ict_basedir = self.train_dataset.ict_basedir
        self.model.set_neutral_ict(self.train_dataset.ict_basedir)
        
        # make logdir --------------------------------------------------------------------------------------------
        os.makedirs(self.opts.log_dir, exist_ok=True)
        import datetime
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        
        self.opts.log_dir = os.path.join(self.opts.log_dir, now+"-all")
        os.makedirs(self.opts.log_dir, exist_ok=True)

        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/train/rig", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/valid/mesh", exist_ok=True)
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
        self.logger = open(os.path.join(self.opts.log_dir, "log.txt"), 'w')
        print(f'Saving log at: {self.opts.log_dir}')
        
        print(self.train_dataset.get_data_config())
        print(train_sampler.get_sampler_config())
        print(self.valid_dataset.get_data_config())
        print(valid_sampler.get_sampler_config())
        
        self.logger.write(self.train_dataset.get_data_config())
        self.logger.write(train_sampler.get_sampler_config())
        self.logger.write(self.valid_dataset.get_data_config())
        self.logger.write(valid_sampler.get_sampler_config())
                
        global_step = 0
        BEST_LOSS = 100_000_000
        BEST_EPOCH = 0
        start_epoch = self.opts.start_epoch
        
        th_eye = torch.eye(self.opts.seg_dim, self.opts.seg_dim)
        gt_seg = th_eye[self.model.ict_vert_segment.cpu().detach()]
        
        check_usage = False
        for epoch in range(start_epoch, epochs):
            print(f"[{epoch:03d}/{epochs:03d}][Train]")
            
            running_losses = {
                "total": 0
            }
            self.model.train()
            train_counter = 0
            
            len_train_data = len(self.train_dataloader)
            len_valid_data = len(self.valid_dataloader)
            pbar = tqdm(enumerate(self.train_dataloader), total=len_train_data, position=0, ncols=100)
                                             
            for index, batch in pbar:
                                
                self.optimizer.zero_grad()
                # ------------------------------------------------------------------------------------------------
                # _, id_coeff, gt_rig_params, template, dfn_info, operators_path, vertices, faces, img, mesh_data = batch
                
                loss_dict, pred_vertices, _, pred_exp, pred_id, pred_seg = self.model(
                    batch, 
                    return_all=True, 
                    stage=stage, 
                    epoch=epoch
                )
                
                # get total loss
                loss = 0
                for key, value in loss_dict.items():
                    key_ = key.split("_")[0]
                    tmp = value*self.loss_lambda[key_]
                    loss += tmp
                    # running_losses[key] += tmp
                    if key in running_losses.keys():
                        running_losses[key] += tmp
                    else:
                        running_losses[key] = tmp
                loss_dict["total"] = loss 

                # running loss
                running_losses["total"] += loss_dict["total"]
                
                # ------------------------------------------------------------------------------------------------
                # backward
                loss.backward()
                self.optimizer.step()
                
                # ------------------------------------------------------------------------------------------------
                global_step += 1
                train_counter += 1
                
                # for visualization
                gt_id_coeff = batch.id_coeff.cpu()
                gt_rig = batch.gt_rig_params.cpu()
                gt_rig = gt_rig[:, :self.opts.rig_dim]
                vertices = batch.vertices.cpu().squeeze()
                faces = batch.faces.cpu()
                mesh_data = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data.cpu().numpy()]
                
                interv = round(len_train_data / 10)
                if train_counter % interv == 1:
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Train] "
                    for key, value in running_losses.items():
                        log_text += f"{key}: {value:.6f} "
                    self.logger.write(log_text+"\n")
                    
                    frame = BS//2
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
                            c_list=[seg for seg in pred_seg[frame:frame+2]]*2
                            
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
            ## write log in txtfile
            t_l = running_losses["total"]/train_counter
            train_log_ = f"[{epoch:03d}/{epochs:03d}][Train] Avg Loss: {t_l}\n"
            self.logger.write(train_log_)

            # scheduler
            if self.opts.design == 'nfr' or self.opts.use_scheduler:
                self.scheduler.step()
            
            # log
            if self.opts.tb:
                self.log_loss(self.writer_train, running_losses, epoch, train_counter)

            # save model
            if epoch % 100 == 0:
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_{epoch:03d}.pth')
            
            # validation -----------------------------------------------------------------------------------------
            if epoch % self.opts.validate_epoch == 0:
                self.model.eval()
                print(f"[{epoch:03d}/{epochs:03d}][Valid]")
                running_losses_val = {
                    "total": 0
                }
                counter = 0
                for index, batch in tqdm(enumerate(self.valid_dataloader), total=len_valid_data, ncols=100):
                    counter += 1
                    # ------------------------------------------------------------------------------------------------
                    # _, id_coeff, gt_rig_params, template, dfn_info, operators_path, vertices, faces, img, mesh_data = batch
                        
                    with torch.no_grad():
                        loss_dict, pred_vertices, _, pred_exp, pred_id, pred_seg = self.model(
                            batch, 
                            stage=stage,
                            return_all=True
                        )

                    # get total loss
                    loss = 0
                    for key, value in loss_dict.items():
                        key_ = key.split("_")[0]
                        tmp = value.item()*self.loss_lambda[key_]
                        loss += tmp
                        # running_losses[key] += tmp
                        if key in running_losses_val.keys():
                            running_losses_val[key] += tmp
                        else:
                            running_losses_val[key] = tmp
                    loss_dict["total"] = loss 

                    # running loss
                    running_losses_val["total"] += loss_dict["total"]
                    # ------------------------------------------------------------------------------------------------

                    # for visualization
                    gt_id_coeff = batch.id_coeff.cpu()
                    gt_rig = batch.gt_rig_params.cpu()
                    gt_rig = gt_rig[:, :self.opts.rig_dim]
                    vertices = batch.vertices.cpu().squeeze()
                    faces = batch.faces.cpu()
                    mesh_data = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data.cpu().numpy()]
                    
                    interv = round(len_valid_data /5)
                    if index % interv == 0:
                    # if index % 2 == 0:
                        log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Valid] "
                        for key, value in running_losses_val.items():
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
                    if self.opts.debug:
                        break
                ## write log
                v_l = running_losses_val["total"]/counter
                train_log_ = f"[{epoch:03d}/{epochs:03d}][Valid] Avg Loss: {v_l}\n"
                self.logger.write(train_log_)

                # log
                if self.opts.tb:
                    self.log_loss(self.writer_valid, running_losses_val, epoch, counter)
                
                # best loss
                if running_losses_val["total"]/counter < BEST_LOSS:
                    BEST_LOSS = running_losses_val["total"]/counter
                    BEST_EPOCH = epoch
                    print(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6f} - Best epoch: {BEST_EPOCH:03d}\n")
                    self.logger.write(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6f}\n")
                    torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_best.pth')
    
    # Unsupervised Learning of Robust Spectral Shape Matching
    def train_ULRSSM(self, epochs, stage=9):
        
        # define loss lamdba -------------------------------------------------------------------------------------
        self.loss_lambda = {
            "d": 1.0, # dummy
            "l": self.opts.batch_size,
            "align": self.opts.lambda_align,
            "recon": self.opts.lambda_vert, # gt vertex recon
            "vert": 1.0, # == blendshape recon
            "norm":  self.opts.lambda_normal,
            "jacob": self.opts.lambda_jacob,
            "temp":  self.opts.lambda_temp,
            "nll":   self.opts.lambda_seg,
        }
        
        # define optimizer
        if stage == 9:
            self.optimizer = torch.optim.Adam(
                self.model.get_mesh_ae_parameters(),
                lr=self.opts.lr, 
                betas=(0.5, 0.999)
            )
        elif stage == 8:
            self.optimizer = torch.optim.Adam(
                [*self.model.mesh_encoder.parameters()],
                lr=self.opts.lr, 
                betas=(0.5, 0.999)
            )
    
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-4
        )
        
        # define dataset -----------------------------------------------------------------------------------------
        # BS = self.opts.batch_size
        # self.train_dataset = IDMeshDataset(self.opts, is_train=True)
        # train_sampler = IDMeshSampler(
        #     self.train_dataset.len_list, 
        #     self.opts.batch_size,
        #     shuffle=True,
        #     balance=False,
        #     n_sampling=self.opts.n_sampling,
        #     n_=8
        # )
        # self.train_dataloader = torch.utils.data.DataLoader(
        #     self.train_dataset, 
        #     batch_sampler=train_sampler, 
        #     collate_fn=partial(collate_wrapper_ID, device=opts.device), 
        #     num_workers=0
        # )
        
        # self.valid_dataset = IDMeshDataset(self.opts, is_valid=True)
        # valid_sampler = IDMeshSampler(
        #     self.valid_dataset.len_list, 
        #     self.opts.batch_size,
        #     shuffle=True,
        #     balance=False,
        #     n_sampling=self.opts.n_sampling,
        #     n_=64
        # )
        # self.valid_dataloader = torch.utils.data.DataLoader(
        #     self.valid_dataset, 
        #     batch_sampler=valid_sampler, 
        #     collate_fn=partial(collate_wrapper_ID, device=opts.device), 
        #     num_workers=0
        # )
        BS = self.opts.batch_size
        self.train_dataset = MeshDataset(self.opts, is_train=True)
        train_sampler = MeshSampler(
            self.train_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            n_sampling=self.opts.n_sampling,
            n_=8
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_sampler=train_sampler,
            collate_fn=partial(collate_wrapper, device=opts.device), 
            num_workers=0
        )
        
        self.valid_dataset = MeshDataset(self.opts, is_valid=True)
        valid_sampler = MeshSampler(
            self.valid_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            n_sampling=self.opts.n_sampling,
            n_=64
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset, 
            batch_sampler=valid_sampler,
            collate_fn=partial(collate_wrapper, device=opts.device), 
            num_workers=0
        )
        
        ## pass 
        #self.model.ict_basedir = self.train_dataset.ict_basedir
        self.model.set_neutral_ict(self.train_dataset.ict_basedir)
        
        # make logdir --------------------------------------------------------------------------------------------
        os.makedirs(self.opts.log_dir, exist_ok=True)
        import datetime
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        
        self.opts.log_dir = os.path.join(self.opts.log_dir, now+"-all")
        os.makedirs(self.opts.log_dir, exist_ok=True)

        os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/train/rig", exist_ok=True)
        os.makedirs(f"{self.opts.log_dir}/img/valid/mesh", exist_ok=True)
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
        self.logger = open(os.path.join(self.opts.log_dir, "log.txt"), 'w')
        print(f'Saving log at: {self.opts.log_dir}')
        
        print(self.train_dataset.get_data_config())
        print(train_sampler.get_sampler_config())
        print(self.valid_dataset.get_data_config())
        print(valid_sampler.get_sampler_config())
        
        self.logger.write(self.train_dataset.get_data_config())
        self.logger.write(train_sampler.get_sampler_config())
        self.logger.write(self.valid_dataset.get_data_config())
        self.logger.write(valid_sampler.get_sampler_config())
                
        global_step = 0
        BEST_LOSS = 100_000_000
        BEST_EPOCH = 0
        start_epoch = self.opts.start_epoch
        
        th_eye = torch.eye(self.opts.seg_dim, self.opts.seg_dim)
        gt_seg = th_eye[self.model.ict_vert_segment.cpu().detach()]
        
        check_usage = False
        for epoch in range(start_epoch, epochs):
            print(f"[{epoch:03d}/{epochs:03d}][Train]")
            
            running_losses = {
                "total": 0
            }
            self.model.train()
            train_counter = 0
            
            len_train_data = len(self.train_dataloader)
            len_valid_data = len(self.valid_dataloader)
            # pbar = tqdm(enumerate(self.train_dataloader), total=len_train_data, position=0, ncols=100)
            pbar = tqdm(enumerate(self.train_dataloader), total=len_train_data, position=0)
                                             
            for index, batch in pbar:
                                
                self.optimizer.zero_grad()
                # ------------------------------------------------------------------------------------------------
                # _, id_coeff, gt_rig_params, template, dfn_info, operators_path, vertices, faces, img, mesh_data = batch
                
                loss_dict, pred_vertices, _, pred_exp, pred_id, pred_seg = self.model(
                    batch, 
                    return_all=True, 
                    stage=stage,
                    epoch=epoch
                )
                
                # get total loss
                loss = 0
                # tttt = "log"
                for key, value in loss_dict.items():
                    key_ = key.split("_")[0]
                    tmp = value*self.loss_lambda[key_]
                    loss += tmp

                    # # log text -------------------------
                    # tkey=key.split("_")[-1]
                    # tttt += f"|{tkey} {tmp:.3g}"
                    # # ----------------------------------

                    if key in running_losses.keys():
                        running_losses[key] += tmp
                    else:
                        running_losses[key] = tmp
                loss_dict["total"] = loss 
                
                ### pbar.set_description(f"loss: {loss:.5f}")
                # pbar.set_description(tttt)

                # running loss
                running_losses["total"] += loss_dict["total"]
                
                # ------------------------------------------------------------------------------------------------
                # backward
                loss.backward()
                self.optimizer.step()
                
                # ------------------------------------------------------------------------------------------------
                global_step += 1
                train_counter += 1
                
                # for visualization
                gt_id_coeff = batch.id_coeff.cpu()
                gt_rig = batch.gt_rig_params.cpu()
                vertices = batch.vertices.cpu().squeeze()
                faces = batch.faces.cpu()
                mesh_data = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data.cpu().numpy()]
                
                interv = round(len_train_data / 10)
                if train_counter % interv == 1 and pred_vertices is not None:
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Train] "
                    for key, value in running_losses.items():
                        log_text += f"{key}: {value:.6f} "
                    self.logger.write(log_text+"\n")
                    
                    frame = BS//2
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
                            c_list=[seg for seg in pred_seg[frame:frame+2]]*2
                            
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
            
            ## write log in txtfile
            t_l = running_losses["total"]/train_counter
            train_log_ = f"[{epoch:03d}/{epochs:03d}][Train] Avg Loss: {t_l}\n"
            self.logger.write(train_log_)

            # scheduler
            if self.opts.design == 'nfr' or self.opts.use_scheduler:
                self.scheduler.step()
            
            # log
            if self.opts.tb:
                self.log_loss(self.writer_train, running_losses, epoch, train_counter)

            # save model
            if epoch % 100 == 0:
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_{epoch:03d}.pth')
            
            # validation -----------------------------------------------------------------------------------------
            self.model.eval()
            print(f"[{epoch:03d}/{epochs:03d}][Valid]")
            running_losses_val = {
                "total": 0
            }
            counter = 0
            for index, batch in tqdm(enumerate(self.valid_dataloader), total=len_valid_data, ncols=100):
                counter += 1
                # ------------------------------------------------------------------------------------------------
                # _, id_coeff, gt_rig_params, template, dfn_info, operators_path, vertices, faces, img, mesh_data = batch
                    
                with torch.no_grad():
                    loss_dict, pred_vertices, _, pred_exp, pred_id, pred_seg = self.model(
                        batch, 
                        stage=stage,
                        return_all=True
                    )

                # get total loss
                loss = 0
                for key, value in loss_dict.items():
                    key_ = key.split("_")[0]
                    tmp = value.item()*self.loss_lambda[key_]
                    loss += tmp
                    if key in running_losses_val.keys():
                        running_losses_val[key] += tmp
                    else:
                        running_losses_val[key] = tmp
                loss_dict["total"] = loss 

                # running loss
                running_losses_val["total"] += loss_dict["total"]
                # ------------------------------------------------------------------------------------------------

                # for visualization
                gt_id_coeff = batch.id_coeff.cpu()
                gt_rig = batch.gt_rig_params.cpu()
                vertices = batch.vertices.cpu().squeeze()
                faces = batch.faces.cpu()
                mesh_data = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data.cpu().numpy()]
                
                interv = round(len_valid_data /5)
                if index % interv == 0 and pred_vertices is not None:
                # if index % 2 == 0:
                    log_text = f"[{epoch:03d}/{epochs:03d}][{index:04d}][Valid] "
                    for key, value in running_losses_val.items():
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
                if self.opts.debug:
                    break
            # log
            if self.opts.tb:
                self.log_loss(self.writer_valid, running_losses_val, epoch, counter)
            
            v_l = running_losses_val["total"]/counter
            train_log_ = f"[{epoch:03d}/{epochs:03d}][Valid] Avg Loss: {v_l}\n"
            self.logger.write(train_log_)

            # best loss
            if running_losses_val["total"]/counter < BEST_LOSS:
                BEST_LOSS = running_losses_val["total"]/counter
                BEST_EPOCH = epoch
                best_txt= f"[Valid] Best Loss: {BEST_LOSS:.6f} - Best epoch: {BEST_EPOCH:03d}"
                print(f"[{epoch:03d}/{epochs:03d}]{best_txt}\n")
                self.logger.write(f"[{epoch:03d}/{epochs:03d}]{best_txt}\n")
                torch.save(self.model.state_dict(), f'{self.opts.log_dir}/model_best.pth')
        
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
    if opts.design == "new3":
        trainer.train_ULRSSM(epochs=opts.max_epoch, stage=8)
        # trainer.train_ULRSSM(epochs=opts.max_epoch, stage=9)
    if opts.design == "new5":
        trainer.train_stage1(epochs=opts.max_epoch, stage=21)
    if opts.design == "new_enc":
        trainer.train_stage1(epochs=opts.max_epoch, stage=21)
    else:
        trainer.train_stage1(epochs=opts.max_epoch, stage=1)
    