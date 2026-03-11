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
from utils.matplotlib_rnd import plot_image_array, plot_image_array_seg, vis_rig, plot_image_array_points, vis_mesh_key_weight
from utils.ckpt_utils import *
from utils.exp_utils import plateau_hat_points

from models.baseline import CageNet
from models.NGBC import NeuralGeneralizedBarycentricCoordinate, NeuralGeneralizedBarycentricCoordinateLBS, NeuralGeneralizedBarycentricCoordinateCBD, NeuralStrainDisplacement
from utils.mesh_utils import compute_vertex_strain, calc_norm_torch
# from models.NGBCv2 import NeuralBarycentricCoordinatev2, NeuralBarycentricCoordinatev3

import torch.multiprocessing as mp

# sys.path = list(set(sys.path))

def Options():
    parser = argparse.ArgumentParser(description='neural generalized barycentric coordinate for FA retargeting')
    parser.add_argument('-c', '--config', default='config/train_CBD.yml', help='config file path')
    parser.add_argument("--device",       type=str,   default="cuda:0")
    
    parser.add_argument("--log_dir",      type=str,   default="eval_CBD")

    parser.add_argument("--version",      type=int,   default=1,      help='train method (1: baseline, 2: ours, 3: NBC++)')
    
    parser.add_argument("--data_selection",      type=int,   default=-1,
                        help='select dataset (-1: all, 0: voca, 1:biwi, 2: mf_SEN, 3: coma, 4: mf_ROM, 5: mf all)')
    
    #### Choose a last layer activation for key_weight_model()
    parser.add_argument("--last_activation", default="relu", choices=["relu", "elu", "softmax", "softplus", "none", "sqrelu"],
        help="Choose a last layer activation for NGBC.key_weight_model()"
    )
    
    parser.add_argument("--data_toggle",dest='data_toggle', action='store_true')
    parser.set_defaults(data_toggle=False)
    
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

    parser.add_argument("--use_eval_data2",dest='use_eval_data2', action='store_true')
    parser.set_defaults(use_eval_data2=False)
    
    parser.add_argument("--realtest",dest='realtest', action='store_true')
    parser.set_defaults(realtest=False)
    
    parser.add_argument("--no_eval_metric",dest='no_eval_metric', action='store_true')
    parser.set_defaults(no_eval_metric=False)
    parser.add_argument("--no_vis_interv",dest='no_vis_interv', action='store_true')
    parser.set_defaults(no_vis_interv=False)
    ## ---- eval lbs --------
    parser.add_argument("--eval_use_lbs",dest='eval_use_lbs', action='store_true')
    parser.set_defaults(eval_use_lbs=False)
    parser.add_argument("--eval_use_hybrid", dest="eval_use_hybrid", action="store_true")
    parser.set_defaults(eval_use_hybrid=False)
    parser.add_argument("--eval_use_hybrid_separate", dest="eval_use_hybrid_separate", action="store_true")
    parser.set_defaults(eval_use_hybrid_separate=False) 
    parser.add_argument("--eval_cross_retarget", dest="eval_cross_retarget", action="store_true")
    parser.set_defaults(eval_cross_retarget=False)
    parser.add_argument("--tgt_vert_path",         type=str,   default=None)    
    parser.add_argument("--tgt_norm_path",         type=str,   default=None)    
    parser.add_argument("--tgt_obj_path",         type=str,   default=None)    

    ## at train_CBD.py
    parser.add_argument("--num_lbs_joints", type=int, default=4, help='number of joints for LBS')
    parser.add_argument("--lbs_pretrained_epochs", type=int, default=50, help='number of epochs to pretrain LBS')
    parser.add_argument("--use_lbs", dest='use_lbs',  action='store_true')
    parser.set_defaults(use_lbs=False)
    parser.add_argument("--vis_joint_pos",dest='vis_joint_pos', action='store_true')
    parser.set_defaults(vis_joint_pos=False)
    parser.add_argument("--hybrid_lbs_epoch", type=int, default=-1) # stage2 폴더명에서 from_lbs_ckpt_XXX 못읽을 때 수동 override 용
    parser.add_argument("--use_hyb_delta_lbs_input",dest='use_hyb_delta_lbs_input', action='store_true')
    parser.set_defaults(use_hyb_delta_lbs_input=False)
    parser.add_argument("--use_hyb_concat_lbs",dest='use_hyb_concat_lbs', action='store_true')
    parser.set_defaults(use_hyb_concat_lbs=False)
    parser.add_argument("--use_finetune_lbs",dest='use_finetune_lbs', action='store_true')
    parser.set_defaults(use_finetune_lbs=False)
    
    parser.add_argument("--use_lbs_joint_center", dest='use_lbs_joint_center',  action='store_true')
    parser.set_defaults(use_lbs_joint_center=False)     
    parser.add_argument("--use_exp_joint_predict", dest='use_exp_joint_predict',  action='store_true')
    parser.set_defaults(use_exp_joint_predict=False)
    parser.add_argument("--use_joint_predict", dest='use_joint_predict',  action='store_true')
    parser.set_defaults(use_joint_predict=False)
    parser.add_argument("--use_weighted_joint_pos", dest='use_weighted_joint_pos',  action='store_true')
    parser.set_defaults(use_weighted_joint_pos=False)
    parser.add_argument("--no_use_translation", dest='no_use_translation', action='store_true')
    parser.set_defaults(no_use_translation=False)
    parser.add_argument("--use_lbs_laplacian",dest='use_lbs_laplacian', action='store_true')
    parser.set_defaults(use_lbs_laplacian=False)
    parser.add_argument("--use_lbs_ent",dest='use_lbs_ent', action='store_true')
    parser.set_defaults(use_lbs_ent=False)
    parser.add_argument("--use_lbs_t",dest='use_lbs_t', action='store_true')
    parser.set_defaults(use_lbs_t=False)
    parser.add_argument("--use_lbs_R",dest='use_lbs_R', action='store_true')
    parser.set_defaults(use_lbs_R=False)
    parser.add_argument("--use_lbs_bal",dest='use_lbs_bal', action='store_true')
    parser.set_defaults(use_lbs_bal=False)

    ## strain displacement options ---
    parser.add_argument("--use_strain", dest='use_strain', action='store_true')
    parser.set_defaults(use_strain=False)
    parser.add_argument("--strain_dim", type=int, default=1, help='1: norm only, 2: norm+trace')
    parser.add_argument("--strain_full_grad", dest='strain_full_grad', action='store_true')
    parser.set_defaults(strain_full_grad=False)
    parser.add_argument("--eval_use_strain_disp", dest="eval_use_strain_disp", action="store_true")
    parser.set_defaults(eval_use_strain_disp=False)

    ## ---- eval lbs --------

    parser.add_argument("--tb",           action='store_true')
    parser.set_defaults(is_train=True)
    
    parser.add_argument("--use_t_mask",dest='use_t_mask', action='store_true')
    parser.set_defaults(use_t_mask=False)
    
    parser.add_argument("--laplacian",dest='laplacian', action='store_true')
    parser.set_defaults(laplacian=False)
    
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

import cv2
import glob

def images_to_video_cv(img_dir, out_path, fps=30):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    assert len(imgs) > 0, "No images found"

    first = cv2.imread(imgs[0])
    h, w, _ = first.shape

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    for img_path in imgs:
        img = cv2.imread(img_path)
        writer.write(img)

    writer.release()

# --- Loss Functions ---
def gaussian_kernel1d(kernel_size=5, sigma=1.0):
    x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel

def apply_gaussian_filter(tensor, kernel_size=5, sigma=1.0):
    """Applies a Gaussian filter to the tensor.
    Args:
        tensor (torch.tensor): input tensor
        kernel_size (int): size of the kernel
        sigma (float): sigma value for gaussian filter
    Returns:
        filtered_tensor
    """
    kernel_size = int(kernel_size)

    # Generate the 1D Gaussian kernel
    kernel = gaussian_kernel1d(kernel_size, sigma)
    kernel = kernel.reshape(1, 1, -1).to(tensor.device) # (out_channels, in_channels, kernel_size)
    kernel = kernel.repeat(tensor.size(1), 1, 1) # [128, 1, kernel_size]
    tensor = tensor.transpose(0, 1).unsqueeze(0) # [1, 128, T]

    filtered_tensor = F.conv1d(tensor, kernel, padding=(kernel_size // 2), groups=tensor.size(1))

    # Transpose back to original shape
    filtered_tensor = filtered_tensor.squeeze(0).transpose(0, 1)
    return filtered_tensor

class Trainer():
    def __init__(self, opts):
        # set opts
        self.opts = opts
        self.set_seed(self.opts)
        self.device = opts.device
        last_act_list = ["relu", "elu", "softmax", "softplus", "none", "sqrelu"]
        last_act_list = [self.opts.last_activation==l_act for l_act in last_act_list]
        if opts.version==0:
            #from models import NFS
            from utils.nfr_utils import get_dfn_info
            from utils.mesh_utils import get_mesh_operators
            self.get_dfn_info = get_dfn_info
            self.get_mesh_operators = get_mesh_operators
            
            if opts.NFR:
                from evaluation import Trainer
                trainer = Trainer(opts)
                self.model = trainer.model
            else:
                from models.NFS import NFS
                self.model = NFS(self.opts, None, print_param=True).to(self.device)
#             if opts.NFR:
#                 from evaluation import Trainer
#             else:
#                 from eval_CBD import Trainer
            
        elif opts.version==1:
            self.model = CageNet(
                device=self.device,
                optim_cage=self.opts.optim_cage
            )
        elif opts.version==3: # old before everthing separate
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
                is_train=True,
                use_pou = ~self.opts.no_pou,
                device=self.device,
                #hid_dim=128 if self.opts.use_data2 or self.opts.use_data3 else 256,
                hid_dim=128 if self.opts.align_latent else 256,
            )
        
        elif opts.version==6:
            self.model = NeuralGeneralizedBarycentricCoordinateLBS(
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
                is_train=True,
                use_pou = ~self.opts.no_pou,
                device=self.device,
                #hid_dim=128 if self.opts.use_data2 or self.opts.use_data3 else 256,
                hid_dim=128 if self.opts.align_latent else 256,
            )
            
        elif opts.version==7 or opts.version==8: # copied from train_CBD.py
            self.model = NeuralGeneralizedBarycentricCoordinateLBS(
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
                is_train=True,
                use_pou = ~self.opts.no_pou,
                device=self.device,
                #hid_dim=128 if self.opts.use_data2 or self.opts.use_data3 else 256,
                hid_dim=128 if self.opts.align_latent else 256,
            )
            self.model_CBD = NeuralGeneralizedBarycentricCoordinateCBD(
                opts, num_layers=4,
                num_cage_vertices=self.opts.num_cage_v, #32
                use_exp_recon=False, # not used yet
                use_shp_recon=False, # not used yet
                use_shp=False,
                use_relu=last_act_list[0],
                use_elu=last_act_list[1],
                use_softmax=last_act_list[2],
                use_softplus=last_act_list[3],
                no_activation=last_act_list[4],
                is_train=True,
                use_pou = ~self.opts.no_pou,
                device=self.device,
                #hid_dim=128 if self.opts.use_data2 or self.opts.use_data3 else 256,
                hid_dim=128 if self.opts.align_latent else 256,
            )

        elif opts.version == 9:
            self.model = NeuralGeneralizedBarycentricCoordinateLBS(
                opts, num_layers=4,
                num_cage_vertices=self.opts.num_cage_v,
                use_exp_recon=False, use_shp_recon=False, use_shp=False,
                use_relu=last_act_list[0], use_elu=last_act_list[1],
                use_softmax=last_act_list[2], use_softplus=last_act_list[3],
                no_activation=last_act_list[4],
                is_train=True, use_pou=~self.opts.no_pou, device=self.device,
                hid_dim=128 if self.opts.align_latent else 256,
            )
            strain_dim = self.opts.strain_dim if self.opts.use_strain else 0
            self.model_disp = NeuralStrainDisplacement(
                opts, hid_dim=256, num_layers=4,
                strain_dim=strain_dim, device=self.device,
            )

        else:
            raise NotImplementedError('No matching model version')


        if opts.version == 7: # LBS pretrained + CBD training
            
            parent_dir = os.path.dirname(self.opts.ckpt)
            self.lbs_epoch = self._parse_lbs_epoch_from_stage2_dir(self.opts.ckpt)
            self._load_weight(self.model, name="lbs",ckpt_dir=parent_dir, epoch=self.lbs_epoch)
            self._load_weight(self.model_CBD, name="cbd",ckpt_dir=self.opts.ckpt, epoch=self.opts.start_epoch)
        
        elif opts.version == 8: # LBS + CBD joint training
            self._load_weight(self.model, name="lbs", ckpt_dir=self.opts.ckpt, epoch=self.opts.start_epoch)
            self._load_weight(self.model_CBD, name="cbd", ckpt_dir=self.opts.ckpt, epoch=self.opts.start_epoch)
            self.lbs_epoch = self.opts.start_epoch # joint training, so start_epoch -> lbs_epoch

        elif opts.version == 9: # LBS + Strain Displacement joint training
            self._load_weight(self.model, name="lbs", ckpt_dir=self.opts.ckpt, epoch=self.opts.start_epoch)
            self._load_weight(self.model_disp, name="disp", ckpt_dir=self.opts.ckpt, epoch=self.opts.start_epoch)
            self.lbs_epoch = self.opts.start_epoch

        elif opts.version == 3: # corresponds to version 5 in train_CBD.py
            ckpt_has_cbd = False        
            if self.opts.ckpt is not None:
                if self.opts.continue_ckpt:
                    ckpt = glob.glob(os.path.join(self.opts.ckpt, f"*_{self.opts.start_epoch:03d}.pth"))[0]
                else:
                    ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
                    
                ckpt_dict = torch.load(ckpt, map_location="cpu")
                ckpt_has_cbd = any(k.startswith("cbd_") or k.startswith("key_d_model") for k in ckpt_dict.keys())

            if self.opts.use_lbs and self.opts.start_stage == 2 and ckpt_has_cbd:
                # stage2 ckpt에는 CBD branch 키가 포함되어 있으므로
                # load_state_dict 전에 반드시 branch를 만들어야 함
                self.model.load_CBD_brach(self.opts.use_lbs)
                self.cbd_loaded = True  
            
            self.load_weight()
                # load weight
        else:
            self.load_weight()
        
        # # load weight
        # if opts.version!=0:
        #     self.load_weight()
        
    
    def load_weight(self):
        if self.opts.ckpt:
            print(self.opts.ckpt)
            if self.opts.continue_ckpt:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, f"*_{self.opts.start_epoch:03d}.pth"))[0]
            else:
                ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
            print(f"Loading... {ckpt}")
            
            ckpt_dict = torch.load(ckpt, map_location=self.device)            
            self.model.load_state_dict(ckpt_dict,strict=False)
        else:
            print('no ckpt found, training from scratch!')
    
    def _load_weight(self, model, name="lbs", ckpt_dir=None, epoch=None):
        if ckpt_dir:
            print(f"Loading... {ckpt_dir}")
            if self.opts.continue_ckpt:
                if opts.version == 7:
                    ckpt = glob.glob(os.path.join(ckpt_dir, f"*_{epoch:03d}.pth"))[0]
                elif opts.version == 8 or opts.version == 9:
                    ckpt = glob.glob(os.path.join(ckpt_dir, f"*_{name}_{epoch:03d}.pth"))[0]
            else:
                if opts.version == 7:
                    ckpt = glob.glob(os.path.join(ckpt_dir, f"*_best.pth"))[0]
                elif opts.version == 8 or opts.version == 9:
                    ckpt = glob.glob(os.path.join(ckpt_dir, f"*_{name}_best.pth"))[0]
            ckpt_dict = torch.load(ckpt)            
            model.load_state_dict(ckpt_dict)
            print(f"Loaded! {ckpt}")
        else:
            print(f'no {name} ckpt found, training from scratch!')
    
    def _parse_lbs_epoch_from_stage2_dir(self, stage2_dir: str) -> int:
                name = os.path.basename(stage2_dir)
                parts = name.split('_')
                try:
                    return int(parts[-1])
                except:
                    raise ValueError(f"cannot parse lbs epoch from stage2 dir name: {name}")
    

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
        self.opts.log_dir = os.path.join(self.opts.log_dir, ckpt_path+'-eval',selection+'-pca_data'+f'_e{self.opts.start_epoch:02d}')
        
        os.makedirs(self.opts.log_dir, exist_ok=True)
        if self.opts.no_vis_interv == False:
            os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        else:
            os.makedirs(f"{self.opts.log_dir}/img-full", exist_ok=True)
        
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
                mesh_data_num = batch.mesh_data.cpu().numpy().astype(int)
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
            if self.opts.no_vis_interv == False:
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
            else:
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
                save_logdir = f"{self.opts.log_dir}/img-full"
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
        
        anim_name = f"animation_cbd_{self.opts.num_cage_v}_e{self.opts.start_epoch}"
            
        if self.opts.no_vis_interv:
            images_to_video_cv(
            f"{self.opts.log_dir}/img-full",
            f"{self.opts.log_dir}/{anim_name}.mp4",
            fps=30
            )
            print("animation done!")


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
            self.opts.log_dir = self.opts.log_dir + '-masked' + f'_e{self.opts.start_epoch:02d}'
            
        if self.opts.laplacian:
            self.opts.log_dir = self.opts.log_dir + '-laplacian' + f'_e{self.opts.start_epoch:02d}'
        
        # if self.opts.use_t_mask:
        #     self.opts.log_dir = self.opts.log_dir + '-masked'
            
        # if self.opts.laplacian:
        #     self.opts.log_dir = self.opts.log_dir + '-laplacian'
            
        os.makedirs(self.opts.log_dir, exist_ok=True)

        if self.opts.no_vis_interv == False:
            os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        else:
            os.makedirs(f"{self.opts.log_dir}/img-full", exist_ok=True)
        # os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        
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
        if self.opts.laplacian:
            losses_val["Lap"] = 0.0
            
        mesh_data = self.dataset.data_name
                
        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)
        for index, batch in pbar:
            # if index == 0:
            #     inner_mask = plateau_hat_points(batch.template)
            #     mmm = batch.template.shape[0] / torch.count_nonzero(inner_mask)
            #     print('multiply', mmm)
                
            # model forward ----------------------------------------------------------------------------------
            with torch.no_grad():
                
                if self.opts.version==0:
                    ### only for NFS #############################################################
                    if self.opts.NFR==False:
                        ## only onces!!!
                        if index==0:
                            src_mesh = trimesh.Trimesh(
                                vertices=batch.template[0].cpu().numpy(),
                                faces=batch.faces[0].cpu().numpy()
                            )
                            if self.opts.laplacian:
                                tmp_L = igl.cotmatrix(src_mesh.vertices, src_mesh.faces)
                                src_L = torch.sparse_csc_tensor(
                                    torch.LongTensor(tmp_L.indptr).to(device),
                                    torch.LongTensor(tmp_L.indices).to(device),
                                    torch.FloatTensor(tmp_L.data).to(device),
                                    tmp_L.shape
                                )
                            ## common routine
                            dfn_info = self.get_dfn_info(src_mesh, map_location=self.device)
                            src_operators = self.get_mesh_operators(src_mesh)
                            img = self.model.renderer.render_img(src_mesh).float().to(self.device)
                            img_feat = self.model.get_img_feat(img)
                            vert_feat = self.model.get_local_feature(
                                batch.template[0][None], batch.faces[0], img_feat, at='verts'
                            ).float()
                            tri_feat = self.model.get_local_feature(
                                batch.template[0][None], batch.faces[0], img_feat, at='faces'
                            ).float()

                            pred_id_coeff  = self.model.encode_id(vert_feat, dfn_info)
                            pred_seg_coeff = self.model.encode_seg(vert_feat, dfn_info) if self.opts.design=='new2' else None
                        else:
                            if (batch.template[0].cpu().numpy() - src_mesh.vertices).mean() != 0:
                                src_mesh = trimesh.Trimesh(
                                    vertices=batch.template[0].cpu().numpy(),
                                    faces=batch.faces[0].cpu().numpy()
                                )
                                if self.opts.laplacian:
                                    tmp_L = igl.cotmatrix(src_mesh.vertices, src_mesh.faces)
                                    src_L = torch.sparse_csc_tensor(
                                    torch.LongTensor(tmp_L.indptr).to(device),
                                    torch.LongTensor(tmp_L.indices).to(device),
                                    torch.FloatTensor(tmp_L.data).to(device),
                                    tmp_L.shape
                                )
                                
                                ## common routine
                                dfn_info = self.get_dfn_info(src_mesh, map_location=self.device)
                                src_operators = self.get_mesh_operators(src_mesh)
                                img = self.model.renderer.render_img(src_mesh).float().to(self.device)
                                img_feat = self.model.get_img_feat(img)
                                vert_feat = self.model.get_local_feature(
                                    batch.template[0][None], batch.faces[0], img_feat, at='verts'
                                ).float()
                                tri_feat = self.model.get_local_feature(
                                    batch.template[0][None], batch.faces[0], img_feat, at='faces'
                                ).float()

                                pred_id_coeff  = self.model.encode_id(vert_feat, dfn_info)
                                pred_seg_coeff = self.model.encode_seg(vert_feat, dfn_info) if self.opts.design=='new2' else None

                        # vert_feat_exp = []
                        # for gt_v in batch.vertices:
                        #     _tmp_ = self.model.get_local_feature(gt_v[None], batch.faces[0], img_feat).float()
                        #     vert_feat_exp.append(_tmp_)
                        # vert_feat_exp = torch.vstack(vert_feat_exp)
                        vert_feat_exp =  self.model.get_local_feature(batch.vertices, batch.faces[0], img_feat).float()

                        with torch.no_grad():
                            pred_exp_coeff = self.model.encode_exp(vert_feat_exp, dfn_info, batch_process=True, verbose=False)# [W, Rig]
                            
                            #pred_exp = apply_gaussian_filter(
                            #    pred_exp, kernel_size=5, sigma=1.0
                            #)

                            inputs = (
                                tri_feat if self.opts.dec_type=='jacob' else vert_feat,
                                pred_exp_coeff, pred_id_coeff, pred_seg_coeff,
                                None, batch.template[0][None], batch.faces[0], src_operators
                            )
                            pred_vertices, _ = self.model.decode(inputs, tgt_mesh=src_mesh, batch_process=True)
                    ##############################################################################
                    else:
                        if index==0:
                            src_verts = batch.template[0]
                            src_faces = batch.faces[0]
                            src_m = trimesh.Trimesh(
                                vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                            )
                            if self.opts.laplacian:
                                tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                                src_L = torch.sparse_csc_tensor(
                                    torch.LongTensor(tmp_L.indptr).to(device),
                                    torch.LongTensor(tmp_L.indices).to(device),
                                    torch.FloatTensor(tmp_L.data).to(device),
                                    tmp_L.shape
                                )

                            src_img = self.model.renderer.render_img(src_m).float().to(device)
                            src_img_feat = self.model.get_img_feat(src_img)[None]
                            src_dfn_info = self.get_dfn_info(src_m, map_location=device)
                            src_operators = self.get_mesh_operators(src_m)
                        else:
                            if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:            
                                src_verts = batch.template[0]
                                src_faces = batch.faces[0]
                                src_m = trimesh.Trimesh(
                                    vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                                )
                                if self.opts.laplacian:
                                    tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                                    src_L = torch.sparse_csc_tensor(
                                    torch.LongTensor(tmp_L.indptr).to(device),
                                    torch.LongTensor(tmp_L.indices).to(device),
                                    torch.FloatTensor(tmp_L.data).to(device),
                                    tmp_L.shape
                                )

                                src_img = self.model.renderer.render_img(src_m).float().to(device)
                                src_img_feat = self.model.get_img_feat(src_img)[None]
                                src_dfn_info = self.get_dfn_info(src_m, map_location=device)
                                src_operators = self.get_mesh_operators(src_m)

                        with torch.no_grad():
                            inputs_v = trainer.model.get_inputs(batch.vertices, batch.faces[0])# [B, V, 3+3]

                            ## get expression
                            self.model.model.update_precomputes(src_dfn_info)
                            pred_exp = self.model.model.encode(inputs_v, src_img.to(device), N_F=src_m.faces.shape[0])
                            
                            pred_vertices, _, _ = self.model.calc_new_mesh(
                                src_verts, src_faces, pred_exp, src_operators, src_dfn_info, src_img
                            )
                    
                else:
                    if index==0:
                        src_verts = batch.template[0]
                        src_faces = batch.faces[0]
                        src_m = trimesh.Trimesh(
                            vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                        )
                        if self.opts.laplacian:
                            tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                            src_L = torch.sparse_csc_tensor(
                                torch.LongTensor(tmp_L.indptr).to(device),
                                torch.LongTensor(tmp_L.indices).to(device),
                                torch.FloatTensor(tmp_L.data).to(device),
                                tmp_L.shape
                            )
                    else:
                        if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:            
                            src_verts = batch.template[0]
                            src_faces = batch.faces[0]
                            src_m = trimesh.Trimesh(
                                vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                            )
                            if self.opts.laplacian:
                                tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                                src_L = torch.sparse_csc_tensor(
                                    torch.LongTensor(tmp_L.indptr).to(device),
                                    torch.LongTensor(tmp_L.indices).to(device),
                                    torch.FloatTensor(tmp_L.data).to(device),
                                    tmp_L.shape
                                )
                    if self.opts.version==1:
                        # NEURAL CAGE
                        pred_vertices, _ = trainer.model.retarget(
                            batch.template, batch.vertices, batch.template
                        )
                    elif self.opts.version==21:
                        # NEURAL CAGE
                        pred_vertices, recon_vertices, recon_source, exp_z, \
                        pred_source, _, _, _, _, _, _ = self.model(
                            batch.template, batch.vertices, 
                            batch.template_normal, batch.vertices_normal,
                            batch.mesh_data, epoch=0
                        )
                        ## Use only displacement
                        pred_vertices = pred_vertices - pred_source + batch.template
                    elif self.opts.version==22:
                        (
                            pred_vertices, _, pred_source, _, 
                            src_exp_z, _, _, _, _, _, _, _, _, _
                        ) = self.model(
                            batch.template, batch.vertices, 
                            batch.template_normal, batch.vertices_normal,
                            batch.mesh_data, epoch=0
                        )
                    else:
                        # Ours
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
                    
                    if self.opts.laplacian:
                        losses_val["Lap"] += F.mse_loss(
                            src_L @ batch.vertices*inner_mask, src_L @ pred_vertices*inner_mask
                        ).item() * denom # * mmm
                        
                else:                    
                    if self.opts.laplacian:                    
                        losses_val["Lap"] += F.mse_loss(
                            src_L @ batch.vertices, src_L @ pred_vertices
                        ).item() * denom
                        
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
            # # visualization for debugging
            # if self.opts.batch_size > 1:
            #     interv_val = round(len_data / 10)
            #     if index % interv_val == 0:
            #         vertices = batch.vertices.cpu()
            #         faces = batch.faces.cpu()
            #         pred_vertices_ = pred_vertices.detach().cpu()
                    
            #         v_list = [
            #             vertices[0],
            #             vertices[1],
            #             vertices[HB],
            #             vertices[-1],
            #             pred_vertices_[0],
            #             pred_vertices_[1],
            #             pred_vertices_[HB],
            #             pred_vertices_[-1],
            #         ]
            #         len_v = len(v_list)
            #         f_list=[faces[0]] * len_v
            #         save_logdir = f"{self.opts.log_dir}/img"
            #         save_img_name = f"{index:04d}"
                    
            #         plot_image_array(
            #             v_list, f_list, 
            #             rot_list=[[0,0,0]]*len_v,
            #             size=1, bg_black=False, mode='shade', 
            #             logdir=save_logdir, 
            #             name=save_img_name, save=True
            #         )
            #         # 11649/(11649+6309) + 6309/(11649+6309)

            if self.opts.no_vis_interv == False:
                interv_val = round(len_data / 5)
                if index % interv_val == 0:
                    # for visualization
                    vertices = batch.vertices.cpu()
                    # faces = batch.faces.cpu()
                    faces = batch.faces[0].cpu()
                                    
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
            else:
                # for visualization
                vertices = batch.vertices.cpu()
                # faces = batch.faces.cpu()
                faces = batch.faces[0].cpu()
                                
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
                save_logdir = f"{self.opts.log_dir}/img-full"
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
        
        anim_name = f"animation_cbd_{self.opts.num_cage_v}_e{self.opts.start_epoch}"
            
        if self.opts.no_vis_interv:
            images_to_video_cv(
            f"{self.opts.log_dir}/img-full",
            f"{self.opts.log_dir}/{anim_name}.mp4",
            fps=30
            )
            print("animation done!")
        ##########################################################################################################
        
    
    def evaluate2Cross(self, tgt_vert_path, tgt_norm_path, tgt_obj_path):
        """
            cross-retargeting task
        """
        assert opts.version == 3, "evaluate2Cross is for version 3 only"

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

        trg_name = os.path.splitext(os.path.basename(tgt_vert_path))[0]

        if self.opts.use_t_mask:
            self.opts.log_dir = self.opts.log_dir + '-masked' + f'_e{self.opts.start_epoch:02d}_{trg_name}'
            
        if self.opts.laplacian:
            self.opts.log_dir = self.opts.log_dir + '-laplacian' + f'_e{self.opts.start_epoch:02d}_{trg_name}'
        
        # if self.opts.use_t_mask:
        #     self.opts.log_dir = self.opts.log_dir + '-masked'
            
        # if self.opts.laplacian:
        #     self.opts.log_dir = self.opts.log_dir + '-laplacian'
            
        os.makedirs(self.opts.log_dir, exist_ok=True)

        if self.opts.no_vis_interv == False:
            os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        else:
            os.makedirs(f"{self.opts.log_dir}/img-full", exist_ok=True)
        # os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        
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
        
        # ------------------------------------------
        # 1. Load target mesh (fixed)
        # ------------------------------------------
        tgt_neu_vert = torch.from_numpy(np.load(tgt_vert_path)).float().to(device)
        tgt_neu_norm = torch.from_numpy(np.load(tgt_norm_path)).float().to(device)
        
        # Load target faces for rendering
        tgt_mesh = trimesh.load(tgt_obj_path, process=False)
        if isinstance(tgt_mesh, trimesh.Scene):
            tgt_mesh = trimesh.util.concatenate(tuple(tgt_mesh.geometry.values()))
        tgt_faces = tgt_mesh.faces

        # add batch dim
        tgt_neu_vert = tgt_neu_vert.unsqueeze(0)  # [1, M, 3]
        tgt_neu_norm = tgt_neu_norm.unsqueeze(0)
        
        # eval loop ##############################################################################################        
        
        len_data = len(self.dataloader)
        denom = 1 / len_data
        
        # if self.opts.NFR:
        #     self.model.model.eval()
        # else:
        #     self.model.eval()
        
        self.model.eval()
                
        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)
        for index, batch in pbar:
            # if index == 0:
            #     inner_mask = plateau_hat_points(batch.template)
            #     mmm = batch.template.shape[0] / torch.count_nonzero(inner_mask)
            #     print('multiply', mmm)
                
            # model forward ----------------------------------------------------------------------------------
            with torch.no_grad():   
                if index==0:
                    src_verts = batch.template[0]
                    src_faces = batch.faces[0]
                    src_m = trimesh.Trimesh(
                        vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                    )
                    if self.opts.laplacian:
                        tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                        src_L = torch.sparse_csc_tensor(
                            torch.LongTensor(tmp_L.indptr).to(device),
                            torch.LongTensor(tmp_L.indices).to(device),
                            torch.FloatTensor(tmp_L.data).to(device),
                            tmp_L.shape
                        )
                else:
                    if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:            
                        src_verts = batch.template[0]
                        src_faces = batch.faces[0]
                        src_m = trimesh.Trimesh(
                            vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                        )
                        if self.opts.laplacian:
                            tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                            src_L = torch.sparse_csc_tensor(
                                torch.LongTensor(tmp_L.indptr).to(device),
                                torch.LongTensor(tmp_L.indices).to(device),
                                torch.FloatTensor(tmp_L.data).to(device),
                                tmp_L.shape
                            )
                # if self.opts.version==1:
                #     # NEURAL CAGE
                #     pred_vertices, _ = trainer.model.retarget(
                #         batch.template, batch.vertices, batch.template
                #     )
                # elif self.opts.version==21:
                #     # NEURAL CAGE
                #     pred_vertices, recon_vertices, recon_source, exp_z, \
                #     pred_source, _, _, _, _, _, _ = self.model(
                #         batch.template, batch.vertices, 
                #         batch.template_normal, batch.vertices_normal,
                #         batch.mesh_data, epoch=0
                #     )
                #     ## Use only displacement
                #     pred_vertices = pred_vertices - pred_source + batch.template
                # elif self.opts.version==22:
                #     (
                #         pred_vertices, _, pred_source, _, 
                #         src_exp_z, _, _, _, _, _, _, _, _, _
                #     ) = self.model(
                #         batch.template, batch.vertices, 
                #         batch.template_normal, batch.vertices_normal,
                #         batch.mesh_data, epoch=0
                #     )
                # else:
                # Ours
                # pred_vertices, _, _, _, _, _, _ = self.model_CBD.retarget(
                #     batch.template, batch.vertices, 
                #     batch.template_normal, batch.vertices_normal,
                #     mesh_data=batch.mesh_data, epoch=0
                # )
                
                src_neu_vert = batch.template
                src_def_vert = batch.vertices
                src_neu_norm = batch.template_normal
                src_def_norm = batch.vertices_normal
                # 3-2 Cross retarget (CBD branch)
                pred_vertices, pred_source = self.model.retarget(
                    src_neu_vert, src_neu_norm,
                    src_def_vert, src_def_norm,
                    tgt_neu_vert.expand(src_neu_vert.shape[0], -1, -1),
                    tgt_neu_norm.expand(src_neu_vert.shape[0], -1, -1),
                    mesh_data=batch.mesh_data
                    )
            
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

            if self.opts.no_vis_interv == False:
                interv_val = round(len_data / 5)
                if index % interv_val == 0:
                    # for visualization
                    vertices = batch.vertices.cpu()
                    # faces = batch.faces.cpu()
                    # faces = batch.faces[0].cpu()
                    faces_s = batch.faces[0].cpu()
                    faces_t = tgt_faces
                                    
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
                    f_list=[faces_s, faces_t]
                    save_logdir = f"{self.opts.log_dir}/img"
                    save_img_name = f"{index:04d}"
                    
                    plot_image_array(
                        v_list, f_list, 
                        rot_list=[[0,0,0]]*len_v,
                        size=1, bg_black=False, mode='shade', 
                        logdir=save_logdir, 
                        name=save_img_name, save=True
                    )
            else:
                # for visualization
                vertices = batch.vertices.cpu()
                # faces = batch.faces.cpu()
                # faces = batch.faces[0].cpu()
                faces_s = batch.faces[0].cpu()
                faces_t = tgt_faces
                                
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
                f_list=[faces_s, faces_t]
                save_logdir = f"{self.opts.log_dir}/img-full"
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
        # log_text = f"[Eval] "
        # for key, value in losses_val.items():
        #     txt = f"{key}: {value:.6e} "
        #     print(txt)
        #     log_text += txt
        # self.logger.write(log_text+"\n")
        print('done!')
        
        anim_name = f"animation_cbd_{self.opts.num_cage_v}_e{self.opts.start_epoch}"
            
        if self.opts.no_vis_interv:
            images_to_video_cv(
            f"{self.opts.log_dir}/img-full",
            f"{self.opts.log_dir}/{anim_name}_{trg_name}.mp4",
            fps=30
            )
            print("animation done!")
        ##########################################################################################################
     
    
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
                        src_operators = self.get_mesh_operators(src_mesh)
                        img = self.model.renderer.render_img(src_mesh).float().to(self.device)
                        img_feat = self.model.get_img_feat(img)
                        vert_feat = self.model.get_local_feature(
                            batch.template[0][None], batch.faces[0], img_feat, at='verts'
                        ).float()
                        tri_feat = self.model.get_local_feature(
                            batch.template[0][None], batch.faces[0], img_feat, at='faces'
                        ).float()
                        
                        with torch.no_grad():
                            pred_id_coeff  = self.model.encode_id(vert_feat, dfn_info)
                            pred_seg_coeff = self.model.encode_seg(vert_feat, dfn_info) if self.opts.design=='new2' else None # [1, V, Seg]
                    else:
                        if (batch.template[0].cpu().numpy() - src_mesh.vertices).mean() != 0:
                            src_mesh = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())
                    
                            ## common routine
                            dfn_info = self.get_dfn_info(src_mesh, map_location=self.device)
                            src_operators = self.get_mesh_operators(src_mesh)
                            img = self.model.renderer.render_img(src_mesh).float().to(self.device)
                            img_feat = self.model.get_img_feat(img)
                            vert_feat = self.model.get_local_feature(
                                batch.template[0][None], batch.faces[0], img_feat, at='verts'
                            ).float()
                            tri_feat = self.model.get_local_feature(
                                batch.template[0][None], batch.faces[0], img_feat, at='faces'
                            ).float()
                            
                            with torch.no_grad():
                                pred_id_coeff  = self.model.encode_id(vert_feat, dfn_info)
                                
                                pred_seg_coeff = self.model.encode_seg(vert_feat, dfn_info) if self.opts.design=='new2' else None # [1, V, Seg]

                    vert_feat_exp = []
                    for gt_v in batch.vertices:
                        _tmp_ = self.model.get_local_feature(gt_v[None], batch.faces[0], img_feat).float()
                        vert_feat_exp.append(_tmp_)
                    vert_feat_exp = torch.vstack(vert_feat_exp)

                    with torch.no_grad():
                        pred_exp_coeff = self.model.encode_exp(vert_feat_exp, dfn_info, batch_process=True, verbose=False)# [W, Rig]
                        inputs = (
                            tri_feat if self.opts.dec_type=="jacob" else vert_feat, 
                            pred_exp_coeff, pred_id_coeff, pred_seg_coeff,
                            None, batch.template[0][None], batch.faces[0], src_operators
                        )
                        pred_vertices, _ = self.model.decode(inputs, tgt_mesh=src_mesh, batch_process=True)

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
   

    def evaluateLBS(self):
        
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
        # import pdb;pdb.set_trace()
        if self.opts.log_dir == 'eval_CBD': # default
            self.opts.log_dir = os.path.join(self.opts.log_dir, ckpt_path+'-eval',str(self.opts.data_selection)+'-pca_data'+f'_e{self.opts.start_epoch:02d}')
        
        os.makedirs(self.opts.log_dir, exist_ok=True)
        if self.opts.no_vis_interv == False:
            os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        else:
            os.makedirs(f"{self.opts.log_dir}/img-full", exist_ok=True)

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
                # pred_vertices, recon_vertices, recon_source, exp_z, pred_source, _ = self.model(
                #     batch.template, batch.vertices, 
                #     batch.template_normal, batch.vertices_normal,
                #     batch.mesh_data, epoch=0
                # )
                ## vLBS hybrid model prediction   ----------------------------------------------------------------
                pred_vertices, recon_vertices, recon_source, _, pred_source, _, _, key_weight, W_lbs, T_lbs = self.model(
                    batch.template, batch.vertices,
                    batch.template_normal, batch.vertices_normal,
                    batch.mesh_data, epoch=0, 
                    out_kw=True, stage=1 # 1: LBS stage 2: LBS + CBD
                )
            # ------------------------------------------------------------------------------------------------
            
            # Metric -----------------------------------------------------------------------------------------
            with torch.no_grad():
                mesh_data_num = batch.mesh_data.cpu().numpy().astype(int)
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf', 'ict'])[mesh_data_num]
                
                HB = batch.vertices.shape[0] // 2
            
            if self.opts.no_eval_metric == False:
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
            if self.opts.no_vis_interv == False:
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
            else:
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
                save_logdir = f"{self.opts.log_dir}/img-full"
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
        
        if self.opts.eval_use_lbs:
            if self.opts.no_use_translation:
                anim_name = f"animation_lbs_{self.opts.num_lbs_joints}_e{self.opts.start_epoch}"
            else:
                anim_name = f"animation_lbs_{self.opts.num_lbs_joints}_9dof_e{self.opts.start_epoch}"
        
        if self.opts.no_vis_interv:
            images_to_video_cv(
            f"{self.opts.log_dir}/img-full",
            f"{self.opts.log_dir}/{anim_name}.mp4",
            fps=30
            )
            print("animation done!")
  

    def evaluateLBS2(self):
        """
            self-retargeting task on real
        """
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        print("Running LBS-only evaluation on EvalDataset (real test set)")
        BS = self.opts.batch_size
        HB = BS // 2
        device=self.device
        
        if self.opts.data_selection == -1:
            raise NotImplementedError('only works for individual data')
        data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict']
        selection = data_name_list[self.opts.data_selection]
            
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
            self.opts.log_dir = self.opts.log_dir + '-masked' + f'_e{self.opts.start_epoch:02d}'
            
        if self.opts.laplacian:
            self.opts.log_dir = self.opts.log_dir + '-laplacian' + f'_e{self.opts.start_epoch:02d}'
            
        os.makedirs(self.opts.log_dir, exist_ok=True)
        
        if self.opts.no_vis_interv == False:
            os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        else:
            os.makedirs(f"{self.opts.log_dir}/img-full", exist_ok=True)
            
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
        if self.opts.laplacian:
            losses_val["Lap"] = 0.0
            
        mesh_data = self.dataset.data_name
        
        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)
        for index, batch in pbar:                
            # model forward ----------------------------------------------------------------------------------
            with torch.no_grad():
                if index==0:
                    src_verts = batch.template[0]
                    src_faces = batch.faces[0]
                    src_m = trimesh.Trimesh(
                        vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                    )
                    if self.opts.laplacian:
                        tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                        src_L = torch.sparse_csc_tensor(
                            torch.LongTensor(tmp_L.indptr).to(device),
                            torch.LongTensor(tmp_L.indices).to(device),
                            torch.FloatTensor(tmp_L.data).to(device),
                            tmp_L.shape
                        )
                else:
                    if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:            
                        src_verts = batch.template[0]
                        src_faces = batch.faces[0]
                        src_m = trimesh.Trimesh(
                            vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                        )
                        if self.opts.laplacian:
                            tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                            src_L = torch.sparse_csc_tensor(
                                torch.LongTensor(tmp_L.indptr).to(device),
                                torch.LongTensor(tmp_L.indices).to(device),
                                torch.FloatTensor(tmp_L.data).to(device),
                                tmp_L.shape
                            )
                
                # LBS-only forward
                pred_vertices, _, _, _, _, _, _, _, _, _ = self.model(
                    batch.template,
                    batch.vertices,
                    batch.template_normal,
                    batch.vertices_normal,
                    mesh_data=batch.mesh_data, epoch=0, 
                    out_kw=True, stage=1 # 1: LBS stage 2: LBS + CBD
                )
                
            # Metric -----------------------------------------------------------------------------------------
            with torch.no_grad():
                mesh_data_num = batch.mesh_data.cpu().numpy().astype(int)
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf', 'ict'])[mesh_data_num]
                
                HB = batch.vertices.shape[0] // 2
            
            if self.opts.no_eval_metric == False:
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
            if self.opts.no_vis_interv == False:
                interv_val = round(len_data / 5)
                if index % interv_val == 0:
                    # for visualization
                    vertices = batch.vertices.cpu()
                    # faces = batch.faces.cpu()
                    faces = batch.faces[0].cpu()
                                    
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
            else:
                # for visualization
                vertices = batch.vertices.cpu()
                # faces = batch.faces.cpu()
                faces = batch.faces[0].cpu()
                                
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
                save_logdir = f"{self.opts.log_dir}/img-full"
                save_img_name = f"{index:04d}"
                
                plot_image_array(
                    v_list, f_list, 
                    rot_list=[[0,0,0]]*len_v,
                    size=1, bg_black=False, mode='shade', 
                    logdir=save_logdir, 
                    name=save_img_name, save=True
                )
        ##########################################################################################################
        
        log_text = f"[Eval] "
        for key, value in losses_val.items():
            txt = f"{key}: {value:.6e} "
            print(txt)
            log_text += txt
        self.logger.write(log_text+"\n")                
        print('done!')
        
        if self.opts.eval_use_lbs:
            if self.opts.no_use_translation:
                anim_name = f"animation_lbs_{self.opts.num_lbs_joints}_e{self.opts.start_epoch}"
            else:
                anim_name = f"animation_lbs_{self.opts.num_lbs_joints}_9dof_e{self.opts.start_epoch}"
            
        if self.opts.no_vis_interv:
            images_to_video_cv(
            f"{self.opts.log_dir}/img-full",
            f"{self.opts.log_dir}/{anim_name}.mp4",
            fps=30
            )
            print("animation done!")

     
    def evaluateLBS3(self):
        """
            self-retargeting task on real with version 6 
        """
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        assert opts.version == 6, "evaluateLBS3 is for version 6 only"
        print("Running LBS-only evaluation on EvalDataset (real test set)")
        BS = self.opts.batch_size
        HB = BS // 2
        device=self.device
        
        if self.opts.data_selection == -1:
            raise NotImplementedError('only works for individual data')
        data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict']
        selection = data_name_list[self.opts.data_selection]
            
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
            self.opts.log_dir = self.opts.log_dir + '-masked' + f'_e{self.opts.start_epoch:02d}'
            
        if self.opts.laplacian:
            self.opts.log_dir = self.opts.log_dir + '-laplacian' + f'_e{self.opts.start_epoch:02d}'
            
        os.makedirs(self.opts.log_dir, exist_ok=True)
        
        if self.opts.no_vis_interv == False:
            os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        else:
            os.makedirs(f"{self.opts.log_dir}/img-full", exist_ok=True)
            
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
        if self.opts.laplacian:
            losses_val["Lap"] = 0.0
            
        mesh_data = self.dataset.data_name
        
        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)
        for index, batch in pbar:                
            # model forward ----------------------------------------------------------------------------------
            with torch.no_grad():
                if index==0:
                    src_verts = batch.template[0]
                    src_faces = batch.faces[0]
                    src_m = trimesh.Trimesh(
                        vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                    )
                    if self.opts.laplacian:
                        tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                        src_L = torch.sparse_csc_tensor(
                            torch.LongTensor(tmp_L.indptr).to(device),
                            torch.LongTensor(tmp_L.indices).to(device),
                            torch.FloatTensor(tmp_L.data).to(device),
                            tmp_L.shape
                        )
                else:
                    if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:            
                        src_verts = batch.template[0]
                        src_faces = batch.faces[0]
                        src_m = trimesh.Trimesh(
                            vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                        )
                        if self.opts.laplacian:
                            tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                            src_L = torch.sparse_csc_tensor(
                                torch.LongTensor(tmp_L.indptr).to(device),
                                torch.LongTensor(tmp_L.indices).to(device),
                                torch.FloatTensor(tmp_L.data).to(device),
                                tmp_L.shape
                            )
                
                # LBS-only forward
                pred_vertices, _, _, _, _, _, _, _, _, _ = self.model(
                    batch.template,
                    batch.vertices,
                    batch.template_normal,
                    batch.vertices_normal,
                    mesh_data=batch.mesh_data, epoch=0, 
                    out_kw=True,
                )
                
            # Metric -----------------------------------------------------------------------------------------
            with torch.no_grad():
                mesh_data_num = batch.mesh_data.cpu().numpy().astype(int)
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf', 'ict'])[mesh_data_num]
                
                HB = batch.vertices.shape[0] // 2
            
            if self.opts.no_eval_metric == False:
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
            if self.opts.no_vis_interv == False:
                interv_val = round(len_data / 5)
                if index % interv_val == 0:
                    # for visualization
                    vertices = batch.vertices.cpu()
                    # faces = batch.faces.cpu()
                    faces = batch.faces[0].cpu()
                                    
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
            else:
                # for visualization
                vertices = batch.vertices.cpu()
                # faces = batch.faces.cpu()
                faces = batch.faces[0].cpu()
                                
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
                save_logdir = f"{self.opts.log_dir}/img-full"
                save_img_name = f"{index:04d}"
                
                plot_image_array(
                    v_list, f_list, 
                    rot_list=[[0,0,0]]*len_v,
                    size=1, bg_black=False, mode='shade', 
                    logdir=save_logdir, 
                    name=save_img_name, save=True
                )
        ##########################################################################################################
        
        log_text = f"[Eval] "
        for key, value in losses_val.items():
            txt = f"{key}: {value:.6e} "
            print(txt)
            log_text += txt
        self.logger.write(log_text+"\n")                
        print('done!')
        
        if self.opts.eval_use_lbs:
            if self.opts.no_use_translation:
                anim_name = f"animation_lbs_{self.opts.num_lbs_joints}_e{self.opts.start_epoch}"
            else:
                anim_name = f"animation_lbs_{self.opts.num_lbs_joints}_9dof_e{self.opts.start_epoch}"
            
        if self.opts.no_vis_interv:
            images_to_video_cv(
            f"{self.opts.log_dir}/img-full",
            f"{self.opts.log_dir}/{anim_name}.mp4",
            fps=30
            )
            print("animation done!")
     
     
    def evaluateLBS3Cross(self, tgt_vert_path, tgt_norm_path, tgt_obj_path):
        """
            cross-retargeting task on real with version 6 
        """
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        assert opts.version == 6, "evaluateLBS3 is for version 6 only"
        print("Running LBS-only cross-retargeting evaluation on EvalDataset(real test set)")
        BS = self.opts.batch_size
        HB = BS // 2
        device=self.device
        
        if self.opts.data_selection == -1:
            raise NotImplementedError('only works for individual data')
        data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict']
        selection = data_name_list[self.opts.data_selection]
            
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
        
        trg_name = os.path.splitext(os.path.basename(tgt_vert_path))[0]

        if self.opts.use_t_mask:
            self.opts.log_dir = self.opts.log_dir + '-masked' + f'_e{self.opts.start_epoch:02d}_{trg_name}'
            
        if self.opts.laplacian:
            self.opts.log_dir = self.opts.log_dir + '-laplacian' + f'_e{self.opts.start_epoch:02d}_{trg_name}'
            
        os.makedirs(self.opts.log_dir, exist_ok=True)
        
        if self.opts.no_vis_interv == False:
            os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        else:
            os.makedirs(f"{self.opts.log_dir}/img-full", exist_ok=True)
            
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
        
        
        len_data = len(self.dataloader)
        denom = 1 / len_data
        
        if self.opts.NFR:
            self.model.model.eval()
        else:
            self.model.eval()
            
        mesh_data = self.dataset.data_name
        
        # ------------------------------------------
        # 1. Load target mesh (fixed)
        # ------------------------------------------
        tgt_neu_vert = torch.from_numpy(np.load(tgt_vert_path)).float().to(device)
        tgt_neu_norm = torch.from_numpy(np.load(tgt_norm_path)).float().to(device)
        
        # Load target faces for rendering
        tgt_mesh = trimesh.load(tgt_obj_path, process=False)
        if isinstance(tgt_mesh, trimesh.Scene):
            tgt_mesh = trimesh.util.concatenate(tuple(tgt_mesh.geometry.values()))
        tgt_faces = tgt_mesh.faces

        # add batch dim
        tgt_neu_vert = tgt_neu_vert.unsqueeze(0)  # [1, M, 3]
        tgt_neu_norm = tgt_neu_norm.unsqueeze(0)
        
        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)
        for index, batch in pbar:                
            # model forward ----------------------------------------------------------------------------------
            with torch.no_grad():
                if index==0:
                    src_verts = batch.template[0]
                    src_faces = batch.faces[0]
                    src_m = trimesh.Trimesh(
                        vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                    )
                    if self.opts.laplacian:
                        tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                        src_L = torch.sparse_csc_tensor(
                            torch.LongTensor(tmp_L.indptr).to(device),
                            torch.LongTensor(tmp_L.indices).to(device),
                            torch.FloatTensor(tmp_L.data).to(device),
                            tmp_L.shape
                        )
                else:
                    if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:            
                        src_verts = batch.template[0]
                        src_faces = batch.faces[0]
                        src_m = trimesh.Trimesh(
                            vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                        )
                        if self.opts.laplacian:
                            tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                            src_L = torch.sparse_csc_tensor(
                                torch.LongTensor(tmp_L.indptr).to(device),
                                torch.LongTensor(tmp_L.indices).to(device),
                                torch.FloatTensor(tmp_L.data).to(device),
                                tmp_L.shape
                            )
                
                # LBS-only forward
                # pred_vertices, _, _, _, _, _, _, _, _, _ = self.model(
                #     batch.template,
                #     batch.vertices,
                #     batch.template_normal,
                #     batch.vertices_normal,
                #     mesh_data=batch.mesh_data, epoch=0, 
                #     out_kw=True,
                # )
                
                pred_vertices, _ = self.model.retarget(
                    batch.template, batch.template_normal,
                    batch.vertices, batch.vertices_normal,
                    tgt_neu_vert.expand(batch.template.shape[0], -1, -1),
                    tgt_neu_norm.expand(batch.template.shape[0], -1, -1),
                )
                
            # Metric -----------------------------------------------------------------------------------------
            with torch.no_grad():
                mesh_data_num = batch.mesh_data.cpu().numpy().astype(int)
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf', 'ict'])[mesh_data_num]
                
                HB = batch.vertices.shape[0] // 2
            
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
            if self.opts.no_vis_interv == False:
                interv_val = round(len_data / 5)
                if index % interv_val == 0:
                    # for visualization
                    vertices = batch.vertices.cpu()
                    # faces = batch.faces.cpu()
                    faces_s = batch.faces[0].cpu()
                    faces_t = tgt_faces
                                    
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
                    f_list=[faces_s, faces_t]
                    save_logdir = f"{self.opts.log_dir}/img"
                    save_img_name = f"{index:04d}"
                    
                    plot_image_array(
                        v_list, f_list, 
                        rot_list=[[0,0,0]]*len_v,
                        size=1, bg_black=False, mode='shade', 
                        logdir=save_logdir, 
                        name=save_img_name, save=True
                    )
            else:
                # for visualization
                vertices = batch.vertices.cpu()
                # faces = batch.faces.cpu()
                faces_s = batch.faces[0].cpu()
                faces_t = tgt_faces
                                
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
                f_list=[faces_s, faces_t]
                save_logdir = f"{self.opts.log_dir}/img-full"
                save_img_name = f"{index:04d}"
                
                plot_image_array(
                    v_list, f_list, 
                    rot_list=[[0,0,0]]*len_v,
                    size=1, bg_black=False, mode='shade', 
                    logdir=save_logdir, 
                    name=save_img_name, save=True
                )
        ##########################################################################################################
        
        print('done!')
        
        if self.opts.eval_use_lbs:
            if self.opts.no_use_translation:
                anim_name = f"animation_lbs_cross_{self.opts.num_lbs_joints}_e{self.opts.start_epoch}"
            else:
                anim_name = f"animation_lbs_cross_{self.opts.num_lbs_joints}_9dof_e{self.opts.start_epoch}"
            
        if self.opts.no_vis_interv:
            images_to_video_cv(
            f"{self.opts.log_dir}/img-full",
            f"{self.opts.log_dir}/{anim_name}_{trg_name}.mp4",
            fps=30
            )
            print("animation done!")
   
    
    def evaluateHybrid(self):
        """
        self-retargeting task on real

        LBS:  parent_dir/model_{lbs_epoch}.pth
        CBD:  stage2_dir/model_{start_epoch}.pth  (== opts.start_epoch)
        """
        ##########################################################################################################
        # helper function ----------------------------------------------------------------------------------------
        def _parse_lbs_epoch_from_stage2_dir(stage2_dir: str) -> int:
            name = os.path.basename(stage2_dir)
            parts = name.split('_')
            try:
                return int(parts[-1])
            except:
                raise ValueError(f"cannot parse lbs epoch from stage2 dir name: {name}")
        
        def _load_state_partial(ckpt_path: str, prefixes: tuple):
            sd = torch.load(ckpt_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            part = {k: v for k, v in sd.items() if k.startswith(prefixes)}
            missing, unexpected = self.model.load_state_dict(part, strict=False)
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        print("Running LBS+CBD hybrid evaluation on EvalDataset (real test set)")
        BS = self.opts.batch_size
        HB = BS // 2
        device=self.device
        
        stage2_dir = self.opts.ckpt
        assert stage2_dir is not None, "--ckpt must be stage2 logdir path"
        
        parent_dir = os.path.dirname(stage2_dir)
        
        lbs_epoch = self.opts.hybrid_lbs_epoch
        if lbs_epoch < 0:
            lbs_epoch = _parse_lbs_epoch_from_stage2_dir(stage2_dir)
        assert lbs_epoch >= 0, f"cannot parse lbs_epoch from stage2 dir name: {stage2_dir} (use --hybrid_lbs_epoch)"
        
        lbs_ckpt = os.path.join(parent_dir, f"model_{lbs_epoch:03d}.pth")
        cbd_epoch = self.opts.start_epoch
        print(self.opts.ckpt)
        if self.opts.continue_ckpt:
            cbd_ckpt = glob.glob(os.path.join(self.opts.ckpt, f"*_{cbd_epoch:03d}.pth"))[0]
        else:
            cbd_ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
        # cbd_ckpt = os.path.join(stage2_dir, f"model_{cbd_epoch:03d}.pth")

        assert os.path.isfile(lbs_ckpt), f"missing LBS ckpt: {lbs_ckpt}"
        assert os.path.isfile(cbd_ckpt), f"missing CBD ckpt: {cbd_ckpt}"
        
        if self.opts.data_selection == -1:
            raise NotImplementedError('only works for individual data')
        data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict']
        selection = data_name_list[self.opts.data_selection]
            
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
            self.opts.log_dir = self.opts.log_dir + '-masked' + f'_hybrid_lbse{lbs_epoch:02d}_cbde{self.opts.start_epoch:02d}'
            
        if self.opts.laplacian:
            self.opts.log_dir = self.opts.log_dir + '-laplacian' + f'_hybrid_cbde{self.opts.start_epoch:02d}_lbse{lbs_epoch:02d}'
            
        os.makedirs(self.opts.log_dir, exist_ok=True)
        
        if self.opts.no_vis_interv == False:
            os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        else:
            os.makedirs(f"{self.opts.log_dir}/img-full", exist_ok=True)
            
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
        
        if self.opts.NFR:
            self.model.model.eval()
        else:
            self.model.eval()
        
        # -----------------------------
        # 4) load weights (LBS first, then CBD only)
        # -----------------------------
        # (중요) stage2 forward 쓰려면 CBD branch가 init 되어있어야 함
        if hasattr(self.model, "load_CBD_brach"):
            self.model.load_CBD_brach(True)

        # LBS 관련 prefix들 (필요하면 joint center 모델 prefix도 여기 포함)
        # lbs_prefix = ("lbs_exp_z_model", "lbs_weight_model", "lbs_pose_model", "lbs_joint_center_model", "shape_model")
        lbs_prefix = ("lbs_exp_z_model", "lbs_weight_model", "lbs_pose_model")
        cbd_prefix = ("cbd_exp_z_model", "key_weight_model", "key_d_model")
        
        import pdb;pdb.set_trace()
        
        _load_state_partial(lbs_ckpt, lbs_prefix)
        print(f"[lbs partial load] {os.path.basename(lbs_ckpt)}")
        _load_state_partial(cbd_ckpt, cbd_prefix)    
        print(f"[cbd partial load] {os.path.basename(cbd_ckpt)}")

        losses_val = {
            "MSE": 0.0
        }
        
        if self.opts.use_t_mask:
            losses_val["MSE-in"] = 0.0
            losses_val["MSE-out"] = 0.0
        if self.opts.laplacian:
            losses_val["Lap"] = 0.0
            
        mesh_data = self.dataset.data_name
        
        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)
        
        for index, batch in pbar:                
            # model forward ----------------------------------------------------------------------------------
            with torch.no_grad():
                if index==0:
                    src_verts = batch.template[0]
                    src_faces = batch.faces[0]
                    src_m = trimesh.Trimesh(
                        vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                    )
                    if self.opts.laplacian:
                        tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                        src_L = torch.sparse_csc_tensor(
                            torch.LongTensor(tmp_L.indptr).to(device),
                            torch.LongTensor(tmp_L.indices).to(device),
                            torch.FloatTensor(tmp_L.data).to(device),
                            tmp_L.shape
                        )
                else:
                    if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:            
                        src_verts = batch.template[0]
                        src_faces = batch.faces[0]
                        src_m = trimesh.Trimesh(
                            vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                        )
                        if self.opts.laplacian:
                            tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                            src_L = torch.sparse_csc_tensor(
                                torch.LongTensor(tmp_L.indptr).to(device),
                                torch.LongTensor(tmp_L.indices).to(device),
                                torch.FloatTensor(tmp_L.data).to(device),
                                tmp_L.shape
                            )
                
                pred_vertices, _, _, _, _, _, _, _, _, _ = self.model(
                    batch.template,
                    batch.vertices,
                    batch.template_normal,
                    batch.vertices_normal,
                    mesh_data=batch.mesh_data, epoch=0, 
                    out_kw=True, stage=2 # 1: LBS stage 2: LBS + CBD
                )
                
            # Metric -----------------------------------------------------------------------------------------
            with torch.no_grad():
                mesh_data_num = batch.mesh_data.cpu().numpy().astype(int)
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf', 'ict'])[mesh_data_num]
                
                HB = batch.vertices.shape[0] // 2
            
            if self.opts.no_eval_metric == False:
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
            if self.opts.no_vis_interv == False:
                interv_val = round(len_data / 5)
                if index % interv_val == 0:
                    # for visualization
                    vertices = batch.vertices.cpu()
                    # faces = batch.faces.cpu()
                    faces = batch.faces[0].cpu()
                                    
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
            else:
                # for visualization
                vertices = batch.vertices.cpu()
                # faces = batch.faces.cpu()
                faces = batch.faces[0].cpu()
                                
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
                save_logdir = f"{self.opts.log_dir}/img-full"
                save_img_name = f"{index:04d}"
                
                plot_image_array(
                    v_list, f_list, 
                    rot_list=[[0,0,0]]*len_v,
                    size=1, bg_black=False, mode='shade', 
                    logdir=save_logdir, 
                    name=save_img_name, save=True
                )
        ##########################################################################################################
        
        log_text = f"[Eval] "
        for key, value in losses_val.items():
            txt = f"{key}: {value:.6e} "
            print(txt)
            log_text += txt
        self.logger.write(log_text+"\n")                
        print('done!')
        
        anim_name = f"animation_hybrid_e{self.opts.start_epoch}"
            
        if self.opts.no_vis_interv:
            images_to_video_cv(
            f"{self.opts.log_dir}/img-full",
            f"{self.opts.log_dir}/{anim_name}.mp4",
            fps=30
            )
            print("animation done!")
    
    
    def evaluateHybridSeparate(self):
        """
        self-retargeting task on real

        LBS:  parent_dir/model_{lbs_epoch}.pth
        CBD:  cbd_ckpt_path
        """
        assert opts.version == 7 or opts.version == 8, "evaluateLBS3 is for version 7 or 8 only"
        ##########################################################################################################
        # helper function ----------------------------------------------------------------------------------------
        # def _parse_lbs_epoch_from_stage2_dir(stage2_dir: str) -> int:
        #     name = os.path.basename(stage2_dir)
        #     parts = name.split('_')
        #     try:
        #         return int(parts[-1])
        #     except:
        #         raise ValueError(f"cannot parse lbs epoch from stage2 dir name: {name}")
        
        # def _load_state_partial(ckpt_path: str, prefixes: tuple):
        #     sd = torch.load(ckpt_path, map_location="cpu")
        #     if isinstance(sd, dict) and "state_dict" in sd:
        #         sd = sd["state_dict"]
        #     part = {k: v for k, v in sd.items() if k.startswith(prefixes)}
        #     missing, unexpected = self.model.load_state_dict(part, strict=False)
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        print("Running LBS+CBD hybrid evaluation on EvalDataset (real test set)")
        BS = self.opts.batch_size
        HB = BS // 2
        device=self.device
        
        stage2_dir = self.opts.ckpt
        assert stage2_dir is not None, "--ckpt must be stage2 logdir path"
        
        # parent_dir = os.path.dirname(stage2_dir)
        
        # lbs_ckpt = os.path.join(parent_dir, f"model_{lbs_epoch:03d}.pth")
        stage2_epoch = self.opts.start_epoch
        print(self.opts.ckpt)
        if self.opts.continue_ckpt:
            stage2_ckpt = glob.glob(os.path.join(self.opts.ckpt, f"*_{stage2_epoch:03d}.pth"))[0]
        else:
            stage2_ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
        # cbd_ckpt = os.path.join(stage2_dir, f"model_{cbd_epoch:03d}.pth")

        # assert os.path.isfile(lbs_ckpt), f"missing LBS ckpt: {lbs_ckpt}"
        # assert os.path.isfile(cbd_ckpt), f"missing CBD ckpt: {cbd_ckpt}"
        assert os.path.isfile(stage2_ckpt), f"missing stage 2 CBD ckpt: {stage2_ckpt}"
        
        if self.opts.data_selection == -1:
            raise NotImplementedError('only works for individual data')
        data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict']
        selection = data_name_list[self.opts.data_selection]
            
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
        
        dir_name = f'hybrid_lbse{self.lbs_epoch:02d}_cbde{self.opts.start_epoch:02d}'
        if self.opts.use_t_mask:
            self.opts.log_dir = self.opts.log_dir + f'-masked_{dir_name}'
            
        if self.opts.laplacian:
            self.opts.log_dir = self.opts.log_dir + f'-laplacian_{dir_name}'
            
        os.makedirs(self.opts.log_dir, exist_ok=True)
        
        if self.opts.no_vis_interv == False:
            os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        else:
            os.makedirs(f"{self.opts.log_dir}/img-full", exist_ok=True)
            
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
        
        if self.opts.NFR:
            self.model.model.eval()
        else:
            self.model.eval()
            self.model_CBD.eval()
        
        # -----------------------------
        # 4) load weights (LBS first, then CBD only)
        # -----------------------------
        # done at init time   
        
        losses_val = {
            "MSE": 0.0
        }
        
        if self.opts.use_t_mask:
            losses_val["MSE-in"] = 0.0
            losses_val["MSE-out"] = 0.0
        if self.opts.laplacian:
            losses_val["Lap"] = 0.0
            
        mesh_data = self.dataset.data_name
        
        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)
        
        for index, batch in pbar:                
            # model forward ----------------------------------------------------------------------------------
            with torch.no_grad():
                if index==0:
                    src_verts = batch.template[0]
                    src_faces = batch.faces[0]
                    src_m = trimesh.Trimesh(
                        vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                    )
                    if self.opts.laplacian:
                        tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                        src_L = torch.sparse_csc_tensor(
                            torch.LongTensor(tmp_L.indptr).to(device),
                            torch.LongTensor(tmp_L.indices).to(device),
                            torch.FloatTensor(tmp_L.data).to(device),
                            tmp_L.shape
                        )
                else:
                    if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:            
                        src_verts = batch.template[0]
                        src_faces = batch.faces[0]
                        src_m = trimesh.Trimesh(
                            vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy()
                        )
                        if self.opts.laplacian:
                            tmp_L = igl.cotmatrix(src_m.vertices, src_m.faces)
                            src_L = torch.sparse_csc_tensor(
                                torch.LongTensor(tmp_L.indptr).to(device),
                                torch.LongTensor(tmp_L.indices).to(device),
                                torch.FloatTensor(tmp_L.data).to(device),
                                tmp_L.shape
                            )
                
                # LBS-only forward
                # pred_vertices, _, _, _, _, _, _, _, _, _ = self.model(
                #     batch.template,
                #     batch.vertices,
                #     batch.template_normal,
                #     batch.vertices_normal,
                #     mesh_data=batch.mesh_data, epoch=0, 
                #     out_kw=True, stage=2 # 1: LBS stage 2: LBS + CBD
                # )
                
                pred_vertices, recon_vertices, recon_source, exp_z, pred_source, t_mask, key_d, pred_key_weight, W_lbs, T_lbs = self.model(
                        batch.template, batch.vertices, batch.template_normal, batch.vertices_normal,
                        batch.mesh_data, epoch=0                   
                    )
                if self.opts.use_hyb_delta_lbs_input:
                    pred_vertices_CBD, recon_vertices_CBD, recon_source_CBD, exp_z_CBD, pred_source_CBD, t_mask_CBD, pred_key_weight_CBD = self.model_CBD(
                        batch.template, batch.vertices, batch.template_normal, batch.vertices_normal,
                        batch.mesh_data, epoch=0, lbs_output = pred_vertices,
                    )
                elif self.opts.use_hyb_concat_lbs:
                    ## add pred_vertices to batch_template_v
                    pred_vertices_CBD, recon_vertices_CBD, recon_source_CBD, exp_z_CBD, pred_source_CBD, t_mask_CBD, pred_key_weight_CBD = self.model_CBD(
                        batch.template, batch.vertices, batch.template_normal, batch.vertices_normal,
                        batch.mesh_data, epoch=0, lbs_output = pred_vertices, lbs_source = pred_source
                    )    
                else: # default
                    pred_vertices_CBD, recon_vertices_CBD, recon_source_CBD, exp_z_CBD, pred_source_CBD, t_mask_CBD, pred_key_weight_CBD = self.model_CBD(
                        batch.template, batch.vertices, batch.template_normal, batch.vertices_normal,
                        batch.mesh_data, epoch=0 
                    )
                pred_vertices = pred_vertices + pred_vertices_CBD # expressed face
                pred_source = pred_source + pred_source_CBD # neutral face
                
            # Metric -----------------------------------------------------------------------------------------
            with torch.no_grad():
                mesh_data_num = batch.mesh_data.cpu().numpy().astype(int)
                mesh_data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf', 'ict'])[mesh_data_num]
                
                HB = batch.vertices.shape[0] // 2
            
            if self.opts.no_eval_metric == False:
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
            if self.opts.no_vis_interv == False:
                interv_val = round(len_data / 5)
                if index % interv_val == 0:
                    # for visualization
                    vertices = batch.vertices.cpu()
                    # faces = batch.faces.cpu()
                    faces = batch.faces[0].cpu()
                                    
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
            else:
                # for visualization
                vertices = batch.vertices.cpu()
                # faces = batch.faces.cpu()
                faces = batch.faces[0].cpu()
                                
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
                save_logdir = f"{self.opts.log_dir}/img-full"
                save_img_name = f"{index:04d}"
                
                plot_image_array(
                    v_list, f_list, 
                    rot_list=[[0,0,0]]*len_v,
                    size=1, bg_black=False, mode='shade', 
                    logdir=save_logdir, 
                    name=save_img_name, save=True
                )
        ##########################################################################################################
        
        log_text = f"[Eval] "
        for key, value in losses_val.items():
            txt = f"{key}: {value:.6e} "
            print(txt)
            log_text += txt
        self.logger.write(log_text+"\n")                
        print('done!')
        
        anim_name = f"animation_hybrid_e{self.opts.start_epoch}"
            
        if self.opts.no_vis_interv:
            images_to_video_cv(
            f"{self.opts.log_dir}/img-full",
            f"{self.opts.log_dir}/{anim_name}.mp4",
            fps=30
            )
            print("animation done!")


    def evaluateStrainDisp(self):
        """
        Self-retargeting evaluation for v9: LBS + Strain Displacement.
        """
        assert opts.version == 9, "evaluateStrainDisp is for version 9 only"
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        print("Running LBS+StrainDisp evaluation on EvalDataset (real test set)")
        BS = self.opts.batch_size
        HB = BS // 2
        device = self.device

        stage2_dir = self.opts.ckpt
        assert stage2_dir is not None, "--ckpt must be logdir path"

        if self.opts.data_selection == -1:
            raise NotImplementedError('only works for individual data')
        data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict']
        selection = data_name_list[self.opts.data_selection]

        self.dataset = EvalDataset(data_name=selection, toggle=False)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.opts.batch_size,
            collate_fn=partial(CBD_collate_wrapper_eval, device=self.device),
        )
        ##########################################################################################################

        ###### Logging ###########################################################################################
        os.makedirs(self.opts.log_dir, exist_ok=True)

        ckpt_path = self.opts.ckpt.split('/')[-1]
        self.opts.log_dir = os.path.join(self.opts.log_dir, ckpt_path+'-eval', selection)

        if self.opts.start_epoch > 0:
            epoch_tag = f'e{self.opts.start_epoch:03d}'
        else:
            epoch_tag = 'eBest'
        strain_tag = ''
        if self.opts.use_strain:
            strain_tag = '_strain'
            if self.opts.strain_full_grad:
                strain_tag += '_fullgrad'
        dir_name = f'straindisp_{epoch_tag}{strain_tag}'
        if self.opts.use_t_mask:
            self.opts.log_dir = self.opts.log_dir + f'-masked_{dir_name}'
        else:
            self.opts.log_dir = self.opts.log_dir + f'_{dir_name}'

        os.makedirs(self.opts.log_dir, exist_ok=True)

        if self.opts.no_vis_interv == False:
            os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        else:
            os.makedirs(f"{self.opts.log_dir}/img-full", exist_ok=True)

        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)

        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), opts)

        self.logger = open(os.path.join(self.opts.log_dir, "log.txt"), 'w')
        print(f'Saving log at: {self.opts.log_dir}')

        print(self.dataset.get_data_config())
        self.logger.write(self.dataset.get_data_config())
        #---------------------------------------------------------------------------------------------------------
        ##########################################################################################################

        # eval loop ##############################################################################################
        len_data = len(self.dataloader)
        denom = 1 / len_data

        self.model.eval()
        self.model_disp.eval()

        losses_val = {"MSE": 0.0}
        if self.opts.use_t_mask:
            losses_val["MSE-in"] = 0.0
            losses_val["MSE-out"] = 0.0

        mesh_data = self.dataset.data_name

        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)

        for index, batch in pbar:
            # model forward ----------------------------------------------------------------------------------
            with torch.no_grad():
                ## 1. LBS forward
                pred_lbs, recon_vertices, recon_source, exp_z, pred_source, t_mask, key_d, pred_key_weight, W_lbs, T_lbs = self.model(
                    batch.template, batch.vertices, batch.template_normal, batch.vertices_normal,
                    batch.mesh_data, epoch=0
                )

                ## 2. Compute strain
                strain = None
                if self.opts.use_strain:
                    if self.opts.strain_dim == 2:
                        sn, st = compute_vertex_strain(
                            pred_lbs, batch.template, batch.faces[0] if batch.faces.dim() == 3 else batch.faces,
                            return_trace=True
                        )
                        strain = torch.cat([sn, st], dim=-1)
                    else:
                        strain = compute_vertex_strain(
                            pred_lbs, batch.template,
                            batch.faces[0] if batch.faces.dim() == 3 else batch.faces,
                            return_trace=False
                        )

                ## 3. LBS normals + DispNet
                faces_for_norm = batch.faces[0] if batch.faces.dim() == 3 else batch.faces
                lbs_norm = calc_norm_torch(pred_lbs, faces_for_norm, at='verts')
                displacement, _ = self.model_disp(
                    pred_lbs, lbs_norm,
                    strain=strain
                )

                ## 4. Final composition
                pred_vertices = pred_lbs + displacement

            # Metric -----------------------------------------------------------------------------------------
            with torch.no_grad():
                mesh_data_num = batch.mesh_data.cpu().numpy().astype(int)

            if self.opts.no_eval_metric == False:
                if self.opts.use_t_mask:
                    inner_mask = plateau_hat_points(batch.template)
                    outter_mask = 1 - inner_mask

                    losses_val['MSE-in'] += F.mse_loss(
                        batch.vertices*inner_mask, pred_vertices*inner_mask
                    ).item() * denom

                    losses_val['MSE-out'] += F.mse_loss(
                        batch.template*outter_mask, pred_vertices*outter_mask
                    ).item() * denom

                losses_val['MSE'] += F.mse_loss(
                    batch.vertices, pred_vertices
                ).item() * denom
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
                os.makedirs(save_vert_logdir, exist_ok=True)
                curr_batch = pred_vertices.shape[0]
                for b_idx in range(curr_batch):
                    save_vert_name = f"{save_vert_logdir}/{index*curr_batch + b_idx:06d}.npy"
                    np.save(save_vert_name, pred_vertices[b_idx].detach().cpu().numpy())

            # ------------------------------------------------------------------------------------------------
            # Render: GT | strain heatmap on LBS | displacement heatmap on pred
            is_full = self.opts.no_vis_interv
            if not is_full:
                interv_val = round(len_data / 5)
                should_render = (index % interv_val == 0)
            else:
                should_render = True

            if should_render:
                from PIL import Image
                faces_cpu = batch.faces[0].cpu().numpy() if batch.faces.dim() == 3 else batch.faces.cpu().numpy()
                save_logdir = f"{self.opts.log_dir}/{'img-full' if is_full else 'img'}"
                save_path = os.path.join(save_logdir, f"{index:04d}.png")

                gt_np = batch.vertices[0].cpu().numpy()
                lbs_np = pred_lbs[0].cpu().numpy()
                pred_np = pred_vertices[0].cpu().numpy()
                disp_mag = np.linalg.norm(displacement[0].cpu().numpy(), axis=-1)  # [V]

                panel_specs = [
                    (gt_np, np.zeros(gt_np.shape[0])[:, None], 'YlOrRd', 'GT'),
                    (pred_np, np.zeros(pred_np.shape[0])[:, None], 'YlOrRd', 'Pred'),
                ]
                if strain is not None:
                    strain_np = strain[0].cpu().numpy().squeeze()
                    panel_specs.append((lbs_np, strain_np[:, None], 'coolwarm', 'strain_on_LBS'))
                panel_specs.append((pred_np, disp_mag[:, None], 'YlOrRd', 'disp_on_Pred'))

                panels = []
                tmp_dir = os.path.join(save_logdir, '_tmp')
                os.makedirs(tmp_dir, exist_ok=True)
                for verts, weights, cmap, title in panel_specs:
                    tmp_path = os.path.join(tmp_dir, f'{index:04d}_{title}.png')
                    vmax = max(float(weights.max()), 1e-6)
                    vis_mesh_key_weight(
                        verts, faces_cpu, weights, cage_idx=0,
                        cmap=cmap, vmin=0, vmax=vmax,
                        view_yrots=(0,),
                        save_path=tmp_path, close=True, title=title,
                        shade=True,
                    )
                    panels.append(Image.open(tmp_path))
                    os.remove(tmp_path)

                # stitch horizontally
                total_w = sum(p.width for p in panels)
                max_h = max(p.height for p in panels)
                stitched = Image.new('RGB', (total_w, max_h), (255, 255, 255))
                x_off = 0
                for p in panels:
                    stitched.paste(p, (x_off, 0))
                    x_off += p.width
                stitched.save(save_path)
        ##########################################################################################################

        log_text = f"[Eval] "
        for key, value in losses_val.items():
            txt = f"{key}: {value:.6e} "
            print(txt)
            log_text += txt
        self.logger.write(log_text+"\n")
        print('done!')

        anim_name = f"animation_straindisp_e{self.opts.start_epoch}"

        if self.opts.no_vis_interv:
            images_to_video_cv(
            f"{self.opts.log_dir}/img-full",
            f"{self.opts.log_dir}/{anim_name}.mp4",
            fps=30
            )
            print("animation done!")



    def evaluateStrainDispCross(self, tgt_vert_path, tgt_norm_path, tgt_obj_path):
        """
        Cross-retarget for v9: LBS + Strain Displacement.
        source = test dataset sequence (expressions)
        target = external neutral mesh
        """
        assert opts.version == 9, "evaluateStrainDispCross is for version 9 only"
        BS = self.opts.batch_size
        device = self.device

        if self.opts.data_selection == -1:
            raise NotImplementedError('only works for individual data')
        data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict']
        selection = data_name_list[self.opts.data_selection]

        self.dataset = EvalDataset(data_name=selection, toggle=False)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.opts.batch_size,
            collate_fn=partial(CBD_collate_wrapper_eval, device=self.device),
        )

        ###### Logging
        os.makedirs(self.opts.log_dir, exist_ok=True)
        trg_name = os.path.splitext(os.path.basename(tgt_vert_path))[0]
        ckpt_path = self.opts.ckpt.split('/')[-1]
        self.opts.log_dir = os.path.join(self.opts.log_dir, ckpt_path+'-eval', selection)

        epoch_tag = f'e{self.opts.start_epoch:03d}' if self.opts.start_epoch > 0 else 'eBest'
        strain_tag = '_strain' if self.opts.use_strain else ''
        if self.opts.use_strain and self.opts.strain_full_grad:
            strain_tag += '_fullgrad'
        dir_name = f'straindisp_cross_{trg_name}_{epoch_tag}{strain_tag}'
        self.opts.log_dir = self.opts.log_dir + f'_{dir_name}'

        os.makedirs(self.opts.log_dir, exist_ok=True)
        if self.opts.no_vis_interv:
            os.makedirs(f"{self.opts.log_dir}/img-full", exist_ok=True)
        else:
            os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)

        with open(os.path.join(self.opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(self.opts), f, indent=4)
        self.dump_yaml(os.path.join(self.opts.log_dir, "train_opts.yml"), opts)
        self.logger = open(os.path.join(self.opts.log_dir, "log.txt"), 'w')
        print(f'Saving log at: {self.opts.log_dir}')

        ###### Load target mesh
        self.model.eval()
        self.model_disp.eval()
        print("Running Cross-Retarget Evaluation (StrainDisp)")

        tgt_neu_vert = torch.from_numpy(np.load(tgt_vert_path)).float().to(device).unsqueeze(0)
        tgt_neu_norm = torch.from_numpy(np.load(tgt_norm_path)).float().to(device).unsqueeze(0)

        tgt_mesh = trimesh.load(tgt_obj_path, process=False)
        if isinstance(tgt_mesh, trimesh.Scene):
            tgt_mesh = trimesh.util.concatenate(tuple(tgt_mesh.geometry.values()))
        tgt_faces = torch.from_numpy(tgt_mesh.faces.astype(np.int64)).to(device)
        tgt_faces_np = tgt_mesh.faces

        len_data = len(self.dataloader)
        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)

        for index, batch in pbar:
            with torch.no_grad():
                B_cur = batch.template.shape[0]

                ## 1. LBS retarget: source expression -> target mesh
                pred_lbs, pred_source = self.model.retarget(
                    batch.template, batch.template_normal,
                    batch.vertices, batch.vertices_normal,
                    tgt_neu_vert.expand(B_cur, -1, -1),
                    tgt_neu_norm.expand(B_cur, -1, -1),
                )

                ## 2. Compute strain on target mesh
                strain = None
                if self.opts.use_strain:
                    tgt_template = tgt_neu_vert.expand(B_cur, -1, -1)
                    if self.opts.strain_dim == 2:
                        sn, st = compute_vertex_strain(pred_lbs, tgt_template, tgt_faces, return_trace=True)
                        strain = torch.cat([sn, st], dim=-1)
                    else:
                        strain = compute_vertex_strain(pred_lbs, tgt_template, tgt_faces, return_trace=False)

                ## 3. LBS normals + DispNet
                lbs_norm = calc_norm_torch(pred_lbs, tgt_faces, at='verts')
                displacement, _ = self.model_disp(
                    pred_lbs, lbs_norm,
                    strain=strain
                )

                ## 4. Final composition
                pred_vertices = pred_lbs + displacement

            # Save vertices
            if self.opts.save_vert:
                save_vert_logdir = f"{self.opts.log_dir}/verts"
                os.makedirs(save_vert_logdir, exist_ok=True)
                for b_idx in range(B_cur):
                    np.save(
                        f"{save_vert_logdir}/{index*B_cur + b_idx:06d}.npy",
                        pred_vertices[b_idx].detach().cpu().numpy()
                    )

            # Render
            is_full = self.opts.no_vis_interv
            if not is_full:
                interv_val = round(len_data / 5)
                should_render = (index % interv_val == 0)
            else:
                should_render = True

            if should_render:
                from PIL import Image
                save_logdir = f"{self.opts.log_dir}/{'img-full' if is_full else 'img'}"
                save_path = os.path.join(save_logdir, f"{index:04d}.png")

                lbs_np = pred_lbs[0].cpu().numpy()
                pred_np = pred_vertices[0].cpu().numpy()
                disp_mag = np.linalg.norm(displacement[0].cpu().numpy(), axis=-1)

                panel_specs = [
                    (lbs_np, np.zeros(lbs_np.shape[0])[:, None], 'YlOrRd', 'LBS_retarget'),
                    (pred_np, np.zeros(pred_np.shape[0])[:, None], 'YlOrRd', 'Pred'),
                ]
                if strain is not None:
                    strain_np = strain[0].cpu().numpy().squeeze()
                    panel_specs.append((lbs_np, strain_np[:, None], 'coolwarm', 'strain_on_LBS'))
                panel_specs.append((pred_np, disp_mag[:, None], 'YlOrRd', 'disp_on_Pred'))

                panels = []
                tmp_dir = os.path.join(save_logdir, '_tmp')
                os.makedirs(tmp_dir, exist_ok=True)
                for verts, weights, cmap, title in panel_specs:
                    tmp_path = os.path.join(tmp_dir, f'{index:04d}_{title}.png')
                    vmax = max(float(weights.max()), 1e-6)
                    vis_mesh_key_weight(
                        verts, tgt_faces_np, weights, cage_idx=0,
                        cmap=cmap, vmin=0, vmax=vmax,
                        view_yrots=(0,),
                        save_path=tmp_path, close=True, title=title,
                        shade=True,
                    )
                    panels.append(Image.open(tmp_path))
                    os.remove(tmp_path)

                total_w = sum(p.width for p in panels)
                max_h = max(p.height for p in panels)
                stitched = Image.new('RGB', (total_w, max_h), (255, 255, 255))
                x_off = 0
                for p in panels:
                    stitched.paste(p, (x_off, 0))
                    x_off += p.width
                stitched.save(save_path)

        print('done!')
        anim_name = f"animation_straindisp_cross_{trg_name}_e{self.opts.start_epoch}"
        if self.opts.no_vis_interv:
            images_to_video_cv(
                f"{self.opts.log_dir}/img-full",
                f"{self.opts.log_dir}/{anim_name}.mp4",
                fps=30
            )
            print("animation done!")

    def evaluateHybridSeparateCross(self, tgt_vert_path, tgt_norm_path, tgt_obj_path):
        """
        Cross-retarget:
        source = test dataset sequence
        target = external neutral mesh (.npy)

        tgt_vert_path : path to target neutral vertices (.npy)
        tgt_norm_path : path to target neutral normals (.npy)
        """
        assert opts.version == 7 or opts.version == 8, "evaluateHybridSeparateCross is for version 7 or 8 only"
        ##########################################################################################################
        # helper function ----------------------------------------------------------------------------------------
        # def _parse_lbs_epoch_from_stage2_dir(stage2_dir: str) -> int:
        #     name = os.path.basename(stage2_dir)
        #     parts = name.split('_')
        #     try:
        #         return int(parts[-1])
        #     except:
        #         raise ValueError(f"cannot parse lbs epoch from stage2 dir name: {name}")
        
        # def _load_state_partial(ckpt_path: str, prefixes: tuple):
        #     sd = torch.load(ckpt_path, map_location="cpu")
        #     if isinstance(sd, dict) and "state_dict" in sd:
        #         sd = sd["state_dict"]
        #     part = {k: v for k, v in sd.items() if k.startswith(prefixes)}
        #     missing, unexpected = self.model.load_state_dict(part, strict=False)
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        print("Running LBS+CBD hybrid evaluation on EvalDataset (real test set)")
        BS = self.opts.batch_size
        HB = BS // 2
        len_data = len(self.dataloader)
        device=self.device
        
        stage2_dir = self.opts.ckpt
        assert stage2_dir is not None, "--ckpt must be stage2 logdir path"
        
        # parent_dir = os.path.dirname(stage2_dir)
        
        # lbs_ckpt = os.path.join(parent_dir, f"model_{lbs_epoch:03d}.pth")
        stage2_epoch = self.opts.start_epoch
        print(self.opts.ckpt)
        if self.opts.continue_ckpt:
            stage2_ckpt = glob.glob(os.path.join(self.opts.ckpt, f"*_{stage2_epoch:03d}.pth"))[0]
        else:
            stage2_ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
        # cbd_ckpt = os.path.join(stage2_dir, f"model_{cbd_epoch:03d}.pth")

        # assert os.path.isfile(lbs_ckpt), f"missing LBS ckpt: {lbs_ckpt}"
        # assert os.path.isfile(cbd_ckpt), f"missing CBD ckpt: {cbd_ckpt}"
        assert os.path.isfile(stage2_ckpt), f"missing stage 2 CBD ckpt: {stage2_ckpt}"
        
        if self.opts.data_selection == -1:
            raise NotImplementedError('only works for individual data')
        data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict']
        selection = data_name_list[self.opts.data_selection]
            
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
        
        trg_name = os.path.splitext(os.path.basename(tgt_vert_path))[0]
       
        ckpt_path = self.opts.ckpt.split('/')[-1]
        self.opts.log_dir = os.path.join(self.opts.log_dir, ckpt_path+'-eval', selection)
        
        dir_name = f'hybrid_cross_{trg_name}_lbse{self.lbs_epoch:02d}_cbde{self.opts.start_epoch:02d}'
        if self.opts.use_t_mask:
            self.opts.log_dir = self.opts.log_dir + f'-masked_{dir_name}_{trg_name}'
            
        if self.opts.laplacian:
            self.opts.log_dir = self.opts.log_dir + f'-laplacian_{dir_name}_{trg_name}'
            
        os.makedirs(self.opts.log_dir, exist_ok=True)
        
        if self.opts.no_vis_interv == False:
            os.makedirs(f"{self.opts.log_dir}/img", exist_ok=True)
        else:
            os.makedirs(f"{self.opts.log_dir}/img-full", exist_ok=True)
            
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
        if self.opts.NFR:
            self.model.model.eval()
        else:
            self.model.eval()
            self.model_CBD.eval()
                
        print("Running Cross-Retarget Evaluation")

        # ------------------------------------------
        # 1. Load target mesh (fixed)
        # ------------------------------------------
        tgt_neu_vert = torch.from_numpy(np.load(tgt_vert_path)).float().to(device)
        tgt_neu_norm = torch.from_numpy(np.load(tgt_norm_path)).float().to(device)
        
        # Load target faces for rendering
        tgt_mesh = trimesh.load(tgt_obj_path, process=False)
        if isinstance(tgt_mesh, trimesh.Scene):
            tgt_mesh = trimesh.util.concatenate(tuple(tgt_mesh.geometry.values()))
        tgt_faces = tgt_mesh.faces

        # add batch dim
        tgt_neu_vert = tgt_neu_vert.unsqueeze(0)  # [1, M, 3]
        tgt_neu_norm = tgt_neu_norm.unsqueeze(0)

        # ------------------------------------------
        # 2. Dataset (source sequence)
        # ------------------------------------------
        data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict']
        selection = data_name_list[self.opts.data_selection]

        self.dataset = EvalDataset(data_name=selection, toggle=False)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.opts.batch_size,
            collate_fn=partial(CBD_collate_wrapper_eval, device=self.device),
        )

        # ------------------------------------------
        # 3. Loop over source sequence
        # ------------------------------------------
        for index, batch in enumerate(self.dataloader):

            with torch.no_grad():

                src_neu_vert = batch.template
                src_def_vert = batch.vertices
                src_neu_norm = batch.template_normal
                src_def_norm = batch.vertices_normal

                # 3-1 LBS forward (source only)
                pred_vertices_lbs, pred_source_lbs = self.model.retarget(
                    src_neu_vert, src_neu_norm,
                    src_def_vert, src_def_norm,
                    tgt_neu_vert.expand(src_neu_vert.shape[0], -1, -1),
                    tgt_neu_norm.expand(src_neu_vert.shape[0], -1, -1),
                )

                # 3-2 Cross retarget (CBD branch)
                pred_vertices_cbd, pred_source_cbd = self.model_CBD.retarget(
                    src_neu_vert, src_neu_norm,
                    src_def_vert, src_def_norm,
                    tgt_neu_vert.expand(src_neu_vert.shape[0], -1, -1),
                    tgt_neu_norm.expand(src_neu_vert.shape[0], -1, -1),
                    mesh_data=batch.mesh_data,
                    lbs_output=pred_vertices_lbs if self.opts.use_hyb_concat_lbs else None,
                    lbs_source=pred_source_lbs if self.opts.use_hyb_concat_lbs else None
                )

                # 3-3 combine LBS + CBD
                pred_vertices = pred_vertices_lbs + pred_vertices_cbd # expressed face
                pred_source = pred_source_lbs + pred_source_cbd # neutral face

            # 저장 / 렌더 등은 여기서 처리
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
            # Render: GT | strain heatmap on LBS | displacement heatmap on pred
            is_full = self.opts.no_vis_interv
            if not is_full:
                interv_val = round(len_data / 5)
                should_render = (index % interv_val == 0)
            else:
                should_render = True

            if should_render:
                from PIL import Image
                faces_cpu = batch.faces[0].cpu().numpy() if batch.faces.dim() == 3 else batch.faces.cpu().numpy()
                save_logdir = f"{self.opts.log_dir}/{'img-full' if is_full else 'img'}"
                save_path = os.path.join(save_logdir, f"{index:04d}.png")

                gt_np = batch.vertices[0].cpu().numpy()
                lbs_np = pred_lbs[0].cpu().numpy()
                pred_np = pred_vertices[0].cpu().numpy()
                disp_mag = np.linalg.norm(displacement[0].cpu().numpy(), axis=-1)  # [V]

                panels = []
                tmp_dir = os.path.join(save_logdir, '_tmp')
                os.makedirs(tmp_dir, exist_ok=True)

                # Panel 1: GT, Panel 2: Pred (shaded)
                panel_specs = [
                    (gt_np, np.zeros(gt_np.shape[0])[:, None], 'YlOrRd', 'GT'),
                    (pred_np, np.zeros(pred_np.shape[0])[:, None], 'YlOrRd', 'Pred'),
                ]
                # Panel 3: strain on LBS (only if use_strain)
                if strain is not None:
                    strain_np = strain[0].cpu().numpy().squeeze()
                    panel_specs.append((lbs_np, strain_np[:, None], 'coolwarm', 'strain_on_LBS'))
                # Panel 4: displacement on pred
                panel_specs.append((pred_np, disp_mag[:, None], 'YlOrRd', 'disp_on_Pred'))

                for verts, weights, cmap, title in panel_specs:
                    tmp_path = os.path.join(tmp_dir, f'{index:04d}_{title}.png')
                    vmax = max(float(weights.max()), 1e-6)
                    vis_mesh_key_weight(
                        verts, faces_cpu, weights, cage_idx=0,
                        cmap=cmap, vmin=0, vmax=vmax,
                        view_yrots=(0,),
                        save_path=tmp_path, close=True, title=title,
                        shade=True,
                    )
                    panels.append(Image.open(tmp_path))
                    os.remove(tmp_path)

                total_w = sum(p.width for p in panels)
                max_h = max(p.height for p in panels)
                stitched = Image.new('RGB', (total_w, max_h), (255, 255, 255))
                x_off = 0
                for p in panels:
                    stitched.paste(p, (x_off, 0))
                    x_off += p.width
                stitched.save(save_path)
        ##########################################################################################################
        
        # log_text = f"[Eval] "
        # for key, value in losses_val.items():
        #     txt = f"{key}: {value:.6e} "
        #     print(txt)
        #     log_text += txt
        # self.logger.write(log_text+"\n")                
        print('done!')
        
        anim_name = f"animation_hybrid_cross_e{self.opts.start_epoch}"
            
        if self.opts.no_vis_interv:
            images_to_video_cv(
            f"{self.opts.log_dir}/img-full",
            f"{self.opts.log_dir}/{anim_name}_{trg_name}.mp4",
            fps=30
            )
            print("animation done!")
    
    
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
        
        ## NBC++ (version 2)
        python eval_CBD.py --version 21 --ckpt ./ckpts_CBD3/2025-11-26-14-46-32-NGBCv1 --data_selection 2 --realtest --batch_size 1 --save_vert --use_t_mask
        python eval_CBD.py --version 21 --ckpt ./ckpts_CBD3/2025-11-21-12-50-24-NGBCv1 --data_selection 2 --realtest --batch_size 1 --save_vert --use_t_mask
    """
    mp.set_start_method('spawn', force=True)
    
    # argparse configs
    opts = Options()

    # base configs (yaml)
    # import pdb;pdb.set_trace()
    if opts.version==0:
        opts.config='config/train.yml'
        opts_yaml = yaml.load(open(opts.config), Loader=yaml.FullLoader)
    else:
        config = f'{opts.ckpt}/train_opts.yml'
        opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)

    # update with argparse configs
    opts_ = vars(opts)
    # import pdb;pdb.set_trace()

    opts_yaml.update(opts_)
    opts = argparse.Namespace(**opts_yaml)
    
    if opts.version==0:
        if opts.NFR==False:
            # extras
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
        else:
            opts.design="nfr"
            opts.dec_type="jacob"
            
    print('loaded version:', opts.version)
    
    ## load model
    trainer = Trainer(opts)
    
    if opts.realtest:
        if opts.use_eval_data2:
            trainer.evaluate3() ## nfs test dataloader
        else:
            if opts.eval_cross_retarget:
                if opts.eval_use_lbs:
                    if opts.version == 6:
                        trainer.evaluateLBS3Cross(opts.tgt_vert_path, opts.tgt_norm_path, opts.tgt_obj_path)
                elif opts.eval_use_strain_disp:
                    trainer.evaluateStrainDispCross(opts.tgt_vert_path, opts.tgt_norm_path, opts.tgt_obj_path)
                elif opts.eval_use_hybrid_separate:
                    trainer.evaluateHybridSeparateCross(opts.tgt_vert_path, opts.tgt_norm_path, opts.tgt_obj_path)
                else: # CBD
                    trainer.evaluate2Cross(opts.tgt_vert_path, opts.tgt_norm_path, opts.tgt_obj_path) ## real test frames
            else:
                if opts.eval_use_lbs:
                    if opts.version == 6:
                        trainer.evaluateLBS3()
                    else:
                        trainer.evaluateLBS2()
                elif opts.eval_use_hybrid:
                    trainer.evaluateHybrid()
                elif opts.eval_use_hybrid_separate:
                    trainer.evaluateHybridSeparate()
                elif opts.eval_use_strain_disp:
                    trainer.evaluateStrainDisp()
                else:
                    trainer.evaluate2() ## real test frames
    else:
        if opts.eval_use_lbs:
            trainer.evaluateLBS()
        else:
            trainer.evaluate() ## pca test data

