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

import sys
from pathlib import Path
abs_path = str(Path(__file__).parents[0].absolute())
sys.path+=[abs_path]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataloader_CBD import (
    CBDDataset,
    CBDdataSampler,
    CBD_collate_wrapper
)

# from utils.mesh_utils import Renderer #, calc_cent
from utils.matplotlib_rnd import plot_image_array, plot_image_array_seg, vis_rig
from utils.ckpt_utils import *

from utils.exp_utils import Model_mk1, Model_mk3_1
from utils.remesh_utils import compute_MVC_vertexwise, apply_MVC_weights_batch, build_padded_neighbors, pca_normal_axis_vectorized

def Options():
    parser = argparse.ArgumentParser(description='neural cage for FA')
    parser.add_argument('-c', '--config', default='config/train_CBD.yml', help='config file path')
    parser.add_argument("--device",       type=str,   default="cuda:0")
    
    parser.add_argument("--tb",           action='store_true')
    parser.add_argument("--log_dir",      type=str,   default="ckpts_CBD")
    parser.add_argument("--max_epoch",    type=int,   default=100,  help='number of epochs')
    parser.add_argument("--start_epoch",  type=int,   default=0,      help='number of epochs')
    parser.add_argument("--lr",           type=float, default=0.0002, help='learning rate')
    
    parser.add_argument("--batch_size",   type=int,   default=8,      help='batch size')

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

    parser.set_defaults(is_train=True)
    
    
    parser.add_argument("--optim_cage",dest='optim_cage', action='store_true')
    parser.set_defaults(optim_cage=False)
    
    
    args = parser.parse_args()
    return args

# --- Loss Functions ---

def mvc_loss(mvc_weights):
    """ penalize MVC with negative values """
    neg_loss = torch.nn.functional.relu(-mvc_weights) ** 2
    return torch.mean(neg_loss)

def p2f_loss(before_v, after_v, normals_before, normals_after):
    """ Point-to-Surface Loss
    Args:
        before_v: (B, N, 3) vertices source mesh
        after_v: (B, N, 3) vertices deformed mesh
        normals_before: (B, N, 3) normals from pca plane in source mesh
        normals_after: (B, N, 3) normals from pca plane in deformed mesh
    Returns
        loss (float)
    """
    #device = before_v.device
    
    def distance(verts, norms):
        # B, V, _ = verts.shape
        # dists = torch.zeros(B, V, device=device)
        # for i, neighbors_idx in enumerate(neighbors_map):
        #     if len(neighbors_idx) < 3: continue
            
        #     neighborhood = verts[:, neighbors_idx.to(device), :]
        #     centroid = torch.mean(neighborhood, dim=1)
        #     _, _, V_svd = torch.linalg.svd(neighborhood - centroid.unsqueeze(1))
        #     normals = V_svd[:, -1, :]
            
        #     dist_vec = verts[:, i, :] - centroid
        #     dists[:, i] = torch.abs(torch.sum(dist_vec * normals, dim=1))
        dists = torch.abs(torch.sum(verts * norms, dim=-1))
        return dists

    before_dist = distance(before_v, normals_before)
    after_dist = distance(after_v, normals_after)
    return F.mse_loss(before_dist, after_dist)

def pca_normal_axis(verts, neighbors_map):
    """
    Args:
        verts (torch.tensor): (B, N, 3) vertices
        neighbors_map (list(int):
    Returns:
        normal_axis: (B, N, 3)
    """
    B, V, _ = verts.shape
    
    normal_axis = torch.zeros_like(verts)
    for i, neighbors_idx in enumerate(neighbors_map):
        if len(neighbors_idx) < 2: continue

        neighborhood = verts[:, neighbors_idx, :]
        centroid = torch.mean(neighborhood, dim=1)
        _, _, V_svd = torch.linalg.svd(neighborhood - centroid.unsqueeze(1))
        normal_axis[:, i, :] = V_svd[:, -1, :]
    return normal_axis

def norm_loss(normals_before, normals_after):
    """ PCA Normal Loss 
    Args:
        before_v: (B, N, 3) vertices source mesh
        after_v: (B, N, 3) vertices deformed mesh
        normals_before: (B, N, 3) normals from pca plane in source mesh
        normals_after: (B, N, 3) normals from pca plane in deformed mesh
    Returns
        loss (float)
    """
    # device = before_v.device
    
    # normals_before = pca_normals(before_v)
    # normals_after  = pca_normals(after_v)
    return torch.mean(1.0 - F.cosine_similarity(normals_before, normals_after, dim=-1))

class CageNet(nn.Module):
    """
        Simple implementation of 'Neural Cages for Detail-preserving 3d Deformations'
    """
    def __init__(self, 
                 in_dim=3,
                 hid_dim=512,
                 out_dim=3, 
                 device='cpu',
                 optim_cage=False,
                ):
        super(CageNet, self).__init__()
        
        # pointnet encoder
        self.device=device
        self.optim_cage = optim_cage
        
        
        # self.cage_v = nn.Parameter(torch.rand(128, 3))
        import trimesh        
        test_cage = trimesh.load("test_cage.obj") # 512 vertices 988 faces
        ## may need a better mesh!
        self.C = test_cage.vertices.shape[0]        
        
        self.cage_v = torch.tensor(test_cage.vertices * 1.5).float().to(device)
        if self.optim_cage:
            self.cage_v = nn.Parameter(self.cage_v)
        self.cage_f = torch.tensor(test_cage.faces).long().to(device)
        
        # poinnet encoder
        self.encoder = Model_mk3_1(in_dim, hid_dim).to(device)
        
        # atlasnet decoder
        self.nc_decoder = Model_mk1(in_dim+hid_dim, out_dim).to(device)
        self.nd_decoder = Model_mk1(in_dim+hid_dim+hid_dim, out_dim).to(device)
        
        
        
    def forward(self, source_mesh, deform_mesh, epoch):
        """
        Args:
            source_mesh (torch.tensor) [1, N, 3]: input source mesh
            deform_mesh (torch.tensor) [B, N, 3]: input deformed mesh
        Return:
            predicted deformed mesh
        """
        _, V, _ = source_mesh.shape
        B, V, _ = deform_mesh.shape
        
        #shares same encoder!
        x = torch.cat([deform_mesh, source_mesh], dim=0) # [B+1, N, 3]
        out, _ = self.encoder(x) # [B+1, 512]
        out = out.unsqueeze(1) # [B+1, 1, 512]
        
        #t_code, s_code = out.unsqueeze(1).chunk(2)
        t_code, s_code = out[:B], out[B:] # [B, 1, 512] & [1, 1, 512]
        
        cage_v = self.cage_v.view(1, -1, 3) # [1, C, 3]
        
        x_nc = torch.cat([
            s_code.expand(-1, self.C, -1), 
            cage_v
        ],dim=-1) # [1, C, 512+3]
        source_cage_v = self.nc_decoder(x_nc) + cage_v# [1, C, 3]
        source_cage_v_expand = source_cage_v.expand(B, -1, -1) # [B, C, 3]
        
        
        x_nd = torch.cat([
            s_code.expand(B, self.C, -1),
            t_code.expand(-1, self.C, -1),
            source_cage_v_expand
        ],dim=-1) # [B, C, 512+512+3]
        deform_cage_v = self.nd_decoder(x_nd) + source_cage_v # [B, C, 3]
        
        mvc = compute_MVC_vertexwise(
            source_mesh.squeeze(0), 
            source_cage_v.squeeze(0), 
            self.cage_f
        ) # [N, C]
        
        predicted_mesh = mvc @ deform_cage_v ## [N, C] @ [B, C, 3] -> [B, N, 3]
        
        return predicted_mesh, mvc
        

class Trainer():
    def __init__(self, opts):
        # set opts
        self.opts = opts
        self.set_seed(self.opts)
        self.device = opts.device
        
        self.model = CageNet(device=self.device, optim_cage=self.opts.optim_cage)
        
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
            self.model.load_state_dict(ckpt_dict)
        else:
            print('no ckpt found, training from scratch!')
    
    
    def train_stage1(self, epochs):
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.opts.lr, betas=(0.9, 0.999))
        
        # if self.opts.use_scheduler:
        #     self.scheduler = torch.optim.lr_scheduler.StepLR(
        #         self.optimizer, 
        #         step_size=self.opts.sc_step, 
        #         gamma=self.opts.sc_gamma
        #     )
            
        ##########################################################################################################
        # define dataset -----------------------------------------------------------------------------------------
        BS = self.opts.batch_size
        self.train_dataset = CBDDataset(self.opts, is_train=True)

        # self.neighbor_maps = {
        #     i: [torch.tensor(n, dtype=torch.long) for n in trimesh.Trimesh(
        #         vertices=mesh_info[id_list[0]], faces=mesh_info['face']
        #     ).vertex_neighbors]
        #     for i, (mesh_info, id_list) in enumerate([
        #         (self.train_dataset.voca_mesh, self.train_dataset.voca_id_list),
        #         (self.train_dataset.biwi_mesh, self.train_dataset.biwi_id_list),
        #         (self.train_dataset.mf_mesh, self.train_dataset.mf_id_list)
        #     ])
        # }
        self.neighbor_maps = {
            i: igl.adjacency_list(mesh_info['face'])
            for i, mesh_info in enumerate([
                self.train_dataset.voca_mesh,
                self.train_dataset.biwi_mesh,
                self.train_dataset.mf_mesh,
            ])
        }
        self.neighbor_pad_mask = {}
        for i in self.neighbor_maps.keys():
            # (idx_pad, mask)
            self.neighbor_pad_mask[i] = build_padded_neighbors(self.neighbor_maps[i], device=self.device) 
        
        train_sampler = CBDdataSampler(
            self.train_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            n_sampling=opts.n_sampling,
            n_=self.opts.batch_size,
            is_train=True
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_sampler=train_sampler, 
            # batch_size=8, shuffle=True,
            collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
            num_workers=0,
        )
        
        
        self.valid_dataset = CBDDataset(self.opts, is_valid=True)
        
        valid_sampler = CBDdataSampler(
            self.valid_dataset.len_list, 
            self.opts.batch_size,
            shuffle=True,
            balance=False,
            n_sampling=opts.n_sampling,
            n_=self.opts.batch_size,
            is_train=False
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
        
        self.opts.log_dir = os.path.join(self.opts.log_dir, now+"-CBD")
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
        #---------------------------------------------------------------------------------------------------------
        ##########################################################################################################
        
        
        
        # training loop ##########################################################################################
        global_step = 0
        BEST_LOSS = 100_000_000
        BEST_EPOCH = 0
        start_epoch = self.opts.start_epoch
                
        # define loss lamdba 
        self.loss_lambda = {
            "mvc": self.opts.lambda_mvc,
            "align": self.opts.lambda_align,
            "p2f": self.opts.lambda_p2f,
            "norm": self.opts.lambda_norm,
            # symm 
        }
        
        check_usage = False
        
        len_train_data = len(self.train_dataloader)
        len_valid_data = len(self.valid_dataloader)
        interv_train = round(len_train_data / 10)
        for epoch in range(start_epoch, epochs):
            print(f"[{epoch:03d}/{epochs:03d}][Train]")
            
            ## for logging loss!
            running_losses = {
                "mvc": 0.0,
                "align": 0.0,
                "p2f": 0.0,
                "norm": 0.0,
                "total": 0.0
            }
            
            self.model.train()
            train_counter = 0
            
            pbar = tqdm(enumerate(self.train_dataloader), total=len_train_data, position=0, ncols=100)
                                             
            for index, batch in pbar:
                
                ## for debugging 
                # print(batch.vertices.shape, batch.template.shape)
                #mode = np.array(['voca', 'biwi', 'mf'])[batch.mesh_data]
                #self.train_dataset.vis_mesh(batch.vertices.cpu(), mesh=mode, tag=f"{index:06d}")
                #self.train_dataset.vis_mesh(batch.template.cpu(), mesh=mode, tag=f"{index:06d}")
                
                
                self.optimizer.zero_grad()
                
                # model prediction -------------------------------------------------------------------------------
                pred_vertices, mvc_weights = self.model(batch.template, batch.vertices, epoch=epoch)
                # ------------------------------------------------------------------------------------------------
                
                
                ##################################################################################################
                # ------------------------------------------------------------------------------------------------
                
                mesh_data = np.array(['voca', 'biwi', 'mf'])[batch.mesh_data.cpu().numpy()]
                                
                template_expanded = batch.template.expand_as(pred_vertices)
                #neighbors = self.neighbor_maps[batch.mesh_data.item()]
                
                idx_pad, mask = self.neighbor_pad_mask[batch.mesh_data.item()]
                normals_before = pca_normal_axis_vectorized(template_expanded, idx_pad, mask)
                normals_after = pca_normal_axis_vectorized(pred_vertices, idx_pad, mask)
                
                #normals_before_ = pca_normal_axis(template_expanded, neighbors)
                #normals_after_ = pca_normal_axis(pred_vertices, neighbors)
                #(normals_before * normals_before_).sum(dim=-1).abs().mean().item()
                
                loss_dict = {} # make it as a dictionary
                                
                loss_dict['mvc'] = mvc_loss(mvc_weights)
                loss_dict['align'] = F.mse_loss(batch.vertices, pred_vertices)
                loss_dict['p2f']   = p2f_loss(template_expanded, pred_vertices, normals_before, normals_after)                
                loss_dict['norm']  = norm_loss(normals_before, normals_after)
                # 
                # ------------------------------------------------------------------------------------------------
                ##################################################################################################
                               
                
                # get total loss (lambda weights are multiplied here!)
                loss = 0
                for key, value in loss_dict.items():
                    key_ = key.split("_")[0]
                    tmp = value*self.loss_lambda[key_]
                    loss += tmp
                    running_losses[key] += tmp
                loss_dict["total"] = loss 

                # running loss
                running_losses["total"] += loss_dict["total"]
                pbar.set_description(f"total loss: {loss:.5f}")
                # ------------------------------------------------------------------------------------------------
                # backward
                loss.backward()
                self.optimizer.step()                
                # ------------------------------------------------------------------------------------------------
                
                global_step += 1
                train_counter += 1
                
                # for visualization
                vertices = batch.vertices.cpu()
                faces = batch.faces.cpu()
                
                
                interv_train = round(len_train_data / 10)
                if train_counter % interv_train == 1:
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
            
            ### scheduler (not used - for now...)
            # if self.opts.use_scheduler:
            #     self.scheduler.step()
            
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
                "mvc": 0.0,
                "align": 0.0,
                "p2f": 0.0,
                "norm": 0.0,
                "total": 0.0
            }
            
            counter = 0
            pbar = tqdm(enumerate(self.valid_dataloader), total=len_valid_data, ncols=100)
            for index, batch in pbar:
                counter += 1
                
                # model validation -------------------------------------------------------------------------------
                with torch.no_grad():
                    pred_vertices, mvc_weights = self.model(batch.template, batch.vertices, epoch=epoch)
                # ------------------------------------------------------------------------------------------------
                
                
                ##################################################################################################
                # ------------------------------------------------------------------------------------------------ 
                with torch.no_grad():
                    template_expanded = batch.template.expand_as(pred_vertices)
                    neighbors = self.neighbor_maps[batch.mesh_data.item()]

                    normals_before = pca_normal_axis(template_expanded, neighbors)                
                    normals_after = pca_normal_axis(pred_vertices, neighbors)

                    loss_dict = {} # make it as a dictionary                                
                    loss_dict['mvc'] = mvc_loss(mvc_weights)
                    loss_dict['align'] = F.mse_loss(batch.vertices, pred_vertices)
                    loss_dict['p2f']   = p2f_loss(template_expanded, pred_vertices, normals_before, normals_after)                
                    loss_dict['norm']  = norm_loss(normals_before, normals_after)
                # ------------------------------------------------------------------------------------------------
                ##################################################################################################
                
                # get total loss
                loss = 0
                for key, value in loss_dict.items():
                    key_ = key.split("_")[0]
                    tmp = value.item()*self.loss_lambda[key_]
                    loss += tmp
                    running_losses_val[key] += tmp
                loss_dict["total"] = loss 

                # running loss
                running_losses_val["total"] += loss_dict["total"]
            
                pbar.set_description(f"total loss: {loss:.5f}")
                
                # ------------------------------------------------------------------------------------------------
                # for visualization
                vertices = batch.vertices.cpu()
                faces = batch.faces.cpu()
                mesh_data = np.array(['voca', 'biwi', 'mf'])[batch.mesh_data.cpu().numpy()]
                
                interv_val = round(len_valid_data / 5)
                if index % interv_val == 0:
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
            if running_losses_val["total"]/counter < BEST_LOSS:
                BEST_LOSS = running_losses_val["total"]/counter
                BEST_EPOCH = epoch
                print(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6f} - Best epoch: {BEST_EPOCH:03d}\n")
                self.logger.write(f"[{epoch:03d}/{epochs:03d}] Best Loss: {BEST_LOSS:.6f}\n")
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
    trainer.train_stage1(epochs=opts.max_epoch)

