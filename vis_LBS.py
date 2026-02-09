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
from utils.matplotlib_rnd import plot_image_array, plot_image_array_seg, vis_rig, plot_image_array_points, vis_mesh_key_weight, vis_mesh_with_joint_points
from utils.ckpt_utils import *
from utils.exp_utils import plateau_hat_points

from models.baseline import CageNet
from models.NGBC import NeuralGeneralizedBarycentricCoordinate

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
    ## ---- eval lbs --------
    parser.add_argument("--vis_joint_pos",dest='vis_joint_pos', action='store_true')
    parser.set_defaults(vis_joint_pos=False)
    parser.add_argument("--vis_partition_per_joint",dest='vis_partition_per_joint', action='store_true')
    parser.set_defaults(vis_partition_per_joint=False)
    parser.add_argument("--lib", choices=["mpl", "o3d", "pyv"], default='mpl',
        help="Choose a visualization method to use (matplotlib, open3d, pyvista)"
    )
    ## ---- eval lbs --------
    parser.add_argument("--tb",           action='store_true')
    parser.set_defaults(is_train=True)
    parser.add_argument("--use_t_mask",dest='use_t_mask', action='store_true')
    parser.set_defaults(use_t_mask=False)
    parser.add_argument("--save_vert",dest='save_vert', action='store_true')
    parser.set_defaults(save_vert=False)
    parser.add_argument("--save_gt",dest='save_gt', action='store_true')
    parser.set_defaults(save_gt=False)
    
    args = parser.parse_args()
    return args

import matplotlib.cm as cm
import cv2
# import open3d as o3d
# import pyvista as pv

# def vis_frame_pyvista_3views(
#     V, F, C,
#     mesh_alpha=0.35,
#     joint_radius=0.008,
#     yrots=(0, 25, -25),              # degrees
#     screenshot=None,                 # "xxx.png" 저장 경로
#     window_size=(1920, 640),
#     bg='white',
#     cmap_name='tab20'
# ):
#     # torch -> numpy
#     if hasattr(V, "detach"): V = V.detach().cpu().numpy()
#     if hasattr(F, "detach"): F = F.detach().cpu().numpy()
#     if hasattr(C, "detach"): C = C.detach().cpu().numpy()

#     # pyvista faces format
#     faces = np.hstack([np.full((F.shape[0],1), 3), F]).astype(np.int32)
#     mesh = pv.PolyData(V, faces)

#     joints = pv.PolyData(C)
#     J = C.shape[0]
#     joints["jid"] = np.arange(J)

#     # sphere glyphs (real spheres)
#     sphere = pv.Sphere(radius=joint_radius, theta_resolution=16, phi_resolution=16)
#     glyph = joints.glyph(scale=False, geom=sphere)

#     # colors for legend
#     cmap = cm.get_cmap(cmap_name, max(J, 3))
#     colors = [cmap(i)[:3] for i in range(J)]  # RGB

#     # 1x3 plot
#     p = pv.Plotter(shape=(1, 3), off_screen=(screenshot is not None), window_size=window_size)
#     p.set_background(bg)

#     # camera base: look at mesh center
#     center = mesh.center
#     bounds = np.array(mesh.bounds)  # xmin,xmax,ymin,ymax,zmin,zmax
#     extent = np.max([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
#     dist = 2.2 * extent + 1e-6

#     for k, ang in enumerate(yrots):
#         p.subplot(0, k)
#         p.add_mesh(mesh, color='lightgray', opacity=mesh_alpha, smooth_shading=True)

#         # colored spheres by jid (glyph inherited scalars)
#         p.add_mesh(glyph, scalars="jid", cmap=cmap_name, show_scalar_bar=False)

#         # camera: rotate around +Y axis
#         rad = np.deg2rad(ang)
#         eye = center + np.array([dist*np.sin(rad), 0.15*extent, dist*np.cos(rad)])
#         p.camera_position = [eye.tolist(), center.tolist(), [0, 1, 0]]
#         p.add_text(f"y={ang}°", font_size=14)

#         if k == 0:
#             # legend on first panel (can get big; adjust if J large)
#             legend_entries = [(str(i), colors[i]) for i in range(J)]
#             p.add_legend(legend_entries, bcolor=None, face=None, size=(0.25, 0.25), loc="lower_left")

#     p.show(screenshot=screenshot)
    
    
# from PIL import Image
# from matplotlib import colormaps
# def vis_frame_o3d_3views_save(
#     V, F, C,
#     save_path,
#     mesh_alpha=0.35,
#     joint_radius=0.008,
#     yrots=(0, 25, -25),
#     width=640, height=640,
#     cmap_name='tab20'
# ):
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     if hasattr(V, "detach"): V = V.detach().cpu().numpy()
#     if hasattr(F, "detach"): F = F.detach().cpu().numpy()
#     if hasattr(C, "detach"): C = C.detach().cpu().numpy()

#     # mesh
#     mesh = o3d.geometry.TriangleMesh(
#         o3d.utility.Vector3dVector(V),
#         o3d.utility.Vector3iVector(F)
#     )
#     mesh.compute_vertex_normals()

#     mesh_mat = o3d.visualization.rendering.MaterialRecord()
#     mesh_mat.shader = "defaultLitTransparency"
#     mesh_mat.base_color = [0.7, 0.7, 0.7, mesh_alpha]

#     # spheres
#     J = C.shape[0]
#     cmap = colormaps.get_cmap(cmap_name)
#     spheres = []
#     for j in range(J):
#         sp = o3d.geometry.TriangleMesh.create_sphere(radius=joint_radius, resolution=12)
#         sp.translate(C[j])
#         col = cmap((j % 20) / 19.0)[:3] if J > 20 else cmap(j / max(J-1,1))[:3]
#         sp.paint_uniform_color([float(col[0]), float(col[1]), float(col[2])])
#         sp.compute_vertex_normals()
#         spheres.append(sp)

#     sphere_mat = o3d.visualization.rendering.MaterialRecord()
#     sphere_mat.shader = "defaultLit"

#     r = o3d.visualization.rendering.OffscreenRenderer(width, height)
#     scene = r.scene
#     scene.set_background([1, 1, 1, 1])
#     scene.add_geometry("mesh", mesh, mesh_mat)
#     for j, sp in enumerate(spheres):
#         scene.add_geometry(f"j{j}", sp, sphere_mat)

#     # camera
#     bbox = mesh.get_axis_aligned_bounding_box()
#     center = bbox.get_center()
#     extent = np.max(bbox.get_extent()) + 1e-6
#     dist = 2.2 * extent
#     up = np.array([0, 1, 0], dtype=np.float64)

#     imgs = []
#     for ang in yrots:
#         rad = np.deg2rad(ang)
#         eye = center + np.array([dist*np.sin(rad), 0.15*extent, dist*np.cos(rad)])
#         scene.camera.look_at(center, eye, up)
#         img = r.render_to_image()
#         imgs.append(np.asarray(img))

#     r.release_resources()

#     # concat horizontally
#     concat = np.concatenate(imgs, axis=1)  # (H, W*3, 3/4)
#     Image.fromarray(concat).save(save_path)


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
        elif opts.version==3: # copied from train_CBD.py
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
        else:
            raise NotImplementedError('No matching model version')
        
        # load weight
        if opts.version!=0:
            self.load_weight()
    
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
            
            
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
   
    def visLBS(self):
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
            self.opts.log_dir = os.path.join(self.opts.log_dir, ckpt_path+'-eval',str(self.opts.data_selection)+'-pca_data')
        
        os.makedirs(self.opts.log_dir, exist_ok=True)

        save_logdir = None
        if self.opts.vis_joint_pos:
            save_logdir = f"{self.opts.log_dir}/img-visjoint"
            os.makedirs(save_logdir, exist_ok=True)
        else:
            save_logdir = f"{self.opts.log_dir}/img-full"
            os.makedirs(save_logdir, exist_ok=True)

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
        check_usage = False
        
        len_data = len(self.dataloader)
        denom = 1 / len_data
        
        self.model.eval()
        
        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)
        for index, batch in pbar:
            # model forward ----------------------------------------------------------------------------------
            with torch.no_grad():
                src_verts = batch.template[0]
                src_faces = batch.faces[0]
                src_m = trimesh.Trimesh(vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy())
                # LBS-only forward
                if self.opts.vis_joint_pos:
                    if self.opts.vis_partition_per_joint and index == 0:
                        pred_vertices, _, _, _, _, _, _, _, W_lbs, _, C_lbs = self.model(
                            batch.template,
                            batch.vertices,
                            batch.template_normal,
                            batch.vertices_normal,
                            mesh_data=batch.mesh_data, epoch=0, 
                            out_kw=True, stage=1 # 1: LBS stage 2: LBS + CBD 
                        )
                    else:                    
                        pred_vertices, _, _, _, _, _, _, _, _, _, C_lbs = self.model(
                            batch.template,
                            batch.vertices,
                            batch.template_normal,
                            batch.vertices_normal,
                            mesh_data=batch.mesh_data, epoch=0, 
                            out_kw=True, stage=1 # 1: LBS stage 2: LBS + CBD 
                        )
                else:
                    pred_vertices, _, _, _, _, _, _, _, _, _ = self.model(
                        batch.template,
                        batch.vertices,
                        batch.template_normal,
                        batch.vertices_normal,
                        mesh_data=batch.mesh_data, epoch=0, 
                        out_kw=True, stage=1 # 1: LBS stage 2: LBS + CBD 
                    )
                HB = batch.vertices.shape[0] // 2
            
            if self.opts.vis_partition_per_joint and index == 0:
                # import pdb;pdb.set_trace()
                save_joint_weight_logdir = f"{self.opts.log_dir}/img-visweight"
                os.makedirs(save_joint_weight_logdir, exist_ok=True)
                num_joints = W_lbs[0].shape[-1]
                print("saving joint weights... \n")
                W_all = W_lbs[0].detach().cpu().numpy()     # (N,J)
                Wf_all = W_all[src_faces].mean(axis=1)      # (F,J)
                vmin_global = 0.0
                vmax_global = np.percentile(Wf_all, 99.5)
                for nj in range(num_joints):
                    vis_mesh_key_weight(
                    verts=pred_vertices[0].detach().cpu().numpy(),      # (N,3)
                    faces=src_faces.detach().cpu().numpy(),             # (F,3)
                    key_weight=W_lbs[0].detach().cpu().numpy(),         # (N,K)
                    cage_idx=nj,                                        # int
                    yrot=20,
                    vmin=vmin_global, vmax=vmax_global,
                    view_yrots=(0, 90, 180),
                    save_path=f"{save_joint_weight_logdir}/joint_{nj:03d}.png",
                    )
                print("DONE! \n")

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
            # for visualization
            vertices = batch.vertices.cpu() # [B, N, 3] 
            faces = batch.faces.cpu() # already [F, 3]        
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
            save_img_name = f"{index:04d}"
            
            if self.opts.vis_joint_pos:
                C_lbs = C_lbs.detach().cpu() # keep torch ok (plot 내부에서 numpy로 바꿔도 됨)
                if self.opts.use_weighted_joint_pos: 
                    if index == 0: # iterate only once, use_weighted_joint_pos means fixed position
                        if self.opts.lib == 'mpl':
                            # C_list = [
                            #     None, # GT 에는 안찍기
                            #     C_lbs[0],
                            # ]
                            # plot_image_array_points(
                            #     v_list,   
                            #     f_list,   
                            #     rot_list=[[0,0,0]] * len_v,
                            #     size=1, bg_black=False, mode='shade',
                            #     logdir=save_logdir,
                            #     name=save_img_name, save=True,
                            #     points_list=C_list,  
                            #     points_color='r',
                            #     points_size=2,
                            # )
                            vis_mesh_with_joint_points(
                            verts=pred_vertices[0],
                            faces=faces,
                            C=C_lbs[0],
                            yrot=20,
                            save_path=f"{save_logdir}/{save_img_name}.png"
                            )
                        # elif self.opts.lib == 'o3d':
                        #     save_dir = f"{save_logdir}-o3d"
                        #     os.makedirs(save_dir, exist_ok=True)
                        #     save_path = f"{save_dir}/{save_img_name}.png"
                        #     vis_frame_o3d_3views_save(
                        #         V=pred_vertices[0],   # (N,3)
                        #         F=faces,
                        #         C=C_lbs[0],           # (J,3)
                        #         save_path=save_path  # ← 핵심
                        #     )
                        # elif self.opts.lib == 'pysv':
                        #     import pdb;pdb.set_trace()
                        #     save_dir = f"{save_logdir}-pyv"
                        #     os.makedirs(save_dir, exist_ok=True)
                        #     save_path = f"{save_dir}/{save_img_name}.png"
                        #     vis_frame_pyvista_3views(
                        #         V=pred_vertices[0],   # (N,3)
                        #         F=faces,
                        #         C=C_lbs[0],           # (J,3)
                        #         screenshot=save_path  # ← 핵심
                        #     )
                    else: 
                        continue
                else:
                    if self.opts.lib == 'mpl':
                        # C_list = [
                        #     None, # GT 에는 안찍기
                        #     C_lbs[0],
                        # ]
                        # plot_image_array_points(
                        #     v_list,   
                        #     f_list,   
                        #     rot_list=[[0,0,0]] * len_v,
                        #     size=1, bg_black=False, mode='shade',
                        #     logdir=save_logdir,
                        #     name=save_img_name, save=True,
                        #     points_list=C_list,  
                        #     points_color='r',
                        #     points_size=2,
                        # )
                        vis_mesh_with_joint_points(
                        verts=pred_vertices[0],
                        faces=faces,
                        C=C_lbs[0],
                        yrot=20,
                        save_path=f"{save_logdir}/{save_img_name}.png"
                        )
                    # elif self.opts.lib == 'o3d':
                    #     save_dir = f"{save_logdir}-o3d"
                    #     os.makedirs(save_dir, exist_ok=True)
                    #     save_path = f"{save_dir}/{save_img_name}.png"
                    #     vis_frame_o3d_3views_save(
                    #         V=pred_vertices[0],   # (N,3)
                    #         F=faces,
                    #         C=C_lbs[0],           # (J,3)
                    #         save_path=save_path  # ← 핵심
                    #     )
                    # elif self.opts.lib == 'pyv':
                    #     import pdb;pdb.set_trace()
                    #     save_dir = f"{save_logdir}-pyv"
                    #     os.makedirs(save_dir, exist_ok=True)
                    #     save_path = f"{save_dir}/{save_img_name}.png"
                    #     vis_frame_pyvista_3views(
                    #         V=pred_vertices[0],   # (N,3)
                    #         F=faces,
                    #         C=C_lbs[0],           # (J,3)
                    #         screenshot=save_path  # ← 핵심
                    #     )
            
            else:
                plot_image_array(
                    v_list, f_list, 
                    rot_list=[[0,0,0]]*len_v,
                    size=1, bg_black=False, mode='shade', 
                    logdir=save_logdir, 
                    name=save_img_name, save=True
                )
        ##########################################################################################################
        # make video
        if self.opts.use_weighted_joint_pos == False: # only make video if joint position is fixed
            if self.opts.vis_joint_pos:
                vid_name = f"{self.opts.log_dir}/animation_visjoint.mp4"
            else:
                vid_name = f"{self.opts.log_dir}/animation.mp4"
            images_to_video_cv(
            save_logdir,
            vid_name,
            fps=30
            )
        print('done!')

    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    
    def visLBS2(self):
        """
            self-retargeting task
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
            
        os.makedirs(self.opts.log_dir, exist_ok=True)
        save_logdir = None
        if self.opts.vis_joint_pos:
            save_logdir = f"{self.opts.log_dir}/img-visjoint"
            os.makedirs(save_logdir, exist_ok=True)
        else:
            save_logdir = f"{self.opts.log_dir}/img-full"
            os.makedirs(save_logdir, exist_ok=True)
            
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
        check_usage = False
        
        len_data = len(self.dataloader)
        denom = 1 / len_data
        
        self.model.eval()    
        mesh_data = self.dataset.data_name
        
        pbar = tqdm(enumerate(self.dataloader), total=len_data, ncols=100)
        for index, batch in pbar:                
            # model forward ----------------------------------------------------------------------------------
            with torch.no_grad():
                src_verts = batch.template[0]
                src_faces = batch.faces[0]
                src_m = trimesh.Trimesh(vertices=src_verts.cpu().numpy(), faces=src_faces.cpu().numpy())
                # LBS-only forward
                if self.opts.vis_joint_pos:
                    if self.opts.vis_partition_per_joint and index == 0:
                        pred_vertices, _, _, _, _, _, _, _, W_lbs, _, C_lbs = self.model(
                            batch.template,
                            batch.vertices,
                            batch.template_normal,
                            batch.vertices_normal,
                            mesh_data=batch.mesh_data, epoch=0, 
                            out_kw=True, stage=1 # 1: LBS stage 2: LBS + CBD 
                        )
                    else:                    
                        pred_vertices, _, _, _, _, _, _, _, _, _, C_lbs = self.model(
                            batch.template,
                            batch.vertices,
                            batch.template_normal,
                            batch.vertices_normal,
                            mesh_data=batch.mesh_data, epoch=0, 
                            out_kw=True, stage=1 # 1: LBS stage 2: LBS + CBD 
                        )
                else:
                    pred_vertices, _, _, _, _, _, _, _, _, _ = self.model(
                        batch.template,
                        batch.vertices,
                        batch.template_normal,
                        batch.vertices_normal,
                        mesh_data=batch.mesh_data, epoch=0, 
                        out_kw=True, stage=1 # 1: LBS stage 2: LBS + CBD 
                    )
                HB = batch.vertices.shape[0] // 2
            # ------------------------------------------------------------------------------------------------

            if self.opts.vis_partition_per_joint and index == 0:
                save_joint_weight_logdir = f"{self.opts.log_dir}/img-visweight"
                os.makedirs(save_joint_weight_logdir, exist_ok=True)
                num_joints = W_lbs[0].shape[-1]
                print("saving joint weights... \n")
                W_all = W_lbs[0].detach().cpu().numpy()     # (N,J)
                Wf_all = W_all[src_faces.detach().cpu().numpy()].mean(axis=1)      # (F,J)
                vmin_global = 0.0
                vmax_global = np.percentile(Wf_all, 99.5)
                for nj in range(num_joints):
                    vis_mesh_key_weight(
                    verts=pred_vertices[0].detach().cpu().numpy(),     # (N,3)
                    faces=src_faces.detach().cpu().numpy(),                # (F,3)
                    key_weight=W_lbs[0].detach().cpu().numpy(),         # (N,K)
                    cage_idx=nj,              # int
                    yrot=20,
                    vmin=vmin_global, vmax=vmax_global,
                    view_yrots=(0, 90, 180),
                    save_path=f"{save_joint_weight_logdir}/joint_{nj:03d}.png",
                    )
                print("Done ! \n")
            # import pdb;pdb.set_trace()

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
            f_list=[faces] * len_v # 복사해주기 (v_list 갯수만큼)
            save_img_name = f"{index:04d}"

            if self.opts.vis_joint_pos:
                C_lbs = C_lbs.detach().cpu() # keep torch ok (plot 내부에서 numpy로 바꿔도 됨)
                if self.opts.use_weighted_joint_pos: 
                    if self.opts.lib == 'mpl':
                        # C_list = [
                        #     None, # GT 에는 안찍기
                        #     C_lbs[0],
                        # ]
                        # plot_image_array_points(
                        #     v_list,   
                        #     f_list,   
                        #     rot_list=[[0,0,0]] * len_v,
                        #     size=1, bg_black=False, mode='shade',
                        #     logdir=save_logdir,
                        #     name=save_img_name, save=True,
                        #     points_list=C_list,  
                        #     points_color='r',
                        #     points_size=2,
                        # )
                        vis_mesh_with_joint_points(
                        verts=pred_vertices[0],
                        faces=faces,
                        C=C_lbs[0],
                        yrot=0,
                        save_path=f"{save_logdir}/{save_img_name}.png"
                        )
                    # elif self.opts.lib == 'o3d':
                    #     save_dir = f"{save_logdir}-o3d"
                    #     os.makedirs(save_dir, exist_ok=True)
                    #     save_path = f"{save_dir}/{save_img_name}.png"
                    #     vis_frame_o3d_3views_save(
                    #         V=pred_vertices[0],   # (N,3)
                    #         F=faces,
                    #         C=C_lbs[0],           # (J,3)
                    #         save_path=save_path  # ← 핵심
                    #     )
                    # elif self.opts.lib == 'pyv':
                    #     import pdb;pdb.set_trace()
                    #     save_dir = f"{save_logdir}-pyv"
                    #     os.makedirs(save_dir, exist_ok=True)
                    #     save_path = f"{save_dir}/{save_img_name}.png"
                    #     vis_frame_pyvista_3views(
                    #         V=pred_vertices[0],   # (N,3)
                    #         F=faces,
                    #         C=C_lbs[0],           # (J,3)
                    #         screenshot=save_path  # ← 핵심
                    #     )
                    break # iterate only once, use_weighted_joint_pos means fixed position
                else:
                    if self.opts.lib == 'mpl':
                        # C_list = [
                        #     None, # GT 에는 안찍기
                        #     C_lbs[0],
                        # ]
                        # plot_image_array_points(
                        #     v_list,   
                        #     f_list,   
                        #     rot_list=[[0,0,0]] * len_v,
                        #     size=1, bg_black=False, mode='shade',
                        #     logdir=save_logdir,
                        #     name=save_img_name, save=True,
                        #     points_list=C_list,  
                        #     points_color='r',
                        #     points_size=2,
                        # )
                        vis_mesh_with_joint_points(
                        verts=pred_vertices[0],
                        faces=faces,
                        C=C_lbs[0],
                        yrot=0,
                        save_path=f"{save_logdir}/{save_img_name}.png"
                        )
                    # elif self.opts.lib == 'o3d':
                    #     save_dir = f"{save_logdir}-o3d"
                    #     os.makedirs(save_dir, exist_ok=True)
                    #     save_path = f"{save_dir}/{save_img_name}.png"
                    #     vis_frame_o3d_3views_save(
                    #         V=pred_vertices[0],   # (N,3)
                    #         F=faces,
                    #         C=C_lbs[0],           # (J,3)
                    #         save_path=save_path  # ← 핵심
                    #     )
                    # elif self.opts.lib == 'pyv':
                    #     import pdb;pdb.set_trace()
                    #     save_dir = f"{save_logdir}-pyv"
                    #     os.makedirs(save_dir, exist_ok=True)
                    #     save_path = f"{save_dir}/{save_img_name}.png"
                    #     vis_frame_pyvista_3views(
                    #         V=pred_vertices[0],   # (N,3)
                    #         F=faces,
                    #         C=C_lbs[0],           # (J,3)
                    #         screenshot=save_path  # ← 핵심
                    #     )
                        
            else:
                plot_image_array(
                    v_list, f_list, 
                    rot_list=[[0,0,0]]*len_v,
                    size=1, bg_black=False, mode='shade', 
                    logdir=save_logdir, 
                    name=save_img_name, save=True
                )
        
        ##########################################################################################################
        # make video
        if self.opts.use_weighted_joint_pos == False: # only make video if joint position is fixed
            if self.opts.vis_joint_pos:
                vid_name = f"{self.opts.log_dir}/animation_visjoint.mp4"
            else:
                vid_name = f"{self.opts.log_dir}/animation.mp4"
            images_to_video_cv(
            save_logdir,
            vid_name,
            fps=30
            )
        print("done")

    
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
    for '--data_selection'
    #### voca | biwi | mf_SEN | coma | mf_ROM | mf_all |
    ####   0  |   1  |    2   |   3  |    4   |    5   |
    
    ###############
    ## run commands
        ## on PCA
    python vis_LBS.py  --vis_partition_per_joint --vis_joint_pos --data_selection 3 --continue_ckpt --start_epoch 200 --ckpt "./ckpts_CBD/2026-01-21-12-59-49-NGBC++v5" --version 3 --batch_size 1 --use_t_mask
        ## on Real 
    ## 64 joints
    python vis_LBS.py  --vis_partition_per_joint --vis_joint_pos --realtest --data_selection 3 --continue_ckpt --start_epoch 200 --ckpt "./ckpts_CBD/2026-01-21-12-59-49-NGBC++v5" --version 3 --batch_size 1 --use_t_mask
    ## 32 joints
    python vis_LBS.py  --vis_partition_per_joint --vis_joint_pos --realtest --data_selection 3 --continue_ckpt --start_epoch 300 --ckpt "./ckpts_CBD/2026-01-21-12-59-22-NGBC++v5" --version 3 --batch_size 1 --use_t_mask

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
    
    print('loaded version:', opts.version)
    
    ## load model
    trainer = Trainer(opts)
    
    if opts.realtest:
        trainer.visLBS2() 
    else:
        trainer.visLBS()
        

