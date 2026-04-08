"""
eval_comp_v2.py — NFS/NFR evaluation (based on vis_CBD.py, confirmed jitter-free).

Usage:
    # NFR self-retarget (mf_ROM id12)
    python eval_comp_v2.py --NFR --src_data 4 --src_id 12 --tgt_data 4 --tgt_id 12 --make_video

    # NFR cross-retarget (mf_ROM id12 → ict id9)
    python eval_comp_v2.py --NFR --src_data 4 --src_id 12 --tgt_data 5 --tgt_id 9 --make_video

    # NFS self-retarget
    python eval_comp_v2.py --ckpt ckpts_comparison/NFS-best --src_data 4 --src_id 12 --tgt_data 4 --tgt_id 12 --make_video

    # NFS cross-retarget
    python eval_comp_v2.py --ckpt ckpts_comparison/NFS-best --src_data 4 --src_id 12 --tgt_data 5 --tgt_id 9 --make_video
"""
import os
import glob
import json
import yaml
import random
import cv2

import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
import trimesh
import igl
import pickle


import torch
import torch.nn.functional as F

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

from utils.remesh_utils import ICT_face_model
import torch.multiprocessing as mp


# ── Metric helpers ──────────────────────────────────────────────────────────

def _build_cot_laplacian(verts_np, faces_np):
    return igl.cotmatrix(verts_np.astype(np.float64), faces_np.astype(np.int32))

def _laplacian_error(L_sp, pred, gt):
    B = pred.shape[0]
    total = 0.0
    for b in range(B):
        Lp = L_sp @ pred[b].numpy().astype(np.float64)
        Lg = L_sp @ gt[b].numpy().astype(np.float64)
        total += float(np.mean(np.sum((Lp - Lg) ** 2, axis=-1)))
    return total / B

def _normal_consistency(pred, gt, faces):
    pred_n = calc_norm_torch(pred, faces)
    gt_n = calc_norm_torch(gt, faces)
    cos_sim = F.cosine_similarity(pred_n, gt_n, dim=-1)
    return (1.0 - cos_sim).mean().item()

def _edge_length_distortion(pred, gt, edges):
    e0, e1 = edges[:, 0], edges[:, 1]
    len_pred = torch.sqrt(((pred[:, e0] - pred[:, e1]) ** 2).sum(dim=-1))
    len_gt = torch.sqrt(((gt[:, e0] - gt[:, e1]) ** 2).sum(dim=-1))
    return ((len_pred - len_gt).abs() / (len_gt + 1e-8)).mean().item()

def _build_edges(faces_np):
    e = np.concatenate([faces_np[:, [0,1]], faces_np[:, [1,2]], faces_np[:, [0,2]]], axis=0)
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    return torch.tensor(e, dtype=torch.long)

def images_to_video(img_dir, out_path, fps=30):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not imgs:
        print(f"[WARN] No images in {img_dir}")
        return
    first = cv2.imread(imgs[0])
    h, w, _ = first.shape
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in imgs:
        writer.write(cv2.imread(p))
    writer.release()
    print(f"Video saved: {out_path}")


def Options():
    parser = argparse.ArgumentParser(description='neural generalized barycentric coordinate for FA retargeting')
    parser.add_argument('-c', '--config', default='config/train_CBD.yml', help='config file path')
    parser.add_argument("--device",       type=str,   default="cuda:0")
    
    parser.add_argument("--log_dir",      type=str,   default="eval_comp")

    parser.add_argument("--version",      type=int,   default=0,      help='0: NFS/NFR')

    # Source / Target (CLI control)
    parser.add_argument("--src_data", type=int, default=4,
                        help='Source dataset (0:voca 1:biwi 2:mf_SEN 3:coma 4:mf_ROM 5:ict 6:ict-cap)')
    parser.add_argument("--src_id", type=int, default=12, help='Source identity index')
    parser.add_argument("--tgt_data", type=int, default=4,
                        help='Target dataset (same encoding). If same as src_data+src_id → self-retarget')
    parser.add_argument("--tgt_id", type=int, default=12, help='Target identity index')
    parser.add_argument("--exp_num", type=int, default=0, help='Expression number (for ict-cap)')

    parser.add_argument("--batch_size",   type=int,   default=1)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--ckpt",         type=str,   default=None)
    parser.add_argument("--continue_ckpt",dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)

    parser.add_argument("--use_t_mask",dest='use_t_mask', action='store_true')
    parser.set_defaults(use_t_mask=True)

    parser.add_argument("--save_vert",dest='save_vert', action='store_true')
    parser.set_defaults(save_vert=False)
    parser.add_argument("--save_gt",dest='save_gt', action='store_true')
    parser.set_defaults(save_gt=False)

    parser.add_argument("--NFR",dest='NFR', action='store_true')
    parser.set_defaults(NFR=False)

    parser.add_argument("--max_frames", type=int, default=-1, help='Max frames (-1=all)')
    parser.add_argument("--make_video", dest='make_video', action='store_true')
    parser.set_defaults(make_video=False)
    parser.add_argument("--no_vis", dest='no_vis', action='store_true')
    parser.set_defaults(no_vis=False)

    parser.add_argument("--use_NFR",dest='use_NFR', action='store_true')
    parser.set_defaults(use_NFR=False)
    parser.add_argument("--optim_cage",dest='optim_cage', action='store_true')
    parser.set_defaults(optim_cage=False)
    parser.add_argument("--align_latent",dest='align_latent', action='store_true')
    parser.set_defaults(align_latent=False)
    parser.add_argument("--tb", action='store_true')
    parser.set_defaults(is_train=True)
    parser.add_argument("--use_eval_data2",dest='use_eval_data2', action='store_true')
    parser.set_defaults(use_eval_data2=False)
    parser.add_argument("--realtest",dest='realtest', action='store_true')
    parser.set_defaults(realtest=False)

    args = parser.parse_args()
    return args

class Pipeline():
    def __init__(self, opts):
        # set opts
        self.opts = opts
        self.set_seed(self.opts)
        self.device = opts.device
        
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
        """Run single eval from CLI args."""
        opts = self.opts
        self.save_vis_data(opts.src_data, opts.src_id, opts.tgt_data, opts.tgt_id, opts.exp_num)
        
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
                
        model_tag = 'NFR' if opts.NFR else 'NFS'
        self.opts.log_dir = f'./{opts.log_dir}/{model_tag}/{file_name}'
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
        
        
        if os.path.exists(self.opts.log_dir_vert) and opts.save_vert:
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
                        if opts.max_frames > 0 and index >= opts.max_frames:
                            break
                        if index==0:
                            src_verts = batch.template[0].cpu().numpy()
                            src_faces = batch.faces[0].cpu().numpy()
                            src_m = trimesh.Trimesh(vertices=src_verts, faces=src_faces)
                            src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
                            src_operators = get_mesh_operators(src_m)
                            src_img = self.model.renderer.render_img(src_m).float().to(device)
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
                            pbar.set_description(f"NFS self | MSE: {losses_val['MSE']:.5e}")

                            pred_outputs_np = pred_outputs.detach().cpu().numpy()

                            CurrBS=pred_outputs_np.shape[0]
                            for b_idx in range(CurrBS):
                                if opts.save_vert:
                                    save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
                                    np.save(save_out_name, pred_outputs_np[b_idx])
                                if self.opts.save_gt:
                                    save_gt_name = GT_log_dir+f'/{index*CurrBS+b_idx:06d}.npy'
                                    np.save(save_gt_name, batch.vertices[b_idx].cpu().numpy())
                                if not opts.no_vis:
                                    _faces_cpu = batch.faces[0].cpu()
                                    v_list = [batch.vertices[b_idx].cpu(), batch.template[0].cpu(),
                                              torch.tensor(pred_outputs_np[b_idx])]
                                    f_list = [_faces_cpu] * 3
                                    plot_image_array(v_list, f_list, rot_list=[[0,0,0]]*3,
                                                     size=1, bg_black=False, mode='shade',
                                                     logdir=f"{self.opts.log_dir}/img",
                                                     name=f"{index*CurrBS+b_idx:06d}", save=True)
                    if opts.make_video:
                        images_to_video(f"{self.opts.log_dir}/img",
                                        os.path.join(self.opts.log_dir, "eval_nfs_self.mp4"))
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
                        if opts.max_frames > 0 and index >= opts.max_frames:
                            break
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
                            vert_feat_exp = torch.vstack(vert_feat_exp) * 1.3
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
                                if opts.save_vert:
                                    np.save(self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy', pred_outputs_np[b_idx])
                                if not opts.no_vis:
                                    _faces_cpu = batch.faces[0].cpu()
                                    _tgt_faces_cpu = torch.tensor(tgt_f).long() if isinstance(tgt_f, np.ndarray) else tgt_faces.cpu()
                                    v_list = [batch.vertices[b_idx].cpu(), torch.tensor(pred_outputs_np[b_idx])]
                                    f_list = [_faces_cpu, _tgt_faces_cpu]
                                    plot_image_array(v_list, f_list, rot_list=[[0,0,0]]*2,
                                                     size=1, bg_black=False, mode='shade',
                                                     logdir=f"{self.opts.log_dir}/img",
                                                     name=f"{index*CurrBS+b_idx:06d}", save=True)
                    if opts.make_video:
                        images_to_video(f"{self.opts.log_dir}/img",
                                        os.path.join(self.opts.log_dir, "eval_nfs_cross.mp4"))
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
            #                     pbar.set_description('src chng!?')
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
            #         tgt_img = self.model.renderer.render_img(tgt_m).float().to(device)
            #         tgt_operators = self.model.get_mesh_operators(tgt_m)
                    
            #         pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
            #         for index, batch in pbar:
            #             if index==0:
            #                 src_m = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())

            #                 src_img = self.model.renderer.render_img(src_m).float().to(device)
            #                 src_img_feat = self.model.get_img_feat(src_img)[None]
            #                 src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
            #             else:
            #                 if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:
            #                     pbar.set_description('src chng!?')
            #                     src_m = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())
            #                     src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
            #                     src_img = self.model.renderer.render_img(src_m).float().to(device)
            #                     src_img_feat = self.model.get_img_feat(src_img)

            #             with torch.no_grad():
            #                 vert_feat_exp = []
            #                 for b_v in batch.vertices:
            #                     _tmp_ = self.model.get_local_feature(b_v[None], batch.faces[0], src_img_feat).float()
            #                     vert_feat_exp.append(_tmp_)
            #                 vert_feat_exp = torch.vstack(vert_feat_exp)
            #                 pred_exp_coeff = self.model.encode_exp(
            #                     vert_feat_exp, src_dfn_info, batch_process=True, verbose=False
            #                 )# [W, Rig]

            #                 inputs = (
            #                     tgt_tri_feat if self.opts.dec_type=='jacob' else tgt_vert_feat,
            #                     pred_exp_coeff, pred_id_coeff, pred_seg_coeff,
            #                     None, tgt_verts[None], tgt_faces, tgt_operators
            #                 )
            #                 pred_outputs, _ = self.model.decode(inputs, batch_process=True)

            #                 pred_outputs_np = pred_outputs.detach().cpu().numpy()

            #                 CurrBS=pred_outputs_np.shape[0]
            #                 for b_idx in range(CurrBS):
            #                     save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
            #                     np.save(save_out_name, pred_outputs_np[b_idx])
            else:
                ## from NFR checkpoint
                if SELF_RETARGET:
                    print('self-retargeting! (src == tgt)')
                    L_sp, edges = None, None
                    total = {"mse": 0.0, "mse_in": 0.0, "mse_out": 0.0, "l2": 0.0, "lap": 0.0,
                             "norm_cos": 0.0, "edge_dist": 0.0, "l2_max_sum": 0.0}
                    all_pv = []
                    n_frames = 0

                    pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
                    for index, batch in pbar:
                        if opts.max_frames > 0 and index >= opts.max_frames:
                            break
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
                            inputs_v = self.model.get_inputs(batch.vertices, batch.faces[0])
                            self.model.model.update_precomputes(src_dfn_info)
                            pred_exp = self.model.model.encode(inputs_v, src_img.to(device), N_F=src_m.faces.shape[0])
                            pred_outputs, _, _ = self.model.calc_new_mesh(
                                src_verts, src_faces, pred_exp, src_operators, src_dfn_info, src_img
                            )
                            pred_outputs_np = pred_outputs.detach().cpu().numpy()
                            gt_v = batch.vertices.cpu()
                            pred_cpu = pred_outputs.cpu()

                            # Metrics
                            if L_sp is None:
                                _fnp = src_faces.cpu().numpy()
                                L_sp = _build_cot_laplacian(src_verts.cpu().numpy(), _fnp)
                                edges = _build_edges(_fnp)
                            faces_t = torch.tensor(src_faces.cpu().numpy(), dtype=torch.long)
                            B = gt_v.shape[0]
                            for b in range(B):
                                _gt, _pr = gt_v[b:b+1], pred_cpu[b:b+1]
                                total["mse"] += F.mse_loss(_gt, _pr).item()
                                if opts.use_t_mask:
                                    t_mask = plateau_hat_points(src_verts.unsqueeze(0).cpu())
                                    inv_mask = 1.0 - t_mask
                                    total["mse_in"] += F.mse_loss(_gt * t_mask, _pr * t_mask).item()
                                    total["mse_out"] += F.mse_loss(_gt * inv_mask, _pr * inv_mask).item()
                                pv_l2 = torch.sqrt(((_gt - _pr) ** 2).sum(dim=-1))
                                total["l2"] += pv_l2.mean().item()
                                total["l2_max_sum"] += pv_l2.max(dim=-1).values.mean().item()
                                all_pv.append(pv_l2.numpy())
                                total["lap"] += _laplacian_error(L_sp, _pr, _gt)
                                total["norm_cos"] += _normal_consistency(_pr, _gt, faces_t)
                                total["edge_dist"] += _edge_length_distortion(_pr, _gt, edges)
                                n_frames += 1

                            CurrBS=pred_outputs_np.shape[0]
                            for b_idx in range(CurrBS):
                                if opts.save_vert:
                                    np.save(self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy', pred_outputs_np[b_idx])
                                if opts.save_gt:
                                    os.makedirs(self.opts.log_dir+'/GT_verts', exist_ok=True)
                                    np.save(self.opts.log_dir+f'/GT_verts/{index*CurrBS+b_idx:06d}.npy', gt_v[b_idx].numpy())
                                if not opts.no_vis:
                                    _faces_cpu = batch.faces[0].cpu()
                                    v_list = [gt_v[b_idx], batch.template[0].cpu(), torch.tensor(pred_outputs_np[b_idx])]
                                    f_list = [_faces_cpu] * 3
                                    plot_image_array(v_list, f_list, rot_list=[[0,0,0]]*3,
                                                     size=1, bg_black=False, mode='shade',
                                                     logdir=f"{self.opts.log_dir}/img",
                                                     name=f"{index*CurrBS+b_idx:06d}", save=True)

                        pbar.set_description(f"NFR self | mse: {total['mse']/max(n_frames,1):.5e}")

                    # Save metrics
                    if n_frames > 0:
                        inv = 1.0 / n_frames
                        all_pv_np = np.concatenate(all_pv, axis=0).flatten()
                        results = {
                            "model": "nfr", "mode": "self",
                            "MSE": total["mse"]*inv, "MSE_inner": total["mse_in"]*inv,
                            "MSE_outer": total["mse_out"]*inv, "L2_mean": total["l2"]*inv,
                            "L2_max_mean": total["l2_max_sum"]*inv,
                            "L2_median": float(np.median(all_pv_np)),
                            "L2_p95": float(np.percentile(all_pv_np, 95)),
                            "L2_p99": float(np.percentile(all_pv_np, 99)),
                            "L2_max": float(np.max(all_pv_np)),
                            "Laplacian_err": total["lap"]*inv,
                            "Normal_cos_dist": total["norm_cos"]*inv,
                            "Edge_len_distortion": total["edge_dist"]*inv,
                            "num_frames": n_frames,
                        }
                        with open(os.path.join(self.opts.log_dir, "results.json"), 'w') as f:
                            json.dump(results, f, indent=4)
                        print(f"\nResults saved: {self.opts.log_dir}/results.json")
                        for k, v in results.items():
                            if isinstance(v, float): print(f"  {k}: {v:.6e}")
                        np.save(os.path.join(self.opts.log_dir, "per_vertex_l2.npy"), all_pv_np)
                    if opts.make_video:
                        images_to_video(f"{self.opts.log_dir}/img",
                                        os.path.join(self.opts.log_dir, "eval_nfr_self.mp4"))
                else:
                    print('cross-retargeting!')
                    tgt_m = trimesh.Trimesh(vertices=tgt_v, faces=tgt_f)
                    tgt_dfn_info = nfr_utils.get_dfn_info(tgt_m, map_location=device)
                    tgt_verts = tgt_v_th[0]
                    tgt_faces = torch.from_numpy(tgt_m.faces).to(device)
                    tgt_img = self.model.renderer.render_img(tgt_m).float().to(device)
                    tgt_operators = self.model.get_mesh_operators(tgt_m)
                    
                    pbar = tqdm(enumerate(src_dataloader), total=len_dataloader, ncols=100)
                    for index, batch in pbar:
                        if opts.max_frames > 0 and index >= opts.max_frames:
                            break
                        if index==0:
                            src_m = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())

                            src_img = self.model.renderer.render_img(src_m).float().to(device)
                            src_img_feat = self.model.get_img_feat(src_img)[None]
                            src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
                        else:
                            if (batch.template[0].cpu().numpy() - src_m.vertices).mean() != 0:
                                src_m = trimesh.Trimesh(vertices=batch.template[0].cpu().numpy(), faces=batch.faces[0].cpu().numpy())

                                src_img = self.model.renderer.render_img(src_m).float().to(device)
                                src_img_feat = self.model.get_img_feat(src_img)[None]
                                src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)

                        with torch.no_grad():
                            inputs_v = self.model.get_inputs(batch.vertices, batch.faces[0])# [B, V, 3+3]

                            ## get expression
                            self.model.model.update_precomputes(src_dfn_info)
                            pred_exp = self.model.model.encode(inputs_v, src_img.to(device), N_F=src_m.faces.shape[0])
                            pred_outputs, _, _ = self.model.calc_new_mesh(
                                tgt_verts, tgt_faces, pred_exp, tgt_operators, tgt_dfn_info, tgt_img
                            )
                            pred_outputs_np = pred_outputs.detach().cpu().numpy()

                            CurrBS=pred_outputs_np.shape[0]
                            for b_idx in range(CurrBS):
                                save_out_name = self.opts.log_dir_vert+f'/{index*CurrBS+b_idx:06d}.npy'
                                np.save(save_out_name, pred_outputs_np[b_idx])

                                # Save visualization image (cross-retarget NFR)
                                _faces_cpu = batch.faces[0].cpu() if batch.faces.dim() == 3 else batch.faces.cpu()
                                _tgt_faces_cpu = torch.tensor(tgt_f).long() if isinstance(tgt_f, np.ndarray) else tgt_f.cpu()
                                v_list = [
                                    batch.vertices[b_idx].cpu(),
                                    torch.tensor(pred_outputs_np[b_idx]),
                                ]
                                f_list = [_faces_cpu, _tgt_faces_cpu]
                                img_save_dir = self.opts.log_dir_vert.replace('/verts', '/img')
                                os.makedirs(img_save_dir, exist_ok=True)
                                plot_image_array(
                                    v_list, f_list,
                                    rot_list=[[0,0,0]]*len(v_list),
                                    size=1, bg_black=False, mode='shade',
                                    logdir=img_save_dir,
                                    name=f"{index*CurrBS+b_idx:06d}", save=True)
                    if opts.make_video:
                        images_to_video(f"{self.opts.log_dir}/img",
                                        os.path.join(self.opts.log_dir, "eval_cross.mp4"))
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
    # NFR self-retarget (mf_ROM id12)
    python eval_comp_v2.py --NFR --src_data 4 --src_id 12 --tgt_data 4 --tgt_id 12 --make_video --max_frames 200

    # NFR cross-retarget (mf_ROM id12 → ict id9)
    python eval_comp_v2.py --NFR --src_data 4 --src_id 12 --tgt_data 5 --tgt_id 9 --make_video --max_frames 200

    # NFS self-retarget
    python eval_comp_v2.py --ckpt ckpts_comparison/NFS-best --src_data 4 --src_id 12 --tgt_data 4 --tgt_id 12 --make_video

    # NFS cross-retarget
    python eval_comp_v2.py --ckpt ckpts_comparison/NFS-best --src_data 4 --src_id 12 --tgt_data 5 --tgt_id 9 --make_video
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

