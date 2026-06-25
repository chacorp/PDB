import os
import glob
import json
import yaml
import numpy as np
import torch
import argparse
import igl
import pickle
import trimesh

import sys
from pathlib import Path
__abs_path__ = str(Path.cwd().parents[0].absolute())
__utils_path__ = f'{__abs_path__}/utils'

for __util_path__ in [__abs_path__, __utils_path__]:
    if not __util_path__ in sys.path:
        sys.path+=[__util_path__]

# from matplotrender import *
from utils.matplotlib_rnd import (
    render_mesh_diff,
    plot_image_array
)

from dataloader_CBD import (
    EvalDataset,
    CBDDataBatch_eval,
    CBD_collate_wrapper_eval,
)
from functools import partial
from tqdm import tqdm
from glob import glob
import igl

from tqdm import tqdm
from PIL import Image

import ipywidgets as widgets
import matplotlib.pyplot as plt

def get_mesh(selection, dataset, SELECT_MESH):
    if selection=='ict'or selection=='ict-cap':
        # id_vecs = np.load(f'{__abs_path__}/data/ICT_live_100/iden_vecs.npy')
        # id_vecs = np.load(f'{__abs_path__}/ict_face_pt/random_identity_vecs.npy')[:101]
        id_vecs = torch.load(f'{__abs_path__}/ict_face_pt/ict_id_vecs_test.pt').numpy()
        id_disps = dataset.ict_face_model.get_id_disp(id_vecs[SELECT_MESH])
        mesh_v = dataset.ict_face_model.neutral_verts.squeeze() + id_disps.squeeze()
        
        return mesh_v, dataset.ict_face_model.faces
    else:
        if selection=='voca':
            mesh = dataset.voca_mesh
        elif selection=='biwi':
            # mesh = dataset.biwi_mesh            
            biwi_trimesh = trimesh.load(f'{__abs_path__}/test-mesh/BIWI.ply')
            with open(f'{__abs_path__}/test-mesh/biwi_templates.pkl', 'rb') as f:
                mesh = pickle.load(f)
                
            mesh['face']=biwi_trimesh.faces
        elif selection=='mf_SEN':
            mesh = dataset.mf_SEN_mesh
        elif selection=='coma':
            mesh = dataset.coma_mesh
        elif selection=='mf_ROM':
            mesh = dataset.mf_ROM_mesh

        mesh_list = [idname for idname in mesh.keys() if idname!='face']
        mesh_v = mesh[mesh_list[SELECT_MESH]]

        if selection=='biwi':
            m_align = np.load(f'{__abs_path__}/utils/biwi/align.npy')
            mesh_v = np.concatenate((mesh_v,np.ones((mesh_v.shape[0],1))), axis=1) @ m_align.T
        return mesh_v, mesh['face']





data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict','ict-cap'] # 0 1 2 3 4 5

SRC_SELECT_DATA=4
SRC_SELECT_MESH=12
EXP_NUM=0

TGT_SELECT_DATA=6
TGT_SELECT_mesh=6

src_selection = data_name_list[SRC_SELECT_DATA]
tgt_selection = data_name_list[TGT_SELECT_DATA]

device='cuda:0'

src_dataset = EvalDataset(data_name=src_selection, toggle=False) # if eve-s01
src_v, src_f = get_mesh(src_selection, src_dataset, SRC_SELECT_MESH)
src_n = igl.per_vertex_normals(src_v, src_f)

tgt_dataset = EvalDataset(data_name=tgt_selection, toggle=False) # if eve-s01
tgt_v, tgt_f = get_mesh(tgt_selection, tgt_dataset, TGT_SELECT_mesh)
tgt_n = igl.per_vertex_normals(tgt_v, tgt_f)

M_SCALE = 0.5
print(src_v.shape, tgt_v.shape)
v_list=[ src_v, tgt_v ]
f_list=[ src_f, tgt_f ]
SIZE=3
rot_list=[[0,-10,0]]

# plot_mesh_gouraud(v_list, f_list, mesh_scale=M_SCALE,
#         rot_list=rot_list*len(v_list), size=SIZE, mode='shade', bg_black=False)



# v_list=[v * M_SCALE for v in v_list]
# rot_list=[[0,-10,0]]
# plot_image_array(
#     v_list, f_list,
#     rot_list=rot_list*len(v_list), 
#     size=SIZE, bg_black=False, mode='shade', save=True,
#     logdir=f"fig/", name=f"{file_name}-id_{TGT_SELECT_mesh:02d}-neutral", 
# )



#######################################################
ckpt_path = '2025-10-09-02-04-49-NGBCv5' ### [chosen] comparison
# ckpt_path = '2025-10-11-23-12-33-NGBCv5'
# ckpt_path = '2025-10-09-02-04-49-NGBCv5_'
# ckpt_path = '2025-09-29-14-48-17-NGBCv8'
# ckpt_path = '2025-09-27-12-04-29-NGBCv5'
# ckpt_path = '2025-09-27-08-05-11-NGBCv5'
if 'ict' in src_selection:
    file_name = f'{src_selection}-ID_{SRC_SELECT_MESH:03d}_test-to-{tgt_selection}-ID_{TGT_SELECT_mesh:03d}_test_{EXP_NUM:02d}'
elif 'ict' in tgt_selection:
    # file_name = f'{src_selection}-ID_{SRC_SELECT_MESH:03d}_test-to-{tgt_selection}_test-ID_{TGT_SELECT_mesh:03d}'
    file_name = f'{src_selection}_test-to-ict_test-ID_{TGT_SELECT_mesh:03d}'
else:
    if 'mf' in tgt_selection and TGT_SELECT_mesh < 11:
        file_name = f'{src_selection}_test-to-{tgt_selection}_train'
    else:
        file_name = f'{src_selection}_test-to-{tgt_selection}_test'
# file_name = f'{src_selection}_test-to-{tgt_selection}_val'
# file_name = f'{src_selection}_test-to-{tgt_selection}_train'
#######################################################

a2b_retarget_path = __abs_path__ + '/eval_CBD/'+ckpt_path+'/'+file_name
print(a2b_retarget_path)
mesh_file_paths = sorted(glob(a2b_retarget_path+'/*.npy'))
len_mesh_file_paths=len(mesh_file_paths)
print(len_mesh_file_paths)

a2b_retarget_path = __abs_path__ + '/eval_CBD/nfr/'+file_name
print(a2b_retarget_path)
nfr_mesh_file_paths = sorted(glob(a2b_retarget_path+'/*.npy'))
len_nfr_mesh_file_paths=len(nfr_mesh_file_paths)
print(len_nfr_mesh_file_paths)

a2b_retarget_path = __abs_path__ + '/eval_CBD/nfs/'+file_name
print(a2b_retarget_path)
nfs_mesh_file_paths = sorted(glob(a2b_retarget_path+'/*.npy'))
len_nfs_mesh_file_paths=len(nfs_mesh_file_paths)
print(len_nfs_mesh_file_paths)

if src_selection == 'ict-cap':
    GT_file_name=f'{src_selection}_test-to-{src_selection}_test_{EXP_NUM:02d}'    
else:
    GT_file_name=f'{src_selection}_test-to-{src_selection}_test'
    
GT_a2b_retarget_path = __abs_path__ + '/eval_CBD/GT-'+GT_file_name
print(GT_a2b_retarget_path)
GT_mesh_file_paths = sorted(glob(GT_a2b_retarget_path+'/*.npy'))
GT_len_mesh_file_paths=len(GT_mesh_file_paths)
print(GT_len_mesh_file_paths)

# source/sihun/NeuralFacialAnimation/eval_CBD/nfs/mf_ROM_test-to-ict_test-ID_000
# source/sihun/NeuralFacialAnimation/eval_CBD/nfr/mf_ROM_test-to-ict_test-ID_000

print('total_len:', GT_len_mesh_file_paths)

our_data_path = '/data/sihun/CVPR_user_study/ours'
gt_data_path = '/data/sihun/CVPR_user_study/GT'
nfs_data_path = '/data/sihun/CVPR_user_study/nfs'
nfr_data_path = '/data/sihun/CVPR_user_study/nfr'

curr_gt_data_path = f'{gt_data_path}/{GT_file_name}'
print(curr_gt_data_path)

curr_our_data_path = f'{our_data_path}/{file_name}'
print(curr_our_data_path)

curr_nfs_data_path = f'{nfs_data_path}/{file_name}'
print(curr_nfs_data_path)

curr_nfr_data_path = f'{nfr_data_path}/{file_name}'
print(curr_nfr_data_path)

os.makedirs(curr_gt_data_path, exist_ok=True)
os.makedirs(curr_our_data_path, exist_ok=True)
os.makedirs(curr_nfs_data_path, exist_ok=True)
os.makedirs(curr_nfr_data_path, exist_ok=True)

SIZE=6
rot_list=[[0,0,0]]
M_SCALE=0.6

SELF_RETARGET = SRC_SELECT_MESH==TGT_SELECT_mesh
if SELF_RETARGET:
    print('SELF_RETARGET!')
else:
    print('CROSS_RETARGET!')



for frame in tqdm(range(GT_len_mesh_file_paths)):
    if os.path.exists(f"{curr_our_data_path}/ours-{file_name}--{frame:06d}.png"):
        continue
    _our_= np.load(mesh_file_paths[frame]).squeeze()
    _nfr_= np.load(nfr_mesh_file_paths[frame]).squeeze()
    _nfs_= np.load(nfs_mesh_file_paths[frame]).squeeze()
    _GT_= np.load(GT_mesh_file_paths[frame]).squeeze()
    
    plt.clf()
    if SELF_RETARGET:
        plot_image_array(
            [_GT_* M_SCALE], [src_f],
            rot_list=rot_list, size=SIZE, bg_black=False, mode='shade',
            logdir=curr_gt_data_path, name=f"GT-{file_name}--{frame:06d}",
            save=True
        )
    plot_image_array(
        [_our_ * M_SCALE], [tgt_f],
        rot_list=rot_list, size=SIZE, bg_black=False, mode='shade',
        logdir=curr_our_data_path, name=f"ours-{file_name}--{frame:06d}",
        save=True
    )
    plot_image_array(
        [_nfs_ * M_SCALE], [tgt_f],
        rot_list=rot_list, size=SIZE, bg_black=False, mode='shade',
        logdir=curr_nfs_data_path, name=f"nfs-{file_name}--{frame:06d}",
        save=True
    )
    plot_image_array(
        [_nfr_ * M_SCALE], [tgt_f],
        rot_list=rot_list, size=SIZE, bg_black=False, mode='shade',
        logdir=curr_nfr_data_path, name=f"nfr-{file_name}--{frame:06d}",
        save=True
    )
    plt.close('all')
    # break