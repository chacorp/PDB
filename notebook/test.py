import yaml
import argparse

import torch
from torch import nn
#from loss import *

import sys
from pathlib import Path
abs_path = str(Path.cwd().parents[0].absolute())
if not abs_path in sys.path:
    sys.path+=[abs_path, f'{abs_path}/third_party/siren']
else:
    sys.path+=[f'{abs_path}/third_party/siren']
sys.path+=[abs_path, f'{abs_path}/third_party/diffusion-net/src']
import diffusion_net
    
import modules
from meta_modules import HyperNetwork

import pickle
from models.encoder import BaseDiffusionNetEncoder, HyperDiffusionNetEncoder

if __name__ == '__main__':
    config = f'{abs_path}/config/train.yml'
    opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)
    opts = argparse.Namespace(**opts_yaml)

    opts.NFR = False
    opts.dec_type = 'disp'
    # opts.ckpt = '../ckpt_stage1/2024-08-18-23-32-29-all' ## ------> load checkpoint
    # opts.ckpt = '../ckpt_stage1/2024-06-09-10-57-34-all'
    # opts.ckpt = '../ckpt_stage1/2024-07-08-06-27-12-all'
    opts.ckpt = '../ckpts/2025-04-12-08-30-12-all'
    # opts.ckpt = '../ckpts/2025-04-23-13-35-59-all'
    opts.design = 'new2'

    # dummy...
    opts.device="cuda:0"
    opts.data_rand_trans=False
    opts.data_rand_scale=False
    opts.learn_rig_emb=False
    opts.use_decimate = False
    opts.stage1 = True
    opts.scale_exp = 1.0

    in_shape_dict = {'cents&norms':6, 'cents':3, 'cents&norms&seg':7}
    out_shape_dict = {'vert': 3, 'disp': 3, 'jacob': 9}

    # me = BaseDiffusionNetEncoder(
    #                 in_shape=in_shape_dict['cents&norms'],
    #                 pre_computes=None,
    #                 out_shape=128,
    #             )



    with open('/data/sihun/ICT-audio2face/precompute-synth-fullhead/000_dfn_info.pkl', 'rb') as f:
        asd = pickle.load(f)

    # asd
    batch_mass, batch_L, batch_evals, batch_evecs, batch_grad_X, batch_grad_Y = [],[],[],[],[],[]
    # for dfn_info_path in batch.dfn_info:
    # dfn_info = pickle.load(open(dfn_info_path, 'rb')) # DiffusionNet info
    dfn_info = asd
    dfn_info = [_.cuda().float() if type(_) is not torch.Size else _  for _ in dfn_info]

    batch_mass.append(dfn_info[0])
    batch_L.append(dfn_info[1])
    batch_evals.append(dfn_info[2])
    batch_evecs.append(dfn_info[3])
    batch_grad_X.append(dfn_info[4])
    batch_grad_Y.append(dfn_info[5])

    batch_mass=torch.stack(batch_mass).cuda()
    batch_L=torch.stack(batch_L).cuda()
    batch_evals=torch.stack(batch_evals).cuda()
    batch_evecs=torch.stack(batch_evecs).cuda()

    ID_DIM = 32
    me = HyperDiffusionNetEncoder(
            C_in=in_shape_dict['cents&norms'],
            C_width=ID_DIM,
            N_block=2,
            num_hidden_layers=1,
            hyper_num_hidden_layers=1,
            C_out=32,
        ).cuda()
    
    output = me(
            id_in=torch.rand(1,11248, ID_DIM).cuda(), 
            x_in=torch.rand(1,11248,6).cuda(), 
            mass=batch_mass, 
            L=batch_L, 
            evals=batch_evals, 
            evecs=batch_evecs, 
            gradX=batch_grad_X, 
            gradY=batch_grad_Y
        )