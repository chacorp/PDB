import os
import glob
import numpy as np
import torch
import torch.utils.data as data
import random
import pickle

import igl
import scipy
import trimesh
from functools import partial

from utils.keys import get_data_splits, get_identity_num, ICT_KEYS, DATA_KEYS, KEYS
from utils.remesh_utils import map_vertices, decimate_mesh_vertex, ICT_face_model, procrustes_LDM
from utils.exp_utils import PCA_holder, adjacency_matrix
from utils.matplotlib_rnd import plot_image_array


import sys
from pathlib import Path
__abs_path__ = str(Path(__file__).parents[0].absolute())

if not __abs_path__ in sys.path:
    sys.path+=[__abs_path__]

class EvalDataset(data.Dataset):
    def __init__(self,
                 opts=None,
                 is_train=False,
                 data_basedir='/data/sihun',
                 data_name='coma',
                 toggle=True,
                 ict_cap_id_num=2, # 0~20
                 ict_cap_exp_num=0 # 0 or 1
                ):
        super().__init__()
        self.opts = opts
        self.is_train = False
        self.mode = 'test'
        
        #data_name_list = ['voca','mf_SEN','biwi','coma','mf_ROM']
        data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM','ict','ict-cap']
        self.data_name = data_name
        
        d_mask = [dn==data_name for dn in data_name_list]
        # self.mesh_data = torch.tensor([0,1,2,3,4,5])[d_mask]
        self.mesh_data = torch.arange(len(data_name_list))[d_mask]
        
        if not self.data_name in data_name_list:
            #voca, mf_SEN, biwi, coma, mf_ROM
            raise ValueError(f'\nNo data name for that!: [{data_name}] listed dataname: {data_name_list}')
        
        _, self.voca_data_split, self.biwi_data_split, self.mf_data_split, _ = get_data_splits()
        
        self.biwi_base_path = f'{data_basedir}/BIWI_align_deci'
        self.coma_base_path = f'{data_basedir}/VOCA-COMA'
        self.mf_base_path = f'{data_basedir}/multiface_align'

        if toggle:
            self.template_data_basedir = data_basedir # char 
        else:
            self.template_data_basedir = data_basedir+'/pca' # eve-s01
        
        self.len = 0
        self.write_all_data_as_txt()
        
        if self.data_name=='ict':
            #self.iden_vecs = np.zeros((100,))
            # self.iden_vecs = torch.load(f'{__abs_path__}/ict_face_pt/ict_id_vecs_test.pt').numpy()
            self.iden_vecs = np.load(f'{__abs_path__}/data/ICT_live_100/iden_vecs.npy')
            self.expression_vecs = np.load(f'{__abs_path__}/data/ICT_live_100/expression_vecs_test.npy')
            
            self.ict_face_model=ICT_face_model()
            self.ict_len = len(self.iden_vecs)
            self.ict_exp_len = len(self.expression_vecs)
            self.len = self.ict_len * self.ict_exp_len
            self.get_data = self.get_ict

        if self.data_name=='ict-cap':
            self.ict_face_model=ICT_face_model()
            self.iden_vecs = torch.load(f'{__abs_path__}/ict_face_pt/ict_id_vecs_test.pt').numpy()
            
            if ict_cap_id_num > -1:
                self.iden_vecs = self.iden_vecs[ict_cap_id_num]
            else:
                self.iden_vecs = np.zeros((100,))

            if ict_cap_exp_num==0:
                self.expression_vecs = np.load(f'{__abs_path__}/_cap/20240318_MySlate_922_exp_coeffs.npy')
            else:
                self.expression_vecs = np.load(f'{__abs_path__}/_cap/20240325_MySlate_924_exp_coeffs.npy')
            
            self.len = len(self.expression_vecs)
            self.get_data = self.get_ict_cap
            
            
        if self.data_name=='voca':
            self.voca_std = np.load(f"{__abs_path__}/utils/voca/standardization.npy", allow_pickle=True).item()
            with open(f"{self.template_data_basedir}/VOCA-COMA/voca_templates.pkl",'rb') as f:
                self.voca_mesh = pickle.load(f)
            self.get_data = self.get_voca
            
        
        if self.data_name=='biwi':
            ## already std applied
            #self.biwi_std = np.load(f"{__abs_path__}/biwi/standardization.npy", allow_pickle=True).item()
            with open(f"{self.template_data_basedir}/BIWI_align_deci/templates_align_deci.pkl",'rb') as f:
                self.biwi_mesh = pickle.load(f) # meshes
            self.get_data = self.get_biwi
        
        if self.data_name=='mf_SEN':
            self.mf_SEN_std = np.load(f"{__abs_path__}/utils/mf/standardization.npy", allow_pickle=True).item()
            with open(f"{self.template_data_basedir}/multiface_align/mf_templates.pkl",'rb') as f:
                self.mf_SEN_mesh = pickle.load(f)
            self.get_data = self.get_mf_SEN
            # adj_mat = igl.adjacency_matrix(self.mf_SEN_mesh["face"])
            # degree = np.asarray(adj_mat.sum(axis=1)).squeeze()
            # adj_mat_norm = scipy.sparse.diags(1/degree) @ adj_mat
            # self.mf_SEN_adj_matrix = torch.tensor(adj_mat_norm.todense()).float().to_sparse().to(self.device)
            # self.mf_SEN_adj_list = igl.adjacency_list(self.mf_SEN_mesh["face"])

        
        if self.data_name=='coma':
            self.coma_std = np.load(f"{__abs_path__}/utils/voca/standardization.npy", allow_pickle=True).item()
            with open(f"{self.template_data_basedir}/VOCA-COMA/voca_templates.pkl",'rb') as f:
                self.coma_mesh = pickle.load(f)
            self.get_data = self.get_coma
            # adj_mat = igl.adjacency_matrix(self.coma_mesh["face"])
            # degree = np.asarray(adj_mat.sum(axis=1)).squeeze()
            # adj_mat_norm = scipy.sparse.diags(1/degree) @ adj_mat
            # self.coma_adj_matrix = torch.tensor(adj_mat_norm.todense()).float().to_sparse().to(self.device)
            # self.coma_adj_list = igl.adjacency_list(self.coma_mesh["face"])


        if self.data_name=='mf_ROM':
            self.mf_ROM_std = np.load(f"{__abs_path__}/utils/mf/standardization.npy", allow_pickle=True).item()
            with open(f"{self.template_data_basedir}/multiface_align/mf_templates.pkl",'rb') as f:
                self.mf_ROM_mesh = pickle.load(f)
            self.get_data = self.get_mf_ROM
            # adj_mat = igl.adjacency_matrix(self.mf_ROM_mesh["face"])
            # degree = np.asarray(adj_mat.sum(axis=1)).squeeze()
            # adj_mat_norm = scipy.sparse.diags(1/degree) @ adj_mat
            # self.mf_ROM_adj_matrix = torch.tensor(adj_mat_norm.todense()).float().to_sparse().to(self.device)
            # self.mf_ROM_adj_list = igl.adjacency_list(self.mf_ROM_mesh["face"])
    
    
    def get_data_config(self):
        text = "===========[Dataset]===========\n"
        text+= f"[mode]: {self.mode}\n"
        text+= f"[Dataset]: {self.data_name}\n"
        text+= f"------------------------------\n"
        text+= f"[total data]: {self.len}\n"
        text+= "===============================\n"
        return text
        
    def __len__(self):
        return self.len
        #return self.len_anim_all
    
    def load_data_txt(self, txt_file):
        with open(txt_file, 'r') as f:
            tmp = f.readlines()
            tmp = [ line.replace('\n','') for line in tmp]
        return tmp
        
    def write_all_data_as_txt(self):
        """
        reads the folder and writes a list of files in the folder as a .txt format
        """
        if self.data_name=='mf_ROM':
            logger_file = f"{__abs_path__}/utils/data/mf_ROM_vertex_data_{self.mode}.txt"
            if os.path.exists(logger_file):
                print(f'mf ROM data list exists: {logger_file}')
            else:
                logger = open(logger_file, 'w')
                for id_name in self.mf_data_split[self.mode]:
                    vtx_path_list = sorted(glob.glob(os.path.join(f'{self.mf_base_path}/ROM/{self.mode}', 'vertices_npy', id_name, '*')))
                    for vtx_path in vtx_path_list:
                        if '.npy' in vtx_path:
                            continue
                        vtx_path = sorted(glob.glob(os.path.join(vtx_path,'*.npy')))
                        for path in vtx_path:
                            logger.write(f'{path}\n')
                logger.close()
            self.mf_ROM_datalist = self.load_data_txt(logger_file)
            self.len += len(self.mf_ROM_datalist)
        
        if self.data_name=='mf_SEN':
            logger_file = f"{__abs_path__}/utils/data/mf_SEN_vertex_data_{self.mode}.txt"
            if os.path.exists(logger_file):
                print(f'mf SEN data list exists: {logger_file}')
            else:
                logger = open(logger_file, 'w')
                for id_name in self.mf_data_split[self.mode]:
                    vtx_path_list = sorted(glob.glob(os.path.join(f'{self.mf_base_path}/SEN/{self.mode}', 'vertices_npy', id_name, '*')))
                    for vtx_path in vtx_path_list:
                        if '.npy' in vtx_path:
                            continue
                        vtx_path = sorted(glob.glob(os.path.join(vtx_path,'*.npy')))
                        for path in vtx_path:
                            logger.write(f'{path}\n')
                logger.close()
            self.mf_SEN_datalist = self.load_data_txt(logger_file)
            self.len += len(self.mf_SEN_datalist)
            
        if self.data_name=='coma':
            logger_file = f"{__abs_path__}/utils/data/coma_vertex_data_{self.mode}.txt"
            if os.path.exists(logger_file):
                print(f'coma data list exists: {logger_file}')
            else:
                logger = open(logger_file, 'w')
                for id_name in self.voca_data_split[self.mode]:
                    vtx_path_list = sorted(glob.glob(f'{self.coma_base_path}/COMA/{self.mode}/{id_name}/vertices_npy/*'))
                    for vtx_path in vtx_path_list:
                        vtx_path = sorted(glob.glob(os.path.join(vtx_path,'*.npy')))
                        for path in vtx_path:
                            logger.write(f'{path}\n')
                logger.close()
            self.coma_datalist = self.load_data_txt(logger_file)
            self.len += len(self.coma_datalist)
        
        if self.data_name=='voca':
            logger_file = f"{__abs_path__}/utils/data/voca_vertex_data_{self.mode}.txt"
            if os.path.exists(logger_file):
                print(f'voca data list exists: {logger_file}')
            else:
                logger = open(logger_file, 'w')
                for id_name in self.voca_data_split[self.mode]:
                    vtx_path_list = sorted(glob.glob(f'{self.coma_base_path}/VOCASET/{self.mode}/{id_name}/vertices_npy/*'))
                    for vtx_path in vtx_path_list:
                        vtx_path = sorted(glob.glob(os.path.join(vtx_path,'*.npy')))
                        for path in vtx_path:
                            logger.write(f'{path}\n')
                logger.close()
            self.voca_datalist = self.load_data_txt(logger_file)
            self.len += len(self.voca_datalist)

        if self.data_name=='biwi':
            logger_file = f"{__abs_path__}/utils/data/biwi_vertex_data_{self.mode}.txt"
            if os.path.exists(logger_file):
                print(f'biwi data list exists: {logger_file}')
            else:
                logger = open(logger_file, 'w')
                for id_name in self.biwi_data_split[self.mode]:
                    id_path_list = sorted(glob.glob(f'{self.biwi_base_path}/{self.mode}/vertices_npy/{id_name}*'))
                    for id_path in id_path_list:
                        if 'pca' in id_path:
                            continue
                        vtx_path = sorted(glob.glob(f'{id_path}/*.npy'))
                        for path in vtx_path:
                            logger.write(f'{path}\n')
                logger.close()
            self.biwi_datalist = self.load_data_txt(logger_file)
            self.len += len(self.biwi_datalist)
            
    def get_ict(self, index):
        id_index = index // self.ict_exp_len
        index = index % self.ict_exp_len
        
        id_name =f'{id_index:03d}'
        id_coeff  = self.iden_vecs[id_index]
        # id_coeff  = self.iden_vecs
        
        exp_coeff = self.expression_vecs[index]
        faces = self.ict_face_model.faces
        
        vertices, template, _ = self.ict_face_model.apply_coeffs(
            id_coeff, exp_coeff, return_all=True, #region=region_dice
        )
        # exp_coeff = np.concatenate((exp_coeff, np.zeros(75))) # make it size 128
        # exp_coeff = torch.tensor(exp_coeff).float()
         
        vertices=vertices[0]
        template=template[0]
        # import pdb;pdb.set_trace()
        template_normal = igl.per_vertex_normals(template, faces)
        vertices_normal = igl.per_vertex_normals(vertices, faces)
        
        template = torch.tensor(template).float()
        vertices = torch.tensor(vertices).float()
        faces = torch.tensor(faces).long()
        template_normal = torch.tensor(template_normal).float()
        vertices_normal = torch.tensor(vertices_normal).float()
        
        return vertices, template, vertices_normal, template_normal, faces, id_name

    def get_ict_cap(self, index):        
        #id_coeff=np.zeros((100,))
        id_coeff = self.iden_vecs
        
        exp_coeff = self.expression_vecs[index]
        faces = self.ict_face_model.faces
        
        vertices, template, _ = self.ict_face_model.apply_coeffs(
            id_coeff, exp_coeff, return_all=True, #region=region_dice
        ) 
        vertices=vertices[0]
        template=template[0]
        
        template_normal = igl.per_vertex_normals(template, faces)
        vertices_normal = igl.per_vertex_normals(vertices, faces)
        
        template = torch.tensor(template).float()
        vertices = torch.tensor(vertices).float()
        faces = torch.tensor(faces).long()
        template_normal = torch.tensor(template_normal).float()
        vertices_normal = torch.tensor(vertices_normal).float()
        
        return vertices, template, vertices_normal, template_normal, faces, 'id_name'
        
    def get_voca(self, index):
        file_path=self.voca_datalist[index]
        id_name = file_path.split('/')[6]
        
        template_np = self.voca_mesh[id_name]
        template = torch.tensor(template_np).float()
        
        vertices_np = np.load(file_path)
        R, t, _ = procrustes_LDM(vertices_np, template_np)
        vertices_np = vertices_np @ R.T + t
        vertices = torch.tensor(vertices_np).float()
        
        faces_np = self.voca_std['new_f']
        faces = torch.tensor(faces_np).long()
        
        
        template_normal = igl.per_vertex_normals(template_np, faces_np)
        vertices_normal = igl.per_vertex_normals(vertices_np, faces_np)
        template_normal = torch.tensor(template_normal).float()
        vertices_normal = torch.tensor(vertices_normal).float()
        
        return vertices, template, vertices_normal, template_normal, faces, id_name
        
    def get_biwi(self, index):
        file_path=self.biwi_datalist[index]
        id_name = file_path.split('/')[6].split('_')[0]
        
        vertices_np = np.load(file_path)
        vertices = torch.tensor(vertices_np).float()
        
        faces_np = self.biwi_mesh['face']
        faces = torch.tensor(faces_np).long()
        
        template_np = self.biwi_mesh[id_name]
        template = torch.tensor(template_np).float()
        
        template_normal = igl.per_vertex_normals(template_np, faces_np)
        vertices_normal = igl.per_vertex_normals(vertices_np, faces_np)
        template_normal = torch.tensor(template_normal).float()
        vertices_normal = torch.tensor(vertices_normal).float()
        
        return vertices, template, vertices_normal, template_normal, faces, id_name
        
    def get_mf_SEN(self, index):
        file_path=self.mf_SEN_datalist[index]
        id_name = file_path.split('/')[7]
        
        template_np = self.mf_SEN_mesh[id_name]
        template = torch.tensor(template_np).float()
        
        vertices_np = np.load(file_path)
        R, t, _ = procrustes_LDM(vertices_np, template_np)
        vertices_np = vertices_np @ R.T + t
        vertices = torch.tensor(vertices_np).float()
        
        # faces_np = self.mf_SEN_std['new_f']
        # faces = torch.tensor(faces_np).long()
        faces = self.mf_SEN_std['new_f'].long()
        
        
        template_normal = igl.per_vertex_normals(template_np, faces.numpy())
        vertices_normal = igl.per_vertex_normals(vertices_np, faces.numpy())
        template_normal = torch.tensor(template_normal).float()
        vertices_normal = torch.tensor(vertices_normal).float()
        
        return vertices, template, vertices_normal, template_normal, faces, id_name
        
    def get_coma(self, index):
        file_path=self.coma_datalist[index]
        id_name = file_path.split('/')[6]
        
        template_np = self.coma_mesh[id_name]
        template = torch.tensor(template_np).float()
        
        vertices_np = np.load(file_path)
        R, t, _ = procrustes_LDM(vertices_np, template_np)
        vertices_np = vertices_np @ R.T + t
        vertices = torch.tensor(vertices_np).float()
        
        faces_np = self.coma_std['new_f']
        faces = torch.tensor(faces_np).long()
        
        
        template_normal = igl.per_vertex_normals(template_np, faces_np)
        vertices_normal = igl.per_vertex_normals(vertices_np, faces_np)
        template_normal = torch.tensor(template_normal).float()
        vertices_normal = torch.tensor(vertices_normal).float()
        
        return vertices, template, vertices_normal, template_normal, faces, id_name
    
    def get_mf_ROM(self, index):
        file_path=self.mf_ROM_datalist[index]
        id_name = file_path.split('/')[7]
        
        template_np = self.mf_ROM_mesh[id_name]
        template = torch.tensor(template_np).float()
        
        # vertices_np = np.load(file_path)
        # vertices = torch.tensor(vertices_np).float()
        vertices_np = np.load(file_path)
        R, t, _ = procrustes_LDM(vertices_np, template_np)
        vertices_np = vertices_np @ R.T + t
        vertices = torch.tensor(vertices_np).float()
        
        # faces_np = self.mf_ROM_std['new_f']
        # faces = torch.tensor(faces_np).long()
        faces = self.mf_ROM_std['new_f'].long()
        
        template_normal = igl.per_vertex_normals(template_np, faces.numpy())
        vertices_normal = igl.per_vertex_normals(vertices_np, faces.numpy())
        template_normal = torch.tensor(template_normal).float()
        vertices_normal = torch.tensor(vertices_normal).float()
        
        return vertices, template, vertices_normal, template_normal, faces, id_name
            
    def __getitem__(self, index):
        return (*self.get_data(index), self.mesh_data)
            

class CBDDataset(data.Dataset):
    def __init__(self, 
                 opts,
                 toggle=False,
                 data_basedir='/data/sihun', # char-s02
                 is_train=False,
                 is_valid=False,
                 window_size=8, # batch size
                 device='cpu',
                 print_config=False,
                 ict_face_only=False,
                 n_components=100, ## mf, voca, biwi all used 100
                 scale=1.0, # pca data axis scale
                 use_voca=True,
                 use_coma=True,
                 use_biwi=True,
                 use_mf_SEN=True,
                 use_mf_ROM=True,
                 use_ict=False,
                 use_ict_narrow=False,
                ):
        super().__init__()
        # get basenames
        self.opts = opts
        self.is_train = is_train
        self.is_valid = is_valid
        self.device = device
        self.data_basedir = data_basedir
        self.n_components = n_components
        self.scale = scale
        
        if self.opts.use_data0:
            self.n_components=400
            use_voca=False
            use_coma=True
            use_biwi=False
            use_mf_SEN=False
            use_mf_ROM=False
            use_ict=False
            use_ict_narrow=False
        elif self.opts.use_data1:
            use_voca=False
            use_coma=False
            use_biwi=False
            use_mf_SEN=True
            use_mf_ROM=True
            use_ict=False
            use_ict_narrow=False
        elif self.opts.use_data2:
            use_voca=False
            use_coma=False
            use_biwi=False
            use_mf_SEN=True
            use_mf_ROM=True
            use_ict=True
            use_ict_narrow=True
        elif self.opts.use_data3:
            use_voca=False
            use_coma=True
            use_biwi=False
            use_mf_SEN=True
            use_mf_ROM=True
            use_ict=True
            use_ict_narrow=False
        elif self.opts.use_data9:
            use_voca=False
            use_coma=False
            use_biwi=False
            use_mf_SEN=False
            use_mf_ROM=False
            use_ict=True
            use_ict_narrow=True
        elif self.opts.use_data9:
            use_voca=False
            use_coma=False
            use_biwi=False
            use_mf_SEN=False
            use_mf_ROM=False
            use_ict=True
            use_ict_narrow=True
        else:
            pass
            # use_voca=True
            # use_coma=True
            # use_biwi=True
            # use_mf_SEN=True
            # use_mf_ROM=True
            # use_ict=False
            # use_ict_narrow=False
            
        self.use_voca=use_voca
        self.use_coma=use_coma
        self.use_biwi=use_biwi
        self.use_mf_SEN=use_mf_SEN
        self.use_mf_ROM=use_mf_ROM
        self.use_ict=use_ict
        self.use_ict_narrow=use_ict_narrow
        
        self.use_laplacian = self.opts.use_laplacian
        basedir=data_basedir
        if toggle:
            # /data/sihun/pca
            data_basedir=data_basedir+'/pca' # char-s05
        else:
            data_basedir=data_basedir # char-s02
        #self.data_config(self.opts, window_size)
        
        self.mode = 'test'
        if is_train:
            self.mode = 'train'
            self.scale = 1.5
        elif is_valid:
            self.mode = 'val'
        
        total_len = 0
        total_id = 0
        
        _, voca_data_split, biwi_data_split, mf_data_split, _ = get_data_splits()
        
        ## to make no leftover for each mesh id
        self.min_sample = self.n_components % self.opts.batch_size
        self.len_list=[]

        ## add face only and narrow face too
        if self.use_ict:
            self.ict_face_model=ICT_face_model()
                        
            self.iden_vecs, self.expression_vecs = self.get_ict_params()
            self.ict_len = len(self.iden_vecs)
            self.ict_exp_len = len(self.expression_vecs)
            total_id = total_id + self.ict_len
            
            ict_min_sample = self.ict_exp_len % (self.ict_len * self.opts.batch_size)
            
            self.ict_synth_precompute_fh = f'{basedir}/ICT-audio2face/precompute-synth-fullhead'
            self.ict_synth_precompute_fo = f'{basedir}/ICT-audio2face/precompute-synth-face_only'
            ## Added segmentation ##################################################################
            self.ict_seg=torch.tensor(np.load('utils/ict/ICT_segment_onehot_24.npy'))
            ########################################################################################
            if self.use_laplacian:
                self.ict_cotmatrix={}
                ict_cotmatrix_path='utils/ict/ict_cotmatrix.pkl'
                
                if os.path.exists(ict_cotmatrix_path):
                    with open(ict_cotmatrix_path,'rb') as f:
                        self.ict_cotmatrix = pickle.load(f)
                else:
                    for idx, id_coeff in enumerate(self.iden_vecs):
                        id_disps = self.ict_face_model.get_id_disp(id_coeff).squeeze()
                        id_verts = id_disps + self.ict_face_model.neutral_verts
                        # import pdb;pdb.set_trace()
        
                        tmp_L = igl.cotmatrix(id_verts, self.ict_face_model.faces)
                        tmp_L = torch.sparse_csc_tensor(
                            torch.from_numpy(tmp_L.indptr).long(),
                            torch.from_numpy(tmp_L.indices).long(),
                            torch.from_numpy(tmp_L.data).float(),
                            tmp_L.shape
                        )#.to(self.device)
                        #tmp_L = torch.tensor(tmp_L.todense()).float().to_sparse().to(self.device)
                        self.ict_cotmatrix[f"{idx:03d}"] = tmp_L
                        
                    with open(ict_cotmatrix_path,'wb') as f:
                        pickle.dump(self.ict_cotmatrix, f)
                    
            # adj_mat = igl.adjacency_matrix(self.voca_mesh["face"])
            # degree = np.asarray(adj_mat.sum(axis=1)).squeeze()
            # adj_mat_norm = scipy.sparse.diags(1/degree) @ adj_mat
            # self.voca_adj_matrix = torch.tensor(adj_mat_norm.todense()).float().to_sparse().to(self.device)
            # self.voca_adj_list = igl.adjacency_list(self.voca_mesh["face"])
            
            self.len_list.append([(self.ict_exp_len+ict_min_sample), self.get_ict, torch.tensor(5), self.ict_len])

        if self.use_ict_narrow:
            self.ict_face_model_narrow=ICT_face_model(narrow_only=True)
            self.region_num = self.ict_face_model_narrow.get_region_num(
                self.ict_face_model_narrow.neutral_verts
            )
            
            self.iden_vecs, self.expression_vecs = self.get_ict_params()
            self.ict_narrow_len = len(self.iden_vecs)
            self.ict_narrow_exp_len = len(self.expression_vecs)
            total_id = total_id + self.ict_narrow_len
            
            ict_min_sample = self.ict_narrow_exp_len % (self.ict_narrow_len * self.opts.batch_size)
            
            self.ict_synth_precompute_nf = f'{basedir}/ICT-audio2face/precompute-synth-narrow_face'
            ## Added segmentation ##################################################################
            self.ict_narrow_seg=torch.tensor(
                np.load('utils/ict/ICT_segment_onehot_24.npy')
            )[:self.ict_face_model_narrow.v_num]
            ########################################################################################

            if self.use_laplacian:
                self.ict_narrow_cotmatrix={}
                ict_cotmatrix_path='utils/ict/ict_narrow_cotmatrix.pkl'
                
                if os.path.exists(ict_cotmatrix_path):
                    with open(ict_cotmatrix_path,'rb') as f:
                        self.ict_cotmatrix = pickle.load(f)
                else:
                    for idx, id_coeff in enumerate(self.iden_vecs):
                        id_disps = self.ict_face_model_narrow.get_id_disp(
                            id_coeff, region=self.region_num
                        ).squeeze()
                        id_verts = id_disps + self.ict_face_model_narrow.neutral_verts
                        
                        # import pdb;pdb.set_trace()
        
                        tmp_L = igl.cotmatrix(id_verts, self.ict_face_model_narrow.faces)
                        tmp_L = torch.sparse_csc_tensor(
                            torch.from_numpy(tmp_L.indptr).long(),
                            torch.from_numpy(tmp_L.indices).long(),
                            torch.from_numpy(tmp_L.data).float(),
                            tmp_L.shape
                        )#.to(self.device)
                        #tmp_L = torch.tensor(tmp_L.todense()).float().to_sparse().to(self.device)
                        self.ict_narrow_cotmatrix[f"{idx:03d}"] = tmp_L
                        
                    with open(ict_cotmatrix_path,'wb') as f:
                        pickle.dump(self.ict_narrow_cotmatrix, f)
                    
            # adj_mat = igl.adjacency_matrix(self.voca_mesh["face"])
            # degree = np.asarray(adj_mat.sum(axis=1)).squeeze()
            # adj_mat_norm = scipy.sparse.diags(1/degree) @ adj_mat
            # self.voca_adj_matrix = torch.tensor(adj_mat_norm.todense()).float().to_sparse().to(self.device)
            # self.voca_adj_list = igl.adjacency_list(self.voca_mesh["face"])
                    
            self.len_list.append([
                (self.ict_narrow_exp_len+ict_min_sample),
                self.get_ict_narrow, torch.tensor(6), self.ict_narrow_len
            ])
        
        if self.use_voca:
            self.voca_pca_holder_list=[]
            self.voca_id_list=[]
            for id_name in voca_data_split[self.mode]:
                self.voca_id_list.append(id_name)
                npz_file = f"{data_basedir}/VOCA-COMA/VOCASET/{self.mode}/{id_name}_pca.npz"
                self.voca_pca_holder_list.append(PCA_holder(npz_file))
            assert len(self.voca_pca_holder_list) == len(self.voca_id_list), "mismatch in voca"
            
            self.voca_len = len(self.voca_pca_holder_list)
            self.voca_std = np.load("utils/voca/standardization.npy", allow_pickle=True).item()
            with open(f"{data_basedir}/VOCA-COMA/voca_templates.pkl",'rb') as f:
                self.voca_mesh = pickle.load(f)
            total_id = total_id + self.voca_len
            
            ## Added segmentation ##################################################################
            self.voca_seg=torch.tensor(np.load('utils/voca/flame_seg_24.npy'))
            ########################################################################################

            if self.use_laplacian:
                self.voca_cotmatrix={}
                voca_cotmatrix_path='utils/voca/voca_cotmatrix.pkl'
    
                if os.path.exists(voca_cotmatrix_path):
                    with open(voca_cotmatrix_path,'rb') as f:
                        self.voca_cotmatrix = pickle.load(f)
                else:
                    for id_name in voca_data_split[self.mode]:                
                        tmp_L = igl.cotmatrix(self.voca_mesh[id_name], self.voca_mesh['face'])
                        tmp_L = torch.tensor(tmp_L.todense()).float().to_sparse().to(self.device)
                        self.voca_cotmatrix[id_name] = tmp_L
                        
                    with open(voca_cotmatrix_path,'wb') as f:
                        pickle.dump(self.voca_cotmatrix, f)
            # adj_mat = igl.adjacency_matrix(self.voca_mesh["face"])
            # degree = np.asarray(adj_mat.sum(axis=1)).squeeze()
            # adj_mat_norm = scipy.sparse.diags(1/degree) @ adj_mat
            # self.voca_adj_matrix = torch.tensor(adj_mat_norm.todense()).float().to_sparse().to(self.device)
            # self.voca_adj_list = igl.adjacency_list(self.voca_mesh["face"])
                    
            self.len_list.append([(self.n_components+self.min_sample)*self.voca_len, self.get_voca, torch.tensor(0), self.voca_len])
        
        
        if self.use_biwi:
            self.biwi_pca_holder_list=[]
            self.biwi_id_list=[]
            for id_name in biwi_data_split[self.mode]:
                self.biwi_id_list.append(id_name)
                
                npz_file = f"{data_basedir}/BIWI_align_deci/{self.mode}/vertices_npy/{id_name}_pca.npz"
                if not os.path.exists(npz_file):
                    npz_file = f"{data_basedir}/BIWI_align_deci/{self.mode}/{id_name}_pca.npz"
                    
                self.biwi_pca_holder_list.append(PCA_holder(npz_file))
            assert len(self.biwi_pca_holder_list) == len(self.biwi_id_list), "mismatch in biwi"
            
            self.biwi_len = len(self.biwi_pca_holder_list)
            #self.biwi_std = np.load("utils/biwi/standardization.npy", allow_pickle=True).item()
            with open(f"{data_basedir}/BIWI_align_deci/templates_align_deci.pkl",'rb') as f:
                self.biwi_mesh = pickle.load(f) # meshes
            total_id = total_id + self.biwi_len
            
            ## Added segmentation ##################################################################
            self.biwi_seg=torch.tensor(np.load('utils/biwi/biwi_seg_24.npy'))
            ########################################################################################

            if self.use_laplacian:
                self.biwi_cotmatrix={}
                biwi_cotmatrix_path='utils/biwi/biwi_cotmatrix.pkl'
    
                if os.path.exists(biwi_cotmatrix_path):
                    with open(biwi_cotmatrix_path,'rb') as f:
                        self.biwi_cotmatrix = pickle.load(f)
                else:
                    for id_name in biwi_data_split[self.mode]:                
                        tmp_L = igl.cotmatrix(self.biwi_mesh[id_name], self.biwi_mesh['face'])
                        tmp_L = torch.tensor(tmp_L.todense()).float().to_sparse().to(self.device)
                        self.biwi_cotmatrix[id_name] = tmp_L
                        
                    with open(biwi_cotmatrix_path,'wb') as f:
                        pickle.dump(self.biwi_cotmatrix, f)
            # adj_mat = igl.adjacency_matrix(self.biwi_mesh["face"])
            # degree = np.asarray(adj_mat.sum(axis=1)).squeeze()
            # adj_mat_norm = scipy.sparse.diags(1/degree) @ adj_mat
            # self.biwi_adj_matrix = torch.tensor(adj_mat_norm.todense()).float().to_sparse().to(self.device)
            # self.biwi_adj_list = igl.adjacency_list(self.biwi_mesh["face"])
            
            self.len_list.append([(self.n_components+self.min_sample)*self.biwi_len, self.get_biwi, torch.tensor(1), self.biwi_len])

        
        if self.use_mf_SEN:
            self.mf_SEN_pca_holder_list=[]
            self.mf_SEN_id_list=[]
            for id_name in mf_data_split[self.mode]:
                self.mf_SEN_id_list.append(id_name)
                
                npz_file = f"{data_basedir}/multiface_align/SEN/{self.mode}/vertices_npy/{id_name}_pca.npz"
                if not os.path.exists(npz_file):
                    npz_file = f"{data_basedir}/multiface_align/SEN/{self.mode}/{id_name}_pca.npz"
                    
                self.mf_SEN_pca_holder_list.append(PCA_holder(npz_file))
            assert len(self.mf_SEN_pca_holder_list) == len(self.mf_SEN_id_list), "mismatch in mf SEN"
                
            self.mf_SEN_len = len(self.mf_SEN_pca_holder_list)
            self.mf_SEN_std = np.load("utils/mf/standardization.npy", allow_pickle=True).item()
            with open(f"{data_basedir}/multiface_align/mf_templates.pkl",'rb') as f:
                self.mf_SEN_mesh = pickle.load(f)
            total_id = total_id + self.mf_SEN_len

            self.mf_precompute_path=f"{basedir}/multiface_align/precomputes"            
            ## Added segmentation ##################################################################
            self.mf_SEN_seg=torch.tensor(np.load('utils/mf/mf_seg_24.npy'))
            ########################################################################################
            
            if self.use_laplacian:
                self.mf_SEN_cotmatrix={}
                mf_cotmatrix_path='utils/mf/mf_cotmatrix.pkl'
    
                if os.path.exists(mf_cotmatrix_path):
                    with open(mf_cotmatrix_path,'rb') as f:
                        self.mf_SEN_cotmatrix = pickle.load(f)
                else:
                    for id_name in mf_data_split[self.mode]:                
                        tmp_L = igl.cotmatrix(self.mf_SEN_mesh[id_name], self.mf_SEN_mesh['face'])
                        tmp_L = torch.tensor(tmp_L.todense()).float().to_sparse().to(self.device)
                        self.mf_SEN_cotmatrix[id_name] = tmp_L
                        
                    with open(mf_cotmatrix_path,'wb') as f:
                        pickle.dump(self.mf_SEN_cotmatrix, f)
            # adj_mat = igl.adjacency_matrix(self.mf_SEN_mesh["face"])
            # degree = np.asarray(adj_mat.sum(axis=1)).squeeze()
            # adj_mat_norm = scipy.sparse.diags(1/degree) @ adj_mat
            # self.mf_SEN_adj_matrix = torch.tensor(adj_mat_norm.todense()).float().to_sparse().to(self.device)
            # self.mf_SEN_adj_list = igl.adjacency_list(self.mf_SEN_mesh["face"])
            
            self.len_list.append([(self.n_components+self.min_sample)*self.mf_SEN_len, self.get_multiface_SEN, torch.tensor(2), self.mf_SEN_len])

        
        if self.use_coma:
            self.coma_pca_holder_list=[]
            self.coma_id_list=[]
            for id_name in voca_data_split[self.mode]:
                self.coma_id_list.append(id_name)
                npz_file = f"{data_basedir}/VOCA-COMA/COMA/{self.mode}/{id_name}_pca.npz"
                self.coma_pca_holder_list.append(PCA_holder(npz_file))
            assert len(self.coma_pca_holder_list) == len(self.coma_id_list), "mismatch in coma"
            
            self.coma_len = len(self.coma_pca_holder_list)
            self.coma_std = np.load("utils/voca/standardization.npy", allow_pickle=True).item()
            with open(f"{data_basedir}/VOCA-COMA/voca_templates.pkl",'rb') as f:
                self.coma_mesh = pickle.load(f)
            total_id = total_id + self.coma_len

            self.mf_precompute_path=f"{basedir}/multiface_align/precomputes"
            ## Added segmentation ##################################################################
            self.coma_seg=torch.tensor(np.load('utils/voca/flame_seg_24.npy'))
            ########################################################################################

            if self.use_laplacian:
                self.coma_cotmatrix={}
                coma_cotmatrix_path='utils/voca/voca_cotmatrix.pkl'
    
                if os.path.exists(coma_cotmatrix_path):
                    with open(coma_cotmatrix_path,'rb') as f:
                        self.coma_cotmatrix = pickle.load(f)
                else:
                    for id_name in voca_data_split[self.mode]:                
                        tmp_L = igl.cotmatrix(self.coma_mesh[id_name], self.coma_mesh['face'])
                        tmp_L = torch.tensor(tmp_L.todense()).float().to_sparse().to(self.device)
                        self.coma_cotmatrix[id_name] = tmp_L
                        
                    with open(coma_cotmatrix_path,'wb') as f:
                        pickle.dump(self.coma_cotmatrix, f)
            # adj_mat = igl.adjacency_matrix(self.coma_mesh["face"])
            # degree = np.asarray(adj_mat.sum(axis=1)).squeeze()
            # adj_mat_norm = scipy.sparse.diags(1/degree) @ adj_mat
            # self.coma_adj_matrix = torch.tensor(adj_mat_norm.todense()).float().to_sparse().to(self.device)
            # self.coma_adj_list = igl.adjacency_list(self.coma_mesh["face"])
                    
            self.len_list.append([(self.n_components+self.min_sample)*self.coma_len, self.get_coma, torch.tensor(3), self.coma_len])


        if self.use_mf_ROM:
            self.mf_ROM_pca_holder_list=[]
            self.mf_ROM_id_list=[]
            for id_name in mf_data_split[self.mode]:
                self.mf_ROM_id_list.append(id_name)
                
                npz_file = f"{data_basedir}/multiface_align/ROM/{self.mode}/vertices_npy/{id_name}_pca.npz"
                if not os.path.exists(npz_file):
                    npz_file = f"{data_basedir}/multiface_align/ROM/{self.mode}/{id_name}_pca.npz"
                    
                self.mf_ROM_pca_holder_list.append(PCA_holder(npz_file))
            assert len(self.mf_ROM_pca_holder_list) == len(self.mf_ROM_id_list), "mismatch in mf ROM"
                
            self.mf_ROM_len = len(self.mf_ROM_pca_holder_list)
            self.mf_ROM_std = np.load("utils/mf/standardization.npy", allow_pickle=True).item()
            with open(f"{data_basedir}/multiface_align/mf_templates.pkl",'rb') as f:
                self.mf_ROM_mesh = pickle.load(f)
            total_id = total_id + self.mf_ROM_len
            
            ## Added segmentation ##################################################################
            self.mf_ROM_seg=torch.tensor(np.load('utils/mf/mf_seg_24.npy'))
            ########################################################################################
            if self.use_laplacian:
                self.mf_ROM_cotmatrix={}
                mf_cotmatrix_path='utils/mf/mf_cotmatrix.pkl'
    
                if os.path.exists(mf_cotmatrix_path):
                    with open(mf_cotmatrix_path,'rb') as f:
                        self.mf_SEN_cotmatrix = pickle.load(f)
                else:
                    for id_name in mf_data_split[self.mode]:                
                        tmp_L = igl.cotmatrix(self.mf_ROM_mesh[id_name], self.mf_ROM_mesh['face'])
                        tmp_L = torch.tensor(tmp_L.todense()).float().to_sparse().to(self.device)
                        self.mf_ROM_cotmatrix[id_name] = tmp_L
                    
                    with open(mf_cotmatrix_path, 'wb') as f:
                        pickle.dump(self.mf_ROM_cotmatrix, f)
            # adj_mat = igl.adjacency_matrix(self.mf_ROM_mesh["face"])
            # degree = np.asarray(adj_mat.sum(axis=1)).squeeze()
            # adj_mat_norm = scipy.sparse.diags(1/degree) @ adj_mat
            # self.mf_ROM_adj_matrix = torch.tensor(adj_mat_norm.todense()).float().to_sparse().to(self.device)
            # self.mf_ROM_adj_list = igl.adjacency_list(self.mf_ROM_mesh["face"])
            
            self.len_list.append([(self.n_components+self.min_sample)*self.mf_ROM_len, self.get_multiface_ROM, torch.tensor(4), self.mf_ROM_len])
        
        # 2 for -weight and +weight
        
        total_len = (self.n_components+self.min_sample) * total_id
        
        self.total_len = total_len
        self.total_id = total_id
                
        if print_config:
            print(self.get_data_config())
            
        
    def get_data_config(self):
        text = "===========[Dataset]===========\n"
        text+= f"[mode]: {self.mode}\n"
        text+= f"------------------------------\n"
        text+= f"[total id mesh]: {self.total_id}\n"
        text+= f"[pca n_components]: {self.n_components}\n"
        text+= f"[+ remain]: {self.min_sample}\n"
        text+= f"[total pca data]: {self.total_len}\n"
        text+= "===============================\n"
        return text
    
    def __len__(self):
        return self.total_len
    
    def data_config(self, opts=None, window_size=8):
        flag = True if opts is not None else False
        
        self.use_ict_synth_single = opts.use_ict_synth_single if flag else False
        
        self.use_ict_real = opts.use_ict_real if flag else False
        self.use_ict_synth = opts.use_ict_synth if flag else False
        self.use_mf_SEN = opts.use_mf_SEN if flag else False
        self.use_mf_ROM = opts.use_mf_ROM if flag else False
        self.use_voca = opts.use_voca if flag else False
        self.use_coma = opts.use_coma if flag else False
        self.use_biwi = opts.use_biwi if flag else False
        
        if flag:
            self.use_decimate = False
            self.WS = window_size
        else:
            self.use_decimate = self.opts.use_decimate
            self.WS = self.opts.window_size
            
    def get_ict_params(self):
        if self.mode == 'train':
            iden_vecs = np.load('./ict_face_pt/random_identity_vecs.npy')[:111]
            expression_vecs = np.load('./ict_face_pt/random_expression_vecs.npy')
        else:
            iden_vecs = np.load('./data/ICT_live_100/iden_vecs.npy')
            expression_vecs = np.load(f'./data/ICT_live_100/expression_vecs_{self.mode}.npy')

        return iden_vecs, expression_vecs

    def get_ict_narrow(self, index, id_index):
        id_coeff = self.iden_vecs[id_index]
        id_name = f"{id_index:03d}"

        # region_dice = np.random.randint(3, size=(1))
        
        # if np.random.random(1) > 0.5:
        #     exp_coeff = np.random.random((1, 53))
        # else:
        #     exp_coeff = np.random.randint(2, size=(1, 53))
        #     exp_coeff = np.random.random((1, 53))
        # exp_coeff = np.eye(53)[index][None] if index < 53 else np.random.randint(2, size=(1, 53))
        
        if index >= self.ict_narrow_exp_len:
            index = index % self.ict_narrow_exp_len
        # exp_coeff = self.expression_vecs[index] 
        # exp_coeff = exp_coeff * self.scale
        
        if self.mode=='train':
            if np.random.random(1) > 0.5:
                exp_coeff = np.random.random(53)
            else:
                #exp_coeff = np.random.randint(2, size=(1, 53))
                exp_coeff = np.where(np.random.random(53) > 0.9, 1, 0)
        else:
            exp_coeff = self.expression_vecs[index]
            
        # exp_coeff = self.expression_vecs[index]
        # exp_coeff = exp_coeff * self.scale

        # self.ict_face_model_narrow=ICT_face_model(narrow_only=True)
        # self.region_num
        faces = self.ict_face_model_narrow.faces
        
        deformed, template, _ = self.ict_face_model_narrow.apply_coeffs(
            id_coeff, exp_coeff, return_all=True, region=self.region_num
        )
        exp_coeff = np.concatenate((exp_coeff, np.zeros(75))) # make it size 128
        exp_coeff = torch.tensor(exp_coeff).float()        
         
        deformed=deformed[0]
        template=template[0]
        
        template_normal = igl.per_vertex_normals(template, faces)
        deformed_normal = igl.per_vertex_normals(deformed, faces)
        
        template = torch.tensor(template).float()
        deformed = torch.tensor(deformed).float()
        faces = torch.tensor(faces).long()
        template_normal = torch.tensor(template_normal).float()
        deformed_normal = torch.tensor(deformed_normal).float()
        
        return (template, deformed, faces, template_normal, deformed_normal, self.ict_seg, exp_coeff, id_name)
    
    def get_ict(self, index, id_index):
        id_coeff = self.iden_vecs[id_index]
        id_name = f"{id_index:03d}"

        # region_dice = np.random.randint(3, size=(1))
        
        
        if index >= self.ict_exp_len:
            index = index % self.ict_exp_len
        # exp_coeff = self.expression_vecs[index] 
        # exp_coeff = exp_coeff * self.scale

        if self.mode=='train':
            if np.random.random(1) > 0.5:
                exp_coeff = np.random.random(53)
            else:
                #exp_coeff = np.random.randint(2, size=(1, 53))
                exp_coeff = np.where(np.random.random(53) > 0.9, 1, 0)
        else:
            exp_coeff = self.expression_vecs[index]
        # exp_coeff = self.expression_vecs[index] 
        # exp_coeff = exp_coeff * self.scale
        
        faces = self.ict_face_model.faces
        
        deformed, template, _ = self.ict_face_model.apply_coeffs(
            id_coeff, exp_coeff, return_all=True, #region=region_dice
        )
        exp_coeff = np.concatenate((exp_coeff, np.zeros(75))) # make it size 128
        exp_coeff = torch.tensor(exp_coeff).float()        
         
        deformed=deformed[0]
        template=template[0]
        
        template_normal = igl.per_vertex_normals(template, faces)
        deformed_normal = igl.per_vertex_normals(deformed, faces)
        
        template = torch.tensor(template).float()
        deformed = torch.tensor(deformed).float()
        faces = torch.tensor(faces).long()
        template_normal = torch.tensor(template_normal).float()
        deformed_normal = torch.tensor(deformed_normal).float()
        
        return (template, deformed, faces, template_normal, deformed_normal, self.ict_seg, exp_coeff, id_name)
    
    def get_voca(self, index, id_index):
        
        id_name = self.voca_id_list[id_index]
        pca_holder = self.voca_pca_holder_list[id_index]
        template = self.voca_mesh[id_name]
        faces = self.voca_mesh["face"]
                
        # if index < self.n_components:
        if False:
            deformed = pca_holder.sample_from_pca_one_axis(scale=self.scale, select=index, verbose=False)
        else:
            deformed = pca_holder.sample_from_pca(scale=self.scale)
            
        template_normal = igl.per_vertex_normals(template, faces)
        deformed_normal = igl.per_vertex_normals(deformed, faces)
        
        template = torch.tensor(template).float()
        deformed = torch.tensor(deformed).float()
        faces = torch.tensor(faces).long()
        template_normal = torch.tensor(template_normal).float()
        deformed_normal = torch.tensor(deformed_normal).float()
        
        return (template, deformed, faces, template_normal, deformed_normal, self.voca_seg, torch.zeros(128), id_name)

    def get_coma(self, index, id_index):
        
        id_name = self.coma_id_list[id_index]
        pca_holder = self.coma_pca_holder_list[id_index]
        template = self.coma_mesh[id_name]
        faces = self.coma_mesh["face"]
                
        # if index < self.n_components:
        if False:
            deformed = pca_holder.sample_from_pca_one_axis(scale=self.scale, select=index, verbose=False)
        else:
            deformed = pca_holder.sample_from_pca(scale=self.scale)
            
        template_normal = igl.per_vertex_normals(template, faces)
        deformed_normal = igl.per_vertex_normals(deformed, faces)
        
        template = torch.tensor(template).float()
        deformed = torch.tensor(deformed).float()
        faces = torch.tensor(faces).long()
        template_normal = torch.tensor(template_normal).float()
        deformed_normal = torch.tensor(deformed_normal).float()
        
        return (template, deformed, faces, template_normal, deformed_normal, self.coma_seg, torch.zeros(128), id_name)
        
    def get_biwi(self, index, id_index):
        
        id_name = self.biwi_id_list[id_index]
        pca_holder = self.biwi_pca_holder_list[id_index]
        
        # if index < self.n_components:
        if False:
            deformed = pca_holder.sample_from_pca_one_axis(scale=self.scale, select=index, verbose=False)
        else:
            deformed = pca_holder.sample_from_pca(scale=self.scale)
        template = self.biwi_mesh[id_name]
        faces = self.biwi_mesh["face"]
        
        template_normal = igl.per_vertex_normals(template, faces)
        deformed_normal = igl.per_vertex_normals(deformed, faces)
        
        template = torch.tensor(template).float()
        deformed = torch.tensor(deformed).float()
        faces = torch.tensor(faces).long()
        template_normal = torch.tensor(template_normal).float()
        deformed_normal = torch.tensor(deformed_normal).float()
        
        return (template, deformed, faces, template_normal, deformed_normal, self.biwi_seg, torch.zeros(128), id_name)

    
    def get_multiface_SEN(self, index, id_index):
        
        id_name = self.mf_SEN_id_list[id_index]
        pca_holder = self.mf_SEN_pca_holder_list[id_index]
        
        if False:
            deformed = pca_holder.sample_from_pca_one_axis(scale=self.scale, select=index, verbose=False)
        else:
            deformed = pca_holder.sample_from_pca(scale=self.scale)
        template = self.mf_SEN_mesh[id_name]
        faces = self.mf_SEN_mesh["face"]
        
        template_normal = igl.per_vertex_normals(template, faces)
        deformed_normal = igl.per_vertex_normals(deformed, faces)
        
        template = torch.tensor(template).float()
        deformed = torch.tensor(deformed).float()
        faces = torch.tensor(faces).long()
        template_normal = torch.tensor(template_normal).float()
        deformed_normal = torch.tensor(deformed_normal).float()
        
        return (template, deformed, faces, template_normal, deformed_normal, self.mf_SEN_seg, torch.zeros(128), id_name)
    
    
    def get_multiface_ROM(self, index, id_index):
        
        id_name = self.mf_ROM_id_list[id_index]
        pca_holder = self.mf_ROM_pca_holder_list[id_index]
        
        if False:
            deformed = pca_holder.sample_from_pca_one_axis(scale=self.scale, select=index, verbose=False)
        else:
            deformed = pca_holder.sample_from_pca(scale=self.scale)
        template = self.mf_ROM_mesh[id_name]
        faces = self.mf_ROM_mesh["face"]
        
        template_normal = igl.per_vertex_normals(template, faces)
        deformed_normal = igl.per_vertex_normals(deformed, faces)
        
        template = torch.tensor(template).float()
        deformed = torch.tensor(deformed).float()
        faces = torch.tensor(faces).long()
        template_normal = torch.tensor(template_normal).float()
        deformed_normal = torch.tensor(deformed_normal).float()
        
        return (template, deformed, faces, template_normal, deformed_normal, self.mf_ROM_seg, torch.zeros(128), id_name)

        
    def random_rotation_matrix(self, randgen=None):
        """
        Borrowed from https://github.com/nmwsharp/diffusion-net/blob/master/src/diffusion_net/utils.py
        
        Creates a random rotation matrix.
        randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
        """
        # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
        
        if randgen is None:
            randgen = np.random.RandomState()
            
        theta, phi, z = tuple(randgen.rand(3).tolist())
        
        theta = theta * 2.0*np.pi  # Rotation about the pole (Z).
        phi = phi * 2.0*np.pi  # For direction of pole deflection.
        z = z * 2.0 # For magnitude of pole deflection.
        
        # Compute a vector V used for distributing points over the sphere
        # via the reflection I - V Transpose(V).  This formulation of V
        # will guarantee that if x[1] and x[2] are uniformly distributed,
        # the reflected points will be uniform on the sphere.  Note that V
        # has length sqrt(2) to eliminate the 2 in the Householder matrix.
        
        r = np.sqrt(z)
        Vx, Vy, Vz = V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z)
            )
        
        st = np.sin(theta)
        ct = np.cos(theta)
        
        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
        # Construct the rotation matrix  ( V Transpose(V) - I ) R.

        M = (np.outer(V, V) - np.eye(3)).dot(R)
        return M
    
    def random_rotate_points(self, pts, randgen=None):
        R = self.random_rotation_matrix(randgen) 
        R = torch.from_numpy(R).to(device=pts.device, dtype=pts.dtype)
        return torch.matmul(pts, R)
    
    def random_trans_scale(self, template, vertices):
        """
            not used
        """
        ## Random Augmentation ---------------------------------------------------------
        trans, scale = 0.0, 1.0
        if self.opts.data_rand_trans:
            t_range = 0.02
            trans = (torch.rand((1, 3)) - 0.5) * t_range
        if self.opts.data_rand_scale:
            scale = torch.rand((1)).repeat(3) * 0.2 + 0.9 # [0.9 ~ 1.1]
        template = template * scale + trans
        vertices = vertices * scale + trans
        ## -----------------------------------------------------------------------------
        
        return template, vertices
    
    def __getitem__(self, index):
        """
        Args:
            index (int,int,int): batched data indicies 
                idx (int): index for the expression
                id_mesh (int): mesh identity label
                mesh_data (int): label for the data, (0: voca-coma, 1: biwi, 2: mf)
        Returns:
            data
        """
        idx, id_mesh, mesh_data = index
        
        if mesh_data == 0:
            datas = self.get_voca(idx, id_mesh)        
        elif mesh_data == 1:
            datas = self.get_biwi(idx, id_mesh)
        elif mesh_data == 2:
            datas = self.get_multiface_SEN(idx, id_mesh)
        elif mesh_data == 3:
            datas = self.get_coma(idx, id_mesh)
            mesh_data = 0
        elif mesh_data == 4:
            datas = self.get_multiface_ROM(idx, id_mesh)
            mesh_data = 2
        elif mesh_data == 5:
            datas = self.get_ict(idx, id_mesh)
            mesh_data = 5
        elif mesh_data == 6:
            datas = self.get_ict_narrow(idx, id_mesh)
            mesh_data = 5
        else:
            raise ValueError('got wrong number')
            
        mesh_data = torch.tensor(mesh_data)

        # return (*datas, mesh_data)
        (template, deformed, faces, template_normal, deformed_normal, seg, exp_coeff, id_name) = datas

        template, deformed = self.random_trans_scale(template, deformed)
        
        return (template, deformed, faces, template_normal, deformed_normal, seg, exp_coeff, id_name, mesh_data)
    
    def get_slice_idx(self, F_idx, WS):
        """
        Args:
            F_idx (int / np.ndarray): frame index for exp_coeff or vertices
        Returns:
            slice_idx (int): slicing index
        """
        if type(F_idx)==np.ndarray:
            F_idx = F_idx.shape[0]
        if type(F_idx)==torch.Tensor:
            F_idx = F_idx.shape[0]
            
        if self.mode == 'test':
            WS = F_idx
            slice_idx = 0
        else:
            slice_idx = random.randint(0, F_idx-WS)
        return slice_idx
    
    def get_id_num(self, audio_path, template):
        if template.shape[0] == self.ict_face_model.v_idx:
            return self.identity_num[audio_path.split(self.mode)[-1].split('/')[1]]
        
        # elif self.use_voca and template.shape[0] == self.voca_trimesh.vertices.shape[0]:        
        elif self.use_voca and template.shape[0] == self.voca_coma_std['v_idx'].shape[0]:
            return self.identity_num[audio_path.split(self.mode)[-1].split('/')[1]]
        
        elif self.use_biwi and template.shape[0] == self.biwi_trimesh.vertices.shape[0]:
            return self.identity_num[audio_path.split('self.mode')[-1].split('/')[-1].split('_')[0]]
        
        # elif self.use_mf_ROM and template.shape[0] == self.mf_trimesh.vertices.shape[0]:
        #     return
        
        elif self.use_mf_SEN and template.shape[0] == self.mf_trimesh.vertices.shape[0]:
            return self.identity_num[audio_path.split('wav2vec2')[-1].split('/')[-1].split('-SEN')[0]]
    
    def vis_mesh(self, 
                 vertices, # [B, V, 3]
                 faces=None, 
                #  frame=0, 
                 mesh='ict', 
                 tag='', 
                 bg_black=False,
                 size=2,
                 render_mode='shade',
                 logdir='_tmp',
                ):
        if faces is None:
            if mesh == 'ict':
                if vertices.shape[1] == 11248:
                    _, faces = self.ict_face_model.get_random_v_and_f(select=0)
                elif vertices.shape[1] == 9409:
                    _, faces = self.ict_face_model.get_random_v_and_f(select=1)
                else:
                    _, faces = self.ict_face_model.get_random_v_and_f(select=2)
            elif mesh == 'voca':
                faces = self.voca_mesh['face']
            elif mesh == 'biwi':
                faces = self.biwi_mesh['face']
            elif mesh == 'mf':
                faces = self.mf_SEN_mesh['face']
            
        v_list = vertices * 0.8
        len_v = len(v_list)
        
        f_list = [faces]*len_v
        rot_list=[[0,0,0]]*len_v
        
        os.makedirs(logdir, exist_ok=True)
        
        plot_image_array(
            v_list, f_list, rot_list, 
            size=size,
            mode=render_mode,
            bg_black=bg_black, 
            logdir=logdir,
            save=True,
            name=f'{tag}-{mesh}'
        )

class CBDdataSampler(data.Sampler):
    def __init__(self, 
                 len_list, batch_size, shuffle=False, 
                 balance=False, n_sampling=False, n_=4, 
                 reverse=False, is_train=False, is_valid=False
                ):
        self.len_list = len_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reverse = reverse
        self.data = np.array(['voca', 'biwi', 'mf', 'voca', 'mf', 'ict', 'ict'])
        self.n_sampling = n_sampling
        self.n_ = n_
        self.epoch = 0
        
        self.is_train = is_train
        self.mode = 'test'
        if is_train:
            self.mode = 'train'
        elif is_valid:
            self.mode = 'val'
        
        self.indices, self.labels, self.id_mesh = self._get_indices()
        self.total = self.indices.shape[0]
        
        
    def _get_indices(self):
        """
        Returns:
            indices (np.ndarray): indices array for the data [N, Batch]
            labels (np.ndarray): labels for the data, (0: voca-coma, 1: biwi, 2: mf)
            id_mesh (np.ndarray): mesh identity label
        """
        # just empty arrays
        indices = np.zeros(0, dtype=int)
        labels  = np.zeros(0, dtype=int)
        id_mesh = np.zeros(0, dtype=int)
        
        self.len_data=[]
        self.mesh_data=[]
        for len_data, _, mesh_data, id_len_list in self.len_list:
            
            m_data = mesh_data.numpy()
            data_name = self.data[m_data]
            padd = (len_data // id_len_list) % self.batch_size
            
            n_expressions = (len_data // id_len_list) + (self.batch_size-padd)
            
            if self.mode == 'train' and data_name != 'ict':
                n_expressions = 2*n_expressions
            
            _indices = np.tile(np.arange(0, n_expressions, dtype=int), id_len_list)
            _labels = np.arange(0, id_len_list, dtype=int).repeat(n_expressions)
            _id_mesh = np.ones_like(_labels)*m_data
                        
            indices = np.r_[indices, _indices]
            labels = np.r_[labels, _labels]
            id_mesh = np.r_[id_mesh, _id_mesh]
            
            self.len_data.append(len(_indices))
            self.mesh_data.append(data_name)
            
        indices = indices.reshape(-1, self.batch_size)
        labels = labels.reshape(-1, self.batch_size)
        id_mesh = id_mesh.reshape(-1, self.batch_size)
        
        assert indices.shape[0] == labels.shape[0], "miss match!"
        assert indices.shape[0] == id_mesh.shape[0], "miss match!"
        
        return indices, labels, id_mesh
    
    def __iter__(self):
        if self.shuffle:
            idx = np.arange(self.indices.shape[0])
            idx = np.random.permutation(idx)
            
            indices = self.indices[idx]
            labels = self.labels[idx]
            id_mesh = self.id_mesh[idx]
        else:
            indices = self.indices
            labels = self.labels
            id_mesh = self.id_mesh
            
        if self.reverse:
            indices = self.indices[::-1]
            labels = self.labels[::-1]
            id_mesh = self.id_mesh[::-1]
        
        self.length = len(indices)
        
        batch = np.concatenate([indices[:,:,None], labels[:,:,None], id_mesh[:,:,None]], axis=-1)
        batch = batch.tolist()
        return iter(batch)

    def __len__(self):
        return self.total
    
    def get_sampler_config(self):
        text = "========[CBDdataSampler]========\n"
        text += f"[mode]: {self.mode}\n"
        text += f"[Batch size]: {self.batch_size}\n"
        for name, len_data in zip(self.mesh_data, self.len_data):
            text += f"[Batched {name.upper()}]: {len_data}\n"
        text += f"[Batched total len]: {len(self.indices)}\n"
        text += "===============================\n"
        return text
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        

class CBDDataBatch:
    def __init__(self, data):
        """
        Args:
            template: source neutral mesh
            vertices: source deformed mesh
            faces: source mesh trianlge
            mesh_data: dataset index (voca / multiface / biwi)
        """
        if data is not None: # essential !
            transposed_data = list(zip(*data))
                        
            self.template = torch.stack(transposed_data[0], 0) # [B, V, 3]
            #self.template = transposed_data[0][0][None] # [1, V, 3]
            self.vertices = torch.stack(transposed_data[1], 0) # [B, V, 3]
            self.faces = transposed_data[2][0] # # [F, 3]
            
            self.template_normal = torch.stack(transposed_data[3], 0) # [B, V, 3]
            self.vertices_normal = torch.stack(transposed_data[4], 0) # [B, V, 3]
            
            self.mesh_data = transposed_data[-1][0] # 1
            self.segmentation = torch.stack(transposed_data[5], 0) # [B, V, 24]
            
            self.exp_coeff = torch.stack(transposed_data[6], 0) # [B, V, 24]
            self.id_name = transposed_data[7][0] # string
    
    @property
    def get_dfn_info(self): 
        return [self.mass, self.L, self.evals, self.evecs, self.grad_X, self.grad_Y, self.faces]
        
    def set_dfn_info(self, data):
        # DiffusionNet precomputes
        self.mass = data[0]
        self.L = data[1]
        self.evals = data[2]
        self.evecs = data[3]
        self.grad_X = data[4]
        self.grad_Y = data[5]
        #self.faces = torch.stack(dfn_info[6], 0) # -> duplicated!
        
    def set_batch_dfn_info(self, dfn_info):
        # DiffusionNet precomputes
        self.mass = torch.stack(dfn_info[0], 0)
        self.L = torch.stack(dfn_info[1], 0)
        self.evals = torch.stack(dfn_info[2], 0)
        self.evecs = torch.stack(dfn_info[3], 0)
        self.grad_X = torch.stack(dfn_info[4], 0)
        self.grad_Y = torch.stack(dfn_info[5], 0)
        #self.faces = torch.stack(dfn_info[6], 0) # -> duplicated!
    
    def to(self, device='cpu'):
        for id_ in self.__dict__.keys():
            attr = self.__getattribute__(id_)
            if isinstance(attr, torch.Tensor):
                self.__setattr__(id_, attr.to(device))
        return self
        
    # # custom memory pinning method on custom type
    # def pin_memory(self):
    #     self.inp = self.inp.pin_memory()
    #     self.tgt = self.tgt.pin_memory()
    #     return self

def CBD_collate_wrapper(batch, device="cpu"):
    return CBDDataBatch(batch).to(device)

class CBDDataBatch_eval:
    def __init__(self, data):
        """
        Args:
            template: source neutral mesh
            vertices: source deformed mesh
            faces: source mesh trianlge
            mesh_data: dataset index (voca / multiface / biwi)
        """
        if data is not None: # essential !
            transposed_data = list(zip(*data))
            self.vertices = torch.stack(transposed_data[0], 0) # [B, V, 3]            
            self.template = torch.stack(transposed_data[1], 0) # [B, V, 3]            
            
            self.vertices_normal = torch.stack(transposed_data[2], 0) # [B, V, 3]
            self.template_normal = torch.stack(transposed_data[3], 0) # [B, V, 3]
            
            self.faces = torch.stack(transposed_data[4], 0) # # [F, 3]
            self.mesh_data = transposed_data[-1][0]
            self.id_name = transposed_data[-2][0] # string
            
            #                  [     0,      1,      2,      3,      4]
            # data_name_list = ['voca','biwi','mf_SEN','coma','mf_ROM']
            
    def to(self, device='cpu'):
        for id_ in self.__dict__.keys():
            attr = self.__getattribute__(id_)
            if isinstance(attr, torch.Tensor):
                self.__setattr__(id_, attr.to(device))
        return self
        
    # # custom memory pinning method on custom type
    # def pin_memory(self):
    #     self.inp = self.inp.pin_memory()
    #     self.tgt = self.tgt.pin_memory()
    #     return self

def CBD_collate_wrapper_eval(batch, device="cpu"):
    return CBDDataBatch_eval(batch).to(device)

if __name__ == "__main__":
    """
    python dataloader_CBD.py
    """
    import yaml; import argparse
    from tqdm import tqdm
    import time
    
    from utils import plot_image_array, plot_image_array_diff3, vis_rig 
    
    def set_seed(opts):
        # set seed
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opts.seed)
        random.seed(opts.seed)
    
    
    opts_yaml = yaml.load(open('config/train.yml'), Loader=yaml.FullLoader)
    opts_yaml["learn_rig_emb"] = False
    #opts_yaml["device"] = "cuda:0"
    opts_yaml["device"] = "cpu"
    opts = argparse.Namespace(**opts_yaml)
    
    set_seed(opts)

    #### evaluation data
    # eval_dataset = EvalDataset(data_name='mf_SEN')
    # eval_dataloader = torch.utils.data.DataLoader(
    #     eval_dataset,
    #     num_workers=8,
    #     shuffle=False,
    #     batch_size=1,
    # )
    
    # len_eval_dataloader = len(eval_dataloader)
    # pbar = tqdm(enumerate(eval_dataloader), total=len_eval_dataloader)
    # mode = eval_dataset.data_name
    # #import pdb;pdb.set_trace()

    # logdir = 'tmp_'
    # os.makedirs(logdir, exist_ok=True)
    # for idx, data in pbar:
    #     vertices, template, vertices_normal, template_normal, faces = data
    #     pbar.set_description(f"{idx:5d}-{mode},{vertices.shape}, {template.shape}")
        
    #     v_list = [vertices[0] * 0.8, template[0] * 0.8] 
    #     len_v = len(v_list)
        
    #     f_list = [faces[0]]*len_v
    #     plot_image_array(
    #         v_list, f_list, rot_list=[[0,0,0]]*len_v,
    #         size=2, bg_black=True,  logdir=logdir, save=True, name=f'{idx:05d}-{mode}'
    #     )
    #     if idx == 50:
    #         break
    # import pdb;pdb.set_trace()


    
    
    opts.batch_size = 16
    print(f'use batch_size: {opts.batch_size}')
    
    dataset = CBDDataset(
        opts,
        is_train=True, 
        is_valid=False,
        toggle=True,
        use_voca=False,
        use_coma=False,
        use_biwi=False,
        use_mf_SEN=True,
        use_mf_ROM=True,
        use_ict=True,
    )
    print(dataset.get_data_config())
        
    
    sampler = CBDdataSampler(
        dataset.len_list, 
        opts.batch_size,
        shuffle=True,
        balance=False,
        n_sampling=opts.n_sampling,
        n_=opts.batch_size,
        # reverse=True,
        is_train=True,
        # is_train=False,
        # is_valid=True
    )
    print('n_sampling', opts.n_sampling)
    print(sampler.get_sampler_config())
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_sampler=sampler, 
        collate_fn=partial(CBD_collate_wrapper, device=opts.device),
        num_workers=0
    )
    
    import pdb;pdb.set_trace()
#     dataloader = torch.utils.data.DataLoader(
#         dataset, 
#         batch_size=opts.batch_size, 
#         shuffle=False,
#         collate_fn=partial(CBD_collate_wrapper, device=opts.device), 
#         num_workers=0
#     )

    len_dataloader = len(dataloader)
    #data = next(iter(dataloader))
    # iter_dataloader = iter(dataloader)
    # pbar = tqdm(enumerate(dataloader), total=len_dataloader, position=0)
    # rambar = tqdm(range(len_dataloader), total=125, desc='ram', position=1, ncols=100)
    # cpubar = tqdm(range(len_dataloader), total=100, desc='cpu', position=2, ncols=100)
    
    
    
    pbar = tqdm(enumerate(dataloader), total=len_dataloader)
    for idx, batch in pbar:
        #print(batch.vertices.shape, batch.template.shape)
        
        mode = np.array(['voca', 'biwi', 'mf', ',',',','ict'])[batch.mesh_data]
        
        #pbar.set_description(f"{idx}-{mode}")
        pbar.set_description(f"{idx}-{mode},{batch.vertices.shape}, {batch.template.shape}")
        # plot_image_array(
        #         v_list, f_list, 
        #         rot_list=[[0,0,0]] * len_v, 
        #         size=1, bg_black=False, mode='shade',
        #         logdir=save_logdir,
        #         name=save_img_name, save=True
        #     )
        #print(template.min(), template.max())
        
        # data_cat = torch.cat([batch.template.cpu(), batch.vertices.cpu()],dim=0)
        # dataset.vis_mesh(data_cat, mesh=mode,tag=f"{idx:06d}-cat-ntrl_dfrm",size=1)
        
        dataset.vis_mesh(batch.vertices.cpu(), mesh=mode,tag=f"{idx:06d}-dfrm",size=1)
        dataset.vis_mesh(batch.template.cpu(), mesh=mode,tag=f"{idx:06d}-ntrl",size=1)
