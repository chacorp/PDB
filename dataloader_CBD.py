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

from utils import (
    ICT_face_model, 
    procrustes_LDM, 
    plot_image_array, 
    calc_norm_torch
)
from utils.keys import get_data_splits, get_identity_num, ICT_KEYS, DATA_KEYS, KEYS
from utils.remesh_utils import map_vertices, decimate_mesh_vertex
from utils.mesh_utils import get_dfn_info2, get_mesh_operators
from utils.exp_utils import PCA_holder, adjacency_matrix
from tqdm import tqdm
import time

class CBDDataset(data.Dataset):
    def __init__(self, 
                 opts,
                 #data_basedir='/data/sihun',
                 data_basedir='/data/sihun/pca',
                 is_train=False,
                 is_valid=False,
                 window_size=8, # batch size
                 device='cpu',
                 print_config=False,
                 ict_face_only=False,
                 n_components=100, ## mf, voca, biwi all used 100
                ):
        super().__init__()
        # get basenames
        self.opts = opts
        self.is_train = is_train
        self.is_valid = is_valid
        self.device = device
        self.data_basedir = data_basedir
        self.n_components = n_components
        #self.data_config(self.opts, window_size)
        
        self.mode = 'test'
        if is_train:
            self.mode = 'train'
        elif is_valid:
            self.mode = 'val'
        
        total_len = 0
        total_id = 0
        
        _, voca_data_split, biwi_data_split, mf_data_split, _ = get_data_splits()
        
        ## to make no leftover for each mesh id
        self.min_sample = self.n_components % self.opts.batch_size
        self.len_list=[]
        
        
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

        # adj_mat = igl.adjacency_matrix(self.voca_mesh["face"])
        # degree = np.asarray(adj_mat.sum(axis=1)).squeeze()
        # adj_mat_norm = scipy.sparse.diags(1/degree) @ adj_mat
        # self.voca_adj_matrix = torch.tensor(adj_mat_norm.todense()).float().to_sparse().to(self.device)
        # self.voca_adj_list = igl.adjacency_list(self.voca_mesh["face"])
                
        self.len_list.append([(self.n_components+self.min_sample)*self.voca_len, self.get_voca, torch.tensor(0), self.voca_len])
        
        
        
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
        
        # adj_mat = igl.adjacency_matrix(self.biwi_mesh["face"])
        # degree = np.asarray(adj_mat.sum(axis=1)).squeeze()
        # adj_mat_norm = scipy.sparse.diags(1/degree) @ adj_mat
        # self.biwi_adj_matrix = torch.tensor(adj_mat_norm.todense()).float().to_sparse().to(self.device)
        # self.biwi_adj_list = igl.adjacency_list(self.biwi_mesh["face"])
        
        self.len_list.append([(self.n_components+self.min_sample)*self.biwi_len, self.get_biwi, torch.tensor(1), self.biwi_len])
        
        
        self.mf_pca_holder_list=[]
        self.mf_id_list=[]
        for id_name in mf_data_split[self.mode]:
            self.mf_id_list.append(id_name)
            
            npz_file = f"{data_basedir}/multiface_align/SEN/{self.mode}/vertices_npy/{id_name}_pca.npz"
            if not os.path.exists(npz_file):
                npz_file = f"{data_basedir}/multiface_align/SEN/{self.mode}/{id_name}_pca.npz"
                
            self.mf_pca_holder_list.append(PCA_holder(npz_file))
        assert len(self.mf_pca_holder_list) == len(self.mf_id_list), "mismatch in mf"
            
        self.mf_len = len(self.mf_pca_holder_list)
        self.mf_std = np.load("utils/mf/standardization.npy", allow_pickle=True).item()
        with open(f"{data_basedir}/multiface_align/mf_templates.pkl",'rb') as f:
            self.mf_mesh = pickle.load(f)
        total_id = total_id + self.mf_len
        
        # adj_mat = igl.adjacency_matrix(self.mf_mesh["face"])
        # degree = np.asarray(adj_mat.sum(axis=1)).squeeze()
        # adj_mat_norm = scipy.sparse.diags(1/degree) @ adj_mat
        # self.mf_adj_matrix = torch.tensor(adj_mat_norm.todense()).float().to_sparse().to(self.device)
        # self.mf_adj_list = igl.adjacency_list(self.mf_mesh["face"])
        
        self.len_list.append([(self.n_components+self.min_sample)*self.mf_len, self.get_multiface, torch.tensor(2), self.mf_len])
        
        
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
            
    def get_voca(self, index, id_index):
        #pca_index = index % self.n_components
        #id_index = index // (self.n_components+self.min_sample)
        #id_index = index // self.n_components
        #id_index = index % self.voca_len
        
        id_name = self.voca_id_list[id_index]
        pca_holder = self.voca_pca_holder_list[id_index]
        template = self.voca_mesh[id_name]
        faces = self.voca_mesh["face"]
        
        # if index < self.n_components:
        if False:
            vertex = pca_holder.sample_from_pca_one_axis(scale=1.0, select=index, verbose=False)
        else:
            vertex = pca_holder.sample_from_pca(scale=1.0)
            
        template = torch.tensor(template).float()
        vertex = torch.tensor(vertex).float()
        faces = torch.tensor(faces).long()
        return (template, vertex, faces)
            
    def get_biwi(self, index, id_index):
        #pca_index = index % self.n_components
        #id_index = index // (self.n_components+self.min_sample)
        #id_index = index // self.n_components
        #id_index = index % self.biwi_len
        
        id_name = self.biwi_id_list[id_index]
        pca_holder = self.biwi_pca_holder_list[id_index]
        
        # if index < self.n_components:
        if False:
            vertex = pca_holder.sample_from_pca_one_axis(scale=1.0, select=index, verbose=False)
        else:
            vertex = pca_holder.sample_from_pca(scale=1.0)
        template = self.biwi_mesh[id_name]
        faces = self.biwi_mesh["face"]
        
        template = torch.tensor(template).float()
        vertex = torch.tensor(vertex).float()
        faces = torch.tensor(faces).long()
        return (template, vertex, faces)
    
    def get_multiface(self, index, id_index):
        #pca_index = index % self.n_components
        #id_index = index // (self.n_components+self.min_sample)
        #id_index = index // self.n_components
        #id_index = index % self.mf_len
        
        id_name = self.mf_id_list[id_index]
        pca_holder = self.mf_pca_holder_list[id_index]
        
        # if index < self.n_components:
        if False:
            vertex = pca_holder.sample_from_pca_one_axis(scale=1.0, select=index, verbose=False)
        else:
            vertex = pca_holder.sample_from_pca(scale=1.0)
        template = self.mf_mesh[id_name]
        faces = self.mf_mesh["face"]
        
        template = torch.tensor(template).float()
        vertex = torch.tensor(vertex).float()
        faces = torch.tensor(faces).long()
        return (template, vertex, faces)
    
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
            t_range = 0.1
            trans = (torch.rand((1, 3))*t_range - t_range*0.5)
        if self.opts.data_rand_scale:
            scale = torch.rand((1)).repeat(3) * 0.4 + 0.8
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
        else:
            datas = self.get_multiface(idx, id_mesh)
        mesh_data = torch.tensor(mesh_data)
        return (*datas, mesh_data)
    
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
                faces = self.mf_mesh['face']
            
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


class CBDDataset2(data.Dataset):
    def __init__(self, 
                 opts,
                 #data_basedir='/data/sihun',
                 data_basedir='/data/sihun/pca',
                 is_train=False,
                 is_valid=False,
                 window_size=8, # batch size
                 device='cpu',
                 print_config=False,
                 ict_face_only=False,
                 n_components=100, ## mf, voca, biwi all used 100
                ):
        super().__init__()
        # get basenames
        self.opts = opts
        self.is_train = is_train
        self.is_valid = is_valid
        self.device = device
        self.data_basedir = data_basedir
        self.n_components = n_components
        #self.data_config(self.opts, window_size)
        
        self.mode = 'test'
        if is_train:
            self.mode = 'train'
        elif is_valid:
            self.mode = 'val'
        
        total_len = 0
        total_id = 0
        
        _, voca_data_split, biwi_data_split, mf_data_split, _ = get_data_splits()
        
        ## to make no leftover for each mesh id
        self.min_sample = self.n_components % self.opts.batch_size
        self.len_list=[]
        
        
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
                        
        self.len_list.append([(self.n_components+self.min_sample)*self.voca_len, self.get_voca, torch.tensor(0), self.voca_len])
        
        
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
        
        self.len_list.append([(self.n_components+self.min_sample)*self.biwi_len, self.get_biwi, torch.tensor(1), self.biwi_len])
        
        
        self.mf_pca_holder_list=[]
        self.mf_id_list=[]
        for id_name in mf_data_split[self.mode]:
            self.mf_id_list.append(id_name)
            
            npz_file = f"{data_basedir}/multiface_align/SEN/{self.mode}/vertices_npy/{id_name}_pca.npz"
            if not os.path.exists(npz_file):
                npz_file = f"{data_basedir}/multiface_align/SEN/{self.mode}/{id_name}_pca.npz"
                
            self.mf_pca_holder_list.append(PCA_holder(npz_file))
        assert len(self.mf_pca_holder_list) == len(self.mf_id_list), "mismatch in mf"
            
        self.mf_len = len(self.mf_pca_holder_list)
        self.mf_std = np.load("utils/mf/standardization.npy", allow_pickle=True).item()
        with open(f"{data_basedir}/multiface_align/mf_templates.pkl",'rb') as f:
            self.mf_mesh = pickle.load(f)
        total_id = total_id + self.mf_len
        
        self.len_list.append([(self.n_components+self.min_sample)*self.mf_len, self.get_multiface, torch.tensor(2), self.mf_len])
        
        
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
            
    def get_voca(self, index, id_index):
        
        id_name = self.voca_id_list[id_index]
        pca_holder = self.voca_pca_holder_list[id_index]
        template = self.voca_mesh[id_name]

        id_index2 = len(self.voca_id_list)-1 if id_index==0 else id_index - 1
        id_name2 = self.voca_id_list[id_index2]
        pca_holder2 = self.voca_pca_holder_list[id_index2]
        template2 = self.voca_mesh[id_name2]
        
        faces = self.voca_mesh["face"]
        
        # if index < self.n_components:
        if False:
            vertex = pca_holder.sample_from_pca_one_axis(scale=2.0, select=index, verbose=False)
        # else:
        #     vertex = pca_holder.sample_from_pca(scale=2.0)
        else:
            vertex = pca_holder.sample_from_pca(scale=2.0)
            vertex2 = vertex - template + template2
                
        template = torch.tensor(template).float()
        vertex = torch.tensor(vertex).float()
        
        template2 = torch.tensor(template2).float()
        vertex2 = torch.tensor(vertex2).float()
        
        faces = torch.tensor(faces).long()
        return (template, vertex, template2, vertex2, faces)
            
    def get_biwi(self, index, id_index):
        #pca_index = index % self.n_components
        #id_index = index // (self.n_components+self.min_sample)
        #id_index = index // self.n_components
        #id_index = index % self.biwi_len
        
        id_name = self.biwi_id_list[id_index]
        pca_holder = self.biwi_pca_holder_list[id_index]
        template = self.biwi_mesh[id_name]

        id_index2 = len(self.biwi_id_list)-1 if id_index==0 else id_index - 1
        id_name2 = self.biwi_id_list[id_index2]
        pca_holder2 = self.biwi_pca_holder_list[id_index2]
        template2 = self.biwi_mesh[id_name2]
        
        faces = self.biwi_mesh["face"]
        
        # if index < self.n_components:
        if False:
            vertex = pca_holder.sample_from_pca_one_axis(scale=2.0, select=index, verbose=False)
        # else:
        #     vertex = pca_holder.sample_from_pca(scale=2.0)
        else:
            vertex = pca_holder.sample_from_pca(scale=2.0)
            vertex2 = vertex - template + template2
        
        template = torch.tensor(template).float()
        vertex = torch.tensor(vertex).float()
        
        template2 = torch.tensor(template2).float()
        vertex2 = torch.tensor(vertex2).float()
        
        faces = torch.tensor(faces).long()
        return (template, vertex, template2, vertex2, faces)
    
    def get_multiface(self, index, id_index):
        #pca_index = index % self.n_components
        #id_index = index // (self.n_components+self.min_sample)
        #id_index = index // self.n_components
        #id_index = index % self.mf_len
        
        id_name = self.mf_id_list[id_index]
        pca_holder = self.mf_pca_holder_list[id_index]
        template = self.mf_mesh[id_name]

        id_index2 = len(self.mf_id_list)-1 if id_index==0 else id_index - 1
        id_name2 = self.mf_id_list[id_index2]
        pca_holder2 = self.mf_pca_holder_list[id_index2]
        template2 = self.mf_mesh[id_name2]
        
        faces = self.mf_mesh["face"]
        
        # if index < self.n_components:
        if False:
            vertex = pca_holder.sample_from_pca_one_axis(scale=2.0, select=index, verbose=False)
        # else:
        #     vertex = pca_holder.sample_from_pca(scale=2.0)
        else:
            vertex = pca_holder.sample_from_pca(scale=2.0)
            vertex2 = vertex - template + template2
        
        template = torch.tensor(template).float()
        vertex = torch.tensor(vertex).float()
        
        template2 = torch.tensor(template2).float()
        vertex2 = torch.tensor(vertex2).float()
        
        faces = torch.tensor(faces).long()
        return (template, vertex, template2, vertex2, faces)
    
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
            t_range = 0.1
            trans = (torch.rand((1, 3))*t_range - t_range*0.5)
        if self.opts.data_rand_scale:
            scale = torch.rand((1)).repeat(3) * 0.4 + 0.8
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
        else:
            datas = self.get_multiface(idx, id_mesh)
        mesh_data = torch.tensor(mesh_data)
        return (*datas, mesh_data)
    
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
                faces = self.mf_mesh['face']
            
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
                 reverse=False, is_train=True, is_valid=False
                ):
        self.len_list = len_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reverse = reverse
        self.data = np.array(['voca', 'biwi', 'mf'])
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
        
        for len_data, _, mesh_data, id_len_list in self.len_list:
            #print(len_data, id_len_list)
            m_data = mesh_data.numpy()
            
            padd = (len_data // id_len_list) % self.batch_size
            #SS_next += padd
            
            n_expressions = (len_data // id_len_list) +padd
            
            _indices = np.tile(np.arange(0, n_expressions, dtype=int), id_len_list)
            _labels = np.arange(0, id_len_list, dtype=int).repeat(n_expressions)
            _id_mesh = np.ones_like(_labels)*m_data
            
            if self.is_train:
                if m_data == 0: # voca or coma
                    tile_n = 3*(self.batch_size//2)
                if m_data == 1: # biwi
                    tile_n = 4*(self.batch_size//2)
                if m_data == 2: # mf
                    tile_n = 2*(self.batch_size//2)
                _indices = np.tile(_indices, tile_n)
                _labels = np.tile(_labels, tile_n)
                _id_mesh = np.tile(_id_mesh, tile_n)
            
            indices = np.r_[indices, _indices]
            labels = np.r_[labels, _labels]
            id_mesh = np.r_[id_mesh, _id_mesh]

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
        text += f"[mode]: {self.mode}"
        text += f"[Batch size]: {self.batch_size}\n"
        for i, data in enumerate(self.data):
            data_len = len(np.where(self.labels[:,0]==i)[0])
            text += f"[Batched {data.upper()}]: {data_len}\n"
        text += f"[Batched len]: {len(self.indices)}\n"
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
            self.mesh_data = transposed_data[3][0] # 1
    
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

class CBDDataBatch2:
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
                        
            self.template = torch.stack(transposed_data[0]+transposed_data[2], 0) # [B, V, 3]
            #self.template = transposed_data[0][0][None] # [1, V, 3]
            self.vertices = torch.stack(transposed_data[1]+transposed_data[3], 0) # [B, V, 3]
            self.faces = transposed_data[4][0] # # [F, 3]
            self.mesh_data = transposed_data[5][0] # 1
    
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

def CBD_collate_wrapper2(batch, device="cpu"):
    return CBDDataBatch2(batch).to(device)

if __name__ == "__main__":
    """
    python dataloader_CBD.py
    """
    import yaml; import argparse
    from tqdm import tqdm
    
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
    
    
    opts.batch_size = 8
    print(f'use batch_size: {opts.batch_size}')
    
    dataset = CBDDataset(opts, is_train=True, is_valid=False)
    print(dataset.get_data_config())
    
    
    sampler = CBDdataSampler(
        dataset.len_list, 
        opts.batch_size,
        shuffle=False,
        balance=False,
        n_sampling=opts.n_sampling,
        n_=opts.batch_size,
        #reverse=True,
        #is_train=True
        is_train=False
    )
    print('n_sampling', opts.n_sampling)
    print(sampler.get_sampler_config())
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_sampler=sampler, 
        collate_fn=partial(CBD_collate_wrapper, device=opts.device),
        num_workers=0
    )
    
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
    
    
    from utils import plot_image_array, plot_image_array_diff3, vis_rig 
    
    pbar = tqdm(enumerate(dataloader), total=len_dataloader)
    for idx, batch in pbar:
        #import pdb;pdb.set_trace()
        #print(batch.vertices.shape, batch.template.shape)
        
        mode = np.array(['voca', 'biwi', 'mf'])[batch.mesh_data]
        
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
        
        data_cat = torch.cat([batch.template.cpu(), batch.vertices.cpu()],dim=0)
        dataset.vis_mesh(data_cat, mesh=mode,tag=f"{idx:06d}-cat-ntrl_dfrm")
        
        #dataset.vis_mesh(batch.vertices.cpu(), mesh=mode,tag=f"{idx:06d}-dfrm")
        #dataset.vis_mesh(batch.template.cpu(), mesh=mode,tag=f"{idx:06d}-ntrl")
