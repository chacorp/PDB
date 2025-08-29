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
                 data_basedir='/data/sihun',
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
                
        self.len_list.append([self.n_components*self.voca_len, self.get_voca, torch.tensor(0), self.voca_len])
        
        
        
        self.biwi_pca_holder_list=[]
        self.biwi_id_list=[]
        for id_name in biwi_data_split[self.mode]:
            self.biwi_id_list.append(id_name)
            npz_file = f"{data_basedir}/BIWI_align_deci/{self.mode}/vertices_npy/{id_name}_pca.npz"
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
        
        self.len_list.append([self.n_components*self.biwi_len, self.get_biwi, torch.tensor(1), self.biwi_len])
        
        
        self.mf_pca_holder_list=[]
        self.mf_id_list=[]
        for id_name in mf_data_split[self.mode]:
            self.mf_id_list.append(id_name)
            npz_file = f"{data_basedir}/multiface_align/SEN/{self.mode}/vertices_npy/{id_name}_pca.npz"
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
        
        self.len_list.append([self.n_components*self.mf_len, self.get_multiface, torch.tensor(2), self.mf_len])
        
        
        # 2 for -weight and +weight
        self.min_sample = 1
        total_len = self.n_components * total_id * self.min_sample
        
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
        text+= f"[sample range]: {self.min_sample} ([-1, +1] x variance) \n"
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
            
    def get_voca(self, index):
        #pca_index = index % self.n_components
        id_index = index // self.n_components
        
        id_name = self.voca_id_list[id_index]
        pca_holder = self.voca_pca_holder_list[id_index]
        
        # if index < self.n_components:
        if False:
            vertex = pca_holder.sample_from_pca_one_axis(scale=1.0, select=index, verbose=False)
        else:
            vertex = pca_holder.sample_from_pca(scale=1.0)
        template = self.voca_mesh[id_name]
        faces = self.voca_mesh["face"]
        
        template = torch.tensor(template).float()
        vertex = torch.tensor(vertex).float()
        faces = torch.tensor(faces).long()
        return (template, vertex, faces)
            
    def get_biwi(self, index):
        #pca_index = index % self.n_components
        id_index = index // self.n_components
        
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
    
    def get_multiface(self, index):
        #pca_index = index % self.n_components
        id_index = index // self.n_components
        
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
        idx = index
        
        #select = random.randint(0, 2)
        for len_data, get_data, mesh_data, id_sent_len in self.len_list:
        ## for _, get_data, mesh_data, id_sent_len in self.len_list:
            if idx < len_data:
                datas = get_data(idx) if mesh_data == 0 else get_data(idx)
                break
            else:
                idx = idx - len_data
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
                 size=3,
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
    def __init__(self, len_list, batch_size, shuffle=False, balance=False, n_sampling=False, n_=4, reverse=False, is_train=True):
        self.len_list = len_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reverse = reverse
        self.mode = np.array(['voca', 'biwi', 'mf'])
        self.n_sampling = n_sampling
        self.n_ = n_
        self.epoch = 0
        self.is_train = is_train
        
        self.indices, self.labels = self._get_indices()
        self.total = self.indices.shape[0]
        
        
    def _get_indices(self):
        """
        Returns:
            indices (np.ndarray): indices array for the data [N, Batch]
            labels (np.ndarray): labels for the data, just for debugging
        """
        indices = np.zeros(0, dtype=int)
        labels = np.zeros(0, dtype=int) # for debugging
        SS = 0
        
        # m_data == 0: # voca or coma
        # m_data == 1: # biwi
        # m_data == 2: # mf
        for len_data, _, mesh_data, id_len_list in self.len_list:
            #print(len_data, id_len_list)
            m_data = mesh_data.numpy()
            SS_next = SS+len_data
            _indices = np.arange(SS, SS_next, dtype=int)
            
            if self.is_train:
                if m_data == 0: # voca or coma
                    tile_n = 15
                if m_data == 1: # biwi
                    tile_n = 20
                if m_data == 2: # mf
                    tile_n = 10
                _indices = np.tile(_indices, tile_n)
                len_data = len_data * tile_n
                
            if self.n_sampling:
                _remain = len_data % (self.batch_size * self.n_)
            else:
                _remain = len_data % self.batch_size
                
            if _remain > 0:
                indices = np.r_[indices, _indices[0:-_remain]]
                labels = np.r_[labels, np.ones(len_data-_remain) * m_data]
            else:
                indices = np.r_[indices, _indices]
                labels = np.r_[labels, np.ones(len_data) * m_data]
            SS = SS_next
                
        if self.n_sampling:
            indices = indices.reshape(-1, self.batch_size, self.n_).transpose(0, 2, 1).reshape(-1, self.batch_size)
            labels = labels.reshape(-1, self.batch_size, self.n_).transpose(0, 2, 1).reshape(-1, self.batch_size)
        else:
            indices = indices.reshape(-1, self.batch_size)
            labels = labels.reshape(-1, self.batch_size)
        
        assert indices.shape[0] == labels.shape[0], "miss match!"
                
        return indices, labels
    
    def __iter__(self):
        
        idx = np.arange(self.indices.shape[0])
            
        if self.shuffle:
            idx = np.random.permutation(idx)
            # self.indices = self.indices[idx]
            # self.labels = self.labels[idx]
            indices = self.indices[idx]
            labels = self.labels[idx]
        else:
            indices = self.indices
            labels = self.labels
            
        if self.reverse:
            indices = self.indices[::-1]
            labels = self.labels[::-1]
        
        batch = indices.tolist()
        
        # select = np.tile(
        #     np.random.randint(3, size=indices.shape[0]), self.batch_size
        # ).reshape(self.batch_size,-1).transpose()
        
        #------------------------------------
        # select = np.zeros_like(indices) # fullhead
        # # select = np.ones_like(indices) # face_only
        
        # batch = np.concatenate([select[:,:,None], indices[:,:,None]], axis=-1)
        # batch = batch.tolist()
        #------------------------------------
        
        self.length = len(indices)
        
        return iter(batch)

    def __len__(self):
        return self.total
    
    def get_sampler_config(self):
        text = "========[CBDdataSampler]========\n"
        text += f"[Batch size]: {self.batch_size}\n"
        for i, mode in enumerate(self.mode):
            mode_len = len(np.where(self.labels[:,0]==i)[0])
            text += f"[Batched {mode.upper()}]: {mode_len}\n"
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
                        
            #self.template = torch.stack(transposed_data[0], 0)
            self.template = transposed_data[0][0][None] # [1, V, 3]
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

if __name__ == "__main__":
    """
    python dataloader_CBD.py
    """
    
    import yaml; import argparse
    from tqdm import tqdm
    opts_yaml = yaml.load(open('config/train.yml'), Loader=yaml.FullLoader)
    opts_yaml["learn_rig_emb"] = False
    #opts_yaml["device"] = "cuda:0"
    opts_yaml["device"] = "cpu"
    opts = argparse.Namespace(**opts_yaml)
    
    # opts.selection = 4 # BIWI
    # opts.selection = 3 # VOCASET
    opts.selection = 20 # ict and multiface (mf)
    # opts.selection = 1 # ict-capture
    # opts.selection = 2
    
    # opts.selection = 5 # mf SEN
    # opts.selection = 6 # mf ROM
    
    opts.window_size = 1
    print(f'use window_size: {opts.window_size}')
    
    opts.batch_size = 8
    print(f'use batch_size: {opts.batch_size}')
    
    #dataset = InvRigDataset(opts, is_train=True)
    #dataset = MeshDataset(opts, is_train=True, is_valid=False)
    # dataset = MeshDataset(opts, is_train=False, is_valid=True)
    # dataset = MeshDataset(opts, is_train=False, is_valid=False)
    
    dataset = CBDDataset(opts, is_train=True, is_valid=False)
    
    
    # total_len, id_sent_len = dataset.set_multiface_SEN()
    # print(total_len, id_sent_len)
    
    # total_len, id_sent_len = dataset.set_multiface_ROM()
    # print(total_len, id_sent_len)
    
    # total_len, id_sent_len = dataset.set_voca()
    # print(total_len, id_sent_len)
    
    # total_len, id_sent_len = dataset.set_coma()
    # print(total_len, id_sent_len)
    
    # total_len, id_sent_len = dataset.set_biwi()
    # print(total_len, id_sent_len)
    
    # dataset.biwi_len_list
    # total_len, id_sent_len = dataset.set_ict_synth()
    # print(total_len, id_sent_len)
    # total_len, id_sent_len = dataset.set_ict_real()
    # print(total_len, id_sent_len)
    print(dataset.get_data_config())
    # dataset.ict_real_len_list
    
    # dataset = NFSDataset(opts, is_train=True, return_audio_dir=True)
    # print(dataset.get_data_config())
    
    sampler = CBDdataSampler(
        dataset.len_list, 
        opts.batch_size,
        #shuffle=True,
        balance=False,
        n_sampling=opts.n_sampling,
        n_=opts.batch_size,
        reverse=True,
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
        print(batch.vertices.shape, batch.template.shape)
        #(audio_feat, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img), mesh_data, audio_path = batch
        #print(audio_path[0], audio_feat.shape, id_coeff.shape, gt_rig_params.shape, template.shape, vertices.shape)
        
        mode = np.array(['voca', 'biwi', 'mf'])[batch.mesh_data]
#         # print(idx, mode, batch.audio_feat.shape, batch.gt_rig_params.shape, batch.template.shape, batch.vertices.shape, batch.normals.shape)
#         pbar.set_description(f"{idx}-{mode}")
#         # plot_image_array(
#         #         v_list, f_list, 
#         #         rot_list=[[0,0,0]] * len_v, 
#         #         size=1, bg_black=False, mode='shade',
#         #         logdir=save_logdir,
#         #         name=save_img_name, save=True
#         #     )
#         #print(template.min(), template.max())
        dataset.vis_mesh(batch.vertices.cpu(), mesh=mode,tag=f"{idx:06d}")
        dataset.vis_mesh(batch.template.cpu(), mesh=mode,tag=f"{idx:06d}")
