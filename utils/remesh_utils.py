import sys
from pathlib import Path
abs_path = str(Path(__file__).parents[1].absolute())
sys.path+=[abs_path, f"{abs_path}/utils"]

import numpy as np
import torch
from scipy.spatial import cKDTree

from matplotlib_rnd import *

import trimesh
import random

torch.arange(4)
qwe = torch.randperm(4)

class load_obj_mesh():
    def __init__(self, obj_file, to_torch=False, device='cpu'):
        self.device=device
        self._load_obj_mesh(obj_file, to_torch)
        if to_torch:
            self.to_torch()
            
    def to_torch(self):
        self.vertices = torch.tensor(self.vertices).float().to(self.device)
        self.vn = torch.tensor(self.vn).float().to(self.device)
        self.vt = torch.tensor(self.vt).float().to(self.device)
        self.f  = torch.tensor(self.f).long().to(self.device)
        self.ft = torch.tensor(self.ft).long().to(self.device)
        self.fn = torch.tensor(self.fn).long().to(self.device)
        
    def _load_obj_mesh(self, obj_file, to_torch):
        """
        Custom obj reader
    
        Args
        -------
            obj_file (str): file path
        
        Returns
        -------
            obj (EasyDict)
        """
        vertex_position = []
        vertex_normal = []
        vertex_UV = []
        face_indices = []
        face_normal = []
        face_UV = []
        for line in open(obj_file, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                v = list(map(float, values[1:]))
                vertex_position.append(v)
            if values[0] == 'vn':
                vn = list(map(float, values[1:]))
                vertex_normal.append(vn)
            if values[0] == 'vt':
                vt = list(map(float, values[1:]))
                vertex_UV.append(vt)
            if values[0] == 'f':
                f = list(map(lambda x: int(x.split('/')[0]),  values[1:]))
                face_indices.append(f)
                if len(values[1].split('/')) >=2:
                    ft = list(map(lambda x: int(float(x.split('/')[1])),  values[1:]))
                    face_UV.append(ft)
                if len(values[1].split('/')) >=3:
                    ft = list(map(lambda x: int(float(x.split('/')[2])),  values[1:]))
                    face_normal.append(ft)
                    
        self.vertices = np.array(vertex_position)
        self.vn = np.array(vertex_normal)
        self.vt = np.array(vertex_UV)
        self.f  = np.array(face_indices)
        self.ft = np.array(face_UV)
        self.fn = np.array(face_normal)
        
        if self.f.min() > 0:
            self.f  = self.f  - 1
            self.ft = self.ft - 1
            self.fn = self.fn - 1

def find_common_indices(array1, array2):
    common_indices = np.intersect1d(array1, array2)
    return common_indices

def find_indices_in_array1(array1, array2):
    indices = np.array([np.where(array1 == element)[0][0] for element in array2 if element in array1])
    return indices

def compute_origin_vertex_info(original_vertices, subdivided_vertices):
    origin_vertex_indices = []
    threshold = 1e-3

    for j, sub_vertex in enumerate(subdivided_vertices):
        distances = np.sqrt(np.sum((sub_vertex - original_vertices) ** 2, axis=1))

        min_idx = np.argmin(distances)
        if distances[min_idx] <= threshold:
            origin_vertex_indices.append(min_idx)
    return origin_vertex_indices


def compute_added_vertex_info(vertices_A, vertices_B, threshold=1e-8):
    """
    I forgot why I made this ...

    Args
    -------
        vertices_A (np.ndarray): vertices [V, 3]
        vertices_B (np.ndarray): vertices [V, 3]

    Returns
    -------
        added_vertex_info: contains relative_position, index (nn index A), i (index of B)
    """
    tree_A = cKDTree(vertices_A)
    added_vertex_info = []

    for i, vertex_B in enumerate(vertices_B):
        dist, index = tree_A.query(vertex_B, k=1)
        if dist > threshold:
            relative_position = vertex_B - vertices_A[index]
            added_vertex_info.append(np.hstack((relative_position, index, i)))

    return np.array(added_vertex_info)

def reconstruct_mesh(vertices_A, added_vertex_info):
    """
    I forgot why I made this 2...

    Args
    -------
        vertices_A (np.ndarray): vertices [V, 3]
        added_vertex_info (np.ndarray): 

    Returns
    -------
        reconstructed_vertices

    """
    reconstructed_vertices = np.zeros((len(vertices_A), 3))
    
    relative_positions = added_vertex_info[:, :3]
    reference_indices = added_vertex_info[:, 3].astype(int)
    original_order = added_vertex_info[:, 4].astype(int)

    added_vertices_absolute = vertices_A[reference_indices] + relative_positions

    reconstructed_vertices[:len(vertices_A), :] = vertices_A
    reconstructed_vertices[original_order] = added_vertices_absolute

    return reconstructed_vertices

def get_new_mesh(vertices, faces, v_idx, invert=False):
    """make a new mesh using selected vertices

    Args
    -------
        vertices (torch.tensor): [V, 3] array of vertices
        faces (torch.tensor): [F, 3] array of face indices
        v_idx (torch.tensor): [N] list of vertex index to remove from mesh

    Returns
    -------
        updated_verts (torch.tensor): [V', 3] new array of vertices
        updated_faces (torch.tensor): [F', 3] new array of face indices
        updated_verts_idx (torch.tensor): [N] list of vertex index to remove from mesh (fixed)
    """
    max_index = vertices.shape[0]
    new_vertex_indices = torch.arange(max_index)

    if invert:
        mask = torch.zeros(max_index, dtype=torch.bool)
        mask[v_idx] = 1
    else:
        mask = torch.ones(max_index, dtype=torch.bool)
        mask[v_idx] = 0

    updated_verts     = vertices[mask]
    updated_verts_idx = new_vertex_indices[mask]

    index_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(updated_verts_idx)}

    updated_faces = torch.tensor([
                    [index_mapping.get(idx.item(), -1) for idx in face]
                    for face in faces
                ])

    valid_faces = ~torch.any(updated_faces == -1, dim=1)
    updated_faces = updated_faces[valid_faces]
    return updated_verts, updated_faces, updated_verts_idx

def find_corr_indices(vertices, array1, array2):
    """Find corresponding selected vertices indices from the subsampled vertices

    Args:
        vertices (np.array): [V, 3] source vertices
        array1 (np.array): [N] indices for selected vertices (ex. part segmentation)
        array2 (np.array): [M] indices for subsampled vertices (ex. standardized mesh vertices)

    Return
    -------
        indices (np.array): [N ∩ M] corresponding indices in subsampled vertices for selected vertices
    """    
    mask = np.zeros([vertices.shape[0]],dtype=bool)
    mask[array1] = True
    selected_mask = mask[array2]

    arange_ = np.arange(array2.shape[0])
    indices = arange_[selected_mask]
    return indices

def mesh_standardization(
        mesh, 
        mesh_data="voca", 
        util_dir=f"{abs_path}/mesh_utils", 
        return_idx=False
    ):
    """Apply mesh standardization with pre-calculated vertex indices and face indices

    Args
    -------
        mesh (trimesh.Trimesh)
        mesh_data (str)
        util_dir (str)
        return_idx (bool)

    Return
    -------
        new_mesh (trimesh.Trimesh)
    """
    # load info
    basename = os.path.join(util_dir, mesh_data, "standardization.npy")
    info_dict= np.load(basename, allow_pickle=True).item()

    # set new mesh
    new_v = mesh.vertices[info_dict['v_idx']]
    new_f = info_dict['new_f']
    new_mesh = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False, maintain_order=True)
    
    if return_idx:
        return new_mesh, info_dict['v_idx']
    return new_mesh

def compute_average_distance(points, mode='np'):
    """Compute the average Euclidean distance from the origin for a set of points.

    Args
    -------
        points (np.ndarray / torch.tensor): [V, 3] A set of points.
        mode (str): choose between np (numpy) and torch

    Return
    -------
        average_distance (np/torch): Average Euclidean distance.
    """
    if mode=='np':
        distances = np.linalg.norm(points, axis=1)
        average_distance = np.mean(distances)
    else:
        distances = torch.linalg.norm(points, axis=1)
        average_distance = torch.mean(distances)
    return average_distance

def procrustes_LDM(P, Q, LDM=None, mode='np'):
    """# Q = (P*s)@R.T+t
    [P->Q] Find the optimal rotation, translation, and scaling
    that aligns two sets of points P and Q minimizing using landmarks.
    
    Args
    -------
        P (np.ndarray): [V, 3] A set of source points.
        Q (np.ndarray): [V, 3] A set of corresponding target points.

    Returns
    ------- 
        R (np.ndarray): Rotation matrix (R)
        t (np.ndarray): translation vector (t)
        s (np.ndarray): scale factor (s)
    """
    if mode=='np':
        helper = np
    else:
        helper = torch
        
    if LDM is not None:
        P = P[LDM]
        Q = Q[LDM]
    
    # Calculate the centroids of the point sets
    centroid_P = helper.mean(P, axis=0)
    centroid_Q = helper.mean(Q, axis=0)
    
    # Center the points around the origin
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Calculate the average distances from the origin
    avg_dist_P = compute_average_distance(P_centered, mode=mode)
    avg_dist_Q = compute_average_distance(Q_centered, mode=mode)

    # Calculate the scale factor
    s = avg_dist_Q / avg_dist_P

    # Scale the points
    P_scaled = P_centered * s

    # Compute the covariance matrix
    H = P_scaled.T @ Q_centered

    # Perform Singular Value Decomposition
    U, S, Vt = helper.linalg.svd(H)
    
    # Compute the rotation matrix
    R = Vt.T @ U.T

    # Special reflection case
    if helper.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute the translation vector
    t = centroid_Q - R @ (centroid_P * s)

    return R, t, s


def cal_LVE(gt_, pred_, LDM):
    """Lip Vertex Error

    Args
    -------
        pred_ (torch.tensor): predicted vertices
        gt_ (torch.tensor): GT vertices

    Returns
    -------
        max_LVE
        max_LVE_std
        mean_LVE
    """
    gt_ = gt_.detach().cpu().numpy()
    pred_ = pred_.detach().cpu().numpy()
    L2_dis_mouth = np.square(gt_[:,LDM] - pred_[:,LDM]) # [lip_num, T, 3]
    L2_dis_mouth = np.sum(L2_dis_mouth,axis=2) # [T, lip_num]
    L2_dis_mouth_max = np.max(L2_dis_mouth, axis=1) # [T, lip_num]
    L2_dis_mouth_mean = np.mean(L2_dis_mouth, axis=1) # [T, lip_num]
    
    max_LVE = np.mean(L2_dis_mouth_max)
    max_LVE_std = np.std(L2_dis_mouth_max)
    mean_LVE = np.mean(L2_dis_mouth_mean)
    return max_LVE, max_LVE_std, mean_LVE

def align(pred_, gt_, match_index, mode=1):
    """
    Align animations for evaluation

    Args
    -------
        pred_ (np.array): predicted vertices
        gt_ (np.array): GT vertices
        match_index (np.array): selected vertices for kabsch algorithm
        mode (int): select mode
        
    Returns
    -------
        new_pred_np (np.array): aligned vertices
    """
    new_pred_list = []
    T = gt_.shape[0]
    for f_index in range(T):
        if mode == 1: # align per-frame
            R, t, _ = procrustes_LDM(pred_[f_index], gt_[f_index])
            new_pred = (pred_[f_index]) @ R.T + t
        elif mode == 2: # align per-frame with selected vertices
            R, t, _ = procrustes_LDM(pred_[f_index, match_index], gt_[f_index, match_index])
            new_pred = (pred_[f_index]) @ R.T + t
        elif mode == 3: # align to first frame of GT
            R, t, _ = procrustes_LDM(pred_[f_index], gt_[0])
            new_pred = (pred_[f_index]) @ R.T + t
        elif mode == 4: # align using [R|t] from first frames
            R, t, _ = procrustes_LDM(pred_[0], gt_[0])
            new_pred = (pred_[f_index]) @ R.T + t
        else:
            new_pred = pred_
        new_pred_list.append(new_pred.reshape(1, -1, 3))
    new_pred_np = np.concatenate(new_pred_list, axis=0)
    return new_pred_np


def decimate_mesh_vertex(mesh, num_vertex, tolerance=2):
    """
    Decimate the mesh to have approximately the target number of vertices.

    Args
    -------
        mesh (trimesh.Trimesh): Source mesh to decimate.
        num_vertex (int): Target vertex number.

    Returns
    -------
        mesh (trimesh.Trimesh): Decimated mesh.
    """
    # check mesh...
    o_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True, validate=True, maintain_order=True)
    
    #NOTE Euler Characteristic: V - E + F = 2
    num_faces = 100 + 2 * num_vertex
    prev_num_faces = mesh.faces.shape[0]
    
    while abs(mesh.vertices.shape[0] - num_vertex) > tolerance:
        if num_faces == prev_num_faces:
            num_faces = num_faces -1
        elif num_faces > prev_num_faces:
            print('cannot reduce more')
            break
        mesh = mesh.simplify_quadric_decimation(num_faces)
        _, v_map = map_vertices(o_mesh, mesh)
        
        mesh = trimesh.Trimesh(o_mesh.vertices[v_map], mesh.faces, process=True, validate=True, maintain_order=True)
        print("Decimated to", num_faces, "faces, mesh has", mesh.vertices.shape[0], "vertices")
        num_faces -= (mesh.vertices.shape[0] - num_vertex) // 2
        prev_num_faces = num_faces
    print('Output mesh has', mesh.vertices.shape[0], 'vertices and', mesh.faces.shape[0], 'faces')
    return mesh

def decimate_mesh(mesh, num_faces):
    """
    Applying decimation to the mesh

    Args
    ------
        mesh (trimesh.Trimesh): source mesh
        num_faces (int): number of desired triangles

    Returns
    -------
        decimated_mesh (trimesh.Trimesh): decimated source mesh 
    """
    decimated_mesh = mesh.simplify_quadric_decimation(num_faces)
    return decimated_mesh

def map_vertices(original_mesh, decimated_mesh):
    """
    Get nearest neighbor vertex on original mesh

    Args
    ------
        original_mesh (trimesh.Trimesh): source mesh
        decimated_mesh (trimesh.Trimesh): decimated source mesh 

    Returns
    -------
        distances (np.ndarray): distance with the nearest neighbor vertex on original mesh
        vertex_map (np.ndarray): index of the nearest neighbor vertex on original mesh
    """
    tree = cKDTree(original_mesh.vertices)
    distances, vertex_map = tree.query(decimated_mesh.vertices, k=1)
    return distances, vertex_map


def build_padded_neighbors(neighbors_map, device=None):
    """
    Args:
        neighbors_map (list(int)): adjacent vertex indicies
    Returns:
        idx_pad: adjacent vertex indices
        mask: indicates if it is connected
    """
    N = len(neighbors_map)
    M = max((len(v) for v in neighbors_map), default=0)
    idx_pad = torch.full((N, M), -1, dtype=torch.long, device=device)
    mask = torch.zeros((N, M), dtype=torch.bool, device=device)
    for i, idxs in enumerate(neighbors_map):
        if len(idxs) == 0: 
            continue
        L = len(idxs)
        idx_pad[i, :L] = torch.tensor(idxs, dtype=torch.long, device=device)
        mask[i, :L] = True
    return idx_pad, mask

def pca_normal_axis_vectorized(verts, idx_pad, mask, eps=1e-12):
    """
    Args:
        verts (torch.tensor): mesh vertices (B, N, 3)
        idx_pad (torch.tensor): adjacent vertex indices
        mask (torch.tensor): indicates if it is connected
        eps (float, optional): Defaults to 1e-12

    Returns:
        normal axis (sign might be fliped)
    """
    B, N, _ = verts.shape
    assert idx_pad.dim() == 2 and mask.shape == idx_pad.shape
    M = idx_pad.shape[1]
    if M == 0:
        return torch.zeros_like(verts)

    device, dtype = verts.device, verts.dtype

    safe_idx = idx_pad.clamp(min=0)                 # (N, M)
    nbrs = verts[:, safe_idx, :]                    # (B, N, M, 3)
    m = mask.reshape(1, *mask.shape, 1).to(dtype)                  # (1, N, M, 1)
    cnt = mask.sum(dim=1).clamp_min(1).view(1, N, 1, 1).to(dtype)  # (1, N, 1, 1)

    centroid = (nbrs * m).sum(dim=2, keepdim=True) / cnt           # (B, N, 1, 3)
    X = (nbrs - centroid) * m                                      # (B, N, M, 3)

    XT = X.transpose(-2, -1)                                       # (B, N, 3, M)
    Cov = XT @ X                                                   # (B, N, 3, 3)
    Cov = Cov / cnt

    evals, evecs = torch.linalg.eigh(Cov)                          # (B, N, 3), (B, N, 3, 3)
    normals = evecs[..., 0]                                        # (B, N, 3)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(eps)

    few = (mask.sum(dim=1) < 2)
    if few.any():
        normals[:, few, :] = 0

    return normals

class ICT_face_model():
    def __init__(self, 
                 face_only=False, 
                 narrow_only=False, 
                 scale=0.1, 
                 base_dir=None, 
                 use_decimate=False, 
                 device='cpu'):
        """quick load ict face model
        Args:
            face_only (bool): if True, use face region only 
            scale (float): re-scale mesh 
            device (str): device 
        """
        self.device = device
        self.scale  = scale
        base_dir = abs_path if base_dir==None else base_dir
        self.base_dir = base_dir
        
        # get vertices and faces
        ## default: full = face + head + neck
        self.region = {
            0: [11248, 11144], # v_idx, quad_f_idx
            1: [9409, 9230], # v_idx, quad_f_idx # face_only
            2: [6706, 6560], # v_idx, quad_f_idx # narrow_only
        }
        v_idx, quad_f_idx = self.region[0]
        if face_only:
            v_idx, quad_f_idx = self.region[1]
        if narrow_only: # (not used)
            v_idx, quad_f_idx = self.region[2]
        
        self.use_decimate = use_decimate
        self.ict_deci = np.load(f'{base_dir}/utils/ict/ICT_decimate.npz')
        
        ## mesh faces
        quad_Faces = torch.load(f'{base_dir}/ict_face_pt/quad_faces.pt')[:quad_f_idx] #, map_location='cuda:0')
        self.faces = quad_Faces[:, [[0, 1, 2],[0, 2, 3]] ].permute(1, 0, 2).reshape(-1, 3).numpy()
        self.f_num = self.faces.shape[0]
        self.v_num = v_idx

        ## mesh verticies (alignment)
        neutral_verts = (torch.load(f'{base_dir}/ict_face_pt/neutral_verts.pt') * scale) - torch.tensor([0.0, 0.0, 0.5])
        self.neutral_verts = neutral_verts[:v_idx].numpy()

        ## blendshape basis
        self.exp_basis= torch.load(f'{base_dir}/ict_face_pt/exp_basis.pt') * scale
        self.id_basis = torch.load(f'{base_dir}/ict_face_pt/id_basis.pt') * scale
                
        ## send to device
        #self.neutral_verts = self.neutral_verts.to(self.device)
        self.exp_basis= self.exp_basis.numpy()
        self.id_basis = self.id_basis.numpy()
    
    def get_region_num(self, vertices):
        """
        Args:
            vertices: batched vertices or vertices [B, V, 3] or [V, 3]
        """
        N = vertices.shape[-2]
        
        if N == 11248:
            return 0
        elif N == 9409:
            return 1
        else:
            return 2
        
    def get_random_v_and_f(self, select=None, mode='np'):
        """
        Returns:
            tuple(int, np.ndarray)
        """
        if select is None:
            select = random.randint(0, 2)
        
        # 0: face to shoulder | 1: face to neck | 2. narrow face
        v_idx, quad_f_idx = self.region[select]
        
        qf_pth = f'{self.base_dir}/ict_face_pt/quad_faces.pt'
        quad_Faces = torch.load(qf_pth)[:quad_f_idx]
        tri_faces = quad_Faces[:, [[0, 1, 2],[0, 2, 3]] ].permute(1, 0, 2).reshape(-1, 3)
        
        tri_faces = tri_faces.numpy() if mode == 'np' else tri_faces
            
        return v_idx, tri_faces

    def get_mesh(self, mesh_std=False, return_idx=False):
        """
        Args:
            mesh_std (bool): if True, apply `mesh standardization`
        """
        mesh = trimesh.Trimesh(
            vertices=self.neutral_verts, 
            faces=self.faces, 
            process=False, maintain_order=True
        )
        mesh_v_idx = np.arange(self.neutral_verts.shape[0])
        
        ## mesh_standardization
        if mesh_std:
            mesh, mesh_v_idx = mesh_standardization(mesh, mesh_data='ict', return_idx=True)
        
        if return_idx:
            return mesh, mesh_v_idx
        return mesh
    
    def get_random_mesh(self, mesh_std=False, return_idx=False):
        """
        Args:
            mesh_std (bool): if True, apply `mesh standardization`
        """
        id_disps = self.get_id_disp(np.random.rand(100))[0]
        mesh = trimesh.Trimesh(
            vertices=self.neutral_verts + id_disps, 
            faces=self.faces, 
            process=False, maintain_order=True
        )
        mesh_v_idx = np.arange(self.neutral_verts.shape[0])
        
        ## mesh_standardization
        if mesh_std:
            mesh, mesh_v_idx = mesh_standardization(mesh, mesh_data='ict', return_idx=True)
        
        if return_idx:
            return mesh, mesh_v_idx
        return mesh

    def apply_coeffs(self, id_coeff, exp_coeffs=None, mesh_v_idx=None, return_all=False, region=0):
        """
        Args:
            id_coeff (np.ndarray): [100] ICT-facekit identity coeff
            exp_coeffs (np.ndarray): [T, 53] ICT-facekit expression coeff
            mesh_v_idx (np.ndarray): <int> array of std mesh vertex indices
        """
        # id vertices
        id_disps = self.get_id_disp(id_coeff)
        
        # exp vertices
        exp_disp = self.get_exp_disp(exp_coeffs)
        
        id_verts = self.neutral_verts + id_disps
        id_exp_verts = id_verts + exp_disp
        
        # apply std
        if mesh_v_idx is not None:
            id_exp_verts = id_exp_verts[:, mesh_v_idx]
        
        # id_verts = id_verts[:self.region[region][0]]
        # id_exp_verts = id_exp_verts[:self.region[region][0]]
        # import pdb;pdb.set_trace()
        id_verts = id_verts[:, :self.region[region][0]]
        id_exp_verts = id_exp_verts[:, :self.region[region][0]]
        
        if return_all:
            return id_exp_verts, id_verts, exp_disp
        return id_exp_verts
    
    def apply_coeffs_batch(self, id_coeff, exp_coeffs=None, mesh_v_idx=None, return_all=False, region=0):
        """
        Args:
            id_coeff (np.ndarray): [N,100] ICT-facekit identity coeff
            exp_coeffs (np.ndarray): [T, 53] ICT-facekit expression coeff
            mesh_v_idx (np.ndarray): <int> array of std mesh vertex indices
        """
        device = id_coeff.device
        
        # id vertices
        id_disps = self.get_id_disp(id_coeff)
        
        # exp vertices
        exp_disps = self.get_exp_disp(exp_coeffs)
            
        id_verts = self.neutral_verts[:, :self.region[region][0]] + id_disps
        id_exp_verts = id_verts + exp_disps
        
        # apply std
        if mesh_v_idx is not None:
            id_exp_verts = id_exp_verts[:, mesh_v_idx]
            
        id_verts = id_verts[:, :self.region[region][0]]
        id_exp_verts = id_exp_verts[:, :self.region[region][0]]
        
        if return_all:
            return id_exp_verts, id_verts, exp_disps
        return id_exp_verts
    
    def apply_coeffs_batch_torch(self, id_coeff, exp_coeffs=None, mesh_v_idx=None, return_all=False,region=0):
        """
        Args:
            id_coeff (torch.Tensor): [N,100] ICT-facekit identity coeff
            exp_coeffs (torch.Tensor): [T, 53] ICT-facekit expression coeff
            mesh_v_idx (torch.Tensor): <int> array of std mesh vertex indices
        """
        device = id_coeff.device
        
        # id vertices
        #id_disps = self.get_id_disp(id_coeff)
        neutral_verts = self.neutral_verts[:self.region[region][0]]
        
        B = id_coeff.shape[0]
        id_basis_reshaped = torch.from_numpy(self.id_basis.reshape(100, -1)).float().to(device)
        id_disps = torch.mm(id_coeff, id_basis_reshaped).reshape(B, -1, 3)
        id_disps = id_disps[:, :self.region[region][0]]
        
        # exp vertices
        #exp_disps =self.get_exp_disp(exp_coeffs)
        T = exp_coeffs.shape[0]
        exp_basis_reshaped = torch.from_numpy(self.exp_basis.reshape(53, -1)).float().to(device)
        exp_disps = torch.mm(exp_coeffs, exp_basis_reshaped).reshape(T, -1, 3)
        exp_disps = exp_disps[:, :self.region[region][0]]
            
        id_verts = torch.from_numpy(neutral_verts).float().to(device) + id_disps
        id_exp_verts = id_verts + exp_disps
        
        if self.use_decimate:
            mesh_v_idx = self.ict_deci["v_idx"]
        
        if mesh_v_idx is not None:
            id_exp_verts = id_exp_verts[:, mesh_v_idx]
            id_verts = id_verts[:, mesh_v_idx]
            exp_disps = exp_disps[:, mesh_v_idx]
            
        if return_all:
            return id_exp_verts, id_verts, exp_disps
                
        return id_exp_verts
    
    def get_id_disp(self, id_coeff=None, mesh_v_idx=None, region=0):
        """
        Args:
            id_coeff (np.ndarray): [100] / [B, 100] ICT-facekit expression coeff
            mesh_v_idx (np.ndarray): <int> array of std mesh vertex indices
            
        Returns:
            id_disps (np.ndarray): [V, 3] / [B, V, 3]mesh vertices
        """
        # id vertices
        if id_coeff is not None:
            if len(id_coeff.shape) < 2:
                id_coeff = id_coeff[None]
            B = id_coeff.shape[0]
            id_basis_reshaped = self.id_basis.reshape(100, -1)
            id_disps = np.matmul(id_coeff, id_basis_reshaped).reshape(B, -1, 3)
            id_disps = id_disps[:, :self.region[region][0]]
        else:
            id_disps = 0.0

        # apply std
        if mesh_v_idx is not None:
            id_disps = id_disps[:, mesh_v_idx]
            
        return id_disps
    
    def get_exp_disp(self, exp_coeffs=None, mesh_v_idx=None, region=0):
        """
        Args:
            exp_coeffs (np.ndarray): [T, 53] ICT-facekit expression coeff
            mesh_v_idx (np.ndarray): <int> array of std mesh vertex indices

        Returns:
            exp_disps (np.ndarray): [T, V, 3] mesh vertices
        """
        # exp vertices
        if exp_coeffs is not None:
            if len(exp_coeffs.shape) < 2:
                exp_coeffs = exp_coeffs[None]
                
            T = exp_coeffs.shape[0]
            #exp_disps = np.einsum('jk,kls->jls', exp_coeffs.float(), self.exp_basis.to(device))[:, :self.region[region][0]]
            
            exp_basis_reshaped = self.exp_basis.reshape(53, -1)
            exp_disps = np.matmul(exp_coeffs, exp_basis_reshaped).reshape(T, -1, 3)
            exp_disps = exp_disps[:, :self.region[region][0]]
        else:
            exp_disps = 0.0
        
        # apply std
        if mesh_v_idx is not None:
            exp_disps = exp_disps[:, mesh_v_idx]
            
        return exp_disps

    
def compute_MVC(src_v, cage_v, cage_f, eps=1e-8):
    """
    Compute Mean Value Coordinates weights

    Args:
        src_v (torch.tensor): (V, 3) source points
        cage_v (torch.tensor): (Nc, 3) cage/control points
        cage_f (torch.tensor): (F, 3) triangle indices of cage faces
        eps (float): numerical stability epsilon

    Returns:
        weights per cage triangle (V, F, 3)
        NOTE: not normalized
    """
    V = src_v.shape[0]
    F = cage_f.shape[0]

    # face vertex indices
    i0, i1, i2 = cage_f[:, 0], cage_f[:, 1], cage_f[:, 2]

    P0 = cage_v[i0]  # (F, 3)
    P1 = cage_v[i1]
    P2 = cage_v[i2]

    # Expand to (V, F, 3)
    X = src_v[:, None, :]  # (V, 1, 3)
    D0 = P0[None, :, :] - X  # (V, F, 3)
    D1 = P1[None, :, :] - X
    D2 = P2[None, :, :] - X

    d0 = D0.norm(dim=-1).clamp(min=eps)
    d1 = D1.norm(dim=-1).clamp(min=eps)
    d2 = D2.norm(dim=-1).clamp(min=eps)

    U0 = D0 / d0[..., None]
    U1 = D1 / d1[..., None]
    U2 = D2 / d2[..., None]

    # angles
    L0 = (U1 - U2).norm(dim=-1)
    L1 = (U2 - U0).norm(dim=-1)
    L2 = (U0 - U1).norm(dim=-1)

    theta0 = 2 * torch.arcsin(torch.clamp(L0 * 0.5, -1.0, 1.0))
    theta1 = 2 * torch.arcsin(torch.clamp(L1 * 0.5, -1.0, 1.0))
    theta2 = 2 * torch.arcsin(torch.clamp(L2 * 0.5, -1.0, 1.0))

    h = 0.5 * (theta0 + theta1 + theta2)

    near_pi = torch.abs(torch.pi - h) < eps
    not_near_pi = ~near_pi

    s_theta0 = torch.sin(theta0)
    s_theta1 = torch.sin(theta1)
    s_theta2 = torch.sin(theta2)

    c0 = (2 * torch.sin(h) * torch.sin(h - theta0)) / (s_theta1 * s_theta2 + eps) - 1
    c1 = (2 * torch.sin(h) * torch.sin(h - theta1)) / (s_theta2 * s_theta0 + eps) - 1
    c2 = (2 * torch.sin(h) * torch.sin(h - theta2)) / (s_theta0 * s_theta1 + eps) - 1

    s0 = torch.sqrt(torch.clamp(1 - c0**2, 0, 1))
    s1 = torch.sqrt(torch.clamp(1 - c1**2, 0, 1))
    s2 = torch.sqrt(torch.clamp(1 - c2**2, 0, 1))

    # Output weight containers
    W0 = torch.zeros((V, F), dtype=src_v.dtype, device=src_v.device)
    W1 = torch.zeros((V, F), dtype=src_v.dtype, device=src_v.device)
    W2 = torch.zeros((V, F), dtype=src_v.dtype, device=src_v.device)

    # degenerate: h == pi
    sin_theta0 = torch.sin(theta0)
    deg_weight = sin_theta0 * d1 * d2

    W0[near_pi] = deg_weight[near_pi]
    W1[near_pi] = deg_weight[near_pi]
    W2[near_pi] = deg_weight[near_pi]

    # general case
    num0 = (theta0 - c1 * theta2 - c2 * theta1)
    num1 = (theta1 - c2 * theta0 - c0 * theta2)
    num2 = (theta2 - c0 * theta1 - c1 * theta0)

    denom0 = 2 * s1 * s_theta2 * d0
    denom1 = 2 * s2 * s_theta0 * d1
    denom2 = 2 * s0 * s_theta1 * d2

    W0[not_near_pi] = num0[not_near_pi] / (denom0[not_near_pi] + eps)
    W1[not_near_pi] = num1[not_near_pi] / (denom1[not_near_pi] + eps)
    W2[not_near_pi] = num2[not_near_pi] / (denom2[not_near_pi] + eps)

    # stack weights
    return torch.stack([W0, W1, W2], dim=-1)  # (V, F, 3)

def compute_MVC_vertexwise(src_v, cage_v, cage_f, eps=1e-8):
    """
    Compute Mean Value Coordinates per cage vertex for each mesh vertex

    Args:
        src_v (torch.tensor): (V, 3) source points
        cage_v (torch.tensor): (Nc, 3) cage/control points
        cage_f (torch.tensor): (F, 3) triangle indices of cage faces
        eps (float): numerical stability epsilon

    Returns:
        weights per cage vertex (V, Nc)
    """
    V = src_v.shape[0]
    Nc = cage_v.shape[0]
    F = cage_f.shape[0]

    # Get cage face vertices
    i0, i1, i2 = cage_f[:, 0], cage_f[:, 1], cage_f[:, 2]
    P0 = cage_v[i0]  # (F, 3)
    P1 = cage_v[i1]
    P2 = cage_v[i2]

    X = src_v[:, None, :]  # (V, 1, 3)

    D0 = P0[None, :, :] - X  # (V, F, 3)
    D1 = P1[None, :, :] - X
    D2 = P2[None, :, :] - X

    # clamp min by eps to prevent division by 0
    d0 = D0.norm(dim=-1).clamp(min=eps)
    d1 = D1.norm(dim=-1).clamp(min=eps)
    d2 = D2.norm(dim=-1).clamp(min=eps)

    U0 = D0 / d0[..., None]
    U1 = D1 / d1[..., None]
    U2 = D2 / d2[..., None]

    # compute angles between normalized directions
    L0 = (U1 - U2).norm(dim=-1)
    L1 = (U2 - U0).norm(dim=-1)
    L2 = (U0 - U1).norm(dim=-1)

    theta0 = 2 * torch.arcsin(torch.clamp(L0 * 0.5, -0.999999, 0.999999))
    theta1 = 2 * torch.arcsin(torch.clamp(L1 * 0.5, -0.999999, 0.999999))
    theta2 = 2 * torch.arcsin(torch.clamp(L2 * 0.5, -0.999999, 0.999999))
    h = 0.5 * (theta0 + theta1 + theta2)

    near_pi = torch.abs(torch.pi - h) < eps
    not_near_pi = ~near_pi

    s_theta0 = torch.sin(theta0)
    s_theta1 = torch.sin(theta1)
    s_theta2 = torch.sin(theta2)

    # added eps to prevent division by 0
    c0 = (2 * torch.sin(h) * torch.sin(h - theta0)) / (s_theta1 * s_theta2 + eps) - 1
    c1 = (2 * torch.sin(h) * torch.sin(h - theta1)) / (s_theta2 * s_theta0 + eps) - 1
    c2 = (2 * torch.sin(h) * torch.sin(h - theta2)) / (s_theta0 * s_theta1 + eps) - 1

    # added eps to prevent sqrt(0)
    s0 = torch.sqrt(torch.clamp(1 - c0**2, 0, 1)+eps)
    s1 = torch.sqrt(torch.clamp(1 - c1**2, 0, 1)+eps)
    s2 = torch.sqrt(torch.clamp(1 - c2**2, 0, 1)+eps)

    # triangle-wise weights (V, F)
    W0 = torch.zeros((V, F), dtype=src_v.dtype, device=src_v.device)
    W1 = torch.zeros_like(W0)
    W2 = torch.zeros_like(W0)

    deg_weight = s_theta0 * d1 * d2
    W0[near_pi] = deg_weight[near_pi]
    W1[near_pi] = deg_weight[near_pi]
    W2[near_pi] = deg_weight[near_pi]

    num0 = (theta0 - c1 * theta2 - c2 * theta1)
    num1 = (theta1 - c2 * theta0 - c0 * theta2)
    num2 = (theta2 - c0 * theta1 - c1 * theta0)

    denom0 = 2 * s1 * s_theta2 * d0
    denom1 = 2 * s2 * s_theta0 * d1
    denom2 = 2 * s0 * s_theta1 * d2

    # added eps to prevent division by 0
    W0[not_near_pi] = num0[not_near_pi] / (denom0[not_near_pi] + eps)
    W1[not_near_pi] = num1[not_near_pi] / (denom1[not_near_pi] + eps)
    W2[not_near_pi] = num2[not_near_pi] / (denom2[not_near_pi] + eps)

    # Final (V, Nc) weight matrix
    W_vert = torch.zeros((V, Nc), dtype=src_v.dtype, device=src_v.device)
    W_vert.index_add_(1, i0, W0)
    W_vert.index_add_(1, i1, W1)
    W_vert.index_add_(1, i2, W2)

    # Normalize to ensure partition of unity
    W_vert = W_vert / (W_vert.sum(dim=1, keepdim=True) + eps)

    return W_vert  # (V, Nc)


def apply_MVC_weights(W, cage_f, cage_function, eps=1e-8):
    """
    Args:
        W: (V, F, 3) tensor of MVC weights
        cage_f: (F, 3) LongTensor of face indices
        cage_function: (Nc, 3) tensor of control values (e.g., positions or attributes)
    
    Returns:
        deformed target mesh (V, 3)
    """
    V, F, _ = W.shape
    device = W.device

    # Get function values per triangle vertex: (F, 3, 3)
    # [F, 3] index -> [F, 3, 3]
    tri_values = cage_function[cage_f]  # (F, 3, 3)

    # Broadcast to (V, F, 3, 3): function value per source-vertex and triangle
    tri_values = tri_values.unsqueeze(0).expand(V, -1, -1, -1)  # (V, F, 3, 3)
    weights = W.unsqueeze(-1)                                   # (V, F, 3, 1)

    weighted_sum = (weights * tri_values).sum(dim=2)  # (V, F, 3)
    total = weighted_sum.sum(dim=1)                   # (V, 3)
    
    weight_sum = W.sum(dim=(1,2), keepdim=True).clamp(min=eps)  # (V, 1, 1)
    result = total / weight_sum.squeeze(-1)                     # (V, 3)

    return result

def apply_MVC_weights_batch(W, cage_f, deformed_cages, eps=1e-8):
    """
    Apply fixed MVC weights (V, F, 3) to batch of deformed cages (B, Nc, 3)

    Args:
        W (torch.tensor): (V, F, 3) fixed MVC weights
        cage_f (torch.LongTensor): (F, 3) triangle indices
        deformed_cages (torch.tensor): (B, Nc, 3) batch of deformed cage positions

    Returns:
        deformed target mesh per batch (B, V, 3)
    """
    B, Nc, _ = deformed_cages.shape
    V, F, _ = W.shape
    device = deformed_cages.device

    # Get face-wise cage values: (B, F, 3, 3)
    tri_values = deformed_cages[:, cage_f]  # (B, F, 3, 3)

    # Broadcast weights: (1, V, F, 3, 1)
    weights = W[None, :, :, :, None]  # add batch dim, and vector dim

    # Broadcast tri_values: (B, 1, F, 3, 3)
    tri_values = tri_values[:, None, :, :, :]

    # Multiply and sum: (B, V, F, 3)
    weighted_sum = (weights * tri_values).sum(dim=3)

    # Final sum over F: (B, V, 3)
    total = weighted_sum.sum(dim=2)

    # Normalize per vertex
    weight_sum = W.sum(dim=(1, 2), keepdim=True).clamp(min=eps)  # (V, 1, 1)
    result = total / weight_sum.transpose(0, 1)  # (B, V, 3)

    return result

if __name__ == '__main__':
    # original mesh
    pass
#     ict_mesh = trimesh.load('/source/sihun/NFS/tmp/eccv_ict_recon.obj')
    
#     # decimated mesh
#     ict_mesh_decimate = trimesh.load('/source/sihun/NFS/tmp/eccv_ict_recon-decimate_tri.obj')
    
#     # mesh subdivision
#     ict_mesh_subdivide = trimesh.remesh.subdivide(ict_mesh.vertices, ict_mesh.faces, return_index=True)
    