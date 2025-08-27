"""
    For experiments only!!!
    very dirty code...
"""
import os
import pickle
from glob import glob

import numpy as np
import igl

import scipy
from scipy.ndimage import gaussian_filter
from scipy.sparse import  diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

import trimesh

import torch
import torch.nn as nn
import torch.optim
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter_add
from collections import defaultdict


import sys
from pathlib import Path
abs_path = str(Path(__file__).parents[1].absolute())
if not abs_path in sys.path:
    sys.path+=[abs_path]
    # sys.path+=[f'{abs_path}/third_party/siren']
    sys.path+=[f'{abs_path}/third_party/Pointnet_Pointnet2_pytorch/models']
    sys.path+=[f'{abs_path}/third_party/diffusion-net/src']
else:
    # sys.path+=[f'{abs_path}/third_party/siren']
    sys.path+=[f'{abs_path}/third_party/Pointnet_Pointnet2_pytorch/models']
    sys.path+=[f'{abs_path}/third_party/diffusion-net/src']
    
import diffusion_net
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer, STN3d, STNkd
from pointnet_part_seg import get_model, get_loss
from utils.remesh_utils import ICT_face_model

class PCA_holder():
    def __init__(self, npz_file="pca_model.npz"):
        data = np.load(npz_file)
        #proj_coords = np.load("proj_coords.npy")
        
        self.mean_ = data['mean_']
        self.components_ = data['components_']
        self.explained_variance_ = data['explained_variance_']
        self.n_components_ = data['components_'].shape[0]
        #recon = proj_coords @ components + mean


    def sample_from_pca(self, scale=1.0):
        z = np.random.randn(self.n_components_) * np.sqrt(self.explained_variance_) * scale
        z = z @ self.components_ + self.mean_
        return z.reshape(-1,3)
    
def get_colors(vertices):
    min_coord,max_coord = np.min(vertices,axis=0,keepdims=True),np.max(vertices,axis=0,keepdims=True)
    cmap = (vertices-min_coord)/(max_coord-min_coord)
    return cmap
    
def quaternion_to_rotation_matrix(q):
    """
    Quaternion (w, x, y, z) → Rotation matrix
    Args:
        q: (B, 4) or (4,) torch tensor, where q = [w, x, y, z]

    Returns:
        R: (B, 3, 3) or (3, 3) rotation matrix
    """
    if q.ndim == 1:
        q = q.unsqueeze(0)  # (1, 4)

    q = q / q.norm(dim=1, keepdim=True)  # normalize
    w, x, y, z = q.unbind(dim=1)

    B = q.shape[0]

    R = torch.empty((B, 3, 3), dtype=q.dtype, device=q.device)

    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)

    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - x * w)

    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    if R.shape[0] == 1:
        return R[0]  # remove batch dim if input was (4,)
    return R
    
def get_vertex_to_face_map(V, F):
    """
    For each vertex, returns a list of face indices it belongs to.

    Args:
        V (np.ndarray): Vertices array of shape (n, 3)
        F (np.ndarray): Faces array of shape (m, 3)

    Returns:
        vertex_face_map (list of lists): vertex_face_map[i] contains the list of face indices that vertex i belongs to
    """
    num_vertices = V.shape[0]
    vertex_face_map = [[] for _ in range(num_vertices)]

    for face_idx, face in enumerate(F):
        for vertex_idx in face:
            vertex_face_map[vertex_idx].append(face_idx)

    return vertex_face_map

def get_TBN_foreach_vertex(V, F):
    """
    Get TBN matrix for each vertex

    Args:
        V (np.ndarray): vertices (N, 3)
        F (np.ndarray): faces / trianlges (M, 3)

    Returns:
        np.ndarray: matrix (N, 3, 3)
        [
            [Tx, Ty, Tz],
            [Bx, By, Bz],
            [Nx, Ny, Nz],
        ]
    """
    Vn = igl.per_vertex_normals(V, F)
    
    al = igl.adjacency_list(F) # all neighbors
    nb = [nbrs[0] for nbrs in al] # select one
    
    Vt_nb = V[nb] - V
    Vt_nb = Vt_nb/(np.linalg.norm(Vt_nb, axis=1)[:,None]+1e-8)

    # project
    _Vt = np.sum(Vt_nb * Vn, axis=1, keepdims=True)
        
    Vt = Vt_nb - (_Vt * Vn)
    Vt = Vt/(np.linalg.norm(Vt, axis=1, keepdims=True)+1e-8)
    
    Vb = np.cross(Vn, Vt)
    Vb = Vb/(np.linalg.norm(Vb, axis=1, keepdims=True)+1e-8)
    
    return np.concatenate([
        Vt[:,None], 
        Vb[:,None],
        Vn[:,None], 
    ], axis=1)

def rescale(V1 ,V2):
    """rescale V1 to V2"""
    V1mean = (V1.max(0)+V1.min(0))*0.5
    V2mean = (V2.max(0)+V2.min(0))*0.5
    V = (V1-V1mean)/max(V1.max(0)-V1.min(0)) * max(V2.max(0)-V2.min(0)) + V2mean
    return V

def get_TriangleArea_foreach_vertex(V,F):
    """
    Returns (V,)
    """
    DArea = igl.doublearea(V, F) * 0.5
    print(DArea.shape)

    V2F = get_vertex_to_face_map(V,F)
    VArea = []
    for v2f in V2F:
        VArea.append(np.sum(DArea[v2f])/len(v2f))
    VArea = np.array(VArea)
    return VArea

def get_scale_foreach_vertex(V,F):
    """
    Returns (V, 3)
    """
    # Vxyz = []
    # for vf in V[F]: # F, 3, 3
    #     Vxyz.append(vf.max(0)-vf.min(0))
    # Vxyz = np.array(Vxyz)
    
    V2F = get_vertex_to_face_map(V,F)
    Vscale = []
    for v2f in V2F:
        Vscale.append(np.std(V[F[v2f]].reshape(-1,3), axis=0))
    Vscale = np.array(Vscale)
    return Vscale


def get_scale_foreach_triangle(Vsrc, Fsrc):
    """
    get scale ratio of DeformedSourceModelLocalBoudingBox / SourceModelLocalBoudingBox
    Returns (V, 3, 3)
    """
    S_matrices = np.zeros((Vsrc.shape[0], 3))

    all_nbh = igl.adjacency_list(Fsrc) # all neighbors for each vertex    
    for v_idx, nbh in enumerate(all_nbh):

        # get positions in both source and deformed meshes
        local_src = Vsrc[nbh]

        # bounding box: max - min
        bbox_src = np.max(local_src, axis=0) - np.min(local_src, axis=0)

        # prevent divide by zero
        bbox_src[bbox_src == 0] = 1e-8

        S_matrices[v_idx] = bbox_src

    return S_matrices

def smooth(vert, lap, loop=1, t=0.01):
    D_Inv = diags(1 / lap.diagonal())
    _lap = D_Inv @ lap
    
    # a = 1-t
    I_L = scipy.sparse.identity(lap.shape[0]) - _lap*t
    # I_L = (I_L*t).power(loop)
    for i in range(loop):
        I_L = I_L.T @ I_L
    
    vert = I_L @ vert
    return vert

def taubin_smooth(vert, lap, values, loop=2, m=0.01, l=0.01):
    D_Inv = diags(1 / lap.diagonal())
    _lap = D_Inv @ lap
    
    I_L = scipy.sparse.identity(lap.shape[0]) - _lap*l
    I_M = scipy.sparse.identity(lap.shape[0]) - _lap*m
    
    I_L = I_L @ I_M
    for i in range(loop-1):
        I_L = I_L @ I_L
    
    values = I_L @ values
    return values

def mesh_smooth(V, F, values, tau=0.001):

    # Mesh smoothing with libigl
    l = igl.cotmatrix(V, F)  # laplace-beltrami operator in libigl
    m = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC) # mass matrix in libigl
    s = m - tau * l
    return spsolve(s, m @ values)

def random_rotation_matrix(randgen=None):
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
    
def random_rotate_points(pts, randgen=None, return_rot=False):
    R = random_rotation_matrix(randgen) 
    if return_rot:
        return np.matmul(pts, R), R
    return np.matmul(pts, R)

def calc_norm(mesh):
    """
        mesh(trimesh.Trimesh)
    """
    cross1 = lambda x,y:np.cross(x,y)
    fv = mesh.vertices[mesh.faces]

    span = fv[ :, 1:, :] - fv[ :, :1, :]
    norm = cross1(span[:, 0, :], span[:, 1, :])
    norm = norm / (np.linalg.norm(norm, axis=-1)[ :, np.newaxis] + 1e-8)
    norm_v = trimesh.geometry.mean_vertex_normals(mesh.vertices.shape[0], mesh.faces, norm)
    return norm_v, norm

def calc_norm_trimesh(vertices, faces):
    """
        mesh(trimesh.Trimesh)
    """
    cross1 = lambda x,y:np.cross(x,y)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    fv = mesh.vertices[mesh.faces]

    span = fv[ :, 1:, :] - fv[ :, :1, :]
    norm = cross1(span[:, 0, :], span[:, 1, :])
    norm = norm / (np.linalg.norm(norm, axis=-1)[ :, np.newaxis] + 1e-8)
    norm_v = trimesh.geometry.mean_vertex_normals(mesh.vertices.shape[0], mesh.faces, norm)
    return norm_v, norm
    
def compute_triangle_normals(V, F):
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals

def compute_triangle_centers(V, F):
    return (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0


def project_point_to_triangle(p, tri, return_bary=False):
    a, b, c = tri.astype(np.float64)
    ab = b - a
    ac = c - a
    ap = p.astype(np.float64) - a

    d00 = ab @ ab
    d01 = ab @ ac
    d11 = ac @ ac
    d20 = ap @ ab
    d21 = ap @ ac

    denom = d00 * d11 - d01 * d01 + 1e-8
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1 - u)
    w = 1 - u - v

    proj = u * a + v * b + w * c
    if return_bary:
        return proj, np.array([u, v, w], dtype=np.float64)
    return proj

def compute_vertex_normals(V, F):
    V = V.astype(np.float64)
    N = V.shape[0]
    Vn = np.zeros((N, 3), dtype=np.float64)
    tri = V[F]
    n = np.cross(tri[:,1] - tri[:,0], tri[:,2] - tri[:,0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
    for i in range(3):
        np.add.at(Vn, F[:,i], n)
    Vn /= np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-8
    return Vn
    
def blend_rotation_quaternions_dq(tri_mat, weights):
    """
    Blend 3 unit quaternions Q, R_, S using barycentric weights via dual quaternion-style.

    Args:
        tri_mat: (Nx3x3) 
        weights: (3,) barycentric weights (u, v, w)

    Returns:
        (3,3) rotation matrix representing blended rotation
    """

    _R0 = R.from_matrix(tri_mat[0]).as_quat()
    _R1 = R.from_matrix(tri_mat[1]).as_quat()
    _R2 = R.from_matrix(tri_mat[2]).as_quat()
    
    # Weighted sum of quaternions
    quat_blend = weights[0] * _R0 + weights[1] * _R1 + weights[2] * _R2

    # Normalize back to unit quaternion
    quat_blend /= np.linalg.norm(quat_blend)

    # Convert to rotation matrix
    return R.from_quat(quat_blend).as_matrix()
    
def find_closest_valid_feature(V_src, F_src, V_tar, F_tar, val_tar=None, normal_weight=0.5, k=5, is_mat=False):
    """
    Return closest valid points on the target mesh for each source vertex.
    
    Args:
        V_src: (N,3) source vertices
        F_src: (N,3) source triangles
        V_tar: (M,3) target vertices
        F_tar: (T,3) target triangles
        val_tar: (M, C) target per vertex feature
        normal_weight: how much weight to give to normal similarity (0~1)
    """
    if val_tar is None:
        val_tar = V_src
        
    N_src, N_src_fn = calc_norm_trimesh(V_src, F_src)
    #print()
    N = V_src.shape[0]
    tri_centers = compute_triangle_centers(V_tar, F_tar)    # (T,3)
    tri_normals = compute_triangle_normals(V_tar, F_tar)    # (T,3)

    tree = cKDTree(tri_centers)  # Efficient nearest triangle lookup

    if is_mat:
        closest_points = np.zeros((V_src.shape[0], 3, 3))
    else:
        closest_points = np.zeros_like(V_src)

    for i in range(N):
        v_src = V_src[i]
        n_src = N_src[i]

        # 1. Query k nearest triangle centers
        dists, idxs = tree.query(v_src, k=k)  # k-nearest triangle centers
        best_score = float('inf')
        # best_point = None
        best_tri = None

        for j, dist in zip(idxs, dists):
            # tri = V_tar[F_tar[j]]
            tri = F_tar[j]
            tri_normal = tri_normals[j]

            # Distance term
            #dist = np.linalg.norm(tri_centers[j] - v_src)
            # dist = dists[j]

            # Normal similarity (1 - cosine)
            normal_sim = 1 - np.dot(n_src, tri_normal)

            # Combined score (smaller the better)
            score = (1 - normal_weight) * dist + normal_weight * normal_sim

            if score < best_score:
                best_score = score
                # best_point = proj
                best_tri = tri

        # closest_points[i] = best_point
        u,v,w = get_triangle_barycentric(v_src, V_tar[best_tri])
        val_tri = val_tar[best_tri]
        # print(val_tri.shape)
        if is_mat: # val_tar : Mx3x3
            closest_value = blend_rotation_quaternions_dq(val_tri, (u,v,w))
        else: # val_tar Mx3
            closest_value = u*val_tri[0] + v*val_tri[1] + w*val_tri[2]
        closest_points[i] = closest_value

    return closest_points

    
def find_closest_valid_feature_normal_first(V_src, F_src, V_tar, F_tar,
                                            val_tar=None, normal_weight=0.5, k=5, is_mat=False):
    if val_tar is None:
        val_tar = V_src.copy().astype(np.float64)

    src_normals = compute_vertex_normals(V_src, F_src)
    # src_normals, _ = calc_norm_trimesh(V_src, F_src)
    tri_centers = compute_triangle_centers(V_tar, F_tar)
    tri_normals = compute_triangle_normals(V_tar, F_tar)

    nearest_idxs = []
    bary_weights = []

    # No need for KD-tree on centers since we filter by normals first
    for i, v in enumerate(V_src.astype(np.float64)):
        n_src = src_normals[i]
        # dot with all target normals
        dots = tri_normals @ n_src
        valid = np.where(dots > 0)[0]
        
        if valid.size == 0:
            nearest_idxs.append(-1)
            bary_weights.append(np.zeros(3, dtype=np.float64))
            continue
            
        # find closest among valid centers
        centers_valid = tri_centers[valid]
        dists = np.linalg.norm(centers_valid - v, axis=1)
        best = valid[np.argmin(dists)]
        _, bc = project_point_to_triangle(v, V_tar[F_tar[best]], return_bary=True)
        nearest_idxs.append(best)
        bary_weights.append(bc)

    return np.array(nearest_idxs, dtype=int), np.vstack(bary_weights)

def build_vertex_adjacency(F):
    adj = defaultdict(set)
    for tri in F:
        for i in range(3):
            for j in range(i+1,3):
                u, v = tri[i], tri[j]
                adj[u].add(v); adj[v].add(u)
    return {i: list(neigh) for i, neigh in adj.items()}

def build_triangle_adjacency(F):
    T = len(F)
    edge_dict = defaultdict(list)
    for t, tri in enumerate(F):
        for i in range(3):
            a, b = sorted((tri[i], tri[(i+1)%3]))
            edge_dict[(a,b)].append(t)
    adj = [[] for _ in range(T)]
    for tris in edge_dict.values():
        if len(tris) > 1:
            for i in tris:
                for j in tris:
                    if i != j and j not in adj[i]:
                        adj[i].append(j)
    return adj

def find_barycentric_with_strict_adjacency(V_src, F_src, V_tar, F_tar, normal_weight=0.5, k=5):
    N = V_src.shape[0]
    src_normals = compute_vertex_normals(V_src, F_src)
    tri_centers = compute_triangle_centers(V_tar, F_tar)
    tri_normals = compute_triangle_normals(V_tar, F_tar)

    k_eff = min(k, len(F_tar))
    tree = cKDTree(tri_centers)
    dists, idxs = tree.query(V_src, k=k_eff)
    if k_eff == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    # Candidate triangles for each vertex based on distance and normal similarity
    candidates = []
    for i in range(N):
        v, n = V_src[i], src_normals[i]
        cand = []
        for dist, j in zip(dists[i], idxs[i]):
            dot = tri_normals[j] @ n
            if dot <= 0:
                continue
            score = (1 - normal_weight) * dist + normal_weight * (1 - dot)
            cand.append((score, j))
        cand.sort()
        candidates.append([j for _, j in cand])

    src_adj = build_vertex_adjacency(F_src)
    tri_adj_set = [set(neighs) for neighs in build_triangle_adjacency(F_tar)]

    tri_idx = np.full((N,), -1, dtype=int)
    bary = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        neigh_i = [j for j in src_adj.get(i, []) if tri_idx[j] >= 0]
        valid = candidates[i]

        if not neigh_i:
            # No neighbors assigned yet, take top candidate
            if valid:
                tri_idx[i] = valid[0]
                _, bary[i] = project_point_to_triangle(V_src[i], V_tar[F_tar[tri_idx[i]]], return_bary=True)
        else:
            # All previously assigned triangles from neighbors
            neighbor_tris = {tri_idx[j] for j in neigh_i}
            for cand_tri in valid:
                if all(cand_tri in tri_adj_set[ntri] for ntri in neighbor_tris):
                    tri_idx[i] = cand_tri
                    _, bary[i] = project_point_to_triangle(V_src[i], V_tar[F_tar[cand_tri]], return_bary=True)
                    break
            # fallback if nothing found
            if tri_idx[i] == -1 and valid:
                tri_idx[i] = valid[0]
                _, bary[i] = project_point_to_triangle(V_src[i], V_tar[F_tar[tri_idx[i]]], return_bary=True)

    return tri_idx, bary
    
def find_barycentric_with_adjacency(V_src, F_src, V_tar, F_tar, normal_weight=0.5, k=5):
    N = V_src.shape[0]
    src_normals = compute_vertex_normals(V_src, F_src)
    tri_centers = compute_triangle_centers(V_tar, F_tar)
    tri_normals = compute_triangle_normals(V_tar, F_tar)

    k_eff = min(k, len(F_tar))
    tree = cKDTree(tri_centers)
    dists, idxs = tree.query(V_src, k=k_eff)
    # ensure idxs shape (N, k_eff)
    if k_eff == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    # build candidate list per source
    candidates = []
    for i in range(N):
        v, n = V_src[i], src_normals[i]
        cand = []
        for dist, j in zip(dists[i], idxs[i]):
            if tri_normals[j] @ n <= 0:
                continue
            score = (1-normal_weight)*dist + normal_weight*(1 - (tri_normals[j] @ n))
            cand.append((score, j))
        cand.sort(key=lambda x: x[0])
        candidates.append([j for _, j in cand])

    src_adj = build_vertex_adjacency(F_src)
    tri_adj = build_triangle_adjacency(F_tar)

    tri_idx = np.full((N,), -1, dtype=int)
    bary = np.zeros((N,3), dtype=np.float64)

    # initial best
    for i in range(N):
        if candidates[i]:
            best = candidates[i][0]
            tri_idx[i] = best
            _, bary[i] = project_point_to_triangle(V_src[i], V_tar[F_tar[best]], return_bary=True)

    # enforce adjacency
    for i in range(N):
        neigh = [j for j in src_adj.get(i, []) if j < i and tri_idx[j] >= 0]
        if not neigh:
            continue
        for jtri in candidates[i]:
            if all(jtri in tri_adj[tri_idx[j]] for j in neigh):
                tri_idx[i] = jtri
                _, bary[i] = project_point_to_triangle(V_src[i], V_tar[F_tar[jtri]], return_bary=True)
                break

    return tri_idx, bary


def average_rotation_matrix_batch(rot_mats):
    """
    Args:
        rot_mats: (B, N, 3, 3) torch.Tensor

    Returns:
        R_mean: (B, 3, 3)
    """
    assert rot_mats.ndim == 4 and rot_mats.shape[2:] == (3, 3)
    B, N = rot_mats.shape[:2]
    
    M = rot_mats.mean(dim=1)  # (B, 3, 3)
    
    U, _, Vh = torch.linalg.svd(M)  # 각: (B, 3, 3)
    
    R = U @ Vh
    
    det = torch.linalg.det(R)
    mask = det < 0
    if mask.any():
        U[mask, :, -1] *= -1
        R[mask] = U[mask] @ Vh[mask]

    return R



def triangle_to_vertex_map(F, V):
    tri_centers = compute_triangle_centers(V, F)
    nearest = []
    for center, tri in zip(tri_centers, F):
        d = np.linalg.norm(V[tri] - center, axis=1)
        nearest.append(tri[np.argmin(d)])
    return np.array(nearest)

def find_closest_geodesic_valid_feature(
    V_src, F_src, V_tar, F_tar, val_tar=None,
    normal_weight=0.5, k=5, geo_weight=1.0, local_k=5):
    
    if val_tar is None:
        val_tar = V_src.astype(np.float64)
    
    src_normals = compute_vertex_normals(V_src, F_src)
    tri_centers = compute_triangle_centers(V_tar, F_tar)
    tri_normals = compute_triangle_normals(V_tar, F_tar)

    # vertex-to-vertex geodesic distance
    tar_geodesic = compute_geodesic_distance_matrix(V_tar, F_tar)
    tri_to_vert = triangle_to_vertex_map(F_tar, V_tar)  # triangle index → vertex index
    
    candidate_idxs = []
    bary_weights = []

    for i, v in enumerate(V_src.astype(np.float64)):
        n_src = src_normals[i]
        dots = tri_normals @ n_src
        valid = np.where(dots > 0)[0]

        if valid.size == 0:
            candidate_idxs.append(-1)
            bary_weights.append(np.zeros(3))
            continue

        centers_valid = tri_centers[valid]
        dist = np.linalg.norm(centers_valid - v, axis=1)
        scores = (1 - normal_weight) * dist - normal_weight * dots[valid]
        best = valid[np.argmin(scores)]
        proj, bc = project_point_to_triangle(v, V_tar[F_tar[best]], return_bary=True)
        candidate_idxs.append(best)
        bary_weights.append(bc)

    candidate_idxs = np.array(candidate_idxs)
    bary_weights = np.vstack(bary_weights)

    # Local consistency using vertex geodesic distance
    src_nn = NearestNeighbors(n_neighbors=local_k + 1).fit(V_src)
    _, src_neighbors = src_nn.kneighbors(V_src)

    refined_idxs = candidate_idxs.copy()
    for i in range(V_src.shape[0]):
        if candidate_idxs[i] == -1:
            continue
        neighbor_tri_ids = [candidate_idxs[j] for j in src_neighbors[i][1:] if candidate_idxs[j] != -1]
        if len(neighbor_tri_ids) == 0:
            continue
        v_i = tri_to_vert[candidate_idxs[i]]
        v_neighbors = tri_to_vert[neighbor_tri_ids]
        geo_dists = tar_geodesic[v_i, v_neighbors]
        if np.max(geo_dists) > geo_weight:
            # 보정: 후보들 중심에 가까운 삼각형으로 대체
            center = np.mean(V_tar[v_neighbors], axis=0)
            dists = np.linalg.norm(tri_centers - center, axis=1)
            new_best = np.argmin(dists)
            refined_idxs[i] = new_best
            _, new_bc = project_point_to_triangle(V_src[i], V_tar[F_tar[new_best]], return_bary=True)
            bary_weights[i] = new_bc

    return refined_idxs, bary_weights

    from itertools import combinations

def compute_geodesic_distance_matrix(V, F):
    N = V.shape[0]
    # build edge graph
    edges = np.vstack([F[:, [0,1]], F[:, [1,2]], F[:, [2,0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    vi, vj = edges[:,0], edges[:,1]
    lengths = np.linalg.norm(V[vi] - V[vj], axis=1)
    W = sp.coo_matrix((lengths, (vi, vj)), shape=(N,N))
    W = W + W.T
    return csgraph.dijkstra(W, directed=False)

def build_src_adjacency(V_src, F_src):
    adj = [[] for _ in range(len(V_src))]
    for tri in F_src:
        for u,v in combinations(tri,2):
            adj[u].append(v)
            adj[v].append(u)
    # remove duplicates
    return [list(set(nei)) for nei in adj]

def find_geodesic_consistent_correspondence(
        V_src, F_src, V_tar, F_tar,
        normal_weight=0.5,
        smooth_weight=0.1,
        max_iter=5,
        k=5,
    ):
    src_normals = compute_vertex_normals(V_src, F_src)
    tar_normals = compute_vertex_normals(V_tar, F_tar)
    tar_geodesic = compute_geodesic_distance_matrix(V_tar, F_tar)
    src_adj = build_src_adjacency(V_src, F_src)

    dists = cdist(V_src, V_tar)
    dots  = src_normals @ tar_normals.T
    unary = (1 - normal_weight) * dists - normal_weight * dots
    
    candidates = np.argsort(unary, axis=1)[:, :k]
    A = candidates[:, 0].copy()

    for it in range(max_iter):
        changes = 0
        order = np.random.permutation(len(V_src))
        for i in order:
            best_cost = unary[i, A[i]] + smooth_weight * sum(tar_geodesic[A[i], A[n]] for n in src_adj[i])
            best_c   = A[i]
            
            for c in candidates[i]:
                cost = unary[i, c] + smooth_weight * sum(tar_geodesic[c, A[n]] for n in src_adj[i])
                if cost < best_cost:
                    best_cost = cost
                    best_c    = c
            if best_c != A[i]:
                A[i] = best_c
                changes += 1
        if changes == 0:
            break

    v2tris = [[] for _ in range(len(V_tar))]
    for ti, tri in enumerate(F_tar):
        for v in tri:
            v2tris[v].append(ti)

    refined_tri = np.empty(len(V_src), dtype=int)
    bary_ws    = []
    for i, vid in enumerate(A):
        tris = v2tris[vid]
        if not tris:
            refined_tri[i] = -1
            bary_ws.append(np.zeros(3))
            continue
        tri_norms = compute_triangle_normals(V_tar, F_tar)[tris]
        best_t    = tris[np.argmax(tri_norms @ src_normals[i])]
        refined_tri[i] = best_t
        _, bc = project_point_to_triangle(V_src[i], V_tar[F_tar[best_t]], return_bary=True)
        bary_ws.append(bc)

    return refined_tri, np.vstack(bary_ws)


def rodrigues_rotation_matrix(rotvec):
    """
    Args:
        rotvec (np.ndarray): (3,) rotvec = axis * angle
    Returns:
        R (np.ndarray) : (3, 3) rotation matrix
    """
    
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        return np.eye(3)

    axis = rotvec / theta
    x, y, z = axis

    K = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

    
def rodrigues_rotation_matrix_torch(rotvec):
    """
    Args:
        rotvec (torch.tensor): (3,) rotvec = axis * angle
    Returns:
        R (torch.tensor) : (3, 3) rotation matrix
    """
    
    theta = torch.linalg.norm(rotvec)
    if theta < 1e-8:
        return torch.eye(3)

    axis = rotvec / theta
    x, y, z = axis

    K = torch.tensor([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    R = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R
    
def rotation_matrix_from_vectors(v1, v2):
    """v1 → v2 (Rodrigues' formula)

    Args:
        v1, v2 (np.ndarray): (3,)

    Returns:
        R (np.ndarray): (3,3)
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    if np.isclose(dot, 1.0):
        return np.eye(3)
    if np.isclose(dot, -1.0):
        # find other axis
        ortho = np.array([1, 0, 0]) if not np.isclose(v1[0], 1.0) else np.array([0, 1, 0])
        axis = np.cross(v1, ortho)
        axis /= np.linalg.norm(axis)
        return rodrigues_rotation_matrix(axis * np.pi)

    skew = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])

    R = np.eye(3) + skew + (skew @ skew) * ((1 - dot) / (np.linalg.norm(cross) ** 2))
    return R

def rotation_matrix_from_vectors_batch(v1s, v2s):
    """
    Args:
        v1s (np.ndarray): (B, 3)
        v2s (np.ndarray): (B, 3)

    Returns:
        R (np.ndarray): (B, 3, 3)
    """
    v1s = v1s / np.linalg.norm(v1s, axis=1, keepdims=True)
    v2s = v2s / np.linalg.norm(v2s, axis=1, keepdims=True)

    B = v1s.shape[0]
    cross = np.cross(v1s, v2s)
    dot = np.einsum('bi,bi->b', v1s, v2s)

    R_all = np.zeros((B, 3, 3))

    for i in range(B):
        if np.isclose(dot[i], 1.0):
            R_all[i] = np.eye(3)
        elif np.isclose(dot[i], -1.0):
            v1 = v1s[i]
            ortho = np.array([1, 0, 0]) if not np.isclose(v1[0], 1.0) else np.array([0, 1, 0])
            axis = np.cross(v1, ortho)
            axis = axis / np.linalg.norm(axis)
            R_all[i] = rodrigues_rotation_matrix(axis * np.pi)
        else:
            c = cross[i]
            c_norm_sq = np.dot(c, c)
            skew = np.array([
                [0, -c[2], c[1]],
                [c[2], 0, -c[0]],
                [-c[1], c[0], 0]
            ])
            R = np.eye(3) + skew + skew @ skew * ((1 - dot[i]) / c_norm_sq)
            R_all[i] = R

    return R_all

def from_6D_to_rotation_matrix_torch(in_6d, eps=1e-12):
    """
    6D representation (B, 6) → rotation matrix (B, 3, 3) *following Zhou et al. (CVPR 2019)
    Args:
        in_6d (torch.Tensor): (B, 6)
    Returns:
        R (torch.Tensor): (B, 3, 3)
    """
    a1, a2 = in_6d[..., :3], in_6d[..., 3:]
    
    # b1 = a1 / (a1.norm(dim=-1, keepdim=True) + eps)
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=eps)
    
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    # b2 = b2 / (b2.norm(dim=-1, keepdim=True) + eps)
    b2 = torch.nn.functional.normalize(b2, dim=-1, eps=eps)
    
    b3 = torch.cross(b1, b2, dim=-1)
    
    return torch.stack((b1, b2, b3), dim=-2)


def rodrigues_rotation_matrix_torch(rotvec):
    """
    rotvec: (B, 3) rotation vectors
    returns: (B, 3, 3) rotation matrices
    """
    theta = torch.norm(rotvec, dim=-1, keepdim=True)  # (B, 1)
    axis = rotvec / (theta + 1e-8)                    # (B, 3)
    x, y, z = axis.unbind(-1)

    K = torch.stack([
        torch.stack([torch.zeros_like(x), -z, y], dim=-1),
        torch.stack([z, torch.zeros_like(x), -x], dim=-1),
        torch.stack([-y, x, torch.zeros_like(x)], dim=-1),
    ], dim=-2)  # (B, 3, 3)

    I = torch.eye(3, device=rotvec.device).unsqueeze(0)
    theta = theta.unsqueeze(-1)  # (B, 1, 1)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R
    
@torch.no_grad()
def rotation_matrix_from_vectors_batch_fast(v1s, v2s, eps=1e-12):
    """
    Args:
        v1s, v2s: (B, N, 3)
    Returns:
        R: (B, N, 3, 3)
    """
    if v1s.ndim == 2:
        v1s = v1s.unsqueeze(0)
        v2s = v2s.unsqueeze(0)

    B, N = v1s.shape[:2]
    device = v1s.device

    v1 = v1s / (v1s.norm(dim=-1, keepdim=True) + eps)
    v2 = v2s / (v2s.norm(dim=-1, keepdim=True) + eps)

    cross = torch.cross(v1, v2, dim=-1)              # (B, N, 3)
    dot = (v1 * v2).sum(dim=-1)                      # (B, N)
    v_norm_sq = (cross ** 2).sum(dim=-1) + eps       # (B, N)

    # skew-symmetric matrix K
    K = torch.zeros(B, N, 3, 3, device=device)
    K[:, :, 0, 1] = -cross[:, :, 2]
    K[:, :, 0, 2] =  cross[:, :, 1]
    K[:, :, 1, 0] =  cross[:, :, 2]
    K[:, :, 1, 2] = -cross[:, :, 0]
    K[:, :, 2, 0] = -cross[:, :, 1]
    K[:, :, 2, 1] =  cross[:, :, 0]

    I = torch.eye(3, device=device).view(1, 1, 3, 3)
    K2 = K @ K
    coef = ((1 - dot) / v_norm_sq).view(B, N, 1, 1)
    R = I + K + coef * K2  # (B, N, 3, 3)

    # dot ≈ 1
    mask_eq1 = (dot > 1.0 - 1e-4)
    R[mask_eq1] = torch.eye(3, device=device)

    # dot ≈ -1
    mask_eq_neg1 = (dot < -1.0 + 1e-4)
    if mask_eq_neg1.any():
        idx = mask_eq_neg1.nonzero(as_tuple=True)
        v1_neg = v1[idx]  # (M, 3)
        ortho = torch.where(
            (v1_neg[:, 0].abs() < 0.9).unsqueeze(-1),
            torch.tensor([1., 0., 0.], device=device),
            torch.tensor([0., 1., 0.], device=device)
        )
        axis = torch.cross(v1_neg, ortho)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + eps)
        R180 = rodrigues_rotation_matrix_torch(axis * torch.pi)  # (M, 3, 3)
        R[idx[0], idx[1]] = R180

    return R

def get_triangle_normal(VF, eps=1e-12):
    E1 = VF[..., 1] - VF[..., 0]
    E2 = VF[..., 2] - VF[..., 0]

    Vn = torch.cross(E1, E2, dim=-1)
    Vn = Vn / (torch.linalg.norm(Vn, dim=-1, keepdim=True) + eps)
    return Vn

def get_triangle_basis(V, F):
    VF = V[F]
    E1 = VF[:,1] - VF[:,0]
    E2 = VF[:,2] - VF[:,0]

    # Vn = np.cross(E1, E2)
    # Vn = Vn/np.linalg.norm(Vn, axis=-1)[:,None]
    Vn = get_triangle_normal(VF)
    
    # V4 = Vn + VF[:,0]
    # returns [V2-V1, V3-V1, V4-V1]
    return torch.stack([E1, E2, Vn], axis=1).permute(0,2,1)
    
def fast_get_adj_triangle_torch(F):
    """
    Args:
        F: (T, 3) LongTensor
    Returns:
        adj_list: list of lists
    """
    
    edge_dict = defaultdict(list)
    T = F.shape[0]
    for t_idx in range(T):
        tri = F[t_idx]
        edges = [(min(tri[i].item(), tri[(i+1)%3].item()), max(tri[i].item(), tri[(i+1)%3].item())) for i in range(3)]
        for edge in edges:
            edge_dict[edge].append(t_idx)
    
    adj_list = [[] for _ in range(T)]
    for t_indices in edge_dict.values():
        if len(t_indices) < 2:
            continue
        for i in t_indices:
            for j in t_indices:
                if i != j and j not in adj_list[i]:
                    adj_list[i].append(j)
    return adj_list
    
def create_A_sparse_torch(V, F, W=1.0, return_rhs=False):
    """
    Args:
        V (torch.tensor): vertices
        F (torch.tensor): faces
        INV_MAT (torch.tensor): (T, 3, 3)
    """
    T = F.shape[0]
    N = V.shape[0]
    device = V.device
    R_ = 3

    adj_list = fast_get_adj_triangle_torch(F)

    BASIS_MAT = get_triangle_basis(V, F)
    INV_MAT = torch.linalg.inv(BASIS_MAT)
    
    INV_MAT = INV_MAT.transpose(1, 2)  # (T, 3, 4)
    COEFF = torch.cat([
        -INV_MAT.sum(dim=-1, keepdim=True),
        INV_MAT
    ], dim=-1)  # (T, 3, 4)

    rows = []
    cols = []
    data = []

    for i, adj_tri in enumerate(adj_list):
        deg = 1.0 / len(adj_tri)
        coeff_i = COEFF[i]  # (3, 4)
        for r in range(R_):
            row_idx = R_ * i + r

            # self
            rows.extend([row_idx]*4)
            cols.extend([F[i][0], F[i][1], F[i][2], N + i])
            data.extend((-W * coeff_i[r]).tolist())

            # neighbors
            for j in adj_tri:
                coeff_j = COEFF[j]
                rows.extend([row_idx]*4)
                cols.extend([F[j][0], F[j][1], F[j][2], N + j])
                data.extend((W * deg * coeff_j[r]).tolist())

    rows = torch.tensor(rows, dtype=torch.long, device=device)
    cols = torch.tensor(cols, dtype=torch.long, device=device)
    data = torch.tensor(data, dtype=V.dtype, device=device)

    shape = (R_ * T, N + T)
    
    A_sparse = torch.sparse_coo_tensor(
        indices=torch.stack([rows, cols]),
        values=data,
        size=shape
    ).coalesce()

    if return_rhs:
        rhs = torch.zeros((R_ * T, 3), dtype=V.dtype, device=device)
        
        return A_sparse, rhs
    else:
        return A_sparse

def create_A_sparse_torch_batch_34(V, F, W=1.0, return_rhs=False):
    """
    Batch-aware version of create_A_sparse_torch 3x4.

    Args:
        V (torch.Tensor): (B, N, 3) vertices
        F (torch.LongTensor): (B, T, 3) faces
        W (float): weight
        return_rhs (bool): if True, also return a zero RHS of shape (B*T*3, 3)

    Returns:
        A_sparse (torch.sparse_coo_tensor): shape (B*T*3, B*(N+T))
        (optionally rhs (torch.Tensor): (B*T*3, 3))
    """
    B, N, _ = V.shape
    _, T, _ = F.shape
    device = V.device
    R_ = 3

    rows, cols, data = [], [], []

    # per‐batch offset
    N_per = N + T

    for b in range(B):
        Vb = V[b]             # (N,3)
        Fb = F[b]             # (T,3)
        # adjacency per triangle
        adj = fast_get_adj_triangle_torch(Fb)  # list of length T

        # per‐triangle basis → inverse → coeff
        BASIS = get_triangle_basis(Vb, Fb)             # (T,3,3)
        INV   = torch.linalg.inv(BASIS)                # (T,3,3)
        
        INV   = INV.transpose(1, 2)                    # (T,3,3)
        COEFF = torch.cat([
            -INV.sum(dim=-1, keepdim=True),  # (T,3,1)
             INV                             # (T,3,3)
        ], dim=-1)                            # (T,3,4)

        base_row = b * (T * R_)
        base_col = b * N_per

        for i in range(T):
            neigh = adj[i]
            deg = 1.0 / len(neigh)
            Ci = COEFF[i]  # (3,4)

            for r in range(R_):
                row = base_row + i*R_ + r

                # --- self‐triangle term ---
                # 3 vertex cols, 1 triangle‐var col
                rows.extend([row]*4)
                cols.extend([
                    base_col + Fb[i,0],
                    base_col + Fb[i,1],
                    base_col + Fb[i,2],
                    base_col + N + i
                ])
                data.extend((-W * Ci[r]).tolist())

                # --- neighbor triangles ---
                for j in neigh:
                    Cj = COEFF[j]
                    rows.extend([row]*4)
                    cols.extend([
                        base_col + Fb[j,0],
                        base_col + Fb[j,1],
                        base_col + Fb[j,2],
                        base_col + N + j
                    ])
                    data.extend((W * deg * Cj[r]).tolist())

    # build sparse indices
    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    vals = torch.tensor(data, dtype=V.dtype, device=device)
    shape = (B * T * R_, B * N_per)

    A_sparse = torch.sparse_coo_tensor(idx, vals, size=shape).coalesce()

    if return_rhs:
        rhs = torch.zeros((shape[0], 3), dtype=V.dtype, device=device)
        return A_sparse, rhs
    else:
        return A_sparse

def create_A_sparse_torch_batch_33(V, F, W=1.0, return_rhs=False):
    """
    Batch-aware version of create_A_sparse_torch 3x3 (discarded normal.

    Args:
        V (torch.Tensor): (B, N, 3) vertices
        F (torch.LongTensor): (B, T, 3) faces
        W (float): weight
        return_rhs (bool): if True, also return a zero RHS of shape (B*T*3, 3)

    Returns:
        A_sparse (torch.sparse_coo_tensor): shape (B*T*3, B*(N+T))
        (optionally rhs (torch.Tensor): (B*T*3, 3))
    """
    B, N, _ = V.shape
    _, T, _ = F.shape
    device = V.device
    R_ = 3

    rows, cols, data = [], [], []
    
    for b in range(B):
        Vb = V[b]             # (N,3)
        Fb = F[b]             # (T,3)
        # adjacency per triangle
        adj = fast_get_adj_triangle_torch(Fb)  # list of length T

        # per‐triangle basis → inverse → coeff
        BASIS = get_triangle_basis(Vb, Fb)
        INV   = torch.linalg.inv(BASIS)
        INV   = INV.transpose(1, 2) # (T,3,3)
        
        COEFF = torch.cat([
            -INV.sum(dim=-1, keepdim=True),
             INV
        ], dim=-1) # (T,3,4)

        base_row = b * (T * R_)
        base_col = b * N

        for i in range(T):
            neigh = adj[i]
            deg = 1.0 / len(neigh)
            Ci = COEFF[i]  # (3,4)

            for r in range(R_):
                row = base_row + i*R_ + r

                # --- self‐triangle term ---
                # 3 vertex cols, 1 triangle‐var col
                rows.extend([row]*3)
                cols.extend([
                    base_col + Fb[i,0],
                    base_col + Fb[i,1],
                    base_col + Fb[i,2],
                ])
                data.extend((-W * Ci[r, :3]).tolist())

                # --- neighbor triangles ---
                for j in neigh:
                    Cj = COEFF[j]
                    rows.extend([row]*3)
                    cols.extend([
                        base_col + Fb[j,0],
                        base_col + Fb[j,1],
                        base_col + Fb[j,2],
                    ])
                    data.extend((W * deg * Cj[r,:3]).tolist())

    # build sparse indices
    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    vals = torch.tensor(data, dtype=V.dtype, device=device)
    shape = (B * T * R_, B * N)

    A_sparse = torch.sparse_coo_tensor(idx, vals, size=shape).coalesce()

    if return_rhs:
        rhs = torch.zeros((shape[0], 3), dtype=V.dtype, device=device)
        return A_sparse, rhs
    else:
        return A_sparse

def compute_edges(verts_N, faces):
    """
    Computes edges in packed form from the packed version of faces and verts.
    reference: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/structures/meshes.html#Meshes.edges_packed
    Args:
        faces (torch.tensor): (N, 3)
    """
    
    v0, v1, v2 = faces.chunk(3, dim=-1)
    e01 = torch.cat([v0, v1], dim=-1)  # (sum(F_n), 2)
    e12 = torch.cat([v1, v2], dim=-1)  # (sum(F_n), 2)
    e20 = torch.cat([v2, v0], dim=-1)  # (sum(F_n), 2)

    # All edges including duplicates.
    edges = torch.cat([e12, e20, e01], dim=1)  # (sum(F_n)*3, 2)
    
    # rows in edges after sorting will be of the form (v0, v1) where v1 > v0.
    edges, _ = edges.sort(dim=-1)

    # Remove duplicate edges: convert each edge (v0, v1) into an
    # integer hash = V * v0 + v1; this is much faster than edges.unique(dim=1)
    # After finding the unique elements reconstruct the vertex indices as:
    # (v0, v1) = (hash / V, hash % V)
    V = verts_N
    edges_hash = V * edges[..., 0] + edges[..., 1]
    uqe, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
    
    uqe_V = uqe.div(V, rounding_mode='floor')
    return torch.stack([uqe_V, uqe % V], dim=1)

def adjacency_matrix(verts_N, faces):
    V = verts_N
    
    edges = compute_edges(V, faces)
    
    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32)
    A  = torch.sparse_coo_tensor(idx, ones, (V, V))
    return A
    
def laplacian_and_adjacency(verts_N: torch.Tensor, edges: torch.Tensor):
    """
    Reference: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/laplacian_matrices.html
    
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts_N: number of vertices (N) of the mesh (N, 3)
        edges:   tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
        A: Adjacency matrix (V, V)
    """
    V = verts_N

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32)
    A  = torch.sparse_coo_tensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse_coo_tensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32)
    L -= torch.sparse_coo_tensor(idx, ones, (V, V))

    return L, A


def get_output(self, out, x_in, return_inv=False, return_raw=False):
    """
    Args:
        self (class): class
        out (torch.tensor): network output
        x_in (torch.tensor): network input (to refer tensor shape)
        return_inv (bool): if True, return inverse scale matrix
        return_raw (bool): if True, return raw network output
    Returns:
        rotation matrix
    """
    device = x_in.device
    # if out.shape[-2:] == (3,3):
    #     return out
    if self.mode=='v':
        return out
        
    elif self.mode=='rot':
        if self.out_dim == 9:
            return out.reshape(out.shape[0], out.shape[1], 3, 3)
        elif self.out_dim == 6:
            # 1 in diagonal element to cover indentity matrix
            T = torch.zeros((out.shape[0], out.shape[1], 3, 3)).to(device)
            # T[...,0,0], T[..., 0,1], T[..., 0,2] =            1,  out[..., 2],  out[..., 1] 
            # T[...,1,0], T[..., 1,1], T[..., 1,2] =  out[..., 5],            1,  out[..., 0] 
            # T[...,2,0], T[..., 2,1], T[..., 2,2] =  out[..., 4],  out[..., 3],            1

            out = out**2
            T[...,0,0], T[..., 0,1], T[..., 0,2] =            1, -out[..., 2],  out[..., 1] 
            T[...,1,0], T[..., 1,1], T[..., 1,2] =  out[..., 5],            1, -out[..., 0] 
            T[...,2,0], T[..., 2,1], T[..., 2,2] = -out[..., 4],  out[..., 3],            1
            
            # T[...,0,0], T[..., 0,1], T[..., 0,2] +=            0, -out[..., 2],  out[..., 1] 
            # T[...,1,0], T[..., 1,1], T[..., 1,2] +=  out[..., 5],            0, -out[..., 0] 
            # T[...,2,0], T[..., 2,1], T[..., 2,2] += -out[..., 4],  out[..., 3],            0
            return T
        elif self.out_dim == 3:
            # 1 in diagonal element to cover indentity matrix
            T = torch.zeros((out.shape[0], out.shape[1], 3, 3)).to(device)
            # T[...,0,0], T[..., 0,1], T[..., 0,2] =            1,  out[..., 2],  out[..., 1] 
            # T[...,1,0], T[..., 1,1], T[..., 1,2] =  out[..., 2],            1,  out[..., 0] 
            # T[...,2,0], T[..., 2,1], T[..., 2,2] =  out[..., 1],  out[..., 0],            1
            
            out = out**2
            T[...,0,0], T[..., 0,1], T[..., 0,2] =            1, -out[..., 2],  out[..., 1] 
            T[...,1,0], T[..., 1,1], T[..., 1,2] =  out[..., 2],            1, -out[..., 0] 
            T[...,2,0], T[..., 2,1], T[..., 2,2] = -out[..., 1],  out[..., 0],            1
            
            # T[...,0,0], T[..., 0,1], T[..., 0,2] +=            0, -out[..., 2],  out[..., 1] 
            # T[...,1,0], T[..., 1,1], T[..., 1,2] +=  out[..., 2],            0, -out[..., 0] 
            # T[...,2,0], T[..., 2,1], T[..., 2,2] += -out[..., 1],  out[..., 0],            0
            return T
            
    elif self.mode=='q':
        # out = torch.nn.functional.normalize(out, dim=-1) ## normalize output
        
        if self.out_dim == 4:
            if return_raw:
                return quaternion_to_rotation_matrix(out.reshape(-1, 4)).reshape(out.shape[0], out.shape[1], 3, 3), out
            return quaternion_to_rotation_matrix(out.reshape(-1, 4)).reshape(out.shape[0], out.shape[1], 3, 3)
            
    elif self.mode=='6D':
        if self.out_dim == 6:
            if return_raw:
                return from_6D_to_rotation_matrix_torch(out).reshape(out.shape[0], out.shape[1], 3, 3), out
            return from_6D_to_rotation_matrix_torch(out).reshape(out.shape[0], out.shape[1], 3, 3)
                            
    elif self.mode=='scale':
        # prevent negative scale
        out = out**2
        
        if self.out_dim == 3:
            if return_inv:
                return out.unsqueeze(-2) * torch.eye(3)[None,None].to(device), (1/out.unsqueeze(-2)) * torch.eye(3)[None,None].to(device)
            return out.unsqueeze(-2) * torch.eye(3)[None,None].to(device)
            
        elif self.out_dim == 1:
            if return_inv:
                return out.unsqueeze(-2) * torch.eye(3)[None,None].to(device), (1/out.unsqueeze(-2)) * torch.eye(3)[None,None].to(device)
            return out.unsqueeze(-2) * torch.eye(3)[None,None].to(device)

    elif self.mode=='rns': # rotation and scale
        if self.out_dim == 4:
            _tmp_s = out[...,0:1]**2
            out_s = _tmp_s.unsqueeze(-2) * torch.eye(3)[None,None].to(device)
            out_s_inv = (1/_tmp_s.unsqueeze(-2)) * torch.eye(3)[None,None].to(device)
            T = torch.zeros((out.shape[0], out.shape[1], 3, 3)).to(device)
            T[...,0,0], T[..., 0,1], T[..., 0,2] =            1,   out[..., 3],  out[..., 2] 
            T[...,1,0], T[..., 1,1], T[..., 1,2] =   out[..., 3],            1,  out[..., 1] 
            T[...,2,0], T[..., 2,1], T[..., 2,2] =   out[..., 2],  out[..., 1],            1
            out_r=T
            if return_inv:
                return out_r, out_s, out_s_inv
            return out_r, out_s
            
        elif self.out_dim == 7:
            _tmp_s = out[...,0:1]**2
            out_s = _tmp_s.unsqueeze(-2) * torch.eye(3)[None,None].to(device)
            out_s_inv = (1/_tmp_s.unsqueeze(-2)) * torch.eye(3)[None,None].to(device)
            T = torch.zeros((out.shape[0], out.shape[1], 3, 3)).to(device)
            T[...,0,0], T[..., 0,1], T[..., 0,2] =            1,   out[..., 1],  out[..., 2] 
            T[...,1,0], T[..., 1,1], T[..., 1,2] =   out[..., 3],            1,  out[..., 4] 
            T[...,2,0], T[..., 2,1], T[..., 2,2] =   out[..., 5],  out[..., 6],            1
            out_r=T
            if return_inv:
                return out_r, out_s, out_s_inv
            return out_r, out_s
            
        elif self.out_dim == 10:
            _tmp_s = out[...,0:1]**2
            out_s = _tmp_s.unsqueeze(-2) * torch.eye(3)[None,None].to(device)
            out_s_inv = (1/_tmp_s.unsqueeze(-2)) * torch.eye(3)[None,None].to(device)
            out_r=out[..., 1:].reshape(out.shape[0], out.shape[1], 3, 3)
            if return_inv:
                return out_r, out_s, out_s_inv
            return out_r, out_s
    
    return out

def load_batch_dfn_ino(dfn_info_list, device):
    """
    dfn_info_list (list)
    """
    batch_mass, batch_L, batch_evals, batch_evecs, batch_grad_X, batch_grad_Y, batch_faces = [],[],[],[],[],[],[]
    for dfn_info in dfn_info_list:
        # dfn_info = pickle.load(open(dfn_info_path, 'rb')) # DiffusionNet info
        dfn_info = [_.to(device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
        
        batch_mass.append(dfn_info[0])
        batch_L.append(dfn_info[1])
        batch_evals.append(dfn_info[2])
        batch_evecs.append(dfn_info[3])
        batch_grad_X.append(dfn_info[4])
        batch_grad_Y.append(dfn_info[5])
        batch_faces.append(dfn_info[6])
        # batch_grad_X=torch.stack(batch_grad_X).to(self.device)
        # batch_grad_Y=torch.stack(batch_grad_Y).to(self.device)
    
    batch_mass=torch.stack(batch_mass).to(device)
    batch_L=torch.stack(batch_L).to(device)
    batch_evals=torch.stack(batch_evals).to(device)
    batch_evecs=torch.stack(batch_evecs).to(device)
    return batch_mass, batch_L, batch_evals, batch_evecs, batch_grad_X, batch_grad_Y, batch_faces


class Model(nn.Module):
    def __init__(self, in_dim=3, out_dim=9, num_layers=4, mode='rot', use_residual=False, use_adain=False, use_to_out=False, out_type='vertices'):
        super().__init__()
        self.mode = mode
        self.out_dim = out_dim
        
        self.use_residual = use_residual
        self.use_adain = use_adain
        self.use_to_out = use_to_out
        self.out_type = out_type
        
        self.act = nn.ReLU()
        self.layer_in = nn.Linear(in_dim, 64)
        self.layer_out = nn.Linear(64, out_dim)

        self.layers = nn.ModuleList([ 
            nn.Linear(64, 64) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(64)  for _ in range(num_layers)
        ])
        
        self.to_out = nn.ModuleList([ 
            nn.Linear(64, out_dim) for _ in range(num_layers)
        ])

        # self.adain_in = nn.Linear(in_dim, 64)
        self.adain_in = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(), nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), nn.LayerNorm(64), 
                nn.Linear(64, 64),
            )
        self.adains_m = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
        ])
        self.adains_s = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
        ])

        self.forward_func = self.forward_default
        
        if self.use_adain:
            self.forward_func = self.forward_adain

    def forward(self, x_in, return_inv=False, return_raw=False):
        out = self.forward_func(x_in)
        if self.out_type == 'global':
            out = out.mean(-2, keepdims=True)
        return get_output(self, out, x_in, return_inv, return_raw)
        
    def forward_default(self, x_in, return_inv=False):
        out = self.act(self.layer_in(x_in))

        if self.use_to_out:
            to_out = 0
            
        for n, l, t_o in zip(self.norms, self.layers, self.to_out):
            if self.use_residual:
                out = n(self.act(l(out))) + out
            else:
                out = n(self.act(l(out)))
                
            if self.use_to_out:
                to_out += t_o(out)
        
        out = self.layer_out(out)
        
        if self.use_to_out:
            out += to_out
        return out
        
    def forward_adain(self, x_in, return_inv=False):
        out = self.act(self.layer_in(x_in))
        id_in = self.act(self.adain_in(x_in)).mean(-2, keepdims=True) + out.mean(-2, keepdims=True)
        
        if self.use_to_out:
            to_out = 0
            
        for n, l, mu, sigma, t_o in zip(self.norms, self.layers, self.adains_m, self.adains_s, self.to_out):
            if self.use_residual:
                out = n(self.act(l(out))) + out
            else:
                out = n(self.act(l(out)))
            out = out * sigma(id_in) + mu(id_in)
            
            if self.use_to_out:
                to_out += t_o(out)
                
        out = self.layer_out(out)
        
        if self.use_to_out:
            out += to_out
            
        return out
        
class Model2(nn.Module):
    def __init__(self, in_dim=3):
        super().__init__()

        self.act = nn.ReLU()
        self.layer_in = nn.Linear(in_dim, 64)
        self.layer_out_r = nn.Linear(64, 9)
        self.layer_out_s = nn.Linear(64, 1)

        self.layers = nn.ModuleList([
            nn.Linear(64, 64),
            nn.Linear(64, 64), 
            nn.Linear(64, 64),
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(64),
            nn.LayerNorm(64),
            nn.LayerNorm(64),
        ])

    def forward(self, x_in):
        out = self.act(self.layer_in(x_in))

        for n, l, in zip(self.norms, self.layers):
            out = n(self.act(l(out)))
        out_S = self.layer_out_s(out)
        out_R = self.layer_out_r(out)
        
        return out_S, out_R
        

class Model_mk1(nn.Module):
    """
        simple MLP
    """
    def __init__(self, 
                 in_dim=3+512,
                 out_dim=3, 
                 num_layer=4,
                 act='relu'):
        super(Model_mk1, self).__init__()
        
        if act == 'none':
            self.act = lambda x: x
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU()
        
        self.linears = []
        dim_list = [in_dim, 192, 128, 64, out_dim]
        for i in range(len(dim_list)-1):
            MLP = nn.Sequential(
                    nn.Linear(dim_list[i], dim_list[i]//2),
                    self.act,
                    nn.LayerNorm(dim_list[i]//2),
                    nn.Linear(dim_list[i]//2, dim_list[i]//4),
                    self.act,
                    nn.LayerNorm(dim_list[i]//4),
                    nn.Linear(dim_list[i]//4, dim_list[i+1]),
                )
            self.linears.append(MLP)
        self.linears = nn.ModuleList(self.linears)

    def forward(self, x):
        """
            x (torch.tensor) [B, N, C]: input
            out (torch.tensor)
        """
        out = x
        for i in range(len(self.linears)):
            out = self.linears[i](out)
        return out
    
class Model_mk2(nn.Module):
    def __init__(self, in_dim=3, out_dim=9, num_layers=4, mode='rot', use_residual=False, use_adain=False, use_to_out=False, out_type='local'):
        super().__init__()
        self.mode = mode
        self.out_dim = out_dim
        self.out_type = out_type
        
        self.use_residual = use_residual
        self.use_adain = use_adain
        self.use_to_out = use_to_out
        
        self.act = nn.ReLU()
        
        self.layer_in_g = nn.Linear(in_dim, 64)
        self.layer_out_g = nn.Linear(64, out_dim)

        # global prediction ##################
        self.layers_g = nn.ModuleList([ 
            nn.Linear(64, 64) for _ in range(num_layers)
        ])
        self.norms_g = nn.ModuleList([
            nn.LayerNorm(64)  for _ in range(num_layers)
        ])
        ######################################

        self.to_out_g = nn.ModuleList([ 
            nn.Linear(64, out_dim) for _ in range(num_layers)
        ])

        self.adain_in_g = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(), nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), nn.LayerNorm(64), 
                nn.Linear(64, 64),
            )
        self.adains_m_g = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
        ])
        self.adains_s_g = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
        ])

        
        self.layer_in = nn.Linear(in_dim+out_dim, 64)
        self.layer_out = nn.Linear(64, out_dim)
        # local prediction ###################
        self.layers = nn.ModuleList([ 
            nn.Linear(64, 64) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(64)  for _ in range(num_layers)
        ])
        ######################################
        
        self.to_out = nn.ModuleList([ 
            nn.Linear(64, out_dim) for _ in range(num_layers)
        ])

        self.adain_in = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(), nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), nn.LayerNorm(64), 
                nn.Linear(64, 64),
            )
        self.adains_m = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
        ])
        self.adains_s = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
        ])

        self.forward_func = self.forward_default
        
        if self.use_adain:
            self.forward_func = self.forward_adain
        
    def forward(self, x_in, return_inv=False, return_raw=False):
        out = self.forward_func(x_in)
        # out = out + 1e-12
        if self.out_type == 'global':
            out = out.mean(-2, keepdims=True)
        return get_output(self, out, x_in, return_inv, return_raw)
        
    def forward_default(self, x_in, return_inv=False):
        out = self.act(self.layer_in(x_in))

        if self.use_to_out:
            to_out = 0
            
        for n, l, t_o in zip(self.norms, self.layers, self.to_out):
            if self.use_residual:
                out = n(self.act(l(out))) + out
            else:
                out = n(self.act(l(out)))
                
            if self.use_to_out:
                to_out += t_o(out)
        
        out = self.layer_out(out)
        
        if self.use_to_out:
            out += to_out
        return out
        
    def forward_adain(self, x_in, return_inv=False):
        out = self.act(self.layer_in_g(x_in))
        id_in = self.act(self.adain_in_g(x_in)).mean(-2, keepdims=True) + out.mean(-2, keepdims=True)
        
        # global
        if self.use_to_out:
            to_out = 0
            
        for n, l, mu, sigma, t_o in zip(self.norms_g, self.layers_g, self.adains_m_g, self.adains_s_g, self.to_out_g):
            if self.use_residual:
                out = n(self.act(l(out))) + out
            else:
                out = n(self.act(l(out)))
            out = out * sigma(id_in) + mu(id_in)
            
            if self.use_to_out:
                to_out += t_o(out)
                
        out = self.layer_out_g(out)
        
        if self.use_to_out:
            out += to_out
        
        out = out.mean(-2).unsqueeze(-2).repeat(1, x_in.shape[1], 1) # B, N, out_dim

        # local
        x_in_l = torch.cat((x_in, out ),dim=-1 )
        out_l = self.act(self.layer_in(x_in_l))
        if self.use_to_out:
            to_out_l = 0
            
        for n, l, mu, sigma, t_o in zip(self.norms, self.layers, self.adains_m, self.adains_s, self.to_out):
            if self.use_residual:
                out_l = n(self.act(l(out_l))) + out_l
            else:
                out_l = n(self.act(l(out_l)))
            out_l = out_l * sigma(id_in) + mu(id_in)
            
            if self.use_to_out:
                to_out_l += t_o(out_l)
                
        out_l = self.layer_out(out_l)
        
        if self.use_to_out:
            out_l += to_out_l
        
        ## add global + local
        out = out_l + out
        return out

class Model_mk2_2(nn.Module):
    def __init__(self, in_dim=3, out_dim=9, num_layers=4, mode='rot', use_residual=False, use_adain=False, use_to_out=False, out_type='local'):
        super().__init__()
        self.mode = mode
        self.out_dim = out_dim
        self.out_type = out_type
        
        self.use_residual = use_residual
        self.use_adain = use_adain
        self.use_to_out = use_to_out
        
        self.act = nn.ReLU()
        
        self.layer_in_g = nn.Linear(in_dim, 64)
        self.layer_out_g = nn.Linear(64, out_dim)

        # global prediction ##################
        self.layers_g = nn.ModuleList([ 
            nn.Linear(64, 64) for _ in range(num_layers)
        ])
        self.norms_g = nn.ModuleList([
            nn.LayerNorm(64)  for _ in range(num_layers)
        ])
        ######################################

        self.to_out_g = nn.ModuleList([ 
            nn.Linear(64, out_dim) for _ in range(num_layers)
        ])

        self.adain_in_g = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(), nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), nn.LayerNorm(64), 
                nn.Linear(64, 64),
            )
        self.adains_m_g = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
        ])
        self.adains_s_g = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
        ])

        
        self.layer_in = nn.Linear(in_dim, 64)
        # self.layer_in = nn.Linear(in_dim+out_dim, 64)
        self.layer_out = nn.Linear(64, out_dim)
        # local prediction ###################
        self.layers = nn.ModuleList([ 
            nn.Linear(64, 64) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(64)  for _ in range(num_layers)
        ])
        ######################################
        
        self.to_out = nn.ModuleList([ 
            nn.Linear(64, out_dim) for _ in range(num_layers)
        ])

        self.adain_in = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(), nn.LayerNorm(64), 
                # nn.Linear(in_dim+out_dim, 64), nn.ReLU(), nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), nn.LayerNorm(64), 
                nn.Linear(64, 64),
            )
        self.adains_m = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
        ])
        self.adains_s = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
            nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64), nn.ReLU(), #nn.LayerNorm(64), 
                nn.Linear(64, 64),
            ),
        ])

        self.forward_func = self.forward_default
        
        if self.use_adain:
            self.forward_func = self.forward_adain

    def forward(self, x_in, return_inv=False, return_raw=False):
        out, out_l = self.forward_func(x_in)
        
        out = get_output(self, out, x_in, return_inv, return_raw)
        out_l = get_output(self, out_l, x_in, return_inv, return_raw)
        return out, out_l
        
    def forward_default(self, x_in, return_inv=False):
        out = self.act(self.layer_in_g(x_in))
        id_in = self.act(self.adain_in_g(x_in)).mean(-2, keepdims=True) + out.mean(-2, keepdims=True)
        
        # global
        if self.use_to_out:
            to_out = 0
            
        for n, l, t_o in zip(self.norms_g, self.layers_g, self.to_out_g):
            if self.use_residual:
                out = n(self.act(l(out))) + out
            else:
                out = n(self.act(l(out)))
            
            if self.use_to_out:
                to_out += t_o(out)
                
        out = self.layer_out_g(out)
        
        if self.use_to_out:
            out += to_out
        # print(out.shape)
        out = out.mean(-2).unsqueeze(-2).repeat(1, x_in.shape[1], 1) # B, N, 4

        ## apply global transform
        x_in_l = x_in
        if x_in.shape[-1] > 3:
            _mat = get_output(self, out, x_in)
            
            x_in_l[...,:3] = torch.einsum('bnck,bnk->bnc', _mat, x_in[...,:3])
            if x_in.shape[-1] > 6:
                x_in_l[...,3:6] = torch.einsum('bnck,bnk->bnc', _mat, x_in[...,3:6])
                
        # x_in_l = torch.cat((x_in, out ),dim=-1 )
        out_l = self.act(self.layer_in(x_in_l))
        if self.use_to_out:
            to_out_l = 0
            
        for n, l, t_o in zip(self.norms, self.layers, self.self.to_out):
            if self.use_residual:
                out_l = n(self.act(l(out_l))) + out_l
            else:
                out_l = n(self.act(l(out_l)))
            
            if self.use_to_out:
                to_out_l += t_o(out_l)
                
        out_l = self.layer_out(out_l)
        
        if self.use_to_out:
            out_l += to_out_l

        return out, out_l
        
    def forward_adain(self, x_in, return_inv=False):
        out = self.act(self.layer_in_g(x_in))
        id_in = self.act(self.adain_in_g(x_in)).mean(-2, keepdims=True) + out.mean(-2, keepdims=True)
        
        # global
        if self.use_to_out:
            to_out = 0
            
        for n, l, mu, sigma, t_o in zip(self.norms_g, self.layers_g, self.adains_m_g, self.adains_s_g, self.to_out_g):
            if self.use_residual:
                out = n(self.act(l(out))) + out
            else:
                out = n(self.act(l(out)))
            out = out * sigma(id_in) + mu(id_in)
            
            if self.use_to_out:
                to_out += t_o(out)
                
        out = self.layer_out_g(out)
        
        if self.use_to_out:
            out += to_out
        # print(out.shape)
        out = out.mean(-2).unsqueeze(-2).repeat(1, x_in.shape[1], 1) # (B, N, outdim)

        ## apply global transform
        x_in_l = torch.zeros_like(x_in)
        if x_in.shape[-1] > 3:
            _mat = get_output(self, out, x_in)
            
            x_in_l[...,:3] += torch.einsum('bnck,bnk->bnc', _mat, x_in[...,:3])
            if x_in.shape[-1] > 6:
                x_in_l[...,3:6] += torch.einsum('bnck,bnk->bnc', _mat, x_in[...,3:6])
            x_in_l[...,-3:] += x_in[...,-3:]
        
        # local
        # x_in_l = torch.cat((x_in, out),dim=-1 )
        out_l = self.act(self.layer_in(x_in_l))
        # x_in_l => (B, N, indim)
        # print(x_in_l.shape, out.mean(-2, keepdims=True).shape)
        id_in_l = self.act(self.adain_in(x_in_l)).mean(-2, keepdims=True) #+ out.mean(-2, keepdims=True)
        
        if self.use_to_out:
            to_out_l = 0
            
        for n, l, mu, sigma, t_o in zip(self.norms, self.layers, self.adains_m, self.adains_s, self.to_out):
            if self.use_residual:
                out_l = n(self.act(l(out_l))) + out_l
            else:
                out_l = n(self.act(l(out_l)))
            out_l = out_l * sigma(id_in_l) + mu(id_in_l)
            
            if self.use_to_out:
                to_out_l += t_o(out_l)
                
        out_l = self.layer_out(out_l)
        
        if self.use_to_out:
            out_l += to_out_l

        return out, out_l

class Model_mk3(nn.Module):
    """
    PointNet architecture
    """
    def __init__(self, #part_num=50, normal_channel=True):
        in_dim=3, out_dim=9, num_layers=4, mode='rot', use_residual=False, use_adain=False, use_to_out=False):
        super().__init__()
        
        self.mode = mode
        self.out_dim = out_dim
        
        self.use_residual = False # (not used)
        self.use_adain = False # (not used)
        self.use_to_out = False # (not used)
        
        # if normal_channel:
        #     channel = 6
        # else:
        #     channel = 3
        self.out_dim = out_dim
        self.stn = STN3d(in_dim)
        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        self.convs1 = torch.nn.Conv1d(4944-16, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, out_dim, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, x_in, return_inv=False, return_raw=False, no_rot=False):
        out, trans_feat = self.forward_func(x_in)
        # out = out + 1e-12
        # return self.get_output(out, x_in, return_inv, return_raw)
        if no_rot:
            return outs, trans_feat
        outs = get_output(self, out, x_in, return_inv=return_inv, return_raw=return_raw)
        return outs, trans_feat
        
    # def forward_func(self, point_cloud, label):
    def forward_func(self, point_cloud):
        # B, D, N = point_cloud.size()
        # trans = self.stn(point_cloud)
        # point_cloud = point_cloud.transpose(2, 1) # (B, N, D)
        
        B, N, D = point_cloud.size()
        trans = self.stn(point_cloud.transpose(2, 1))
        
        if D > 3:
            # point_cloud, feature = point_cloud.split(3, dim=2)
            if D > 6:
                point_cloud, point_normal, feature = point_cloud[...,:3], point_cloud[...,3:6], point_cloud[...,6:]
            else:
                point_cloud, feature = point_cloud[...,:3], point_cloud[...,3:]
                
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 6:
            point_normal = torch.bmm(point_normal, trans)
            
        if D > 3:
            # point_cloud = torch.cat([point_cloud, feature], dim=2)
            if D > 6:
                point_cloud = torch.cat([point_cloud, point_normal, feature], dim=-1)
            else:
                point_cloud = torch.cat([point_cloud, feature], dim=-1)

        point_cloud = point_cloud.transpose(2, 1) # (B, D, N)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        # out_max = torch.cat([out_max,label.squeeze(1)],1)
        # expand = out_max.view(-1, 2048+16, 1).repeat(1, 1, N)
        expand = out_max.view(-1, 2048, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        net = net.transpose(2, 1).contiguous()
        # net = F.log_softmax(net.view(-1, self.out_dim), dim=-1)
        net = net.view(B, N, self.out_dim) # [B, N, out_dim]

        return net, trans_feat

    
class Model_mk3_1(nn.Module):
    """
    PointNet architecture
    """
    def __init__(self, #part_num=50, normal_channel=True):
        in_dim=3, out_dim=512, num_layers=4, mode='rot'):
        super().__init__()
        
        self.mode = mode
        self.out_dim = out_dim
                
        # if normal_channel:
        #     channel = 6
        # else:
        #     channel = 3
        self.out_dim = out_dim
        self.stn = STN3d(in_dim)
        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.fstn = STNkd(k=128)

    def forward(self, x_in, return_inv=False, return_raw=False, no_rot=False):
        out, trans_feat = self.forward_func(x_in)
        
        # outs = get_output(self, out, x_in, return_inv=return_inv, return_raw=return_raw)
        return out, trans_feat
                
    def forward_func(self, point_cloud):                
        B, N, D = point_cloud.size()
        trans = self.stn(point_cloud.transpose(2, 1))
        
        if D > 3:
            # point_cloud, feature = point_cloud.split(3, dim=2)
            if D > 6:
                point_cloud, point_normal, feature = point_cloud[...,:3], point_cloud[...,3:6], point_cloud[...,6:]
            else:
                point_cloud, feature = point_cloud[...,:3], point_cloud[...,3:]
                
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 6:
            point_normal = torch.bmm(point_normal, trans)
            
        if D > 3:
            # point_cloud = torch.cat([point_cloud, feature], dim=2)
            if D > 6:
                point_cloud = torch.cat([point_cloud, point_normal, feature], dim=-1)
            else:
                point_cloud = torch.cat([point_cloud, feature], dim=-1)

        point_cloud = point_cloud.transpose(2, 1) # (B, D, N)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = self.bn4(self.conv4(net_transformed))
        out_max = torch.max(out4, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 512)
        
        return out_max, trans_feat

class Model_mk3_2(nn.Module):
    def __init__(self, #part_num=50, normal_channel=True):
        in_dim=3, out_dim=9, num_layers=4, mode='rot', use_residual=False, use_adain=False, use_to_out=False):
        super().__init__()
        
        self.mode = mode
        self.out_dim = out_dim
        
        self.use_residual = False # (not used)
        self.use_adain = False # (not used)
        self.use_to_out = False # (not used)
        
        # if normal_channel:
        #     channel = 6
        # else:
        #     channel = 3
        self.out_dim = out_dim
            
        self.stn = STN3d(in_dim)
        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64,   128, 1)
        self.conv3 = torch.nn.Conv1d(128,  128, 1)
        self.conv4 = torch.nn.Conv1d(128,  512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        
        self.convs1 = torch.nn.Conv1d(2048, 256, 1)
        self.convs2 = torch.nn.Conv1d(256,  256, 1)
        self.convs3 = torch.nn.Conv1d(256,  128, 1)
        self.convs4 = torch.nn.Conv1d(128, out_dim, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)


    def forward(self, x_in, return_inv=False, return_raw=False):
        out, trans_feat = self.forward_func(x_in)
        # out = out + 1e-12
        
        outs = get_output(self, out, x_in, return_inv=return_inv, return_raw=return_raw)
        return outs, trans_feat
        
    # def forward_func(self, point_cloud, label):
    def forward_func(self, point_cloud):
        # B, D, N = point_cloud.size()
        # trans = self.stn(point_cloud)
        # point_cloud = point_cloud.transpose(2, 1) # (B, N, D)
        
        B, N, D = point_cloud.size()
        trans = self.stn(point_cloud.transpose(2, 1))
        
        if D > 3:
            # point_cloud, feature = point_cloud.split(3, dim=2)
            if D > 6:
                point_cloud, point_normal, feature = point_cloud[...,:3], point_cloud[...,3:6], point_cloud[...,6:]
            else:
                point_cloud, feature = point_cloud[...,:3], point_cloud[...,3:]
                
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 6:
            point_normal = torch.bmm(point_normal, trans)
            
        if D > 3:
            # point_cloud = torch.cat([point_cloud, feature], dim=2)
            if D > 6:
                point_cloud = torch.cat([point_cloud, point_normal, feature], dim=-1)
            else:
                point_cloud = torch.cat([point_cloud, feature], dim=-1)

        point_cloud = point_cloud.transpose(2, 1) # (B, D, N)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        
        out_max = out_max.view(-1, 2048, 1).repeat(1, 1, N)
        
        net = F.relu(self.bns1(self.convs1(out_max)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        net = net.transpose(2, 1).contiguous()
        # net = F.log_softmax(net.view(-1, self.out_dim), dim=-1)
        net = net.view(B, N, self.out_dim) # [B, N, out_dim]

        return net, trans_feat

## diffusionNet (small)
class Model_mk4(nn.Module):
    def __init__(self, #part_num=50, normal_channel=True):
        in_dim=3, out_dim=9, num_layers=4, hid_shape=64, mode='rot', use_residual=False, use_adain=False, use_to_out=False, outputs_at='vertices'):
        super().__init__()
        
        self.mode = mode
        self.out_dim = out_dim
        
        self.use_residual = False # (not used)
        self.use_adain = False # (not used)
        self.use_to_out = False # (not used)
        self.outputs_at = outputs_at

        """
        Construct a DiffusionNet.

        Parameters:
            C_in (int):                     input dimension 
            C_out (int):                    output dimension 
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient. Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """
        self.dfn = diffusion_net.DiffusionNet(
            C_in=in_dim, 
            C_out=out_dim, 
            C_width=hid_shape, 
            N_block=num_layers, 
            outputs_at=outputs_at, 
            with_gradient_features=True,
            last_activation=None,
        )

    def forward(self, x_in, 
                batch_mass=None,
                batch_L_val=None,
                batch_evals=None,
                batch_evecs=None,
                batch_gradX=None,
                batch_gradY=None,
                batch_faces=None,
                return_inv=False,
                return_raw=False,
               ):
        out = self.dfn(
            x_in, 
            batch_mass, 
            L=batch_L_val,
            evals=batch_evals,
            evecs=batch_evecs,
            gradX=batch_gradX,
            gradY=batch_gradY,
            faces=batch_faces
        )
        if self.outputs_at=='global_mean':
            out = out.unsqueeze(-2)
        
        out = get_output(self, out, x_in, return_inv=return_inv, return_raw=return_raw)
        return out

    
def normalize_homogeneous(V):
    return np.concatenate([V, np.ones((V.shape[0], 1))], axis=1)

def save_pca_to_npz(file_name, pca):
    
    np.savez(f"{file_name}.npz",
             mean_=pca.mean_,
             components_=pca.components_,
             explained_variance_=pca.explained_variance_)

class PCA_holder():
    def __init__(self, npz_file="pca_model.npz"):
        data = np.load(npz_file)
        #proj_coords = np.load("proj_coords.npy")
        
        self.mean_ = data['mean_']
        self.components_ = data['components_']
        self.explained_variance_ = data['explained_variance_']
        self.n_components_ = data['components_'].shape[0]
        #recon = proj_coords @ components + mean

    def sample_from_pca(self, scale=1.0):
        z = np.random.randn(self.n_components_) * np.sqrt(self.explained_variance_) * scale
        z = z @ self.components_ + self.mean_
        return z.reshape(-1,3)

    def sample_from_pca_one_axis(self, scale=1.0, select=-1, verbose=False):
        """
        sample from pca, but within 1/4 explained_variance
        """
        z = np.random.randn(self.n_components_) * np.sqrt(self.explained_variance_) * scale
        
        # masking
        tmp = np.zeros_like(z)
        if select > -1 and select < self.n_components_:
            select = select
        else:
            select = np.random.randint(0, self.n_components_//4)
        
        if verbose:
            print(select)
        tmp[select] = z[select]
        z=tmp
        
        z = z @ self.components_ + self.mean_
        return z.reshape(-1,3)