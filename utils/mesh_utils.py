import os
import sys
from pathlib import Path
abs_path = str(Path(__file__).parents[1].absolute())
sys.path+=[abs_path, f'{abs_path}/third_party/diffusion-net/src']
import diffusion_net

import pickle
import numpy as np
import torch
from torch.utils.dlpack import from_dlpack
import trimesh

from copy import deepcopy
from torch_scatter import scatter_add
from torch_sparse import coalesce, transpose
try:
    from cupyx.scipy.sparse.linalg import SuperLU
except ImportError:
    SuperLU = None

import pytorch3d

from utils.deformation_transfer import Transfer




def sample_point_in_triangle(batch_v, faces, lamdas=None, normalize=False):
    """
    Args:
        batch_v (torch.tensor): [B, V, 3] batched vertices
        faces (torch.tensor): [F, 3]
        lamdas (torch.tensor): [2,] for interpolation v0-v1 -> tmp, tmp-v2 -> new vertices (optional)
        normalize (bool): if True, normalize (for normal vectors)
    Return:
        new_v_T (torch.tensor): [B, V, 3]
    """
    vf_T = batch_v[:, faces]
    
    if lamdas is None:
        lamdas = torch.rand(2).to(vf_T.device)

    new_v_T = vf_T[:,:,0] * lamdas[0] + vf_T[:,:,1] * (1-lamdas[0])
    new_v_T = new_v_T * lamdas[1] + vf_T[:,:,2] * (1-lamdas[1])
    if normalize:
        new_v_T = torch.nn.functional.normalize(new_v_T, p=2, dim=-1)  # [N, 3]
    return new_v_T
    
def barycentric_coordinate(T, P):
    """
    Args:
        T (torch.tensor): triangle vertices [3, 3]
        P (torch.tensor): point on the triangle [3, ]
    Return:
        lambdas (torch.tensor): [3, ]
    """
    A, B, C = T[0], T[1], T[2]
    mat = torch.tensor([
            [A[0], A[1], A[2], 1],
            [B[0], B[1], B[2], 1],
            [C[0], C[1], C[2], 1],
        ]).T  # T
    P_ext = torch.tensor([P[0], P[1], P[2], 1])
    lambdas = torch.linalg.pinv(mat.float()) @ P_ext.float()
    lambdas = lambdas[:3]
    lambdas[2] = 1- lambdas[0] -lambdas[1]
    return lambdas

def calc_cent(vertices, faces, mode='np'):
    """
    Args:
        vertices (torch.tensor): [B, V, 3] vertices 
        faces (torch.tensor): [F, 3] vertex index for each face
    """
    if len(vertices.size()) < 3:
        vertices = vertices.unsqueeze(0)
    fv = vertices[:, faces]
    
    if mode=='np':
        return np.mean(fv, axis=-2)
    else:
        return torch.mean(fv, dim=-2)

def calc_norm_torch(batch_v, face, at='face'):
    """
    Args:
        batch_v (torch.tensor): [B*T, V, 3] vertices for current batch
        face (torch.tensor): [F, 3] vertex index for each face
        at (str): mode 'face', 'vertex' (default: 'face')
        
    Returns:
        norm | face_norm (torch.tensor): corresponding normal vector
    """
    B_S = batch_v.shape[0]
    N_V = batch_v.shape[1]

    batch_vf = batch_v[:, face] # --> [B, F, 3, 3]
    span = batch_vf[..., 1:, :] - batch_vf[..., :1, :] # --> [B, V, 2, 3]
    cross = torch.linalg.cross(span[..., 0, :], span[..., 1, :], dim=-1) # --> [B, F, 3]
    face_norm = torch.nn.functional.normalize(cross, p=2, dim=-1)  # --> [B, F, 3]
    
    if at == 'face':
        return face_norm
    else: # at == 'vertex'
        idx = torch.cat([face[:, 0], face[:, 1], face[:, 2]], dim=0)
        face_norm = face_norm.repeat(1, 3, 1)

        norm = scatter_add(face_norm, idx, dim=1, dim_size=N_V)
        norm = torch.nn.functional.normalize(norm, p=2, dim=-1)  # [N, 3]
        return norm

###### Reference for codes below: https://github.com/dafei-qin/NFR_pytorch/blob/master/myutils.py
def get_dfn_info(mesh, cache_dir=None, device='cuda'):
    """
    Args:
        mesh (trimesh.Trimesh)
    """
    verts_list = torch.from_numpy(mesh.vertices).unsqueeze(0).float()
    face_list = torch.from_numpy(mesh.faces).unsqueeze(0).long()
    frames_list, mass_list, L_list, evals_list, evecs_list, gradX_list, gradY_list = \
        diffusion_net.geometry.get_all_operators(verts_list, face_list, k_eig=128, op_cache_dir=cache_dir)

    dfn_info = [mass_list[0], L_list[0], evals_list[0], evecs_list[0], gradX_list[0], gradY_list[0], torch.from_numpy(mesh.faces)]
    dfn_info = [_.to(device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
    return dfn_info

def get_dfn_info2(vertices, faces, cache_dir=None, device='cuda'):
    """
    Args:
        vertices (np.ndarray)
        faces (np.ndarray)
    """
    if type(vertices)==np.ndarray:
        vertices=torch.from_numpy(vertices)
    if type(faces)==np.ndarray:
        faces=torch.from_numpy(faces)
        
    verts_list = vertices.unsqueeze(0).float()
    face_list  = faces.unsqueeze(0).long()
    frames_list, mass_list, L_list, evals_list, evecs_list, gradX_list, gradY_list = \
        diffusion_net.geometry.get_all_operators(verts_list, face_list, k_eig=128, op_cache_dir=cache_dir)

    dfn_info = [mass_list[0], L_list[0], evals_list[0], evecs_list[0], gradX_list[0], gradY_list[0], faces]
    dfn_info = [_.to(device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
    return dfn_info

def get_span_matrix(batch_vertices, faces):
    """
    Args
        batch_vertices (torch.tensor): [B, V, 3]
        faces (torch.tensor): [F, 3]
    Return
        span (torch.tensor): [B, F, 3, 3] (v2-v1, v3-v1, v4-v1)
    """
    B_v = batch_vertices

    faces  = faces[None].repeat(B_v.shape[0], 1 ,1)
    B, num_faces  = faces.shape[:2]
    batch_indices = torch.arange(B)[:, None, None]
    batch_indices = torch.tile(batch_indices, (1, num_faces, 1))

    B_vf = B_v[batch_indices, faces].permute(0, 1, 3, 2)

    v1, v2, v3 = B_vf[..., 0], B_vf[..., 1], B_vf[..., 2]

    cross = torch.linalg.cross(v2 - v1, v3 - v1, dim=-1)

    vn = torch.nn.functional.normalize(cross, p=2, dim=-1)  # [F, 3]

    v4 = v1 + vn

    span = torch.stack((v2 - v1, v3 - v1, v4 - v1), dim=-1)
    return span

def precompute_neutral_span_inv(template_verts, faces):
    """Precompute (neutral_span^T)^{-1} per face for fast per-step Jacobian computation.

    During training, replaces linalg.solve (template-dependent) with a precomputed
    matmul: Q = neutral_span_inv[f] @ span_deformed^T[b,f].

    Args:
        template_verts: [V, 3] numpy array or tensor (single identity template)
        faces: [F, 3] numpy array or tensor
    Returns:
        neutral_span_inv: [F, 3, 3] CPU tensor — (neutral_span^T)^{-1} per face
    """
    if isinstance(template_verts, np.ndarray):
        template_verts = torch.tensor(template_verts, dtype=torch.float32)
    if isinstance(faces, np.ndarray):
        faces = torch.tensor(faces, dtype=torch.long)
    with torch.no_grad():
        neutral_span = get_span_matrix(template_verts.unsqueeze(0), faces)   # [1, F, 3, 3]
        neutral_span_perm = neutral_span.permute(0, 1, 3, 2)                 # [1, F, 3, 3] (transposed per face)
        # pinv handles near-degenerate (zero-area) faces gracefully
        neutral_span_inv = torch.linalg.pinv(neutral_span_perm).squeeze(0)   # [F, 3, 3]
    return neutral_span_inv.cpu()


def get_jacobian_matrix(verts, faces, template=None, neutral_span_inv=None, return_torch=False):
    """Reference from Deformation Transfer for Triangle Meshes [Sumner and Popovic, 2004]
    Args
        verts (torch.tensor): [B, V, 3] target vertices (deformed)
        faces (torch.tensor): [F, 3]
        template (torch.tensor): [B, V, 3] source vertices — used when neutral_span_inv is None
        neutral_span_inv (torch.tensor): [F, 3, 3] precomputed (neutral_span^T)^{-1};
            if provided, skips template span computation and uses faster matmul.
    Return
        Q (torch.tensor): [B, F, 3, 3] per-face deformation Jacobian
    """
    B = verts.shape[0]
    num_faces = faces.shape[0]

    span_matrix = get_span_matrix(verts, faces)           # [B, F, 3, 3]
    span_perm   = span_matrix.permute(0, 1, 3, 2)         # [B, F, 3, 3]

    if neutral_span_inv is not None:
        # Fast path: precomputed (neutral^T)^{-1} — matmul instead of solve
        nsi = neutral_span_inv.to(verts.device)
        if nsi.dim() == 3:
            nsi = nsi.unsqueeze(0).expand(B, -1, -1, -1)  # [B, F, 3, 3]
        Q = torch.bmm(
            nsi.reshape(-1, 3, 3),
            span_perm.reshape(-1, 3, 3)
        ).reshape(B, num_faces, 3, 3)
    else:
        # Original path: compute template span and solve
        neutral_span_matrix = get_span_matrix(template, faces)
        Q = torch.linalg.solve(neutral_span_matrix.permute(0, 1, 3, 2), span_perm)
    return Q


def compute_vertex_strain(deformed_verts, template_verts, faces, return_trace=False,
                          neutral_span_inv=None):
    """
    Per-vertex Green-Lagrange strain.

    Args:
        deformed_verts: [B, V, 3]
        template_verts: [B, V, 3]
        faces: [F, 3]
        return_trace: if True, also return trace(E) per vertex (signed)

    Returns:
        strain_norm: [B, V, 1] — ||E||_F per vertex (unsigned magnitude)
        strain_trace: [B, V, 1] — trace(E) per vertex (signed, only if return_trace)
    """
    B, V, _ = deformed_verts.shape
    num_faces = faces.shape[0]
    device = deformed_verts.device

    # Per-face deformation gradient: [B, F, 3, 3]
    F_grad = get_jacobian_matrix(deformed_verts, faces, template_verts,
                                  neutral_span_inv=neutral_span_inv)

    # Green-Lagrange: E = 0.5 * (F^T F - I)
    Ft = F_grad.permute(0, 1, 3, 2)  # [B, F, 3, 3]
    FtF = torch.bmm(
        Ft.reshape(-1, 3, 3),
        F_grad.reshape(-1, 3, 3)
    ).reshape(B, num_faces, 3, 3)
    I = torch.eye(3, device=device).reshape(1, 1, 3, 3)
    E = 0.5 * (FtF - I)

    # Per-face Frobenius norm: [B, F]
    strain_norm_face = torch.sqrt((E ** 2).sum(dim=(-2, -1)) + 1e-12)

    # Scatter face values to vertices (average adjacent faces)
    idx = faces.reshape(-1).unsqueeze(0).expand(B, -1)  # [B, 3F]
    ones = torch.ones(B, 3 * num_faces, device=device)

    def _expand3(x):
        return x.unsqueeze(-1).expand(-1, -1, 3).reshape(B, -1)

    norm_vert = scatter_add(_expand3(strain_norm_face), idx, dim=1, dim_size=V)  # [B, V]
    count = scatter_add(ones, idx, dim=1, dim_size=V)  # [B, V]
    strain_norm = (norm_vert / (count + 1e-12)).unsqueeze(-1)  # [B, V, 1]

    if return_trace:
        trace_face = E.diagonal(dim1=-2, dim2=-1).sum(-1)  # [B, F]
        trace_vert = scatter_add(_expand3(trace_face), idx, dim=1, dim_size=V)
        strain_trace = (trace_vert / (count + 1e-12)).unsqueeze(-1)  # [B, V, 1]
        return strain_norm, strain_trace

    return strain_norm


def compute_strain_signal(deformed_verts, template_verts, faces, mode='norm',
                          neutral_span_inv=None):
    """
    Unified strain computation with multiple modes.

    Args:
        deformed_verts: [B, V, 3]
        template_verts: [B, V, 3]
        faces: [F, 3]
        mode: strain mode string
            'norm'       — ||E||_F per vertex (dim=1, unsigned magnitude)
            'norm_trace' — [||E||_F, trace(E)] per vertex (dim=2)
            'full'       — upper-triangle of symmetric E (dim=6: E_xx,E_yy,E_zz,E_xy,E_xz,E_yz)
            'principal'  — eigenvalues of E sorted descending (dim=3)
            'local'      — 1-ring neighbor aggregated: [self_norm, neighbor_mean, neighbor_std] (dim=3)
        neutral_span_inv: [B, F, 3, 3] or None — precomputed inverse span matrices

    Returns:
        strain: [B, V, strain_dim]  where strain_dim depends on mode
    """
    B, V, _ = deformed_verts.shape
    num_faces = faces.shape[0]
    device = deformed_verts.device

    # Common: per-face Green-Lagrange strain tensor E
    F_grad = get_jacobian_matrix(deformed_verts, faces, template_verts,
                                  neutral_span_inv=neutral_span_inv)
    Ft = F_grad.permute(0, 1, 3, 2)
    FtF = torch.bmm(
        Ft.reshape(-1, 3, 3),
        F_grad.reshape(-1, 3, 3)
    ).reshape(B, num_faces, 3, 3)
    I_mat = torch.eye(3, device=device).reshape(1, 1, 3, 3)
    E = 0.5 * (FtF - I_mat)  # [B, F, 3, 3]

    # Common: face→vertex scatter setup
    idx = faces.reshape(-1).unsqueeze(0).expand(B, -1)  # [B, 3F]
    ones = torch.ones(B, 3 * num_faces, device=device)
    count = scatter_add(ones, idx, dim=1, dim_size=V)  # [B, V]
    count_safe = count + 1e-12

    def _scatter_face_to_vert(face_vals):
        """face_vals: [B, F] or [B, F, D] → [B, V] or [B, V, D] (averaged)."""
        if face_vals.dim() == 2:
            expanded = face_vals.unsqueeze(-1).expand(-1, -1, 3).reshape(B, -1)
            return scatter_add(expanded, idx, dim=1, dim_size=V) / count_safe
        else:
            D = face_vals.shape[-1]
            # [B, F, D] → [B, 3F, D] by repeating per face vertex
            expanded = face_vals.unsqueeze(2).expand(-1, -1, 3, -1).reshape(B, -1, D)
            idx_d = idx.unsqueeze(-1).expand(-1, -1, D)
            scattered = torch.zeros(B, V, D, device=device)
            scattered.scatter_add_(1, idx_d, expanded)
            return scattered / count_safe.unsqueeze(-1)

    if mode == 'norm':
        strain_face = torch.sqrt((E ** 2).sum(dim=(-2, -1)) + 1e-12)  # [B, F]
        return _scatter_face_to_vert(strain_face).unsqueeze(-1)  # [B, V, 1]

    elif mode == 'norm_trace':
        norm_face = torch.sqrt((E ** 2).sum(dim=(-2, -1)) + 1e-12)  # [B, F]
        trace_face = E.diagonal(dim1=-2, dim2=-1).sum(-1)  # [B, F]
        norm_v = _scatter_face_to_vert(norm_face)   # [B, V]
        trace_v = _scatter_face_to_vert(trace_face)  # [B, V]
        return torch.stack([norm_v, trace_v], dim=-1)  # [B, V, 2]

    elif mode == 'full':
        # Upper-triangle of symmetric E: [E_xx, E_yy, E_zz, E_xy, E_xz, E_yz]
        E_xx = E[:, :, 0, 0]  # [B, F]
        E_yy = E[:, :, 1, 1]
        E_zz = E[:, :, 2, 2]
        E_xy = E[:, :, 0, 1]
        E_xz = E[:, :, 0, 2]
        E_yz = E[:, :, 1, 2]
        E_6 = torch.stack([E_xx, E_yy, E_zz, E_xy, E_xz, E_yz], dim=-1)  # [B, F, 6]
        return _scatter_face_to_vert(E_6)  # [B, V, 6]

    elif mode == 'principal':
        # Eigenvalues of symmetric E per face, sorted descending → [B, F, 3]
        E_sym = 0.5 * (E + E.permute(0, 1, 3, 2))  # ensure symmetry
        eigvals = torch.linalg.eigvalsh(E_sym.reshape(-1, 3, 3))  # [B*F, 3] ascending
        eigvals = eigvals.flip(-1).reshape(B, num_faces, 3)  # descending [B, F, 3]
        return _scatter_face_to_vert(eigvals)  # [B, V, 3]

    elif mode == 'local':
        # Per-vertex: [self_norm, neighbor_mean_norm, neighbor_std_norm]
        norm_face = torch.sqrt((E ** 2).sum(dim=(-2, -1)) + 1e-12)  # [B, F]
        self_norm = _scatter_face_to_vert(norm_face)  # [B, V]

        # Build vertex adjacency from faces for 1-ring neighbor aggregation
        f_np = faces.cpu().numpy() if faces.is_cuda else faces.numpy()
        edges = set()
        for f in f_np:
            edges.add((f[0], f[1]))
            edges.add((f[1], f[0]))
            edges.add((f[0], f[2]))
            edges.add((f[2], f[0]))
            edges.add((f[1], f[2]))
            edges.add((f[2], f[1]))
        edge_src = torch.tensor([e[0] for e in edges], device=device, dtype=torch.long)
        edge_dst = torch.tensor([e[1] for e in edges], device=device, dtype=torch.long)

        # Gather neighbor norms
        neigh_norms = self_norm[:, edge_src]  # [B, E]
        neigh_idx = edge_dst.unsqueeze(0).expand(B, -1)
        neigh_ones = torch.ones_like(neigh_norms)
        neigh_sum = torch.zeros(B, V, device=device).scatter_add(1, neigh_idx, neigh_norms)
        neigh_count = torch.zeros(B, V, device=device).scatter_add(1, neigh_idx, neigh_ones)
        neigh_mean = neigh_sum / (neigh_count + 1e-12)

        neigh_sq = torch.zeros(B, V, device=device).scatter_add(1, neigh_idx, neigh_norms ** 2)
        neigh_var = neigh_sq / (neigh_count + 1e-12) - neigh_mean ** 2
        neigh_std = torch.sqrt(neigh_var.clamp(min=0) + 1e-12)

        return torch.stack([self_norm, neigh_mean, neigh_std], dim=-1)  # [B, V, 3]

    else:
        raise ValueError(f"Unknown strain mode: {mode}")


def compute_jacobian_det(deformed_verts, template_verts, faces, neutral_span_inv=None):
    """Per-vertex det(F) - 1, scattered from per-face deformation gradients.

    det(F) > 1  → local expansion  (puffing)
    det(F) = 1  → isometric        (no volume change)
    det(F) < 1  → local compression (wrinkle / suction)

    Returning det(F) - 1 so the neutral pose maps to 0 (cleaner colormap centering).

    Args:
        deformed_verts:    [B, V, 3]
        template_verts:    [B, V, 3]
        faces:             [F, 3]
        neutral_span_inv:  [B, F, 3, 3] or None

    Returns:
        [B, V, 1]  per-vertex det(F) - 1
    """
    B, V, _ = deformed_verts.shape
    num_faces = faces.shape[0]
    device = deformed_verts.device

    F_grad = get_jacobian_matrix(deformed_verts, faces, template_verts,
                                  neutral_span_inv=neutral_span_inv)  # [B, F, 3, 3]
    det_face = torch.det(F_grad.reshape(-1, 3, 3)).reshape(B, num_faces) - 1.0  # [B, F]

    # Scatter face → vertex (average adjacent faces)
    idx = faces.reshape(-1).unsqueeze(0).expand(B, -1)           # [B, 3F]
    ones = torch.ones(B, 3 * num_faces, device=device)
    count = scatter_add(ones, idx, dim=1, dim_size=V) + 1e-12     # [B, V]
    expanded = det_face.unsqueeze(-1).expand(-1, -1, 3).reshape(B, -1)  # [B, 3F]
    det_vert = scatter_add(expanded, idx, dim=1, dim_size=V) / count     # [B, V]

    return det_vert.unsqueeze(-1)  # [B, V, 1]


def compute_jacobian_features(deformed_verts, template_verts, faces, neutral_span_inv=None):
    """Per-vertex Jacobian features via polar decomposition F = R @ U.

    Discards rotation R (irrelevant for EDD — head rotation != skin deformation).
    Returns stretch tensor U features:
        [det(U) - 1,  U_upper_tri(6)]  →  7-dim per vertex

    det(U) - 1: area/volume change signal
        > 0  → local expansion  (puffing)
        = 0  → isometric        (no deformation)
        < 0  → local compression (suction / wrinkles)

    U upper-tri [U00, U11, U22, U01, U02, U12]: stretch magnitude and direction.
        Combined with det, gives DiffusionNetEDD full deformation state signal.

    Args:
        deformed_verts   : [B, V, 3]
        template_verts   : [B, V, 3]
        faces            : [F, 3]
        neutral_span_inv : [F, 3, 3] precomputed, optional

    Returns:
        jac_feat : [B, V, 7]  (det(U)-1 + U upper-tri), face-to-vertex scattered
    """
    B, V, _ = deformed_verts.shape
    num_faces = faces.shape[0]
    device = deformed_verts.device

    # Per-face deformation gradient F [B, F, 3, 3]
    F_grad = get_jacobian_matrix(deformed_verts, faces, template_verts,
                                  neutral_span_inv=neutral_span_inv)
    F_flat = F_grad.reshape(-1, 3, 3)   # [B*F, 3, 3]

    # Polar decomposition: F = R @ U  (U = symmetric stretch tensor)
    # torch.linalg.svd: F = U_svd @ S @ Vh  →  R = U_svd @ Vh,  U = Vh.T @ diag(S) @ Vh
    U_svd, S, Vh = torch.linalg.svd(F_flat)                    # [BF,3,3], [BF,3], [BF,3,3]
    U_stretch = torch.bmm(Vh.mT, torch.bmm(torch.diag_embed(S), Vh))  # [BF, 3, 3]

    # det(U): product of singular values (all positive by construction)
    det_U = S.prod(dim=-1) - 1.0          # [BF]  (subtract 1: 0 at rest)

    # Upper-triangular 6 entries of (U - I): deviation from identity stretch
    # Subtracting I so that neutral pose → all zeros (better conditioning for network)
    u00 = U_stretch[:, 0, 0] - 1.0;  u11 = U_stretch[:, 1, 1] - 1.0;  u22 = U_stretch[:, 2, 2] - 1.0
    u01 = U_stretch[:, 0, 1];        u02 = U_stretch[:, 0, 2];         u12 = U_stretch[:, 1, 2]
    face_feat = torch.stack([det_U, u00, u11, u22, u01, u02, u12], dim=-1)  # [BF, 7]
    face_feat = face_feat.reshape(B, num_faces, 7)                           # [B, F, 7]

    # Scatter face features → vertex features (area-weighted average)
    # Use face area as weight: area ∝ ||(v1-v0) × (v2-v0)||
    fv = deformed_verts[:, faces]                                   # [B, F, 3, 3]
    edge1 = fv[:, :, 1, :] - fv[:, :, 0, :]
    edge2 = fv[:, :, 2, :] - fv[:, :, 0, :]
    areas = torch.cross(edge1, edge2, dim=-1).norm(dim=-1)          # [B, F]

    idx = faces.reshape(-1).unsqueeze(0).expand(B, -1)              # [B, 3F]

    # Weight numerator: sum of (area * feat) per vertex
    # feat expanded to 3 vertices per face, then scatter_add
    feat_exp  = face_feat.unsqueeze(2).expand(-1, -1, 3, -1)        # [B, F, 3, 7]
    area_exp  = areas.unsqueeze(2).unsqueeze(3).expand(-1, -1, 3, 7) # [B, F, 3, 7]
    weighted  = (feat_exp * area_exp).reshape(B, -1, 7)             # [B, 3F, 7]
    area_flat = area_exp[..., 0].reshape(B, -1)                     # [B, 3F]

    feat_sum  = torch.zeros(B, V, 7, device=device)
    area_sum  = torch.zeros(B, V,    device=device)
    idx7      = idx.unsqueeze(-1).expand(-1, -1, 7)                 # [B, 3F, 7]
    feat_sum.scatter_add_(1, idx7, weighted)
    area_sum.scatter_add_(1, idx, area_flat)

    jac_feat = feat_sum / (area_sum.unsqueeze(-1) + 1e-12)          # [B, V, 7]
    return jac_feat


# Strain mode → output dim mapping
STRAIN_MODE_DIM = {
    'norm': 1,
    'norm_trace': 2,
    'full': 6,
    'principal': 3,
    'local': 3,
}


def laplacian_smooth_np(vertices, faces, n_iter=10, lambda_factor=0.5):
    """
    Naive uniform Laplacian smoothing (shrinks mesh). n_iter rounds of shrink only.
    Uses uniform weights (average of neighbors) instead of cotangent weights.
    """
    from scipy.sparse import csr_matrix, diags
    V = vertices.shape[0]
    F = faces

    # Build uniform adjacency Laplacian: L = D_inv @ A - I
    # where A is adjacency, D is degree matrix
    rows = np.concatenate([F[:, 0], F[:, 0], F[:, 1], F[:, 1], F[:, 2], F[:, 2]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 2], F[:, 0], F[:, 1]])
    data = np.ones(len(rows), dtype=np.float64)
    A = csr_matrix((data, (rows, cols)), shape=(V, V))
    A = (A > 0).astype(np.float64)  # binary adjacency (remove duplicates)

    degree = np.array(A.sum(axis=1)).flatten()
    D_inv = diags(1.0 / (degree + 1e-12))

    v = vertices.copy().astype(np.float64)
    for _ in range(n_iter):
        avg = (D_inv @ A @ v)  # neighbor average
        v = v + lambda_factor * (avg - v)  # move toward average
    return v.astype(np.float32)


def taubin_smooth_np(vertices, faces, n_iter=10, lambda_factor=0.5, mu_factor=-0.53):
    """
    Taubin smoothing (shrinkage-free) on mesh vertices.
    Linear iteration: exactly n_iter rounds of (shrink + inflate).

    Args:
        vertices: (V, 3) numpy array
        faces: (F, 3) int numpy array
        n_iter: number of smoothing iterations
        lambda_factor: smoothing step (positive)
        mu_factor: inflation step (negative, |mu| > lambda for shrinkage-free)
    Returns:
        smoothed: (V, 3) numpy float32 array
    """
    import igl
    from scipy.sparse import diags
    L = igl.cotmatrix(vertices.astype(np.float64), faces)
    D_inv = diags(1.0 / (L.diagonal() + 1e-12))
    L_norm = D_inv @ L

    v = vertices.copy().astype(np.float64)
    for _ in range(n_iter):
        v = v + lambda_factor * (L_norm @ v)
        v = v + mu_factor * (L_norm @ v)
    return v.astype(np.float32)


def build_smooth_operator(template_verts, faces, n_iter=10,
                          lambda_factor=0.5, mu_factor=-0.53):
    """
    Build Taubin smoothing operator S as a sparse matrix. 
    (Taubin smoothing, introduced by Gabriel Taubin in 1995, is a widely used iterative mesh smoothing algorithm in computer graphics 
    and 3D modeling that removes high-frequency noise while preserving the overall volume and shape of a mesh)
    S is computed from template (neutral) mesh topology — fixed regardless of deformation.
    Usage: smooth_verts = S @ verts (per-column, i.e. S @ verts_Vx3)

    Args:
        template_verts: (V, 3) numpy array — neutral mesh vertices
        faces: (F, 3) int numpy array
        n_iter: number of Taubin iterations
    Returns:
        S: (V, V) scipy sparse matrix
    """
    import igl
    from scipy.sparse import diags, eye
    L = igl.cotmatrix(template_verts.astype(np.float64), faces)
    D_inv = diags(1.0 / (L.diagonal() + 1e-12))
    L_norm = D_inv @ L
    I = eye(L.shape[0])

    S_shrink = I + lambda_factor * L_norm
    S_inflate = I + mu_factor * L_norm
    S_1iter = S_inflate @ S_shrink

    S = I
    for _ in range(n_iter):
        S = S_1iter @ S
    return S


class Normalizer(object):
    def __init__(self, std_path, device, zero_mean=True):
        if zero_mean:
            self.gradients_std = np.load(os.path.join(std_path, 'gradients_std.npy'))
            self.gradients_std = torch.from_numpy(self.gradients_std).to(device).float()
            self.gradients_mean = torch.eye(3).view(-1,).unsqueeze(0).unsqueeze(0).to(device).float()
        else:
            self.gradients_mean, self.gradients_std = np.load(os.path.join(std_path, 'wks_mean_std.npz'), allow_pickle=True)['arr_0'].item().values()
            self.gradients_std = torch.from_numpy(self.gradients_std).to(device).float()
            self.gradients_mean = torch.from_numpy(self.gradients_mean).to(device).float()

    def normalize(self, tensor):
        return (tensor - self.gradients_mean.to(tensor.device)) / self.gradients_std.to(tensor.device)

    def inv_normalize(self, tensor):
        return tensor * self.gradients_std.to(tensor.device) + self.gradients_mean.to(tensor.device)
    
class Normalizer_img(Normalizer):
    def __init__(self, std_path, device):
        import warnings
        warnings.filterwarnings('ignore')
        with open(os.path.join(std_path, 'img_stat.pkl'), 'rb') as f:
            self.gradients_mean, self.gradients_std = pickle.load(f).values()
        self.gradients_std = torch.tensor(self.gradients_std).to(device).float()
        self.gradients_mean = torch.tensor(self.gradients_mean).to(device).float()

def mesh_transform(mesh,
                   mesh_data="voca",
                   util_dir=f"{abs_path}/mesh_utils",
                   inverse=False):
    """
    Args:
        mesh: trimesh
        mesh_data: the type of mesh data ["biwi", "voca"]
    Return
        mesh
    """
    # load
    mat = np.load(f'{util_dir}/{mesh_data}/align.npy')

    # clone mesh
    mesh_new = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    
    # pre-process biwi
    if mesh_data == 'biwi': 
        if not inverse:
            # for BIWI, mean vertex needs to be moved to zero center
            mesh_new.vertices = mesh_new.vertices - mesh_new.vertices.mean(axis=0) # - np.array([0, 0, 0.03])

    # make 4x4 matrix
    if mat.shape == (3, 4):
        tmp_identity = np.eye(4)[3:4,:]
        mat = np.concatenate([mat, tmp_identity], axis=0)
        # print("voca_mat: ", mat)
    
    # make inverse
    if inverse:
        mat = np.linalg.inv(mat)

    # transform
    mesh_v = np.hstack((mesh_new.vertices, np.ones((mesh_new.vertices.shape[0], 1))))
    mesh_new_v = np.dot(mesh_v, mat.T)
    mesh_new.vertices = mesh_new_v[:,:3]
    return mesh_new

def vert_transform_batch(vertices,
                   mesh_data="voca",
                   util_dir=f"{abs_path}/mesh_utils",
                   inverse=False):
    """
    Args:
        vertices (np.array): [V, 3] vertices of the mesh
        mesh_data: the type of mesh data ["biwi", "voca"]
    Return
        mesh
    """
    # load
    mat = np.load(f'{util_dir}/{mesh_data}/align.npy')
    
    # pre-process biwi
    if mesh_data == 'biwi': 
        if not inverse:
            # for BIWI, mean vertex needs to be moved to zero center
            vertices = vertices - vertices.mean(axis=0) # - np.array([0, 0, 0.03])

    # make 4x4 matrix
    if mat.shape == (3, 4):
        tmp_identity = np.eye(4)[3:4,:]
        mat = np.concatenate([mat, tmp_identity], axis=0)
        # print("voca_mat: ", mat)
    
    # make inverse
    if inverse:
        mat = np.linalg.inv(mat)

    # transform
    mesh_v = np.concatenate((vertices, np.ones((*vertices.shape[:-1], 1))), axis=-1)
    mesh_new_v = np.dot(mesh_v, mat.T)
    return mesh_new_v[...,:3]

def get_mesh_operators(mesh):
    N_FACE = mesh.faces.shape[0]
    N_VERTEX = mesh.vertices.shape[0]
    transf = Transfer(mesh, deepcopy(mesh))
    lu_solver = SuperLU(transf.lu)
    idxs, vals = coalesce(from_dlpack(transf.idxs.toDlpack()).long(), from_dlpack(transf.vals.toDlpack()), m=N_FACE *3, n=N_VERTEX)
    idxs, vals = transpose(idxs, vals, m=N_FACE *3, n=N_VERTEX)
    rhs = transf.cupy_A.T
    return lu_solver, idxs, vals, rhs

class Renderer:
    def __init__(self, view_d=6, img_size=1024, fragments=False):

        import warnings
        warnings.filterwarnings('ignore')
        from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras, 
        PointLights, 
        Materials, 
        RasterizationSettings, 
        MeshRenderer,
        MeshRendererWithFragments,
        MeshRasterizer)

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        R, T = look_at_view_transform(view_d, 0, 0) 
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=img_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            cull_backfaces=True
        )
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        self.return_fragment = fragments
        if self.return_fragment:
            rd = MeshRendererWithFragments
        else:
            rd = MeshRenderer
        materials = Materials(
            device=device,
            specular_color=[[0.0, 0.0, 0.0]],
            shininess=100
        )
        # color = [172, 219, 255]
        color = torch.tensor([255, 255, 255]) / 2 / 255
        lights = PointLights(device=device, location=[[0.0, 0.0, 6]])
        renderer = rd(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=pytorch3d.renderer.HardFlatShader(
                device=device, 
                cameras=cameras,
                lights=lights
            )
        )

        self.renderer = renderer
        self.color = color
        self.device = device
        self.materials = materials
        self.img_normalizer = Normalizer_img(f'{abs_path}/data/MF_all_v5', 'cuda:0')

    def renderbatch(self, vertices, faces, reverse=False):
        meshes = pytorch3d.structures.Meshes(verts=vertices, faces=faces)
        meshes.textures = pytorch3d.renderer.TexturesVertex(verts_features=torch.tensor(self.color).to(self.device).unsqueeze(0).unsqueeze(0).expand(len(meshes), meshes.num_verts_per_mesh()[0], -1))
        if self.return_fragment:
            images, fragments = self.renderer(meshes,materials=self.materials)
        else:
            images = self.renderer(meshes,materials=self.materials)
            fragments = None
        if reverse:
            return 1 - images[..., :3], fragments
        return images[..., :3], fragments


    def mesh2img(self, mesh, reverse=False, noise=False):
        mesh = pytorch3d.structures.Meshes(verts=[torch.from_numpy(mesh.vertices).float().to(self.device)], faces=[torch.from_numpy(mesh.faces).to(self.device)])
        mesh.textures = pytorch3d.renderer.TexturesVertex(verts_features=torch.tensor(self.color).to(self.device).unsqueeze(0).expand_as(mesh.verts_packed())[None])
        if self.return_fragment:
            images, fragments = self.renderer(mesh,materials=self.materials)
        else:
            images = self.renderer(mesh,materials=self.materials)
            fragments = None
        if noise:
            images += (torch.randn(images.shape[:3]).unsqueeze(-1) * 0.01).to(images.device)
        if reverse:
            return ((1 - images[0, ..., :3].detach().cpu().numpy()) * 255).astype(np.uint8), fragments
        return (images[0, ..., :3].detach().cpu().numpy() * 255).astype(np.uint8), fragments
    
    def render_img(self, mesh, img_enc='cnn', img_path=''):
        """
        Args:
            mesh (trimesh.Trimesh): triangle mesh
            img_enc (str): encoder type
            img_path (str): save path for rendered image
        Return:
            img (torch.tensor): rendered image [1, 256, 256, 3]
        """
        img, fragments = self.renderbatch(
            [torch.from_numpy(mesh.vertices).float().to(self.device)], 
            [torch.from_numpy(mesh.faces).float().to(self.device)], 
            reverse=True
        )
        zbuf = fragments.zbuf[0, ..., 0].float()
        img = torch.cat((img, zbuf.unsqueeze(0).unsqueeze(-1)), dim=-1)
        img[img == -1] = img.amax(dim=(0, 1, 2))[-1] * 2
        img = self.img_normalizer.normalize(img)
        if img_path != '':
            np.save(img_path, img.cpu().numpy())
        if img_enc == 'cnn':
            img = img[..., :3]
        return img