# Original code from: https://github.com/chacorp/matplotrender/blob/main/src/matplotrender.py

import os
from glob import glob
# import trimesh
import numpy as np
# from scipy import stats

import torch
import torchvision.utils as tvu
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as matclrs
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation #, PillowWriter
# from functools import partial

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from tqdm import tqdm

import subprocess

@torch.no_grad()
def vis_rig(rig, save_fn, normalize=False):
    # rig shape: (bs, T, 53)
    heatmap = rig.unsqueeze(1) # (bs, 1, T, 53)
    heatmap = heatmap.repeat(1, 3, 1, 1) # (bs, 3, T, 53) --> height, width
    tvu.save_image(heatmap, f"{save_fn}", nrow=1, normalize=normalize)
    
def normalize(V):
    V = V - V.mean(axis=0)
    return V / np.max(np.linalg.norm(V, axis=1))

def normalize_homogeneous(V):
    return np.concatenate([V, np.ones((V.shape[0], 1))], axis=1)
    
def frustum(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M

def ortho(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 2.0 / (right - left)
    M[1, 1] = 2.0 / (top - bottom)
    M[2, 2] = -2.0 / (zfar - znear)
    M[3, 3] = 1.0
    M[0, 3] = -(right + left) / (right - left)
    M[1, 3] = -(top + bottom) / (top - bottom)
    M[2, 3] = -(zfar + znear) / (zfar - znear)
    return M

def perspective(fovy, aspect, znear, zfar):
    h = np.tan(0.5*np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)

def transform(V, M):
    V_h = np.hstack([V, np.ones((V.shape[0], 1))])
    return (M @ V_h.T).T[:, :3]
    
def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype=float)

def yrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ c, 0, s, 0],
                      [ 0, 1, 0, 0],
                      [-s, 0, c, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def zrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ c,-s, 0, 0],
                      [ s, c, 0, 0],
                      [ 0, 0, 1, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def xrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ 1, 0, 0, 0],
                      [ 0, c,-s, 0],
                      [ 0, s, c, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def transform_vertices(frame_v, MVP, F, norm=True, no_parsing=False):
    V = frame_v
    if norm:
        V = (V - (V.max(0) + V.min(0)) *0.5) / max(V.max(0) - V.min(0))
    V = np.c_[V, np.ones(len(V))]
    V = V @ MVP.T
    V /= V[:, 3].reshape(-1, 1)
    if no_parsing:
        return V
    VF = V[F]
    return VF

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(-1)[:,None]
    
def fix_triangle_widning(vertices, faces):
    """Flip triangle winding so the mesh has a consistent, outward-facing orientation
    (same as matplotrender's utils.fix_triangle_widning).

    Face-normal based shading and backface culling assume CCW (outward-facing,
    positive signed volume) winding. Meshes exported with the opposite winding
    flip every face normal, so the outer surface gets culled and hidden interior
    geometry (eye sockets, inner mouth) is drawn instead. This detects that case
    from the signed volume and reverses every face's winding if needed.

    Args
        vertices (np.ndarray): [V, 3] vertex positions
        faces (np.ndarray): [F, 3] triangle indices
    Return
        faces (np.ndarray): [F, 3] triangle indices with outward winding
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    signed_volume = np.sum(np.einsum('ij,ij->i', v0, np.cross(v1, v2)))
    if signed_volume < 0:
        faces = faces[:, [0, 2, 1]]
    return faces

def calc_face_norm(vertices, faces, mode='faces'):
    """
    Args
        vertices (np.ndarray): vertices
        faces (np.ndarray): face indices
    """

    fv = vertices[faces]
    span = fv[:, 1:, :] - fv[:, :1, :]
    norm = np.cross(span[:, 0, :], span[:, 1, :])
    norm = norm / (np.linalg.norm(norm, axis=-1)[:, np.newaxis] + 1e-12)
    
    if mode=='faces':
        return norm
    
    # Compute mean vertex normals manually
    vertex_normals = np.zeros(vertices.shape, dtype=np.float64)
    for i, face in enumerate(faces):
        for vertex in face:
            vertex_normals[vertex] += norm[i]

    # Normalize the vertex normals
    norm_v = vertex_normals / (np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis] + 1e-12)
    return norm_v

def render_mesh(ax, V, MVP, F, norm):
    # quad to triangle    
    VF_tri = transform_vertices(V, MVP, F, norm)

    T = VF_tri[:, :, :2]
    Z = -VF_tri[:, :, 2].mean(axis=1)
    zmin, zmax = Z.min(), Z.max()
    Z = (Z - zmin) / (zmax - zmin)

    C = plt.get_cmap("gray")(Z)
    I = np.argsort(Z)
    T, C = T[I, :], C[I, :]

    collection = PolyCollection(T, closed=False, linewidth=0.2, facecolor=C, edgecolor="black")
    ax.add_collection(collection)

def colors_to_cmap(colors):
    '''
    colors_to_cmap(nx3_or_nx4_rgba_array) yields a matplotlib colormap object that, when
    that will reproduce the colors in the given array when passed a list of n evenly
    spaced numbers between 0 and 1 (inclusive), where n is the length of the argument.

    Example:
      cmap = colors_to_cmap(colors)
      zs = np.asarray(range(len(colors)), dtype=np.float) / (len(colors)-1)
      # cmap(zs) should reproduce colors; cmap[zs[i]] == colors[i]
    '''
    colors = np.asarray(colors)
    if colors.shape[1] == 3:
        colors = np.hstack((colors, np.ones((len(colors),1))))
    steps = (0.5 + np.asarray(range(len(colors)-1), dtype=float))/(len(colors) - 1)
    return matclrs.LinearSegmentedColormap(
        'auto_cmap',
        {clrname: ([(0, col[0], col[0])] + 
                   [(step, c0, c1) for (step,c0,c1) in zip(steps, col[:-1], col[1:])] + 
                   [(1, col[-1], col[-1])])
         for (clridx,clrname) in enumerate(['red', 'green', 'blue', 'alpha'])
         for col in [colors[:,clridx]]},
        N=len(colors)
    )

def get_new_mesh(vertices, faces, v_idx, invert=False):
    """Calculate standardized mesh
    Args:
        vertices (np.ndarray): [V, 3] array of vertices 
        faces (np.ndarray): [F, 3] array of face indices 
        v_idx (np.ndarray): [N] list of vertex index to remove from mesh
    Return:
        updated_verts (np.ndarray): [V', 3] new array of vertices 
        updated_faces (np.ndarray): [F', 3] new array of face indices  
        updated_verts_idx (np.ndarray): [N] list of vertex index to remove from mesh (fixed)
    """
    max_index = vertices.shape[0]
    new_vertex_indices = np.arange(max_index)

    if invert:
        mask = np.zeros(max_index, dtype=bool)
        mask[v_idx] = True
    else:
        mask = np.ones(max_index, dtype=bool)
        mask[v_idx] = False

    updated_verts = vertices[mask]
    updated_verts_idx = new_vertex_indices[mask]

    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(updated_verts_idx)}

    updated_faces = np.array([
                    [index_mapping.get(idx, -1) for idx in face]
                    for face in faces
                ])

    valid_faces = ~np.any(updated_faces == -1, axis=1)
    updated_faces = updated_faces[valid_faces]
    
    return updated_verts, updated_faces, updated_verts_idx

def plot_image(V, F, size=6, xrot=0,yrot=0,zrot=0, aspect=30, dist=-6, norm=False):
    """Render an image of a mesh from vertices and face indices
    Args:
        V (torch.tensor): Single mesh vertices
        F (torch.tensor): Face indices of the mesh
    """        
    ## visualize
    fig = plt.figure(figsize=(size,size))
    ax = fig.add_axes([0,0,1,1], xlim=[-1,+1], ylim=[-1,+1], aspect=1, frameon=False)
    
    ## MVP
    model = translate(0, 0, dist) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
    # proj  = perspective(aspect, 1, 1, 100)
    #proj  = perspective(55, 1, 1, 10)
    proj  = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
    MVP   = proj @ model # view is identity

    render_mesh(ax, V, MVP, F, norm)
    fig.show()
    plt.close()

def plot_image_overlap(Vs, Fs, size=6, xrot=0,yrot=0,zrot=0, dist=-6, norm=False):
    """Render an image of a meshs from vertices and face indices in overlapping manner
    Args:
        Vs (torch.tensor): Batched mesh vertices
        Fs (torch.tensor): Batched face indices of the mesh
    """        
    ## visualize
    fig = plt.figure(figsize=(size,size))
    ax = fig.add_axes([0,0,1,1], xlim=[-1,+1], ylim=[-1,+1], aspect=1, frameon=False)
    
    ## MVP
    model = translate(0, 0, -5) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
    #proj  = perspective(30, 1, 1, 100)
    proj  = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
    MVP   = proj @ model # view is identity

    for V, F in zip(Vs, Fs):
        render_mesh(ax, V, MVP, F, norm)
    plt.show()
    plt.close()

def _homogeneous(V):
    return np.concatenate([V, np.ones((V.shape[0], 1))], axis=1)

def vis_mesh_key_weight(
        verts, faces, key_weight, cage_idx,
        SIZE=4, yrot=0, cmap='magma', vmin=None, vmax=None, show_cbar=True,
        view_yrots=(0, 90, 180)
    ):
    """
    Args:
        verts: (N, 3)
        faces: (F, 3) int
        key_weight: (N, K)
        cage_idx: int
    """

    assert key_weight.shape[0] == verts.shape[0], \
        f"key_weight shape[0] ({key_weight.shape[0]}) != verts.shape[0] ({verts.shape[0]})"
    assert 0 <= cage_idx < key_weight.shape[1], \
        f"cage_idx {cage_idx} out of range [0, {key_weight.shape[1]-1}]"

    print("verts min/max:", verts.min(0), verts.max(0))

    Wv = key_weight[:, cage_idx]
    Wf = Wv[faces].mean(axis=1)

    if vmin is None: vmin = float(Wf.min())
    if vmax is None: vmax = float(Wf.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_fn = plt.get_cmap(cmap)

    V = normalize_homogeneous(verts)
    
    view  = translate(0, 0, -4.5)
#     proj  = perspective(55, 1.0, 1.0, 100.0)
    proj = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
    MV   = proj @ view
    
    
    
    num_views = len(view_yrots)
    fig = plt.figure(figsize=(SIZE* num_views, SIZE ))
    
    for j, add_rot in enumerate(view_yrots):
        model = yrotate(yrot+add_rot)
        
        #model = yrotate(yrot)
        V_mu = np.median(V, axis=0)
        V_model = (V-V_mu) @ model.T + V_mu
        
        #MVP   = proj @ view @ model
        V_proj = V_model @ MV.T
        V_proj  = V_proj[:, :3] / V_proj[:, 3:4]  # (N,3), -1~1
    
        VF = V_proj[faces]
        T = VF[:, :, :2]
        Z = -VF[:, :, 2].mean(1)
        order = np.argsort(Z)
    
        T_sorted  = T[order]
        W_sorted  = Wf[order]
    
        C = cmap_fn(norm(W_sorted))  # (F,4)
    
        #fig = plt.figure(figsize=(SIZE, SIZE))
        # ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, 1], ylim=[-1, 1], aspect=1, frameon=False)
        # coll = PolyCollection(T_sorted, closed=True, linewidth=0.2, facecolor=C, edgecolor=C)
        # ax.add_collection(coll)
        # ax.set_xticks([]); ax.set_yticks([])
        # ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax = fig.add_axes([j / num_views, 0, 1 / num_views, 1],
                          xlim=[-1, 1], ylim=[-1, 1], aspect=1, frameon=False)

        coll = PolyCollection(T_sorted, closed=True, linewidth=0.1,
                              facecolor=C, edgecolor=C)
        ax.add_collection(coll)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        # ax.set_title(f"y={yrot + add_rot}° ({mode})", fontsize=10)
    
        if show_cbar:
            # colorbar
            sm = ScalarMappable(norm=norm, cmap=cmap_fn)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
            cbar.set_label(f"key_weight[:, {cage_idx}]")
    
        plt.title(f"Cage vertex {cage_idx} weight")
    plt.show()

def vis_mesh_all_cage_weights(
    verts, faces, key_weight,
    cage_indices=None,
    mode='argmax',              # 'argmax' | 'blend'
    SIZE=4, yrot=0,
    base_cmap='tab20',
    show_legend=False,
    mesh_scale=1.0,
    mesh_trans=np.array([0,0.0,0]),
    light_dir=np.array([0,0,1]),
    view_yrots=(0, 90, 180),
    save=False,
    logdir='.', 
    name='test'
):
    """
    Visualizing per cage weight for each face (triangle).
    

    - mode='argmax': color with the single cage that has maximum weight
    - mode='blend' : average the cage color using cage weight

    Args:
        verts: (N, 3)
        faces: (F, 3) int
        key_weight: (N, K)  # mesh vertex x cage vertex
        cage_indices: list[int] or None
        base_cmap: matplotlib colormap name ('tab20', 'tab20b', ...)
    """
    verts  = np.asarray(verts) * mesh_scale + mesh_trans
    faces  = np.asarray(faces, dtype=int)
    W_full = np.asarray(key_weight)
    N, K = W_full.shape
    assert verts.shape[0] == N, f"verts ({verts.shape[0]}) vs key_weight rows ({N}) mismatch"

    if cage_indices is None:
        cage_indices = list(range(K))
    else:
        cage_indices = list(cage_indices)
        assert all(0 <= i < K for i in cage_indices), "cage_indices out of range"

    M = len(cage_indices)

    ## color palete
    cmap = plt.get_cmap(base_cmap, max(M, 3))  # least 3
    
    # if M <= cmap.N:
    if False:
        base_colors = np.asarray([cmap(i) for i in range(M)])  # (M,4)
    else:
        cmap_wide = plt.get_cmap('nipy_spectral')
        # cmap_wide = plt.get_cmap('gist_rainbow')
        # cmap_wide = plt.get_cmap('rainbow') ##
        # cmap_wide = plt.get_cmap('jet')
        # cmap_wide = plt.get_cmap('cubehelix')
        # cmap_wide = plt.get_cmap('turbo')
        # cmap_wide = plt.get_cmap('gist_ncar')
        base_colors = np.asarray([cmap_wide(i/(M-1)) for i in range(M)])
        # base_colors = np.asarray([cmap_wide(i/(M)) for i in range(M)])

    # (N, M) 선택 케이지 weight
    W_sel = W_full[:, cage_indices]  # (N, M)
    Wf_all = W_sel[faces].mean(axis=1)

    #V = normalize_homogeneous(verts)
    #V = verts #- verts.mean(0, keepdims=True)
    V = normalize_homogeneous(verts)
    view = translate(0, 0, -4.5)
    #proj = perspective(45, 1.0, 1.0, 100.0)
    proj  = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
    MV   = proj @ view
    
    num_views = len(view_yrots)
    fig = plt.figure(figsize=(SIZE* num_views, SIZE ))
    
    for j, add_rot in enumerate(view_yrots):
        model = yrotate(yrot+add_rot)
        
        #V_clip = V @ MVP.T
        # V_clip = transform(V, MVP)
        C = calc_face_norm(verts, faces) @ model[:3,:3].T
        
        # V_mu = np.median(V, axis=0)
        V_mu = np.array([[0, 0, -0.45, 0]])
        V_model = (V - V_mu) @ model.T + V_mu
        #V_model = V_model - np.mean(V_model, axis=0) + V_mu

        V_proj = V_model @ MV.T
        V_ndc  = V_proj[:, :3] / V_proj[:, 3:4]  # (N,3), -1~1
        # V_ndc  = V_clip[:, :3] / V_clip[:, 3:4]
    
        VF = V_ndc[faces]        # (F, 3, 3)
        T  = VF[:, :, :2]        # (F, 3, 2)
        Z  = -VF[:, :, 2].mean(1)
        order = np.argsort(Z)
    
        T_sorted = T[order]
        W_sorted = Wf_all[order]
        C_sorted = C[order]
        NI = np.argwhere(C_sorted[:,2] > 0).squeeze()

        T_sorted = T_sorted[NI]
        W_sorted = W_sorted[NI]
        C_sorted = C_sorted[NI]
        
        C_sorted = (C_sorted @ light_dir)[:,np.newaxis].repeat(3, axis=-1)
        C_sorted = C_sorted*0.7+0.2
    
        if mode == 'argmax':
            labels = np.argmax(W_sorted, axis=-1)  # (F,)
            face_colors = base_colors[labels]     # (F,4)
            face_colors[:,:3] = face_colors[:,:3] *0.7 + C_sorted *.3
            print(face_colors.shape)
        elif mode == 'blend':
            # sum_row = W_sorted.sum(axis=1, keepdims=True) + 1e-12
            # Wn = W_sorted / sum_row
            Wn = W_sorted #/ W_sorted.max()
            # print(Wn.min(), Wn.max())
            rgb = Wn @ base_colors[:, :3]    # (F,3)
            # rgb = rgb * 2
            # print(rgb.min(), rgb.max())
            rgb = (rgb-rgb.min(0)) / (rgb.max(0) - rgb.min(0))
            print(Wn.shape, T_sorted.shape, rgb.shape, C_sorted.shape)
            rgb = rgb *0.7 + C_sorted *.3
            rgb = np.clip(rgb, 0, 1)
            alpha = np.ones((rgb.shape[0], 1))
            face_colors = np.concatenate([rgb, alpha], axis=1)  # (F,4)
        else:
            raise ValueError("mode must be 'argmax' or 'blend'")
    
        ### PLOT
        #fig = plt.figure(figsize=(SIZE, SIZE))
        # ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, 1], ylim=[-1, 1], aspect=1, frameon=False)
        ax = fig.add_axes([j / num_views, 0, 1 / num_views, 1],
                          xlim=[-1, 1], ylim=[-1, 1], aspect=1, frameon=False)

        coll = PolyCollection(T_sorted, closed=True, linewidth=0.1,
                              facecolor=face_colors, edgecolor='black')
        ax.add_collection(coll)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_title(f"y={yrot + add_rot}° ({mode})", fontsize=10)
        # plt.title(f"All cage weights ({mode})")
    
        if show_legend:
            handles = []
            for i, ci in enumerate(cage_indices):
                handles.append(mpatches.Patch(color=base_colors[i], label=f"cage {ci}"))
            
            if len(handles) <= 12:
                ax.legend(handles=handles, loc='upper right', fontsize=8)
            else:
                ax.legend(handles=handles, bbox_to_anchor=(1.02, 1.0), loc='upper left',
                          borderaxespad=0., fontsize=7, ncol=1)

    # plt.show()
    if save:
        plt.savefig('{}/{}.png'.format(logdir, name), bbox_inches = 'tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

def plot_image_array(Vs,
                     Fs,
                     rot_list=None,
                     size=6,
                     norm=False,
                     mode='mesh',
                     linewidth=1,
                     linestyle='solid',
                     light_dir=np.array([0,0,1]),
                     bg_black = True,
                     logdir='.',
                     name='000',
                     use_persp=False,
                     save=False,
                     is_diff=False,
                     diff_base=None,  # list of arrays, one per mesh in Vs, or a single array broadcast to all
                     vmin=None,
                     vmax=None,
                    ):
    """
    Args:
        Vs (list): list of vertices [V, V, V, ...]
        Fs (list): list of face indices [F, F, F, ...]
        rot_list (list): list of euler angle [ [x,y,z], [x,y,z], ...]
        size (int): size of figure
        norm (bool): if True, normalize vertices
        mode (str): mode for rendering [mesh(wireframe), shade, normal]
        linewidth (float): line width for wireframe (kwargs for matplotlib)
        linestyle (str): line style for wireframe (kwargs for matplotlib)
        light_dir (np.array): light direction
        bg_black (bool): if True, use dark_background for plt.style
        logdir (str): directory for saved image
        name (str): name for saved image
        save (bool): if True, save the plot as image
        is_diff (bool): if True, color each mesh by per-face displacement magnitude
            from diff_base (YlOrRd, blended onto the same lit-shading base as
            mode='shade') instead of plain shading -- goes through the exact
            same camera/projection/rotation code as every other mode, so an
            is_diff=True render is directly comparable (same framing) to a
            plain render of the same Vs/Fs/rot_list.
        diff_base: required if is_diff. Either one array (broadcast to all
            meshes) or a list the same length as Vs (one base mesh per entry --
            e.g. each mesh's own neutral/rest template).
        vmin, vmax: fixed displacement range for the color scale (before the
            [0,1] normalization). Default (None) auto-scales per mesh.
    """
    if is_diff and diff_base is None:
        raise ValueError('diff_base is None!')
    diff_bases = None
    if is_diff:
        diff_bases = diff_base if isinstance(diff_base, (list, tuple)) else [diff_base] * len(Vs)
    if mode=='gouraud':
        print("currently WIP!: need to curl by z")
        
    num_meshes = len(Vs)
    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    fig = plt.figure(figsize=(size * num_meshes, size))  # Adjust figure size based on the number of meshes
    
    for idx, (V, F) in enumerate(zip(Vs, Fs)):
        # face-normal culling below needs outward winding (some meshes are inverted)
        F = fix_triangle_widning(V, F)

        # Calculate the position of the subplot for the current mesh
        ax_pos = [idx / num_meshes, 0, 1 / num_meshes, 1]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)

        # xrot, yrot, zrot = rot[0], 90, rot[2]
        if rot_list:
            xrot, yrot, zrot = rot_list[idx]
        else:
            xrot, yrot, zrot = 0,0,0
            
        ### MVP
        # model = translate(0, 0, -3) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        model = (yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot))[:3,:3]
        view = translate(0, 0, -5)
        if use_persp:
            proj  = perspective(55, 1, 1, 100)
        else:
            proj  = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
        #MVP   = proj @ view @ model
        MVP   = proj @ view # view is identity
        V_mu = np.median(V, axis=0)
        
        # V_h_mu = _homogeneous(V - V_mu)
        
        # quad to triangle    
        V_model = (V - V_mu) @ model.T + V_mu
        V_proj = _homogeneous(V_model) @ MVP.T
        V_proj = V_proj[:, :3] / V_proj[:, 3:4]  # (N,3), -1 ~ 1
        VF_tri = V_proj[F]
        # VF_tri = transform_vertices(V, MVP, F, norm)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)

        if is_diff:
            D = diff_bases[idx]
            diff = np.abs(D - V)[F]           # [num_faces, 3, 3]
            diff = np.linalg.norm(diff, axis=1)  # [num_faces, 3]
            diff = np.linalg.norm(diff, axis=1)  # [num_faces]

            C = calc_face_norm(V, F) @ model[:3,:3].T
            I = np.argsort(Z)
            T, C, diff = T[I, :], C[I, :], diff[I]

            NI = np.argwhere(C[:,2] > 0)[:,0]
            T, C, diff = T[NI, :], C[NI, :], diff[NI]

            lit = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)
            lit = np.clip(lit, 0, 1)
            lit = lit*0.7+0.2

            lo = diff.min() if vmin is None else vmin
            hi = diff.max() if vmax is None else vmax
            dn = np.clip((diff - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(diff)
            Dc = plt.get_cmap("YlOrRd")(dn)[:, :3]
            mask = dn[:, np.newaxis]
            C = lit * (1 - mask) + Dc * mask
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        elif mode=='normal':
            C = calc_face_norm(V, F) @ model[:3,:3].T

            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] > 0)[:,0]
            T, C = T[NI, :], C[NI, :]
            C = np.clip(C, 0, 1) if False else C * 0.5 + 0.5
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        elif mode=='shade':
            C = calc_face_norm(V, F) @ model[:3,:3].T
            
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] > 0)[:,0]
            T, C = T[NI, :], C[NI, :]
            
            C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)
            C = np.clip(C, 0, 1)
            C = C*0.7+0.2
            collection = PolyCollection(T, closed=False, linewidth=linewidth,facecolor=C, edgecolor=C)
        elif mode=='gouraud':
#             I = np.argsort(Z)
#             V, F, vidx = get_new_mesh(V, F, I, invert=True)
            
            ### curling by normal
            C = calc_face_norm(V, F, mode='v') #@ model[:3,:3].T
            NI = np.argwhere(C[:,2] > 0.0)[:,0]
            V, F, vidx = get_new_mesh(V, F, NI, invert=True)
            #F = np.flip(F, 1)#F[:,::-1]
            C = calc_face_norm(V, F, mode='v') #@ model[:3,:3].T
            
            #VV = (V-V.min()) / (V.max()-V.min())# world coordinate
            V = transform_vertices(V, MVP, F, norm, no_parsing=True)
            
            #VF_tri = transform_vertices(V, MVP, F, norm)
#             print(V.shape)
            
            triangle_ = tri.Triangulation(V[:,0], V[:,1], triangles=F)
            #print(triangle_.shape)
            C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)
            C = np.clip(C, 0, 1)
            C = C*0.5+0.25
            # VV = (V-V.min()) / (V.max()-V.min()) #screen coordinate
            # cmap = colors_to_cmap(VV)
            cmap = colors_to_cmap(C)
            zs = np.linspace(0.0, 1.0, num=V.shape[0])
            plt.tripcolor(triangle_, zs, cmap=cmap, shading='gouraud')
            
        else:
            C = plt.get_cmap("gray")(Z)
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]
            
            collection = PolyCollection(T, closed=False, linewidth=0.23, facecolor=C, edgecolor='black')
            
        if mode!='gouraud':
            ax.add_collection(collection)
        plt.xticks([])
        plt.yticks([])
    
    if save:
        plt.savefig('{}/{}.png'.format(logdir, name), bbox_inches = 'tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
    # plt.close(plt.gcf())
        

from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.cm import get_cmap
def plot_image_array_grd(Vs, Fs, 
                         Cs=None, 
                         rot_list=None, 
                         size=6, 
                         norm=False, 
                         mode='shade', # not used
                         threshold=0.01,
                         seg_divide=False,
                         is_diff=False,
                         diff_base=None, # required if is_diff is True
                         diff_revert=False,
                         light_dir=np.array([0, 0, 1]),
                         light_sticked=True, # light sticked to the front of the face
                         view_dir=np.array([0, 0, 1]),
                         mesh_trans=np.array([0, 0, 0]),
                         mesh_scale=1.0,
                         c_map="nipy_spectral", 
                         blend=0.5,
                         seg_only=-1,
                         bg_black=True, logdir='.', name='000', save=False, show=True):
    """
    v_list=[ vertices]
    f_list=[ faces ]
    c_list=[ ict_vert_segment ]
    diff_list=[ vertices - ict_tri.vertices ]
    
    plot_image_array_grd(
        v_list, f_list, 
        # c_list, # uncomment to visualize segmentation
        # diff_list, is_diff=True, diff_base=ict_tri.vertices, # uncomment to visualize difference
        rot_list=[[0,-10,0]]*len(v_list), 
        size=SIZE, 
        bg_black=False,
        show=True
    )
    """
    

    num_meshes = len(Vs)
    plt.style.use('dark_background' if bg_black else 'default')
    fig = plt.figure(figsize=(size * num_meshes, size))
    
    if is_diff:
        if diff_base is None:
            raise ValueError('diff_base is None!')
            
    if Cs==None:
        Cs = [None] * num_meshes
    
    # light source data type int -> float
    light_dir_view = light_dir.astype(float)
                
    for idx, (V, F, C) in enumerate(zip(Vs, Fs, Cs)):
                
        if norm:
            V = normalize(V)
            
        if is_diff:
            #C = np.linalg.norm(abs(C), axis=-1)
            C = np.linalg.norm(abs(V-diff_base), axis=-1)
            C = (C - C.min(0)) / (C.max(0) - C.min(0))
            C = 1 - C if diff_revert else C
            #print(C.shape, C.max(0), C.min(0))
            # diff = D_diff[idx]
            # if diff_max > 0:
            #     diff = (diff - diff_min) / (diff_max - diff_min)
        
        V = V * mesh_scale + mesh_trans
    
        ax_pos = [idx / num_meshes, 0, 1 / num_meshes, 1]
        ax = fig.add_axes(ax_pos, xlim=[-1, 1], ylim=[-1, 1], aspect=1, frameon=False)

        xrot, yrot, zrot = rot_list[idx] if rot_list else (0, 0, 0)
        model = translate(0, 0, -5) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        proj = ortho(-1, 1, -1, 1, 1, 100)
        MVP = proj @ model

        # Normal 변환은 inverse transpose의 rotation part
        model_rot = model[:3, :3]
        model_rot_invT = np.linalg.inv(model_rot).T

        # 정점 normal
        vertex_normals_obj = calc_face_norm(V, F, mode='v')
        vertex_normals_view = (model_rot_invT @ vertex_normals_obj.T).T
        
        face_normals_obj = calc_face_norm(V, F)
        face_normals_view = (model_rot_invT @ face_normals_obj.T).T
        keep = (face_normals_view @ view_dir) >= 0
        #F = F[keep]
        
        if light_sticked:
            # light sticked to front face
            intensity = np.clip(vertex_normals_obj @ light_dir_view, 0, 1)
        else:
            # intensity per vertex
            intensity = np.clip(vertex_normals_view @ light_dir_view, 0, 1)
        
        # 정점 변환
        V_proj = transform(V, MVP)

        # depth sorting: triangle 중심 z
        tri_depth = V_proj[F][:, :, 2].mean(axis=1)
        sort_idx = np.argsort(-tri_depth)
        F_sorted = F[sort_idx]

        # Gouraud shading
        triang = tri.Triangulation(V_proj[:, 0], V_proj[:, 1], F_sorted)
                
        vertex_color = intensity[..., np.newaxis].repeat(3, axis=-1)
        vertex_color = vertex_color *0.7 + 0.2
        # vertex_color = vertex_color *0.6 + 0.3
        
        if is_diff:
            Sc = plt.get_cmap("YlOrRd")(C)[...,:3] ## [N, 4]
            mask = C[:,np.newaxis]
            #vertex_color = vertex_color*(1-blend)*(1-mask) + blend*Sc*mask
            vertex_color = vertex_color*(1-mask) + Sc*mask
            
        else:
            # if type(C)==np.ndarray:
            #     C=torch.tensor(C)
            if type(C)==torch.tensor:
                C=C.numpy()
                
            if C.any() != None:
                len_seg = C.shape[-1]
                #S = softmax(C) # softmax
                    
                S=C
                S = S.argmax(-1)#.float()

                if seg_only>0:
                    SF = S[F_sorted]

                    v_mask = (S==seg_only).any(-1)
                    vertex_color = vertex_color[v_mask][0]
                    S = S[v_mask][0]

                    f_mask = (SF==seg_only).any(-1)
                    F_sorted_masked = F_sorted[f_mask]
                    triang = tri.Triangulation(V_proj[:, 0], V_proj[:, 1], F_sorted_masked)

                S = S.astype(float)
                Sc = plt.get_cmap(c_map)(S/len_seg)[...,:3]

                vertex_color = vertex_color*(1-blend) + blend*Sc
                vertex_color = np.clip(vertex_color, 0, 1)

    #         if seg_divide:
    #             n_model = model.copy()
    #             n_model[:3, :3] = model_rot_invT
    #             N_MVP = proj @ n_model
    #             N_proj = transform(vertex_normals_view, N_MVP)

    #             for seg_n in range(len_seg):
    #                 V_proj_seg = V_proj.copy()
    #                 SF = S[F_sorted]

    #                 v_mask = (S==seg_n).any(-1)
    #                 vertex_color_seg = vertex_color[v_mask][0]
    #                 S = S[v_mask][0]

    #                 f_mask = (SF==seg_n).any(-1)
    #                 F_sorted_masked = F_sorted[f_mask]
    #                 V_proj_seg[v_mask, :2] = V_proj_seg[v_mask, :2] + (V_proj_seg[v_mask, :2]*0.5)
    #                 triang = tri.Triangulation(V_proj_seg[:, 0], V_proj_seg[:, 1], F_sorted_masked)

    #                 cmap = colors_to_cmap(vertex_color_seg)
    #                 zs = np.linspace(0.0, 1.0, num=V.shape[0])
    #                 plt.tripcolor(triang, zs, cmap=cmap, shading='gouraud')
    #         else:
        cmap = colors_to_cmap(torch.tensor(vertex_color))
        zs = np.linspace(0.0, 1.0, num=V.shape[0])
        plt.tripcolor(triang, zs, cmap=cmap, shading='gouraud')
            
        ax.set_xticks([])
        ax.set_yticks([])

    if save:
        plt.savefig(f'{logdir}/{name}.png', bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
        
def plot_image_array_VC(V, 
                     F, 
                     VCs,
                     rot_list=None, 
                     size=6, 
                     norm=False, 
                     mode='mesh', 
                     linewidth=1, 
                     linestyle='solid', 
                     light_dir=np.array([0,0,1]),
                     bg_black = True,
                     logdir='.', 
                     name='000', 
                     save=False,
                     draw_base=True,
                    DC=False,
                    threshold=None,
                    ):
    num_meshes = len(VCs)
    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    fig = plt.figure(figsize=(size * num_meshes, size))  # Adjust figure size based on the number of meshes
    
    #xrot, yrot, zrot = rot[0], 90, rot[2]
    if rot_list:
        xrot, yrot, zrot = rot_list[0]
    else:
        xrot, yrot, zrot = 0,0,0
    import pdb;pdb.set_trace()
    ## MVP
    model = translate(0, 0, -2) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
    # proj  = perspective(30, 1, 1, 100)
    proj  = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
    # proj  = perspective(55, 1, 1, 10)
    MVP   = proj @ model # view is identity
    if draw_base:
        # Calculate the position of the subplot for the current mesh
        ax_pos = [0 / num_meshes, 0, 1 / num_meshes, 1]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        VF_tri = transform_vertices(V[0], MVP, F, norm)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)

        C = calc_face_norm(V, F) @ model[:3,:3].T

        I = np.argsort(Z)
        T, C = T[I, :], C[I, :]

        NI = np.argwhere(C[:,2] > 0).squeeze()
        T, C = T[NI, :], C[NI, :]

        C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)            
        C = np.clip(C, 0, 1)

        #C = C*0.5+0.25
        C = C*0.6+0.15

        C = np.clip(C, 0, 1)
        collection = PolyCollection(T, closed=False, linewidth=linewidth,facecolor=C, edgecolor=C)
        ax.add_collection(collection)
        plt.xticks([])
        plt.yticks([])
    
    #for idx, V in enumerate(Vs):
    DDD = np.array(VCs) # B V 3
    if threshold is not None:
        DDD[DDD > threshold] = 0
    diff_min, diff_max = DDD.min(), DDD.max()
    print(diff_min, diff_max)
    
    for idx, vc in enumerate(VCs):
        idx = idx + 1
        # Calculate the position of the subplot for the current mesh
        ax_pos = [idx / num_meshes, 0, 1 / num_meshes, 1]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        vc = vc[F]
        vc = np.linalg.norm(vc, axis=1) # N 3
        
        if diff_max > 0:
            vc = (vc - diff_min) / (diff_max - diff_min)    
            
        vc = vc[I,:]
        vc = vc[NI, :]
        
        if DC:
            mask = vc.mean(1)[:,np.newaxis]
            vc = vc.mean(1)
            vc = plt.get_cmap("YlOrRd")(vc) ## [N, 4]
        else:
            mask = 0.5
            
        C_ = C*(1-mask)+ vc[:,:3]*(mask)
        
        C_ = np.clip(C_, 0, 1)
        collection = PolyCollection(T, closed=False, linewidth=linewidth,facecolor=C_, edgecolor=C_)
        
        ax.add_collection(collection)
        plt.xticks([])
        if DC:
            plt.xlabel(f'min:{VCs[idx-1].min():.5f} | max: {VCs[idx-1].max():.5f}')
        plt.yticks([])
    
    if save:
        plt.savefig('{}/{}.png'.format(logdir, name), bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()
        plt.close()
        
def plot_image_array_diff(Vs,
                     Fs,
                     Ds,
                     rot_list=None,
                     size=6,
                     norm=False,
                     mode='mesh',
                     linewidth=1,
                     linestyle='solid',
                     light_dir=np.array([0,0,1]),
                     bg_black = True,
                    logdir='.',
                    name='000',
                     save=False,
                    draw_base=True,
                    vmin=None,
                    vmax=None,
                    ):
    """
    Renders displacement for each mesh: requires vertices for each mesh sequence

    vmin/vmax: if given, use this fixed displacement range (in mesh units, before
    the [0,1] normalization) instead of each panel's own min/max, so displacement
    color is directly comparable across panels/images. Default (None) preserves
    the original per-panel auto-normalization behavior.
    """
    num_meshes = len(Vs)
    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    fig = plt.figure(figsize=(size * num_meshes, size))  # Adjust figure size based on the number of meshes
    
    ##### 
    V = Ds[0]
    F = Fs[0]

    #xrot, yrot, zrot = rot[0], 90, rot[2]
    if rot_list:
        xrot, yrot, zrot = rot_list[0]
    else:
        xrot, yrot, zrot = 0,0,0
    ## MVP
    model = translate(0, 0, -2) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
    # proj  = perspective(30, 1, 1, 100)
    #proj  = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
    proj  = perspective(55, 1, 1, 10)
    MVP   = proj @ model # view is identity
    
    if draw_base:
        # Calculate the position of the subplot for the current mesh
        ax_pos = [0 / num_meshes, 0, 1 / num_meshes, 1]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        VF_tri = transform_vertices(V, MVP, F, norm)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)

        C = calc_face_norm(V,F) @ model[:3,:3].T

        I = np.argsort(Z)
        T, C = T[I, :], C[I, :]

        NI = np.argwhere(C[:,2] > 0).squeeze()
        T, C = T[NI, :], C[NI, :]

        C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)            
        C = np.clip(C, 0, 1)

        #C = C*0.5+0.25
        C = C*0.6+0.15

        C = np.clip(C, 0, 1)
        collection = PolyCollection(T, closed=False, linewidth=linewidth,facecolor=C, edgecolor=C)
        ax.add_collection(collection)
        plt.xticks([])
        plt.yticks([])
    
    
    for idx, (V, F, D) in enumerate(zip(Vs, Fs, Ds)):
        idx = idx + 1
        # Calculate the position of the subplot for the current mesh
        ax_pos = [idx / num_meshes, 0, 1 / num_meshes, 1]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)

        #xrot, yrot, zrot = rot[0], 90, rot[2]
        if rot_list:
            xrot, yrot, zrot = rot_list[idx]
        else:
            xrot, yrot, zrot = 0,0,0
        ## MVP
        # model = translate(0, 0, -6) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        # # proj  = perspective(30, 1, 1, 100)
        # proj  = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
        # MVP   = proj @ model # view is identity
        # quad to triangle    
        VF_tri = transform_vertices(V, MVP, F, norm)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)

        diff = np.array(abs(D - V))
        diff = diff[F] # N 3 3
        diff = np.linalg.norm(diff, axis=1) # N 3
        diff = np.linalg.norm(diff, axis=1) # N
        
        diff_min, diff_max = (0.0, vmax) if vmax is not None else (diff.min(), diff.max())
        if vmin is not None:
            diff_min = vmin
        if diff_max > diff_min:
            diff = np.clip((diff - diff_min) / (diff_max - diff_min), 0, 1)

        C = calc_face_norm(V,F) @ model[:3,:3].T

        I = np.argsort(Z)
        T, C = T[I, :], C[I, :]
        diff = diff[I]

        NI = np.argwhere(C[:,2] > 0).squeeze()
        T, C = T[NI, :], C[NI, :]
        diff = diff[NI]

        C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)            
        C = np.clip(C, 0, 1)

        #C = C*0.5+0.25
        C = C*0.6+0.15

        Dc = plt.get_cmap("YlOrRd")(diff) ## [N, 4]
        #     diff = diff *0.8
        #     diff = diff[:,:3] - diff[:,3:]
        # print(Dc.shape)
        # print(Dc[:,:3].min(0), Dc[:,:3].max(0))
        mask = diff[:,np.newaxis]
        C = C*(1-mask)+ Dc[:,:3]*(mask)
        # C[:,0]=C[:,0]+diff*0.2
        # C[:,0]=C[:,0]+diff
        C = np.clip(C, 0, 1)
        collection = PolyCollection(T, closed=False, linewidth=linewidth,facecolor=C, edgecolor=C)
        
        ax.add_collection(collection)
        plt.xticks([])
        plt.yticks([])
    
    if save:
        plt.savefig('{}/{}.png'.format(logdir, name), bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_image_array_diff2(Vs, 
                     Fs, 
                     D,
                     rot_list=None, 
                     size=6, 
                     norm=False, 
                     mode='mesh', 
                     linewidth=1, 
                     linestyle='solid', 
                     light_dir=np.array([0,0,1]),
                     bg_black = True,
                    threshold=None,
                    logdir='.', 
                    name='000', 
                     save=False,
                    draw_base=True,
                    c_map = 'YlOrRd'
                    ):
    """
    v_list=[ mesh_base.vertices, mesh1.vertices, mesh2.vertices ]
    f_list=[ mesh_base.faces,    mesh1.faces,    mesh2.faces ]
    d_list= mesh_base.vertices 
    SIZE=4

    plot_image_array_diff2(
        v_list, f_list, d_list,
        rot_list=[[0,-10,0]]*len(v_list), 
        size=SIZE,
    )
    """
    num_meshes = len(Vs)
    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    fig = plt.figure(figsize=(size * num_meshes, size))  # Adjust figure size based on the number of meshes
    
    ##### 
    V = D
    F = Fs[0]

    #xrot, yrot, zrot = rot[0], 90, rot[2]
    if rot_list is not None:
        xrot, yrot, zrot = rot_list[0]
    else:
        xrot, yrot, zrot = 0,0,0
    ## MVP
    model = translate(0, 0, -4) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
    # proj  = perspective(30, 1, 1, 100)
    proj  = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
    # proj  = perspective(55, 1, 1, 10)
    MVP   = proj @ model # view is identity
    
    if draw_base:
        # Calculate the position of the subplot for the current mesh
        ax_pos = [0 / num_meshes, 0, 1 / num_meshes, 1]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        VF_tri = transform_vertices(V, MVP, F, norm)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)

        C = calc_face_norm(V, F) @ model[:3,:3].T

        I = np.argsort(Z)
        T, C = T[I, :], C[I, :]

        NI = np.argwhere(C[:,2] > 0).squeeze()
        T, C = T[NI, :], C[NI, :]

        C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)            
        C = np.clip(C, 0, 1)

        C = C*0.6+0.3

        C = np.clip(C, 0, 1)
        collection = PolyCollection(T, closed=False, linewidth=linewidth,facecolor=C, edgecolor=C)
        ax.add_collection(collection)
        plt.xticks([])
        plt.yticks([])
    
        
    #for idx, V in enumerate(Vs):
    DDD = np.array(abs(D - np.array(Vs)))**2 # B V 3
    DDD = DDD[:, F] # B N 3 3
    DDD = np.linalg.norm(DDD, axis=-1) # B N 3
    DDD = np.linalg.norm(DDD, axis=-1) # B N
    
    #DDD = np.clip(DDD, 0, 0.06)
    if threshold is not None:
        DDD[DDD > threshold] = 0
    diff_min, diff_max = DDD.min(), DDD.max()
    
    for idx, (V, F) in enumerate(zip(Vs, Fs)):
    
        #xrot, yrot, zrot = rot[0], 90, rot[2]
        if rot_list:
            xrot, yrot, zrot = rot_list[idx]
        else:
            xrot, yrot, zrot = 0,0,0
              
        idx = idx + 1
        # Calculate the position of the subplot for the current mesh
        ax_pos = [idx / num_meshes, 0, 1 / num_meshes, 1]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        # quad to triangle & apply MVP
        VF_tri = transform_vertices(V, MVP, F, norm)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)

        diff = DDD[idx-1]
        if diff_max > 0:
            diff = (diff - diff_min) / (diff_max - diff_min)    

        C = calc_face_norm(V, F) @ model[:3,:3].T

        I = np.argsort(Z)
        T, C = T[I, :], C[I, :]
        diff = diff[I]

        NI = np.argwhere(C[:,2] > 0).squeeze()
        T, C = T[NI, :], C[NI, :]
        diff = diff[NI]

        C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)            
        C = np.clip(C, 0, 1)

        #C = C*0.5+0.25
        C = C*0.6+0.3

        Dc = plt.get_cmap(c_map)(diff)## [N, 4]
        mask = diff[:,np.newaxis]
        C = C*(1-mask)+ Dc[:,:3]*(mask)
        C = np.clip(C, 0, 1)
        collection = PolyCollection(T, closed=False, linewidth=linewidth,facecolor=C, edgecolor=C)
        
        ax.add_collection(collection)
        plt.xticks([])
        #plt.xlabel(f'min:{DDD[idx-1].min():.5f} | max: {DDD[idx-1].max():.5f} | norm: {np.linalg.norm(DDD[idx-1], axis=-1):.5f}')
        plt.xlabel(f'min:{DDD[idx-1].min():.5f} | max: {DDD[idx-1].max():.5f}')
        plt.yticks([])
    
    if save:
        plt.savefig('{}/{}.png'.format(logdir, name), bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()
        plt.close()

def setup_plot(bg_black, size, num_meshes):
    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    fig, axes = plt.subplots(1, num_meshes, figsize=(size * num_meshes, size))
    if num_meshes == 1:
        axes = [axes]
    for ax in axes:
        ax.set_xlim(-1, 1)  # Adjusted to prevent cutting off the mesh
        ax.set_ylim(-1, 1)  # Adjusted to prevent cutting off the mesh
        ax.set_aspect('equal')
        ax.axis('off')
    return fig, axes

def transform_and_project(V, F, MVP, norm):
    VF_tri = transform_vertices(V, MVP, F, norm)
    T = VF_tri[:, :, :2]
    Z = -VF_tri[:, :, 2].mean(axis=1)
    zmin, zmax = Z.min(), Z.max()
    Z = (Z - zmin) / (zmax - zmin)
    return T, Z

def prepare_color(C, model, light_dir):
    C = C @ model[:3, :3].T
    C = (C @ light_dir)[:, np.newaxis].repeat(3, axis=-1)
    C = np.clip(C, 0, 1)
    #C = C * 0.6 + 0.3
    return C

def process_mesh(V, F, MVP, norm, model, light_dir, linewidth, c_map, diff=None):
    T, Z = transform_and_project(V, F, MVP, norm)
    C = calc_face_norm(V, F)
    C = prepare_color(C, model, light_dir)
    I = np.argsort(Z)
    T, C = T[I, :], C[I, :]
    C = np.clip(C, 0, 1)
    
    NI = np.argwhere(C[:, 2] > 0).squeeze()
    T, C = T[NI, :], C[NI, :]
    if diff is not None:
        diff = diff[I]
        diff = diff[NI]
        Dc = plt.get_cmap(c_map)(diff)
        mask = diff[:, np.newaxis]
        C = C * (1 - mask) + Dc[:, :3] * mask
        C = np.clip(C, 0, 1)
    C = C * 0.7 + 0.2
    return T, C

def plot_image_array_diff4(
    Vs, 
    Fs, 
    D, 
    rot=(0, 0, 0), 
    size=6, 
    norm=False, 
    linewidth=1, 
    linestyle='solid', 
    light_dir=np.array([0,0,1]), 
    bg_black=True, 
    threshold=None, 
    logdir='.', 
    name='000', 
    save=False, 
    draw_base=True, 
    c_map='YlOrRd'
    ):
    num_meshes = len(Vs) + 1
    fig, axes = setup_plot(bg_black, size, num_meshes)
    
    xrot, yrot, zrot = rot if rot is not None else (0, 0, 0)
    
    model = translate(0, 0, -5) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
    proj = ortho(-1, 1, -1, 1, 1, 100)
    # proj  = perspective(65, 1, 1, 10)
    MVP = proj @ model

    if draw_base:
        T, C = process_mesh(D, Fs[0], MVP, norm, model, light_dir, linewidth, c_map)
        collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        axes[0].add_collection(collection)
        axes[0].axis('off')
        
    D_diff = np.array(abs(D - np.array(Vs)))[:, Fs[0]]
    D_diff = np.linalg.norm(D_diff, axis=-1)
    D_diff = np.linalg.norm(D_diff, axis=-1)
    
    if threshold is not None:
        D_diff[D_diff > threshold] = 0
    diff_min, diff_max = D_diff.min(), D_diff.max()

    for idx, (V, F) in enumerate(zip(Vs, Fs)):
        diff = D_diff[idx]
        if diff_max > 0:
            diff = (diff - diff_min) / (diff_max - diff_min)
        diff[diff>0.02]=1
        
        T, C = process_mesh(V, F, MVP, norm, model, light_dir, linewidth, c_map, diff)
        collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        axes[idx + 1].add_collection(collection)
        axes[idx + 1].set_xlabel(f'min:{D_diff[idx].min():.5f} | max: {D_diff[idx].max():.5f}')
        axes[idx + 1].axis('off')
        plt.xlabel(f'min:{D_diff[idx].min():.5f} | max: {D_diff[idx].max():.5f}')

    if save:
        plt.savefig(f'{logdir}/{name}.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_image_array_diff3(
    Vs, 
    Fs, 
    D, 
    rot=(0, 0, 0), 
    size=6, 
    norm=False, 
    linewidth=1, 
    linestyle='solid', 
    light_dir=np.array([0,0,1]), 
    bg_black=True, 
    threshold=None, 
    logdir='.', 
    name='000', 
    save=False, 
    draw_base=True, 
    c_map='YlOrRd'
    ):
    num_meshes = len(Vs) + 1
    fig, axes = setup_plot(bg_black, size, num_meshes)
    
    xrot, yrot, zrot = rot if rot is not None else (0, 0, 0)
    
    model = translate(0, 0, -5) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
    proj = ortho(-1, 1, -1, 1, 1, 100)
    # proj  = perspective(65, 1, 1, 10)
    MVP = proj @ model

    if draw_base:
        T, C = process_mesh(D, Fs[0], MVP, norm, model, light_dir, linewidth, c_map)
        collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        axes[0].add_collection(collection)
        axes[0].axis('off')
        
#     D_diff = np.array(np.square(D - np.array(Vs)))[:, Fs[0]]
    D_diff = np.array(abs(D - np.array(Vs)))[:, Fs[0]]
    D_diff = np.linalg.norm(D_diff, axis=-1)
    D_diff = np.linalg.norm(D_diff, axis=-1)
    
    if threshold is not None:
        D_diff[D_diff > threshold] = threshold
    diff_min, diff_max = D_diff.min(), D_diff.max()
    
    norm_color = plt.Normalize(vmin=0, vmax=diff_max)
    sm = plt.cm.ScalarMappable(cmap=c_map, norm=norm_color)
    sm.set_array([])

    for idx, (V, F) in enumerate(zip(Vs, Fs)):
        diff = D_diff[idx]
        if diff_max > 0:
            diff = (diff - diff_min) / (diff_max - diff_min)
        
        T, C = process_mesh(V, F, MVP, norm, model, light_dir, linewidth, c_map, diff)
        collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        axes[idx + 1].add_collection(collection)
#         axes[idx + 1].set_xlabel(f'min:{D_diff[idx].min():.5f} | max: {D_diff[idx].max():.5f}')
#         axes[idx + 1].set_title(f'min:{D_diff[idx].min():.2e} | mean: {D_diff[idx].mean():.2e} | max: {D_diff[idx].max():.2e}')
        axes[idx + 1].set_title(f'mean: {D_diff[idx].mean():.2e} | std: {D_diff[idx].std():.2e}')
        axes[idx + 1].axis('off')
        #plt.xlabel(f'min:{D_diff[idx].min():.5f} | max: {D_diff[idx].max():.5f}')

    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.006, pad=0.015)
    cbar.set_label('Mean Squared Error', rotation=90, labelpad=15)
    
    if save:
        plt.savefig(f'{logdir}/{name}.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()
        
def plot_image_array_col(Vs, 
                     Fs, 
                     Cs,
                     rot_list=None, 
                     size=6, 
                     norm=False, 
                     mode='mesh', 
                     linewidth=1, 
                     linestyle='solid', 
                     light_dir=np.array([0,0,1]),
                     bg_black = True,
                     logdir='.', 
                     name='000', 
                     save=False,
                    blend=0.5,
                    ):
    num_meshes = len(Vs)
    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    fig = plt.figure(figsize=(size * num_meshes, size))  # Adjust figure size based on the number of meshes
    
    for idx, (V, F, col) in enumerate(zip(Vs, Fs, Cs)):
        S = col.numpy() if type(col)==torch.Tensor else col
        # Calculate the position of the subplot for the current mesh
        ax_pos = [idx / num_meshes, 0, 1 / num_meshes, 1]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
   
        #xrot, yrot, zrot = rot[0], 90, rot[2]
        if rot_list:
            xrot, yrot, zrot = rot_list[idx]
        else:
            xrot, yrot, zrot = 0,0,0
        ## MVP
        # model = translate(0, 0, -3) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        # proj  = perspective(55, 1, 1, 10)
        model = translate(0, 0, -5) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        proj  = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
        MVP   = proj @ model # view is identity

        VF_tri = transform_vertices(V, MVP, F, norm)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)

        C = calc_face_norm(V, F) @ model[:3,:3].T
        S = S[F]
        
        I = np.argsort(Z)
        T, C = T[I, :], C[I, :]
        S = S[I, :]

        NI = np.argwhere(C[:,2] > 0).squeeze()
        T, C = T[NI, :], C[NI, :]
        S = S[NI, :]

        C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)            
        C = np.clip(C, 0, 1)

        #C = C*0.5+0.25
        C = C*0.7+0.15
        #Sc = torch.from_numpy(S).mode(-1).values.numpy()
        
        Sc = S.mean(1)#.numpy()
        #Sc = Sc[...,:3]

        if blend >= 1.0:
            print('using given color')
            C = Sc
        else:
            C = (C*(1-blend) + Sc*blend)
            C = np.clip(C, 0, 1)
        
        collection = PolyCollection(T, closed=False, linewidth=linewidth,facecolor=C, edgecolor=C)
        
        ax.add_collection(collection)
        plt.xticks([])
        plt.yticks([])
        idx = idx + 1
    
    if save:
        plt.savefig('{}/{}.png'.format(logdir, name), bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_image_array_seg(Vs, 
                     Fs, 
                     Cs,
                     rot_list=None, 
                     size=6, 
                     norm=False, 
                     mode='mesh', 
                     linewidth=1, 
                     linestyle='solid', 
                     light_dir=np.array([0,0,1]),
                     bg_black = True,
                     c_map="nipy_spectral",
                     logdir='.', 
                     name='000', 
                     save=False,
                     draw_base=True,
                     blend = 0.4,
                    ):
    num_meshes = len(Vs)
    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    fig = plt.figure(figsize=(size * num_meshes, size))  # Adjust figure size based on the number of meshes
    
    #xrot, yrot, zrot = rot[0], 90, rot[2]
#     if rot_list:
#         xrot, yrot, zrot = rot_list[0]
#     else:
#         xrot, yrot, zrot = 0,0,0
#     ## MVP
#     model = translate(0, 0, -6) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
#     # proj  = perspective(30, 1, 1, 100)
#     proj  = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
#     MVP   = proj @ model # view is identity
    
        
    for idx, (V, F, seg) in enumerate(zip(Vs, Fs, Cs)):
        
        len_seg = seg.shape[-1]
        #S = seg.argmax(-1)
        #print(seg.argmax(-1).shape)
        S = softmax(seg) # softmax
        S = S.argmax(-1).float()
                
        # Calculate the position of the subplot for the current mesh
        ax_pos = [idx / num_meshes, 0, 1 / num_meshes, 1]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
   
        #xrot, yrot, zrot = rot[0], 90, rot[2]
        if rot_list:
            xrot, yrot, zrot = rot_list[idx]
        else:
            xrot, yrot, zrot = 0,0,0
        ## MVP
        # model = translate(0, 0, -3) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        # proj  = perspective(55, 1, 1, 10)
        model = translate(0, 0, -5) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        proj  = ortho(-1, 1, -1, 1, 1, 100) # Use ortho instead of perspective
        MVP   = proj @ model # view is identity
        VF_tri = transform_vertices(V, MVP, F, norm)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)

        C = calc_face_norm(V, F) @ model[:3,:3].T
        S = S[F]
        
        I = np.argsort(Z)
        T, C = T[I, :], C[I, :]
        S = S[I, :]

        NI = np.argwhere(C[:,2] > 0).squeeze()
        T, C = T[NI, :], C[NI, :]
        S = S[NI, :]

        C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)            
        C = np.clip(C, 0, 1)

        #C = C*0.5+0.25
        C = C*0.7+0.15

        #S = S.mode(-1).values.numpy()
        #S = S.max(-1).values.numpy()
        S = S.mean(-1)
#         S = S.max(-1)
        
        Sc = plt.get_cmap(c_map)(S/len_seg)
        Sc = Sc[...,:3]
        
        
        C = (C*(1-blend) + Sc*blend)
        C = np.clip(C, 0, 1)
        
        collection = PolyCollection(T, closed=False, linewidth=linewidth,facecolor=C, edgecolor=C)
        
        ax.add_collection(collection)
        plt.xticks([])
        plt.yticks([])
        idx = idx + 1
    
    if save:
        plt.savefig('{}/{}.png'.format(logdir, name), bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_points_image(
        Vs, 
        logdir='.', 
        name='train_000', 
        rot_list=None, 
        size=3, 
        norm=False, 
        save=True
    ): 
    num_meshes = len(Vs)
    fig = plt.figure(figsize=(size, size * num_meshes + 1))  # Adjust figure size based on the number of meshes

    for idx, V in enumerate(Vs):
        
        ax_pos = [0, idx / num_meshes, 1, 1 / num_meshes]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=True)

        if rot_list:
            xrot, yrot, zrot = rot_list[idx]
        else:
            xrot, yrot, zrot = 0,0,0
        ## MVP
        model = translate(0, 0, -2) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        proj  = perspective(55, 1, 1, 100)
        # proj  = ortho(-1, 1, -1, 1, 1,100)
        MVP   = proj @ model # view is identity
        
        V = np.c_[V, np.ones(len(V))]
        V = V @ MVP.T
        V /= V[:, 3].reshape(-1, 1)
        
        T = V[:, :2]
        T = T.reshape(-1, 2)
        for t, c_ in zip(T, ['r','g','b']):
            ax.scatter(t[0], t[1], c = c_, marker='.')
    plt.show()
    plt.close()
        
def plot_points_image_array_LDM(
        Vs, 
        LDM, 
        logdir='.', 
        name='train_000', 
        rot_list=None, 
        size=3, 
        norm=False, 
        save=True
    ): 
    num_meshes = len(Vs)
    fig = plt.figure(figsize=(size, size * num_meshes + 1))  # Adjust figure size based on the number of meshes

    for idx, (V, LDM) in enumerate(zip(Vs, LDM)):
        
        ax_pos = [0, idx / num_meshes, 1, 1 / num_meshes]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=True)

        if rot_list:
            xrot, yrot, zrot = rot_list[idx]
        else:
            xrot, yrot, zrot = 0,0,0
        ## MVP
        model = translate(0, 0, -2) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        proj  = perspective(55, 1, 1, 100)
        # proj  = ortho(-1, 1, -1, 1, 1,100)
        MVP   = proj @ model # view is identity
        
        V = np.c_[V, np.ones(len(V))]
        V = V @ MVP.T
        V /= V[:, 3].reshape(-1, 1)
        
        T = V[:, :2]
        T = T.reshape(-1, 2)
    
        ax.scatter(T[:, 0], T[:,1], c = 'b', marker='.')
        ax.scatter(T[LDM,0], T[LDM,1], c = 'r', marker='.')
    if save:
        plt.savefig('{}/{}.png'.format(logdir, name), bbox_inches = 'tight')
    else:
        plt.show()
    plt.close()

def plot_points_image_array_seg(
        Vs, 
        segs, 
        logdir='.', 
        name='train_000', 
        rot_list=None, 
        size=3, 
        norm=False, 
        save=True
    ): 
    num_meshes = len(Vs)
    fig = plt.figure(figsize=(size, size * num_meshes + 1))  # Adjust figure size based on the number of meshes

    
    for idx, (V, seg) in enumerate(zip(Vs, segs)):
        
        min_seg = seg.min()
        len_seg = seg.max() - seg.min()
        
        ax_pos = [0, idx / num_meshes, 1, 1 / num_meshes]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=True)

        if rot_list:
            xrot, yrot, zrot = rot_list[idx]
        else:
            xrot, yrot, zrot = 0,0,0
        ## MVP
        model = translate(0, 0, -2) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        proj  = perspective(55, 1, 1, 100)
        # proj  = ortho(-1, 1, -1, 1, 1,100)
        MVP   = proj @ model # view is identity
        
        V = np.c_[V, np.ones(len(V))]
        V = V @ MVP.T
        V /= V[:, 3].reshape(-1, 1)
        
        T = V[:, :2]
        T = T.reshape(-1, 2)
        

        Z = -V[:, 2]
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)
        
        I = np.argsort(Z)
                
        C = plt.get_cmap("nipy_spectral")((seg-min_seg)/len_seg)
        # C = plt.get_cmap("hsv")(seg)
        T, C = T[I, :], C[I, :]        
        
        ax.scatter(T[:, 0], T[:,1], c=C, marker='.')
    if save:
        plt.savefig('{}/{}.png'.format(logdir, name), bbox_inches = 'tight')
    else:
        plt.show()
    plt.close()
    
def plot_points_image_array_VC(
        Vs, 
        Cs, 
        logdir='.', 
        name='train_000', 
        rot_list=None, 
        size=3, 
        norm=False, 
        save=True
    ): 
    num_meshes = len(Vs)
    fig = plt.figure(figsize=(size, size * num_meshes + 1))  # Adjust figure size based on the number of meshes

    
    for idx, (V, C) in enumerate(zip(Vs, Cs)):
        
        ax_pos = [0, idx / num_meshes, 1, 1 / num_meshes]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=True)

        if rot_list:
            xrot, yrot, zrot = rot_list[idx]
        else:
            xrot, yrot, zrot = 0,0,0
        ## MVP
        model = translate(0, 0, -2) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        proj  = perspective(55, 1, 1, 100)
        # proj  = ortho(-1, 1, -1, 1, 1,100)
        MVP   = proj @ model # view is identity
        
        V = np.c_[V, np.ones(len(V))]
        V = V @ MVP.T
        V /= V[:, 3].reshape(-1, 1)
        
        T = V[:, :2]
        T = T.reshape(-1, 2)        

        Z = -V[:, 2]
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)
        
        I = np.argsort(Z)
        T, C = T[I, :], C[I, :]        
        
        ax.scatter(T[:, 0], T[:,1], c=C, marker='.')
    plt.show()
    plt.close()
    
def plot_points_image_array(
        Vs, 
        segments, 
        logdir='.', 
        name='train_000', 
        rot_list=None, 
        size=3, 
        norm=False, 
        save=True
    ): 
    num_meshes = len(Vs)
    fig = plt.figure(figsize=(size, size * num_meshes + 1))  # Adjust figure size based on the number of meshes
    
    for idx, (V, seg) in enumerate(zip(Vs, segments)):
        V = V.detach().cpu().numpy()
        seg = seg.detach().cpu().numpy()
        
        label = np.argmax(seg, axis=-1).reshape(-1, 1)
        ## Calculate the position of the subplot for the current mesh # (left, bottom, width, height)
        # ax_pos = [idx / num_meshes, 0, 1 / num_meshes, 1]
        ax_pos = [0, idx / num_meshes, 1, 1 / num_meshes]
        ax = fig.add_axes(ax_pos, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=True)

        #xrot, yrot, zrot = rot[0], 90, rot[2]
        if rot_list:
            xrot, yrot, zrot = rot_list[idx]
        else:
            xrot, yrot, zrot = 0,0,0
        ## MVP
        model = translate(0, 0, -2) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
        proj  = perspective(55, 1, 1, 100)
        # proj  = ortho(-1, 1, -1, 1, 1,100)
        MVP   = proj @ model # view is identity
        
        V = np.c_[V, np.ones(len(V))]
        V = V @ MVP.T
        V /= V[:, 3].reshape(-1, 1)
        
        Z = -V[:, :2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)
        
        T = V[:, :2]
        T = T.reshape(-1, 2)
        C = plt.get_cmap("hsv")(label/24)
        # I = np.argsort(Z)
        # T, C = T[I, :], C[I, :]
    
        ax.scatter(T[:, 0], T[:,1], c = C, marker='.')
    if save:
        plt.savefig('{}/{}.png'.format(logdir, name), bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()
        plt.close()

def render_wo_audio(#basedir="tmp",
                   Vs, F, Ds=None,
                   savedir="tmp",
                   savename="temp",
                   figsize=(3,3),
                   fps=30,
                   size=4,
                   y_rot=0,
                   light_dir=np.array([0,0,1]),
                   mode='mesh', 
                   linewidth=1,
                   save=True,
                   bg_black=False,
                  ):
    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    # make dirs
    os.makedirs(savedir, exist_ok=True)
        
    num_meshes = len(Vs)
    size = size
    figsize = (size, size)
    
    ## visualize
    fig = plt.figure(figsize=figsize)
    _r = figsize[0] / figsize[1]
    fig_xlim = [-_r, _r]
    fig_ylim = [-1, +1]
    ax = fig.add_axes([0,0,1,1], xlim=fig_xlim, ylim=fig_ylim, aspect=1, frameon=False)
    

    ## MVP
    model = translate(0, 0, -2) @ yrotate(y_rot)
#     proj  = perspective(55, 1, 1, 10)
    proj  = ortho(-1, 1, -1, 1, 1,100)
    MVP   = proj @ model # view is identity
    
    global D_idx
    D_idx = 0
    def render_mesh(ax, V, MVP, F):
        global D_idx
        # quad to triangle    
        VF_tri = transform_vertices(V, MVP, F, norm=False)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)
                
        if mode=='shade':
            C = calc_face_norm(V, F) @ model[:3,:3].T
            I = np.argsort(Z) # -----------------------> depth sorting
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] > 0)[:,0] # --> culling w/ normal
            T, C = T[NI, :], C[NI, :]
            
            C = np.clip((C @ light_dir), 0, 1) # ------> cliping range 0 - 1
            C = C[:,np.newaxis].repeat(3, axis=-1)
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        elif mode=='diff':
            D = Ds[D_idx]
            diff = np.array(abs(D - V))
            diff = diff[F] # N 3 3
            diff = np.linalg.norm(diff, axis=1) # N 3
            diff = np.linalg.norm(diff, axis=1) # N

            diff_min, diff_max = diff.min(), diff.max()
            if diff_max > 0:
                diff = (diff - diff_min) / (diff_max - diff_min)    

            C = calc_face_norm(V, F) @ model[:3,:3].T

            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]
            diff = diff[I]

            NI = np.argwhere(C[:,2] > 0)[:,0]
            T, C = T[NI, :], C[NI, :]
            diff = diff[NI]

            C = (C @ light_dir)[:,np.newaxis].repeat(3, axis=-1)
            C = np.clip(C, 0, 1)

            #C = C*0.5+0.25
            C = C*0.6+0.15

            Dc = plt.get_cmap("YlOrRd")(diff) ## [N, 4]
            mask = diff[:,np.newaxis]
            C = C*(1-mask)+ Dc[:,:3]*(mask)
            C = np.clip(C, 0, 1)
            collection = PolyCollection(T, closed=False, linewidth=linewidth,facecolor=C, edgecolor=C)
            D_idx =D_idx + 1
        else:
            C = plt.get_cmap("gray")(Z)
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]
            
            NI = np.argwhere(C[:,2] > 0)[:,0]
            T, C = T[NI, :], C[NI, :]
            collection = PolyCollection(T, closed=False, linewidth=0.23, facecolor=C, edgecolor="black")
        ax.add_collection(collection)
    
    def update(V):
        # Cleanup previous collections
        for coll in ax.collections:
            coll.remove()

        # Render meshes for all views
        render_mesh(ax, V, MVP, F)
        
        return ax.collections
    
    plt.tight_layout()
    #tqdm(mesh_vtxs, desc="rnd", ncols=60)
    anim = FuncAnimation(fig, update, frames=Vs, blit=True)
    if save:
        bar = tqdm(total=num_meshes, desc="rendering")
        anim.save(
            f'{savedir}/{savename}.mp4', 
            fps=fps,
            progress_callback=lambda i, n: bar.update(1)
        )
        print(f"saved as: {savedir}/{savename}.mp4")
    else:
        return anim

def render_w_audio(#basedir="tmp",
                   Vs, F,
                   savedir="tmp",
                   savename="temp",
                   audio_fn="tmp",
                   figsize=(3,3),
                   fps=30,
                   y_rot=0,
                   light_dir=np.array([0,0,1]),
                   mode='mesh', 
                   linewidth=1,
                   bg_black=False,
                  ):
    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    # make dirs
    os.makedirs(savedir, exist_ok=True)
        
    num_meshes = len(Vs)
    print(num_meshes)
    size = 4
    
    ## visualize
    fig = plt.figure(figsize=figsize)
    _r = figsize[0] / figsize[1]
    fig_xlim = [-_r, _r]
    fig_ylim = [-1, +1]
    ax = fig.add_axes([0,0,1,1], xlim=fig_xlim, ylim=fig_ylim, aspect=1, frameon=False)
    

    ## MVP
    model = translate(0, 0, -5) @ yrotate(y_rot)
    proj  = ortho(-1, 1, -1, 1, 1,100)
    # proj  = perspective(55, 1, 1, 10)
    # proj  = perspective(45, 1, 1, 100)
    MVP   = proj @ model # view is identity

    def render_mesh(ax, V, MVP, F):        
        # quad to triangle    
        VF_tri = transform_vertices(V, MVP, F, norm=False)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)
        
        if mode=='shade':
            C = calc_face_norm(V, F) @ model[:3,:3].T
            I = np.argsort(Z) # -----------------------> depth sorting
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] > 0)[:,0] # --> culling w/ normal
            T, C = T[NI, :], C[NI, :]
            
            C = np.clip((C @ light_dir), 0, 1) # ------> cliping range 0 - 1
            C = C[:,np.newaxis].repeat(3, axis=-1)
            C = np.clip(C, 0, 1)
            #C = C*0.5+0.25
            C = C*0.6+0.15
            
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        else:
            C = plt.get_cmap("gray")(Z)
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]
            
            NI = np.argwhere(C[:,2] > 0)[:,0]
            T, C = T[NI, :], C[NI, :]
            collection = PolyCollection(T, closed=False, linewidth=0.23, facecolor=C, edgecolor="black")
        ax.add_collection(collection)
    
    def update(V):
        # Cleanup previous collections
        for coll in ax.collections:
            coll.remove()

        # Render meshes for all views
        render_mesh(ax, V, MVP, F)
        return ax.collections
    
    plt.tight_layout()
    anim = FuncAnimation(fig, update, frames=Vs, blit=True)
    
    bar = tqdm(total=num_meshes, desc="rendering")
    anim.save(
        f'{savedir}/tmp2.mp4', 
        fps=fps,
        progress_callback=lambda i, n: bar.update(1)
    )
    plt.close()

    # mux audio and video
    print("[INFO] mux audio and video")
    cmd = f"ffmpeg -y -i {audio_fn} -i {savedir}/tmp2.mp4 -c:v copy -c:a aac {savedir}/{savename}.mp4"
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"saved as: {savedir}/{savename}.mp4")

    # remove tmp files
    subprocess.call(f"rm -f {savedir}/tmp2.mp4", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def update_frame(frame_idx, Vs, Fs, D, axes, linewidth, c_map, norm, light_dir, threshold, rot_list):
    for ax in axes:
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    V = D[frame_idx]
    F = Fs[0]
    
    xrot, yrot, zrot = rot_list[frame_idx] if rot_list is not None else (0, 0, 0)
    
    model = translate(0, 0, -5) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
    # proj  = perspective(65, 1, 1, 10)
    proj = ortho(-1, 1, -1, 1, 1, 100)
    MVP = proj @ model
    
    # Plot the GT mesh
    T, C = process_mesh(V, F, MVP, norm, model, light_dir, linewidth, c_map)
    collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
    axes[0].add_collection(collection)
    axes[0].set_title(f'Frame {frame_idx + 1:03d} * 1e-2', fontsize=6)
    axes[0].axis('off')
    
    #D_diff = np.array([abs(np.array(D[frame_idx]) - np.array(vs[frame_idx])) for vs in Vs])[:, F]
    D_diff = np.array(abs(np.array(D[frame_idx]) - np.array(Vs[:, frame_idx])))[:, F]
    D_diff = np.linalg.norm(D_diff, axis=-1)
    D_diff = np.linalg.norm(D_diff, axis=-1)
    
    if threshold is not None:
        D_diff[D_diff > threshold] = 0
    diff_min, diff_max = D_diff.min(), D_diff.max()
    
    for idx, V in enumerate(Vs):
        diff = D_diff[idx]
        if diff_max > 0:
            diff = (diff - diff_min) / (diff_max - diff_min)
        
        T, C = process_mesh(V[frame_idx], F, MVP, norm, model, light_dir, linewidth, c_map, diff)
        collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        axes[idx + 1].add_collection(collection)
#         axes[idx + 1].set_xlabel(f'Frame {frame_idx + 1} | min:{D_diff.min():.5f} | max: {D_diff.max():.5f}')
        axes[idx + 1].set_title(f'min:{D_diff[idx].min()*100:.3f} | max: {D_diff[idx].max()*100:.3f}', fontsize=6)
        axes[idx + 1].axis('off')
        #plt.xlabel(f'Frame {frame_idx + 1} | min:{D_diff.min():.5f} | max: {D_diff.max():.5f}')
    # Add the text to the figure
    #fig.text(0.5, 0.01, f'min: {diff_min:.4f} | max: {diff_max:.4f}', ha='center', fontsize=12, transform=fig.transFigure)


def render_mesh_diff(Vs, Fs, D, 
                    rot_list=None,
                    size=6,
                    norm=False,
                    linewidth=1,
                    light_dir=np.array([0,0,1]),
                    bg_black=True,
                    threshold=None,
                    c_map='YlOrRd', 
                    savedir=None, 
                    savename="temp",
                    audio_fn=None,
                    fps=30
                    ):
    """
    ex):
    v_list=[ pred_outputs[:frame_num], vertices.numpy()[:frame_num] ]
    f_list=[ ict_full.faces ]
    d_list=[ vertices.numpy()[:frame_num] ]
    render_mesh_diff(
        v_list, 
        f_list, 
        d_list[0],
        size=2, bg_black=False,
        savedir='_tmp',
        savename="temp",
    )
    """
    num_frames = len(D)
    num_meshes = len(Vs) + 1
    fig, axes = setup_plot(bg_black, size, num_meshes)
    Vs = np.array(Vs)
    
    plt.tight_layout()
    #fig.subplots_adjust(top=0.95)  # Adjust the top margin to ensure titles are not cut off
    anim = FuncAnimation(fig, update_frame, frames=num_frames, fargs=(Vs, Fs, D, axes, linewidth, c_map, norm, light_dir, threshold, rot_list), repeat=False)
    
    if savedir is None:
        plt.show()
    else:
        if audio_fn is None:
            anim_name = f'{savedir}/{savename}.mp4'
        else:
            anim_name = f'{savedir}/_tmp_.mp4'
        bar = tqdm(total=num_frames, desc="rendering")
        anim.save(
            anim_name, 
            fps=fps,
            progress_callback=lambda i, n: bar.update(1)
        )
            
    plt.close()
    
    if audio_fn is not None:
        # mux audio and video
        print("[INFO] mux audio and video")
        cmd = f"ffmpeg -y -i {audio_fn} -i {savedir}/_tmp_.mp4 -c:v copy -c:a aac {savedir}/{savename}.mp4"
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"saved as: {savedir}/{savename}.mp4")

        # remove tmp files
        subprocess.call(f"rm -f {savedir}/_tmp_.mp4", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
