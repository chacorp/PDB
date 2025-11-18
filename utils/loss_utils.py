import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Loss Functions ---
def distance_loss(mesh_vertices, cage_vertices, coordinate_weight, tau=0.02, return_e=False):
    """
    Args:
        mesh_vertices: (B, N, 3)
        cage_vertices: (B, C, 3)
        coordinate_weight: (B, N, C)
    Returns:
        loss
    """
    _,C,_=cage_vertices.shape
    
    mesh_vertices_expand = mesh_vertices[:,:,None].repeat(1,1,C,1)
    cage_vertices_expand = cage_vertices[:,None]
    
    mesh_vertices_dist = (mesh_vertices_expand - cage_vertices_expand)**2
    
    return (mesh_vertices_dist * coordinate_weight.unsqueeze(-1) ).mean()
    
def mvc_loss(mvc_weights):
    """ penalize MVC with negative values """
    neg_loss = torch.nn.functional.relu(-mvc_weights) ** 2
    return torch.mean(neg_loss)

def p2f_loss(before_v, after_v, normals_before, normals_after):
    """ Point-to-Surface Loss
    Args:
        before_v: (B, N, 3) vertices source mesh
        after_v: (B, N, 3) vertices deformed mesh
        normals_before: (B, N, 3) normals from pca plane in source mesh
        normals_after: (B, N, 3) normals from pca plane in deformed mesh
    Returns
        loss (float)
    """
    
    def distance(verts, norms):
        dists = torch.abs(torch.sum(verts * norms, dim=-1))
        return dists

    before_dist = distance(before_v, normals_before)
    after_dist = distance(after_v, normals_after)
    return F.mse_loss(before_dist, after_dist)

def pca_normal_axis(verts, neighbors_map):
    """
    Args:
        verts (torch.tensor): (B, N, 3) vertices
        neighbors_map (list(int):
    Returns:
        normal_axis: (B, N, 3)
    """
    B, V, _ = verts.shape
    
    normal_axis = torch.zeros_like(verts)
    for i, neighbors_idx in enumerate(neighbors_map):
        if len(neighbors_idx) < 2: continue

        neighborhood = verts[:, neighbors_idx, :]
        centroid = torch.mean(neighborhood, dim=1)
        _, _, V_svd = torch.linalg.svd(neighborhood - centroid.unsqueeze(1))
        normal_axis[:, i, :] = V_svd[:, -1, :]
    return normal_axis

def norm_loss(normals_before, normals_after):
    """ PCA Normal Loss 
    Args:
        before_v: (B, N, 3) vertices source mesh
        after_v: (B, N, 3) vertices deformed mesh
        normals_before: (B, N, 3) normals from pca plane in source mesh
        normals_after: (B, N, 3) normals from pca plane in deformed mesh
    Returns
        loss (float)
    """
    return torch.mean(1.0 - F.cosine_similarity(normals_before, normals_after, dim=-1))

def non_ict_loss(pred):
    """Reference from Neural Face Rigging for Animating and Retargeting Facial Meshes in the Wild [Qin et al. 2023], Eq.(3)
    L_FACS = {   
         -x, x < 0 
          0, 0 <= x < 1
        x-1, x > 1
    }
    Args:
        pred (torch.tensor): predicted expression code
    
    Returns:
        loss
    """
    
    loss = torch.where(pred < 0, -pred, torch.where(pred > 1, pred - 1, torch.zeros_like(pred))).mean()
    return loss

def laplacian_loss(batch, pred_key_weight, dataset, mesh_data_num, device):
    """
    Args:
        batch: data class
        pred_key_weight: coordinate prediction (B,V,C)
        dataset: train dataset
        device: cpu, cuda
    Returns:
        laplacian smoothing loss
    """
    
    if mesh_data_num == 0:
        L = dataset.voca_cotmatrix[batch.id_name].to(device)
    elif mesh_data_num == 1:
        L = dataset.biwi_cotmatrix[batch.id_name].to(device)
    elif mesh_data_num == 2:
        L = dataset.mf_SEN_cotmatrix[batch.id_name].to(device)
    elif mesh_data_num == 3:
        L = dataset.coma_cotmatrix[batch.id_name].to(device)
    elif mesh_data_num == 4:
        L = dataset.mf_ROM_cotmatrix[batch.id_name].to(device)
    elif mesh_data_num == 5:
        L = dataset.ict_cotmatrix[batch.id_name].to(device)
    else:
        raise ValueError(f'no data for {mesh_data_num}')
        
    loss = 0
    for pred_key_w in pred_key_weight:
        pred_key_w_lap = L @ pred_key_w
        loss += pred_key_w_lap.sum(0).pow(2).mean()
        
    return loss