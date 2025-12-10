import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

# --- Loss Functions ---
def distance_loss(
        mesh_vertices,
        cage_vertices,
        coordinate_weight,
        tau=0.02,
        return_e=False
    ):
    """
    Args:
        mesh_vertices: (B, N, 3)
        cage_vertices: (B, K, 3)
        coordinate_weight: (B, N, K)
    Returns:
        loss
    """
    _,C,_=cage_vertices.shape
    
    mesh_vertices_expand = mesh_vertices[:,:,None].repeat(1,1,C,1)
    cage_vertices_expand = cage_vertices[:,None]
    
    mesh_vertices_dist = torch.square(mesh_vertices_expand - cage_vertices_expand)

    #w = torch.softmax((coordinate_weight / tau), dim=-1)
    w = coordinate_weight
    return (mesh_vertices_dist * w.unsqueeze(-1)).mean()

def distance_loss2(mesh_vertices, cage_vertices, coordinate_weight, tau=0.02, return_e=False):
    """
    Args:
        mesh_vertices: (B, N, 3)
        cage_vertices: (B, K, 3)
        coordinate_weight: (B, N, K)
    Returns:
        loss
    """
    B, V, _ = mesh_vertices.shape
    
    _tau = 1 / tau
    _, C, _ = cage_vertices.shape
    
    mesh_vertices_expand = mesh_vertices[:,:,None].repeat(1,1,C,1)
    cage_vertices_expand = cage_vertices[:,None]
    
    mesh_vertices_dist = torch.linalg.norm(mesh_vertices_expand - cage_vertices_expand, dim=-1)
    # mesh_vertices_dist = torch.square(torch.linalg.norm(mesh_vertices_expand - cage_vertices_expand, dim=-1))
    
    # candidate_idx = torch.argmax(mesh_vertices_dist, dim=1)
    
    mesh_vertices_max = mesh_vertices_dist.max(-2).values.unsqueeze(1)    
    mesh_vertices_dist = 1 - (mesh_vertices_dist / mesh_vertices_max)
    
    return F.mse_loss(mesh_vertices_dist, coordinate_weight) 

class DistanceLoss():
    def __init__(self, device='cpu'):
        self.device=device
        # from utils.keys import ict_data_synth, mf_data_split
        
        # self.ict_pkl_dict={}
        # for id_name in ict_data_synth['train']:
        #     with open(f'/data/sihun/pca/ICT/geodesics/{id_name}.pkl', 'rb') as f:
        #         self.ict_pkl_dict[id_name] = pickle.load(f)#.to(device) # (N, N)
        
        # self.mf_pkl_dict={}
        # for id_name in mf_data_split['train']:
        #     with open(f'/data/sihun/pca/multiface_align/geodesics/{id_name}.pkl', 'rb') as f:
        #         self.mf_pkl_dict[id_name] = pickle.load(f)#.to(device) # (N, N)
            
    def distance_loss2(self,
            mesh_vertices,
            cage_vertices,
            coordinate_weight,
            randperm_idx,
            batch,
            tau=0.02,
            return_e=False
        ):
        """
        Args:
            mesh_vertices: (B, N, 3)
            cage_vertices: (B, K, 3)
            coordinate_weight: (B, N, K)
        Returns:
            loss
        """
        B, V, _ = mesh_vertices.shape
        
        _tau = 1 / tau
        _, C, _ = cage_vertices.shape
        
        mesh_vertices_expand = mesh_vertices[:,:,None].repeat(1,1,C,1)
        cage_vertices_expand = cage_vertices[:,None]
        
        mesh_vertices_dist = torch.linalg.norm(mesh_vertices_expand - cage_vertices_expand, dim=-1)
        # mesh_vertices_dist = torch.square(torch.linalg.norm(mesh_vertices_expand - cage_vertices_expand, dim=-1))
        
        mesh_vertices_max = mesh_vertices_dist.max(-2).values.unsqueeze(1)    
        mesh_vertices_dist = 1 - (mesh_vertices_dist / mesh_vertices_max)
        
        # candidate_idx = torch.argmax(mesh_vertices_dist, dim=1)
        # import pdb;pdb.set_trace()
        
        ## TODO: 
        # load pickle
        mesh_data_num = batch.mesh_data.cpu().numpy()
        mesh_data = np.array(['voca', 'biwi', 'multiface_align', 'voca', 'multiface_align', 'ICT'])[mesh_data_num]
        # with torch.no_grad():
        #     if mesh_data=='ICT':
        #         geodesics=self.ict_pkl_dict[batch.id_name]
        #     else:
        #         geodesics=self.mf_pkl_dict[batch.id_name]
        #         geodesics=geodesics[randperm_idx, randperm_idx].to(mesh_vertices_dist.device)

        # import pdb;pdb.set_trace()
        with torch.no_grad():
            with open(f'/data/sihun/pca/{mesh_data}/geodesics/{batch.id_name}.pkl', 'rb') as f:
                geodesics = pickle.load(f)
                geodesics = geodesics[randperm_idx][:, randperm_idx]
                geodesics = geodesics.to(mesh_vertices_dist.device) # (N, N)
            candidate_idx = torch.argmax(mesh_vertices_dist, dim=1)  # (B, M) M << N
            
            c_geodesics = geodesics[candidate_idx]
            c_geodesics = c_geodesics.transpose(2,1)
            max_gdistance = 1.0
            gd_mask = (c_geodesics < max_gdistance) * 1.0
        # gd_mask = gd_mask
        # GD = ~GD * 1.0
        # get closest vertex index
        # get geodesic
        # multiply
        mesh_vertices_dist = mesh_vertices_dist * gd_mask
        
        return F.mse_loss(mesh_vertices_dist, coordinate_weight) 
    
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