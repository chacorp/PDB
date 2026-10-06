import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

def weight_entropy_loss(
    weights: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Entropy regularization loss for normalized nonnegative weights.

    Args:
        weights: Tensor of shape (..., K), where the last dimension is the
            weight dimension. Each row is expected to be nonnegative and
            approximately sum to 1.
        eps: Small constant for numerical stability.
        reduction: One of {"mean", "sum", "none"}.

    Returns:
        Scalar tensor if reduction is "mean" or "sum".
        Otherwise returns per-row entropy of shape weights.shape[:-1].

    Notes:
        Entropy is:
            H(w) = -sum_i w_i log(w_i)

        If you want to encourage *sparser* weights, minimize H(w).
        This function returns positive entropy, so minimizing the returned
        value pushes weights toward lower-entropy, more peaked distributions.
    """
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Invalid reduction: {reduction}")

    # Clamp only for log stability; keep original weights in multiplication
    log_w = torch.log(weights.clamp_min(eps))
    entropy = -(weights * log_w).sum(dim=-1)

    if reduction == "mean":
        return entropy.mean()
    if reduction == "sum":
        return entropy.sum()
    return entropy

def weight_column_consistency_loss(
    vertices_a: torch.Tensor,
    weights_a: torch.Tensor,
    vertices_b: torch.Tensor,
    weights_b: torch.Tensor,
    vertex_areas_a: torch.Tensor = None,
    vertex_areas_b: torch.Tensor = None,
    temperature: float = 0.1,
    min_column_mass: float = 1e-6,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Encourage the same weight-column index to cover the same facial region.

    Args:
        vertices_a: Vertex positions of shape ``(B, N_a, 3)``.
        weights_a: Nonnegative weights of shape ``(B, N_a, K)``.
        vertices_b: Vertex positions of shape ``(B, N_b, 3)``.
        weights_b: Nonnegative weights of shape ``(B, N_b, K)``.
        vertex_areas_a: Optional vertex areas of shape ``(B, N_a)`` or
            ``(B, N_a, 1)``. Uniform areas are used when omitted.
        vertex_areas_b: Optional vertex areas of shape ``(B, N_b)`` or
            ``(B, N_b, 1)``. Uniform areas are used when omitted.
        temperature: Contrastive softmax temperature.
        min_column_mass: Columns whose normalized area-weighted mass is below
            this value on either mesh are excluded.
        eps: Numerical stability constant.

    Returns:
        Scalar symmetric contrastive loss.
    """
    if vertices_a.ndim != 3 or vertices_b.ndim != 3:
        raise ValueError("vertices must have shape (B, N, 3)")
    if weights_a.ndim != 3 or weights_b.ndim != 3:
        raise ValueError("weights must have shape (B, N, K)")
    if vertices_a.shape[0] != vertices_b.shape[0]:
        raise ValueError("the two inputs must have the same batch size")
    if weights_a.shape[0] != vertices_a.shape[0] or weights_b.shape[0] != vertices_b.shape[0]:
        raise ValueError("vertices and weights must have matching batch sizes")
    if weights_a.shape[1] != vertices_a.shape[1] or weights_b.shape[1] != vertices_b.shape[1]:
        raise ValueError("vertices and weights must have matching vertex counts")
    if weights_a.shape[2] != weights_b.shape[2]:
        raise ValueError("the two weight tensors must have the same number of columns")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    def column_centroids(vertices, weights, vertex_areas):
        if vertex_areas is None:
            areas = torch.ones_like(vertices[..., 0])
        else:
            areas = vertex_areas
            if areas.ndim == 3 and areas.shape[-1] == 1:
                areas = areas.squeeze(-1)
            if areas.shape != vertices.shape[:2]:
                raise ValueError("vertex areas must have shape (B, N) or (B, N, 1)")
            areas = areas.to(device=vertices.device, dtype=vertices.dtype)

        areas = areas.clamp_min(0)
        areas = areas / areas.sum(dim=1, keepdim=True).clamp_min(eps)

        mesh_centroid = (areas[..., None] * vertices).sum(dim=1, keepdim=True)
        centered = vertices - mesh_centroid
        rms_scale = torch.sqrt(
            (areas * centered.square().sum(dim=-1)).sum(dim=1, keepdim=True)
            + eps
        )
        normalized_vertices = centered / rms_scale[..., None]

        column_measure = areas[..., None] * weights
        column_mass = column_measure.sum(dim=1)
        centroids = torch.einsum(
            "bnk,bnd->bkd", column_measure, normalized_vertices
        )
        centroids = centroids / column_mass.clamp_min(eps)[..., None]
        return centroids, column_mass

    centroids_a, mass_a = column_centroids(vertices_a, weights_a, vertex_areas_a)
    centroids_b, mass_b = column_centroids(vertices_b, weights_b, vertex_areas_b)
    distance = torch.cdist(centroids_a, centroids_b, p=2).square()

    losses = []
    for batch_idx in range(distance.shape[0]):
        active = (
            (mass_a[batch_idx] > min_column_mass)
            & (mass_b[batch_idx] > min_column_mass)
        )
        active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
        if active_idx.numel() < 2:
            continue

        pair_distance = distance[batch_idx][active_idx][:, active_idx]
        labels = torch.arange(active_idx.numel(), device=distance.device)
        losses.append(F.cross_entropy(-pair_distance / temperature, labels))
        losses.append(F.cross_entropy(-pair_distance.transpose(0, 1) / temperature, labels))

    if not losses:
        return (weights_a.sum() + weights_b.sum()) * 0.0
    return torch.stack(losses).mean()
    
def _distance_loss_v1_hinge(mesh_vertices, cage_vertices, coordinate_weight, tau=0.01):
    """Original tau-margin hinge version: distances within tau are free."""
    dist = torch.sqrt(((mesh_vertices[:, :, None, :] - cage_vertices[:, None, :, :]) ** 2).sum(dim=-1) + 1e-8)
    penalty = torch.relu(dist - tau) ** 2
    loss = (coordinate_weight * penalty).sum(dim=-1).mean()
    return loss


def _distance_loss_v2_buggy(mesh_vertices, cage_vertices, coordinate_weight, tau=0.02):
    """Kept only for the record: crashes whenever N != K due to the stray unsqueeze(-1)."""
    _, C, _ = cage_vertices.shape
    mesh_vertices_expand = mesh_vertices[:, :, None].repeat(1, 1, C, 1)
    cage_vertices_expand = cage_vertices[:, None]
    mesh_vertices_dist = torch.square(mesh_vertices_expand - cage_vertices_expand).sum(dim=-1)
    w = coordinate_weight
    return (mesh_vertices_dist * w.unsqueeze(-1)).mean()


def _distance_loss_v3_percage(mesh_vertices, cage_vertices, coordinate_weight, eps=1e-8):
    """Per-cage normalized mean squared distance, averaged only over cages with nonzero weight mass."""
    squared_dist = (
        mesh_vertices[:, :, None, :] - cage_vertices[:, None, :, :]
    ).square().sum(dim=-1)  # (B, N, K)

    w = coordinate_weight
    weight_sum = w.sum(dim=1)  # (B, K)

    loss_per_cage = (
        (w * squared_dist).sum(dim=1) / weight_sum.clamp_min(eps)
    )  # (B, K)

    valid = weight_sum > eps

    return loss_per_cage[valid].mean()


def _distance_loss_v4_global(mesh_vertices, cage_vertices, coordinate_weight, eps=1e-8):
    """Single global weighted-mean squared distance across the whole (B, N, K) tensor."""
    squared_dist = (
        mesh_vertices[:, :, None, :] - cage_vertices[:, None, :, :]
    ).square().sum(dim=-1)  # (B, N, K)

    w = coordinate_weight
    return (w * squared_dist).sum() / (w.sum() + eps)


def masked_cage_consistency_loss(cage_full, cage_sampled, key_weight, eps=1e-8):
    """
    Cage-consistency MSE, restricted to the control points the sampled input
    actually has evidence for.

    key_weight (B, N, K) maps the K control points (cage vertices) to the N
    sampled mesh vertices used to produce cage_sampled; a control point whose
    column is all-zero was never referenced by any sampled vertex, so its
    predicted position in cage_sampled carries no information from the input
    and shouldn't be forced to match cage_full.

    Args:
        cage_full: (B, K, 3) cage predicted from the full mesh
        cage_sampled: (B, K, 3) cage predicted from the subsampled mesh
        key_weight: (B, N, K) coordinate weights from the subsampled forward pass
    Returns:
        scalar loss: per-sample mean squared error (averaged over xyz, matching
        F.mse_loss units) over used control points, averaged over the batch
    """
    with torch.no_grad():
        used = (key_weight.detach() > 0).any(dim=1).float()  # (B, K)
        count = used.sum(dim=-1).clamp(min=1.0)  # (B,)

    se = (cage_full - cage_sampled).square().mean(dim=-1)  # (B, K), mean over xyz like F.mse_loss
    per_sample = (se * used).sum(dim=-1) / count  # (B,)
    return per_sample.mean()


def distance_loss(mesh_vertices, cage_vertices, coordinate_weight, eps=1e-8, tau=0.02):
    """
    Args:
        mesh_vertices: (B, N, 3)
        cage_vertices: (B, K, 3)
        coordinate_weight: (B, N, K)
    Returns:
        loss

    Dispatches to a specific historical implementation via the DIST_LOSS_VARIANT
    env var (v1 / v2buggy / v3 / v4 / dl2 / dl3), for the convergence sweep in
    tmp_CBD/. Defaults to v4, the implementation active before the sweep.
    """
    variant = os.environ.get("DIST_LOSS_VARIANT", "v4")
    if variant == "v1":
        v1_tau = float(os.environ.get("DIST_LOSS_TAU", "0.01"))
        return _distance_loss_v1_hinge(mesh_vertices, cage_vertices, coordinate_weight, tau=v1_tau)
    elif variant == "v2buggy":
        return _distance_loss_v2_buggy(mesh_vertices, cage_vertices, coordinate_weight, tau=tau)
    elif variant == "v3":
        return _distance_loss_v3_percage(mesh_vertices, cage_vertices, coordinate_weight, eps=eps)
    elif variant == "v4":
        return _distance_loss_v4_global(mesh_vertices, cage_vertices, coordinate_weight, eps=eps)
    elif variant == "dl2":
        return distance_loss2(mesh_vertices, cage_vertices, coordinate_weight, tau=tau)
    elif variant == "dl3":
        return distance_loss3(mesh_vertices, cage_vertices, coordinate_weight, tau=tau)
    else:
        raise ValueError(f"unknown DIST_LOSS_VARIANT: {variant!r}")


def distance_loss2(
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
    B, V, _ = mesh_vertices.shape
    
    _tau = 1 / tau
    _, C, _ = cage_vertices.shape
    
    mesh_vertices_expand = mesh_vertices[:,:,None].repeat(1,1,C,1)
    cage_vertices_expand = cage_vertices[:,None]
    
    mesh_vertices_dist = torch.linalg.norm(mesh_vertices_expand - cage_vertices_expand, dim=-1)
    
    mesh_vertices_max = mesh_vertices_dist.max(-2).values.unsqueeze(1)    
    mesh_vertices_dist = 1 - (mesh_vertices_dist / mesh_vertices_max)
    
    return F.mse_loss(mesh_vertices_dist, coordinate_weight) 

def distance_loss3(
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
    B, C, _ = cage_vertices.shape
    
    sum_coordinate_weight = coordinate_weight.sum(-2, keepdim=True) # (B, 1, K)
    coordinate_weight_  = coordinate_weight / (sum_coordinate_weight + 1e-12)
    K_mesh_vertices = coordinate_weight_.transpose(2,1) @ mesh_vertices

    ####################################################
    # consider only Σw_i > 0, (i=cage vertex index)
    mask = (sum_coordinate_weight > 0).all(-2)
    
    K_mesh_vertices = K_mesh_vertices[mask].reshape(B,-1,3)
    cage_vertices = cage_vertices[mask].reshape(B,-1,3)
    ####################################################
    
    loss = F.mse_loss(K_mesh_vertices.detach(), cage_vertices)
    
    return loss
    
def distance_loss3_(mesh_vertices, cage_vertices, coordinate_weight, tau=0.02, return_e=False):
    """
    # deprecated version
    Args:
        mesh_vertices: (B, N, 3)
        cage_vertices: (B, K, 3)
        coordinate_weight: (B, N, K)
    Returns:
        loss
    """
    _,C,_=cage_vertices.shape
    # import pdb;pdb.set_trace()
    # _denom = coordinate_weight.sum(1, keepdim=True) # (B, 1, K)
    # _denom[_denom<=0]=1
    num_nonzero = torch.count_nonzero(coordinate_weight, dim=-1)
    
    _coordinate_weight = coordinate_weight / num_nonzero.unsqueeze(-1) # (B, N, K)
    # _coordinate_weight = torch.nan_to_num(_coordinate_weight)
    
    # mean_v = torch.zeros_like(cage_vertices)
    # for i in range(cage_vertices.shape[1]):        
    K_mesh_vertices = _coordinate_weight.transpose(2,1) @ mesh_vertices
    
    return F.mse_loss(K_mesh_vertices, cage_vertices)
    
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
    
    loss = torch.where(
        pred < 0, 
        -pred, 
        torch.where(pred > 1, pred - 1, torch.zeros_like(pred))
    ).mean()
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