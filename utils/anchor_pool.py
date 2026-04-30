"""
anchor_pool.py — Closed-form bind-pose estimation via NFS feature attention pooling.

Universal function: takes any mesh + its NFS seg feature, applies the
ICT-derived (anchor, offset) pair, and returns per-joint bind position.

The same anchor & offset are derived once on ICT mean mesh (precompute_joint_anchors.py)
and reused for every other mesh (training caches, unseen test inputs alike).
"""
from typing import Optional, Union
import torch
import torch.nn.functional as F


def compute_bind_pos(
    verts: torch.Tensor,
    nfs_feat: torch.Tensor,
    anchor: torch.Tensor,
    offset: torch.Tensor,
    temperature: Union[float, torch.Tensor] = 0.1,
    normals: Optional[torch.Tensor] = None,
    offset_frame: str = 'world',
) -> torch.Tensor:
    """
    Args:
        verts:     [V, 3] or [B, V, 3]   target mesh vertex positions
        nfs_feat:  [V, K] or [B, V, K]   per-vertex NFS seg encoder features (K=256 for our setup)
        anchor:    [J, K]                per-joint anchor feature (precomputed on ICT mean)
        offset:    [J, 3]                per-joint residual (precomputed on ICT mean)
        temperature: scalar or [J]       softmax temperature; smaller = sharper
        normals:   [V, 3] or [B, V, 3]   needed only for offset_frame='normal'
        offset_frame:  'world' or 'normal'
            'world':  bind_pos = pool_pos + offset                    (offset interpreted as 3D world vec)
            'normal': bind_pos = pool_pos + |offset|*sign · pool_normal
                      (only the magnitude in normal direction is preserved; tangent dropped)
    Returns:
        bind_pos:  [J, 3] or [B, J, 3]
    """
    has_batch = verts.dim() == 3
    if not has_batch:
        verts = verts.unsqueeze(0)
        nfs_feat = nfs_feat.unsqueeze(0)
        if normals is not None:
            normals = normals.unsqueeze(0)
    B, V, K = nfs_feat.shape
    J = anchor.shape[0]

    feat_n = F.normalize(nfs_feat, dim=-1)                         # [B, V, K]
    anch_n = F.normalize(anchor,   dim=-1)                         # [J, K]
    sim = torch.einsum('bvk,jk->bjv', feat_n, anch_n)              # [B, J, V]

    if isinstance(temperature, torch.Tensor):
        T = temperature.view(1, J, 1).to(sim.device, sim.dtype)
    else:
        T = float(temperature)
    attn = F.softmax(sim / T, dim=-1)                              # [B, J, V]
    pool_pos = torch.einsum('bjv,bvk->bjk', attn, verts)           # [B, J, 3]

    if offset_frame == 'world':
        bind_pos = pool_pos + offset.unsqueeze(0)
    elif offset_frame == 'normal':
        if normals is None:
            raise ValueError('offset_frame="normal" requires `normals`')
        pool_normal = torch.einsum('bjv,bvk->bjk', attn, normals)
        pool_normal = F.normalize(pool_normal, dim=-1)
        # Offset stored as scalar magnitude per joint (signed): along normal direction
        # Convention: positive = outward (along +normal), negative = inward
        offset_mag = offset[:, 0:1] if offset.shape[-1] == 1 else \
                     offset.norm(dim=-1, keepdim=True) * torch.sign(offset[:, 0:1])
        bind_pos = pool_pos + offset_mag.unsqueeze(0) * pool_normal
    else:
        raise ValueError(f'unknown offset_frame: {offset_frame}')

    return bind_pos if has_batch else bind_pos.squeeze(0)


def derive_anchor_offset(
    ict_verts: torch.Tensor,
    ict_feat: torch.Tensor,
    maya_bind: torch.Tensor,
    sigma: torch.Tensor,
):
    """
    One-time derivation on ICT mean mesh.

    For each joint j:
        α[v] ∝ exp(-||V[v] - maya_bind[j]||² / (2 σ_j²))           (σ-weighted spatial Gaussian)
        anchor[j] = Σ α[v] · ict_feat[v]
        pool_pos[j] = Σ α[v] · V[v]
        δ[j]      = maya_bind[j] - pool_pos[j]                      (world-space residual)

    Args:
        ict_verts: [V, 3]
        ict_feat:  [V, K]
        maya_bind: [J, 3]
        sigma:     [J]   per-joint Gaussian radius (from sigma_targets.npy)

    Returns:
        anchor:    [J, K]
        offset:    [J, 3]   (world-space)
        pool_pos:  [J, 3]   (for sanity check)
        alpha_max: [J]      (max attention weight per joint, indicates sharpness)
    """
    V, K = ict_feat.shape
    J = maya_bind.shape[0]
    device = ict_verts.device

    # [J, V] distances²
    d2 = (ict_verts.unsqueeze(0) - maya_bind.unsqueeze(1)).pow(2).sum(dim=-1)   # [J, V]
    sigma_sq = sigma.pow(2).clamp_min(1e-8).view(J, 1)                          # [J, 1]
    logits = -d2 / (2 * sigma_sq)                                                # [J, V]
    alpha = F.softmax(logits, dim=-1)                                            # [J, V]

    anchor   = alpha @ ict_feat                                                  # [J, K]
    pool_pos = alpha @ ict_verts                                                 # [J, 3]
    offset   = maya_bind - pool_pos                                              # [J, 3]
    alpha_max = alpha.max(dim=-1).values                                         # [J]
    return anchor, offset, pool_pos, alpha_max
