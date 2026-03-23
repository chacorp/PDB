"""
models/facial_animation.py
==========================
FacialAnimationModel — inference-time wrapper combining HierarchicalLBS + DiffusionNetEDD.

For training, use model_hlbs and model_edd directly:
    train_hlbs.py          — HLBS-only pre-training
    train_edd.py           — EDD-only training (HLBS frozen)
    train_facial_anim.py   — Stage-based curriculum (HLBS → HLBS+EDD)
"""

import torch
import torch.nn as nn

from models.hierarchical_lbs import HierarchicalLBS
from models.diffusion_edd import DiffusionNetEDD


class FacialAnimationModel(nn.Module):
    """
    Neural Facial Animation: HierarchicalLBS + DiffusionNetEDD.

    Inference-only forward: HLBS → Jacobian features → EDD → final mesh.
    For training use the individual model files above.
    """

    def __init__(
        self,
        rig,                        # RigData from utils.rig_loader.load_rig()
        topo_key: str,              # 'mf', 'biwi', or 'voca'
        faces_np,                   # [F, 3] int32  — mesh faces for Jacobian
        z_dim: int = 256,           # expression code dimension (= hid_dim of HLBS)
        num_layers: int = 4,
        edd_hid_channels: int = 128,
        edd_n_blocks: int = 4,
        edd_k_eig: int = 128,
        device: str = 'cpu',
    ):
        super().__init__()

        self.topo_key = topo_key
        self.register_buffer('faces', torch.tensor(faces_np, dtype=torch.long))

        self.hlbs = HierarchicalLBS(
            rig=rig,
            topology=topo_key,
            in_dim_exp=12,
            hid_dim=z_dim,
            num_layers=num_layers,
            device=device,
        )

        self.edd = DiffusionNetEDD(
            z_dim=z_dim,
            hid_channels=edd_hid_channels,
            n_blocks=edd_n_blocks,
            k_eig=edd_k_eig,
        )

    def register_edd_topology(
        self,
        verts_np,
        faces_np,
        op_cache_dir: str = 'utils/diffusion_ops',
        cuda_device: str = 'cuda:0',
    ) -> None:
        """Register DiffusionNet + Poisson operators. Must call before forward."""
        self.edd.register_topology(
            self.topo_key, verts_np, faces_np,
            op_cache_dir=op_cache_dir,
            cuda_device=cuda_device,
        )

    def set_edd_topology(self, device: torch.device) -> None:
        """Activate operators for this topology."""
        self.edd.set_topology(self.topo_key, device)

    def forward(
        self,
        source_vert: torch.Tensor,   # [B, N, 3]
        deform_vert: torch.Tensor,   # [B, N, 3]
        source_norm: torch.Tensor,   # [B, N, 3]
        deform_norm: torch.Tensor,   # [B, N, 3]
    ):
        """
        Returns:
            pred_deformed : [B, N, 3]
            rigid_v       : [B, N, 3]  HLBS coarse output
            z_exp         : [B, L]     expression latent
        """
        delta     = deform_vert - source_vert
        src_in    = torch.cat([source_vert, source_norm], dim=-1)
        deform_in = torch.cat([delta, deform_norm, src_in], dim=-1)

        rigid_v, z_exp = self.hlbs(source_vert, deform_in, return_z_exp=True)

        from utils.mesh_utils import compute_jacobian_features
        jac_feat      = compute_jacobian_features(rigid_v, source_vert, self.faces)
        disp          = self.edd(jac_feat, z_exp)
        pred_deformed = rigid_v + disp

        return pred_deformed, rigid_v, z_exp

    def reg_loss(self) -> dict:
        """Regularization losses from HierarchicalLBS (delta_W, delta_t)."""
        return self.hlbs.reg_loss()
