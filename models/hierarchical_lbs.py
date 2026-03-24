"""
models/hierarchical_lbs.py
===========================
HierarchicalLBS: LBS with predefined skeleton (84 joints) and Maya bind-skin
initialization.

Per-identity adaptation (generalize to unseen identities):
    skin_weight_net : source_vert [B,N,3] -> delta_W [B,N,J]
    bind_pose_net   : source_vert [B,N,3] -> delta_t [B,J,3]

Both networks take the template (neutral) mesh geometry as input, like PSI.
No identity index is needed — unseen identities are handled automatically.

Standard hierarchical LBS formula:
    G_j = T_world_j  @  B_inv_j          # skinning matrix [4,4]
    v'  = sum_j  w_j * (G_j @ [v; 1])[:3]

    T_world[root]  = T_local[root]
    T_world[j]     = T_world[parent[j]]  @  T_local[j]

At rest pose T_local[j] = I → G_j = I → v' = v  (no deformation)

Expression-driven (per frame):
    lbs_exp_z_model : deform_in [B,N,C] -> z_exp [B,1,L]
    lbs_pose_model  : z_exp    [B,1,L] -> local_rot_6d [B,1,J*6]

Regularization losses (output-space):
    L_W_reg = ||delta_W||^2  (weight stays close to Maya init)
    L_t_reg = ||delta_t||^2  (bind pose offset stays small)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.exp_utils import from_6D_to_rotation_matrix_torch
from utils.rig_loader import load_rig, rig_to_logit_W, RigData
from models.encoder import LinearEncoder


class HierarchicalLBS(nn.Module):
    """
    LBS with predefined 84-joint hierarchy, Maya bind-skin init, and
    geometry-driven per-identity adaptation (generalizes to unseen identities).

    Args:
        rig            : RigData from utils.rig_loader.load_rig()
        topology       : which W_init to use ('mf', 'biwi', 'voca')
        in_dim_exp     : input feature dim for expression encoder (default 12)
        hid_dim        : hidden dim for all MLP networks
        num_layers     : number of MLP hidden layers
        device         : torch device string
    """

    def __init__(
        self,
        rig: RigData,
        topology: str = 'mf',
        in_dim_exp: int = 12,
        hid_dim: int = 256,
        num_layers: int = 4,
        device: str = 'cpu',
    ):
        super().__init__()

        self.device_str = device
        J = len(rig.joint_names)
        self.num_joints = J
        self.joint_names = rig.joint_names

        # ── Fixed buffers from Maya export ────────────────────────────────
        self.register_buffer('B_inv',      rig.B_inv.to(device))        # [J, 4, 4]
        self.register_buffer('parent_idx', rig.parent_idx.to(device))   # [J]
        self.register_buffer('bind_pos',   rig.bind_pos.to(device))     # [J, 3]

        # Precompute T_bind_local[j] = B_inv[parent] @ inv(B_inv[j])
        B_inv_t  = rig.B_inv.to(device)                              # [J, 4, 4]
        B_bind   = torch.linalg.inv(B_inv_t)                         # [J, 4, 4]
        parent   = rig.parent_idx.to(device)
        T_bind_local = torch.zeros_like(B_bind)
        for j in range(J):
            p = parent[j].item()
            if p == -1:
                T_bind_local[j] = B_bind[j]
            else:
                T_bind_local[j] = B_inv_t[p] @ B_bind[j]
        self.register_buffer('T_bind_local', T_bind_local)            # [J, 4, 4]

        # ── Base skin weights (logit space, fixed) ─────────────────────────
        W_init = rig.W_init.get(topology)
        if W_init is None:
            raise KeyError(
                f"topology '{topology}' not found in rig.W_init. "
                f"Available: {list(rig.W_init.keys())}"
            )
        N = W_init.shape[0]
        self.N = N

        logit_W_base = rig_to_logit_W(W_init.to(device))             # [N, J]
        self.register_buffer('logit_W_base', logit_W_base)

        # ── Per-identity adaptation networks (geometry-driven) ────────────
        # skin_weight_net: template_v [B,N,3] -> delta_W [B,N,J]
        self.skin_weight_net = LinearEncoder(
            in_dim=3,
            out_dim=J,
            hid_dim=hid_dim,
            num_layers=num_layers,
            out_type='vertices',
        ).to(device)

        # bind_pose_net: template_v [B,N,3] -> delta_t [B,J,3]
        # out_type='global' → mean pool over N → [B,1,J*3], reshape to [B,J,3]
        self.bind_pose_net = LinearEncoder(
            in_dim=3,
            out_dim=J * 3,
            hid_dim=hid_dim,
            num_layers=num_layers,
            out_type='global',
        ).to(device)

        # ── Expression encoder: deform_in [B,N,C] -> z_exp [B,1,L] ──────
        self.lbs_exp_z_model = LinearEncoder(
            in_dim=in_dim_exp,
            out_dim=hid_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            out_type='global',
        ).to(device)

        # ── BoneTransformNet: z_exp [B,1,L] -> rot6d [B,1,J*6] ──────────
        self.lbs_pose_model = LinearEncoder(
            in_dim=hid_dim,
            out_dim=J * 6,
            hid_dim=hid_dim,
            out_type='global',
        ).to(device)

        self._rot6d = from_6D_to_rotation_matrix_torch

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_skinning_weights(self, source_vert: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source_vert : [B, N, 3]
        Returns:
            W : [B, N, J]  rows sum to 1
        """
        B = source_vert.shape[0]
        delta_W = self.skin_weight_net(source_vert)                   # [B, N, J]
        logit_W = self.logit_W_base.unsqueeze(0) + delta_W            # [B, N, J]
        return F.softmax(logit_W, dim=-1), delta_W                    # [B, N, J]

    def _get_adjusted_B_inv(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_t : [B, J, 3]
        Returns:
            B_inv_id : [B, J, 4, 4]
        """
        B, J = delta_t.shape[:2]
        dev, dtype = self.B_inv.device, self.B_inv.dtype

        eye3   = torch.eye(3, device=dev, dtype=dtype).expand(B, J, 3, 3)
        neg_dt = (-delta_t).unsqueeze(-1)                              # [B, J, 3, 1]
        top    = torch.cat([eye3, neg_dt], dim=-1)                     # [B, J, 3, 4]
        bot    = torch.cat([
            torch.zeros(B, J, 1, 3, device=dev, dtype=dtype),
            torch.ones( B, J, 1, 1, device=dev, dtype=dtype),
        ], dim=-1)                                                      # [B, J, 1, 4]
        T_neg = torch.cat([top, bot], dim=-2)                          # [B, J, 4, 4]

        B_inv_base = self.B_inv.unsqueeze(0).expand(B, -1, -1, -1)    # [B, J, 4, 4]
        return torch.bmm(
            T_neg.reshape(B * J, 4, 4),
            B_inv_base.reshape(B * J, 4, 4),
        ).reshape(B, J, 4, 4)

    def _chain_hierarchy(self, T_delta: torch.Tensor) -> torch.Tensor:
        """
        Forward Kinematics:
            T_world[j] = T_world[parent] @ T_bind_local[j] @ T_delta[j]

        Args:
            T_delta : [B, J, 4, 4]
        Returns:
            T_world : [B, J, 4, 4]
        """
        B, J, _, _ = T_delta.shape
        T_bind_local = self.T_bind_local.unsqueeze(0).expand(B, -1, -1, -1)

        T_world_list = [None] * J
        for j in range(J):
            p        = self.parent_idx[j].item()
            combined = torch.bmm(T_bind_local[:, j], T_delta[:, j])   # [B, 4, 4]
            if p == -1:
                T_world_list[j] = combined
            else:
                T_world_list[j] = torch.bmm(T_world_list[p], combined)

        return torch.stack(T_world_list, dim=1)                        # [B, J, 4, 4]

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        source_vert: torch.Tensor,
        deform_in: torch.Tensor,
        return_z_exp: bool = False,
    ):
        """
        Args:
            source_vert  : [B, N, 3]  template/neutral mesh vertices
            deform_in    : [B, N, C]  expression input features
            return_z_exp : if True, return (rigid_v, z_exp)

        Returns:
            rigid_v : [B, N, 3]
            z_exp   : [B, L]  (only if return_z_exp=True)
        """
        B, N, _ = source_vert.shape
        J = self.num_joints
        device = source_vert.device

        # 1. Per-identity skin weights and bind-pose offset from template geometry
        W, delta_W = self._get_skinning_weights(source_vert)           # [B,N,J], [B,N,J]
        delta_t = self.bind_pose_net(source_vert)                      # [B,1,J*3]
        delta_t = delta_t.squeeze(1).reshape(B, J, 3)                 # [B,J,3]

        # 2. Expression latent (per frame)
        z_exp = self.lbs_exp_z_model(deform_in)                        # [B, 1, L]
        z_exp_flat = z_exp.squeeze(1)                                  # [B, L]

        # 3. Local joint rotations (6D → 3x3)
        rot6d = self.lbs_pose_model(z_exp)                             # [B, 1, J*6]
        rot6d = rot6d.reshape(B * J, 6)
        local_R = self._rot6d(rot6d).reshape(B, J, 3, 3)              # [B, J, 3, 3]

        # 4. Build local 4x4 transforms (no in-place ops)
        zeros_col = torch.zeros(B, J, 3, 1, device=device, dtype=source_vert.dtype)
        zeros_row = torch.zeros(B, J, 1, 3, device=device, dtype=source_vert.dtype)
        ones_val  = torch.ones( B, J, 1, 1, device=device, dtype=source_vert.dtype)
        top       = torch.cat([local_R, zeros_col], dim=-1)            # [B, J, 3, 4]
        bot       = torch.cat([zeros_row, ones_val], dim=-1)           # [B, J, 1, 4]
        T_local   = torch.cat([top, bot], dim=-2)                      # [B, J, 4, 4]

        # 5. FK chain → world transforms
        T_world = self._chain_hierarchy(T_local)                       # [B, J, 4, 4]

        # 6. Skinning matrices G = T_world @ B_inv(delta_t)
        B_inv_id = self._get_adjusted_B_inv(delta_t)                   # [B, J, 4, 4]
        G = torch.bmm(
            T_world.reshape(B * J, 4, 4),
            B_inv_id.reshape(B * J, 4, 4),
        ).reshape(B, J, 4, 4)                                          # [B, J, 4, 4]

        # 7. LBS: v' = sum_j w_j * (G_j @ [v;1])[:3]
        v_h = torch.cat([
            source_vert,
            torch.ones(B, N, 1, device=device, dtype=source_vert.dtype),
        ], dim=-1)                                                      # [B, N, 4]
        v_per_joint = torch.einsum('bjkl,bnl->bnjk', G[:, :, :3, :], v_h)  # [B,N,J,3]
        rigid_v     = torch.einsum('bnj,bnjk->bnk', W, v_per_joint)        # [B, N, 3]

        if return_z_exp:
            return rigid_v, z_exp_flat
        return rigid_v

    # ── Regularization ────────────────────────────────────────────────────

    def reg_loss(self, source_vert: torch.Tensor, edges=None) -> dict:
        """
        Compute regularization losses on delta_W, delta_t, and optionally W smoothness.

        Args:
            source_vert : [B, N, 3]  template vertices
            edges       : [E, 2] long tensor of mesh edges (optional).
                          Smoothness = mean over edges of ||W_i - W_j||^2
                          (Dirichlet energy: penalizes adjacent vertices with different weights)
        Returns:
            dict with 'L_W_reg', 'L_t_reg', 'L_W_smooth'
        """
        B, _, _ = source_vert.shape
        J = self.num_joints

        W, delta_W = self._get_skinning_weights(source_vert)           # [B, N, J]
        delta_t = self.bind_pose_net(source_vert).squeeze(1).reshape(B, J, 3)

        L_W_reg    = (delta_W ** 2).mean()
        L_t_reg    = (delta_t ** 2).mean()

        if edges is not None:
            W_i = W[:, edges[:, 0], :]          # [B, E, J]
            W_j = W[:, edges[:, 1], :]          # [B, E, J]
            L_W_smooth = ((W_i - W_j) ** 2).mean()
        else:
            L_W_smooth = source_vert.new_zeros(())

        return {'L_W_reg': L_W_reg, 'L_t_reg': L_t_reg, 'L_W_smooth': L_W_smooth}


# ── Quick sanity check ────────────────────────────────────────────────────
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.rig_loader import load_rig, print_summary

    rig = load_rig('maya_rig', topologies=['mf'])
    print_summary(rig)

    model = HierarchicalLBS(rig, topology='mf', device='cpu')
    print(f"\nHierarchicalLBS: {sum(p.numel() for p in model.parameters()):,} trainable params")

    B, N = 2, 5223
    src = torch.randn(B, N, 3)
    din = torch.randn(B, N, 12)

    out = model(src, din)
    print(f"Input: {src.shape}  Output: {out.shape}")
    assert out.shape == src.shape

    regs = model.reg_loss(src)
    print(f"reg_loss: L_W_reg={regs['L_W_reg'].item():.2e}  L_t_reg={regs['L_t_reg'].item():.2e}")

    out2, z = model(src, din, return_z_exp=True)
    print(f"z_exp: {z.shape}")
    print("OK")
