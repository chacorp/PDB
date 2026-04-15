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
        freeze_adapt: bool = False,
        use_joint_trans: bool = False,
        smooth_delta_W: int = 0,
        smooth_delta_W_alpha: float = 0.5,
        full_prediction: bool = False,
    ):
        super().__init__()

        self.device_str = device
        self.freeze_adapt = freeze_adapt
        self.use_joint_trans = use_joint_trans
        self.smooth_delta_W_iters = smooth_delta_W
        self.smooth_delta_W_alpha = smooth_delta_W_alpha
        self._mesh_edges = None
        self._mesh_edges_by_N = None
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

        # Multi-topology: disabled by default, enabled via enable_multi_topo()
        self._logit_W_by_N = None

        # ── Per-identity adaptation networks (geometry-driven) ────────────
        # skin_weight_net: [template_v, template_n] [B,N,6] -> delta_W [B,N,J]
        self.skin_weight_net = LinearEncoder(
            in_dim=6,
            out_dim=J,
            hid_dim=hid_dim,
            num_layers=num_layers,
            out_type='vertices',
        ).to(device)

        # bind_pose_net: [template_v, template_n] [B,N,6] -> delta_t [B,J,3]
        # out_type='global' → mean pool over N → [B,1,J*3], reshape to [B,J,3]
        self.bind_pose_net = LinearEncoder(
            in_dim=6,
            out_dim=J * 3,
            hid_dim=hid_dim,
            num_layers=num_layers,
            out_type='global',
        ).to(device)

        # ── Freeze adaptation networks if requested ─────────────────────
        if self.freeze_adapt:
            for p in self.skin_weight_net.parameters():
                p.requires_grad_(False)
            for p in self.bind_pose_net.parameters():
                p.requires_grad_(False)

        # ── Expression encoder: deform_in [B,N,C] -> z_exp [B,1,L] ──────
        self.lbs_exp_z_model = LinearEncoder(
            in_dim=in_dim_exp,
            out_dim=hid_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            out_type='global',
        ).to(device)

        # ── BoneTransformNet: z_exp [B,1,L] -> rot6d [B,1,J*6] (+ trans [B,1,J*3]) ─
        pose_out_dim = J * 9 if use_joint_trans else J * 6
        self.lbs_pose_model = LinearEncoder(
            in_dim=hid_dim,
            out_dim=pose_out_dim,
            hid_dim=hid_dim,
            out_type='global',
        ).to(device)

        self._rot6d = from_6D_to_rotation_matrix_torch

    # ── Multi-topology support ─────────────────────────────────────────────

    def enable_multi_topo(self, rig):
        """Enable multi-topology delta mode. Call after __init__ when using 2+ datasets."""
        self._logit_W_by_N = {}
        dev = self.logit_W_base.device
        for topo_name, W_t in rig.W_init.items():
            n = W_t.shape[0]
            self._logit_W_by_N[n] = rig_to_logit_W(W_t.to(dev))

    # ── Internal helpers ──────────────────────────────────────────────────

    def set_mesh_edges(self, faces):
        """Precompute mesh edges for delta_W smoothing. Caches per vertex count."""
        import numpy as _np
        f_np = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else faces
        if f_np.ndim == 1:
            f_np = f_np.reshape(-1, 3)
        e = _np.concatenate([f_np[:, [0,1]], f_np[:, [1,2]], f_np[:, [0,2]]], axis=0)
        e = _np.sort(e, axis=1)
        e = _np.unique(e, axis=0)
        n_verts = int(e.max()) + 1
        if self._mesh_edges_by_N is None:
            self._mesh_edges_by_N = {}
        self._mesh_edges_by_N[n_verts] = torch.tensor(e, dtype=torch.long, device=self.logit_W_base.device)
        self._mesh_edges = self._mesh_edges_by_N[n_verts]

    def _smooth_delta_W(self, delta_W: torch.Tensor) -> torch.Tensor:
        """Laplacian smoothing on delta_W: (1-α)*self + α*mean(neighbors)."""
        if self._mesh_edges_by_N is None or self.smooth_delta_W_iters <= 0:
            return delta_W
        B, N, J = delta_W.shape
        edges = self._mesh_edges_by_N.get(N)
        if edges is None:
            return delta_W  # No edges for this topology, skip
        alpha = self.smooth_delta_W_alpha
        for _ in range(self.smooth_delta_W_iters):
            src_idx = torch.cat([edges[:, 0], edges[:, 1]], dim=0)
            tgt_idx = torch.cat([edges[:, 1], edges[:, 0]], dim=0)
            neighbor_vals = delta_W[:, src_idx, :]                        # [B, 2E, J]
            neighbor_sum = torch.zeros_like(delta_W)
            neighbor_cnt = torch.zeros(B, N, 1, device=delta_W.device)
            tgt_exp = tgt_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, J)
            neighbor_sum.scatter_add_(1, tgt_exp, neighbor_vals)
            cnt_ones = torch.ones(B, tgt_idx.shape[0], 1, device=delta_W.device)
            neighbor_cnt.scatter_add_(1, tgt_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1), cnt_ones)
            neighbor_mean = neighbor_sum / neighbor_cnt.clamp(min=1)
            delta_W = (1 - alpha) * delta_W + alpha * neighbor_mean
        return delta_W

    def _get_skinning_weights(self, source_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source_feat : [B, N, 6]  (template_v + template_n)
        Returns:
            W : [B, N, J]  rows sum to 1
            delta_W : [B, N, J]
        """
        B, N = source_feat.shape[:2]
        if self.freeze_adapt:
            delta_W = source_feat.new_zeros(B, N, self.num_joints)
        else:
            delta_W = self.skin_weight_net(source_feat)               # [B, N, J]

        if self._logit_W_by_N is not None:
            # Multi-topology: select base by vertex count, fallback to full pred
            base = self._logit_W_by_N.get(N)
            logit_W = base.unsqueeze(0) + delta_W if base is not None else delta_W
        else:
            # Single topology: original behavior
            logit_W = self.logit_W_base.unsqueeze(0) + delta_W        # [B, N, J]

        # Smooth final logit_W (not delta_W)
        logit_W = self._smooth_delta_W(logit_W)

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
        source_normal: torch.Tensor = None,
        return_z_exp: bool = False,
        z_exp_override: torch.Tensor = None,
    ):
        """
        Args:
            source_vert   : [B, N, 3]  template/neutral mesh vertices
            deform_in     : [B, N, C]  expression input features
            source_normal : [B, N, 3]  template/neutral mesh normals
            return_z_exp  : if True, return (rigid_v, z_exp)
            z_exp_override: [B, L] if provided, skip internal encoder and use this

        Returns:
            rigid_v : [B, N, 3]
            z_exp   : [B, L]  (only if return_z_exp=True)
        """
        B, N, _ = source_vert.shape
        J = self.num_joints
        device = source_vert.device

        # 1. Per-identity skin weights and bind-pose offset from template geometry
        source_feat = torch.cat([source_vert, source_normal], dim=-1)  # [B, N, 6]
        W, delta_W = self._get_skinning_weights(source_feat)           # [B,N,J], [B,N,J]
        if self.freeze_adapt:
            delta_t = source_vert.new_zeros(B, J, 3)
        else:
            delta_t = self.bind_pose_net(source_feat)                  # [B,1,J*3]
            delta_t = delta_t.squeeze(1).reshape(B, J, 3)             # [B,J,3]

        # 2. Expression latent (per frame)
        if z_exp_override is not None:
            z_exp_flat = z_exp_override                                # [B, L]
            z_exp = z_exp_flat.unsqueeze(1)                            # [B, 1, L]
        else:
            z_exp = self.lbs_exp_z_model(deform_in)                    # [B, 1, L]
            z_exp_flat = z_exp.squeeze(1)                              # [B, L]
        
        # 3. Local joint rotations (6D → 3x3) and optional translation
        pose_out = self.lbs_pose_model(z_exp)                           # [B, 1, J*(6+3?)]
        pose_out = pose_out.squeeze(1)                                  # [B, J*D]

        rot6d = pose_out[:, :J * 6].reshape(B * J, 6)
        local_R = self._rot6d(rot6d).reshape(B, J, 3, 3)              # [B, J, 3, 3]

        if self.use_joint_trans:
            local_t = pose_out[:, J * 6:].reshape(B, J, 3, 1)         # [B, J, 3, 1]
        else:
            local_t = torch.zeros(B, J, 3, 1, device=device, dtype=source_vert.dtype)

        # 4. Build local 4x4 transforms
        zeros_row = torch.zeros(B, J, 1, 3, device=device, dtype=source_vert.dtype)
        ones_val  = torch.ones( B, J, 1, 1, device=device, dtype=source_vert.dtype)
        top       = torch.cat([local_R, local_t], dim=-1)              # [B, J, 3, 4]
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

    # ── Cross-retargeting ─────────────────────────────────────────────────

    def retarget(
        self,
        src_neu_vert: torch.Tensor,
        src_neu_norm: torch.Tensor,
        src_def_vert: torch.Tensor,
        src_def_norm: torch.Tensor,
        tgt_neu_vert: torch.Tensor,
        tgt_neu_norm: torch.Tensor,
    ):
        """
        Cross-retargeting: apply SOURCE expression to TARGET identity.

        - Expression (joint rotations): extracted from SOURCE deform_in
        - Identity (skin weights W, bind pose delta_t): from TARGET template

        Args:
            src_neu_vert : [B, N_s, 3]  source neutral vertices
            src_neu_norm : [B, N_s, 3]  source neutral normals
            src_def_vert : [B, N_s, 3]  source deformed vertices
            src_def_norm : [B, N_s, 3]  source deformed normals
            tgt_neu_vert : [B, N_t, 3]  target neutral vertices
            tgt_neu_norm : [B, N_t, 3]  target neutral normals

        Returns:
            rigid_v : [B, N_t, 3]  retargeted vertices on target mesh
        """
        B = src_def_vert.shape[0]
        N_t = tgt_neu_vert.shape[1]
        J = self.num_joints
        device = src_def_vert.device

        # ── Expression from SOURCE ──────────────────────────────────────
        delta_src = src_def_vert - src_neu_vert
        src_in    = torch.cat([src_neu_vert, src_neu_norm], dim=-1)
        deform_in = torch.cat([delta_src, src_def_norm, src_in], dim=-1)

        z_exp = self.lbs_exp_z_model(deform_in)                        # [B, 1, L]

        pose_out = self.lbs_pose_model(z_exp).squeeze(1)               # [B, J*D]
        rot6d    = pose_out[:, :J * 6].reshape(B * J, 6)
        local_R  = self._rot6d(rot6d).reshape(B, J, 3, 3)

        if self.use_joint_trans:
            local_t = pose_out[:, J * 6:].reshape(B, J, 3, 1)
        else:
            local_t = torch.zeros(B, J, 3, 1, device=device, dtype=src_def_vert.dtype)

        zeros_row = torch.zeros(B, J, 1, 3, device=device, dtype=src_def_vert.dtype)
        ones_val  = torch.ones( B, J, 1, 1, device=device, dtype=src_def_vert.dtype)
        top       = torch.cat([local_R, local_t], dim=-1)
        bot       = torch.cat([zeros_row, ones_val], dim=-1)
        T_local   = torch.cat([top, bot], dim=-2)                      # [B, J, 4, 4]
        T_world   = self._chain_hierarchy(T_local)                     # [B, J, 4, 4]

        # ── Identity from TARGET ────────────────────────────────────────
        tgt_feat = torch.cat([tgt_neu_vert, tgt_neu_norm], dim=-1)     # [B, N_t, 6]
        W_tgt, _ = self._get_skinning_weights(tgt_feat)                # [B, N_t, J]
        if self.freeze_adapt:
            delta_t_tgt = tgt_neu_vert.new_zeros(B, J, 3)
        else:
            delta_t_tgt = self.bind_pose_net(tgt_feat).squeeze(1).reshape(B, J, 3)

        B_inv_tgt = self._get_adjusted_B_inv(delta_t_tgt)              # [B, J, 4, 4]
        G = torch.bmm(
            T_world.reshape(B * J, 4, 4),
            B_inv_tgt.reshape(B * J, 4, 4),
        ).reshape(B, J, 4, 4)

        # ── LBS on target mesh ──────────────────────────────────────────
        v_h = torch.cat([
            tgt_neu_vert,
            torch.ones(B, N_t, 1, device=device, dtype=tgt_neu_vert.dtype),
        ], dim=-1)
        v_per_joint = torch.einsum('bjkl,bnl->bnjk', G[:, :, :3, :], v_h)
        rigid_v     = torch.einsum('bnj,bnjk->bnk', W_tgt, v_per_joint)

        return rigid_v

    # ── Regularization ────────────────────────────────────────────────────

    def reg_loss(self, source_feat: torch.Tensor, edges=None) -> dict:
        """
        Compute regularization losses on delta_W, delta_t, and optionally W smoothness.

        Args:
            source_feat : [B, N, 6]  template vertices + normals
            edges       : [E, 2] long tensor of mesh edges (optional).
                          Smoothness = mean over edges of ||W_i - W_j||^2
                          (Dirichlet energy: penalizes adjacent vertices with different weights)
        Returns:
            dict with 'L_W_reg', 'L_t_reg', 'L_W_smooth'
        """
        B, _, _ = source_feat.shape
        J = self.num_joints

        W, delta_W = self._get_skinning_weights(source_feat)           # [B, N, J]
        delta_t = self.bind_pose_net(source_feat).squeeze(1).reshape(B, J, 3)

        L_W_reg    = (delta_W ** 2).mean()
        L_t_reg    = (delta_t ** 2).mean()

        if edges is not None:
            W_i = W[:, edges[:, 0], :]          # [B, E, J]
            W_j = W[:, edges[:, 1], :]          # [B, E, J]
            L_W_smooth = ((W_i - W_j) ** 2).mean()
        else:
            L_W_smooth = source_feat.new_zeros(())

        return {'L_W_reg': L_W_reg, 'L_t_reg': L_t_reg, 'L_W_smooth': L_W_smooth}


class HierarchicalLBS_FullPred(nn.Module):
    """
    Full-prediction HierarchicalLBS.

    Unlike HierarchicalLBS which predicts delta_W on top of Maya base,
    this model predicts full skin weights and bind pose directly.
    Maya init is used only as Phase 1 supervision target (annealed away).
    """

    def __init__(
        self,
        rig: RigData,
        topology: str = 'mf',
        in_dim_exp: int = 12,
        hid_dim: int = 256,
        num_layers: int = 4,
        device: str = 'cpu',
        use_joint_trans: bool = False,
        smooth_W: int = 0,
        smooth_W_alpha: float = 0.5,
        dfn_skin: bool = False,
        dfn_bind: bool = False,
        dfn_exp: bool = False,
        nfs_feat_dim: int = 0,
        nfs_concat: bool = False,
        adain_pos_norm: bool = False,
        freeze_bind_pose: bool = False,
    ):
        super().__init__()

        self.device_str = device
        self.use_joint_trans = use_joint_trans
        self.freeze_bind_pose = freeze_bind_pose
        self.dfn_skin = dfn_skin
        self.dfn_bind = dfn_bind
        self.dfn_exp = dfn_exp
        self.nfs_feat_dim = nfs_feat_dim
        self.nfs_concat = nfs_concat
        self.adain_pos_norm = adain_pos_norm
        self.smooth_W_iters = smooth_W
        self.smooth_W_alpha = smooth_W_alpha
        self._mesh_edges = None
        self._mesh_edges_by_N = None

        J = len(rig.joint_names)
        self.num_joints = J
        self.joint_names = rig.joint_names

        # ── Fixed hierarchy buffers ──────────────────────────────────────
        self.register_buffer('parent_idx', rig.parent_idx.to(device))
        self.register_buffer('bind_pos',   rig.bind_pos.to(device))

        # Log hierarchy verification
        _p = rig.parent_idx.numpy()
        _has_child = set(int(_p[j]) for j in range(J) if _p[j] >= 0)
        _leaves = [j for j in range(J) if j not in _has_child]
        print(f"[HLBS] Hierarchy: {J} joints, {len(_leaves)} leaves")
        print(f"  Root(0)→Neck({_p[1]=='0' or _p[1]}): parent={_p[1]}")
        print(f"  Neck(1) children: {[j for j in range(J) if _p[j]==1]}")
        print(f"  Leaf joints: {_leaves[:10]}{'...' if len(_leaves) > 10 else ''}")

        B_inv_t = rig.B_inv.to(device)
        B_bind  = torch.linalg.inv(B_inv_t)
        parent  = rig.parent_idx.to(device)
        T_bind_local = torch.zeros_like(B_bind)
        for j in range(J):
            p = parent[j].item()
            if p == -1:
                T_bind_local[j] = B_bind[j]
            else:
                T_bind_local[j] = B_inv_t[p] @ B_bind[j]
        self.register_buffer('T_bind_local', T_bind_local)
        if freeze_bind_pose:
            self.register_buffer('B_inv_fixed', B_inv_t)  # [J, 4, 4]

        # ── Maya init as supervision target (not structural base) ────────
        # Per-topology init targets: {topology_key: W_target [N, J]}
        self._init_targets = {}
        for topo_key, W_t in rig.W_init.items():
            self._init_targets[topo_key] = W_t.to(device)
        self.register_buffer('bind_pos_target', rig.bind_pos.to(device).clone())

        # mesh_data label → topology key mapping
        self._mesh_data_to_topo = {
            2: 'mf',    # mesh_data == 2 → mf (SEN/ROM)
            5: 'ict',   # mesh_data == 5 → ict
            1: 'biwi',
            0: 'voca',
        }

        # ── Full prediction networks ─────────────────────────────────────
        # nfs_concat mode: concat [pos+norm(6) + seg_feat(256)] = 262, AdaIN on pos+norm(6), 4 layers
        # nfs_feat only mode: seg_feat(256) only, AdaIN on seg_feat(256), 2 layers
        # original mode: pos+norm(6), AdaIN on pos+norm(6), num_layers
        if nfs_feat_dim > 0 and nfs_concat:
            _skin_in = 6 + nfs_feat_dim
            _bind_in = 6 + nfs_feat_dim
            _adain_dim = 6 if adain_pos_norm else None  # None = same as in_dim
            _n_layers = num_layers
        elif nfs_feat_dim > 0:
            _skin_in = nfs_feat_dim
            _bind_in = nfs_feat_dim
            _adain_dim = None
            _n_layers = 2
        else:
            _skin_in = 6
            _bind_in = 6
            _adain_dim = None
            _n_layers = num_layers

        # LayerNorm for seg features (normalize scale across topologies)
        self.nfs_layer_norm = nn.LayerNorm(nfs_feat_dim) if nfs_feat_dim > 0 else None

        if dfn_skin:
            from models.encoder import BaseDiffusionNetEncoder
            self.skin_weight_net = BaseDiffusionNetEncoder(
                in_shape=6, out_shape=J, hid_shape=hid_dim,
                N_block=num_layers, outputs_at='vertices',
                with_grad=True, last_activation=None,
            ).to(device)
        else:
            self.skin_weight_net = LinearEncoder(
                in_dim=_skin_in, out_dim=J, hid_dim=hid_dim,
                num_layers=_n_layers, out_type='vertices',
                adain_in_dim=_adain_dim,
            ).to(device)

        if dfn_bind:
            from models.encoder import BaseDiffusionNetEncoder
            self.bind_pose_net = BaseDiffusionNetEncoder(
                in_shape=6, out_shape=J * 3, hid_shape=hid_dim,
                N_block=num_layers, outputs_at='global_mean',
                with_grad=True, last_activation=None,
            ).to(device)
        else:
            self.bind_pose_net = LinearEncoder(
                in_dim=_bind_in, out_dim=J * 3, hid_dim=hid_dim,
                num_layers=_n_layers, out_type='global',
                adain_in_dim=_adain_dim,
            ).to(device)

        # ── Expression encoder + pose model ─────────────────────────────
        if dfn_exp:
            from models.encoder import BaseDiffusionNetEncoder
            self.lbs_exp_z_model = BaseDiffusionNetEncoder(
                in_shape=in_dim_exp, out_shape=hid_dim, hid_shape=hid_dim,
                N_block=num_layers, outputs_at='global_mean',
                with_grad=True, last_activation=None,
            ).to(device)
        else:
            self.lbs_exp_z_model = LinearEncoder(
                in_dim=in_dim_exp, out_dim=hid_dim, hid_dim=hid_dim,
                num_layers=num_layers, out_type='global',
            ).to(device)

        # Pose model always LinearEncoder (input is z_exp [B, 1, L], no mesh structure)
        pose_out_dim = J * 9 if use_joint_trans else J * 6
        self.lbs_pose_model = LinearEncoder(
            in_dim=hid_dim, out_dim=pose_out_dim, hid_dim=hid_dim,
            out_type='global',
        ).to(device)

        self._rot6d = from_6D_to_rotation_matrix_torch

    # ── DiffusionNet precompute ─────────────────────────────────────────

    def update_dfn_precomputes(self, dfn_info):
        """Update DiffusionNet operators for DiffusionNet modules.
        Call when mesh topology changes (e.g., new identity or new dataset).
        """
        if self.dfn_skin:
            self.skin_weight_net.update_precomputes(dfn_info)
        if self.dfn_bind:
            self.bind_pose_net.update_precomputes(dfn_info)
        if self.dfn_exp:
            self.lbs_exp_z_model.update_precomputes(dfn_info)

    # ── Mesh edges ───────────────────────────────────────────────────────

    def set_mesh_edges(self, faces):
        import numpy as _np
        f_np = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else faces
        if f_np.ndim == 1:
            f_np = f_np.reshape(-1, 3)
        e = _np.concatenate([f_np[:, [0,1]], f_np[:, [1,2]], f_np[:, [0,2]]], axis=0)
        e = _np.sort(e, axis=1)
        e = _np.unique(e, axis=0)
        self._mesh_edges = torch.tensor(e, dtype=torch.long, device=self.parent_idx.device)

    def _smooth_logit_W(self, logit_W):
        if self._mesh_edges is None or self.smooth_W_iters <= 0:
            return logit_W
        edges = self._mesh_edges
        alpha = self.smooth_W_alpha
        B, N, J = logit_W.shape
        for _ in range(self.smooth_W_iters):
            src_idx = torch.cat([edges[:, 0], edges[:, 1]], dim=0)
            tgt_idx = torch.cat([edges[:, 1], edges[:, 0]], dim=0)
            neighbor_vals = logit_W[:, src_idx, :]
            neighbor_sum = torch.zeros_like(logit_W)
            neighbor_cnt = torch.zeros(B, N, 1, device=logit_W.device)
            tgt_exp = tgt_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, J)
            neighbor_sum.scatter_add_(1, tgt_exp, neighbor_vals)
            cnt_ones = torch.ones(B, tgt_idx.shape[0], 1, device=logit_W.device)
            neighbor_cnt.scatter_add_(1, tgt_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1), cnt_ones)
            neighbor_mean = neighbor_sum / neighbor_cnt.clamp(min=1)
            logit_W = (1 - alpha) * logit_W + alpha * neighbor_mean
        return logit_W

    # ── Feature prep helper ───────────────────────────────────────────────

    def _prepare_feat(self, source_vert, source_normal, nfs_feat=None):
        """Prepare skin_input and adain_input based on mode."""
        source_feat = torch.cat([source_vert, source_normal], dim=-1)  # [B, N, 6]
        if nfs_feat is not None:
            nfs_normed = self.nfs_layer_norm(nfs_feat)
            if self.nfs_concat:
                skin_input = torch.cat([source_feat, nfs_normed], dim=-1)  # [B, N, 262]
                _adain = source_feat if self.adain_pos_norm else None
            else:
                skin_input = nfs_normed  # [B, N, 256]
                _adain = None
        else:
            skin_input = source_feat
            _adain = None
        return skin_input, _adain

    # ── Core ─────────────────────────────────────────────────────────────

    def _get_skinning_weights(self, source_feat, adain_input=None):
        if self.dfn_skin:
            logit_W = self.skin_weight_net(source_feat)  # [B, N, J]
        else:
            logit_W = self.skin_weight_net(source_feat, adain_input=adain_input)  # [B, N, J]
        logit_W = self._smooth_logit_W(logit_W)
        return F.softmax(logit_W, dim=-1), logit_W

    def _get_bind_pose(self, source_feat, adain_input=None):
        B = source_feat.shape[0]
        J = self.num_joints
        if self.dfn_bind:
            joint_pos = self.bind_pose_net(source_feat).squeeze(1).reshape(B, J, 3)
        else:
            joint_pos = self.bind_pose_net(source_feat, adain_input=adain_input).squeeze(1).reshape(B, J, 3)

        dev, dtype = self.parent_idx.device, source_feat.dtype
        eye3    = torch.eye(3, device=dev, dtype=dtype).expand(B, J, 3, 3)
        neg_jp  = (-joint_pos).unsqueeze(-1)
        top     = torch.cat([eye3, neg_jp], dim=-1)
        bot     = torch.cat([
            torch.zeros(B, J, 1, 3, device=dev, dtype=dtype),
            torch.ones( B, J, 1, 1, device=dev, dtype=dtype),
        ], dim=-1)
        B_inv_id = torch.cat([top, bot], dim=-2)
        return B_inv_id, joint_pos

    def _chain_hierarchy(self, T_delta):
        B, J, _, _ = T_delta.shape
        T_bind_local = self.T_bind_local.unsqueeze(0).expand(B, -1, -1, -1)
        T_world_list = [None] * J
        for j in range(J):
            p = self.parent_idx[j].item()
            combined = torch.bmm(T_bind_local[:, j], T_delta[:, j])
            if p == -1:
                T_world_list[j] = combined
            else:
                T_world_list[j] = torch.bmm(T_world_list[p], combined)
        return torch.stack(T_world_list, dim=1)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, source_vert, deform_in, source_normal=None, return_z_exp=False, z_exp_override=None, nfs_feat=None):
        B, N, _ = source_vert.shape
        J = self.num_joints
        device = source_vert.device

        skin_input, _adain = self._prepare_feat(source_vert, source_normal, nfs_feat)
        W, _ = self._get_skinning_weights(skin_input, adain_input=_adain)
        if self.freeze_bind_pose:
            B_inv_id = self.B_inv_fixed.unsqueeze(0).expand(B, -1, -1, -1)  # [B, J, 4, 4]
        else:
            B_inv_id, _ = self._get_bind_pose(skin_input, adain_input=_adain)

        if z_exp_override is not None:
            z_exp_flat = z_exp_override
        else:
            z_exp = self.lbs_exp_z_model(deform_in)
            z_exp_flat = z_exp.squeeze(1) if z_exp.dim() == 3 else z_exp  # [B, L]

        pose_out = self.lbs_pose_model(z_exp_flat.unsqueeze(1)).squeeze(1)
        if self.use_joint_trans:
            rot6d   = pose_out[:, :J*6].reshape(B * J, 6)
            local_t = pose_out[:, J*6:].reshape(B, J, 3, 1)
        else:
            rot6d   = pose_out.reshape(B * J, 6)
            local_t = torch.zeros(B, J, 3, 1, device=device, dtype=source_vert.dtype)

        local_R   = self._rot6d(rot6d).reshape(B, J, 3, 3)
        zeros_row = torch.zeros(B, J, 1, 3, device=device, dtype=source_vert.dtype)
        ones_val  = torch.ones( B, J, 1, 1, device=device, dtype=source_vert.dtype)
        top       = torch.cat([local_R, local_t], dim=-1)
        bot       = torch.cat([zeros_row, ones_val], dim=-1)
        T_local   = torch.cat([top, bot], dim=-2)

        T_world = self._chain_hierarchy(T_local)
        G = torch.bmm(
            T_world.reshape(B * J, 4, 4),
            B_inv_id.reshape(B * J, 4, 4),
        ).reshape(B, J, 4, 4)

        v_h = torch.cat([source_vert, torch.ones(B, N, 1, device=device, dtype=source_vert.dtype)], dim=-1)
        v_per_joint = torch.einsum('bjkl,bnl->bnjk', G[:, :, :3, :], v_h)
        rigid_v     = torch.einsum('bnj,bnjk->bnk', W, v_per_joint)

        if return_z_exp:
            return rigid_v, z_exp_flat
        return rigid_v

    # ── Cross-retargeting ─────────────────────────────────────────────────

    def retarget(
        self,
        src_neu_vert: torch.Tensor,
        src_neu_norm: torch.Tensor,
        src_def_vert: torch.Tensor,
        src_def_norm: torch.Tensor,
        tgt_neu_vert: torch.Tensor,
        tgt_neu_norm: torch.Tensor,
        tgt_nfs_feat: torch.Tensor = None,
    ):
        """
        Cross-retargeting: apply SOURCE expression to TARGET identity.

        Expression (joint rotations): extracted from SOURCE deform_in
        Identity (skin weights W, bind pose): from TARGET template geometry

        Args:
            src_neu_vert : [B, N_s, 3]
            src_neu_norm : [B, N_s, 3]
            src_def_vert : [B, N_s, 3]
            src_def_norm : [B, N_s, 3]
            tgt_neu_vert : [B, N_t, 3]
            tgt_neu_norm : [B, N_t, 3]

        Returns:
            rigid_v : [B, N_t, 3]
        """
        B = src_def_vert.shape[0]
        N_t = tgt_neu_vert.shape[1]
        J = self.num_joints
        device = src_def_vert.device

        # ── Expression from SOURCE ──────────────────────────────────────
        delta_src = src_def_vert - src_neu_vert
        src_in    = torch.cat([src_neu_vert, src_neu_norm], dim=-1)
        deform_in = torch.cat([delta_src, src_def_norm, src_in], dim=-1)

        z_exp = self.lbs_exp_z_model(deform_in)
        z_exp_flat = z_exp.squeeze(1) if z_exp.dim() == 3 else z_exp
        pose_out = self.lbs_pose_model(z_exp_flat.unsqueeze(1)).squeeze(1)

        if self.use_joint_trans:
            rot6d   = pose_out[:, :J*6].reshape(B * J, 6)
            local_t = pose_out[:, J*6:].reshape(B, J, 3, 1)
        else:
            rot6d   = pose_out.reshape(B * J, 6)
            local_t = torch.zeros(B, J, 3, 1, device=device, dtype=src_def_vert.dtype)

        local_R   = self._rot6d(rot6d).reshape(B, J, 3, 3)
        zeros_row = torch.zeros(B, J, 1, 3, device=device, dtype=src_def_vert.dtype)
        ones_val  = torch.ones( B, J, 1, 1, device=device, dtype=src_def_vert.dtype)
        top       = torch.cat([local_R, local_t], dim=-1)
        bot       = torch.cat([zeros_row, ones_val], dim=-1)
        T_local   = torch.cat([top, bot], dim=-2)
        T_world   = self._chain_hierarchy(T_local)

        # ── Identity from TARGET ────────────────────────────────────────
        tgt_skin_input, tgt_adain = self._prepare_feat(tgt_neu_vert, tgt_neu_norm, tgt_nfs_feat)
        W_tgt, _ = self._get_skinning_weights(tgt_skin_input, adain_input=tgt_adain)
        if self.freeze_bind_pose:
            B_inv_tgt = self.B_inv_fixed.unsqueeze(0).expand(B, -1, -1, -1)
        else:
            B_inv_tgt, _ = self._get_bind_pose(tgt_skin_input, adain_input=tgt_adain)

        G = torch.bmm(
            T_world.reshape(B * J, 4, 4),
            B_inv_tgt.reshape(B * J, 4, 4),
        ).reshape(B, J, 4, 4)

        # ── LBS on target mesh ──────────────────────────────────────────
        v_h = torch.cat([
            tgt_neu_vert,
            torch.ones(B, N_t, 1, device=device, dtype=tgt_neu_vert.dtype),
        ], dim=-1)
        v_per_joint = torch.einsum('bjkl,bnl->bnjk', G[:, :, :3, :], v_h)
        rigid_v     = torch.einsum('bnj,bnjk->bnk', W_tgt, v_per_joint)

        return rigid_v

    # ── Phase 1 init supervision ─────────────────────────────────────────

    def init_loss(self, source_vert, source_normal=None, mesh_data=None, perm_idx=None, nfs_feat=None):
        """
        Maya init supervision loss for Phase 1 warm-up.
        If mesh_data is provided, uses per-topology W target.
        If perm_idx is provided, slices W target accordingly.
        """
        skin_input, _adain = self._prepare_feat(source_vert, source_normal, nfs_feat)
        W, _ = self._get_skinning_weights(skin_input, adain_input=_adain)
        _, joint_pos = self._get_bind_pose(skin_input, adain_input=_adain)
        losses = {}

        # Find appropriate W target
        W_target = None
        if mesh_data is not None:
            md = mesh_data.item() if isinstance(mesh_data, torch.Tensor) else mesh_data
            topo_key = self._mesh_data_to_topo.get(md)
            if topo_key and topo_key in self._init_targets:
                W_target = self._init_targets[topo_key]
        else:
            for k, v in self._init_targets.items():
                if v.shape[0] >= source_vert.shape[1]:
                    W_target = v
                    break

        if W_target is not None:
            N = source_vert.shape[1]
            if perm_idx is not None:
                W_target = W_target[perm_idx, :]
            elif W_target.shape[0] > N:
                W_target = W_target[:N, :]
            losses['L_W_init'] = F.mse_loss(W, W_target.unsqueeze(0).expand_as(W))

        losses['L_bind_init'] = F.mse_loss(joint_pos, self.bind_pos_target.unsqueeze(0).expand_as(joint_pos))
        return losses

    # ── Regional weight constraint ──────────────────────────────────────

    # Target joint indices for constrained regions (all 66 joints)
    _CONSTRAINED_JOINTS = list(range(66))

    def _build_regional_weight_constraints(self, alpha=0.5, adaptive=False, topologies=('ict',)):
        """Build per-joint min threshold and dominant vertex masks from Maya init.
        Call once after model creation. Builds for specified topologies.

        Args:
            topologies: tuple of topo keys to apply RWC (e.g. ('ict',) or ('ict', 'mf'))
        """
        self._rwc_per_topo = {}

        for topo_key in topologies:
            W_maya = self._init_targets.get(topo_key)
            if W_maya is None:
                print(f"[WARN] No {topo_key} init target for regional weight constraints")
                continue

            dom_counts = {}
            for j in self._CONSTRAINED_JOINTS:
                dom_mask = W_maya[:, j] > 0.01
                if dom_mask.sum() > 0:
                    dom_counts[j] = dom_mask.sum().item()

            if not dom_counts:
                continue

            max_n_dom = max(dom_counts.values())
            rwc = {}
            for j in self._CONSTRAINED_JOINTS:
                dom_mask = W_maya[:, j] > 0.01
                if dom_mask.sum() == 0:
                    continue
                mean_w = W_maya[dom_mask, j].mean().item()

                if adaptive:
                    n_dom = dom_counts[j]
                    alpha_j = alpha + (1.0 - alpha) * (1.0 - n_dom / max_n_dom)
                else:
                    alpha_j = alpha

                threshold = alpha_j * mean_w
                rwc[j] = {
                    'dom_verts': dom_mask,
                    'threshold': threshold,
                    'mean_maya': mean_w,
                    'alpha': alpha_j,
                }

            self._rwc_per_topo[topo_key] = rwc
            print(f"[HLBS] RWC built for '{topo_key}': {len(rwc)} joints "
                  f"(base_alpha={alpha}, adaptive={adaptive})")
            for j, info in rwc.items():
                print(f"  [{j:2d}] {self.joint_names[j]:>40s}  "
                      f"alpha={info['alpha']:.3f}  thr={info['threshold']:.4f}  "
                      f"maya_mean={info['mean_maya']:.4f}")

        # Backward compat
        self._rwc = self._rwc_per_topo.get('ict', {})

    def regional_weight_constraint_loss(self, source_vert, source_normal=None,
                                         mesh_data=None, perm_idx=None, nfs_feat=None):
        """
        Regional weight constraint loss for maintaining anatomically
        meaningful skinning weights on specific joints.

        Two components:
          1) L_rwc_init: MSE to Maya W on constrained joint-vertex pairs
          2) L_rwc_min:  soft penalty when W drops below per-joint threshold

        Applies to topologies registered in _rwc_per_topo.
        """
        losses = {}

        if not hasattr(self, '_rwc_per_topo') or not self._rwc_per_topo:
            # Backward compat: fall back to self._rwc (ICT only)
            if not hasattr(self, '_rwc') or not self._rwc:
                return losses
            self._rwc_per_topo = {'ict': self._rwc}

        # Determine current topology
        topo_key = None
        if mesh_data is not None:
            md = mesh_data.item() if isinstance(mesh_data, torch.Tensor) else mesh_data
            topo_key = self._mesh_data_to_topo.get(md)

        if topo_key is None or topo_key not in self._rwc_per_topo:
            return losses

        rwc = self._rwc_per_topo[topo_key]
        if not rwc:
            return losses

        skin_input, _adain = self._prepare_feat(source_vert, source_normal, nfs_feat)
        W, _ = self._get_skinning_weights(skin_input, adain_input=_adain)  # [B, N, J]

        W_maya = self._init_targets.get(topo_key)
        if W_maya is None:
            return losses

        N = source_vert.shape[1]

        # Slice W_maya to match current vertex count (region select)
        if perm_idx is not None:
            W_maya_sliced = W_maya[perm_idx, :]
        elif W_maya.shape[0] > N:
            W_maya_sliced = W_maya[:N, :]
        else:
            W_maya_sliced = W_maya

        init_loss = 0.0
        min_w_loss = 0.0
        count = 0

        for j, info in rwc.items():
            dom = info['dom_verts']
            thr = info['threshold']

            # Slice dominant mask to match current vertex count
            if perm_idx is not None:
                dom_sliced = dom[perm_idx]
            elif dom.shape[0] > N:
                dom_sliced = dom[:N]
            else:
                dom_sliced = dom

            if dom_sliced.sum() == 0:
                continue

            # 1) Init supervision on this joint's dominant vertices
            W_pred_j = W[:, dom_sliced, j]                          # [B, n_dom]
            W_tgt_j  = W_maya_sliced[dom_sliced, j].unsqueeze(0).expand_as(W_pred_j)
            init_loss = init_loss + F.mse_loss(W_pred_j, W_tgt_j)

            # 2) Min weight penalty
            deficit = torch.relu(thr - W_pred_j)                    # [B, n_dom]
            min_w_loss = min_w_loss + (deficit ** 2).mean()

            count += 1

        if count > 0:
            losses['L_rwc_init'] = init_loss / count
            losses['L_rwc_min'] = min_w_loss / count

        return losses

    # ── Hierarchy locality loss ──────────────────────────────────────────

    def _build_hierarchy_constraints(self, topologies=('ict',)):
        """Build leaf-joint dominant region masks from Maya init.
        For each leaf joint, store vertices where W_maya > 0.01 (has weight).
        """
        J = self.num_joints
        parent_idx = self.parent_idx.cpu().numpy()

        # Find leaf joints (no children)
        has_child = set()
        for j in range(J):
            p = parent_idx[j]
            if p >= 0:
                has_child.add(p)
        self._leaf_joints = [j for j in range(J) if j not in has_child]

        self._hier_per_topo = {}
        for topo_key in topologies:
            W_maya = self._init_targets.get(topo_key)
            if W_maya is None:
                continue

            hier = {}
            for j in self._leaf_joints:
                region = W_maya[:, j] > 0.01  # vertices where this leaf has weight
                if region.sum() == 0:
                    continue
                hier[j] = region  # [N] bool mask
            self._hier_per_topo[topo_key] = hier

            print(f"[HLBS] Hierarchy constraints built for '{topo_key}': "
                  f"{len(hier)}/{len(self._leaf_joints)} leaf joints")
            for j, mask in hier.items():
                print(f"  [{j:2d}] {self.joint_names[j]:>40s}  region_verts={mask.sum().item()}")

    def hierarchy_locality_loss(self, source_vert, source_normal=None,
                                mesh_data=None, perm_idx=None, nfs_feat=None,
                                margin=0.0):
        """
        Leaf joint locality loss: each leaf joint should have the maximum
        predicted weight in its Maya-defined region (W_maya > 0.01).

        loss = mean over leaf joints of:
            relu(max_other_weight - W_pred[:, leaf] + margin)
        """
        losses = {}

        if not hasattr(self, '_hier_per_topo') or not self._hier_per_topo:
            return losses

        # Determine topology
        topo_key = None
        if mesh_data is not None:
            md = mesh_data.item() if isinstance(mesh_data, torch.Tensor) else mesh_data
            topo_key = self._mesh_data_to_topo.get(md)

        if topo_key is None or topo_key not in self._hier_per_topo:
            return losses

        hier = self._hier_per_topo[topo_key]
        if not hier:
            return losses

        skin_input, _adain = self._prepare_feat(source_vert, source_normal, nfs_feat)
        W, _ = self._get_skinning_weights(skin_input, adain_input=_adain)  # [B, N, J]
        N = source_vert.shape[1]

        total_loss = 0.0
        count = 0

        for j, region in hier.items():
            # Slice region mask for current vertex count
            if perm_idx is not None:
                region_sliced = region[perm_idx]
            elif region.shape[0] > N:
                region_sliced = region[:N]
            else:
                region_sliced = region

            if region_sliced.sum() == 0:
                continue

            W_region = W[:, region_sliced, :]        # [B, n_region, J]
            W_leaf = W_region[:, :, j]               # [B, n_region]

            # Max weight among all OTHER joints
            W_others = W_region.clone()
            W_others[:, :, j] = -1e9                 # exclude self
            max_other = W_others.max(dim=-1).values  # [B, n_region]

            # Penalty when any other joint exceeds this leaf
            violation = torch.relu(max_other - W_leaf + margin)
            total_loss = total_loss + violation.mean()
            count += 1

        if count > 0:
            losses['L_hier'] = total_loss / count

        return losses

    # ── Regularization ───────────────────────────────────────────────────

    def reg_loss(self, source_feat, edges=None):
        W, _ = self._get_skinning_weights(source_feat)
        L_W_smooth = source_feat.new_zeros(())
        if edges is not None:
            W_i = W[:, edges[:, 0], :]
            W_j = W[:, edges[:, 1], :]
            L_W_smooth = ((W_i - W_j) ** 2).mean()
        return {'L_W_smooth': L_W_smooth}


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
