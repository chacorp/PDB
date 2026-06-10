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
        _po = getattr(rig, 'process_order', None)
        if _po is None:
            _po = torch.arange(len(rig.joint_names), dtype=torch.int64)
        self.register_buffer('process_order', _po.to(device))           # [J] long

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
        for j in self.process_order.tolist():
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
        use_gmm_hybrid: bool = False,
        init_log_sigma: float = -1.2,
        gmm_mode: str = 'additive',
        residual_scale: float = 2.0,
        sigma_targets: 'np.ndarray | torch.Tensor | None' = None,
        face_joint_idx: list = None,
        base_joint_idx: int = None,
        face_mask_r0: float = 1.0,
        face_mask_r1: float = 2.25,
        bind_pose_mode: str = 'net',
        joint_anchors: 'np.ndarray | torch.Tensor | None' = None,
        joint_offsets: 'np.ndarray | torch.Tensor | None' = None,
        attn_temperature_init: float = 0.1,
        helper_joint_idx: 'list | None' = None,
        bind_pose_base_residual: bool = False,
    ):
        super().__init__()

        self.device_str = device
        self.use_joint_trans = use_joint_trans
        self.freeze_bind_pose = freeze_bind_pose
        self.use_gmm_hybrid = use_gmm_hybrid
        self.gmm_mode = gmm_mode
        assert gmm_mode in ('additive', 'multiplicative', 'residual'), f'unknown gmm_mode: {gmm_mode}'
        self.residual_scale = float(residual_scale)
        # Bind-pose source: 'net' = bind_pose_net MLP (default), 'anchor_pool' = closed-form NFS attention
        assert bind_pose_mode in ('net', 'anchor_pool'), f'unknown bind_pose_mode: {bind_pose_mode}'
        self.bind_pose_mode = bind_pose_mode
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

        # ── Option A: Face-mask mode ─────────────────────────────────────
        # If face_joint_idx is set, softmax is restricted to these joints on
        # face vertices (determined by plateau_hat mask). Non-face vertices
        # get W=1 on base_joint_idx. This forces the network to only predict
        # within the face region, and routes non-face rigid motion to a base.
        self.use_face_mask = face_joint_idx is not None and len(face_joint_idx) > 0
        self.face_mask_r0 = face_mask_r0
        self.face_mask_r1 = face_mask_r1
        if self.use_face_mask:
            face_idx_t = torch.tensor(sorted(face_joint_idx), dtype=torch.long, device=device)
            self.register_buffer('face_joint_idx', face_idx_t)
            self.base_joint_idx = int(base_joint_idx) if base_joint_idx is not None else 2
            print(f'[HLBS FaceMask] enabled | face joints: {len(face_joint_idx)}/{J} | '
                  f'base_idx={self.base_joint_idx} ({self.joint_names[self.base_joint_idx]}) | '
                  f'r0={face_mask_r0}, r1={face_mask_r1}')

        # ── Helper joint reparameterization (parent + residual) ──────────
        # When helper_joint_idx is provided, bind_pose_net's raw output for
        # helper joints is treated as a parent-relative residual instead of an
        # absolute position. Net output for non-helpers stays absolute.
        # The residual tensor [B, n_helpers, 3] is stored as a side-effect
        # attribute (_last_bind_pose_residual) for L2 penalty in the trainer.
        if helper_joint_idx is not None and len(helper_joint_idx) > 0:
            h_t = torch.tensor(sorted(helper_joint_idx), dtype=torch.long, device=device)
            self.register_buffer('helper_joint_idx_buf', h_t)
            self._helper_joint_set = set(int(x) for x in helper_joint_idx)
            print(f'[HLBS HelperReparam] enabled | {len(helper_joint_idx)} helpers '
                  f'predicted as parent + residual')
        else:
            self._helper_joint_set = set()
        self._last_bind_pose_residual = None

        # ── Anchor-pool bind-pose mode ───────────────────────────────────
        # Closed-form: bind_pos = pool_attn(NFS_feat, anchor) + offset
        # Anchors & offsets derived once on ICT mean (precompute_joint_anchors.py).
        if self.bind_pose_mode == 'anchor_pool':
            assert joint_anchors is not None and joint_offsets is not None, \
                'bind_pose_mode="anchor_pool" requires joint_anchors and joint_offsets'
            if not isinstance(joint_anchors, torch.Tensor):
                joint_anchors = torch.tensor(joint_anchors, dtype=torch.float32)
            if not isinstance(joint_offsets, torch.Tensor):
                joint_offsets = torch.tensor(joint_offsets, dtype=torch.float32)
            assert joint_anchors.shape[0] == J, \
                f'joint_anchors shape {tuple(joint_anchors.shape)} mismatches J={J}'
            assert joint_offsets.shape == (J, 3), \
                f'joint_offsets shape {tuple(joint_offsets.shape)} != ({J}, 3)'
            self.register_buffer('joint_anchors', joint_anchors.to(device))    # [J, 256]
            self.register_buffer('joint_offsets', joint_offsets.to(device))    # [J, 3]
            # Per-joint learnable temperature (sharper/smoother attention per joint)
            self.attn_temperature = nn.Parameter(
                torch.full((J,), float(attn_temperature_init), device=device))
            print(f'[HLBS BindPose] anchor_pool enabled | anchors={tuple(joint_anchors.shape)} '
                  f'| offset_norm range=[{joint_offsets.norm(dim=-1).min():.4f}, '
                  f'{joint_offsets.norm(dim=-1).max():.4f}] '
                  f'| init T={attn_temperature_init}')

        # ── Fixed hierarchy buffers ──────────────────────────────────────
        self.register_buffer('parent_idx', rig.parent_idx.to(device))
        self.register_buffer('bind_pos',   rig.bind_pos.to(device))
        _po = getattr(rig, 'process_order', None)
        if _po is None:
            _po = torch.arange(len(rig.joint_names), dtype=torch.int64)
        self.register_buffer('process_order', _po.to(device))

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
        # Per-topology bind_pos buffers (used for L_bind_reg in net mode to
        # supervise each identity toward its own topology's Maya GT).
        for topo_key, bp in rig.bind_pos_dict.items():
            self.register_buffer(f'bind_pos_target_{topo_key}', bp.to(device).clone())

        # ── bind pose base-residual reparameterization ───────────────────
        # When enabled, bind_pose_net predicts a RESIDUAL for ALL joints on top
        # of a single fixed base (ICT mean bind pose — symmetric, topology-
        # agnostic). joint_pos = bind_pos_base + raw. Universal base means no
        # per-topology base needed at inference (works for unseen topologies).
        self._bind_pose_base_residual = bool(bind_pose_base_residual)
        if self._bind_pose_base_residual:
            _base = rig.bind_pos_dict.get('ict', rig.bind_pos)
            self.register_buffer('bind_pos_base', _base.to(device).clone())
            print(f'[HLBS BindPoseResidual] enabled | base = ICT mean bind pose '
                  f'(joint_pos = base + net_residual for all {J} joints)')

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

        # ── GMM hybrid params ────────────────────────────────────────────
        # logit_total = skin_weight_net_output + (-||v - μ||² / (2σ²))
        # μ from bind_pose_net (per-identity), σ learnable per-joint scalar.
        # If `sigma_targets` is provided, it's used as per-joint init AND as
        # shrinkage target (via sigma_shrink_loss) to prevent σ from inflating.
        if use_gmm_hybrid:
            if sigma_targets is not None:
                # Per-joint σ from category-based spec (np.ndarray or tensor, shape [J])
                if not isinstance(sigma_targets, torch.Tensor):
                    sigma_targets = torch.tensor(sigma_targets, dtype=torch.float32)
                sigma_targets = sigma_targets.to(device).float()
                assert sigma_targets.shape == (J,), \
                    f'sigma_targets shape {tuple(sigma_targets.shape)} != ({J},)'
                log_sigma_target = torch.log(sigma_targets.clamp_min(1e-4))
                self.log_sigma = nn.Parameter(log_sigma_target.clone())
                self.register_buffer('log_sigma_target', log_sigma_target)
                _init_msg = (f'per-joint from targets  '
                             f'range log σ=[{log_sigma_target.min():+.2f}, {log_sigma_target.max():+.2f}]  '
                             f'σ=[{sigma_targets.min():.3f}, {sigma_targets.max():.3f}]')
            else:
                self.log_sigma = nn.Parameter(
                    torch.full((J,), float(init_log_sigma), device=device)
                )
                # No target → penalty will be skipped even if lambda>0
                self.register_buffer('log_sigma_target',
                                     torch.full((J,), float(init_log_sigma), device=device))
                _init_msg = f'uniform init log_σ={init_log_sigma}'

            # zero-init last layer of skin_weight_net → initial W ≈ pure Gaussian
            _last = None
            for m in self.skin_weight_net.modules():
                if isinstance(m, nn.Linear):
                    _last = m
            if _last is not None:
                nn.init.zeros_(_last.weight)
                if _last.bias is not None:
                    nn.init.zeros_(_last.bias)
                print(f'[HLBS] GMM hybrid enabled: {_init_msg}, '
                      f'skin_weight_net last layer zero-initialized, mode={gmm_mode}')

        # bind_pose_net: built only in 'net' mode. anchor_pool / freeze use closed-form path.
        if self.bind_pose_mode == 'net':
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
        else:
            self.bind_pose_net = None  # not used; freed for param savings

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
        n_verts = int(f_np.max()) + 1
        edges_t = torch.tensor(e, dtype=torch.long, device=self.parent_idx.device)
        self._mesh_edges = edges_t
        # Also populate the per-N dict used by weight_smoothness_loss /
        # weight_quality_metric — without this, those silently skip and log 0.
        if self._mesh_edges_by_N is None:
            self._mesh_edges_by_N = {}
        self._mesh_edges_by_N[n_verts] = edges_t

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

    def _get_skinning_weights(self, source_feat, adain_input=None,
                               source_vert=None, joint_pos=None,
                               dist_sq_override=None):
        if self.dfn_skin:
            logit_net = self.skin_weight_net(source_feat)  # [B, N, J]
        else:
            logit_net = self.skin_weight_net(source_feat, adain_input=adain_input)  # [B, N, J]
        logit_net = self._smooth_logit_W(logit_net)
        # Cache raw smoothed logit_net for net_center_loss / diagnostic access via extras
        self._last_logit_net = logit_net

        # GMM hybrid: combine Gaussian bias with network logits
        #   additive       : softmax(logit_net + logit_gauss)           [original]
        #   multiplicative : softmax(logit_gauss) * sigmoid(logit_net)  [net refines WITHIN Gaussian support]
        _have_dist_input = (dist_sq_override is not None) or (source_vert is not None and joint_pos is not None)
        if self.use_gmm_hybrid and _have_dist_input:
            if dist_sq_override is not None:
                # Geodesic distance² lookup, [B, N, J] (bypass euclidean compute)
                dist_sq = dist_sq_override
            else:
                # Euclidean: [B, N, J, 3] = [B, N, 1, 3] - [B, 1, J, 3]
                diff = source_vert.unsqueeze(2) - joint_pos.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1)                       # [B, N, J]
            sigma_sq = torch.exp(2 * self.log_sigma).view(1, 1, -1)     # [1, 1, J]
            logit_gauss = -dist_sq / (2 * sigma_sq)                     # [B, N, J]

            if self.gmm_mode == 'multiplicative':
                W_prior = F.softmax(logit_gauss, dim=-1)                 # [B, N, J]  RBF support
                modul   = torch.sigmoid(logit_net)                       # [B, N, J]  in [0,1]
                W_raw   = W_prior * modul
                W_pre   = W_raw / (W_raw.sum(dim=-1, keepdim=True) + 1e-8)
                logit_total = W_pre.clamp_min(1e-8).log()
            elif self.gmm_mode == 'residual':
                # µ-centered bounded residual: net acts as bounded refinement
                # whose impact decays as exp(logit_gauss) away from µ_j.
                #   logit_total = logit_gauss + exp(logit_gauss) · tanh(net/s) · s
                # Far from µ (window→0): residual disappears; near µ: ±s.
                s = self.residual_scale
                bounded = torch.tanh(logit_net / s) * s                  # [-s, +s]
                window  = torch.exp(logit_gauss)                          # (0, 1]
                logit_total = logit_gauss + window * bounded
                W_pre = None  # softmax below
            else:  # 'additive'
                logit_total = logit_net + logit_gauss
                W_pre = None  # softmax below
        else:
            logit_total = logit_net
            W_pre = None

        # ── Option A: face-mask mode ────────────────────────────────────
        # Face region (t_mask≈1): softmax restricted to self.face_joint_idx.
        # Non-face region (t_mask≈0): W routed to self.base_joint_idx.
        # Smooth blending via plateau_hat preserves partition-of-unity.
        if self.use_face_mask and source_vert is not None:
            from utils.exp_utils import plateau_hat_points
            B, N, J = logit_total.shape
            t_mask = plateau_hat_points(
                source_vert, r0=self.face_mask_r0, r1=self.face_mask_r1,
            ).squeeze(-1)                                               # [B, N]

            logit_face = logit_total.index_select(-1, self.face_joint_idx)   # [B, N, J_face]
            W_face = F.softmax(logit_face, dim=-1)                            # sums to 1 over face joints

            W_full = torch.zeros(B, N, J, device=source_vert.device,
                                 dtype=source_vert.dtype)
            W_full.index_copy_(-1, self.face_joint_idx,
                               W_face * t_mask.unsqueeze(-1))                 # scale by t_mask
            # Route non-face mass to base joint (add, so base may coexist in face set)
            W_full[..., self.base_joint_idx] = (
                W_full[..., self.base_joint_idx] + (1.0 - t_mask)
            )
            return W_full, logit_total

        if W_pre is not None:
            return W_pre, logit_total
        return F.softmax(logit_total, dim=-1), logit_total

    @staticmethod
    def _build_B_inv(joint_pos):
        """[B, J, 3] joint world positions → [B, J, 4, 4] inverse bind matrix."""
        B, J, _ = joint_pos.shape
        dev, dtype = joint_pos.device, joint_pos.dtype
        eye3    = torch.eye(3, device=dev, dtype=dtype).expand(B, J, 3, 3)
        neg_jp  = (-joint_pos).unsqueeze(-1)
        top     = torch.cat([eye3, neg_jp], dim=-1)
        bot     = torch.cat([
            torch.zeros(B, J, 1, 3, device=dev, dtype=dtype),
            torch.ones( B, J, 1, 1, device=dev, dtype=dtype),
        ], dim=-1)
        return torch.cat([top, bot], dim=-2)

    def _get_bind_pose(self, source_feat, adain_input=None):
        """Net-based bind pose prediction (mode='net' path).

        Two reparameterizations:
        - base_residual (bind_pose_base_residual=True): ALL joints predicted as
          a residual on a single fixed base (ICT mean bind pose) —
              joint_pos = bind_pos_base + raw
          _last_bind_pose_residual = raw [B, J, 3] (all joints; trainer applies
          differentiated L2: weak for helpers, strong for non-helpers).
        - legacy (default): non-helper joints absolute, helper joints
          parent-relative residual (joint_pos[h] = joint_pos[parent] + raw[h]).
        """
        B = source_feat.shape[0]
        J = self.num_joints
        if self.dfn_bind:
            raw = self.bind_pose_net(source_feat).squeeze(1).reshape(B, J, 3)
        else:
            raw = self.bind_pose_net(source_feat, adain_input=adain_input).squeeze(1).reshape(B, J, 3)

        if self._bind_pose_base_residual:
            joint_pos = self.bind_pos_base.unsqueeze(0) + raw            # [B, J, 3]
            self._last_bind_pose_residual = raw                          # all joints
        elif self._helper_joint_set:
            joint_pos = raw.clone()
            for j in self._helper_joint_set:
                p = int(self.parent_idx[j].item())
                if p >= 0:
                    joint_pos[:, j] = joint_pos[:, p] + raw[:, j]
            self._last_bind_pose_residual = raw.index_select(1, self.helper_joint_idx_buf)
        else:
            joint_pos = raw
            self._last_bind_pose_residual = None
        return self._build_B_inv(joint_pos), joint_pos

    def _get_bind_pose_anchor(self, source_vert, nfs_feat, bind_pos_cache=None):
        """
        Closed-form anchor-pool bind pose.

        Args:
            source_vert:   [B, V, 3]
            nfs_feat:      [B, V, K]   per-vertex NFS feat (must be supplied in this mode)
            bind_pos_cache:[B, J, 3]   if not None, use cached value (skips attention pool).

        Returns:
            B_inv_id:  [B, J, 4, 4]
            joint_pos: [B, J, 3]
        """
        if bind_pos_cache is not None:
            joint_pos = bind_pos_cache
        else:
            assert nfs_feat is not None, \
                'bind_pose_mode="anchor_pool" requires nfs_feat (or bind_pos_cache)'
            from utils.anchor_pool import compute_bind_pos
            joint_pos = compute_bind_pos(
                source_vert, nfs_feat,
                self.joint_anchors, self.joint_offsets,
                temperature=self.attn_temperature,
                offset_frame='world',
            )
        return self._build_B_inv(joint_pos), joint_pos

    def _build_fk_levels(self):
        """Group joints by tree depth. Joints at the same depth are independent
        → one batched matmul per depth level instead of a 66-step Python loop
        with per-joint .item() syncs. Built once, then cached on self._fk_levels.
        """
        parent = self.parent_idx.tolist()
        J = len(parent)
        depth = [-1] * J

        def _depth(j):
            if depth[j] >= 0:
                return depth[j]
            p = parent[j]
            depth[j] = 0 if p < 0 else _depth(p) + 1
            return depth[j]

        for j in range(J):
            _depth(j)
        dev = self.parent_idx.device
        levels = []
        for d in range(1, max(depth) + 1):
            js = [j for j in range(J) if depth[j] == d]
            ps = [parent[j] for j in js]
            levels.append((
                torch.tensor(js, dtype=torch.long, device=dev),
                torch.tensor(ps, dtype=torch.long, device=dev),
            ))
        self._fk_levels = levels

    def _chain_hierarchy(self, T_delta):
        B, J, _, _ = T_delta.shape
        if getattr(self, '_fk_levels', None) is None:
            self._build_fk_levels()
        T_bind_local = self.T_bind_local.unsqueeze(0).expand(B, -1, -1, -1)
        # combined[b,j] = T_bind_local[j] @ T_delta[j] — all joints in parallel.
        combined = torch.matmul(T_bind_local, T_delta)               # [B, J, 4, 4]
        # Roots (depth 0) need no parent → T_world == combined for them.
        T_world = combined
        # Walk depth levels; each level's joints are independent (one matmul).
        for j_idx, p_idx in self._fk_levels:
            parent_T = T_world.index_select(1, p_idx)                # [B, n, 4, 4]
            child_T  = combined.index_select(1, j_idx)               # [B, n, 4, 4]
            T_world  = T_world.index_copy(1, j_idx,
                                          torch.matmul(parent_T, child_T))
        return T_world

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, source_vert, deform_in, source_normal=None, return_z_exp=False, z_exp_override=None, nfs_feat=None, return_extras=False, bind_pos_cache=None, dist_sq_geo=None, reuse_identity=False):
        B, N, _ = source_vert.shape
        J = self.num_joints
        device = source_vert.device

        # Identity-only block: W / joint_pos / B_inv_id depend solely on
        # (source_vert, source_normal, nfs_feat, bind_pos_cache, dist_sq_geo) —
        # NOT on deform_in. reuse_identity=True reuses the values cached by the
        # immediately preceding forward (e.g. neutral recon reusing the main pass),
        # skipping a redundant _prepare_feat + bind_pose_net + skin_weight_net.
        if reuse_identity and getattr(self, '_last_W', None) is not None:
            W         = self._last_W
            joint_pos = self._last_joint_pos
            B_inv_id  = self._last_B_inv
        else:
            skin_input, _adain = self._prepare_feat(source_vert, source_normal, nfs_feat)

            # Bind pose first — μ needed for GMM-hybrid skinning
            if self.freeze_bind_pose:
                if bind_pos_cache is not None:
                    # Per-topo / per-id frozen bind_pos passed in via cache (preferred)
                    joint_pos = bind_pos_cache
                    B_inv_id = self._build_B_inv(joint_pos)
                else:
                    B_inv_id = self.B_inv_fixed.unsqueeze(0).expand(B, -1, -1, -1)  # [B, J, 4, 4]
                    joint_pos = self.bind_pos_target.unsqueeze(0).expand(B, -1, -1) # [B, J, 3]
            elif self.bind_pose_mode == 'anchor_pool':
                B_inv_id, joint_pos = self._get_bind_pose_anchor(
                    source_vert, nfs_feat, bind_pos_cache=bind_pos_cache)
            else:  # 'net'
                B_inv_id, joint_pos = self._get_bind_pose(skin_input, adain_input=_adain)

            W, _ = self._get_skinning_weights(
                skin_input, adain_input=_adain,
                source_vert=source_vert, joint_pos=joint_pos,
                dist_sq_override=dist_sq_geo,
            )
            self._last_W         = W
            self._last_joint_pos = joint_pos
            self._last_B_inv     = B_inv_id

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

        if return_extras:
            extras = {
                'W': W, 'joint_pos': joint_pos,
                'local_R': local_R, 'local_t': local_t.squeeze(-1),  # [B,J,3,3], [B,J,3]
                'T_world': T_world,
                'logit_net': getattr(self, '_last_logit_net', None),  # [B,N,J] for net_center_loss
            }
            if return_z_exp:
                return rigid_v, z_exp_flat, extras
            return rigid_v, extras
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
        tgt_bind_pos_cache: torch.Tensor = None,
        tgt_dist_sq_geo: torch.Tensor = None,
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
        if self.freeze_bind_pose:
            if tgt_bind_pos_cache is not None:
                tgt_joint_pos = tgt_bind_pos_cache
                B_inv_tgt = self._build_B_inv(tgt_joint_pos)
            else:
                B_inv_tgt = self.B_inv_fixed.unsqueeze(0).expand(B, -1, -1, -1)
                tgt_joint_pos = self.bind_pos_target.unsqueeze(0).expand(B, -1, -1)
        elif self.bind_pose_mode == 'anchor_pool':
            B_inv_tgt, tgt_joint_pos = self._get_bind_pose_anchor(
                tgt_neu_vert, tgt_nfs_feat, bind_pos_cache=tgt_bind_pos_cache)
        else:  # 'net'
            B_inv_tgt, tgt_joint_pos = self._get_bind_pose(tgt_skin_input, adain_input=tgt_adain)
        W_tgt, _ = self._get_skinning_weights(
            tgt_skin_input, adain_input=tgt_adain,
            source_vert=tgt_neu_vert, joint_pos=tgt_joint_pos,
            dist_sq_override=tgt_dist_sq_geo,
        )

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

    # ── σ shrinkage loss ─────────────────────────────────────────────────

    def sigma_shrink_loss(self, active_idx=None):
        """
        Penalize log_sigma drift ABOVE target (one-sided: σ can shrink freely,
        but growing past target is penalized). Applied only to `active_idx`
        joints if provided (frozen joints' σ is irrelevant since their W=0).

        Returns: scalar loss.
        """
        if not self.use_gmm_hybrid:
            return self.log_sigma.new_tensor(0.0)
        # Relu(log_σ - target)²  → zero when log_σ ≤ target, grows quadratically above
        excess = F.relu(self.log_sigma - self.log_sigma_target)
        if active_idx is not None and len(active_idx) > 0:
            if not torch.is_tensor(active_idx):
                active_idx = torch.tensor(active_idx, dtype=torch.long,
                                          device=self.log_sigma.device)
            excess = excess.index_select(0, active_idx)
        return (excess ** 2).mean()

    # ── Phase 1 init supervision ─────────────────────────────────────────

    def init_loss(self, source_vert, source_normal=None, mesh_data=None, perm_idx=None, nfs_feat=None, dist_sq_geo=None):
        """
        Maya init supervision loss for Phase 1 warm-up.
        If mesh_data is provided, uses per-topology W target.
        If perm_idx is provided, slices W target accordingly.
        """
        skin_input, _adain = self._prepare_feat(source_vert, source_normal, nfs_feat)
        _, joint_pos = self._get_bind_pose(skin_input, adain_input=_adain)
        W, _ = self._get_skinning_weights(
            skin_input, adain_input=_adain,
            source_vert=source_vert, joint_pos=joint_pos,
            dist_sq_override=dist_sq_geo,
        )
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

    def regional_weight_constraint_loss(self, W, N, mesh_data=None, perm_idx=None):
        """
        Regional weight constraint loss for maintaining anatomically
        meaningful skinning weights on specific joints.

        Two components:
          1) L_rwc_init: MSE to Maya W on constrained joint-vertex pairs
          2) L_rwc_min:  soft penalty when W drops below per-joint threshold

        Args:
            W: [B, N, J] predicted skin weights (from forward pass)
            N: number of vertices
        """
        losses = {}

        if not hasattr(self, '_rwc_per_topo') or not self._rwc_per_topo:
            if not hasattr(self, '_rwc') or not self._rwc:
                return losses
            self._rwc_per_topo = {'ict': self._rwc}

        topo_key = None
        if mesh_data is not None:
            md = mesh_data.item() if isinstance(mesh_data, torch.Tensor) else mesh_data
            topo_key = self._mesh_data_to_topo.get(md)

        if topo_key is None or topo_key not in self._rwc_per_topo:
            return losses

        rwc = self._rwc_per_topo[topo_key]
        if not rwc:
            return losses

        W_maya = self._init_targets.get(topo_key)
        if W_maya is None:
            return losses

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

    def hierarchy_locality_loss(self, W, N, mesh_data=None, perm_idx=None, margin=0.0):
        """
        Leaf joint locality loss: each leaf joint should have the maximum
        predicted weight in its Maya-defined region (W_maya > 0.01).

        Args:
            W: [B, N, J] predicted skin weights (from forward pass)
            N: number of vertices
        """
        losses = {}

        if not hasattr(self, '_hier_per_topo') or not self._hier_per_topo:
            return losses

        topo_key = None
        if mesh_data is not None:
            md = mesh_data.item() if isinstance(mesh_data, torch.Tensor) else mesh_data
            topo_key = self._mesh_data_to_topo.get(md)

        if topo_key is None or topo_key not in self._hier_per_topo:
            return losses

        hier = self._hier_per_topo[topo_key]
        if not hier:
            return losses

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

    # ── Distance-based weight locality loss ──────────────────────────────

    def distance_weight_loss(self, W, source_vert, dist_sq_override=None):
        """
        Encourage weight to be high for joints close to the vertex.

        L_dist = mean_v ( sum_j W[v, j] * dist²(v, j) )

        dist² = Euclidean ||v - p_j||² by default, or geodesic dist²
        from joint home_vertex when dist_sq_override is supplied [B, N, J].

        Args:
            W: [B, N, J] predicted skin weights
            source_vert: [B, N, 3] template vertices
            dist_sq_override: optional [B, N, J] geodesic dist² lookup
        """
        if dist_sq_override is not None:
            dist_sq = dist_sq_override
        else:
            p = self.bind_pos_target                          # [J, 3]
            diff = source_vert.unsqueeze(2) - p.view(1, 1, -1, 3)  # [B, N, J, 3]
            dist_sq = (diff ** 2).sum(dim=-1)                 # [B, N, J]
        L_dist = (W * dist_sq).sum(dim=-1).mean()         # mean over B, N
        return {'L_dist': L_dist}

    # ── Mesh2Animation-inspired skin-weight regularizers ──────────────────

    def _w_smoothness_area(self, W, source_vert, faces):
        """Area-weighted 1-ring weight Dirichlet energy:

            E = Σ_i A_i · mean_{j∈N(i)} ‖W_i − W_j‖²  /  Σ_i A_i

        A_i = barycentric vertex area (from the template → constant, no grad).
        Area weighting makes this a surface integral ∫‖∇W‖²dS, so it is
        DENSITY-UNBIASED: densely tessellated regions (lips/eyes) are no
        longer over-penalized just for having more edges — which is what
        made narrow weight regions collapse under the uniform variant.
        Differentiable w.r.t. W. Returns scalar, or None if edges missing.
        """
        N = W.shape[1]
        edges = None if self._mesh_edges_by_N is None else self._mesh_edges_by_N.get(N)
        if edges is None:
            return None
        B = W.shape[0]
        f = faces[0] if faces.dim() == 3 else faces       # [F, 3] (shared topo)
        e0, e1 = edges[:, 0], edges[:, 1]
        diff2 = ((W.index_select(1, e0) - W.index_select(1, e1)) ** 2).sum(-1)  # [B,E] grad→W
        idx_e = torch.cat([e0, e1])                       # [2E]
        val_e = torch.cat([diff2, diff2], dim=1)          # [B, 2E]
        idx_b = idx_e.unsqueeze(0).expand(B, -1)
        var = torch.zeros(B, N, device=W.device, dtype=diff2.dtype).scatter_add_(
            1, idx_b, val_e)
        cnt = torch.zeros(B, N, device=W.device).scatter_add_(
            1, idx_b, torch.ones(B, idx_e.shape[0], device=W.device))
        var = var / cnt.clamp_min(1.0).to(var.dtype)      # [B, N] local variation
        # per-vertex barycentric area — from template (input) → constant
        with torch.no_grad():
            V = source_vert.detach().float()
            v0, v1, v2 = V[:, f[:, 0]], V[:, f[:, 1]], V[:, f[:, 2]]
            tri = 0.5 * torch.linalg.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)  # [B,F]
            A = torch.zeros(B, N, device=W.device)
            for c in range(3):
                A.scatter_add_(1, f[:, c].unsqueeze(0).expand(B, -1), tri / 3.0)
        A = A.to(var.dtype)
        E = (A * var).sum(dim=1) / A.sum(dim=1).clamp_min(1e-8)   # [B]
        return E.mean()

    def weight_smoothness_loss(self, W, source_vert, faces):
        """#1 — area-weighted 1-ring weight smoothness (Mesh2Animation L_ss).

        Area-weighted (mass-matrix style) so it is density-unbiased — unlike
        a uniform edge sum, it does NOT over-smooth finely tessellated
        regions, so narrow weight regions are not forced to collapse.
        Caller must skip when subsampled (mesh edges index the full vertex
        set). Uses mesh connectivity only → works on any topology.
        """
        E = self._w_smoothness_area(W, source_vert, faces)
        return {} if E is None else {'L_wlap': E}

    def weight_ref_loss(self, W, source_vert, joint_pos):
        """#3 — reference-weight prior (Mesh2Animation L_id, Eq. 7).

        For each vertex, the reference weight is a hard one-hot of its closest
        BONE (the joint→parent segment), exactly as the paper's GRS module.
        L_wref = MSE(W, W_ref) pulls predicted weights toward a geometrically
        valid joint assignment. joint_pos are the predicted bind-pose joints,
        so the reference co-evolves with training.
        """
        B, N, J = W.shape
        p = joint_pos                                     # [B, J, 3]
        par_idx = torch.where(self.parent_idx < 0,
                              torch.arange(J, device=p.device), self.parent_idx)
        par = p.index_select(1, par_idx)                  # [B, J, 3] (root → self)
        seg = par - p                                     # [B, J, 3]
        seg_len2 = (seg ** 2).sum(-1).clamp_min(1e-12)    # [B, J]
        v = source_vert.unsqueeze(2) - p.unsqueeze(1)     # [B, N, J, 3]
        t = (v * seg.unsqueeze(1)).sum(-1) / seg_len2.unsqueeze(1)   # [B, N, J]
        t = t.clamp(0.0, 1.0)
        foot = p.unsqueeze(1) + t.unsqueeze(-1) * seg.unsqueeze(1)   # [B, N, J, 3]
        d2 = ((source_vert.unsqueeze(2) - foot) ** 2).sum(-1)        # [B, N, J]
        closest = d2.argmin(dim=-1)                       # [B, N]
        W_ref = F.one_hot(closest, num_classes=J).to(W.dtype)        # [B, N, J]
        return {'L_wref': F.mse_loss(W, W_ref)}

    @torch.no_grad()
    def weight_quality_metric(self, W, source_vert, faces):
        """#2 — GT-free skin-weight quality (Mesh2Animation static structure
        smoothness, Fig. 7). Same area-weighted 1-ring energy as the L_wlap
        loss, computed without grad as a diagnostic. Lower = smoother weights,
        mesh-resolution / topology independent. Returns scalar or None.

        NOTE: identical formula to weight_smoothness_loss — when --lambda_wlap
        is on this metric tracks the loss (not an independent gauge); it is
        most informative on baseline / L_wref-only runs.
        """
        return self._w_smoothness_area(W.float(), source_vert, faces)

    def inside_mesh_loss(self, joint_pos, vertices, vertex_normals):
        """Hinge penalty: joint protrudes outside mesh → penalize, inside → 0.

        Differentiable signed-distance approximation
        (Skin Tokens, Zhang 2026, "Bone-Mesh Containment" reward; supervised
        adaptation):

          For each joint J:
            1. find nearest vertex V_k on the mesh (argmin Euclidean dist).
            2. signed distance ≈ (J − V_k) · n_k where n_k is V_k's outward
               vertex normal.
               → > 0  if J is on the OUTER side of V_k  (joint outside)
               → < 0  if J is on the INNER side          (joint inside)
            3. hinge:  max(0, signed_dist)² .
            4. mean over batch and joints.

        Cost: B·J·N pairwise distances + 1 argmin + 2 gathers. For ICT
        N=11248, J=66, B=8: ~6M floats, trivial on GPU.

        Approximation note: at concavities the nearest-vertex-normal can flip
        sign incorrectly. For closed head meshes this is rare; if needed,
        replace with libigl signed_distance (igl.signed_distance) for exact
        SDF (CPU, slower).
        """
        # joint_pos: [B, J, 3], vertices: [B, N, 3], vertex_normals: [B, N, 3]
        B, J, _ = joint_pos.shape
        diff = joint_pos.unsqueeze(2) - vertices.unsqueeze(1)        # [B, J, N, 3]
        d2 = (diff ** 2).sum(-1)                                      # [B, J, N]
        nearest_idx = d2.argmin(dim=-1)                               # [B, J]
        idx_e = nearest_idx.unsqueeze(-1).expand(-1, -1, 3)
        V_near = torch.gather(vertices, 1, idx_e)                     # [B, J, 3]
        N_near = torch.gather(vertex_normals, 1, idx_e)               # [B, J, 3]
        signed_dist = ((joint_pos - V_near) * N_near).sum(dim=-1)     # [B, J]
        hinge = signed_dist.clamp(min=0.0)
        return {'L_inside': hinge.pow(2).mean()}


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
