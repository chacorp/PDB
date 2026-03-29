"""
models/diffusion_edd.py
=======================
DiffusionNet Error-Driven Displacement (EDD) network.

Given:
    jac_feat  [B, V, 7]   Jacobian features from compute_jacobian_features()
                           (deviation from neutral: [det(U)-1, u00-1, u11-1, u22-1, u01, u02, u12])
    z_exp     [B, z_dim]  expression code (from HierarchicalLBS encoder)

Outputs:
    disp      [B, V, 3]   per-vertex residual displacement (add to LBS output)

Pipeline:
    1. Broadcast z_exp over vertices → [B, V, z_dim]
    2. Concatenate with jac_feat     → [B, V, 7+z_dim]
    3. DiffusionNet (outputs_at='faces') → per-face Jacobian [B, F, 9]
    4. Poisson solve (deformation_gradient.apply) → [B, V, 3] displacement

Topology-agnostic: DiffusionNet weights are shared; operators (DiffusionNet spectral
operators + Poisson Transfer operators) are precomputed per topology via
register_topology() and swapped at runtime via set_topology().

Usage:
    edd = DiffusionNetEDD(z_dim=256)
    edd.register_topology('mf', verts_np, faces_np, op_cache_dir='utils/diffusion_ops')
    edd.set_topology('mf', device)
    disp = edd(jac_feat, z_exp)   # [B, V, 3]
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

__abs_path__ = str(Path(__file__).parents[1].absolute())
__diffnet_path__ = os.path.join(__abs_path__, 'third_party', 'diffusion-net', 'src')
for _p in [__abs_path__, __diffnet_path__]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import diffusion_net
from utils.deformation_transfer import deformation_gradient as DGrad


class DiffusionNetEDD(nn.Module):
    """
    Error-Driven Displacement via DiffusionNet + Poisson solve.

    Args:
        z_dim:        dimension of expression code z_exp
        hid_channels: DiffusionNet hidden channel width  (default 128)
        n_blocks:     number of DiffusionNet blocks       (default 4)
        k_eig:        Laplacian eigenvectors per topology  (default 128)
    """

    def __init__(
        self,
        z_dim: int,
        hid_channels: int = 128,
        n_blocks: int = 4,
        k_eig: int = 128,
    ):
        super().__init__()

        self.z_dim  = z_dim
        self.k_eig  = k_eig
        C_in  = 7 + z_dim   # Jacobian features + expression code
        C_out = 9            # flattened 3×3 Jacobian per face

        self.dfn = diffusion_net.DiffusionNet(
            C_in=C_in,
            C_out=C_out,
            C_width=hid_channels,
            N_block=n_blocks,
            outputs_at='faces',
            with_gradient_features=True,
        )

        # Per-topology DiffusionNet spectral operators
        # key → dict(mass, L, evals, evecs, gradX, gradY, faces)
        self._dfn_ops: dict = {}

        # Per-topology Poisson solve operators
        # key → (solver, idxs, vals, rhs)   from utils.deformation_transfer.Transfer
        self._poisson_ops: dict = {}

        # Currently active topology (set by set_topology)
        self._active_dfn: dict = {}
        self._active_poisson: tuple = ()
        self.active_topo: str = ''

    # ------------------------------------------------------------------
    # Operator registration / topology management
    # ------------------------------------------------------------------

    def register_topology(
        self,
        topo_key: str,
        verts_np: np.ndarray,
        faces_np: np.ndarray,
        op_cache_dir: str = 'utils/diffusion_ops',
        cuda_device: str = 'cuda:0',
    ) -> None:
        """
        Precompute and register DiffusionNet + Poisson operators for one topology.

        Args:
            topo_key:    identifier string, e.g. 'mf', 'biwi', 'voca'
            verts_np:    [V, 3] float32 — neutral-pose vertices
            faces_np:    [F, 3] int32   — triangle indices
            op_cache_dir: directory for DiffusionNet .npz cache
            cuda_device:  CUDA device string for Poisson Transfer (e.g. 'cuda:0')
        """
        V_t = torch.tensor(verts_np, dtype=torch.float32)
        F_t = torch.tensor(faces_np, dtype=torch.int32)

        # ---- DiffusionNet spectral operators ----
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
            V_t, F_t, k_eig=self.k_eig, op_cache_dir=op_cache_dir
        )
        self._dfn_ops[topo_key] = dict(
            mass=mass,
            L=L,
            evals=evals,
            evecs=evecs,
            gradX=gradX,
            gradY=gradY,
            faces=F_t.long(),         # [F, 3]
        )

        # ---- Poisson solve operators via Transfer ----
        try:
            from utils.deformation_transfer import Transfer
            from utils.nfr_utils import Mesh
            neutral = Mesh(verts_np.astype(np.float64), faces_np.astype(np.int32))
            transfer = Transfer(neutral, neutral, area=True, device=cuda_device)
            # Convert cupy idxs/vals → torch tensors for torch_sparse spmm
            # Use numpy roundtrip (init-time only, so CPU↔GPU copy is acceptable)
            import cupy as _cupy
            idxs_t = torch.tensor(_cupy.asnumpy(transfer.idxs), dtype=torch.long).to(cuda_device)
            vals_t = torch.tensor(_cupy.asnumpy(transfer.vals), dtype=torch.float32).to(cuda_device)
            self._poisson_ops[topo_key] = (
                transfer.solver,
                idxs_t,
                vals_t,
                transfer.ATarea,   # scipy sparse, used only for .shape (3F, V)
            )
            print(
                f"[DiffusionNetEDD] Registered '{topo_key}': "
                f"V={verts_np.shape[0]}, F={faces_np.shape[0]}, "
                f"Poisson OK"
            )
        except Exception as e:
            print(
                f"[DiffusionNetEDD] WARNING: Poisson operators failed for '{topo_key}': {e}\n"
                f"  Forward will skip Poisson solve and return zero displacement."
            )
            self._poisson_ops[topo_key] = None

    def set_topology(self, topo_key: str, device: torch.device) -> None:
        """
        Activate a registered topology for the next forward() call(s).
        Move DiffusionNet operators to the given device.

        Args:
            topo_key: registered topology key
            device:   torch device
        """
        assert topo_key in self._dfn_ops, (
            f"Topology '{topo_key}' not registered. "
            f"Call register_topology() first. Available: {list(self._dfn_ops)}"
        )
        self.active_topo = topo_key

        ops = self._dfn_ops[topo_key]
        # Move sparse tensors to device; keep non-tensor entries as-is
        def _to(v):
            return v.to(device) if isinstance(v, torch.Tensor) else v

        self._active_dfn = {k: _to(v) for k, v in ops.items()}
        self._active_poisson = self._poisson_ops.get(topo_key)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        jac_feat: torch.Tensor,
        z_exp:    torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            jac_feat: [B, V, 7]   Jacobian features (neutral-relative stretch)
            z_exp:    [B, z_dim]  expression code (optional, skipped if z_dim=0)

        Returns:
            disp: [B, V, 3]  residual displacement to add to LBS output
        """
        assert self.active_topo, "Call set_topology() before forward()."

        B, V, _ = jac_feat.shape
        device   = jac_feat.device

        # 1. Condition: broadcast z_exp and concatenate → [B, V, 7+z_dim]
        if self.z_dim > 0 and z_exp is not None:
            x = torch.cat([jac_feat, z_exp.unsqueeze(1).expand(-1, V, -1)], dim=-1)
        else:
            x = jac_feat

        # 2. Build per-batch operator lists (same topology for entire batch)
        ops = self._active_dfn
        L    = ops['L'].to(device)
        gradX = ops['gradX'].to(device)
        gradY = ops['gradY'].to(device)

        batch_L     = [L] * B
        batch_mass  = ops['mass'].to(device).unsqueeze(0).expand(B, -1)
        batch_evals = ops['evals'].to(device).unsqueeze(0).expand(B, -1)
        batch_evecs = ops['evecs'].to(device).unsqueeze(0).expand(B, -1, -1)
        batch_gradX = [gradX] * B
        batch_gradY = [gradY] * B
        faces_batch = ops['faces'].to(device).unsqueeze(0)  # [1, F, 3]

        # 3. DiffusionNet → per-face Jacobian [B, F, 9]
        g_pred = self.dfn(
            x, batch_mass,
            L=batch_L,
            evals=batch_evals,
            evecs=batch_evecs,
            gradX=batch_gradX,
            gradY=batch_gradY,
            faces=faces_batch,
        )

        # 4. Poisson solve → [B, V, 3]
        if self._active_poisson is None:
            # Poisson ops not available (no CUDA / cupy): return zeros
            return torch.zeros(B, V, 3, device=device, dtype=jac_feat.dtype)

        solver, idxs, vals, rhs = self._active_poisson
        disp = DGrad.apply(g_pred, solver, idxs, vals, rhs.shape)
        disp = disp - disp.mean(dim=[0, 1], keepdim=True)

        return disp
