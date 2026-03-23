"""
utils/rig_loader.py
===================
Load and parse Maya rig export data into tensors for HierarchicalLBS.

Expected files (produced by maya_rig/export_rig.py):
    maya_rig/rig_info.json          -- joint hierarchy + bindPreMatrix
    maya_rig/skin_weights.npy       -- [5223, 84] float32  (MF topology)
    maya_rig/skin_weights_biwi.npy  -- [2560, 84] (produced by transfer_weights.py)
    maya_rig/skin_weights_voca.npy  -- [3525, 84]

Convention note
---------------
Maya stores matrices in ROW-major format (translation in last row):
    [R | 0]
    [t | 1]
For standard column-major LBS  (G @ [v;1])  we must TRANSPOSE:
    B_inv_col = B_inv_maya.T
Sanity check: B_inv_col @ [bind_pos; 1] == [0,0,0,1]  ✓ (verified)

Usage
-----
    from utils.rig_loader import load_rig, rig_to_logit_W

    rig = load_rig()                   # loads from maya_rig/ by default
    # rig.B_inv        [J, 4, 4]  column-major inverse bind matrices
    # rig.parent_idx   [J]        int64, -1 for root
    # rig.bind_pos     [J, 3]     joint world positions at bind pose
    # rig.joint_names  list[str]  length J
    # rig.W_init       dict  topology -> [V, J] float32, rows sum to 1

    logit_W = rig_to_logit_W(rig.W_init['mf'])   # [N, J] for nn.Parameter init
"""

import os
import json
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RigData:
    joint_names : List[str]           # [J]
    parent_idx  : torch.Tensor        # [J]      int64, -1 = root
    bind_pos    : torch.Tensor        # [J, 3]   float32
    B_inv       : torch.Tensor        # [J, 4, 4] float32, column-major (already transposed)
    W_init      : Dict[str, torch.Tensor] = field(default_factory=dict)
    # key -> [V, J] float32, rows sum to 1.0


def load_rig(
    rig_dir: str = 'maya_rig',
    topologies: Optional[List[str]] = None,
    device: str = 'cpu',
) -> RigData:
    """
    Load Maya rig export into RigData.

    Args:
        rig_dir    : directory containing rig_info.json and skin_weights*.npy
        topologies : which skin weight files to load (default: all available)
        device     : torch device for returned tensors
    """
    if topologies is None:
        topologies = ['mf', 'biwi', 'voca']

    # ------------------------------------------------------------------
    # 1. Parse rig_info.json
    # ------------------------------------------------------------------
    rig_path = os.path.join(rig_dir, 'rig_info.json')
    if not os.path.exists(rig_path):
        raise FileNotFoundError(
            f"rig_info.json not found at '{rig_path}'. "
            "Run maya_rig/export_rig.py inside Maya first."
        )

    with open(rig_path) as f:
        rig_info = json.load(f)

    joints = rig_info['joints']   # list of dicts, topologically sorted
    J = len(joints)

    joint_names  = [None] * J
    parent_arr   = np.full(J, -1,         dtype=np.int64)
    bind_pos_arr = np.zeros((J, 3),       dtype=np.float32)
    B_inv_arr    = np.zeros((J, 4, 4),    dtype=np.float32)

    for jdata in joints:
        idx = jdata['index']
        joint_names[idx]    = jdata['name']
        parent_arr[idx]     = jdata['parent_index']
        bind_pos_arr[idx]   = jdata['bind_world_pos']
        # Maya row-major → column-major via transpose
        B_inv_arr[idx]      = np.array(jdata['bind_pre_matrix'], dtype=np.float32).T

    # Validate topological order
    for idx, p in enumerate(parent_arr):
        if p != -1 and p >= idx:
            raise ValueError(
                f"Joint '{joint_names[idx]}' (idx={idx}) has parent_idx={p} >= itself. "
                "Joints must be in topological order (parent before child)."
            )

    # ------------------------------------------------------------------
    # 2. Load skin weights
    # ------------------------------------------------------------------
    weight_files = {
        'mf'  : os.path.join(rig_dir, 'skin_weights_mf.npy'),
        'biwi': os.path.join(rig_dir, 'skin_weights_biwi.npy'),
        'voca': os.path.join(rig_dir, 'skin_weights_voca.npy'),
    }

    W_init = {}
    for topo in topologies:
        if topo not in weight_files:
            raise KeyError(f"Unknown topology '{topo}'. Available: {list(weight_files)}")
        fpath = weight_files[topo]
        if not os.path.exists(fpath):
            print(f"[rig_loader] WARNING: '{fpath}' not found — skipping '{topo}'")
            continue
        W_np = np.load(fpath).astype(np.float32)   # [V, J]
        if W_np.shape[1] != J:
            raise ValueError(
                f"skin_weights for '{topo}': expected {J} columns, got {W_np.shape[1]}"
            )
        # Renormalize rows (should already be 1.0, but enforce)
        row_sums = W_np.sum(axis=1, keepdims=True)
        W_np /= row_sums + 1e-12
        W_init[topo] = torch.tensor(W_np, dtype=torch.float32).to(device)

    return RigData(
        joint_names=joint_names,
        parent_idx=torch.tensor(parent_arr,   dtype=torch.int64).to(device),
        bind_pos=torch.tensor(bind_pos_arr,   dtype=torch.float32).to(device),
        B_inv=torch.tensor(B_inv_arr,         dtype=torch.float32).to(device),
        W_init=W_init,
    )


def rig_to_logit_W(W_init: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert skin weights to logit (log) space for nn.Parameter initialization.

    In HierarchicalLBS:
        W = softmax(logit_W_base + delta_W, dim=-1)
    where logit_W_base = rig_to_logit_W(W_init) is fixed,
    and delta_W is a learnable per-identity offset (initialized to 0).

    Args:
        W_init  : [N, J] skin weights, rows sum to ~1
    Returns:
        logit_W : [N, J]
    """
    return torch.log(W_init + eps)


def sanity_check(rig: RigData) -> None:
    """Verify B_inv: B_inv @ [bind_pos; 1] should equal [0, 0, 0, 1] for all joints."""
    J = len(rig.joint_names)
    pos_h = torch.cat([rig.bind_pos, torch.ones(J, 1)], dim=-1)  # [J, 4]
    # B_inv [J,4,4] @ pos_h [J,4,1] → [J,4]
    result = torch.bmm(rig.B_inv, pos_h.unsqueeze(-1)).squeeze(-1)  # [J, 4]
    expected = torch.zeros_like(result)
    expected[:, 3] = 1.0   # expected: [0, 0, 0, 1] for every joint
    err = (result - expected).abs()
    max_err = err.max().item()
    print(f"[sanity_check] B_inv @ [bind_pos;1]: max_abs_err = {max_err:.2e}  (should be ~0)")
    if max_err > 1e-3:
        print(f"  WARNING: large error. Check coordinate conventions.")
    else:
        print(f"  OK — all joints verified.")


def print_summary(rig: RigData) -> None:
    J = len(rig.joint_names)
    print(f"RigData: {J} joints")
    print(f"  B_inv   : {tuple(rig.B_inv.shape)}")
    print(f"  parent_idx sample: {rig.parent_idx[:5].tolist()}")
    for i in range(min(6, J)):
        p = rig.parent_idx[i].item()
        pname = rig.joint_names[p] if p >= 0 else 'None'
        print(f"  [{i:2d}] {rig.joint_names[i]:35s} parent=[{p:2d}] {pname}")
    if J > 6:
        print(f"  ... ({J-6} more)")
    for topo, W in rig.W_init.items():
        rs = W.sum(dim=-1)
        print(f"  W_init['{topo}']: {tuple(W.shape)}  "
              f"sum: min={rs.min():.4f} max={rs.max():.4f}  "
              f"nnz/row: {(W > 0).float().sum(-1).mean():.1f}")


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    rig = load_rig('maya_rig', topologies=['mf'])
    print_summary(rig)
    sanity_check(rig)
