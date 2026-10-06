"""
Standalone rotation-canonicalization module (T-Net-style): given a possibly
rotated mesh, estimates the rotation matrix so the input can be aligned back
toward its canonical orientation before being consumed by downstream models
(key_weight_model / key_d branch in NGBC).

Reuses LinearEncoder as the backbone (mirrors how PointNet's T-Net is just a
mini point-cloud encoder + global pooling + a matrix regression head), and
the existing Zhou et al. 6D representation (`from_6D_to_rotation_matrix_torch`)
so the output is a valid rotation matrix by construction -- no orthogonality
regularizer needed, unlike vanilla T-Net.

This module is intentionally NOT embedded inside LinearEncoder itself: it is
meant to be instantiated once per independently-robustified branch (one
instance in front of key_weight_model, a separate instance in front of the
key_d branch), each estimating its own rotation from its own input.
"""
import torch
import torch.nn as nn

from models.encoder import LinearEncoder
from utils.exp_utils import from_6D_to_rotation_matrix_torch


class RotationEstimator(nn.Module):
    """
    Estimates R_hat ~= R_applied^{-1} (= R_applied^T, since R_applied is a
    rotation matrix) from a (possibly rotated) mesh, using the same
    row-vector convention as the rest of the codebase (verts_rot = verts @ R,
    see dataloader_CBD.py's random_rotate_points / utils/exp_utils.py).

    Aligning back to canonical: verts_aligned = verts_rot @ R_hat
    """
    def __init__(self, in_dim=6, hid_dim=128, num_layers=4):
        super().__init__()
        self.encoder = LinearEncoder(
            in_dim=in_dim, hid_dim=hid_dim, out_dim=6,
            num_layers=num_layers, out_type='global',
        )

    def forward(self, x_in):
        """
        Args:
            x_in: (B, N, in_dim) -- raw (possibly rotated) vertex features,
                  e.g. vertex position concatenated with vertex normal.
        Returns:
            R_hat: (B, 3, 3)
        """
        out = self.encoder(x_in)       # (B, 1, 6)
        out = out.squeeze(1)           # (B, 6)
        return from_6D_to_rotation_matrix_torch(out)  # (B, 3, 3)

    @staticmethod
    def align(verts, R_hat):
        """verts: (B, N, 3), R_hat: (B, 3, 3) -> (B, N, 3)"""
        return torch.bmm(verts, R_hat)
