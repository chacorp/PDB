"""PDB with target-conditioned per-control transform [R | s]."""

import math
from easydict import EasyDict

import torch
from torch import nn
from torch.nn import functional as F

from models.NGBC import NeuralGeneralizedBarycentricCoordinate
from models.encoder import LinearEncoder
from utils.exp_utils import plateau_hat_points, from_6D_to_rotation_matrix_torch


class PDBplus(NeuralGeneralizedBarycentricCoordinate):
    """
    controls: [B, M, 3] (out_type=0) or [B, M, 4, 3] (out_type=4, [R|t])
    """

    def __init__(self, opts=None, hid_dim=256, num_cage_vertices=512,
                 num_layers=4, device='cpu', **kwargs):
        if opts is None:
            opts = EasyDict(in_type=1, out_type=0)
        if opts.out_type not in (0, 4):
            raise ValueError('PDBplus requires out_type 0 (control displacements) '
                             'or 4 (LBS [R|t] per control).')
        if kwargs.get('use_shp', False):
            raise ValueError('PDBplus uses target conditioning; use_shp must be False.')
            
        super().__init__(
                opts=opts, 
                hid_dim=hid_dim,
                num_cage_vertices=num_cage_vertices,
                num_layers=num_layers, 
                device=device, 
                **kwargs
            )
        
        self.target_z_model = LinearEncoder(
            in_dim=self.in_dim, 
            out_dim=hid_dim, 
            num_layers=num_layers,
            out_type='global',
        )
        self.target_transform_model = LinearEncoder(
            in_dim=hid_dim, 
            out_dim=num_cage_vertices * 7,
            num_layers=num_layers, 
            out_type='global',
        )
        # Start close to identity rotation and unit scale. Nonzero weights
        # allow gradients into the target encoder on the first backward pass.
        head = self.target_transform_model.layer_out
        nn.init.normal_(head.weight, mean=0.0, std=1e-4)
        
        with torch.no_grad():
            head.bias.copy_(head.bias.new_tensor(
                [1., 0., 0., 0., 1., 0., 0.]
            ).repeat(num_cage_vertices))
        self.to(device)

    def _neutral_features(self, vertices, normals):
        """
        Args:
            vertices (torch.tensor): [B, N, 3]
            normals (torch.tensor): [B, N, 3]
        Returns:
            features (torch.tensor): [B, N, C]
        """
        features = vertices
        if self.in_type > 0:
            if normals is None:
                raise ValueError('Normals are required for in_type > 0.')
            features = torch.cat((features, normals), dim=-1)
        if self.in_type == 2:
            features = torch.cat((features, plateau_hat_points(vertices)), dim=-1)
        return features

    @staticmethod
    def rotation_from_6d(values):
        """
        Args:
            values (torch.tensor): [..., 6]
        Returns:
            R (torch.tensor): [..., 3, 3]
        """
        eps = 1e-6
        a1, a2 = values[..., :3], values[..., 3:]
        default_axis = torch.zeros_like(a1)
        default_axis[..., 0] = 1
        
        a1 = torch.where(a1.norm(dim=-1, keepdim=True) > eps, a1, default_axis)
        b1 = F.normalize(a1, dim=-1, eps=eps)
        perpendicular = a2 - (a2 * b1).sum(-1, keepdim=True) * b1
        # The coordinate axis least aligned with b1 cannot be parallel to it.
        fallback = F.one_hot(b1.abs().argmin(dim=-1), 3).to(values)
        
        a2 = torch.where(perpendicular.norm(dim=-1, keepdim=True) > eps, a2, fallback)
        return from_6D_to_rotation_matrix_torch(torch.cat((a1, a2), dim=-1), eps=eps)

    def predict_target_parameters(self, target_vert, target_norm):
        """
        Args:
            target_vert (torch.tensor): [B, N, 3]
            target_norm (torch.tensor): [B, N, 3]
        Returns:
            (weights, rotations, scales): [B, N, M], [B, M, 3, 3], [B, M]
        """
        features = self._neutral_features(target_vert, target_norm)
        weights = self.key_weight_model(features, N=self.NZ)
        target_z = self.target_z_model(features)
        raw = self.target_transform_model(target_z).reshape(
            target_vert.shape[0], self.num_cage_vertices, 7
        )
        rotations = self.rotation_from_6d(raw[..., :6])
        # Positive isotropic scale, approximately 1 at initialization.
        scales = (F.softplus(raw[..., 6]) / math.log(2.)).clamp_min(1e-6)
        return weights, rotations, scales

    @staticmethod
    def transform_controls(controls, rotations, scales):
        """
        Args:
            controls (torch.tensor): [B, M, 3]
            rotations (torch.tensor): [B, M, 3, 3]
            scales (torch.tensor): [B, M]
        Returns:
            controls (torch.tensor): [B, M, 3]
        """
        return torch.einsum('bcij,bcj->bci', rotations, controls) * scales.unsqueeze(-1)

    @staticmethod
    def conjugate_transforms(controls, rotations, scales):
        """
        Args:
            controls (torch.tensor): [B, M, 4, 3]
            rotations (torch.tensor): [B, M, 3, 3]
            scales (torch.tensor): [B, M]
        Returns:
            controls (torch.tensor): [B, M, 4, 3]
        """
        R = controls[..., :3, :].transpose(-1, -2)          # (B, M, 3, 3)
        t = controls[..., 3, :]                             # (B, M, 3)
        R_new = rotations @ R @ rotations.transpose(-1, -2)
        t_new = torch.einsum('bcij,bcj->bci', rotations, t) * scales.unsqueeze(-1)
        return torch.cat((R_new.transpose(-1, -2), t_new.unsqueeze(-2)), dim=-2)

    def reconstruct(self, target_neutral, weights, controls, rotations, scales):
        """
        Args:
            target_neutral (torch.tensor): [B, N, 3]
            weights (torch.tensor): [B, N, M]
            controls (torch.tensor): [B, M, 3] or [B, M, 4, 3]
            rotations (torch.tensor): [B, M, 3, 3]
            scales (torch.tensor): [B, M]
        Returns:
            vertices (torch.tensor): [B, N, 3]
        """
        if self.out_type == 4:
            transformed = self.conjugate_transforms(controls, rotations, scales)
            return self.apply_key_d(weights, transformed, target_neutral)
            
        transformed = self.transform_controls(controls, rotations, scales)
        return target_neutral + torch.einsum('bnc,bcd->bnd', weights, transformed)

    def _encode_controls(self, source_vert, deform_vert, source_norm, deform_norm):
        """
        Args:
            source_vert, deform_vert, source_norm, deform_norm (torch.tensor): [B, N, 3]
        Returns:
            (z, controls, mask): [B, 1, L], [B, M, 3] or [B, M, 4, 3], [B, N, 1]
        """
        mask = plateau_hat_points(source_vert)
        _, pair = self.process_input(
            source_vert, deform_vert, source_norm, deform_norm, mask
        )
        z = self.exp_z_model(pair)
        
        controls = self.reshape_key_d(self.key_d_model(z), source_vert.shape[0])
        
        return z, controls, mask

    def forward(self, source_vert, deform_vert, source_norm, deform_norm,
                mesh_data=0, hat_mask=None, epoch=0, out_kw=False):
        """
        Args:
            source_vert, deform_vert, source_norm, deform_norm (torch.tensor): [B, N, 3]
        Returns:
            out_kw=False: (pred_deformed [B, N, 3], recon_deformed [B, N, 3] or 0, 0,
                           z [B, 1, L], pred_source [B, N, 3], mask [B, N, 1],
                           weights [B, N, M], key_s, key_d)
            out_kw=True:  (pred_deformed, recon_deformed, 0, z, key_d, weights)
        """
        z, key_d, mask = self._encode_controls(
            source_vert, deform_vert, source_norm, deform_norm
        )
        _, key_s, _ = self._encode_controls(
            source_vert, source_vert, source_norm, source_norm
        )
        
        weights, rotations, scales = self.predict_target_parameters(
            source_vert, source_norm
        )
        
        pred_deformed = self.reconstruct(
            source_vert, weights, key_d, rotations, scales
        )
        pred_source = self.reconstruct(
            source_vert, weights, key_s, rotations, scales
        )
        recon_deformed = 0
        if self.is_train and self.use_exp_recon:
            recon_deformed = self.recon_exp_model[mesh_data](z).reshape_as(source_vert) + source_vert
        if out_kw:
            return pred_deformed, recon_deformed, 0, z, key_d, weights
        return pred_deformed, recon_deformed, 0, z, pred_source, mask, weights, key_s, key_d

    @torch.no_grad()
    def retarget(self, src_neu_vert, src_neu_norm, src_def_vert, src_def_norm,
                 tgt_neu_vert, tgt_neu_norm, mesh_data=0, out_kw=False, recon_out=True):
        """
        Args:
            src_neu_vert, src_neu_norm, src_def_vert, src_def_norm (torch.tensor): [B, Ns, 3]
            tgt_neu_vert, tgt_neu_norm (torch.tensor): [B, Nt, 3]
        Returns:
            out_kw=False: (pred_deformed [B, Nt, 3], pred_source [B, Nt, 3])
            out_kw=True:  (pred_deformed, pred_source, z_d [B, 1, L], key_d,
                           z_s [B, 1, L], key_s, weights [B, Nt, M])
        """
        z_d, key_d, _ = self._encode_controls(
            src_neu_vert, src_def_vert, src_neu_norm, src_def_norm
        )
        z_s, key_s, _ = self._encode_controls(
            src_neu_vert, src_neu_vert, src_neu_norm, src_neu_norm
        )
        
        
        weights, rotations, scales = self.predict_target_parameters(
            tgt_neu_vert, tgt_neu_norm
        )
        
        pred_deformed = self.reconstruct(
            tgt_neu_vert, weights, key_d, rotations, scales
        )
        pred_source = self.reconstruct(
            tgt_neu_vert, weights, key_s, rotations, scales
        )
        
        if out_kw:
            return pred_deformed, pred_source, z_d, key_d, z_s, key_s, weights
        return pred_deformed, pred_source

    def _resolve_transforms(self, target_vert, target_norm, rotations, scales):
        """
        Args:
            target_vert, target_norm (torch.tensor): [B, N, 3]
            rotations (torch.tensor or None): [B, M, 3, 3]
            scales (torch.tensor or None): [B, M]
        Returns:
            (rotations, scales): [B, M, 3, 3], [B, M]
        """
        if target_vert is None:
            raise ValueError('tgt_neu_vert is required for displacement reconstruction.')
        if (rotations is None) != (scales is None):
            raise ValueError('Pass both target_rotations and target_scales.')
            
        if rotations is None:
            if self.in_type > 0 and target_norm is None:
                raise ValueError('Pass cached target R/s or tgt_neu_norm; W alone is insufficient.')
                
            _, rotations, scales = self.predict_target_parameters(target_vert, target_norm)
        return rotations, scales

    @torch.no_grad()
    def retarget_animation(self, 
            src_neu_vert, 
            src_neu_norm,
            src_def_vert, 
            src_def_norm,
            key_weight, 
            tgt_neu_vert=None, 
            return_source=False, 
            *,
            tgt_neu_norm=None, 
            target_rotations=None,
            target_scales=None
        ):
        """
        Args:
            src_neu_vert, src_neu_norm, src_def_vert, src_def_norm (torch.tensor): [B, Ns, 3]
            key_weight (torch.tensor): [B, Nt, M]
            tgt_neu_vert, tgt_neu_norm (torch.tensor): [B, Nt, 3]
            target_rotations (torch.tensor): [B, M, 3, 3]
            target_scales (torch.tensor): [B, M]
            
        Returns:
            (output [B, Nt, 3], key_d) (+ neutral [B, Nt, 3] if return_source)
        """
        rotations, scales = self._resolve_transforms(
            tgt_neu_vert, tgt_neu_norm, target_rotations, target_scales
        )
        _, key_d, _ = self._encode_controls(
            src_neu_vert, src_def_vert, src_neu_norm, src_def_norm
        )
        output = self.reconstruct(
            tgt_neu_vert, key_weight, key_d, rotations, scales
        )
        
        if return_source:
            _, key_s, _ = self._encode_controls(
                src_neu_vert, src_neu_vert, src_neu_norm, src_neu_norm
            )
            neutral = self.reconstruct(
                tgt_neu_vert, key_weight, key_s, rotations, scales
            )
            return output, key_d, neutral
        return output, key_d

    @torch.no_grad()
    def blendshape(self, 
            exp_z,
            key_weight, 
            tgt_neu_vert=None,
            src_neu_vert=None, 
            src_neu_norm=None, 
            *, 
            tgt_neu_norm=None,
            target_rotations=None,
            target_scales=None
        ):
        """
        Args:
            exp_z (torch.tensor): [B, 1, L]
            key_weight (torch.tensor): [B, Nt, M]
            tgt_neu_vert, tgt_neu_norm (torch.tensor): [B, Nt, 3]
            target_rotations (torch.tensor): [B, M, 3, 3]
            target_scales (torch.tensor): [B, M]
        Returns:
            (vertices [B, Nt, 3], controls)
        """
        rotations, scales = self._resolve_transforms(
            tgt_neu_vert, tgt_neu_norm, target_rotations, target_scales
        )
        controls = self.reshape_key_d(self.key_d_model(exp_z), exp_z.shape[0])
        return self.reconstruct(
            tgt_neu_vert, key_weight, controls, rotations, scales
        ), controls

    def cyclic_loss(self, *args, **kwargs):
        raise NotImplementedError(
            'PDBplus does not reuse NGBC cyclic_loss: it omits target R/s. '
            'Disable use_cyclic_loss until a transform-aware cycle objective is added.')

    def log_parameter_num(self):
        lines = ['========< PDBplus >========']
        for name, module in self.named_children():
            lines.append(f'[{name}]: {self.count_parameters(module)}')
        lines.append(f'[total]: {self.count_parameters(self)}')
        return '\n'.join(lines) + '\n'
