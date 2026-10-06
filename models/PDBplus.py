"""Point-based deformation with target-conditioned control rotation and scale.

W: [batch, target_vertices, controls]; R: [batch, controls, 3, 3];
s: [batch, controls]. Displacement rows are transformed as s * C * R.T
before blending. No fixed deformation-mode dictionary is introduced.
"""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from models.NGBC import NeuralGeneralizedBarycentricCoordinate
from models.encoder import LinearEncoder
from utils.exp_utils import plateau_hat_points, from_6D_to_rotation_matrix_torch


class PDBplus(NeuralGeneralizedBarycentricCoordinate):
    """Reuse NGBC encoders and add a target-to-control transform predictor.

    Supported: out_type=0, use_shp=False. Source control outputs remain
    canonical displacements, including key_s/key_d in the legacy tuples.
    Target transforms are available through predict_target_parameters().
    """

    def __init__(self, opts=None, hid_dim=256, num_cage_vertices=512,
                 num_layers=4, device='cpu', **kwargs):
        if opts is None:
            opts = SimpleNamespace(in_type=1, out_type=0)
        if opts.out_type != 0:
            raise ValueError('PDBplus requires out_type=0 (control displacements).')
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

        # Same global LinearEncoder architecture as exp_z_model, but the
        # input contains only target neutral features (6 channels normally).
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
        """Reuse the repository's Gram-Schmidt convention (axes as rows).

        Safeguard zero/parallel axes; normal inputs use the existing converter
        unchanged. Columns vs rows are not swapped silently: application is
        explicitly C @ R.T, matching the public formulation.
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
        """Differentiable target prediction; cache the result during inference.

        Returns W [bs,Np,Nc], R [bs,Nc,3,3], s [bs,Nc].
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
        """Each control row c_i is mapped to s_i * c_i @ R_i.T."""
        return torch.einsum('bcij,bcj->bci', rotations, controls) * scales.unsqueeze(-1)

    def reconstruct(self, target_neutral, weights, controls, rotations, scales):
        transformed = self.transform_controls(controls, rotations, scales)
        return target_neutral + torch.einsum('bnc,bcd->bnd', weights, transformed)

    def _encode_controls(self, source_vert, deform_vert, source_norm, deform_norm):
        mask = plateau_hat_points(source_vert)
        _, pair = self.process_input(source_vert, deform_vert, source_norm, deform_norm, mask)
        z = self.exp_z_model(pair)
        controls = self.reshape_key_d(self.key_d_model(z), source_vert.shape[0])
        return z, controls, mask

    def forward(self, source_vert, deform_vert, source_norm, deform_norm,
                mesh_data=0, hat_mask=None, epoch=0, out_kw=False):
        """Self-retargeting with NGBC-compatible return tuple.

        key_s and key_d are canonical control displacements (before R/s),
        suitable for source subsampling consistency. pred_source is a mesh
        position, not a displacement. Neutral output is learned, not forced.
        """
        z, key_d, mask = self._encode_controls(
            source_vert, deform_vert, source_norm, deform_norm
        )
        _, key_s, _ = self._encode_controls(
            source_vert, source_vert, source_norm, source_norm
        )
        weights, rotations, scales = self.predict_target_parameters(source_vert, source_norm)
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
        z_d, key_d, _ = self._encode_controls(
            src_neu_vert, src_def_vert, src_neu_norm, src_def_norm
        )
        z_s, key_s, _ = self._encode_controls(
            src_neu_vert, src_neu_vert, src_neu_norm, src_neu_norm
        )
        weights, rotations, scales = self.predict_target_parameters(tgt_neu_vert, tgt_neu_norm)
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
        """Keep the original latent-editing API, using target control transforms."""
        rotations, scales = self._resolve_transforms(
            tgt_neu_vert, tgt_neu_norm, target_rotations, target_scales)
        controls = self.reshape_key_d(self.key_d_model(exp_z), exp_z.shape[0])
        return self.reconstruct(tgt_neu_vert, key_weight, controls, rotations, scales), controls

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
