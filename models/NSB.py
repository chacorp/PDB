import torch
import torch.nn as nn

from models.encoder import MLP, LinearEncoder


class NeuralSparseBlendpoint(nn.Module):
    """
        Neural Sparse Blendpoint

        Learnable, shared control-point modes `K` combined with an expression
        latent `z` through a K-conditioned FiLM coefficient predictor to infer
        signed mode coefficients `A`. `delta_C` is a fixed linear combination
        of `K` weighted by `A` -- see docs/control_mode_blending_plan.md.

        Public API:
            predict_weights(z)                    -> A [bs, N_K]
            synthesize(A)                          -> delta_C [bs, N_C, 3]
            forward(z, return_coefficients=False)  -> flat_delta_C [bs, 1, 3*N_C]
            forward(z, return_coefficients=True)   -> (flat_delta_C, A)
    """
    def __init__(self,
                 latent_dim,
                 num_controls,
                 num_modes=128,
                 num_layers=2,
                 init_std=0.01,
                 device='cpu',
                ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_controls = num_controls
        self.num_modes = num_modes
        self.num_layers = num_layers

        # learnable geometric modes: gradient flows both to feature extraction
        # (mode_encoder) and to synthesis (synthesize) -- never detached.
        self.modes = nn.Parameter(torch.randn(num_modes, num_controls, 3) * init_std)

        self.mode_encoder = MLP(
            [3 * num_controls, latent_dim, latent_dim],
            act='lrelu', nrm='layer',
        ).to(device)

        self.coefficient_model = LinearEncoder(
            in_dim=latent_dim, out_dim=1, hid_dim=latent_dim,
            num_layers=num_layers, use_residual=True,
            out_type='vertices', no_activation=True, use_pou=False,
            act='lrelu', nrm='layer',
        ).to(device)

    def predict_weights(self, z):
        """
        Args:
            z (torch.Tensor): [bs, 1, L] expression latent
        Returns:
            A (torch.Tensor): [bs, N_K] signed mode coefficients
        """
        mode_features = self.mode_encoder(self.modes.flatten(1))  # [N_K, L]
        mode_features = mode_features.unsqueeze(0).expand(z.shape[0], -1, -1)  # [bs, N_K, L]
        A = self.coefficient_model(mode_features, id_in=z).squeeze(-1)  # [bs, N_K]
        return A

    def synthesize(self, A):
        """
        Args:
            A (torch.Tensor): [bs, N_K] signed mode coefficients
        Returns:
            delta_C (torch.Tensor): [bs, N_C, 3] control point displacement
        """
        delta_C = torch.einsum('br,rcd->bcd', A, self.modes)
        return delta_C

    def forward(self, z, return_coefficients=False):
        """
        Args:
            z (torch.Tensor): [bs, 1, L] expression latent
        Returns:
            flat_delta_C (torch.Tensor): [bs, 1, 3*N_C]
            (flat_delta_C, A) if return_coefficients=True
        """
        bs = z.shape[0]
        A = self.predict_weights(z)
        delta_C = self.synthesize(A)
        flat_delta_C = delta_C.reshape(bs, 1, 3 * self.num_controls)

        if return_coefficients:
            return flat_delta_C, A
        return flat_delta_C
