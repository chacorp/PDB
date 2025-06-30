import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
abs_path = str(Path(__file__).parents[1].absolute())
diffusionnet_path=f'{abs_path}/third_party/diffusion-net/src'
if not diffusionnet_path in sys.path:
    sys.path+=[diffusionnet_path]
import diffusion_net

class DiffusionNetBlock2(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_width, mlp_hidden_dims,
                 dropout=True, 
                 diffusion_method='spectral',
                 with_gradient_features=True, 
                 with_gradient_rotations=True):
        super(DiffusionNetBlock2, self).__init__()

        # Specified dimensions
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.C_width, method=diffusion_method)
        
        self.MLP_C = 2*self.C_width
      
        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(self.C_width, with_gradient_rotations=self.with_gradient_rotations)
            self.MLP_C += self.C_width
        
        # MLPs
        self.mlp = MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)


    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY, return_residual=False):

        # Manage dimensions
        B = x_in.shape[0] # batch dimension
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width))
        
        # Diffusion block 
        x_diffuse = self.diffusion(x_in, L, mass, evals, evecs)
        
        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            # Handle gradX/gradY as batched tensors or lists
            if isinstance(gradX, torch.Tensor):
                gradX = gradX.unsqueeze(0).repeat(B, 1, 1)
                gradY = gradY.unsqueeze(0).repeat(B, 1, 1)

            # Batch matrix multiplication: (B,N,N) @ (B,N,C) -> (B,N,C)
            x_gradX = torch.bmm(gradX, x_diffuse)
            x_gradY = torch.bmm(gradY, x_diffuse)
            x_grad = torch.stack([x_gradX, x_gradY], dim=-1)
            
            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad) 

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        
        # Apply the mlp
        x0_out = self.mlp(feature_combined)


        if return_residual:
            return x0_out, x0_out + x_in
            
        # Skip connection
        x0_out = x0_out + x_in
        return x0_out


class AdaINDiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """
    def __init__(self, C_width, mlp_hidden_dims, 
                 ID_dims=128, 
                 dropout=True, 
                 diffusion_method='spectral',
                 with_gradient_features=True, 
                 with_gradient_rotations=True):
        super(AdaINDiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = diffusion_net.LearnedTimeDiffusion(self.C_width, method=diffusion_method)
        self.gradient_features = diffusion_net.SpatialGradientFeatures(self.C_width, with_gradient_rotations=self.with_gradient_rotations)
        
        self.MLP_C = 3 * self.C_width
        
        # MLPs (activation included)
        self.mlp = diffusion_net.MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)
        
        # AdaIN
        self.norms = nn.LayerNorm(self.C_width)
        self.adain_s = diffusion_net.MiniMLP([ID_dims] + [self.C_width//2] + [self.C_width], dropout=self.dropout)
        self.adain_m = diffusion_net.MiniMLP([ID_dims] + [self.C_width//2] + [self.C_width], dropout=self.dropout)
        
        # self.exp_mlp = diffusion_net.MiniMLP([self.C_width+EXP_dims] + [self.C_width] + [self.C_width*2], dropout=self.dropout)
        # self.exp_w = nn.Linear(self.C_width, self.C_width)
        # self.exp_b = nn.Linear(self.C_width, self.C_width)
        
    def forward(self, id_in, x_in, mass, L, evals, evecs, gradX, gradY):
        
        # Manage dimensions
        B = x_in.shape[0] # batch dimension
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width))
        
        # Diffusion block 
        x_diffuse = self.diffusion(x_in, L, mass, evals, evecs)
        if type(gradX) != list:
            gradX = [gradX for i in range(B)]
            gradY = [gradY for i in range(B)]
            
        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_grads = [] # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching
            for b in range(B):
                # gradient after diffusion
                x_gradX = torch.mm(gradX[b], x_diffuse[b,...])
                x_gradY = torch.mm(gradY[b], x_diffuse[b,...])
                # x_gradX = torch.mm(gradX[b, ...], x_diffuse[b,...])
                # x_gradY = torch.mm(gradY[b, ...], x_diffuse[b,...])

                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
            x_grad = torch.stack(x_grads, dim=0)

            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad) 

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)
        
        # Apply the mlp
        x0_out = self.mlp(feature_combined)
        x0_out = self.norms(x0_out)
        
        # Skip connection
        x0_out = x0_out + x_in

        ## AdaIN ---------------------
        x0_out = x0_out * self.adain_s(id_in) + self.adain_m(id_in)
        ## ---------------------------
        
        return x0_out