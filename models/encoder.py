import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import sys
from pathlib import Path
abs_path = str(Path(__file__).parents[1].absolute())
diffusionnet_path=f'{abs_path}/third_party/diffusion-net/src'
siren_path = f'{abs_path}/third_party/siren'

if not diffusionnet_path in sys.path:
    sys.path+=[diffusionnet_path]
import diffusion_net

if not siren_path in sys.path:
    sys.path+=[siren_path]
import modules
from meta_modules import HyperNetwork

# from torchmeta.modules import (MetaModule, MetaSequential)
from models.blocks import *

class BaseDiffusionNetEncoder(nn.Module):
    # reference: https://github.com/dafei-qin/NFR_pytorch/blob/e3553faa77f65240ec20167aec6e814473233890/mymodel.py#L17
    def __init__(self, in_shape=6, out_shape=128, hid_shape=256, pre_computes=None, N_block=4, outputs_at='global_mean', with_grad=True, last_activation=None):
        super(BaseDiffusionNetEncoder, self).__init__()
        
        self.dfn = diffusion_net.DiffusionNet(
            C_in=in_shape, 
            C_out=out_shape, 
            C_width=hid_shape, 
            N_block=N_block, 
            outputs_at=outputs_at, 
            with_gradient_features=with_grad,
            last_activation=last_activation,
        )
        if pre_computes:
            self.update_precomputes(pre_computes)
        else:
            print("[DiffusionNet] causion: no pre_computes provided!")

    def update_precomputes(self, pre_computes):
        if len(pre_computes[0].shape) > 1:
            self.mass = nn.Parameter(pre_computes[0].squeeze(0), requires_grad=False)

            self.L_ind = nn.Parameter(pre_computes[1]._indices()[1:], requires_grad=False)
            self.L_val = nn.Parameter(pre_computes[1]._values(), requires_grad=False)
            self.L_size = pre_computes[1].size()[1:]
            self.evals = nn.Parameter(pre_computes[2].squeeze(0), requires_grad=False)
            self.evecs = nn.Parameter(pre_computes[3].squeeze(0), requires_grad=False)
            self.grad_X_ind  = nn.Parameter(pre_computes[4]._indices()[1:], requires_grad=False)
            self.grad_X_val  = nn.Parameter(pre_computes[4]._values(),  requires_grad=False)
            self.grad_X_size = pre_computes[4].size()[1:]
            self.grad_Y_ind  = nn.Parameter(pre_computes[5]._indices()[1:], requires_grad=False)
            self.grad_Y_val  = nn.Parameter(pre_computes[5]._values(),  requires_grad=False)
            self.grad_Y_size = pre_computes[5].size()[1:]

            self.faces = nn.Parameter(pre_computes[6].long(), requires_grad=False)
            
        else:
            self.mass = nn.Parameter(pre_computes[0], requires_grad=False)

            self.L_ind = nn.Parameter(pre_computes[1]._indices(), requires_grad=False)
            self.L_val = nn.Parameter(pre_computes[1]._values(), requires_grad=False)
            self.L_size = pre_computes[1].size()
            self.evals = nn.Parameter(pre_computes[2], requires_grad=False)
            self.evecs = nn.Parameter(pre_computes[3], requires_grad=False)
            self.grad_X_ind = nn.Parameter(pre_computes[4]._indices(), requires_grad=False)
            self.grad_X_val = nn.Parameter(pre_computes[4]._values(), requires_grad=False)
            self.grad_X_size =pre_computes[4].size()
            self.grad_Y_ind = nn.Parameter(pre_computes[5]._indices(), requires_grad=False)
            self.grad_Y_val = nn.Parameter(pre_computes[5]._values(), requires_grad=False)
            self.grad_Y_size = pre_computes[5].size()

            self.faces = nn.Parameter(pre_computes[6].unsqueeze(0).long(), requires_grad=False)

    def forward(self,
                inputs,
                batch_mass=None,
                batch_L_val=None,
                batch_evals=None,
                batch_evecs=None,
                batch_gradX=None,
                batch_gradY=None
               ):
        """
        Args:
            inputs (torch.tensor): [vertex position, vertex normal]
        """
        self.L = torch.sparse_coo_tensor(self.L_ind, self.L_val, self.L_size, device=inputs.device)
        batch_size = inputs.shape[0]
        if batch_mass is not None:
            batch_L = [torch.sparse_coo_tensor(self.L_ind, batch_L_val[i], self.L_size, device=inputs.device) for i in range(len(batch_L_val))]
        else:
            batch_L = [self.L for b in range(batch_size)]
            if batch_size > 1:
                batch_mass = self.mass.expand(batch_size, -1)
                batch_evals = self.evals.expand(batch_size, -1)
                batch_evecs = self.evecs.expand(batch_size, -1, -1)
            else:
                batch_mass = self.mass.unsqueeze(0).expand(batch_size, -1)
                batch_evals = self.evals.unsqueeze(0).expand(batch_size, -1)
                batch_evecs = self.evecs.unsqueeze(0).expand(batch_size, -1, -1)

        if batch_gradX is not None:
            gradX = [torch.sparse_coo_tensor(self.grad_X_ind, gX, self.grad_X_size, device=inputs.device) for gX in batch_gradX ]
            gradY = [torch.sparse_coo_tensor(self.grad_Y_ind, gY, self.grad_Y_size, device=inputs.device) for gY in batch_gradY ]
        else:
            gradX = [torch.sparse_coo_tensor(self.grad_X_ind, self.grad_X_val, self.grad_X_size, device=inputs.device) for b in range(batch_size)]
            gradY = [torch.sparse_coo_tensor(self.grad_Y_ind, self.grad_Y_val, self.grad_Y_size, device=inputs.device) for b in range(batch_size)]
            
        ## device???
        outputs = self.dfn(inputs, batch_mass, L=batch_L, evals=batch_evals, evecs=batch_evecs, gradX=gradX, gradY=gradY, faces=self.faces)
        return outputs
    
class MLP(nn.Sequential):
    '''
    A simple MLP with configurable hidden layer sizes.
    '''
    def __init__(self, layer_sizes, num_gn=32, dropout=False, act='relu', name="MLP", p=.5):
        super(MLP, self).__init__()

        if act == 'none':
            self.act = lambda x: x
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU()
            
        self.N_layers = len(layer_sizes)
        layers = []
        norms = []
        for i in range(self.N_layers-1):
            if dropout and i > 0:
                layers.append(nn.Dropout(p=p))    
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
        for j in range(self.N_layers-2): # no norm for the last layer
            # norms.append(nn.GroupNorm(num_gn, layer_sizes[j+1]))
            norms.append(nn.InstanceNorm1d(layer_sizes[j+1]))
            
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        
    def forward(self, x):
        for i in range(self.N_layers-2):
            tmp = self.layers[i](x)#.transpose(-1, -2)
            out = self.act(self.norms[i](tmp))#.transpose(-1, -2)
        out = self.layers[-1](out)
        return out
        
class AdaINDiffusionNetEncoder(nn.Module):
    def __init__(self, C_in, C_out, C_width=128, id_dim=128,
                 pre_computes=None, N_block=4, 
                 last_activation=None, outputs_at='vertices', 
                 mlp_hidden_dims=None, dropout=True, 
                 with_gradient_features=True, 
                 with_gradient_rotations=True, 
                 diffusion_method='spectral'
                 ):
        super(AdaINDiffusionNetEncoder, self).__init__()
        """
        Construct a DiffusionNet with AdaIN.

        Parameters:
            C_in (int):                     input dimension 
            C_out (int):                    output dimension 
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient. Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """


        ## Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ['vertices', 'edges', 'faces', 'global_mean']: raise ValueError("invalid setting for outputs_at")

        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        
        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations
        
        ## Set up the network
        self.adain_in = nn.Sequential(
            nn.Linear(id_dim,      id_dim//2), nn.ReLU(), nn.LayerNorm(id_dim//2), 
            nn.Linear(id_dim//2, id_dim//2), nn.ReLU(), nn.LayerNorm(id_dim//2), 
            nn.Linear(id_dim//2, id_dim),
        )
        self.act = nn.ReLU()

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
       
        # DiffusionNet blocks
        self.blocks = nn.ModuleList()
        for i_block in range(self.N_block):
            block = AdaINDiffusionNetBlock(C_width = C_width,
                                    mlp_hidden_dims = mlp_hidden_dims, # list
                                    ID_dims = id_dim,
                                    dropout = dropout,
                                    diffusion_method = diffusion_method,
                                    with_gradient_features = with_gradient_features, 
                                    with_gradient_rotations = with_gradient_rotations)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])
            
        if pre_computes:
            self.update_precomputes(pre_computes)
        
    def update_precomputes(self, pre_computes):
        if len(pre_computes[0].shape) > 1:
            self.mass = nn.Parameter(pre_computes[0].squeeze(0), requires_grad=False)

            self.L_ind = nn.Parameter(pre_computes[1]._indices()[1:], requires_grad=False)
            self.L_val = nn.Parameter(pre_computes[1]._values(), requires_grad=False)
            self.L_size = pre_computes[1].size()[1:]
            self.evals = nn.Parameter(pre_computes[2].squeeze(0), requires_grad=False)
            self.evecs = nn.Parameter(pre_computes[3].squeeze(0), requires_grad=False)
            self.grad_X_ind  = nn.Parameter(pre_computes[4]._indices()[1:], requires_grad=False)
            self.grad_X_val  = nn.Parameter(pre_computes[4]._values(),  requires_grad=False)
            self.grad_X_size = pre_computes[4].size()[1:]
            self.grad_Y_ind  = nn.Parameter(pre_computes[5]._indices()[1:], requires_grad=False)
            self.grad_Y_val  = nn.Parameter(pre_computes[5]._values(),  requires_grad=False)
            self.grad_Y_size = pre_computes[5].size()[1:]

            self.faces = nn.Parameter(pre_computes[6].long(), requires_grad=False)
            
        else:
            self.mass = nn.Parameter(pre_computes[0], requires_grad=False)

            self.L_ind = nn.Parameter(pre_computes[1]._indices(), requires_grad=False)
            self.L_val = nn.Parameter(pre_computes[1]._values(), requires_grad=False)
            self.L_size = pre_computes[1].size()
            self.evals = nn.Parameter(pre_computes[2], requires_grad=False)
            self.evecs = nn.Parameter(pre_computes[3], requires_grad=False)
            self.grad_X_ind = nn.Parameter(pre_computes[4]._indices(), requires_grad=False)
            self.grad_X_val = nn.Parameter(pre_computes[4]._values(), requires_grad=False)
            self.grad_X_size =pre_computes[4].size()
            self.grad_Y_ind = nn.Parameter(pre_computes[5]._indices(), requires_grad=False)
            self.grad_Y_val = nn.Parameter(pre_computes[5]._values(), requires_grad=False)
            self.grad_Y_size = pre_computes[5].size()

            self.faces = nn.Parameter(pre_computes[6].unsqueeze(0).long(), requires_grad=False)

    def from_6D_to_rotation_matrix_torch(self, in_6d, eps=1e-12):
        """
        6D representation (B, 6) → rotation matrix (B, 3, 3) *following Zhou et al. (CVPR 2019)
        Args:
            in_6d (torch.Tensor): (B, 6)
        Returns:
            R (torch.Tensor): (B, 3, 3)
        """
        a1, a2 = in_6d[..., :3], in_6d[..., 3:]
        
        b1 = torch.nn.functional.normalize(a1, dim=-1, eps=eps)
        
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = torch.nn.functional.normalize(b2, dim=-1, eps=eps)
        
        b3 = torch.cross(b1, b2, dim=-1)
        
        return torch.stack((b1, b2, b3), dim=-2)
        
    def forward(self, x_in, id_in, #mass=None, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None, faces=None):
                batch_mass=None,
                batch_L_val=None,
                batch_evals=None,
                batch_evecs=None,
                batch_gradX=None,
                batch_gradY=None,
                batch_faces=None,
                ):
        """
        A forward pass on the DiffusionNet.

        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].

        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet, not all are strictly necessary.

        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]

        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """
        if True:
            inputs = x_in
            self.L = torch.sparse_coo_tensor(self.L_ind, self.L_val, self.L_size, device=inputs.device)
            batch_size = inputs.shape[0]
            if batch_mass is not None:
                batch_L = [torch.sparse_coo_tensor(self.L_ind, batch_L_val[i], self.L_size, device=inputs.device) for i in range(len(batch_L_val))]
            else:
                batch_L = [self.L for b in range(batch_size)]
                if batch_size > 1:
                    batch_mass = self.mass.expand(batch_size, -1)
                    batch_evals = self.evals.expand(batch_size, -1)
                    batch_evecs = self.evecs.expand(batch_size, -1, -1)
                else:
                    batch_mass = self.mass.unsqueeze(0).expand(batch_size, -1)
                    batch_evals = self.evals.unsqueeze(0).expand(batch_size, -1)
                    batch_evecs = self.evecs.unsqueeze(0).expand(batch_size, -1, -1)

            if batch_gradX is not None:
                gradX = [torch.sparse_coo_tensor(self.grad_X_ind, gX, self.grad_X_size, device=inputs.device) for gX in batch_gradX ]
                gradY = [torch.sparse_coo_tensor(self.grad_Y_ind, gY, self.grad_Y_size, device=inputs.device) for gY in batch_gradY ]
            else:
                gradX = [torch.sparse_coo_tensor(self.grad_X_ind, self.grad_X_val, self.grad_X_size, device=inputs.device) for b in range(batch_size)]
                gradY = [torch.sparse_coo_tensor(self.grad_Y_ind, self.grad_Y_val, self.grad_Y_size, device=inputs.device) for b in range(batch_size)]
        
            mass = batch_mass; L=batch_L; evals=batch_evals; evecs=batch_evecs; gradX=gradX; gradY=gradY; faces=self.faces
        else:
            mass = batch_mass; L=batch_L_val; evals=batch_evals; evecs=batch_evecs; gradX=batch_gradX; gradY=batch_gradY; faces=batch_faces
            
        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.C_in: 
            raise ValueError("DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(self.C_in,x_in.shape[-1]))
        N = x_in.shape[-2]
        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if L != None: L = L.unsqueeze(0)
            if evals != None: evals = evals.unsqueeze(0)
            if evecs != None: evecs = evecs.unsqueeze(0)
            if gradX != None: gradX = gradX.unsqueeze(0)
            if gradY != None: gradY = gradY.unsqueeze(0)
            if edges != None: edges = edges.unsqueeze(0)
            if faces != None: faces = faces.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False
        
        else: raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")
        
        # Apply the first linear layer
        x = self.first_lin(x_in)

        if len( id_in.shape ) < 3:
            id_in = id_in.unsqueeze(-2)
        id_in = self.act(self.adain_in(id_in))#.mean(-2, keepdims=True)# + out.mean(-2, keepdims=True)
        
        # Apply each of the blocks
        for b in self.blocks:
            x = b(id_in, x, mass, L, evals, evecs, gradX, gradY)
        
        # Apply the last linear layer
        x_out = self.last_lin(x)
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)
            
            
        if self.C_out == 3:
            # directly predict displacement
            x_out = x_in[...,:3] - x_out
        elif self.C_out == 6:
            # predict transformation
            x_out = self.from_6D_to_rotation_matrix_torch(x_out)
            x_out = torch.einsum('bnck,bnk->bnc', x_out, x_in[...,:3])
        elif self.C_out == 9:
            # predict transformation
            x_out = x_out.reshape(x_in.shape[0], x_in.shape[1], 3, 3)
            x_out = torch.einsum('bnck,bnk->bnc', x_out, x_in[...,:3])

        return x_out
    
class DoubleDiffusionNetEncoder(nn.Module):
    def __init__(self, C_in, C_out, C_width=128, N_block=4, ID_out=128, EXP_out=128,
                 last_activation=None, mlp_hidden_dims=None, dropout=True, 
                 with_gradient_features=True, with_gradient_rotations=True, diffusion_method='spectral'):
        super(DoubleDiffusionNetEncoder, self).__init__()
        """
        Construct a DiffusionNet.

        Parameters:
            C_in (int):                     input dimension 
            C_out (int):                    output dimension 
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient. Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """


        ## Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation
        
        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        
        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations
        
        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
       
        # DiffusionNet blocks
        self.blocks = nn.ModuleList()
        for i_block in range(self.N_block):
            block = DiffusionNetBlock(C_width = C_width,
                                    mlp_hidden_dims = mlp_hidden_dims, # list
                                    dropout = dropout,
                                    diffusion_method = diffusion_method,
                                    with_gradient_features = with_gradient_features, 
                                    with_gradient_rotations = with_gradient_rotations)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])
        
        # MLP for face identity (shape)
        self.id_mlps = nn.ModuleList()
        for i_mlp in range(self.N_block):
            mlp = MLP([self.C_width]+self.mlp_hidden_dims+[self.C_width], dropout=self.dropout)
            self.id_mlps.append(mlp)
            self.add_module("id_mlp_"+str(i_mlp), self.id_mlps[-1])
        self.last_id_mlp = nn.Linear(C_width, ID_out)
        
        # MLP for face expression
        self.exp_mlps = nn.ModuleList()
        for i_mlp in range(self.N_block):
            mlp = MLP([self.C_width]+self.mlp_hidden_dims+[self.C_width], dropout=self.dropout)
            self.exp_mlps.append(mlp)
            self.add_module("exp_mlp_"+str(i_mlp), self.exp_mlps[-1])
        self.last_exp_mlp = nn.Linear(C_width, EXP_out)

    def update_precomputes(self, pre_computes):
        if len(pre_computes[0].shape) > 1:
            self.mass = nn.Parameter(pre_computes[0].squeeze(0), requires_grad=False)

            self.L_ind = nn.Parameter(pre_computes[1]._indices()[1:], requires_grad=False)
            self.L_val = nn.Parameter(pre_computes[1]._values(), requires_grad=False)
            self.L_size = pre_computes[1].size()[1:]
            self.evals = nn.Parameter(pre_computes[2].squeeze(0), requires_grad=False)
            self.evecs = nn.Parameter(pre_computes[3].squeeze(0), requires_grad=False)
            self.grad_X_ind  = nn.Parameter(pre_computes[4]._indices()[1:], requires_grad=False)
            self.grad_X_val  = nn.Parameter(pre_computes[4]._values(),  requires_grad=False)
            self.grad_X_size = pre_computes[4].size()[1:]
            self.grad_Y_ind  = nn.Parameter(pre_computes[5]._indices()[1:], requires_grad=False)
            self.grad_Y_val  = nn.Parameter(pre_computes[5]._values(),  requires_grad=False)
            self.grad_Y_size = pre_computes[5].size()[1:]

            self.faces = nn.Parameter(pre_computes[6].long(), requires_grad=False)
            
        else:
            self.mass = nn.Parameter(pre_computes[0], requires_grad=False)

            self.L_ind = nn.Parameter(pre_computes[1]._indices(), requires_grad=False)
            self.L_val = nn.Parameter(pre_computes[1]._values(), requires_grad=False)
            self.L_size = pre_computes[1].size()
            self.evals = nn.Parameter(pre_computes[2], requires_grad=False)
            self.evecs = nn.Parameter(pre_computes[3], requires_grad=False)
            self.grad_X_ind = nn.Parameter(pre_computes[4]._indices(), requires_grad=False)
            self.grad_X_val = nn.Parameter(pre_computes[4]._values(), requires_grad=False)
            self.grad_X_size =pre_computes[4].size()
            self.grad_Y_ind = nn.Parameter(pre_computes[5]._indices(), requires_grad=False)
            self.grad_Y_val = nn.Parameter(pre_computes[5]._values(), requires_grad=False)
            self.grad_Y_size = pre_computes[5].size()

            self.faces = nn.Parameter(pre_computes[6].unsqueeze(0).long(), requires_grad=False)
    
    def forward(self, x_in, mass=None, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None, faces=None):
        """
        A forward pass on the DiffusionNet.

        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].

        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet, not all are strictly necessary.

        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]

        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """

        self.L = torch.sparse_coo_tensor(self.L_ind, self.L_val, self.L_size, device=x_in.device)
        batch_size = x_in.shape[0]
        L = [self.L for b in range(batch_size)]
        faces = self.faces
        mass = self.mass
        if batch_size > 1:
            mass = self.mass.expand(batch_size, -1)
            evals = self.evals.expand(batch_size, -1)
            evecs = self.evecs.expand(batch_size, -1, -1)
        else:
            mass = self.mass.unsqueeze(0).expand(batch_size, -1)
            evals = self.evals.unsqueeze(0).expand(batch_size, -1)
            evecs = self.evecs.unsqueeze(0).expand(batch_size, -1, -1)

        gradX = [torch.sparse_coo_tensor(self.grad_X_ind, self.grad_X_val, self.grad_X_size, device=x_in.device) for b in range(batch_size)]
        gradY = [torch.sparse_coo_tensor(self.grad_Y_ind, self.grad_Y_val, self.grad_Y_size, device=x_in.device) for b in range(batch_size)]


        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.C_in: 
            raise ValueError("DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(self.C_in,x_in.shape[-1]))
        N = x_in.shape[-2]
        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if L != None: L = L.unsqueeze(0)
            if evals != None: evals = evals.unsqueeze(0)
            if evecs != None: evecs = evecs.unsqueeze(0)
            if gradX != None: gradX = gradX.unsqueeze(0)
            if gradY != None: gradY = gradY.unsqueeze(0)
            if edges != None: edges = edges.unsqueeze(0)
            if faces != None: faces = faces.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False
        
        else: raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")
        
        # Apply the first linear layer
        x = self.first_lin(x_in)

        x_id = 0
        x_exp = 0
        # Apply each of the blocks
        denom = 1 / torch.sum(mass, dim=-1, keepdim=True).unsqueeze(-1)
        for b, id_mlp, exp_mlp in zip(self.blocks, self.id_mlps, self.exp_mlps):
            #x = b(x, mass, L, evals, evecs, gradX, gradY)
            res, x = b(x, mass, L, evals, evecs, gradX, gradY,return_residual=True)
            
            res_mean = torch.sum(res * mass.unsqueeze(-1), dim=-2).unsqueeze(1) * denom
            x_id = id_mlp(res_mean) + x_id
            # x_id = id_mlp(res_mean + x_id) 
            # tmp_id = torch.sum(id_mlp(res) * mass.unsqueeze(-1), dim=-2).unsqueeze(1) * denom
            # x_id = tmp_id + x_id

            x_exp = exp_mlp(res_mean) + x_exp 
            # x_exp = exp_mlp(res_mean + x_exp) 
            # tmp_exp = torch.sum(exp_mlp(res) * mass.unsqueeze(-1), dim=-2).unsqueeze(1) * denom
            # x_exp = tmp_exp + x_exp
            
        
        # Apply the last linear layer
        x_out = self.last_lin(x)
        x_id = self.last_id_mlp(x_id).squeeze(1)
        x_exp = self.last_id_mlp(x_exp).squeeze(1)
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out, x_id, x_exp
    
class HyperDiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_width, mlp_hidden_dims, hyper_latent_dim,
                 dropout=True, 
                 diffusion_method='spectral',
                 num_hidden_layers=3,
                 hyper_num_hidden_layers=3,
                 with_gradient_features=True, 
                 with_gradient_rotations=True):
        super(HyperDiffusionNetBlock, self).__init__()

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
        
        # MLPs
        #self.mlp = MiniMLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout)
        self.mlp = modules.FCBlock(
            in_features=self.MLP_C, 
            out_features=self.C_width, 
            num_hidden_layers=num_hidden_layers,
            hidden_features=self.C_width//2, 
            outermost_linear=True, 
            nonlinearity='sine'
        )
        
        self.latent_dim = hyper_latent_dim
        self.hyper_net = HyperNetwork(
            hyper_in_features=self.latent_dim,
            hyper_hidden_layers=hyper_num_hidden_layers,
            hyper_hidden_features=self.C_width,
            hypo_module=self.mlp
        )
        
    def forward(self, id_in, x_in, mass, L, evals, evecs, gradX, gradY):
        hypo_params = self.hyper_net(id_in)
        
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
        #x0_out = self.mlp(feature_combined)
        #model_input = {'coords':feature_combined}
        x0_out = self.mlp(feature_combined, params=hypo_params)

        # Skip connection
        x0_out = x0_out + x_in

        return x0_out
    
class HyperDiffusionNet(nn.Module):
    def __init__(self, C_in, C_out, C_width=128, N_block=4, 
                 last_activation=None, outputs_at='vertices', mlp_hidden_dims=None, dropout=True, 
                 num_hidden_layers=3, hyper_num_hidden_layers=3,
                 with_gradient_features=True, with_gradient_rotations=True, diffusion_method='spectral'):
        super(HyperDiffusionNet, self).__init__()
        """
        Construct a DiffusionNet.

        Parameters:
            C_in (int):                     input dimension 
            C_out (int):                    output dimension 
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient. Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """


        ## Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ['vertices', 'edges', 'faces', 'global_mean']: raise ValueError("invalid setting for outputs_at")

        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        
        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ['spectral', 'implicit_dense']: raise ValueError("invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations
        
        ## Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)
       
        # DiffusionNet blocks
        self.blocks = nn.ModuleList()
        for i_block in range(self.N_block):
            block = HyperDiffusionNetBlock(C_width = C_width,
                                    mlp_hidden_dims = mlp_hidden_dims, # list
                                    hyper_latent_dim = C_width, # int
                                    num_hidden_layers = num_hidden_layers,
                                    hyper_num_hidden_layers = hyper_num_hidden_layers,
                                    dropout = dropout,
                                    diffusion_method = diffusion_method,
                                    with_gradient_features = with_gradient_features, 
                                    with_gradient_rotations = with_gradient_rotations)

            self.blocks.append(block)
            self.add_module("block_"+str(i_block), self.blocks[-1])

    
    def forward(self, id_in, x_in, mass, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None, faces=None):
        """
        A forward pass on the DiffusionNet.

        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].

        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet, not all are strictly necessary.

        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]

        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """


        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.C_in: 
            raise ValueError("DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(self.C_in,x_in.shape[-1]))
        N = x_in.shape[-2]
        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if L != None: L = L.unsqueeze(0)
            if evals != None: evals = evals.unsqueeze(0)
            if evecs != None: evecs = evecs.unsqueeze(0)
            if gradX != None: gradX = gradX.unsqueeze(0)
            if gradY != None: gradY = gradY.unsqueeze(0)
            if edges != None: edges = edges.unsqueeze(0)
            if faces != None: faces = faces.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False
        
        else: raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")
        
        # Apply the first linear layer
        x = self.first_lin(x_in)
      
        # Apply each of the blocks
        for b in self.blocks:
            x = b(id_in, x, mass, L, evals, evecs, gradX, gradY)
        
        # Apply the last linear layer
        x = self.last_lin(x)

        # Remap output to faces/edges if requested
        if self.outputs_at == 'vertices': 
            x_out = x
        
        elif self.outputs_at == 'edges': 
            # Remap to edges
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xe = torch.gather(x_gather, 1, edges_gather)
            x_out = torch.mean(xe, dim=-1)
        
        elif self.outputs_at == 'faces': 
            # Remap to faces
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xf = torch.gather(x_gather, 1, faces_gather)
            x_out = torch.mean(xf, dim=-1)
        
        elif self.outputs_at == 'global_mean': 
            # Produce a single global mean ouput.
            # Using a weighted mean according to the point mass/area is discretization-invariant. 
            # (A naive mean is not discretization-invariant; it could be affected by sampling a region more densely)
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)
        
        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out
