import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
abs_path = str(Path(__file__).parents[1].absolute())
sys.path+=[abs_path, f'{abs_path}/third_party/diffusion-net/src']
import diffusion_net

class BaseDecoder(nn.Module):
    """
    Decoder from NFR
    """
    def __init__(self, 
                 in_dim, 
                 hid_dim=128, 
                 num_gn=32, 
                 num_layer=6,
                 out_shape=9, 
                 act='relu'):
        super(BaseDecoder, self).__init__()
        
        if act == 'none':
            self.act = lambda x: x
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU()
        
        self.linears = [nn.Linear(in_dim, hid_dim, bias=False)]
        for _ in range(num_layer):
            self.linears.append(nn.Linear(hid_dim, hid_dim, bias=False))

        # Linear layers
        self.linears = nn.ModuleList(self.linears)
        
        # GROUP NORMs ## Input: (N,C,∗) 
        self.gns = [nn.GroupNorm(num_gn, hid_dim) for _ in range(len(self.linears))]
        self.gns = nn.ModuleList(self.gns)
        
        self.linear_out = nn.Linear(hid_dim, out_shape)

    def forward(self, x):
        """
            x (torch.tensor) [B, N, C]: input
            out (torch.tensor)
        """
        out = x
        for i in range(len(self.linears)):
            out = self.linears[i](out)
            out = torch.transpose(self.act(self.gns[i](torch.transpose(out, -1, -2))), -1, -2)
        out = self.linear_out(out)
        return out
    
class SkinningDecoder(nn.Module):
    def __init__(self, 
                 in_dim,
                 id_dim=100,
                 exp_dim=128,
                 seg_dim=20,
                 hid_dim=256, 
                 num_layer=6,
                 num_gn=32, 
                 out_shape=9, 
                 act='relu',
                 exclude_MLP=False,
                ):
        super().__init__()
        self.out_shape = out_shape
        self.in_dim = in_dim

        self.seg_dim = seg_dim
        self.exclude_MLP = exclude_MLP
        
        if act == 'none':
            self.act = lambda x: x
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU()
        
        self.linears = [nn.Linear(in_dim + exp_dim + id_dim, hid_dim, bias=False)]
        for _ in range(num_layer):
            self.linears.append(nn.Linear(hid_dim, hid_dim, bias=False))
                    
        # Linear layers
        self.linears = nn.ModuleList(self.linears)
        
        # GROUP NORMs
        self.gns = [nn.GroupNorm(num_gn, hid_dim) for _ in range(len(self.linears))]
        self.gns = nn.ModuleList(self.gns)
        
        self.linear_out = nn.Linear(hid_dim, out_shape)
        #self.I_norm = nn.InstanceNorm1d(hid_dim)
        skinning_layer = [
            nn.Linear(seg_dim, hid_dim, bias=False),
            self.act,
            nn.Linear(hid_dim, hid_dim, bias=False),
            self.act,
            nn.Linear(hid_dim, exp_dim, bias=False),
        ]
        self.skinning_layer = nn.Sequential(*skinning_layer)

    ## skinning
    def apply_skinning(self, exp_code, seg_code):
        skinning_weight = self.skinning_layer(seg_code) # [B, V, exp]
        
        # masking and amplify relevent expression 
        focused_exp = skinning_weight * exp_code
        return focused_exp
    
    def forward(self, x, exp_code, id_code, seg_code, eps=1e-12):
        """batch is 1, V = number of vertices

        Args
        ----
            x: (torch.tensor) [B, V, C]: input mesh vertex/face features
            exp_code: (torch.tensor) [B, exp]
            id_code: (torch.tensor) [B, 1, ID]
            seg_code: (torch.tensor) [1, V, S]: input mesh vertex/face segments

        Returns
        -------
            out: (torch.tensor)
        """
        
        focused_exp = self.apply_skinning(exp_code, seg_code) # [B, V, exp]
        
        out = torch.cat([x, focused_exp, id_code], dim=-1) #[B, V, C+exp+ID]
        
        for i in range(len(self.linears)):
            tmp = self.linears[i](out)
            out = torch.transpose(self.act(self.gns[i](torch.transpose(tmp, -1, -2))), -1, -2)
        out = self.linear_out(out)
        
        return out
    

class BaseDiffusionNetDecoder(nn.Module):
    # reference: https://github.com/dafei-qin/NFR_pytorch/blob/e3553faa77f65240ec20167aec6e814473233890/mymodel.py#L17
    def __init__(self, in_shape=6, out_shape=128, hid_shape=256, pre_computes=None, N_block=4, outputs_at='global_mean', with_grad=True, last_activation=None):
        super(BaseDiffusionNetDecoder, self).__init__()
        
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
            print("[DiffusionNet] warning: no pre_computes provided!")
            
        self.InstNorm1d = nn.InstanceNorm1d(hid_dim)
        skinning_layer = [
            nn.Linear(seg_dim, hid_dim, bias=False),
            self.act,
            self.InstNorm1d,
            nn.Linear(hid_dim, hid_dim, bias=False),
            self.act,
            self.InstNorm1d,
            nn.Linear(hid_dim, exp_dim, bias=False),
        ]
        self.skinning_layer = nn.Sequential(*skinning_layer)
    
    def update_precomputes(self, pre_computes):
        #import pdb;pdb.set_trace()
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

    ## skinning
    def apply_skinning(self, exp_code, seg_code):
        skinning_weight = self.skinning_layer(seg_code) # [B, V, exp]
        
        # masking and amplify relevent expression 
        focused_exp = skinning_weight * exp_code
        return focused_exp
    
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