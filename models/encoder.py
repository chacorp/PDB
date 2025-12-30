import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data

import sys
from pathlib import Path
__abs_path__ = str(Path(__file__).parents[1].absolute())
__p_net_path__ = f'{__abs_path__}/third_party/Pointnet_Pointnet2_pytorch/models'
__d_net_path__=f'{__abs_path__}/third_party/diffusion-net/src'
# __siren_path__ = f'{__abs_path__}/third_party/siren'


# for __util_path__ in [__abs_path__, __p_net_path__, __d_net_path__, __siren_path__]:
for __util_path__ in [__abs_path__, __p_net_path__, __d_net_path__]:
    if not __util_path__ in sys.path:
        sys.path+=[__util_path__]

## diffusionnet
import diffusion_net

## pointnet
# from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer, STN3d, STNkd
from pointnet_utils import STN3d, STNkd

## siren
# import modules
# from meta_modules import HyperNetwork

# from torchmeta.modules import (MetaModule, MetaSequential)
from models.blocks import *


class PointNetEncoder_small(nn.Module):
    def __init__(self, in_dim=3, out_dim=512, global_feat=True, feature_transform=False,no_norm_layer=False):
        super(PointNetEncoder_small, self).__init__()
        self.out_dim = out_dim
        
        self.stn = STN3d(in_dim)
        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.out_dim, 1)
        self.bn1 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(64)
        self.bn2 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(128)
        self.bn3 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(self.out_dim)
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x_in):
        """
        Args:
            x_in: (B,N,D)
        Returns
            (B,feat,N)
        """
        x = x_in.transpose(2, 1) # (B,N,D) -> (B,D,N)
        
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_dim)
        
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.out_dim, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
            
class PointNet_small(nn.Module):
    """
    PointNet architecture
    """
    def __init__(self, 
                in_dim=3, out_dim=3, hid_dim=512,
                 mode='rot', # (not used)
                 out_type='vertices',
                 use_softmax=False, use_relu=False, use_elu=False,
                 use_least_N=False, use_least_N_on_V=False,
                 use_gate_layer=False, no_norm_layer=False,
                 tau=1e-2
                ):
        super().__init__()
        
        self.mode = mode
        self.out_dim = out_dim
        self._tau = 1 / tau
        self.out_type = out_type
                
        self.hid_dim = self.out_dim if self.out_type == 'global' else hid_dim 
        self.global_feat = True if self.out_type == 'global' else False
        
        self.use_softmax=use_softmax
        self.use_relu=use_relu
        self.use_elu=use_elu
        self.use_least_N = use_least_N
        self.use_least_N_on_V = use_least_N_on_V
        self.use_gate_layer = use_gate_layer
        
        # if act == 'sigmoid':
        #     self.act = nn.Sigmoid()
        # elif act == 'relu':
        #     self.act = nn.ReLU()
        # elif act == 'lrelu':
        #     self.act = nn.LeakyReLU(0.2)
        # elif act == 'elu':
        #     self.act = nn.ELU(0.2)
        # elif act=='softplus':
        #     self.act = nn.Softplus()
        # else:
        #     self.act = lambda x: x
            
        self.pnt_enc = PointNetEncoder_small(
            in_dim=in_dim,
            out_dim=self.hid_dim,
            global_feat=self.global_feat,
            feature_transform=True,
            no_norm_layer=no_norm_layer,
        )
        
        if self.out_type == 'global':
            self.layer1 = nn.Linear(self.hid_dim, 256)
            self.layer2 = nn.Linear(256, 256)
            self.layer3 = nn.Linear(256, 128)
            self.layer4 = nn.Linear(128, self.out_dim)
            #self.dropout = nn.Dropout(p=0.4)
            self.bns1 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(256)
            self.bns2 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(256)
            self.bns3 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(128)
            
        else:
            # self.convs1 = torch.nn.Conv1d(4944-16, 256, 1)
            self.layer1 = nn.Conv1d(64+self.hid_dim, 256, 1)
            self.layer2 = nn.Conv1d(256, 256, 1)
            self.layer3 = nn.Conv1d(256, 128, 1)
            self.layer4 = nn.Conv1d(128, self.out_dim, 1)
            self.bns1 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(256)
            self.bns2 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(256)
            self.bns3 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(128)
    
    def least_N_zeros_gate(self, out, N: int=128, dim: int = -1):
        """
        forward: hard top-(K-N) mask || backward: softmax(tau)
        
        Args:
            out: (*, K)
            N: least number of zero (keep = K - N)
        
        Return
            mask: range in [0,1] (forward = 0/1, backward = soft)
        """
        K = out.size(dim)
        keep = max(K - N, 0)
        
        if keep == 0:
            soft = torch.softmax(out * self._tau, dim=dim)
            return (torch.zeros_like(soft) - soft).detach() + soft
        
        # soft path
        soft = torch.softmax(out * self._tau, dim=dim) # (*,K)
        
        # hard top-(K-N) mask (forward)
        topk = torch.topk(out, keep, dim=dim)
        hard = torch.zeros_like(out).scatter(dim, topk.indices, 1.0)
        
        mask = (hard - soft).detach() + soft
        return mask

    def forward(self, x_in, N=128, return_all=False):
        B, V, C = x_in.shape
        
        out, trans_feat = self.forward_func(x_in)
        
        # if self.out_type == 'global':
        #     out = out.mean(-2, keepdims=True)
        
        if self.use_softmax:
            out = F.normalize(out, dim=-2) # normalize for each column (key points)
            out = torch.softmax((out * self._tau), dim=-1) # softmax for each mesh vertex
            
        if self.use_relu:
            out = F.normalize(out, dim=-2) # normalize for each column (key points)
            out = F.relu(out)
            
            out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
            
        if self.use_elu:
            out = F.normalize(out, dim=-2) # normalize for each column (key points)
            out = F.elu(out,alpha=0.5)
            
            out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
            
        if self.use_least_N:
            out = F.normalize(out, dim=-2) # normalize for each column (key points)
            out = F.relu(out)
            
            mask = self.least_N_zeros_gate(out, N=N, dim=-1)
            out = out * mask
            out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
            
        if self.use_least_N_on_V:
            NZ = V // 16
            out = F.normalize(out, dim=-2) # normalize for each column (key points)
            mask = self.least_N_zeros_gate(out, N=NZ, dim=-2) # on vertex dimension!
            out = F.relu(out) * mask
            out = out / (out.sum(dim=-1, keepdim=True)+1e-12)

        if return_all:
            return out, trans_feat
        return out
        
    def forward_func(self, point_cloud):
        """
        Args:
            point_cloud: (B,N,D) input point cloud with D dimension features
        """
        
        B, N, D = point_cloud.size()
        
        out, trans, trans_feat = self.pnt_enc(point_cloud)
        # out.shape => (B feat N)
        
        out = F.relu(self.bns1(self.layer1(out)))
        out = F.relu(self.bns2(self.layer2(out)))
        out = F.relu(self.bns3(self.layer3(out)))
        out = self.layer4(out)
        
        if self.out_type == 'global':
            out = out.view(B, 1, self.out_dim)
            return out, trans_feat
            
        out = out.transpose(2, 1).contiguous()        
        # net = F.log_softmax(net.view(-1, self.out_dim), dim=-1)
        out = out.view(B, N, self.out_dim) # [B, N, out_dim]
        return out, trans_feat

class PointNet_large(nn.Module):
    """
    PointNet architecture 2
    """
    def __init__(self, 
                in_dim=3, out_dim=3, hid_dim=512,
                 mode='rot', # (not used)
                 out_type='vertices',
                 use_softmax=False, use_relu=False, use_elu=False,
                 use_least_N=False, use_least_N_on_V=False,
                 use_gate_layer=False,
                 tau=1e-2
                ):
        super().__init__()

        assert out_dim//4 > 1, f'out_dim is too small! {out_dim}'
        
        self.mode = mode
        self.out_dim = out_dim
        self._tau = 1 / tau
        self.out_type = out_type
                
        self.hid_dim = self.out_dim if self.out_type == 'global' else hid_dim 
        self.global_feat = True if self.out_type == 'global' else False
        
        self.use_softmax=use_softmax
        self.use_relu=use_relu
        self.use_elu=use_elu
        self.use_least_N = use_least_N
        self.use_least_N_on_V = use_least_N_on_V
        self.use_gate_layer = use_gate_layer
        
        # if act == 'sigmoid':
        #     self.act = nn.Sigmoid()
        # elif act == 'relu':
        #     self.act = nn.ReLU()
        # elif act == 'lrelu':
        #     self.act = nn.LeakyReLU(0.2)
        # elif act == 'elu':
        #     self.act = nn.ELU(0.2)
        # elif act=='softplus':
        #     self.act = nn.Softplus()
        # else:
        #     self.act = lambda x: x
            
        self.pnt_enc = PointNetEncoder_small(
            in_dim=in_dim,
            out_dim=self.hid_dim,
            global_feat=self.global_feat,
            feature_transform=True
        )
        
        if self.out_type == 'global':
            self.layer1 = nn.Linear(self.hid_dim, 256)
            self.layer2 = nn.Linear(256, 256)
            self.layer3 = nn.Linear(256, self.out_dim//4)
            self.layer4 = nn.Linear(self.out_dim//4, self.out_dim//2)
            self.layer5 = nn.Linear(self.out_dim//2, self.out_dim)
            #self.dropout = nn.Dropout(p=0.4)
            self.bns1 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(256)
            self.bns2 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(256)
            self.bns3 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(self.out_dim//4)
            self.bns4 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(self.out_dim//2)
            
        else:
            # self.convs1 = torch.nn.Conv1d(4944-16, 256, 1)
            self.layer1 = nn.Conv1d(64+self.hid_dim, 256, 1)
            self.layer2 = nn.Conv1d(256, 256, 1)
            self.layer3 = nn.Conv1d(256, self.out_dim//4, 1)
            self.layer4 = nn.Conv1d(self.out_dim//4, self.out_dim//2, 1)
            self.layer5 = nn.Conv1d(self.out_dim//2, self.out_dim, 1)
            self.bns1 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(256)
            self.bns2 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(256)
            self.bns3 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(self.out_dim//4)
            self.bns4 = nn.Identity() if no_norm_layer else nn.BatchNorm1d(self.out_dim//2)
    
    def least_N_zeros_gate(self, out, N: int=128, dim: int = -1):
        """
        forward: hard top-(K-N) mask || backward: softmax(tau)
        
        Args:
            out: (*, K)
            N: least number of zero (keep = K - N)
        
        Return
            mask: range in [0,1] (forward = 0/1, backward = soft)
        """
        K = out.size(dim)
        keep = max(K - N, 0)
        
        if keep == 0:
            soft = torch.softmax(out * self._tau, dim=dim)
            return (torch.zeros_like(soft) - soft).detach() + soft
        
        # soft path
        soft = torch.softmax(out * self._tau, dim=dim) # (*,K)
        
        # hard top-(K-N) mask (forward)
        topk = torch.topk(out, keep, dim=dim)
        hard = torch.zeros_like(out).scatter(dim, topk.indices, 1.0)
        
        mask = (hard - soft).detach() + soft
        return mask

    def forward(self, x_in, N=128, return_all=False):
        B, V, C = x_in.shape
        
        out, trans_feat = self.forward_func(x_in)
        
        # if self.out_type == 'global':
        #     out = out.mean(-2, keepdims=True)
        
        if self.use_softmax:
            out = F.normalize(out, dim=-2) # normalize for each column (key points)
            out = torch.softmax((out * self._tau), dim=-1) # softmax for each mesh vertex
            
        if self.use_relu:
            out = F.normalize(out, dim=-2) # normalize for each column (key points)
            out = F.relu(out)
            
            out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
            
        if self.use_elu:
            out = F.normalize(out, dim=-2) # normalize for each column (key points)
            out = F.elu(out, alpha=0.5)
            
            out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
            
        if self.use_least_N:
            out = F.normalize(out, dim=-2) # normalize for each column (key points)
            out = F.relu(out)
            
            mask = self.least_N_zeros_gate(out, N=N, dim=-1)
            out = out * mask
            out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
            
        if self.use_least_N_on_V:
            NZ = V // 16
            out = F.normalize(out, dim=-2) # normalize for each column (key points)
            mask = self.least_N_zeros_gate(out, N=NZ, dim=-2) # on vertex dimension!
            out = F.relu(out) * mask
            out = out / (out.sum(dim=-1, keepdim=True)+1e-12)

        if return_all:
            return out, trans_feat
        return out
        
    def forward_func(self, point_cloud):
        """
        Args:
            point_cloud: (B,N,D) input point cloud with D dimension features
        """
        
        B, N, D = point_cloud.size()
        
        out, trans, trans_feat = self.pnt_enc(point_cloud)
        # out.shape => (B feat N)
        
        out = F.relu(self.bns1(self.layer1(out)))
        out = F.relu(self.bns2(self.layer2(out)))
        out = F.relu(self.bns3(self.layer3(out)))
        out = F.relu(self.bns4(self.layer4(out)))
        out = self.layer5(out)
        
        if self.out_type == 'global':
            out = out.view(B, 1, self.out_dim)
            return out, trans_feat
            
        out = out.transpose(2, 1).contiguous()        
        # net = F.log_softmax(net.view(-1, self.out_dim), dim=-1)
        out = out.view(B, N, self.out_dim) # [B, N, out_dim]
        return out, trans_feat


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
    def __init__(self, layer_sizes, num_gn=32, dropout=False, act='relu', nrm='layer', name="MLP", p=.5):
        super(MLP, self).__init__()

        if act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(0.2)
        elif act == 'elu':
            self.act = nn.ELU(0.2)
        elif act=='softplus':
            self.act = nn.Softplus()
        else:
            self.act = lambda x: x
        
        self.nrm=nrm
        if nrm == 'group':
            norm_func=nn.GroupNorm
        elif nrm == 'inst':
            norm_func=nn.InstanceNorm1d
        elif nrm == 'batch':
            norm_func=nn.BatchNorm1d
        elif nrm == 'layer':
            norm_func=nn.LayerNorm
        else:
            norm_func=nn.Identity
            
        self.num_layers = len(layer_sizes)
        
        layers = []
        norms = []
        for i in range(self.num_layers-2):
            if dropout and i > 0:
                layers.append(nn.Dropout(p=p))
            
            layers.append(
                nn.Linear(layer_sizes[i], layer_sizes[i+1])
            )
            norms.append(
                norm_func(num_gn, layer_sizes[i+1])
                if nrm == 'group' else
                norm_func(layer_sizes[i+1])
            )            
        layers.append(
            nn.Linear(layer_sizes[-2], layer_sizes[-1])
        )
        
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        
    def detect_transpose(self, x):
        if self.nrm == 'group' or self.nrm == 'inst' or self.nrm == 'batch':
            x = x.transpose(-1, -2)
        return x
        
    def forward(self, x):
        out = x
        for i in range(self.num_layers-2):
            tmp = self.layers[i](out)
            tmp = self.detect_transpose(tmp)
            
            tmp = self.act(self.norms[i](tmp))
            out = self.detect_transpose(tmp)
            
        out = self.layers[-1](out)
        return out

class LinearFeatureExtractor(nn.Module):
    def __init__(self, 
                 in_dim=3,
                 out_dim=128,
                 hid_dim=128,
                 num_layers=4,
                 use_residual=False,
                 out_type='vertices',
                 act='lrelu',
                 nrm='none'
                ):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.use_residual = use_residual
                
        self.out_type = out_type
                
        self.layer_in = nn.Linear(in_dim, hid_dim)
        self.layer_out = nn.Linear(hid_dim, out_dim)
        
        self.layers = nn.ModuleList([
            MLP([hid_dim, hid_dim, hid_dim, hid_dim, hid_dim, hid_dim, hid_dim], act=act, nrm=nrm)
            for _ in range(num_layers)
        ])
                
    def forward(self, x_in):
        # B, V, C = x_in.shape        
        out = self.layer_in(x_in)
        
        for layer in self.layers:
            l_out = layer(out)
            
            if self.use_residual:
                out = l_out + out
            else:
                out = l_out
            
        if self.out_type=='global':
            out = self.layer_out(out).mean(-2, keepdims=True)
        return out

class LinearEncoder(nn.Module):
    def __init__(self, 
                 in_dim=3, out_dim=3, hid_dim=128, num_layers=4, 
                 mode='rot', use_residual=False, out_type='vertices',
                 use_softmax=False, use_relu=False, use_softplus=False, 
                 use_elu=False, use_sqrelu=False,
                 use_least_N=False, use_least_N_on_V=False,no_activation=False,
                 use_gate_layer=False,
                 use_pou=False,
                 act='lrelu', nrm='layer',
                 tau=1e-2, use_K=False, K_dim=8,
                ):
        super().__init__()
        
        self.mode = mode
        self.out_dim = out_dim
        self._tau = 1 / tau
        
        self.use_residual = use_residual
        
        self.use_softmax=use_softmax
        self.use_relu=use_relu
        self.use_sqrelu=use_sqrelu
        self.use_elu=use_elu
        self.use_softplus=use_softplus
        self.use_least_N = use_least_N
        self.use_least_N_on_V = use_least_N_on_V
        self.use_gate_layer = use_gate_layer
        self.no_activation=no_activation
        self.use_pou = use_pou
        
        self.out_type = out_type
        self.use_K = use_K
        self.K_dim = K_dim
                
        self.layer_in = nn.Linear(in_dim, hid_dim)
        self.layer_out = nn.Linear(hid_dim, out_dim)

        self.layers = nn.ModuleList([
            MLP([hid_dim, hid_dim, hid_dim], act=act, nrm=nrm)
            for _ in range(num_layers)
        ])

        ## adaptive layer Norm
        self.adain_in = MLP(
            [in_dim, hid_dim, hid_dim, hid_dim, hid_dim, hid_dim], 
            act=act, nrm=nrm
        )
        
        self.adains_m = nn.ModuleList([
            MLP([hid_dim, hid_dim, hid_dim], act=act, nrm=nrm)
            for _ in range(num_layers)
        ])
        self.adains_s = nn.ModuleList([
            MLP([hid_dim, hid_dim, hid_dim], act=act, nrm=nrm)
            for _ in range(num_layers)
        ])
        
        if self.use_gate_layer:
            self.gate_layer = nn.Sequential(
                MLP([hid_dim, hid_dim, hid_dim, hid_dim, out_dim], act=act, nrm=nrm),
                nn.Sigmoid(),
            )
        
    def least_N_zeros_gate(self, out, N: int=128, dim: int = -1, tau: float = 0.01):
        """
        forward: hard top-(K-N) mask || backward: softmax(tau)
        
        Args:
            out: (*, K)
            N: least number of zero (keep = K - N)
        
        Return
            mask: range in [0,1] (forward = 0/1, backward = soft)
        """
        K = out.size(dim)
        keep = max(K - N, 0)
        
        if keep == 0:
            soft = torch.softmax(out * self._tau, dim=dim)
            return (torch.zeros_like(soft) - soft).detach() + soft
    
        # soft path for gradients
        soft = torch.softmax(out * self._tau, dim=dim) # (*,K)
    
        # hard top-(K-N) mask (forward)
        topk = torch.topk(out, keep, dim=dim)
        hard = torch.zeros_like(out).scatter(dim, topk.indices, 1.0)
    
        # Straight-Through estimator
        mask = (hard - soft).detach() + soft
        return mask
        
    def forward(self, x_in, id_in=None, return_id_in=False, N=128, return_inv=False, return_raw=False):
        B, V, C = x_in.shape
        
        out, id_out = self.forward_func(x_in, id_in, return_id_in)
        
        if self.out_type == 'global':
            out = out.mean(-2, keepdims=True)

        if not self.no_activation:
            if self.use_softmax:
                out = F.normalize(out, dim=-2) # normalize for each column (key points)
                out = torch.softmax((out * self._tau), dim=-1) # softmax for each mesh vertex
                
            if self.use_relu:
                out = F.normalize(out, dim=-2) # normalize for each column (key points)
                out = F.relu(out)
                
                #out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
            
            if self.use_sqrelu:
                out = F.normalize(out, dim=-2) # normalize for each column (key points)
                out = F.relu(out)**2
                
            if self.use_softplus:
                out = F.normalize(out, dim=-2) # normalize for each column (key points)
                out = F.softplus(out)
                
                #out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
    
            if self.use_elu:
                out = F.normalize(out, dim=-2) # normalize for each column (key points)
                out = F.elu(out, alpha=0.5)
                
                #out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
                
            if self.use_least_N:
                out = F.normalize(out, dim=-2) # normalize for each column (key points)
                out = F.relu(out)
                
                mask = self.least_N_zeros_gate(out, N=N, dim=-1)
                out = out * mask
                #out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
                
            if self.use_least_N_on_V:
                NZ = V // 16
                out = F.normalize(out, dim=-2) # normalize for each column (key points)
                out = F.relu(out)
                
                mask = self.least_N_zeros_gate(out, N=NZ, dim=-2) # on vertex dimension!
                out = out * mask
                #out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
            
            if self.use_pou and not self.use_softmax:
                out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
        else:
            if self.use_pou:
                out = out / (out.sum(dim=-1, keepdim=True)+1e-12)

        if return_id_in:
            return out, id_out
        return out
        
        
    def forward_func(self, x_in, id_in=None, return_id_in=False, return_inv=False):
        out = self.layer_in(x_in)
        
        if id_in is None:
            id_in = self.adain_in(x_in).mean(-2, keepdims=True) + out.mean(-2, keepdims=True)
        
        for layer, mu, sigma in zip(self.layers, self.adains_m, self.adains_s):
            l_out = layer(out)
            l_out = l_out * sigma(id_in) + mu(id_in)
            
            if self.use_residual:
                out = l_out + out
            else:
                out = l_out
                
        if self.use_gate_layer:
            out = self.layer_out(out) * self.gate_layer(id_in)
        else:
            out = self.layer_out(out)

        if return_id_in:
            return out, id_in
        return out, None

class LinearEncoder2(nn.Module):
    def __init__(self, 
                 in_dim=3, out_dim=3, hid_dim=128, num_layers=4, 
                 mode='rot', use_residual=False, out_type='vertices',
                 use_softmax=False, use_relu=False, use_softplus=False, 
                 use_elu=False, use_sqrelu=False,
                 use_least_N=False, use_least_N_on_V=False,
                 no_activation=False,
                 use_gate_layer=False,
                 use_pou=False,
                 use_id_feat=True,
                 act='lrelu', nrm='layer',
                 tau=1e-2, use_K=False, K_dim=8,
                ):
        super().__init__()
        
        self.mode = mode
        self.out_dim = out_dim
        self._tau = 1 / tau
        
        self.use_residual = use_residual
        
        self.use_softmax=use_softmax
        self.use_relu=use_relu
        self.use_sqrelu=use_sqrelu
        self.use_elu=use_elu
        self.use_softplus=use_softplus
        self.use_least_N = use_least_N
        self.use_least_N_on_V = use_least_N_on_V
        self.use_gate_layer = use_gate_layer
        self.no_activation=no_activation
        self.use_pou = use_pou
        
        self.out_type = out_type
        self.use_K = use_K
        self.K_dim = K_dim
                
        self.layer_in = nn.Linear(in_dim, hid_dim)
        self.layer_out = nn.Linear(hid_dim, out_dim)

        self.layers = nn.ModuleList([
            MLP([hid_dim, hid_dim, hid_dim], act=act, nrm=nrm)
            for _ in range(num_layers)
        ])

        ## for feature transform
        self.use_id_feat = use_id_feat
        if self.use_id_feat:
            self.id_feat_in = MLP(
                [in_dim, hid_dim, hid_dim, hid_dim, hid_dim, hid_dim, hid_dim], 
                act=act, nrm=nrm
            )
        else:
            self.id_feat_in = MLP(
                [hid_dim, hid_dim, hid_dim, hid_dim], 
                act=act, nrm=nrm
            )
            
        self.STN_a = nn.ModuleList([
            MLP([hid_dim, hid_dim, hid_dim], act=act, nrm=nrm)
            for _ in range(num_layers)
        ])
        self.STN_b = nn.ModuleList([
            MLP([hid_dim, hid_dim, hid_dim], act=act, nrm=nrm)
            for _ in range(num_layers)
        ])
        
        if self.use_gate_layer:
            self.gate_layer = nn.Sequential(
                MLP([hid_dim, hid_dim, hid_dim, hid_dim, out_dim], act=act, nrm=nrm),
                nn.Sigmoid(),
            )
        
    def forward(self, x_in, id_in=None, return_id_in=False, N=128, return_inv=False, return_raw=False):
        B, V, C = x_in.shape
        
        out = self.forward_func(x_in, id_in, return_id_in)
        
        if self.out_type == 'global':
            out = out.mean(-2, keepdims=True)

        if not self.no_activation:
            out = F.normalize(out, dim=-2) # normalize for each column (key points)
            if self.use_softmax:
                out = torch.softmax((out * self._tau), dim=-1) # softmax for each mesh vertex
                
            if self.use_relu:            
                out = F.relu(out)
                
            if self.use_sqrelu:
                out = torch.square(F.relu(out))
                
            if self.use_softplus:
                out = F.softplus(out)
                
            if self.use_elu:
                out = F.elu(out, alpha=0.5)
                
            if self.use_pou and not self.use_softmax:
                out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
        else:
            if self.use_pou:
                out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
            
        if return_id_in:
            return out, id_in
        return out
        
        
    def forward_func(self, x_in, id_in=None, return_id_in=False):
        out = self.layer_in(x_in)

        if self.use_id_feat:
            id_in = self.id_feat_in(x_in).mean(-2, keepdims=True) + out.mean(-2, keepdims=True)
        else:
            id_in = self.id_feat_in(id_in)
        
        for layer, mu, sigma in zip(self.layers, self.STN_b, self.STN_a):
            l_out = layer(out)
            l_out = l_out * sigma(id_in) + mu(id_in)
            
            if self.use_residual:
                out = l_out + out
            else:
                out = l_out
                
        if self.use_gate_layer:
            out = self.layer_out(out) * self.gate_layer(id_in)
        else:
            out = self.layer_out(out)

        if return_id_in:
            return out, id_in
        return out
        
class LinearDecoder2(nn.Module):
    def __init__(self, 
                 in_dim=3, out_dim=3, hid_dim=128, num_layers=4, 
                 mode='rot', use_residual=False, out_type='vertices',
                 use_id_feat=True,
                 act='lrelu', nrm='layer',
                 tau=1e-2, use_K=False, K_dim=8,
                ):
        super().__init__()
        
        self.mode = mode
        self.out_dim = out_dim
        self._tau = 1 / tau
        
        self.use_residual = use_residual
                
        self.out_type = out_type
        self.use_K = use_K
        self.K_dim = K_dim
                
        self.layer_in = nn.Linear(in_dim, hid_dim)
        self.layer_out = nn.Linear(hid_dim, hid_dim)
        self.layer_pos_out = MLP([hid_dim, hid_dim, out_dim], act=act, nrm='none')
        self.layer_nrm_out = MLP([hid_dim, hid_dim, out_dim], act=act, nrm='none')

        self.layers = nn.ModuleList([
            MLP([hid_dim, hid_dim, hid_dim], act=act, nrm=nrm)
            for _ in range(num_layers)
        ])

        ## for feature transform
        self.use_id_feat = use_id_feat
        if self.use_id_feat:
            self.id_feat_in = MLP(
                [in_dim, hid_dim, hid_dim, hid_dim, hid_dim, hid_dim, hid_dim], 
                act=act, nrm=nrm
            )
        else:
            self.id_feat_in = MLP(
                [hid_dim, hid_dim, hid_dim, hid_dim], 
                act=act, nrm=nrm
            )
        
        self.STN_a = nn.ModuleList([
            MLP([hid_dim, hid_dim, hid_dim], act=act, nrm=nrm)
            for _ in range(num_layers)
        ])
        self.STN_b = nn.ModuleList([
            MLP([hid_dim, hid_dim, hid_dim], act=act, nrm=nrm)
            for _ in range(num_layers)
        ])
        
    def forward(self, x_in, id_in=None, N=128, return_id_in=False):
        B, V, C = x_in.shape
        
        out = self.layer_in(x_in)

        if self.use_id_feat:
            id_in = self.id_feat_in(x_in).mean(-2, keepdims=True) + out.mean(-2, keepdims=True)
        else:
            id_in = self.id_feat_in(id_in)
        
        for layer, mu, sigma in zip(self.layers, self.STN_b, self.STN_a):
            l_out = layer(out)
            l_out = l_out * sigma(id_in) + mu(id_in)
            
            if self.use_residual:
                out = l_out + out
            else:
                out = l_out
             
        out = self.layer_out(out).mean(-2, keepdims=True)
        
        ## cage vertex position
        o_pos = self.layer_pos_out(out)
        ## cage vertex normal
        o_nrm = self.layer_nrm_out(out)
        o_nrm = F.normalize(o_nrm, dim=-1)
        
        if return_id_in:
            return o_pos, o_nrm, id_in
        return o_pos, o_nrm, None


class ControlVertexEncoder(nn.Module):
    def __init__(self, 
                 in_dim=3, out_dim=3, hid_dim=128, K_dim=256, num_layers=4, num_heads=8,
                 use_residual=False, act='lrelu', nrm='layer',
                 
                ):
        super().__init__()
        
        self.out_dim = out_dim
        self.use_residual = use_residual
        self.num_layers = num_layers
        
        self.K_dim = K_dim # num vertex

        ## control vertex
        self.v_latent = nn.Parameter(torch.randn(self.K_dim, hid_dim))
        self.v_layer_out = nn.Linear(hid_dim, out_dim)
        
        self.v_layers = nn.ModuleList([
            MLP([hid_dim, hid_dim//2, hid_dim//2, hid_dim], act=act, nrm=nrm)
            for _ in range(num_layers)
        ])
        self.v_attn = nn.ModuleList([
            nn.MultiheadAttention(hid_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        
        ## coordinates
        self.p_layer_in = nn.Linear(in_dim, hid_dim)
        self.p_layer_out = nn.Linear(hid_dim, self.K_dim)
        
        self.p_layers = nn.ModuleList([
            MLP([hid_dim, hid_dim//2, hid_dim//2, hid_dim], act=act, nrm=nrm)
            for _ in range(num_layers)
        ])        
        self.p_attn = nn.ModuleList([
            nn.MultiheadAttention(hid_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
    
    def forward(self, x_in):
        B, V, C = x_in.shape
        
        v_out = self.v_latent[None].repeat(B,1,1)
        p_out = self.p_layer_in(x_in)

        for i in range(self.num_layers):            
            v_feat_, v_attn_weight = self.v_attn[i](v_out, p_out, p_out)
            p_feat_, p_attn_weight = self.p_attn[i](p_out, v_out, v_out)
            
            v_out = v_out + v_feat_
            p_out = p_out + p_feat_
            
            v_out = self.v_layers[i](v_out)
            p_out = self.p_layers[i](p_out)
            
        v_out = self.v_layer_out(v_out)
        p_out = self.p_layer_out(p_out)
            
        return p_out, v_out

class ExpressionEncoder(nn.Module):
    def __init__(self, 
                 in_dim=3, out_dim=3, hid_dim=128, K_dim=256, num_layers=4, num_heads=8,
                 use_residual=False, act='lrelu', nrm='layer',
                 
                ):
        super().__init__()
        
        self.out_dim = out_dim
        self.use_residual = use_residual
        self.num_layers = num_layers
        
        self.K_dim = K_dim # num vertex

        ## control vertex
        self.v_latent = nn.Parameter(torch.randn(self.K_dim, hid_dim))
        self.v_layer_out = nn.Linear(hid_dim, out_dim)
        
        self.v_layers = nn.ModuleList([
            MLP([hid_dim, hid_dim//2, hid_dim//2, hid_dim], act=act, nrm=nrm)
            for _ in range(num_layers)
        ])
        self.v_attn = nn.ModuleList([
            nn.MultiheadAttention(hid_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        
        ## coordinates
        self.p_layer_in = nn.Linear(in_dim, hid_dim)
        self.p_layer_out = nn.Linear(hid_dim, self.K_dim)
        
        self.p_layers = nn.ModuleList([
            MLP([hid_dim, hid_dim//2, hid_dim//2, hid_dim], act=act, nrm=nrm)
            for _ in range(num_layers)
        ])        
        self.p_attn = nn.ModuleList([
            nn.MultiheadAttention(hid_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
    
    def forward(self, v_in):
        B, K, C = v_in.shape
        
        v_out = self.v_latent[None].repeat(B,1,1)
        p_out = self.p_layer_in(x_in)

        for i in range(self.num_layers):            
            v_feat_, v_attn_weight = self.v_attn[i](v_out, p_out, p_out)
            p_feat_, p_attn_weight = self.p_attn[i](p_out, v_out, v_out)
            
            v_out = v_out + v_feat_
            p_out = p_out + p_feat_
            
            v_out = self.v_layers[i](v_out)
            p_out = self.p_layers[i](p_out)
            
        v_out = self.v_layer_out(v_out)
        p_out = self.p_layer_out(p_out)
            
        return p_out, v_out