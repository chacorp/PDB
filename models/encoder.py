import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import sys
from pathlib import Path
__abs_path__ = str(Path(__file__).parents[1].absolute())
__p_net_path__ = f'{__abs_path__}/third_party/Pointnet_Pointnet2_pytorch/models'
__d_net_path__=f'{__abs_path__}/third_party/diffusion-net/src'
__siren_path__ = f'{__abs_path__}/third_party/siren'


for __util_path__ in [__abs_path__, __p_net_path__, __d_net_path__, __siren_path__]:
    if not __util_path__ in sys.path:
        sys.path+=[__util_path__]

## diffusionnet
import diffusion_net

## pointnet
# from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer, STN3d, STNkd
from pointnet_utils import STN3d, STNkd

## siren
import modules
from meta_modules import HyperNetwork

# from torchmeta.modules import (MetaModule, MetaSequential)
from models.blocks import *


class PointNetEncoder_small(nn.Module):
    def __init__(self, in_dim=3, out_dim=512, global_feat=True, feature_transform=False):
        super(PointNetEncoder_small, self).__init__()
        self.out_dim = out_dim
        
        self.stn = STN3d(in_dim)
        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.out_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.out_dim)
        
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
                 use_gate_layer=False,
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
            feature_transform=True
        )
        
        if self.out_type == 'global':
            self.layer1 = nn.Linear(self.hid_dim, 256)
            self.layer2 = nn.Linear(256, 256)
            self.layer3 = nn.Linear(256, 128)
            self.layer4 = nn.Linear(128, self.out_dim)
            #self.dropout = nn.Dropout(p=0.4)
            self.bns1 = nn.BatchNorm1d(256)
            self.bns2 = nn.BatchNorm1d(256)
            self.bns3 = nn.BatchNorm1d(128)
            
        else:
            # self.convs1 = torch.nn.Conv1d(4944-16, 256, 1)
            self.layer1 = nn.Conv1d(64+self.hid_dim, 256, 1)
            self.layer2 = nn.Conv1d(256, 256, 1)
            self.layer3 = nn.Conv1d(256, 128, 1)
            self.layer4 = nn.Conv1d(128, self.out_dim, 1)
            self.bns1 = nn.BatchNorm1d(256)
            self.bns2 = nn.BatchNorm1d(256)
            self.bns3 = nn.BatchNorm1d(128)
    
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
            self.bns1 = nn.BatchNorm1d(256)
            self.bns2 = nn.BatchNorm1d(256)
            self.bns3 = nn.BatchNorm1d(self.out_dim//4)
            self.bns4 = nn.BatchNorm1d(self.out_dim//2)
            
        else:
            # self.convs1 = torch.nn.Conv1d(4944-16, 256, 1)
            self.layer1 = nn.Conv1d(64+self.hid_dim, 256, 1)
            self.layer2 = nn.Conv1d(256, 256, 1)
            self.layer3 = nn.Conv1d(256, self.out_dim//4, 1)
            self.layer4 = nn.Conv1d(self.out_dim//4, self.out_dim//2, 1)
            self.layer5 = nn.Conv1d(self.out_dim//2, self.out_dim, 1)
            self.bns1 = nn.BatchNorm1d(256)
            self.bns2 = nn.BatchNorm1d(256)
            self.bns3 = nn.BatchNorm1d(self.out_dim//4)
            self.bns4 = nn.BatchNorm1d(self.out_dim//2)
    
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
    def __init__(self, layer_sizes, num_gn=32, dropout=False, act='relu', nrm='batch', name="MLP", p=.5):
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
            
            out = self.act(self.norms[i](tmp))
            out = self.detect_transpose(out)
        out = self.layers[-1](out)
        return out

class LinearEncoder(nn.Module):
    def __init__(self, 
                 in_dim=3, out_dim=3, hid_dim=128, num_layers=4, 
                 mode='rot', use_residual=False, out_type='vertices',
                 use_softmax=False, use_relu=False, use_softplus=False, use_elu=False,
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
        
    def forward(self, x_in, N=128, return_inv=False, return_raw=False):
        B, V, C = x_in.shape
        
        out = self.forward_func(x_in)
        
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
            
        return out
        
        
    def forward_func(self, x_in, return_inv=False):
        out = self.layer_in(x_in)
        
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
                    
        return out

class LinearEncoder2(nn.Module):
    def __init__(self,
                 in_dim=3, style_dim=100, out_dim=3, hid_dim=128,
                 num_layers=4, use_style=True, out_type='vertices',
                 use_softmax=False, use_relu=False, use_softplus=False, use_elu=False,
                 use_least_N=False, use_least_N_on_V=False,
                 no_activation=False, 
                 use_pou=False,
                 use_residual=True, use_K=False, K_dim=8,
                 act='lrelu', nrm='layer',
                 use_gate_layer=False,
                ):
        super().__init__()
                
        self.in_dim = in_dim
        self.style_dim = style_dim
        self.out_dim = out_dim
        self.use_style=use_style
        self.out_type = out_type
        self.use_softmax = use_softmax
        self.use_relu = use_relu
        self.use_elu = use_elu
        self.use_softplus = use_softplus
        self.no_activation = no_activation

        self.use_residual=use_residual
        self.use_least_N=use_least_N # not used
        self.use_least_N_on_V=use_least_N_on_V # not used
        self.use_pou=use_pou

        self.use_gate_layer = use_gate_layer
        self.use_K = use_K
        self.K_dim = K_dim
        
        self.act = nn.LeakyReLU(0.2)
                
        self.layer_in = nn.Linear(in_dim, hid_dim)
        self.layer_out = nn.Linear(hid_dim, out_dim)

        self.layers = nn.ModuleList([
            MLP([hid_dim, hid_dim, hid_dim], act=act, nrm=nrm)
            for _ in range(num_layers)
        ])
        adain_dim = style_dim if use_style else in_dim
        self.adain_in = MLP(
            [adain_dim, hid_dim, hid_dim, hid_dim, hid_dim, hid_dim], 
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

    def forward(self, x_in, style, N=128):
        """
            x_in: (B, N, 3)
        """
        out = self.layer_in(x_in)
        
        id_in = self.adain_in(style if self.use_style else x_in)
        
        for l, mu, sigma in zip(self.layers, self.adains_m, self.adains_s):
            l_out = l(out)
            s_out = sigma(id_in)
            m_out = mu(id_in)
            l_out = (l_out * s_out) + m_out
            
            if self.use_residual:
                out = l_out + out
            else:
                out = l_out
        
        if self.use_gate_layer:
            out = self.layer_out(out) * self.gate_layer(self.act(id_in))
        else:
            out = self.layer_out(out)
        
        if self.out_type == 'global':
            out = out.mean(-2, keepdims=True)

        if not self.no_activation:
            if self.use_softmax:
                out = F.normalize(out, dim=-2) # normalize for each column (key points)
                out = torch.softmax((out) * self._tau, dim=-1) # softmax for each mesh vertex
    
            if self.use_relu:
                out = F.normalize(out, dim=-2) # normalize for each column (key points)
                out = F.relu(out)
                
                # out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
                
            if self.use_elu:
                out = F.normalize(out, dim=-2) # normalize for each column (key points)
                out = F.elu(out)
                
                # out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
    
            if self.use_softplus:
                out = F.normalize(out, dim=-2) # normalize for each column (key points)
                out = F.softplus(out)
                
                # out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
                
            if self.use_least_N:
                out = F.normalize(out, dim=-2) # normalize for each column (key points)
                out = F.relu(out)
                
                mask = self.least_N_zeros_gate(out, N=N, dim=-1)
                out = out * mask
                # out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
                
            if self.use_least_N_on_V:
                NZ = V // 16
                out = F.normalize(out, dim=-2) # normalize for each column (key points)
                out = F.relu(out)
                
                mask = self.least_N_zeros_gate(out, N=NZ, dim=-2) # on vertex dimension!
                out = out * mask
                # out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
        
            if self.use_pou and not self.use_softmax:
                out = out / (out.sum(dim=-1, keepdim=True)+1e-12)
            
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
