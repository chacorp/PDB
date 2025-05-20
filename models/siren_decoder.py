# Original code from https://github.com/ycjungSubhuman/DeepDeformable3DCaricatures/blob/main/surface_net.py

import torch
from torch import nn
#from loss import *

import sys
from pathlib import Path
abs_path = str(Path(__file__).parents[1].absolute())
if not abs_path in sys.path:
    sys.path+=[abs_path, f'{abs_path}/third_party/siren']
else:
    sys.path+=[f'{abs_path}/third_party/siren']
    
try:
    import modules
    from meta_modules import HyperNetwork
    
except:
    import pdb;pdb.set_trace()
    raise ImportError("siren git repo required!")


class SurfaceDeformationField(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_shape=3,
                 latent_dim=128, 
                 model_type='sine', 
                 hyper_hidden_layers=1, 
                 num_hidden_layers=3, 
                 hyper_hidden_features=256, 
                 hidden_num=128, 
                 **kwargs):
        super().__init__()

        # latent code embedding for training subjects
        self.latent_dim = latent_dim
        # self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        # nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        # Deform-Net
        self.deform_net = modules.SingleBVPNet(
            type=model_type,
            mode='mlp', 
            hidden_features=hidden_num,
            num_hidden_layers=num_hidden_layers,
            in_features=in_dim,
            out_features=out_shape
        )
        # Hyper-Net
        self.hyper_net = HyperNetwork(
            hyper_in_features=self.latent_dim,
            hyper_hidden_layers=hyper_hidden_layers,
            hyper_hidden_features=hyper_hidden_features,
            hypo_module=self.deform_net
        )
        #print(self)

    # def get_hypo_net_weights(self, model_input):
    #     instance_idx = model_input['instance_idx']
    #     embedding = self.latent_codes(instance_idx)
    #     hypo_params = self.hyper_net(embedding)
    #     return hypo_params, embedding

    # def get_latent_code(self,instance_idx):
    #     embedding = self.latent_codes(instance_idx)
    #     return embedding
    
    def forward(self, vtx_input, id_input):
        # {
        #     'coords': torch.from_numpy(self.coords).float(),
        #     'positions': torch.from_numpy(self.positions).float(),
        #     'weight': torch.from_numpy(self.weights).float(),
        #     'instance_idx':torch.Tensor([self.instance_idx]).squeeze().long()
        # }
        ## batch*vertices , 1, xyz
        ## model(torch.rand(2,1,3),torch.rand(2,128))
        # tensor([[[ 0.0228,  0.0007,  0.0282]],
        #         [[-0.0702, -0.0810,  0.1221]]], grad_fn=<AddBackward0>)
        
        # get network weights for Deform-net using Hyper-net 
        hypo_params = self.hyper_net(id_input)

        model_input = {'coords':vtx_input}        
        model_output = self.deform_net(model_input, params=hypo_params)
        displacement = model_output['model_out']
        # V_new = coords + displacement # deform into template space

        # model_out = {
        #     'model_in':model_output['model_in'], 
        #     'model_out':V_new, 
        #     'latent_vec':embedding, 
        #     'hypo_params':hypo_params}

        #losses = surface_deformation_pos_loss(model_out, gt)
        return displacement