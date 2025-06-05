## Code adapted from [Esser, Rombach 2021]: https://compvis.github.io/taming-transformers/
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def swish(x):
    return x*torch.sigmoid(x)

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        z_flattened = z.view(-1, self.e_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # d1 = torch.sum(z_flattened ** 2, dim=1, keepdim=True)
        # d2 = torch.sum(self.embedding.weight**2, dim=1)
        # d3 = torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z) 
        min_encodings.scatter_(1, min_encoding_indices, 1)


        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 2, 1).contiguous()
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_distance(self, z):
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        d = torch.reshape(d, (z.shape[0], -1, z.shape[2])).permute(0,2,1).contiguous()
        return d

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

        return z_q


class VectorBases(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    ____________________________________________
    """

    def __init__(self,
                n_e=128,
                e_dim=2048,
                tau=0.1,
                device='cpu',
                opts=None
        ):
        super(VectorBases, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.tau = tau # temperature

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
        self.device = device

        self.opts = opts
        self.criterion = nn.MSELoss()
        self.lambda_orth = opts.lambda_orth if self.opts is not None else 1.0
        self.lambda_norm = opts.lambda_norm if self.opts is not None else 1.0

    def init_from_mesh_feature(self, mesh, example_feature):
        """
            mesh (trimesh.Trimesh): mesh
            example_feature (torch.tensor): per vertex feature 
        """
        import trimesh
        # import pdb;pdb.set_trace()
        _, face_index = trimesh.sample.sample_surface_even(mesh, count=self.opts.n_corr)
        # Vs = mesh.vertices[mesh.faces[face_index]] #[Num verts, triangle indices (3), val (3)]
        weight = example_feature[mesh.faces[face_index]].mean(1).float().requires_grad_()
        #print(self.embedding.weight.mean())
        self.embedding = nn.Embedding.from_pretrained(weight)
        #print(self.embedding.weight.mean())
        # self.embedding.weight.copy_(weight)       

    def forward(self, z):
        """
            z (torch.tensor): shape matching feature (will not be optimized!)
        """
        # [B, N, C] -> [B*N, C]
        z_flattened = z.view(-1, self.e_dim)
        
        ## no need to pass gradient to feature!
        d = torch.matmul(z_flattened, self.embedding.weight.t())
        norm_a = torch.norm(z_flattened, dim=1, keepdim=True)
        norm_b = torch.norm(self.embedding.weight, dim=1, keepdim=True)
        d = d / (norm_a * norm_b.t())

        N,M = d.shape
        temperature = np.sqrt(N*M*1.0) * self.tau
        
        d = F.softmax(d * temperature, dim=1)
        d = d.detach()

        # soft correspondence
        z_q = torch.matmul(d, self.embedding.weight).view(z.shape)

        # # one hot (hard correspondence)
        # min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        # min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z) 
        # min_encodings.scatter_(1, min_encoding_indices, 1)
        # # get quantized latent vectors
        # z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        return z_q

    def get_loss(self, loss_dict):
        """
            loss_dict (dict): loss dictionary
        """
        orth_loss = self.criterion(
            torch.eye(self.n_e).to(self.device),
            self.embedding.weight @ self.embedding.weight.t()
        ) * self.lambda_orth

        norm_loss = torch.norm(self.embedding.weight) * self.lambda_norm
        
        loss_dict['l_orth'] = orth_loss
        loss_dict['l_norm'] = norm_loss
        return loss_dict

    def get_distance(self, z):
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        d = torch.reshape(d, (z.shape[0], -1, z.shape[2])).permute(0,2,1).contiguous()
        return d