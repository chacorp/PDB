import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
import numpy as np

#import igl
import sys
import pickle

from pathlib import Path
abs_path = str(Path(__file__).parents[1].absolute())
sys.path+=[abs_path, f'{abs_path}/mesh_utils']

from tqdm import tqdm

from utils.nfr_utils import Normalizer, get_dfn_info, calc_cent
from utils.remesh_utils import ICT_face_model
from utils.mesh_utils import Renderer, calc_norm_torch, get_mesh_operators, get_jacobian_matrix
from utils.mesh_utils import compute_wks_autoscale
from utils.nfr_utils import reconstruct_jacobians

from models.CNN import TextureEncoder
from models.decoder import SkinningDecoder, AdaINDiffusionNetDecoder, BaseDiffusionNetDecoder
from models.encoder import BaseDiffusionNetEncoder, DiffusionNetEncoder, DoubleDiffusionNetEncoder
# from models.NFR import *
# from models.siren_decoder import SurfaceDeformationField

class NFS(nn.Module):
    def __init__(self, 
                 opts, 
                 mesh_dfn_info=None, 
                 print_param=True, 
                 is_train=False,
                 design="nfr",
                ):
        super().__init__()
        """
        Args:
            opts (Namespace): config options
        """
        self.opts = opts
        self.device = self.opts.device if opts is not None else 'cpu'
        self.is_train = self.opts.is_train if opts is not None else False
        self.ict_face_only = self.opts.ict_face_only if opts is not None else True
        self.design = self.opts.design if opts is not None else design
        self.scale_exp = self.opts.scale_exp if opts is not None else 1.0
        if self.scale_exp != 1.0:
            print(f"[ using scaled expression code!!: {self.scale_exp} ]")
        
        self.enc_in_key = self.opts.enc_feature_type if opts is not None else 'cents&norms'
        self.dec_in_key = self.opts.dec_feature_type if opts is not None else 'cents&norms'
        self.out_key = self.opts.dec_type if opts is not None else 'disp'
        
        self.img_feat_dim = self.opts.img_feat_dim if opts is not None else 128
        self.ltn_dim = self.opts.ltn_dim if opts is not None else 256
        self.rig_dim = self.opts.rig_dim if opts is not None else 128
        self.id_dim = self.opts.id_dim if opts is not None else 128
        self.seg_dim = self.opts.seg_dim if opts is not None else 20
        self.use_decimate = self.opts.use_decimate if opts is not None else False
        self.corr_dim = self.opts.corr_dim if opts is not None else 2048
        
        ### mesh autoencoder
        in_shape_dict = {'cents&norms':6, 'cents':3, 'cents&norms&seg':7, 'wks':128, 'cents&norms&disp':9}
        out_shape_dict = {'vert': 3, 'disp': 3, 'jacob': 9}
        self.in_shape_dict = in_shape_dict
        
        self.compute_wks_autoscale = compute_wks_autoscale
        
        if self.design == 'new2': # added segment encoder, skinning block (MLP)
            # self.img_encoder = TextureEncoder()
            # self.img_fc = nn.Linear(128, self.img_feat_dim)
            
            self.mesh_id_encoder = BaseDiffusionNetEncoder(
                in_shape=in_shape_dict[self.enc_in_key],#+self.img_feat_dim,
                pre_computes=mesh_dfn_info,
                out_shape=self.id_dim,
            )
            self.mesh_exp_encoder = BaseDiffusionNetEncoder(
                in_shape=in_shape_dict[self.enc_in_key],#+self.img_feat_dim,
                pre_computes=mesh_dfn_info,
                out_shape=self.rig_dim,
            )
            self.mesh_seg_encoder = BaseDiffusionNetEncoder(
                in_shape=in_shape_dict[self.enc_in_key],#+self.img_feat_dim,
                pre_computes=mesh_dfn_info,
                out_shape=self.seg_dim,
                outputs_at='vertices',
            )
            self.mesh_decoder = SkinningDecoder(
                in_dim=in_shape_dict[self.enc_in_key],#+self.img_feat_dim, 
                id_dim=self.id_dim,
                exp_dim=self.rig_dim,
                seg_dim=self.seg_dim,
                out_shape=out_shape_dict[self.out_key],
            )
        elif self.design == 'new3':
            self.mesh_encoder = DoubleDiffusionNetEncoder(
                C_in=in_shape_dict[self.enc_in_key],
                C_out=self.ltn_dim,
                C_width=256,
                ID_out=self.id_dim, 
                EXP_out=self.rig_dim,
            )
            self.mesh_decoder = AdaINDiffusionNetDecoder(
                #C_in=in_shape_dict[self.dec_in_key]+self.ltn_dim+self.rig_dim, 
                C_in=self.ltn_dim+self.rig_dim, 
                C_out=3, # displacement
                ID_dims=self.id_dim, 
            )
        elif self.design == 'new4':
            self.mesh_id_encoder = BaseDiffusionNetEncoder(
                in_shape=in_shape_dict[self.enc_in_key],
                out_shape=self.id_dim,
            )
            self.mesh_exp_encoder = AdaINDiffusionNetDecoder(
                C_in=in_shape_dict[self.enc_in_key], 
                C_out=self.rig_dim,
                ID_dims=self.id_dim, 
                outputs_at='global_mean',
            )
            self.mesh_decoder = AdaINDiffusionNetDecoder(
                #C_in=in_shape_dict[self.dec_in_key]+self.ltn_dim+self.rig_dim, 
                C_in=self.ltn_dim+self.rig_dim, 
                C_out=3, # displacement
                ID_dims=self.id_dim, 
                outputs_at='vertices',
            )
        elif self.design == 'new5':
            self.mesh_id_encoder = BaseDiffusionNetEncoder(
                in_shape=in_shape_dict[self.enc_in_key],
                pre_computes=mesh_dfn_info,
                out_shape=self.id_dim,
            )
            self.mesh_exp_encoder = BaseDiffusionNetEncoder(
                in_shape=in_shape_dict[self.enc_in_key],
                pre_computes=mesh_dfn_info,
                out_shape=self.rig_dim,
            )
            self.mesh_decoder = BaseDiffusionNetDecoder(
                # in_shape=self.ltn_dim,
                emb_dim=self.corr_dim, 
                id_dim=self.id_dim, 
                exp_dim=self.rig_dim,
                pre_computes=mesh_dfn_info,
                out_shape=out_shape_dict[self.out_key],
                outputs_at='vertices',
                use_canon=self.opts.use_canon,
                device=self.device,
                opts=self.opts
            )
        # elif self.design == 'new3':             
        #     self.mesh_id_encoder = BaseDiffusionNetEncoder(
        #         in_shape=in_shape_dict[self.enc_in_key]+self.img_feat_dim,
        #         pre_computes=mesh_dfn_info,
        #         out_shape=self.id_dim,
        #     )
        #     self.mesh_exp_encoder = BaseDiffusionNetEncoder(
        #         in_shape=in_shape_dict[self.enc_in_key]+self.img_feat_dim,
        #         pre_computes=mesh_dfn_info,
        #         out_shape=self.rig_dim,
        #     )
        #     self.mesh_seg_encoder = BaseDiffusionNetEncoder(
        #         in_shape=in_shape_dict[self.enc_in_key]+self.img_feat_dim,
        #         pre_computes=mesh_dfn_info,
        #         out_shape=self.seg_dim,
        #         outputs_at='vertices',
        #     )
        #     self.mesh_decoder = BaseDiffusionNetEncoder(
        #         in_dim=in_shape_dict[self.enc_in_key]+self.img_feat_dim+self.seg_dim, 
        #         pre_computes=mesh_dfn_info,
        #         out_shape=out_shape_dict[self.out_key],
        #         outputs_at='vertices',
        #     )
        # elif self.design == 'new4':
        #     self.mesh_id_encoder = BaseDiffusionNetEncoder(
        #         in_shape=in_shape_dict[self.enc_in_key],
        #         pre_computes=mesh_dfn_info,
        #         out_shape=self.id_dim,
        #     )
        #     self.mesh_exp_encoder = BaseDiffusionNetEncoder(
        #         in_shape=in_shape_dict[self.enc_in_key],
        #         pre_computes=mesh_dfn_info,
        #         out_shape=self.exp_dim,
        #     )
        #     self.mesh_decoder = AdaINDiffusionNetDecoder(
        #         C_in=in_shape_dict[self.enc_in_key]+self.rig_dim,
        #         C_width=128,
        #         ID_dims=self.id_dim,
        #         C_out=out_shape_dict[self.out_key],
        #     )
        else:
            raise NotImplementedError
            
        # print all layer params number
        if print_param:
            self.print_parameter_num()
        
        self.criterion = nn.MSELoss()
        self.calc_norm_torch = calc_norm_torch
        self.calc_cent = calc_cent
        self.get_jacobian_matrix = get_jacobian_matrix
        
        
        self.ict_face_model = ICT_face_model(face_only=False, device=self.device, use_decimate=self.use_decimate)
        # self.ict_face_model_fo = ICT_face_model(face_only=True, device=self.device, use_decimate=self.use_decimate)
        
        self.ict_neutral = torch.from_numpy(self.ict_face_model.neutral_verts).float().to(self.device)
        self.ict_faces = torch.from_numpy(self.ict_face_model.faces).long().to(self.device)
        
        if self.design != 'new5':
            self.renderer = Renderer(view_d=2.5, img_size=256, fragments=True)
            
        if self.design == 'new5' and self.opts.use_canon:
            #### NOTE: hard coded
            print('loading example_feature (from hard coded path!)')
            example_feature = torch.load('/data/sihun/ICT-audio2face/precompute-synth-fullhead/100_diff3f.pth')
            self.mesh_decoder.canonical_bases.init_from_mesh_feature(self.ict_face_model.get_mesh(), example_feature)
        #---------------------------------------------------------------------------------
        if self.opts.seg_dim == 20:
            seg_npy = f'{abs_path}/utils/ict/ICT_segment_onehot.npy'
        elif self.opts.seg_dim == 24:
            seg_npy = f'{abs_path}/utils/ict/ICT_segment_onehot_24.npy'
        elif self.opts.seg_dim == 14:
            seg_npy = f'{abs_path}/utils/ict/ICT_segment_onehot_14.npy'
        elif self.opts.seg_dim == 6:
            seg_npy = f'{abs_path}/utils/ict/ICT_segment_onehot_06.npy'
        else:
            raise NotImplementedError(f"no segment map for seg_dim: {self.opts.seg_dim}")
        
        self.ict_vert_segment = torch.from_numpy(np.load(seg_npy)).to(self.device)
        self.ict_vert_segment = self.ict_vert_segment.argmax(-1).long()
        # self.ict_vert_segment_fo = self.ict_vert_segment[:self.ict_face_model_fo.v_num]
        #---------------------------------------------------------------------------------
        
        
        #---------------------------------------------------------------------------------
        self.get_mesh_operators = get_mesh_operators
        
        if self.opts.dec_type=='jacob':            
            from utils.deformation_transfer import deformation_gradient
            self.normalizer = Normalizer(f"{abs_path}/{self.opts.std_file}", self.device)
            self.myfunc = deformation_gradient.apply
        #---------------------------------------------------------------------------------
        
        #---------------------------------------------------------------------------------
        if self.design == 'new3':
            from utils.fmap_utils import RegularizedFMNet, SURFMNetLoss, Similarity, SquaredFrobeniusLoss
            self.fmap_net = RegularizedFMNet(bidirectional=True)
            self.surfmnet_loss = SURFMNetLoss(
                float(self.opts.lambda_w_bij), 
                float(self.opts.lambda_w_orth), 
                float(self.opts.lambda_w_lap)
            ) # Orthogonality + Bijectivity + Laplacian Commutativity
            self.permutation_similarity = Similarity(tau=0.07)
            self.align_loss = SquaredFrobeniusLoss()
        #---------------------------------------------------------------------------------
    
    def compute_permutation_matrix(self, feat_x, feat_y, bidirectional=False, normalize=True):
        if normalize:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)
        similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

        # sinkhorn normalization
        Pxy = self.permutation_similarity(similarity)

        if bidirectional:
            Pyx = self.permutation_similarity(similarity.transpose(1, 2))
            return Pxy, Pyx
        else:
            return Pxy

    def print_parameter_num(self):
        """Prints parameters
        """
        print("===========< NFS >===========")
        if self.design=='nfr':
            print(f"[img_encoder]: \t{self.count_parameters(self.img_encoder)}")
            print(f"[img_fc]: \t{self.count_parameters(self.img_fc)}")
        elif self.design=='new3':
            print(f"[mesh_encoder]: \t{self.count_parameters(self.mesh_encoder)}")
            print(f"[mesh_decoder]: \t{self.count_parameters(self.mesh_decoder)}")
        else:
            print(f"[mesh_decoder]: \t{self.count_parameters(self.mesh_decoder)}")
        
            print(f"[mesh_id_encoder]: \t{self.count_parameters(self.mesh_id_encoder)}")
            print(f"[mesh_exp_encoder]: \t{self.count_parameters(self.mesh_exp_encoder)}")
            
        if 'new2' in self.design:
            print(f"[mesh_seg_encoder]: \t{self.count_parameters(self.mesh_seg_encoder)}")
        print("-------------------------------")
        print(f"[total]: \t{self.count_parameters(self)}")
        print("===============================") 
        
    def count_parameters(self, model):
        try:
            return sum(p.numel() for p in model if p.requires_grad)
        except:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def set_neutral_ict(self, ict_basedir):
        self.ict_basedir = ict_basedir
        self.ict_precompute = f'{self.ict_basedir}/precompute-synth-fullhead'
        self.ict_precompute_fo = f'{self.ict_basedir}/precompute-synth-face_only'
        self.ict_precompute_nf = f'{self.ict_basedir}/precompute-synth-narrow_face'
        
        if self.use_decimate:
            v_idx = torch.from_numpy(self.ict_face_model.ict_deci["v_idx"])
            self.ict_vert_segment = self.ict_vert_segment[v_idx]
            self.ict_neutral = self.ict_neutral[v_idx]
            self.ict_faces = torch.from_numpy(self.ict_face_model.ict_deci["new_f"]).long().to(self.device)
            
            self.neu_dfn_info = pickle.load(open(os.path.join(self.ict_precompute, f"100_dfn_info-deci.pkl"), 'rb'))
            neu_operators = os.path.join(self.ict_precompute, f"100_operators-deci.pkl")
            self.neu_operators = pickle.load(open(neu_operators, mode='rb'))
            self.neu_img = torch.from_numpy(np.load(os.path.join(self.ict_precompute, f"100_img.npy"))).to(self.device).float()
        else:
            self.neu_dfn_info  = pickle.load(open(os.path.join(self.ict_precompute, f"100_dfn_info.pkl"), 'rb'))
            neu_operators = os.path.join(self.ict_precompute, f"100_operators.pkl")
            self.neu_operators = pickle.load(open(neu_operators, mode='rb'))
            self.neu_img = torch.from_numpy(np.load(os.path.join(self.ict_precompute, f"100_img.npy"))).to(self.device).float()
            
            self.neu_fo_dfn_info  = pickle.load(open(os.path.join(self.ict_precompute_fo, f"100_dfn_info.pkl"), 'rb'))
            neu_operators = os.path.join(self.ict_precompute_fo, f"100_operators.pkl")
            self.neu_fo_operators = pickle.load(open(neu_operators, mode='rb'))
            self.neu_fo_img = torch.from_numpy(np.load(os.path.join(self.ict_precompute_fo, f"100_img.npy"))).to(self.device).float()
            
            self.neu_nf_dfn_info  = pickle.load(open(os.path.join(self.ict_precompute_nf, f"100_dfn_info.pkl"), 'rb'))
            neu_operators = os.path.join(self.ict_precompute_nf, f"100_operators.pkl")
            self.neu_nf_operators = pickle.load(open(neu_operators, mode='rb'))
            self.neu_nf_img = torch.from_numpy(np.load(os.path.join(self.ict_precompute_nf, f"100_img.npy"))).to(self.device).float()
            
        self.local_feat_ict=None
        print('loading cache (ict neutral mesh)')
            
    def get_mesh_encoder_parameters(self):
        param_list = [
            *self.mesh_id_encoder.parameters(), 
            *self.mesh_exp_encoder.parameters(),
        ]
        
        if 'new2' in self.design:
            param_list = param_list + [*self.mesh_seg_encoder.parameters()]
        return param_list
    
    def get_mesh_decoder_parameters(self):
        return self.mesh_decoder.parameters()
        
    def get_mesh_autoencoder_parameters(self):
        """no audio encoder"""
        param_list = [
            *self.mesh_id_encoder.parameters(), 
            *self.mesh_exp_encoder.parameters(),
            *self.mesh_decoder.parameters(), 
        ]
        if 'new2' in self.design:
            param_list = param_list + [*self.mesh_seg_encoder.parameters()]
        return param_list
    
    def get_mesh_ae_parameters(self):
        """no audio encoder"""
        param_list = [
            *self.mesh_encoder.parameters(), 
            *self.mesh_decoder.parameters(), 
        ]
        return param_list
    
    def get_img_feat(self, img):
        """
        Args:
            img (torch.tensor): rendered image [1,H,W,C] or [H,W,C]
        Returns:
            image feature
        """
        if len(img.shape) < 4:
            img = img[None]
        return self.img_fc(self.img_encoder(img[..., :3].permute(0, 3, 1, 2)))
          
    def get_img_feat_from_mesh(self, tgt_mesh, return_feat=False):
        """
        Args
        --------
            tgt_mesh (trimesh): trimesh.Trimesh
            return_feat (bool): if True, return image feature
        Returns
        --------
            tgt_img (torch.tensor): rendered image
        """
        tgt_img = self.renderer.render_img(tgt_mesh).float().to(self.device)
        if return_feat:
            tgt_img_feat = self.get_img_feat(tgt_img).unsqueeze(1) #------------------- [1, 1, 128]
            return tgt_img_feat
        else:
            return tgt_img
    
    def get_precomputes(self, tgt_mesh, no_img=False):
        if not no_img:
            tgt_img = self.renderer.render_img(tgt_mesh).float().to(self.device)
            
        tgt_dfn_info = get_dfn_info(tgt_mesh, map_location=self.device)
        
        if not no_img:
            return tgt_dfn_info
        return tgt_img, tgt_dfn_info
    
    def calc_vert(self, pred_jacob, myfunc, operators):
        """Poisson solve
        Args
        --------
            pred_jacob (torch.tensor): jacobian of the mesh
            myfunc (torch.autograd.Function): deformation_gradient
            operators (tuple): tuple of mesh operators for poisson solving \
                               (SuperLU, face_to_vert_idxs, face_to_vert_values, cupy csr matrix)
        Return
        --------
            out_pred (torch.tensor): deformed vertices of the mesh
        """
        lu_solver, idxs, vals, rhs = operators
        pred_vert = myfunc(pred_jacob, lu_solver, idxs, vals, rhs.shape)
        pred_vert = pred_vert.float() # float64 -> float32
        pred_vert = pred_vert - pred_vert.mean(dim=1, keepdim=True)
        return pred_vert
    
    def L2_regularization(self, pred):
        """
        Args:
            pred (torch.tensor): predicted expression code
        
        Returns:
            loss
        """
        #loss = F.normalize(pred, p=2.0, dim=-1, eps=1e-12)
        loss = LA.norm(pred, dim=-1).mean()
        return loss
    
    def non_ict_loss(self, pred):
        """Reference from Neural Face Rigging for Animating and Retargeting Facial Meshes in the Wild [Qin et al. 2023], Eq.(3)
        L_FACS = {   
             -x, x < 0 
              0, 0 <= x < 1
            x-1, x > 1
        }
        Args:
            pred (torch.tensor): predicted expression code
        
        Returns:
            loss
        """
        
        loss = torch.where(pred < 0, -pred, torch.where(pred > 1, pred - 1, torch.zeros_like(pred))).mean()
        return loss
        
    def label_smoothing(self, one_hot, smoothing=0.2):
        """Reference: function borrrowed from DiffusionNet github -> but it is not used!
        Args:
            one_hot (torch.tensor):
            smoothing (float):
        """
        n_class = one_hot.shape[-1]
        one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        return one_hot
        
    def get_local_feature(self, vertices, faces, img_feat, at='verts'):
        """
        Args
        ----------
            vertices (torch.tensor): [B, V, 3] vertices from the mesh
            faces (torch.tensor): [F, 3] vertex indices of each triangle
        
        Returns
        ----------
            local_feat (torch.tensor) [B, V, 134] or [B, F, 134]
        """
        if at=='verts':
            verts_pos = vertices # [1, V, 3]
            verts_nrm = self.calc_norm_torch(verts_pos, faces, at='verts') # [1, V, 3]
            
            B, V, _ = vertices.shape
            verts_img_feat = img_feat.repeat(1, V, 1) # [1, V, 128]
            local_feat = torch.cat([verts_pos, verts_nrm, verts_img_feat], dim=-1) # [1, V, 3+3+128]
        else:
            verts_pos = vertices # [1, V, 3]
            tri_centr = self.calc_cent(verts_pos.squeeze(0), faces, mode='torch').unsqueeze(0)
            tri_norms = self.calc_norm_torch(verts_pos, faces)

            B, F = vertices.shape[0], faces.shape[0]
            tri_img_feat = img_feat.repeat(1, F, 1)
            local_feat = torch.cat([tri_centr, tri_norms, tri_img_feat], dim=-1) # [1, V, 3+3+128]
            
        return local_feat   
    
    def get_local_feature_no_img(self, displacements, vertices, faces, dfn_info, at='verts'):
        """
        Args
        ----------
            displacements (torch.tensor): [B, V, 3] vertices from the mesh
            vertices (torch.tensor): [B, V, 3] vertices from the mesh
            faces (torch.tensor): [F, 3] vertex indices of each triangle
        
        Returns
        ----------
            local_feat (torch.tensor) [B, V or F, enc_feature_type]
        """
        
        if at=='verts':
            if self.opts.enc_feature_type == 'wks':
                local_feat = self.compute_wks_autoscale(dfn_info[2].unsqueeze(0), dfn_info[3].unsqueeze(0), dfn_info[0].unsqueeze(0))
            else:
                verts_pos = vertices # [1, V, 3]
                verts_nrm = self.calc_norm_torch(verts_pos, faces, at='verts') # [1, V, 3]
                
                #B, V, _ = vertices.shape
                
                if self.in_shape_dict[self.enc_in_key] == 6:
                    local_feat = torch.cat([verts_pos, verts_nrm], dim=-1) # [1, V, 3+3+3]
                if self.in_shape_dict[self.enc_in_key] == 9:
                    local_feat = torch.cat([verts_pos, verts_nrm, displacements], dim=-1) # [1, V, 3+3+3]
        else:
            verts_pos = vertices # [1, V, 3]
            tri_centr = self.calc_cent(verts_pos.squeeze(0), faces, mode='torch').unsqueeze(0)
            tri_norms = self.calc_norm_torch(verts_pos, faces)
            verts_disp = displacements+vertices
            tri_disp = self.calc_cent(verts_disp.squeeze(0), faces, mode='torch').unsqueeze(0) - tri_centr

            #B, F = vertices.shape[0], faces.shape[0]
            
            if self.in_shape_dict[self.enc_in_key] == 6:
                local_feat = torch.cat([tri_centr, tri_norms], dim=-1) # [1, V, 3+3]
            if self.in_shape_dict[self.enc_in_key] == 9:
                local_feat = torch.cat([tri_centr, tri_norms, tri_disp], dim=-1) # [1, V, 3+3+3]
            
        return local_feat
        
    def encode_(self, vert_feat, dfn_info, verbose=False):
        """batch size is 1
        Args
        ----------
            vert_feat (torch.tensor): [B, V, 6] per-vertex features
            dfn_info (list): mesh precompute features for DiffusionNet
        
        Returns
        ----------
            vtx_corr_feat (torch.tensor): [B, V, 256] per-vertex correspondence feature
            id_code (torch.tensor): [B, ID] identity code per batch
            exp_code (torch.tensor): [B, EXP] expression code per batch
        """
        if verbose:
            print("encoding ...")
        self.mesh_encoder.update_precomputes(dfn_info)
        vtx_corr_feat, id_code, exp_code = self.mesh_encoder(vert_feat)
        return vtx_corr_feat, id_code, exp_code
    
    def encode_id(self, vert_feat, dfn_info, verbose=False):
        """batch size is 1
        Args
        ----------
            vert_feat (torch.tensor): [B, V, 6] pre-vertex features
            dfn_info (list): mesh precompute features for DiffusionNet
        
        Returns
        ----------
            id_code (torch.tensor): [B, ID] identity code per batch
        """
        if verbose:
            print("encoding id ...")
        self.mesh_id_encoder.update_precomputes(dfn_info)
        id_code = self.mesh_id_encoder(vert_feat)
        return id_code
    
    def encode_exp(self, vert_feat, dfn_info, batch_process=True, verbose=False):
        """batch is frame
        Args
        ----------
            vert_feat (torch.tensor): [B, V, 6] pre-vertex features
            dfn_info (list): mesh precompute features for DiffusionNet
        
        Returns
        ----------
            exp_code (torch.tensor): [B, Exp] expression code per batch
        """
        self.mesh_exp_encoder.update_precomputes(dfn_info)
        if batch_process:
            exp_code = self.mesh_exp_encoder(vert_feat) # [1, Rig]
        else:
            exp_code = []
            
            if verbose:
                pbar = tqdm(vert_feat, desc="encoding exp...")
            else:
                pbar = vert_feat
                
            for v_f in pbar:
                exp_c = self.mesh_exp_encoder(v_f[None])
                exp_code.append(exp_c)
            exp_code = torch.vstack(exp_code) #------------- [W, Rig]
        return exp_code
    
    def encode_exp_adain(self, vert_feat, id_code, dfn_info, batch_process=True, verbose=False):
        """batch is frame
        Args
        ----------
            vert_feat (torch.tensor): [B, V, 6] pre-vertex features
            id_code (torch.tensor): [B, V, ID] ID code (repeated per vertex)
            dfn_info (list): mesh precompute features for DiffusionNet
        
        Returns
        ----------
            exp_code (torch.tensor): [B, Exp] expression code per batch
        """
        self.mesh_exp_encoder.update_precomputes(dfn_info)
        if batch_process:
            exp_code = self.mesh_exp_encoder(vert_feat, id_code) # [B, Exp]
        else:
            exp_code = []
            
            if verbose:
                pbar = tqdm(vert_feat, id_code, desc="encoding exp...")
            else:
                pbar = zip(vert_feat, id_code)
                
            for v_f, id_ in pbar:
                exp_c = self.mesh_exp_encoder(v_f[None], id_[None])
                exp_code.append(exp_c)
            exp_code = torch.vstack(exp_code) #------------- [B, Exp]
        return exp_code
    
    def encode_seg(self, vert_feat, dfn_info):
        """batch size is 1
        Args
        ----------
            vert_feat (torch.tensor): [B, V, 6] pre-vertex features
            dfn_info (list): mesh precompute features for DiffusionNet
        
        Returns
        ----------
            seg_code (torch.tensor): [B, Exp] segment code per batch
        """
        self.mesh_seg_encoder.update_precomputes(dfn_info)
        seg_code = self.mesh_seg_encoder(vert_feat) # [1, ID]
        return seg_code
    
    def get_inputs_ict(self, pred_exp_coeff, region=0, v_num=None, faces_ict=None):
        """get decoder input from ict neutral face mesh
        Args:
            pred_exp_coeff (torch.tensor): not used in the process
            region (int): label for ict face region [0: face to shoulder, 1: face to neck, 2: narrow face]
        
        Returns:
            inputs_ict (tuple): inputs for decoder
        """
        
        if v_num is None or faces_ict is None:
            v_num, faces_ict = self.ict_face_model.get_random_v_and_f(region)
        
        faces_ict = faces_ict.to(self.device)
        template_ict = self.ict_neutral[None,:v_num]
        
        if region == 0:
            neu_img = self.neu_img
            operators_ict = self.neu_operators
            dfn_info_ict = self.neu_dfn_info
        elif region == 1:
            neu_img = self.neu_fo_img
            operators_ict = self.neu_fo_operators
            dfn_info_ict = self.neu_fo_dfn_info
        else:
            neu_img = self.neu_nf_img
            operators_ict = self.neu_nf_operators
            dfn_info_ict = self.neu_nf_dfn_info
            
        img_feat_ict = self.get_img_feat(neu_img).unsqueeze(1) # [B, 1, 128]
        
        local_feat_ict = self.get_local_feature(template_ict, faces_ict, img_feat_ict)
        B = pred_exp_coeff.shape[0]
        
        with torch.no_grad(): ## no need to pass it to id encoder
            pred_id_coeff_ict = self.encode_id(local_feat_ict, dfn_info_ict).repeat(B, 1)# [1, ID]
        
        if 'new2' in self.design:
            pred_seg_coeff_ict = self.encode_seg(local_feat_ict, dfn_info_ict)# [1, V, Seg]
        else:
            pred_seg_coeff_ict = None
        
        if self.opts.dec_type=='jacob':
            local_feat_ict = self.get_local_feature(template_ict, faces_ict, img_feat_ict, at='faces')
            
        return (
            local_feat_ict.repeat(B, 1, 1), 
            pred_exp_coeff, 
            pred_id_coeff_ict, 
            pred_seg_coeff_ict.repeat(B, 1, 1), 
            None, # style_emb #(not used)
            template_ict.repeat(B, 1, 1), 
            faces_ict, 
            operators_ict,
        )
    
    def get_inputs_ict_no_img(self, pred_exp_coeff, region=0, v_num=None, faces_ict=None):
        """get decoder input from ict neutral face mesh
        Args:
            pred_exp_coeff (torch.tensor): not used in the process
            region (int): label for ict face region [0: face to shoulder, 1: face to neck, 2: narrow face]
        
        Returns:
            inputs_ict (tuple): inputs for decoder
        """
        
        if v_num is None or faces_ict is None:
            v_num, faces_ict = self.ict_face_model.get_random_v_and_f(region)
        
        faces_ict = faces_ict.to(self.device)
        template_ict = self.ict_neutral[None,:v_num]
        
        if region == 0:
            operators_ict = self.neu_operators
            dfn_info_ict = self.neu_dfn_info
        elif region == 1:
            operators_ict = self.neu_fo_operators
            dfn_info_ict = self.neu_fo_dfn_info
        else:
            operators_ict = self.neu_nf_operators
            dfn_info_ict = self.neu_nf_dfn_info
        
        local_feat_ict = self.get_local_feature_no_img(template_ict*0, template_ict, faces_ict, dfn_info_ict)
        
        B = pred_exp_coeff.shape[0]

        if self.design=='new3':
            self.mesh_encoder.update_precomputes(dfn_info_ict)
            vtx_corr_feat, pred_id_coeff_ict, pred_exp_coeff = self.mesh_encoder(local_feat_ict.repeat(B, 1, 1))
            # with torch.no_grad(): ## no need to pass it to id encoder
            #     pred_id_coeff_ict = self.encode_id(local_feat_ict, dfn_info_ict).repeat(B, 1)# [1, ID]
            return (
                vtx_corr_feat, 
                pred_exp_coeff, 
                pred_id_coeff_ict, 
                None, # 
                None, # (not used)
                template_ict.repeat(B, 1, 1), 
                faces_ict, 
                operators_ict,
            )
        else:
            with torch.no_grad(): ## no need to pass it to id encoder
                pred_id_coeff_ict = self.encode_id(local_feat_ict, dfn_info_ict).repeat(B, 1)# [1, ID]
            
            if 'new2' in self.design:
                pred_seg_coeff_ict = self.encode_seg(local_feat_ict, dfn_info_ict).repeat(B, 1, 1)# [B, V, Seg]
            else:
                pred_seg_coeff_ict = None
            
        if self.opts.dec_type=='jacob':
            local_feat_ict = self.get_local_feature_no_img(template_ict*0, template_ict, faces_ict, dfn_info_ict, at='faces')
        
        return (
            local_feat_ict.repeat(B, 1, 1), 
            pred_exp_coeff, 
            pred_id_coeff_ict, 
            pred_seg_coeff_ict, # 
            None, # style_emb #(not used)
            template_ict.repeat(B, 1, 1), 
            faces_ict, 
            operators_ict,
        )
        
    @torch.no_grad()
    def encode_mesh(self,
                    gt_vertices,
                    src_mesh, 
                    tgt_mesh, 
                    batch_process=False,
                    use_filter=False,
                    kernel_size=5,
                    sigma=1.2,
                    use_scale=False,
                    scale=1.0,
                ):
        
        ## target mesh local feature
        tgt_img = self.renderer.render_img(tgt_mesh).float().to(self.device)
        tgt_img_feat = self.get_img_feat(tgt_img).unsqueeze(1) #------------------- [1, 1, 128]
        tgt_dfn_info = get_dfn_info(tgt_mesh, map_location=self.device)
        tgt_verts = torch.from_numpy(tgt_mesh.vertices)[None].to(self.device).float()
        tgt_faces = torch.from_numpy(tgt_mesh.faces).to(self.device)
        vert_feat = self.get_local_feature(tgt_verts, tgt_faces, tgt_img_feat)# [1, V, 3+3+128]
                
        with torch.no_grad():
            pred_id_coeff = self.encode_id(vert_feat, tgt_dfn_info)

            if 'new2' in self.design:
                pred_seg_coeff = self.encode_seg(vert_feat, tgt_dfn_info)# [1, V, Seg]
            else:
                pred_seg_coeff = None
        
        #pred_id_coeff = pred_id_coeff.unsqueeze(1) # ----------------------- [1, 1, ID]
        torch.cuda.empty_cache()
        ##------------------------------------------------------------------------------
        
        
        ## source mesh local feature
        gt_vertices = gt_vertices.to(self.device)
        src_img = self.renderer.render_img(src_mesh).float().to(self.device)
        src_img_feat = self.get_img_feat(src_img).unsqueeze(1) #------------------- [1, 1, 128]
        src_dfn_info = get_dfn_info(src_mesh, map_location=self.device)
        src_faces = torch.from_numpy(src_mesh.faces).to(self.device)
        
        vert_feat_exp = []
        for gt_v in gt_vertices:
            _exp = self.get_local_feature(gt_v[None], src_faces, src_img_feat).float()# [W, V, 3+3+128]
            vert_feat_exp.append(_exp)
        vert_feat_exp = torch.vstack(vert_feat_exp)
        #print(vert_feat_exp.shape)
        
        with torch.no_grad():
            pred_exp_coeff = self.encode_exp(vert_feat_exp, src_dfn_info, batch_process=batch_process, verbose=True)# [W, Rig]
        torch.cuda.empty_cache()
        ##------------------------------------------------------------------------------
        
        if use_filter:
            pred_exp_coeff = self.apply_gaussian_filter(
                pred_exp_coeff, 
                kernel_size=kernel_size, 
                sigma=sigma
            )
        
        if use_scale:
            pred_exp_coeff = pred_exp_coeff * scale
            
        return pred_exp_coeff, pred_id_coeff, pred_seg_coeff, vert_feat
    
    def encode_mesh_grad(self, vert_feat, vert_feat_exp, dfn_info):
        pred_seg_coeff = None
        pred_id_coeff = self.encode_id(vert_feat, dfn_info) # [1, ID]
        pred_exp_coeff = self.encode_exp(vert_feat_exp, dfn_info) # [W, Rig]
        
        if 'new2' in self.design:
            pred_seg_coeff = self.encode_seg(vert_feat, dfn_info) # [1, V, Seg]
                
        return pred_id_coeff, pred_exp_coeff, pred_seg_coeff
     
    def extract_mesh_feature(self, dfn_info, verbose=False):
        """batch size is 1
        Args
        ----------
            dfn_info (list): mesh precompute features for DiffusionNet
        
        Returns
        ----------
            mesh_feat (torch.tensor): [B, feat] identity code per batch
        """
        if verbose:
            print("extracting feature ...")
        self.mesh_fe.update_precomputes(dfn_info)
        
        wks = self.compute_wks_autoscale(
            evals=dfn_info[2].squeeze(0),
            evecs=dfn_info[3].squeeze(0),
            mass=dfn_info[0].squeeze(0)
        )
        mesh_feat = self.mesh_fe(wks)
        return mesh_feat
    
    
    def gaussian_kernel1d(self, kernel_size=5, sigma=1.0):
        x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel

    def apply_gaussian_filter(self, tensor, kernel_size, sigma):
        """Applies a Gaussian filter to the tensor.
        Args:
            tensor (torch.tensor): input tensor
            kernel_size (int): size of the kernel
            sigma (float): sigma value for gaussian filter
        Returns:
            filtered_tensor
        """
        kernel_size = int(kernel_size)

        # Generate the 1D Gaussian kernel
        kernel = self.gaussian_kernel1d(kernel_size, sigma)
        kernel = kernel.reshape(1, 1, -1).to(self.device) # (out_channels, in_channels, kernel_size)
        kernel = kernel.repeat(tensor.size(1), 1, 1) # [128, 1, kernel_size]
        tensor = tensor.transpose(0, 1).unsqueeze(0) # [1, 128, T]

        filtered_tensor = F.conv1d(tensor, kernel, padding=(kernel_size // 2), groups=tensor.size(1))

        # Transpose back to original shape
        filtered_tensor = filtered_tensor.squeeze(0).transpose(0, 1)
        return filtered_tensor
    
    def decode_grad(self, inputs, batch_process=True):
        """
        Args
        ----------
            inputs (tuple):
                local_feat (torch.tensor): [1, V, 6]
                pred_exp_coeff (torch.tensor): [B, Exp]
                pred_id_coeff (torch.tensor): [B, ID]
                pred_seg_coeff (torch.tensor): [B, V, Seg]
                style_emb (torch.tensor): [B, feat] (not used)
                template (torch.tensor): [B, V, 3] vertices
                faces (torch.tensor): [F, 3] triangle indicies
                operators (tuple): SuperLU, Laplacian mat ...
            batch_process (bool): if True, process in batch
        
        Returns
        ----------
            pred_outputs, pred_jacobians: predicted vertices and jacobians
        """
        (local_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff, _, template, faces, operators) = inputs
        
        B, _ = pred_exp_coeff.shape
        _, V, _  = local_feat.shape
        
        if batch_process:
            #local_feat = local_feat.repeat(W, 1, 1)
            pred_exp_coeff_repeat = pred_exp_coeff.unsqueeze(1).repeat(1, V, 1)
            pred_id_coeff_repeat = pred_id_coeff.unsqueeze(1).repeat(1, V, 1)
            
            if 'nfr' in self.design:
                inputs = torch.cat([local_feat, pred_exp_coeff_repeat, pred_id_coeff_repeat], dim=-1)
                pred_outputs = self.mesh_decoder(inputs) # [B, V or VF, out_dim]
            elif self.design == 'new3':
                inputs = torch.cat([local_feat, pred_exp_coeff_repeat], dim=-1)
                pred_outputs = self.mesh_decoder(inputs, pred_id_coeff_repeat)
            else:
                pred_outputs = self.mesh_decoder(local_feat, pred_exp_coeff_repeat, pred_id_coeff_repeat, pred_seg_coeff)
        else:
            pred_outputs = []
            pred_id_coeff_repeat = pred_id_coeff.unsqueeze(1).repeat(1, V, 1)  #--------------- [1, V, ID]
            
            for pred_rig in pred_exp_coeff.unsqueeze(1):
                if 'nfr' in self.design:
                    single_pred_rig = pred_rig[None].repeat(1, V, 1), #--------------- [1, V, Rig]
                    
                    inputs = torch.cat([local_feat, single_pred_rig, pred_id_coeff_repeat], dim=-1).float() #-- [1, V, 134+Rig+ID]
                    tmp_vertices = self.mesh_decoder(inputs) # [B, V or VF, out_dim]
                elif self.design == 'new3':
                    single_pred_exp = pred_exp[None].repeat(1, V, 1), #--------------- [1, V, Rig]
                    
                    inputs = torch.cat([local_feat, single_pred_exp], dim=-1).float() #-- [1, V, 134+Rig+ID]
                    tmp_vertices = self.mesh_decoder(inputs, pred_id_coeff_repeat) # [B, V or VF, out_dim]
                else:
                    tmp_vertices = self.mesh_decoder(local_feat, pred_rig[None], pred_id_coeff_repeat, pred_seg_coeff)
                    
                pred_outputs.append(tmp_vertices)
            pred_outputs = torch.vstack(pred_outputs)  #-------------------- [W, V, 3]
            
        if self.opts.dec_type=='jacob':
            pred_jacobians = self.normalizer.inv_normalize(pred_outputs)
            pred_jacobians = reconstruct_jacobians(pred_jacobians, repr='matrix')
            
            pred_outputs = self.calc_vert(pred_jacobians, self.myfunc, operators)
        else:
            if self.opts.dec_type=='disp':
                pred_outputs = template + pred_outputs
            pred_jacobians = self.get_jacobian_matrix(pred_outputs, faces, template, return_torch=True)
            
        return pred_outputs, pred_jacobians
    
    @torch.no_grad()
    def decode(self, 
               inputs, 
               tgt_mesh=None, # required if dec_type=='jacob'
               scale_exp=1.0, 
               return_exp=False,
               batch_process=False,
               use_filter=False,
               kernel_size=5,
               scale_mask=torch.ones(53),
               sigma=1.0
              ):
        """
        ## Note: batch_size is always 1 | used at inference
        Args
        ----------
            inputs:
                local_feat (torch.tensor): [1, V, 6]
                pred_exp_coeff (torch.tensor): [B, 1, Exp]
                pred_id_coeff (torch.tensor): [B, 1, ID]
                pred_seg_coeff (torch.tensor): [B, V, Seg]
                style_emb (torch.tensor): [B, feat]
                template (torch.tensor): [V, 3] vertices
                faces (torch.tensor): [F, 3] triangle indicies
                operators
            tgt_mesh (trimesh.Trimesh):
            scale_exp (float): scale the expression code
            return_exp (bool): if True, return epxression code
        Returns
        ----------
            pred_outputs: [batch_size, W, V, 3]
        """
        (local_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff, style_emb, template, faces, operators) = inputs
        
        if scale_exp != 1.0:
            pred_exp_coeff = pred_exp_coeff * scale_exp
        
        if use_filter:
            pred_exp_coeff = self.apply_gaussian_filter(
                pred_exp_coeff, 
                kernel_size=kernel_size, 
                sigma=sigma
            )
        W, _ = pred_exp_coeff.shape
        V = local_feat.shape[1]
        
        if batch_process:
            local_feat = local_feat.repeat(W, 1, 1)
            pred_exp_coeff_repeat = pred_exp_coeff.unsqueeze(1).repeat(1, V, 1)
            pred_id_coeff_repeat = pred_id_coeff.unsqueeze(1).repeat(W, V, 1)
            
            if 'nfr' in self.design:
                inputs = torch.cat([local_feat, pred_exp_coeff_repeat, pred_id_coeff_repeat], dim=-1)
                pred_outputs = self.mesh_decoder(inputs) # [B, V or VF, out_dim]
                
            else:
                pred_outputs = self.mesh_decoder(local_feat, pred_exp_coeff_repeat, pred_id_coeff_repeat, pred_seg_coeff)
        else:
            pred_outputs = []
            pred_id_coeff_repeat = pred_id_coeff.repeat(1, V, 1) #--------------- [1, V, ID]
        
            for pred_rig in tqdm(pred_exp_coeff, desc="decoding..."): # pred_rig -> [Rig]
                
                pred_rig = pred_rig[None]
                if 'nfr' in self.design:
                    pred_rig = pred_rig.repeat(1, V, 1) #--------------- [1, V, Rig]
                    input_frame = torch.cat([local_feat, pred_rig, pred_id_coeff_repeat], dim=-1).float() # [1, V, 134+Rig+ID]
                    with torch.no_grad():
                        tmp_vertices = self.mesh_decoder(input_frame) # [B, V or VF, out_dim]
                else:
                    with torch.no_grad():
                        tmp_vertices = self.mesh_decoder(local_feat, pred_rig, pred_id_coeff_repeat, pred_seg_coeff)
                            
                pred_outputs.append(tmp_vertices)
            pred_outputs = torch.vstack(pred_outputs)  #-------------------- [W, V, 3]
            ## empty_cache
            torch.cuda.empty_cache()
        
        
        ## converting outputs
        ## NFR design requires Poisson solving to obtain final vertices
        if self.opts.dec_type=='jacob':
            if operators is None:
                operators = self.get_mesh_operators(tgt_mesh)
            
            pred_jacobians = self.normalizer.inv_normalize(pred_outputs)
            pred_jacobians = reconstruct_jacobians(pred_jacobians, repr='matrix')
            
            with torch.no_grad():
                pred_outputs = self.calc_vert(pred_jacobians, self.myfunc, operators)
        else:
            if self.opts.dec_type=='disp':
                pred_outputs = template + pred_outputs
            pred_jacobians = self.get_jacobian_matrix(pred_outputs, faces, template, return_torch=True)
            
        if return_exp:
            return pred_outputs, pred_jacobians, pred_exp_coeff
        else:
            return pred_outputs, pred_jacobians
        
            
    def forward(self, batch, teacher_forcing=True, return_all=False, stage=9, epoch=0):
        if stage == 1:
            ### train with ICT only + train mesh autoencoder only (learn rig space)
            return self.stage1_forward(batch, teacher_forcing, return_all, epoch)
        elif stage == 11:
            ### train mesh decoder only
            return self.stage11_forward(batch, teacher_forcing, return_all, epoch)
        elif stage == 8:
            ### train mesh id encoder only
            return self.stage8_forward(batch, teacher_forcing, return_all, epoch)
        elif stage == 9:
            ### train mesh single encoder only
            return self.stage9_forward(batch, teacher_forcing, return_all, epoch)
        elif stage == 21:
            ### train mesh single encoder only
            return self.stage21_forward(batch, teacher_forcing, return_all, epoch)
        else:
            return NotImplementedError
    
    def stage1_forward(self, batch, teacher_forcing=True, return_all=False, epoch=0):
        """
        Args
        ----------
            batch (tuple):
                audio_feat (torch.Tensor):    [B, W, 768] (not used !!!)
                id_coeff (torch.Tensor):      [B, 128]    ICT-face model id coeff
                gt_rig_params (torch.Tensor): [B, 128] ICT-face model exp_coeff 
                template (torch.Tensor):      [B, V, 3] mesh vertices
                dfn_info (list(torch.Tensor)): DiffusionNet precomputes (mass, L, eval, evecs, gradX, gradY, faces)
                operators (list(torch.Tensor)): mesh operators for Poisson solve (NFR)
                vertices (torch.Tensor):      [B, V, 3] sequence of mesh vertices (animation)
                faces (torch.Tensor):         [F, 3] mesh faces indices
                img (torch.Tensor):           [B, H, W, C] rendered mesh image
            teacher_forcing (bool): used for Transformer model (not used !!!)
            return_all (bool): if True, return loss and all predicted outputs
            
        Returns
        ----------
            pred_outputs (torch.Tensor): [B, W, V, 3]
        """
        
        #_, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img, mesh_data = batch
        mesh_data = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data] 
        ## Send to device --------------------------------------------------------------
        gt_id_coeff = batch.id_coeff
        gt_rig_params = batch.gt_rig_params[:, :self.opts.rig_dim]
        template = batch.template
        gt_vertices = batch.vertices
        faces = batch.faces
        img = batch.img
        gt_normals = batch.normals
        dfn_info  = pickle.load(open(batch.dfn_info, 'rb'))
        operators = pickle.load(open(batch.operators, mode='rb'))
        ##------------------------------------------------------------------------------
        
        ## Encoding --------------------------------------------------------------------
        # source expression face
        vert_feat_exp = self.get_local_feature_no_img(torch.zeros_like(template), gt_vertices, faces, dfn_info)  # [B, V, 6]

        # target neutral face
        vert_feat = self.get_local_feature_no_img(torch.zeros_like(template), template, faces, dfn_info) # [1, V, 6]
        
        pred_id_coeff, pred_exp_coeff, pred_seg_coeff = self.encode_mesh_grad(vert_feat, vert_feat_exp, dfn_info)
        ##------------------------------------------------------------------------------
        
        if epoch < 100 and mesh_data == 'ict' and self.opts.warmup:
            inputs = (vert_feat, gt_rig_params, pred_id_coeff, pred_seg_coeff, None, template, faces, operators)
        else:
            inputs = (vert_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff, None, template, faces, operators)
        pred_outputs, pred_jacobians = self.decode_grad(inputs)
        
        ## Loss function ---------------------------------------------------------------
        loss = {}
            
        loss["recon_vDec"] = self.criterion(gt_vertices, pred_outputs)
        
        pred_normals = self.calc_norm_torch(pred_outputs, faces, at='verts')
        loss["norm_vDec"] = self.criterion(gt_normals, pred_normals)
        
        gt_jacobians = self.get_jacobian_matrix(gt_vertices, faces, template, return_torch=True)
        loss["jacob_vDec"] = self.criterion(gt_jacobians, pred_jacobians)
        
        if mesh_data == 'ict':
            loss["vert_rIEnc"] = self.criterion(gt_id_coeff, pred_id_coeff)
            loss["vert_rEEnc"] = self.criterion(gt_rig_params, pred_exp_coeff)
            
            region_num = self.ict_face_model.get_region_num(gt_vertices)
            v_num, faces = self.ict_face_model.get_random_v_and_f(region_num, mode='torch')
                
            if not self.opts.no_BP:
                pred_ICT_recon = self.ict_face_model.apply_coeffs_batch_torch(pred_id_coeff[:, :100], pred_exp_coeff[:, :53], region=region_num)
                loss["vert_vICT"] = self.criterion(gt_vertices, pred_ICT_recon)
        
            if not self.opts.no_BR:
                inputs_ict = self.get_inputs_ict_no_img(pred_exp_coeff.detach(), region=region_num, v_num=v_num, faces_ict=faces)
                gt_recon_ICT = self.ict_face_model.apply_coeffs_batch_torch(inputs_ict[2][:, :100], pred_exp_coeff[:, :53].detach(), region=region_num)
                pred_outputs_ict, _ = self.decode_grad(inputs_ict)
                loss["vert_vICT"] = self.criterion(gt_recon_ICT, pred_outputs_ict)

        else:
            loss["vert_rIEnc"] = self.non_ict_loss(pred_id_coeff)
            loss["vert_rEEnc"] = self.non_ict_loss(pred_exp_coeff)
            
        if return_all:
            return loss, pred_outputs, None, pred_exp_coeff, pred_id_coeff, pred_seg_coeff
        return loss
    
    def stage8_forward(self, batch, teacher_forcing=True, return_all=False, epoch=0):
        """
        Args
        ----------
            batch (tuple):
                id_coeff (torch.Tensor):      [B, 128]    ICT-face model id coeff
                gt_rig_params (torch.Tensor): [B, 128] ICT-face model exp_coeff 
                template (torch.Tensor):      [B, V, 3] mesh vertices
                dfn_info (list(torch.Tensor)): DiffusionNet precomputes (mass, L, eval, evecs, gradX, gradY, faces)
                operators (list(torch.Tensor)): mesh operators for Poisson solve (NFR)
                vertices (torch.Tensor):      [B, V, 3] sequence of mesh vertices (animation)
                faces (torch.Tensor):         [F, 3] mesh faces indices
            teacher_forcing (bool): used for Transformer model (not used !!!)
            return_all (bool): if True, return loss and all predicted outputs
            
        Returns
        ----------
            pred_outputs (torch.Tensor): [B, W, V, 3]

        TODO
        - [ ] visualize correspondence using feature similarity
        """
        
        #_, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img, mesh_data = batch
        mesh_data = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data] 
        ## Send to device --------------------------------------------------------------
        gt_id_coeff = batch.id_coeff
        gt_rig_params = batch.gt_rig_params[:, :self.opts.rig_dim]
        gt_normals = batch.normals
        gt_vertices = batch.vertices.to(self.device)
        template = batch.template
        faces = batch.faces
        dfn_info = pickle.load(open(batch.dfn_info, 'rb'))
        B = gt_vertices.shape[0]

        operators = pickle.load(open(batch.operators, mode='rb'))
        ##------------------------------------------------------------------------------
        
        ## Encoding --------------------------------------------------------------------
         
        # target neutral face
        vert_feat = self.get_local_feature_no_img(gt_vertices-template, template, faces, dfn_info) # [1, V, 9]
        # local_feat = self.compute_wks_autoscale(batch_evals, batch_evecs, batch_mass)
        
        # returns tuple
        pred_vtx_corr_feat, pred_id_coeff, pred_exp_coeff = self.encode_(vert_feat, dfn_info) # [B, V, 256], [B, ID], [B, EXP]
        ##------------------------------------------------------------------------------
        
        
        ## Loss function ---------------------------------------------------------------
        losses = {}
        if self.design == 'new3':
            ## ID loss
            vert_feat_zero = self.get_local_feature_no_img(torch.zeros_like(template), template, faces, dfn_info) # [1, V, 9]
            pred_vtx_corr_feat_zero, pred_id_coeff_zero, _ = self.encode_(vert_feat_zero, dfn_info) # [B, V, 256], [B, ID], [B, EXP]
            
            losses["d_ID"] = self.criterion(gt_id_coeff, pred_id_coeff_zero)
            losses["d_feat"] = self.criterion(pred_vtx_corr_feat, pred_vtx_corr_feat_zero)
            
            # losses = self.unsupervised_shape_matching_loss(
            #     pred_vtx_corr_feat, dfn_info, pred_vtx_corr_feat_zero, dfn_info, losses=losses)
            
            losses = self.unsupervised_shape_matching_loss(
                pred_vtx_corr_feat, dfn_info, losses=losses)

        if mesh_data == 'ict':
            losses["vert_rIEnc"] = self.criterion(gt_id_coeff, pred_id_coeff)
            losses["vert_rEEnc"] = self.criterion(gt_rig_params, pred_exp_coeff)
                
            region_num = self.ict_face_model.get_region_num(gt_vertices)
            if not self.opts.no_BP:
                pred_ICT_recon = self.ict_face_model.apply_coeffs_batch_torch(pred_id_coeff[:, :100], pred_exp_coeff[:, :53], region=region_num)                
                # losses["vert_vICT"] = self.criterion(gt_vertices, pred_ICT_recon)
                losses["recon_vICT"] = self.criterion(gt_vertices, pred_ICT_recon)

        else: # (not used in this train loop)
            losses["vert_rIEnc"] = self.non_ict_loss(pred_id_coeff[:, :100]) + self.L2_regularization(pred_id_coeff[:, 100:])
            losses["vert_rEEnc"] = self.non_ict_loss(pred_exp_coeff[:, :53]) + self.L2_regularization(pred_exp_coeff[:, 53:])
            

        if return_all:
            return losses, pred_ICT_recon, None, pred_exp_coeff, pred_id_coeff, None
        return losses

    def stage9_forward(self, batch, teacher_forcing=True, return_all=False, epoch=0):
        """
        Args
        ----------
            batch (tuple):
                audio_feat (torch.Tensor):    [B, W, 768] (not used !!!)
                id_coeff (torch.Tensor):      [B, 128]    ICT-face model id coeff
                gt_rig_params (torch.Tensor): [B, 128] ICT-face model exp_coeff 
                template (torch.Tensor):      [B, V, 3] mesh vertices
                dfn_info (list(torch.Tensor)): DiffusionNet precomputes (mass, L, eval, evecs, gradX, gradY, faces)
                operators (list(torch.Tensor)): mesh operators for Poisson solve (NFR)
                vertices (torch.Tensor):      [B, V, 3] sequence of mesh vertices (animation)
                faces (torch.Tensor):         [F, 3] mesh faces indices
                img (torch.Tensor):           [B, H, W, C] rendered mesh image
            teacher_forcing (bool): used for Transformer model (not used !!!)
            return_all (bool): if True, return loss and all predicted outputs
            
        Returns
        ----------
            pred_outputs (torch.Tensor): [B, W, V, 3]
        """
        
        #_, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img, mesh_data = batch
        mesh_data = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data] 
        ## Send to device --------------------------------------------------------------
        gt_id_coeff = batch.id_coeff
        gt_rig_params = batch.gt_rig_params[:, :self.opts.rig_dim]
        gt_normals = batch.normals
        gt_vertices = batch.vertices.to(self.device)
        template = batch.template
        faces = batch.faces
        dfn_info = pickle.load(open(batch.dfn_info, 'rb'))
        B = gt_vertices.shape[0]        
        operators = pickle.load(open(batch.operators, mode='rb'))
        ##------------------------------------------------------------------------------
        
        ## Encoding --------------------------------------------------------------------
        # mesh_feat = extract_mesh_feature(self, dfn_info, verbose=False)
        
        # target neutral face
        vert_feat = self.get_local_feature_no_img(gt_vertices-template, template, faces, dfn_info) # [1, V, 9]
        # local_feat = self.compute_wks_autoscale(batch_evals, batch_evecs, batch_mass)
        
        # returns tuple
        pred_vtx_corr_feat, pred_id_coeff, pred_exp_coeff = self.encode_(vert_feat, dfn_info) # [B, V, 256], [B, ID], [B, EXP]
        ##------------------------------------------------------------------------------
        
        
        ## Decoding --------------------------------------------------------------------
        self.mesh_decoder.update_precomputes(dfn_info)
        inputs = (pred_vtx_corr_feat, pred_exp_coeff, pred_id_coeff, None, None, template, faces, operators)
        pred_outputs, pred_jacobians = self.decode_grad(inputs)
        ##------------------------------------------------------------------------------
        
        ## Loss function ---------------------------------------------------------------
        losses = {}
        if self.design == 'new3':
            ## ID loss
            vert_feat_zero = self.get_local_feature_no_img(torch.zeros_like(template), template, faces, dfn_info) # [1, V, 9]
            pred_vtx_corr_feat_zero, pred_id_coeff_zero, _ = self.encode_(vert_feat_zero, dfn_info) # [B, V, 256], [B, ID], [B, EXP]
            # losses["d_ID"] = self.criterion(gt_id_coeff, pred_id_coeff_zero)
            # losses["d_feat"] = self.criterion(pred_vtx_corr_feat, pred_vtx_corr_feat_zero)
            
            # losses = self.unsupervised_shape_matching_loss(template, faces, dfn_info)
            if torch.rand(1) < 0.5:
            # if False:
                losses = self.unsupervised_shape_matching_loss(
                    pred_vtx_corr_feat, dfn_info, pred_vtx_corr_feat_zero, dfn_info, losses=losses)
            else:
                losses = self.unsupervised_shape_matching_loss(
                    pred_vtx_corr_feat, dfn_info, losses=losses)

        
        losses["recon_vDec"] = self.criterion(gt_vertices, pred_outputs)
        
        pred_normals = self.calc_norm_torch(pred_outputs, faces, at='verts')
        losses["norm_vDec"] = self.criterion(gt_normals, pred_normals)
    
        with torch.no_grad():
            gt_jacobians = self.get_jacobian_matrix(gt_vertices, faces, template, return_torch=True)
        losses["jacob_vDec"] = self.criterion(gt_jacobians, pred_jacobians)

        if mesh_data == 'ict':
            losses["vert_rIEnc"] = self.criterion(gt_id_coeff, pred_id_coeff)
            losses["vert_rEEnc"] = self.criterion(gt_rig_params, pred_exp_coeff)
            
            region_num = self.ict_face_model.get_region_num(gt_vertices)
            v_num, faces = self.ict_face_model.get_random_v_and_f(region_num, mode='torch')
            ict_seg = self.ict_vert_segment[:v_num]
                
            if not self.opts.no_BP:
                pred_ICT_recon = self.ict_face_model.apply_coeffs_batch_torch(pred_id_coeff[:, :100], pred_exp_coeff[:, :53], region=region_num)                
                losses["vert_vICT"] = self.criterion(gt_vertices, pred_ICT_recon)
        
            if not self.opts.no_BR:
                inputs_ict = self.get_inputs_ict_no_img(pred_exp_coeff.detach(), region=region_num, v_num=v_num, faces_ict=faces)
                gt_recon_ICT = self.ict_face_model.apply_coeffs_batch_torch(inputs_ict[2][:, :100], pred_exp_coeff[:, :53].detach(), region=region_num)
                pred_outputs_ict, _ = self.decode_grad(inputs_ict)
                losses["vert_vICT"] = self.criterion(gt_recon_ICT, pred_outputs_ict)

            if self.design == 'new2':
                pred_seg_log = F.log_softmax(pred_seg_coeff, dim=-1) ## [V, Seg]
                losses["nll_vSeg"] = 0
                for pred_s_l in pred_seg_log:
                    losses["nll_vSeg"] += F.nll_loss(pred_s_l, ict_seg)
                losses["nll_vSeg"] = losses["nll_vSeg"] / pred_seg_log.shape[0]
        else:
            # if not self.opts.no_BR:
            #     inputs_ict = self.get_inputs_ict(pred_exp_coeff.detach(), region=region_num, v_num=v_num, faces_ict=faces)
            #     gt_recon_ICT = self.ict_face_model.apply_coeffs_batch_torch(inputs_ict[2][:, :100], pred_exp_coeff[:, :53].detach(), region=region_num)
            #     pred_outputs_ict, _ = self.decode_grad(inputs_ict)
            #     losses["vert_vICT"] = self.criterion(gt_recon_ICT, pred_outputs_ict)
            losses["vert_rIEnc"] = self.non_ict_loss(pred_id_coeff[:, :100]) + self.L2_regularization(pred_id_coeff[:, 100:])
            losses["vert_rEEnc"] = self.non_ict_loss(pred_exp_coeff[:, :53]) + self.L2_regularization(pred_exp_coeff[:, 53:])
            

        if return_all:
            return losses, pred_outputs, None, pred_exp_coeff, pred_id_coeff, None
        return losses
    
    def stage21_forward(self, batch, teacher_forcing=True, return_all=False, epoch=0):
        """
        Args
        ----------
            batch (tuple):
                audio_feat (torch.Tensor):    [B, W, 768] (not used !!!)
                id_coeff (torch.Tensor):      [B, 128]    ICT-face model id coeff
                gt_rig_params (torch.Tensor): [B, 128] ICT-face model exp_coeff 
                template (torch.Tensor):      [B, V, 3] mesh vertices
                dfn_info (list(torch.Tensor)): DiffusionNet precomputes (mass, L, eval, evecs, gradX, gradY, faces)
                operators (list(torch.Tensor)): mesh operators for Poisson solve (NFR)
                vertices (torch.Tensor):      [B, V, 3] sequence of mesh vertices (animation)
                faces (torch.Tensor):         [F, 3] mesh faces indices
                img (torch.Tensor):           [B, H, W, C] rendered mesh image
            teacher_forcing (bool): used for Transformer model (not used !!!)
            return_all (bool): if True, return loss and all predicted outputs
            
        Returns
        ----------
            pred_outputs (torch.Tensor): [B, W, V, 3]
        """
        
        #_, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img, mesh_data = batch
        mesh_data = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data] 
        ## Send to device --------------------------------------------------------------
        gt_id_coeff = batch.id_coeff
        gt_rig_params = batch.gt_rig_params[:, :self.opts.rig_dim]
        gt_normals = batch.normals
        gt_vertices = batch.vertices.to(self.device)
        template = batch.template
        faces = batch.faces
        dfn_info = pickle.load(open(batch.dfn_info, 'rb'))
        B, V, _ = gt_vertices.shape#[0]

        ## dummy ########################################################################
        corr_feat = batch.corr_feat
        # import pdb;pdb.set_trace()
        # corr_feat = torch.rand(B, V, 2048).to(self.device)
        ################################################################################# 

        # batch_mass, batch_L, batch_evals, batch_evecs, batch_grad_X, batch_grad_Y = [],[],[],[],[],[]
        # for dfn_info_path in batch.dfn_info:
        #     dfn_info = torch.load(dfn_info_path)
        #     dfn_info = [_.to(self.device).float() if type(_) is not torch.Size else _  for _ in dfn_info]            
        #     batch_mass.append(dfn_info[0])
        #     batch_L.append(dfn_info[1])
        #     batch_evals.append(dfn_info[2])
        #     batch_evecs.append(dfn_info[3])
        #     batch_grad_X.append(dfn_info[4])
        #     batch_grad_Y.append(dfn_info[5])
        # batch_mass=torch.stack(batch_mass).to(self.device)
        # batch_L=torch.stack(batch_L).to(self.device)
        # batch_evals=torch.stack(batch_evals).to(self.device)
        # batch_evecs=torch.stack(batch_evecs).to(self.device)
        
        operators = pickle.load(open(batch.operators, mode='rb'))
        ##------------------------------------------------------------------------------
        
        ## Encoding --------------------------------------------------------------------
        # source expression face
        vert_feat_exp = self.get_local_feature_no_img(torch.zeros_like(template), gt_vertices, faces, dfn_info)  # [B, V, 6]

        # target neutral face
        vert_feat = self.get_local_feature_no_img(torch.zeros_like(template), template, faces, dfn_info) # [1, V, 6]
        
        pred_id_coeff, pred_exp_coeff, _ = self.encode_mesh_grad(vert_feat, vert_feat_exp, dfn_info)
        ##------------------------------------------------------------------------------
        
        

        ## Decoding --------------------------------------------------------------------
        self.mesh_decoder.update_precomputes(dfn_info)

        # target inputs
        if epoch < 100 and mesh_data == 'ict' and self.opts.warmup:
            inputs = (corr_feat, gt_rig_params, pred_id_coeff, None, None, template, faces, operators)
        else:
            inputs = (corr_feat, pred_exp_coeff, pred_id_coeff, None, None, template, faces, operators)
        pred_outputs, pred_jacobians = self.decode_grad(inputs)
        ##------------------------------------------------------------------------------
        
        ## Loss function ---------------------------------------------------------------
        losses = {}
        
        # loss for canonical bases
        # if self.opts.use_canon:
        #     losses = self.mesh_decoder.canonical_bases.get_loss(losses)

        losses["recon_vDec"] = self.criterion(gt_vertices, pred_outputs)
        
        pred_normals = self.calc_norm_torch(pred_outputs, faces, at='verts')
        losses["norm_vDec"] = self.criterion(gt_normals, pred_normals)
    
        with torch.no_grad():
            gt_jacobians = self.get_jacobian_matrix(gt_vertices, faces, template, return_torch=True)
        losses["jacob_vDec"] = self.criterion(gt_jacobians, pred_jacobians)

        if mesh_data == 'ict':
            losses["vert_rIEnc"] = self.criterion(gt_id_coeff, pred_id_coeff)
            losses["vert_rEEnc"] = self.criterion(gt_rig_params, pred_exp_coeff)
            
            region_num = self.ict_face_model.get_region_num(gt_vertices)
            v_num, faces = self.ict_face_model.get_random_v_and_f(region_num, mode='torch')
            ict_seg = self.ict_vert_segment[:v_num]
                
            if not self.opts.no_BP:
                pred_ICT_recon = self.ict_face_model.apply_coeffs_batch_torch(pred_id_coeff[:, :100], pred_exp_coeff[:, :53], region=region_num)                
                losses["vert_vICT"] = self.criterion(gt_vertices, pred_ICT_recon)
                #losses["recon_vICT"] = self.criterion(gt_vertices, pred_ICT_recon)
        
            if not self.opts.no_BR:
                inputs_ict = self.get_inputs_ict_no_img(pred_exp_coeff.detach(), region=region_num, v_num=v_num, faces_ict=faces)
                gt_recon_ICT = self.ict_face_model.apply_coeffs_batch_torch(inputs_ict[2][:, :100], pred_exp_coeff[:, :53].detach(), region=region_num)
                pred_outputs_ict, _ = self.decode_grad(inputs)
                losses["vert_vICT"] = self.criterion(gt_recon_ICT, pred_outputs_ict)
                #losses["recon_vICT"] = self.criterion(gt_recon_ICT, pred_outputs_ict)

        else:
            # if not self.opts.no_BR:
            #     inputs_ict = self.get_inputs_ict(pred_exp_coeff.detach(), region=region_num, v_num=v_num, faces_ict=faces)
            #     gt_recon_ICT = self.ict_face_model.apply_coeffs_batch_torch(inputs_ict[2][:, :100], pred_exp_coeff[:, :53].detach(), region=region_num)
            #     pred_outputs_ict, _ = self.decode_grad(inputs_ict)
            #     losses["vert_vICT"] = self.criterion(gt_recon_ICT, pred_outputs_ict)
            losses["vert_rIEnc"] = self.non_ict_loss(pred_id_coeff[:, :100]) + self.L2_regularization(pred_id_coeff[:, 100:])
            losses["vert_rEEnc"] = self.non_ict_loss(pred_exp_coeff[:, :53]) + self.L2_regularization(pred_exp_coeff[:, 53:])
            
        if return_all:
            return losses, pred_outputs, None, pred_exp_coeff, pred_id_coeff, None
        return losses
    
    # def unsupervised_shape_matching_loss(self, template, faces, dfn_info):
    def unsupervised_shape_matching_loss(self, 
        feat_x, 
        dfn_info_x,
        feat_y=None, 
        dfn_info_y=None,
        losses={},
        ):
        """
        Unsupervised Learning of Robust Spectral Shape Matching
        
        """
        # vert_feat_x = self.get_local_feature_no_img(
        #     torch.zeros_like(template[0:1]), template[0:1], faces, dfn_info
        # ) # [1, V, 9]
        # feat_x, _, _ = self.encode_(vert_feat_x, dfn_info)
        
        B = feat_x.shape[0]

        if feat_y is None:
            dfn_info_y = self.neu_dfn_info
            vert_feat_y = self.get_local_feature_no_img(
                torch.zeros_like(self.ict_neutral)[None], self.ict_neutral[None], self.ict_faces, self.neu_dfn_info
            ) # [1, V, 9]
            feat_y, _, _ = self.encode_(vert_feat_y, dfn_info_y)
            feat_y = feat_y.repeat(B,1,1)
        
        # HB = B//2
        evecs_x = dfn_info_x[3][None].repeat(B,1,1) # [1, V, 128]
        evals_x = dfn_info_x[2][None].repeat(B,1) # [1, 128]
        evecs_trans_x = dfn_info_x[3].T * dfn_info_x[0][None] # [128, V]
        evecs_trans_x = evecs_trans_x[None].repeat(B,1,1) #[1, 128, V]
        
        evecs_y = dfn_info_y[3][None].repeat(B,1,1) # [1, V, 128]
        evals_y = dfn_info_y[2][None].repeat(B,1) # [1, 128]
        evecs_trans_y = dfn_info_y[3].T * dfn_info_y[0][None] # [128, V]
        evecs_trans_y = evecs_trans_y[None].repeat(B,1,1) #[1, 128, V]
        
        
        Cxy, Cyx = self.fmap_net(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
        
        # dict 
        # losses = self.surfmnet_loss(Cxy, Cyx, evals_x, evals_x)
        l_bij, l_orth, l_lap = self.surfmnet_loss(Cxy, Cyx, evals_x, evals_x)
        
        Pxy, Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)
        
        Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))
        Cyx_est = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y))
        # losses['l_align'] = self.align_loss(Cxy, Cxy_est) + self.align_loss(Cyx, Cyx_est)
        l_align = self.align_loss(Cxy, Cxy_est) + self.align_loss(Cyx, Cyx_est)
        
        # for key in losses.keys():
        for key, l_value in zip(['l_bij','l_orth','l_lap','l_align'],[l_bij,l_orth,l_lap,l_align]):
            if key in losses.keys():
                losses[key]+=l_value
            else:
                losses[key]=l_value

        # losses['l_orth']=l_orth
        # losses['l_lap']=l_lap
        # losses['l_align']=l_align
        return losses
    
    @torch.no_grad()
    def evaluate(self, batch, batch_process=True, return_all=False, stage=1, epoch=0, newid=None, mode=''):

        ### train with ICT only + train mesh autoencoder only (learn rig space)
        if mode == 'invrig':
            return self.stage1_invrig(batch, newid, batch_process, return_all, epoch)
        else:
            return self.stage1_evaluate(batch, batch_process, return_all, epoch)
    
    def stage1_evaluate(self, batch, batch_process=False, return_all=False, epoch=0):
        """
        ## Note: B (batch_size) is always 1
        Args:
            batch: (tuple)
                audio_feat:    [B, W, 768] audio feature from wav2vec 2.0 -> not used here
                id_coeff:      [1, 128]    ICT-face model id coeff
                gt_rig_params: [B, W, 128] ICT-face model exp_coeff 
                template:      [B, V, 3] mesh vertices
                dfn_info (list): DiffusionNet information
                operators (list): mesh operators
                vertices:      [B, W, V, 3] sequence of mesh vertices (animation)
                faces:         [B, F, 3] mesh faces indices
                img:           [B, 256, 256, 3] rendered mesh image
            batch_process: (bool) if True, process in batch
            return_all: (bool) if True, return loss and all predicted outputs
        Return:
            pred_outputs: (torch.tensor) [B, W, V, 3]
        """
        mesh_data = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data.cpu().numpy()]
        
        ## Send to device --------------------------------------------------------------
        # gt_id_coeff = batch.id_coeff.to(self.device)
        # gt_rig_params = batch.gt_rig_params.to(self.device)
        template = batch.template.to(self.device)
        gt_vertices = batch.vertices.to(self.device)
        faces = batch.faces.to(self.device)
        img = batch.img.to(self.device)
        #gt_normals = batch.normals.squeeze()
        operators = pickle.load(open(batch.operators, mode='rb'))
        dfn_info = pickle.load(open(batch.get_dfn_info, mode='rb'))
        ##------------------------------------------------------------------------------
        
        
        ## Encoding --------------------------------------------------------------------
        ## source & target mesh local feature
        img_feat = self.get_img_feat(img).unsqueeze(1) #--------------------------------------- [1, 1, 128]
        vert_feat = self.get_local_feature(template, faces, img_feat)#------------------ [1, V, 3+3+128]
        vert_feat_exp = self.get_local_feature(gt_vertices, faces, img_feat)#----------- [W, V, 3+3+128]
        
        with torch.no_grad():
            pred_id_coeff = self.encode_id(vert_feat, dfn_info)#---------------- [1, ID]
            if 'new2' in self.design:
                pred_seg_coeff = self.encode_seg(vert_feat, dfn_info)# [1, V, Seg]
            else:
                pred_seg_coeff = None
        torch.cuda.empty_cache()
        ##------------------------------------------------------------------------------
        
        
        ##------------------------------------------------------------------------------
        if self.scale_exp != 1:
            pred_exp_coeff = pred_exp_coeff * self.scale_exp
        ##------------------------------------------------------------------------------
        
        ## Decoding --------------------------------------------------------------------
        with torch.no_grad():
            if self.opts.dec_type=='jacob':
                faces_feat = self.get_local_feature(template, faces, img_feat, at='faces')
                local_feat = faces_feat #----------------------------------------------- [1, V, 3+3+128]
            else:
                local_feat = vert_feat #------------------------------------------------ [1, V, 3+3+128]
        
            #operators = pickle.load(open(operators[0], mode='rb'))
            style_emb = None
            inputs = (local_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff, style_emb, template, faces, operators)
            pred_outputs, _ = self.decode(inputs, batch_process=False)
        torch.cuda.empty_cache()
        ##------------------------------------------------------------------------------
        
        loss = {}
        if self.opts.dec_type=='jacob':
            gt_vertices_zm = gt_vertices - gt_vertices[:, 0:1].mean(dim=1, keepdim=True) # because no translation included
            loss["recon_vDec"] = self.criterion(gt_vertices_zm, pred_outputs)
        else:
            loss["recon_vDec"] = self.criterion(gt_vertices, pred_outputs)
            
        if return_all:
            if self.design == 'nfr': 
                return loss, pred_outputs, None, pred_exp_coeff, pred_id_coeff, None
            else:
                return loss, pred_outputs, None, pred_exp_coeff, pred_id_coeff, pred_seg_coeff
        return loss
    
    def stage1_invrig(self, batch, newid, batch_process=False, return_all=False, epoch=0):
        """
        ## Note: B (batch_size) is always 1
        Args:
            batch (tuple):
                audio_feat:    [B, W, 768] audio feature from wav2vec 2.0 -> not used here
                id_coeff:      [1, 128]    ICT-face model id coeff
                gt_rig_params: [B, W, 128] ICT-face model exp_coeff 
                template:      [B, V, 3] mesh vertices
                dfn_info (list): DiffusionNet information
                operators (list): mesh operators
                vertices:      [B, W, V, 3] sequence of mesh vertices (animation)
                faces:         [B, F, 3] mesh faces indices
                img:           [B, 256, 256, 3] rendered mesh image
            newid (tuple):
                tgt_coeff:     [1, 128]    ICT-face model id coeff
                tgt_idx: (int)
            batch_process: (bool) if True, process in batch
            return_all: (bool) if True, return loss and all predicted outputs
        Return:
            pred_outputs: (torch.tensor) [B, W, V, 3]
        """
        (audio_feat, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img), mesh_data = batch
        B, W, _ = audio_feat.shape
        V = template.shape[1]
        
        ## Send to device --------------------------------------------------------------
        #gt_id_coeff = id_coeff.to(self.device).float()
        gt_rig_params = gt_rig_params.to(self.device).float().squeeze(0)
        template = template.to(self.device).float()
        gt_vertices = vertices.to(self.device).float().squeeze(0)
        faces = faces.to(self.device).squeeze(0)
        img = img.to(self.device).float()
        
        
        # get New target mesh (neutral face and animated)
        (tgt_coeff, tgt_idx) = newid
        #p_mode = 'face_only' if self.ict_face_only else 'fullhead'
        tgt_img = np.load(os.path.join(self.ict_precompute, f"{tgt_idx:03d}_img.npy")) # ----- [1, 256, 256, 3]
        tgt_img = torch.from_numpy(tgt_img).to(self.device).float()
        tgt_coeff = torch.from_numpy(tgt_coeff).to(self.device).float()
        tgt_gt_rig = gt_rig_params[:,:53].to(self.device).float()
        tgt_dfn_info  = pickle.load(open(os.path.join(self.ict_precompute, f"{tgt_idx:03d}_dfn_info.pkl"), 'rb')) # list[ ... ]
        tgt_operators = os.path.join(self.ict_precompute, f"{tgt_idx:03d}_operators.pkl") # list[ ... ]
        tgt_operators = pickle.load(open(tgt_operators, mode='rb'))
        
        # displacements
        tgt_id_disps = torch.einsum('k,kls->ls', tgt_coeff, self.ict_face_model.id_basis)[:self.ict_face_model.v_num]
        tgt_exp_disp = torch.einsum('jk,kls->jls', tgt_gt_rig, self.ict_face_model.exp_basis)[:,:self.ict_face_model.v_num]
        
        tgt_template = self.ict_neutral.to(self.device) + tgt_id_disps # neutral face
        tgt_gt_vertices = tgt_template + tgt_exp_disp # facial expressions
        
        # send to device
        tgt_template = tgt_template[None].to(self.device).float()
        tgt_gt_vertices = tgt_gt_vertices.to(self.device).float()
        ##------------------------------------------------------------------------------
        
        
        
        
        ## Encoding --------------------------------------------------------------------
        ## target mesh feature
        tgt_img_feat = self.get_img_feat(tgt_img).unsqueeze(1) #------------------------------- [1, 1, 128]
        vert_feat = self.get_local_feature(tgt_template, faces, tgt_img_feat)#---------- [1, V, 3+3+128]
        
        ## source mesh feature
        img_feat = self.get_img_feat(img).unsqueeze(1) #--------------------------------------- [1, 1, 128]
        vert_feat_exp = self.get_local_feature(gt_vertices, faces, img_feat)#----------- [W, V, 3+3+128]

        with torch.no_grad():
            # target id_code
            pred_id_coeff = self.encode_id(vert_feat, tgt_dfn_info)#-------------------- [1, ID]

            # target seg_code
            if 'new2' in self.design:
                pred_seg_coeff = self.encode_seg(vert_feat, tgt_dfn_info)# [1, V, Seg]

            # source exp_code
            pred_exp_coeff = self.encode_exp(vert_feat_exp, dfn_info, batch_process=batch_process)# [W, Rig]
        torch.cuda.empty_cache()
        
                
        ## Decoding --------------------------------------------------------------------
        with torch.no_grad():
            if self.opts.dec_type=='jacob':
                faces_feat = self.get_local_feature(tgt_template, faces, tgt_img_feat, at='faces')
                local_feat = faces_feat #--------------------------------------------------- [1, V, 3+3+128]
            else:
                local_feat = vert_feat #---------------------------------------------------- [1, V, 3+3+128]
            
                operators = pickle.load(open(operators[0], mode='rb'))
                style_emb = None
                inputs = (local_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff, style_emb, tgt_template, faces, tgt_operators)
                pred_outputs, pred_jacobians = self.decode(inputs)
        torch.cuda.empty_cache()
        ##------------------------------------------------------------------------------
            
        loss = {}
        if self.opts.dec_type=='jacob':
            loss["recon_vDec"] = self.criterion(tgt_gt_vertices, pred_outputs)
        else:
            loss["recon_vDec"] = self.criterion(tgt_gt_vertices, pred_outputs)
            
        if return_all:
            if self.design == 'nfr': 
                return loss, pred_outputs, None, pred_exp_coeff, pred_id_coeff, None
            else:
                return loss, pred_outputs, None, pred_exp_coeff, pred_id_coeff, pred_seg_coeff
        return loss
    
    def get_mesh_skinning_weight(self, pred_seg, no_grad=False):
        if no_grad:
            with torch.no_grad():
                return self.mesh_decoder.skinning_layer(pred_seg) # [B, V, exp]
        else:
            return self.mesh_decoder.skinning_layer(pred_seg) # [B, V, exp]
        
    @torch.no_grad()
    def inference(self, 
                     gt_vertices, 
                     src_mesh, 
                     tgt_mesh, 
                     batch_process=False, 
                     return_all=False,
                     return_exp=False,
                     return_seg=False, 
                     use_filter=False,
                     use_scale=False,
                     scale_mask=torch.ones(53)*2.0,
                     kernel_size=3,
                     sigma=1.0
                    ):
        """
        ## Note: batch_size is always 1
        Args:
            gt_vertices (torch.tensor): mesh animation vertices
            src_mesh (trimesh): source triangle mesh
            tgt_mesh (trimesh): target triangle mesh
        Return:
            pred_outputs: [batch_size, W, V, 3]
        """
        ## get data and send to device
        if len(gt_vertices.shape) < 3:
            gt_vertices = gt_vertices.reshape(gt_vertices.shape[0], -1, 3)
            
        pred_exp_coeff, pred_id_coeff, pred_seg_coeff, vert_feat = self.encode_mesh(
            gt_vertices,
            src_mesh, 
            tgt_mesh, 
            batch_process=batch_process
        )
        ## empty_cache
        torch.cuda.empty_cache()
        ##------------------------------------------------------------------------------
        
        
        ##------------------------------------------------------------------------------
        if self.scale_exp != 1.0:
            pred_exp_coeff = pred_exp_coeff * self.scale_exp
                    
        if use_scale:
            pred_exp_coeff[:,:53] = pred_exp_coeff[:,:53] * scale_mask[None].to(device=pred_exp_coeff.device)
            
        if use_filter:
            pred_exp_coeff = self.apply_gaussian_filter(
                pred_exp_coeff.squeeze(1), 
                kernel_size=kernel_size, 
                sigma=sigma
            )
        ##------------------------------------------------------------------------------
        
        if self.design == 'nfr':
            pred_exp_coeff = pred_exp_coeff.squeeze()
        
        ## Decode ----------------------------------------------------------------------
        tgt_verts = torch.from_numpy(tgt_mesh.vertices)[None].to(self.device)
        tgt_faces = torch.from_numpy(tgt_mesh.faces).to(self.device)
        tgt_operators = None
        
        if self.opts.dec_type=='jacob':
            tgt_operators = self.get_mesh_operators(tgt_mesh)
            tgt_img = self.renderer.render_img(tgt_mesh).float().to(self.device)
            tgt_img_feat = self.get_img_feat(tgt_img).unsqueeze(1) #------------------- [1, 1, 128]
            faces_feat = self.get_local_feature(tgt_verts, tgt_faces, tgt_img_feat, at='faces')
            local_feat = faces_feat # [1, F, 3+3+128]
        else:
            local_feat = vert_feat # [1, V, 3+3+128]

        style_emb = None # depricated 
        inputs = (local_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff, style_emb, tgt_verts, tgt_faces, tgt_operators)
            
        with torch.no_grad():
            pred_outputs, pred_jacobians = self.decode(
                inputs,
                batch_process=batch_process,
            )
            
        ## empty_cache
        torch.cuda.empty_cache()
        ##------------------------------------------------------------------------------
        
        if return_all:
            return pred_outputs, (vert_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff, style_emb, tgt_verts, tgt_faces, tgt_operators)
        if return_exp:
            return pred_outputs, pred_exp_coeff
        if return_seg:
            return pred_outputs, pred_seg_coeff
        return pred_outputs
