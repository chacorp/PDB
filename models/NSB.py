import sys

from pathlib import Path
__abs_path__ = str(Path(__file__).parents[1].absolute())
__mesh_util_path__ = f'{__abs_path__}/mesh_utils'

for __util_path__ in [__abs_path__, __mesh_util_path__]:
    if not __util_path__ in sys.path:
        sys.path+=[__util_path__]

import torch
import torch.nn as nn
from utils.exp_utils import Model, Model_mk2_1
from utils.exp_utils import plateau_hat_points


class ControlModeKeyD(nn.Module):
    """
        K-conditioned FiLM coefficient predictor + fixed linear synthesis.

        Learnable, shared control-point modes `K` combined with an expression
        latent `z` through a K-conditioned FiLM coefficient predictor to infer
        signed mode coefficients `A`. `delta_C` is a fixed linear combination
        of `K` weighted by `A` -- see docs/control_mode_blending_plan.md.

        This is the drop-in replacement for NGBC.key_d_model (out_type=0 /
        delta form only): forward(z) returns a flat [bs, 1, 3*N_C] tensor,
        matching what NGBC.reshape_key_d expects.

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

        from models.encoder import MLP, LinearEncoder

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


class NeuralSparseBlendpoint(nn.Module):
    """
        Full retargeting model, structured identically to
        models.NGBC.NeuralGeneralizedBarycentricCoordinate.

        `key_weight_model` and `exp_z_model` are kept the same as NGBC.
        `key_d_model` is replaced with `ControlModeKeyD`: a shared learnable
        set of control-point modes `K`, conditioned on the expression latent
        `z` via FiLM, predicting signed mode coefficients `A` that are
        linearly combined with `K` to produce `delta_C`.

        First supported combination only: out_type=0 (delta form), use_shp=False.
        Other combinations raise NotImplementedError explicitly.
    """
    def __init__(self,
                 opts=None,
                 in_dim=3,
                 out_dim=3,
                 hid_dim=256,
                 num_cage_vertices=512,
                 num_layers=4,
                 N_list=[3525, 2560, 5223], # voca biwi mf
                 use_softmax=False,
                 use_relu=True,
                 use_elu=False,
                 use_softplus=False,
                 use_least_N=False,
                 use_least_N_on_V=False,
                 least_number_of_zeros=256, # for sparsity (not used)
                 is_train=False,
                 tau=0.05,
                 device='cpu',
                 use_exp_recon=False, # was not necessary
                 use_shp_recon=False, # necessary for training, but not needed for inference
                 use_shp=False,
                 use_pou=True,
                 use_full_vertex=False, # default: false (= delta form)
                 no_activation=False,
                 num_modes=128,
                 mode_num_layers=2,
                 mode_init_std=0.01,
                ):
        super().__init__()
        self.opts = opts

        self.is_train = is_train

        self.num_layers = num_layers
        self.num_cage_vertices = num_cage_vertices
        self.NZ = least_number_of_zeros
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_shp_recon = use_shp_recon
        self.use_shp = use_shp
        if not self.use_shp:
            self.use_shp_recon = False
            print('shape model not used!, use_shp_recon set to False')
        self.use_exp_recon = use_exp_recon
        self.use_full_vertex = use_full_vertex
        self.no_activation = no_activation
        self.use_pou = use_pou

        ###### NN input type settings
        ## key_weight model | key_d_model
        # 0: (src_p),        (def_p, src_p)
        # 1: (src_p, src_n), (def_p, def_n, src_p, src_n)

        self.in_type = 1
        self.out_type = 1 # vertex
        ######

        if self.opts is not None:
            self.in_type = self.opts.in_type
            self.out_type = self.opts.out_type

        if self.in_type == 0:
            self.in_dim = 3
            in_dim_exp = self.in_dim*2
        elif self.in_type == 1:
            self.in_dim = 6
            in_dim_exp = self.in_dim*2
        elif self.in_type == 2:
            self.in_dim = 6+1
            in_dim_exp = 6+6+1
        else:
            raise NotImplementedError('in_type not implemented')

        if self.out_type == 0: # delta form
            self.use_full_vertex = False
            self.out_dim = 3
        elif self.out_type == 1: # vertex (linear precision)
            self.use_full_vertex = True
            self.out_dim = 3
        elif self.out_type == 2: # transform matrix
            self.use_full_vertex = False
            self.out_dim = 9 # (6D + translation 3) will be reshaped into 3x4 matrix
            M_ = num_cage_vertices
            num_cage_vertices = num_cage_vertices * 4

            from utils.exp_utils import from_6D_to_rotation_matrix_torch as _6D_to_rot_
            self._6D_to_rot_ = _6D_to_rot_
        elif self.out_type == 3: # transform matrix (compskin setting)
            self.use_full_vertex = False
            self.out_dim = 6 # (6D + translation 3) will be reshaped into 3x4 matrix
            M_ = num_cage_vertices
            num_cage_vertices = num_cage_vertices * 4

            from utils.exp_utils import create_BN
            self._6D_to_rot_ = create_BN
        else:
            raise NotImplementedError('out_type not implemented')

        # first version: control-mode key_d_model only supports delta form
        # without shape conditioning. Other combinations must fail loudly
        # rather than silently falling back to a different behavior.
        if self.out_type != 0:
            raise NotImplementedError(
                f'NeuralSparseBlendpoint currently only supports out_type=0 (delta form), got out_type={self.out_type}'
            )
        if self.use_shp:
            raise NotImplementedError(
                'NeuralSparseBlendpoint currently does not support use_shp=True (shape conditioning)'
            )

        from models.encoder import LinearEncoder

        M = num_cage_vertices
        L = hid_dim
        NZ= least_number_of_zeros

        # coordinate predictor -- identical to NGBC.key_weight_model
        self.key_weight_model = LinearEncoder(
            in_dim=self.in_dim, out_dim=M,
            use_softmax=use_softmax,
            use_relu=use_relu, # default setting
            use_elu=use_elu,
            use_softplus=use_softplus,
            use_least_N=use_least_N,
            use_least_N_on_V=use_least_N_on_V,
            no_activation=no_activation,
            use_pou=use_pou,
        ).to(device)

        # expression encoder -- identical to NGBC.exp_z_model
        if self.use_shp:
            self.exp_z_model = Model_mk2_1(
                in_dim=in_dim_exp,
                style_dim=L,
                out_dim=L,
                num_layers=self.num_layers,
                use_style=True,
                out_type='global',
            ).to(device)
        else:
            self.exp_z_model = LinearEncoder(
                in_dim=in_dim_exp,
                out_dim=L,
                num_layers=self.num_layers,
                out_type='global',
            ).to(device)

        # shape encoder (unreachable while use_shp=True is blocked above; kept for interface parity)
        if self.use_shp:
            self.shape_model = LinearEncoder(
                in_dim=self.in_dim,
                out_dim=L,
                out_type='global'
            ).to(device)

        # cage displacement predictor -- replaced with the control-mode head
        self.key_d_model = ControlModeKeyD(
            latent_dim=L,
            num_controls=M,
            num_modes=num_modes,
            num_layers=mode_num_layers,
            init_std=mode_init_std,
            device=device,
        )

        if self.is_train:
            if self.use_exp_recon:
                self.recon_exp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)

            if self.use_shp_recon:
                self.recon_shp_model = nn.ModuleList([
                    LinearEncoder(in_dim=L, out_dim=N*3, num_layers=self.num_layers, out_type='global')
                    for N in N_list
                ]).to(device)

    def reshape_key_d(self, key_d, B):
        if self.out_type == 2:
            # cage transform matrix
            key_d = key_d.reshape(B,self.num_cage_vertices, 9)
            tmp_R, tmp_t = key_d[...,:6], key_d[...,6:]

            tmp_R = self._6D_to_rot_(tmp_R).reshape(B, -1, 3, 3)
            key_d = torch.cat([tmp_R, tmp_t[..., None]], dim=-1) # (B, M, 3, 4)
            key_d = key_d.permute(0,1,3,2).reshape(B, -1, 3) # (B, M4, 3)

        elif self.out_type == 3:
            # cage transform matrix
            key_d = key_d.reshape(B,self.num_cage_vertices, 6)
            key_d = self._6D_to_rot_(key_d)# (B, M, 3, 4)
            key_d = key_d.permute(0,1,3,2).reshape(B, -1, 3) # (B, M4, 3)

        else:
            key_d = key_d.reshape(B, self.num_cage_vertices, 3)

        return key_d

    @staticmethod
    def count_parameters(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def log_parameter_num(self):
        """Per-submodule + total trainable parameter count, for logging at train start."""
        log_txt = "========< NeuralSparseBlendpoint >========\n"
        log_txt += f"[key_weight_model]: \t{self.count_parameters(self.key_weight_model)}\n"
        if self.use_shp:
            log_txt += f"[shape_model]: \t{self.count_parameters(self.shape_model)}\n"
        log_txt += f"[exp_z_model]: \t{self.count_parameters(self.exp_z_model)}\n"
        log_txt += f"[key_d_model]: \t{self.count_parameters(self.key_d_model)}\n"
        if self.is_train and self.use_exp_recon:
            log_txt += f"[recon_exp_model]: \t{self.count_parameters(self.recon_exp_model)}\n"
        if self.is_train and self.use_shp_recon:
            log_txt += f"[recon_shp_model]: \t{self.count_parameters(self.recon_shp_model)}\n"
        log_txt += "-------------------------------\n"
        log_txt += f"[total]: \t{self.count_parameters(self)}\n"
        log_txt += "============================================================\n"
        return log_txt

    def process_input(self, source_vert, deform_vert, source_norm, deform_norm, hat_mask):
        source_in = source_vert
        deform_in = deform_vert - source_vert # as a delta
        # deform_in = deform_vert # as a vertex

        if self.in_type > 0:
            source_in = torch.cat([source_in, source_norm], dim=-1)
            deform_in = torch.cat([deform_in, deform_norm], dim=-1)

        deform_in = torch.cat([deform_in, source_in], dim=-1)

        if self.in_type==2:
            source_in = torch.cat([source_in, hat_mask], dim=-1)
            deform_in = torch.cat([deform_in, hat_mask], dim=-1)
        return source_in, deform_in

    def forward(self, source_vert, deform_vert, source_norm, deform_norm, mesh_data, hat_mask=None, epoch=0, out_kw=False):
        """
        Args:
            source_vert (torch.tensor): [B, N, 3] source mesh vertices
            deform_vert (torch.tensor): [B, N, 3] deformed mesh vertices
            source_norm (torch.tensor): [B, N, 3] source mesh vertex normals
            deform_norm (torch.tensor): [B, N, 3] deformed mesh vertex normals
            mesh_data (int): indicator for data (0: voca, 1: biwi, 2: multiface)
            epoch (int): train epoch (epoch != iteration)
        Returns:
            (pred_deformed, recon_deformed, recon_source):
            predicted deformed mesh using weight and displacement,
            reconstructed deformed mesh
            reconstructed source mesh
        """
        B, N, _ = deform_vert.shape

        hat_mask = plateau_hat_points(source_vert)
        source_in, deform_in = self.process_input(source_vert, deform_vert, source_norm, deform_norm, hat_mask)

        if self.use_shp:
            z_ID_B = self.shape_model(source_in) # (B, 1, L)

            exp_z = self.exp_z_model(deform_in, z_ID_B) # (B, 1, L)
            key_d = self.key_d_model(exp_z, z_ID_B)
        else:
            exp_z = self.exp_z_model(deform_in) # (B, 1, L)
            key_d = self.key_d_model(exp_z)

        key_d = self.reshape_key_d(key_d, B)

        key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
        # --> (B, N, 4M) if self.opts.out_type == 2
        delta_v = torch.einsum('bnc,bci->bni',key_weight,key_d)

        if self.use_full_vertex:
            pred_deformed = delta_v
        else:
            pred_deformed = delta_v + source_vert

        ## necessary
        if self.use_shp_recon:
            recon_source = self.recon_shp_model[mesh_data](z_ID_B)
            recon_source = recon_source.reshape(B, -1, 3)
        else:
            recon_source = 0

        ## unnecessary
        if self.use_exp_recon:
            recon_delta_v = self.recon_exp_model[mesh_data](exp_z)
            recon_delta_v = recon_delta_v.reshape(B, -1, 3)
            recon_deformed = recon_delta_v + source_vert
        else:
            recon_deformed = 0

        ## source cage (key_s / pred_cage_s): forward the "zero deformation" input
        ## through the same exp/key predictors to get the cage at rest. Computed
        ## unconditionally (not just for use_full_vertex/out_type==1) since it's
        ## needed for cage-consistency and distance losses regardless of out_type.
        source_in_s, deform_in_s = self.process_input(source_vert, source_vert, source_norm, deform_norm, hat_mask)

        if self.use_shp:
            exp_z_s = self.exp_z_model(deform_in_s, z_ID_B) # (B, 1, L)
            key_s = self.key_d_model(exp_z_s, z_ID_B)
        else:
            exp_z_s = self.exp_z_model(deform_in_s) # (B, 1, L)
            key_s = self.key_d_model(exp_z_s)
        key_s = self.reshape_key_d(key_s, B)

        pred_source = torch.einsum('bnc,bci->bni',key_weight,key_s)

        if out_kw:
            return pred_deformed, recon_deformed, recon_source, exp_z, key_d, key_weight

        return pred_deformed, recon_deformed, recon_source, exp_z, pred_source, hat_mask, key_weight, key_s, key_d

    def cyclic_loss(self,
            source_vert,
            source_norm,
            pred_cage_d,
            pred_cage_s,
            hat_mask=None,
            mode=1,
        ):
        """
        cyclic loss
        """
        B = source_vert.shape[0]

        # source_vert (target identity) and deform_vert (this batch's own identity) can have
        # different vertex counts -- unlike forward()/process_input(), we can't build a
        # source-vs-deform delta here. deform_vert/deform_norm are only used for B above;
        # source_in (target identity only) is all key_weight_model needs.
        source_in = source_vert

        hat_mask = plateau_hat_points(source_vert)

        if self.in_type > 0:
            source_in = torch.cat([source_in, source_norm], dim=-1)

        if self.in_type==2:
            source_in = torch.cat([source_in, hat_mask], dim=-1)

        key_weight = self.key_weight_model(source_in, N=self.NZ) # (B, N, M)
        key_weight = key_weight.detach()
        # --> (B, N, 4M) if self.opts.out_type == 2

        def_v = torch.einsum('bnc,bci->bni',key_weight,pred_cage_d)
        neu_v = torch.einsum('bnc,bci->bni',key_weight,pred_cage_s)

        if self.use_full_vertex:
            pred_deformed = def_v
            pred_source = neu_v
        else:
            pred_deformed = def_v + source_vert
            pred_source = neu_v + source_vert

        deform_in_td = pred_deformed - source_vert
        deform_in_ts = source_vert - source_vert
        if self.in_type > 0:
            deform_in_td = torch.cat([deform_in_td, source_norm], dim=-1)
            deform_in_ts = torch.cat([deform_in_ts, source_norm], dim=-1)
        deform_in_td = torch.cat([deform_in_td, source_in], dim=-1)
        deform_in_ts = torch.cat([deform_in_ts, source_in], dim=-1)

        exp_z_td = self.exp_z_model(deform_in_td)
        key_d_td = self.key_d_model(exp_z_td)
        v_td = self.reshape_key_d(key_d_td, B)

        exp_z_ts = self.exp_z_model(deform_in_ts)
        key_d_ts = self.key_d_model(exp_z_ts)
        v_ts = self.reshape_key_d(key_d_ts, B)

        loss_neu = torch.nn.functional.mse_loss(v_ts, pred_cage_s)
        loss_def = torch.nn.functional.mse_loss(v_td, pred_cage_d)
        loss = loss_neu + loss_def
        if mode == 2:
            loss_recon = torch.nn.functional.mse_loss(pred_source, source_vert) # newly added
            loss = loss + loss_recon
        return loss, pred_deformed, pred_source

    def retarget(self,
                 src_neu_vert, src_neu_norm, src_def_vert, src_def_norm, tgt_neu_vert, tgt_neu_norm,
                 mesh_data=0, out_kw=False, recon_out=True):
        """
        Args:
            src_neu_vert (torch.tensor): [B, N, 3] source neutral mesh vertex positions
            src_neu_norm (torch.tensor): [B, N, 3] source neutral mesh vertex normals

            src_def_vert (torch.tensor): [B, N, 3] source deformed mesh vertex positions
            src_def_norm (torch.tensor): [B, N, 3] source deformed mesh vertex normals

            tgt_neu_vert (torch.tensor): [B, M, 3] target neutral mesh vertex positions
            tgt_neu_norm (torch.tensor): [B, M, 3] target neutral mesh vertex normals

            mesh_data (int): indicator for data (0: voca, 1: biwi, 2: multiface) -- not used!

        Returns:
            (pred_deformed, pred_source):
            predicted target deformation and neutral mesh using weight and cage prediction
        """
        B, N, _ = src_def_vert.shape

        tgt_in = tgt_neu_vert
        src_in = src_neu_vert

        deform_in_d = src_def_vert-src_neu_vert # as a delta
        deform_in_s = src_neu_vert-src_neu_vert # as a delta

        if self.in_type > 0:
            tgt_in = torch.cat([tgt_in, tgt_neu_norm], dim=-1)
            src_in = torch.cat([src_in, src_neu_norm], dim=-1)
            deform_in_d = torch.cat([deform_in_d, src_def_norm], dim=-1)
            deform_in_s = torch.cat([deform_in_s, src_def_norm], dim=-1)

        deform_in_d = torch.cat([deform_in_d, src_in], dim=-1)
        deform_in_s = torch.cat([deform_in_s, src_in], dim=-1)

        if self.in_type == 2:
            src_hat_mask = plateau_hat_points(src_neu_vert)
            src_in = torch.cat([src_in, src_hat_mask], dim=-1)
            deform_in_s = torch.cat([deform_in_s, src_hat_mask], dim=-1)
            deform_in_d = torch.cat([deform_in_d, src_hat_mask], dim=-1)

            tgt_hat_mask = plateau_hat_points(tgt_neu_vert)
            tgt_in = torch.cat([tgt_in, tgt_hat_mask], dim=-1)

        with torch.no_grad():
            if self.use_shp:
                z_ID_B = self.shape_model(src_in) # (B, 1, L)

                exp_z_d = self.exp_z_model(deform_in_d, z_ID_B) # (B, 1, L)
                key_d = self.key_d_model(exp_z_d, z_ID_B)

                exp_z_s = self.exp_z_model(deform_in_s, z_ID_B) # (B, 1, L)
                key_s = self.key_d_model(exp_z_s, z_ID_B)
            else:
                exp_z_d = self.exp_z_model(deform_in_d) # (B, 1, L)
                key_d = self.key_d_model(exp_z_d)

                exp_z_s = self.exp_z_model(deform_in_s) # (B, 1, L)
                key_s = self.key_d_model(exp_z_s)

            key_d = self.reshape_key_d(key_d, B)
            key_s = self.reshape_key_d(key_s, B)

            key_weight = self.key_weight_model(tgt_in, N=self.NZ) # (B, N, K)
            # --> (B, N, 4K) if self.opts.out_type == 2

            delta_dv = torch.einsum('bnc,bci->bni',key_weight,key_d)
            delta_sv = torch.einsum('bnc,bci->bni',key_weight,key_s)

        if self.use_full_vertex:
            pred_deformed = delta_dv
            pred_source = delta_sv
        else:
            pred_deformed = delta_dv + tgt_neu_vert
            pred_source = delta_sv + tgt_neu_vert

        if out_kw:
            return pred_deformed, pred_source, exp_z_d, key_d, exp_z_s, key_s, key_weight

        return pred_deformed, pred_source

    @torch.no_grad()
    def predict_coordinate(self, tgt_neu_vert, tgt_neu_norm):
        """
        Args:
            tgt_neu_vert (torch.tensor): [B, N, 3] target neutral mesh vertex positions
            tgt_neu_norm (torch.tensor): [B, N, 3] target neutral mesh vertex normals
        Returns:
            key_weight, cooridnate w.r.t the cage vertices  [B, N, M]

        """
        B, N, _ = tgt_neu_vert.shape
        tgt_in = tgt_neu_vert

        if self.in_type > 0:
            tgt_in = torch.cat([tgt_in, tgt_neu_norm], dim=-1)

        if self.in_type == 2:
            tgt_hat_mask = plateau_hat_points(tgt_neu_vert)
            tgt_in = torch.cat([tgt_in, tgt_hat_mask], dim=-1)

        key_weight = self.key_weight_model(
            tgt_in,
            N=self.NZ # (not used!)
        ) # (B, N, M)

        return key_weight

    @torch.no_grad()
    def retarget_animation(
        self,
        src_neu_vert,
        src_neu_norm,
        src_def_vert,
        src_def_norm,
        key_weight,
        tgt_neu_vert=None, # required if the model is delta prediction!
        return_source=False,
    ):
        """
        Args:
            src_neu_vert (torch.tensor): [B, N, 3] source neutral mesh vertex positions
            src_neu_norm (torch.tensor): [B, N, 3] source neutral mesh vertex normals

            src_def_vert (torch.tensor): [B, N, 3] source deformed mesh vertex positions
            src_def_norm (torch.tensor): [B, N, 3] source deformed mesh vertex normals

            key_weight (torch.tensor):   [B, M, K] target neutral mesh vertex normals
            tgt_neu_vert (torch.tensor): [B, M, 3] target neutral mesh vertex positions

            return_source (bool): if True, returns reconstructed source mesh
        Returns:
            (pred_deformed): [B, M, 3] predicted target deformed mesh vertices
        """
        B, N, _ = src_neu_vert.shape
        src_neu_in = src_neu_vert
        src_def_in = src_def_vert - src_neu_vert # as a delta (optional)

        if self.in_type > 0:
            src_neu_in = torch.cat([src_neu_in, src_neu_norm], dim=-1)
            src_def_in = torch.cat([src_def_in, src_def_norm], dim=-1)

        src_def_in = torch.cat([src_def_in, src_neu_in], dim=-1)

        if self.in_type == 2:
            src_hat_mask = plateau_hat_points(src_neu_vert)
            src_neu_in = torch.cat([src_neu_in, src_hat_mask], dim=-1)
            src_def_in = torch.cat([src_def_in, src_hat_mask], dim=-1)

        if self.use_shp:
            z_ID_B = self.shape_model(src_neu_in) # (B, 1, L)

            src_exp_z = self.exp_z_model(src_def_in, z_ID_B) # (B, 1, L)
            src_key_d = self.key_d_model(src_exp_z, z_ID_B) # (B, 3K)
        else:
            src_exp_z = self.exp_z_model(src_def_in) # (B, 1, L)
            src_key_d = self.key_d_model(src_exp_z) # (B, 3K)

        src_key_d = self.reshape_key_d(src_key_d, B) # (B, K, 3)

        tgt_def_v = torch.einsum('bnc,bci->bni', key_weight, src_key_d)

        if self.use_full_vertex:
            # absolute position
            pred_deformed = tgt_def_v
        else:
            # displacement
            pred_deformed = tgt_def_v + tgt_neu_vert

        return pred_deformed, src_key_d

    @torch.no_grad()
    def blendshape(
        self,
        exp_z,
        key_weight,
        tgt_neu_vert=None, # required if the model is delta prediction!
        src_neu_vert=None,
        src_neu_norm=None,
    ):
        """
        Animate target mesh using blendshape coefficient
        *available if the model is trained with `align_latent==True`

        Args:
            exp_z (torch.tensor): [B, 1, 128] blendshape coefficient for the expression

            key_weight (torch.tensor):   [B, M, K] target neutral mesh vertex normals
            tgt_neu_vert (torch.tensor): [B, M, 3] target neutral mesh vertex positions

            src_neu_vert (torch.tensor): [B, N, 3] source neutral mesh vertex positions # iff self.use_shp==True
            src_neu_norm (torch.tensor): [B, N, 3] source neutral mesh vertex normals # iff self.use_shp==True
        Returns:
            (pred_deformed): [B, M, 3] predicted target deformed mesh vertices
        """
        B = exp_z.shape[0]

        if self.use_shp:
            src_in = src_neu_vert
            if self.in_type > 0:
                src_in = torch.cat([src_in, src_neu_norm], dim=-1)
            if self.in_type == 2:
                src_hat_mask = plateau_hat_points(src_neu_vert)
                src_in = torch.cat([src_in, src_hat_mask], dim=-1)
            z_ID_B = self.shape_model(src_in) # (B, 1, L)

            key_d = self.key_d_model(exp_z, z_ID_B) # (B, 3K)
        else:
            key_d = self.key_d_model(exp_z) # (B, 3K)

        key_d = self.reshape_key_d(key_d, B) # (B, K, 3)

        cage_v = torch.einsum('bnc,bci->bni', key_weight, key_d)

        if self.use_full_vertex:
            pred_deformed = cage_v
        else:
            pred_deformed = cage_v + tgt_neu_vert

        return pred_deformed, key_d
