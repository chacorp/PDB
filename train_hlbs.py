"""
train_hlbs.py — HierarchicalLBS standalone pre-training.

HLBS only: predicts coarse LBS deformation toward smooth_GT.
EDD is not involved. Analogous to Stage-1 in train_stage_disp.py.

Usage:
    python train_hlbs.py --config configs/train.yml \
        --rig_path utils/mf/rig_info.json \
        --topo_key mf \
        --num_identities 13 \
        --smooth_n_iter 16 \
        --use_data1 --data_toggle --batch_size 16 --max_epoch 200 --tb
"""
import os
import sys
import json
import argparse
import glob
import random
import time
import numpy as np
import yaml

import warnings
warnings.filterwarnings("ignore", message="torch.sparse.SparseTensor.*is deprecated")

import torch
import torch.nn.functional as F
# TF32 matmul on Ampere+ (A5000 등) — matmul 가속. Volta(V100)엔 TF32 HW가
# 없어 이 플래그는 그냥 무시됨(완전 무해, no-op).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils.matplotlib_rnd import plot_image_array
from dataloader_CBD import CBDDataset, CBDdataSampler, CBD_collate_wrapper
from utils.vis_loader import CheckpointVisLoader


class Logger:
    def __init__(self, file_path):
        self.file_path = file_path
    def write(self, txt):
        with open(self.file_path, 'a') as f:
            f.write(txt)


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yml")

    # rig / topology
    parser.add_argument("--rig_path", type=str, required=True,
                        help='Directory containing rig_info.json and skin_weights*.npy (from maya_rig/export_rig.py)')
    parser.add_argument("--topo_key", type=str, default='mf',
                        choices=['mf', 'biwi', 'voca', 'ict'],
                        help='Which skin weight topology to use')
    parser.add_argument("--num_identities", type=int, default=13)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)

    # regularization
    parser.add_argument("--lambda_W_reg", type=float, default=1e-4,
                        help='L2 reg on delta_W')
    parser.add_argument("--lambda_t_reg", type=float, default=1e-4,
                        help='L2 reg on delta_t')
    parser.add_argument("--lambda_neu", type=float, default=1.0,
                        help='Neutral reconstruction loss: forward(src, delta=0) == src')
    parser.add_argument("--lambda_W_smooth", type=float, default=0.0,
                        help='Dirichlet smoothness on W: mean ||W_i - W_j||^2 over edges. '
                             '0 = disabled (default)')

    # ablation
    parser.add_argument("--freeze_adapt", dest='freeze_adapt', action='store_true',
                        help='Freeze skin_weight_net & bind_pose_net (delta_W=0, delta_t=0). '
                             'Only joint transforms are learned. Ablation for Maya init quality.')
    parser.set_defaults(freeze_adapt=False)
    parser.add_argument("--use_joint_trans", dest='use_joint_trans', action='store_true',
                        help='Predict per-joint local translation in addition to rotation (6+3=9 DOF per joint)')
    parser.set_defaults(use_joint_trans=False)
    parser.add_argument("--smooth_delta_W", type=int, default=0,
                        help='Laplacian smoothing iterations on delta_W in forward pass (0=off)')
    parser.add_argument("--smooth_delta_W_alpha", type=float, default=0.5,
                        help='Smoothing blend ratio (0=no smooth, 1=full neighbor average)')

    # surface losses
    parser.add_argument("--lambda_normal", type=float, default=0.0,
                        help='Normal consistency loss (1 - cos(n_pred, n_gt))')
    parser.add_argument("--normal_full_only", action='store_true',
                        help='Compute recon-normal ONLY on full-mesh batches; skip on '
                             'subsampled (permed) batches instead of kNN-PCA approximation. '
                             'Scale --lambda_normal up (~4x for mix4) to compensate frequency.')
    parser.add_argument("--lambda_curvature", type=float, default=0.0,
                        help='Curvature loss (Laplacian difference)')

    # full prediction mode
    parser.add_argument("--full_prediction", dest='full_prediction', action='store_true',
                        help='Use HierarchicalLBS_FullPred (no Maya base dependency)')
    parser.set_defaults(full_prediction=False)
    parser.add_argument("--init_phase_epochs", type=int, default=20,
                        help='Total Phase 1 epochs. Behavior depends on --init_mode.')
    parser.add_argument("--init_hold_epochs", type=int, default=0,
                        help='Hold epochs before annealing (only used in hold_anneal mode)')
    parser.add_argument("--init_mode", type=str, default='anneal',
                        choices=['anneal', 'hold_anneal', 'hold_cutoff'],
                        help='anneal: linear decay 0→K. '
                             'hold_anneal: hold H epochs then anneal to K. '
                             'hold_cutoff: hold H epochs then drop to 0.')
    parser.add_argument("--lambda_init", type=float, default=1.0,
                        help='Init supervision loss weight')

    # bind pose regularization (independent of lambda_init, not annealed)
    parser.add_argument("--lambda_bind_reg", type=float, default=0.0,
                        help='Bind pose MSE to Maya init (always on, 0=disabled). '
                             'Excludes helper joints when --use_helpers=1.')
    parser.add_argument("--lambda_helper_residual", type=float, default=0.0,
                        help='L2 penalty on HELPER joint bind_pose residual (weak). '
                             'Helpers may drift further from base. Active when '
                             '--use_helpers=1 (or base_residual mode).')
    parser.add_argument("--lambda_bind_residual", type=float, default=0.0,
                        help='[bind_pose_base_residual mode] L2 penalty on NON-helper '
                             'joint residual (strong — keep near ICT-mean base). '
                             'Differentiated: non-helper strong, helper weak.')
    parser.add_argument("--lambda_cross_cyclic", type=float, default=0.0,
                        help='Cyclic consistency loss for cross-retarget (Track B). '
                             'After first retarget produces pred_tgt_def (A→B), runs a '
                             'reverse retarget (B→A) using pred_tgt_def as new source-deformed '
                             'and src_neu(A) as new target-neutral. '
                             'L_cross_cyclic = MSE(reconstructed src_def, original src_def). '
                             'Encourages bijective expression encoding/decoding.')
    parser.add_argument("--lambda_mirror", type=float, default=0.0,
                        help='Bilateral symmetry loss for L/R helper joint pairs. '
                             'L_mirror = MSE(pos[L], mirror_x(pos[R])), with '
                             'mirror_x([x,y,z]) = [-x, y, z]. Encourages predicted '
                             'helper positions to be left-right symmetric. Active only '
                             'when --use_helpers=1.')
    parser.add_argument("--helper_joints_json", type=str,
                        default='maya_rig/hybrid/helper_joints_v1.json',
                        help='Path to helper_joints config (used for mirror_pairs).')
    parser.add_argument("--lambda_w_mirror", type=float, default=0.0,
                        help='Bilateral symmetry PRIOR on skinning weights W. '
                             'L_w_mirror = mean|W[v,j] - W[mir(v), swap(j)]|, where mir = per-topology '
                             'L/R vertex mirror map (nearest to x-flipped position) and swap = joint '
                             'L/R swap (Left<->Right, midline->self). Penalizes network non-equivariance '
                             '(asymmetric W under symmetric input). Full-mesh batches only.')
    parser.add_argument("--w_mirror_mode", type=str, default='vertex',
                        choices=['vertex', 'moment'],
                        help="L_w_mirror mechanism. 'vertex': per-vertex W[v]~W[mir(v),swap] "
                             "(needs vertex mirror map; strongest, directly targets A_W). "
                             "'moment': match each joint's weight-field moments (mass/centroid/"
                             "covariance) to its mirror joint (map-free, needs only joint swap; "
                             "robust when topology is not vertex-symmetric, e.g. mf).")

    # regional weight constraint
    parser.add_argument("--lambda_rwc", type=float, default=0.0,
                        help='Regional weight constraint loss weight (0=disabled)')
    parser.add_argument("--rwc_alpha", type=float, default=0.5,
                        help='Min threshold = alpha * mean(Maya weight) per constrained joint')
    parser.add_argument("--rwc_adaptive", dest='rwc_adaptive', action='store_true',
                        help='Adaptive alpha: joints with fewer dominant vertices get stronger constraint')
    parser.set_defaults(rwc_adaptive=False)
    parser.add_argument("--rwc_topologies", type=str, default='ict',
                        help='Comma-separated topologies for RWC (e.g. "ict" or "ict,mf")')

    # hierarchy locality loss (leaf joints dominate their regions)
    parser.add_argument("--lambda_hier", type=float, default=0.0,
                        help='Hierarchy locality loss weight (0=disabled)')
    parser.add_argument("--hier_margin", type=float, default=0.0,
                        help='Margin for hierarchy loss (leaf must exceed others by this much)')

    # distance-based weight locality loss
    parser.add_argument("--lambda_dist", type=float, default=0.0,
                        help='Distance-based weight locality: W should be high for close joints')
    # Mesh2Animation-inspired skin-weight regularizers / metric
    parser.add_argument("--lambda_wlap", type=float, default=0.0,
                        help='#1 1-ring Laplacian weight smoothness loss (topology-independent).')
    parser.add_argument("--lambda_wref", type=float, default=0.0,
                        help='#3 reference-weight prior: MSE to closest-bone one-hot (L_id).')
    parser.add_argument("--lambda_ortho", type=float, default=0.0,
                        help='Simplicits eq7-inspired off-diagonal weight orthogonality: decorrelate per-joint weight columns (mean_v W_i*W_j -> 0, i!=j) to induce sparse/disjoint skinning regions. diag NOT forced (partition-of-unity). 0=off.')
    parser.add_argument("--lambda_inside_bind", type=float, default=0.0,
                        help='Hinge penalty when bind-pose joints protrude outside the '
                             'source mesh (SkinTokens "Bone-Mesh Containment" idea, '
                             'supervised). 0 = off. Approx SDF via nearest-vertex-normal.')
    parser.add_argument("--lambda_inside_def", type=float, default=0.0,
                        help='Same penalty applied to DEFORMED joint positions vs the '
                             'predicted deformed mesh — catches joints that pop out '
                             'after FK chain. Subsample → skipped (faces invalid).')
    parser.add_argument("--w_metric", type=int, default=1, choices=[0, 1],
                        help='#2 log GT-free weight-quality metric to TB (diagnostic, no loss).')

    # freeze bind pose (use Maya init directly, no bind_pose_net prediction)
    parser.add_argument("--freeze_bind_pose", dest='freeze_bind_pose', action='store_true',
                        help='Fix bind pose to Maya init (skip bind_pose_net). '
                             'With --per_id_bind_pose_dir set, uses per-id v3 GT '
                             'instead (Stage-2: bind pose = oracle GT).')
    parser.set_defaults(freeze_bind_pose=False)
    parser.add_argument("--bind_pose_base_residual", type=int, default=0, choices=[0, 1],
                        help='1: bind_pose_net predicts a residual for ALL joints on '
                             'top of a fixed ICT-mean base (joint_pos = base + residual). '
                             'Topology-agnostic universal base. 0: legacy (non-helper '
                             'absolute, helper parent+residual).')
    parser.add_argument("--bind_pose_net_ckpt", type=str, default=None,
                        help='[Stage-3] Load bind_pose_net.* weights from this '
                             'checkpoint (e.g. a Stage-1 bind_only_*.pth).')
    parser.add_argument("--freeze_bind_pose_net", type=int, default=0, choices=[0, 1],
                        help='[Stage-3] Freeze bind_pose_net params (requires_grad=False). '
                             'Net still runs forward — uses its PREDICTED bind pose, '
                             'not GT — so skin/transform branches adapt to the real '
                             'inference distribution. Differs from --freeze_bind_pose '
                             '(which bypasses the net and uses GT/mean directly).')

    # bind-pose mode: 'net' (default MLP) | 'anchor_pool' (closed-form NFS attention)
    parser.add_argument("--bind_pose_mode", type=str, default='net',
                        choices=['net', 'anchor_pool'],
                        help='net: bind_pose_net MLP (default). '
                             'anchor_pool: closed-form NFS attention with precomputed (anchor, offset).')
    parser.add_argument("--joint_anchors_npy", type=str, default=None,
                        help='Path to joint_anchors.npy [J, K] (from precompute_joint_anchors.py)')
    parser.add_argument("--joint_offsets_npy", type=str, default=None,
                        help='Path to joint_offsets.npy [J, 3] (world-space residuals)')
    parser.add_argument("--attn_temperature_init", type=float, default=0.1,
                        help='Initial softmax temperature for anchor pooling (per-joint, learnable)')
    parser.add_argument("--bind_pos_cache_dir", type=str, default=None,
                        help='Per-id bind_pos cache (lookup {id_name}_bind_pos.npy). '
                             'In anchor_pool mode, defaults to --nfs_feat_dir. '
                             'Build with precompute_per_id_bind_pos.py.')

    # GMM hybrid: add Gaussian bias to skin weight prediction
    parser.add_argument("--use_gmm_hybrid", dest='use_gmm_hybrid', action='store_true',
                        help='Add Gaussian bias (based on predicted joint_pos) to skin_weight_net logits')
    parser.set_defaults(use_gmm_hybrid=False)
    parser.add_argument("--use_geodesic_gauss", action='store_true', default=False,
                        help='Replace Euclidean ||v - μ_j||² with precomputed geodesic dist²[topo, j, v] '
                             'in: (1) GMM Gauss bias, (2) L_net_center, (3) distance_weight_loss. '
                             'Requires geo_dist_{topo}.npy in --geo_dist_dir (default: --rig_path). '
                             'Falls back to Euclidean per-batch when topology has no table.')
    parser.add_argument("--geo_dist_dir", type=str, default=None,
                        help='Directory with geo_dist_{topo}.npy [J, V] tables '
                             '(from precompute_geo_dist.py). Defaults to --rig_path.')
    parser.add_argument("--per_id_bind_pose_dir", type=str, default=None,
                        help='Dir with {id_name}_bind_pos_landmark.npy per-id GT bind_pose '
                             '(from precompute_per_id_bind_pos_landmark.py). When set, '
                             'L_bind_reg target switches from per-topo mean to per-id GT '
                             'where available; per-topo mean used as fallback for missing IDs.')
    # ── ICT cross-retargeting paired supervision (Track B) ──
    parser.add_argument("--lambda_cross_retarget", type=float, default=0.0,
                        help='Weight for cross-id retarget loss. When > 0, builds a CrossPairICTDataset '
                             'and at every train step computes L_cross = MSE(model.retarget(src_id_A, '
                             'tgt_id_B_neutral), tgt_id_B_def) — supervising explicit identity vs '
                             'expression disentanglement using shared z_FACS pairs.')
    parser.add_argument("--cross_pair_iden_vecs", type=str,
                        default='ict_face_pt/random_identity_vecs.npy',
                        help='Path to .npy or .pt file with iden_vecs [N, 100] for cross-pair sampling. '
                             'Default = TRAINING IDs (random_identity_vecs[:111]). NEVER point this at '
                             'data/ICT_live_100/iden_vecs.npy — those are the TEST 100 IDs; using them '
                             'here would leak test identities into training.')
    parser.add_argument("--cross_pair_length", type=int, default=0,
                        help='Items-per-epoch for cross-pair loader (0=N_id*53). Lower = fewer pairs/epoch.')

    # ── Vertex subsample augmentation ──
    parser.add_argument("--subsample_ratio", type=float, default=0.0,
                        help='Per-batch vertex subsample ratio (0 = disabled, 0.5 = use half). '
                             'Applies same perm to src_v / gt_v / normals / nfs_feat / geo_dist; '
                             'sets batch.perm_idx so existing perm-aware paths (rwc, hier, init) work. '
                             'Skips face-dependent losses (normal, curvature). Landmark vertices are '
                             'pinned to ensure anchor preservation.')
    parser.add_argument("--subsample_mode", type=str, default='random',
                        choices=['random', 'fps', 'importance', 'importance_strict',
                                 'mixed', 'mix3', 'mix4'],
                        help='random: uniform random. '
                             'fps: farthest point sampling (PointNet++ style spatial spread). '
                             'importance: soft contour weighting w = 1 + α·bump (non-contour '
                             'still possible). '
                             'importance_strict: hard contour-only (w = bump; non-contour w=0). '
                             'mixed: per-sample alternation (even=random, odd=importance). '
                             'mix3: per-batch 1/3 of full / random / importance(α=99). '
                             'mix4: per-batch 1/4 of full / fps / random / importance_strict '
                             '(each non-full mode draws its own ratio per batch).')
    parser.add_argument("--subsample_mix3_alpha", type=float, default=99.0,
                        help='[mix3 mode] alpha for the importance sub-mode (near-100% contour).')
    parser.add_argument("--subsample_ratio_min", type=float, default=0.1,
                        help='[mix4 mode] lower bound for per-batch ratio sampling.')
    parser.add_argument("--subsample_ratio_max", type=float, default=0.5,
                        help='[mix4 mode] upper bound for fps/random ratio. '
                             'For importance_strict the cap is min(this, n_contour/N) '
                             'per topology (ICT ≈ 0.44, MF ≈ 0.38).')
    parser.add_argument("--subsample_contour_r0", type=float, default=0.1,
                        help='[importance mode] inner radius (full weight) of bump around each '
                             'contour landmark vertex.')
    parser.add_argument("--subsample_contour_r1", type=float, default=0.3,
                        help='[importance mode] outer radius (zero weight) of bump.')
    parser.add_argument("--subsample_contour_alpha", type=float, default=4.0,
                        help='[importance mode] contour oversample factor. weight = 1 + α·bump; '
                             'contour vertices are sampled ~(1+α)× more often than far vertices.')
    parser.add_argument("--subsample_contour_kind", type=str, default='segment',
                        choices=['point', 'segment', 'region'],
                        help='[importance mode] bump geometry. '
                             'point=ball at each landmark (union of balls, max over centers); '
                             'segment=curve via segments connecting consecutive landmarks '
                             '(true contour band); '
                             'region=per-region plateau (centroid+max-radius ball, filled inside).')
    parser.add_argument("--subsample_region_falloff", type=float, default=0.1,
                        help='[importance/region mode] absolute falloff distance beyond region '
                             'radius. r0=R (max landmark dist from centroid), r1=R+falloff.')
    parser.add_argument("--normal_knn_k", type=int, default=16,
                        help='[subsample mode] kNN neighborhood size for PCA-based pred normal '
                             'estimation when faces are unusable (Hoppe 1992 / Klasing 2009). '
                             'GT normal uses pre-computed batch.vertices_normal directly.')
    parser.add_argument("--init_log_sigma", type=float, default=-1.2,
                        help='Initial log σ for GMM hybrid (σ = exp(-1.2) ≈ 0.3). '
                             'Overridden by --sigma_targets_npy if given.')
    parser.add_argument("--gmm_mode", type=str, default='additive',
                        choices=['additive', 'multiplicative', 'residual'],
                        help='additive: softmax(net+gauss). '
                             'multiplicative: softmax(gauss) * sigmoid(net), net refines within RBF support. '
                             'residual: softmax(gauss + exp(gauss)·tanh(net/s)·s) — bounded µ-centered residual.')
    parser.add_argument("--residual_scale", type=float, default=2.0,
                        help='Residual scale s in `gmm_mode=residual`. Bounds net contribution to ±s in log space.')
    parser.add_argument("--sigma_targets_npy", type=str, default=None,
                        help='Path to [J]-vector .npy of per-joint σ targets '
                             '(from precompute_sigma_targets.py). If set, used as '
                             'init AND as shrinkage target for log_sigma.')
    parser.add_argument("--lambda_sigma_reg", type=float, default=0.0,
                        help='Weight for σ shrinkage penalty: relu(log_σ - log_σ_target)².'
                             ' Applied only to active (face) joints if active_joints_json is set. '
                             '0 = disabled.')
    parser.add_argument("--lambda_net_center", type=float, default=0.0,
                        help='Net-output-center anchor loss: ||mean_v(softmax(logit_net)·v) - μ_pred||². '
                             'Forces logit_net distribution to be centered at the predicted joint position. '
                             'Net keeps shape freedom but center is pinned. 0 = disabled.')

    # Option A: face-mask mode (restrict W prediction to face joints, non-face → base joint)
    parser.add_argument("--use_helpers", type=int, default=1, choices=[0, 1],
                        help='Enable helper-joint-aware training. When 1: '
                             '(a) L_bind_reg excludes helpers (so helper position '
                             'is not locked to parent), '
                             '(b) [planned] helper position as parent+residual with L2 reg, '
                             '(c) [planned] L_mirror loss for L/R helper pairs. '
                             'When 0: helpers loaded from active_joints_json are treated as '
                             'regular face joints (full L_bind_reg, no mirror, no residual reg).')
    parser.add_argument("--active_joints_json", type=str, default=None,
                        help='Path to active_joints.json from analyze_active_joints.py. '
                             'If set, enables face-mask mode in HLBS.')
    parser.add_argument("--face_mask_r0", type=float, default=1.0,
                        help='Plateau inner radius for face region (where t_mask=1)')
    parser.add_argument("--face_mask_r1", type=float, default=2.25,
                        help='Plateau outer radius (where t_mask=0)')
    parser.add_argument("--no_face_mask", action='store_true',
                        help='Ablation: disable the face mask (softmax over ALL '
                             'joints, no non-face->base routing). Relies on weight '
                             'locality (wref/wlap) + recon to keep non-face inert.')
    parser.set_defaults(no_face_mask=False)

    # DiffusionNet options (per-module)
    parser.add_argument("--dfn_skin", dest='dfn_skin', action='store_true',
                        help='Use DiffusionNet for skin_weight_net')
    parser.set_defaults(dfn_skin=False)
    parser.add_argument("--dfn_bind", dest='dfn_bind', action='store_true',
                        help='Use DiffusionNet for bind_pose_net')
    parser.set_defaults(dfn_bind=False)
    parser.add_argument("--dfn_exp", dest='dfn_exp', action='store_true',
                        help='Use DiffusionNet for lbs_exp_z_model')
    parser.set_defaults(dfn_exp=False)

    # NFS pretrained feature
    parser.add_argument("--nfs_feat_dir", type=str, default=None,
                        help='Directory with NFS per-identity features (*_nfs_feat.npy). '
                             'If set, skin_weight_net and bind_pose_net use these as input.')
    parser.add_argument("--nfs_feat_dim", type=int, default=256,
                        help='Per-vertex NFS/Diff3F feature dim (matches --nfs_feat_dir files).')
    parser.add_argument("--nfs_proj_dim", type=int, default=0,
                        help='If >0, project nfs/Diff3F feature down to this dim via small MLP (Linear->ELU->Linear) before LayerNorm+concat. Learned task-aware bottleneck/denoise. 0=raw.')
    parser.add_argument("--nfs_concat", dest='nfs_concat', action='store_true',
                        help='Concat seg feat with pos+norm as input [262], 4 layers.')
    parser.set_defaults(nfs_concat=False)
    parser.add_argument("--use_corrective", type=int, default=0,
                        help="N corrective blend shapes (NBS-style baseline, single-stage); 0=off")
    parser.add_argument("--nfs_on_cpu", dest='nfs_on_cpu', action='store_true',
                        help='Keep NFS features on CPU and transfer to GPU per-batch (saves VRAM).')
    parser.set_defaults(nfs_on_cpu=False)
    parser.add_argument("--adain_pos_norm", dest='adain_pos_norm', action='store_true',
                        help='AdaIN conditioning on pos+norm [6] only. Without: AdaIN on full input.')
    parser.set_defaults(adain_pos_norm=False)

    # NFS encoder
    parser.add_argument("--nfs_ckpt", type=str, default=None,
                        help='Pretrained NFS checkpoint. If set, use NFS expression encoder for z_exp.')

    # target
    parser.add_argument("--target", type=str, default='gt',
                        choices=['gt', 'smooth_gt'],
                        help='"gt": HLBS → GT, EDD learns true LBS residual (default/recommended). '
                             '"smooth_gt": HLBS → smooth_GT, EDD learns wrinkle = GT-smooth_GT.')
    parser.add_argument("--smooth_n_iter", type=int, default=16,
                        help='Taubin smoothing iters (used when --target smooth_gt, default 16)')

    # training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--eval_iter", type=int, default=25,
                        help='Visualize every N epochs via vis_loader')
    parser.add_argument("--val_every", type=int, default=5,
                        help='Run validation every N epochs (default: 5)')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sc_step", type=int, default=1000000)
    parser.add_argument("--sc_gamma", type=float, default=0.5)
    parser.add_argument("--lambda_vert", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--profile", action='store_true',
                        help='Print per-section timing (data/fwd/bwd/opt) every 50 batches.')
    parser.add_argument("--amp", action='store_true',
                        help='bf16 autocast (mixed precision) for the forward pass. '
                             'Revert by simply dropping this flag.')
    parser.add_argument("--skip_p3d_install", action='store_true',
                        help='Skip pytorch3d auto-install when GPU FPS kernel is '
                             'missing (probe only, warn, then fall back to slow '
                             'Python FPS). Same as env SKIP_P3D_INSTALL=1.')
    # ── Caricaturization data augmentation ──
    parser.add_argument("--caricat_aug_dir", type=str, default='',
                        help='Directory with caricaturized augmented neutral meshes '
                             '({id_name}_aug.npy [N,3], per ICT/MF identity). Empty '
                             'disables. Variants: nfs_features_seg_aug, '
                             'nfs_features_seg_aug_masked, '
                             'nfs_features_seg_aug_masked_lowEye (Sela CVIU 2015 based).')
    parser.add_argument("--caricat_prob", type=float, default=0.0,
                        help='Per-item probability of replacing the neutral template '
                             'with its caricaturized variant. Deformation delta '
                             '(ICT exp_disp / MF procrustes-aligned offset) is then '
                             'added to the aug template, preserving expression. '
                             'Requires --caricat_aug_dir. 0.0 disables.')

    # mask
    parser.add_argument("--no_t_mask", dest='no_t_mask', action='store_true')
    parser.set_defaults(no_t_mask=False)
    parser.add_argument("--use_t_mask", dest='no_t_mask', action='store_false')

    # data
    parser.add_argument("--use_data0", dest='use_data0', action='store_true')
    parser.set_defaults(use_data0=False)
    parser.add_argument("--use_data1", dest='use_data1', action='store_true')
    parser.set_defaults(use_data1=False)
    parser.add_argument("--use_data2", dest='use_data2', action='store_true')
    parser.set_defaults(use_data2=False)
    parser.add_argument("--use_data3", dest='use_data3', action='store_true')
    parser.set_defaults(use_data3=False)
    parser.add_argument("--use_data4", dest='use_data4', action='store_true',
                        help='COMA + BIWI + MF + ICT (full data incl. ICT).')
    parser.set_defaults(use_data4=False)
    # curriculum
    parser.add_argument("--curriculum", dest='curriculum', action='store_true',
                        help='Curriculum learning: Phase 1 = ICT single-basis only, '
                             'Phase 2 = full data (ICT + MF)')
    parser.set_defaults(curriculum=False)
    parser.add_argument("--curriculum_epochs", type=int, default=30,
                        help='Number of epochs for curriculum Phase 1 (single-basis ICT)')

    parser.add_argument("--data_toggle", dest='data_toggle', action='store_true')
    parser.add_argument("--no_fullhead", dest='no_fullhead', action='store_true',
                        help='Exclude ICT fullhead region (11248 verts) to save GPU memory')
    parser.set_defaults(no_fullhead=False)
    parser.set_defaults(data_toggle=False)

    # checkpoint
    parser.add_argument("--ckpt", type=str, default=None,
                        help='Checkpoint dir to resume from')
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--continue_ckpt", dest='continue_ckpt', action='store_true')
    parser.set_defaults(continue_ckpt=False)
    parser.add_argument("--init_ckpt", type=str, default='',
                        help='Stage-2: initialize model weights from this checkpoint FILE '
                             '(strict=False). Unlike --continue_ckpt, starts a FRESH run dir, '
                             'optimizer, and epoch counter.')
    parser.add_argument("--mf_idpca_prob", type=float, default=0.0,
                        help='Per-sample prob of swapping the MF identity template to a pseudo-id '
                             'sampled from the mf identity-PCA space (delta-transfer, like caricat). '
                             'Requires mfpca_XX.npy in --mf_idpca_dir + matching bind GT '
                             '(nfs_features_seg/mfpca_XX_bind_pos_landmark.npy) and Diff3F feat '
                             '(diff3f_feat_raw/mfpca_XX_nfs_feat.npy).')
    parser.add_argument("--mf_idpca_dir", type=str, default='mf_idpca_aug')
    parser.add_argument("--caricat_ict_off", action='store_true',
                        help='Disable caricaturization aug for ICT samples (111 ids already give '
                             'ample identity diversity; caricat budget goes to MF instead).')
    parser.set_defaults(caricat_ict_off=False)
    parser.add_argument("--caricat_mf_off", action='store_true',
                        help='Disable the (PCA-path) MF caricaturization aug. NOTE: before '
                             '2026-07-08 the train-time MF getters had NO caricat at all — '
                             'historical runs were effectively ICT-caricat-only.')
    parser.set_defaults(caricat_mf_off=False)
    parser.add_argument("--bind_reg_exclude_extra", action='store_true',
                        help='With use_helpers=0 + a manual active_joints_json: exclude the '
                             'listed helper_joint_idx from bind-pose GT supervision (L_bind_reg) '
                             'WITHOUT enabling helper reparam/mirror. For equal-budget controls: '
                             'extra joints must learn placement freely — their template positions '
                             'are untrusted parked slots and must never be GT-anchored.')
    parser.set_defaults(bind_reg_exclude_extra=False)
    parser.add_argument("--cross_pair_mf_prob", type=float, default=0.0,
                        help='Probability of drawing the cross-retarget batch from the MF '
                             'cross-id pair dataset (per-id PCA expression sampling + delta '
                             'transfer pseudo-GT) instead of the ICT pair dataset. 0=ICT only. '
                             'Requires --lambda_cross_retarget > 0.')
    parser.add_argument("--helper_stage2_mode", type=str, default='heads',
                        choices=['heads', 'rows'],
                        help="Stage-2 helper mechanism. 'heads': new helper-only modules "
                             "(use with --use_helper_stage2). 'rows': NO new modules — train "
                             "only the helper ROWS of the three frozen final layers "
                             "(skin/pose/bind layer_out); per-joint rows are independent so "
                             "base outputs stay bit-exact (~35K effective params).")
    parser.add_argument("--use_helper_stage2", action='store_true',
                        help='Stage-2 helper branch: frozen base rig + trainable helper-only '
                             'heads (helper_logit_net / helper_pose_head / helper_bind_head). '
                             'Use with --init_ckpt (base-only stage-1 ckpt) + --freeze_base_rig '
                             '+ --use_helpers 1 + manual active_joints_json.')
    parser.set_defaults(use_helper_stage2=False)
    parser.add_argument("--freeze_base_rig", action='store_true',
                        help='Stage-2: freeze ALL stage-1 rig params (skinning/bind/exp/pose nets); '
                             'train ONLY the corrective branch (corr_blend/corr_coef). '
                             'Use with --init_ckpt + --use_corrective.')
    parser.set_defaults(freeze_base_rig=False)

    # data path
    parser.add_argument("--data_basedir", type=str, default="/data/sihun",
                        help='Base directory for datasets')

    # logging
    parser.add_argument("--log_dir", type=str, default="./ckpts_hlbs")
    parser.add_argument("--tb", dest='tb', action='store_true')
    parser.set_defaults(tb=False)
    parser.add_argument("--debug", dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument("--vis_frames", type=str, default="config/vis_frames.yml")

    # dataloader compat fields (required by CBDDataset but unused here)
    parser.add_argument("--version", type=int, default=10)
    parser.add_argument("--use_strain", dest='use_strain', action='store_true')
    parser.set_defaults(use_strain=False)
    parser.add_argument("--strain_dim", type=int, default=1)
    parser.add_argument("--use_lbs", dest='use_lbs', action='store_true')
    parser.set_defaults(use_lbs=False)
    parser.add_argument("--use_lbs_joint_center", dest='use_lbs_joint_center', action='store_true')
    parser.set_defaults(use_lbs_joint_center=False)
    parser.add_argument("--no_use_translation", dest='no_use_translation', action='store_true')
    parser.set_defaults(no_use_translation=False)
    parser.add_argument("--use_laplacian", dest='use_laplacian', action='store_true')
    parser.set_defaults(use_laplacian=False)
    parser.add_argument("--data_rand_trans", dest='data_rand_trans', action='store_true')
    parser.set_defaults(data_rand_trans=False)
    parser.add_argument("--data_rand_scale", dest='data_rand_scale', action='store_true')
    parser.set_defaults(data_rand_scale=False)
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--use_decimate", dest='use_decimate', action='store_true')
    parser.set_defaults(use_decimate=False)
    parser.add_argument("--use_data9", dest='use_data9', action='store_true')
    parser.set_defaults(use_data9=False)

    return parser.parse_args()


class HLBSTrainer:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # landmark→vertex-index map cache for _build_bump_strict (keyed per topo)
        self._lm_to_vidx_cache = {}

        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)

        from utils.rig_loader import load_rig
        from models.hierarchical_lbs import HierarchicalLBS

        if not opts.full_prediction:
            rig = load_rig(opts.rig_path)
            self.model = HierarchicalLBS(
                rig=rig,
                topology=opts.topo_key,
                in_dim_exp=12,
                hid_dim=opts.hid_dim,
                num_layers=opts.num_layers,
                device=str(self.device),
                freeze_adapt=opts.freeze_adapt,
                use_joint_trans=opts.use_joint_trans,
                smooth_delta_W=opts.smooth_delta_W,
                smooth_delta_W_alpha=opts.smooth_delta_W_alpha,
            ).to(self.device)

            # Enable multi-topology delta if rig has 2+ topologies loaded
            if len(rig.W_init) >= 2:
                self.model.enable_multi_topo(rig)
                print(f"[HLBS] Multi-topology delta enabled ({list(rig.W_init.keys())})")
        else:
            self.model = None  # built in train_full_prediction

        if not opts.full_prediction:
            if opts.freeze_adapt:
                print("[HLBS] freeze_adapt=True: delta_W=0, delta_t=0 (Maya init only, joint transforms learned)")
                opts.lambda_W_reg = 0.0
                opts.lambda_t_reg = 0.0
                opts.lambda_W_smooth = 0.0

            if opts.ckpt and opts.continue_ckpt:
                paths = sorted(glob.glob(os.path.join(opts.ckpt, f"model_hlbs_{opts.start_epoch:03d}.pth")))
                if not paths:
                    paths = sorted(glob.glob(os.path.join(opts.ckpt, "model_hlbs_best.pth")))
                if paths:
                    self.model.load_state_dict(torch.load(paths[0], map_location=self.device))
                    print(f"Resumed HLBS from: {paths[0]}")

        # ── NFS expression encoder (optional) ─────────────────────────────
        self.nfs_encoder = None
        self.nfs_z_adapter = None
        if opts.nfs_ckpt:
            self._load_nfs_encoder(opts)

        self.vis_loader = CheckpointVisLoader(opts, device=self.device)
        self._edges = None   # lazily computed from first batch faces

    def _load_nfs_encoder(self, opts):
        """Load NFS expression encoder standalone (no full NFS model needed)."""
        import yaml, pickle, trimesh
        from models.encoder import BaseDiffusionNetEncoder
        from models.CNN import TextureEncoder
        from utils.nfr_utils import get_dfn_info

        print(f"[HLBS] Loading NFS expression encoder from: {opts.nfs_ckpt}")

        # ── Read NFS config ──
        nfs_dir = os.path.dirname(opts.nfs_ckpt)
        with open(os.path.join(nfs_dir, "train_opts.yml")) as f:
            nfs_cfg = yaml.safe_load(f)
        nfs_rig_dim = nfs_cfg.get('rig_dim', 128)
        nfs_img_feat_dim = nfs_cfg.get('img_feat_dim', 128)

        # ── Build standalone modules ──
        in_shape = 6 + nfs_img_feat_dim   # pos(3) + normal(3) + img_feat(128)
        exp_encoder = BaseDiffusionNetEncoder(
            in_shape=in_shape, pre_computes=None, out_shape=nfs_rig_dim,
        ).to(self.device)
        img_encoder = TextureEncoder().to(self.device)
        img_fc = torch.nn.Linear(128, nfs_img_feat_dim).to(self.device)

        # ── Load weights from NFS checkpoint ──
        ckpt = torch.load(opts.nfs_ckpt, map_location=self.device)
        skip_suffixes = {'mass', 'L_ind', 'L_val', 'evals', 'evecs', 'grad_X', 'grad_Y', 'faces'}

        def _extract(prefix):
            out = {}
            for k, v in ckpt.items():
                if k.startswith(prefix):
                    short = k[len(prefix):]
                    if not any(short.startswith(s) or s in short for s in skip_suffixes):
                        out[short] = v
            return out

        exp_encoder.load_state_dict(_extract('mesh_exp_encoder.'), strict=False)
        img_encoder.load_state_dict(_extract('img_encoder.'), strict=False)
        img_fc.load_state_dict(_extract('img_fc.'), strict=False)
        print(f"  Loaded exp_encoder, img_encoder, img_fc weights")

        # ── Freeze all ──
        for m in [exp_encoder, img_encoder, img_fc]:
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()

        self.nfs_exp_encoder = exp_encoder
        self.nfs_img_encoder = img_encoder
        self.nfs_img_fc = img_fc
        self.nfs_encoder = exp_encoder  # for None-check in train loop

        # ── Precompute DiffusionNet operators for MF template ──
        with open("/data/sihun/pca/multiface_align/mf_templates.pkl", 'rb') as f:
            templates = pickle.load(f)
        faces_np = np.array(templates['face'], dtype=np.int32)
        first_id = [k for k in templates if k != 'face'][0]
        verts_np = templates[first_id].astype(np.float32)
        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        self._nfs_dfn_info = get_dfn_info(mesh, map_location=self.device)
        print(f"  DiffusionNet operators precomputed")

        # ── Precompute image feature from prerendered neutral image ──
        img_npy_path = "data/MF_all_v5/m--20180426--0000--002643814--GHS_neutral_img.npy"
        img_np = np.load(img_npy_path)                                  # [256, 256, 3]
        img_t = torch.tensor(img_np, dtype=torch.float32, device=self.device)
        img_t = img_t.unsqueeze(0).permute(0, 3, 1, 2)                 # [1, 3, 256, 256]
        with torch.no_grad():
            self._nfs_img_feat = img_fc(img_encoder(img_t))             # [1, img_feat_dim]
        print(f"  Image feature precomputed from {img_npy_path}")

        # ── Adapter: NFS z_GE [B, rig_dim] → HLBS z_exp [B, hid_dim] ──
        if nfs_rig_dim != opts.hid_dim:
            self.nfs_z_adapter = torch.nn.Linear(nfs_rig_dim, opts.hid_dim).to(self.device)
            print(f"  z_adapter: {nfs_rig_dim} → {opts.hid_dim}")
        else:
            self.nfs_z_adapter = torch.nn.Identity()

        # ── Freeze HLBS's own expression encoder ──
        for p in self.model.lbs_exp_z_model.parameters():
            p.requires_grad_(False)
        print("[HLBS] NFS encoder ready. HLBS lbs_exp_z_model frozen.")

    @torch.no_grad()
    def _nfs_encode_exp(self, gt_v, gt_n):
        """Encode expression using NFS pretrained encoder.
        Args:
            gt_v: [B, V, 3] expression vertices
            gt_n: [B, V, 3] expression normals
        Returns:
            z_exp: [B, hid_dim]
        """
        B, V, _ = gt_v.shape
        # vertex feature: pos + normal + img_feat (broadcast over V)
        vert_feat = torch.cat([gt_v, gt_n], dim=-1)                    # [B, V, 6]
        img_feat = self._nfs_img_feat.expand(B, -1)                    # [B, 128]
        img_exp = img_feat.unsqueeze(1).expand(-1, V, -1)              # [B, V, 128]
        nfs_input = torch.cat([vert_feat, img_exp], dim=-1)            # [B, V, 134]
        # DiffusionNet encode
        self.nfs_exp_encoder.update_precomputes(self._nfs_dfn_info)
        z_ge = self.nfs_exp_encoder(nfs_input)                         # [B, 128]
        return self.nfs_z_adapter(z_ge)                                # [B, hid_dim]

    # DTU3D 73-point landmark groupings (1-indexed) + segment connectivity for
    # contour-curve-based importance sampling. Per-vertex distance to nearest
    # segment → plateau_hat bump → weight = 1 + alpha * bump.
    _DTU3D_BROW_R = (1, 3, 5, 7)
    _DTU3D_BROW_L = (9, 11, 13, 15)
    _DTU3D_EYE_R  = (17, 19, 21, 23)
    _DTU3D_EYE_L  = (25, 27, 29, 31)
    _DTU3D_NOSE   = (36, 37, 43, 44)
    _DTU3D_MOUTH  = (40, 47, 48, 50, 52, 53, 54, 55, 56)
    _DTU3D_SEGMENTS = [
        # Brow (open curves)
        (1, 3), (3, 5), (5, 7),
        (9, 11), (11, 13), (13, 15),
        # Eye (closed loops)
        (17, 19), (19, 21), (21, 23), (23, 17),
        (25, 27), (27, 29), (29, 31), (31, 25),
        # Nose wings (two short separate curves)
        (36, 37), (43, 44),
        # Mouth ring (closed) + 40-50 nose-base→top-lip connector
        (47, 48), (48, 50), (50, 52), (52, 53),
        (53, 54), (54, 55), (55, 56), (56, 47),
        (40, 50),
    ]
    # Per-region groupings for region-mode plateau (centroid+max-radius ball).
    _REGION_GROUPS = [
        ('brow_R', _DTU3D_BROW_R),
        ('brow_L', _DTU3D_BROW_L),
        ('eye_R',  _DTU3D_EYE_R),
        ('eye_L',  _DTU3D_EYE_L),
        ('nose',   _DTU3D_NOSE),
        ('mouth',  _DTU3D_MOUTH),
    ]

    @staticmethod
    def _point_to_segments_min_dist(V, seg_starts, seg_ends):
        """Distance from each vertex to the closest line segment.

        V          : [N, 3]
        seg_starts : [S, 3]
        seg_ends   : [S, 3]
        Returns    : [N] min distance to any segment.
        """
        seg     = seg_ends - seg_starts                          # [S, 3]
        seg_len = (seg ** 2).sum(-1).clamp_min(1e-12)            # [S]
        diff    = V.unsqueeze(1) - seg_starts.unsqueeze(0)       # [N, S, 3]
        t       = (diff * seg.unsqueeze(0)).sum(-1) / seg_len.unsqueeze(0)
        t       = t.clamp(0.0, 1.0)
        foot    = seg_starts.unsqueeze(0) + t.unsqueeze(-1) * seg.unsqueeze(0)
        return (V.unsqueeze(1) - foot).norm(dim=-1).min(dim=-1).values

    @staticmethod
    def _fps(V, K, seed=None):
        """Farthest Point Sampling [PointNet++, Qi 2017].

        V    : [N, 3]
        K    : int (number of samples)
        seed : optional int (used only by the manual fallback path)
        Returns: [K] long indices.

        Prefers PyTorch3D's CUDA kernel (sample_farthest_points) when GPU build
        is available — orders of magnitude faster than the Python loop fallback
        (no per-iteration GPU→CPU sync from .item()).
        """
        try:
            from pytorch3d.ops import sample_farthest_points
            # pytorch3d returns (selected_points, selected_indices) — take the
            # SECOND value as indices (taking the first was a latent bug:
            # selected_points has shape [1, K, 3], not [1, K]).
            _, idx = sample_farthest_points(V.unsqueeze(0), K=K)        # [1, K]
            return idx.squeeze(0).long()
        except Exception:
            pass
        # Fallback: pure-PyTorch manual FPS (slow due to .item() sync per step).
        N = V.shape[0]
        device = V.device
        indices = torch.zeros(K, dtype=torch.long, device=device)
        distances = torch.full((N,), float('inf'), device=device)
        if seed is not None:
            g = torch.Generator(device=device).manual_seed(seed)
            farthest = int(torch.randint(0, N, (1,), generator=g, device=device).item())
        else:
            farthest = int(torch.randint(0, N, (1,), device=device).item())
        for i in range(K):
            indices[i] = farthest
            dist = ((V - V[farthest]) ** 2).sum(dim=-1)
            distances = torch.minimum(distances, dist)
            farthest = int(distances.argmax().item())
        return indices

    def _build_bump_strict(self, V_b, lm_vidx, N, r0, r1, kind, region_falloff):
        """Compute per-vertex importance bump in [0, 1] for STRICT mode.
        Outside contour: bump=0 (zero weight → excluded from multinomial).

        Returns: bump [N] or None if landmarks unavailable.
        """
        from utils.exp_utils import plateau_hat_r, plateau_hat_points
        if lm_vidx is None:
            return None
        # lm_to_vidx depends only on (lm_vidx tensor, N) — both fixed per topology.
        # Cache it: avoids ~73 per-element .item() GPU syncs on every call.
        cache_key = (id(lm_vidx), int(N))
        lm_to_vidx = self._lm_to_vidx_cache.get(cache_key)
        if lm_to_vidx is None:
            lm_to_vidx = {}
            lm_vidx_list = lm_vidx.cpu().tolist()        # single sync
            for lm in (self._DTU3D_BROW_R + self._DTU3D_BROW_L
                       + self._DTU3D_EYE_R + self._DTU3D_EYE_L
                       + self._DTU3D_NOSE  + self._DTU3D_MOUTH):
                idx0 = lm - 1
                if 0 <= idx0 < len(lm_vidx_list):
                    v_idx = int(lm_vidx_list[idx0])
                    if 0 <= v_idx < N:
                        lm_to_vidx[lm] = v_idx
            self._lm_to_vidx_cache[cache_key] = lm_to_vidx
        if not lm_to_vidx:
            return None
        if kind == 'point':
            C = torch.stack([V_b[v] for v in lm_to_vidx.values()], dim=0)
            return plateau_hat_points(V_b, C, r0=r0, r1=r1).max(dim=-1).values
        elif kind == 'region':
            region_bumps = []
            for _, lm_ids in self._REGION_GROUPS:
                valid_vidx = [lm_to_vidx[lm] for lm in lm_ids if lm in lm_to_vidx]
                if len(valid_vidx) < 2:
                    continue
                pts = V_b[torch.tensor(valid_vidx, dtype=torch.long, device=V_b.device)]
                centroid = pts.mean(dim=0)
                R = (pts - centroid).norm(dim=-1).max().item()
                dist_r = (V_b - centroid).norm(dim=-1)
                region_bumps.append(plateau_hat_r(dist_r, r0=R, r1=R + region_falloff))
            if not region_bumps:
                return None
            return torch.stack(region_bumps, dim=-1).max(dim=-1).values
        else:  # 'segment'
            seg_pairs = [(a, b_) for (a, b_) in self._DTU3D_SEGMENTS
                         if a in lm_to_vidx and b_ in lm_to_vidx]
            if not seg_pairs:
                return None
            seg_starts = torch.stack([V_b[lm_to_vidx[a]]  for a, _ in seg_pairs], dim=0)
            seg_ends   = torch.stack([V_b[lm_to_vidx[b_]] for _, b_ in seg_pairs], dim=0)
            dist = self._point_to_segments_min_dist(V_b, seg_starts, seg_ends)
            return plateau_hat_r(dist, r0=r0, r1=r1)

    def _build_subsample_perm(self, src_v, batch, ratio, mode):
        """Build [B, K] long perm_idx for vertex subsampling.

        Modes:
          - 'random'             : uniform random.
          - 'fps'                : farthest point sampling (uniform spatial spread).
          - 'importance'         : weighted by w = 1 + α·bump (soft, allows non-contour).
          - 'importance_strict'  : weight = bump (zero outside contour; multinomial
                                   only picks from contour-area vertices).
          - 'mixed'              : per-sample alternation (even=random, odd=importance).
          - 'mix3'               : per-batch 1/3 of full/random/importance(α=99).
          - 'mix4'               : per-batch 1/4 of full/fps/random/importance_strict.
                                   For each non-full mode, ratio is sampled per batch
                                   from a uniform range:
                                     fps/random       : U[ratio_min, ratio_max]
                                     importance_strict: U[ratio_min, n_contour/N]
                                   (max for importance auto-capped by contour vertex
                                   count to avoid CAPPED sampling.)
        """
        B, N, _ = src_v.shape
        opts = self.opts
        r0    = getattr(opts, 'subsample_contour_r0',    0.1)
        r1    = getattr(opts, 'subsample_contour_r1',    0.3)
        alpha = getattr(opts, 'subsample_contour_alpha', 4.0)
        kind  = getattr(opts, 'subsample_contour_kind', 'segment')
        region_falloff = getattr(opts, 'subsample_region_falloff', 0.1)
        ratio_min = getattr(opts, 'subsample_ratio_min', 0.1)
        ratio_max = getattr(opts, 'subsample_ratio_max', 0.5)

        # ── mix3 (legacy 3-way): full / random / importance(soft, high α) ──
        if mode == 'mix3':
            roll = int(torch.randint(0, 3, (1,)).item())
            if roll == 0:
                return None
            mode  = 'random' if roll == 1 else 'importance'
            if mode == 'importance':
                alpha = getattr(opts, 'subsample_mix3_alpha', 99.0)

        # ── mix4: full / fps / random / importance_strict, per-batch dynamic ratio ──
        if mode == 'mix4':
            roll = int(torch.randint(0, 4, (1,)).item())
            if roll == 0:
                return None                                              # full mesh
            mode = ['fps', 'random', 'importance_strict'][roll - 1]
            if mode == 'importance_strict':
                # Need to know n_contour for THIS batch (first sample's topology).
                idn0 = batch.id_name[0] if hasattr(batch, 'id_name') else ''
                topo0 = ('ict' if idn0.startswith('ict_') else
                         'mf'  if idn0.startswith('m--')   else None)
                lm_vidx_0 = self._landmark_vidx_per_topo.get(topo0) if topo0 else None
                bump0 = self._build_bump_strict(src_v[0], lm_vidx_0, N,
                                                r0, r1, kind, region_falloff)
                if bump0 is None:
                    mode = 'random'
                    ratio = float(np.random.uniform(ratio_min, ratio_max))
                else:
                    n_contour = int((bump0 > 1e-8).sum().item())
                    imp_max = max(ratio_min, min(ratio_max, n_contour / N))
                    ratio = float(np.random.uniform(ratio_min, imp_max))
            else:
                ratio = float(np.random.uniform(ratio_min, ratio_max))

        K = int(round(N * ratio))
        if K >= N or K <= 0:
            return None

        out = []
        for b in range(B):
            idn = batch.id_name[b] if hasattr(batch, 'id_name') else ''
            topo = ('ict' if idn.startswith('ict_') else
                    'mf'  if idn.startswith('m--')   else None)
            lm_vidx = self._landmark_vidx_per_topo.get(topo) if topo else None
            V_b = src_v[b]                                                # [N, 3]

            _eff_mode = mode
            if mode == 'mixed':
                _eff_mode = 'random' if (b % 2 == 0) else 'importance'

            if _eff_mode == 'fps':
                # Random first-point seed each call → stochastic across epochs.
                perm = self._fps(V_b, K)

            elif _eff_mode == 'importance_strict':
                bump = self._build_bump_strict(V_b, lm_vidx, N,
                                               r0, r1, kind, region_falloff)
                if bump is None:
                    perm = torch.randperm(N, device=src_v.device)[:K]
                else:
                    # weight = bump; vertices with bump=0 cannot be picked.
                    # Cap K if contour vertex count is smaller.
                    n_contour = int((bump > 1e-8).sum().item())
                    K_eff = min(K, n_contour)
                    perm = torch.multinomial(bump + 1e-12, K_eff, replacement=False)
                    if K_eff < K:
                        # Pad with random non-contour to keep [B, K] tensor shape consistent.
                        mask = torch.ones(N, dtype=torch.bool, device=src_v.device)
                        mask[perm] = False
                        remain = mask.nonzero(as_tuple=False).squeeze(-1)
                        extra = remain[torch.randperm(len(remain), device=src_v.device)[:K - K_eff]]
                        perm = torch.cat([perm, extra])

            elif _eff_mode == 'importance':
                # Soft importance: w = 1 + α·bump (existing behavior).
                bump = self._build_bump_strict(V_b, lm_vidx, N,
                                               r0, r1, kind, region_falloff)
                if bump is None:
                    perm = torch.randperm(N, device=src_v.device)[:K]
                else:
                    w = 1.0 + alpha * bump
                    perm = torch.multinomial(w, K, replacement=False)

            else:  # 'random'
                perm = torch.randperm(N, device=src_v.device)[:K]

            out.append(perm)
        return torch.stack(out, dim=0)                                    # [B, K]

    @torch.no_grad()
    def _visualize_curriculum_bases(self, epoch, save_dir):
        """Visualize all 53 ICT expression bases: GT vs pred for each basis."""
        from utils.keys import ICT_KEYS
        from utils.remesh_utils import ICT_face_model
        import igl

        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()

        ict_model = ICT_face_model()
        # Use first identity
        iden_vecs = np.load('ict_face_pt/random_identity_vecs.npy')
        id_coeff = iden_vecs[0]

        v_gt_list, v_pred_list = [], []
        for basis_idx in range(53):
            exp_coeff = np.zeros(53)
            exp_coeff[basis_idx] = 1.0

            v_num, faces_np = ict_model.get_random_v_and_f(select=0)  # fullhead
            deformed, template, _ = ict_model.apply_coeffs(id_coeff, exp_coeff, return_all=True, region=0)
            deformed = deformed[0]
            template = template[0]

            template_t = torch.tensor(template).float().unsqueeze(0).to(self.device)
            deformed_t = torch.tensor(deformed).float().unsqueeze(0).to(self.device)
            template_n = torch.tensor(igl.per_vertex_normals(template, faces_np)).float().unsqueeze(0).to(self.device)
            deformed_n = torch.tensor(igl.per_vertex_normals(deformed, faces_np)).float().unsqueeze(0).to(self.device)

            delta = deformed_t - template_t
            src_in = torch.cat([template_t, template_n], dim=-1)
            deform_in = torch.cat([delta, deformed_n, src_in], dim=-1)

            # Get NFS feat for visualization if available
            _vis_nfs_feat = None
            if hasattr(self, '_nfs_feat_cache') and self._nfs_feat_cache:
                _key = f"ict_{0:03d}"  # first identity
                if _key in self._nfs_feat_cache:
                    _f = self._nfs_feat_cache[_key]
                    N_cur = template_t.shape[1]
                    if _f.shape[0] > N_cur:
                        _f = _f[:N_cur]
                    _vis_nfs_feat = _f.unsqueeze(0)

            pred = self.model(template_t, deform_in, source_normal=template_n, nfs_feat=_vis_nfs_feat)

            faces_cpu = torch.tensor(faces_np).long()
            v_gt_list.append(deformed_t[0].cpu())
            v_pred_list.append(pred[0].cpu())

        # Plot in groups of 8 (GT row + pred row)
        for start in range(0, 53, 8):
            end = min(start + 8, 53)
            v_list = [v_gt_list[i] for i in range(start, end)]
            v_list += [v_pred_list[i] for i in range(start, end)]
            f_list = [faces_cpu] * len(v_list)
            names = [ICT_KEYS[i] for i in range(start, end)]
            plot_image_array(
                v_list, f_list, rot_list=[[0, 0, 0]] * len(v_list),
                size=1, bg_black=False, mode='shade',
                logdir=save_dir,
                name=f"{epoch:03d}_bases_{start:02d}-{end-1:02d}", save=True)

        self.model.train()

    def train(self, epochs):
        opts = self.opts
        BS = opts.batch_size

        # Collect trainable params: HLBS model + NFS adapter (if any)
        train_params = list(self.model.parameters())
        if self.nfs_z_adapter is not None and not isinstance(self.nfs_z_adapter, torch.nn.Identity):
            train_params += list(self.nfs_z_adapter.parameters())
        self.optimizer = torch.optim.AdamW(
            train_params, lr=opts.lr, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opts.sc_step, gamma=opts.sc_gamma)

        # When target is GT, smooth data is not needed — override to skip loading
        if opts.target == 'gt':
            opts.smooth_n_iter = 0
        
        train_ds = CBDDataset(opts, is_train=True, toggle=opts.data_toggle, data_basedir=opts.data_basedir)
        valid_ds = CBDDataset(opts, is_valid=True, toggle=opts.data_toggle, data_basedir=opts.data_basedir)
        _region_min = 1 if opts.no_fullhead else 0
        train_sampler = CBDdataSampler(train_ds.len_list, BS, shuffle=True,  balance=False, is_train=True, region_min=_region_min)
        valid_sampler = CBDdataSampler(valid_ds.len_list, BS, shuffle=True,  balance=False, is_valid=True, region_min=_region_min)
        _nw = opts.num_workers
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_sampler=train_sampler,
            collate_fn=partial(CBD_collate_wrapper, device='cpu'),
            num_workers=_nw, persistent_workers=(_nw > 0),
            pin_memory=torch.cuda.is_available())
        valid_loader = torch.utils.data.DataLoader(
            valid_ds, batch_sampler=valid_sampler,
            collate_fn=partial(CBD_collate_wrapper, device='cpu'), num_workers=0)

        # logging
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        resume_mode = opts.ckpt and opts.continue_ckpt
        if resume_mode and os.path.isdir(opts.ckpt):
            opts.log_dir = opts.ckpt
        else:
            freeze_tag = "-frozenAdapt" if opts.freeze_adapt else ""
            trans_tag = "-jTrans" if opts.use_joint_trans else ""
            sdw_tag = f"-sdw{opts.smooth_delta_W}a{opts.smooth_delta_W_alpha}" if opts.smooth_delta_W > 0 else ""
            nfs_tag = "-nfsEnc" if opts.nfs_ckpt else ""
            surf_tag = ""
            if opts.lambda_normal > 0: surf_tag += f"-nrm{opts.lambda_normal}"
            if opts.lambda_curvature > 0: surf_tag += f"-crv{opts.lambda_curvature}"
            wsm_tag = f"-Wsm{opts.lambda_W_smooth}" if opts.lambda_W_smooth > 0 else ""
            cur_tag = f"-cur{opts.curriculum_epochs}" if opts.curriculum else ""
            tag = f"-HLBS-{opts.topo_key}-s{opts.smooth_n_iter}{freeze_tag}{trans_tag}{sdw_tag}{nfs_tag}{surf_tag}{wsm_tag}{cur_tag}"
            opts.log_dir = os.path.join(opts.log_dir, now + tag)

        os.makedirs(opts.log_dir, exist_ok=True)
        os.makedirs(f"{opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{opts.log_dir}/img/train/cross_retarget", exist_ok=True)
        os.makedirs(f"{opts.log_dir}/img/valid/mesh", exist_ok=True)

        with open(os.path.join(opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(opts), f, indent=4)
        with open(os.path.join(opts.log_dir, "train_opts.yml"), 'w') as f:
            yaml.dump(vars(opts), f, sort_keys=False)

        writer_train = writer_valid = None
        if opts.tb:
            writer_train = SummaryWriter(log_dir=os.path.join(opts.log_dir, "train"))
            writer_valid = SummaryWriter(log_dir=os.path.join(opts.log_dir, "valid"))

        logger = Logger(os.path.join(opts.log_dir, "log.txt"))
        print(f'Log: {logger.file_path}')

        if opts.target == 'smooth_gt':
            tgt_desc = f"smooth_GT (Taubin n_iter={opts.smooth_n_iter})"
            smooth_line = f"  smooth_n_iter : {opts.smooth_n_iter}\n"
        else:
            tgt_desc = "GT (full target, no smoothing)"
            smooth_line = ""
        config_text = (
            f"=== HLBS Training ===\n"
            f"  topo_key      : {opts.topo_key}\n"
            f"  num_identities: {opts.num_identities}\n"
            f"  target        : {opts.target}  ({tgt_desc})\n"
            f"{smooth_line}"
            f"  lambda_W_reg  : {opts.lambda_W_reg}\n"
            f"  lambda_t_reg  : {opts.lambda_t_reg}\n"
            f"  lambda_neu    : {opts.lambda_neu}\n"
            f"  lambda_W_smooth: {opts.lambda_W_smooth}\n"
            f"  freeze_adapt  : {opts.freeze_adapt}\n"
            f"  use_joint_trans: {opts.use_joint_trans}\n"
            f"=====================\n"
        )
        print(config_text)
        logger.write(config_text)
        logger.write(train_ds.get_data_config())

        loss_lambda = {
            "recon-lbs":   opts.lambda_vert,
            "recon-neu":   opts.lambda_neu,
            "lbs-W-reg":   opts.lambda_W_reg,
            "lbs-t-reg":   opts.lambda_t_reg,
            "lbs-W-smooth": opts.lambda_W_smooth,
            "recon-normal": opts.lambda_normal,
            "recon-curvature": opts.lambda_curvature,
        }

        BEST_LOSS  = 1e8
        BEST_EPOCH = 0
        len_train  = len(train_loader)
        len_valid  = len(valid_loader)
        interv     = max(1, round(len_train / 10))
        interv_val = max(1, round(len_valid / 3))

        # Precompute mesh edges for delta_W forward smoothing (if enabled)
        if opts.smooth_delta_W > 0:
            print(f"[HLBS] delta_W forward smoothing: iters={opts.smooth_delta_W}, alpha={opts.smooth_delta_W_alpha}")

        # ── Curriculum: Phase 1 loader (ICT single-basis only) ──────────
        curriculum_transitioned = False
        if opts.curriculum:
            # Filter len_list to ICT only (mesh_data == 5)
            ict_len_list = [x for x in train_ds.len_list if x[2].item() == 5]
            # Override expression count: 53 bases × identities
            n_ict_ids = ict_len_list[0][3]
            pad = 53 % BS
            ict_len_list[0] = [53 + (BS - pad if pad else 0), ict_len_list[0][1], ict_len_list[0][2], n_ict_ids]
            train_ds.curriculum_single_basis = True
            cur_sampler = CBDdataSampler(ict_len_list, BS, shuffle=True, balance=False, is_train=True, region_min=_region_min)
            cur_loader = torch.utils.data.DataLoader(
                train_ds, batch_sampler=cur_sampler,
                collate_fn=partial(CBD_collate_wrapper, device='cpu'),
                num_workers=_nw, persistent_workers=(_nw > 0),
            pin_memory=torch.cuda.is_available())
            print(f"[Curriculum] Phase 1: ICT single-basis for {opts.curriculum_epochs} epochs "
                  f"({len(cur_loader)} batches/epoch, {n_ict_ids} identities × 53 bases)")

        for epoch in range(opts.start_epoch, epochs + 1):
            # ── Curriculum phase transition ──────────────────────────────
            if opts.curriculum and not curriculum_transitioned:
                if epoch < opts.curriculum_epochs:
                    active_loader = cur_loader
                else:
                    # Switch to full data
                    train_ds.curriculum_single_basis = False
                    active_loader = train_loader
                    if epoch == opts.curriculum_epochs:
                        curriculum_transitioned = True
                        print(f"[Curriculum] Phase 2: switching to full data at epoch {epoch}")
            else:
                active_loader = train_loader
            # ── Train ────────────────────────────────────────────────────────
            self.model.train()
            running = {"recon-lbs": 0.0, "recon-neu": 0.0, "recon-normal": 0.0, "recon-curvature": 0.0, "lbs-W-reg": 0.0, "lbs-t-reg": 0.0, "lbs-W-smooth": 0.0, "total": 0.0}
            cnt = 0

            _len_active = len(active_loader)
            pbar = tqdm(enumerate(active_loader), total=_len_active, ncols=120,
                        desc=f"[{epoch:03d}] Train HLBS")
            for idx, batch in pbar:
                batch = batch.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()

                src_v  = batch.template
                src_n  = batch.template_normal
                gt_v   = batch.vertices
                gt_n   = batch.vertices_normal
                smt_v  = batch.smooth_vertices

                # Register mesh edges for this topology if not cached
                if opts.smooth_delta_W > 0:
                    N_cur = src_v.shape[1]
                    if self.model._mesh_edges_by_N is None or N_cur not in self.model._mesh_edges_by_N:
                        _faces = batch.faces[0] if batch.faces.dim() == 3 else batch.faces
                        self.model.set_mesh_edges(_faces)

                target_v = smt_v if opts.target == 'smooth_gt' else gt_v

                delta     = gt_v - src_v
                src_in    = torch.cat([src_v, src_n], dim=-1)        # [B, N, 6]
                deform_in = torch.cat([delta, gt_n, src_in], dim=-1) # [B, N, 12]

                # z_exp from NFS encoder or HLBS internal encoder
                z_exp_ext = None
                if self.nfs_encoder is not None:
                    z_exp_ext = self._nfs_encode_exp(gt_v, gt_n)      # [B, hid_dim]

                pred_lbs = self.model(src_v, deform_in, source_normal=src_n, z_exp_override=z_exp_ext)  # [B, N, 3]

                if opts.no_t_mask:
                    loss_dict = {"recon-lbs": F.mse_loss(target_v, pred_lbs)}
                else:
                    from utils.exp_utils import plateau_hat_points
                    t_mask   = plateau_hat_points(src_v)
                    inv_mask = 1.0 - t_mask
                    loss_dict = {
                        "recon-lbs": (
                            F.mse_loss(target_v * t_mask,  pred_lbs * t_mask)
                            + F.mse_loss(src_v * inv_mask, pred_lbs * inv_mask)
                        )
                    }

                # ── Neutral reconstruction loss ───────────────────────────
                # When delta=0 (no deformation), pred_lbs should equal src_v
                if opts.lambda_neu > 0:
                    delta_zero   = torch.zeros_like(src_v)
                    neu_deform_in = torch.cat([delta_zero, src_n, src_v, src_n], dim=-1)
                    pred_neutral = self.model(src_v, neu_deform_in, source_normal=src_n, nfs_feat=_nfs_feat if hasattr(self, '_nfs_feat_cache') else None)
                    if opts.no_t_mask:
                        loss_dict["recon-neu"] = F.mse_loss(src_v, pred_neutral)
                    else:
                        loss_dict["recon-neu"] = (
                            F.mse_loss(src_v * t_mask,    pred_neutral * t_mask)
                            + F.mse_loss(src_v * inv_mask, pred_neutral * inv_mask)
                        )

                # ── Normal consistency loss ───────────────────────────────
                if opts.lambda_normal > 0:
                    from utils.mesh_utils import calc_norm_torch
                    pred_n = calc_norm_torch(pred_lbs, batch.faces, at='verts')
                    gt_n_recomputed = calc_norm_torch(target_v, batch.faces, at='verts')
                    normal_diff = 1 - F.cosine_similarity(pred_n, gt_n_recomputed, dim=-1)  # [B, V]
                    if not opts.no_t_mask:
                        normal_diff = normal_diff * t_mask.squeeze(-1)
                    loss_dict["recon-normal"] = normal_diff.mean()

                # ── Curvature loss (Laplacian difference) ────────────────────
                if opts.lambda_curvature > 0:
                    from train_edd_real import _uniform_laplacian
                    loss_dict["recon-curvature"] = F.mse_loss(
                        _uniform_laplacian(pred_lbs, batch.faces),
                        _uniform_laplacian(target_v, batch.faces))

                # ── Regularization + smoothness ───────────────────────────
                # Lazily build mesh edges per topology for smoothness loss
                edges = None
                if opts.lambda_W_smooth > 0:
                    N_cur = src_v.shape[1]
                    if not hasattr(self, '_edges_by_N'):
                        self._edges_by_N = {}
                    if N_cur not in self._edges_by_N:
                        f_raw = batch.faces.cpu()
                        f_np = f_raw[0].numpy() if f_raw.dim() == 3 else f_raw.numpy()
                        e = np.concatenate([f_np[:, [0,1]], f_np[:, [1,2]], f_np[:, [0,2]]], axis=0)
                        e = np.sort(e, axis=1)
                        e = np.unique(e, axis=0)
                        self._edges_by_N[N_cur] = torch.tensor(e, dtype=torch.long, device=self.device)
                    edges = self._edges_by_N[N_cur]
                src_feat = torch.cat([src_v, src_n], dim=-1)
                regs = self.model.reg_loss(src_feat, edges=edges)
                loss_dict["lbs-W-reg"]    = regs["L_W_reg"]
                loss_dict["lbs-t-reg"]    = regs["L_t_reg"]
                if opts.lambda_W_smooth > 0:
                    loss_dict["lbs-W-smooth"] = regs["L_W_smooth"]

                loss = sum(loss_dict[k] * loss_lambda[k] for k in loss_dict)
                loss.backward()
                self.optimizer.step()

                for k in running:
                    if k != "total" and k in loss_dict:
                        running[k] += (loss_dict[k] * loss_lambda[k]).item()
                running["total"] += loss.item()
                cnt += 1
                pbar.set_description(f"[{epoch:03d}] lbs: {loss_dict['recon-lbs']:.5e}")

                _interv = max(1, round(_len_active / 10))
                if idx % _interv == 1:
                    inv = 1.0 / cnt
                    log_text = f"[{epoch:03d}/{epochs:03d}][{idx:04d}][Train] "
                    log_text += " ".join(f"{k}: {v*inv:.6e}" for k, v in running.items())
                    logger.write(log_text + "\n")

                    HB = BS // 2
                    _s = lambda i: min(i, BS-1)
                    _d = lambda t: t.detach().float().cpu()  # .float(): bf16(AMP)→fp32 for numpy/vis
                    faces_cpu = batch.faces.cpu()
                    v_list = [
                        _d(gt_v[0]),        _d(gt_v[_s(1)]),
                        _d(gt_v[_s(HB)]),   _d(gt_v[BS-1]),
                    ]
                    if opts.target == 'smooth_gt':
                        v_list += [
                            _d(smt_v[0]),       _d(smt_v[_s(1)]),
                            _d(smt_v[_s(HB)]),  _d(smt_v[BS-1]),
                        ]
                    v_list += [
                        _d(pred_lbs[0]),    _d(pred_lbs[_s(1)]),
                        _d(pred_lbs[_s(HB)]), _d(pred_lbs[BS-1]),
                    ]
                    f_list = [faces_cpu] * len(v_list)
                    plot_image_array(
                        v_list, f_list, rot_list=[[0,0,0]]*len(v_list),
                        size=1, bg_black=False, mode='shade',
                        logdir=f"{opts.log_dir}/img/train/mesh",
                        name=f"{epoch:03d}_{idx:04d}", save=True)

                if opts.debug:
                    break

            if epoch != 0:
                self.scheduler.step()
            if writer_train:
                for k, v in running.items():
                    writer_train.add_scalar(k, v / cnt, epoch)

            if epoch % opts.save_interval == 0:
                torch.save(self.model.state_dict(),
                           f'{opts.log_dir}/model_hlbs_{epoch:03d}.pth')

            if epoch % opts.eval_iter == 0:
                self.vis_loader.visualize(
                    self.model, None, epoch,
                    save_dir=f'{opts.log_dir}/img/eval',
                    mode='hlbs',
                    smooth_n_iter=opts.smooth_n_iter,
                    no_t_mask=opts.no_t_mask,
                    nfs_feat_cache=self._nfs_feat_cache if hasattr(self, '_nfs_feat_cache') else None,
                )
                if opts.curriculum and epoch > 0:
                    self._visualize_curriculum_bases(
                        epoch, save_dir=f'{opts.log_dir}/img/eval_bases')

            # ── Valid ────────────────────────────────────────────────────────
            if epoch == 0 or epoch % opts.val_every != 0:
                continue

            self.model.eval()
            running_val = {"recon-lbs": 0.0, "recon-neu": 0.0, "recon-normal": 0.0, "recon-curvature": 0.0, "total": 0.0}
            vcnt = 0

            pbar = tqdm(enumerate(valid_loader), total=len_valid, ncols=120,
                        desc=f"[{epoch:03d}] Valid HLBS")
            for idx, batch in pbar:
                batch = batch.to(self.device, non_blocking=True)
                vcnt += 1
                with torch.no_grad():
                    src_v  = batch.template
                    src_n  = batch.template_normal
                    gt_v   = batch.vertices
                    gt_n   = batch.vertices_normal
                    smt_v  = batch.smooth_vertices

                    delta     = gt_v - src_v
                    src_in    = torch.cat([src_v, src_n], dim=-1)
                    deform_in = torch.cat([delta, gt_n, src_in], dim=-1)

                    z_exp_ext = None
                    if self.nfs_encoder is not None:
                        z_exp_ext = self._nfs_encode_exp(gt_v, gt_n)

                    pred_lbs  = self.model(src_v, deform_in, source_normal=src_n, z_exp_override=z_exp_ext)

                    target_v_val = smt_v if opts.target == 'smooth_gt' else gt_v
                    val_loss = F.mse_loss(target_v_val, pred_lbs).item() * loss_lambda["recon-lbs"]
                    running_val["recon-lbs"] += val_loss
                    running_val["total"]     += val_loss

                    # Neutral reconstruction loss (val)
                    if opts.lambda_neu > 0:
                        delta_zero    = torch.zeros_like(src_v)
                        neu_deform_in = torch.cat([delta_zero, src_n, src_v, src_n], dim=-1)
                        pred_neutral  = self.model(src_v, neu_deform_in, source_normal=src_n)
                        val_neu = F.mse_loss(src_v, pred_neutral).item() * loss_lambda["recon-neu"]
                        running_val["recon-neu"] += val_neu
                        running_val["total"]     += val_neu

                    # Normal consistency loss (val)
                    if opts.lambda_normal > 0:
                        from utils.mesh_utils import calc_norm_torch
                        pred_n = calc_norm_torch(pred_lbs, batch.faces, at='verts')
                        gt_n_val = calc_norm_torch(target_v_val, batch.faces, at='verts')
                        val_nrm = (1 - F.cosine_similarity(pred_n, gt_n_val, dim=-1)).mean().item() * loss_lambda["recon-normal"]
                        running_val["recon-normal"] += val_nrm
                        running_val["total"]        += val_nrm

                    # Curvature loss (val)
                    if opts.lambda_curvature > 0:
                        from train_edd_real import _uniform_laplacian
                        val_crv = F.mse_loss(
                            _uniform_laplacian(pred_lbs, batch.faces),
                            _uniform_laplacian(target_v_val, batch.faces)).item() * loss_lambda["recon-curvature"]
                        running_val["recon-curvature"] += val_crv
                        running_val["total"]           += val_crv

                pbar.set_description(f"[{epoch:03d}] val lbs: {val_loss:.5e}")

                if idx % interv_val == 0:
                    BS_v = batch.vertices.shape[0]
                    HB_v = BS_v // 2
                    _s = lambda i: min(i, BS_v-1)
                    faces_cpu = batch.faces.cpu()
                    v_list = [
                        gt_v[0].cpu(),        gt_v[_s(1)].cpu(),
                        gt_v[_s(HB_v)].cpu(), gt_v[BS_v-1].cpu(),
                    ]
                    if opts.target == 'smooth_gt':
                        v_list += [
                            smt_v[0].cpu(),        smt_v[_s(1)].cpu(),
                            smt_v[_s(HB_v)].cpu(), smt_v[BS_v-1].cpu(),
                        ]
                    v_list += [
                        pred_lbs[0].cpu(),        pred_lbs[_s(1)].cpu(),
                        pred_lbs[_s(HB_v)].cpu(), pred_lbs[BS_v-1].cpu(),
                    ]
                    f_list = [faces_cpu] * len(v_list)
                    plot_image_array(
                        v_list, f_list, rot_list=[[0,0,0]]*len(v_list),
                        size=1, bg_black=False, mode='shade',
                        logdir=f"{opts.log_dir}/img/valid/mesh",
                        name=f"{epoch:03d}_{idx:04d}", save=True)

                if opts.debug:
                    break

            if writer_valid:
                for k, v in running_val.items():
                    writer_valid.add_scalar(k, v / vcnt, epoch)

            total_val = running_val["total"] / vcnt
            # Per-component val line — always written so log.txt covers every val epoch
            parts = " ".join(f"{k}: {running_val[k]/vcnt:.6e}"
                             for k in ["recon-lbs", "recon-neu", "recon-normal", "recon-curvature"])
            val_line = (f"[{epoch:03d}] Val: {parts} total: {total_val:.6e} "
                        f"(Best: {min(total_val, BEST_LOSS):.6e} "
                        f"[{epoch if total_val < BEST_LOSS else BEST_EPOCH}])")
            print(val_line); logger.write(val_line + "\n")
            if total_val < BEST_LOSS:
                BEST_LOSS  = total_val
                BEST_EPOCH = epoch
                torch.save(self.model.state_dict(), f'{opts.log_dir}/model_hlbs_best.pth')
                print(f"[{epoch:03d}] Best updated: {BEST_LOSS:.6e}")
                logger.write(f"[{epoch:03d}] Best updated: {BEST_LOSS:.6e}\n")


    def train_full_prediction(self, epochs):
        """Train with HierarchicalLBS_FullPred: full W/bind prediction + Phase 1 annealing."""
        from models.hierarchical_lbs import HierarchicalLBS_FullPred
        opts = self.opts
        BS = opts.batch_size

        # ── Load active-joint config (Option A: face-mask mode) ──────────
        self._face_joint_idx = None
        self._base_joint_idx = None
        self._helper_joint_idx = None
        self._use_helpers = bool(getattr(opts, 'use_helpers', 1))
        if getattr(opts, 'active_joints_json', None):
            with open(opts.active_joints_json) as f:
                _aj = json.load(f)
            self._face_joint_idx = _aj['face_joint_idx']
            self._base_joint_idx = _aj['base_joint_idx']
            _helper_list = _aj.get('helper_joint_idx', [])
            # use_helpers=0 → don't apply helper-aware logic (helpers learned as regular joints).
            # bind_reg_exclude_extra: even with use_helpers=0 (no reparam/mirror),
            # keep the helper idx list at TRAINER level so L_bind_reg excludes
            # those joints from bind-pose GT supervision — their template
            # positions are untrusted parked slots; placement must be learned.
            # (model ctor stays gated by _use_helpers → no HelperReparam.)
            self._helper_joint_idx = (
                list(_helper_list)
                if (self._use_helpers or getattr(opts, 'bind_reg_exclude_extra', False))
                else [])
            print(f"[Option A] Loaded {len(self._face_joint_idx)} face joints, "
                  f"base={_aj['base_joint_name']} (idx={self._base_joint_idx}), "
                  f"helpers={len(_helper_list)} "
                  f"({'use_helpers=ON' if self._use_helpers else 'use_helpers=OFF — treated as regular face joints'}) "
                  f"from {opts.active_joints_json}")
            if getattr(opts, 'no_face_mask', False):
                # Ablation: disable face mask -> softmax over ALL joints, no
                # non-face->base routing (use_face_mask becomes False in model).
                self._face_joint_idx = None
                print("[face-mask] DISABLED via --no_face_mask")

        # ── Load helper mirror pairs (for L_mirror loss) ─────────────────
        # Reads name pairs from helper_joints_*.json; resolves to joint indices
        # via rig.joint_names (loaded later via rig). Stored as [n_pairs, 2] long.
        self._mirror_pair_idx = None
        if (self._use_helpers and getattr(opts, 'lambda_mirror', 0) > 0
                and getattr(opts, 'helper_joints_json', None)
                and os.path.exists(opts.helper_joints_json)):
            with open(opts.helper_joints_json) as f:
                _hj = json.load(f)
            self._mirror_pair_names = _hj.get('mirror_pairs', [])
            # rig.joint_names is available after rig loading below; resolve there.
            print(f"[L_mirror] {len(self._mirror_pair_names)} mirror pairs declared "
                  f"in {opts.helper_joints_json}")
        else:
            self._mirror_pair_names = []

        # ── Load σ targets (per-joint init + shrinkage target) ───────────
        self._sigma_targets = None
        if getattr(opts, 'sigma_targets_npy', None):
            self._sigma_targets = np.load(opts.sigma_targets_npy).astype(np.float32)
            print(f"[σ targets] Loaded shape={self._sigma_targets.shape}, "
                  f"range=[{self._sigma_targets.min():.3f}, {self._sigma_targets.max():.3f}] "
                  f"from {opts.sigma_targets_npy}")

        # ── Load anchor-pool bind-pose assets ────────────────────────────
        self._joint_anchors = None
        self._joint_offsets = None
        if getattr(opts, 'bind_pose_mode', 'net') == 'anchor_pool':
            assert opts.joint_anchors_npy and opts.joint_offsets_npy, \
                'bind_pose_mode=anchor_pool requires --joint_anchors_npy and --joint_offsets_npy'
            self._joint_anchors = np.load(opts.joint_anchors_npy).astype(np.float32)
            self._joint_offsets = np.load(opts.joint_offsets_npy).astype(np.float32)
            print(f"[anchor_pool] anchors={self._joint_anchors.shape} "
                  f"offsets={self._joint_offsets.shape} (||δ|| range "
                  f"[{np.linalg.norm(self._joint_offsets, axis=-1).min():.4f}, "
                  f"{np.linalg.norm(self._joint_offsets, axis=-1).max():.4f}]) "
                  f"from {opts.joint_anchors_npy}")

        # ── Per-id bind_pos cache (anchor_pool default: nfs_feat_dir) ────
        self._bind_pos_cache = {}
        _cache_dir = getattr(opts, 'bind_pos_cache_dir', None)
        if _cache_dir is None and getattr(opts, 'bind_pose_mode', 'net') == 'anchor_pool':
            _cache_dir = getattr(opts, 'nfs_feat_dir', None)
        if _cache_dir:
            import glob as _glob
            files = _glob.glob(os.path.join(_cache_dir, '*_bind_pos.npy'))
            for fp in files:
                k = os.path.basename(fp).replace('_bind_pos.npy', '')
                self._bind_pos_cache[k] = torch.tensor(np.load(fp), dtype=torch.float32).to(self.device)
            if files:
                print(f"[bind_pos cache] Loaded {len(self._bind_pos_cache)} per-id "
                      f"from {_cache_dir}")
            elif getattr(opts, 'bind_pose_mode', 'net') == 'anchor_pool':
                print(f"[bind_pos cache] WARNING: no *_bind_pos.npy in {_cache_dir}. "
                      f"Run precompute_per_id_bind_pos.py first, or expect online (slower) computation.")

        # ── Per-id bind_pose GT cache (Phase B: landmark-based supervision) ──
        self._per_id_bind_pose_gt = {}    # id_name → [J, 3] tensor on self.device
        if getattr(opts, 'per_id_bind_pose_dir', None):
            import glob as _glob
            _bp_dirs = [opts.per_id_bind_pose_dir]
            # Also load aug bind pose GT from the caricat dir; filenames there
            # are {id}_aug_bind_pos_landmark.npy → keys auto-suffix '_aug',
            # matching the id_name the dataloader emits for caricat samples.
            if (getattr(opts, 'caricat_aug_dir', '')
                    and getattr(opts, 'caricat_prob', 0) > 0
                    and os.path.isdir(opts.caricat_aug_dir)):
                _bp_dirs.append(opts.caricat_aug_dir)
            for _bp_dir in _bp_dirs:
                for _fp in _glob.glob(os.path.join(_bp_dir, '*_bind_pos_landmark.npy')):
                    _k = os.path.basename(_fp).replace('_bind_pos_landmark.npy', '')
                    self._per_id_bind_pose_gt[_k] = torch.from_numpy(
                        np.load(_fp)).to(self.device).float()
            if self._per_id_bind_pose_gt:
                _n_aug = sum(1 for k in self._per_id_bind_pose_gt if k.endswith('_aug'))
                print(f"[per-id bind_pose GT] Loaded {len(self._per_id_bind_pose_gt)} "
                      f"({_n_aug} aug) from {_bp_dirs}")
            else:
                print(f"[per-id bind_pose GT] WARNING: no *_bind_pos_landmark.npy in "
                      f"{_bp_dirs}. L_bind_reg falls back to per-topo mean.")

        # ── Per-topo geodesic dist tables (for GMM Gauss / L_net_center / L_dist) ──
        self._geo_dist_per_topo = {}    # topo → [J, V] tensor on self.device
        if getattr(opts, 'use_geodesic_gauss', False):
            _gd_dir = getattr(opts, 'geo_dist_dir', None) or opts.rig_path
            for _topo in ('ict', 'mf', 'biwi', 'coma'):
                _p = os.path.join(_gd_dir, f'geo_dist_{_topo}.npy')
                if os.path.exists(_p):
                    self._geo_dist_per_topo[_topo] = torch.from_numpy(
                        np.load(_p)).to(self.device).float()                # [J, V]
            if self._geo_dist_per_topo:
                _msg = ', '.join(f'{k}{tuple(v.shape)}'
                                 for k, v in self._geo_dist_per_topo.items())
                print(f"[geo_dist] loaded: {_msg} from {_gd_dir}")
            else:
                print(f"[geo_dist] WARNING: --use_geodesic_gauss set but no "
                      f"geo_dist_{{topo}}.npy in {_gd_dir}. Falling back to Euclidean.")

        # ── Cross-pair ICT loader (Track B) ──
        self._cross_pair_loader = None
        self._cross_pair_iter = None
        if getattr(opts, 'lambda_cross_retarget', 0) > 0:
            try:
                from dataloader_cross_pair import CrossPairICTDataset, cross_pair_collate
                from utils.remesh_utils import ICT_face_model
                _ict_fm = ICT_face_model()
                _path = opts.cross_pair_iden_vecs
                if _path.endswith('.npy'):
                    _iden_vecs = np.load(_path)
                else:
                    _iden_vecs = torch.load(_path).numpy()
                _length = opts.cross_pair_length if opts.cross_pair_length > 0 \
                    else (_iden_vecs.shape[0] * 53)
                _cross_ds = CrossPairICTDataset(
                    _ict_fm, _iden_vecs, expression_vecs=None,
                    mode='train', length=_length,
                )
                _cross_nw = opts.num_workers
                self._cross_pair_loader = torch.utils.data.DataLoader(
                    _cross_ds, batch_size=opts.batch_size, num_workers=_cross_nw,
                    persistent_workers=(_cross_nw > 0),
                    pin_memory=torch.cuda.is_available(),
                    shuffle=True, collate_fn=partial(cross_pair_collate, device='cpu'),
                )
                self._cross_pair_iter = iter(self._cross_pair_loader)
                print(f"[cross-pair] CrossPairICTDataset: {_iden_vecs.shape[0]} ids × 53 exp → "
                      f"{_length} pairs/epoch (batch={opts.batch_size})")
            except Exception as _e:
                print(f"[cross-pair] WARNING: failed to build CrossPairICTDataset: {_e}. "
                      f"Disabling cross-retarget supervision.")
                self._cross_pair_loader = None

        # ── MF cross-id pair loader (delta-transfer pseudo-GT) ──────────
        self._cross_pair_mf_loader = None
        if (getattr(opts, 'lambda_cross_retarget', 0) > 0
                and getattr(opts, 'cross_pair_mf_prob', 0) > 0):
            try:
                from dataloader_cross_pair import CrossPairMFDataset, cross_pair_collate
                _mf_ds = CrossPairMFDataset(data_basedir=opts.data_basedir, mode='train')
                _mf_nw = opts.num_workers
                self._cross_pair_mf_loader = torch.utils.data.DataLoader(
                    _mf_ds, batch_size=opts.batch_size, num_workers=_mf_nw,
                    persistent_workers=(_mf_nw > 0),
                    pin_memory=torch.cuda.is_available(),
                    shuffle=True, collate_fn=partial(cross_pair_collate, device='cpu'))
                self._cross_pair_mf_iter = iter(self._cross_pair_mf_loader)
                print(f"[cross-pair-mf] CrossPairMFDataset: {len(_mf_ds._ids)} ids, "
                      f"{len(_mf_ds._holders)} PCA holders -> {len(_mf_ds)} pairs/epoch "
                      f"(prob={opts.cross_pair_mf_prob})")
            except Exception as _e:
                print(f"[cross-pair-mf] WARNING: failed to build CrossPairMFDataset: {_e}. "
                      f"ICT-only cross pairs.")
                self._cross_pair_mf_loader = None

        # ── Per-topo landmark vertex indices (for subsample anchor pinning) ──
        self._landmark_vidx_per_topo = {}
        if getattr(opts, 'subsample_ratio', 0) > 0:
            _lv_dir = getattr(opts, 'geo_dist_dir', None) or opts.rig_path
            for _topo in ('ict', 'mf'):
                _p = os.path.join(_lv_dir, f'landmark_vidx_{_topo}.npy')
                if os.path.exists(_p):
                    self._landmark_vidx_per_topo[_topo] = torch.from_numpy(
                        np.load(_p)).to(self.device).long()
            if self._landmark_vidx_per_topo:
                _msg = ', '.join(f'{k}{tuple(v.shape)}'
                                 for k, v in self._landmark_vidx_per_topo.items())
                print(f"[subsample] landmark anchors loaded: {_msg}")

        # ── Build FullPred model ─────────────────────────────────────────
        from utils.rig_loader import load_rig
        rig = load_rig(opts.rig_path)

        # Resolve mirror pair names → indices using rig.joint_names
        if self._mirror_pair_names:
            _name_to_idx = {n: i for i, n in enumerate(rig.joint_names)}
            _pair_idx = []
            for (n_l, n_r) in self._mirror_pair_names:
                if n_l in _name_to_idx and n_r in _name_to_idx:
                    _pair_idx.append([_name_to_idx[n_l], _name_to_idx[n_r]])
                else:
                    print(f"[L_mirror] WARNING: pair ({n_l}, {n_r}) not found in rig.joint_names")
            if _pair_idx:
                self._mirror_pair_idx = torch.tensor(_pair_idx, dtype=torch.long,
                                                    device=self.device)
                print(f"[L_mirror] resolved {len(_pair_idx)}/{len(self._mirror_pair_names)} pairs")
            else:
                self._mirror_pair_idx = None

        # ── Joint L/R swap map for L_w_mirror (skinning-weight symmetry prior) ──
        self._joint_swap_idx = None
        self._wmirror_map_by_N = {}
        self._wmirror_ref_by_N = {}          # N -> symmetric reference verts (mean mesh)
        if getattr(opts, 'lambda_w_mirror', 0) > 0:
            _n2i = {n: i for i, n in enumerate(rig.joint_names)}
            def _swap_name(n):
                for a, b in (('Left', 'Right'), ('_L_', '_R_')):
                    if a in n:
                        return n.replace(a, b)
                    if b in n:
                        return n.replace(b, a)
                return n
            _sw = [_n2i.get(_swap_name(n), i) for i, n in enumerate(rig.joint_names)]
            self._joint_swap_idx = torch.tensor(_sw, dtype=torch.long, device=self.device)
            _nlat = int((self._joint_swap_idx != torch.arange(len(_sw), device=self.device)).sum())
            print(f"[L_w_mirror] joint swap map: {_nlat}/{len(_sw)} lateral joints "
                  f"(lambda_w_mirror={opts.lambda_w_mirror})")
            # Symmetric reference meshes -> clean per-vertex mirror map (index
            # correspondence is a TOPOLOGY property, identity-independent). ict_mean
            # is exactly symmetric (~1e-6); mf topology is not vertex-symmetric so its
            # map stays ~0.025-approximate regardless (best available).
            import trimesh as _tm
            for _rp in ('third_party/coupe.computational-caricaturization/inputs/ict_mean.obj',
                        'third_party/coupe.computational-caricaturization/inputs/mf_mean.obj'):
                if os.path.exists(_rp):
                    _rv = np.asarray(_tm.load(_rp, process=False).vertices)
                    self._wmirror_ref_by_N[int(_rv.shape[0])] = _rv
                    print(f"[L_w_mirror] symmetric ref {os.path.basename(_rp)} N={_rv.shape[0]}")

        self.model = HierarchicalLBS_FullPred(
            rig=rig,
            topology=opts.topo_key,
            in_dim_exp=12,
            hid_dim=opts.hid_dim,
            num_layers=opts.num_layers,
            device=str(self.device),
            use_joint_trans=opts.use_joint_trans,
            smooth_W=opts.smooth_delta_W,
            smooth_W_alpha=opts.smooth_delta_W_alpha,
            dfn_skin=opts.dfn_skin,
            dfn_bind=opts.dfn_bind,
            dfn_exp=opts.dfn_exp,
            nfs_feat_dim=opts.nfs_feat_dim if opts.nfs_feat_dir else 0,
            nfs_concat=opts.nfs_concat if hasattr(opts, 'nfs_concat') else False,
            adain_pos_norm=opts.adain_pos_norm if hasattr(opts, 'adain_pos_norm') else False,
            nfs_proj_dim=getattr(opts, 'nfs_proj_dim', 0),
            use_corrective=getattr(opts, 'use_corrective', 0),
            use_helper_stage2=getattr(opts, 'use_helper_stage2', False),
            freeze_bind_pose=opts.freeze_bind_pose if hasattr(opts, 'freeze_bind_pose') else False,
            use_gmm_hybrid=opts.use_gmm_hybrid if hasattr(opts, 'use_gmm_hybrid') else False,
            init_log_sigma=opts.init_log_sigma if hasattr(opts, 'init_log_sigma') else -1.2,
            gmm_mode=getattr(opts, 'gmm_mode', 'additive'),
            residual_scale=getattr(opts, 'residual_scale', 2.0),
            sigma_targets=getattr(self, '_sigma_targets', None),
            bind_pose_mode=getattr(opts, 'bind_pose_mode', 'net'),
            joint_anchors=getattr(self, '_joint_anchors', None),
            joint_offsets=getattr(self, '_joint_offsets', None),
            attn_temperature_init=getattr(opts, 'attn_temperature_init', 0.1),
            face_joint_idx=getattr(self, '_face_joint_idx', None),
            base_joint_idx=getattr(self, '_base_joint_idx', None),
            face_mask_r0=getattr(opts, 'face_mask_r0', 1.0),
            face_mask_r1=getattr(opts, 'face_mask_r1', 2.25),
            helper_joint_idx=(self._helper_joint_idx if self._use_helpers
                              and self._helper_joint_idx else None),
            bind_pose_base_residual=bool(getattr(opts, 'bind_pose_base_residual', 0)),
        ).to(self.device)
        print(f"[HLBS FullPred] {sum(p.numel() for p in self.model.parameters()):,} params")

        # ── Stage-3: load + freeze bind_pose_net from a Stage-1 checkpoint ──
        # --bind_pose_net_ckpt : load only bind_pose_net.* (+ helper buffer) weights
        # --freeze_bind_pose_net : set requires_grad=False on bind_pose_net so the
        #   net still runs forward (predicted bind pose, NOT GT) but is not updated;
        #   skin_weight / expression / pose branches keep learning.
        _bpn_ckpt = getattr(opts, 'bind_pose_net_ckpt', None)
        if _bpn_ckpt and os.path.isfile(_bpn_ckpt):
            _sd = torch.load(_bpn_ckpt, map_location=self.device, weights_only=False)
            _bpn_sd = {k: v for k, v in _sd.items() if k.startswith('bind_pose_net.')}
            _msg = self.model.load_state_dict(_bpn_sd, strict=False)
            print(f"[Stage-3] loaded {len(_bpn_sd)} bind_pose_net tensors "
                  f"from {_bpn_ckpt}")
        if getattr(opts, 'freeze_bind_pose_net', 0):
            _n = 0
            for p in self.model.bind_pose_net.parameters():
                p.requires_grad_(False); _n += p.numel()
            self.model.bind_pose_net.eval()
            print(f"[Stage-3] bind_pose_net frozen ({_n:,} params, requires_grad=False)")

        # Load NFS pretrained features
        self._nfs_feat_cache = {}
        self._nfs_feat_dim = getattr(opts, 'nfs_feat_dim', 256)
        self._nfs_on_cpu = getattr(opts, 'nfs_on_cpu', False)
        if opts.nfs_feat_dir:
            import glob as _glob
            feat_files = _glob.glob(os.path.join(opts.nfs_feat_dir, '*_nfs_feat.npy'))
            for fp in feat_files:
                fname = os.path.basename(fp).replace('_nfs_feat.npy', '')
                t = torch.tensor(np.load(fp), dtype=torch.float32)
                if not self._nfs_on_cpu:
                    t = t.to(self.device)
                self._nfs_feat_cache[fname] = t
            total_mb = sum(v.numel() * 4 for v in self._nfs_feat_cache.values()) / 1e6
            loc_str = 'CPU' if self._nfs_on_cpu else 'GPU'
            print(f"[NFS feat] Loaded {len(self._nfs_feat_cache)} identity features "
                  f"({total_mb:.1f} MB on {loc_str})")

        # Build regional weight constraints
        if opts.lambda_rwc > 0:
            rwc_topos = tuple(t.strip() for t in opts.rwc_topologies.split(','))
            self.model._build_regional_weight_constraints(
                alpha=opts.rwc_alpha, adaptive=opts.rwc_adaptive, topologies=rwc_topos)

        # Build hierarchy locality constraints
        if opts.lambda_hier > 0:
            rwc_topos = tuple(t.strip() for t in opts.rwc_topologies.split(','))
            self.model._build_hierarchy_constraints(topologies=rwc_topos)

        # Resume from checkpoint if specified
        if opts.ckpt and opts.continue_ckpt:
            ckpt_path = os.path.join(opts.ckpt, f"model_hlbs_{opts.start_epoch:03d}.pth")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(opts.ckpt, "model_hlbs_best.pth")
            if os.path.exists(ckpt_path):
                # strict=False so old ckpts (pre face-mask / sigma-target buffers) load cleanly
                _msg = self.model.load_state_dict(
                    torch.load(ckpt_path, map_location=self.device), strict=False)
                if _msg.missing_keys or _msg.unexpected_keys:
                    print(f"[FullPred resume] missing={len(_msg.missing_keys)} "
                          f"unexpected={len(_msg.unexpected_keys)}")
                print(f"[FullPred] Resumed from: {ckpt_path}")

        # ── Stage-2: initialize from stage-1 ckpt (fresh run dir/optimizer/epoch) ──
        if getattr(opts, 'init_ckpt', '') and not (opts.ckpt and opts.continue_ckpt):
            _sd = torch.load(opts.init_ckpt, map_location=self.device)
            _own = self.model.state_dict()
            # Drop shape-mismatched entries (e.g. face_joint_idx 34 -> 55 when a
            # base-only ckpt initializes a helper-enabled model): strict=False
            # skips missing/unexpected KEYS but still errors on size mismatch.
            _drop = [k for k, v in _sd.items() if k in _own and _own[k].shape != v.shape]
            for k in _drop:
                _sd.pop(k)
            if _drop:
                print(f"[Stage2] dropped shape-mismatched keys: {_drop}")
            _msg = self.model.load_state_dict(_sd, strict=False)
            print(f"[Stage2] init from: {opts.init_ckpt} "
                  f"(missing={len(_msg.missing_keys)} unexpected={len(_msg.unexpected_keys)})")
            if _msg.missing_keys:
                print(f"[Stage2] fresh-init modules: {sorted(set(k.split('.')[0] for k in _msg.missing_keys))}")

        # ── Stage-2: freeze base rig, train helper/corrective branch only ──
        _s2_rows_mode = False
        if getattr(opts, 'freeze_base_rig', False):
            if (getattr(opts, 'helper_stage2_mode', 'heads') == 'rows'
                    and not getattr(opts, 'use_helper_stage2', False)):
                # ── rows mode: NO new modules. Freeze everything, then train ONLY
                # the helper ROWS of the three final layers. Each output unit has
                # its own row (out_j = w_j·h + b_j), rows are independent, so base
                # outputs stay bit-exact. Grad hooks zero non-helper rows; one
                # UNION mask per tensor (multiple hooks would multiply/intersect).
                _s2_rows_mode = True
                for _pp in self.model.parameters():
                    _pp.requires_grad = False
                _hidx = self.model.helper_joint_idx_buf.tolist()
                _J = self.model.num_joints

                def _enable_rows(layer, rows):
                    _mw = torch.zeros_like(layer.weight); _mw[rows] = 1.0
                    _mb = torch.zeros_like(layer.bias);   _mb[rows] = 1.0
                    layer.weight.requires_grad = True
                    layer.bias.requires_grad = True
                    layer.weight.register_hook(lambda g, m=_mw: g * m)
                    layer.bias.register_hook(lambda g, m=_mb: g * m)

                # 1) skin logits: helper rows, re-init W=0 / bias=-4 (stage-1
                #    values were face-masked out of softmax => untrained garbage;
                #    -4 => near-zero initial softmax share, grad still alive)
                _lo = self.model.skin_weight_net.layer_out
                with torch.no_grad():
                    _lo.weight[_hidx] = 0.0
                    _lo.bias[_hidx] = -4.0
                _enable_rows(_lo, _hidx)

                # 2) pose rows: layout [J*6 rot6d | J*3 t]. Identity rot6d bias
                #    => helpers initially move rigidly with parent via FK.
                _po = self.model.lbs_pose_model.layer_out
                _rot_rows = [6 * h + k for h in _hidx for k in range(6)]
                _t_rows = ([_J * 6 + 3 * h + k for h in _hidx for k in range(3)]
                           if opts.use_joint_trans else [])
                _pose_rows = _rot_rows + _t_rows
                with torch.no_grad():
                    _po.weight[_pose_rows] = 0.0
                    _po.bias[_rot_rows] = torch.tensor(
                        [1., 0., 0., 0., 1., 0.], device=_po.bias.device).repeat(len(_hidx))
                    if _t_rows:
                        _po.bias[_t_rows] = 0.0
                _enable_rows(_po, _pose_rows)

                # 3) bind rows: keep stage-1 values (weak L_bind_reg supervised —
                #    plausible positions), just make them trainable.
                _bo = self.model.bind_pose_net.layer_out
                _bind_rows = [3 * h + k for h in _hidx for k in range(3)]
                _enable_rows(_bo, _bind_rows)

                _eff = ((len(_hidx) + len(_pose_rows) + len(_bind_rows))
                        * (_lo.weight.shape[1] + 1))
                print(f"[Stage2-rows] base rig frozen (bit-exact) | trainable helper rows: "
                      f"skin {len(_hidx)} pose {len(_pose_rows)} bind {len(_bind_rows)} "
                      f"(~{_eff/1e3:.1f}K effective params)")
            else:
                _S2_TRAINABLE = ('corr_', 'helper_logit_net', 'helper_pose_head', 'helper_bind_head')
                _n_frz = 0; _n_trn = 0
                for _pn, _pp in self.model.named_parameters():
                    if _pn.startswith(_S2_TRAINABLE):
                        _pp.requires_grad = True;  _n_trn += _pp.numel()
                    else:
                        _pp.requires_grad = False; _n_frz += _pp.numel()
                print(f"[Stage2] base rig FROZEN ({_n_frz/1e6:.2f}M params) | "
                      f"trainable branch: {_n_trn/1e3:.1f}K params")

        # rows mode: weight_decay must be 0 — AdamW's decoupled decay shrinks the
        # WHOLE tensor (incl. frozen base rows with zero grads) every step.
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=opts.lr, betas=(0.9, 0.999),
            weight_decay=(0.0 if _s2_rows_mode else 1e-2))
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opts.sc_step, gamma=opts.sc_gamma)

        if opts.target == 'gt':
            opts.smooth_n_iter = 0

        train_ds = CBDDataset(opts, is_train=True, toggle=opts.data_toggle, data_basedir=opts.data_basedir)
        valid_ds = CBDDataset(opts, is_valid=True, toggle=opts.data_toggle, data_basedir=opts.data_basedir)
        _region_min = 1 if opts.no_fullhead else 0
        train_sampler = CBDdataSampler(train_ds.len_list, BS, shuffle=True,  balance=False, is_train=True, region_min=_region_min)
        valid_sampler = CBDdataSampler(valid_ds.len_list, BS, shuffle=True,  balance=False, is_valid=True, region_min=_region_min)
        _nw = opts.num_workers
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_sampler=train_sampler,
            collate_fn=partial(CBD_collate_wrapper, device='cpu'),
            num_workers=_nw, persistent_workers=(_nw > 0),
            pin_memory=torch.cuda.is_available())
        valid_loader = torch.utils.data.DataLoader(
            valid_ds, batch_sampler=valid_sampler,
            collate_fn=partial(CBD_collate_wrapper, device='cpu'), num_workers=0)

        # ── Logging ──────────────────────────────────────────────────────
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        trans_tag = "-jTrans" if opts.use_joint_trans else ""
        sdw_tag = f"-sdw{opts.smooth_delta_W}a{opts.smooth_delta_W_alpha}" if opts.smooth_delta_W > 0 else ""
        fh_tag = "-noFH" if opts.no_fullhead else ""
        surf_tag = ""
        if opts.lambda_normal > 0: surf_tag += f"-nrm{opts.lambda_normal}"
        if opts.lambda_curvature > 0: surf_tag += f"-crv{opts.lambda_curvature}"
        wsm_tag = f"-Wsm{opts.lambda_W_smooth}" if opts.lambda_W_smooth > 0 else ""
        cur_tag = f"-cur{opts.curriculum_epochs}" if opts.curriculum else ""
        tag = f"-HLBS-FullPred-{opts.topo_key}{trans_tag}{sdw_tag}{fh_tag}{surf_tag}{wsm_tag}{cur_tag}"
        if opts.ckpt and opts.continue_ckpt:
            opts.log_dir = opts.ckpt  # resume into same dir
        else:
            opts.log_dir = os.path.join(opts.log_dir, now + tag)

        os.makedirs(opts.log_dir, exist_ok=True)
        os.makedirs(f"{opts.log_dir}/img/train/mesh", exist_ok=True)
        os.makedirs(f"{opts.log_dir}/img/train/cross_retarget", exist_ok=True)
        os.makedirs(f"{opts.log_dir}/img/valid/mesh", exist_ok=True)

        with open(os.path.join(opts.log_dir, "opts.json"), 'w') as f:
            json.dump(vars(opts), f, indent=4)
        with open(os.path.join(opts.log_dir, "train_opts.yml"), 'w') as f:
            yaml.dump(vars(opts), f, sort_keys=False)

        writer_train = writer_valid = None
        if opts.tb:
            writer_train = SummaryWriter(log_dir=os.path.join(opts.log_dir, "train"))
            writer_valid = SummaryWriter(log_dir=os.path.join(opts.log_dir, "valid"))

        logger = Logger(os.path.join(opts.log_dir, "log.txt"))
        print(f'Log: {logger.file_path}')

        # Determine skin_weight_net config for logging
        try:
            _sw_net = self.model.skin_weight_net
            _sw_in_dim = _sw_net.layer_in.weight.shape[1] if hasattr(_sw_net, 'layer_in') else '?'
            _sw_n_layers = len(_sw_net.layers) if hasattr(_sw_net, 'layers') else '?'
            _adain_layers = [m for m in _sw_net.adain_in.modules() if hasattr(m, 'weight') and m.weight.dim() == 2] if hasattr(_sw_net, 'adain_in') else []
            _adain_dim = _adain_layers[0].weight.shape[1] if _adain_layers else '?'
        except Exception:
            _sw_in_dim = _sw_n_layers = _adain_dim = '?'

        config_text = (
            f"=== HLBS FullPred Training ===\n"
            f"  topo_key       : {opts.topo_key}\n"
            f"  use_joint_trans: {opts.use_joint_trans}\n"
            f"  init_phase     : {opts.init_phase_epochs} epochs ({opts.init_mode})\n"
            f"  lambda_init    : {opts.lambda_init}\n"
            f"  lambda_vert    : {opts.lambda_vert}\n"
            f"  lambda_neu     : {opts.lambda_neu}\n"
            f"  lambda_rwc     : {opts.lambda_rwc} (adaptive={getattr(opts, 'rwc_adaptive', False)})\n"
            f"  lambda_hier    : {opts.lambda_hier} (margin={opts.hier_margin})\n"
            f"  lambda_dist    : {opts.lambda_dist}\n"
            f"  use_gmm_hybrid : {getattr(opts, 'use_gmm_hybrid', False)} "
            f"(mode={getattr(opts, 'gmm_mode', 'additive')}, init_log_σ={getattr(opts, 'init_log_sigma', -1.2)})\n"
            f"  σ targets      : {getattr(opts, 'sigma_targets_npy', None)} "
            f"(λ_sigma_reg={getattr(opts, 'lambda_sigma_reg', 0.0)})\n"
            f"  bind_pose_mode : {getattr(opts, 'bind_pose_mode', 'net')} "
            f"(anchors={getattr(opts, 'joint_anchors_npy', None)}, "
            f"cache={getattr(opts, 'bind_pos_cache_dir', None)})\n"
            f"  face_mask      : {self._face_joint_idx is not None} ("
            f"{len(self._face_joint_idx) if self._face_joint_idx else 0}/{self.model.num_joints} joints, "
            f"r0={opts.face_mask_r0}, r1={opts.face_mask_r1})\n"
            f"  nfs_feat_dir   : {opts.nfs_feat_dir}\n"
            f"  nfs_concat     : {getattr(opts, 'nfs_concat', False)}\n"
            f"  adain_pos_norm : {getattr(opts, 'adain_pos_norm', False)}\n"
            f"  skin_weight_net: in={_sw_in_dim}, layers={_sw_n_layers}, adain_in={_adain_dim}\n"
            f"  dfn_skin/bind/exp: {opts.dfn_skin}/{opts.dfn_bind}/{opts.dfn_exp}\n"
            f"==============================\n"
        )
        print(config_text)
        logger.write(config_text)
        logger.write(train_ds.get_data_config())

        # ── Precompute mesh edges ────────────────────────────────────────
        if opts.smooth_delta_W > 0:
            print(f"[FullPred] delta_W forward smoothing: iters={opts.smooth_delta_W}, alpha={opts.smooth_delta_W_alpha}")

        BEST_LOSS = 1e8
        BEST_EPOCH = 0
        len_train = len(train_loader)
        len_valid = len(valid_loader)
        interv = max(1, round(len_train / 10))
        interv_val = max(1, round(len_valid / 3))
        K = opts.init_phase_epochs

        # ── Curriculum: Phase 1 loader (ICT single-basis only) ──────────
        curriculum_transitioned = False
        if opts.curriculum:
            ict_len_list = [x for x in train_ds.len_list if x[2].item() == 5]
            n_ict_ids = ict_len_list[0][3]
            pad = 53 % BS
            ict_len_list[0] = [53 + (BS - pad if pad else 0), ict_len_list[0][1], ict_len_list[0][2], n_ict_ids]
            train_ds.curriculum_single_basis = True
            cur_sampler = CBDdataSampler(ict_len_list, BS, shuffle=True, balance=False, is_train=True, region_min=_region_min)
            cur_loader = torch.utils.data.DataLoader(
                train_ds, batch_sampler=cur_sampler,
                collate_fn=partial(CBD_collate_wrapper, device='cpu'),
                num_workers=_nw, persistent_workers=(_nw > 0),
            pin_memory=torch.cuda.is_available())
            print(f"[Curriculum] Phase 1: ICT single-basis for {opts.curriculum_epochs} epochs "
                  f"({len(cur_loader)} batches/epoch, {n_ict_ids} identities × 53 bases)")

        for epoch in range(opts.start_epoch, epochs + 1):
            # ── Curriculum phase transition ──────────────────────────────
            if opts.curriculum and not curriculum_transitioned:
                if epoch < opts.curriculum_epochs:
                    active_loader = cur_loader
                else:
                    train_ds.curriculum_single_basis = False
                    active_loader = train_loader
                    if epoch == opts.curriculum_epochs:
                        curriculum_transitioned = True
                        print(f"[Curriculum] Phase 2: switching to full data at epoch {epoch}")
            else:
                active_loader = train_loader
            # Phase 1 init loss scheduling
            H = opts.init_hold_epochs
            if opts.init_mode == 'anneal':
                lambda_init = opts.lambda_init * max(0.0, 1.0 - epoch / K) if K > 0 else 0.0
            elif opts.init_mode == 'hold_anneal':
                if epoch < H:
                    lambda_init = opts.lambda_init
                elif epoch < K:
                    lambda_init = opts.lambda_init * (1.0 - (epoch - H) / max(K - H, 1))
                else:
                    lambda_init = 0.0
            elif opts.init_mode == 'hold_cutoff':
                lambda_init = opts.lambda_init if epoch < H else 0.0

            # ── Train ────────────────────────────────────────────────────
            self.model.train()
            running = {"recon-lbs": 0.0, "recon-neu": 0.0, "recon-normal": 0.0, "init-W": 0.0, "init-bind": 0.0, "L_bind_reg": 0.0, "L_bind_residual": 0.0, "L_helper_residual": 0.0, "L_mirror": 0.0, "L_w_mirror": 0.0, "L_rwc_init": 0.0, "L_rwc_min": 0.0, "L_hier": 0.0, "L_dist": 0.0, "L_wlap": 0.0, "L_wref": 0.0, "L_ortho": 0.0, "L_inside_bind": 0.0, "L_inside_def": 0.0, "metric-W_smooth": 0.0, "L_sigma": 0.0, "L_net_center": 0.0, "L_cross_retarget": 0.0, "total": 0.0}
            cnt = 0

            _len_active = len(active_loader)
            pbar = tqdm(enumerate(active_loader), total=_len_active, ncols=120,
                        desc=f"[{epoch:03d}] Train FullPred (phase{'1' if lambda_init > 0 else '2'})")
            _use_amp = getattr(opts, 'amp', False) and torch.cuda.is_available()
            _prof = getattr(opts, 'profile', False)
            _tprof = None
            if _prof:
                _prof_acc = {'data': 0.0, 'fwd': 0.0, 'bwd': 0.0, 'opt': 0.0}
                _prof_n = 0
                _prof_sync = (torch.cuda.synchronize if torch.cuda.is_available()
                              else (lambda: None))
                _prof_prev = time.perf_counter()
                # one-shot op-level trace: wait 10 / warmup 2 / active 4 steps
                if not getattr(self, '_torch_prof_done', False):
                    _tprof = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA],
                        schedule=torch.profiler.schedule(wait=10, warmup=2, active=4),
                        record_shapes=False, with_stack=False)
                    _tprof.start()
            for idx, batch in pbar:
                if _prof:
                    _t_data = time.perf_counter()
                    _prof_acc['data'] += _t_data - _prof_prev
                batch = batch.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()

                # bf16 autocast over the whole forward+loss; backward stays outside.
                # (loop body has no continue/break/return → manual enter/exit safe.)
                _amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                          enabled=_use_amp)
                _amp_ctx.__enter__()

                src_v  = batch.template
                src_n  = batch.template_normal
                gt_v   = batch.vertices
                gt_n   = batch.vertices_normal
                target_v = gt_v

                # Register mesh edges for this topology if not cached
                # (needed by smooth_delta_W op and by L_wlap / weight-quality metric)
                if (opts.smooth_delta_W > 0 or getattr(opts, 'lambda_wlap', 0) > 0
                        or getattr(opts, 'w_metric', 0)):
                    N_cur = src_v.shape[1]
                    if self.model._mesh_edges_by_N is None or N_cur not in self.model._mesh_edges_by_N:
                        _faces = batch.faces[0] if batch.faces.dim() == 3 else batch.faces
                        self.model.set_mesh_edges(_faces)

                # Update DiffusionNet precomputes when topology changes
                if opts.dfn_skin or opts.dfn_bind or opts.dfn_exp:
                    N_cur = src_v.shape[1]
                    if not hasattr(self, '_dfn_cached_N') or self._dfn_cached_N != N_cur:
                        import trimesh as _tm
                        from utils.nfr_utils import get_dfn_info
                        _f = batch.faces[0].cpu().numpy() if batch.faces.dim() == 3 else batch.faces.cpu().numpy()
                        _v = src_v[0].cpu().numpy()
                        _mesh = _tm.Trimesh(vertices=_v, faces=_f, process=False)
                        _dfn = get_dfn_info(_mesh, cache_dir='dfn_cache', map_location=self.device)
                        self.model.update_dfn_precomputes(_dfn)
                        self._dfn_cached_N = N_cur

                # Get NFS features if available
                _nfs_feat = None
                if opts.nfs_feat_dir and self._nfs_feat_cache:
                    B_cur = src_v.shape[0]
                    N_cur = src_v.shape[1]
                    feats = []
                    for b in range(B_cur):
                        id_key = batch.id_name[b]
                        if id_key in self._nfs_feat_cache:
                            f = self._nfs_feat_cache[id_key]
                            if f.shape[0] > N_cur:
                                f = f[:N_cur]
                            if self._nfs_on_cpu:
                                f = f.to(self.device, non_blocking=True)
                            feats.append(f)
                        else:
                            feats.append(torch.zeros(N_cur, self._nfs_feat_dim, device=self.device))
                    _nfs_feat = torch.stack(feats, dim=0)  # [B, N, 256]

                # ── Subsample augmentation (Track A) ──
                # Subsample to K = round(N * ratio) vertices; pin landmark anchors;
                # apply same perm to src_v / src_n / gt_v / gt_n / nfs_feat.
                # Sets batch.perm_idx so existing perm-aware paths (geo_dist, rwc, hier,
                # init_loss W target) handle their own slicing.
                if getattr(opts, 'subsample_ratio', 0) > 0:
                    _perm = self._build_subsample_perm(
                        src_v, batch, opts.subsample_ratio, opts.subsample_mode)
                    if _perm is not None:
                        idx3 = _perm.unsqueeze(-1).expand(-1, -1, 3)
                        src_v = torch.gather(src_v, 1, idx3)
                        src_n = torch.gather(src_n, 1, idx3)
                        gt_v  = torch.gather(gt_v,  1, idx3)
                        gt_n  = torch.gather(gt_n,  1, idx3)
                        target_v = gt_v
                        if _nfs_feat is not None:
                            idxF = _perm.unsqueeze(-1).expand(-1, -1, _nfs_feat.shape[-1])
                            _nfs_feat = torch.gather(_nfs_feat, 1, idxF)
                        batch.perm_idx = _perm

                is_permed = hasattr(batch, 'perm_idx') and batch.perm_idx is not None

                delta     = gt_v - src_v
                src_in    = torch.cat([src_v, src_n], dim=-1)
                deform_in = torch.cat([delta, gt_n, src_in], dim=-1)

                # Per-id bind_pos cache lookup (anchor_pool mode optimization)
                _bind_pos_cache = None
                if self._bind_pos_cache:
                    _items = []
                    _all_hit = True
                    for b in range(src_v.shape[0]):
                        k = batch.id_name[b]
                        if k in self._bind_pos_cache:
                            _items.append(self._bind_pos_cache[k])
                        else:
                            _all_hit = False; break
                    if _all_hit:
                        _bind_pos_cache = torch.stack(_items, dim=0)  # [B, J, 3]

                # freeze_bind_pose mode: build the per-batch frozen bind_pos cache.
                # Priority per id: per-id v3 GT (Phase B landmark cache) > per-topo mean.
                # → Stage-2 training (bind pose fixed at per-id GT, learn skin/transform).
                if (_bind_pos_cache is None
                        and getattr(opts, 'freeze_bind_pose', False)
                        and hasattr(batch, 'id_name')):
                    _items = []
                    _n_per_id = 0; _n_mean = 0
                    for b in range(src_v.shape[0]):
                        idn = batch.id_name[b]
                        if idn in self._per_id_bind_pose_gt:           # per-id v3 GT
                            _items.append(self._per_id_bind_pose_gt[idn])
                            _n_per_id += 1
                        elif idn.startswith('ict_') and hasattr(self.model, 'bind_pos_target_ict'):
                            _items.append(self.model.bind_pos_target_ict); _n_mean += 1
                        elif idn.startswith('m--') and hasattr(self.model, 'bind_pos_target_mf'):
                            _items.append(self.model.bind_pos_target_mf); _n_mean += 1
                        else:
                            _items.append(self.model.bind_pos_target); _n_mean += 1
                    _bind_pos_cache = torch.stack(_items, dim=0).detach()  # [B, J, 3]

                # Per-batch geodesic dist² lookup (replaces Euclidean ||v - μ||²)
                _dist_sq_geo = None
                if self._geo_dist_per_topo:
                    B_cur = src_v.shape[0]; N_cur = src_v.shape[1]
                    J_cur = self.model.num_joints
                    _per_b = []
                    _all_hit_geo = True
                    for b in range(B_cur):
                        idn = batch.id_name[b] if hasattr(batch, 'id_name') else ''
                        topo = ('ict'  if idn.startswith('ict_')  else
                                'mf'   if idn.startswith('m--')   else
                                'biwi' if idn.startswith('biwi_') else
                                'coma' if idn.startswith('coma_') else None)
                        if topo and topo in self._geo_dist_per_topo:
                            gd = self._geo_dist_per_topo[topo]      # [J, V_topo]
                            gd_t = gd.t()                            # [V_topo, J]
                            if is_permed:
                                _perm = batch.perm_idx[b].to(gd_t.device).long()
                                gd_t = gd_t.index_select(0, _perm)   # [N, J]
                            elif gd_t.shape[0] != N_cur:
                                if gd_t.shape[0] > N_cur:
                                    gd_t = gd_t[:N_cur]
                                else:
                                    _pad = torch.zeros(N_cur - gd_t.shape[0], J_cur,
                                                       device=gd_t.device, dtype=gd_t.dtype)
                                    gd_t = torch.cat([gd_t, _pad], dim=0)
                            _per_b.append(gd_t)
                        else:
                            _all_hit_geo = False; break
                    if _all_hit_geo:
                        _dist_sq_geo = (torch.stack(_per_b, dim=0)) ** 2   # [B, N, J]

                _need_extras = (opts.lambda_rwc > 0 or opts.lambda_hier > 0
                                or opts.lambda_bind_reg > 0 or opts.lambda_dist > 0
                                or opts.lambda_net_center > 0
                                or getattr(opts, 'lambda_wlap', 0) > 0
                                or getattr(opts, 'lambda_wref', 0) > 0
                                or getattr(opts, 'lambda_inside_bind', 0) > 0
                                or getattr(opts, 'lambda_inside_def', 0) > 0
                                or getattr(opts, 'w_metric', 0))
                if _need_extras:
                    pred_lbs, _extras = self.model(
                        src_v, deform_in, source_normal=src_n,
                        nfs_feat=_nfs_feat, return_extras=True,
                        bind_pos_cache=_bind_pos_cache,
                        dist_sq_geo=_dist_sq_geo)
                    _W = _extras['W']          # [B, N, J]
                    _joint_pos = _extras['joint_pos']  # [B, J, 3] or None
                else:
                    pred_lbs = self.model(src_v, deform_in, source_normal=src_n,
                                          nfs_feat=_nfs_feat, bind_pos_cache=_bind_pos_cache,
                                          dist_sq_geo=_dist_sq_geo)

                # ── Recon loss ───────────────────────────────────────────
                if opts.no_t_mask:
                    loss_dict = {"recon-lbs": F.mse_loss(target_v, pred_lbs)}
                else:
                    from utils.exp_utils import plateau_hat_points
                    t_mask   = plateau_hat_points(src_v)
                    inv_mask = 1.0 - t_mask
                    loss_dict = {
                        "recon-lbs": (
                            F.mse_loss(target_v * t_mask,  pred_lbs * t_mask)
                            + F.mse_loss(src_v * inv_mask, pred_lbs * inv_mask)
                        )
                    }

                # ── Neutral recon loss ───────────────────────────────────
                if opts.lambda_neu > 0:
                    delta_zero = torch.zeros_like(src_v)
                    neu_deform_in = torch.cat([delta_zero, src_n, src_v, src_n], dim=-1)
                    # reuse_identity: W / joint_pos / B_inv_id are identity-only
                    # and were just computed by the main forward above — skip the
                    # redundant bind_pose_net + skin_weight_net pass.
                    pred_neutral = self.model(src_v, neu_deform_in, source_normal=src_n,
                                              nfs_feat=_nfs_feat if hasattr(self, '_nfs_feat_cache') else None,
                                              bind_pos_cache=_bind_pos_cache,
                                              dist_sq_geo=_dist_sq_geo,
                                              reuse_identity=True)
                    if opts.no_t_mask:
                        loss_dict["recon-neu"] = F.mse_loss(src_v, pred_neutral)
                    else:
                        loss_dict["recon-neu"] = (
                            F.mse_loss(src_v * t_mask,    pred_neutral * t_mask)
                            + F.mse_loss(src_v * inv_mask, pred_neutral * inv_mask)
                        )

                # ── Normal consistency loss ─────────────────────────────────
                # Non-permed: classic face-based per-vertex normals (calc_norm_torch).
                # Permed: GT normals are pre-computed on full mesh (in batch.vertices_normal)
                # and already gathered into gt_n at the sampled indices — use directly.
                # Pred normals must still be estimated since pred_lbs is the model output:
                # use PCA-on-kNN (Hoppe '92, Klasing '09; PyTorch3D estimate_pointcloud_normals)
                # with sign alignment against template normal src_n.
                if opts.lambda_normal > 0 and not (getattr(opts, 'normal_full_only', False) and is_permed):
                    if is_permed:
                        from pytorch3d.ops import estimate_pointcloud_normals
                        k = max(4, int(getattr(opts, 'normal_knn_k', 16)))
                        pred_n_raw = estimate_pointcloud_normals(
                            pred_lbs, neighborhood_size=k, disambiguate_directions=False)
                        pred_sign = torch.sign((pred_n_raw * src_n).sum(dim=-1, keepdim=True))
                        pred_sign = torch.where(pred_sign == 0, torch.ones_like(pred_sign), pred_sign)
                        pred_n      = pred_n_raw * pred_sign
                        gt_n_recomp = gt_n                            # pre-computed, already permed
                    else:
                        from utils.mesh_utils import calc_norm_torch
                        pred_n      = calc_norm_torch(pred_lbs, batch.faces, at='verts')
                        gt_n_recomp = calc_norm_torch(target_v, batch.faces, at='verts')
                    normal_diff = 1 - F.cosine_similarity(pred_n, gt_n_recomp, dim=-1)
                    if not opts.no_t_mask:
                        normal_diff = normal_diff * t_mask.squeeze(-1)
                    loss_dict["recon-normal"] = normal_diff.mean()

                # ── Curvature loss (Laplacian difference) ────────────────
                if opts.lambda_curvature > 0 and not is_permed:
                    from train_edd_real import _uniform_laplacian
                    loss_dict["recon-curvature"] = F.mse_loss(
                        _uniform_laplacian(pred_lbs, batch.faces),
                        _uniform_laplacian(target_v, batch.faces))

                # ── Phase 1: init supervision ────────────────────────────
                if lambda_init > 0:
                    _md = batch.mesh_data if hasattr(batch, 'mesh_data') else None
                    init_losses = self.model.init_loss(src_v, source_normal=src_n, mesh_data=_md,
                                                       nfs_feat=_nfs_feat, dist_sq_geo=_dist_sq_geo)
                    for k, v in init_losses.items():
                        loss_dict[k] = v

                # ── Bind pose regularization (per-topology / per-id supervision in net mode) ──
                # Skipped in anchor_pool mode (redundant by construction).
                # Target priority: per-id GT (Phase B landmark-based) > per-topo mean (fallback)
                if (opts.lambda_bind_reg > 0 and _joint_pos is not None
                        and getattr(opts, 'bind_pose_mode', 'net') == 'net'):
                    B_cur = _joint_pos.shape[0]
                    targets = []
                    n_per_id = 0; n_per_topo = 0
                    for b in range(B_cur):
                        idn = batch.id_name[b] if hasattr(batch, 'id_name') else ''
                        # 1st: per-id GT cache (landmark-based)
                        if idn in self._per_id_bind_pose_gt:
                            targets.append(self._per_id_bind_pose_gt[idn])
                            n_per_id += 1
                            continue
                        # 2nd: per-topo mean fallback
                        if idn.startswith('ict_') and hasattr(self.model, 'bind_pos_target_ict'):
                            tgt = self.model.bind_pos_target_ict
                        elif idn.startswith('m--') and hasattr(self.model, 'bind_pos_target_mf'):
                            tgt = self.model.bind_pos_target_mf
                        else:
                            tgt = self.model.bind_pos_target
                        targets.append(tgt)
                        n_per_topo += 1
                    target_batch = torch.stack(targets, dim=0)               # [B, J, 3]
                    # Under random scale/trans aug, the mesh (hence predicted
                    # joint_pos) lives in the augmented frame. Map the fixed bind
                    # GT into that frame so L_bind_reg is consistent: GT*scale+trans
                    # (per sample). No-op when aug off (scale=1, trans=0 / None).
                    if getattr(batch, 'scale', None) is not None:
                        _bsc = batch.scale.to(target_batch.device).unsqueeze(1)  # [B,1,3]
                        _btr = batch.trans.to(target_batch.device).unsqueeze(1)  # [B,1,3]
                        target_batch = target_batch * _bsc + _btr
                    # Exclude helper joints: their per-id GT = parent's per-id pos,
                    # so MSE would lock helpers onto parent, killing the helper's
                    # degree of freedom. Restrict L_bind_reg to non-helper joints.
                    if self._helper_joint_idx:
                        J_total = _joint_pos.shape[1]
                        non_helper = [j for j in range(J_total)
                                      if j not in self._helper_joint_idx]
                        nh_idx = torch.tensor(non_helper, dtype=torch.long,
                                              device=_joint_pos.device)
                        loss_dict["L_bind_reg"] = F.mse_loss(
                            _joint_pos.index_select(1, nh_idx),
                            target_batch.index_select(1, nh_idx).detach())
                    else:
                        loss_dict["L_bind_reg"] = F.mse_loss(_joint_pos, target_batch.detach())

                # ── Bind-pose residual L2 regularization (differentiated) ─
                # base_residual mode: _last_bind_pose_residual is [B, J, 3] for ALL
                #   joints. Split into non-helper (strong λ_bind_residual — stay near
                #   ICT-mean base) vs helper (weak λ_helper_residual — freer to move).
                # legacy mode: _last_bind_pose_residual is helper-only → L_helper_residual.
                _res = getattr(self.model, '_last_bind_pose_residual', None)
                if _res is not None and bool(getattr(opts, 'bind_pose_base_residual', 0)):
                    if self._helper_joint_idx:
                        J_tot = _res.shape[1]
                        _h = torch.tensor(self._helper_joint_idx, dtype=torch.long,
                                          device=_res.device)
                        _nh = torch.tensor([j for j in range(J_tot)
                                            if j not in self._helper_joint_idx],
                                           dtype=torch.long, device=_res.device)
                        if getattr(opts, 'lambda_bind_residual', 0) > 0:
                            loss_dict["L_bind_residual"] = (
                                _res.index_select(1, _nh) ** 2).sum(dim=-1).mean()
                        if getattr(opts, 'lambda_helper_residual', 0) > 0:
                            loss_dict["L_helper_residual"] = (
                                _res.index_select(1, _h) ** 2).sum(dim=-1).mean()
                    elif getattr(opts, 'lambda_bind_residual', 0) > 0:
                        loss_dict["L_bind_residual"] = (_res ** 2).sum(dim=-1).mean()
                elif (_res is not None and self._use_helpers
                        and getattr(opts, 'lambda_helper_residual', 0) > 0):
                    # legacy: helper-only residual
                    loss_dict["L_helper_residual"] = (_res ** 2).sum(dim=-1).mean()

                # ── Bilateral mirror loss for L/R helper pairs ──────────
                # L_mirror = MSE(pos[L], mirror_x(pos[R])), mirror_x([x,y,z]) = [-x, y, z].
                # Encourages predicted helper positions to be left-right symmetric.
                if (self._use_helpers and getattr(opts, 'lambda_mirror', 0) > 0
                        and self._mirror_pair_idx is not None
                        and _joint_pos is not None):
                    L_idx = self._mirror_pair_idx[:, 0]
                    R_idx = self._mirror_pair_idx[:, 1]
                    pos_L = _joint_pos.index_select(1, L_idx)               # [B, P, 3]
                    pos_R = _joint_pos.index_select(1, R_idx)               # [B, P, 3]
                    mirror_R = pos_R.clone()
                    mirror_R[..., 0] = -mirror_R[..., 0]                    # flip x
                    loss_dict["L_mirror"] = F.mse_loss(pos_L, mirror_R)

                # ── σ shrinkage penalty (active joints only if face-mask on) ──
                if opts.lambda_sigma_reg > 0 and getattr(self.model, 'use_gmm_hybrid', False):
                    _act = self._face_joint_idx
                    loss_dict["L_sigma"] = self.model.sigma_shrink_loss(active_idx=_act)

                # ── Net-center anchor loss ───────────────────────────────
                # Euclidean: ||Σ_v W_net[v,j]·v_pos - joint_pos||²        (1차 모멘트 매칭)
                # Geodesic : Σ_v W_net[v,j] · geo_dist²[topo, j, v]       (home_vertex 주변 집중)
                if (opts.lambda_net_center > 0 and _need_extras
                        and _extras.get('logit_net') is not None
                        and (_joint_pos is not None or _dist_sq_geo is not None)):
                    logit_net = _extras['logit_net']                     # [B, V, J]
                    W_net = F.softmax(logit_net, dim=1)                   # softmax over V per joint
                    if _dist_sq_geo is not None:
                        # Restrict to face joints if defined
                        if self._face_joint_idx:
                            _face_idx_t = torch.tensor(self._face_joint_idx,
                                                       dtype=torch.long, device=W_net.device)
                            W_a = W_net.index_select(2, _face_idx_t)         # [B, V, J_face]
                            d_a = _dist_sq_geo.index_select(2, _face_idx_t)  # [B, V, J_face]
                        else:
                            W_a = W_net; d_a = _dist_sq_geo
                        loss_dict["L_net_center"] = (W_a * d_a).sum(dim=1).mean()
                    else:
                        mu_fit = torch.einsum('bvj,bvk->bjk', W_net, src_v)  # [B, J, 3]
                        if self._face_joint_idx:
                            _face_idx_t = torch.tensor(self._face_joint_idx,
                                                       dtype=torch.long, device=mu_fit.device)
                            mu_fit_a = mu_fit.index_select(1, _face_idx_t)
                            mu_tgt_a = _joint_pos.index_select(1, _face_idx_t)
                        else:
                            mu_fit_a = mu_fit; mu_tgt_a = _joint_pos
                        loss_dict["L_net_center"] = F.mse_loss(mu_fit_a, mu_tgt_a.detach())

                # ── Regional weight constraint (uses pre-computed W) ────
                if opts.lambda_rwc > 0:
                    _md = batch.mesh_data if hasattr(batch, 'mesh_data') else None
                    _perm = getattr(batch, 'perm_idx', None)
                    rwc_losses = self.model.regional_weight_constraint_loss(
                        _W, src_v.shape[1], mesh_data=_md, perm_idx=_perm)
                    for k, v in rwc_losses.items():
                        loss_dict[k] = v

                # ── Distance-based weight locality (uses pre-computed W) ─
                if opts.lambda_dist > 0:
                    dist_losses = self.model.distance_weight_loss(_W, src_v,
                                                                  dist_sq_override=_dist_sq_geo)
                    for k, v in dist_losses.items():
                        loss_dict[k] = v

                # ── #1 area-weighted weight smoothness (Mesh2Animation) ──
                # density-unbiased 1-ring ‖ΔW‖²; faces invalid under subsample → skip.
                if getattr(opts, 'lambda_wlap', 0) > 0 and not is_permed and _W is not None:
                    for k, v in self.model.weight_smoothness_loss(
                            _W, src_v, batch.faces).items():
                        loss_dict[k] = v

                # ── L/R skinning-weight symmetry prior (network equivariance) ──
                # W is identity-level skinning; for a bilaterally-symmetric face
                # topology it should be L/R symmetric (genuine asymmetry lives in
                # the per-frame joint transforms, not W). Full-mesh batches only.
                # Two modes (see --w_mirror_mode): 'vertex' = per-vertex mirror,
                # 'moment' = map-free per-joint moment matching.
                if (getattr(opts, 'lambda_w_mirror', 0) > 0 and not is_permed
                        and _W is not None and self._joint_swap_idx is not None):
                    _wmode = getattr(opts, 'w_mirror_mode', 'vertex')
                    if _wmode == 'moment':
                        # map-free: match each joint's weight-field moments to its
                        # mirror joint (mass/centroid/covariance over vertex positions).
                        _swp = self._joint_swap_idx
                        _eps = 1e-6
                        _p = src_v                                       # [B,N,3]
                        _m = _W.sum(dim=1)                               # [B,J] 0th (mass)
                        _wp = torch.einsum('bnj,bnd->bjd', _W, _p)       # [B,J,3] 1st
                        _c = _wp / (_m.unsqueeze(-1) + _eps)             # centroid
                        _pp = _p.unsqueeze(-1) * _p.unsqueeze(-2)        # [B,N,3,3]
                        _S = torch.einsum('bnj,bnde->bjde', _W, _pp) / (_m[..., None, None] + _eps)
                        _cov = _S - _c.unsqueeze(-1) * _c.unsqueeze(-2)  # [B,J,3,3] 2nd central
                        _flip = torch.tensor([-1., 1., 1.], device=_W.device)
                        _c_s = _c.index_select(1, _swp) * _flip          # flip_x(centroid_swap)
                        _cov_s = _cov.index_select(1, _swp).clone()      # M·cov·Mᵀ, M=diag(-1,1,1)
                        _cov_s[..., 0, 1] *= -1; _cov_s[..., 1, 0] *= -1
                        _cov_s[..., 0, 2] *= -1; _cov_s[..., 2, 0] *= -1
                        _m_s = _m.index_select(1, _swp)
                        _per_j = ((_m - _m_s).abs()
                                  + (_c - _c_s).norm(dim=-1)
                                  + (_cov - _cov_s).flatten(start_dim=-2).norm(dim=-1))  # [B,J]
                        if self._face_joint_idx:
                            _fj = torch.tensor(self._face_joint_idx, dtype=torch.long, device=_W.device)
                            _per_j = _per_j.index_select(1, _fj)
                        loss_dict['L_w_mirror'] = _per_j.mean()
                    else:
                        # per-vertex: W[v,:] ~ W[mir(v), swap(:)] (needs vertex map).
                        _Ncur = src_v.shape[1]
                        if _Ncur not in self._wmirror_map_by_N:
                            from scipy.spatial import cKDTree
                            # Prefer symmetric mean-mesh ref (identity-independent map);
                            # fall back to the batch's own neutral otherwise.
                            if _Ncur in self._wmirror_ref_by_N:
                                _Vref = self._wmirror_ref_by_N[_Ncur]; _rtag = 'mean-mesh'
                            else:
                                _Vref = src_v[0].detach().cpu().numpy(); _rtag = 'batch-mesh'
                            _Vc = _Vref - _Vref.mean(0)
                            _Vm = _Vc.copy(); _Vm[:, 0] *= -1
                            _d, _mir = cKDTree(_Vc).query(_Vm)
                            self._wmirror_map_by_N[_Ncur] = torch.tensor(
                                _mir, dtype=torch.long, device=src_v.device)
                            print(f"[L_w_mirror] built vertex mirror map N={_Ncur} "
                                  f"src={_rtag} (mean pair dist {float(_d.mean()):.2e})")
                        _mirT = self._wmirror_map_by_N[_Ncur]
                        _Wm = _W.index_select(1, _mirT).index_select(2, self._joint_swap_idx)
                        loss_dict['L_w_mirror'] = (_W - _Wm).abs().sum(dim=-1).mean()

                # ── #3 reference-weight prior (Mesh2Animation L_id) ──────
                # MSE to closest-bone one-hot; needs predicted joint_pos.
                if (getattr(opts, 'lambda_wref', 0) > 0
                        and _W is not None and _joint_pos is not None):
                    for k, v in self.model.weight_ref_loss(
                            _W, src_v, _joint_pos).items():
                        loss_dict[k] = v

                # ── Simplicits eq7-inspired weight orthogonality (adapted) ──
                # off-diagonal Gram of per-joint weight columns -> 0:
                # decorrelate joints' influence over vertices => disjoint,
                # per-vertex-sparse skinning. diag NOT forced to 1 (our W is
                # partition-of-unity, unlike Simplicits free eigenmodes).
                # Valid on subsampled batches (MC over present vertices).
                if getattr(opts, 'lambda_ortho', 0) > 0 and _W is not None:
                    _Wm = _W                                          # [B, N, J]
                    _G = torch.bmm(_Wm.transpose(1, 2), _Wm) / _Wm.shape[1]  # [B, J, J]
                    _Jn = _G.shape[-1]
                    _off = _G * (1.0 - torch.eye(_Jn, device=_G.device, dtype=_G.dtype))
                    loss_dict['L_ortho'] = (_off ** 2).sum(dim=(1, 2)).mean()

                # ── #2 GT-free weight-quality metric (diagnostic, no loss) ─
                if (getattr(opts, 'w_metric', 0) and not is_permed
                        and _W is not None):
                    _wq = self.model.weight_quality_metric(_W, src_v, batch.faces)
                    if _wq is not None:
                        running["metric-W_smooth"] += float(_wq)

                # ── Bone-Mesh Containment: bind-pose joints ──────────────
                # Hinge penalty when a bind joint pops out of the (subsampled)
                # source mesh. Uses dataloader-provided src_n; works permed too.
                if (getattr(opts, 'lambda_inside_bind', 0) > 0
                        and _joint_pos is not None):
                    for k, v in self.model.inside_mesh_loss(
                            _joint_pos, src_v, src_n).items():
                        loss_dict[f'L_inside_bind'] = v

                # ── Bone-Mesh Containment: deformed joint positions ──────
                # Deformed joint = T_world[j][:3, 3]. Check against pred_lbs
                # (deformed mesh). Skip when permed: pred_lbs faces invalid
                # → kNN-PCA normals would be needed, deferred.
                if (getattr(opts, 'lambda_inside_def', 0) > 0
                        and not is_permed
                        and _need_extras and 'T_world' in _extras):
                    from utils.mesh_utils import calc_norm_torch
                    _Tw = _extras['T_world']                # [B, J, 4, 4]
                    _def_jp = _Tw[..., :3, 3]                # [B, J, 3]
                    _pred_n = calc_norm_torch(pred_lbs, batch.faces, at='verts')
                    for k, v in self.model.inside_mesh_loss(
                            _def_jp, pred_lbs, _pred_n).items():
                        loss_dict[f'L_inside_def'] = v

                # ── Hierarchy locality loss (uses pre-computed W) ────────
                if opts.lambda_hier > 0:
                    _md = batch.mesh_data if hasattr(batch, 'mesh_data') else None
                    _perm = getattr(batch, 'perm_idx', None)
                    hier_losses = self.model.hierarchy_locality_loss(
                        _W, src_v.shape[1], mesh_data=_md, perm_idx=_perm,
                        margin=opts.hier_margin)
                    for k, v in hier_losses.items():
                        loss_dict[k] = v

                # ── Cross-id retarget loss (Track B) ─────────────────────
                if (getattr(opts, 'lambda_cross_retarget', 0) > 0
                        and self._cross_pair_loader is not None):
                    # Draw from MF pair loader with prob cross_pair_mf_prob
                    # (delta-transfer pseudo-GT), else from the ICT pair loader.
                    _cp_is_mf = (getattr(self, '_cross_pair_mf_loader', None) is not None
                                 and np.random.random() < getattr(opts, 'cross_pair_mf_prob', 0))
                    if _cp_is_mf:
                        try:
                            cb = next(self._cross_pair_mf_iter)
                        except StopIteration:
                            self._cross_pair_mf_iter = iter(self._cross_pair_mf_loader)
                            cb = next(self._cross_pair_mf_iter)
                    else:
                        try:
                            cb = next(self._cross_pair_iter)
                        except StopIteration:
                            self._cross_pair_iter = iter(self._cross_pair_loader)
                            cb = next(self._cross_pair_iter)
                    cb = cb.to(self.device)
                    B_c, V_c, _ = cb.tgt_template.shape

                    # NFS feat for tgt id (W uses tgt's identity)
                    _tgt_nfs = None
                    if opts.nfs_feat_dir and self._nfs_feat_cache:
                        _feats = []
                        for b in range(B_c):
                            _idn = cb.tgt_id_name[b]
                            if _idn in self._nfs_feat_cache:
                                _f = self._nfs_feat_cache[_idn]
                                if _f.shape[0] > V_c: _f = _f[:V_c]
                                if self._nfs_on_cpu: _f = _f.to(self.device, non_blocking=True)
                                _feats.append(_f)
                            else:
                                _feats.append(torch.zeros(V_c, self._nfs_feat_dim, device=self.device))
                        _tgt_nfs = torch.stack(_feats, dim=0)

                    # Per-id bind_pos cache for tgt (Phase B GT > legacy > per-topo mean)
                    _tgt_bp = None
                    _bp_items = []
                    for b in range(B_c):
                        _idn = cb.tgt_id_name[b]
                        if _idn in self._per_id_bind_pose_gt:
                            _bp_items.append(self._per_id_bind_pose_gt[_idn])
                        elif _idn in self._bind_pos_cache:
                            _bp_items.append(self._bind_pos_cache[_idn])
                        elif hasattr(self.model, 'bind_pos_target_ict'):
                            _bp_items.append(self.model.bind_pos_target_ict)
                        else:
                            _bp_items = None; break
                    if _bp_items is not None and len(_bp_items) == B_c:
                        _tgt_bp = torch.stack(_bp_items, dim=0).detach()

                    # geo_dist² for tgt (topology follows the drawn pair loader)
                    _cp_topo = 'mf' if _cp_is_mf else 'ict'
                    _tgt_geo = None
                    if _cp_topo in self._geo_dist_per_topo:
                        _gd = self._geo_dist_per_topo[_cp_topo].t()      # [V, J]
                        if _gd.shape[0] >= V_c:
                            _gd = _gd[:V_c]
                            _tgt_geo = (_gd.unsqueeze(0).expand(B_c, -1, -1)) ** 2

                    # ── Subsample cross-retarget batch ─────────────────
                    # Same perm for src and tgt (both ICT, same N). All cb
                    # tensors gather; _tgt_nfs and _tgt_geo also gather; _tgt_bp
                    # is per-joint so unaffected. _tgt_bp / V_c updated.
                    _cb_permed = False
                    if (getattr(opts, 'subsample_ratio', 0) > 0
                            and cb.src_template.shape[1] == cb.tgt_template.shape[1]):
                        cb.id_name = cb.src_id_name   # topo detect in _build_subsample_perm
                        _cperm = self._build_subsample_perm(
                            cb.src_template, cb,
                            opts.subsample_ratio, opts.subsample_mode)
                        if _cperm is not None:
                            _cb_permed = True
                            _idx3 = _cperm.unsqueeze(-1).expand(-1, -1, 3)
                            cb.src_template        = torch.gather(cb.src_template, 1, _idx3)
                            cb.src_template_normal = torch.gather(cb.src_template_normal, 1, _idx3)
                            cb.src_vertices        = torch.gather(cb.src_vertices, 1, _idx3)
                            cb.src_vertices_normal = torch.gather(cb.src_vertices_normal, 1, _idx3)
                            cb.tgt_template        = torch.gather(cb.tgt_template, 1, _idx3)
                            cb.tgt_template_normal = torch.gather(cb.tgt_template_normal, 1, _idx3)
                            cb.tgt_vertices        = torch.gather(cb.tgt_vertices, 1, _idx3)
                            cb.tgt_vertices_normal = torch.gather(cb.tgt_vertices_normal, 1, _idx3)
                            if _tgt_nfs is not None:
                                _idxF = _cperm.unsqueeze(-1).expand(-1, -1, _tgt_nfs.shape[-1])
                                _tgt_nfs = torch.gather(_tgt_nfs, 1, _idxF)
                            if _tgt_geo is not None:
                                _idxJ = _cperm.unsqueeze(-1).expand(-1, -1, _tgt_geo.shape[-1])
                                _tgt_geo = torch.gather(_tgt_geo, 1, _idxJ)
                            V_c = cb.tgt_template.shape[1]

                    # Forward retarget: A → B
                    pred_tgt = self.model.retarget(
                        cb.src_template, cb.src_template_normal,
                        cb.src_vertices,  cb.src_vertices_normal,
                        cb.tgt_template,  cb.tgt_template_normal,
                        tgt_nfs_feat=_tgt_nfs,
                        tgt_bind_pos_cache=_tgt_bp,
                        tgt_dist_sq_geo=_tgt_geo,
                    )
                    loss_dict["L_cross_retarget"] = F.mse_loss(pred_tgt, cb.tgt_vertices)

                    # ── Cyclic retarget: B → A reconstruction ────────────
                    # Use pred_tgt (B's predicted deformed) as the new source-deformed,
                    # cb.tgt_template (B's neutral) as new source-neutral, and
                    # cb.src_template (A's neutral) as the new target.
                    # Expected: reconstructed src_def should match cb.src_vertices.
                    # Notes:
                    #  - pred_tgt's normals are unknown at runtime; we use cb.tgt_vertices_normal
                    #    as a practical proxy (close to true normals once pred_tgt ≈ GT).
                    #  - NFS feat / bind pos cache must reflect the SOURCE id (A) now in the
                    #    target slot of the retarget call.
                    pred_src_recon = None
                    if getattr(opts, 'lambda_cross_cyclic', 0) > 0:
                        _src_nfs = None
                        if opts.nfs_feat_dir and self._nfs_feat_cache:
                            _feats = []
                            for b in range(B_c):
                                _idn = cb.src_id_name[b]
                                if _idn in self._nfs_feat_cache:
                                    _f = self._nfs_feat_cache[_idn]
                                    if _f.shape[0] > V_c: _f = _f[:V_c]
                                    if self._nfs_on_cpu: _f = _f.to(self.device, non_blocking=True)
                                    _feats.append(_f)
                                else:
                                    _feats.append(torch.zeros(V_c, self._nfs_feat_dim, device=self.device))
                            _src_nfs = torch.stack(_feats, dim=0)

                        _src_bp = None
                        _bp_items_src = []
                        for b in range(B_c):
                            _idn = cb.src_id_name[b]
                            if _idn in self._per_id_bind_pose_gt:
                                _bp_items_src.append(self._per_id_bind_pose_gt[_idn])
                            elif _idn in self._bind_pos_cache:
                                _bp_items_src.append(self._bind_pos_cache[_idn])
                            elif hasattr(self.model, 'bind_pos_target_ict'):
                                _bp_items_src.append(self.model.bind_pos_target_ict)
                            else:
                                _bp_items_src = None; break
                        if _bp_items_src is not None and len(_bp_items_src) == B_c:
                            _src_bp = torch.stack(_bp_items_src, dim=0).detach()

                        pred_src_recon = self.model.retarget(
                            cb.tgt_template,        cb.tgt_template_normal,       # new src_neu (B)
                            pred_tgt,               cb.tgt_vertices_normal,       # new src_def (B)
                            cb.src_template,        cb.src_template_normal,       # new tgt_neu (A)
                            tgt_nfs_feat=_src_nfs,
                            tgt_bind_pos_cache=_src_bp,
                            tgt_dist_sq_geo=_tgt_geo,                              # ICT geo (shared)
                        )
                        loss_dict["L_cross_cyclic"] = F.mse_loss(pred_src_recon, cb.src_vertices)

                # ── Total loss ───────────────────────────────────────────
                loss_lambda = {
                    "recon-lbs": opts.lambda_vert,
                    "recon-neu": opts.lambda_neu,
                    "recon-normal": opts.lambda_normal,
                    "recon-curvature": opts.lambda_curvature,
                    "L_W_init": lambda_init,
                    "L_bind_init": lambda_init,
                    "L_bind_reg": opts.lambda_bind_reg,
                    "L_helper_residual": getattr(opts, 'lambda_helper_residual', 0.0),
                    "L_bind_residual": getattr(opts, 'lambda_bind_residual', 0.0),
                    "L_mirror": getattr(opts, 'lambda_mirror', 0.0),
                    "L_w_mirror": getattr(opts, 'lambda_w_mirror', 0.0),
                    "L_rwc_init": opts.lambda_rwc,
                    "L_rwc_min": opts.lambda_rwc,
                    "L_hier": opts.lambda_hier,
                    "L_dist": opts.lambda_dist,
                    "L_wlap": getattr(opts, 'lambda_wlap', 0.0),
                    "L_wref": getattr(opts, 'lambda_wref', 0.0),
                    "L_ortho": getattr(opts, 'lambda_ortho', 0.0),
                    "L_inside_bind": getattr(opts, 'lambda_inside_bind', 0.0),
                    "L_inside_def":  getattr(opts, 'lambda_inside_def', 0.0),
                    "L_sigma": opts.lambda_sigma_reg,
                    "L_net_center": opts.lambda_net_center,
                    "L_cross_retarget": opts.lambda_cross_retarget,
                    "L_cross_cyclic": getattr(opts, 'lambda_cross_cyclic', 0.0),
                }
                loss = sum(loss_dict[k] * loss_lambda.get(k, 0.0) for k in loss_dict)
                _amp_ctx.__exit__(None, None, None)   # backward must run outside autocast
                if _prof:
                    _prof_sync(); _t_fwd = time.perf_counter()
                    _prof_acc['fwd'] += _t_fwd - _t_data
                loss.backward()
                if _prof:
                    _prof_sync(); _t_bwd = time.perf_counter()
                    _prof_acc['bwd'] += _t_bwd - _t_fwd
                self.optimizer.step()
                if _prof:
                    _prof_sync(); _t_opt = time.perf_counter()
                    _prof_acc['opt'] += _t_opt - _t_bwd
                    _prof_prev = _t_opt
                    _prof_n += 1
                    if _prof_n % 50 == 0:
                        _n = 50
                        print(f"[profile] data {_prof_acc['data']/_n*1e3:6.1f}ms | "
                              f"fwd {_prof_acc['fwd']/_n*1e3:6.1f}ms | "
                              f"bwd {_prof_acc['bwd']/_n*1e3:6.1f}ms | "
                              f"opt {_prof_acc['opt']/_n*1e3:6.1f}ms | "
                              f"total {sum(_prof_acc.values())/_n*1e3:6.1f}ms/batch")
                        for _k in _prof_acc:
                            _prof_acc[_k] = 0.0
                    if _tprof is not None:
                        _tprof.step()
                        if _prof_n == 16:
                            _tprof.stop()
                            print("\n[torch.profiler] top ops by CUDA time "
                                  "(steps 12-15):")
                            print(_tprof.key_averages().table(
                                sort_by='cuda_time_total', row_limit=25))
                            _tprof = None
                            self._torch_prof_done = True

                for k in running:
                    if k != "total" and k in loss_dict:
                        running[k] += (loss_dict[k] * loss_lambda.get(k, 1.0)).item()
                    elif k == "init-W" and "L_W_init" in loss_dict:
                        running[k] += (loss_dict["L_W_init"] * lambda_init).item()
                    elif k == "init-bind" and "L_bind_init" in loss_dict:
                        running[k] += (loss_dict["L_bind_init"] * lambda_init).item()
                running["total"] += loss.item()
                cnt += 1
                pbar.set_description(
                    f"[{epoch:03d}] lbs:{loss_dict['recon-lbs']:.4e} init:{lambda_init:.2f}")

                _interv = max(1, round(_len_active / 10))
                if idx % _interv == 1:
                    inv = 1.0 / cnt
                    log_text = f"[{epoch:03d}/{epochs:03d}][{idx:04d}][Train] "
                    log_text += " ".join(f"{k}: {v*inv:.6e}" for k, v in running.items())
                    logger.write(log_text + "\n")

                    # Skip mesh vis when subsampled — batch.faces refers to original
                    # vertex indices and would index out-of-range on subsampled verts.
                    if not is_permed:
                        HB = BS // 2
                        _d = lambda t: t.detach().float().cpu()  # .float(): bf16(AMP)→fp32 for numpy/vis
                        _s = lambda i: min(i, BS-1)
                        faces_cpu = batch.faces.cpu()
                        v_list = [
                            _d(gt_v[0]),        _d(gt_v[_s(1)]),
                            _d(gt_v[_s(HB)]),   _d(gt_v[BS-1]),
                            _d(pred_lbs[0]),    _d(pred_lbs[_s(1)]),
                            _d(pred_lbs[_s(HB)]), _d(pred_lbs[BS-1]),
                        ]
                        f_list = [faces_cpu] * len(v_list)
                        plot_image_array(
                            v_list, f_list, rot_list=[[0,0,0]]*len(v_list),
                            size=1, bg_black=False, mode='shade',
                            logdir=f"{opts.log_dir}/img/train/mesh",
                            name=f"{epoch:03d}_{idx:04d}", save=True)

                    # Cross-retarget vis — match main mesh vis pattern: 4 GT + 4 pred.
                    # Skip when the cross-retarget batch was subsampled (cb.faces
                    # still refers to the original full vertex set → out-of-range).
                    if (getattr(opts, 'lambda_cross_retarget', 0) > 0
                            and self._cross_pair_loader is not None
                            and not _cb_permed):
                        _d = lambda t: t.detach().float().cpu()  # .float(): bf16(AMP)→fp32 for numpy/vis
                        B_c_vis = cb.src_template.shape[0]
                        _sc = lambda i: min(i, B_c_vis - 1)
                        HBc = B_c_vis // 2
                        faces_cr = cb.faces.cpu()
                        # cross_retarget.png: 4 GT (cb.tgt_vertices) + 4 pred (pred_tgt)
                        v_list_cr = [
                            _d(cb.tgt_vertices[0]),     _d(cb.tgt_vertices[_sc(1)]),
                            _d(cb.tgt_vertices[_sc(HBc)]), _d(cb.tgt_vertices[B_c_vis-1]),
                            _d(pred_tgt[0]),            _d(pred_tgt[_sc(1)]),
                            _d(pred_tgt[_sc(HBc)]),     _d(pred_tgt[B_c_vis-1]),
                        ]
                        f_list_cr = [faces_cr] * len(v_list_cr)
                        plot_image_array(
                            v_list_cr, f_list_cr, rot_list=[[0,0,0]]*len(v_list_cr),
                            size=1, bg_black=False, mode='shade',
                            logdir=f"{opts.log_dir}/img/train/cross_retarget",
                            name=f"{epoch:03d}_{idx:04d}", save=True)
                        # cross_cyclic.png (if active): 4 GT (cb.src_vertices) + 4 recon (pred_src_recon)
                        if pred_src_recon is not None:
                            v_list_cyc = [
                                _d(cb.src_vertices[0]),     _d(cb.src_vertices[_sc(1)]),
                                _d(cb.src_vertices[_sc(HBc)]), _d(cb.src_vertices[B_c_vis-1]),
                                _d(pred_src_recon[0]),      _d(pred_src_recon[_sc(1)]),
                                _d(pred_src_recon[_sc(HBc)]), _d(pred_src_recon[B_c_vis-1]),
                            ]
                            f_list_cyc = [faces_cr] * len(v_list_cyc)
                            plot_image_array(
                                v_list_cyc, f_list_cyc, rot_list=[[0,0,0]]*len(v_list_cyc),
                                size=1, bg_black=False, mode='shade',
                                logdir=f"{opts.log_dir}/img/train/cross_retarget",
                                name=f"{epoch:03d}_{idx:04d}_cyclic", save=True)

                if opts.debug:
                    break

            if epoch != 0:
                self.scheduler.step()
            if writer_train:
                for k, v in running.items():
                    writer_train.add_scalar(k, v / cnt, epoch)

            if epoch % opts.save_interval == 0:
                torch.save(self.model.state_dict(),
                           f'{opts.log_dir}/model_hlbs_{epoch:03d}.pth')

            if epoch % opts.eval_iter == 0:
                self.vis_loader.visualize(
                    self.model, None, epoch,
                    save_dir=f'{opts.log_dir}/img/eval',
                    mode='hlbs',
                    smooth_n_iter=opts.smooth_n_iter,
                    no_t_mask=opts.no_t_mask,
                    nfs_feat_cache=self._nfs_feat_cache if hasattr(self, '_nfs_feat_cache') else None,
                )
                if opts.curriculum and epoch > 0:
                    self._visualize_curriculum_bases(
                        epoch, save_dir=f'{opts.log_dir}/img/eval_bases')

            # ── Valid ────────────────────────────────────────────────────
            if epoch == 0 or epoch % opts.val_every != 0:
                continue

            self.model.eval()
            running_val = {"recon-lbs": 0.0, "recon-neu": 0.0,
                           "recon-normal": 0.0, "L_bind_reg": 0.0,
                           "L_sigma": 0.0, "L_net_center": 0.0,
                           "total": 0.0}
            vcnt = 0

            pbar = tqdm(enumerate(valid_loader), total=len_valid, ncols=120,
                        desc=f"[{epoch:03d}] Valid FullPred")
            for idx, batch in pbar:
                batch = batch.to(self.device, non_blocking=True)
                vcnt += 1
                with torch.no_grad():
                    src_v  = batch.template
                    src_n  = batch.template_normal
                    gt_v   = batch.vertices
                    gt_n   = batch.vertices_normal

                    # Get NFS features for valid batch
                    _nfs_feat = None
                    if opts.nfs_feat_dir and self._nfs_feat_cache:
                        B_cur = src_v.shape[0]
                        N_cur = src_v.shape[1]
                        feats = []
                        for b in range(B_cur):
                            id_key = batch.id_name[b]
                            if id_key in self._nfs_feat_cache:
                                f = self._nfs_feat_cache[id_key]
                                if f.shape[0] > N_cur:
                                    f = f[:N_cur]
                                if self._nfs_on_cpu:
                                    f = f.to(self.device, non_blocking=True)
                                feats.append(f)
                            else:
                                feats.append(torch.zeros(N_cur, self._nfs_feat_dim, device=self.device))
                        _nfs_feat = torch.stack(feats, dim=0)

                    # Update DiffusionNet precomputes for val topology
                    if opts.dfn_skin or opts.dfn_bind or opts.dfn_exp:
                        N_cur = src_v.shape[1]
                        if not hasattr(self, '_dfn_cached_N') or self._dfn_cached_N != N_cur:
                            import trimesh as _tm
                            from utils.nfr_utils import get_dfn_info
                            _f = batch.faces[0].cpu().numpy() if batch.faces.dim() == 3 else batch.faces.cpu().numpy()
                            _v = src_v[0].cpu().numpy()
                            _mesh = _tm.Trimesh(vertices=_v, faces=_f, process=False)
                            _dfn = get_dfn_info(_mesh, cache_dir='dfn_cache', map_location=self.device)
                            self.model.update_dfn_precomputes(_dfn)
                            self._dfn_cached_N = N_cur

                    delta     = gt_v - src_v
                    src_in    = torch.cat([src_v, src_n], dim=-1)
                    deform_in = torch.cat([delta, gt_n, src_in], dim=-1)

                    # Per-id bind_pos cache lookup (val) ─ same as train
                    _bind_pos_cache_val = None
                    if self._bind_pos_cache:
                        _items = []; _all_hit = True
                        for b in range(src_v.shape[0]):
                            k = batch.id_name[b]
                            if k in self._bind_pos_cache:
                                _items.append(self._bind_pos_cache[k])
                            else:
                                _all_hit = False; break
                        if _all_hit:
                            _bind_pos_cache_val = torch.stack(_items, dim=0)

                    # Per-batch geodesic dist² (val) — same logic as train
                    _dist_sq_geo_val = None
                    if self._geo_dist_per_topo:
                        _per_b = []; _all_hit = True
                        for b in range(src_v.shape[0]):
                            idn = batch.id_name[b] if hasattr(batch, 'id_name') else ''
                            topo = ('ict'  if idn.startswith('ict_')  else
                                    'mf'   if idn.startswith('m--')   else
                                    'biwi' if idn.startswith('biwi_') else
                                    'coma' if idn.startswith('coma_') else None)
                            if topo and topo in self._geo_dist_per_topo:
                                _gd = self._geo_dist_per_topo[topo].t()       # [V, J]
                                if _gd.shape[0] != src_v.shape[1]:
                                    _gd = _gd[:src_v.shape[1]] if _gd.shape[0] > src_v.shape[1] else _gd
                                _per_b.append(_gd)
                            else:
                                _all_hit = False; break
                        if _all_hit:
                            _dist_sq_geo_val = (torch.stack(_per_b, dim=0)) ** 2

                    pred_lbs, _extras = self.model(
                        src_v, deform_in, source_normal=src_n,
                        nfs_feat=_nfs_feat,
                        bind_pos_cache=_bind_pos_cache_val,
                        dist_sq_geo=_dist_sq_geo_val,
                        return_extras=True)
                    _joint_pos_val = _extras.get('joint_pos')
                    _logit_net_val = _extras.get('logit_net')

                    target_v_val = gt_v
                    val_loss = F.mse_loss(target_v_val, pred_lbs).item() * opts.lambda_vert
                    running_val["recon-lbs"] += val_loss
                    running_val["total"]     += val_loss

                    if opts.lambda_neu > 0:
                        delta_zero = torch.zeros_like(src_v)
                        neu_deform_in = torch.cat([delta_zero, src_n, src_v, src_n], dim=-1)
                        pred_neutral = self.model(src_v, neu_deform_in, source_normal=src_n,
                                                  nfs_feat=_nfs_feat,
                                                  bind_pos_cache=_bind_pos_cache_val,
                                                  dist_sq_geo=_dist_sq_geo_val)
                        val_neu = F.mse_loss(src_v, pred_neutral).item() * opts.lambda_neu
                        running_val["recon-neu"] += val_neu
                        running_val["total"]     += val_neu

                    # ── recon-normal (mirror of train) ───────────────────
                    if opts.lambda_normal > 0:
                        from utils.mesh_utils import calc_norm_torch
                        _pn = calc_norm_torch(pred_lbs, batch.faces, at='verts')
                        _gn = calc_norm_torch(target_v_val, batch.faces, at='verts')
                        val_nrm = (1 - F.cosine_similarity(_pn, _gn, dim=-1)).mean().item() * opts.lambda_normal
                        running_val["recon-normal"] += val_nrm
                        running_val["total"]        += val_nrm

                    # ── L_bind_reg (per-id GT > per-topo mean fallback) ───
                    if (opts.lambda_bind_reg > 0 and _joint_pos_val is not None
                            and getattr(opts, 'bind_pose_mode', 'net') == 'net'):
                        B_cur = _joint_pos_val.shape[0]
                        _targets = []
                        for b in range(B_cur):
                            idn = batch.id_name[b] if hasattr(batch, 'id_name') else ''
                            if idn in self._per_id_bind_pose_gt:
                                _targets.append(self._per_id_bind_pose_gt[idn])
                            elif idn.startswith('ict_') and hasattr(self.model, 'bind_pos_target_ict'):
                                _targets.append(self.model.bind_pos_target_ict)
                            elif idn.startswith('m--') and hasattr(self.model, 'bind_pos_target_mf'):
                                _targets.append(self.model.bind_pos_target_mf)
                            else:
                                _targets.append(self.model.bind_pos_target)
                        _tgt_batch = torch.stack(_targets, dim=0)
                        if self._helper_joint_idx:
                            _J = _joint_pos_val.shape[1]
                            _nh = [j for j in range(_J) if j not in self._helper_joint_idx]
                            _nh_t = torch.tensor(_nh, dtype=torch.long, device=_joint_pos_val.device)
                            v_bind = F.mse_loss(
                                _joint_pos_val.index_select(1, _nh_t),
                                _tgt_batch.index_select(1, _nh_t)).item()
                        else:
                            v_bind = F.mse_loss(_joint_pos_val, _tgt_batch).item()
                        v_bind *= opts.lambda_bind_reg
                        running_val["L_bind_reg"] += v_bind
                        running_val["total"]      += v_bind

                    # ── L_sigma (parameter-only, but lambda-scaled) ───────
                    if opts.lambda_sigma_reg > 0 and getattr(self.model, 'use_gmm_hybrid', False):
                        v_sig = self.model.sigma_shrink_loss(active_idx=self._face_joint_idx).item() \
                                * opts.lambda_sigma_reg
                        running_val["L_sigma"] += v_sig
                        running_val["total"]   += v_sig

                    # ── L_net_center (geodesic or euclidean variant) ──────
                    if (opts.lambda_net_center > 0 and _logit_net_val is not None
                            and (_joint_pos_val is not None or _dist_sq_geo_val is not None)):
                        _Wn = F.softmax(_logit_net_val, dim=1)
                        if _dist_sq_geo_val is not None:
                            if self._face_joint_idx:
                                _fi = torch.tensor(self._face_joint_idx, dtype=torch.long,
                                                   device=_Wn.device)
                                _Wa = _Wn.index_select(2, _fi)
                                _Da = _dist_sq_geo_val.index_select(2, _fi)
                            else:
                                _Wa, _Da = _Wn, _dist_sq_geo_val
                            v_nc = (_Wa * _Da).sum(dim=1).mean().item()
                        else:
                            _mu = torch.einsum('bvj,bvk->bjk', _Wn, src_v)
                            if self._face_joint_idx:
                                _fi = torch.tensor(self._face_joint_idx, dtype=torch.long,
                                                   device=_mu.device)
                                _mu_a = _mu.index_select(1, _fi)
                                _jp_a = _joint_pos_val.index_select(1, _fi)
                            else:
                                _mu_a, _jp_a = _mu, _joint_pos_val
                            v_nc = F.mse_loss(_mu_a, _jp_a).item()
                        v_nc *= opts.lambda_net_center
                        running_val["L_net_center"] += v_nc
                        running_val["total"]        += v_nc

                pbar.set_description(f"[{epoch:03d}] val lbs: {val_loss:.5e}")

                if idx % interv_val == 0:
                    BS_v = batch.vertices.shape[0]
                    HB_v = BS_v // 2
                    _s = lambda i: min(i, BS_v-1)
                    faces_cpu = batch.faces[0].cpu() if batch.faces.dim() == 3 else batch.faces.cpu()
                    v_list = [
                        gt_v[0].cpu(),        gt_v[_s(1)].cpu(),
                        gt_v[_s(HB_v)].cpu(), gt_v[BS_v-1].cpu(),
                        pred_lbs[0].cpu(),        pred_lbs[_s(1)].cpu(),
                        pred_lbs[_s(HB_v)].cpu(), pred_lbs[BS_v-1].cpu(),
                    ]
                    f_list = [faces_cpu] * len(v_list)
                    plot_image_array(
                        v_list, f_list, rot_list=[[0,0,0]]*len(v_list),
                        size=1, bg_black=False, mode='shade',
                        logdir=f"{opts.log_dir}/img/valid/mesh",
                        name=f"{epoch:03d}_{idx:04d}", save=True)

                if opts.debug:
                    break

            if writer_valid:
                for k, v in running_val.items():
                    writer_valid.add_scalar(k, v / vcnt, epoch)

            total_val = running_val["total"] / vcnt
            # Per-component val line — written every val epoch (incl. best-update)
            parts = " ".join(f"{k}: {running_val[k]/vcnt:.6e}"
                             for k in running_val if k != "total")
            updated = total_val < BEST_LOSS
            cur_best   = total_val if updated else BEST_LOSS
            cur_best_e = epoch     if updated else BEST_EPOCH
            val_line = (f"[{epoch:03d}] Val: {parts} total: {total_val:.6e} "
                        f"(Best: {cur_best:.6e} [{cur_best_e}])")
            print(val_line); logger.write(val_line + "\n")
            if updated:
                BEST_LOSS  = total_val
                BEST_EPOCH = epoch
                torch.save(self.model.state_dict(), f'{opts.log_dir}/model_hlbs_best.pth')
                msg = f"[{epoch:03d}] Best updated: {BEST_LOSS:.6e}"
                print(msg); logger.write(msg + "\n")


def _ensure_pytorch3d_gpu(skip_install=False):
    """Verify pytorch3d's FPS CUDA kernel is available; reinstall from source if not.

    mix4 subsampling rolls FPS. Without the CUDA kernel, pytorch3d's
    sample_farthest_points falls back to a Python loop with per-step .item()
    syncs — seconds per batch. On a CUDA box we require the GPU build.

    Probe + (re)build run in subprocesses so the parent picks up a fresh
    pytorch3d after install. Build env is set per GPU arch (V100=7.0,
    A5000/RTX30=8.6, A100=8.0, RTX40=8.9, etc.) so the compile doesn't
    waste time on unrelated archs and avoids default-arch errors.

    Opt out by setting env var SKIP_P3D_INSTALL=1 (probes only, no install).
    """
    import os as _os
    import subprocess
    if not torch.cuda.is_available():
        return
    probe = (
        'import torch;'
        'from pytorch3d.ops import sample_farthest_points;'
        'sample_farthest_points(torch.randn(1,256,3,device="cuda"),K=32);'
        'print("P3D_GPU_OK")'
    )
    chk = subprocess.run([sys.executable, '-c', probe],
                         capture_output=True, text=True)
    if chk.returncode == 0 and 'P3D_GPU_OK' in chk.stdout:
        print('[pytorch3d] GPU FPS kernel verified.')
        return
    if skip_install or _os.environ.get('SKIP_P3D_INSTALL', '0') == '1':
        print('[pytorch3d] GPU FPS unavailable; install skipped '
              '(--skip_p3d_install or SKIP_P3D_INSTALL=1) — '
              'falling back to slow Python FPS.')
        print(f'  reason: {(chk.stderr or chk.stdout).strip()[-300:]}')
        return
    # Build env: derive CUDA arch from the actual GPU; align CUDA_HOME with
    # the toolkit matching torch's bundled CUDA version (cu124 → cuda-12.4).
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"{major}.{minor}"
    torch_cuda = (torch.version.cuda or '').replace('.', '-')   # '12.4' → '12-4'
    candidate_cuda_homes = [
        _os.environ.get('CUDA_HOME', ''),
        f"/usr/local/cuda-{torch.version.cuda}" if torch.version.cuda else '',
        '/usr/local/cuda',
    ]
    cuda_home = next((p for p in candidate_cuda_homes if p and _os.path.isdir(p)), '')
    env = _os.environ.copy()
    if cuda_home:
        env['CUDA_HOME'] = cuda_home
        env['PATH'] = f"{cuda_home}/bin:" + env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = f"{cuda_home}/lib64:" + env.get('LD_LIBRARY_PATH', '')
    env['TORCH_CUDA_ARCH_LIST'] = arch
    env['FORCE_CUDA'] = '1'
    print(f'[pytorch3d] GPU FPS unavailable — building from source '
          f'(arch={arch}, CUDA_HOME={cuda_home or "<unset>"})')
    print(f'  reason: {(chk.stderr or chk.stdout).strip()[-300:]}')
    subprocess.run([
        sys.executable, '-m', 'pip', 'uninstall', '-y', 'pytorch3d',
    ], check=False, env=env)
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
        'fvcore', 'iopath',
    ], check=False, env=env)
    subprocess.run([
        sys.executable, '-m', 'pip', 'install',
        '--no-build-isolation', '--no-cache-dir',
        'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9',
    ], check=False, env=env)
    chk2 = subprocess.run([sys.executable, '-c', probe],
                          capture_output=True, text=True)
    if chk2.returncode == 0 and 'P3D_GPU_OK' in chk2.stdout:
        print('[pytorch3d] reinstall OK — GPU FPS kernel now available.')
    else:
        print('[pytorch3d] WARNING: still no GPU FPS after reinstall; '
              'mix4 FPS will use the slow Python fallback.')
        print(f'  reason: {(chk2.stderr or chk2.stdout).strip()[-300:]}')


if __name__ == "__main__":
    opts = Options()
    _ensure_pytorch3d_gpu(skip_install=getattr(opts, 'skip_p3d_install', False))

    if os.path.exists(opts.config):
        opts_yaml = yaml.load(open(opts.config), Loader=yaml.FullLoader)
        opts_dict = vars(opts)
        # yaml is the base; override ONLY with args explicitly passed on the CLI.
        # (vars(opts) also holds argparse defaults — using it wholesale would let
        #  every default clobber the yaml value, which neutered --config before.)
        cli_keys = {tok[2:].split('=')[0]
                    for tok in sys.argv[1:] if tok.startswith('--')}
        merged = dict(opts_yaml)
        for k, v in opts_dict.items():
            if k in cli_keys or k not in merged:
                merged[k] = v
        opts = argparse.Namespace(**merged)

    if opts.target == 'smooth_gt':
        assert opts.smooth_n_iter > 0, "target=smooth_gt requires --smooth_n_iter > 0"

    trainer = HLBSTrainer(opts)
    if opts.full_prediction:
        trainer.train_full_prediction(epochs=opts.max_epoch)
    else:
        trainer.train(epochs=opts.max_epoch)
