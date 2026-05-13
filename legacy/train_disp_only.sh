#!/bin/bash
## DispNet-only pre-training with GT oracle signals
## Experiments: all 5 strain modes x smooth_n_iter

set -e

COMMON_ARGS="\
    --config configs/train.yml \
    --tb \
    --lr 1e-4 \
    --sc_step 100000 \
    --in_type 1 \
    --batch_size 16 \
    --version 9 \
    --use_strain \
    --last_activation relu \
    --use_lbs_joint_center \
    --num_lbs_joints 64 \
    --save_interval 50 \
    --eval_iter 25 \
    --use_data1 \
    --no_t_mask \
    --smooth_n_iter 16 \
    --max_epoch 300"

########################################
## Strain mode: norm (dim=1)
########################################
CMD_NORM="python train_disp_only.py $COMMON_ARGS --strain_mode norm"

########################################
## Strain mode: norm_trace (dim=2)
########################################
CMD_NORM_TRACE="python train_disp_only.py $COMMON_ARGS --strain_mode norm_trace"

########################################
## Strain mode: full (dim=6)
########################################
CMD_FULL="python train_disp_only.py $COMMON_ARGS --strain_mode full"

########################################
## Strain mode: principal (dim=3)
########################################
CMD_PRINCIPAL="python train_disp_only.py $COMMON_ARGS --strain_mode principal"

########################################
## Strain mode: local (dim=3)
########################################
CMD_LOCAL="python train_disp_only.py $COMMON_ARGS --strain_mode local"

########################################
## Smooth n_iter variants (with norm mode)
########################################
CMD_NORM_S8="python train_disp_only.py $COMMON_ARGS --strain_mode norm --smooth_n_iter 8"
CMD_NORM_S32="python train_disp_only.py $COMMON_ARGS --strain_mode norm --smooth_n_iter 32"

########################################
## v2: principal strain + true EDD + t_mask
########################################
CMD_PRINCIPAL_V2="python train_disp_only.py $COMMON_ARGS \
    --strain_mode principal --use_true_edd --use_t_mask"

CMD_NORM_TRACE_V2="python train_disp_only.py $COMMON_ARGS \
    --strain_mode norm_trace --use_true_edd --use_t_mask"

########################################
## v3: v2 + z-score normalization
########################################
CMD_PRINCIPAL_V3="python train_disp_only.py $COMMON_ARGS \
    --strain_mode principal --use_true_edd --use_t_mask \
    --norm_stats_file norm_stats/norm_stats_principal_s16_trueedd_masked.npz"

########################################
## Run: change CMD variable below
########################################
echo "Running: $CMD_PRINCIPAL_V3"
eval $CMD_PRINCIPAL_V3
