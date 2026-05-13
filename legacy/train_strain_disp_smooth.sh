#!/bin/bash
## v10: LBS + Strain-conditioned DispNet with Smooth GT Decomposition
## LBS targets smooth_GT, DispNet targets wrinkle (GT - smooth_GT)
## Based on v9 (train_strain_disp.sh) with --smooth_n_iter flag

set -e

########################################
## Experiment: smooth_n_iter=16 (default choice)
## With strain, stop-grad
########################################
CMD_SMOOTH16="python train_CBD.py \
    --max_epoch 500 \
    --tb \
    --lr 1e-4 \
    --sc_step 100000 \
    --in_type 1 \
    --out_type 1 \
    --batch_size 16 \
    --version 9 \
    --use_strain \
    --strain_dim 1 \
    --smooth_n_iter 16 \
    --last_activation 'relu' \
    --use_lbs_joint_center \
    --num_lbs_joints 64 \
    --save_interval 50 \
    --use_data1 \
    --no_t_mask"

########################################
## Experiment: smooth_n_iter=8
########################################
CMD_SMOOTH8="python train_CBD.py \
    --max_epoch 500 \
    --tb \
    --lr 1e-4 \
    --sc_step 100000 \
    --in_type 1 \
    --out_type 1 \
    --batch_size 16 \
    --version 9 \
    --use_strain \
    --strain_dim 1 \
    --smooth_n_iter 8 \
    --last_activation 'relu' \
    --use_lbs_joint_center \
    --num_lbs_joints 64 \
    --save_interval 50 \
    --use_data1 \
    --no_t_mask"

########################################
## Experiment: smooth_n_iter=32
########################################
CMD_SMOOTH32="python train_CBD.py \
    --max_epoch 500 \
    --tb \
    --lr 1e-4 \
    --sc_step 100000 \
    --in_type 1 \
    --out_type 1 \
    --batch_size 16 \
    --version 9 \
    --use_strain \
    --strain_dim 1 \
    --smooth_n_iter 32 \
    --last_activation 'relu' \
    --use_lbs_joint_center \
    --num_lbs_joints 64 \
    --save_interval 50 \
    --use_data1 \
    --no_t_mask"

########################################
## Run selected experiment
## Change CMD to CMD_SMOOTH8, CMD_SMOOTH16, or CMD_SMOOTH32
########################################
echo "Running: $CMD_SMOOTH16"
eval $CMD_SMOOTH16
