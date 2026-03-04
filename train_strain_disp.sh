#!/bin/bash
## v9: LBS + Strain-conditioned Displacement Network
## Progressive experiments:
##   Exp 1: Naive baseline (no strain) — DispNet(lbs_deformed) → delta
##   Exp 2: With strain (||E||_F) as additional input
## Training mode: joint training (LBS + DispNet), stop-grad on strain by default

set -e

########################################
## Experiment 1: Naive (no strain)
## DispNet input: lbs_deformed_pos + lbs_deformed_norm (6D)
########################################
CMD_EXP1="python train_CBD.py \
    --max_epoch 500 \
    --tb \
    --lr 1e-4 \
    --sc_step 100000 \
    --in_type 1 \
    --out_type 1 \
    --batch_size 16 \
    --version 9 \
    --last_activation 'relu' \
    --use_lbs_joint_center \
    --num_lbs_joints 64 \
    --save_interval 50 \
    --use_data1 \
    --data_toggle \
    --no_t_mask"

########################################
## Experiment 2: With strain (||E||_F)
## DispNet input: lbs_deformed_pos + lbs_deformed_norm + strain (7D)
## stop-grad on strain (default: strain_full_grad=False)
########################################
CMD_EXP2="python train_CBD.py \
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
    --last_activation 'relu' \
    --use_lbs_joint_center \
    --num_lbs_joints 64 \
    --save_interval 50 \
    --use_data1 \
    --data_toggle \
    --no_t_mask"

########################################
## Experiment 2b: With strain + full grad through LBS
## (strain_full_grad=True — gradient flows from DispNet through strain to LBS)
########################################
CMD_EXP2B="python train_CBD.py \
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
    --strain_full_grad \
    --last_activation 'relu' \
    --use_lbs_joint_center \
    --num_lbs_joints 64 \
    --save_interval 50 \
    --use_data1 \
    --data_toggle \
    --no_t_mask"

########################################
## Run selected experiment
## Change CMD to CMD_EXP1, CMD_EXP2, or CMD_EXP2B
########################################
echo "Running: $CMD_EXP1"
eval $CMD_EXP1
