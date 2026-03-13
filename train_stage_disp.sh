#!/bin/bash
## Stage-based training: Stage1 LBS→smooth_GT, Stage2 Joint LBS+DispNet
## Experiments: smooth_n_iter = {8, 16, 32}

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
    --strain_mode norm \
    --last_activation relu \
    --use_lbs_joint_center \
    --num_lbs_joints 64 \
    --save_interval 50 \
    --eval_iter 25 \
    --use_data1 \
    --no_t_mask \
    --lbs_pretrained_epochs 200 \
    --max_epoch 500"

########################################
## Stage training: smooth_n_iter=8
########################################
CMD_S8="python train_stage_disp.py $COMMON_ARGS --smooth_n_iter 8"

########################################
## Stage training: smooth_n_iter=16
########################################
CMD_S16="python train_stage_disp.py $COMMON_ARGS --smooth_n_iter 16"

########################################
## Stage training: smooth_n_iter=32
########################################
CMD_S32="python train_stage_disp.py $COMMON_ARGS --smooth_n_iter 32"

########################################
## Stage training + strain match loss: smooth_n_iter=16
########################################
CMD_S16_SM="python train_stage_disp.py $COMMON_ARGS --smooth_n_iter 16 \
    --use_strain_match --lambda_strain_match 0.1 --strain_match_loss_type mse"

########################################
## Stage training + pre-trained DispNet: smooth_n_iter=16
## (fill in --disp_ckpt path after train_disp_only.py finishes)
########################################
CMD_S16_PRETRAINED="python train_stage_disp.py $COMMON_ARGS --smooth_n_iter 16 \
    --disp_ckpt ./ckpts_CBD/DISP_CKPT_DIR/model_disp_best.pth"

########################################
## v2: principal strain + norm_trace matching + true EDD + t_mask
########################################
CMD_S16_V2="python train_stage_disp.py $COMMON_ARGS --smooth_n_iter 16 \
    --strain_mode principal --strain_match_mode norm_trace \
    --use_strain_match --lambda_strain_match 0.1 --strain_match_loss_type mse \
    --use_true_edd --use_t_mask"

CMD_S16_V2_PRETRAINED="python train_stage_disp.py $COMMON_ARGS --smooth_n_iter 16 \
    --strain_mode principal --strain_match_mode norm_trace \
    --use_strain_match --lambda_strain_match 0.1 --strain_match_loss_type mse \
    --use_true_edd --use_t_mask \
    --disp_ckpt ./ckpts_CBD/DISP_CKPT_DIR/model_disp_best.pth"

########################################
## Run: change CMD variable below
########################################
echo "Running: $CMD_S16_V2"
eval $CMD_S16_V2
