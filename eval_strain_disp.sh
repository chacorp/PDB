#!/bin/bash
## v9: Eval LBS + Strain-conditioned Displacement Network
## Matches train_strain_disp.sh experiments
## Usage: Update --ckpt path and --start_epoch, then run selected block

set -e

## Common flags
COMMON="python eval_CBD.py \
    --version 9 \
    --eval_use_strain_disp \
    --no_vis_interv \
    --data_selection 4 \
    --continue_ckpt \
    --in_type 1 \
    --out_type 1 \
    --last_activation relu \
    --use_lbs_joint_center \
    --num_lbs_joints 64 \
    --use_data1 \
    --no_t_mask \
    --realtest \
    --batch_size 1 \
    --save_vert"

########################################
## Experiment 1: Naive (no strain)
########################################
## >>> Update ckpt path after training <<<
EXP1_CKPT="./ckpts_CBD/<EXP1_FOLDER>"

# specific epoch
# eval $COMMON --ckpt "$EXP1_CKPT" --start_epoch 50
# eval $COMMON --ckpt "$EXP1_CKPT" --start_epoch 100
# eval $COMMON --ckpt "$EXP1_CKPT" --start_epoch 200
# eval $COMMON --ckpt "$EXP1_CKPT" --start_epoch 500

# best
# eval $COMMON --ckpt "$EXP1_CKPT"

########################################
## Experiment 2: With strain (stop-grad)
########################################
## >>> Update ckpt path after training <<<
EXP2_CKPT="./ckpts_CBD/<EXP2_FOLDER>"

# eval $COMMON --use_strain --strain_dim 1 --ckpt "$EXP2_CKPT" --start_epoch 50
# eval $COMMON --use_strain --strain_dim 1 --ckpt "$EXP2_CKPT" --start_epoch 100
# eval $COMMON --use_strain --strain_dim 1 --ckpt "$EXP2_CKPT" --start_epoch 200
# eval $COMMON --use_strain --strain_dim 1 --ckpt "$EXP2_CKPT" --start_epoch 500

# best
# eval $COMMON --use_strain --strain_dim 1 --ckpt "$EXP2_CKPT"

########################################
## Experiment 2b: With strain (full-grad)
########################################
## >>> Update ckpt path after training <<<
EXP2B_CKPT="./ckpts_CBD/<EXP2B_FOLDER>"

# eval $COMMON --use_strain --strain_dim 1 --strain_full_grad --ckpt "$EXP2B_CKPT" --start_epoch 50
# eval $COMMON --use_strain --strain_dim 1 --strain_full_grad --ckpt "$EXP2B_CKPT" --start_epoch 100
# eval $COMMON --use_strain --strain_dim 1 --strain_full_grad --ckpt "$EXP2B_CKPT" --start_epoch 200
# eval $COMMON --use_strain --strain_dim 1 --strain_full_grad --ckpt "$EXP2B_CKPT" --start_epoch 500

# best
# eval $COMMON --use_strain --strain_dim 1 --strain_full_grad --ckpt "$EXP2B_CKPT"
