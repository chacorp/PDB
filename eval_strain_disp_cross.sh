#!/bin/bash
## v9: Cross-retarget eval for LBS + Strain Displacement
## Source: expression sequence (--data_selection)
## Target: external neutral mesh (biwi_ready)

TARGET_DIR="/source/inyup/NeuralFacialAnimation/utils/data/biwi_ready"
TARGET_OBJ_DIR="/source/inyup/NeuralFacialAnimation/utils/data/test_data"

## Common flags
COMMON="python eval_CBD.py \
    --version 9 \
    --eval_cross_retarget \
    --eval_use_strain_disp \
    --no_vis_interv \
    --last_activation relu \
    --use_lbs_joint_center \
    --num_lbs_joints 64 \
    --use_data1 \
    --data_selection 3 \
    --realtest \
    --batch_size 1 \
    --save_vert"

########################################
## Experiment 1: Naive (no strain)
########################################
EXP1_CKPT="./ckpts_CBD/2026-03-04-19-26-56-NGBC++v9"

for v_path in ${TARGET_DIR}/*.npy; do
    if [[ "$v_path" == *_normal.npy ]]; then
        continue
    fi
    base=$(basename "$v_path" .npy)
    n_path="${TARGET_DIR}/${base}_normal.npy"
    o_path="${TARGET_OBJ_DIR}/${base}.obj"
    echo "Cross-retarget Exp1 on: $base"
    eval $COMMON --ckpt "$EXP1_CKPT" \
        --tgt_vert_path "$v_path" \
        --tgt_norm_path "$n_path" \
        --tgt_obj_path "$o_path"
done

########################################
## Experiment 2: With strain (stop-grad)
########################################
# EXP2_CKPT="./ckpts_CBD/<EXP2_FOLDER>"
# for v_path in ${TARGET_DIR}/*.npy; do
#     if [[ "$v_path" == *_normal.npy ]]; then continue; fi
#     base=$(basename "$v_path" .npy)
#     n_path="${TARGET_DIR}/${base}_normal.npy"
#     o_path="${TARGET_OBJ_DIR}/${base}.obj"
#     echo "Cross-retarget Exp2 on: $base"
#     eval $COMMON --use_strain --strain_dim 1 --ckpt "$EXP2_CKPT" \
#         --tgt_vert_path "$v_path" \
#         --tgt_norm_path "$n_path" \
#         --tgt_obj_path "$o_path"
# done
