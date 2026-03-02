#!/bin/bash
# analyze_CBD.sh
# Run LBS+CBD analysis on a trained version-8 checkpoint.
#
# Usage:
#   bash analyze_CBD.sh
#
# To override the checkpoint path, set CKPT env variable:
#   CKPT=./ckpts_CBD/2026-XX-XX-my-full-data bash analyze_CBD.sh

CKPT=${CKPT:-"./ckpts_CBD/2026-02-13-07-54-53-NGBC++v8"}

# --------------------------------------------------------------------------
# Standard analysis: Multiface ROM test set, best checkpoint
# --------------------------------------------------------------------------
python analyze_CBD.py \
    --version 8 \
    --ckpt "$CKPT" \
    --data_selection 4 \
    --realtest \
    --batch_size 1 \
    --num_top_cage 32 \
    --log_dir ./analysis_CBD

# --------------------------------------------------------------------------
# Weight visualization only (fast, no per-frame residual loop):
# --------------------------------------------------------------------------
# python analyze_CBD.py \
#     --version 8 \
#     --ckpt "$CKPT" \
#     --data_selection 4 \
#     --realtest \
#     --batch_size 1 \
#     --num_top_cage 32 \
#     --log_dir ./analysis_CBD \
#     --weight_vis_only

# --------------------------------------------------------------------------
# Limit to first N frames (for quick debugging):
# --------------------------------------------------------------------------
# python analyze_CBD.py \
#     --version 8 \
#     --ckpt "$CKPT" \
#     --data_selection 4 \
#     --realtest \
#     --batch_size 1 \
#     --num_top_cage 32 \
#     --log_dir ./analysis_CBD \
#     --max_frames 50

# --------------------------------------------------------------------------
# Specific saved epoch (instead of best):
# --------------------------------------------------------------------------
# python analyze_CBD.py \
#     --version 8 \
#     --ckpt "$CKPT" \
#     --data_selection 4 \
#     --realtest \
#     --batch_size 1 \
#     --num_top_cage 32 \
#     --log_dir ./analysis_CBD \
#     --start_epoch 500 \
#     --continue_ckpt
