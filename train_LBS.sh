## tmux session script for training CBD hybrid models
## runs training in a detached tmux session >> doesn't stop when terminal is closed
##########
## tips ##
##########
## 1. attaching session to current terminal (bash shell)
# tmux attach -t lbs_train
## 2. listing sessions
# tmux ls
## 3. killing session
# tmux kill-session -t lbs_train
## 4. showing status w/o attaching
# tmux capture-pane -pt lbs_train | tail -n 20

## !/usr/bin/env bash
set -e

## change if needed
SESSION_NAME="lbs_train"

# 이미 세션이 있으면 붙기
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Session already exists. Attaching..."
    tmux attach -t $SESSION_NAME
    exit 0
fi

# 새 세션 생성 (detach 모드)
tmux new-session -d -s $SESSION_NAME


## version should always be 6 for LBS training
# python train_CBD.py \
#     --max_epoch 600 \
#     --tb \
#     --lr 1e-4 \
#     --sc_step 100 \
#     --in_type 1 \
#     --out_type 1 \
#     --batch_size 16 \
#     --version 6 \
#     --use_data0 \
#     --last_activation 'relu' \
#     --use_lbs \
#     --use_lbs_joint_center \
#     --num_lbs_joints 32 \
#     --lbs_pretrained_epochs 600 \
#     --data_toggle \
#     --use_perm

## full data training
# CMD="python train_CBD.py \
#     --max_epoch 500 \
#     --tb \
#     --lr 1e-4 \
#     --sc_step 100 \
#     --in_type 1 \
#     --out_type 1 \
#     --batch_size 16 \
#     --version 6 \
#     --last_activation 'relu' \
#     --use_lbs \
#     --use_lbs_joint_center \
#     --num_lbs_joints 64 \
#     --lbs_pretrained_epochs 600 \
#     --data_toggle \
#     --use_perm"

    ## continue
    CMD="python train_CBD.py \
        --continue_ckpt \
        --start_epoch 150 \
        --ckpt './ckpts_CBD/2026-02-11-08-37-13-NGBC++v6' \
        --log_dir "./ckpts_CBD/2026-02-11-08-37-13-NGBC++v6" \
        --max_epoch 500 \
        --tb \
        --lr 1e-4 \
        --sc_step 100 \
        --in_type 1 \
        --out_type 1 \
        --batch_size 16 \
        --version 6 \
        --last_activation 'relu' \
        --use_lbs \
        --use_lbs_joint_center \
        --num_lbs_joints 64 \
        --lbs_pretrained_epochs 600 \
        --data_toggle \
        --use_perm"


tmux send-keys -t $SESSION_NAME "$CMD" C-m

echo "Training started in tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
