## version should always be 7 for Hybrid training

####################################
####### Multiface only #########
python train_CBD.py \
    --max_epoch 1000 \
    --tb \
    --lr 1e-4 \
    --sc_step 100000 \
    --in_type 1 \
    --out_type 1 \
    --batch_size 16 \
    --version 7 \
    --use_data1 \
    --last_activation 'relu' \
    --use_lbs \
    --use_lbs_joint_center \
    --num_lbs_joints 32 \
    --use_perm \
    --save_interval 25 \
    --continue_ckpt \
    --start_epoch 500 \
    --ckpt '/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-07-03-NGBC++v5' \
    --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-07-03-NGBC++v5" \
    --start_stage 2 \
    --num_cage_v 512 \
    --data_toggle

