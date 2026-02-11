#########################
## with Multiface only ##
#########################

############
## Hybrid ##
############
## this all is must "version 7" >> train_vLBSHybrid2()

###############
## concat input
## 64 joints + 512 cage vertices + residualized input
python train_CBD.py \
    --use_hyb_concat_lbs \
    --max_epoch 800 \
    --tb \
    --lr 1e-4 \
    --sc_step 100000 \
    --in_type 1 \
    --out_type 1 \
    --batch_size 16 \
    --version 7 \
    --use_data1 \
    --last_activation 'relu' \
    --use_lbs_joint_center \
    --num_lbs_joints 64 \
    --use_perm \
    --save_interval 25 \
    --continue_ckpt \
    --start_epoch 300 \
    --ckpt './ckpts_CBD/2026-02-03-11-10-40-NGBC++v5' \
    --log_dir "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5" \
    --num_cage_v 512 \
    --data_toggle


###############
## 32 joints + 512 cage vertices (not actually done)
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
    --use_lbs_joint_center \
    --num_lbs_joints 32 \
    --use_perm \
    --save_interval 25 \
    --continue_ckpt \
    --start_epoch 500 \
    --ckpt '/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-07-03-NGBC++v5' \
    --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-07-03-NGBC++v5" \
    --num_cage_v 512 \
    --data_toggle

###############
## 64 joints + 512 cage vertices
python train_CBD.py \
    --max_epoch 800 \
    --tb \
    --lr 1e-4 \
    --sc_step 100000 \
    --in_type 1 \
    --out_type 1 \
    --batch_size 16 \
    --version 7 \
    --use_data1 \
    --last_activation 'relu' \
    --use_lbs_joint_center \
    --num_lbs_joints 64 \
    --use_perm \
    --save_interval 25 \
    --continue_ckpt \
    --start_epoch 300 \
    --ckpt './ckpts_CBD/2026-02-03-11-10-40-NGBC++v5' \
    --log_dir "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5" \
    --num_cage_v 512 \
    --data_toggle

    ## continue
    python train_CBD.py \
    --max_epoch 800 \
    --tb \
    --lr 1e-4 \
    --sc_step 100000 \
    --in_type 1 \
    --out_type 1 \
    --batch_size 16 \
    --version 7 \
    --use_data1 \
    --last_activation 'relu' \
    --use_lbs_joint_center \
    --num_lbs_joints 64 \
    --use_perm \
    --save_interval 25 \
    --continue_ckpt \
    --start_epoch 725 \
    --ckpt './ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-13-55-18-NGBC++v7-stage2-from_lbs_ckpt_300' \
    --log_dir "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-13-55-18-NGBC++v7-stage2-from_lbs_ckpt_300" \
    --num_cage_v 512 \
    --data_toggle

#####################
## residualized input
## 64 joints + 512 cage vertices
python train_CBD.py \
    --use_hyb_delta_lbs_input \
    --max_epoch 800 \
    --tb \
    --lr 1e-4 \
    --sc_step 100000 \
    --in_type 1 \
    --out_type 1 \
    --batch_size 16 \
    --version 7 \
    --use_data1 \
    --last_activation 'relu' \
    --use_lbs_joint_center \
    --num_lbs_joints 64 \
    --use_perm \
    --save_interval 25 \
    --continue_ckpt \
    --start_epoch 300 \
    --ckpt './ckpts_CBD/2026-02-03-11-10-40-NGBC++v5' \
    --log_dir "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5" \
    --num_cage_v 512 \
    --data_toggle

    ## continue 
    python train_CBD.py \
    --use_hyb_delta_lbs_input \
    --max_epoch 800 \
    --tb \
    --lr 1e-4 \
    --sc_step 100000 \
    --in_type 1 \
    --out_type 1 \
    --batch_size 16 \
    --version 7 \
    --use_data1 \
    --last_activation 'relu' \
    --use_lbs_joint_center \
    --num_lbs_joints 64 \
    --use_perm \
    --save_interval 25 \
    --continue_ckpt \
    --start_epoch 625 \
    --ckpt './ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-15-24-51-NGBC++v7-stage2-from_lbs_ckpt_300' \
    --log_dir "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-15-24-51-NGBC++v7-stage2-from_lbs_ckpt_300" \
    --num_cage_v 512 \
    --data_toggle
