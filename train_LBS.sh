## version should always be 6 for LBS training

python train_CBD.py \
    --max_epoch 600 \
    --tb \
    --lr 1e-4 \
    --sc_step 100 \
    --in_type 1 \
    --out_type 1 \
    --batch_size 16 \
    --version 6 \
    --use_data0 \
    --last_activation 'relu' \
    --use_lbs \
    --use_lbs_joint_center \
    --num_lbs_joints 32 \
    --lbs_pretrained_epochs 600 \
    --data_toggle \
    --use_perm



