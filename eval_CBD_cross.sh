#############################################################
## full data cross-retargeting test on stylized character ###
#############################################################

## LBS
TARGET_DIR="/source/inyup/NeuralFacialAnimation/utils/data/biwi_ready"
TARGET_OBJ_DIR="/source/inyup/NeuralFacialAnimation/utils/data/test_data"

for v_path in ${TARGET_DIR}/*.npy; do

    # normal 파일은 스킵
    if [[ "$v_path" == *_normal.npy ]]; then
        continue
    fi

    base=$(basename "$v_path" .npy)

    n_path="${TARGET_DIR}/${base}_normal.npy"
    o_path="${TARGET_OBJ_DIR}/${base}.obj"

    echo "Running cross-retarget on: $base"

    python eval_CBD.py \
        --eval_cross_retarget \
        --eval_use_lbs \
        --version 6 \
        --tgt_vert_path "$v_path" \
        --tgt_norm_path "$n_path" \
        --tgt_obj_path "$o_path" \
        --no_vis_interv \
        --data_selection 3 \
        --continue_ckpt \
        --start_epoch 200 \
        --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-11-08-37-13-NGBC++v6" \
        --realtest \
        --batch_size 1 \
        --save_vert \
        --use_t_mask

done



## NBC







## NFS







## LBS + NBC 