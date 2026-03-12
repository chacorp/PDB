#!/bin/bash
## v9 (original, 13-dim, use_source_template): Cross-retargeting eval
## Runs epochs 200, 300, 400, 500, best in parallel

CKPT=${1:-"./ckpts_CBD/2026-03-09-18-11-33-NGBC++v9"}
CKPT_NAME=$(basename ${CKPT})
EVAL_BASE="./eval_CBD/${CKPT_NAME}-eval"

TGT_VERT_PATH="/data/sihun/multiface_align/002421669_tgt_verts.npy"
TGT_NORM_PATH="/data/sihun/multiface_align/002421669_tgt_normals.npy"
TGT_OBJ_PATH="/data/sihun/multiface_align/obj/m--20190529--1300--002421669--GHS_mesh.obj"

BASE_FLAGS="
    --version 9
    --eval_use_strain_disp
    --eval_cross_retarget
    --no_vis_interv
    --ckpt ${CKPT}
    --last_activation relu
    --use_lbs_joint_center
    --num_lbs_joints 64
    --use_strain
    --strain_dim 1
    --use_source_template
    --smooth_n_iter 0
    --use_t_mask
    --use_data1
    --data_toggle
    --realtest
    --batch_size 1
    --save_vert
    --tgt_vert_path ${TGT_VERT_PATH}
    --tgt_norm_path ${TGT_NORM_PATH}
    --tgt_obj_path  ${TGT_OBJ_PATH}
"

EPOCHS=(200 300 400 500)

for EP in "${EPOCHS[@]}"; do
    LOG_DIR="${EVAL_BASE}/cross_e${EP}"
    mkdir -p ${LOG_DIR}
    echo "=== [Cross] mf_ROM epoch ${EP} ==="
    python eval_CBD.py ${BASE_FLAGS} --continue_ckpt --start_epoch ${EP} --data_selection 4 \
        --log_dir ${LOG_DIR} \
        > ${LOG_DIR}/eval.log 2>&1 &
done

LOG_DIR="${EVAL_BASE}/cross_best"
mkdir -p ${LOG_DIR}
echo "=== [Cross] mf_ROM best ==="
python eval_CBD.py ${BASE_FLAGS} --data_selection 4 \
    --log_dir ${LOG_DIR} \
    > ${LOG_DIR}/eval.log 2>&1 &

echo "All 5 cross-retarget evals launched. Results in ${EVAL_BASE}/"
echo "Waiting..."
wait
echo "Done!"
