#!/bin/bash
## v9: Eval LBS + Strain-conditioned Displacement Network
## Self-retargeting (MSE metrics) + Cross-retargeting (visualization)

set -e

CKPT="./ckpts_CBD/2026-03-10-19-07-09-NGBC++v9"
EPOCH=100

## Cross-retargeting target neutral mesh (FILL IN after locating a test identity)
##   TGT_VERT_PATH : .npy  [V, 3] target neutral vertex positions
##   TGT_NORM_PATH : .npy  [V, 3] target neutral vertex normals
##   TGT_OBJ_PATH  : .obj        target neutral mesh (for face topology)
## Val identity: m--20190529--1300--002421669--GHS (val split, used for visual cross-retarget only)
TGT_VERT_PATH="/data/sihun/multiface_align/002421669_tgt_verts.npy"
TGT_NORM_PATH="/data/sihun/multiface_align/002421669_tgt_normals.npy"
TGT_OBJ_PATH="/data/sihun/multiface_align/obj/m--20190529--1300--002421669--GHS_mesh.obj"

COMMON_FLAGS="
    --version 9
    --eval_use_strain_disp
    --no_vis_interv
    --continue_ckpt
    --start_epoch ${EPOCH}
    --ckpt ${CKPT}
    --last_activation relu
    --use_lbs_joint_center
    --num_lbs_joints 64
    --use_strain
    --strain_dim 1
    --use_t_mask
    --use_data1
    --data_toggle
    --realtest
    --batch_size 1
    --save_vert
"

########################################
## Self-retargeting  →  MSE / MSE-in / MSE-out
########################################
# echo "=== [Self] mf_SEN epoch ${EPOCH} ==="
# python eval_CBD.py ${COMMON_FLAGS} --data_selection 2

echo "=== [Self] mf_ROM epoch ${EPOCH} ==="
python eval_CBD.py ${COMMON_FLAGS} --data_selection 4

########################################
## Cross-retargeting  →  visualization only (no GT)
## Fill TGT_* paths above to enable
########################################
if [ "${TGT_VERT_PATH}" != "FILL_ME.npy" ]; then
    # echo "=== [Cross] mf_SEN → target epoch ${EPOCH} ==="
    # python eval_CBD.py ${COMMON_FLAGS} \
    #     --eval_cross_retarget \
    #     --data_selection 2 \
    #     --tgt_vert_path ${TGT_VERT_PATH} \
    #     --tgt_norm_path ${TGT_NORM_PATH} \
    #     --tgt_obj_path  ${TGT_OBJ_PATH}

    echo "=== [Cross] mf_ROM → target epoch ${EPOCH} ==="
    python eval_CBD.py ${COMMON_FLAGS} \
        --eval_cross_retarget \
        --data_selection 4 \
        --tgt_vert_path ${TGT_VERT_PATH} \
        --tgt_norm_path ${TGT_NORM_PATH} \
        --tgt_obj_path  ${TGT_OBJ_PATH}
else
    echo "[Cross] Skipped — fill in TGT_VERT/NORM/OBJ_PATH above"
fi

########################################
## Previous experiments (archived)
########################################
# EXP1_CKPT="./ckpts_CBD/2026-03-04-19-26-56-NGBC++v9"   # no strain
# EXP2B_CKPT="./ckpts_CBD/2026-03-05-10-05-18-NGBC++v9"  # strain full-grad
