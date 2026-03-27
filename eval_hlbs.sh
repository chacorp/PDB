#!/bin/bash
## Eval HierarchicalLBS: Self-retargeting (metrics) + Cross-retargeting (visualization)

set -e

CKPT="./ckpts_hlbs/FILL_ME"   # ← fill with actual checkpoint dir
EPOCH=-1                       # -1 = best, or specific epoch number

RIG_PATH="utils/mf/rig_info.json"

## Cross-retargeting target neutral mesh
TGT_VERT_PATH="/data/sihun/multiface_align/002421669_tgt_verts.npy"
TGT_NORM_PATH="/data/sihun/multiface_align/002421669_tgt_normals.npy"
TGT_OBJ_PATH="/data/sihun/multiface_align/obj/m--20190529--1300--002421669--GHS_mesh.obj"

COMMON_FLAGS="
    --rig_path ${RIG_PATH}
    --topo_key mf
    --start_epoch ${EPOCH}
    --ckpt ${CKPT}
    --batch_size 1
    --save_vert
    --use_t_mask
"

########################################
## Self-retargeting  →  full metrics
########################################
echo "=== [Self] mf_ROM epoch ${EPOCH} ==="
python eval_hlbs.py ${COMMON_FLAGS} --data_selection mf_ROM

########################################
## Cross-retargeting  →  visualization only
########################################
if [ -f "${TGT_VERT_PATH}" ]; then
    echo "=== [Cross] mf_ROM → target epoch ${EPOCH} ==="
    python eval_hlbs.py ${COMMON_FLAGS} \
        --cross_retarget \
        --data_selection mf_ROM \
        --tgt_vert_path ${TGT_VERT_PATH} \
        --tgt_norm_path ${TGT_NORM_PATH} \
        --tgt_obj_path  ${TGT_OBJ_PATH} \
        --make_video
else
    echo "[Cross] Skipped — tgt_vert_path not found: ${TGT_VERT_PATH}"
fi
