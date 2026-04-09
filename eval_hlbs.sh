#!/bin/bash
# ============================================================================
# eval_hlbs.sh — HLBS evaluation guide
# ============================================================================
# data_selection: voca, biwi, mf_SEN, coma, mf_ROM, ict, ict-cap
# tgt_dataset:    mf(0-12), voca(0-11), biwi(0-13), coma(0-11), ict(0-19)
# ============================================================================

# Set your checkpoint path here
CKPT="./ckpts_hlbs/YOUR_CKPT_DIR"
# Example:
# CKPT="./ckpts_hlbs/2026-03-31-13-53-44-HLBS-FullPred-mf-jTrans"
# CKPT="./ckpts_hlbs_delta_smooth/2026-04-06-13-34-25-HLBS-mf-s0-jTrans-sdw3a0.5-nrm0.5-crv0.5-cur50"

# Common flags (adjust per checkpoint)
COMMON="--rig_path maya_rig/hybrid --topo_key mf --use_joint_trans --hid_dim 128 --data_toggle --batch_size 1 --use_t_mask --make_video"

# Add --full_prediction for FullPred checkpoints
# Add --smooth_delta_W 3 for delta mode with forward smoothing

# ── Self-retarget ───────────────────────────────────────────────────────────

# Self-retarget on mf_ROM (best checkpoint)
python eval_hlbs.py --ckpt $CKPT $COMMON \
    --data_selection mf_ROM \
    --full_prediction

# Self-retarget on mf_ROM (specific epoch)
python eval_hlbs.py --ckpt $CKPT $COMMON \
    --data_selection mf_ROM \
    --full_prediction --start_epoch 300

# Self-retarget on ict-cap (id2, exp0)
python eval_hlbs.py --ckpt $CKPT $COMMON \
    --data_selection ict-cap \
    --src_identity 2 --exp_num 0 \
    --full_prediction

# Self-retarget on coma
python eval_hlbs.py --ckpt $CKPT $COMMON \
    --data_selection coma \
    --full_prediction

# ── Cross-retarget ──────────────────────────────────────────────────────────

# Cross-retarget: mf_ROM → ICT id2
python eval_hlbs.py --ckpt $CKPT $COMMON \
    --data_selection mf_ROM \
    --cross_retarget --tgt_dataset ict --tgt_identity 2 \
    --full_prediction

# Cross-retarget: mf_ROM → ICT id9
python eval_hlbs.py --ckpt $CKPT $COMMON \
    --data_selection mf_ROM \
    --cross_retarget --tgt_dataset ict --tgt_identity 9 \
    --full_prediction

# Cross-retarget: mf_ROM → BIWI id3
python eval_hlbs.py --ckpt $CKPT $COMMON \
    --data_selection mf_ROM \
    --cross_retarget --tgt_dataset biwi --tgt_identity 3 \
    --full_prediction

# Cross-retarget: mf_ROM → VOCA id0
python eval_hlbs.py --ckpt $CKPT $COMMON \
    --data_selection mf_ROM \
    --cross_retarget --tgt_dataset voca --tgt_identity 0 \
    --full_prediction

# Cross-retarget: mf_ROM → COMA id0
python eval_hlbs.py --ckpt $CKPT $COMMON \
    --data_selection mf_ROM \
    --cross_retarget --tgt_dataset coma --tgt_identity 0 \
    --full_prediction

# ── Delta mode example ──────────────────────────────────────────────────────

# Delta mode self-retarget (no --full_prediction, add --smooth_delta_W if used)
# CKPT_DELTA="./ckpts_hlbs_delta_smooth/..."
# python eval_hlbs.py --ckpt $CKPT_DELTA $COMMON \
#     --data_selection mf_ROM \
#     --smooth_delta_W 3

# ── Options ─────────────────────────────────────────────────────────────────
# --start_epoch N     Load specific epoch (-1 = best, default)
# --src_identity N    Source identity for ict-cap (-1 = default 2)
# --exp_num N         Expression sequence for ict-cap (0 or 1)
# --save_vert         Save predicted vertices as .npy
# --save_gt           Save GT vertices as .npy
# --save_obj          Save predicted meshes as .obj
# --no_vis            Skip image rendering (metrics only)
# --no_video          Skip video generation
# --data_basedir P    Data root (default: /data/sihun)
#
# Output: eval_hlbs/{ckpt_name}-eval/e{epoch}-{self|cross-tgt}/data_selection/
#         ├── results.json
#         ├── per_vertex_l2.npy
#         ├── img/
#         └── eval_hlbs.mp4
