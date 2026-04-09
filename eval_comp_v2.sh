#!/bin/bash
# ============================================================================
# eval_comp_v2.sh — NFS / NFR evaluation guide
# ============================================================================
# Data index:  0:voca  1:biwi  2:mf_SEN  3:coma  4:mf_ROM  5:ict  6:ict-cap
# MF identity: 12=test(002645310), 11=val(002421669), 0-10=train
# ICT identity: 0-19 (test), uses ict_id_vecs_test.pt
# ============================================================================

# ── NFR ─────────────────────────────────────────────────────────────────────

# NFR self-retarget (mf_ROM test id)
python eval_comp_v2.py --NFR \
    --src_data 4 --src_id 12 --tgt_data 4 --tgt_id 12 \
    --make_video --use_t_mask

# NFR self-retarget (quick debug, 200 frames)
python eval_comp_v2.py --NFR \
    --src_data 4 --src_id 12 --tgt_data 4 --tgt_id 12 \
    --make_video --max_frames 200

# NFR cross-retarget (mf_ROM → ICT id2)
python eval_comp_v2.py --NFR \
    --src_data 4 --src_id 12 --tgt_data 5 --tgt_id 2 \
    --make_video --max_frames 200

# NFR cross-retarget (mf_ROM → ICT id9)
python eval_comp_v2.py --NFR \
    --src_data 4 --src_id 12 --tgt_data 5 --tgt_id 9 \
    --make_video --max_frames 200

# NFR self-retarget (ict-cap id2, exp0)
python eval_comp_v2.py --NFR \
    --src_data 6 --src_id 2 --tgt_data 6 --tgt_id 2 --exp_num 0 \
    --make_video --max_frames 200

# ── NFS ─────────────────────────────────────────────────────────────────────

# NFS self-retarget (mf_ROM test id)
python eval_comp_v2.py --ckpt ./ckpts_comparison/NFS-best \
    --src_data 4 --src_id 12 --tgt_data 4 --tgt_id 12 \
    --make_video --use_t_mask

# NFS self-retarget (quick debug, 200 frames)
python eval_comp_v2.py --ckpt ./ckpts_comparison/NFS-best \
    --src_data 4 --src_id 12 --tgt_data 4 --tgt_id 12 \
    --make_video --max_frames 200

# NFS cross-retarget (mf_ROM → ICT id2)
python eval_comp_v2.py --ckpt ./ckpts_comparison/NFS-best \
    --src_data 4 --src_id 12 --tgt_data 5 --tgt_id 2 \
    --make_video --max_frames 200

# NFS cross-retarget (mf_ROM → ICT id9)
python eval_comp_v2.py --ckpt ./ckpts_comparison/NFS-best \
    --src_data 4 --src_id 12 --tgt_data 5 --tgt_id 9 \
    --make_video --max_frames 200

# NFS self-retarget (ict-cap id2, exp0)
python eval_comp_v2.py --ckpt ./ckpts_comparison/NFS-best \
    --src_data 6 --src_id 2 --tgt_data 6 --tgt_id 2 --exp_num 0 \
    --make_video --max_frames 200

# ── NFR cross-retarget to other datasets ────────────────────────────────────

# NFR cross: mf_ROM → VOCA id0
python eval_comp_v2.py --NFR \
    --src_data 4 --src_id 12 --tgt_data 0 --tgt_id 0 \
    --make_video --max_frames 200

# NFR cross: mf_ROM → BIWI id0
python eval_comp_v2.py --NFR \
    --src_data 4 --src_id 12 --tgt_data 1 --tgt_id 0 \
    --make_video --max_frames 200

# NFR cross: mf_ROM → COMA id0
python eval_comp_v2.py --NFR \
    --src_data 4 --src_id 12 --tgt_data 3 --tgt_id 0 \
    --make_video --max_frames 200

# ── NFS cross-retarget to other datasets ────────────────────────────────────

# NFS cross: mf_ROM → VOCA id0
python eval_comp_v2.py --ckpt ./ckpts_comparison/NFS-best \
    --src_data 4 --src_id 12 --tgt_data 0 --tgt_id 0 \
    --make_video --max_frames 200

# NFS cross: mf_ROM → BIWI id0
python eval_comp_v2.py --ckpt ./ckpts_comparison/NFS-best \
    --src_data 4 --src_id 12 --tgt_data 1 --tgt_id 0 \
    --make_video --max_frames 200

# NFS cross: mf_ROM → COMA id0
python eval_comp_v2.py --ckpt ./ckpts_comparison/NFS-best \
    --src_data 4 --src_id 12 --tgt_data 3 --tgt_id 0 \
    --make_video --max_frames 200

# ── Source from COMA (data=3) ───────────────────────────────────────────────

# NFR self-retarget on COMA (all identities in test)
python eval_comp_v2.py --NFR \
    --src_data 3 --src_id 0 --tgt_data 3 --tgt_id 0 \
    --make_video --max_frames 200

# NFS self-retarget on COMA
python eval_comp_v2.py --ckpt ./ckpts_comparison/NFS-best \
    --src_data 3 --src_id 0 --tgt_data 3 --tgt_id 0 \
    --make_video --max_frames 200

# ── Identity index reference ────────────────────────────────────────────────
# VOCA (0-11):  FaceTalk_170904_00128_TA, FaceTalk_170811_03275_TA, ...
#   test: FaceTalk_170809_00138_TA(id0 in test), FaceTalk_170731_00024_TA(id1)
# BIWI (0-13):  F1-F8 (female), M1-M6 (male)
# COMA (0-11):  Same as VOCA templates
# MF (0-12):    0-10=train, 11=val, 12=test
# ICT (0-19):   Test identities from ict_id_vecs_test.pt

# ── Options ─────────────────────────────────────────────────────────────────
# --save_vert       Save predicted vertices as .npy
# --save_gt         Save GT vertices as .npy (self-retarget only)
# --no_vis          Skip image rendering (metrics only)
# --max_frames N    Limit to N frames (-1 = all)
# --make_video      Generate .mp4 from rendered images
# --use_t_mask      Use T-zone mask for inner/outer MSE
#
# Output: eval_comp/{NFS|NFR}/{src}_test-to-{tgt}_test[-masked]/
#         ├── results.json   (metrics)
#         ├── per_vertex_l2.npy
#         ├── img/           (rendered frames)
#         └── verts/         (if --save_vert)
