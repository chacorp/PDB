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
