

########################################################################
######################## Multiface only
######################## Hybrid LBS + CBD >> version 4 
########################################################################
## 64 joints - 9dof + 512 CBD
        ## w/ 500
python eval_CBD.py \
--no_vis_interv \
--data_selection 4 \
--continue_ckpt \
--start_epoch 500 \
--ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-13-55-18-NGBC++v7-stage2-from_lbs_ckpt_300" \
--version 4 \
--eval_use_hybrid_separate \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask

        ## w/ 600
python eval_CBD.py \
--no_vis_interv \
--data_selection 4 \
--continue_ckpt \
--start_epoch 600 \
--ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-13-55-18-NGBC++v7-stage2-from_lbs_ckpt_300" \
--version 4 \
--eval_use_hybrid_separate \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask

        ## w/ 700
python eval_CBD.py \
--no_vis_interv \
--data_selection 4 \
--continue_ckpt \
--start_epoch 700 \
--ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-13-55-18-NGBC++v7-stage2-from_lbs_ckpt_300" \
--version 4 \
--eval_use_hybrid_separate \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask

        ## w/ 800
python eval_CBD.py \
--no_vis_interv \
--data_selection 4 \
--continue_ckpt \
--start_epoch 800 \
--ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-13-55-18-NGBC++v7-stage2-from_lbs_ckpt_300/2026-02-10-14-30-10-NGBC++v7-stage2-from_lbs_ckpt_725" \
--version 4 \
--eval_use_hybrid_separate \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask

        ## w/ best
python eval_CBD.py \
--no_vis_interv \
--data_selection 4 \
--ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-13-55-18-NGBC++v7-stage2-from_lbs_ckpt_300/2026-02-10-14-30-10-NGBC++v7-stage2-from_lbs_ckpt_725" \
--version 4 \
--eval_use_hybrid_separate \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask

