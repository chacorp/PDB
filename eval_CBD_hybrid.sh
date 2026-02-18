

########################################################################
######################## Multiface only
######################## Hybrid LBS + CBD >> version 4 
########################################################################

##########################################
## 64 joints - 9dof + 512 CBD finetute lbs at stage 2 train at char-s02
python eval_CBD.py \
--version 8 \
--eval_use_hybrid_separate \
--no_vis_interv \
--data_selection 4 \
--continue_ckpt \
--start_epoch 800 \
--ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-16-17-38-51-NGBC++v7-stage2-from_lbs_ckpt_300" \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask

##########################################
## 64 joints - 9dof + 512 CBD default at char-s03
# python eval_CBD.py \
# --version 7 \
# --eval_use_hybrid_separate \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 500 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-13-06-52-18-NGBC++v7-stage2-from_lbs_ckpt_300" \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

# ##########################################
# ## 64 joints - 9dof + 512 CBD concat input at eve-s01
# python eval_CBD.py \
# --version 7 \
# --eval_use_hybrid_separate \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 500 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-13-07-05-00-NGBC++v7-stage2-from_lbs_ckpt_300" \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

##########################################
## 64 joints - 9dof + 512 CBD joint train at char-s02
python eval_CBD.py \
--version 8 \
--eval_use_hybrid_separate \
--no_vis_interv \
--data_selection 4 \
--continue_ckpt \
--start_epoch 200 \
--ckpt "./ckpts_CBD/2026-02-13-07-54-53-NGBC++v8" \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask

python eval_CBD.py \
--version 8 \
--eval_use_hybrid_separate \
--no_vis_interv \
--data_selection 4 \
--continue_ckpt \
--start_epoch 300 \
--ckpt "./ckpts_CBD/2026-02-13-07-54-53-NGBC++v8" \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask

python eval_CBD.py \
--version 8 \
--eval_use_hybrid_separate \
--no_vis_interv \
--data_selection 4 \
--continue_ckpt \
--start_epoch 400 \
--ckpt "./ckpts_CBD/2026-02-13-07-54-53-NGBC++v8" \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask

python eval_CBD.py \
--version 8 \
--eval_use_hybrid_separate \
--no_vis_interv \
--data_selection 4 \
--continue_ckpt \
--start_epoch 500 \
--ckpt "./ckpts_CBD/2026-02-13-07-54-53-NGBC++v8" \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask

python eval_CBD.py \
--version 8 \
--eval_use_hybrid_separate \
--no_vis_interv \
--data_selection 4 \
--ckpt "./ckpts_CBD/2026-02-13-07-54-53-NGBC++v8" \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask































################ deprecated ##################

# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 500 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-11-08-27-57-NGBC++v7-stage2-from_lbs_ckpt_300" \
# --version 4 \
# --use_hyb_concat_lbs \
# --eval_use_hybrid_separate \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ 600
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 600 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-11-08-27-57-NGBC++v7-stage2-from_lbs_ckpt_300" \
# --version 4 \
# --use_hyb_concat_lbs \
# --eval_use_hybrid_separate \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ 700
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 700 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-11-08-27-57-NGBC++v7-stage2-from_lbs_ckpt_300/2026-02-12-06-39-41-NGBC++v7-stage2-from_lbs_ckpt_650" \
# --version 4 \
# --use_hyb_concat_lbs \
# --eval_use_hybrid_separate \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ 800
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 800 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-11-08-27-57-NGBC++v7-stage2-from_lbs_ckpt_300/2026-02-12-06-39-41-NGBC++v7-stage2-from_lbs_ckpt_650" \
# --version 4 \
# --use_hyb_concat_lbs \
# --eval_use_hybrid_separate \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ best
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-11-08-27-57-NGBC++v7-stage2-from_lbs_ckpt_300/2026-02-12-06-39-41-NGBC++v7-stage2-from_lbs_ckpt_650" \
# --version 4 \
# --use_hyb_concat_lbs \
# --eval_use_hybrid_separate \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask




##########################################
## 64 joints - 9dof + 512 CBD delta input
        ## w/ 500
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 500 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-15-24-51-NGBC++v7-stage2-from_lbs_ckpt_300" \
# --version 4 \
# --use_hyb_delta_lbs_input \
# --eval_use_hybrid_separate \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ 600
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 600 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-15-24-51-NGBC++v7-stage2-from_lbs_ckpt_300" \
# --version 4 \
# --use_hyb_delta_lbs_input \
# --eval_use_hybrid_separate \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ 700
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 700 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-15-24-51-NGBC++v7-stage2-from_lbs_ckpt_300/2026-02-10-14-20-33-NGBC++v7-stage2-from_lbs_ckpt_625" \
# --version 4 \
# --use_hyb_delta_lbs_input \
# --eval_use_hybrid_separate \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

        ## w/ 800
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 800 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-15-24-51-NGBC++v7-stage2-from_lbs_ckpt_300/2026-02-10-14-20-33-NGBC++v7-stage2-from_lbs_ckpt_625" \
# --version 4 \
# --use_hyb_delta_lbs_input \
# --eval_use_hybrid_separate \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ best
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-15-24-51-NGBC++v7-stage2-from_lbs_ckpt_300/2026-02-10-14-20-33-NGBC++v7-stage2-from_lbs_ckpt_625" \
# --version 4 \
# --use_hyb_delta_lbs_input \
# --eval_use_hybrid_separate \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask





##################
## residualized
## 64 joints - 9dof + 512 CBD
        ## w/ 500
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 500 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-13-55-18-NGBC++v7-stage2-from_lbs_ckpt_300" \
# --version 4 \
# --eval_use_hybrid_separate \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ 600
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 600 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-13-55-18-NGBC++v7-stage2-from_lbs_ckpt_300" \
# --version 4 \
# --eval_use_hybrid_separate \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ 700
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 700 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-13-55-18-NGBC++v7-stage2-from_lbs_ckpt_300" \
# --version 4 \
# --eval_use_hybrid_separate \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ 800
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 800 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-13-55-18-NGBC++v7-stage2-from_lbs_ckpt_300/2026-02-10-14-30-10-NGBC++v7-stage2-from_lbs_ckpt_725" \
# --version 4 \
# --eval_use_hybrid_separate \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ best
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --ckpt "./ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-09-13-55-18-NGBC++v7-stage2-from_lbs_ckpt_300/2026-02-10-14-30-10-NGBC++v7-stage2-from_lbs_ckpt_725" \
# --version 4 \
# --eval_use_hybrid_separate \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

