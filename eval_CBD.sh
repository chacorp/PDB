 ## NBC 
        ## w/ 200
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 3 \
# --continue_ckpt \
# --start_epoch 200 \
# --ckpt "./ckpts_CBD/2026-01-26-09-29-24-NGBCv5/2026-01-28-17-35-52-NGBCv5" \
# --version 3 \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ 300
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 3 \
# --continue_ckpt \
# --start_epoch 300 \
# --ckpt "./ckpts_CBD/2026-01-26-09-29-24-NGBCv5/2026-01-28-17-35-52-NGBCv5" \
# --version 3 \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ 400
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 3 \
# --continue_ckpt \
# --start_epoch 400 \
# --ckpt "./ckpts_CBD/2026-01-26-09-29-24-NGBCv5/2026-01-28-17-35-52-NGBCv5" \
# --version 3 \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ 500
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 3 \
# --continue_ckpt \
# --start_epoch 500 \
# --ckpt "./ckpts_CBD/2026-01-26-09-29-24-NGBCv5/2026-01-28-17-35-52-NGBCv5" \
# --version 3 \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ best
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 3 \
# --ckpt "./ckpts_CBD/2026-01-26-09-29-24-NGBCv5/2026-01-28-17-35-52-NGBCv5" \
# --version 3 \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask


########################################################################
######################## Multiface only
######################## Hybrid LBS + CBD
## 64 joints - 9dof + 512 CBD
        ## w/ 500
python eval_CBD.py \
--eval_use_hybrid \
--no_vis_interv \
--data_selection 4 \
--continue_ckpt \
--start_epoch 500 \
--ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-07-18-10-40-NGBC++v5-stage2-from_lbs_ckpt_300" \
--version 3 \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask

        ## w/ 600
python eval_CBD.py \
--eval_use_hybrid \
--no_vis_interv \
--data_selection 4 \
--continue_ckpt \
--start_epoch 600 \
--ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-07-18-10-40-NGBC++v5-stage2-from_lbs_ckpt_300" \
--version 3 \
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
--ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-07-18-10-40-NGBC++v5-stage2-from_lbs_ckpt_300/2026-02-08-10-25-47-NGBC++v5-stage2-from_lbs_ckpt_650" \
--version 3 \
--eval_use_hybrid \
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
--ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-07-18-10-40-NGBC++v5-stage2-from_lbs_ckpt_300/2026-02-08-10-25-47-NGBC++v5-stage2-from_lbs_ckpt_650" \
--version 3 \
--eval_use_hybrid \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask

        ## w/ best
python eval_CBD.py \
--no_vis_interv \
--data_selection 4 \
--ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-07-18-10-40-NGBC++v5-stage2-from_lbs_ckpt_300/2026-02-08-10-25-47-NGBC++v5-stage2-from_lbs_ckpt_650" \
--version 3 \
--eval_use_hybrid \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask


######################## LBS only 
## 64 joints - 9dof
#          ## w/ 200
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 200 \
# --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5" \
# --version 3 \
# --eval_use_lbs \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## w/ 300
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 300 \
# --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5" \
# --version 3 \
# --eval_use_lbs \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask
#         ## w/ 400
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 400 \
# --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5" \
# --version 3 \
# --eval_use_lbs \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask
#         ## NBC w/ 500
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 500 \
# --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5" \
# --version 3 \
# --eval_use_lbs \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask
# #         ## best
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5" \
# --version 3 \
# --eval_use_lbs \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask


## 128 joints - 9dof
         ## w/ 200
python eval_CBD.py \
--no_vis_interv \
--data_selection 4 \
--continue_ckpt \
--start_epoch 200 \
--ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-29-44-NGBC++v5" \
--version 3 \
--eval_use_lbs \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask

        ## w/ 300
python eval_CBD.py \
--no_vis_interv \
--data_selection 4 \
--continue_ckpt \
--start_epoch 300 \
--ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-29-44-NGBC++v5" \
--version 3 \
--eval_use_lbs \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask
        ## w/ 400
python eval_CBD.py \
--no_vis_interv \
--data_selection 4 \
--continue_ckpt \
--start_epoch 400 \
--ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-29-44-NGBC++v5" \
--version 3 \
--eval_use_lbs \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask
        ## NBC w/ 500
python eval_CBD.py \
--no_vis_interv \
--data_selection 4 \
--continue_ckpt \
--start_epoch 500 \
--ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-29-44-NGBC++v5" \
--version 3 \
--eval_use_lbs \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask
#         ## best
python eval_CBD.py \
--no_vis_interv \
--data_selection 4 \
--ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-29-44-NGBC++v5" \
--version 3 \
--eval_use_lbs \
--realtest \
--batch_size 1 \
--save_vert \
--use_t_mask




        ########################## CBD only ##########################
## MF only (multiface all : sen + rom)
#         ## NBC w/ 200
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 200 \
# --ckpt "./ckpts_CBD/2026-02-03-10-59-23-NGBCv5" \
# --version 3 \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask
#         ## NBC w/ 300
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 300 \
# --ckpt "./ckpts_CBD/2026-02-03-10-59-23-NGBCv5" \
# --version 3 \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask
#         ## NBC w/ 400
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 400 \
# --ckpt "./ckpts_CBD/2026-02-03-10-59-23-NGBCv5" \
# --version 3 \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask
#         ## NBC w/ 500
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --continue_ckpt \
# --start_epoch 500 \
# --ckpt "./ckpts_CBD/2026-02-03-10-59-23-NGBCv5" \
# --version 3 \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask
#         ## best
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 4 \
# --ckpt "./ckpts_CBD/2026-02-03-10-59-23-NGBCv5" \
# --version 3 \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask





########################################################################
######################## coma only ##########################

# #     ## w/ weighted joint pos + 64 joints
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 3 \
# --continue_ckpt \
# --start_epoch 400 \
# --ckpt "./ckpts_CBD/2026-01-21-12-59-49-NGBC++v5" \
# --version 3 \
# --eval_use_lbs \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## e300
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 3 \
# --continue_ckpt \
# --start_epoch 500 \
# --ckpt "./ckpts_CBD/2026-01-21-12-59-49-NGBC++v5" \
# --version 3 \
# --eval_use_lbs \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

#         ## best loss
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 3 \
# --ckpt "./ckpts_CBD/2026-01-21-12-59-49-NGBC++v5" \
# --version 3 \
# --eval_use_lbs \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask 


#     ## w/ weighted joint pos + 128 joints
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 3 \
# --continue_ckpt \
# --start_epoch 200 \
# --ckpt "./ckpts_CBD/2026-01-21-13-01-39-NGBC++v5" \
# --version 3 \
# --eval_use_lbs \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask
        ## e300
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 3 \
# --continue_ckpt \
# --start_epoch 300 \
# --ckpt "./ckpts_CBD/2026-01-21-13-01-39-NGBC++v5" \
# --version 3 \
# --eval_use_lbs \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask


     ## w/ weighted joint pos + 256 joints 
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 3 \
# --continue_ckpt \
# --start_epoch 200 \
# --ckpt "./ckpts_CBD/2026-01-21-13-01-46-NGBC++v5" \
# --version 3 \
# --eval_use_lbs \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask
    ## w/ weighted joint pos + 512 joints (not yet trained)
# python eval_CBD.py \
# --no_vis_interv \
# --data_selection 3 \
# --continue_ckpt \
# --start_epoch 200 \
# --ckpt "./ckpts_CBD/2026-01-21-13-01-53-NGBC++v5" \
# --version 3 \
# --eval_use_lbs \
# --realtest \
# --batch_size 1 \
# --save_vert \
# --use_t_mask

