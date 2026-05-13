#########################
## with Multiface only ##
#########################

############
## Hybrid ##
############

    #####################
    ## residualized input
    #####################
    ## with 32 joint pos extracted from 9 DOF prediction
    ## with residualized input over LBS prediction
    # python train_CBD.py \
    # --use_hyb_delta_lbs_input \
    # --max_epoch 1000 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 100000 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data1 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 32 \
    # --use_lbs_joint_center \
    # --use_perm \
    # --save_interval 25 \
    # --continue_ckpt \
    # --start_epoch 500 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-07-03-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-07-03-NGBC++v5" \
    # --start_stage 2 \
    # --num_cage_v 512 \
    # --data_toggle


    ## with 64 joint pos extracted from 9 DOF prediction
    ## with residualized input over LBS prediction
    # python train_CBD.py \
    # --use_hyb_delta_lbs_input \
    # --max_epoch 800 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 100000 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data1 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 64 \
    # --use_lbs_joint_center \
    # --use_perm \
    # --save_interval 25 \
    # --continue_ckpt \
    # --start_epoch 300 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5" \
    # --start_stage 2 \
    # --num_cage_v 512 \
    # --data_toggle

    #     ## continue
    #     python train_CBD.py \
    #     --use_hyb_delta_lbs_input \
    #     --max_epoch 800 \
    #     --tb \
    #     --lr 1e-4 \
    #     --sc_step 100000 \
    #     --in_type 1 \
    #     --out_type 1 \
    #     --batch_size 16 \
    #     --version 5 \
    #     --use_data1 \
    #     --last_activation 'relu' \
    #     --use_lbs \
    #     --num_lbs_joints 64 \
    #     --use_lbs_joint_center \
    #     --use_perm \
    #     --save_interval 25 \
    #     --continue_ckpt \
    #     --start_epoch 650 \
    #     --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-07-18-10-40-NGBC++v5-stage2-from_lbs_ckpt_300" \
    #     --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5/2026-02-07-18-10-40-NGBC++v5-stage2-from_lbs_ckpt_300" \
    #     --start_stage 2 \
    #     --num_cage_v 512 \
    #     --data_toggle


    ## with 128 joint pos extracted from 9 DOF prediction
    ## with residualized input over LBS prediction 
    ## w/ model_best.pth (477) 
    # python train_CBD.py \
    # --use_hyb_delta_lbs_input \
    # --max_epoch 977 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 100000 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data1 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 128 \
    # --use_lbs_joint_center \
    # --use_perm \
    # --save_interval 25 \
    # --start_epoch 477 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-29-44-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-29-44-NGBC++v5" \
    # --start_stage 2 \
    # --num_cage_v 512 \
    # --data_toggle


    ################
    ## naive default
    ################
    
    ## with 32 joint pos extracted from 9 DOF prediction
    # python train_CBD.py \
    # --max_epoch 1000 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 100000 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data1 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 32 \
    # --use_lbs_joint_center \
    # --use_perm \
    # --save_interval 25 \
    # --continue_ckpt \
    # --start_epoch 500 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-07-03-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-07-03-NGBC++v5" \
    # --start_stage 2 \
    # --num_cage_v 512 
    # # --data_toggle \
    # # --use_perm \
    # # --use_exp_joint_predict \
    # # --use_data2 --data_toggle \
    # # --use_lbs_laplacian \
    # # --use_lbs_ent \
    # # --use_lbs_t \
    # # --use_lbs_R \
    # # --use_lbs_bal \

        ## continue
        # python train_CBD.py \
        # --max_epoch 1000 \
        # --tb \
        # --lr 1e-4 \
        # --sc_step 100000 \
        # --in_type 1 \
        # --out_type 1 \
        # --batch_size 16 \
        # --version 5 \
        # --use_data1 \
        # --last_activation 'relu' \
        # --use_lbs \
        # --num_lbs_joints 32 \
        # --use_lbs_joint_center \
        # --use_perm \
        # --save_interval 25 \
        # --continue_ckpt \
        # --start_epoch 800 \
        # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-07-03-NGBC++v5/2026-02-06-11-46-17-NGBC++v5-stage2-from_lbs_ckpt_500" \
        # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-07-03-NGBC++v5/2026-02-06-11-46-17-NGBC++v5-stage2-from_lbs_ckpt_500" \
        # --start_stage 2 \
        # --num_cage_v 512 
        # # --data_toggle \
        # # --use_perm \
        # # --use_exp_joint_predict \
        # # --use_data2 --data_toggle \
        # # --use_lbs_laplacian \
        # # --use_lbs_ent \
        # # --use_lbs_t \
        # # --use_lbs_R \
        # # --use_lbs_bal \


    ## with 64 joint pos extracted from 9 DOF prediction
    # python train_CBD.py \
    # --max_epoch 800 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 100000 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data1 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 64 \
    # --use_lbs_joint_center \
    # --use_perm \
    # --save_interval 25 \
    # --continue_ckpt \
    # --start_epoch 300 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-10-40-NGBC++v5" \
    # --start_stage 2 \
    # --num_cage_v 512 \
    # --data_toggle


    ## with 128 joint pos extracted from 9 DOF prediction
    ## w/ best loss epoch (477)
    # python train_CBD.py \
    # --max_epoch 977 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 100000 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data1 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 128 \
    # --use_lbs_joint_center \
    # --use_perm \
    # --save_interval 25 \
    # --start_epoch 650 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-29-44-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-29-44-NGBC++v5" \
    # --start_stage 2 \
    # --num_cage_v 512 \
    # --data_toggle

        # ## continue
        #     python train_CBD.py \
        #     --max_epoch 977 \
        #     --tb \
        #     --lr 1e-4 \
        #     --sc_step 100000 \
        #     --in_type 1 \
        #     --out_type 1 \
        #     --batch_size 16 \
        #     --version 5 \
        #     --use_data1 \
        #     --last_activation 'relu' \
        #     --use_lbs \
        #     --num_lbs_joints 128 \
        #     --use_lbs_joint_center \
        #     --use_perm \
        #     --save_interval 25 \
        #     --continue_ckpt \
        #     --start_epoch 650 \
        #     --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-29-44-NGBC++v5/2026-02-07-19-39-26-NGBC++v5-stage2-from_lbs_ckpt_477" \
        #     --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-02-03-11-29-44-NGBC++v5/2026-02-07-19-39-26-NGBC++v5-stage2-from_lbs_ckpt_477" \
        #     --start_stage 2 \
        #     --num_cage_v 512 \
        #     --data_toggle


###########
### CBD ###
###########

## with CB 512 (default)
# python train_CBD.py \
# --num_cage_v 512 \
# --max_epoch 500 \
# --tb \
# --lr 1e-4 \
# --sc_step 100000 \
# --in_type 1 \
# --out_type 1 \
# --batch_size 16 \
# --version 5 \
# --use_data1 \
# --last_activation 'relu' \
# --use_perm \
# --data_toggle \
# --save_interval 25 \


###########
### LBS ###
###########

    ##################################
    ### 9 dof joint pos extraction ###
    ##################################

    ## with 32 joint 
    # python train_CBD.py \
    # --max_epoch 500 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 100000 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data1 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --use_lbs_joint_center \
    # --num_lbs_joints 32 \
    # --lbs_pretrained_epochs 600 \
    # --data_toggle \
    # --use_perm \
    # --continue_ckpt \
    # --start_epoch 100 \
    # --ckpt './ckpts_CBD/2026-02-03-11-07-03-NGBC++v5' \
    # --log_dir './ckpts_CBD/2026-02-03-11-07-03-NGBC++v5'

    ## with 64 joint 
    # python train_CBD.py \
    # --max_epoch 500 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 100000 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data1 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --use_lbs_joint_center \
    # --num_lbs_joints 64 \
    # --lbs_pretrained_epochs 600 \
    # --data_toggle \
    # --use_perm \
    # --continue_ckpt \
    # --start_epoch 50 \
    # --ckpt './ckpts_CBD/2026-02-03-11-10-40-NGBC++v5' \
    # --log_dir './ckpts_CBD/2026-02-03-11-10-40-NGBC++v5'

    ## with 128 joint 
    # python train_CBD.py \
    # --max_epoch 500 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 100000 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data1 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --use_lbs_joint_center \
    # --num_lbs_joints 128 \
    # --lbs_pretrained_epochs 600 \
    # --data_toggle \
    # --use_perm \
    # --continue_ckpt \
    # --start_epoch 200 \
    # --ckpt './ckpts_CBD/2026-02-03-11-29-44-NGBC++v5' \
    # --log_dir './ckpts_CBD/2026-02-03-11-29-44-NGBC++v5'


#############################################################################################################################


####################
## with COMA only ## >> moved to Multiface_only experiment
####################

############
## Hybrid ##
############

    ## with 32 joint pos extracted from 9 DOF prediction
    # python train_CBD.py \
    # --max_epoch 700 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 100000 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 32 \
    # --use_lbs_joint_center \
    # --use_perm \
    # --save_interval 25 \
    # --continue_ckpt \
    # --start_epoch 200 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-26-10-49-50-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-26-10-49-50-NGBC++v5" \
    # --start_stage 2 \
    # --num_cage_v 512 \
    # --data_toggle
    # # --use_perm \
    # # --use_exp_joint_predict \
    # # --use_data2 --data_toggle \
    # # --use_lbs_laplacian \
    # # --use_lbs_ent \
    # # --use_lbs_t \
    # # --use_lbs_R \
    # # --use_lbs_bal \
    
    ## with 64 joint pos extracted from 9 DOF prediction
    # python train_CBD.py \
    # --max_epoch 1000 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 100000 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 64 \
    # --use_lbs_joint_center \
    # --use_perm \
    # --save_interval 25 \
    # --continue_ckpt \
    # --start_epoch 500 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-26-10-46-17-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-26-10-46-17-NGBC++v5" \
    # --start_stage 2 \
    # --num_cage_v 512 \
    # --data_toggle
    # # --use_perm \
    # # --use_exp_joint_predict \
    # # --use_data2 --data_toggle \
    # # --use_lbs_laplacian \
    # # --use_lbs_ent \
    # # --use_lbs_t \
    # # --use_lbs_R \
    # # --use_lbs_bal \

    ## with 128 joint pos extracted from 9 DOF prediction
    # python train_CBD.py \
    # --max_epoch 900 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 100000 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 128 \
    # --use_lbs_joint_center \
    # --use_perm \
    # --save_interval 25 \
    # --continue_ckpt \
    # --start_epoch 775 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-26-10-52-30-NGBC++v5/2026-01-31-12-23-10-NGBC++v5-stage2-from_lbs_ckpt_400" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-26-10-52-30-NGBC++v5/2026-01-31-12-23-10-NGBC++v5-stage2-from_lbs_ckpt_400" \
    # --start_stage 2 \
    # --num_cage_v 512 \
    # --data_toggle
    # # --use_perm \
    # # --use_exp_joint_predict \
    # # --use_data2 --data_toggle \
    # # --use_lbs_laplacian \
    # # --use_lbs_ent \
    # # --use_lbs_t \
    # # --use_lbs_R \
    # # --use_lbs_bal \

###########
### CBD ###
###########

    ## CBD
    ## with CV 1024 (default)
    # python train_CBD.py \
    # --max_epoch 600 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 10 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_perm \
    # # --num_lbs_joints 8 \
    # # --lbs_pretrained_epochs 600 \
    # # --use_lbs \
    # # --start_epoch 103 \
    # # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-14-07-21-58-NGBC++v5" \
    # # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-14-07-21-58-NGBC++v5" \
    # # --use_lbs_joint_center \
    # # --use_exp_joint_predict \
    # # --use_data2 --data_toggle \
    # # --use_lbs_laplacian \
    # # --use_lbs_ent \
    # # --use_lbs_t \
    # # --use_lbs_R \
    # # --use_lbs_bal \
    
    ## with CV 512
    # python train_CBD.py \
    # --num_cage_v 512 \
    # --max_epoch 500 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 100000 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_perm \
    # --data_toggle \
    # --save_interval 25 \
    # --continue_ckpt \
    # --start_epoch 150 \
    # --ckpt '/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-26-09-29-24-NGBCv5' \
    # --log_dir '/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-26-09-29-24-NGBCv5' 
    # # --num_lbs_joints 8 \
    # # --lbs_pretrained_epochs 600 \
    # # --use_lbs \
    # # --use_lbs_joint_center \
    # # --use_exp_joint_predict \
    # # --use_data2 --data_toggle \
    # # --use_lbs_laplacian \
    # # --use_lbs_ent \
    # # --use_lbs_t \
    # # --use_lbs_R \
    # # --use_lbs_bal \


###########
### LBS ###
###########

    ###################################
    ### 9 dof joint pos extraction >> seclected as main experiment
    ###################################

    ## LBS (64 joints) 
    # python train_CBD.py \
    # --max_epoch 500 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 100000 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 64 \
    # --use_lbs_joint_center \
    # --no_use_translation \
    # --use_weighted_joint_pos \
    # --lbs_pretrained_epochs 500 \
    # --data_toggle \
    # --use_perm \
    # --save_interval 25 \
    # --continue_ckpt \
    # --start_epoch 300 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-12-59-49-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-12-59-49-NGBC++v5" \

    ## with joint position extraction from 9 DOF pose prediciton
    # python train_CBD.py \
    # --max_epoch 500 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 10 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --use_lbs_joint_center \
    # --num_lbs_joints 64 \
    # --lbs_pretrained_epochs 600 \
    # --data_toggle \
    # --use_perm \
    # # --no_use_translation \
    # # --use_weighted_joint_pos \
    # # --save_interval 25 \
    # # --continue_ckpt \
    # # --start_epoch 200 \
    # # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-12-59-49-NGBC++v5" \
    # # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-12-59-49-NGBC++v5" \
    # # --continue_ckpt \
    # # --use_exp_joint_predict \
    # # --use_data2 --data_toggle \
    # # --use_lbs_laplacian \
    # # --use_lbs_ent \
    # # --use_lbs_t \
    # # --use_lbs_R \
    # # --use_lbs_bal \


    # ## LBS (32 joints) - use_perm 끄고,  at eve-s01 experiment 
        ## with joint position extraction from 9 DOF pose prediciton
    # python train_CBD.py \
    # --max_epoch 600 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 10 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --use_lbs_joint_center \
    # --num_lbs_joints 32 \
    # --lbs_pretrained_epochs 600 \
    # --data_toggle \
    # --use_perm \
    #     # --no_use_translation \
    #     # --use_weighted_joint_pos \
    #     # --save_interval 25 \
    #     # --start_epoch 164 \
    #     # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-13-01-39-NGBC++v5" \
    #     # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-13-01-39-NGBC++v5"
    #     # --continue_ckpt \
    #     # --use_exp_joint_predict \
    #     # --use_data2 --data_toggle \
    #     # --use_lbs_laplacian \
    #     # --use_lbs_ent \
    #     # --use_lbs_t \
    #     # --use_lbs_R \
    #     # --use_lbs_bal \


    # ## LBS (64 joints) - use_perm 끄고,  at eve-s01 experiment 
        ## with joint position extraction from 9 DOF pose prediciton
    # python train_CBD.py \
    # --max_epoch 600 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 10 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --use_lbs_joint_center \
    # --num_lbs_joints 64 \
    # --lbs_pretrained_epochs 600 \
    # --data_toggle \
    # --use_perm \
    # # --vis_joint_pos \ 
    #     # --no_use_translation \
    #     # --use_weighted_joint_pos \
    #     # --save_interval 25 \
    #     # --start_epoch 164 \
    #     # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-13-01-39-NGBC++v5" \
    #     # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-13-01-39-NGBC++v5"
    #     # --continue_ckpt \
    #     # --use_exp_joint_predict \
    #     # --use_data2 --data_toggle \
    #     # --use_lbs_laplacian \
    #     # --use_lbs_ent \
    #     # --use_lbs_t \
    #     # --use_lbs_R \
    #     # --use_lbs_bal \


    ###################################
    ### use_weighted_joint_pos == True
    ###################################

    # ## LBS (8 joints) - use_perm 끄고, 
    # python train_CBD.py \
    # --max_epoch 600 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 10 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 8 \
    # --use_lbs_joint_center \
    # --no_use_translation \
    # --use_weighted_joint_pos \
    # --lbs_pretrained_epochs 600 \
    # --use_perm \
    # --start_epoch 91 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-12-55-57-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-12-55-57-NGBC++v5" \
    # # # --use_exp_joint_predict \
    # # # --use_data2 --data_toggle \
    # # # --use_lbs_laplacian \
    # # # --use_lbs_ent \
    # # # --use_lbs_t \
    # # # --use_lbs_R \
    # # # --use_lbs_bal \


    # ## LBS (16 joints) - use_perm 끄고, 
    # python train_CBD.py \
    # --max_epoch 600 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 10 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 16 \
    # --use_lbs_joint_center \
    # --no_use_translation \
    # --use_weighted_joint_pos \
    # --lbs_pretrained_epochs 600 \
    # --use_perm \
    # --start_epoch 225 \
    # --continue_ckpt \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-12-58-19-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-12-58-19-NGBC++v5" \
    # --save_interval 25
    # # --use_exp_joint_predict \
    # # --use_data2 --data_toggle \
    # # --use_lbs_laplacian \
    # # --use_lbs_ent \
    # # --use_lbs_t \
    # # --use_lbs_R \
    # # --use_lbs_bal \


    # ## LBS (32 joints) (64 at eve-s01  , 128 at eve-s01  로 먼저 가보자)
    # python train_CBD.py \
    # --max_epoch 600 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 10 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 32 \
    # --use_lbs_joint_center \
    # --no_use_translation \
    # --use_weighted_joint_pos \
    # --lbs_pretrained_epochs 600 \
    # --use_perm \
    # --save_interval 25 \
    # --continue_ckpt \
    # --start_epoch 250 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-12-59-22-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-12-59-22-NGBC++v5" \
    # # --data_toggle \
    # # --use_exp_joint_predict \
    # # --use_data2 --data_toggle \
    # # --use_lbs_laplacian \
    # # --use_lbs_ent \
    # # --use_lbs_t \
    # # --use_lbs_R \
    # # --use_lbs_bal \


    # ## LBS (64 joints) (64 at char-s02-nbcpp-02)
    # python train_CBD.py \
    # --max_epoch 500 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 100000 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 64 \
    # --use_lbs_joint_center \
    # --no_use_translation \
    # --use_weighted_joint_pos \
    # --lbs_pretrained_epochs 500 \
    # --use_perm \
    # --save_interval 25 \
    # --continue_ckpt \
    # --start_epoch 300 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-12-59-49-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-12-59-49-NGBC++v5" \
    # --start_stage 2
    # # --data_toggle \
    # # --use_perm \
    # # --use_exp_joint_predict \
    # # --use_data2 --data_toggle \
    # # --use_lbs_laplacian \
    # # --use_lbs_ent \
    # # --use_lbs_t \
    # # --use_lbs_R \
    # # --use_lbs_bal \


    ## LBS (128 joints) - use_perm 끄고, at eve-s01-nbcpp-01
    # python train_CBD.py \
    # --max_epoch 600 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 10 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 128 \
    # --use_lbs_joint_center \
    # --no_use_translation \
    # --use_weighted_joint_pos \
    # --lbs_pretrained_epochs 600 \
    # --data_toggle \
    # --use_perm \
    # --save_interval 25 \
    # --start_epoch 164 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-13-01-39-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-13-01-39-NGBC++v5"
    # # --continue_ckpt \
    # # --use_exp_joint_predict \
    # # --use_data2 --data_toggle \
    # # --use_lbs_laplacian \
    # # --use_lbs_ent \
    # # --use_lbs_t \
    # # --use_lbs_R \
    # # --use_lbs_bal \


    # ## LBS (256 joints) - use_perm 끄고,  at eve-s01 experiment 14
    # python train_CBD.py \
    # --max_epoch 600 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 10 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 256 \
    # --use_lbs_joint_center \
    # --no_use_translation \
    # --use_weighted_joint_pos \
    # --lbs_pretrained_epochs 600 \
    # --data_toggle \
    # --use_perm \
    # --save_interval 25 \
    # --continue_ckpt \
    # --start_epoch 50 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-13-01-46-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-13-01-46-NGBC++v5"
    # # --continue_ckpt \
    # # --use_exp_joint_predict \
    # # --use_data2 --data_toggle \
    # # --use_lbs_laplacian \
    # # --use_lbs_ent \
    # # --use_lbs_t \
    # # --use_lbs_R \
    # # --use_lbs_bal \


    # ## LBS (512 joints) (32 at char-s02, 64 at eve-s01 experiment 15
    # python train_CBD.py \
    # --max_epoch 600 \
    # --tb \
    # --lr 1e-4 \
    # --sc_step 10 \
    # --in_type 1 \
    # --out_type 1 \
    # --batch_size 16 \
    # --version 5 \
    # --use_data0 \
    # --last_activation 'relu' \
    # --use_lbs \
    # --num_lbs_joints 512 \
    # --use_lbs_joint_center \
    # --no_use_translation \
    # --use_weighted_joint_pos \
    # --lbs_pretrained_epochs 600 \
    # --data_toggle \
    # --use_perm \
    # --continue_ckpt \
    # --start_epoch 300 \
    # --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-13-01-53-NGBC++v5" \
    # --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-21-13-01-53-NGBC++v5"
    # # --continue_ckpt \
    # # --use_exp_joint_predict \
    # # --use_data2 --data_toggle \
    # # --use_lbs_laplacian \
    # # --use_lbs_ent \
    # # --use_lbs_t \
    # # --use_lbs_R \
    # # --use_lbs_bal \










################################################################################# old ########################################################################

# python train_CBD.py \
# --max_epoch 200 \
# --tb \
# --lr 2E-4 \
# --sc_step 10 \
# --version 5 \
# --batch_size 32 \
# --num_cage_v 512 \
# --in_type 1 \
# --out_type 1


# ## LBS only (WIP)
#     ## no joint center 
#     ## 2026-01-14-07-21-58-NGBC++v5
# python train_CBD.py \
# --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-14-07-21-58-NGBC++v5" \
# --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-14-07-21-58-NGBC++v5" \
# --max_epoch 600 \
# --tb \
# --lr 1e-4 \
# --sc_step 10 \
# --in_type 1 \
# --out_type 1 \
# --batch_size 32 \
# --version 5 \
# --use_lbs \
# --num_lbs_joints 8 \
# --lbs_pretrained_epochs 600 \
# # --use_lbs_joint_center \
# # --use_exp_joint_predict \
# # --use_data2 --data_toggle \
# # --use_lbs_laplacian \
# # --use_lbs_ent \
# # --use_lbs_t \
# # --use_lbs_R \
# # --use_lbs_bal \


#     ## (WIP)
#     ## joint center w/ expression for center prediction
#     ## 2026-01-14-07-23-30-NGBC++v5
# python train_CBD.py \
# --ckpt "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-14-07-23-30-NGBC++v5" \
# --log_dir "/source/inyup/NeuralFacialAnimation/ckpts_CBD/2026-01-14-07-23-30-NGBC++v5" \
# --max_epoch 600 \
# --tb \
# --lr 1e-4 \
# --sc_step 10 \
# --in_type 1 \
# --out_type 1 \
# --batch_size 32 \
# --version 5 \
# --use_lbs \
# --num_lbs_joints 8 \
# --lbs_pretrained_epochs 600 \
# --use_lbs_joint_center \
# --use_exp_joint_predict \
# # --use_data2 --data_toggle \
# # --use_lbs_laplacian \
# # --use_lbs_ent \
# # --use_lbs_t \
# # --use_lbs_R \
# # --use_lbs_bal \
# # --debug_stage


## LBS only (WIP)
    ## weighted joint center prediction
    ## 2026-01-15-17-31-09-NGBC++v5
# python train_CBD.py \
# --max_epoch 600 \
# --tb \
# --lr 1e-4 \
# --sc_step 10 \
# --in_type 1 \
# --out_type 1 \
# --batch_size 32 \
# --version 5 \
# --use_lbs \
# --num_lbs_joints 8 \
# --lbs_pretrained_epochs 600 \
# --no_use_translation \
# --use_lbs_joint_center \
# --use_weighted_joint_pos \
# # --use_exp_joint_predict \
# # --use_data2 --data_toggle \
# # --use_lbs_laplacian \
# # --use_lbs_ent \
# # --use_lbs_t \
# # --use_lbs_R \
# # --use_lbs_bal \


## LBS only (WIP)
    ## # of joints: 16  
    ## joint center prediction: weighted  
    ## use_perm = True, 
    ## dataset: ict + Multiface
    ## 2026-01-16-12-25-45-NGBC++v5
# python train_CBD.py \
# --max_epoch 600 \
# --tb \
# --lr 1e-4 \
# --sc_step 10 \
# --in_type 1 \
# --out_type 1 \
# --batch_size 32 \
# --version 5 \
# --use_lbs \
# --num_lbs_joints 16 \
# --lbs_pretrained_epochs 600 \
# --no_use_translation \
# --use_lbs_joint_center \
# --use_weighted_joint_pos \
# --use_data2 --data_toggle \ 
# # --use_exp_joint_predict \
# # --use_lbs_laplacian \
# # --use_lbs_ent \
# # --use_lbs_t \
# # --use_lbs_R \
# # --use_lbs_bal \


## LBS only (WIP) 
    ## # of joints: 16  
    ## joint center prediction: weighted  
    ## use_perm = True, 
    ## dataset: VOCA, COMA, BIWI, Multiface
    ## last_activation: ReLU
# python train_CBD.py \
# --max_epoch 600 \
# --tb \
# --lr 1e-4 \
# --sc_step 100000 \
# --in_type 1 \
# --out_type 1 \
# --batch_size 32 \
# --version 5 \
# --use_lbs \
# --num_lbs_joints 16 \
# --lbs_pretrained_epochs 600 \
# --no_use_translation \
# --use_lbs_joint_center \
# --use_weighted_joint_pos \
# --data_toggle \
# --last_activation 'relu' \
# # --use_exp_joint_predict \
# # --use_lbs_laplacian \
# # --use_lbs_ent \
# # --use_lbs_t \
# # --use_lbs_R \
# # --use_lbs_bal \


## LBS only (WIP) 
    ## # of joints: 16  
    ## joint center prediction: weighted  
    ## use_perm = True, 
    ## dataset: VOCA, COMA, BIWI, Multiface
    ## last_activation: ReLU    
    ## use_lbs_ent
# python train_CBD.py \
# --max_epoch 600 \
# --tb \
# --lr 1e-4 \
# --sc_step 100000 \
# --in_type 1 \
# --out_type 1 \
# --batch_size 32 \
# --version 5 \
# --use_lbs \
# --num_lbs_joints 16 \
# --lbs_pretrained_epochs 600 \
# --no_use_translation \
# --use_lbs_joint_center \
# --use_weighted_joint_pos \
# --data_toggle \
# --last_activation 'relu' \
# --use_lbs_ent \
# # --use_lbs_laplacian \
# # --use_exp_joint_predict \
# # --use_lbs_t \
# # --use_lbs_R \
# # --use_lbs_bal \


## LBS + CBD (option 1)
    ## no joint center
    ## ## lets do it after we get LBS only 600 epoch
# python train_CBD.py \
# --max_epoch 600 \
# --tb \
# --lr 1e-4 \
# --sc_step 10 \
# --in_type 1 \
# --out_type 1 \
# --batch_size 32 \
# --version 5 \
# --use_lbs \
# --num_lbs_joints 8 \
# --lbs_pretrained_epochs 200 \
# # --use_lbs_joint_center \
# # --use_exp_joint_predict \
# # --use_data2 --data_toggle \
# # --use_lbs_laplacian \
# # --use_lbs_ent \
# # --use_lbs_t \
# # --use_lbs_R \
# # --use_lbs_bal \
# # --debug_stage

    ## joint center w/ expression for center prediction
    ## lets do it after we get LBS only 600 epoch
# python train_CBD.py \
# --max_epoch 600 \
# --tb \
# --lr 1e-4 \
# --sc_step 10 \
# --in_type 1 \
# --out_type 1 \
# --batch_size 32 \
# --version 5 \
# --use_lbs \
# --num_lbs_joints 8 \
# --lbs_pretrained_epochs 400 \
# --use_lbs_joint_center \
# --use_exp_joint_predict \
# --use_data2 --data_toggle \
# --use_lbs_laplacian \
# --use_lbs_ent \
# --use_lbs_t \
# --use_lbs_R \
# --use_lbs_bal \
# --debug_stage 
