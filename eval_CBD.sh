# CKPT="2025-10-05-22-59-45-NGBCv8"
# CKPT="2025-10-05-23-06-44-NGBCv8"
# CKPT="2025-09-26-10-16-32-NGBCv5"
# CKPT="2025-09-29-14-48-17-NGBCv8"
# CKPT="2025-10-08-02-55-02-NGBCv5"
# CKPT="2025-10-09-01-04-55-NGBCv5"
# CKPT="2025-10-09-10-56-49-NGBCv5"
# CKPT="2025-10-09-02-04-49-NGBCv5" 
# CKPT="2025-10-11-23-12-33-NGBCv5" 
# CKPT="2025-10-15-20-15-53-NGBCv5" 
# CKPT="2025-10-17-19-12-00-NGBCv55" 




# char-s04  2025-10-21-21-00-46-NGBCv5 #1	0	ReLU	hard # scp -r -P 30853 root@143.248.249.137:/source/sihun/NeuralFacialAnimation/ckpts_CBD/2025-10-21-21-00-46-NGBCv5 /source/sihun/NeuralFacialAnimation/ckpts_CBD/2025-10-21-21-00-46-NGBCv5
# char-s03  2025-10-21-21-00-34-NGBCv5 #1	1	ReLU	hard # scp -r -P 32140 root@143.248.249.186:/source/sihun/NeuralFacialAnimation/ckpts_CBD/2025-10-21-21-00-34-NGBCv5 /source/sihun/NeuralFacialAnimation/ckpts_CBD/2025-10-21-21-00-34-NGBCv5
# eve-s05   2025-10-21-21-07-13-NGBCv5 #1	2	ReLU	hard # scp -r -P 32531 root@143.248.249.98:/source/sihun/NeuralFacialAnimation/ckpts_CBD/2025-10-21-21-07-13-NGBCv5 /source/sihun/NeuralFacialAnimation/ckpts_CBD/2025-10-21-21-07-13-NGBCv5
# char-s02  2025-10-21-21-28-38-NGBCv5 #1	1	none 	hard # scp -r -P 30337 root@143.248.249.193:/source/sihun/NeuralFacialAnimation/ckpts_CBD/2025-10-21-21-28-38-NGBCv5 /source/sihun/NeuralFacialAnimation/ckpts_CBD/2025-10-21-21-28-38-NGBCv5
# eve-s05   2025-10-21-21-24-21-NGBCv5 #1	1	softplus	hard # scp -r -P 32531 root@143.248.249.98:/source/sihun/NeuralFacialAnimation/ckpts_CBD/2025-10-21-21-24-21-NGBCv5 /source/sihun/NeuralFacialAnimation/ckpts_CBD/2025-10-21-21-24-21-NGBCv5
# char-s02  2025-10-21-21-39-27-NGBCv5 #1	1	ReLU	soft# scp -r -P 30337 root@143.248.249.193:/source/sihun/NeuralFacialAnimation/ckpts_CBD/2025-10-21-21-39-27-NGBCv5 /source/sihun/NeuralFacialAnimation/ckpts_CBD/2025-10-21-21-39-27-NGBCv5

# CKPT="2025-10-21-21-00-46-NGBCv5" # 1 0 relu hardPOU
# CKPT="2025-10-21-21-00-34-NGBCv5" # 1 1 relu hardPOU
# CKPT="2025-10-21-21-07-13-NGBCv5" # 1 2 relu hardPOU
# CKPT="2025-10-21-21-24-21-NGBCv5" # 1 1 softplus hardPOU
# CKPT="2025-10-21-21-28-38-NGBCv5" # 1 1 none hardPOU
# CKPT="2025-10-21-21-39-27-NGBCv5" # 1 1 none softPOU

# CKPT="2025-10-20-00-15-40-CBD" # NeuralCage
# CKPT="2025-10-22-16-22-45-NGBCv5" # 1 0 
# CKPT="2025-10-23-13-03-54-NGBCv5" # 1 1
# CKPT="2025-10-23-17-32-49-NGBCv5" # 1 1
# CKPT="2025-10-24-18-11-21-NGBCv5" # 1 1 128
# CKPT="2025-10-24-18-03-51-NGBCv5" # 1 1 256
# CKPT="2025-09-28-14-50-34-NGBCv5" # 1 1 softplus
# CKPT="2025-10-03-22-40-03-NGBCv5" # 1 1 sPOU
# CKPT="2025-09-30-16-28-51-NGBCv5" # 1 1 None
# CKPT="2025-10-08-02-55-02-NGBCv5" # 1 1 ELU(0.5)

#python eval_CBD.py --ckpt ./ckpts_CBD/2025-10-22-16-26-26-NGBCv5 --data_selection 4 --realtest --use_eval_data2 --use_t_mask --save_vert --batch_size 1 # 1 2 

# python eval_CBD.py --ckpt ./ckpts_CBD/2025-10-22-16-26-26-NGBCv5 --data_selection 4 --realtest --use_eval_data2 --use_t_mask --use_data2 --save_vert --batch_size 1 # 1 2 


# CKPT='./ckpts_CBD/2025-10-20-00-15-40-CBD'
# VN=1
# # # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 0 --realtest --use_t_mask --batch_size 1
# # # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 1 --realtest --use_t_mask --batch_size 1
# # # # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --use_t_mask --save_vert --batch_size 1
# # # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 3 --realtest --use_t_mask --batch_size 1
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --use_t_mask --save_vert --batch_size 1
# # # # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --use_t_mask --save_vert --batch_size 1


# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask


# CKPT='./ckpt_stage1/2024-08-18-23-32-29-all'
# # # CKPT='./ckpt_stage1/2024-07-08-06-27-12-all'
# VN=0
# # # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 0 --realtest --use_t_mask --batch_size 1
# # # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 1 --realtest --use_t_mask --batch_size 1
# # # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --use_t_mask --save_vert --batch_size 1
# # # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 3 --realtest --use_t_mask --batch_size 1
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --use_t_mask --save_vert --batch_size 1
# # # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --use_t_mask --save_vert --batch_size 1


# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask

# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --batch_size 1 --save_vert --use_t_mask
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --batch_size 1 --save_vert --use_t_mask
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --batch_size 1 --save_vert --use_t_mask



# CKPT='./ckpts_CBD3/2025-11-26-14-46-32-NGBCv1'
# VN=21

# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 0 --realtest --use_t_mask --batch_size 1
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 1 --realtest --use_t_mask --batch_size 1
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --use_t_mask --save_vert --batch_size 1
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 3 --realtest --use_t_mask --batch_size 1
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --use_t_mask --save_vert --batch_size 1
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --use_t_mask --save_vert --batch_size 1

# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --batch_size 1 --save_vert --use_t_mask --start_epoch 800 --continue_ckpt
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --batch_size 1 --save_vert --use_t_mask --start_epoch 800 --continue_ckpt
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --batch_size 1 --save_vert --use_t_mask --start_epoch 800 --continue_ckpt


# CKPT='./ckpts_CBD3/2025-11-21-12-50-24-NGBCv1'
# CKPT='./ckpts_CBD3/2025-12-04-00-29-05-NGBCv1'
# VN=21
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --batch_size 1 --save_vert --use_t_mask --start_epoch 800 --continue_ckpt
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --batch_size 1 --save_vert --use_t_mask --start_epoch 800 --continue_ckpt
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --batch_size 1 --save_vert --use_t_mask --start_epoch 800 --continue_ckpt



## NFS ############################################################ 
# CKPT='./ckpt_stage1/2024-07-08-06-27-12-all'  ### NFS
CKPT='./ckpt_stage1/2024-08-18-23-32-29-all'  ### NFS
# # CKPT='./ckpt_stage1/2024-08-18-23-32-29-all'  ### NFS

VN=0
python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask
python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask
python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask

# # # python eval_CBD.py --version 0 --ckpt ./ckpt_stage1/2024-07-08-06-27-12-all

# CKPT='./ckpt_stage1/exp_019_ICT_MF-jacob_NFR'  ### NFR
# VN=0
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask --use_NFR
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask --use_NFR
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask --use_NFR

# python eval_CBD.py --version 0 --ckpt ./ckpt_stage1/exp_019_ICT_MF-jacob_NFR --use_NFR
# python eval_CBD.py --version 0 --ckpt 'NFR-pretrained' --NFR --data_selection 2 --realtest --use_t_mask --batch_size 1 --save_vert
# python eval_CBD.py --version 0 --ckpt 'NFR-pretrained' --NFR --data_selection 4 --realtest --use_t_mask --batch_size 1 --save_vert
# python eval_CBD.py --version 0 --ckpt 'NFR-pretrained' --NFR --data_selection 5 --realtest --use_t_mask --batch_size 1 --save_vert
##############################################################


# CKPT='NFR-pretrained'
# VN=0
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask --NFR
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask --NFR
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask --NFR
# CKPT='./ckpt_stage1/exp_019_ICT_MF-jacob_NFR'
# VN=0
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 0 --realtest --use_t_mask --batch_size 1 --use_NFR
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 1 --realtest --use_t_mask --batch_size 1 --use_NFR
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --use_t_mask --save_vert --batch_size 1 --use_NFR
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 3 --realtest --use_t_mask --batch_size 1 --use_NFR
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --use_t_mask --save_vert --batch_size 1 --use_NFR
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --use_t_mask --save_vert --batch_size 1 --use_NFR

############### ours ##########################################################################################
# # CKPT='./ckpts_CBD/2025-10-22-22-30-30-NGBCv5'
# # CKPT='./ckpts_CBD/2025-10-09-02-04-49-NGBCv5'
# CKPT='./ckpts_CBD/2025-11-03-14-56-32-NGBCv5' 
# CKPT='./ckpts_CBD/2025-10-23-17-32-49-NGBCv5' # design study
# CKPT='./ckpts_CBD/2025-11-05-16-32-53-NGBCv5' 
# CKPT='./ckpts_CBD/2025-11-08-22-53-46-NGBCv5' 
# CKPT='./ckpts_CBD/2025-11-09-21-21-43-NGBCv5'  ### Ours
# # CKPT='./ckpts_CBD/2025-11-11-01-01-20-NGBCv5' 
# CKPT='./ckpts_CBD/2026-02-25-17-14-34-NGBCv5' 
CKPT='./ckpts_CBD/2026-01-13-14-02-16-NGBCv5-sqrelu'


VN=5
# # # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 0 --realtest --use_t_mask --batch_size 1 --align_latent 
# # # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 1 --realtest --use_t_mask --batch_size 1 --align_latent 
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --use_t_mask --batch_size 1 --save_vert --align_latent --continue_ckpt --start_epoch 50
# # # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 3 --realtest --use_t_mask --batch_size 1 --save_vert --align_latent --continue_ckpt --start_epoch 50
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --use_t_mask --batch_size 1 --save_vert --align_latent --continue_ckpt --start_epoch 50
# # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --use_t_mask --batch_size 1 --save_vert --align_latent --continue_ckpt --start_epoch 50


# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --batch_size 1 --save_vert --align_latent --continue_ckpt --start_epoch 50 --laplacian --use_t_mask
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --batch_size 1 --save_vert --align_latent --continue_ckpt --start_epoch 50 --laplacian --use_t_mask
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --batch_size 1 --save_vert --align_latent --continue_ckpt --start_epoch 50 --laplacian --use_t_mask

### for relu
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --batch_size 1 --save_vert --align_latent --laplacian --use_t_mask
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --batch_size 1 --save_vert --align_latent --laplacian --use_t_mask
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --batch_size 1 --save_vert --align_latent --laplacian --use_t_mask

# #### for sqrelu
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --batch_size 1 --save_vert --align_latent --laplacian --use_t_mask --last_activation 'sqrelu'
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --batch_size 1 --save_vert --align_latent --laplacian --use_t_mask --last_activation 'sqrelu'
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 5 --realtest --batch_size 1 --save_vert --align_latent --laplacian --use_t_mask --last_activation 'sqrelu'
########################################################################################################################



# for CKPT in '2025-09-26-10-16-32-NGBCv5' '2025-10-22-16-22-45-NGBCv5' '2025-10-22-16-26-26-NGBCv5' '2025-10-03-22-40-03-NGBCv5' '2025-09-28-14-50-34-NGBCv5' '2025-09-30-16-28-51-NGBCv5' '2025-10-08-02-55-02-NGBCv5' '2025-10-23-17-32-49-NGBCv5' '2025-10-24-18-03-51-NGBCv5' '2025-10-24-18-11-21-NGBCv5';
# for CKPT in '2025-09-28-14-50-34-NGBCv5' '2025-09-30-16-28-51-NGBCv5'; 
# do
#     echo "evaluating $CKPT"
#     CKPTPATH='./ckpts_CBD/'$CKPT
#     VN=5
#     # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 0 --realtest --use_t_mask --batch_size 1 # voca
#     # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 1 --realtest --use_t_mask --batch_size 1 # biwi
#     # python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 2 --realtest --use_t_mask --batch_size 1 # mf_SEN
#     python eval_CBD.py --version $VN --ckpt $CKPTPATH --data_selection 3 --realtest --use_t_mask --save_vert --batch_size 1 # coma
#     python eval_CBD.py --version $VN --ckpt $CKPTPATH --data_selection 4 --realtest --use_t_mask --save_vert --batch_size 1 # mf_ROM
# done


# CKPT='./ckpts_CBD/2025-10-08-02-55-02-NGBCv5'
# VN=5
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 3 --realtest --use_t_mask --save_vert --batch_size 1 --last_activation 'elu'
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --use_t_mask --save_vert --batch_size 1 --last_activation 'elu'

# CKPT='./ckpts_CBD/2025-09-28-14-50-34-NGBCv5'
# VN=5
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 3 --realtest --use_t_mask --save_vert --batch_size 1 --last_activation 'softplus'
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --use_t_mask --save_vert --batch_size 1 --last_activation 'softplus'


# CKPT='./ckpts_CBD/2025-09-30-16-28-51-NGBCv5'
# VN=5
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 3 --realtest --use_t_mask --save_vert --batch_size 1 --last_activation 'none'
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --use_t_mask --save_vert --batch_size 1 --last_activation 'none'


# CKPT='./ckpts_CBD/2025-10-03-22-40-03-NGBCv5'
# VN=5
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 3 --realtest --use_t_mask --save_vert --batch_size 1 --no_pou
# python eval_CBD.py --version $VN --ckpt $CKPT --data_selection 4 --realtest --use_t_mask --save_vert --batch_size 1 --no_pou


# IT=1
# OT=2

# VN=5
# # VN=1

# NC=512
# # NC=256
# # NC=128

# LA='relu'
# # LA='softplus'
# # LA='none'
# # LA='elu'

# ################# trained with voca / mf / biwi ##################################
# python eval_CBD.py --version $VN --ckpt ./ckpts_CBD/${CKPT} --in_type $IT --out_type $OT --data_selection 0 --last_activation $LA --realtest --num_cage_v $NC --use_eval_data2 --use_t_mask # --save_vert
# python eval_CBD.py --version $VN --ckpt ./ckpts_CBD/${CKPT} --in_type $IT --out_type $OT --data_selection 1 --last_activation $LA --realtest --num_cage_v $NC --use_eval_data2 --use_t_mask
# python eval_CBD.py --version $VN --ckpt ./ckpts_CBD/${CKPT} --in_type $IT --out_type $OT --data_selection 2 --last_activation $LA --realtest --num_cage_v $NC --use_eval_data2 --use_t_mask #--save_vert
# python eval_CBD.py --version $VN --ckpt ./ckpts_CBD/${CKPT} --in_type $IT --out_type $OT --data_selection 3 --last_activation $LA --realtest --num_cage_v $NC --use_eval_data2 --use_t_mask #--save_vert #--save_gt
# # python eval_CBD.py --version $VN --ckpt ./ckpts_CBD/${CKPT} --in_type $IT --out_type $OT --data_selection 3 --last_activation $LA --realtest --num_cage_v $NC --use_eval_data2 --save_vert --no_pou #--save_gt
# python eval_CBD.py --version $VN --ckpt ./ckpts_CBD/${CKPT} --in_type $IT --out_type $OT --data_selection 4 --last_activation $LA --realtest --num_cage_v $NC --use_eval_data2 --use_t_mask #--save_vert

# # # ## ICT
# # python eval_CBD.py --version $VN --ckpt ./ckpts_CBD/${CKPT} --in_type $IT --out_type $OT --data_selection 5 --last_activation $LA --num_cage_v $NC --use_eval_data2 --save_vert
##################################################################################


################# trained with use_data2
# python eval_CBD.py --version $VN --ckpt ./ckpts_CBD/${CKPT} --in_type $IT --out_type $OT --data_selection 0 --last_activation $LA --realtest --num_cage_v $NC --use_eval_data2 --use_data2 ### voca
# python eval_CBD.py --version $VN --ckpt ./ckpts_CBD/${CKPT} --in_type $IT --out_type $OT --data_selection 1 --last_activation $LA --realtest --num_cage_v $NC --use_eval_data2 --use_data2 ### biwi
# python eval_CBD.py --version $VN --ckpt ./ckpts_CBD/${CKPT} --in_type $IT --out_type $OT --data_selection 2 --last_activation $LA --realtest --num_cage_v $NC --use_eval_data2 --use_data2 --save_vert ### mf_SEN
# python eval_CBD.py --version $VN --ckpt ./ckpts_CBD/${CKPT} --in_type $IT --out_type $OT --data_selection 3 --last_activation $LA --realtest --num_cage_v $NC --use_eval_data2 --use_data2 ### coma
# python eval_CBD.py --version $VN --ckpt ./ckpts_CBD/${CKPT} --in_type $IT --out_type $OT --data_selection 4 --last_activation $LA --realtest --num_cage_v $NC --use_eval_data2 --use_data2 --save_vert ### mf_ROM

# # ## ICT
# python eval_CBD.py --version $VN --ckpt ./ckpts_CBD/${CKPT} --in_type $IT --out_type $OT --data_selection 5 --last_activation $LA --num_cage_v $NC --use_eval_data2 --use_data2 --save_vert --batch_size 1
##################################################################################