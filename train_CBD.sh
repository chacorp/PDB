# python train_CBD.py \
#     --max_epoch 200 \
#     --version 1 \
#     --batch_size 16 \
#     --in_type 1 \
#     --out_type 1 \
#     --optim_cage --data_toggle

# python train_CBD.py \
# --max_epoch 200 \
# --lr 1E-4 \
# --sc_step 20 \
# --version 5 \
# --batch_size 16 \
# --num_cage_v 512 \
# --in_type 1 \
# --out_type 1 \
# --last_activation 'relu' --use_data2 --start_epoch 100 --ckpt './ckpts_CBD/2025-10-09-02-04-49-NGBCv5' --continue_ckpt --data_toggle

# python train_CBD.py --max_epoch 200 --lr 15E-5 --sc_step 10 --version 5 --batch_size 16 --num_cage_v 512 --in_type 1 --out_type 1 --last_activation 'relu' --use_data2 --data_toggle
# python train_CBD.py --max_epoch 200 --lr 1E-4 --sc_step 10 --version 5 --batch_size 16 --num_cage_v 512 --in_type 1 --out_type 1 --last_activation 'relu' --start_epoch 100 --ckpt './ckpts_CBD/2025-10-09-02-04-49-NGBCv5' --continue_ckpt --data_toggle

# python train_CBD.py --max_epoch 200 --lr 2E-4 --sc_step 25 --version 5 --batch_size 16 --num_cage_v 512 --in_type 1 --out_type 1 --last_activation 'relu' --use_data0 --data_toggle
# python train_CBD.py --max_epoch 2000 --lr 1E-4 --sc_step 1000 --batch_size 8 --num_cage_v 512 --in_type 1 --out_type 1 --last_activation 'relu' --data_toggle --use_data8 --log_dir ckpts_CBD --version 5 --align_latent --start_epoch 780 --ckpt './ckpts_CBD/2026-02-25-17-14-34-NGBCv5-relu' --continue_ckpt


# python train_CBD.py --max_epoch 200 --lr 1E-4 --sc_step 20 --batch_size 8 --num_cage_v 512 --in_type 1 --out_type 1 --last_activation 'relu' --data_toggle --use_data1 --log_dir ckpts_CBD --version 5 --start_epoch 100 --ckpt './ckpts_CBD/2025-09-27-08-05-11-NGBCv5' --continue_ckpt
python train_CBD.py --max_epoch 200 --lr 1E-4 --sc_step 20 --batch_size 8 --num_cage_v 512 --in_type 1 --out_type 1 --last_activation 'relu' --data_toggle --use_data1 --log_dir ckpts_CBD --version 5 --start_epoch 100 --ckpt './ckpts_CBD/2025-10-23-17-32-49-NGBCv5' --continue_ckpt

# python train_CBD.py --max_epoch 1000 --lr 1E-4 --sc_step 20 --batch_size 8 --num_cage_v 512 --in_type 1 --out_type 1 --last_activation 'relu' --data_toggle --use_data3 --log_dir ckpts_CBD --version 5 --start_epoch 820 --ckpt './ckpts_CBD/2026-04-02-02-04-44-NGBCv5-dist' --continue_ckpt --align_latent --use_dist_loss