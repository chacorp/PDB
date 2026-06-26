# python train_CBD.py \
# --max_epoch 200 \
# --lr 2E-4 \
# --sc_step 20 \
# --version 8 \
# --batch_size 32 \
# --num_cage_v 512 \
# --in_type 1 \
# --out_type 1 \
# --use_data3 \
# --last_activation 'relu'

# python train_CBD.py \
# --max_epoch 400 \
# --lr 2E-4 \
# --sc_step 20 \
# --version 5 \
# --batch_size 16 \
# --num_cage_v 512 \
# --in_type 1 \
# --out_type 1 \
# --last_activation 'relu' --use_data3 --start_epoch 150 --ckpt './ckpts_CBD/2025-10-09-02-07-57-NGBCv5' --continue_ckpt --data_toggle

# python train_CBD.py \
# --max_epoch 400 \
# --lr 2E-4 \
# --sc_step 20 \
# --version 5 \
# --batch_size 16 \
# --num_cage_v 512 \
# --in_type 1 \
# --out_type 1 \
# --last_activation 'relu' --use_data2 --start_epoch 100 --ckpt './ckpts_CBD/2025-10-09-02-04-49-NGBCv5' --continue_ckpt --data_toggle --use_laplacian

# python train_CBD.py --max_epoch 200 --version 1 --lr 2E-6 --batch_size 16 --in_type 1 --out_type 1 --data_toggle --use_data2

# python train_CBD.py --max_epoch 1000 --lr 1E-4 --sc_step 1000 --batch_size 8 --num_cage_v 512 --in_type 1 --out_type 1 --last_activation 'relu' --data_toggle --use_data2 --log_dir ckpts_CBD --version 5 --align_latent


python train_CBD.py --max_epoch 1000 --lr 1E-4 --sc_step 1000 --batch_size 8 --num_cage_v 512 --in_type 1 --out_type 1 --last_activation 'relu' --data_toggle --use_data2 --log_dir ckpts_CBD --version 5 --align_latent --start_epoch 360 --ckpt './ckpts_CBD/2026-02-18-09-43-05-NGBCv5' --continue_ckpt
