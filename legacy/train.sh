# python train.py \
#     --log_dir "ckpts" \
#     --max_epoch 500 \
#     --stage1 --tb \
#     --dec_type 'disp' \
#     --design 'new2' \
#     --seg_dim 20 \
#     --batch_size 1 \
#     --window_size 8

# python train.py \
#     --log_dir "ckpts" \
#     --max_epoch 2000 \
#     --stage1 --tb \
#     --dec_type 'disp' \
#     --design 'new2' \
#     --seg_dim 20 \
#     --batch_size 8 \
#     --ckpt "ckpts/2025-04-23-13-35-59-all"
# #     --ckpt "ckpt_stage1/2024-08-18-23-32-29-all"

python trainD.py \
    --config config/train.yml \
    --log_dir "ckpts_new" \
    --max_epoch 200 \
    --stage1 --tb \
    --dec_type 'disp' \
    --design 'new2dn' \
    --seg_dim 20 \
    --batch_size 4 \
    --ckpt "ckpt_stage1/2024-06-09-10-57-34-all" \
    --window_size 1


# python trainD.py \
#     --config config/train.yml \
#     --log_dir "ckpts_new2" \
#     --max_epoch 400 \
#     --stage1 --tb \
#     --dec_type 'disp' \
#     --design 'new2dn_mk2' \
#     --seg_dim 20 \
#     --batch_size 8 \
#     --ckpt "ckpt_stage1/2024-06-09-10-57-34-all" \
#     --window_size 1

    

# accelerate launch -m \
#   --config_file default_config.yaml \
#   --machine_rank 0 \
#   --main_process_ip 0.0.0.0 \
#   --main_process_port 20055 \
#   --num_machines 1 \
#   --num_processes 1 \
#   python trainD.py \
#     --config config/train.yml \
#     --log_dir "ckpts_new" \
#     --max_epoch 200 \
#     --stage1 --tb \
#     --dec_type 'disp' \
#     --design 'new2dn' \
#     --seg_dim 20 \
#     --batch_size 8 \
#     --ckpt "ckpt_stage1/2024-06-09-10-57-34-all" \
#     --window_size 1
  

# python train.py \
#     --config config/train2.yml \
#     --log_dir "ckpts_new" \
#     --max_epoch 2000 \
#     --stage1 --tb \
#     --dec_type 'disp' \
#     --design 'new2' \
#     --batch_size 8

# python train2.py \
#     --config config/train2.yml \
#     --log_dir "ckpts_new" \
#     --max_epoch 2000 \
#     --stage1 --tb \
#     --dec_type 'disp' \
#     --design 'new4' \
#     --batch_size 8

# python train2.py \
#     --config config/train2.yml \
#     --log_dir "ckpts_new" \
#     --max_epoch 2000 \
#     --stage1 --tb \
#     --dec_type 'disp' \
#     --design 'new3' \
#     --batch_size 2

# python train2.py \
#     --config config/train2.yml \
#     --log_dir "ckpts_new" \
#     --max_epoch 2000 \
#     --stage1 --tb \
#     --dec_type 'disp' \
#     --design 'new2' \
#     --use_canon \
#     --batch_size 4