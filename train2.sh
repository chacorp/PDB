# ## canon: ict neutral
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

## canon: shpere
python trainD.py \
    --config config/train.yml \
    --log_dir "ckpts_new2" \
    --max_epoch 400 \
    --stage1 --tb \
    --dec_type 'disp' \
    --design 'new2dn_mk2' \
    --seg_dim 20 \
    --batch_size 8 \
    --norm_canon \
    --ckpt "ckpt_stage1/2024-06-09-10-57-34-all" \
    --window_size 1