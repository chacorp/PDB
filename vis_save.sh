# python vis_CBD.py --version 5 --ckpt ./ckpts_CBD/2026-04-27-15-36-49-NGBCv5 --align_latent --continue_ckpt --start_epoch 990
# python vis_CBD.py --version 5 --ckpt ./ckpts_CBD/2026-04-12-04-07-19-NGBCv5-dist
# python vis_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-10-23-17-32-49-NGBCv5
python vis_CBD.py --version 5 --ckpt ./ckpts_CBD/2026-02-25-17-14-34-NGBCv5-relu --align_latent
python vis_CBD.py --version 5 --ckpt ./ckpts_CBD/2026-04-27-15-36-49-NGBCv5 --align_latent


# python vis_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-11-05-16-32-53-NGBCv5 --align_latent --continue_ckpt --start_epoch 100
# python vis_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-11-08-22-53-46-NGBCv5 --align_latent --continue_ckpt --start_epoch 400
# python vis_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-11-09-21-21-43-NGBCv5 --align_latent --continue_ckpt --start_epoch 50

# python vis_CBD.py --version 0 --ckpt 'NFR-pretrained' --NFR

# python vis_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-11-03-14-56-32-NGBCv5 --align_latent --save_gt
# python vis_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-10-22-16-26-26-NGBCv5

# python vis_CBD.py --version 1 --ckpt ./ckpts_CBD/2025-10-20-00-15-40-CBD
# python vis_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-10-22-22-30-30-NGBCv5 --align_latent
# python vis_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-10-09-02-04-49-NGBCv5 --align_latent
# python vis_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-10-27-01-04-59-NGBCv5 --align_latent
# python vis_CBD.py --version 5 --ckpt ./ckpts_CBD/2025-10-22-16-26-26-NGBCv5 --save_gt

# scp -r -P 30675 root@143.248.249.98:/source/sihun/NeuralFacialAnimation/ckpts_CBD/2025-10-22-22-30-30-NGBCv5 /source/sihun/NeuralFacialAnimation/ckpts_CBD/2025-10-22-22-30-30-NGBCv5

## NFS
# python vis_CBD.py --version 0 --ckpt ./ckpt_stage1/2024-07-08-06-27-12-all
# python vis_CBD.py --version 0 --ckpt ./ckpt_stage1/2024-08-18-23-32-29-all

## NFR
# python vis_CBD.py --version 0 --NFR