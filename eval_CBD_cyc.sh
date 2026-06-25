## ============================================================
## Cyclic consistency evaluation — NFS / NFR / Ours
## Results are saved under eval_cyc/
## ============================================================

SRC=ict
TGT=mf_SEN
LOG=eval_cyc

## NFS ############################################################
# CKPT='./ckpt_stage1/2024-08-18-23-32-29-all'
# python eval_CBD_cyc.py --version 0 --ckpt $CKPT --log_dir $LOG --batch_size 1 --data_selection 4
# python eval_CBD_cyc.py --version 0 --ckpt $CKPT --log_dir $LOG --batch_size 1 --data_selection 6
# python eval_CBD_cyc.py --version 0 --ckpt $CKPT \
#     --src_data $SRC --tgt_data $TGT \
#     --log_dir $LOG --symmetric --laplacian \
#     --batch_size 1

## NFR ############################################################
CKPT='./ckpt_stage1/exp_019_ICT_MF-jacob_NFR'
python eval_CBD_cyc.py --version 0 --ckpt $CKPT --log_dir $LOG --batch_size 1 --data_selection 4 --use_NFR
python eval_CBD_cyc.py --version 0 --ckpt $CKPT --log_dir $LOG --batch_size 1 --data_selection 6 --use_NFR
# python eval_CBD_cyc.py --version 0 --ckpt $CKPT --NFR \
#     --src_data $SRC --tgt_data $TGT \
#     --log_dir $LOG --symmetric --laplacian \
#     --batch_size 1

# ## Ours ###########################################################
# CKPT='./ckpts_CBD/2026-02-25-17-14-34-NGBCv5-relu'
# python eval_CBD_cyc.py --version 5 --ckpt $CKPT --align_latent --log_dir $LOG --batch_size 1 --data_selection 4
# python eval_CBD_cyc.py --version 5 --ckpt $CKPT --align_latent --log_dir $LOG --batch_size 1 --data_selection 6

# ## Ours ###########################################################
# CKPT='./ckpts_CBD/2026-04-02-02-04-44-NGBCv5-dist'
# python eval_CBD_cyc.py --version 5 --ckpt $CKPT --align_latent --log_dir $LOG --batch_size 1 --data_selection 4
# python eval_CBD_cyc.py --version 5 --ckpt $CKPT --align_latent --log_dir $LOG --batch_size 1 --data_selection 6


## design ###########################################################
# CKPT='./ckpts_CBD/2025-10-23-17-32-49-NGBCv5'
# python eval_CBD_cyc.py --version 5 --ckpt $CKPT --log_dir $LOG --batch_size 1 --data_selection 0
# python eval_CBD_cyc.py --version 5 --ckpt $CKPT --log_dir $LOG --batch_size 1 --data_selection 1
# python eval_CBD_cyc.py --version 5 --ckpt $CKPT --log_dir $LOG --batch_size 1 --data_selection 2
# python eval_CBD_cyc.py --version 5 --ckpt $CKPT --log_dir $LOG --batch_size 1 --data_selection 3
# python eval_CBD_cyc.py --version 5 --ckpt $CKPT --log_dir $LOG --batch_size 1 --data_selection 4

# CKPT='./ckpts_CBD/2026-04-12-04-07-19-NGBCv5-dist'
# python eval_CBD_cyc.py --version 5 --ckpt $CKPT --log_dir $LOG --batch_size 1 --data_selection 0
# python eval_CBD_cyc.py --version 5 --ckpt $CKPT --log_dir $LOG --batch_size 1 --data_selection 1
# python eval_CBD_cyc.py --version 5 --ckpt $CKPT --log_dir $LOG --batch_size 1 --data_selection 2
# python eval_CBD_cyc.py --version 5 --ckpt $CKPT --log_dir $LOG --batch_size 1 --data_selection 3
# python eval_CBD_cyc.py --version 5 --ckpt $CKPT --log_dir $LOG --batch_size 1 --data_selection 4
