## ============================================================
## Cyclic consistency evaluation — NFS / NFR / Ours
## Results are saved under eval_cyc/
## ============================================================

SRC=ict
TGT=mf_SEN
LOG=eval_cyc

## NFS ############################################################
CKPT='./ckpt_stage1/2024-08-18-23-32-29-all'
python eval_CBD_cyc.py --version 0 --ckpt $CKPT \
    --src_data $SRC --tgt_data $TGT \
    --log_dir $LOG --symmetric --laplacian \
    --batch_size 1

## NFR ############################################################
CKPT='./ckpt_stage1/exp_019_ICT_MF-jacob_NFR'
python eval_CBD_cyc.py --version 0 --ckpt $CKPT --NFR \
    --src_data $SRC --tgt_data $TGT \
    --log_dir $LOG --symmetric --laplacian \
    --batch_size 1

## Ours ###########################################################
CKPT='./ckpts_CBD/2026-02-25-17-14-34-NGBCv5-relu'
python eval_CBD_cyc.py --version 5 --ckpt $CKPT --align_latent \
    --src_data $SRC --tgt_data $TGT \
    --log_dir $LOG --symmetric --laplacian \
    --batch_size 1
