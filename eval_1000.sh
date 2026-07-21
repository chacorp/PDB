## ============================================================
## Eval sweep for the two 1000-epoch checkpoints:
##   - ckpts_CBD/2026-02-25-17-14-34-NGBCv5-relu
##   - ckpts_CBD/2026-04-27-15-36-49-NGBCv5
## Both are version=5, align_latent=True, last_activation=relu.
## Order: eval_CBD_lp.py -> eval_CBD.py -> eval_CBD_cyc.py
## Within eval_CBD.py / eval_CBD_lp.py: data_selection 2,4 first, 5 (ict) last
## Within eval_CBD_cyc.py: data_selection 2,4 first, 6 (ICT->MF cross) last
## Results saved under eval_1000/
## ============================================================

set -e

LOG=eval_1000

MODELS=(
    "./ckpts_CBD/2026-02-25-17-14-34-NGBCv5-relu"
    "./ckpts_CBD/2026-04-27-15-36-49-NGBCv5"
)

## ================= [1/3] eval_CBD_lp.py =================
for CKPT in "${MODELS[@]}"; do
    for DS in 2 4; do
        python eval_CBD_lp.py --version 5 --ckpt $CKPT --align_latent --last_activation relu \
            --data_selection $DS --batch_size 1 --log_dir $LOG
    done
done
for CKPT in "${MODELS[@]}"; do
    python eval_CBD_lp.py --version 5 --ckpt $CKPT --align_latent --last_activation relu \
        --data_selection 5 --batch_size 1 --log_dir $LOG
done

## ================= [2/3] eval_CBD.py =================
for CKPT in "${MODELS[@]}"; do
    for DS in 2 4; do
        python eval_CBD.py --version 5 --ckpt $CKPT --align_latent --last_activation relu \
            --data_selection $DS --realtest --batch_size 1 --save_vert --laplacian --use_t_mask --log_dir $LOG
    done
done
for CKPT in "${MODELS[@]}"; do
    python eval_CBD.py --version 5 --ckpt $CKPT --align_latent --last_activation relu \
        --data_selection 5 --realtest --batch_size 1 --save_vert --laplacian --use_t_mask --log_dir $LOG
done

## ================= [3/3] eval_CBD_cyc.py =================
for CKPT in "${MODELS[@]}"; do
    for DS in 2 4; do
        python eval_CBD_cyc.py --version 5 --ckpt $CKPT --align_latent \
            --data_selection $DS --batch_size 1 --log_dir $LOG
    done
done
for CKPT in "${MODELS[@]}"; do
    python eval_CBD_cyc.py --version 5 --ckpt $CKPT --align_latent \
        --data_selection 6 --batch_size 1 --log_dir $LOG
done
