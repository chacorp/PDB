#!/usr/bin/env bash
# ============================================================================
# eval_design_study.sh 와 같은 프로토콜로 ckpts_CBD9 의 4개 모델(num_cage_v 512/256/128/64)을
# best 체크포인트(*_best.pth, eval 스크립트 기본값)로 평가한다.
#
# 주의: num_cage_v 는 train_opts.yml 에서 읽힌다. align_latent 는 argparse 기본값(False)이
# yml 값(True)을 덮어쓰므로 반드시 명시한다.
# ============================================================================
set -euo pipefail

VN=5
DATASETS=(1 2 3 4)  # 1=biwi, 2=mf_SEN, 3=coma, 4=mf_ROM
CKPTS=(
  ./ckpts_CBD9/2026-10-27-03-49-38-NGBCv5  # 512
  ./ckpts_CBD9/2026-10-27-09-48-28-NGBCv5  # 256
  ./ckpts_CBD9/2026-10-27-15-38-55-NGBCv5  # 128
  ./ckpts_CBD9/2026-10-30-00-43-11-NGBCv5  # 64
)

for CKPT in "${CKPTS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    echo "--- eval_CBD.py (self-retargeting) : $CKPT, data_selection=$DS ---"
    python eval_CBD.py --version $VN --ckpt "$CKPT" --data_selection $DS \
      --realtest --use_t_mask --save_vert --batch_size 1 \
      --last_activation relu --align_latent

    echo "--- eval_CBD_cyc.py (cyclic self-consistency) : $CKPT, data_selection=$DS ---"
    python eval_CBD_cyc.py --version $VN --ckpt "$CKPT" --data_selection $DS \
      --batch_size 1 \
      --last_activation relu --align_latent
  done
done
