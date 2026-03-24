#!/usr/bin/env bash
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate vqdm_lora

python ~/ProjetIA/multiclass/eval_multiclass_consistency.py \
  --img_root ~/ProjetIA/multiclass/eval_samples/dcgan_ai3 \
  --classifier_ckpt ~/ProjetIA/multiclass/cls_ai3/best_resnet18.pth \
  --img_size 128 \
  --batch_size 64 \
  --num_workers 4 \
  --out_csv ~/ProjetIA/multiclass/dcgan_ai3_consistency.csv