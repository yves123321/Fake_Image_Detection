#!/usr/bin/env bash
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate vqdm_lora

python ~/ProjetIA/multiclass/dcgan_onefolder.py \
  --mode train \
  --train_dir ~/data/vqdm_ai3/train/000 \
  --save_dir ~/ProjetIA/multiclass/runs_dcgan_ai3/000 \
  --epochs 40 \
  --batch_size 16 \
  --img_size 64 \
  --num_workers 4

python ~/ProjetIA/multiclass/dcgan_onefolder.py \
  --mode train \
  --train_dir ~/data/vqdm_ai3/train/001 \
  --save_dir ~/ProjetIA/multiclass/runs_dcgan_ai3/001 \
  --epochs 40 \
  --batch_size 16 \
  --img_size 64 \
  --num_workers 4

python ~/ProjetIA/multiclass/dcgan_onefolder.py \
  --mode train \
  --train_dir ~/data/vqdm_ai3/train/002 \
  --save_dir ~/ProjetIA/multiclass/runs_dcgan_ai3/002 \
  --epochs 40 \
  --batch_size 16 \
  --img_size 64 \
  --num_workers 4