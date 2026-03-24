#!/usr/bin/env bash
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate vqdm_lora

python ~/ProjetIA/multiclass/dcgan_onefolder.py \
  --mode sample \
  --ckpt ~/ProjetIA/multiclass/runs_dcgan_ai3/000/G_last.pth \
  --outdir ~/ProjetIA/multiclass/eval_samples/dcgan_ai3/000 \
  --n 300 \
  --batch_size 128 \
  --img_size 64

python ~/ProjetIA/multiclass/dcgan_onefolder.py \
  --mode sample \
  --ckpt ~/ProjetIA/multiclass/runs_dcgan_ai3/001/G_last.pth \
  --outdir ~/ProjetIA/multiclass/eval_samples/dcgan_ai3/001 \
  --n 300 \
  --batch_size 128 \
  --img_size 64

python ~/ProjetIA/multiclass/dcgan_onefolder.py \
  --mode sample \
  --ckpt ~/ProjetIA/multiclass/runs_dcgan_ai3/002/G_last.pth \
  --outdir ~/ProjetIA/multiclass/eval_samples/dcgan_ai3/002 \
  --n 300 \
  --batch_size 128 \
  --img_size 64