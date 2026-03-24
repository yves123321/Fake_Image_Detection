#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=~/data/vqdm_subset_raw
SAVE_ROOT=~/ProjetIA/report_exp/runs
RESULTS_CSV=~/ProjetIA/report_exp/results.csv
DETECTOR=~/ProjetIA/detector_ckpt_gpu/best_resnet18.pth

mkdir -p ~/ProjetIA/report_exp
mkdir -p "${SAVE_ROOT}"

COMMON_ARGS="\
  --data_root ${DATA_ROOT} \
  --save_root ${SAVE_ROOT} \
  --results_csv ${RESULTS_CSV} \
  --epochs 20 \
  --batch_size 64 \
  --num_workers 8 \
  --sample_n 4000 \
  --sample_batch_size 256 \
  --detector_ckpt ${DETECTOR} \
  --seed 42 \
"

# =========================
# Resolution sweep
# =========================
python ~/ProjetIA/report_exp/train_and_eval_supervised_generators.py \
  ${COMMON_ARGS} \
  --run_name acgan_res32_f100 \
  --model acgan \
  --img_size 32 \
  --data_frac 1.0

python ~/ProjetIA/report_exp/train_and_eval_supervised_generators.py \
  ${COMMON_ARGS} \
  --run_name acgan_res64_f100 \
  --model acgan \
  --img_size 64 \
  --data_frac 1.0

python ~/ProjetIA/report_exp/train_and_eval_supervised_generators.py \
  ${COMMON_ARGS} \
  --run_name cvae_res32_f100 \
  --model cvae \
  --img_size 32 \
  --latent_dim 64 \
  --data_frac 1.0

python ~/ProjetIA/report_exp/train_and_eval_supervised_generators.py \
  ${COMMON_ARGS} \
  --run_name cvae_res64_f100 \
  --model cvae \
  --img_size 64 \
  --latent_dim 64 \
  --data_frac 1.0

python ~/ProjetIA/report_exp/run_dcgan_pair_experiment.py \
  --data_root ~/data/vqdm_subset_raw \
  --save_root ~/ProjetIA/report_exp/runs \
  --results_csv ~/ProjetIA/report_exp/results.csv \
  --run_name dcgan_res32_f100 \
  --img_size 32 \
  --data_frac 1.0 \
  --epochs 20 \
  --batch_size 64 \
  --num_workers 8 \
  --detector_ckpt ~/ProjetIA/detector_ckpt_gpu/best_resnet18.pth

python ~/ProjetIA/report_exp/run_dcgan_pair_experiment.py \
  --data_root ~/data/vqdm_subset_raw \
  --save_root ~/ProjetIA/report_exp/runs \
  --results_csv ~/ProjetIA/report_exp/results.csv \
  --run_name dcgan_res64_f100 \
  --img_size 64 \
  --data_frac 1.0 \
  --epochs 20 \
  --batch_size 64 \
  --num_workers 8 \
  --detector_ckpt ~/ProjetIA/detector_ckpt_gpu/best_resnet18.pth

# =========================
# Data scale sweep (img_size=64)
# =========================
python ~/ProjetIA/report_exp/train_and_eval_supervised_generators.py \
  ${COMMON_ARGS} \
  --run_name acgan_res64_f025 \
  --model acgan \
  --img_size 64 \
  --data_frac 0.25

python ~/ProjetIA/report_exp/train_and_eval_supervised_generators.py \
  ${COMMON_ARGS} \
  --run_name acgan_res64_f050 \
  --model acgan \
  --img_size 64 \
  --data_frac 0.50

python ~/ProjetIA/report_exp/train_and_eval_supervised_generators.py \
  ${COMMON_ARGS} \
  --run_name cvae_res64_f025 \
  --model cvae \
  --img_size 64 \
  --latent_dim 64 \
  --data_frac 0.25

python ~/ProjetIA/report_exp/train_and_eval_supervised_generators.py \
  ${COMMON_ARGS} \
  --run_name cvae_res64_f050 \
  --model cvae \
  --img_size 64 \
  --latent_dim 64 \
  --data_frac 0.50

python ~/ProjetIA/report_exp/results_to_tables.py \
  --csv ${RESULTS_CSV} \
  --out_md ~/ProjetIA/report_exp/summary.md \
  --out_tex ~/ProjetIA/report_exp/summary.tex

python ~/ProjetIA/report_exp/run_dcgan_pair_experiment.py \
  --data_root ~/data/vqdm_subset_raw \
  --save_root ~/ProjetIA/report_exp/runs \
  --results_csv ~/ProjetIA/report_exp/results.csv \
  --run_name dcgan_res64_f025 \
  --img_size 64 \
  --data_frac 0.25 \
  --epochs 20 \
  --batch_size 64 \
  --num_workers 8 \
  --detector_ckpt ~/ProjetIA/detector_ckpt_gpu/best_resnet18.pth

python ~/ProjetIA/report_exp/run_dcgan_pair_experiment.py \
  --data_root ~/data/vqdm_subset_raw \
  --save_root ~/ProjetIA/report_exp/runs \
  --results_csv ~/ProjetIA/report_exp/results.csv \
  --run_name dcgan_res64_f050 \
  --img_size 64 \
  --data_frac 0.50 \
  --epochs 20 \
  --batch_size 64 \
  --num_workers 8 \
  --detector_ckpt ~/ProjetIA/detector_ckpt_gpu/best_resnet18.pth