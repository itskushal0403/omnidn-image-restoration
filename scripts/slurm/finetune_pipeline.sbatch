#!/bin/bash
# Stage 2 for one model size, run sequentially inside one job (the mtech QoS allows only 3 submitted jobs):
#   2a fidelity fine-tune  ->  2b perceptual fine-tune (from 2a)  ->  weight-interpolation sweep on the proxy sets
# Usage: sbatch scripts/slurm/finetune_pipeline.sbatch <fid_config> <perc_config> <tag>
#SBATCH --job-name=dn-ft
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/m25csa016/denoising_competition/logs/%x-%j.out
set -euo pipefail
PROJECT=/csehome/m25csa016/denoising_competition
PY=/scratch/m25csa016/envs/denoise/bin/python
export TORCH_HOME=/scratch/m25csa016/denoising_competition/cache/torch
export TORCHINDUCTOR_CACHE_DIR=/scratch/m25csa016/denoising_competition/cache/inductor
export PYTHONUNBUFFERED=1
cd "$PROJECT"
FID=$1; PERC=$2; TAG=$3
fid_name=$(grep '^name:' "$FID" | awk '{print $2}')
perc_name=$(grep '^name:' "$PERC" | awk '{print $2}')
echo "host=$(hostname) fid=$fid_name perc=$perc_name"
[ -f checkpoints/$fid_name/ema_final.pth ]  || $PY -m dncomp.train --config "$FID"
[ -f checkpoints/$perc_name/ema_final.pth ] || $PY -m dncomp.train --config "$PERC"
$PY scripts/interpolate_and_eval.py --a checkpoints/$fid_name/ema_final.pth --b checkpoints/$perc_name/ema_final.pth \
    --alphas 0 0.2 0.4 0.6 0.8 1.0 --out runs/interp_${TAG}.csv
echo "PIPELINE DONE"
