#!/bin/bash
# Train one model on one GPU.  Usage:  sbatch scripts/slurm/train.sbatch configs/omnidn_s.yaml [key=value ...]
# Auto-resumes from checkpoints/<name>/last.pth, so the same command can simply be re-submitted.
#SBATCH --job-name=dn-train
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/scratch/m25csa016/denoising_competition/logs/%x-%j.out

set -euo pipefail
PROJECT=/csehome/m25csa016/denoising_competition
PY=/scratch/m25csa016/envs/denoise/bin/python
export TORCH_HOME=/scratch/m25csa016/denoising_competition/cache/torch
export TORCHINDUCTOR_CACHE_DIR=/scratch/m25csa016/denoising_competition/cache/inductor
export PYTHONUNBUFFERED=1
cd "$PROJECT"
CONFIG=$1; shift
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) config=$CONFIG overrides=$*"
# run in background so the batch shell can forward SIGUSR1 (sent 3 min before the time limit) to python
$PY -m dncomp.train --config "$CONFIG" --override "$@" &
PID=$!
trap "kill -USR1 $PID" USR1
wait $PID || wait $PID
