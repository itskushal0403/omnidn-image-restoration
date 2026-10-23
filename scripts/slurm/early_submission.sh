#!/bin/bash
# SAFETY SUBMISSION (in case the deadline is 11:59 AM, before stage 1 ends at ~13:40).
# Runs on the login node with nohup; nothing that is already running is touched.
#   1. at START_AT, snapshot the running OmniDN-S stage-1 checkpoint (EMA weights)
#   2. 45-min metric-aligned fine-tune (PSNR-loss + exact SSIM, stage-2 degradation mix) on the 1-h `test` partition
#   3. package it into submission/ (model.py + weights + FLOP check) and keep a copy in submission_versions/v1_early
#   4. evaluate on both proxy sets (exact metrics incl. DISTS)
# Usage: nohup scripts/slurm/early_submission.sh 07:00 > logs/early_submission.log 2>&1 &
START_AT=${1:-07:00}
PROJECT=/csehome/m25csa016/denoising_competition
PY=/scratch/m25csa016/envs/denoise/bin/python
export TORCH_HOME=/scratch/m25csa016/denoising_competition/cache/torch
export TORCHINDUCTOR_CACHE_DIR=/scratch/m25csa016/denoising_competition/cache/inductor
export PYTHONUNBUFFERED=1
cd "$PROJECT"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

target=$(date -d "$START_AT" +%s); [ "$target" -lt "$(date +%s)" ] && target=$(date -d "tomorrow $START_AT" +%s)
log "waiting until $(date -d @$target)"
while [ "$(date +%s)" -lt "$target" ]; do sleep 60; done

mkdir -p checkpoints/early
cp checkpoints/omnidn_s_stage1/last.pth checkpoints/early/s_stage1_snapshot.pth
log "snapshot taken: $($PY -c "import torch;print(torch.load('checkpoints/early/s_stage1_snapshot.pth',map_location='cpu',weights_only=False)['state'])")"

FT=checkpoints/omnidn_s_early_ft/ema_final.pth
for attempt in 1 2 3; do
  [ -f $FT ] && break
  log "fine-tune attempt $attempt"
  srun -p test --gres=gpu:1 -c 12 --mem=48G -t 00:58:00 $PY -m dncomp.train --config configs/omnidn_s_ft_fid.yaml \
    --override name=omnidn_s_early_ft init=checkpoints/early/s_stage1_snapshot.pth train.time_budget_h=0.75 \
    data.num_workers=10 val.every=1500 val.dists_every=3000 > logs/early_ft_attempt$attempt.log 2>&1
  log "attempt $attempt exit=$?"
done
SRC=$FT; [ -f $FT ] || { SRC=checkpoints/early/s_stage1_snapshot.pth; log "fine-tune failed -> packaging the raw snapshot"; }

$PY scripts/package_submission.py --ckpt $SRC > logs/early_package.log 2>&1
log "packaged $SRC (exit=$?):"; cat logs/early_package.log
rm -rf submission_versions/v1_early && mkdir -p submission_versions && cp -r submission submission_versions/v1_early
echo "source checkpoint: $SRC (packaged $(date))" > submission_versions/v1_early/VERSION.txt

# end-to-end check of the packaged folder exactly as an organiser would run it (skimage + official DISTS)
mkdir -p runs/early_submission
for s in proxy_test_v2 proxy_test_ood_v2; do
  srun -p test --gres=gpu:1 -c 6 --mem=24G -t 00:20:00 $PY scripts/verify_submission.py --sub submission_versions/v1_early \
    --data data/val/$s --out runs/early_submission/$s.csv > logs/early_verify_$s.log 2>&1
  log "verify $s:"; grep -v -i warn logs/early_verify_$s.log | head -25
done
log "EARLY SUBMISSION READY"
