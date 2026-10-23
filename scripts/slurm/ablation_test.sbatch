#!/bin/bash
# Short (<1 h) ablation on the `test` partition: fine-tune from a snapshot, then evaluate on the proxy test (fp32 + DISTS).
# Usage: sbatch scripts/slurm/ablation_test.sbatch <config> <run_name> <init_ckpt> [override ...]
#SBATCH --job-name=dn-abl
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:59:00
#SBATCH --output=/scratch/m25csa016/denoising_competition/logs/%x-%j.out
PROJECT=/csehome/m25csa016/denoising_competition
PY=/scratch/m25csa016/envs/denoise/bin/python
export TORCH_HOME=/scratch/m25csa016/denoising_competition/cache/torch
export TORCHINDUCTOR_CACHE_DIR=/scratch/m25csa016/denoising_competition/cache/inductor
export PYTHONUNBUFFERED=1
cd "$PROJECT"
CFG=$1; NAME=$2; INIT=$3; shift 3
$PY -m dncomp.train --config "$CFG" --override name=$NAME init=$INIT train.time_budget_h=0.72 data.num_workers=10 \
    val.every=100000 "$@"
$PY - <<PYEOF
import json
from dncomp.data.datasets import PairedFolderDataset
from dncomp.evaluate import evaluate, load_model_from_ckpt, summarize
from DISTS_pytorch import DISTS
d = DISTS().cuda().eval()
m, _ = load_model_from_ckpt("checkpoints/$NAME/ema_final.pth")
df = evaluate(m, PairedFolderDataset("data/val/proxy_test_v2"), dists_model=d, amp=False)
df.to_csv("runs/$NAME/proxy_test_v2_eval.csv", index=False)
print("FINAL", "$NAME", json.dumps(summarize(df)["all"]))
PYEOF
