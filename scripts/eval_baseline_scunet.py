"""Reference "competitor": the public pretrained blind denoiser SCUNet (Zhang et al., MIR 2023; KAIR release
`scunet_color_real_psnr.pth`), evaluated on our proxy test sets with the exact metrics + its FLOPs at 256².

    srun -p test --gres=gpu:1 python scripts/eval_baseline_scunet.py
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dncomp.data.datasets import PairedFolderDataset  # noqa: E402
from dncomp.evaluate import evaluate, summarize  # noqa: E402
from dncomp.metrics.competition_metrics import count_flops  # noqa: E402
from dncomp.models.third_party.network_scunet import SCUNet  # noqa: E402

model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
model.load_state_dict(torch.load(ROOT / "checkpoints/third_party/scunet_color_real_psnr.pth", map_location="cpu"))
print("FLOPs:", count_flops(model))
model = model.cuda().eval()
from DISTS_pytorch import DISTS  # noqa: E402

d = DISTS().cuda().eval()
for s in ["proxy_test_v2", "proxy_test_ood_v2"]:
    df = evaluate(model, PairedFolderDataset(ROOT / f"data/val/{s}"), dists_model=d, amp=False, batch_size=8)
    df.to_csv(ROOT / f"runs/baselines/scunet_{s}.csv", index=False)
    sm = summarize(df)
    print(s, json.dumps(sm["all"]), json.dumps(sm["domain"]))
