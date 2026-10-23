"""Side-by-side examples (corrupted input | model output | ground truth) from the proxy test, per domain.

    python scripts/make_visual_results.py --ckpt checkpoints/<run>/<file>.pth [--out docs/figures/visual_results.png]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dncomp.data.datasets import read_rgb  # noqa: E402
from dncomp.evaluate import load_model_from_ckpt  # noqa: E402
from dncomp.metrics.competition_metrics import psnr_uint8, ssim_uint8  # noqa: E402

SPECS = ["gauss_sk_default", "sp_sk_default", "motion_h9+gauss_sk", "gauss_s25+jpeg_q30", "ca_roll2+gauss_s15",
         "speckle_v0.05", "poisson_sk_default", "gblur_k5+gauss_s15", "motion_d15"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/val/proxy_test_v2")
    ap.add_argument("--out", default="docs/figures/visual_results.png")
    ap.add_argument("--n", type=int, default=9)
    args = ap.parse_args()
    root = ROOT / args.data
    meta = pd.read_csv(root / "meta.csv")
    rng = np.random.default_rng(0)
    picks = []
    for i, spec in enumerate(SPECS[: args.n]):
        dom = ("natural", "medical", "lowlight")[i % 3]
        cand = meta[(meta.spec == spec) & (meta.domain == dom)]
        cand = cand if len(cand) else meta[meta.spec == spec]
        if len(cand):
            picks.append(cand.iloc[int(rng.integers(len(cand)))])
    model, _ = load_model_from_ckpt(ROOT / args.ckpt)
    fig, axes = plt.subplots(len(picks), 3, figsize=(10, 3.4 * len(picks)))
    for r, row in enumerate(picks):
        lq, gt = read_rgb(root / "lq" / f"{row['name']}.png"), read_rgb(root / "gt" / f"{row['name']}.png")
        x = torch.from_numpy(lq).permute(2, 0, 1)[None].float().div(255).cuda()
        out = (model(x).clamp(0, 1) * 255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
        for c, (img, title) in enumerate(((lq, "input"), (out, "OmniDN output"), (gt, "ground truth"))):
            axes[r, c].imshow(img)
            sub = "" if c == 2 else f"  {psnr_uint8(gt, img):.2f} dB / {ssim_uint8(gt, img):.3f}"
            axes[r, c].set_title(f"{title}{sub}" + (f"\n[{row['domain']}] {row['spec']}" if c == 0 else ""), fontsize=8)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / args.out, dpi=85)
    print("saved", args.out)


if __name__ == "__main__":
    main()
