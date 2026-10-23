"""Study: should the model snap near-gray outputs to exact gray?

Medical images (X-ray, MRI, CT, ultrasound) are gray, but test corruptions (e.g. skimage noise, chromatic aberration)
act per channel, so a restored image can keep a faint colour tint, which costs PSNR / SSIM. A ~0-FLOP fix: if the
output's mean chroma |out_c - mean_c(out)| is below a threshold t, replace it by its channel mean.

This script measures, for several t, the PSNR / SSIM change on every proxy-test image, split by whether the GT is
truly gray, so we can pick a t that helps gray images and never touches colour ones.

    srun -p test --gres=gpu:1 python scripts/study_gray_consistency.py --ckpt checkpoints/<run>/ema_best.pth
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dncomp.data.datasets import PairedFolderDataset  # noqa: E402
from dncomp.evaluate import load_model_from_ckpt, quantize  # noqa: E402
from dncomp.losses.losses import ssim_skimage  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/val/proxy_test_v2")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.002, 0.004, 0.006, 0.008, 0.012, 0.02])
    args = ap.parse_args()
    model, _ = load_model_from_ckpt(ROOT / args.ckpt)
    loader = DataLoader(PairedFolderDataset(ROOT / args.data), batch_size=16, num_workers=4)
    rows = []
    for b in loader:
        lq, gt = b["lq"].cuda(), b["gt"].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(lq).float().clamp(0, 1)
        g = quantize(gt)
        gt_gray = (g - g.mean(1, keepdim=True)).abs().amax(dim=(1, 2, 3)) < 1.0
        chroma = (out - out.mean(1, keepdim=True)).abs().mean(dim=(1, 2, 3))
        variants = {"none": out, "gray": out.mean(1, keepdim=True).expand_as(out)}
        res = {}
        for k, v in variants.items():
            p = quantize(v)
            mse = (p - g).pow(2).mean(dim=(1, 2, 3))
            res[k] = (10 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-10)), ssim_skimage(p, g, 255.0))
        for i in range(lq.shape[0]):
            rows.append(dict(name=b["name"][i], domain=b["domain"][i], gt_gray=bool(gt_gray[i]), chroma=float(chroma[i]),
                             psnr=float(res["none"][0][i]), ssim=float(res["none"][1][i]),
                             psnr_gray=float(res["gray"][0][i]), ssim_gray=float(res["gray"][1][i])))
    df = pd.DataFrame(rows)
    print(df.groupby(["domain", "gt_gray"]).agg(n=("psnr", "size"), chroma_med=("chroma", "median"),
                                                chroma_max=("chroma", "max")).to_string())
    base = df[["psnr", "ssim"]].mean()
    print(f"\nbaseline: PSNR {base.psnr:.4f}  SSIM {base.ssim:.5f}")
    for t in args.thresholds:
        snap = df.chroma < t
        psnr = df.psnr.where(~snap, df.psnr_gray).mean()
        ssim = df.ssim.where(~snap, df.ssim_gray).mean()
        wrong = int((snap & ~df.gt_gray).sum())
        print(f"t={t:.3f}: snapped {int(snap.sum()):3d} (non-gray GT snapped: {wrong:2d})  "
              f"PSNR {psnr:.4f} ({psnr - base.psnr:+.4f})  SSIM {ssim:.5f} ({ssim - base.ssim:+.5f})")
    df.to_csv(ROOT / "runs" / "baselines" / "gray_consistency_study.csv", index=False)


if __name__ == "__main__":
    main()
