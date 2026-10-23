"""Evaluation on a paired set with the competition's exact metrics, on GPU.

The model output is clamped to [0, 1] and ROUNDED to uint8 (exactly what the submission does). Then:
  * PSNR : 10·log10(255² / MSE) per image, float64           == skimage.metrics.peak_signal_noise_ratio
  * SSIM : skimage-default SSIM (7×7 uniform, sample cov)      == skimage.metrics.structural_similarity
  * DISTS: official DISTS_pytorch on the uint8 images / 255
Per-image rows are returned so results can be broken down by domain / degradation family.

CLI:  python -m dncomp.evaluate --ckpt checkpoints/<run>/ema_best.pth --data data/val/proxy_test_v2 [--dists]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.datasets import PairedFolderDataset
from .losses.losses import ssim_skimage


def quantize(x: torch.Tensor) -> torch.Tensor:
    """float [0,1] -> uint8-valued float64 tensor in [0, 255] (round, never truncate)."""
    return (x.double().clamp(0, 1) * 255.0).round()


@torch.no_grad()
def evaluate(model, dataset, device="cuda", batch_size=16, dists_model=None, amp=True, identity=False) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    rows = []
    model_was_training = model.training if model is not None else False
    if model is not None:
        model.eval()
    for b in loader:
        lq, gt = b["lq"].to(device), b["gt"].to(device)
        if identity:
            out = lq
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(lq)
        p, g = quantize(out.float()), quantize(gt)
        mse = (p - g).pow(2).mean(dim=(1, 2, 3))
        psnr = 10 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-10))
        ssim = ssim_skimage(p, g, data_range=255.0)
        if dists_model is not None:
            d = dists_model(g.float() / 255.0, p.float() / 255.0, batch_average=False).flatten()
        for i in range(lq.shape[0]):
            r = {k: b[k][i] for k in ("name", "domain", "spec", "family")}
            r.update(psnr=float(psnr[i]), ssim=float(ssim[i]))
            if dists_model is not None:
                r["dists"] = float(d[i])
            rows.append(r)
    if model is not None and model_was_training:
        model.train()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    cols = [c for c in ("psnr", "ssim", "dists") if c in df]
    out = {"all": df[cols].mean().round(4).to_dict()}
    for key in ("domain", "family"):
        out[key] = df.groupby(key)[cols].mean().round(4).to_dict(orient="index")
    return out


def load_model_from_ckpt(path, device="cuda", use_ema=True):
    from .models.omnidn import build_model

    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["model_name"], **ck.get("model_kwargs", {}))
    sd = ck["ema"] if (use_ema and "ema" in ck) else ck["model"]
    model.load_state_dict(sd)
    return model.to(device).eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint (.pth); omit with --identity for the 'no restoration' baseline")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--data", default="data/val/proxy_test_v2")
    ap.add_argument("--dists", action="store_true")
    ap.add_argument("--raw", action="store_true", help="use raw weights instead of EMA")
    ap.add_argument("--out", default=None, help="csv path for per-image results")
    args = ap.parse_args()

    ds = PairedFolderDataset(args.data)
    model = None
    if not args.identity:
        model, _ = load_model_from_ckpt(args.ckpt, use_ema=not args.raw)
    dists = None
    if args.dists:
        from DISTS_pytorch import DISTS

        dists = DISTS().cuda().eval()
    df = evaluate(model, ds, dists_model=dists, identity=args.identity)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
    print(json.dumps(summarize(df), indent=1))


if __name__ == "__main__":
    main()
