"""End-to-end check of the packaged submission exactly as an organiser would run it:
read corrupted PNG -> submission/inference.Restorer -> uint8 output -> skimage PSNR / SSIM + official DISTS vs GT.
Uses ONLY the submission folder (model.py, weights, inference.py), not our training package.

    python scripts/verify_submission.py [--sub submission] [--data data/val/proxy_test_v2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submission")
    ap.add_argument("--data", default="data/val/proxy_test_v2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / args.sub))
    from inference import Restorer  # the submitted code

    from DISTS_pytorch import DISTS

    r = Restorer(ROOT / args.sub / "weights" / "omnidn.pth")
    dists = DISTS().to(r.device).eval()
    root = ROOT / args.data
    meta = pd.read_csv(root / "meta.csv")
    rows = []
    for m in meta.itertuples():
        lq = np.asarray(Image.open(root / "lq" / f"{m.name}.png").convert("RGB"))
        gt = np.asarray(Image.open(root / "gt" / f"{m.name}.png").convert("RGB"))
        out = r(lq)
        assert out.shape == (256, 256, 3) and out.dtype == np.uint8
        t = lambda a: torch.from_numpy(a).permute(2, 0, 1).float().div(255).unsqueeze(0).to(r.device)
        with torch.no_grad():
            d = float(dists(t(gt), t(out)))
        rows.append(dict(name=m.name, domain=m.domain, family=m.family,
                         psnr=peak_signal_noise_ratio(gt, out, data_range=255),
                         ssim=structural_similarity(gt, out, channel_axis=2, data_range=255), dists=d))
    df = pd.DataFrame(rows)
    res = {"all": df[["psnr", "ssim", "dists"]].mean().round(4).to_dict(),
           "domain": df.groupby("domain")[["psnr", "ssim", "dists"]].mean().round(4).to_dict(orient="index")}
    print(json.dumps(res, indent=1))
    if args.out:
        df.to_csv(ROOT / args.out, index=False)


if __name__ == "__main__":
    main()
