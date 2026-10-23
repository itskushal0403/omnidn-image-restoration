"""Stress test: extreme severities beyond the proxy test, on the proxy-test GT images (one spec per image, fixed seed).
Compares "identity" (input returned) with a packaged submission folder, using the exact metrics (skimage PSNR/SSIM).

    python scripts/stress_test.py --sub submission_versions/v1_early
"""
import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.util import random_noise

ROOT = Path(__file__).resolve().parents[1]


def f2u(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def jpeg(img, q):
    b = io.BytesIO()
    Image.fromarray(img).save(b, format="JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))


def motion(img, k, ang):
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1)
    ker = cv2.warpAffine(ker, M, (k, k))
    return cv2.filter2D(img, -1, ker / ker.sum())


SPECS = {
    "gauss_s75": lambda im, r: f2u(random_noise(im, "gaussian", var=(75 / 255) ** 2, rng=r)),
    "sp_a0.25": lambda im, r: f2u(random_noise(im, "s&p", amount=0.25, rng=r)),
    "speckle_v0.2": lambda im, r: f2u(random_noise(im, "speckle", var=0.2, rng=r)),
    "poisson_peak5": lambda im, r: f2u(np.random.default_rng(r).poisson(im / 255.0 * 5) / 5),
    "jpeg_q5": lambda im, r: jpeg(im, 5),
    "motion_k25": lambda im, r: motion(im, 25, 30),
    "gblur_s4": lambda im, r: cv2.GaussianBlur(im, (0, 0), 4),
    "gauss_s50+jpeg_q20": lambda im, r: jpeg(f2u(random_noise(im, "gaussian", var=(50 / 255) ** 2, rng=r)), 20),
    "motion_k15+gauss_s25": lambda im, r: f2u(random_noise(motion(im, 15, 70), "gaussian", var=(25 / 255) ** 2, rng=r)),
    "ca_roll4+sp_a0.05": lambda im, r: f2u(random_noise(np.stack([np.roll(im[..., 0], 4, 1), im[..., 1],
                                                                   np.roll(im[..., 2], -4, 1)], 2), "s&p", rng=r)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submission")
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / args.sub))
    from inference import Restorer

    r = Restorer(ROOT / args.sub / "weights" / "omnidn.pth")
    meta = pd.read_csv(ROOT / "data/val/proxy_test_v2/meta.csv").drop_duplicates("name").iloc[::10].head(args.n)
    gts = {p: np.asarray(Image.open(ROOT / "data/val/proxy_test_v2/gt" / f"{p}.png").convert("RGB")) for p in meta.name}
    rows = []
    for i, (name, gt) in enumerate(gts.items()):
        for spec, fn in SPECS.items():
            lq = fn(gt.copy(), 1000 + i)
            out = r(lq)
            rows.append(dict(spec=spec, domain=meta.set_index("name").domain[name],
                             in_psnr=psnr(gt, lq, data_range=255), out_psnr=psnr(gt, out, data_range=255),
                             in_ssim=ssim(gt, lq, channel_axis=2, data_range=255),
                             out_ssim=ssim(gt, out, channel_axis=2, data_range=255)))
    df = pd.DataFrame(rows)
    t = df.groupby("spec")[["in_psnr", "out_psnr", "in_ssim", "out_ssim"]].mean().round(3)
    t["gain_dB"] = (t.out_psnr - t.in_psnr).round(2)
    print(t.sort_values("out_psnr").to_string())
    out = ROOT / "runs" / f"stress_test_{Path(args.sub).name}.csv"
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
