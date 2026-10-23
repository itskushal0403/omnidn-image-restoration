"""Materialise interpolated candidate checkpoints and score them under protocol variants:
  * SSIM skimage default (7x7 uniform)  — what our loss / selection optimises
  * SSIM gaussian_weights=True (11x11, sigma 1.5, Wang et al.) — a plausible alternative call by the organisers
  * PSNR and DISTS as usual.
Everything is computed exactly like the organisers would (uint8 outputs, skimage on CPU, official DISTS).

    python scripts/eval_candidates_protocols.py --alphas 0 0.4 0.6 0.8 1.0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dncomp.data.datasets import read_rgb  # noqa: E402
from dncomp.models.omnidn import build_model  # noqa: E402

PAIRS = {"s": ("omnidn_s_ft_fid", "omnidn_s_ft_perc"), "m": ("omnidn_m_ft_fid", "omnidn_m_ft_perc")}


def mix(a_path, b_path, alpha):
    ca = torch.load(ROOT / a_path, map_location="cpu", weights_only=False)
    cb = torch.load(ROOT / b_path, map_location="cpu", weights_only=False)
    sd = {k: (1 - alpha) * ca["ema"][k].float() + alpha * cb["ema"][k].float() for k in ca["ema"]}
    return ca, sd


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--sets", nargs="+", default=["proxy_test_v2", "proxy_test_ood_v2"])
    args = ap.parse_args()
    from DISTS_pytorch import DISTS

    dists = DISTS().cuda().eval()
    out_dir = ROOT / "checkpoints" / "candidates"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for m, (fid, perc) in PAIRS.items():
        for a in args.alphas:
            ck, sd = mix(f"checkpoints/{fid}/ema_final.pth", f"checkpoints/{perc}/ema_final.pth", a)
            name = f"{m}_a{a:.1f}"
            torch.save({"ema": sd, "model_name": ck["model_name"], "model_kwargs": ck.get("model_kwargs", {}),
                        "note": f"(1-{a})*{fid} + {a}*{perc} (EMA weights)"}, out_dir / f"{name}.pth")
            model = build_model(ck["model_name"], **ck.get("model_kwargs", {})).cuda().eval()
            model.load_state_dict(sd)
            for s in args.sets:
                root = ROOT / "data" / "val" / s
                meta = pd.read_csv(root / "meta.csv")
                for r in meta.itertuples():
                    lq, gt = read_rgb(root / "lq" / f"{r.name}.png"), read_rgb(root / "gt" / f"{r.name}.png")
                    x = torch.from_numpy(lq).permute(2, 0, 1)[None].float().div(255).cuda()
                    y = model(x).clamp(0, 1)
                    out = (y * 255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
                    t = lambda arr: torch.from_numpy(arr).permute(2, 0, 1)[None].float().div(255).cuda()
                    mse = np.mean((gt.astype(np.float64) - out) ** 2)
                    rows.append(dict(cand=name, set=s, name=r.name, domain=r.domain,
                                     psnr=10 * np.log10(255 ** 2 / max(mse, 1e-10)),
                                     ssim=structural_similarity(gt, out, channel_axis=2, data_range=255),
                                     ssim_gauss=structural_similarity(gt, out, channel_axis=2, data_range=255,
                                                                      gaussian_weights=True, sigma=1.5,
                                                                      use_sample_covariance=False),
                                     dists=float(dists(t(gt), t(out)))))
            print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "runs" / "candidates_protocols.csv", index=False)
    print(df.groupby(["set", "cand"])[["psnr", "ssim", "ssim_gauss", "dists"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
