"""Weight interpolation between a fidelity model and a perceptual model, scored with the exact competition metrics.

θ(α) = (1 − α)·θ_A + α·θ_B   (ESRGAN-style network interpolation: free in FLOPs; B must be fine-tuned from A)

    python scripts/interpolate_and_eval.py --a checkpoints/<A>/ema_final.pth --b checkpoints/<B>/ema_final.pth \
        --alphas 0 0.25 0.5 0.75 1 --out runs/interp_<name>.csv [--save-alpha 0.5 --save-path checkpoints/x.pth]

Prints PSNR / SSIM / DISTS per α on the proxy test (and on the OOD set) so the operating point can be chosen.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dncomp.data.datasets import PairedFolderDataset  # noqa: E402
from dncomp.evaluate import evaluate  # noqa: E402
from dncomp.models.omnidn import build_model  # noqa: E402


def load(path):
    ck = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    return ck, (ck["ema"] if "ema" in ck else ck["model"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--sets", nargs="+", default=["data/val/proxy_test_v2", "data/val/proxy_test_ood_v2"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-alpha", type=float, default=None)
    ap.add_argument("--save-path", default=None)
    args = ap.parse_args()

    ck_a, sa = load(args.a)
    _, sb = load(args.b)
    model = build_model(ck_a["model_name"], **ck_a.get("model_kwargs", {})).cuda()
    from DISTS_pytorch import DISTS

    dists = DISTS().cuda().eval()
    sets = {Path(s).name: PairedFolderDataset(ROOT / s) for s in args.sets}
    rows = []
    for a in args.alphas:
        mix = {k: (1 - a) * sa[k].float() + a * sb[k].float() for k in sa}
        model.load_state_dict(mix)
        for name, ds in sets.items():
            df = evaluate(model, ds, dists_model=dists, amp=False)  # fp32 == exactly what the submission computes
            r = {"alpha": a, "set": name, **df[["psnr", "ssim", "dists"]].mean().round(4).to_dict()}
            for dom, g in df.groupby("domain"):
                r[f"psnr_{dom}"] = round(g.psnr.mean(), 3)
            rows.append(r)
            print(r, flush=True)
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    if args.out:
        res.to_csv(ROOT / args.out, index=False)
    if args.save_alpha is not None:
        a = args.save_alpha
        mix = {k: (1 - a) * sa[k].float() + a * sb[k].float() for k in sa}
        torch.save({"ema": mix, "model_name": ck_a["model_name"], "model_kwargs": ck_a.get("model_kwargs", {}),
                    "note": f"interp alpha={a} between {args.a} and {args.b}"}, ROOT / args.save_path)
        print("saved", args.save_path)


if __name__ == "__main__":
    main()
