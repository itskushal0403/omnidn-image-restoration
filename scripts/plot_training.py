"""Plot validation curves of all runs (runs/*/val_metrics.csv) against the reference baselines.

    python scripts/plot_training.py   ->  docs/figures/training_curves.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = {"identity (no restoration)": (25.43, 0.618, 0.283), "SCUNet pretrained (79 GFLOPs)": (30.03, 0.807, 0.230)}


COLS = ["iter", "hours", "psnr", "ssim", "dists", "psnr_lowlight", "psnr_medical", "psnr_natural"]


def read_val_csv(path):
    """Robust reader: early runs wrote a header without 'dists' but appended rows with it."""
    rows = []
    for line in open(path).read().splitlines()[1:]:
        v = [float(x) if x else float("nan") for x in line.split(",")]
        if len(v) == 7:
            v = v[:4] + [float("nan")] + v[4:]
        rows.append(v)
    return pd.DataFrame(rows, columns=COLS)


def main():
    runs = sorted((ROOT / "runs").glob("*_stage1/val_metrics.csv"))  # fine-tune runs restart the clock
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for csv in runs:
        df = read_val_csv(csv)
        for ax, m in zip(axes, ("psnr", "ssim", "dists")):
            if m in df:
                d = df.dropna(subset=[m])
                ax.plot(d.hours, d[m], marker="o" if m == "dists" else None, ms=3, label=csv.parent.name)
    for ax, (m, lab) in zip(axes, (("psnr", "PSNR (dB) ↑"), ("ssim", "SSIM ↑"), ("dists", "DISTS ↓"))):
        for i, (name, vals) in enumerate(REF.items()):
            ax.axhline(vals[("psnr", "ssim", "dists").index(m)], ls="--", c=f"C{7 + i}", lw=1, label=name)
        ax.set_xlabel("training hours")
        ax.set_title(lab + " on proxy_test_v2 (300-pair subset)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "training_curves.png"
    fig.savefig(out, dpi=90)
    print("saved", out)


if __name__ == "__main__":
    main()
