"""Render a gallery of every degradation in the training engine (and random composed chains) for inspection.

    python scripts/make_degradation_gallery.py      ->  docs/figures/degradation_gallery.png
                                                        docs/figures/degradation_chains.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dncomp.data.datasets import read_rgb  # noqa: E402
from dncomp.data.degradations import OPS, DegradationComposer, to_float, to_uint8  # noqa: E402
from dncomp.data.prep_utils import DATA_ROOT  # noqa: E402

OUT = ROOT / "docs" / "figures"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gt_dir = DATA_ROOT / "val" / "proxy_test_v1" / "gt"
    names = sorted(gt_dir.glob("*.png"))
    rng = np.random.default_rng(1)
    picks = {d: [p for p in names if f"_{d[:3]}_" in p.name] for d in ("natural", "medical", "lowlight")}
    base = {d: read_rgb(v[len(v) // 3]) for d, v in picks.items() if v}

    # 1) every op applied to the same natural image (2 random severities each)
    img = base["natural"]
    cols = 6
    rows = int(np.ceil((len(OPS) + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = axes.ravel()
    axes[0].imshow(img)
    axes[0].set_title("clean", fontsize=9)
    for i, op in enumerate(OPS, start=1):
        out = to_uint8(np.clip(op.fn(to_float(img), rng), 0, 1))
        axes[i].imshow(out)
        axes[i].set_title(f"{op.name}\n({op.family}, w={op.weight})", fontsize=8)
    for a in axes:
        a.axis("off")
    fig.suptitle("Training degradation engine — one random sample of every op", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "degradation_gallery.png", dpi=80)
    plt.close(fig)

    # 2) random composed chains on each domain
    comp = DegradationComposer()
    fig, axes = plt.subplots(len(base), 6, figsize=(18, 3.4 * len(base)))
    for r, (dom, im) in enumerate(base.items()):
        axes[r, 0].imshow(im)
        axes[r, 0].set_title(f"{dom}: clean", fontsize=9)
        for c in range(1, 6):
            lq, ops = comp(im, rng)
            axes[r, c].imshow(lq)
            axes[r, c].set_title("\n".join([", ".join(ops[i:i + 2]) for i in range(0, len(ops), 2)]) or "identity",
                                 fontsize=7)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle("Random degradation chains as seen during training", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "degradation_chains.png", dpi=80)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
