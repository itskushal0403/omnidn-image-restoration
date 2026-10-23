"""Exact re-implementation of the competition's evaluation protocol.

All four axes from the task PDF (Section 5):
  * FLOPs  - fvcore / ptflops on a fixed (1, 3, 256, 256) input       (lower is better)
  * PSNR   - skimage.metrics.peak_signal_noise_ratio on uint8 images    (higher is better)
  * SSIM   - skimage.metrics.structural_similarity on uint8 images     (higher is better)
  * DISTS  - official DISTS-pytorch implementation                     (lower is better)

Full-reference metrics are computed per image on uint8 [0, 255] 256x256 RGB arrays
and averaged over the set, exactly like the organisers describe.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ----------------------------------------------------------------------------- PSNR / SSIM
def psnr_uint8(gt: np.ndarray, pred: np.ndarray) -> float:
    """PSNR between two HxWx3 uint8 images (data_range inferred = 255 for uint8)."""
    assert gt.dtype == np.uint8 and pred.dtype == np.uint8
    return float(peak_signal_noise_ratio(gt, pred, data_range=255))


def ssim_uint8(gt: np.ndarray, pred: np.ndarray) -> float:
    """SSIM between two HxWx3 uint8 images, skimage defaults (7x7 uniform window)."""
    assert gt.dtype == np.uint8 and pred.dtype == np.uint8
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=255))


# ----------------------------------------------------------------------------- DISTS
class DISTSMetric:
    """Thin wrapper around the official DISTS_pytorch package (no gradients)."""

    def __init__(self, device: str | torch.device = "cuda"):
        from DISTS_pytorch import DISTS

        self.device = torch.device(device)
        self.model = DISTS().to(self.device).eval()

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        # DISTS expects float RGB in [0, 1], shape (N, 3, H, W)
        return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    @torch.no_grad()
    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        x = self._to_tensor(gt).to(self.device)
        y = self._to_tensor(pred).to(self.device)
        return float(self.model(x, y, batch_average=True).item())


# ----------------------------------------------------------------------------- FLOPs
@dataclass
class FlopReport:
    fvcore_gflops: float | None = None   # fvcore counts one multiply-add as one "flop"
    ptflops_gmacs: float | None = None   # ptflops reports MACs
    params_m: float = 0.0
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        fv = f"{self.fvcore_gflops:.3f}" if self.fvcore_gflops is not None else "n/a"
        pt = f"{self.ptflops_gmacs:.3f}" if self.ptflops_gmacs is not None else "n/a"
        return f"fvcore={fv} G | ptflops={pt} GMac | params={self.params_m:.3f} M"


def count_flops(model: torch.nn.Module, size: int = 256, device: str = "cpu") -> FlopReport:
    """Measure complexity the way the organisers do: a (1, 3, 256, 256) input tensor."""
    model = model.to(device).eval()
    rep = FlopReport(params_m=sum(p.numel() for p in model.parameters()) / 1e6)
    x = torch.rand(1, 3, size, size, device=device)

    try:
        from fvcore.nn import FlopCountAnalysis

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fca = FlopCountAnalysis(model, x)
            fca.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            rep.fvcore_gflops = fca.total() / 1e9
            skipped = fca.unsupported_ops()
            if skipped:
                rep.notes.append("fvcore skipped ops: " + ", ".join(f"{k}x{v}" for k, v in skipped.items()))
    except Exception as e:  # pragma: no cover - diagnostic only
        rep.notes.append(f"fvcore failed: {e!r}")

    try:
        from ptflops import get_model_complexity_info

        with torch.no_grad():
            macs, _ = get_model_complexity_info(
                model, (3, size, size), as_strings=False, print_per_layer_stat=False, verbose=False
            )
        rep.ptflops_gmacs = macs / 1e9
    except Exception as e:  # pragma: no cover
        rep.notes.append(f"ptflops failed: {e!r}")
    return rep


# ----------------------------------------------------------------------------- aggregation
@dataclass
class MetricAccumulator:
    """Accumulates per-image metrics and reports set averages (+ per-group breakdowns)."""

    dists: DISTSMetric | None = None
    rows: list[dict] = field(default_factory=list)

    def add(self, gt: np.ndarray, pred: np.ndarray, **tags) -> dict:
        row = {"psnr": psnr_uint8(gt, pred), "ssim": ssim_uint8(gt, pred), **tags}
        if self.dists is not None:
            row["dists"] = self.dists(gt, pred)
        self.rows.append(row)
        return row

    def summary(self, by: str | None = None) -> dict:
        import pandas as pd

        df = pd.DataFrame(self.rows)
        cols = [c for c in ("psnr", "ssim", "dists") if c in df]
        if by is None:
            return df[cols].mean().to_dict()
        return df.groupby(by)[cols].mean().to_dict(orient="index")
