"""Training losses, each aligned with an evaluation metric (docs/research/03_losses_and_metrics.md).

All losses take predictions / targets as float tensors in [0, 1], shape (N, 3, H, W), and return a scalar.

| loss            | purpose                                                                 |
|-----------------|-------------------------------------------------------------------------|
| Charbonnier     | robust L1-like fidelity (main stage; L1 beats L2 even on PSNR)          |
| FFT-L1          | frequency-domain L1 — sharper edges, helps deblurring                   |
| PSNRLoss        | mean over images of 10·log10(MSE_i) == exactly (minus) the mean PSNR    |
| SSIMLoss        | 1 − SSIM, **bit-exact replica of skimage's default** (7×7 uniform)      |
| DISTSLoss       | official DISTS network (fp32, frozen) — the perceptual metric itself    |
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target).pow(2) + self.eps2).mean()


class FFTLoss(nn.Module):
    """L1 distance between the 2-D real FFTs (real and imaginary parts), computed in fp32."""

    def forward(self, pred, target):
        p = torch.fft.rfft2(pred.float(), norm="backward")
        t = torch.fft.rfft2(target.float(), norm="backward")
        return (torch.view_as_real(p) - torch.view_as_real(t)).abs().mean() / (pred.shape[-1] * pred.shape[-2]) ** 0.5


class PSNRLoss(nn.Module):
    """mean_i 10·log10(MSE_i) — minimising it maximises the per-image-averaged PSNR used by the organisers."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = 10.0 / math.log(10.0)

    def forward(self, pred, target):
        mse = (pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3))
        return self.scale * torch.log(mse + self.eps).mean()


def ssim_skimage(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0, win: int = 7) -> torch.Tensor:
    """Per-image SSIM identical to ``skimage.metrics.structural_similarity(..., channel_axis, data_range)``
    with default arguments (uniform 7×7 window, sample covariance, K1=0.01, K2=0.03, 3-px border cropped).
    Returns a tensor of shape (N,)."""
    x, y = x.float(), y.float()
    f = lambda t: F.avg_pool2d(t, win, stride=1, padding=0)  # == uniform_filter followed by the border crop
    cov_norm = win * win / (win * win - 1.0)
    ux, uy = f(x), f(y)
    vx = cov_norm * (f(x * x) - ux * ux)
    vy = cov_norm * (f(y * y) - uy * uy)
    vxy = cov_norm * (f(x * y) - ux * uy)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    return s.mean(dim=(2, 3)).mean(dim=1)


class SSIMLoss(nn.Module):
    def forward(self, pred, target):
        return 1.0 - ssim_skimage(pred, target).mean()


class DISTSLoss(nn.Module):
    """Official DISTS as a loss. Frozen, always fp32 (it silently breaks in fp16), full 256×256 images.
    A random flip shared by both images makes it harder for the network to 'game' the metric."""

    def __init__(self):
        super().__init__()
        from DISTS_pytorch import DISTS

        self.net = DISTS()
        self.net.requires_grad_(False)
        self.net.eval()

    def train(self, mode: bool = True):  # keep the metric network in eval mode always
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, pred, target):
        if torch.rand(()) < 0.5:
            pred, target = pred.flip(-1), target.flip(-1)
        with torch.autocast(device_type=pred.device.type, enabled=False):
            return self.net(target.float(), pred.float().clamp(0, 1), require_grad=True, batch_average=True)


class CombinedLoss(nn.Module):
    """Weighted sum of the losses above. Weights of 0 are skipped (and the module is not even built)."""

    def __init__(self, charbonnier=1.0, fft=0.0, psnr=0.0, ssim=0.0, dists=0.0):
        super().__init__()
        self.w = dict(charbonnier=charbonnier, fft=fft, psnr=psnr, ssim=ssim, dists=dists)
        builders = dict(charbonnier=CharbonnierLoss, fft=FFTLoss, psnr=PSNRLoss, ssim=SSIMLoss, dists=DISTSLoss)
        self.terms = nn.ModuleDict({k: builders[k]() for k, v in self.w.items() if v > 0})

    def forward(self, pred, target):
        logs, total = {}, 0.0
        for k, fn in self.terms.items():
            v = fn(pred, target)
            logs[k] = float(v.detach())
            total = total + self.w[k] * v
        return total, logs
