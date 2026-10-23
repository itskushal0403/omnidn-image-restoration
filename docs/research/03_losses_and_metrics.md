# Research notes 3 — Metric internals, losses, perception–distortion balance

*Compiled 2026-10-19. skimage (v0.26.0) and DISTS_pytorch (0.1) sources were read directly.*

## 1. skimage SSIM — exact behaviour

**Defaults** (`structural_similarity(gt, pred, channel_axis=2, data_range=255)`):

| Setting | Value |
|---|---|
| Window | `win_size=7`, **uniform** (`scipy.ndimage.uniform_filter`), `gaussian_weights=False` |
| Constants | `K1=0.01`, `K2=0.03` |
| Covariance | `use_sample_covariance=True` → variances × 49/48 |
| `data_range` | inferred as 255 for uint8 (float inputs without it raise an error) |

How the score is formed:
- Each channel is computed separately; the SSIM map is cropped by 3 px (250×250); the mean is taken per channel, then over channels.
- `channel_axis` *must* be passed for HxWx3 input (otherwise a ValueError is raised). The organisers therefore use
  `channel_axis=2` (or -1).

**Exact PyTorch match** (verified to ~1e-14 in fp64, ~1e-5 in fp32):

```python
def ssim_sk(x, y, R=1.0, win=7):            # x, y: N,C,H,W
    f = lambda t: F.avg_pool2d(t, win, 1, 0)   # == uniform_filter + crop(3)
    cn = win * win / (win * win - 1)
    ux, uy = f(x), f(y)
    vx = cn * (f(x * x) - ux * ux); vy = cn * (f(y * y) - uy * uy); vxy = cn * (f(x * y) - ux * uy)
    C1, C2 = (0.01 * R) ** 2, (0.03 * R) ** 2
    S = (2 * ux * uy + C1) * (2 * vxy + C2) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))
    return S.mean((2, 3)).mean(1)              # per-image SSIM
```

- The `gaussian_weights=True` variant (11×11, σ=1.5) scores 0.01–0.04 lower, but rankings correlate.
- **Optimise the default (uniform) version.**

## 2. PSNR

- **uint8** → `data_range = 255`, MSE in float64. A perfect image gives inf PSNR.
- **uint8 conversion**: we must **round** (`np.round` / `+0.5`), not truncate. Truncation costs ~0.1–0.2 dB at ≥40 dB.
- **Mean per-image PSNR** is exactly optimised by `mean_i log(MSE_i)` = **NAFNet / HINet `PSNRLoss`**: `10/ln10 · log(mean((pred−gt)²) + 1e-8)`
  per image. This weights each image by 1/MSE_i, so easy images (clean X-rays, flat backgrounds) count equally.
- **Zhao et al.** (TCI'17, [1511.08861](https://arxiv.org/abs/1511.08861)):
  - L1-trained networks beat L2 *even on PSNR*.
  - 0.84·MS-SSIM + 0.16·L1 was best on both PSNR and SSIM.
  - L1 → L2 fine-tune gives the lowest L2.
  - → Train with L1 / Charbonnier, then fine-tune with log-MSE.

## 3. DISTS (official package)

**What it computes:**
- **Backbone**: VGG16 (ImageNet) with **L2pooling** (Hanning 3×3) instead of max-pool.
- **Feature stages**: raw input + relu1_2 / 2_2 / 3_3 / 4_3 / 5_3 (1475 channels).
- **Statistics**: per channel, **global** spatial mean / variance / covariance → texture term S1 and structure term S2 (c=1e-6),
  learned α, β.
- **Score**: `1 − Σ(α·S1 + β·S2)`.
- **Input**: RGB in [0,1], ImageNet normalisation, **no resize at 256**.
- **Global colour / brightness errors count**, via the raw-pixel stage.

**Using it as a loss:**
- Call `D(x, y, require_grad=True, batch_average=True)`.
- **Pitfalls:**
  - α / β still have `requires_grad=True` → call `D.requires_grad_(False)`.
  - **Fails under fp16** (c=1e-6 underflows) → compute in fp32 outside autocast.
  - Statistics are global → compute on full 256 images.
  - Ports (piq / pyiqa) give different numbers → validate with the official package only.

**Evidence** (Ding et al. IJCV'21):
- Pretrain with MAE, then fine-tune with DISTS.
- DISTS-only denoising leaves noise in smooth regions (25.56 dB / 0.775 / DISTS 0.175 vs MAE 26.44 / 0.809 / 0.189).
- → **Never DISTS alone.**
- Adversarially vulnerable (Ghildyal & Liu TMLR'23) → keep pixel losses dominant, EMA, apply random shared flips / shifts to both
  images inside the loss (E-LPIPS idea).

## 4. Perception–distortion balance

- **Theory**: Blau & Michaeli CVPR'18; Freirich et al. NeurIPS'21 — optimal tradeoff points are reachable by interpolating between MMSE
  and perceptual estimators → justifies **weight interpolation** (ESRGAN; free in FLOPs; the perceptual model must be fine-tuned from
  the fidelity model).
- **WGSR** (CVPR'24): **DISTS instead of LPIPS as the perceptual term → +0.3 dB PSNR** at equal / better perceptual quality; a
  wavelet-domain L1 + GAN on detail subbands gave +0.5 dB more.
- **FFT-L1** as an auxiliary loss (MIMO-UNet; AIM'25 winner L1 + 0.05·FFT) mainly helps PSNR on blur.
- **LoViF 2026**: small perceptual weights (0.005), or perceptual early → pure fidelity at the end; the top entry used L1 only.

## 5. GAN / diffusion

- **Real-ESRGAN fine-tune**: U-Net-SN discriminator, L1 1 + VGG 1 + GAN 0.1, EMA 0.999, init from the PSNR model.
- **AIM 2025 efficient perceptual SR**: stage 1 L1 + 0.05 FFT → stage 2 L1 + 0.1 perceptual + LDL + 0.1 GAN.
- **For our ranking (2 of 3 quality metrics are fidelity)**: a GAN typically costs 1–2 dB + SSIM in SR → likely a net loss. If used,
  keep the GAN weight ≤0.005–0.01 and blend back by interpolation.
- **One-step diffusion** (OSEDiff ~2.3 TMACs, SinSR ~119M params) is not viable.

## 6. Training tricks at fixed size

- **RCAN-it**:
  - Gains: 2× iterations +0.05 dB, large-patch fine-tune +0.05 dB, SiLU +0.04 dB, big batch + big lr = same quality ~4× faster.
  - Losses: mixup −0.04 dB, stochastic depth −0.45 dB.
- **Restormer progressive patches**: +0.07 dB.
- **Finish training at the eval resolution 256** (NAFNet's global pooling then sees test statistics; no TLC needed).
- **EMA 0.999**; flips / rot90 augmentation; no TTA (×8 FLOPs).

## 7. Choosing the operating point under a rank-average score

- Ranks depend on position relative to other teams, not on margins. 2/3 quality metrics are fidelity metrics → lean fidelity.
- **Weight interpolation**: DISTS gains flatten early while PSNR / SSIM losses keep growing → choose the "knee".

## Recommended loss recipe (starting points, to be validated by sweeps)

| Stage | Loss | Optimiser / schedule |
|---|---|---|
| 1 Fidelity | Charbonnier (ε=1e-3) + 0.05·FFT-L1 | AdamW lr 1e-3 (3e-4 big models), β=(0.9, 0.9), cosine → 1e-7, progressive patches, EMA 0.999 |
| 1b Metric-aligned (last 10–20%) | PSNRLoss + λ_s·(1 − SSIM_sk7), λ_s ∈ {10, 30, 100} (dB units) | lr 1e-4 → 0, 256 crops |
| 2 Perceptual balance | PSNRLoss + λ_s(1 − SSIM) + λ_D·DISTS (fp32, full 256, shared random flip / shift), λ_D ∈ {10, 25, 50, 100} | lr 5e-5–1e-4, 10–20% of the stage-1 iterations; GAN optional at ≤0.005 |
| Select | Interpolate stage-1b ↔ stage-2 EMA weights, α ∈ [0, 1] | Pick α by the simulated rank on the validation set (start: largest α with ΔPSNR ≥ −0.15 dB, ΔSSIM ≥ −0.003) |
