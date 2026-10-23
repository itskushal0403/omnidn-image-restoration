# Research notes 1 — Blind / all-in-one / generalizable restoration

*Compiled 2026-10-19 from a literature sweep (2022–2026). Numbers marked "unverified" come from abstracts or
third-party summaries.*

## Key takeaways

- Our degradation list (Gaussian / Poisson / salt-and-pepper / speckle noise, Gaussian / motion / defocus blur, JPEG, chromatic
  aberration, mixtures) matches **ImageNet-C** (`imagecorruptions`) and **albumentations** almost 1:1. It does **not** match the
  "all-in-one" benchmarks (haze / rain / snow / low-light / blur). So the relevant prior work is *blind denoising* +
  *practical degradation models* (BSRGAN, Real-ESRGAN, SCUNet), not prompt-based all-in-one networks.
- **Data + degradation-pipeline design is the biggest lever**, more than architecture. This holds in every 2025–2026 challenge report
  and in FoundIR / GenDeg / ABAIR.
- **Diffusion / flow restorers are a poor fit when single-pass FLOPs are counted.** The NTIRE 2026 denoising report notes they
  still lag on PSNR. The practical "small generative model" is a regression network plus a perceptual / adversarial fine-tune.
- **Training-only tricks cost zero inference FLOPs:** feature-statistic noise injection, dropout, masking, re-parameterization,
  distillation, EMA, weight interpolation.

## 1. All-in-one methods

| Method | Venue | Core idea | Unseen/mixed? | Cost | Use for us |
|---|---|---|---|---|---|
| PromptIR | NeurIPS'23 [2306.13090](https://arxiv.org/abs/2306.13090) | Restormer + learnable prompts | Weak OOD; 25.9 dB on composite CDD-11 | ~36M, ~130 GMACs | Low |
| InstructIR | ECCV'24 [2401.16468](https://arxiv.org/abs/2401.16468) | NAFNet + text-instruction conditioning | Needs a correct prompt | ~16M + text encoder | Low |
| OneRestore | ECCV'24 [2407.04621](https://arxiv.org/abs/2407.04621) | Composite weather / low-light with a scene-descriptor query | Closed set | moderate | Low–med (idea of composite training) |
| RAM / RAM++ | ECCV'24 [2409.19403](https://arxiv.org/abs/2409.19403) / [2509.12039](https://arxiv.org/abs/2509.12039) | Masked-image pretraining on degraded inputs → fine-tune; RAM++ adds adaptive masking + DINOv2 | Claims robustness | Restormer + DINOv2 | Medium (masked pretraining) |
| AdaIR | ICLR'25 [2403.14614](https://arxiv.org/abs/2403.14614) | Frequency sub-band modulation | Weak on unseen pixelation | Restormer-size | Low–med |
| MoCE-IR | CVPR'25 [2411.18466](https://arxiv.org/abs/2411.18466) | Mixture of complexity experts; bias toward cheap experts | Used in a LoViF-26 3rd-place entry | 25M / ~81 GMACs (S: ~37) | Medium; a static trace would count all experts |
| DFPIR | CVPR'25 [2505.12630](https://arxiv.org/abs/2505.12630) | CLIP-guided degradation-aware feature perturbation | Closed set | Restormer + CLIP | Low |
| Perceive-IR | arXiv [2408.15994](https://arxiv.org/abs/2408.15994) | CLIP quality perceiver + difficulty-adaptive contrastive loss | Claims zero-shot | Heavy | Low |
| **ABAIR** | arXiv [2411.18412](https://arxiv.org/abs/2411.18412) | Pretrain on 450K images with synthetic degradations → LoRA adapters | **Region-wise "degradation CutMix": 31.11 dB vs 30.63 (one global degradation) vs 27.68 (two stacked global degradations)**; per-pixel degradation-segmentation auxiliary head +0.1 dB | 41M | **High (data recipe)** |
| **AnyIR** | arXiv [2504.14249](https://arxiv.org/abs/2504.14249) | Prompt-free U-Net; attention branch + gated-conv branch; spatial-frequency fusion | Best on CDD-11 (29.69), zero-shot underwater / desnow | **AnyIR-T 5.7M / 26 GMACs** | **High (block template)** |
| **DCPT** | ICLR'25 [2501.15510](https://arxiv.org/abs/2501.15510) | Pretrain the encoder to *classify the degradation*, then restore | NAFNet 5-task 27.76→29.84; mixed +5 dB; zero-shot denoising +1.1 dB | training-only | **High** |
| FoundIR / v2 | ICCV'25 [2412.01427](https://arxiv.org/abs/2412.01427) / [2512.09282](https://arxiv.org/abs/2512.09282) | 1M real paired images; v2: the task mixing proportions matter | OOD gains from scale | large | Med–high (mixing ratios) |
| GenDeg | CVPR'25 [2411.17687](https://arxiv.org/abs/2411.17687) | Diffusion generates 550K diverse degraded images | Significant OOD gains | offline | Medium (principle: diversify) |
| R2R | CVPR'26 | Retrieval bank of degradation priors → light backbone | — | ~91% fewer MACs | Medium |

## 2. Generalization to unseen degradations

- **Masked Image Training** (CVPR'23, [2303.13132](https://arxiv.org/abs/2303.13132)): mask input pixels and attention features.
  Trained on σ=15 Gaussian only, it generalizes to speckle / Poisson / s&p, but over-smooths. → Use a *low* mask ratio only.
- **RNINet** (ICLR'25): plain U-Net + **noise injection into per-channel feature mean / std** (DSU-style). +1.7 dB over Masked Training on
  speckle, wins on s&p / Poisson / mixtures, 10× faster. → **High**: training-only, ~0 inference FLOPs.
- **CLIPDenoising** (CVPR'24): frozen CLIP-ResNet encoder generalizes (incl. low-dose CT). The encoder FLOPs count against us → low.
- **Noise-to-Gaussian translation** (CVPR'26 Findings; Poisson2Gaussian +0.75 dB): the idea is subsumed by training on all noise types.
- **Dropout family**: Reflash Dropout (CVPR'22, channel dropout p≈0.5 before the last conv — helps most in *multi-degradation*
  training); Navigating Beyond Dropout (CVPR'24); Adaptive Dropout (CVPR'25). → **High**, free at inference.
- **"Networks are slacking off"** (NeurIPS'23, [2305.15134](https://arxiv.org/abs/2305.15134)): networks overfit the degradation when
  content is complex relative to the degradation range. → Make degradation diversity very wide.
- **Degradation pipelines**:
  - **BSRGAN** — random shuffle of the degradation order.
  - **Real-ESRGAN** — second-order degradations, sinc ringing.
  - **SCUNet** — Gaussian + Poisson + speckle + JPEG + camera noise, shuffle, double degradation. Closest to our list.
- **Scale**:
  - NTIRE'25 denoising winner pretrained on ~2M images; NTIRE'26 report: LSDIR vs DIV2K-only ≈ +3 dB.
  - But gains saturate at moderate data size for small models (Klug & Heckel). For small models, **degradation diversity > raw image count**.

## 3. Challenge lessons

- **NTIRE 2025 Denoising** (σ=50) winner SRC-B, 31.20 dB:
  - Restormer + NAFNet hybrid, ~2M-image pretraining, quality-filtered data.
  - Alternating L1 / L2 / **stationary-wavelet (SWT) loss**, progressive patches 256→768.
- **NTIRE 2026 Denoising**:
  - Winner: Restormer + HAT + MambaIR, LSDIR + Unsplash data. Diffusion models still lag on PSNR.
  - One team saw a 0.9 dB val→test gap → **keep an honest held-out val set.**
- **NTIRE 2025 Efficient SR** winners (~0.13–0.17M params): re-parameterization (ConvLoRA, ECB), **distillation**, EMA,
  multi-stage L1→L2, LSDIR.
- **NTIRE 2026 Efficient Low-Light Enhancement** — *closest to our ranking scheme* (rank aggregation incl. SSIM, LPIPS,
  **DISTS**):
  - Winner: lightweight Retinexformer + structural re-param.
  - Loss **L1 + LPIPS + DISTS + SSIM + TV**, EMA, progressive resolution.
  - Metrics disagree → checkpoint selection on the combined rank matters.
- **LoViF 2026 Real-World All-in-One**: FoundIR as extra data, pseudo-label test-time retraining, MoCE-IR backbone.

## 4. Generative (diffusion / flow) vs regression + GAN

- **Diffusion / flow**:
  - ResShift / SinSR: ~118M-param UNet, ~75 GFLOPs per pass + VQGAN.
  - SD-based one-step (OSEDiff, TSD-SR, PiSA-SR): ~1B params.
  - → Infeasible at low FLOPs.
- **Perception–distortion tradeoff** (Blau & Michaeli CVPR'18): realism costs PSNR. The conflict is milder for VGG-feature distances,
  and DISTS is VGG-based.
- **PMRF** (ICLR'25): posterior mean → rectified flow; needs a 2nd net + steps.
- **Recommendation**: regression net + perceptual / adversarial fine-tune + **weight interpolation** (ESRGAN-style) to pick the
  operating point that minimizes summed rank.

## 5. Medical images

- **Physical noise models**:
  - MRI magnitude → Rician.
  - X-ray / CT → Poisson quantum + electronic Gaussian; low-dose CT image-domain noise is spatially correlated / streaky.
  - Ultrasound → multiplicative speckle.
- **The organisers most likely apply the *same synthetic* corruptions to medical images**, so *content coverage* matters more than
  physics. Images are grayscale (probably replicated to 3 channels) with big black backgrounds, where SSIM punishes residual noise or
  colour tint.

## 6. Low-light

- **Noise is signal-dependent** (variance ∝ intensity), so relative noise is largest in the dark.
- **skimage `poisson` mode**: scales by the number of unique values (~256 for uint8).
- **The target is most likely the *clean dark image*** (not an enhanced one) → **do not brighten**.
- **LOL ground truth already contains native noise.**

## Top-10 actionable ideas (priority order)

1. **Degradation engine that mirrors the likely generator.** imagecorruptions + albumentations + skimage defaults; 1–3 ops in random
   order; mild→extreme severities; 5–10% identity pairs.
2. **Hold-out validation with unseen families.** Unseen families for validation (pixelate, zoom / glass blur, contrast, ISO noise,
   down-up, posterize…). ABAIR **region-wise degradation CutMix**. Checkpoint selection by 4-metric rank-sum.
3. **Content mixture.** Natural (LSDIR, DIV2K / Flickr2K, Unsplash) + low-light + medical; train on **whole images resized to 256** as
   well as crops.
4. **Compact backbone at ~5–20 GMACs** (NAFNet-style U-Net or AnyIR-T-like gated-conv blocks); no CLIP / DINO encoders.
5. **Staged losses + weight interpolation.**
   - Stage 1: Charbonnier / L1 + FFT / SWT.
   - Stage 2: + skimage-matched SSIM + small DISTS / LPIPS (+ optional light GAN).
   - Then interpolate weights.
6. **Free regularizers**: RNINet feature noise injection, dropout before the last conv, low-ratio input masking.
7. **Distill from a large teacher** (NAFNet-64 / Restormer) + structural re-parameterization + EMA (+0.2–0.5 dB at fixed FLOPs in
   NTIRE efficient tracks).
8. **DCPT-style degradation-classification pretraining** or an auxiliary degradation-segmentation head (removed at inference).
9. **Domain handling**: grayscale detection → output the channel mean; never brighten low-light; over-sample Poisson / speckle on dark
   images.
10. **Measurement hygiene**: count FLOPs with both fvcore and ptflops; no TTA; EMA / model soups; exact skimage SSIM.
