# Project plan — small generative model for blind image restoration

> **Plan vs. final (read this first).** This plan was written before the 30-hour deadline was known. What was
> actually built (see `docs/02_experiment_log.md` and `docs/report/technical_report.pdf`):
> - **Teacher→student distillation:** replaced by training two sizes (S 2.96 G, M 11.4 G) in parallel. A later KD
>   ablation lost at equal wall-clock.
> - **Data:** stored at long side ≤ 512 (75.6k samples).
> - **Not built:** FFT blocks, re-parameterisation and the auxiliary heads.
> - **Final submission:** OmniDN-S with a weight blend of two DISTS-weighted fine-tunes.

*Written 2026-10-19 after the research phase (see `docs/research/01–04`). This is the "why" document: every major design
decision below links back to evidence. It will be updated as experiments confirm or refute the choices.*

---

## 0. What we are optimising

The final score is built from **four independent ranks** (FLOPs ↓, PSNR ↑, SSIM ↑, DISTS ↓), equally weighted. Consequences:

1. **FLOPs is 25% of the score.** A 60–150 GMAC model (what most people will build) gives that rank away. We aim for a
   **small model (≈1–5 GMACs)** whose quality is close to large models. This is achieved by distilling a large teacher into it,
   with training-only tricks that cost zero inference FLOPs.
2. **Two of the three quality metrics are fidelity metrics (PSNR, SSIM)** and one is perceptual (DISTS). So we stay
   fidelity-first, and buy DISTS with a small, carefully tuned perceptual stage plus weight interpolation, rather than a
   full GAN (which would cost 1–2 dB PSNR).
3. **The test set is unknown and diverse** (natural / low-light / medical; unknown and mixed degradations). Generalization
   comes mainly from **data and degradation diversity**, not from the architecture.

**Fair-play policy.** We measure FLOPs exactly as the organisers do (fvcore + ptflops at 1×3×256×256). We never hide
computation in ops the counters skip (no custom autograd wrappers around convs, no input-dependent routing that a fixed test
tensor would under-count). If we use FFT blocks (which the counters report as 0), we disclose their real cost in the report.

---

## 1. Pipeline overview

```
 ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐   ┌──────────────────┐
 │ 1. Clean data │──▶│ 2. Degradation    │──▶│ 3. Teacher (large) │──▶│ 4. Student (small)│
 │ natural/low-  │   │ engine (on-the-fly│   │ ~60–100 GMACs      │KD │ ~1–5 GMACs        │
 │ light/medical │   │ random mixtures)  │   │ fidelity training  │──▶│ fidelity + KD     │
 └──────────────┘   └──────────────────┘   └───────────────────┘   └────────┬─────────┘
                                                                              │
          ┌────────────────────────────────────────────────────────────────────┘
          ▼
 ┌──────────────────────┐   ┌───────────────────────┐   ┌─────────────────────────┐
 │ 5. Metric-aligned     │──▶│ 6. Perceptual stage    │──▶│ 7. Weight interpolation  │──▶ submission
 │ fine-tune (PSNR-loss  │   │ (+DISTS, optional tiny │   │ pick α on our proxy test │
 │ + exact skimage SSIM) │   │ GAN) — "generative"    │   │ by simulated rank-sum    │
 └──────────────────────┘   └───────────────────────┘   └─────────────────────────┘
```

Every stage is evaluated on our **proxy test set** (Section 4), which mimics the hidden test set.

---

## 2. Data (clean ground-truth images)

All sources are public / Creative Commons / standard research benchmarks (rule 3.1). Raw downloads are converted into
a uniform store, `data/clean/<domain>/<source>/*.png` (max side 1024, lossless), plus a `manifest.csv` recording source,
license, size and split.

| Domain | Sources (planned) | ≈ #images |
|---|---|---|
| Natural | LSDIR (main, 85k pool), DIV2K, Flickr2K, Unsplash-Lite subset, FFHQ subset | ~60k |
| Low-light (clean dark content) | LOL v1/v2 (real + synthetic), SICE under-exposed members, SID long exposure (dark-rendered), plus darkened natural images synthesised on the fly | ~5k + synthetic |
| Medical | Chest X-ray (NIH-14, Montgomery/Shenzhen, Indiana, pediatric), dental panoramic X-ray, MRI (IXI T1/T2/PD slices, BraTS slices), CT (LoDoPaB-CT), ultrasound (BUSI, HC18), histology, fundus, dermoscopy | ~30k |

**Sampling during training:** ~55% natural, ~35% medical, ~10% low-light (to be tuned). 256×256 patches come from two
modes:
- random crops after a random downscale (texture scale diversity);
- **whole images resized to 256** — the hidden test images are probably whole images at 256.

**Held out entirely** (never trained on): Kodak24, CBSD68, McMaster, Urban100, LSDIR-val, LOL/LOL-v2 test splits, and
held-out patients/subjects of each medical set → used for the proxy test.

---

## 3. Degradation engine (the most important component)

The engine is modular: each degradation is a function with a severity sampler. A *composer* draws 1–3 of them (BSRGAN/SCUNet
style, mostly physical order, sometimes shuffled) and returns the corrupted image plus labels of what was applied.
All outputs are **clipped and quantised to uint8**, exactly as the test images will be.

| Family | Variants | Why |
|---|---|---|
| Gaussian noise | per-channel / gray / channel-correlated / spatially-correlated; σ 1–75; **skimage default var 0.01 (σ=25.5)** | listed |
| Poisson / shot | skimage rule (2^⌈log2 #unique⌉), peak λ 1–1000, Poisson-Gaussian, camera ISP noise (Brooks unprocessing), row/column banding | listed, low-light |
| Salt & pepper / impulse | per-element (skimage, **amount 0.05 default**) / per-pixel (albumentations) / salt only / pepper only / random-valued / dead pixels | listed |
| Speckle | skimage `x + x·n` (default var 0.01), Gamma multiplicative (ultrasound) | listed |
| Gaussian blur | iso / aniso kernels, **cv2 k ∈ {3,5,7,9,15}**, σ 0.3–5 | listed |
| Motion blur | linear (any angle, 5–31 px), random camera trajectories | listed |
| Defocus blur | disk PSF r 1–10 (anti-aliased), box, median | listed |
| JPEG | q 5–95 (PIL/cv2, 4:2:0), double JPEG, WebP, JPEG2000 | listed |
| Chromatic aberration | lateral (radial R/B scaling, albumentations-style shifts) + axial (extra R/B blur) | listed |
| Medical-specific | Rician (MRI), X-ray quantum noise, low-dose CT-like correlated noise | medical domain |
| "Unknown" extras | down-up resampling, pixelate, posterize/banding, ringing/over-sharpening, ISO noise, glass/zoom blur, vignetting/contrast/brightness (mild), dust/scratches | "may contain other corruptions" |
| Spatial mixing | **region-wise degradation CutMix** (ABAIR: +0.5 dB) | generalization |
| Identity | ~3–5% clean→clean pairs | must not damage clean images |

The engine is validated visually: a gallery script renders every family at every severity so you can inspect it
(`docs/figures/degradation_gallery.png`).

---

## 4. Proxy test set and evaluation protocol

`data/val/proxy_test_v1/` contains a fixed set of **~300 pairs** (seeded, built once), stored as uint8 PNG:
- **Domains:** natural (150), low-light (50), medical (100: X-ray, MRI, CT, ultrasound, …).
- **Degradation specs:** 50% *library-default* specs (what a TA would most likely type), 50% random mixtures.
- **OOD split:** a separate *OOD* split uses degradation families **withheld from training during development**, to measure
  true generalization.

Metrics are computed by `src/dncomp/metrics/competition_metrics.py`, which matches the organisers exactly: skimage PSNR/SSIM on
uint8 (`channel_axis=2, data_range=255`), official DISTS, FLOPs via fvcore and ptflops. Results are broken down per domain and
per degradation family.

**Rank simulation:** we keep a pool of "competitor-like" models (our own baselines of various sizes plus public pretrained
restorers) and compute where each candidate would rank on all four axes. This is how the final operating point is chosen.

---

## 5. Model family — "OmniDN"

A NAFNet-style U-Net. Research (doc 04) shows this family gives the best PSNR per counted MAC in the 1–20 G range.

- **Block:** LayerNorm → 1×1 conv → 3×3 depthwise → SimpleGate → simplified channel attention → 1×1, plus a gated FFN.
- **Block allocation:** 4 downsamplings (bottleneck 16×16) with most blocks deep. A block at 1/16 resolution costs the same
  MACs as at full resolution but sees 16× more context, which is essential for motion and defocus blur.
- **Candidate upgrades, each validated by ablation:**
  1. a global-attention block at the 16×16 bottleneck (~0.1 G);
  2. FFT global-filter blocks at deep levels (with fair-play disclosure);
  3. CoordConv input channels (chromatic aberration is radial);
  4. ECB-style re-parameterised depthwise convs (train multi-branch, fuse to one conv for inference: +~0.1 dB, 0 extra FLOPs);
  5. an auxiliary degradation-classification head (DCPT / ABAIR), dropped at inference.
- **Generalization regularizers (0 inference cost):** feature-statistics noise injection (RNINet, ICLR'25), channel dropout
  before the last conv, low-ratio input masking.

**Sizes:**

| Name | Role | Target GMACs |
|---|---|---|
| OmniDN-T | teacher | ~60–100 |
| OmniDN-S | main student | ~3–5 |
| OmniDN-XS | ultra-light student | ~1 |

The submission size is chosen from the Pareto curve (Section 7).

---

## 6. Training recipe (starting points; each is checked on the proxy test)

| Stage | Loss | Notes |
|---|---|---|
| A. Teacher | Charbonnier + 0.05·FFT-L1 | AdamW, cosine schedule, progressive patch 128→256, EMA 0.999, bf16, long schedule on 1 GPU (A6000) |
| B. Student | Charbonnier(s, GT) + λ·L1(s, teacher) + small feature-affinity KD | teacher labels fresh degradations on the fly (infinite data); EMA |
| C. Metric-aligned fine-tune | PSNR-loss (per-image log-MSE, = mean PSNR) + λ_s·(1 − SSIM_skimage-exact) | finish at exactly 256×256 |
| D. Perceptual / "generative" | C + λ_D·DISTS (fp32, full image), optional tiny GAN (≤0.005) | fine-tuned from C |
| E. Interpolation | θ = (1−α)·θ_C + α·θ_D | α chosen on the proxy test by simulated rank-sum |

**Inference wrapper (submission):**
- fixed 256×256 input → model → clamp → **round** (not truncate) to uint8;
- optional grayscale-consistency fix for medical images (if the input content is gray, output gray) — kept only if it helps on
  the proxy test.

---

## 7. Experiments (each one is logged in `docs/02_experiment_log.md`)

1. **Baselines:** identity, classical filters, public pretrained restorers → calibrates the proxy test.
2. **Architecture ablations** at student size (short schedules): plain NAFNet-16 vs deep-skewed vs +attention vs +FFT vs
   +CoordConv vs +regularizers.
3. **Loss ablations:** L1 vs Charbonnier+FFT vs PSNR-loss; SSIM weight; DISTS weight.
4. **Teacher training** (long).
5. **Student KD vs no-KD;** Pareto family (XS / S / M).
6. **Perceptual stage + α sweep** → final operating point.
7. **Robustness:** OOD degradation families, each domain separately.

---

## 8. Compute and storage

**Compute:**
- `mtech` QoS allows **2 GPUs / 2 running jobs**, 36 h per job.
- Long runs are split into chained jobs that resume from checkpoints automatically (`scripts/slurm/`).
- Plan: teacher on the A6000 (cn07); ablations and students on the A30s.

**Storage:**

| Item | Size |
|---|---|
| Processed clean data | ≈ 90 GB |
| Temporary raw downloads (processed one by one, then deleted) | ≈ 30 GB at a time |
| Checkpoints / runs | ≈ 20 GB |
| Proxy test | < 1 GB |
| **Total new** | **≈ 150 GB** |

Current `/scratch` quota is 200 GB, with 109 GB already used → **a quota increase is needed** (requested from the user).

---

## 9. Repository layout

```
denoising_competition/
├── README.md                 project overview + how to run each step
├── docs/
│   ├── 01_plan.md            this file
│   ├── 02_experiment_log.md  every run: config, result, conclusion
│   └── research/             literature notes (01–04)
├── configs/                  one YAML per experiment
├── src/dncomp/               python package
│   ├── data/                 datasets, degradation engine, preprocessing
│   ├── models/               OmniDN + baselines + re-parameterisation
│   ├── losses/               Charbonnier, FFT, PSNR-loss, exact SSIM, DISTS, KD
│   ├── metrics/              exact competition metrics + FLOPs
│   └── utils/
├── scripts/
│   ├── data/                 download + preprocess scripts (one per source)
│   └── slurm/                job scripts (train / eval / chain)
├── submission/               final inference.py + weights + FLOP check
├── tests/                    unit tests (SSIM matches skimage, rounding, FLOPs, …)
└── data, checkpoints, runs, logs → symlinks into /scratch (heavy files)
```
