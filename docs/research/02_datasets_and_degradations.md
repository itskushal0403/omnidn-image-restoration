# Research notes 2 — Datasets and degradation synthesis

*Compiled 2026-10-19. "OK" = the URL / HF dataset was verified reachable without login on that date.*

## A. Datasets

### A1. Natural (high quality, used as clean GT)

| Dataset | #imgs / res | License | Source (verified) | Size | Notes |
|---|---|---|---|---|---|
| **LSDIR** | 84,991 train, ~2K | research benchmark | `danjacobellis/LSDIR` (197 parquet shards ×0.5 GB, no login) | 97.8 GB | Best main source; very clean |
| **DIV2K** | 800 + 100 val, 2K PNG | academic | `data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip` (OK) | 3.5 + 0.45 GB | Excellent |
| **Flickr2K** | 2,650, 2K PNG | research | `yangtao9009/Flickr2K` | 11.7 GB | Excellent |
| OST | 10,324 outdoor | research | `Iceclear/DF2K-OST` | 19.7 GB (with DF2K) | JPEG source → downscale ≥2× |
| Unsplash Lite | 25k | Unsplash Lite terms | TSV of URLs; `images.unsplash.com/...?w=1024&fm=png` | ~25 GB at 1024 | Good after downscale |
| FFHQ | 70k faces 1024 | CC BY-NC-SA 4.0 | `marcosv/ffhq-dataset` (per-file) | subset | Take ~3k |
| Kodak24 / CBSD68 / McMaster / Urban100 | small | benchmarks | `danjacobellis/kodak`, `deepinv/CBSD68`, `eugenesiow/Urban100` | <0.2 GB | **Validation only** |
| WED | 4,744 | research | download link dead (404) | — | skip |

### A2. Low-light

| Dataset | Content | Source | Size | Notes |
|---|---|---|---|---|
| LOL v1 | 500 pairs 600×400 | `Benxelua/LOL_v1` | 0.66 GB | Low frames are noisy |
| LOL-v2 real / synthetic | 789 / 1,000 pairs | `okhater/lolv2-real`, `okhater/lolv2-synthetic` | 0.6 / 0.5 GB | Synthetic low frames are cleaner |
| SID (Sony) | long-exposure raw = very clean GT | `storage.googleapis.com/isl-datasets/SID/Sony2025.zip` | 26.9 GB | Needs rawpy |
| ExDark | 7,363 real dark JPEGs (no GT) | `dronefreak/ExDark` | 0.3 GB | Noisy; only as heavily downscaled dark content |
| SICE | 589 multi-exposure sequences | `okhater/SICE` | 1.9 GB | Under-exposed members are fairly clean |

Low-light synthesis:
- **Gamma darkening** `y = β·(α·x)^γ`, γ∼U(1.5, 5).
- **Physically correct: Brooks et al. 2019 unprocessing.** Inverse tone curve / gamma / CCM / WB → darken → shot + read noise
  (`log_read = 2.18·log_shot + 1.2 + N(0, 0.26)`) → re-process.

### A3. Medical (no-login / permissive first)

| Modality | Dataset | Format | License | Source (verified) | Size |
|---|---|---|---|---|---|
| CXR | NIH ChestX-ray14 | 112k 1024² | unrestricted (cite) | `timm/nih-chest-xray-14` (JPEG q95) / `iraqigold/nih-chest-xray-14-flat` (PNG) | 21.6 / 42 GB |
| CXR | Montgomery + Shenzhen | 800, 3–4K PNG | public (NLM) | `openi.nlm.nih.gov/imgs/collections/...` | 4.4 GB |
| CXR | Indiana (Open-i) | 7,470 PNG | NLM | `.../NLMCXR_png.tgz` | 1.4 GB |
| CXR | Pediatric pneumonia | 5,856 JPEG | CC BY 4.0 | `hf-vision/chest-xray-pneumonia` | 1.2 GB |
| Dental | Panoramic X-ray | 9,974 | CC BY-NC 4.0 | `liodon-ai/dental-panoramic-xray-yolo` | 7.6 GB |
| MRI | **IXI** T1 / T2 / PD | NIfTI volumes, 16-bit | **CC BY-SA 3.0** | `biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar` etc. | 12.8 GB |
| MRI | BraTS 2023 GLI | volumes | CC BY 4.0 (mirror) | `MedOtter/brats2023-gli-dataset` | 13.4 GB |
| CT | **LoDoPaB-CT** GT | 362² float HDF5 slices | **CC BY 4.0** | Zenodo 3384092 (`ground_truth_validation.zip` 1.6 GB) | 1.6–16 GB |
| US | BUSI / HC18 | 780 / 1,334 | CC BY 4.0 | `MedOtter/BUSI`, Zenodo 1327317 | 0.3 GB |
| Histo | NCT-CRC-HE-100K / BreakHis | 224² / 700×460 | CC BY 4.0 / research | Zenodo 1214456, `hirundo-io/BreaKHis-original` | 11.7 / 0.2 GB |
| Fundus | FIVES / HRF | 800 2048² / 45 | CC BY 4.0 | `tyluan/FIVES`, `Zomba/HRF-high-resolution-fundus` | 1.8 GB |
| Derm | ISIC 2019 | 25k | CC BY-NC 4.0 | ISIC S3 bucket | 9.8 GB |
| Avoid | CheXpert, MURA, VinDr, PadChest, fastMRI, Mayo LDCT | — | restrictive terms / registration | — | — |

Processing notes:
- **Volumes** (IXI, BraTS) → axial slices from the middle 60% of the volume; drop background-only slices.
- **MRI** → 0.5–99.5 percentile windowing. **CT** → HU windowing (lung / soft tissue) → 8-bit.

## B. Degradations

### B1. Reference implementations

- **Real-ESRGAN** — 2-stage pipeline:
  - Blur: iso / aniso / generalized / plateau kernels, k=21, σ∈[0.2, 3], sinc p=0.1.
  - Resize: up / down / keep [0.2, 0.7, 0.1].
  - Gaussian noise: σ∈[1, 30], 40% gray.
  - Poisson: scale [0.05, 3].
  - JPEG: q∈[30, 95].
  - Final sinc: p=0.8.
- **BSRGAN** — random permutation of the order:
  - Blur: 2×, k 7–25.
  - Resize: down / up.
  - Gaussian noise: 2–25 (40% color / 40% gray / 20% channel-correlated).
  - JPEG: q 30–95.
  - Camera ISP noise: p=0.25.
  - BSRGAN+ adds Poisson, speckle, USM.
- **SCUNet** (blind denoising):
  - Gaussian σ∈[2, 50] always.
  - Poisson `10^U[2,4]`.
  - Speckle.
  - JPEG q∈[20, 95] always.
  - Camera noise.
  - Resize ×[0.5, 2].
  - Double degradation, shuffle.
- **ImageNet-C** (`imagecorruptions`), severity 1→5, [0,1] scale:

| corruption | severities 1→5 |
|---|---|
| gaussian_noise | σ .08 / .12 / .18 / .26 / .38 |
| shot_noise | λ 60 / 25 / 12 / 5 / 3 (`poisson(x·λ)/λ`) |
| impulse_noise | amount .03 / .06 / .09 / .17 / .27 |
| speckle_noise | σ .15 / .2 / .35 / .45 / .6 |
| gaussian_blur | σ 1 / 2 / 3 / 4 / 6 |
| defocus_blur | disk radius 3 / 4 / 6 / 8 / 10 |
| motion_blur | (radius, σ) (10,3) … (20,15) |
| jpeg_compression | q 25 / 18 / 15 / 10 / 7 |
| pixelate | .6 / .5 / .4 / .3 / .25 |

  Also: glass / zoom blur, contrast, brightness, fog, frost, snow, spatter, saturate, elastic.
- **albumentations 2.x** defaults:
  - GaussNoise std (0.2, 0.44) per channel.
  - ISONoise.
  - MultiplicativeNoise (0.9, 1.1).
  - ShotNoise (0.1, 0.3).
  - SaltAndPepper amount (0.01, 0.06) — *same mask for all channels*.
  - ImageCompression.
  - Downscale.
  - GaussianBlur σ (0.5, 3).
  - MotionBlur k (3, 7).
  - Defocus r (3, 10).
  - ChromaticAberration (primary ±0.02, secondary ±0.05).
  - Also: RicianNoise, Posterize, RingingOvershoot, Halftone, Dithering, Vignetting…
- **3D Common Corruptions**: ISO noise, low-light, colour quantization, bit error, H.265 artifacts, …

### B2. Physically-motivated models

| Model | Recipe |
|---|---|
| Poisson-Gaussian | `y = x + N(0, a·x + b)` (= skimage `localvar`) |
| Camera ISP noise | Brooks unprocess; row / column banding |
| Correlated noise | white noise ⊛ Gaussian(σ 0.5–1.5) |
| **Rician (MRI)** | `|x + n₁ + i·n₂|`, σ = 1–10% of max |
| X-ray quantum | `Poisson(x·N)/N`, N 50–2000, + Gaussian 1–5/255 |
| Low-dose CT | sinogram `Poisson(I₀·e^{−p})` → FBP (streaky correlated noise) |
| Ultrasound speckle | `x·Gamma(L, 1/L)`, L 1–10, or `x + x·n` |
| Impulse | fixed salt / pepper, random-valued, dead / hot pixels |
| Quantization | posterize 3–6 bits, palette quantization, dithering |
| Compression | JPEG q 5–95 (4:2:0), double JPEG, WebP, JPEG2000 |
| Chromatic aberration | lateral: scale R / B by 1 ± 0.2–1% about the centre; axial: extra blur on R / B |
| Defocus | disk PSF r 1–8 |
| Motion | linear 5–25 px any angle; random trajectories (Boracchi & Foi / DeblurGAN) |
| Resolution loss | down-up ×0.25–0.9, random interpolation; pixelate |
| Others | vignetting, haze `I = J·t + A(1−t)`, rain, dust / scratches, ringing (sinc), over-sharpening halos |

### B3. What a TA script most likely does — exact library behaviour

`skimage.util.random_noise` (v0.26) — converts with `img_as_float`, `clip=True`:

- `gaussian`: **var=0.01 → σ = 25.5/255** (default).
- `poisson`: `vals = 2**ceil(log2(#unique values))` → uint8 photo ⇒ vals=256 (mild noise). **Images with few unique values
  (dark / low dynamic range) get much stronger noise.**
- `s&p` / `salt` / `pepper`: **amount=0.05**, decided **per array element** → colored impulses on RGB (~14% of pixels touched).
- `speckle`: `x + x·N(0, 0.01)`.

Consequences:
1. Grayscale medical images replicated to 3 channels get **independent per-channel** noise → the model must map colour noise back to gray.
2. Clipping makes noise in black regions half-rectified (brightens them) → the model must learn to output 0 there.
3. Use the library defaults verbatim with high probability.

Other common defaults:

| Library call | Default |
|---|---|
| `cv2.GaussianBlur(k, 0)` | σ = 0.3·((k−1)/2 − 1) + 0.8 (k = 3/5/7/9/15 → σ = 0.8/1.1/1.4/1.7/2.6) |
| PIL JPEG | q = 75 |
| `cv2.imwrite` JPEG | q = 95 |
| "Motion blur" | typically a horizontal line kernel k = 5–21 via `cv2.filter2D` |

Benchmark severities:

| Degradation | Typical severities |
|---|---|
| Gaussian noise | σ 15 / 25 / 50 |
| JPEG | q 10–40 |
| Poisson | peak 1–30 |
| Impulse | 0.01–0.1 |

## Recommendation from this sweep

- **Dataset mix**: ~100k images (~65k natural, ~5.5k low-light, ~30–37k medical), stored at max side 1024 PNG ≈ 86 GB (≈55 GB at 768).
- **Raw downloads**: process in streaming fashion.
- **Sampling ratio**: natural 55% / medical 35% / low-light 10%.
- **Degradation sampler**: 1 / 2 / 3 ops with p 0.45 / 0.35 / 0.20; physical order 80%, shuffled 20%; 3% identity; 30% exact library
  defaults; Gaussian + JPEG ≈10% of draws.
