# Mini Competition — a small generative model for blind image restoration

Goal: one model that restores **any** corrupted 256×256 RGB image (noise of many kinds, blur, JPEG, chromatic
aberration, mixtures, unknown corruptions) across natural, low-light and medical images.

It is ranked on four equally weighted axes: **FLOPs ↓, PSNR ↑, SSIM ↑, DISTS ↓**.

Our answer is **OmniDN-S**, an efficient NAFNet-style U-Net with a global-attention bottleneck (**2.963 GFLOPs**,
6.66 M params). It is trained on 75,602 clean samples (60,948 images + 14,654 native crops) from 21 public sources
(natural, low-light, medical), corrupted on the fly by a 21-operator degradation engine. It is then fine-tuned on losses
that match the competition metrics (PSNR-loss, exact skimage SSIM, official DISTS), and its final weights blend two
DISTS-weighted fine-tunes.

## Final result (submitted: `submission/` = `submission_versions/v3_final`)

| Model | GFLOPs @256² | Proxy test (600 pairs) PSNR / SSIM / DISTS | Unseen corruptions (140 pairs) |
|---|---|---|---|
| Identity (input returned) | 0 | 25.43 / 0.618 / 0.283 | 25.59 / 0.592 / 0.289 |
| SCUNet, public pretrained | 79.0 | 30.03 / 0.807 / 0.230 | 30.25 / 0.806 / 0.230 |
| **OmniDN-S (submitted)** | **2.96** | **33.92 / 0.9032 / 0.1168** | **31.38 / 0.8568 / 0.1664** |

These are our own held-out sets (organiser-style evaluation of the submission folder), not the hidden test.
Per domain on the proxy set: natural 30.25 / low-light 38.34 / medical 36.59 dB.

**Deliverables:**
- Technical report: [`docs/report/technical_report.pdf`](docs/report/technical_report.pdf) (2 pages; source `.md`,
  builder `scripts/build_report_pdf.py`).
- Code + weights: [`submission/`](submission/) (see its README).
- Upload-ready zip: `deliverables/`.

| Read this | for |
|---|---|
| [`docs/01_plan.md`](docs/01_plan.md) | the design and *why* each decision was made |
| [`docs/research/`](docs/research/) | literature notes behind the design (4 topics) |
| [`docs/02_experiment_log.md`](docs/02_experiment_log.md) | every run: what, why, result |
| [`docs/figures/`](docs/figures/) | degradation gallery, training curves, visual results |
| [`docs/report/`](docs/report/) | the 1–2 page technical report (PDF + Markdown source + figure) |

---

## End-to-end flow

```
 raw public datasets ──► scripts/data/prep_*.py ──► data/clean/<domain>/<source>/*.png + data/manifests/*.csv
                                                         │
                       scripts/build_proxy_test.py ◄─────┤ (held-out "val" rows only)
                       data/val/proxy_test_v1  (530 pairs, TA-style degradations)
                       data/val/proxy_test_ood_v1 (140 pairs, unseen corruption families)
                                                         │
 src/dncomp/data/degradations.py ──► TrainDataset ──► src/dncomp/train.py ──► checkpoints/<run>/
   (21 ops, random chains, CutMix)     (on-the-fly pairs)  (stage 1: fidelity)      │
                                                                                     ▼
                                       configs/*_ft_*.yaml (stage 2: PSNR + exact-SSIM [+ DISTS])
                                                                                     │
                       scripts/interpolate.py / dncomp.evaluate ──► pick the final weights
                                                                                     │
                       scripts/package_submission.py ──► submission/ (model.py, weights, inference.py)
```

## Repository layout

| Path | Contents |
|---|---|
| `src/dncomp/data/prep_utils.py` | uniform clean-image store (resize rules, manifest format) |
| `src/dncomp/data/degradations.py` | **the degradation engine** (21 ops, composer, CutMix) |
| `src/dncomp/data/datasets.py` | training stream and paired evaluation dataset |
| `src/dncomp/models/omnidn.py` | the OmniDN architecture + size presets |
| `src/dncomp/losses/losses.py` | Charbonnier, FFT, PSNR-loss, skimage-exact SSIM, DISTS |
| `src/dncomp/metrics/competition_metrics.py` | the organisers' metrics (skimage / DISTS / fvcore / ptflops) |
| `src/dncomp/train.py` | training loop (bf16, EMA, wall-clock cosine, auto-resume) |
| `src/dncomp/evaluate.py` | exact-metric evaluation, per domain / degradation family |
| `scripts/data/` | one download+convert script per dataset |
| `scripts/slurm/` | SLURM job scripts |
| `submission/` | the deliverable: `model.py`, `weights/omnidn.pth`, `inference.py`, `check_flops.py` |
| `data/`, `checkpoints/`, `runs/`, `logs/` | symlinks to `/scratch/m25csa016/denoising_competition/` (heavy files) |

## How to run each step

```bash
PY=/scratch/m25csa016/envs/denoise/bin/python        # conda env (PyTorch 2.7.1 + CUDA 12.6)
cd /csehome/m25csa016/denoising_competition

# 1. data (each script downloads, converts, writes a manifest; already done)
$PY scripts/data/prep_div2k.py            # ... one per source

# 2. proxy test set + degradation gallery
$PY scripts/build_proxy_test.py
$PY scripts/make_degradation_gallery.py

# 3. training (one GPU per job; auto-resumes if re-submitted)
sbatch --job-name=dn-S scripts/slurm/train.sbatch configs/omnidn_s.yaml
tensorboard --logdir runs/                # curves; or read runs/<run>/val_metrics.csv

# 4. evaluation with the exact competition metrics
srun -p test --gres=gpu:1 $PY -m dncomp.evaluate --ckpt checkpoints/<run>/ema_best.pth --dists

# 5. package the submission and verify FLOPs / I/O contract
$PY scripts/package_submission.py --ckpt checkpoints/<run>/<final>.pth
cd submission && $PY inference.py --input noisy.png --output restored.png
```

## Model sizes (measured at 1×3×256×256)

| Preset | fvcore (G) | ptflops (GMac) | Params |
|---|---|---|---|
| omnidn_xs | 1.71 | 1.75 | 3.8 M |
| **omnidn_s** | **2.96** | **3.01** | 6.7 M |
| omnidn_m | 11.36 | 11.47 | 26.4 M |
| *ref: NAFNet-32 / Restormer* | *16 / 140* | | |

## Compute

- **GPUs:** NVIDIA A30 24 GB (cn01–05) and RTX A6000 48 GB (cn07). The `mtech` QoS allows 2 GPUs per user at a time.
- **Software:** conda env `/scratch/m25csa016/envs/denoise`, Python 3.11, PyTorch 2.7.1+cu126.
