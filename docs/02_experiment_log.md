# Experiment log

Every run, why it was run, and what it showed. Times are IST. Metrics are on `proxy_test_v1` unless stated otherwise
(PSNR / SSIM on uint8 via skimage-exact code; DISTS via the official package).

## 2026-10-19

### 16:35 — smoke test (`smoke_s`, deleted afterwards)
- **What:** OmniDN-S, 7 min on 1 A30; at that point only the chest X-ray subset was available.
- **Result:**
  - ~290–310 img/s at batch 32 (bf16, `torch.compile` ≈ 3.5 min of start-up).
  - Proxy-test PSNR 30.8 → 33.1 dB (X-ray only).
  - All machinery works: EMA, validation, DISTS, checkpoints.
- **Bug found:** the DataLoader stopped after one virtual epoch (100k samples). Fixed: the iterator is restarted on
  `StopIteration`.

### 17:20 — stage 1 launched: `omnidn_s_stage1` (job 44013) and `omnidn_m_stage1` (job 44014)
- **Why two sizes:** we cannot know the competitors' FLOPs in advance. Training S (3.0 G) and M (11.4 G) gives two
  real points on the FLOPs/quality curve to choose from at the end.
- **Setup:**
  - ~71k training images at launch.
  - Loss: Charbonnier + 0.05·FFT.
  - AdamW (S: lr 1e-3, M: 6e-4), wall-clock cosine schedule over 19.5 h, EMA 0.999, batch 32.
- **Plan:** restart (auto-resume) once the remaining sources (LOL-v2, BreaKHis, more CT) finish, so they enter the
  training pool.

### 17:34 — data fix + clean restart: `omnidn_s_stage1` (job 44016) and `omnidn_m_stage1` (job 44017)
- **Problem found by the data agent:** LOL v1 is a pixel-identical subset of LOLv2-real, and their official splits
  disagree (86 LOL v1 *train* images sit in LOLv2-real *val*). Training on one and validating on the other would leak.
- **Fix:**
  - `lol_v1` excluded everywhere (`data/manifests_excluded/`).
  - New clean **`proxy_test_v2`**: 600 pairs from 300 held-out images (natural 140 / low-light 50 / medical 110; 23
    sources). OOD set: 140 pairs.
  - Both runs restarted **from scratch** (only ~15 min were lost) on the full pool.
- **Training pool:** ~92k images (natural 35.6k incl. LSDIR 20k; low-light ~3k dark images; medical ~44k from 14
  sources).
- **Budget:** 20 h wall-clock cosine schedule. Throughput (A30): S ≈ 304 img/s, M ≈ 196 img/s.

### Reference baselines on proxy_test_v2 (`runs/baselines/`)

| Model | GFLOPs @256² | PSNR | SSIM | DISTS | OOD PSNR / SSIM / DISTS |
|---|---|---|---|---|---|
| Identity (input returned) | 0 | 25.43 | 0.618 | 0.283 | 25.59 / 0.592 / 0.289 |
| SCUNet pretrained (`scunet_color_real_psnr`, public blind denoiser) | 79.0 | 30.03 | 0.807 | 0.230 | 30.25 / 0.806 / 0.230 |

- **Identity, hardest specs:** s&p 0.10 (14.4 dB), Gaussian σ=50 (15.5 dB), motion blur + s&p, motion blur + Gaussian noise.
- **SCUNet** stands in for a strong "download a pretrained denoiser" competitor; it is trained on noise only (no blur, CA or
  impulse noise).

### Gray-consistency post-fix — rejected
- **Idea:** snap near-gray outputs to exact gray (0 FLOPs).
- **Result** on an early S checkpoint: best threshold +0.005 dB.
- **Why it fails:** the model already outputs near-gray images for gray GT (median chroma 0.004). Dark low-light colour
  images have similarly low chroma (0.009), so any useful threshold starts graying them.
- **Decision:** not used (script kept: `scripts/study_gray_consistency.py`).

### 21:40 — stage-1 progress after 4 h (full-set numbers at 21:55 for S `ema_best`)

| Model | GMACs | PSNR | SSIM | DISTS | OOD PSNR / SSIM / DISTS |
|---|---|---|---|---|---|
| Identity | 0 | 25.43 | 0.618 | 0.283 | 25.59 / 0.592 / 0.289 |
| SCUNet pretrained | 79 | 30.03 | 0.807 | 0.230 | 30.25 / 0.806 / 0.230 |
| **OmniDN-S @ 4 h** | **3.0** | **33.16** | **0.893** | **0.163** | 31.27 / 0.854 / 0.198 |

(Val-subset curves: `docs/figures/training_curves.png`; M is ~+0.4 dB above S at equal time.)

**Per-degradation analysis** (`runs/baselines/S_4h_*.csv`):
- **Where S dominates:**
  - impulse noise — s&p 0.05: 40.0 vs SCUNet 26.3
  - chromatic aberration — 40.9 vs 28.2
  - blur and blur + noise mixtures: +2–4 dB
- **Where the 26× larger SCUNet still matches or slightly beats S** (pure Gaussian noise / JPEG):
  - JPEG q30: 32.1 vs 32.6
  - Gaussian σ=50: 30.2 vs 30.6
  - Gaussian σ=25 + JPEG: 31.5 vs 32.2
- **Hardest family:** motion blur (≈28–29 dB).
- **OOD:** zoom blur is not restored at all (24.4 vs 24.5 identity); everything else improves.

**Decision → stage-2 degradation mix:** emphasise the families listed in the task PDF (Gaussian noise 0.25, JPEG 0.18,
motion blur 0.09, …) and halve the "extra" families (banding, posterize, webp, …) — fine-tuning on the target
distribution. Implemented as `DegradationComposer(weights=...)` in the `*_ft_*.yaml` configs.

### 21:50 — stage-2 automation
- **Pipeline** (`scripts/slurm/finetune_pipeline.sbatch`), one job per size: 2a fidelity fine-tune → 2b perceptual fine-tune
  (from 2a) → weight-interpolation sweep (α ∈ {0, 0.2, …, 1}) on both proxy sets.
- **Watcher** (`scripts/slurm/stage2_watcher.sh`, nohup on the login node) submits each pipeline when the needed stage-1
  checkpoints exist (the QoS allows only 3 submitted jobs).
- **S stage 2a distils from M** (`teacher:` in the config).
  - Distillation loss = MSE(student, teacher) / MSE(student, GT).detach(), so its gradient scale matches PSNRLoss.
  - The first log-MSE version made the KD gradient ~10× too strong (S already matches M at ~45 dB).

### 22:20–00:30 — submission packaging verified end-to-end; deadline safety net
- **v0 package** (`scripts/package_submission.py`, S `ema_best` at 22:20):
  - FLOPs at 1×3×256×256: fvcore **2.963 G**, ptflops-pytorch **3.014 GMac**, ptflops-aten **3.003 GMac**; 6.66 M params.
  - I/O contract OK.
- **`scripts/verify_submission.py`** runs only the `submission/` folder, organiser-style: PNG → Restorer → uint8 → skimage
  PSNR / SSIM + official DISTS.
  - Result: **33.2451 dB / 0.8941**, *identical* to our fp32 evaluator on the same weights.
  - bf16 evaluation underestimates by only 0.006 dB; the final interpolation sweep now runs in fp32 anyway.
- **Deadline ambiguity** ("tomorrow 11:59" may be AM, before stage 1 ends at ~13:40):
  `scripts/slurm/early_submission.sh` (nohup) will, at 07:00, snapshot S stage 1 → 45-min metric-aligned fine-tune on
  the 1-h `test` partition → package into `submission/` + `submission_versions/v1_early/` → verify on both proxy sets.
  Running jobs are not touched.

## 2026-10-20

### 00:35–01:17 — Ablation A: DISTS loss weight (S, 43-min fine-tunes on the `test` partition)
- **Setup:**
  - All three runs start from the same S stage-1 snapshot (22:20, scores 33.25 / 0.894 / 0.161).
  - Loss = PSNR-loss + 30·(1−SSIM) + λ·DISTS, with the stage-2 degradation mix.
  - Full proxy_test_v2, fp32.

| λ_DISTS | PSNR | SSIM | DISTS | Δ vs snapshot |
|---|---|---|---|---|
| 0 (fidelity only) | **33.43** | **0.898** | 0.157 | +0.18 dB, +0.004 SSIM |
| 40 | 33.16 | 0.892 | 0.124 | −0.09 dB, −0.002, **−0.037 DISTS** |
| 120 | 32.87 | 0.886 | **0.117** | −0.37 dB, −0.008, −0.044 |

- **Conclusions:**
  - The metric-aligned fidelity fine-tune helps by itself.
  - DISTS responds strongly to a modest weight (0.157 → 0.124 at λ=40), with diminishing returns beyond (λ=120 buys only
    0.007 more for −0.3 dB).
  - The final perceptual weight should be in the 40–50 range, with the operating point picked by weight interpolation
    (sweep running).

### 01:19 — Ablation B: weight interpolation between the λ=0 and λ=40 fine-tunes (`runs/interp_abl_d0_d40.csv`)

| α | PSNR | SSIM | DISTS | OOD PSNR / DISTS |
|---|---|---|---|---|
| 0 | 33.429 | 0.8980 | 0.1570 | 31.308 / 0.1963 |
| 0.35 | 33.396 | 0.8975 | 0.1478 | 31.290 / 0.1890 |
| 0.5 | 33.365 | 0.8969 | 0.1429 | 31.271 / 0.1851 |
| 0.65 | 33.321 | 0.8960 | 0.1373 | 31.244 / 0.1808 |
| 0.8 | 33.263 | 0.8948 | 0.1313 | 31.209 / 0.1761 |
| 1.0 | 33.159 | 0.8924 | 0.1242 | 31.149 / 0.1701 |

- **Shape of the trade-off:** PSNR / SSIM cost is convex in α (tiny at first) while the DISTS gain is ~linear, so a middle α buys
  most of the perceptual gain cheaply.
  - α=0.5: −0.06 dB, −0.001 SSIM for −0.014 DISTS.
  - Same shape on the OOD set.
- **λ=120 as the endpoint:** interpolating towards it is slightly *worse* at equal PSNR
  (33.36 / 0.1455 vs 33.365 / 0.1429). → **Final perceptual configs set to λ_DISTS = 40.**

### 01:20–02:03 — Ablations C/D (from the same 22:20 S snapshot, 43 min each, `test` partition)

| Variant | PSNR | SSIM | DISTS | Note |
|---|---|---|---|---|
| fidelity fine-tune (ref., λ_SSIM=30) | **33.429** | 0.8980 | 0.157 | 22.2k iters |
| C: + distillation from M (rel-MSE, w=0.5) | 33.381 | 0.8972 | 0.158 | teacher forward cuts throughput 2.4× (9.2k iters) |
| D: λ_SSIM = 100 | 33.369 | **0.8989** | 0.157 | +0.0009 SSIM for −0.06 dB |

- **Decision C:** drop distillation (same wall-clock, lower PSNR). S stage 2 = plain fidelity fine-tune → perceptual.
- **Decision D:** keep λ_SSIM = 30 (balanced; λ=100 trades PSNR rank for a very small SSIM gain).

### 07:00–07:47 — early safety submission `submission_versions/v1_early` (automatic)
- **Built from:** S stage-1 snapshot at iter 446,880 (13.3 h) → 45-min fidelity fine-tune → packaged.
- **FLOPs:** fvcore 2.963 G, ptflops 3.01 GMac.
- **Verified end-to-end** with the submission folder only (skimage + official DISTS):

| Set | PSNR | SSIM | DISTS | natural / low-light / medical PSNR |
|---|---|---|---|---|
| proxy_test_v2 | **33.93** | **0.905** | **0.148** | 30.26 / 38.39 / 36.58 |
| proxy_test_ood_v2 | 31.53 | 0.860 | 0.191 | 27.33 / 37.23 / 32.91 |

### 11:56 — process note
- **What happened:** the stage-2 watcher was restarted to apply decision C. A `pkill` pattern matched its own shell, so the new
  watcher only started at 11:56.
- **Impact:** none — stage 1 is still running (ends ~13:40).

### 12:10 — stress test at extreme severities (`scripts/stress_test.py`, v1_early, 60 proxy GT images × 10 specs)

| Spec | input PSNR / SSIM | output PSNR / SSIM | gain |
|---|---|---|---|
| motion k15 + Gaussian σ25 | 18.90 / 0.167 | 26.94 / 0.722 | +8.0 dB |
| motion k25 | 23.59 / 0.593 | 27.29 / 0.742 | +3.7 dB |
| JPEG q5 | 25.40 / 0.659 | 27.60 / 0.762 | +2.2 dB |
| Gaussian σ75 | 12.43 / 0.115 | 28.59 / 0.789 | +16.2 dB |
| Gaussian blur σ4 | 24.66 / 0.621 | 28.67 / 0.787 | +4.0 dB |
| Poisson peak 5 | 13.67 / 0.198 | 29.18 / 0.816 | +15.5 dB |
| Gaussian σ50 + JPEG q20 | 20.16 / 0.338 | 29.28 / 0.811 | +9.1 dB |
| speckle var 0.2 | 17.27 / 0.413 | 31.17 / 0.872 | +13.9 dB |
| s&p 0.25 | 10.60 / 0.097 | 32.84 / 0.913 | +22.2 dB |
| CA roll 4 + s&p 0.05 | 15.89 / 0.251 | 34.02 / 0.945 | +18.1 dB |

- **Result:** no failure mode, even far outside the typical severities.
- **Weakest cases:** long motion blur and JPEG q5 — information-destroying degradations, already up-weighted in the stage-2 mix.

### 13:34 — stage 1 finished (20 h each; S 669,511 iters, M 435,000 iters) — full-set evaluation (fp32, exact metrics)

| Model | GMACs | proxy PSNR / SSIM / DISTS | OOD PSNR / SSIM / DISTS | proxy PSNR natural / low-light / medical |
|---|---|---|---|---|
| S stage-1 final | 3.0 | 33.96 / 0.904 / 0.150 | 31.54 / 0.860 / 0.192 | 30.30 / 38.41 / 36.61 |
| M stage-1 final | 11.4 | 34.46 / 0.910 / 0.141 | 31.67 / 0.861 / 0.186 | 30.82 / 38.85 / 37.10 |
| v1_early (S @13.3 h + 45-min ft) | 3.0 | 33.93 / 0.905 / 0.148 | 31.53 / 0.860 / 0.191 | 30.26 / 38.39 / 36.58 |

- **M vs S:** M leads by +0.50 dB in-distribution but only **+0.13 dB on unseen corruption families** — extra capacity mostly
  buys in-distribution fit.
- **Fine-tune vs schedule:** the metric-aligned fine-tune (v1_early) recovers as much as the last 6.7 h of the stage-1
  schedule.
- **Stage-2 pipelines** (jobs 44123 / 44124) started automatically at 13:35.

### 16:37 — stage 2 finished for S and M (fidelity fine-tune → perceptual fine-tune λ_DISTS=40 → interpolation)
- **Stage 2a** (PSNR-loss + 30·(1−SSIM), listed-family degradation mix, 1.5 h): +0.08 dB, +0.002 SSIM, −0.003 DISTS for both
  models.
- **Stage 2b + weight interpolation** (`runs/interp_s.csv`, `runs/interp_m.csv`), proxy_test_v2:

| α | S PSNR / SSIM / DISTS | M PSNR / SSIM / DISTS |
|---|---|---|
| 0.0 | 34.044 / 0.9059 / 0.1465 | 34.560 / 0.9122 / 0.1378 |
| 0.2 | 34.044 / 0.9059 / 0.1409 | 34.555 / 0.9122 / 0.1319 |
| 0.4 | 34.023 / 0.9055 / 0.1342 | 34.530 / 0.9118 / 0.1247 |
| 0.6 | 33.980 / 0.9046 / 0.1263 | 34.485 / 0.9109 / 0.1163 |
| 0.8 | 33.912 / 0.9032 / 0.1177 | 34.417 / 0.9095 / 0.1072 |
| 1.0 | 33.811 / 0.9009 / 0.1103 | 34.320 / 0.9074 / 0.1000 |

- **OOD set:** same shape (S α0 → α1: 31.51 → 31.31 dB, DISTS 0.191 → 0.161).

### 16:45 — rank simulation (`scripts/rank_simulation.py`, `runs/rank_simulation_final.txt`)
- **Inputs:** candidates scored as 0.75·proxy + 0.25·OOD.
- **Bug found and fixed:** the competitors' SSIM / DISTS-vs-PSNR slope was taken from the two most different candidates, one of
  them perceptual → positive DISTS slope (nonsense). It is now taken from the two fidelity-only models (`--slope-from Sa0.0 Ma0.0`).
- **Mean summed rank** over field offsets Δ ∈ {0, 0.5, 1, 1.5, 2, 3} dB:

| Candidate | Mean summed rank |
|---|---|
| S α0.8 | 37.73 |
| S α0.6 | 38.14 |
| M α0.8 | 38.19 |
| M α0.6 | 38.36 |
| S α1.0 | 38.60 |
| … | … |
| M α0.0 | 42.05 |
| S α0.0 | 42.46 |

- **Robust finding:** a strong perceptual blend (α ≈ 0.6–0.8) is best in *every* scenario.
- **S vs M:** M wins if the field is within ~1 dB of us, S if it is ≥ 1.5 dB below.
- **Follow-ups** (running): an independent 4-lens review workflow, SSIM-variant robustness (`scripts/eval_candidates_protocols.py`),
  and direct λ_DISTS = 15 fine-tunes (S and M) to test whether training directly at the trade-off beats interpolation.

### 16:47 — protocol robustness: SSIM `gaussian_weights=True` vs skimage default (`runs/candidates_protocols.csv`)
- **Result:** every candidate scores ≈0.0055 lower with the Gaussian-window SSIM (e.g. S α0.6: 0.9046 → 0.8989;
  M α0.6: 0.9109 → 0.9053). Ordering and gaps between candidates are unchanged on both sets.
  → The choice is robust to which SSIM call the organisers use.
- **Consistency check:** the CPU skimage numbers equal our GPU evaluator's.
- **Saved:** all candidate interpolations in `checkpoints/candidates/{s,m}_a{0.0,0.4,0.6,0.8,1.0}.pth`.

### 17:00 — independent 4-lens decision review (workflow; analyses under `runs/decision_review/<lens>/`)

| Lens | Recommendation | Confidence | Main argument |
|---|---|---|---|
| Rank theory (150-scenario robustness grid, N = 20/40/80) | **S α0.8** | 0.72 | Lowest mean / max regret on expected rank-sum (0.90 / 5.9 vs M α0.8 2.99 / 10.4); S better in 72% of scenarios; P(top-3) 0.72 vs 0.64 |
| Devil's advocate (steelman M, low α, α=1) | **S α0.8** | 0.60 | M wins only if ≳30% of the field is within 0.25 dB of us or FLOPs are bimodal; P(1st) for S ≈ 2× M; α 0.7–0.9 best in 21/22 scenarios |
| Generalization risk (per-image bootstrap, shift-aware sim) | M α0.8 | 0.55 | On *final placement* M is slightly better unless the hidden set is mostly unseen families; S still best on rank-sum |
| Measurement / protocol | **S α0.8** | 0.60 | Counter choice / SSIM variant / DISTS-pipeline variants never flip the order; found 2 package-robustness bugs (below) |

- **Decision:** OmniDN-S, α = 0.8 (interpolation of `omnidn_s_ft_fid` and `omnidn_s_ft_perc` EMA weights).
- **α:** α = 0.8 is the minimax-regret choice in every lens.
- **Caveat:** M is preferable only if the competitor field is unexpectedly strong *and* few competitors sit in the 3–11 G
  FLOPs band.

### 17:05 — package hardening (both 0-FLOP, fp32-identical)
1. **`LayerNorm2d` statistics in fp32.** Under `model.half()` (x−μ)² overflowed fp16 and S collapsed to 13.1 dB.
   - Now: max deviation vs fp32 is 1 uint8 level (mean 0.06 levels).
2. **`output_uint8_levels=True` in the submitted CONFIG.** `forward()` clamps to [0,1] and snaps to the 256 uint8 levels.
   - Why: if the organisers call the network directly and convert with `(y*255).astype(uint8)` (truncation, no clamp),
     wrap-around previously cost ≈4.5 dB.
   - Now: rounding, truncation and PIL conversion all give the identical image.
- **Packaging:** `submission/` re-packaged from `checkpoints/candidates/s_a0.8.pth`.
  - FLOPs unchanged: fvcore 2.963 G, ptflops 3.014 / 3.003 GMac, 6.664 M params.
  - An adversarial package-review workflow (I/O contract, FLOP measurement, metric re-implementation, packaging) is running.

### 17:20–17:45 — adversarial package review (workflow, 12 agents) → fixes → final verification
- **No metric or FLOP defects:**
  - An independent from-scratch organiser-style evaluation reproduces 33.9116 / 0.9032 / 0.1176.
  - FLOP counts are exact and input-independent under fvcore, ptflops (both backends) and FlopCounterMode, on CPU and CUDA.
  - The package is bit-identical under torch 2.5 / 2.7 / 2.14; jit.trace, torch.export and ONNX export work.
- **Fixed (wrapper only, outputs bit-identical):**
  - `inference.py` loaded `model.py` with a bare import (failed for package / by-path imports and could load another
    team's `model.py`) → now loaded by path under a unique module name.
  - CLI: output-path handling, file-name preservation, collision detection, skips hidden or corrupt files.
  - Robust input conversion: palette, 16-bit, float overshoot, HxWx1, CHW.
  - TF32 off on GPU (CPU/GPU parity).
  - README + requirements.
- **Final verification of `submission/` = `submission_versions/v2_final`** (S, α=0.8), organiser-style, fp32:

| Set | PSNR | SSIM | DISTS | natural / low-light / medical PSNR |
|---|---|---|---|---|
| proxy_test_v2 (600) | **33.912** | **0.9032** | **0.1176** | 30.23 / 38.34 / 36.59 |
| proxy_test_ood_v2 (140) | 31.380 | 0.8569 | 0.1678 | 27.25 / 37.07 / 32.70 |

- **Stress test** (extreme severities): +2.1 to +22.4 dB on every spec, no failure mode.

### 18:11 — direct λ_DISTS=15 fine-tunes (S, M; 1.5 h from the fidelity fine-tune) vs weight interpolation
- **Direct λ=15 models:**

| Model | proxy PSNR / SSIM / DISTS | OOD PSNR / SSIM / DISTS |
|---|---|---|
| S λ15 | 33.971 / 0.9040 / 0.1213 | 31.422 / 0.8575 / 0.1703 |
| M λ15 | 34.462 / 0.9102 / 0.1091 | 31.603 / 0.8599 / 0.1644 |

- **vs interpolation:** at equal PSNR, direct training beats fid↔λ40 interpolation by ≈0.004 DISTS (SSIM −0.0004).
- **λ15 ↔ λ40 blends** (`runs/interp_s_p15_p40.csv`):

| Blend | proxy PSNR / SSIM / DISTS | OOD PSNR / SSIM / DISTS |
|---|---|---|
| 0.65·λ15 + 0.35·λ40 | 33.936 / 0.9034 / 0.1179 | 31.393 / 0.8570 / 0.1673 |
| **0.55·λ15 + 0.45·λ40** | **33.922 / 0.9032 / 0.1168** | **31.383 / 0.8568 / 0.1664** |
| previous choice: 0.2·fid + 0.8·λ40 | 33.912 / 0.9032 / 0.1177 | 31.381 / 0.8569 / 0.1679 |

- **Decision:** switch to 0.55·λ15 + 0.45·λ40. It Pareto-dominates the previous point on both sets (PSNR +0.010 / +0.002,
  DISTS −0.0009 / −0.0015, SSIM equal within 1e-4) at the same DISTS-leaning operating point.

### 18:35 — FINAL SUBMISSION v3 (`submission/` = `submission_versions/v3_final`), verified organiser-style (fp32)
- **Model:** OmniDN-S, weights = 0.55·`omnidn_s_ft_perc15` + 0.45·`omnidn_s_ft_perc` (EMA).
- **FLOPs:** fvcore 2.963 G, ptflops 3.014 / 3.003 GMac; 6.664 M params.

| Set | PSNR | SSIM | DISTS |
|---|---|---|---|
| proxy_test_v2 (600 pairs) | **33.922** | **0.9032** | **0.1168** |
| proxy_test_ood_v2 (140 pairs) | **31.383** | **0.8568** | **0.1664** |

- **Stress test:** +2.1 to +22.4 dB on every extreme spec.

### 18:45 — technical report (workflow: writer → 3 critics → reviser)
- **Output:** `docs/report/technical_report.pdf` (2 A4 pages); source `technical_report.md`, figure `fig_tradeoff.png`,
  builder `scripts/build_report_pdf.py`.
- **Fact-check:** every number checked against the repo. The fact-check critic caught that the first draft still
  described v2 (α=0.8); the reviser rewrote it around v3.
- **Count correction:** the true training-set size is 75,602 samples (60,948 images + 14,654 crops; the earlier "~90k /
  ~92k" in README and log was an over-estimate).

## Final state (2026-10-20, 19:00)
- **Submission:** `submission/` (= `submission_versions/v3_final`) — OmniDN-S, 2.963 GFLOPs.
- **Scores (organiser-style):**
  - proxy test: 33.92 / 0.9032 / 0.1168
  - unseen-corruption set: 31.38 / 0.8568 / 0.1664
- **Report:** `docs/report/technical_report.pdf`.
- **Upload-ready archive:** `deliverables/omnidn_s_submission.zip` (submission folder + report).
