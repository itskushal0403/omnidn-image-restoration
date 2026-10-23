# Research notes 4 — Efficient architectures, distillation, and how FLOPs are counted

*Compiled 2026-10-19. fvcore (0.1.5.post20221221) and ptflops (0.7.5) sources were read **and** measured op-by-op.*

## 1. How the FLOP counters count (measured)

**Both tools count MACs** (one multiply-add = 1). fvcore says so explicitly; ptflops returns MACs.

| Op | fvcore | ptflops `pytorch` backend (**actual code default**) | ptflops `aten` |
|---|---|---|---|
| nn.Conv2d (dense / depthwise / dilated / strided) | exact, no bias | exact + bias | exact + bias |
| `F.conv2d` (functional) | counted | **0** | counted |
| nn.Linear | counted | counted | counted |
| `x @ y`, `torch.einsum` | counted | **0** | counted |
| `torch.matmul`, `bmm` | counted | counted | counted |
| `F.scaled_dot_product_attention` | **0** | 0 | counted (CPU) |
| elementwise `*`, `+`, sigmoid, tanh | 0 | 0 | 0 |
| GELU / ReLU modules | 0 | 1 / elem | 0 |
| nn.LayerNorm / GroupNorm | 5 / elem | 1–2 / elem | 0 |
| AdaptiveAvgPool2d | 1 / elem | 1–2 / elem | 0 |
| `x.mean()`, PixelShuffle / Unshuffle, cat, pad, roll | 0 | 0 | 0 |
| bilinear interpolate | 4 / out elem | 1 / out elem | 0 |
| `torch.fft.rfft2` / `irfft2` | **0** | 0 | 0 |
| custom `autograd.Function` | 0 | 0 | 0 |

**Implications:**
- Convs, linears and matmuls dominate the count.
- Elementwise gating (SimpleGate), SCA multiply and parameter-free attention are free — *and genuinely cheap*.
- **FFT is free in the counter but not in reality.** Honest cost ≈ 5·C·HW·log₂(HW) FLOPs per FFT pair — at C=64, 256² that is about
  one 64→64 1×1 conv.
- **Fair-play policy for this project:** we do not hide computation in uncounted ops (no custom autograd wrappers around convs, no
  input-dependent routing). Any FFT usage is disclosed with its real cost estimate in the report.

### NAFNet cost at 1×3×256×256 (measured, fvcore)

| Width | SIDD config (36 blocks) | GoPro config (28 bottleneck blocks) |
|---|---|---|
| 8 | 1.10 G (1.9M params) | 1.07 G |
| 16 | 4.14 G | 4.09 G (4.35M params) |
| 24 | 9.13 G | — |
| 32 | 16.06 G | 15.97 G |
| 64 | 63.3 G | 63.1 G |

**Rule of thumb:** a NAFBlock costs ≈ **6c²·HW MACs at every U-Net level** (channels ×2, pixels ÷4). A block at 1/16 resolution costs the
same as one at full resolution but has a 16× larger receptive field → put most blocks deep.

## 2. Efficient architectures: quality vs GMACs (at 256²)

| Model | GMACs | SIDD | GoPro | Note |
|---|---|---|---|---|
| CGNet-tiny ablation | 0.45 | 39.32 | – | short schedule |
| NAFNet w16 (student, DCKD) | ~2.5 | – | 31.17 → 31.43 with DCKD | |
| AIM'25 Efficient Deblur winner (NAFNet + rep + EMA + global module) | ~5.6 | – | +0.96 dB over NAFNet-C16 | |
| FSM-Net (NTIRE'26 #2, FFT attention) | ~4.5 | – | – | FFT attn +0.18, global bottleneck +0.21 |
| NAFNet w32 | 16 | 39.96 | 32.85 | |
| NAFNet with only 9 blocks, same 16 G | 16 | 39.78 | 31.79 | **depth matters for blur** |
| FNAFNet32 (+FFT) | 23 | – | 33.12 | +0.17 dB over NAFNet32 |
| NAFNet-64 / CGNet | 63 / 62 | 40.30 / 40.39 | 33.71 / 33.77 | |
| Restormer | 140 | 40.02 | 32.92 | |
| FFTformer / LoFormer / EVSSM | 126–131 | – | 34.1–34.5 | heavier, slower to train |

**Verdict:**
- In the 1–20 G range, **NAFNet / CGNet-style conv U-Nets give the best PSNR per MAC**; no 2024–2026 Mamba / transformer beats them per MAC.
- Denoising saturates early (0.45 G → 39.3, 16 G → 40.0 on SIDD).
- Deblurring is capacity- and receptive-field-hungry.

## 3. Efficient-challenge lessons

- **NTIRE'24 ESR winner SPAN** — parameter-free attention.
- **NTIRE'25 ESR winner DSCLoRA** — ConvLoRA + affinity KD; only +0.006–0.037 dB over SPAN-Tiny. A 52-ch teacher was *worse* than a
  28-ch one (capacity gap).
- **NTIRE'26 ESR** — pruning + KD; staged patch growth; L1→L2; FFT loss; EMA.
- **Re-parameterization**: ECB ≈ **+0.1 dB**; RepVGG / ACB / DBB < 0.05 dB.
- **AIM'25 Efficient Deblurring winner** — NAFNet + re-param + EMA + local-pooled SCA + global bottleneck module, 4 progressive stages.

## 4. Knowledge distillation

- **DCKD (AAAI'25)** — NAFNet-32 → 16-ch: output KD +0.04–0.09 dB; contrastive KD +0.21–0.26 dB.
- **Mobile-AI'26 denoising (scarce data)** — KD closed a 1.63 dB gap to 0.05 dB (100·L2(GT) + 900·L2(teacher) + 50·L1).
- **With unlimited synthetic pairs, expect ~0.1–0.3 dB.** Gains are larger for tiny students, deblurring, and domains with little data
  (medical / low-light).
- **Recipe:**
  - Teacher ~10–30× the student MACs, same family.
  - Student loss = Charbonnier(s, GT) + λ·L1(s, T) (λ≈0.5–1) + 0.01–0.1·affinity loss on 2–3 levels.
  - EMA, L2 fine-tune at the end.

## 5. Cheap large receptive field (for motion / defocus blur)

- **Deep, narrow U-Net**: 4 downsamplings (16×16 bottleneck), most blocks at 1/8–1/16 resolution.
- **Global attention at the 16×16 bottleneck**: 256 tokens, ~0.1 G total.
- **FFT global filter** (Res-FFT / FFC / GFNet; valid at fixed 256²).
- **CGNet GCE** (cascaded stride=kernel depthwise convs + nearest upsample).
- **Large-kernel depthwise convs** (≤7 for cuDNN speed).
- **Fixed-256 specifics**:
  - **CoordConv channels** — chromatic aberration and lens blur vary with radius.
  - **Training at 256²** — makes global pooling / FFT statistics match test time.

## 6. Training-speed notes (A30 / A6000)

- **Small nets are launch- / bandwidth-bound**: `torch.compile`, channels_last, bf16 AMP, large batches.
- **Precision**: DISTS loss and FFT in fp32.
- **Re-param branches** double training cost → add them for the final runs only.

## 7. Recommendations from this sweep

| Role | Proposal | GMACs |
|---|---|---|
| Teacher | NAFNet / CGNet width-64 U-Net + FFT blocks at deep levels + global-attention bottleneck | 60–100 |
| Student | Width-16 deep-skewed U-Net (enc [1,1,2,6], ~10 middle, dec [1,1,1,1]) + local-pooled SCA + FFT / global block + ECB re-param + KD + EMA | 2–5 |
| Ultra-small | Width-8, deep-skewed, optional 2× pixel-unshuffle stem, heavy KD | <1 |

**Strategy:** build a **Pareto family** (~0.3 / 1 / 4 G) with the same recipe; pick the submission point from validation.
