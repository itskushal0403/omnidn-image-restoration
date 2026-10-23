"""OmniDN — a small, efficient U-Net for blind image restoration.

Built on NAFNet (Chen et al., ECCV 2022), the family with the best PSNR per counted MAC in the 1–20 GMAC range
(docs/research/04). Changes for this competition:

1. **Deep-skewed block allocation.** A NAFBlock costs ~6·c²·H·W MACs at *every* U-Net level (channels ×2, pixels ÷4),
   but a block at 1/16 resolution sees 16× more context. So most blocks sit at the bottom, which helps motion and
   defocus blur.
2. **Global attention at the 16×16 bottleneck.** 256 tokens; full self-attention there is cheap (~0.1 GMAC) and gives
   every pixel an image-wide receptive field. Matmuls use ``torch.matmul`` so every FLOP counter sees them.
3. **CoordConv input.** Two coordinate channels: chromatic aberration and lens blur depend on the distance from the
   image centre.
4. **Plain-op LayerNorm** (no custom autograd function), which ``torch.compile`` can fuse.

All compute-heavy ops (convolutions, attention matmuls) are standard modules / torch.matmul that fvcore and ptflops
count; the uncounted remainder is only cheap elementwise work (norms, gating, activations; ~4% extra in real FLOPs).
Nothing is hidden from the counters (fair-play policy, docs/01_plan.md).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW tensors."""

    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        # statistics in fp32: under model.half() (x - mu)^2 overflows fp16 (|x - mu| reaches ~1e4) and the output
        # collapses; in fp32 this is an exact no-op
        dt = x.dtype
        x32 = x.float()
        mu = x32.mean(1, keepdim=True)
        var = (x32 - mu).pow(2).mean(1, keepdim=True)
        x = ((x32 - mu) * torch.rsqrt(var + self.eps)).to(dt)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """NAFNet block: [LN → 1×1 → DW3×3 → SimpleGate → SCA → 1×1] + [LN → 1×1 → SimpleGate → 1×1]."""

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        ffn = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.conv2(self.conv1(self.norm1(inp)))
        x = self.sg(x)
        x = x * self.sca(x)
        y = inp + self.conv3(x) * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class GlobalAttentionBlock(nn.Module):
    """Multi-head self-attention over all tokens of a (small) feature map + gated depthwise FFN."""

    def __init__(self, c: int, heads: int = 4, ffn_expand: int = 2):
        super().__init__()
        self.heads = heads
        self.scale = (c // heads) ** -0.5
        self.norm1 = LayerNorm2d(c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)
        ffn = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.ffn_in = nn.Conv2d(c, ffn, 1)
        self.ffn_dw = nn.Conv2d(ffn, ffn, 3, padding=1, groups=ffn)
        self.ffn_out = nn.Conv2d(ffn // 2, c, 1)
        self.sg = SimpleGate()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        b, c, h, w = inp.shape
        q, k, v = self.qkv(self.norm1(inp)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale            # b, heads, N, N
        attn = attn.float().softmax(dim=-1).to(v.dtype)
        out = torch.matmul(v, attn.transpose(-2, -1)).reshape(b, c, h, w)   # b, heads, d, N
        y = inp + self.proj(out) * self.beta
        x = self.ffn_out(self.sg(self.ffn_dw(self.ffn_in(self.norm2(y)))))
        return y + x * self.gamma


class OmniDN(nn.Module):
    def __init__(
        self,
        width: int = 16,
        enc_blocks=(1, 1, 2, 4),
        mid_blocks: int = 10,
        attn_blocks: int = 2,
        dec_blocks=(1, 1, 1, 1),
        attn_heads: int = 4,
        coord_conv: bool = True,
        in_ch: int = 3,
        output_uint8_levels: bool = False,
    ):
        super().__init__()
        self.coord_conv = coord_conv
        # submission contract: clamp to [0, 1] and snap to the 256 uint8 levels inside forward(), so that ANY
        # float->uint8 conversion (rounding, truncation, PIL) yields exactly the same image (0 FLOPs, no parameters)
        self.output_uint8_levels = output_uint8_levels
        self.intro = nn.Conv2d(in_ch + (2 if coord_conv else 0), width, 3, padding=1)
        self.ending = nn.Conv2d(width, in_ch, 3, padding=1)

        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        c = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c *= 2

        # bottleneck: NAF blocks with attention blocks interleaved evenly
        mid: list[nn.Module] = [NAFBlock(c) for _ in range(mid_blocks)]
        if attn_blocks:
            step = max(1, len(mid) // attn_blocks)
            for i in range(attn_blocks):
                mid.insert(min(len(mid), (i + 1) * step + i), GlobalAttentionBlock(c, heads=attn_heads))
        self.middle = nn.Sequential(*mid)

        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.pad_to = 2 ** len(enc_blocks)

    @staticmethod
    def _coord(x):
        """Two coordinate channels in [-1, 1] (CoordConv); recomputed per call (negligible cost, trace-friendly)."""
        h, w = x.shape[-2:]
        ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        return torch.stack(torch.meshgrid(ys, xs, indexing="ij"))[None].expand(x.shape[0], -1, -1, -1)

    def forward(self, x):
        """x: float RGB (N, 3, H, W) in [0, 1] -> restored image of the same shape."""
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.pad_to, (-w) % self.pad_to   # no-op at the competition size 256×256
        x0 = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        x = torch.cat([x0, self._coord(x0)], 1) if self.coord_conv else x0
        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = dec(up(x) + skip)
        x = self.ending(x) + x0
        x = x[..., :h, :w]
        if self.output_uint8_levels:
            x = torch.round(x.clamp(0.0, 1.0) * 255.0) / 255.0
        return x


# ------------------------------------------------------------------------------------------- presets
PRESETS = {
    # name: kwargs   (GMACs at 1x3x256x256 are measured by scripts/count_flops.py and recorded in the README)
    "omnidn_xs": dict(width=12, enc_blocks=(1, 1, 2, 4), mid_blocks=10, attn_blocks=2, dec_blocks=(1, 1, 1, 1)),
    "omnidn_s": dict(width=16, enc_blocks=(1, 1, 2, 4), mid_blocks=10, attn_blocks=2, dec_blocks=(1, 1, 1, 1)),
    "omnidn_m": dict(width=32, enc_blocks=(1, 1, 2, 4), mid_blocks=10, attn_blocks=2, dec_blocks=(1, 1, 1, 1)),
}


def build_model(name: str = "omnidn_s", **overrides) -> OmniDN:
    cfg = dict(PRESETS[name])
    cfg.update(overrides)
    return OmniDN(**cfg)
