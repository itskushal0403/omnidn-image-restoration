"""Inference for the submitted denoising model (OmniDN-S).

Input : one corrupted RGB image, 256×256×3, uint8 [0, 255] (numpy array, PIL image or file path). Gray, gray+alpha,
        RGBA, palette, float [0, 1] and 16-bit inputs are converted to RGB uint8 first. Arrays must be RGB order
        (if you read with cv2.imread, convert BGR->RGB).
Output: one restored RGB image, 256×256×3, **uint8 [0, 255]** (rounded, not truncated).

Python API
----------
    from inference import Restorer               # or: from submission.inference import Restorer
    r = Restorer()                               # loads weights/omnidn.pth once (GPU if available, else CPU)
    out_u8 = r(img_u8)                           # numpy HxWx3 uint8 -> numpy HxWx3 uint8

Raw network (e.g. for FLOP counting): model.build_model() maps a float RGB tensor (N, 3, 256, 256) in [0, 1] to a
float tensor in [0, 1] that already lies on the 256 uint8 levels (k / 255).

Command line
------------
    python inference.py --input noisy.png --output restored.png
    python inference.py --input noisy_dir/ --output restored_dir/

FLOPs: the network itself is `build_model()` from model.py; `python check_flops.py` measures it on a
(1, 3, 256, 256) tensor with fvcore and ptflops, exactly like the organisers.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
# load the sibling model.py by path under a unique name: works with `python inference.py`, `python -m pkg.inference`,
# `from pkg.inference import Restorer`, importlib-by-path, and cannot be shadowed by another module called "model"
_spec = importlib.util.spec_from_file_location("omnidn_submission_model", HERE / "model.py")
_model_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _model_mod
_spec.loader.exec_module(_model_mod)
build_model = _model_mod.build_model
DEFAULT_WEIGHTS = HERE / "weights" / "omnidn.pth"


class Restorer:
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda":  # exact fp32 math on GPU (TF32 would change ~2% of pixels by 1 level vs CPU)
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        self.model = build_model()
        state = torch.load(weights, map_location="cpu", weights_only=True)  # plain state_dict
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, img) -> np.ndarray:
        img = _as_rgb_uint8(img)
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        y = self.model(x)
        y = (y.clamp(0, 1) * 255.0).round().to(torch.uint8)     # float -> uint8 with ROUNDING
        return np.ascontiguousarray(y[0].permute(1, 2, 0).cpu().numpy())


def _as_rgb_uint8(img) -> np.ndarray:
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    if isinstance(img, Image.Image):                 # L, LA, P, RGBA, CMYK -> RGB; 16-bit modes go via numpy
        img = np.asarray(img) if img.mode.startswith("I") else img.convert("RGB")
    img = np.asarray(img)
    if img.ndim == 4 and img.shape[0] == 1:          # 1xHxWxC
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 2, 3, 4):
        img = img.transpose(1, 2, 0)                  # CxHxW -> HxWxC
    if img.dtype != np.uint8:
        kind, a = img.dtype.kind, img.astype(np.float32)
        if kind == "f" and a.max() <= 1.5:            # float [0,1] (tolerates small overshoot)
            a = a * 255.0
        elif kind in "ui" and img.dtype.itemsize >= 2 and a.max() > 255:   # 16-bit data
            a = a / 257.0
        img = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[2] in (1, 2):                        # gray / gray+alpha
        img = np.repeat(img[..., :1], 3, axis=2)
    elif img.shape[2] == 4:
        img = img[..., :3]
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {img.shape}")
    return np.array(img, dtype=np.uint8, copy=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--output", required=True, help="output file or directory")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    r = Restorer(args.weights, args.device)
    src, dst = Path(args.input), Path(args.output)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in exts)
        dst.mkdir(parents=True, exist_ok=True)
        # keep the input file name for PNGs (incl. upper-case .PNG); other formats -> <stem>.png (lossless)
        outs = [dst / (f.name if f.suffix.lower() == ".png" else f.stem + ".png") for f in files]
        if len(set(outs)) != len(outs):
            raise SystemExit("two input files map to the same output name (e.g. a.png and a.jpg)")
    else:
        files = [src]
        if dst.is_dir() or args.output.endswith(("/", "\\")) or dst.suffix == "":
            dst = dst / (src.stem + ".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        outs = [dst]
    failed = []
    for f, o in zip(files, outs):
        try:
            out = r(Image.open(f))
            fmt = "PNG" if o.suffix.lower() == ".png" else None
            Image.fromarray(out).save(o, format=fmt, **({"quality": 100, "subsampling": 0} if o.suffix.lower() in {".jpg", ".jpeg"} else {}))
        except Exception as e:  # keep going; report at the end
            failed.append((f, e))
            print(f"FAILED {f}: {e}", file=sys.stderr)
    print(f"restored {len(files) - len(failed)} image(s) -> {dst}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
