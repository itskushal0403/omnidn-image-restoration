"""Package a trained checkpoint into the self-contained `submission/` folder.

    python scripts/package_submission.py --ckpt checkpoints/<run>/<file>.pth [--raw]

Produces
  submission/model.py            standalone copy of the architecture + build_model() with the chosen configuration
  submission/weights/omnidn.pth  plain PyTorch state_dict (EMA weights by default), fp32
and then runs submission/check_flops.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submission"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--raw", action="store_true", help="export raw weights instead of EMA weights")
    args = ap.parse_args()

    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    state = ck["model"] if args.raw else ck["ema"]
    state = {k: v.float().contiguous() for k, v in state.items()}
    from dncomp.models.omnidn import PRESETS

    cfg = dict(PRESETS[ck["model_name"]])
    cfg.update(ck.get("model_kwargs", {}))
    cfg["output_uint8_levels"] = True  # submitted network returns values already on the 256 uint8 levels

    src = (ROOT / "src" / "dncomp" / "models" / "omnidn.py").read_text()
    head, _ = src.split("# ------------------------------------------------------------------------------------------- presets")
    model_py = head + (
        "# ------------------------------------------------------------------------------------------- submitted model\n"
        f"# exported from: {args.ckpt} ({'raw' if args.raw else 'EMA'} weights)\n"
        f"CONFIG = {cfg!r}\n\n\n"
        "def build_model() -> OmniDN:\n"
        "    return OmniDN(**CONFIG)\n"
    )
    (SUB / "model.py").write_text(model_py)
    (SUB / "weights").mkdir(exist_ok=True)
    torch.save(state, SUB / "weights" / "omnidn.pth")
    print("wrote", SUB / "model.py", "and", SUB / "weights" / "omnidn.pth")
    import shutil

    shutil.rmtree(SUB / "__pycache__", ignore_errors=True)
    subprocess.run([sys.executable, "-B", "check_flops.py"], cwd=SUB, check=True)
    shutil.rmtree(SUB / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
