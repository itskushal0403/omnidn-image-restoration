"""Measure the submitted model's complexity exactly as the organisers do: a (1, 3, 256, 256) input with fvcore and
ptflops (both of its backends). Also checks the output shape / dtype contract.

    python check_flops.py
"""
import warnings

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import Restorer, build_model  # noqa: E402  (model.py loaded by path inside inference.py)

warnings.filterwarnings("ignore")


def main():
    model = build_model().eval()
    x = torch.rand(1, 3, 256, 256)
    params = sum(p.numel() for p in model.parameters()) / 1e6

    from fvcore.nn import FlopCountAnalysis

    fca = FlopCountAnalysis(model, x)
    fca.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
    print(f"fvcore             : {fca.total() / 1e9:.3f} GFLOPs (fvcore counts 1 multiply-add as 1 flop)")

    from ptflops import get_model_complexity_info

    for backend in ("pytorch", "aten"):
        try:
            macs, _ = get_model_complexity_info(model, (3, 256, 256), as_strings=False, backend=backend,
                                                print_per_layer_stat=False, verbose=False)
            print(f"ptflops ({backend:7s})  : {macs / 1e9:.3f} GMACs")
        except TypeError:  # very old ptflops without a backend argument
            macs, _ = get_model_complexity_info(model, (3, 256, 256), as_strings=False,
                                                print_per_layer_stat=False, verbose=False)
            print(f"ptflops            : {macs / 1e9:.3f} GMACs")
            break
    print(f"parameters         : {params:.3f} M")

    r = Restorer(device="cpu")
    out = r((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
    assert out.shape == (256, 256, 3) and out.dtype == np.uint8
    print("I/O contract       : OK (256x256x3 uint8 -> 256x256x3 uint8)")


if __name__ == "__main__":
    main()
