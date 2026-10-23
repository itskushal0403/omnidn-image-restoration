# OmniDN-S — submission package

A small restoration network (NAFNet-style U-Net with a global-attention bottleneck) that removes noise, blur,
compression and other artifacts from 256×256 RGB images (natural, low-light and medical).

| File | Content |
|---|---|
| `model.py` | network definition + `build_model()` (the exact submitted configuration) |
| `weights/omnidn.pth` | trained weights, plain PyTorch `state_dict` (fp32, 26.8 MB) |
| `inference.py` | `Restorer` class + command line: 256×256×3 uint8 in → 256×256×3 uint8 out |
| `check_flops.py` | FLOPs / parameters with fvcore and ptflops at (1, 3, 256, 256) + I/O check |
| `requirements.txt` | dependencies (inference needs only torch, numpy, Pillow) |

## Usage

```bash
pip install -r requirements.txt
python inference.py --input noisy.png       --output restored.png
python inference.py --input noisy_folder/   --output restored_folder/     # keeps file names, writes PNG
python check_flops.py
```

```python
from inference import Restorer            # or: from submission.inference import Restorer
r = Restorer()                            # GPU if available, else CPU
restored = r(noisy)                       # numpy HxWx3 uint8 RGB -> numpy HxWx3 uint8 RGB
```

**Input conventions:**
- **Restorer input:** RGB images — numpy arrays (HxWx3 uint8), PIL images or file paths.
  - If you load with `cv2.imread`, convert BGR→RGB first.
  - Gray, RGBA, palette, float [0,1] and 16-bit inputs are converted automatically.
- **Output conversion:** rounds to uint8.
- **Raw network** (`model.build_model()`): takes a float tensor (N, 3, 256, 256) in [0, 1] and returns a float tensor in
  [0, 1] that already lies on the 256 uint8 levels. So `(y * 255).round()`, `(y * 255).astype(uint8)` and PIL conversion
  all give the identical image.

## Complexity (1 × 3 × 256 × 256)

| Counter | Value |
|---|---|
| fvcore | 2.963 GFLOPs (fvcore counts one multiply-add as one flop) |
| ptflops (pytorch backend) | 3.014 GMac |
| ptflops (aten backend) | 3.003 GMac |
| Parameters | 6.66 M |

Notes:
- **Fixed compute graph:** the network has no input-dependent control flow, so counts do not depend on the input tensor.
- **Precision:** fp32 is the reference. Under `model.half()` the output stays within ±1 uint8 level.
