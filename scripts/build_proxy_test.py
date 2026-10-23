"""Build the fixed PROXY TEST SET that stands in for the organisers' hidden test set.

Design (docs/01_plan.md, Section 4):
* Clean images come only from held-out `split=val` rows of the manifests (never trained on): natural, low-light, medical.
* Each image is brought to 256×256 the way a TA plausibly would: plain resize (aspect ignored), centre-square resize,
  or a crop.
* Degradations are produced with the **library calls a TA would most likely write** (skimage.util.random_noise,
  cv2.GaussianBlur, cv2.filter2D motion kernels, PIL JPEG, np.roll chromatic aberration, ...), deliberately NOT with our
  training engine, so the evaluation is independent of the training code.
* Two sets:
    proxy_test_v2      in-family degradations (single + mixed), 2 specs per image
    proxy_test_ood_v2  families our training engine never produces (glass blur, zoom blur, uniform noise, JPEG2000,
                       salt-only + blur, ...) to measure generalization to unseen corruptions.

Output: data/val/<set>/{gt,lq}/<name>.png + meta.csv [name, domain, source, spec, family, geometry]

    python scripts/build_proxy_test.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.util import random_noise
from skimage.filters import gaussian as sk_gaussian

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dncomp.data.datasets import load_manifests, read_rgb  # noqa: E402
from dncomp.data.prep_utils import DATA_ROOT  # noqa: E402

SEED = 2026
QUOTA = {"natural": 140, "lowlight": 50, "medical": 110}


# ============================================================================================ TA-style primitives
def f2u(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def u2f(x):
    return x.astype(np.float64) / 255.0


def sk_noise(mode, **kw):
    def fn(img, rng):
        return f2u(random_noise(img, mode=mode, rng=int(rng.integers(1 << 30)), **kw))
    return fn


def cv_gblur(k):
    return lambda img, rng: cv2.GaussianBlur(img, (k, k), 0)


def sk_gblur(s):
    return lambda img, rng: f2u(sk_gaussian(img, sigma=s, channel_axis=-1))


def motion(k, angle=0):
    def fn(img, rng):
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1
        if angle:
            M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1)
            ker = cv2.warpAffine(ker, M, (k, k))
        return cv2.filter2D(img, -1, ker / ker.sum())
    return fn


def defocus(r):
    def fn(img, rng):
        k = 2 * r + 1
        ker = np.zeros((k, k), np.float32)
        cv2.circle(ker, (r, r), r, 1, -1)
        return cv2.filter2D(img, -1, ker / ker.sum())
    return fn


def box(k):
    return lambda img, rng: cv2.blur(img, (k, k))


def jpeg(q):
    def fn(img, rng):
        b = io.BytesIO()
        Image.fromarray(img).save(b, format="JPEG", quality=q)
        return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
    return fn


def ca_roll(s):
    def fn(img, rng):
        out = img.copy()
        out[..., 0] = np.roll(img[..., 0], s, axis=1)
        out[..., 2] = np.roll(img[..., 2], -s, axis=1)
        return out
    return fn


def ca_radial(scale):
    def fn(img, rng):
        h, w = img.shape[:2]
        out = img.copy()
        for ch, s in ((0, 1 + scale), (2, 1 - scale)):
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
            out[..., ch] = cv2.warpAffine(img[..., ch], M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return out
    return fn


# ---------------------------------------------------------------- OOD primitives (not produced by the training engine)
def glass_blur(sigma=0.7, delta=1, iters=2):
    def fn(img, rng):
        x = u2f(img)
        x = sk_gaussian(x, sigma=sigma, channel_axis=-1)
        h, w = x.shape[:2]
        for _ in range(iters):
            for yy in range(h - delta, delta, -1):
                for xx in range(w - delta, delta, -1):
                    dx, dy = rng.integers(-delta, delta, size=2)
                    y2, x2 = yy + dy, xx + dx
                    x[yy, xx], x[y2, x2] = x[y2, x2].copy(), x[yy, xx].copy()
        return f2u(sk_gaussian(x, sigma=sigma, channel_axis=-1))
    return fn


def zoom_blur(max_zoom=1.11):
    def fn(img, rng):
        x = u2f(img)
        out = x.copy()
        zs = np.arange(1, max_zoom, 0.02)
        h, w = x.shape[:2]
        for z in zs:
            zh, zw = int(np.ceil(h * z)), int(np.ceil(w * z))
            big = cv2.resize(x, (zw, zh), interpolation=cv2.INTER_LINEAR)
            t, l = (zh - h) // 2, (zw - w) // 2
            out += big[t:t + h, l:l + w]
        return f2u(out / (len(zs) + 1))
    return fn


def uniform_noise(a):
    return lambda img, rng: f2u(u2f(img) + rng.uniform(-a, a, img.shape))


def jp2(rate):
    def fn(img, rng):
        ok, buf = cv2.imencode(".jp2", img[..., ::-1], [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, rate])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1] if ok else img
    return fn


def chain(*fns):
    def fn(img, rng):
        for f in fns:
            img = f(img, rng)
        return img
    return fn


# ============================================================================================ spec tables
SINGLE = {
    "gauss_sk_default": ("noise", sk_noise("gaussian")),                    # var 0.01 -> sigma 25.5
    "gauss_s15": ("noise", sk_noise("gaussian", var=(15 / 255) ** 2)),
    "gauss_s50": ("noise", sk_noise("gaussian", var=(50 / 255) ** 2)),
    "poisson_sk_default": ("noise", sk_noise("poisson")),
    "sp_sk_default": ("noise", sk_noise("s&p")),                            # amount 0.05
    "sp_a0.10": ("noise", sk_noise("s&p", amount=0.10)),
    "pepper_a0.02": ("noise", sk_noise("pepper", amount=0.02)),
    "speckle_sk_default": ("noise", sk_noise("speckle")),                   # var 0.01
    "speckle_v0.05": ("noise", sk_noise("speckle", var=0.05)),
    "gblur_cv_k5": ("blur", cv_gblur(5)),
    "gblur_cv_k9": ("blur", cv_gblur(9)),
    "gblur_sk_s2": ("blur", sk_gblur(2.0)),
    "motion_h9": ("blur", motion(9)),
    "motion_d15": ("blur", motion(15, 45)),
    "defocus_r3": ("blur", defocus(3)),
    "box_k5": ("blur", box(5)),
    "jpeg_q10": ("jpeg", jpeg(10)),
    "jpeg_q30": ("jpeg", jpeg(30)),
    "ca_roll2": ("chromatic", ca_roll(2)),
    "ca_radial1pct": ("chromatic", ca_radial(0.01)),
}
MIXED = {
    "gblur_k5+gauss_s15": chain(cv_gblur(5), sk_noise("gaussian", var=(15 / 255) ** 2)),
    "motion_h9+gauss_sk": chain(motion(9), sk_noise("gaussian")),
    "gauss_s25+jpeg_q30": chain(sk_noise("gaussian", var=(25 / 255) ** 2), jpeg(30)),
    "poisson_sk+jpeg_q50": chain(sk_noise("poisson"), jpeg(50)),
    "gblur_k3+sp_a0.03": chain(cv_gblur(3), sk_noise("s&p", amount=0.03)),
    "speckle_sk+jpeg_q40": chain(sk_noise("speckle"), jpeg(40)),
    "ca_roll2+gauss_s15": chain(ca_roll(2), sk_noise("gaussian", var=(15 / 255) ** 2)),
    "defocus_r2+gauss_s10+jpeg_q40": chain(defocus(2), sk_noise("gaussian", var=(10 / 255) ** 2), jpeg(40)),
    "gblur_sk_s1.5+poisson_sk": chain(sk_gblur(1.5), sk_noise("poisson")),
    "motion_d11+sp_a0.02": chain(motion(11, 30), sk_noise("s&p", amount=0.02)),
}
OOD = {
    "glass_blur": glass_blur(),
    "zoom_blur": zoom_blur(),
    "uniform_noise_a0.15": uniform_noise(0.15),
    "jpeg2000_r40": jp2(40),
    "salt_a0.05+gblur_k3": chain(sk_noise("salt", amount=0.05), cv_gblur(3)),
    "localvar_signal_dep": lambda img, rng: f2u(random_noise(img, mode="localvar", rng=int(rng.integers(1 << 30)),
                                                             local_vars=0.002 + 0.03 * u2f(img) + 1e-6)),
    "gauss_s35+motion_h7": chain(sk_noise("gaussian", var=(35 / 255) ** 2), motion(7)),
}


# ============================================================================================ geometry
def to_256(img, rng):
    h, w = img.shape[:2]
    r = rng.random()
    if r < 0.4:
        return cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA), "resize"
    if r < 0.7:
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        return cv2.resize(img[y0:y0 + s, x0:x0 + s], (256, 256), interpolation=cv2.INTER_AREA), "square"
    f = rng.uniform(256 / min(h, w), 1.0) if min(h, w) > 256 else 1.0
    img = cv2.resize(img, (max(256, round(w * f)), max(256, round(h * f))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    y0, x0 = rng.integers(0, h - 255), rng.integers(0, w - 255)
    return img[y0:y0 + 256, x0:x0 + 256], "crop"


def pick_images(rng) -> pd.DataFrame:
    df = load_manifests("val")
    df = df[[(DATA_ROOT / p).exists() for p in df.path]]  # sources still being written may list missing files
    picks = []
    for dom, n in QUOTA.items():
        sub = df[df.domain == dom]
        if len(sub) == 0:
            continue
        # balance sources: round-robin over sources
        groups = [g.sample(frac=1, random_state=int(rng.integers(1 << 30))) for _, g in sub.groupby("source")]
        rows, i = [], 0
        while len(rows) < min(n, len(sub)):
            for g in groups:
                if i < len(g) and len(rows) < n:
                    rows.append(g.iloc[i])
            i += 1
        picks.append(pd.DataFrame(rows))
    return pd.concat(picks, ignore_index=True)


def build(name, specs_for_image, imgs, rng):
    out = DATA_ROOT / "val" / name
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "lq").mkdir(parents=True, exist_ok=True)
    meta = []
    for k, r in enumerate(imgs.itertuples()):
        gt, geom = to_256(read_rgb(DATA_ROOT / r.path), rng)
        for spec, family, fn in specs_for_image(rng):
            lq = fn(gt.copy(), rng)
            assert lq.shape == (256, 256, 3) and lq.dtype == np.uint8, spec
            nm = f"{k:04d}_{r.domain[:3]}_{spec}"
            cv2.imwrite(str(out / "gt" / f"{nm}.png"), gt[..., ::-1])
            cv2.imwrite(str(out / "lq" / f"{nm}.png"), lq[..., ::-1])
            meta.append(dict(name=nm, domain=r.domain, source=r.source, spec=spec, family=family, geometry=geom))
    pd.DataFrame(meta).to_csv(out / "meta.csv", index=False)
    print(f"{name}: {len(meta)} pairs from {len(imgs)} images ->", out)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2", help="suffix of the output sets (v1 = first build, contained lol_v1)")
    ver = ap.parse_args().version
    rng = np.random.default_rng(SEED)
    imgs = pick_images(rng)
    print(imgs.groupby(["domain", "source"]).size().to_string())
    single, mixed = list(SINGLE.items()), list(MIXED.items())

    def main_specs(rng):
        s = single[int(rng.integers(len(single)))]
        m = list(mixed[int(rng.integers(len(mixed)))])
        return [(s[0], s[1][0], s[1][1]), (m[0], "mixed", m[1])]

    build(f"proxy_test_{ver}", main_specs, imgs, rng)
    ood = list(OOD.items())
    ood_imgs = imgs.sample(n=min(140, len(imgs)), random_state=SEED)
    build(f"proxy_test_ood_{ver}", lambda rng: [(lambda s: (s[0], "ood", s[1]))(ood[int(rng.integers(len(ood)))])],
          ood_imgs, rng)


if __name__ == "__main__":
    main()
