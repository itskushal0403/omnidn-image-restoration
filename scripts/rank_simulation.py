"""Rank simulation: which candidate minimises the expected SUM of ranks over FLOPs, PSNR, SSIM, DISTS?

The competition ranks every axis independently, so the answer depends on the (unknown) competitor field.
We sample many plausible fields under explicit assumptions and report each candidate's expected total rank.

Assumptions (deliberately simple, stated in the report):
  * N competitors (default 40).
  * FLOPs (G, at 256²) ~ mixture: 25% "tiny" log-U[0.2, 5], 45% "typical" log-U[5, 60], 30% "large" log-U[60, 400].
  * Quality: competitor PSNR ~ N(our_ref_psnr − Δ, σ_q) on our proxy scale. Δ = how far the field's average is below us
    (scenarios Δ ∈ {0, 1, 2, 3} dB), σ_q = 1.5 dB.
  * SSIM and DISTS follow PSNR with the slopes measured between our own S and M models, plus independent noise
    (σ 0.01 SSIM, 0.02 DISTS; perceptually-tuned competitors make DISTS noisier).

    python scripts/rank_simulation.py --cands "S:3.0:33.96:0.904:0.150" "M:11.4:34.46:0.910:0.141"
"""
import argparse

import numpy as np
import pandas as pd


def parse(c):
    n, f, p, s, d = c.split(":")
    return dict(name=n, flops=float(f), psnr=float(p), ssim=float(s), dists=float(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", required=True, help="name:gflops:psnr:ssim:dists")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--slope-from", nargs=2, default=None,
                    help="two FIDELITY-only candidates (no perceptual tuning) defining the SSIM/DISTS-vs-PSNR slopes "
                         "of the competitor field; the first one is also the reference level")
    args = ap.parse_args()
    cands = [parse(c) for c in args.cands]
    by_name = {c["name"]: c for c in cands}
    if args.slope_from:  # slopes must come from models WITHOUT perceptual tuning (a perceptual model has a lower PSNR
        a, b = (by_name[n] for n in args.slope_from)  # *and* a lower DISTS, which would flip the DISTS slope's sign)
    else:
        a, b = cands[0], cands[-1]
    ref = a
    hi, lo = (a, b) if a["psnr"] >= b["psnr"] else (b, a)
    dp = max(hi["psnr"] - lo["psnr"], 1e-3)
    k_s, k_d = (hi["ssim"] - lo["ssim"]) / dp, (hi["dists"] - lo["dists"]) / dp
    rng = np.random.default_rng(0)
    rows = []
    for delta in args.deltas:
        tot = {c["name"]: np.zeros(4) for c in cands}
        for _ in range(args.trials):
            g = rng.random(args.n)
            fl = np.where(g < 0.25, np.exp(rng.uniform(np.log(0.2), np.log(5), args.n)),
                          np.where(g < 0.70, np.exp(rng.uniform(np.log(5), np.log(60), args.n)),
                                   np.exp(rng.uniform(np.log(60), np.log(400), args.n))))
            ps = rng.normal(ref["psnr"] - delta, 1.5, args.n)
            ss = ref["ssim"] + k_s * (ps - ref["psnr"]) + rng.normal(0, 0.01, args.n)
            ds = ref["dists"] + k_d * (ps - ref["psnr"]) + rng.normal(0, 0.02, args.n)
            for c in cands:  # each candidate is evaluated as OUR single submission against the same field
                tot[c["name"]] += [1 + (fl < c["flops"]).sum(), 1 + (ps > c["psnr"]).sum(),
                                   1 + (ss > c["ssim"]).sum(), 1 + (ds < c["dists"]).sum()]
        for c in cands:
            r = tot[c["name"]] / args.trials
            rows.append(dict(field_below_us_dB=delta, candidate=c["name"], flops_rank=r[0], psnr_rank=r[1],
                             ssim_rank=r[2], dists_rank=r[3], total=r.sum()))
    df = pd.DataFrame(rows).round(2)
    print(df.to_string(index=False))
    print("\nbest candidate per scenario:")
    print(df.loc[df.groupby("field_below_us_dB").total.idxmin(), ["field_below_us_dB", "candidate", "total"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
