"""Training entry point (one GPU per run).

    python -m dncomp.train --config configs/omnidn_s.yaml

Features
--------
* bf16 autocast + channels_last + optional ``torch.compile``.
* EMA of weights (evaluated and exported instead of raw weights).
* **Wall-clock cosine schedule**: progress = max(iter / max_iters, elapsed / time_budget), so a run always finishes its
  LR decay before the deadline, whatever the actual throughput turns out to be.
* Auto-resume from ``checkpoints/<name>/last.pth`` (SLURM re-submissions just continue). SIGTERM / SIGUSR1 → save
  and exit cleanly.
* Periodic evaluation on the proxy test set with the exact competition metrics; logs to TensorBoard + CSV.
* Fine-tuning: ``init`` = path of a previous checkpoint (weights only, fresh optimiser / schedule).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data.datasets import PairedFolderDataset, TrainDataset
from .data.degradations import DegradationComposer
from .evaluate import evaluate
from .losses.losses import CombinedLoss
from .models.omnidn import build_model

PROJECT = Path(__file__).resolve().parents[2]


def log(msg):
    print(time.strftime("[%m-%d %H:%M:%S] ") + msg, flush=True)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))  # warm-up of the EMA decay
        ema_p = list(self.model.parameters())
        p = [x.detach() for x in model.parameters()]
        torch._foreach_lerp_(ema_p, p, 1.0 - d)


def lr_factor(progress, warmup_frac, min_factor):
    if progress < warmup_frac:
        return max(progress / warmup_frac, 1e-3)
    p = (progress - warmup_frac) / max(1e-8, 1.0 - warmup_frac)
    return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="key.sub=value overrides")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for ov in args.override:
        k, v = ov.split("=", 1)
        d = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = yaml.safe_load(v)

    name = cfg["name"]
    ckpt_dir = PROJECT / "checkpoints" / name
    run_dir = PROJECT / "runs" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)
    tc, dc, vc = cfg["train"], cfg["data"], cfg.get("val", {})

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    torch.manual_seed(tc.get("seed", 0))

    # ------------------------------------------------------------------ model / optimiser / loss
    model = build_model(cfg["model"]["name"], **cfg["model"].get("kwargs", {})).to(device)
    model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], betas=tuple(tc.get("betas", (0.9, 0.9))),
                            weight_decay=tc.get("weight_decay", 0.0))
    loss_fn = CombinedLoss(**cfg["loss"]).to(device)
    ema = EMA(model, tc.get("ema", 0.999))

    state = {"iter": 0, "elapsed": 0.0, "best": -1e9}
    last = ckpt_dir / "last.pth"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        log(f"resumed from {last} at iter {state['iter']} ({state['elapsed'] / 3600:.2f} h elapsed)")
    elif cfg.get("init"):
        ck = torch.load(PROJECT / cfg["init"], map_location="cpu", weights_only=False)
        sd = ck["ema"] if "ema" in ck else ck
        model.load_state_dict(sd)
        ema.model.load_state_dict(sd)
        log(f"initialised weights from {cfg['init']}")

    train_model = torch.compile(model) if tc.get("compile", True) else model

    # optional knowledge distillation: a frozen (larger) teacher's output as an extra target
    teacher, kd_weight = None, 0.0
    if cfg.get("teacher"):
        tck = torch.load(PROJECT / cfg["teacher"]["ckpt"], map_location="cpu", weights_only=False)
        teacher = build_model(tck["model_name"], **tck.get("model_kwargs", {}))
        teacher.load_state_dict(tck["ema"])
        teacher = teacher.to(device).to(memory_format=torch.channels_last).eval().requires_grad_(False)
        kd_weight = cfg["teacher"].get("weight", 1.0)
        log(f"distilling from teacher {cfg['teacher']['ckpt']} (relative-MSE weight {kd_weight})")

    # ------------------------------------------------------------------ data
    composer = DegradationComposer(**dc.get("composer", {}))
    ds = TrainDataset(patch=dc.get("patch", 256), domain_weights=dc.get("domain_weights"),
                      darken_prob=dc.get("darken_prob", 0.1), gray_prob=dc.get("gray_prob", 0.04),
                      composer=composer, seed=state["iter"] + tc.get("seed", 0))
    log(ds.summary())
    loader = DataLoader(ds, batch_size=dc["batch_size"], num_workers=dc.get("num_workers", 12), pin_memory=True,
                        drop_last=True, persistent_workers=True, prefetch_factor=4)
    val_ds = PairedFolderDataset(PROJECT / vc.get("data", "data/val/proxy_test_v2"), limit=vc.get("limit"))
    dists_net = None
    if vc.get("dists_every"):
        from DISTS_pytorch import DISTS

        dists_net = DISTS().to(device).eval()

    writer = SummaryWriter(run_dir / "tb")
    log(f"run={name} params={n_params:.2f}M  batch={dc['batch_size']}  time_budget={tc['time_budget_h']} h  "
        f"max_iters={tc['max_iters']}")

    # ------------------------------------------------------------------ checkpointing / signals
    def save(path, extra=None):
        ck = {"model": model.state_dict(), "ema": ema.model.state_dict(), "opt": opt.state_dict(), "state": state,
              "model_name": cfg["model"]["name"], "model_kwargs": cfg["model"].get("kwargs", {}), "config": cfg}
        if extra:
            ck.update(extra)
        tmp = Path(str(path) + ".tmp")
        torch.save(ck, tmp)
        os.replace(tmp, path)

    stop = {"flag": False}

    def _sig(signum, frame):
        log(f"signal {signum} received -> checkpoint and exit")
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGUSR1, _sig)

    # ------------------------------------------------------------------ loop
    budget_s = tc["time_budget_h"] * 3600
    t_start = time.time() - state["elapsed"]
    t_ckpt = time.time()
    t_log = time.time()
    iters_since_log = 0
    model.train()
    data_iter = iter(loader)
    while True:
        it = state["iter"]
        state["elapsed"] = time.time() - t_start
        progress = max(it / tc["max_iters"], state["elapsed"] / budget_s)
        if progress >= 1.0 or stop["flag"]:
            break
        lr = tc["lr"] * lr_factor(progress, tc.get("warmup_frac", 0.02), tc.get("min_lr_factor", 1e-3))
        for g in opt.param_groups:
            g["lr"] = lr

        try:
            batch = next(data_iter)
        except StopIteration:  # end of a (virtual) epoch: the stream is infinite, just restart it
            data_iter = iter(loader)
            batch = next(data_iter)
        lq = batch["lq"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        gt = batch["gt"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = train_model(lq)
        loss, parts = loss_fn(out.float(), gt)
        if teacher is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t_out = teacher(lq).float().clamp(0, 1)
            # MSE to the teacher, normalised per image by the (detached) MSE to the ground truth: its gradient has the
            # same scale as PSNRLoss's, so weight w behaves like training on the blended target (gt + w·teacher)/(1 + w).
            mse_t = (out.float() - t_out).pow(2).mean(dim=(1, 2, 3))
            mse_g = (out.float() - gt.float()).pow(2).mean(dim=(1, 2, 3)).detach()
            kd = 10.0 / math.log(10.0) * (mse_t / (mse_g + 1e-8)).mean()
            loss = loss + kd_weight * kd
            parts["kd_rel"] = float(kd.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))
        if torch.isfinite(loss):
            opt.step()
            ema.update(model, it)
        state["iter"] = it + 1
        iters_since_log += 1

        if (it + 1) % tc.get("log_every", 100) == 0:
            dt = time.time() - t_log
            ips = iters_since_log * dc["batch_size"] / dt
            t_log, iters_since_log = time.time(), 0
            writer.add_scalar("train/loss", float(loss), it + 1)
            writer.add_scalar("train/lr", lr, it + 1)
            writer.add_scalar("train/grad_norm", float(gn), it + 1)
            writer.add_scalar("train/img_per_s", ips, it + 1)
            for k, v in parts.items():
                writer.add_scalar(f"train/{k}", v, it + 1)
            log(f"it {it + 1:7d}  loss {float(loss):.5f}  " + " ".join(f"{k}={v:.4f}" for k, v in parts.items())
                + f"  lr {lr:.2e}  gn {float(gn):.3f}  {ips:.0f} img/s  prog {progress * 100:.1f}%")

        if (it + 1) % vc.get("every", 2000) == 0:
            do_dists = dists_net is not None and (it + 1) % vc["dists_every"] == 0
            df = evaluate(ema.model, val_ds, dists_model=dists_net if do_dists else None)
            m = df[[c for c in ("psnr", "ssim", "dists") if c in df]].mean().to_dict()
            per_dom = df.groupby("domain")[["psnr", "ssim"]].mean()
            for k, v in m.items():
                writer.add_scalar(f"val/{k}", v, it + 1)
            for dom, r in per_dom.iterrows():
                writer.add_scalar(f"val_{dom}/psnr", r.psnr, it + 1)
            row = {"iter": it + 1, "hours": state["elapsed"] / 3600, "psnr": m["psnr"], "ssim": m["ssim"],
                   "dists": m.get("dists", float("nan")),  # fixed column order (DISTS only every dists_every)
                   **{f"psnr_{d}": per_dom.psnr.get(d, float("nan")) for d in ("lowlight", "medical", "natural")}}
            pd.DataFrame([row]).to_csv(run_dir / "val_metrics.csv", mode="a",
                                       header=not (run_dir / "val_metrics.csv").exists(), index=False)
            log("VAL " + json.dumps({k: round(v, 4) for k, v in row.items()}))
            score = m["psnr"] + 30 * m["ssim"]  # selection proxy (dB-equivalent); final choice is made offline
            if score > state["best"]:
                state["best"] = score
                torch.save({"ema": ema.model.state_dict(), "model_name": cfg["model"]["name"],
                            "model_kwargs": cfg["model"].get("kwargs", {}), "iter": it + 1, "val": row},
                           ckpt_dir / "ema_best.pth")
            if (it + 1) % vc.get("snapshot_every", 20000) == 0:
                torch.save({"ema": ema.model.state_dict(), "model_name": cfg["model"]["name"],
                            "model_kwargs": cfg["model"].get("kwargs", {}), "iter": it + 1, "val": row},
                           ckpt_dir / f"ema_iter{it + 1:07d}.pth")

        if time.time() - t_ckpt > tc.get("ckpt_minutes", 20) * 60:
            save(last)
            t_ckpt = time.time()

    save(last)
    torch.save({"ema": ema.model.state_dict(), "model_name": cfg["model"]["name"],
                "model_kwargs": cfg["model"].get("kwargs", {}), "iter": state["iter"]}, ckpt_dir / "ema_final.pth")
    finished = progress >= 1.0
    log(f"{'finished' if finished else 'stopped'} at iter {state['iter']} after {state['elapsed'] / 3600:.2f} h")
    sys.exit(0 if finished else 3)


if __name__ == "__main__":
    main()
