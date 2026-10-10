"""Training loop (bf16, gradient accumulation, early stopping, exact resume) and evaluation passes."""
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from flaresatformer.losses import FocalDiceLoss
from flaresatformer.metrics import object_counts, pixel_counts, prf
from flaresatformer.models.factory import param_groups


def warmup_cosine(total_steps, warmup_frac):
    warmup = max(1, int(total_steps * warmup_frac))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def autocast(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bool(amp) and device.type == "cuda")


def capture_rng(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def _atomic_save(obj, path):
    tmp = Path(f"{path}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, accum, grad_clip, device, amp):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    params = [p for group in optimizer.param_groups for p in group["params"]]
    total, seen = 0.0, 0
    for i, (x, y, _) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, amp):
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        (loss / accum).backward()
        total += loss.item() * x.shape[0]
        seen += x.shape[0]
        if (i + 1) % accum == 0 or i + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return total / seen


@torch.no_grad()
def evaluate_counts(model, loader, thresholds, device, amp, loss_fn=None):
    """Pixel tp/fp/fn summed over the loader for every threshold, without storing probabilities."""
    model.eval()
    thr = torch.as_tensor(np.asarray(thresholds, dtype=np.float32), device=device).view(-1, 1)
    counts = torch.zeros(len(thresholds), 3, dtype=torch.float64, device=device)
    loss_sum, seen = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, amp):
            logits = model(x)
        logits = logits.float()
        if loss_fn is not None:
            loss_sum += loss_fn(logits, y).item() * x.shape[0]
            seen += x.shape[0]
        pred = torch.sigmoid(logits).flatten().unsqueeze(0) > thr  # (T, pixels)
        target = (y.flatten() > 0.5).unsqueeze(0)
        counts[:, 0] += (pred & target).sum(dim=1)
        counts[:, 1] += (pred & ~target).sum(dim=1)
        counts[:, 2] += (~pred & target).sum(dim=1)
    return counts.cpu().numpy(), (loss_sum / seen if seen else float("nan"))


def fit(model, train_loader, val_loader, cfg, run_dir, device, generator, stop_after=None):
    """Trains with early stopping on val F1@0.5; resumes exactly from run_dir/last.pt if present."""
    tc = cfg["train"]
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    accum = max(1, tc["effective_batch_size"] // tc["batch_size"])
    optimizer = torch.optim.AdamW(param_groups(model, tc["lr"], tc["encoder_lr_mult"], cfg["model"]["pretrained"]),
                                  weight_decay=tc["weight_decay"])
    total_steps = math.ceil(len(train_loader) / accum) * tc["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(total_steps, tc["warmup_frac"]))
    loss_fn = FocalDiceLoss(tc["focal_alpha"], tc["focal_gamma"], tc["dice_weight"])

    state = {"epoch": 0, "best_f1": -1.0, "bad_epochs": 0, "history": []}
    last_path = run_dir / "last.pt"
    if last_path.exists():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        state = ckpt["state"]
        restore_rng(ckpt["rng"], generator)
        print(f"resumed {run_dir} at epoch {state['epoch']}", flush=True)

    epochs_this_call = 0
    while state["epoch"] < tc["epochs"] and state["bad_epochs"] < tc["patience"]:
        if stop_after is not None and epochs_this_call >= stop_after:
            break
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, accum, tc["grad_clip"], device, tc["amp"])
        counts, val_loss = evaluate_counts(model, val_loader, [0.5], device, tc["amp"], loss_fn)
        metrics = prf(*counts[0])
        state["epoch"] += 1
        epochs_this_call += 1
        state["history"].append({"epoch": state["epoch"], "train_loss": train_loss, "val_loss": val_loss,
                                 "val_precision": metrics["precision"], "val_recall": metrics["recall"],
                                 "val_f1": metrics["f1"], "lr": optimizer.param_groups[-1]["lr"],
                                 "epoch_time_s": time.time() - start})
        if metrics["f1"] > state["best_f1"]:
            state["best_f1"], state["bad_epochs"] = metrics["f1"], 0
            _atomic_save(model.state_dict(), run_dir / "best.pt")
        else:
            state["bad_epochs"] += 1
        pd.DataFrame(state["history"]).to_csv(run_dir / "history.csv", index=False)
        _atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "state": state, "rng": capture_rng(generator)}, last_path)
        print(f"epoch {state['epoch']:3d} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"F1 {metrics['f1']:.4f} P {metrics['precision']:.4f} R {metrics['recall']:.4f}", flush=True)
    return state


@torch.no_grad()
def collect_patch_stats(model, loader, meta, thresholds, device, amp, tol):
    """One row per (patch, threshold) with pixel and object counts plus the patch metadata."""
    model.eval()
    rows = []
    for x, y, idx in loader:
        with autocast(device, amp):
            logits = model(x.to(device))
        probs = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
        gts = y[:, 0].numpy() > 0.5
        for b, i in enumerate(idx.tolist()):
            for name, t in thresholds.items():
                pred = probs[b] > t
                tp, fp, fn = pixel_counts(pred, gts[b])
                rows.append({**meta[i], "threshold_name": name, "threshold": float(t),
                             "tp": tp, "fp": fp, "fn": fn, **object_counts(pred, gts[b], tol)})
    return pd.DataFrame(rows)


@torch.no_grad()
def measure_throughput(model, in_channels, device, amp, batch_size=16, iters=10):
    model.eval()
    x = torch.randn(batch_size, in_channels, 256, 256, device=device)
    with autocast(device, amp):
        model(x)  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        with autocast(device, amp):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return batch_size * iters / (time.perf_counter() - start)
