"""One experiment run: split -> cache -> normalization -> fit -> threshold tuning -> test statistics."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from flaresatformer.data.cache import build_cache, load_cache
from flaresatformer.data.dataset import FlareDataset, normalization_for
from flaresatformer.data.splits import subsample_train
from flaresatformer.engine import collect_patch_stats, evaluate_counts, fit, measure_throughput
from flaresatformer.metrics import prf, threshold_grid
from flaresatformer.models.factory import build_model
from flaresatformer.utils import BAND_SETS, git_commit, lib_versions, load_yaml, resolve_path, save_yaml, set_seed

META_COLUMNS = ["tiff_file", "source", "country", "continent", "path_row", "group"]
SMOKE_SIZES = {"train": 64, "val": 32, "test": 32}
VOLATILE_RUN_KEYS = ("git_commit", "versions")


def default_run_dir(cfg, fold, seed, frac=1.0, tag=None, kind="main", smoke=False):
    out = resolve_path(cfg["paths"]["outputs_dir"])
    name = cfg["model"]["name"] + (f"_{tag}" if tag else "")
    if smoke:
        return out / "smoke" / name
    if kind == "lr_search":
        return out / "lr_search" / name / f"lr{cfg['train']['lr']:g}" / f"fold{fold}" / f"seed{seed}"
    return out / name / cfg["data"]["bands"] / f"frac{round(frac * 100):03d}" / f"fold{fold}" / f"seed{seed}"


def _stable(config):
    config = yaml.safe_load(yaml.safe_dump(config))  # normalize types exactly like the file on disk
    config["run"] = {k: v for k, v in config.get("run", {}).items() if k not in VOLATILE_RUN_KEYS}
    return config


def _check_or_write_config(cfg, run_info, run_dir):
    config = {**cfg, "run": {**run_info, "git_commit": git_commit(), "versions": lib_versions()}}
    path = run_dir / "config.yaml"
    if path.exists() and _stable(load_yaml(path)) != _stable(config):
        raise ValueError(f"{run_dir} was started with a different config; restore it or delete the folder to restart")
    save_yaml(config, path)


def _loader(dataset, batch_size, num_workers, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, generator=generator,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)


def run_experiment(cfg, fold, seed, frac=1.0, run_dir=None, tag=None, smoke=False, evaluate_test=True):
    run_dir = Path(run_dir or default_run_dir(cfg, fold, seed, frac, tag, smoke=smoke))
    if (run_dir / "DONE").exists():
        print(f"[skip] {run_dir} is DONE")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    _check_or_write_config(cfg, {"fold": fold, "seed": seed, "frac": frac, "tag": tag, "smoke": smoke,
                                 "evaluate_test": evaluate_test}, run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    paths, tc = cfg["paths"], cfg["train"]
    bands = BAND_SETS[cfg["data"]["bands"]]

    frame = pd.read_csv(resolve_path(paths["splits_dir"]) / f"fold{fold}.csv", dtype=str, keep_default_na=False)
    cache_dir = resolve_path(paths["cache_dir"])
    key = build_cache(frame[["tiff_file", "mask_file"]], bands, cache_dir, resolve_path(paths["data_root"]))
    images, masks, index = load_cache(cache_dir, key)
    frame = subsample_train(frame, frac)
    frame["row"] = frame["tiff_file"].map(index)
    parts = {role: frame[frame["role"] == role].reset_index(drop=True) for role in ("train", "val", "test")}
    if smoke:
        parts = {role: df.sample(min(len(df), SMOKE_SIZES[role]), random_state=seed).reset_index(drop=True)
                 for role, df in parts.items()}

    mean, std, scale = normalization_for(cfg, images, parts["train"]["row"].to_numpy(), bands)
    (run_dir / "norm.json").write_text(json.dumps({"mean": mean, "std": std, "scale": scale}))

    def dataset(role, augment=False):
        return FlareDataset(images, masks, parts[role]["row"].to_numpy(), mean, std, scale, augment)

    generator = torch.Generator().manual_seed(seed)
    workers, eval_bs = cfg["data"]["num_workers"], cfg["eval"]["batch_size"]
    train_loader = _loader(dataset("train", augment=True), tc["batch_size"], workers, shuffle=True, generator=generator)
    val_loader = _loader(dataset("val"), eval_bs, workers)

    model = build_model(cfg).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    state = fit(model, train_loader, val_loader, cfg, run_dir, device, generator)

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    grid = threshold_grid(**cfg["eval"]["thresholds"])
    counts, _ = evaluate_counts(model, val_loader, grid, device, tc["amp"])
    f1s = np.array([prf(*c)["f1"] for c in counts])
    tuned = float(grid[int(np.argmax(f1s))])
    (run_dir / "threshold.json").write_text(json.dumps(
        {"tuned": tuned, "val_f1_tuned": float(f1s.max()), "val_f1_050": float(f1s[np.isclose(grid, 0.5)][0])}))

    if evaluate_test:
        meta = parts["test"][META_COLUMNS].to_dict("records")
        stats = collect_patch_stats(model, _loader(dataset("test"), eval_bs, workers), meta,
                                    {"t050": 0.5, "tuned": tuned}, device, tc["amp"], cfg["eval"]["object_tolerance"])
        stats.to_csv(run_dir / "test_patches.csv", index=False)

    history = pd.DataFrame(state["history"])
    cost = {"params_total": sum(p.numel() for p in model.parameters()),
            "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None,
            "epoch_time_s_mean": float(history["epoch_time_s"].mean()),
            "epochs_run": int(state["epoch"]),
            "inference_patches_per_s": measure_throughput(model, len(bands), device, tc["amp"])}
    (run_dir / "cost.json").write_text(json.dumps(cost, indent=2))
    (run_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[done] {run_dir} | best val F1 {state['best_f1']:.4f} | tuned threshold {tuned}")
    return run_dir
