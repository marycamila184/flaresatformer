"""Turns per-patch test statistics into tables and cluster-bootstrap confidence intervals."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flaresatformer.metrics import object_prf, prf

COUNT_COLUMNS = ["tp", "fp", "fn", "gt_objects", "gt_detected", "pred_objects", "pred_fp_objects"]
KEYS = ["variant", "bands", "frac", "fold", "seed"]


def collect_runs(outputs_dir):
    outputs_dir = Path(outputs_dir)
    rows = []
    for done in sorted(outputs_dir.rglob("DONE")):
        run_dir = done.parent
        if run_dir.relative_to(outputs_dir).parts[0] in ("smoke", "lr_search"):
            continue
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        run = cfg["run"]
        rows.append({"variant": cfg["model"]["name"] + (f"_{run['tag']}" if run.get("tag") else ""),
                     "model": cfg["model"]["name"], "bands": cfg["data"]["bands"], "frac": float(run["frac"]),
                     "fold": int(run["fold"]), "seed": int(run["seed"]), "run_dir": str(run_dir)})
    return pd.DataFrame(rows, columns=["variant", "model", "bands", "frac", "fold", "seed", "run_dir"])


def load_patches(runs):
    frames = []
    for run in runs.itertuples(index=False):
        df = pd.read_csv(Path(run.run_dir) / "test_patches.csv", dtype={"path_row": str}, keep_default_na=False)
        for key in KEYS:
            df[key] = getattr(run, key)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def metrics_from_counts(df):
    s = df[COUNT_COLUMNS].sum()
    out = {f"pixel_{k}": v for k, v in prf(s.tp, s.fp, s.fn).items()}
    out.update({f"object_{k}": v for k, v in object_prf(s.gt_objects, s.gt_detected, s.pred_objects, s.pred_fp_objects).items()})
    for source in ("urban", "fire"):
        neg = df[df.source == source]
        out[f"{source}_false_alarm_rate"] = float((neg.pred_objects > 0).mean()) if len(neg) else float("nan")
        out[f"{source}_fp_pixels_per_patch"] = float(neg.fp.mean()) if len(neg) else float("nan")
    return out


def per_run_metrics(patches, threshold_name):
    """One row per run plus a pooled 'all' row per seed (each patch is tested exactly once over the 4 folds)."""
    sub = patches[patches.threshold_name == threshold_name]
    rows = []
    for key, g in sub.groupby(KEYS):
        row = dict(zip(KEYS, key))
        rows.append({**row, "fold": str(row["fold"]), "n_folds": 1, **metrics_from_counts(g)})
    pooled = ["variant", "bands", "frac", "seed"]
    for key, g in sub.groupby(pooled):
        rows.append({**dict(zip(pooled, key)), "fold": "all", "n_folds": g.fold.nunique(), **metrics_from_counts(g)})
    return pd.DataFrame(rows)


def summarize(per_run):
    metrics = [c for c in per_run.columns if c not in KEYS + ["n_folds"]]
    grouped = per_run.groupby(["variant", "bands", "frac", "fold"])
    out = grouped[metrics].mean().add_suffix("_mean").join(grouped[metrics].std(ddof=1).add_suffix("_std"))
    out["n_seeds"] = grouped.size()
    return out.reset_index()


def paper_table(summary):
    cols = ["pixel_f1", "pixel_iou", "object_f1", "urban_false_alarm_rate", "fire_false_alarm_rate"]
    s = summary[summary.frac == 1.0]
    out = s[["variant", "bands", "fold"]].copy()
    for c in cols:
        out[c] = [f"{m:.3f} $\\pm$ {d:.3f}" if pd.notna(d) else f"{m:.3f}" for m, d in zip(s[f"{c}_mean"], s[f"{c}_std"])]
    return out


def per_continent(patches, threshold_name):
    sub = patches[(patches.threshold_name == threshold_name) & (patches.source == "flare") & (patches.frac == 1.0)]
    rows = []
    for (variant, bands, continent, seed), g in sub.groupby(["variant", "bands", "continent", "seed"]):
        s = g[["tp", "fp", "fn"]].sum()
        rows.append({"variant": variant, "bands": bands, "continent": continent, "seed": seed,
                     "pixel_f1": prf(s.tp, s.fp, s.fn)["f1"]})
    return pd.DataFrame(rows).groupby(["variant", "bands", "continent"]).pixel_f1.agg(["mean", "std"]).reset_index()


def _f1(sums, metric):
    tp, fp, fn, gt_o, gt_d, pr_o, pr_fp = np.moveaxis(np.asarray(sums, dtype=float), -1, 0)

    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(b, dtype=float), where=b > 0)

    if metric == "pixel":
        return div(2 * tp, 2 * tp + fp + fn)
    precision, recall = div(pr_o - pr_fp, pr_o), div(gt_d, gt_o)
    return div(2 * precision * recall, precision + recall)


def _delta_rows(variant, baseline, fold, point_a, point_b, boot_a, boot_b):
    rows = []
    for metric in ("pixel", "object"):
        deltas = _f1(boot_a, metric) - _f1(boot_b, metric)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        rows.append({"variant": variant, "baseline": baseline, "fold": fold, "metric": f"{metric}_f1",
                     "delta": float(_f1(point_a, metric) - _f1(point_b, metric)), "ci_low": float(lo), "ci_high": float(hi)})
    return rows


def bootstrap_delta(patches, variant, baseline="unet", threshold_name="t050", bands="b267", n_boot=2000, seed=0):
    """Cluster bootstrap of F1(variant) - F1(baseline): test path/rows resampled within each fold, seeds averaged."""
    sub = patches[(patches.threshold_name == threshold_name) & (patches.bands == bands) & (patches.frac == 1.0)]
    seed_mean = sub.groupby(["variant", "fold", "path_row", "tiff_file"])[COUNT_COLUMNS].mean()
    a, b = seed_mean.loc[variant].align(seed_mean.loc[baseline], join="inner")
    clusters_a = a.groupby(level=["fold", "path_row"]).sum()
    clusters_b = b.groupby(level=["fold", "path_row"]).sum()
    rng = np.random.default_rng(seed)
    total_a, total_b = np.zeros((n_boot, len(COUNT_COLUMNS))), np.zeros((n_boot, len(COUNT_COLUMNS)))
    rows = []
    for fold in sorted(clusters_a.index.get_level_values("fold").unique()):
        ca, cb = clusters_a.loc[fold].to_numpy(float), clusters_b.loc[fold].to_numpy(float)
        idx = rng.integers(0, len(ca), size=(n_boot, len(ca)))
        boot_a, boot_b = ca[idx].sum(axis=1), cb[idx].sum(axis=1)
        total_a += boot_a
        total_b += boot_b
        rows += _delta_rows(variant, baseline, str(fold), ca.sum(axis=0), cb.sum(axis=0), boot_a, boot_b)
    rows += _delta_rows(variant, baseline, "all", clusters_a.to_numpy(float).sum(axis=0),
                        clusters_b.to_numpy(float).sum(axis=0), total_a, total_b)
    return pd.DataFrame(rows)


def lr_search_summary(outputs_dir):
    rows = []
    for done in sorted((Path(outputs_dir) / "lr_search").rglob("DONE")):
        cfg = yaml.safe_load((done.parent / "config.yaml").read_text())
        hist = pd.read_csv(done.parent / "history.csv")
        rows.append({"model": cfg["model"]["name"], "lr": cfg["train"]["lr"], "best_val_f1": hist.val_f1.max(),
                     "best_epoch": int(hist.loc[hist.val_f1.idxmax(), "epoch"])})
    df = pd.DataFrame(rows, columns=["model", "lr", "best_val_f1", "best_epoch"])
    df["chosen"] = df.groupby("model").best_val_f1.transform("max") == df.best_val_f1
    return df


def cost_table(runs):
    rows = [{"variant": r.variant, **json.loads((Path(r.run_dir) / "cost.json").read_text())}
            for r in runs.itertuples(index=False) if r.frac == 1.0]
    return pd.DataFrame(rows).groupby("variant").mean(numeric_only=True).reset_index()
