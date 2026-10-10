import numpy as np
import pandas as pd

from flaresatformer.aggregate import bootstrap_delta, metrics_from_counts, per_continent, per_run_metrics, summarize
from flaresatformer.figures import cost_vs_f1, f1_per_continent, learning_curve


def _row(**kw):
    base = {"variant": "unet", "bands": "b267", "frac": 1.0, "fold": 1, "seed": 1, "threshold_name": "t050",
            "source": "flare", "continent": "Asia", "path_row": "001", "tiff_file": "x",
            "tp": 0, "fp": 0, "fn": 0, "gt_objects": 0, "gt_detected": 0, "pred_objects": 0, "pred_fp_objects": 0}
    return {**base, **kw}


def _patches():
    rows = []
    for variant, tp in (("unet", 8), ("segformer", 6)):
        for seed in (1, 2):
            for fold in (1, 2):
                continent = "Asia" if fold == 1 else "Africa"
                for k in range(10):
                    rows.append(_row(variant=variant, seed=seed, fold=fold, continent=continent, path_row=f"{fold}{k % 5}",
                                     tiff_file=f"f{fold}_{k}", tp=tp, fp=2, fn=10 - tp, gt_objects=1,
                                     gt_detected=int(tp > 6), pred_objects=1))
                rows.append(_row(variant=variant, seed=seed, fold=fold, source="urban", continent="",
                                 path_row=f"u{fold}", tiff_file=f"u{fold}", fp=3, pred_objects=1, pred_fp_objects=1))
    return pd.DataFrame(rows)


def test_metrics_from_counts():
    m = metrics_from_counts(pd.DataFrame([
        _row(tp=8, fp=2, fn=2, gt_objects=2, gt_detected=1, pred_objects=2, pred_fp_objects=1),
        _row(source="urban", fp=4, pred_objects=1, pred_fp_objects=1),
        _row(source="urban")]))
    assert m["pixel_precision"] == 8 / 14 and m["pixel_recall"] == 0.8
    assert m["object_recall"] == 0.5 and m["object_precision"] == 1 / 3
    assert m["urban_false_alarm_rate"] == 0.5 and m["urban_fp_pixels_per_patch"] == 2.0
    assert np.isnan(m["fire_false_alarm_rate"])


def test_per_run_and_summary_shapes():
    per_run = per_run_metrics(_patches(), "t050")
    assert len(per_run) == 2 * 2 * 2 + 2 * 2  # (variant, fold, seed) + pooled (variant, seed)
    summary = summarize(per_run)
    row = summary[(summary.variant == "unet") & (summary.fold == "1")].iloc[0]
    assert row.n_seeds == 2 and row.pixel_f1_std == 0.0


def test_bootstrap_identical_models_have_zero_delta():
    p = _patches()
    d = bootstrap_delta(pd.concat([p, p[p.variant == "unet"].assign(variant="copy")]), "copy", n_boot=200)
    assert (d.delta == 0).all() and (d.ci_low == 0).all() and (d.ci_high == 0).all()


def test_bootstrap_detects_worse_model():
    d = bootstrap_delta(_patches(), "segformer", n_boot=200)
    pixel = d[(d.metric == "pixel_f1") & (d.fold == "all")].iloc[0]
    assert pixel.delta < 0 and pixel.ci_high < 0


def test_figures_render(tmp_path):
    p = _patches()
    per_run = per_run_metrics(pd.concat([p, p[p.variant == "unet"].assign(frac=0.5)]), "t050")
    learning_curve(per_run, tmp_path / "lc.png")
    f1_per_continent(per_continent(p, "t050"), tmp_path / "continent.png")
    cost = pd.DataFrame({"variant": ["unet", "segformer"], "inference_patches_per_s": [100.0, 50.0]})
    cost_vs_f1(cost, summarize(per_run), tmp_path / "cost.png")
    assert all((tmp_path / n).stat().st_size > 1000 for n in ("lc.png", "continent.png", "cost.png"))
