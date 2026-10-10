import numpy as np
import tifffile
import torch

from flaresatformer.run import run_experiment
from flaresatformer.viz import overlay, rgb_composite, swir2
from scripts.infer import load_run, predict


def test_rendering_helpers_shapes_and_ranges():
    patch = np.random.default_rng(0).random((256, 256, 10)).astype(np.float32)
    rgb = rgb_composite(patch)
    assert rgb.shape == (256, 256, 3) and rgb.min() >= 0 and rgb.max() <= 1
    mask = np.zeros((256, 256), bool)
    mask[10, 10] = True
    out = overlay(swir2(patch), mask)
    assert out.shape == (256, 256, 3) and out[10, 10, 0] > out[10, 10, 2]


def test_infer_loads_a_finished_run_and_predicts(synthetic_cfg, synthetic_root, tmp_path):
    run_dir = run_experiment(synthetic_cfg, fold=1, seed=1, run_dir=tmp_path / "run", smoke=True)
    cfg, model, norm, threshold = load_run(run_dir, torch.device("cpu"))
    patch = tifffile.imread(next((synthetic_root / "dataset" / "flare_patches").iterdir()))
    pred = predict(model, cfg, norm, threshold, patch, torch.device("cpu"))
    assert pred.shape == (256, 256) and pred.dtype == bool
