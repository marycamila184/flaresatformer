import json

import pandas as pd
import pytest

from flaresatformer.run import run_experiment

ARTIFACTS = ("config.yaml", "norm.json", "history.csv", "best.pt", "last.pt", "threshold.json",
             "test_patches.csv", "cost.json", "DONE")


def test_run_experiment_end_to_end(synthetic_cfg, tmp_path):
    run_dir = run_experiment(synthetic_cfg, fold=1, seed=1, run_dir=tmp_path / "run")
    for name in ARTIFACTS:
        assert (run_dir / name).exists(), name
    split = pd.read_csv(f"{synthetic_cfg['paths']['splits_dir']}/fold1.csv")
    stats = pd.read_csv(run_dir / "test_patches.csv")
    assert len(stats) == 2 * (split.role == "test").sum()
    assert set(stats.source) == {"flare", "urban", "fire"}
    assert 0.05 <= json.loads((run_dir / "threshold.json").read_text())["tuned"] <= 0.95


def test_done_run_is_skipped(synthetic_cfg, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "DONE").write_text("x")
    assert run_experiment(synthetic_cfg, fold=1, seed=1, run_dir=run_dir) == run_dir
    assert not (run_dir / "config.yaml").exists()


def test_resume_with_changed_config_is_refused(synthetic_cfg, tmp_path):
    run_dir = tmp_path / "run"
    run_experiment(synthetic_cfg, fold=1, seed=1, run_dir=run_dir, smoke=True)
    (run_dir / "DONE").unlink()
    synthetic_cfg["train"]["lr"] = 0.123
    with pytest.raises(ValueError, match="different config"):
        run_experiment(synthetic_cfg, fold=1, seed=1, run_dir=run_dir, smoke=True)
