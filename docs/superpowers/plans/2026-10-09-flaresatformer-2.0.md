# FlareSatFormer 2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v1 pipeline with a reproducible, resumable experiment pipeline comparing UNet, SegFormer (pretrained and scratch), Swin-UNet, TransUNet and Prithvi-EO-2.0 for subpixel gas flare segmentation.

**Architecture:** Our own PyTorch training loop (`flaresatformer/engine.py`). Model libraries are used only as factories (HF `transformers`, `timm`, `terratorch` + `peft`). YAML configs feed a single CLI (`scripts/train.py`). A queue (`scripts/run_queue.py`) expands the experiment matrix and skips finished runs. Each run writes per-patch test statistics, which `scripts/aggregate.py` turns into tables, bootstrap CIs and figures.

**Tech Stack:** Python 3.12, torch 2.11 (cu128), transformers 5.19, timm 1.0.30, terratorch 1.2.13, peft 0.21, scikit-learn, scipy, pandas, tifffile, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-flaresatformer-2.0-design.md`

## Global Constraints

- Package name: `flaresatformer` (not `flaresat`).
- Channel order in the TIFFs: B1 B2 B3 B4 B5 B6 B7 B9 B10 B11. Main bands: indices `[1, 5, 6]` (`b267`). HLS variant: indices `[1, 2, 3, 4, 5, 6]` (`hls6`).
- Splits are fixed (`random_state=42`). Seeds never change splits.
- Same protocol for all models. Only `train.lr` is tuned per model (priority 0).
- Effective batch size 16. bf16 autocast only on CUDA. Gradient clipping at 1.0.
- venv at `~/venvs/flaresatformer`. Cache at `~/.cache/flaresatformer`. Nothing that needs symlinks may live on the USB drive.
- `outputs/` is git-ignored. `splits/` and `results/` are versioned.
- Scripts run as modules from the repo root: `~/venvs/flaresatformer/bin/python -m scripts.<name>`.
- **No training runs or GPU smoke tests in this implementation session** (user request). Unit tests run on CPU with synthetic data.

**Deviations from the spec (intentional):**
- FRH projects decoder features with a 1×1 conv to 32 channels *before* upsampling. Upsampling 768-channel features to 256×256 would cost about 1.6 GB of VRAM per batch.
- The cache is memory-mapped and relies on the OS page cache (62 GB RAM) instead of an explicit RAM copy. The effect is the same, and it works with DataLoader workers.
- No `scripts/evaluate.py`. Test evaluation runs at the end of every training run (`test_patches.csv`), and `scripts/aggregate.py` recomputes everything from those files.

## Review Focus

1. **Resuming after editing a config:** a person changes `lr` in a YAML and reruns the queue. A half-finished run must refuse to resume, not silently mix settings. Test in Task 8 (`test_resume_with_changed_config_is_refused`).
2. **Running main experiments before lr tuning:** the queue must block priority ≥1 jobs for models whose `train.lr_tuned` is false, unless `--allow-untuned` is passed. Test in Task 8 (`test_untuned_models_are_blocked_from_main_runs`).
3. **Stale cache:** if the CSVs or the band set change, a new cache key must be used, never the old arrays. Test in Task 3 (`test_key_changes_with_bands_or_file_list`).
4. **NaN/Inf pixels in Landsat TIFFs:** non-finite values must not poison normalization stats or losses. They are replaced by 0 and counted at cache build. Test in Task 3 (`test_non_finite_values_become_zero`).
5. **Empty subsets in metrics:** a continent with no predictions, or urban/fire subsets with no positives, must yield 0 (or NaN for absent sources), never a ZeroDivisionError. Tests in Task 5 (`test_prf_is_zero_not_nan_on_empty_inputs`) and Task 9 (`test_metrics_from_counts`).

## File Structure

```
configs/base.yaml                      common protocol, paths
configs/models/<model>.yaml            per-model overrides (batch size, lr, flags)
configs/experiments.yaml               experiment matrix with priorities
flaresatformer/utils.py                config merge, seeding, band tables, provenance
flaresatformer/data/splits.py          scene parsing, folds, validation, learning-curve subsampling
flaresatformer/data/cache.py           TIFF -> float16 .npy band cache
flaresatformer/data/dataset.py         normalization + FlareDataset (flip/rot90)
flaresatformer/losses.py               Focal (alpha_t) + batch Dice
flaresatformer/metrics.py              pixel/object metrics, threshold grid
flaresatformer/models/frh.py           conv_bn_relu, FullResHead, UpsampleHead, make_head
flaresatformer/models/{unet,segformer,swin_unet,transunet,prithvi}.py
flaresatformer/models/factory.py       build_model, param_groups
flaresatformer/engine.py               fit/resume, evaluate_counts, collect_patch_stats, throughput
flaresatformer/run.py                  run_experiment, default_run_dir
flaresatformer/experiments.py          Job, expand, select_jobs
flaresatformer/aggregate.py            per-run metrics, summaries, bootstrap, lr-search summary
flaresatformer/figures.py              paper figures
flaresatformer/viz.py                  RGB/B7/overlay rendering
scripts/{make_splits,build_cache,train,run_queue,aggregate,infer,make_readme_figure}.py
splits/fold{1..4}.csv
tests/...
```

Code blocks whose first line is `# file: <path>` are the exact content of that file (first line excluded).

---

### Task 1: Preserve v1, clean up and scaffold the package

**Files:**
- Delete: `models/`, `train/`, `inference/`, `flaresatformer.egg-info/`
- Modify: `pyproject.toml`, `requirements.txt`, `.gitignore`
- Create: `configs/base.yaml`, `configs/models/*.yaml`, `configs/experiments.yaml`, `flaresatformer/__init__.py`, `flaresatformer/data/__init__.py`, `flaresatformer/models/__init__.py`, `scripts/__init__.py`, `flaresatformer/utils.py`
- Test: `tests/test_utils.py`

**Interfaces:**
- Produces: `REPO_ROOT`, `CONFIG_DIR`, `BAND_SETS`, `HLS_NAMES`, `PRITHVI_MEAN`, `PRITHVI_STD`, `deep_merge(base, override) -> dict`, `load_yaml(path)`, `save_yaml(data, path)`, `load_config(model, overrides=None, config_dir=CONFIG_DIR) -> dict`, `resolve_path(path) -> Path`, `set_seed(seed, deterministic=True)`, `git_commit() -> str`, `lib_versions() -> dict`.

- [ ] **Step 1: Commit the last v1 state** (the uncommitted TransUNet edits and the deleted outputs), so nothing is lost:

```bash
git add -A models train && git commit -m "chore: snapshot last v1 pipeline state before 2.0 rewrite"
```

- [ ] **Step 2: Remove v1 code.** It stays available in history.

```bash
git rm -r -q models train inference && rm -rf flaresatformer.egg-info
```

- [ ] **Step 3: Write the scaffolding files**

```toml
# file: pyproject.toml
[project]
name = "flaresatformer"
version = "2.0.0"
requires-python = ">=3.12"
dependencies = []

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["flaresatformer*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = ["slow: needs pretrained weight downloads or long CPU time"]
addopts = "-m 'not slow'"
```

```text
# file: requirements.txt
# Install torch first from the CUDA 12.8 index:
#   uv pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
transformers==5.19.0
timm==1.0.30
terratorch==1.2.13
peft==0.21.2
numpy==2.5.2
scipy==1.18.1
pandas==3.0.6
scikit-learn==1.9.1
tifffile==2026.9.20
pyyaml==6.0.3
matplotlib==3.11.2
jinja2
pytest==9.1.1
```

Append to `.gitignore`:

```text

# Experiment outputs (only results/ is versioned)
outputs/
```

```yaml
# file: configs/base.yaml
paths:
  data_root: .                       # CSV paths (dataset/...) are relative to this folder
  dataset_dir: dataset
  splits_dir: splits
  cache_dir: ~/.cache/flaresatformer
  outputs_dir: outputs
  results_dir: results

data:
  bands: b267                        # b267 = B2, B6, B7 | hls6 = B2..B7
  num_workers: 4

model:
  pretrained: true
  frh: true
  norm: dataset                      # dataset | prithvi
  grad_checkpointing: false

train:
  epochs: 80
  patience: 15
  effective_batch_size: 16
  batch_size: 16                     # physical batch; models override, accumulation fills the rest
  lr: 3.0e-4
  lr_tuned: false                    # set true after priority 0 picks the lr
  encoder_lr_mult: 0.1
  weight_decay: 0.01
  warmup_frac: 0.05
  grad_clip: 1.0
  amp: true
  focal_alpha: 0.25
  focal_gamma: 2.0
  dice_weight: 1.0

eval:
  batch_size: 16
  object_tolerance: 1
  thresholds: {start: 0.05, stop: 0.95, step: 0.05}
```

```yaml
# file: configs/models/unet.yaml
model:
  name: unet
  pretrained: false
  frh: false
train:
  batch_size: 16
```

```yaml
# file: configs/models/segformer.yaml
model:
  name: segformer
train:
  batch_size: 16
```

```yaml
# file: configs/models/segformer_scratch.yaml
model:
  name: segformer_scratch
  pretrained: false
train:
  batch_size: 16
```

```yaml
# file: configs/models/swin_unet.yaml
model:
  name: swin_unet
train:
  batch_size: 8
```

```yaml
# file: configs/models/transunet.yaml
model:
  name: transunet
train:
  batch_size: 8
```

```yaml
# file: configs/models/prithvi.yaml
model:
  name: prithvi
  norm: prithvi
  lora_r: 16
  lora_alpha: 32
  freeze_encoder: false              # fallback if LoRA does not fit in 8 GB
  grad_checkpointing: true
train:
  batch_size: 4
```

```yaml
# file: configs/experiments.yaml
models: [unet, segformer, segformer_scratch, swin_unet, transunet, prithvi]
folds: [1, 2, 3, 4]
experiments:
  # P0: lr search on the validation split of fold 1 only (test fold untouched)
  - {priority: 0, kind: lr_search, models: all, folds: [1], seeds: [1], lrs: [1.0e-4, 3.0e-4, 1.0e-3], epochs: 30}
  # P1/P2: main comparison, 3 seeds
  - {priority: 1, models: all, seeds: [1]}
  - {priority: 2, models: all, seeds: [2, 3]}
  # P3: Prithvi with all six HLS bands, SegFormer without the full-resolution head
  - {priority: 3, models: [prithvi], seeds: [1], tag: hls6, overrides: {data: {bands: hls6}}}
  - {priority: 3, models: [segformer], seeds: [1], tag: nofrh, overrides: {model: {frh: false}}}
  # P4: learning curve (100% comes from P1)
  - {priority: 4, models: [unet, segformer, prithvi], seeds: [1], fracs: [0.1, 0.25, 0.5]}
```

Create empty `flaresatformer/__init__.py`, `flaresatformer/data/__init__.py`, `flaresatformer/models/__init__.py` and `scripts/__init__.py`.

- [ ] **Step 4: Write the failing test**

```python
# file: tests/test_utils.py
import torch

from flaresatformer.utils import BAND_SETS, deep_merge, load_config, set_seed

MODELS = ["unet", "segformer", "segformer_scratch", "swin_unet", "transunet", "prithvi"]


def test_deep_merge_overrides_nested_without_mutating():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_merge(base, {"a": {"b": 10}})
    assert out == {"a": {"b": 10, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1


def test_every_model_config_loads_with_required_keys():
    for name in MODELS:
        cfg = load_config(name)
        assert cfg["model"]["name"] == name
        assert cfg["data"]["bands"] in BAND_SETS
        assert cfg["train"]["effective_batch_size"] % cfg["train"]["batch_size"] == 0


def test_overrides_win():
    assert load_config("unet", {"train": {"lr": 1.0}})["train"]["lr"] == 1.0


def test_set_seed_is_reproducible():
    set_seed(3)
    a = torch.rand(5)
    set_seed(3)
    assert torch.equal(a, torch.rand(5))
```

- [ ] **Step 5: Run it.** Run `~/venvs/flaresatformer/bin/python -m pytest tests/test_utils.py -q`. Expected: FAIL (`ModuleNotFoundError: flaresatformer.utils`).

- [ ] **Step 6: Implement**

```python
# file: flaresatformer/utils.py
"""Configuration, seeding and provenance helpers shared by every module."""
import copy
import os
import random
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

# Channel indices inside the 10-band patches: B1 B2 B3 B4 B5 B6 B7 B9 B10 B11
BAND_SETS = {"b267": [1, 5, 6], "hls6": [1, 2, 3, 4, 5, 6]}
HLS_NAMES = {1: "BLUE", 2: "GREEN", 3: "RED", 4: "NIR_NARROW", 5: "SWIR_1", 6: "SWIR_2"}
# Prithvi-EO-2.0 normalization (HLS reflectance x 10000), from terratorch prithvi_vit.PRITHVI_V2_MEAN/STD
PRITHVI_MEAN = {1: 1087.0, 2: 1342.0, 3: 1433.0, 4: 2734.0, 5: 1958.0, 6: 1363.0}
PRITHVI_STD = {1: 2248.0, 2: 2179.0, 3: 2178.0, 4: 1850.0, 5: 1242.0, 6: 1049.0}
LIBS = ("torch", "transformers", "timm", "terratorch", "peft", "numpy", "scikit-learn")


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_config(model, overrides=None, config_dir=CONFIG_DIR):
    config_dir = Path(config_dir)
    cfg = deep_merge(load_yaml(config_dir / "base.yaml"), load_yaml(config_dir / "models" / f"{model}.yaml"))
    return deep_merge(cfg, overrides)


def resolve_path(path):
    path = Path(os.path.expanduser(str(path)))
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def lib_versions():
    versions = {}
    for lib in LIBS:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versions[lib] = None
    return versions
```

- [ ] **Step 7: Run tests.** Expected: 4 passed.
- [ ] **Step 8: Commit.** `git add -A && git commit -m "feat: scaffold flaresatformer 2.0 package, configs and experiment matrix"`

---

### Task 2: Splits

**Files:**
- Create: `flaresatformer/data/splits.py`, `scripts/make_splits.py`, `tests/conftest.py`, `splits/fold{1..4}.csv` (generated)
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: `REPO_ROOT`, `resolve_path`, `load_config` (Task 1).
- Produces: `parse_scene(path) -> (path_row: str, scene: str)`, `load_sources(dataset_dir) -> DataFrame[tiff_file, mask_file, source, group, path_row, scene, country, continent]`, `make_fold_frames(df, num_folds=4, val_fraction=0.1, seed=42) -> list[DataFrame]` (adds `role` ∈ {train, val, test}), `subsample_train(frame, frac, seed=42) -> DataFrame`. Fixtures `synthetic_root`, `synthetic_splits`, `synthetic_cfg`.

- [ ] **Step 1: Write the shared synthetic fixtures**

```python
# file: tests/conftest.py
import numpy as np
import pandas as pd
import pytest
import tifffile

from flaresatformer.data.splits import load_sources, make_fold_frames
from flaresatformer.utils import load_config

CONTINENTS = ["Asia", "Africa", "Europe", "North America"]


def _write_patch(path, rng, blobs=()):
    img = (rng.random((256, 256, 10)) * 0.3).astype(np.float32)
    mask = np.zeros((256, 256), dtype=np.uint8)
    for r, c in blobs:
        img[r:r + 2, c:c + 2, 5] = 0.8
        img[r:r + 2, c:c + 2, 6] = 1.0
        mask[r:r + 2, c:c + 2] = 255
    tifffile.imwrite(path, img)
    mask_path = str(path).replace("_patches", "_masks").replace("_patch.tiff", "_mask.tiff")
    tifffile.imwrite(mask_path, mask)


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    """Tiny dataset with the real CSV layout and file naming (flare/urban/fire)."""
    root = tmp_path_factory.mktemp("flaresat")
    ds = root / "dataset"
    for sub in ("flare", "urban", "fire"):
        (ds / f"{sub}_patches").mkdir(parents=True)
        (ds / f"{sub}_masks").mkdir(parents=True)
    rng = np.random.default_rng(0)

    flare = []
    for i in range(48):
        p = i % 12
        name = f"fire_LC8{p:03d}0502019{100 + i // 12:03d}LGN00_{i}_{i}_patch.tiff"
        _write_patch(ds / "flare_patches" / name, rng, [tuple(rng.integers(10, 240, size=2))])
        flare.append({"cntry_name": f"Country{p % 4}", "continent": CONTINENTS[p % 4],
                      "tiff_file": f"dataset/flare_patches/{name}",
                      "mask_file": f"dataset/flare_masks/{name.replace('_patch.tiff', '_mask.tiff')}"})

    urban = []
    for j in range(16):
        city = f"City{j % 8}"
        name = f"urban_LC8{20 + j % 8:03d}0602019230LGN00_{city}_{j}_patch.tiff"
        _write_patch(ds / "urban_patches" / name, rng)
        urban.append({"city": city, "tiff_file": f"dataset/urban_patches/{name}",
                      "mask_file": f"dataset/urban_masks/{name.replace('_patch.tiff', '_mask.tiff')}"})

    fire = []
    for j in range(16):
        name = f"LC08_L1TP_{30 + j % 8:03d}070_20200901_20200901_01_RT_p{j:05d}.tiff"
        _write_patch(ds / "fire_patches" / name, rng)
        fire.append({"tiff_file": f"dataset/fire_patches/{name}", "mask_file": f"dataset/fire_masks/{name}"})

    pd.DataFrame(flare).to_csv(ds / "flare_dataset.csv", index=False)
    pd.DataFrame(urban).to_csv(ds / "urban_dataset.csv", index=False)
    pd.DataFrame(fire).to_csv(ds / "fire_dataset.csv", index=False)
    return root


@pytest.fixture(scope="session")
def synthetic_splits(synthetic_root):
    out = synthetic_root / "splits"
    out.mkdir(exist_ok=True)
    for k, frame in enumerate(make_fold_frames(load_sources(synthetic_root / "dataset")), start=1):
        frame.to_csv(out / f"fold{k}.csv", index=False)
    return out


@pytest.fixture
def synthetic_cfg(synthetic_root, synthetic_splits, tmp_path):
    return load_config("unet", {
        "paths": {"data_root": str(synthetic_root), "splits_dir": str(synthetic_splits),
                  "cache_dir": str(synthetic_root / "cache"), "outputs_dir": str(tmp_path / "outputs")},
        "data": {"num_workers": 0},
        "model": {"base_filters": 8},
        "train": {"epochs": 2, "patience": 5, "batch_size": 4, "effective_batch_size": 4, "amp": False},
        "eval": {"batch_size": 4},
    })
```

- [ ] **Step 2: Write the failing tests**

```python
# file: tests/test_splits.py
import pandas as pd
import pytest

from flaresatformer.data.splits import load_sources, make_fold_frames, parse_scene, subsample_train
from flaresatformer.utils import REPO_ROOT


def test_parse_scene_handles_all_naming_schemes():
    assert parse_scene("dataset/flare_patches/fire_LC81940382019006LGN00_14_23_patch.tiff") == ("194038", "194038_2019006")
    assert parse_scene("dataset/fire_patches/LC08_L1TP_134039_20200901_20200901_01_RT_p00053.tiff") == ("134039", "134039_20200901")
    assert parse_scene("dataset/urban_patches/urban_LC80090582019230LGN00_Cali_2_patch.tiff") == ("009058", "009058_2019230")


def test_parse_scene_rejects_unknown_names():
    with pytest.raises(ValueError):
        parse_scene("dataset/x/random.tiff")


@pytest.fixture(scope="module")
def frames(synthetic_root):
    return make_fold_frames(load_sources(synthetic_root / "dataset"))


def test_each_patch_is_test_in_exactly_one_fold(frames):
    counts = pd.concat([f[f.role == "test"] for f in frames])["tiff_file"].value_counts()
    assert set(counts) == {1}
    assert len(counts) == len(frames[0])


def test_test_groups_never_leak_into_train_or_val(frames):
    for f in frames:
        for source, sub in f.groupby("source"):
            assert set(sub[sub.role == "test"].group).isdisjoint(sub[sub.role != "test"].group), source


def test_val_path_rows_never_appear_in_train(frames):
    for f in frames:
        for source, sub in f.groupby("source"):
            assert set(sub[sub.role == "val"].path_row).isdisjoint(sub[sub.role == "train"].path_row), source
            assert (sub.role == "val").any(), source


def test_splits_are_deterministic(synthetic_root, frames):
    for a, b in zip(frames, make_fold_frames(load_sources(synthetic_root / "dataset"))):
        pd.testing.assert_frame_equal(a, b)


def test_subsample_is_nested_and_keeps_val_and_test(frames):
    f = frames[0]
    small, mid = subsample_train(f, 0.25), subsample_train(f, 0.5)
    assert set(small[small.role == "train"].tiff_file) <= set(mid[mid.role == "train"].tiff_file)
    assert len(small[small.role != "train"]) == len(f[f.role != "train"])
    n_train = (f.role == "train").sum()
    assert 0.25 * n_train <= (small.role == "train").sum() < n_train
    assert subsample_train(f, 1.0) is f


def test_real_csvs_parse_and_split():
    df = load_sources(REPO_ROOT / "dataset")
    assert len(df) == 11505 + 683 + 265
    first = make_fold_frames(df)[0]
    val_share = (first.role == "val").sum() / (first.role != "test").sum()
    assert 0.05 < val_share < 0.2
```

- [ ] **Step 3: Run it.** Run `python -m pytest tests/test_splits.py -q`. Expected: FAIL (`ModuleNotFoundError: flaresatformer.data.splits`).

- [ ] **Step 4: Implement**

```python
# file: flaresatformer/data/splits.py
"""Spatial cross-validation splits: test folds by continent/city/path-row, validation grouped by WRS-2 path/row."""
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

NUM_FOLDS = 4
SEED = 42
COLUMNS = ["tiff_file", "mask_file", "source", "group", "path_row", "scene", "country", "continent"]
_PRE_COLLECTION = re.compile(r"LC8(\d{3})(\d{3})(\d{7})")           # LC8 PPP RRR YYYYDDD LGN00
_COLLECTION_1 = re.compile(r"LC08_\w{4}_(\d{3})(\d{3})_(\d{8})")    # LC08_L1TP_PPPRRR_YYYYMMDD
_TEST_GROUP = {"flare": "continent", "urban": "city", "fire": None}  # None -> path_row


def parse_scene(path):
    name = str(path).rsplit("/", 1)[-1]
    match = _PRE_COLLECTION.search(name) or _COLLECTION_1.search(name)
    if match is None:
        raise ValueError(f"Cannot parse Landsat path/row from {name}")
    path_row = match.group(1) + match.group(2)
    return path_row, f"{path_row}_{match.group(3)}"


def load_sources(dataset_dir):
    frames = []
    for source, group_col in _TEST_GROUP.items():
        df = pd.read_csv(f"{dataset_dir}/{source}_dataset.csv", dtype=str, keep_default_na=False)
        parsed = df["tiff_file"].map(parse_scene)
        df["path_row"] = [p for p, _ in parsed]
        df["scene"] = [s for _, s in parsed]
        df["source"] = source
        df["group"] = df[group_col] if group_col else df["path_row"]
        df["country"] = df["cntry_name"] if "cntry_name" in df else ""
        df["continent"] = df["continent"] if "continent" in df else ""
        frames.append(df[COLUMNS])
    return pd.concat(frames, ignore_index=True)


def _test_folds(df, num_folds):
    fold = np.full(len(df), -1)
    for source in df["source"].unique():
        idx = np.flatnonzero(df["source"].to_numpy() == source)
        groups = df["group"].to_numpy()[idx]
        for k, (_, test) in enumerate(GroupKFold(n_splits=num_folds).split(idx, groups=groups)):
            fold[idx[test]] = k
    return fold


def make_fold_frames(df, num_folds=NUM_FOLDS, val_fraction=0.1, seed=SEED):
    test_fold = _test_folds(df, num_folds)
    frames = []
    for k in range(num_folds):
        role = np.where(test_fold == k, "test", "train").astype(object)
        for source in df["source"].unique():
            idx = np.flatnonzero((df["source"].to_numpy() == source) & (role == "train"))
            splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
            _, val = next(splitter.split(idx, groups=df["path_row"].to_numpy()[idx]))
            role[idx[val]] = "val"
        frame = df.copy()
        frame["role"] = role
        frames.append(frame)
    return frames


def subsample_train(frame, frac, seed=SEED):
    """Keeps whole path/rows of the training set until `frac` of its patches is reached; nested across fracs."""
    if frac >= 1.0:
        return frame
    train = frame[frame["role"] == "train"]
    groups = np.array(sorted(train["path_row"].unique()))
    np.random.default_rng(seed).shuffle(groups)
    counts = train.groupby("path_row").size().reindex(groups).to_numpy()
    keep = set(groups[: np.searchsorted(np.cumsum(counts), frac * len(train)) + 1])
    mask = (frame["role"] != "train") | frame["path_row"].isin(keep)
    return frame[mask].reset_index(drop=True)
```

- [ ] **Step 5: Run tests.** Expected: 8 passed.

- [ ] **Step 6: Script + generate the real splits**

```python
# file: scripts/make_splits.py
"""Generate the versioned fold files: python -m scripts.make_splits"""
import argparse

from flaresatformer.data.splits import load_sources, make_fold_frames
from flaresatformer.utils import resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--out-dir", default="splits")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = make_fold_frames(load_sources(resolve_path(args.dataset_dir)), val_fraction=args.val_fraction)
    for k, frame in enumerate(frames, start=1):
        frame.to_csv(out_dir / f"fold{k}.csv", index=False)
        test_groups = sorted(frame[(frame.role == "test") & (frame.source == "flare")].group.unique())
        print(f"fold{k}: flare test = {test_groups}")
        print(frame.groupby(["source", "role"]).size().unstack(fill_value=0), "\n")


if __name__ == "__main__":
    main()
```

Run `python -m scripts.make_splits`. Expected: four folds printed, with flare test groups Asia | Africa | Europe+Oceania | North+South America.

- [ ] **Step 7: Commit.** `git add -A && git commit -m "feat: spatial splits with path/row-grouped validation and nested subsampling"`

---

### Task 3: Band cache

**Files:**
- Create: `flaresatformer/data/cache.py`, `scripts/build_cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `synthetic_root`, `synthetic_splits` fixtures; `BAND_SETS`, `resolve_path`, `load_yaml`, `CONFIG_DIR`.
- Produces: `cache_key(files, bands) -> str`, `build_cache(files_df, bands, cache_dir, data_root, log_every=500) -> key`, `load_cache(cache_dir, key) -> (images (N,C,256,256) float16 memmap, masks (N,1,256,256) uint8 memmap, index {tiff_file: row})`.

- [ ] **Step 1: Write the failing tests**

```python
# file: tests/test_cache.py
import numpy as np
import pandas as pd
import pytest
import tifffile

from flaresatformer.data.cache import build_cache, cache_key, load_cache


@pytest.fixture
def files(synthetic_splits):
    return pd.read_csv(synthetic_splits / "fold1.csv", dtype=str)[["tiff_file", "mask_file"]]


def test_cache_matches_direct_tiff_read(files, synthetic_root, tmp_path):
    key = build_cache(files, [1, 5, 6], tmp_path, synthetic_root)
    images, masks, index = load_cache(tmp_path, key)
    row = files.iloc[3]
    raw = tifffile.imread(synthetic_root / row.tiff_file)[:, :, [1, 5, 6]].transpose(2, 0, 1)
    np.testing.assert_allclose(images[index[row.tiff_file]], raw, atol=1e-3)
    assert masks.dtype == np.uint8 and set(np.unique(masks)) <= {0, 1}
    gt = tifffile.imread(synthetic_root / row.mask_file) > 0
    np.testing.assert_array_equal(masks[index[row.tiff_file], 0], gt)


def test_cache_is_reused_not_rebuilt(files, synthetic_root, tmp_path):
    key = build_cache(files, [1, 5, 6], tmp_path, synthetic_root)
    stamp = (tmp_path / f"images_{key}.npy").stat().st_mtime_ns
    assert build_cache(files, [1, 5, 6], tmp_path, synthetic_root) == key
    assert (tmp_path / f"images_{key}.npy").stat().st_mtime_ns == stamp


def test_key_changes_with_bands_or_file_list(files):
    key = cache_key(files.tiff_file, [1, 5, 6])
    assert key != cache_key(files.tiff_file, [1, 2, 3, 4, 5, 6])
    assert key != cache_key(files.tiff_file[:-1], [1, 5, 6])


def test_wrong_patch_shape_is_rejected(tmp_path):
    tifffile.imwrite(tmp_path / "bad.tiff", np.zeros((128, 128, 10), np.float32))
    tifffile.imwrite(tmp_path / "bad_mask.tiff", np.zeros((128, 128), np.uint8))
    df = pd.DataFrame({"tiff_file": ["bad.tiff"], "mask_file": ["bad_mask.tiff"]})
    with pytest.raises(ValueError, match="expected shape"):
        build_cache(df, [1, 5, 6], tmp_path / "cache", tmp_path)


def test_non_finite_values_become_zero(tmp_path):
    img = np.full((256, 256, 10), 0.2, np.float32)
    img[0, 0, 5] = np.nan
    img[1, 1, 6] = np.inf
    tifffile.imwrite(tmp_path / "p.tiff", img)
    tifffile.imwrite(tmp_path / "m.tiff", np.zeros((256, 256), np.uint8))
    df = pd.DataFrame({"tiff_file": ["p.tiff"], "mask_file": ["m.tiff"]})
    key = build_cache(df, [1, 5, 6], tmp_path / "cache", tmp_path)
    images, _, _ = load_cache(tmp_path / "cache", key)
    assert np.isfinite(images[0].astype(np.float32)).all()
    assert images[0, 1, 0, 0] == 0 and images[0, 2, 1, 1] == 0
```

- [ ] **Step 2: Run it.** Expected: FAIL (`ModuleNotFoundError: flaresatformer.data.cache`).

- [ ] **Step 3: Implement**

```python
# file: flaresatformer/data/cache.py
"""One-off extraction of the needed bands from the 10-band TIFFs into float16 .npy memmaps."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tifffile

PATCH_SHAPE = (256, 256, 10)


def cache_key(files, bands):
    return hashlib.sha1(json.dumps([list(files), list(bands)]).encode()).hexdigest()[:12]


def _paths(cache_dir, key):
    cache_dir = Path(cache_dir)
    return cache_dir / f"images_{key}.npy", cache_dir / f"masks_{key}.npy", cache_dir / f"index_{key}.json"


def build_cache(files_df, bands, cache_dir, data_root, log_every=500):
    files = files_df["tiff_file"].tolist()
    mask_files = files_df["mask_file"].tolist()
    key = cache_key(files, bands)
    img_path, mask_path, index_path = _paths(cache_dir, key)
    if index_path.exists():  # written last, so its presence means the arrays are complete
        return key
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    data_root = Path(data_root)
    tmp_img, tmp_mask = img_path.with_suffix(".tmp.npy"), mask_path.with_suffix(".tmp.npy")
    n = len(files)
    images = np.lib.format.open_memmap(tmp_img, mode="w+", dtype=np.float16, shape=(n, len(bands), 256, 256))
    masks = np.lib.format.open_memmap(tmp_mask, mode="w+", dtype=np.uint8, shape=(n, 1, 256, 256))
    non_finite = {}
    for i, (tiff_file, mask_file) in enumerate(zip(files, mask_files)):
        img = tifffile.imread(data_root / tiff_file)
        if img.shape != PATCH_SHAPE:
            raise ValueError(f"{tiff_file}: expected shape {PATCH_SHAPE}, got {img.shape}")
        img = img[:, :, bands]
        bad = ~np.isfinite(img)
        if bad.any():
            non_finite[tiff_file] = int(bad.sum())
            img = np.where(bad, 0.0, img)
        images[i] = img.transpose(2, 0, 1).astype(np.float16)
        mask = tifffile.imread(data_root / mask_file)
        if mask.shape != PATCH_SHAPE[:2]:
            raise ValueError(f"{mask_file}: expected shape {PATCH_SHAPE[:2]}, got {mask.shape}")
        masks[i, 0] = mask > 0
        if log_every and (i + 1) % log_every == 0:
            print(f"cache {key}: {i + 1}/{n}", flush=True)
    images.flush()
    masks.flush()
    del images, masks
    os.replace(tmp_img, img_path)
    os.replace(tmp_mask, mask_path)
    with open(index_path, "w") as f:
        json.dump({"files": files, "bands": list(bands), "non_finite": non_finite}, f)
    if non_finite:
        print(f"cache {key}: replaced non-finite values with 0 in {len(non_finite)} patches")
    return key


def load_cache(cache_dir, key):
    img_path, mask_path, index_path = _paths(cache_dir, key)
    with open(index_path) as f:
        files = json.load(f)["files"]
    images = np.load(img_path, mmap_mode="r")
    masks = np.load(mask_path, mmap_mode="r")
    return images, masks, {name: row for row, name in enumerate(files)}
```

```python
# file: scripts/build_cache.py
"""Build the band caches used for training: python -m scripts.build_cache --bands b267 hls6"""
import argparse

import pandas as pd

from flaresatformer.data.cache import build_cache
from flaresatformer.utils import BAND_SETS, CONFIG_DIR, load_yaml, resolve_path


def main():
    paths = load_yaml(CONFIG_DIR / "base.yaml")["paths"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", nargs="+", default=["b267"], choices=sorted(BAND_SETS))
    parser.add_argument("--cache-dir", default=paths["cache_dir"])
    args = parser.parse_args()

    files = pd.read_csv(resolve_path(paths["splits_dir"]) / "fold1.csv", dtype=str)[["tiff_file", "mask_file"]]
    for name in args.bands:
        key = build_cache(files, BAND_SETS[name], resolve_path(args.cache_dir), resolve_path(paths["data_root"]))
        print(f"{name}: cache key {key}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests.** Expected: 5 passed.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: float16 band cache with integrity key and non-finite handling"`

---

### Task 4: Dataset and normalization

**Files:**
- Create: `flaresatformer/data/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `PRITHVI_MEAN`, `PRITHVI_STD` (Task 1).
- Produces: `compute_norm_stats(images, rows, chunk=64) -> (mean list, std list)`, `normalization_for(cfg, images, train_rows, bands) -> (mean, std, scale)`, `normalize(image_chw float32, mean, std, scale) -> ndarray`, `FlareDataset(images, masks, rows, mean, std, scale=1.0, augment=False)` whose items are `(image (C,256,256) float32, mask (1,256,256) float32, index int)`.

- [ ] **Step 1: Write the failing tests**

```python
# file: tests/test_dataset.py
import numpy as np
import torch

from flaresatformer.data.dataset import FlareDataset, compute_norm_stats, normalization_for


def _toy():
    rng = np.random.default_rng(0)
    images = rng.random((4, 3, 256, 256)).astype(np.float16)
    masks = (images[:, :1] > 0.9).astype(np.uint8)
    return images, masks


def test_norm_stats_match_numpy():
    images, _ = _toy()
    mean, std = compute_norm_stats(images, [0, 2, 3], chunk=2)
    ref = images[[0, 2, 3]].astype(np.float64)
    np.testing.assert_allclose(mean, ref.mean(axis=(0, 2, 3)), rtol=1e-6)
    np.testing.assert_allclose(std, ref.std(axis=(0, 2, 3)), rtol=1e-4)


def test_item_is_normalized_with_binary_mask():
    images, masks = _toy()
    x, y, i = FlareDataset(images, masks, [1], mean=[0.5] * 3, std=[2.0] * 3)[0]
    assert x.shape == (3, 256, 256) and y.shape == (1, 256, 256) and i == 0
    torch.testing.assert_close(x, (torch.from_numpy(images[1].astype(np.float32)) - 0.5) / 2.0)
    assert set(torch.unique(y).tolist()) <= {0.0, 1.0}


def test_augmentation_moves_image_and_mask_together():
    images, masks = _toy()
    ds = FlareDataset(images, masks, [0], mean=[0.0] * 3, std=[1.0] * 3, augment=True)
    torch.manual_seed(0)
    changed = False
    for _ in range(20):
        x, y, _ = ds[0]
        assert torch.equal((x[0] > 0.9).float(), y[0])  # the mask was derived from band 0 > 0.9
        changed |= not torch.equal(y, torch.from_numpy(masks[0].astype(np.float32)))
    assert changed


def test_prithvi_normalization_uses_hls_stats():
    mean, std, scale = normalization_for({"model": {"norm": "prithvi"}}, None, None, [1, 5, 6])
    assert mean == [1087.0, 1958.0, 1363.0] and std == [2248.0, 1242.0, 1049.0] and scale == 10000.0
```

- [ ] **Step 2: Run it.** Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# file: flaresatformer/data/dataset.py
"""Torch dataset over the band cache, with per-fold normalization and flip/rot90 augmentation."""
import numpy as np
import torch
from torch.utils.data import Dataset

from flaresatformer.utils import PRITHVI_MEAN, PRITHVI_STD


def compute_norm_stats(images, rows, chunk=64):
    rows = np.sort(np.asarray(rows))
    total = np.zeros(images.shape[1])
    total_sq = np.zeros(images.shape[1])
    count = 0
    for start in range(0, len(rows), chunk):
        batch = np.asarray(images[rows[start:start + chunk]], dtype=np.float64)
        total += batch.sum(axis=(0, 2, 3))
        total_sq += np.square(batch).sum(axis=(0, 2, 3))
        count += batch.shape[0] * batch.shape[2] * batch.shape[3]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 0.0))
    return mean.tolist(), np.maximum(std, 1e-6).tolist()


def normalization_for(cfg, images, train_rows, bands):
    """Returns (mean, std, scale), applied as (x * scale - mean) / std."""
    if cfg["model"].get("norm", "dataset") == "prithvi":
        return [PRITHVI_MEAN[b] for b in bands], [PRITHVI_STD[b] for b in bands], 10000.0
    mean, std = compute_norm_stats(images, train_rows)
    return mean, std, 1.0


def normalize(image, mean, std, scale=1.0):
    mean = np.asarray(mean, dtype=np.float32)[:, None, None]
    std = np.asarray(std, dtype=np.float32)[:, None, None]
    return (image.astype(np.float32) * np.float32(scale) - mean) / std


class FlareDataset(Dataset):
    def __init__(self, images, masks, rows, mean, std, scale=1.0, augment=False):
        self.images, self.masks = images, masks
        self.rows = np.asarray(rows)
        self.mean, self.std, self.scale = mean, std, scale
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = normalize(self.images[row], self.mean, self.std, self.scale)
        mask = self.masks[row].astype(np.float32)
        if self.augment:
            # torch RNG is seeded per DataLoader worker from the loader generator -> reproducible runs
            k = int(torch.randint(4, (1,)))
            image, mask = np.rot90(image, k, axes=(1, 2)), np.rot90(mask, k, axes=(1, 2))
            if torch.rand(1).item() < 0.5:
                image, mask = image[:, :, ::-1], mask[:, :, ::-1]
            if torch.rand(1).item() < 0.5:
                image, mask = image[:, ::-1, :], mask[:, ::-1, :]
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(mask)), i
```

- [ ] **Step 4: Run tests.** Expected: 4 passed.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: dataset with per-fold normalization and geometric augmentation"`

---

### Task 5: Losses and metrics

**Files:**
- Create: `flaresatformer/losses.py`, `flaresatformer/metrics.py`
- Test: `tests/test_losses.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces: `FocalLoss(alpha, gamma)`, `BatchDiceLoss(smooth=1.0)`, `FocalDiceLoss(alpha=0.25, gamma=2.0, dice_weight=1.0)`, all with `(logits, targets) -> scalar`. Also `pixel_counts(pred, gt) -> (tp, fp, fn)`, `prf(tp, fp, fn) -> {precision, recall, f1, iou}`, `object_counts(pred, gt, tol=1) -> {gt_objects, gt_detected, pred_objects, pred_fp_objects}`, `object_prf(gt_objects, gt_detected, pred_objects, pred_fp_objects) -> {precision, recall, f1}`, `threshold_grid(start, stop, step) -> ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
# file: tests/test_losses.py
import math

import torch

from flaresatformer.losses import BatchDiceLoss, FocalDiceLoss, FocalLoss


def test_focal_alpha_weights_classes_differently():
    loss, zero = FocalLoss(alpha=0.25, gamma=2.0), torch.zeros(1, 1, 1, 1)
    assert math.isclose(loss(zero, torch.ones(1, 1, 1, 1)).item(), 0.25 * 0.25 * math.log(2), rel_tol=1e-6)
    assert math.isclose(loss(zero, torch.zeros(1, 1, 1, 1)).item(), 0.75 * 0.25 * math.log(2), rel_tol=1e-6)


def test_batch_dice_is_finite_for_empty_masks_and_confident_negatives():
    value = BatchDiceLoss()(torch.full((2, 1, 8, 8), -20.0), torch.zeros(2, 1, 8, 8))
    assert torch.isfinite(value) and value.item() < 1e-3


def test_batch_dice_perfect_and_inverted():
    t = torch.zeros(1, 1, 8, 8)
    t[0, 0, 2:4, 2:4] = 1
    assert BatchDiceLoss(smooth=0)(t * 40 - 20, t).item() < 1e-6
    assert BatchDiceLoss(smooth=0)(20 - t * 40, t).item() > 0.99


def test_focal_dice_backpropagates():
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    FocalDiceLoss()(logits, (torch.rand(2, 1, 8, 8) > 0.9).float()).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
```

```python
# file: tests/test_metrics.py
import numpy as np

from flaresatformer.metrics import object_counts, object_prf, pixel_counts, prf, threshold_grid


def _mask(*points, shape=(16, 16)):
    m = np.zeros(shape, bool)
    for r, c in points:
        m[r, c] = True
    return m


def test_pixel_counts_and_prf():
    assert pixel_counts(_mask((1, 1), (8, 8)), _mask((1, 1), (1, 2), (5, 5))) == (1, 1, 2)
    assert prf(1, 1, 2) == {"precision": 0.5, "recall": 1 / 3, "f1": 0.4, "iou": 0.25}


def test_prf_is_zero_not_nan_on_empty_inputs():
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    assert object_prf(0, 0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_object_detection_with_one_pixel_tolerance():
    gt = _mask((4, 4), (4, 5), (12, 12))   # two objects
    pred = _mask((5, 6), (0, 15))          # first touches object 1 diagonally; second is an isolated FP
    assert object_counts(pred, gt, tol=1) == {"gt_objects": 2, "gt_detected": 1, "pred_objects": 2, "pred_fp_objects": 1}


def test_object_detection_strict_without_tolerance():
    gt, pred = _mask((4, 4)), _mask((4, 5))
    assert object_counts(pred, gt, tol=0)["gt_detected"] == 0
    assert object_counts(pred, gt, tol=1)["gt_detected"] == 1


def test_empty_masks():
    z = np.zeros((8, 8), bool)
    assert object_counts(z, z) == {"gt_objects": 0, "gt_detected": 0, "pred_objects": 0, "pred_fp_objects": 0}


def test_threshold_grid_contains_half():
    g = threshold_grid(0.05, 0.95, 0.05)
    assert len(g) == 19 and 0.5 in g and g[0] == 0.05 and g[-1] == 0.95
```

- [ ] **Step 2: Run them.** Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# file: flaresatformer/losses.py
"""Focal + batch-level Dice loss shared by every model."""
import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Binary focal loss with class-dependent alpha_t: alpha for positives, 1 - alpha for negatives."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        logits, targets = logits.float(), targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class BatchDiceLoss(nn.Module):
    """Soft Dice over the whole batch: all-negative patches (urban, fire) never produce degenerate per-image terms."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs, targets = torch.sigmoid(logits.float()), targets.float()
        intersection = (probs * targets).sum()
        return 1 - (2 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice = BatchDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.focal(logits, targets) + self.dice_weight * self.dice(logits, targets)
```

```python
# file: flaresatformer/metrics.py
"""Pixel- and object-level metrics from binary masks or summed counts."""
import numpy as np
from scipy import ndimage

STRUCT = np.ones((3, 3), dtype=bool)  # 8-connectivity


def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def pixel_counts(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    return int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())


def prf(tp, fp, fn):
    return {"precision": _safe_div(tp, tp + fp), "recall": _safe_div(tp, tp + fn),
            "f1": _safe_div(2 * tp, 2 * tp + fp + fn), "iou": _safe_div(tp, tp + fp + fn)}


def object_counts(pred, gt, tol=1):
    """A GT flare is detected if a predicted pixel lies within `tol` px of it; a predicted object far from every GT is a FP."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    gt_lab, n_gt = ndimage.label(gt, STRUCT)
    pred_lab, n_pred = ndimage.label(pred, STRUCT)
    pred_zone = ndimage.binary_dilation(pred, STRUCT, iterations=tol) if tol and n_pred else pred
    gt_zone = ndimage.binary_dilation(gt, STRUCT, iterations=tol) if tol and n_gt else gt
    detected = np.unique(gt_lab[pred_zone & (gt_lab > 0)]).size
    matched = np.unique(pred_lab[gt_zone & (pred_lab > 0)]).size
    return {"gt_objects": int(n_gt), "gt_detected": int(detected),
            "pred_objects": int(n_pred), "pred_fp_objects": int(n_pred - matched)}


def object_prf(gt_objects, gt_detected, pred_objects, pred_fp_objects):
    precision = _safe_div(pred_objects - pred_fp_objects, pred_objects)
    recall = _safe_div(gt_detected, gt_objects)
    return {"precision": precision, "recall": recall, "f1": _safe_div(2 * precision * recall, precision + recall)}


def threshold_grid(start, stop, step):
    return np.round(np.arange(start, stop + step / 2, step), 4)
```

- [ ] **Step 4: Run tests.** Expected: 10 passed.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: focal(alpha_t)+batch dice loss and pixel/object metrics"`

---

### Task 6: Models

**Files:**
- Create: `flaresatformer/models/frh.py`, `unet.py`, `segformer.py`, `swin_unet.py`, `transunet.py`, `prithvi.py`, `factory.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `BAND_SETS`, `HLS_NAMES` (Task 1).
- Produces: `build_model(cfg) -> nn.Module` (forward `(B,C,256,256) -> (B,1,256,256)` logits; every model has `.encoder_parameters() -> list[Parameter]` and `.head`). Also `param_groups(model, lr, encoder_lr_mult, pretrained) -> list[dict]` (encoder group first), `conv_bn_relu`, `FullResHead`, `UpsampleHead`, `make_head(in_ch, dec_ch, frh)`.

- [ ] **Step 1: Write the failing tests**

```python
# file: tests/test_models.py
import pytest
import torch

from flaresatformer.models.factory import build_model, param_groups
from flaresatformer.models.frh import FullResHead, UpsampleHead
from flaresatformer.utils import load_config


def _cfg(name, bands="b267", **model):
    return load_config(name, {"data": {"bands": bands},
                              "model": {"pretrained": False, "grad_checkpointing": False, **model}})


@pytest.mark.parametrize("name", ["unet", "segformer", "segformer_scratch", "swin_unet", "transunet"])
@pytest.mark.parametrize("bands", ["b267", "hls6"])
def test_forward_returns_full_resolution_logits(name, bands):
    model = build_model(_cfg(name, bands)).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3 if bands == "b267" else 6, 256, 256))
    assert out.shape == (2, 1, 256, 256)


def test_frh_can_be_disabled():
    assert isinstance(build_model(_cfg("segformer", frh=False)).head, UpsampleHead)
    assert isinstance(build_model(_cfg("segformer")).head, FullResHead)


def test_param_groups_split_encoder_from_head():
    model = build_model(_cfg("swin_unet"))
    groups = param_groups(model, lr=1e-3, encoder_lr_mult=0.1, pretrained=True)
    assert [g["lr"] for g in groups] == pytest.approx([1e-4, 1e-3])
    assert sum(p.numel() for p in groups[0]["params"]) == sum(p.numel() for p in model.encoder.parameters())
    assert len(param_groups(model, 1e-3, 0.1, pretrained=False)) == 1


def test_transunet_backward_with_grad_checkpointing():
    model = build_model(_cfg("transunet", grad_checkpointing=True)).train()
    model(torch.randn(2, 3, 256, 256)).mean().backward()
    assert model.head.fuse[-1].weight.grad is not None


@pytest.mark.slow
@pytest.mark.parametrize("bands", ["b267", "hls6"])
def test_prithvi_forward_and_only_lora_trainable_in_encoder(bands):
    model = build_model(_cfg("prithvi", bands)).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3 if bands == "b267" else 6, 256, 256)).shape == (2, 1, 256, 256)
    trainable = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
```

- [ ] **Step 2: Run it.** Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the shared heads**

```python
# file: flaresatformer/models/frh.py
"""Output heads shared by the transformer models (spec 4.1)."""
import torch
import torch.nn.functional as F
from torch import nn


def conv_bn_relu(in_ch, out_ch, kernel_size=3):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class FullResHead(nn.Module):
    """Fuses decoder features with a shallow full-resolution branch over the raw input.

    Decoder features are projected to `width` channels before upsampling to keep 256x256 activations small.
    """

    def __init__(self, in_ch, dec_ch, width=32, out_ch=1):
        super().__init__()
        self.image = nn.Sequential(conv_bn_relu(in_ch, width), conv_bn_relu(width, width))
        self.dec_proj = conv_bn_relu(dec_ch, width, kernel_size=1)
        self.dec = conv_bn_relu(width, width)
        self.fuse = nn.Sequential(conv_bn_relu(2 * width, width), nn.Conv2d(width, out_ch, 1))

    def forward(self, x, dec):
        dec = F.interpolate(self.dec_proj(dec), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.image(x), self.dec(dec)], dim=1))


class UpsampleHead(nn.Module):
    """FRH ablation: 1x1 classifier on decoder features, then bilinear upsampling to the input size."""

    def __init__(self, dec_ch, out_ch=1):
        super().__init__()
        self.classifier = nn.Conv2d(dec_ch, out_ch, 1)

    def forward(self, x, dec):
        return F.interpolate(self.classifier(dec), size=x.shape[-2:], mode="bilinear", align_corners=False)


def make_head(in_ch, dec_ch, frh=True):
    return FullResHead(in_ch, dec_ch) if frh else UpsampleHead(dec_ch)
```

- [ ] **Step 4: Implement the models**

```python
# file: flaresatformer/models/unet.py
"""UNet baseline, kept identical to the FlareSat/FlareSatFormer v1 architecture for continuity."""
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels, base_filters=32, dropout=0.1, out_channels=1):
        super().__init__()
        f = base_filters
        self.c1, self.c2, self.c3, self.c4 = ConvBlock(in_channels, f), ConvBlock(f, 2 * f), ConvBlock(2 * f, 4 * f), ConvBlock(4 * f, 8 * f)
        self.c5 = ConvBlock(8 * f, 16 * f)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(dropout)
        self.u6, self.c6 = nn.ConvTranspose2d(16 * f, 8 * f, 2, stride=2), ConvBlock(16 * f, 8 * f)
        self.u7, self.c7 = nn.ConvTranspose2d(8 * f, 4 * f, 2, stride=2), ConvBlock(8 * f, 4 * f)
        self.u8, self.c8 = nn.ConvTranspose2d(4 * f, 2 * f, 2, stride=2), ConvBlock(4 * f, 2 * f)
        self.u9, self.c9 = nn.ConvTranspose2d(2 * f, f, 2, stride=2), ConvBlock(2 * f, f)
        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.drop(self.pool(c1)))
        c3 = self.c3(self.drop(self.pool(c2)))
        c4 = self.c4(self.drop(self.pool(c3)))
        c5 = self.c5(self.drop(self.pool(c4)))
        c6 = self.c6(self.drop(torch.cat([self.u6(c5), c4], dim=1)))
        c7 = self.c7(self.drop(torch.cat([self.u7(c6), c3], dim=1)))
        c8 = self.c8(self.drop(torch.cat([self.u8(c7), c2], dim=1)))
        c9 = self.c9(self.drop(torch.cat([self.u9(c8), c1], dim=1)))
        return self.head(c9)

    def encoder_parameters(self):
        return []
```

```python
# file: flaresatformer/models/segformer.py
"""SegFormer MiT-B2 (Hugging Face) with the decode-head classifier replaced by the shared output head."""
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from transformers.utils import logging as hf_logging

from flaresatformer.models.frh import make_head

hf_logging.set_verbosity_error()


class SegFormerSeg(nn.Module):
    def __init__(self, in_channels, pretrained=True, frh=True, checkpoint="nvidia/mit-b2"):
        super().__init__()
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                checkpoint, num_labels=1, num_channels=in_channels, ignore_mismatched_sizes=True)
        else:
            config = SegformerConfig.from_pretrained(checkpoint, num_labels=1, num_channels=in_channels)
            self.net = SegformerForSemanticSegmentation(config)
        self.net.decode_head.classifier = nn.Identity()  # logits become the fused 1/4-resolution features
        self.head = make_head(in_channels, self.net.config.decoder_hidden_size, frh)

    def forward(self, x):
        return self.head(x, self.net(pixel_values=x).logits)

    def encoder_parameters(self):
        return list(self.net.segformer.parameters())
```

```python
# file: flaresatformer/models/swin_unet.py
"""SwinV2-T encoder (timm, ImageNet-1k, window 8 for 256 px) with a UNet-style conv decoder and the shared head."""
import timm
import torch
import torch.nn.functional as F
from torch import nn

from flaresatformer.models.frh import conv_bn_relu, make_head


class SwinUNet(nn.Module):
    def __init__(self, in_channels, pretrained=True, frh=True, encoder="swinv2_tiny_window8_256", dec_channels=(256, 128, 64)):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True, in_chans=in_channels)
        fmt = getattr(self.encoder, "output_fmt", "NCHW")
        self.nhwc = getattr(fmt, "value", fmt) == "NHWC"
        channels = self.encoder.feature_info.channels()  # strides 4, 8, 16, 32
        ins = [channels[3], *dec_channels[:-1]]
        skips = channels[2::-1]
        self.blocks = nn.ModuleList(
            nn.Sequential(conv_bn_relu(i + s, o), conv_bn_relu(o, o)) for i, s, o in zip(ins, skips, dec_channels))
        self.head = make_head(in_channels, dec_channels[-1], frh)

    def forward(self, x):
        feats = self.encoder(x)
        if self.nhwc:
            feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        y = feats[-1]
        for skip, block in zip(feats[-2::-1], self.blocks):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))
        return self.head(x, y)

    def encoder_parameters(self):
        return list(self.encoder.parameters())

    def set_grad_checkpointing(self):
        self.encoder.set_grad_checkpointing(True)
```

```python
# file: flaresatformer/models/transunet.py
"""TransUNet: R50-ViT-B/16 hybrid encoder (timm, ImageNet-21k) + cascaded upsampler with ResNet skips."""
import timm
import torch
import torch.nn.functional as F
from torch import nn

from flaresatformer.models.frh import conv_bn_relu, make_head

SKIP_CHANNELS = (512, 256, 64)  # ResNet stage 1 (1/8), stage 0 (1/4), stem (1/2)


class TransUNet(nn.Module):
    def __init__(self, in_channels, pretrained=True, frh=True, encoder="vit_base_r50_s16_224.orig_in21k",
                 img_size=256, dec_channels=(256, 128, 64, 16)):
        super().__init__()
        self.vit = timm.create_model(encoder, pretrained=pretrained, img_size=img_size, num_classes=0, in_chans=in_channels)
        self.grid = img_size // 16
        self._skips = {}
        backbone = self.vit.patch_embed.backbone
        for name, module in (("s8", backbone.stages[1]), ("s4", backbone.stages[0]), ("s2", backbone.stem.norm)):
            module.register_forward_hook(self._store(name))
        self.conv_more = conv_bn_relu(self.vit.embed_dim, 512)
        ins = (512, *dec_channels[:2])
        self.blocks = nn.ModuleList(
            nn.Sequential(conv_bn_relu(i + s, o), conv_bn_relu(o, o)) for i, s, o in zip(ins, SKIP_CHANNELS, dec_channels[:3]))
        self.last = nn.Sequential(conv_bn_relu(dec_channels[2], dec_channels[3]), conv_bn_relu(dec_channels[3], dec_channels[3]))
        self.head = make_head(in_channels, dec_channels[3], frh)

    def _store(self, name):
        def hook(module, inputs, output):
            self._skips[name] = output
        return hook

    def forward(self, x):
        tokens = self.vit.forward_features(x)[:, self.vit.num_prefix_tokens:]
        b, _, d = tokens.shape
        y = self.conv_more(tokens.transpose(1, 2).reshape(b, d, self.grid, self.grid))
        for name, block in zip(("s8", "s4", "s2"), self.blocks):
            skip = self._skips[name]
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))
        self._skips.clear()
        y = self.last(F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.head(x, y)

    def encoder_parameters(self):
        return list(self.vit.parameters())

    def set_grad_checkpointing(self):
        self.vit.set_grad_checkpointing(True)
```

```python
# file: flaresatformer/models/prithvi.py
"""Prithvi-EO-2.0-300M (terratorch) with LoRA adapters, UperNet decoder and the shared output head."""
import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from torch.utils.checkpoint import checkpoint

from flaresatformer.models.frh import make_head
from flaresatformer.utils import HLS_NAMES

LORA_TARGETS = r".*blocks\.\d+\.attn\.(qkv|proj)"
NECKS = [{"name": "SelectIndices", "indices": [5, 11, 17, 23]},
         {"name": "ReshapeTokensToImage"},
         {"name": "LearnedInterpolateToPyramidal"}]


class _Checkpointed(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.block, *args, use_reentrant=False, **kwargs)
        return self.block(*args, **kwargs)


class PrithviSeg(nn.Module):
    def __init__(self, bands, pretrained=True, frh=True, lora_r=16, lora_alpha=32, freeze_encoder=False):
        super().__init__()
        from terratorch.datasets import HLSBands
        from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory

        net = EncoderDecoderFactory().build_model(
            task="segmentation", backbone="prithvi_eo_v2_300", backbone_pretrained=pretrained,
            backbone_bands=[HLSBands[HLS_NAMES[b]] for b in bands], backbone_num_frames=1,
            decoder="UperNetDecoder", decoder_channels=256, num_classes=1, necks=NECKS)
        self.encoder, self.neck, self.decoder = net.encoder, net.neck, net.decoder
        self.encoder.requires_grad_(False)
        if not freeze_encoder:
            config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=LORA_TARGETS)
            self.encoder = inject_adapter_in_model(config, self.encoder)
            for name, param in self.encoder.named_parameters():
                param.requires_grad = "lora_" in name
        self.head = make_head(len(bands), self.decoder.out_channels, frh)

    def forward(self, x):
        feats = self.neck(self.encoder(x), image_size=tuple(x.shape[-2:]))
        return self.head(x, self.decoder(list(feats)))

    def encoder_parameters(self):
        return []  # backbone frozen; LoRA adapters are new parameters and train at the full lr

    def set_grad_checkpointing(self):
        self.encoder.blocks = nn.ModuleList(_Checkpointed(block) for block in self.encoder.blocks)
```

```python
# file: flaresatformer/models/factory.py
"""Builds models from the resolved config and splits parameters into lr groups."""
from flaresatformer.utils import BAND_SETS


def build_model(cfg):
    m = cfg["model"]
    bands = BAND_SETS[cfg["data"]["bands"]]
    channels = len(bands)
    name = m["name"]
    if name == "unet":
        from flaresatformer.models.unet import UNet
        model = UNet(in_channels=channels, base_filters=m.get("base_filters", 32))
    elif name in ("segformer", "segformer_scratch"):
        from flaresatformer.models.segformer import SegFormerSeg
        model = SegFormerSeg(channels, pretrained=m["pretrained"], frh=m["frh"])
    elif name == "swin_unet":
        from flaresatformer.models.swin_unet import SwinUNet
        model = SwinUNet(channels, pretrained=m["pretrained"], frh=m["frh"])
    elif name == "transunet":
        from flaresatformer.models.transunet import TransUNet
        model = TransUNet(channels, pretrained=m["pretrained"], frh=m["frh"])
    elif name == "prithvi":
        from flaresatformer.models.prithvi import PrithviSeg
        model = PrithviSeg(bands, pretrained=m["pretrained"], frh=m["frh"], lora_r=m.get("lora_r", 16),
                           lora_alpha=m.get("lora_alpha", 32), freeze_encoder=m.get("freeze_encoder", False))
    else:
        raise ValueError(f"Unknown model: {name}")
    if m.get("grad_checkpointing"):
        model.set_grad_checkpointing()
    return model


def param_groups(model, lr, encoder_lr_mult, pretrained):
    encoder_ids = {id(p) for p in model.encoder_parameters()} if pretrained else set()
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in trainable if id(p) in encoder_ids], "lr": lr * encoder_lr_mult},
              {"params": [p for p in trainable if id(p) not in encoder_ids], "lr": lr}]
    return [g for g in groups if g["params"]]
```

- [ ] **Step 5: Run tests.** Run `python -m pytest tests/test_models.py -q`. Expected: 13 passed, 2 deselected. Then run the slow Prithvi test once: `python -m pytest tests/test_models.py -m slow -q`. Expected: 2 passed (downloads weights the first time).
- [ ] **Step 6: Commit.** `git add -A && git commit -m "feat: model zoo (UNet, SegFormer, Swin-UNet, TransUNet, Prithvi+LoRA) with full-resolution head"`

---

### Task 7: Training engine

**Files:**
- Create: `flaresatformer/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `FocalDiceLoss` (Task 5), `prf`, `pixel_counts`, `object_counts` (Task 5), `param_groups` (Task 6).
- Produces: `warmup_cosine(total_steps, warmup_frac) -> callable`, `autocast(device, amp)`, `evaluate_counts(model, loader, thresholds, device, amp, loss_fn=None) -> (counts ndarray (T,3) [tp,fp,fn], mean_loss)`, `fit(model, train_loader, val_loader, cfg, run_dir, device, generator, stop_after=None) -> state dict {epoch, best_f1, bad_epochs, history}`, `collect_patch_stats(model, loader, meta, thresholds: dict, device, amp, tol) -> DataFrame`, `measure_throughput(model, in_channels, device, amp, batch_size=16, iters=10) -> float`. Loaders yield `(x, y, index)`.

- [ ] **Step 1: Write the failing tests**

```python
# file: tests/test_engine.py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flaresatformer.engine import collect_patch_stats, evaluate_counts, fit, warmup_cosine
from flaresatformer.utils import load_config, set_seed

CPU = torch.device("cpu")


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.Dropout2d(0.3), nn.ReLU(), nn.Conv2d(4, 1, 1))

    def forward(self, x):
        return self.net(x)

    def encoder_parameters(self):
        return []


class Identity(nn.Module):
    def forward(self, x):
        return x[:, :1]


def _loaders(seed):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(12, 3, 16, 16, generator=g)
    ds = TensorDataset(x, (x[:, 2:3] > 1.0).float(), torch.arange(12))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=4, shuffle=True, generator=gen, drop_last=True), DataLoader(ds, batch_size=4), gen


def _cfg(epochs, patience=100, lr=3e-2):
    return load_config("unet", {"model": {"pretrained": False}, "train": {
        "epochs": epochs, "patience": patience, "batch_size": 4, "effective_batch_size": 8, "amp": False, "lr": lr}})


def test_warmup_cosine_shape():
    f = warmup_cosine(100, 0.1)
    assert f(0) == 0.1 and f(9) == 1.0 and abs(f(100)) < 1e-9 and 0 < f(55) < 1


def test_fit_learns_and_writes_artifacts(tmp_path):
    set_seed(0)
    train, val, gen = _loaders(0)
    state = fit(Tiny(), train, val, _cfg(epochs=40), tmp_path, CPU, gen)
    assert state["epoch"] == 40
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
    assert pd.read_csv(tmp_path / "history.csv").val_f1.max() > 0.5


def test_early_stopping(tmp_path):
    set_seed(0)
    train, val, gen = _loaders(0)
    state = fit(Tiny(), train, val, _cfg(epochs=50, patience=2, lr=0.0), tmp_path, CPU, gen)
    assert state["epoch"] < 50 and state["bad_epochs"] == 2


def test_resume_matches_uninterrupted_run(tmp_path):
    set_seed(0)
    a = Tiny()
    train, val, gen = _loaders(0)
    fit(a, train, val, _cfg(epochs=4), tmp_path / "a", CPU, gen)

    set_seed(0)
    b = Tiny()
    train, val, gen = _loaders(0)
    fit(b, train, val, _cfg(epochs=4), tmp_path / "b", CPU, gen, stop_after=2)  # "crash" after 2 epochs
    set_seed(123)  # a fresh process: everything must come back from last.pt
    resumed = Tiny()
    train, val, gen = _loaders(999)
    fit(resumed, train, val, _cfg(epochs=4), tmp_path / "b", CPU, gen)

    for pa, pb in zip(a.state_dict().values(), resumed.state_dict().values()):
        torch.testing.assert_close(pa, pb)
    drop = ["epoch_time_s"]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "a" / "history.csv").drop(columns=drop),
                                  pd.read_csv(tmp_path / "b" / "history.csv").drop(columns=drop))


def _fixed_loader():
    x = torch.tensor([-1.0, 0.0, 2.0, 3.0]).view(1, 1, 2, 2).repeat(1, 3, 1, 1)
    y = torch.tensor([0.0, 1.0, 1.0, 0.0]).view(1, 1, 2, 2)
    return DataLoader(TensorDataset(x, y, torch.arange(1)), batch_size=1)


def test_evaluate_counts_matches_manual():
    # sigmoid -> [0.27, 0.50, 0.88, 0.95]; >0.5 -> [0,0,1,1]; >0.9 -> [0,0,0,1]
    counts, _ = evaluate_counts(Identity(), _fixed_loader(), [0.5, 0.9], CPU, amp=False)
    np.testing.assert_array_equal(counts, [[1, 1, 1], [0, 1, 2]])


def test_collect_patch_stats_one_row_per_patch_and_threshold():
    df = collect_patch_stats(Identity(), _fixed_loader(), [{"tiff_file": "a", "source": "flare"}],
                             {"t050": 0.5, "tuned": 0.9}, CPU, amp=False, tol=1)
    assert list(df.threshold_name) == ["t050", "tuned"]
    assert df.iloc[0][["tp", "fp", "fn"]].tolist() == [1, 1, 1]
    assert df.iloc[0].tiff_file == "a"
```

- [ ] **Step 2: Run it.** Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# file: flaresatformer/engine.py
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
```

- [ ] **Step 4: Run tests.** Expected: 7 passed.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: training engine with exact resume, early stopping and evaluation passes"`

---

### Task 8: Run orchestration, train CLI and experiment queue

**Files:**
- Create: `flaresatformer/run.py`, `flaresatformer/experiments.py`, `scripts/train.py`, `scripts/run_queue.py`
- Test: `tests/test_run.py`, `tests/test_experiments.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `default_run_dir(cfg, fold, seed, frac=1.0, tag=None, kind="main", smoke=False) -> Path`, `run_experiment(cfg, fold, seed, frac=1.0, run_dir=None, tag=None, smoke=False, evaluate_test=True) -> Path` (writes `config.yaml, norm.json, history.csv, best.pt, last.pt, threshold.json, test_patches.csv, cost.json, DONE`). Also `Job` (fields `priority, model, fold, seed, frac, kind, lr, epochs, tag, overrides`; methods `config()`, `run_dir()`, `command()`), `expand(spec) -> list[Job]`, `load_experiments(path) -> list[Job]`, `select_jobs(jobs, max_priority=None, only_priority=None, allow_untuned=False) -> (runnable, done, blocked)`, each a list of `(Job, Path)`.

- [ ] **Step 1: Write the failing tests**

```python
# file: tests/test_run.py
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
```

```python
# file: tests/test_experiments.py
from collections import Counter

from flaresatformer.experiments import expand, load_experiments, select_jobs


def test_full_matrix_size_and_order():
    jobs = load_experiments()
    assert Counter(j.priority for j in jobs) == {0: 18, 1: 24, 2: 48, 3: 8, 4: 36}
    assert [j.priority for j in jobs] == sorted(j.priority for j in jobs)


def test_every_job_has_its_own_run_dir():
    jobs = load_experiments()
    assert len({j.run_dir() for j in jobs}) == len(jobs)


def test_lr_search_jobs_skip_test_and_pin_lr():
    job = next(j for j in load_experiments() if j.kind == "lr_search")
    cmd = job.command()
    assert "--no-test" in cmd and "--lr" in cmd and job.epochs == 30 and job.fold == 1


def _spec(tmp_path):
    out = {"paths": {"outputs_dir": str(tmp_path)}}
    return {"models": ["unet"], "folds": [1, 2], "experiments": [
        {"priority": 0, "kind": "lr_search", "models": "all", "folds": [1], "seeds": [1], "lrs": [1e-3], "epochs": 30, "overrides": out},
        {"priority": 1, "models": "all", "seeds": [1], "overrides": out}]}


def test_untuned_models_are_blocked_from_main_runs(tmp_path):
    runnable, done, blocked = select_jobs(expand(_spec(tmp_path)))
    assert [j.kind for j, _ in runnable] == ["lr_search"] and len(blocked) == 2 and not done
    runnable, _, blocked = select_jobs(expand(_spec(tmp_path)), allow_untuned=True)
    assert len(runnable) == 3 and not blocked


def test_done_runs_are_skipped(tmp_path):
    jobs = expand(_spec(tmp_path))
    jobs[0].run_dir().mkdir(parents=True)
    (jobs[0].run_dir() / "DONE").write_text("x")
    runnable, done, _ = select_jobs(jobs, allow_untuned=True)
    assert len(done) == 1 and len(runnable) == 2
```

- [ ] **Step 2: Run them.** Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# file: flaresatformer/run.py
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
```

```python
# file: flaresatformer/experiments.py
"""Expands configs/experiments.yaml into concrete jobs and decides which ones may run."""
import json
import sys
from dataclasses import dataclass

from flaresatformer.run import default_run_dir
from flaresatformer.utils import CONFIG_DIR, load_config, load_yaml


@dataclass(frozen=True)
class Job:
    priority: int
    model: str
    fold: int
    seed: int
    frac: float = 1.0
    kind: str = "main"
    lr: float | None = None
    epochs: int | None = None
    tag: str | None = None
    overrides: str = "{}"  # JSON string keeps the dataclass hashable

    def config(self):
        cfg = load_config(self.model, json.loads(self.overrides))
        if self.lr is not None:
            cfg["train"]["lr"] = self.lr
        if self.epochs is not None:
            cfg["train"]["epochs"] = self.epochs
        return cfg

    def run_dir(self):
        return default_run_dir(self.config(), self.fold, self.seed, self.frac, self.tag, self.kind)

    def command(self):
        cmd = [sys.executable, "-m", "scripts.train", "--model", self.model, "--fold", str(self.fold),
               "--seed", str(self.seed), "--frac", str(self.frac), "--overrides", self.overrides,
               "--run-dir", str(self.run_dir())]
        if self.tag:
            cmd += ["--tag", self.tag]
        if self.lr is not None:
            cmd += ["--lr", str(self.lr)]
        if self.epochs is not None:
            cmd += ["--epochs", str(self.epochs)]
        if self.kind == "lr_search":
            cmd += ["--no-test"]
        return cmd


def expand(spec):
    jobs = []
    for entry in spec["experiments"]:
        models = spec["models"] if entry["models"] == "all" else entry["models"]
        folds = spec["folds"] if entry.get("folds", "all") == "all" else entry["folds"]
        overrides = json.dumps(entry.get("overrides", {}), sort_keys=True)
        for seed in entry["seeds"]:
            for frac in entry.get("fracs", [1.0]):
                for fold in folds:
                    for model in models:
                        for lr in entry.get("lrs", [None]):
                            jobs.append(Job(priority=entry["priority"], model=model, fold=fold, seed=seed, frac=frac,
                                            kind=entry.get("kind", "main"), lr=lr, epochs=entry.get("epochs"),
                                            tag=entry.get("tag"), overrides=overrides))
    return sorted(jobs, key=lambda job: job.priority)  # stable: seed -> frac -> fold -> model inside a priority


def load_experiments(path=CONFIG_DIR / "experiments.yaml"):
    return expand(load_yaml(path))


def select_jobs(jobs, max_priority=None, only_priority=None, allow_untuned=False):
    runnable, done, blocked = [], [], []
    for job in jobs:
        if only_priority is not None and job.priority != only_priority:
            continue
        if max_priority is not None and job.priority > max_priority:
            continue
        cfg = job.config()
        run_dir = default_run_dir(cfg, job.fold, job.seed, job.frac, job.tag, job.kind)
        if (run_dir / "DONE").exists():
            done.append((job, run_dir))
        elif job.kind == "main" and not cfg["train"].get("lr_tuned", False) and not allow_untuned:
            blocked.append((job, run_dir))
        else:
            runnable.append((job, run_dir))
    return runnable, done, blocked
```

```python
# file: scripts/train.py
"""Train and evaluate one run: python -m scripts.train --model unet --fold 1 --seed 1 [--smoke]"""
import argparse
import json
import shutil
from pathlib import Path

from flaresatformer.run import default_run_dir, run_experiment
from flaresatformer.utils import deep_merge, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--overrides", default="{}", help="JSON merged into the config")
    parser.add_argument("--tag")
    parser.add_argument("--run-dir")
    parser.add_argument("--no-test", action="store_true", help="lr search: validation only")
    parser.add_argument("--smoke", action="store_true", help="2 epochs on 64/32/32 patches; prints time and peak VRAM")
    args = parser.parse_args()

    overrides = json.loads(args.overrides)
    if args.smoke:
        overrides = deep_merge(overrides, {"train": {"epochs": 2}})
    cfg = load_config(args.model, overrides)
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    kind = "lr_search" if args.no_test else "main"
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(cfg, args.fold, args.seed, args.frac, args.tag, kind, args.smoke)
    if args.smoke:
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir = run_experiment(cfg, args.fold, args.seed, args.frac, run_dir, args.tag, smoke=args.smoke,
                             evaluate_test=not args.no_test)
    if args.smoke:
        print((run_dir / "cost.json").read_text())


if __name__ == "__main__":
    main()
```

```python
# file: scripts/run_queue.py
"""Run the experiment matrix in priority order, skipping finished runs: python -m scripts.run_queue --max-priority 1"""
import argparse
import subprocess

from flaresatformer.experiments import load_experiments, select_jobs
from flaresatformer.utils import CONFIG_DIR, REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default=str(CONFIG_DIR / "experiments.yaml"))
    parser.add_argument("--max-priority", type=int)
    parser.add_argument("--only-priority", type=int)
    parser.add_argument("--allow-untuned", action="store_true", help="run main jobs even if train.lr_tuned is false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed run")
    args = parser.parse_args()

    runnable, done, blocked = select_jobs(load_experiments(args.experiments), args.max_priority,
                                          args.only_priority, args.allow_untuned)
    print(f"{len(done)} done | {len(runnable)} to run | {len(blocked)} blocked "
          f"(set train.lr and train.lr_tuned: true in configs/models/*.yaml after priority 0)")
    for job, run_dir in runnable:
        print(f"[P{job.priority}] {run_dir}", flush=True)
        if args.dry_run:
            continue
        if subprocess.run(job.command(), cwd=REPO_ROOT).returncode != 0:
            print(f"[failed] {run_dir}")
            if not args.keep_going:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests.** Expected: 8 passed. Also run `python -m scripts.run_queue --dry-run --max-priority 0`. Expected: `0 done | 18 to run | 0 blocked`, plus 18 `[P0]` lines.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: resumable run orchestration, train CLI and prioritized experiment queue"`

---

### Task 9: Aggregation, statistics and figures

**Files:**
- Create: `flaresatformer/aggregate.py`, `flaresatformer/figures.py`, `scripts/aggregate.py`
- Test: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: `prf`, `object_prf` (Task 5); run folder layout (Task 8).
- Produces: `collect_runs(outputs_dir) -> DataFrame[variant, model, bands, frac, fold, seed, run_dir]`, `load_patches(runs)`, `metrics_from_counts(df) -> dict`, `per_run_metrics(patches, threshold_name)` (fold column is a string, with `"all"` for the pooled rows), `summarize(per_run)`, `paper_table(summary)`, `per_continent(patches, threshold_name)`, `bootstrap_delta(patches, variant, baseline="unet", threshold_name="t050", bands="b267", n_boot=2000, seed=0)`, `lr_search_summary(outputs_dir)`, `cost_table(runs)`. Figures: `learning_curve(per_run, path)`, `f1_per_continent(continent_df, path)`, `cost_vs_f1(cost_df, summary_df, path)`.

- [ ] **Step 1: Write the failing tests**

```python
# file: tests/test_aggregate.py
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
```

- [ ] **Step 2: Run it.** Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# file: flaresatformer/aggregate.py
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
```

```python
# file: flaresatformer/figures.py
"""Paper figures. Each model keeps a fixed color from the validated categorical palette plus its own marker."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODEL_ORDER = ["unet", "segformer", "segformer_scratch", "swin_unet", "transunet", "prithvi"]
MODEL_LABELS = {"unet": "UNet", "segformer": "SegFormer-B2", "segformer_scratch": "SegFormer-B2 (scratch)",
                "swin_unet": "Swin-UNet", "transunet": "TransUNet", "prithvi": "Prithvi-EO-2.0"}
MODEL_COLORS = {"unet": "#2a78d6", "segformer": "#eb6834", "segformer_scratch": "#1baf7a",
                "swin_unet": "#eda100", "transunet": "#e87ba4", "prithvi": "#008300"}
MODEL_MARKERS = {"unet": "o", "segformer": "s", "segformer_scratch": "D", "swin_unet": "^", "transunet": "v", "prithvi": "P"}
INK, INK_MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def _axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def _marker_style(model):
    return {"color": MODEL_COLORS[model], "marker": MODEL_MARKERS[model], "label": MODEL_LABELS[model]}


def learning_curve(per_run, path):
    df = per_run[(per_run.fold == "all") & (per_run.bands == "b267") & per_run.variant.isin(MODEL_ORDER)]
    fig, ax = _axes((5.5, 3.6))
    for model in MODEL_ORDER:
        g = df[df.variant == model].groupby("frac").pixel_f1.mean()
        if len(g) < 2:
            continue
        x = g.index.to_numpy() * 100
        ax.plot(x, g.to_numpy(), linewidth=2, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, **_marker_style(model))
        ax.annotate(MODEL_LABELS[model], (x[-1], g.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100], ["10%", "25%", "50%", "100%"])
    ax.minorticks_off()
    ax.set_xlabel("Training data used", color=INK)
    ax.set_ylabel("Pixel F1 (test, all folds)", color=INK)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _save(fig, path)


def f1_per_continent(continent_df, path):
    df = continent_df[(continent_df.bands == "b267") & continent_df.variant.isin(MODEL_ORDER)]
    continents = sorted(df.continent.unique())
    fig, ax = _axes((6.0, 0.8 + 0.6 * len(continents)))
    offsets = np.linspace(-0.3, 0.3, len(MODEL_ORDER))
    for offset, model in zip(offsets, MODEL_ORDER):
        g = df[df.variant == model].set_index("continent").reindex(continents)
        if g["mean"].isna().all():
            continue
        ax.scatter(g["mean"], np.arange(len(continents)) + offset, s=45, edgecolors=SURFACE, linewidths=1.5,
                   zorder=3, **_marker_style(model))
    ax.set_yticks(range(len(continents)), continents)
    ax.set_xlabel("Pixel F1 (test)", color=INK)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=8)
    _save(fig, path)


def cost_vs_f1(cost_df, summary_df, path):
    s = summary_df[(summary_df.fold == "all") & (summary_df.bands == "b267") & (summary_df.frac == 1.0)].set_index("variant")
    c = cost_df.set_index("variant")
    fig, ax = _axes((5.5, 3.6))
    for model in MODEL_ORDER:
        if model not in s.index or model not in c.index:
            continue
        x, y = c.loc[model, "inference_patches_per_s"], s.loc[model, "pixel_f1_mean"]
        ax.scatter(x, y, s=70, edgecolors=SURFACE, linewidths=1.5, zorder=3, **_marker_style(model))
        ax.annotate(MODEL_LABELS[model], (x, y), xytext=(6, 4), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Inference throughput (patches/s, RTX 4070 Laptop)", color=INK)
    ax.set_ylabel("Pixel F1 (test, all folds)", color=INK)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    _save(fig, path)
```

```python
# file: scripts/aggregate.py
"""Aggregate finished runs into results/ (tables, bootstrap CIs, figures): python -m scripts.aggregate"""
import argparse

import pandas as pd

from flaresatformer import figures
from flaresatformer.aggregate import (bootstrap_delta, collect_runs, cost_table, load_patches, lr_search_summary,
                                      paper_table, per_continent, per_run_metrics, summarize)
from flaresatformer.utils import CONFIG_DIR, load_yaml, resolve_path

THRESHOLDS = ("t050", "tuned")


def main():
    paths = load_yaml(CONFIG_DIR / "base.yaml")["paths"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default=paths["outputs_dir"])
    parser.add_argument("--results-dir", default=paths["results_dir"])
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    outputs, results = resolve_path(args.outputs_dir), resolve_path(args.results_dir)
    (results / "figures").mkdir(parents=True, exist_ok=True)

    lr = lr_search_summary(outputs)
    if len(lr):
        lr.to_csv(results / "lr_search.csv", index=False)
        print("lr search (chosen):\n", lr[lr.chosen].to_string(index=False))

    runs = collect_runs(outputs)
    if runs.empty:
        print("no finished main runs yet")
        return
    patches = load_patches(runs)
    summaries = {}
    for t in THRESHOLDS:
        per_run = per_run_metrics(patches, t)
        per_run.to_csv(results / f"metrics_per_run_{t}.csv", index=False)
        summaries[t] = summarize(per_run)
        summaries[t].to_csv(results / f"summary_{t}.csv", index=False)
        paper_table(summaries[t]).to_latex(results / f"table_{t}.tex", index=False)
        per_continent(patches, t).to_csv(results / f"per_continent_{t}.csv", index=False)

    main_runs = patches[(patches.bands == "b267") & (patches.frac == 1.0)]
    if "unet" in set(main_runs.variant):
        deltas = [bootstrap_delta(patches, v, threshold_name=t, n_boot=args.n_boot).assign(threshold=t)
                  for t in THRESHOLDS for v in sorted(set(main_runs.variant) - {"unet"})]
        if deltas:
            pd.concat(deltas).to_csv(results / "bootstrap_vs_unet.csv", index=False)

    cost = cost_table(runs)
    cost.to_csv(results / "cost.csv", index=False)
    figures.learning_curve(per_run_metrics(patches, "t050"), results / "figures" / "learning_curve.png")
    figures.f1_per_continent(per_continent(patches, "t050"), results / "figures" / "f1_per_continent.png")
    figures.cost_vs_f1(cost, summaries["t050"], results / "figures" / "cost_vs_f1.png")
    print(f"results written to {results}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests.** Expected: 5 passed. Open `tmp` figures from the test (or rerun the test code in a REPL) and look at them for label collisions.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: aggregation with cluster bootstrap CIs, paper tables and figures"`

---

### Task 10: Inference, README figure and README

**Files:**
- Create: `flaresatformer/viz.py`, `scripts/infer.py`, `scripts/make_readme_figure.py`, `assets/flaresat_example.png`
- Delete: `assets/active_fire_1.png` (shared with the FlareSat repo, replaced)
- Modify: `README.md`
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: `build_model`, `normalize`, `BAND_SETS`, `run_experiment` (tests).
- Produces: `stretch(channel)`, `rgb_composite(patch_hw10) -> (H,W,3)`, `swir2(patch) -> (H,W)`, `overlay(gray, mask) -> (H,W,3)`. Also `scripts.infer.load_run(run_dir, device) -> (cfg, model, norm, threshold)` and `scripts.infer.predict(model, cfg, norm, threshold, patch, device) -> bool mask`.

- [ ] **Step 1: Write the failing tests**

```python
# file: tests/test_viz.py
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
```

- [ ] **Step 2: Run it.** Expected: FAIL (`ModuleNotFoundError: flaresatformer.viz`).

- [ ] **Step 3: Implement**

```python
# file: flaresatformer/viz.py
"""Rendering helpers for patches (RGB composite, B7, mask overlay) shared by inference and the README figure."""
import numpy as np

FLARE_RED = (0.890, 0.286, 0.282)  # #e34948


def stretch(channel, low=2, high=98):
    lo, hi = np.percentile(channel, [low, high])
    return np.clip((channel - lo) / (hi - lo + 1e-9), 0, 1)


def rgb_composite(patch):
    """Natural color from a 10-band patch: B4, B3, B2 are channels 3, 2, 1."""
    return np.dstack([stretch(patch[:, :, c]) for c in (3, 2, 1)])


def swir2(patch):
    return np.clip(patch[:, :, 6], 0, 1)  # B7, reflectance 0-1


def overlay(gray, mask, color=FLARE_RED):
    rgb = np.dstack([gray] * 3).astype(np.float32)
    rgb[mask.astype(bool)] = color
    return rgb
```

```python
# file: scripts/infer.py
"""Predict flare masks with one or more finished runs and save a comparison panel per patch.

python -m scripts.infer --run-dirs outputs/unet/b267/frac100/fold1/seed1 outputs/segformer/b267/frac100/fold1/seed1 --n 4
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402

from flaresatformer.data.dataset import normalize  # noqa: E402
from flaresatformer.figures import MODEL_LABELS  # noqa: E402
from flaresatformer.models.factory import build_model  # noqa: E402
from flaresatformer.utils import BAND_SETS, load_yaml, resolve_path  # noqa: E402
from flaresatformer.viz import overlay, rgb_composite, swir2  # noqa: E402


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    cfg = load_yaml(run_dir / "config.yaml")
    cfg["model"]["pretrained"] = False  # weights come from best.pt; skip pretrained downloads
    model = build_model(cfg)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    norm = json.loads((run_dir / "norm.json").read_text())
    threshold = json.loads((run_dir / "threshold.json").read_text())["tuned"]
    return cfg, model.to(device).eval(), norm, threshold


@torch.no_grad()
def predict(model, cfg, norm, threshold, patch, device):
    bands = BAND_SETS[cfg["data"]["bands"]]
    x = normalize(np.nan_to_num(patch)[:, :, bands].transpose(2, 0, 1), norm["mean"], norm["std"], norm["scale"])
    prob = torch.sigmoid(model(torch.from_numpy(x)[None].to(device)).float())[0, 0].cpu().numpy()
    return prob > threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--patches", nargs="*", help="tiff_file paths as in the split CSV; default: random test flares")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="inference/output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = [load_run(r, device) for r in args.run_dirs]
    cfg0 = runs[0][0]
    data_root = resolve_path(cfg0["paths"]["data_root"])
    split = pd.read_csv(resolve_path(cfg0["paths"]["splits_dir"]) / f"fold{cfg0['run']['fold']}.csv",
                        dtype=str, keep_default_na=False)
    if args.patches:
        chosen = split[split.tiff_file.isin(args.patches)]
    else:
        chosen = split[(split.role == "test") & (split.source == "flare")].sample(args.n, random_state=args.seed)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in chosen.itertuples(index=False):
        patch = np.nan_to_num(tifffile.imread(data_root / row.tiff_file))
        gt = tifffile.imread(data_root / row.mask_file) > 0
        b7 = swir2(patch)
        panels = [("RGB (B4, B3, B2)", rgb_composite(patch)), ("SWIR 2 (B7)", b7), ("Reference", overlay(b7, gt))]
        for cfg, model, norm, threshold in runs:
            label = MODEL_LABELS.get(cfg["model"]["name"], cfg["model"]["name"])
            panels.append((label, overlay(b7, predict(model, cfg, norm, threshold, patch, device))))
        fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.9))
        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.savefig(out_dir / f"{Path(row.tiff_file).stem}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(out_dir / f"{Path(row.tiff_file).stem}.png")


if __name__ == "__main__":
    main()
```

```python
# file: scripts/make_readme_figure.py
"""Build the README example figure from one flare patch.

python -m scripts.make_readme_figure --tiff dataset/flare_patches/<patch>.tiff --mask dataset/flare_masks/<mask>.tiff
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from flaresatformer.utils import resolve_path  # noqa: E402
from flaresatformer.viz import overlay, rgb_composite, swir2  # noqa: E402

RING = "#eda100"
INK = "#0b0b0b"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiff", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", default="assets/flaresat_example.png")
    args = parser.parse_args()

    patch = np.nan_to_num(tifffile.imread(resolve_path(args.tiff)))
    mask = tifffile.imread(resolve_path(args.mask)) > 0
    rows, cols = np.nonzero(mask)
    center = (cols.mean(), rows.mean())
    b7 = swir2(patch)
    panels = [("RGB (B4, B3, B2)", rgb_composite(patch)), ("SWIR 2 (B7)", b7), ("Flare pixels (red)", overlay(b7, mask))]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.add_patch(plt.Circle(center, 16, fill=False, color=RING, linewidth=1.5))
        ax.set_title(title, fontsize=11, color=INK)
        ax.axis("off")
    fig.savefig(resolve_path(args.out), dpi=150, bbox_inches="tight", facecolor="white")
    print(resolve_path(args.out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests.** Expected: 2 passed.

- [ ] **Step 5: Generate the README figure.** Pick a flare patch with a clearly visible flare (20–80 mask pixels, low cloud). Run `python -m scripts.make_readme_figure --tiff ... --mask ...`. Open the PNG and check that the flare is obvious and the ring is centered. Then `git rm assets/active_fire_1.png`.

- [ ] **Step 6: Rewrite `README.md`.** Keep the intro and the Araucária photo. Replace the rest with:
  - **Dataset table:** real counts (11,505 flare patches from 83 countries in 6 continents, 683 urban, 265 fire; 256×256×10, bands B1–B7, B9, B10, B11), with the new figure.
  - **Models table:** the 6 models plus pretraining, as in spec §4.2.
  - **Protocol summary:** splits, loss, optimizer, lr search, seeds, metrics, bootstrap.
  - **Reproduction:** venv, `make_splits`, `build_cache`, `train --smoke`, `run_queue` by priority, `aggregate`, `infer`.
  - **Results:** say they are pending, filled in from `results/` after the runs.
  - **Link:** FlareSat as the v1 reference.

- [ ] **Step 7: Full test suite.** Run `python -m pytest -q`. Expected: all pass (slow tests deselected).
- [ ] **Step 8: Commit.** `git add -A && git commit -m "feat: inference panels, new README figure and README for FlareSatFormer 2.0"`

---

## After this plan (user-run, not in this session)

1. `python -m scripts.build_cache --bands b267 hls6`: about 20–40 min from the USB drive.
2. `python -m scripts.train --model <m> --fold 1 --smoke` for the 6 models. Adjust `train.batch_size` and `grad_checkpointing` in the YAMLs until each fits in 8 GB.
3. `python -m scripts.run_queue --only-priority 0`, then `python -m scripts.aggregate`. Set the chosen `train.lr` and `lr_tuned: true` per model.
4. `python -m scripts.run_queue --max-priority 2` (then 3, 4), and `python -m scripts.aggregate`.
