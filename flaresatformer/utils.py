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
