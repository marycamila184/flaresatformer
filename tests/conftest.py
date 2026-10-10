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
