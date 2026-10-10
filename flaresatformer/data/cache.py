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
