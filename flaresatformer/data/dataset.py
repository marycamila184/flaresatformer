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
