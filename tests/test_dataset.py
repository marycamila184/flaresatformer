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
