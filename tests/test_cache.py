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
