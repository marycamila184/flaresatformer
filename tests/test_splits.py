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
