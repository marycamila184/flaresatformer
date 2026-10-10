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
