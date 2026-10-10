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
