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
