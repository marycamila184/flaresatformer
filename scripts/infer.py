"""Predict flare masks with one or more finished runs and save a comparison panel per patch.

python -m scripts.infer --run-dirs outputs/unet/b267/frac100/fold1/seed1 outputs/segformer/b267/frac100/fold1/seed1 --n 4
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402

from flaresatformer.data.dataset import normalize  # noqa: E402
from flaresatformer.figures import MODEL_LABELS  # noqa: E402
from flaresatformer.models.factory import build_model  # noqa: E402
from flaresatformer.utils import BAND_SETS, load_yaml, resolve_path  # noqa: E402
from flaresatformer.viz import overlay, rgb_composite, swir2  # noqa: E402


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    cfg = load_yaml(run_dir / "config.yaml")
    cfg["model"]["pretrained"] = False  # weights come from best.pt; skip pretrained downloads
    model = build_model(cfg)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    norm = json.loads((run_dir / "norm.json").read_text())
    threshold = json.loads((run_dir / "threshold.json").read_text())["tuned"]
    return cfg, model.to(device).eval(), norm, threshold


@torch.no_grad()
def predict(model, cfg, norm, threshold, patch, device):
    bands = BAND_SETS[cfg["data"]["bands"]]
    x = normalize(np.nan_to_num(patch)[:, :, bands].transpose(2, 0, 1), norm["mean"], norm["std"], norm["scale"])
    prob = torch.sigmoid(model(torch.from_numpy(x)[None].to(device)).float())[0, 0].cpu().numpy()
    return prob > threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--patches", nargs="*", help="tiff_file paths as in the split CSV; default: random test flares")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="inference/output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = [load_run(r, device) for r in args.run_dirs]
    cfg0 = runs[0][0]
    data_root = resolve_path(cfg0["paths"]["data_root"])
    split = pd.read_csv(resolve_path(cfg0["paths"]["splits_dir"]) / f"fold{cfg0['run']['fold']}.csv",
                        dtype=str, keep_default_na=False)
    if args.patches:
        chosen = split[split.tiff_file.isin(args.patches)]
    else:
        chosen = split[(split.role == "test") & (split.source == "flare")].sample(args.n, random_state=args.seed)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in chosen.itertuples(index=False):
        patch = np.nan_to_num(tifffile.imread(data_root / row.tiff_file))
        gt = tifffile.imread(data_root / row.mask_file) > 0
        b7 = swir2(patch)
        panels = [("RGB (B4, B3, B2)", rgb_composite(patch)), ("SWIR 2 (B7)", b7), ("Reference", overlay(b7, gt))]
        for cfg, model, norm, threshold in runs:
            label = MODEL_LABELS.get(cfg["model"]["name"], cfg["model"]["name"])
            panels.append((label, overlay(b7, predict(model, cfg, norm, threshold, patch, device))))
        fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.9))
        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.savefig(out_dir / f"{Path(row.tiff_file).stem}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(out_dir / f"{Path(row.tiff_file).stem}.png")


if __name__ == "__main__":
    main()
