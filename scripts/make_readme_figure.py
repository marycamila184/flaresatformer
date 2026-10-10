"""Build the README example figure from one flare patch.

python -m scripts.make_readme_figure --tiff dataset/flare_patches/<patch>.tiff --mask dataset/flare_masks/<mask>.tiff
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from flaresatformer.utils import resolve_path  # noqa: E402
from flaresatformer.viz import overlay, rgb_composite, stretch  # noqa: E402

BOX = "#eda100"
INK = "#0b0b0b"
ZOOM = 64


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiff", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", default="assets/flaresat_example.png")
    args = parser.parse_args()

    patch = np.nan_to_num(tifffile.imread(resolve_path(args.tiff)))
    mask = tifffile.imread(resolve_path(args.mask)) > 0
    rows, cols = np.nonzero(mask)
    # square zoom window centred on the bounding box of all flare pixels
    r0 = int(np.clip((rows.min() + rows.max()) // 2 - ZOOM // 2, 0, mask.shape[0] - ZOOM))
    c0 = int(np.clip((cols.min() + cols.max()) // 2 - ZOOM // 2, 0, mask.shape[1] - ZOOM))
    b7 = stretch(patch[:, :, 6], 2, 99.9)
    zoom = overlay(b7, mask)[r0:r0 + ZOOM, c0:c0 + ZOOM]
    panels = [("RGB (B4, B3, B2)", rgb_composite(patch), True), ("SWIR 2 (B7)", b7, True),
              ("Zoom: flare pixels in red", zoom, False)]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
    for ax, (title, img, boxed) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if boxed:
            ax.add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), ZOOM, ZOOM, fill=False, color=BOX, linewidth=1.5))
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    for side in axes[2].spines.values():
        side.set_color(BOX)
        side.set_linewidth(1.5)
    fig.savefig(resolve_path(args.out), dpi=150, bbox_inches="tight", facecolor="white")
    print(resolve_path(args.out))


if __name__ == "__main__":
    main()
