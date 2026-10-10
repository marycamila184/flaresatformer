"""Rendering helpers for patches (RGB composite, B7, mask overlay) shared by inference and the README figure."""
import numpy as np

FLARE_RED = (0.890, 0.286, 0.282)  # #e34948


def stretch(channel, low=2, high=98):
    lo, hi = np.percentile(channel, [low, high])
    return np.clip((channel - lo) / (hi - lo + 1e-9), 0, 1)


def rgb_composite(patch):
    """Natural color from a 10-band patch: B4, B3, B2 are channels 3, 2, 1."""
    return np.dstack([stretch(patch[:, :, c]) for c in (3, 2, 1)])


def swir2(patch):
    return np.clip(patch[:, :, 6], 0, 1)  # B7, reflectance 0-1


def overlay(gray, mask, color=FLARE_RED):
    rgb = np.dstack([gray] * 3).astype(np.float32)
    rgb[mask.astype(bool)] = color
    return rgb
