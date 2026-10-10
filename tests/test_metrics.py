import numpy as np

from flaresatformer.metrics import object_counts, object_prf, pixel_counts, prf, threshold_grid


def _mask(*points, shape=(16, 16)):
    m = np.zeros(shape, bool)
    for r, c in points:
        m[r, c] = True
    return m


def test_pixel_counts_and_prf():
    assert pixel_counts(_mask((1, 1), (8, 8)), _mask((1, 1), (1, 2), (5, 5))) == (1, 1, 2)
    assert prf(1, 1, 2) == {"precision": 0.5, "recall": 1 / 3, "f1": 0.4, "iou": 0.25}


def test_prf_is_zero_not_nan_on_empty_inputs():
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    assert object_prf(0, 0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_object_detection_with_one_pixel_tolerance():
    gt = _mask((4, 4), (4, 5), (12, 12))   # two objects
    pred = _mask((5, 6), (0, 15))          # first touches object 1 diagonally; second is an isolated FP
    assert object_counts(pred, gt, tol=1) == {"gt_objects": 2, "gt_detected": 1, "pred_objects": 2, "pred_fp_objects": 1}


def test_object_detection_strict_without_tolerance():
    gt, pred = _mask((4, 4)), _mask((4, 5))
    assert object_counts(pred, gt, tol=0)["gt_detected"] == 0
    assert object_counts(pred, gt, tol=1)["gt_detected"] == 1


def test_empty_masks():
    z = np.zeros((8, 8), bool)
    assert object_counts(z, z) == {"gt_objects": 0, "gt_detected": 0, "pred_objects": 0, "pred_fp_objects": 0}


def test_threshold_grid_contains_half():
    g = threshold_grid(0.05, 0.95, 0.05)
    assert len(g) == 19 and 0.5 in g and g[0] == 0.05 and g[-1] == 0.95
