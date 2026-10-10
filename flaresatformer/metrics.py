"""Pixel- and object-level metrics from binary masks or summed counts."""
import numpy as np
from scipy import ndimage

STRUCT = np.ones((3, 3), dtype=bool)  # 8-connectivity


def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def pixel_counts(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    return int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())


def prf(tp, fp, fn):
    return {"precision": _safe_div(tp, tp + fp), "recall": _safe_div(tp, tp + fn),
            "f1": _safe_div(2 * tp, 2 * tp + fp + fn), "iou": _safe_div(tp, tp + fp + fn)}


def object_counts(pred, gt, tol=1):
    """A GT flare is detected if a predicted pixel lies within `tol` px of it; a predicted object far from every GT is a FP."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    gt_lab, n_gt = ndimage.label(gt, STRUCT)
    pred_lab, n_pred = ndimage.label(pred, STRUCT)
    pred_zone = ndimage.binary_dilation(pred, STRUCT, iterations=tol) if tol and n_pred else pred
    gt_zone = ndimage.binary_dilation(gt, STRUCT, iterations=tol) if tol and n_gt else gt
    detected = np.unique(gt_lab[pred_zone & (gt_lab > 0)]).size
    matched = np.unique(pred_lab[gt_zone & (pred_lab > 0)]).size
    return {"gt_objects": int(n_gt), "gt_detected": int(detected),
            "pred_objects": int(n_pred), "pred_fp_objects": int(n_pred - matched)}


def object_prf(gt_objects, gt_detected, pred_objects, pred_fp_objects):
    precision = _safe_div(pred_objects - pred_fp_objects, pred_objects)
    recall = _safe_div(gt_detected, gt_objects)
    return {"precision": precision, "recall": recall, "f1": _safe_div(2 * precision * recall, precision + recall)}


def threshold_grid(start, stop, step):
    return np.round(np.arange(start, stop + step / 2, step), 4)
