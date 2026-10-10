import math

import torch

from flaresatformer.losses import BatchDiceLoss, FocalDiceLoss, FocalLoss


def test_focal_alpha_weights_classes_differently():
    loss, zero = FocalLoss(alpha=0.25, gamma=2.0), torch.zeros(1, 1, 1, 1)
    assert math.isclose(loss(zero, torch.ones(1, 1, 1, 1)).item(), 0.25 * 0.25 * math.log(2), rel_tol=1e-6)
    assert math.isclose(loss(zero, torch.zeros(1, 1, 1, 1)).item(), 0.75 * 0.25 * math.log(2), rel_tol=1e-6)


def test_batch_dice_is_finite_for_empty_masks_and_confident_negatives():
    value = BatchDiceLoss()(torch.full((2, 1, 8, 8), -20.0), torch.zeros(2, 1, 8, 8))
    assert torch.isfinite(value) and value.item() < 1e-3


def test_batch_dice_perfect_and_inverted():
    t = torch.zeros(1, 1, 8, 8)
    t[0, 0, 2:4, 2:4] = 1
    assert BatchDiceLoss(smooth=0)(t * 40 - 20, t).item() < 1e-6
    assert BatchDiceLoss(smooth=0)(20 - t * 40, t).item() > 0.99


def test_focal_dice_backpropagates():
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    FocalDiceLoss()(logits, (torch.rand(2, 1, 8, 8) > 0.9).float()).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
