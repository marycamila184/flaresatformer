"""Focal + batch-level Dice loss shared by every model."""
import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Binary focal loss with class-dependent alpha_t: alpha for positives, 1 - alpha for negatives."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        logits, targets = logits.float(), targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class BatchDiceLoss(nn.Module):
    """Soft Dice over the whole batch: all-negative patches (urban, fire) never produce degenerate per-image terms."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs, targets = torch.sigmoid(logits.float()), targets.float()
        intersection = (probs * targets).sum()
        return 1 - (2 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice = BatchDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.focal(logits, targets) + self.dice_weight * self.dice(logits, targets)
