import torch

def compute_metrics(preds, targets, threshold=0.5, eps=1e-7):
    """
    Computes Precision, Recall and F1-score for binary segmentation.

    Args:
        preds (torch.Tensor): probabilities or logits (B, 1, H, W)
        targets (torch.Tensor): binary masks (B, 1, H, W)
        threshold (float): threshold to binarize predictions
        eps (float): numerical stability

    Returns:
        precision, recall, f1 (floats)
    """

    # If logits, apply sigmoid
    if preds.max() > 1 or preds.min() < 0:
        preds = torch.sigmoid(preds)

    preds = (preds > threshold).float()
    targets = targets.float()

    # Flatten
    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = torch.sum(preds * targets)
    fp = torch.sum(preds * (1 - targets))
    fn = torch.sum((1 - preds) * targets)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return (
        precision.item(),
        recall.item(),
        f1.item(),
    )
