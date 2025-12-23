import torch.nn as nn
from terratorch.models import PrithviModel


class PrithviSegmentation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.model = PrithviModel(
            backbone="prithvi_eo_v2",
            task="segmentation",
            num_classes=1,
            in_channels=in_channels,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x):
        # logits: (B, 1, H, W)
        return self.model(x)

    def freeze_backbone(self):
        self.model.freeze_backbone()

    def unfreeze_backbone(self):
        self.model.unfreeze_backbone()
