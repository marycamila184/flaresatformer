"""Output heads shared by the transformer models (spec 4.1)."""
import torch
import torch.nn.functional as F
from torch import nn


def conv_bn_relu(in_ch, out_ch, kernel_size=3):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class FullResHead(nn.Module):
    """Fuses decoder features with a shallow full-resolution branch over the raw input.

    Decoder features are projected to `width` channels before upsampling to keep 256x256 activations small.
    """

    def __init__(self, in_ch, dec_ch, width=32, out_ch=1):
        super().__init__()
        self.image = nn.Sequential(conv_bn_relu(in_ch, width), conv_bn_relu(width, width))
        self.dec_proj = conv_bn_relu(dec_ch, width, kernel_size=1)
        self.dec = conv_bn_relu(width, width)
        self.fuse = nn.Sequential(conv_bn_relu(2 * width, width), nn.Conv2d(width, out_ch, 1))

    def forward(self, x, dec):
        dec = F.interpolate(self.dec_proj(dec), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.image(x), self.dec(dec)], dim=1))


class UpsampleHead(nn.Module):
    """FRH ablation: 1x1 classifier on decoder features, then bilinear upsampling to the input size."""

    def __init__(self, dec_ch, out_ch=1):
        super().__init__()
        self.classifier = nn.Conv2d(dec_ch, out_ch, 1)

    def forward(self, x, dec):
        return F.interpolate(self.classifier(dec), size=x.shape[-2:], mode="bilinear", align_corners=False)


def make_head(in_ch, dec_ch, frh=True):
    return FullResHead(in_ch, dec_ch) if frh else UpsampleHead(dec_ch)
