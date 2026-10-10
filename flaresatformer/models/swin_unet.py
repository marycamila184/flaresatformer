"""SwinV2-T encoder (timm, ImageNet-1k, window 8 for 256 px) with a UNet-style conv decoder and the shared head."""
import timm
import torch
import torch.nn.functional as F
from torch import nn

from flaresatformer.models.frh import conv_bn_relu, make_head


class SwinUNet(nn.Module):
    def __init__(self, in_channels, pretrained=True, frh=True, encoder="swinv2_tiny_window8_256", dec_channels=(256, 128, 64)):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True, in_chans=in_channels)
        fmt = getattr(self.encoder, "output_fmt", "NCHW")
        self.nhwc = getattr(fmt, "value", fmt) == "NHWC"
        channels = self.encoder.feature_info.channels()  # strides 4, 8, 16, 32
        ins = [channels[3], *dec_channels[:-1]]
        skips = channels[2::-1]
        self.blocks = nn.ModuleList(
            nn.Sequential(conv_bn_relu(i + s, o), conv_bn_relu(o, o)) for i, s, o in zip(ins, skips, dec_channels))
        self.head = make_head(in_channels, dec_channels[-1], frh)

    def forward(self, x):
        feats = self.encoder(x)
        if self.nhwc:
            feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        y = feats[-1]
        for skip, block in zip(feats[-2::-1], self.blocks):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))
        return self.head(x, y)

    def encoder_parameters(self):
        return list(self.encoder.parameters())

    def set_grad_checkpointing(self):
        self.encoder.set_grad_checkpointing(True)
