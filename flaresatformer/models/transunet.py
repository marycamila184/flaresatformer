"""TransUNet: R50-ViT-B/16 hybrid encoder (timm, ImageNet-21k) + cascaded upsampler with ResNet skips."""
import timm
import torch
import torch.nn.functional as F
from torch import nn

from flaresatformer.models.frh import conv_bn_relu, make_head

SKIP_CHANNELS = (512, 256, 64)  # ResNet stage 1 (1/8), stage 0 (1/4), stem (1/2)


class TransUNet(nn.Module):
    def __init__(self, in_channels, pretrained=True, frh=True, encoder="vit_base_r50_s16_224.orig_in21k",
                 img_size=256, dec_channels=(256, 128, 64, 16)):
        super().__init__()
        self.vit = timm.create_model(encoder, pretrained=pretrained, img_size=img_size, num_classes=0, in_chans=in_channels)
        self.grid = img_size // 16
        self._skips = {}
        backbone = self.vit.patch_embed.backbone
        for name, module in (("s8", backbone.stages[1]), ("s4", backbone.stages[0]), ("s2", backbone.stem.norm)):
            module.register_forward_hook(self._store(name))
        self.conv_more = conv_bn_relu(self.vit.embed_dim, 512)
        ins = (512, *dec_channels[:2])
        self.blocks = nn.ModuleList(
            nn.Sequential(conv_bn_relu(i + s, o), conv_bn_relu(o, o)) for i, s, o in zip(ins, SKIP_CHANNELS, dec_channels[:3]))
        self.last = nn.Sequential(conv_bn_relu(dec_channels[2], dec_channels[3]), conv_bn_relu(dec_channels[3], dec_channels[3]))
        self.head = make_head(in_channels, dec_channels[3], frh)

    def _store(self, name):
        def hook(module, inputs, output):
            self._skips[name] = output
        return hook

    def forward(self, x):
        tokens = self.vit.forward_features(x)[:, self.vit.num_prefix_tokens:]
        b, _, d = tokens.shape
        y = self.conv_more(tokens.transpose(1, 2).reshape(b, d, self.grid, self.grid))
        for name, block in zip(("s8", "s4", "s2"), self.blocks):
            skip = self._skips[name]
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))
        self._skips.clear()
        y = self.last(F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.head(x, y)

    def encoder_parameters(self):
        return list(self.vit.parameters())

    def set_grad_checkpointing(self):
        self.vit.set_grad_checkpointing(True)
