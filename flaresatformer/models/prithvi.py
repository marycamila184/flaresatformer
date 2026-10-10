"""Prithvi-EO-2.0-300M (terratorch) with LoRA adapters, UperNet decoder and the shared output head."""
import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from torch.utils.checkpoint import checkpoint

from flaresatformer.models.frh import make_head
from flaresatformer.utils import HLS_NAMES

LORA_TARGETS = r".*blocks\.\d+\.attn\.(qkv|proj)"
NECKS = [{"name": "SelectIndices", "indices": [5, 11, 17, 23]},
         {"name": "ReshapeTokensToImage"},
         {"name": "LearnedInterpolateToPyramidal"}]


class _Checkpointed(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.block, *args, use_reentrant=False, **kwargs)
        return self.block(*args, **kwargs)


class PrithviSeg(nn.Module):
    def __init__(self, bands, pretrained=True, frh=True, lora_r=16, lora_alpha=32, freeze_encoder=False):
        super().__init__()
        from terratorch.datasets import HLSBands
        from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory

        net = EncoderDecoderFactory().build_model(
            task="segmentation", backbone="prithvi_eo_v2_300", backbone_pretrained=pretrained,
            backbone_bands=[HLSBands[HLS_NAMES[b]] for b in bands], backbone_num_frames=1,
            decoder="UperNetDecoder", decoder_channels=256, num_classes=1, necks=NECKS)
        self.encoder, self.neck, self.decoder = net.encoder, net.neck, net.decoder
        self.encoder.requires_grad_(False)
        if not freeze_encoder:
            config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=LORA_TARGETS)
            self.encoder = inject_adapter_in_model(config, self.encoder)
            for name, param in self.encoder.named_parameters():
                param.requires_grad = "lora_" in name
        self.head = make_head(len(bands), self.decoder.out_channels, frh)

    def forward(self, x):
        feats = self.neck(self.encoder(x), image_size=tuple(x.shape[-2:]))
        return self.head(x, self.decoder(list(feats)))

    def encoder_parameters(self):
        return []  # backbone frozen; LoRA adapters are new parameters and train at the full lr

    def set_grad_checkpointing(self):
        self.encoder.blocks = nn.ModuleList(_Checkpointed(block) for block in self.encoder.blocks)
