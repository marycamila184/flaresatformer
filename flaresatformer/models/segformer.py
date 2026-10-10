"""SegFormer MiT-B2 (Hugging Face) with the decode-head classifier replaced by the shared output head."""
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from transformers.utils import logging as hf_logging

from flaresatformer.models.frh import make_head

hf_logging.set_verbosity_error()


class SegFormerSeg(nn.Module):
    def __init__(self, in_channels, pretrained=True, frh=True, checkpoint="nvidia/mit-b2"):
        super().__init__()
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                checkpoint, num_labels=1, num_channels=in_channels, ignore_mismatched_sizes=True)
        else:
            config = SegformerConfig.from_pretrained(checkpoint, num_labels=1, num_channels=in_channels)
            self.net = SegformerForSemanticSegmentation(config)
        self.net.decode_head.classifier = nn.Identity()  # logits become the fused 1/4-resolution features
        self.head = make_head(in_channels, self.net.config.decoder_hidden_size, frh)

    def forward(self, x):
        return self.head(x, self.net(pixel_values=x).logits)

    def encoder_parameters(self):
        return list(self.net.segformer.parameters())
