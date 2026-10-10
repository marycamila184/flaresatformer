"""Builds models from the resolved config and splits parameters into lr groups."""
from flaresatformer.utils import BAND_SETS


def build_model(cfg):
    m = cfg["model"]
    bands = BAND_SETS[cfg["data"]["bands"]]
    channels = len(bands)
    name = m["name"]
    if name == "unet":
        from flaresatformer.models.unet import UNet
        model = UNet(in_channels=channels, base_filters=m.get("base_filters", 32))
    elif name in ("segformer", "segformer_scratch"):
        from flaresatformer.models.segformer import SegFormerSeg
        model = SegFormerSeg(channels, pretrained=m["pretrained"], frh=m["frh"])
    elif name == "swin_unet":
        from flaresatformer.models.swin_unet import SwinUNet
        model = SwinUNet(channels, pretrained=m["pretrained"], frh=m["frh"])
    elif name == "transunet":
        from flaresatformer.models.transunet import TransUNet
        model = TransUNet(channels, pretrained=m["pretrained"], frh=m["frh"])
    elif name == "prithvi":
        from flaresatformer.models.prithvi import PrithviSeg
        model = PrithviSeg(bands, pretrained=m["pretrained"], frh=m["frh"], lora_r=m.get("lora_r", 16),
                           lora_alpha=m.get("lora_alpha", 32), freeze_encoder=m.get("freeze_encoder", False))
    else:
        raise ValueError(f"Unknown model: {name}")
    if m.get("grad_checkpointing"):
        model.set_grad_checkpointing()
    return model


def param_groups(model, lr, encoder_lr_mult, pretrained):
    encoder_ids = {id(p) for p in model.encoder_parameters()} if pretrained else set()
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in trainable if id(p) in encoder_ids], "lr": lr * encoder_lr_mult},
              {"params": [p for p in trainable if id(p) not in encoder_ids], "lr": lr}]
    return [g for g in groups if g["params"]]
