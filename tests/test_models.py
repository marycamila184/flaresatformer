import pytest
import torch

from flaresatformer.models.factory import build_model, param_groups
from flaresatformer.models.frh import FullResHead, UpsampleHead
from flaresatformer.utils import load_config


def _cfg(name, bands="b267", **model):
    return load_config(name, {"data": {"bands": bands},
                              "model": {"pretrained": False, "grad_checkpointing": False, **model}})


@pytest.mark.parametrize("name", ["unet", "segformer", "segformer_scratch", "swin_unet", "transunet"])
@pytest.mark.parametrize("bands", ["b267", "hls6"])
def test_forward_returns_full_resolution_logits(name, bands):
    model = build_model(_cfg(name, bands)).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3 if bands == "b267" else 6, 256, 256))
    assert out.shape == (2, 1, 256, 256)


def test_frh_can_be_disabled():
    assert isinstance(build_model(_cfg("segformer", frh=False)).head, UpsampleHead)
    assert isinstance(build_model(_cfg("segformer")).head, FullResHead)


def test_param_groups_split_encoder_from_head():
    model = build_model(_cfg("swin_unet"))
    groups = param_groups(model, lr=1e-3, encoder_lr_mult=0.1, pretrained=True)
    assert [g["lr"] for g in groups] == pytest.approx([1e-4, 1e-3])
    assert sum(p.numel() for p in groups[0]["params"]) == sum(p.numel() for p in model.encoder.parameters())
    assert len(param_groups(model, 1e-3, 0.1, pretrained=False)) == 1


def test_transunet_backward_with_grad_checkpointing():
    model = build_model(_cfg("transunet", grad_checkpointing=True)).train()
    model(torch.randn(2, 3, 256, 256)).mean().backward()
    assert model.head.fuse[-1].weight.grad is not None


@pytest.mark.slow
@pytest.mark.parametrize("bands", ["b267", "hls6"])
def test_prithvi_forward_and_only_lora_trainable_in_encoder(bands):
    model = build_model(_cfg("prithvi", bands)).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3 if bands == "b267" else 6, 256, 256)).shape == (2, 1, 256, 256)
    trainable = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
