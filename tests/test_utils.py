import torch

from flaresatformer.utils import BAND_SETS, deep_merge, load_config, set_seed

MODELS = ["unet", "segformer", "segformer_scratch", "swin_unet", "transunet", "prithvi"]


def test_deep_merge_overrides_nested_without_mutating():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_merge(base, {"a": {"b": 10}})
    assert out == {"a": {"b": 10, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1


def test_every_model_config_loads_with_required_keys():
    for name in MODELS:
        cfg = load_config(name)
        assert cfg["model"]["name"] == name
        assert cfg["data"]["bands"] in BAND_SETS
        assert cfg["train"]["effective_batch_size"] % cfg["train"]["batch_size"] == 0


def test_overrides_win():
    assert load_config("unet", {"train": {"lr": 1.0}})["train"]["lr"] == 1.0


def test_set_seed_is_reproducible():
    set_seed(3)
    a = torch.rand(5)
    set_seed(3)
    assert torch.equal(a, torch.rand(5))
