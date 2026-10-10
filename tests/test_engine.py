import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flaresatformer.engine import collect_patch_stats, evaluate_counts, fit, warmup_cosine
from flaresatformer.utils import load_config, set_seed

CPU = torch.device("cpu")


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.Dropout2d(0.3), nn.ReLU(), nn.Conv2d(4, 1, 1))

    def forward(self, x):
        return self.net(x)

    def encoder_parameters(self):
        return []


class Identity(nn.Module):
    def forward(self, x):
        return x[:, :1]


def _loaders(seed):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(12, 3, 16, 16, generator=g)
    ds = TensorDataset(x, (x[:, 2:3] > 1.0).float(), torch.arange(12))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=4, shuffle=True, generator=gen, drop_last=True), DataLoader(ds, batch_size=4), gen


def _cfg(epochs, patience=100, lr=3e-2):
    return load_config("unet", {"model": {"pretrained": False}, "train": {
        "epochs": epochs, "patience": patience, "batch_size": 4, "effective_batch_size": 8, "amp": False, "lr": lr}})


def test_warmup_cosine_shape():
    f = warmup_cosine(100, 0.1)
    assert f(0) == 0.1 and f(9) == 1.0 and abs(f(100)) < 1e-9 and 0 < f(55) < 1


def test_fit_learns_and_writes_artifacts(tmp_path):
    set_seed(0)
    train, val, gen = _loaders(0)
    state = fit(Tiny(), train, val, _cfg(epochs=40), tmp_path, CPU, gen)
    assert state["epoch"] == 40
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
    assert pd.read_csv(tmp_path / "history.csv").val_f1.max() > 0.5


def test_early_stopping(tmp_path):
    set_seed(0)
    train, val, gen = _loaders(0)
    state = fit(Tiny(), train, val, _cfg(epochs=50, patience=2, lr=0.0), tmp_path, CPU, gen)
    assert state["epoch"] < 50 and state["bad_epochs"] == 2


def test_resume_matches_uninterrupted_run(tmp_path):
    set_seed(0)
    a = Tiny()
    train, val, gen = _loaders(0)
    fit(a, train, val, _cfg(epochs=4), tmp_path / "a", CPU, gen)

    set_seed(0)
    b = Tiny()
    train, val, gen = _loaders(0)
    fit(b, train, val, _cfg(epochs=4), tmp_path / "b", CPU, gen, stop_after=2)  # "crash" after 2 epochs
    set_seed(123)  # a fresh process: everything must come back from last.pt
    resumed = Tiny()
    train, val, gen = _loaders(999)
    fit(resumed, train, val, _cfg(epochs=4), tmp_path / "b", CPU, gen)

    for pa, pb in zip(a.state_dict().values(), resumed.state_dict().values()):
        torch.testing.assert_close(pa, pb)
    drop = ["epoch_time_s"]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "a" / "history.csv").drop(columns=drop),
                                  pd.read_csv(tmp_path / "b" / "history.csv").drop(columns=drop))


def _fixed_loader():
    x = torch.tensor([-1.0, 0.0, 2.0, 3.0]).view(1, 1, 2, 2).repeat(1, 3, 1, 1)
    y = torch.tensor([0.0, 1.0, 1.0, 0.0]).view(1, 1, 2, 2)
    return DataLoader(TensorDataset(x, y, torch.arange(1)), batch_size=1)


def test_evaluate_counts_matches_manual():
    # sigmoid -> [0.27, 0.50, 0.88, 0.95]; >0.5 -> [0,0,1,1]; >0.9 -> [0,0,0,1]
    counts, _ = evaluate_counts(Identity(), _fixed_loader(), [0.5, 0.9], CPU, amp=False)
    np.testing.assert_array_equal(counts, [[1, 1, 1], [0, 1, 2]])


def test_collect_patch_stats_one_row_per_patch_and_threshold():
    df = collect_patch_stats(Identity(), _fixed_loader(), [{"tiff_file": "a", "source": "flare"}],
                             {"t050": 0.5, "tuned": 0.9}, CPU, amp=False, tol=1)
    assert list(df.threshold_name) == ["t050", "tuned"]
    assert df.iloc[0][["tp", "fp", "fn"]].tolist() == [1, 1, 1]
    assert df.iloc[0].tiff_file == "a"


def test_resume_is_exact_with_worker_processes_and_augmentation(tmp_path):
    from flaresatformer.data.dataset import FlareDataset
    from flaresatformer.run import _loader

    rng = np.random.default_rng(0)
    images = rng.standard_normal((12, 3, 16, 16)).astype(np.float16)
    masks = (images[:, 2:3] > 1.0).astype(np.uint8)

    def loaders(seed):
        gen = torch.Generator().manual_seed(seed)
        train_ds = FlareDataset(images, masks, np.arange(12), [0.0] * 3, [1.0] * 3, augment=True)
        val_ds = FlareDataset(images, masks, np.arange(12), [0.0] * 3, [1.0] * 3)
        return _loader(train_ds, 4, 2, shuffle=True, generator=gen), _loader(val_ds, 4, 2), gen

    set_seed(0)
    a = Tiny()
    train, val, gen = loaders(0)
    fit(a, train, val, _cfg(epochs=4), tmp_path / "a", CPU, gen)

    set_seed(0)
    b = Tiny()
    train, val, gen = loaders(0)
    fit(b, train, val, _cfg(epochs=4), tmp_path / "b", CPU, gen, stop_after=2)
    set_seed(123)
    resumed = Tiny()
    train, val, gen = loaders(999)
    fit(resumed, train, val, _cfg(epochs=4), tmp_path / "b", CPU, gen)

    for pa, pb in zip(a.state_dict().values(), resumed.state_dict().values()):
        torch.testing.assert_close(pa, pb)
