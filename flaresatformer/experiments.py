"""Expands configs/experiments.yaml into concrete jobs and decides which ones may run."""
import json
import sys
from dataclasses import dataclass

from flaresatformer.run import default_run_dir
from flaresatformer.utils import CONFIG_DIR, load_config, load_yaml


@dataclass(frozen=True)
class Job:
    priority: int
    model: str
    fold: int
    seed: int
    frac: float = 1.0
    kind: str = "main"
    lr: float | None = None
    epochs: int | None = None
    tag: str | None = None
    overrides: str = "{}"  # JSON string keeps the dataclass hashable

    def config(self):
        cfg = load_config(self.model, json.loads(self.overrides))
        if self.lr is not None:
            cfg["train"]["lr"] = self.lr
        if self.epochs is not None:
            cfg["train"]["epochs"] = self.epochs
        return cfg

    def run_dir(self):
        return default_run_dir(self.config(), self.fold, self.seed, self.frac, self.tag, self.kind)

    def command(self):
        cmd = [sys.executable, "-m", "scripts.train", "--model", self.model, "--fold", str(self.fold),
               "--seed", str(self.seed), "--frac", str(self.frac), "--overrides", self.overrides,
               "--run-dir", str(self.run_dir())]
        if self.tag:
            cmd += ["--tag", self.tag]
        if self.lr is not None:
            cmd += ["--lr", str(self.lr)]
        if self.epochs is not None:
            cmd += ["--epochs", str(self.epochs)]
        if self.kind == "lr_search":
            cmd += ["--no-test"]
        return cmd


def expand(spec):
    jobs = []
    for entry in spec["experiments"]:
        models = spec["models"] if entry["models"] == "all" else entry["models"]
        folds = spec["folds"] if entry.get("folds", "all") == "all" else entry["folds"]
        overrides = json.dumps(entry.get("overrides", {}), sort_keys=True)
        for seed in entry["seeds"]:
            for frac in entry.get("fracs", [1.0]):
                for fold in folds:
                    for model in models:
                        for lr in entry.get("lrs", [None]):
                            jobs.append(Job(priority=entry["priority"], model=model, fold=fold, seed=seed, frac=frac,
                                            kind=entry.get("kind", "main"), lr=lr, epochs=entry.get("epochs"),
                                            tag=entry.get("tag"), overrides=overrides))
    return sorted(jobs, key=lambda job: job.priority)  # stable: seed -> frac -> fold -> model inside a priority


def load_experiments(path=CONFIG_DIR / "experiments.yaml"):
    return expand(load_yaml(path))


def select_jobs(jobs, max_priority=None, only_priority=None, allow_untuned=False):
    runnable, done, blocked = [], [], []
    for job in jobs:
        if only_priority is not None and job.priority != only_priority:
            continue
        if max_priority is not None and job.priority > max_priority:
            continue
        cfg = job.config()
        run_dir = default_run_dir(cfg, job.fold, job.seed, job.frac, job.tag, job.kind)
        if (run_dir / "DONE").exists():
            done.append((job, run_dir))
        elif job.kind == "main" and not cfg["train"].get("lr_tuned", False) and not allow_untuned:
            blocked.append((job, run_dir))
        else:
            runnable.append((job, run_dir))
    return runnable, done, blocked
