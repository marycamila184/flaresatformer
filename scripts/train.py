"""Train and evaluate one run: python -m scripts.train --model unet --fold 1 --seed 1 [--smoke]"""
import argparse
import json
import shutil
from pathlib import Path

from flaresatformer.run import default_run_dir, run_experiment
from flaresatformer.utils import deep_merge, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--overrides", default="{}", help="JSON merged into the config")
    parser.add_argument("--tag")
    parser.add_argument("--run-dir")
    parser.add_argument("--no-test", action="store_true", help="lr search: validation only")
    parser.add_argument("--smoke", action="store_true", help="2 epochs on 64/32/32 patches; prints time and peak VRAM")
    args = parser.parse_args()

    overrides = json.loads(args.overrides)
    if args.smoke:
        overrides = deep_merge(overrides, {"train": {"epochs": 2}})
    cfg = load_config(args.model, overrides)
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    kind = "lr_search" if args.no_test else "main"
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(cfg, args.fold, args.seed, args.frac, args.tag, kind, args.smoke)
    if args.smoke:
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir = run_experiment(cfg, args.fold, args.seed, args.frac, run_dir, args.tag, smoke=args.smoke,
                             evaluate_test=not args.no_test)
    if args.smoke:
        print((run_dir / "cost.json").read_text())


if __name__ == "__main__":
    main()
