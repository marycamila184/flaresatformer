"""Aggregate finished runs into results/ (tables, bootstrap CIs, figures): python -m scripts.aggregate"""
import argparse

import pandas as pd

from flaresatformer import figures
from flaresatformer.aggregate import (bootstrap_delta, collect_runs, cost_table, load_patches, lr_search_summary,
                                      paper_table, per_continent, per_run_metrics, summarize)
from flaresatformer.utils import CONFIG_DIR, load_yaml, resolve_path

THRESHOLDS = ("t050", "tuned")


def main():
    paths = load_yaml(CONFIG_DIR / "base.yaml")["paths"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default=paths["outputs_dir"])
    parser.add_argument("--results-dir", default=paths["results_dir"])
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    outputs, results = resolve_path(args.outputs_dir), resolve_path(args.results_dir)
    (results / "figures").mkdir(parents=True, exist_ok=True)

    lr = lr_search_summary(outputs)
    if len(lr):
        lr.to_csv(results / "lr_search.csv", index=False)
        print("lr search (chosen):\n", lr[lr.chosen].to_string(index=False))

    runs = collect_runs(outputs)
    if runs.empty:
        print("no finished main runs yet")
        return
    patches = load_patches(runs)
    summaries = {}
    for t in THRESHOLDS:
        per_run = per_run_metrics(patches, t)
        per_run.to_csv(results / f"metrics_per_run_{t}.csv", index=False)
        summaries[t] = summarize(per_run)
        summaries[t].to_csv(results / f"summary_{t}.csv", index=False)
        paper_table(summaries[t]).to_latex(results / f"table_{t}.tex", index=False)
        per_continent(patches, t).to_csv(results / f"per_continent_{t}.csv", index=False)

    main_runs = patches[(patches.bands == "b267") & (patches.frac == 1.0)]
    if "unet" in set(main_runs.variant):
        deltas = [bootstrap_delta(patches, v, threshold_name=t, n_boot=args.n_boot).assign(threshold=t)
                  for t in THRESHOLDS for v in sorted(set(main_runs.variant) - {"unet"})]
        if deltas:
            pd.concat(deltas).to_csv(results / "bootstrap_vs_unet.csv", index=False)

    cost = cost_table(runs)
    cost.to_csv(results / "cost.csv", index=False)
    figures.learning_curve(per_run_metrics(patches, "t050"), results / "figures" / "learning_curve.png")
    figures.f1_per_continent(per_continent(patches, "t050"), results / "figures" / "f1_per_continent.png")
    figures.cost_vs_f1(cost, summaries["t050"], results / "figures" / "cost_vs_f1.png")
    print(f"results written to {results}")


if __name__ == "__main__":
    main()
