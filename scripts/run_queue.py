"""Run the experiment matrix in priority order, skipping finished runs: python -m scripts.run_queue --max-priority 1"""
import argparse
import subprocess

from flaresatformer.experiments import load_experiments, select_jobs
from flaresatformer.utils import CONFIG_DIR, REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default=str(CONFIG_DIR / "experiments.yaml"))
    parser.add_argument("--max-priority", type=int)
    parser.add_argument("--only-priority", type=int)
    parser.add_argument("--allow-untuned", action="store_true", help="run main jobs even if train.lr_tuned is false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed run")
    args = parser.parse_args()

    runnable, done, blocked = select_jobs(load_experiments(args.experiments), args.max_priority,
                                          args.only_priority, args.allow_untuned)
    print(f"{len(done)} done | {len(runnable)} to run | {len(blocked)} blocked "
          f"(set train.lr and train.lr_tuned: true in configs/models/*.yaml after priority 0)")
    for job, run_dir in runnable:
        print(f"[P{job.priority}] {run_dir}", flush=True)
        if args.dry_run:
            continue
        if subprocess.run(job.command(), cwd=REPO_ROOT).returncode != 0:
            print(f"[failed] {run_dir}")
            if not args.keep_going:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
