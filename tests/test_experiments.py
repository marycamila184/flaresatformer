from collections import Counter

from flaresatformer.experiments import expand, load_experiments, select_jobs


def test_full_matrix_size_and_order():
    jobs = load_experiments()
    assert Counter(j.priority for j in jobs) == {0: 18, 1: 24, 2: 48, 3: 8, 4: 36}
    assert [j.priority for j in jobs] == sorted(j.priority for j in jobs)


def test_every_job_has_its_own_run_dir():
    jobs = load_experiments()
    assert len({j.run_dir() for j in jobs}) == len(jobs)


def test_lr_search_jobs_skip_test_and_pin_lr():
    job = next(j for j in load_experiments() if j.kind == "lr_search")
    cmd = job.command()
    assert "--no-test" in cmd and "--lr" in cmd and job.epochs == 30 and job.fold == 1


def _spec(tmp_path):
    out = {"paths": {"outputs_dir": str(tmp_path)}}
    return {"models": ["unet"], "folds": [1, 2], "experiments": [
        {"priority": 0, "kind": "lr_search", "models": "all", "folds": [1], "seeds": [1], "lrs": [1e-3], "epochs": 30, "overrides": out},
        {"priority": 1, "models": "all", "seeds": [1], "overrides": out}]}


def test_untuned_models_are_blocked_from_main_runs(tmp_path):
    runnable, done, blocked = select_jobs(expand(_spec(tmp_path)))
    assert [j.kind for j, _ in runnable] == ["lr_search"] and len(blocked) == 2 and not done
    runnable, _, blocked = select_jobs(expand(_spec(tmp_path)), allow_untuned=True)
    assert len(runnable) == 3 and not blocked


def test_done_runs_are_skipped(tmp_path):
    jobs = expand(_spec(tmp_path))
    jobs[0].run_dir().mkdir(parents=True)
    (jobs[0].run_dir() / "DONE").write_text("x")
    runnable, done, _ = select_jobs(jobs, allow_untuned=True)
    assert len(done) == 1 and len(runnable) == 2
