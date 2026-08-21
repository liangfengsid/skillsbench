"""Harbor adapter: result parsing + driver command construction (no Docker)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_RESULTS = _REPO / "benchmark" / "harbor_adapter" / "results.py"
_DRIVER = _REPO / "benchmark" / "scripts" / "run_terminalbench_with_harbor.py"
_BENCH = _REPO / "benchmark"


def _load(path: Path, name: str, extra_paths=()):
    for p in extra_paths:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_reward_from_verifier_prefers_reward_key():
    results = _load(_RESULTS, "harbor_adapter_results")
    assert results.reward_from_verifier({"rewards": {"reward": 1, "other": 0}}) == 1.0
    assert results.reward_from_verifier({"rewards": {"score": 0.0}}) == 0.0
    assert results.reward_from_verifier({"reward": "1"}) == 1.0
    assert results.reward_from_verifier({}) == 0.0
    assert results.reward_from_verifier(None) == 0.0


def test_short_task_id_strips_dataset_prefix():
    results = _load(_RESULTS, "harbor_adapter_results")
    assert results.short_task_id("terminal-bench/cad-model") == "cad-model"
    assert results.short_task_id("cad-model") == "cad-model"
    assert results.short_task_id("cad-model/") == "cad-model"


def test_evaluation_from_trial_success_and_error():
    results = _load(_RESULTS, "harbor_adapter_results")
    ok = results.evaluation_from_trial(
        {
            "task_name": "terminal-bench/cad-model",
            "trial_name": "cad-model__abc",
            "verifier_result": {"rewards": {"reward": 1.0}},
        }
    )
    assert ok["eval_mode"] == "harbor"
    assert ok["task_id"] == "cad-model"
    assert ok["task_success"] is True
    assert ok["reward"] == 1.0

    fail = results.evaluation_from_trial(
        {
            "task_name": "broken-task",
            "verifier_result": {"rewards": {"reward": 0}},
            "exception_info": {"exception_message": "AgentTimeoutError"},
        }
    )
    assert fail["task_success"] is False
    assert fail["error"] == "AgentTimeoutError"


def test_envelope_and_load_trial_results(tmp_path):
    results = _load(_RESULTS, "harbor_adapter_results")
    job = tmp_path / "jobs" / "tb-harbor-test"
    trial = job / "cad-model__xyz"
    trial.mkdir(parents=True)
    (job / "result.json").write_text(
        json.dumps({"job_name": "tb-harbor-test"}), encoding="utf-8"
    )
    payload = {
        "task_name": "cad-model",
        "trial_name": "cad-model__xyz",
        "started_at": "2026-08-14T00:00:00Z",
        "finished_at": "2026-08-14T00:01:00Z",
        "verifier_result": {"rewards": {"reward": 1}},
        "agent_result": {
            "n_input_tokens": 10,
            "n_output_tokens": 5,
            "n_cache_tokens": 2,
            "cost_usd": 0.01,
            "metadata": {"api_calls": 3},
        },
        "exception_info": None,
    }
    (trial / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = results.load_trial_results(job)
    assert len(rows) == 1
    env = results.envelope_from_trial(
        rows[0],
        model="Qwen/Qwen3.6-27B",
        jobs_dir=tmp_path / "jobs",
        dataset_path=_REPO / "benchmark" / "terminal-bench" / "tasks",
        harbor_env="docker",
    )
    assert env["schema"] == "terminalbench.hermes_run.v1"
    assert env["eval_mode"] == "harbor"
    assert env["skillsbench_task_id"] == "cad-model"
    assert env["evaluation"]["task_success"] is True
    assert env["run_conversation_result"]["api_calls"] == 3
    assert env["run_conversation_result"]["total_tokens"] == 15


def test_build_harbor_command_uses_include_task_name():
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    cmd = driver.build_harbor_command(
        harbor_bin=["harbor"],
        dataset_path=Path("/tmp/tasks"),
        agent=driver._HERMES_AGENT,
        model="Qwen/Qwen3.6-27B",
        harbor_env="docker",
        jobs_dir=Path("/tmp/jobs"),
        job_name="tb-test",
        task_ids=["cad-model", "foo"],
        n_concurrent=1,
        max_iterations=90,
    )
    assert cmd[:3] == ["harbor", "run", "-p"]
    assert "--include-task-name" in cmd
    assert "--task-name" not in cmd
    assert cmd.count("cad-model") == 1
    assert "--ak" in cmd and "max_iterations=90" in cmd
    assert cmd[cmd.index("--agent-import-path") + 1] == driver._HERMES_AGENT
    assert "-a" not in cmd


def test_oracle_command_omits_agent_kwargs():
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    cmd = driver.build_harbor_command(
        harbor_bin=["python", "-m", "harbor"],
        dataset_path=Path("/tmp/tasks"),
        agent="oracle",
        model="",
        harbor_env="docker",
        jobs_dir=Path("/tmp/jobs"),
        job_name="oracle-job",
        task_ids=["cad-model"],
        n_concurrent=1,
        max_iterations=90,
    )
    assert cmd[cmd.index("-a") + 1] == "oracle"
    assert "--agent-import-path" not in cmd
    after_run = cmd[cmd.index("run") :]
    assert "-m" not in after_run
    assert "--ak" not in cmd


def test_collect_exclude_task_names_from_flag_and_remainder():
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    names = driver.collect_exclude_task_names(
        ["math-eval-grader"],
        ["--exclude-task-name", "jax-speedrun-gpu", "-x", "math-eval-grader"],
    )
    assert names == ["math-eval-grader", "jax-speedrun-gpu"]


def test_exclude_task_name_omitted_from_harbor_command(tmp_path, capsys):
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    root = tmp_path / "tasks"
    for name in ("keep-me", "math-eval-grader"):
        d = root / name
        d.mkdir(parents=True)
        (d / "instruction.md").write_text("x\n", encoding="utf-8")
    rc = driver.main(
        [
            "--all",
            "--dataset-path",
            str(root),
            "--oracle",
            "--dry-run",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--job-name",
            "dry-ex",
            "--exclude-task-name",
            "math-eval-grader",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "excluding 1 task(s): math-eval-grader" in out
    assert "--include-task-name keep-me" in out
    assert "--include-task-name math-eval-grader" not in out


def test_dry_run_prints_command(tmp_path, capsys):
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    tasks = tmp_path / "tasks" / "cad-model"
    tasks.mkdir(parents=True)
    (tasks / "instruction.md").write_text("do it\n", encoding="utf-8")
    rc = driver.main(
        [
            "--task",
            "cad-model",
            "--dataset-path",
            str(tmp_path / "tasks"),
            "--oracle",
            "--dry-run",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--job-name",
            "dry",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "harbor" in out
    assert "--include-task-name cad-model" in out
    assert "oracle" in out


def test_resume_skips_completed_jsonl(tmp_path, capsys):
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    root = tmp_path / "tasks"
    for name in ("keep-me", "done-already"):
        d = root / name
        d.mkdir(parents=True)
        (d / "instruction.md").write_text("x\n", encoding="utf-8")
    log = tmp_path / "runs.jsonl"
    log.write_text(
        json.dumps(
            {
                "schema": "terminalbench.hermes_run.v1",
                "skillsbench_task_id": "done-already",
                "evaluation": {"task_success": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rc = driver.main(
        [
            "--all",
            "--dataset-path",
            str(root),
            "--oracle",
            "--dry-run",
            "--resume",
            "--log-jsonl",
            str(log),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--job-name",
            "resume-job",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "skipping 1 completed" in out
    assert "1 from jsonl" in out
    assert "--include-task-name keep-me" in out
    assert "done-already" not in out.split("harbor run", 1)[-1]


def _write_harbor_trial(
    jobs_dir: Path,
    job_name: str,
    task_id: str,
    *,
    finished_at: str,
    reward: float = 1.0,
    trial_suffix: str = "xyz",
) -> None:
    trial_dir = jobs_dir / job_name / f"{task_id}__{trial_suffix}"
    trial_dir.mkdir(parents=True)
    payload = {
        "task_name": task_id,
        "trial_name": f"{task_id}__{trial_suffix}",
        "started_at": "2026-08-14T00:00:00Z",
        "finished_at": finished_at,
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": None,
    }
    (trial_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    (jobs_dir / job_name / "result.json").write_text(
        json.dumps({"n_total_trials": 1, "stats": {}}),
        encoding="utf-8",
    )


def test_latest_finished_trials_ignores_job_level_and_in_progress(tmp_path):
    results = _load(_RESULTS, "harbor_adapter_results")
    jobs = tmp_path / "jobs"
    _write_harbor_trial(jobs, "job-a", "cad-model", finished_at="2026-08-14T01:00:00Z")
    _write_harbor_trial(
        jobs,
        "job-b",
        "cad-model",
        finished_at="2026-08-14T02:00:00Z",
        trial_suffix="later",
        reward=0.0,
    )
    in_progress = jobs / "job-b" / "keep-me__wip"
    in_progress.mkdir(parents=True)
    (in_progress / "trial.log").write_text("running\n", encoding="utf-8")

    latest = results.latest_finished_trials(jobs)
    assert set(latest) == {"cad-model"}
    assert latest["cad-model"]["trial_name"] == "cad-model__later"
    assert results.completed_task_ids_from_jobs(jobs) == {"cad-model"}


def test_resume_skips_harbor_result_json_without_jsonl(tmp_path, capsys):
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    root = tmp_path / "tasks"
    for name in ("keep-me", "harbor-done"):
        d = root / name
        d.mkdir(parents=True)
        (d / "instruction.md").write_text("x\n", encoding="utf-8")
    jobs = tmp_path / "jobs"
    _write_harbor_trial(jobs, "tb-harbor-old", "harbor-done", finished_at="2026-08-14T01:00:00Z")
    log = tmp_path / "runs.jsonl"
    rc = driver.main(
        [
            "--all",
            "--dataset-path",
            str(root),
            "--oracle",
            "--dry-run",
            "--resume",
            "--log-jsonl",
            str(log),
            "--jobs-dir",
            str(jobs),
            "--job-name",
            "resume-job",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "from Harbor jobs" in out
    assert "--include-task-name keep-me" in out
    harbor_run = out.split("harbor run", 1)[-1]
    assert "harbor-done" not in harbor_run


def test_apply_harbor_resume_backfills_jsonl(tmp_path):
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    jobs = tmp_path / "jobs"
    _write_harbor_trial(jobs, "old-job", "cad-model", finished_at="2026-08-14T01:00:00Z")
    log = tmp_path / "runs.jsonl"
    kwargs = {
        "model": "m",
        "jobs_dir": jobs,
        "dataset_path": tmp_path / "tasks",
        "harbor_env": "docker",
    }
    plan = driver.apply_harbor_resume(
        ["cad-model", "keep-me"],
        log_jsonl=log,
        jobs_dir=jobs,
        envelope_kwargs=kwargs,
        dry_run=False,
    )
    assert plan["pending"] == ["keep-me"]
    assert plan["skipped"] == ["cad-model"]
    assert plan["n_from_harbor"] == 1
    assert plan["backfilled"] == 1
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["skillsbench_task_id"] == "cad-model"
    assert rows[0]["schema"] == "terminalbench.hermes_run.v1"
    again = driver.apply_harbor_resume(
        ["cad-model", "keep-me"],
        log_jsonl=log,
        jobs_dir=jobs,
        envelope_kwargs=kwargs,
        dry_run=False,
    )
    assert again["backfilled"] == 0
    assert again["n_from_jsonl"] == 1
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1


def test_allocate_job_name_avoids_existing_dir(tmp_path):
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    jobs = tmp_path / "jobs"
    (jobs / "tb-train").mkdir(parents=True)
    name = driver.allocate_job_name(jobs, "tb-train")
    assert name != "tb-train"
    assert name.startswith("tb-train-resume-")
    assert driver.allocate_job_name(jobs, "fresh") == "fresh"


def test_resume_allocates_new_job_name_when_dir_exists(tmp_path, capsys):
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    root = tmp_path / "tasks" / "keep-me"
    root.mkdir(parents=True)
    (root / "instruction.md").write_text("x\n", encoding="utf-8")
    jobs = tmp_path / "jobs"
    (jobs / "resume-job").mkdir(parents=True)
    log = tmp_path / "runs.jsonl"
    log.write_text("", encoding="utf-8")
    rc = driver.main(
        [
            "--all",
            "--dataset-path",
            str(tmp_path / "tasks"),
            "--oracle",
            "--dry-run",
            "--resume",
            "--log-jsonl",
            str(log),
            "--jobs-dir",
            str(jobs),
            "--job-name",
            "resume-job",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "resume-job-resume-" in out
    assert "--job-name resume-job-resume-" in out


def test_harbor_child_env_instruction_prefix():
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    env = driver.harbor_child_env(
        base={"PATH": "/usr/bin"},
        repo=_REPO,
        instruction_prefix="Use frozen skills.",
        hermes_home=Path("/tmp/exp/hermes_home"),
    )
    assert env["HERMES_HARBOR_INSTRUCTION_PREFIX"] == "Use frozen skills."
    assert env["HERMES_HOME"] == str(Path("/tmp/exp/hermes_home").resolve())


def test_skills_overlay_dry_run(tmp_path, capsys):
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    tasks = tmp_path / "tasks" / "cad-model"
    tasks.mkdir(parents=True)
    (tasks / "instruction.md").write_text("x\n", encoding="utf-8")
    overlay = tmp_path / "lib"
    overlay.mkdir()
    (overlay / "evo-x").mkdir()
    (overlay / "evo-x" / "SKILL.md").write_text("# x\n", encoding="utf-8")
    exp = tmp_path / "exp"
    rc = driver.main(
        [
            "--task",
            "cad-model",
            "--dataset-path",
            str(tmp_path / "tasks"),
            "--experiment-dir",
            str(exp),
            "--isolate-hermes-home",
            "--skills-overlay",
            str(overlay),
            "--run-method",
            "coevoskills",
            "--run-phase",
            "frozen_eval",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run skills overlay" in out


def test_overlay_skill_packages(tmp_path):
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    dest.mkdir()
    pack = src / "evo-a"
    pack.mkdir(parents=True)
    (pack / "SKILL.md").write_text("# a\n", encoding="utf-8")
    assert driver.overlay_skill_packages(src, dest) == 1
    assert (dest / "evo-a" / "SKILL.md").is_file()


def test_envelope_extra_fields_baseline_schema():
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    extra = driver.envelope_extra_fields(method="coevoskills", phase="frozen_eval")
    assert extra["method"] == "coevoskills"
    assert extra["phase"] == "frozen_eval"
    assert extra["schema"] == "terminalbench.baseline_run.v1"


def test_harbor_child_env_disables_hot_pool():
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    env = driver.harbor_child_env(
        base={"PATH": "/usr/bin"},
        repo=_REPO,
        max_iterations=60,
        hot_pool=False,
        hermes_home=Path("/tmp/exp/hermes_home"),
    )
    assert env["HERMES_HOT_POOL_ENABLED"] == "0"
    assert "HERMES_HOT_POOL_PATH" not in env
    assert env["HERMES_HOME"] == str(Path("/tmp/exp/hermes_home").resolve())
    assert env["HERMES_HARBOR_MAX_ITERATIONS"] == "60"
    assert str(_REPO) in env["PYTHONPATH"]


def test_isolate_requires_experiment_dir(tmp_path):
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    tasks = tmp_path / "tasks" / "cad-model"
    tasks.mkdir(parents=True)
    (tasks / "instruction.md").write_text("x\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        driver.main(
            [
                "--task",
                "cad-model",
                "--dataset-path",
                str(tmp_path / "tasks"),
                "--isolate-hermes-home",
                "--dry-run",
            ]
        )


def test_dry_run_isolate_and_no_hot_pool(tmp_path, capsys):
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    tasks = tmp_path / "tasks" / "cad-model"
    tasks.mkdir(parents=True)
    (tasks / "instruction.md").write_text("x\n", encoding="utf-8")
    exp = tmp_path / "tb_exp"
    rc = driver.main(
        [
            "--task",
            "cad-model",
            "--dataset-path",
            str(tmp_path / "tasks"),
            "--experiment-dir",
            str(exp),
            "--isolate-hermes-home",
            "--no-hot-pool",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run isolate HERMES_HOME" in out
    assert str(exp / "hermes_home") in out
    assert str(exp / "jobs") in out


def test_prepare_harbor_isolate_seeds_hermes_home(tmp_path, monkeypatch):
    driver = _load(_DRIVER, "run_terminalbench_with_harbor", extra_paths=(_BENCH, _REPO))
    real = tmp_path / "real_hermes"
    polluted = real / "skills" / "polluted"
    polluted.mkdir(parents=True)
    (polluted / "SKILL.md").write_text("# polluted\n", encoding="utf-8")
    (real / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
    bundled = tmp_path / "bundled_skills"
    (bundled / "demo").mkdir(parents=True)
    (bundled / "demo" / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(real))
    exp = tmp_path / "exp"
    dataset = tmp_path / "terminal-bench" / "tasks"
    dataset.mkdir(parents=True)
    manifest = driver.prepare_harbor_isolate(
        experiment_dir=exp,
        dataset_path=dataset,
        source_skills=bundled,
    )
    home = Path(manifest["hermes_home"])
    assert (home / "skills" / "demo" / "SKILL.md").is_file()
    assert not (home / "skills" / "polluted").exists()
    assert (home / "config.yaml").is_symlink()
    assert manifest["isolate_hermes_home"] is True
    assert manifest["bundled_skills_dir"] == str(bundled.resolve())



def test_hermes_harbor_agent_import_optional():
    spec = importlib.util.find_spec("harbor")
    if spec is None:
        pytest.importorskip("harbor")
    from benchmark.harbor_adapter.hermes_agent import HermesHarborAgent

    assert HermesHarborAgent.name() == "hermes"
    assert HermesHarborAgent.SUPPORTS_ATIF is False
