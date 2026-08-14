"""Parse Harbor job/trial result.json into SkillsBench-compatible evaluation dicts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def short_task_id(task_name: str) -> str:
    """``terminal-bench/cad-model`` → ``cad-model``."""
    name = (task_name or "").strip().rstrip("/")
    if "/" in name:
        return name.rsplit("/", 1)[-1]
    return name


def reward_from_verifier(verifier_result: Optional[Dict[str, Any]]) -> float:
    if not isinstance(verifier_result, dict):
        return 0.0
    rewards = verifier_result.get("rewards")
    if isinstance(rewards, dict) and rewards:
        if "reward" in rewards:
            try:
                return float(rewards["reward"])
            except (TypeError, ValueError):
                return 0.0
        try:
            return float(next(iter(rewards.values())))
        except (TypeError, ValueError, StopIteration):
            return 0.0
    raw = verifier_result.get("reward")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def evaluation_from_trial(trial: Dict[str, Any]) -> Dict[str, Any]:
    verifier = trial.get("verifier_result") if isinstance(trial.get("verifier_result"), dict) else {}
    reward = reward_from_verifier(verifier)
    task_success = reward >= 1.0
    exc = trial.get("exception_info")
    error = None
    if isinstance(exc, dict) and exc.get("exception_message"):
        error = str(exc.get("exception_message"))
        if not task_success:
            reward = reward  # keep 0 unless verifier already scored
    return {
        "eval_mode": "harbor",
        "eval_backend": "harbor",
        "task_id": short_task_id(str(trial.get("task_name") or "")),
        "task_success": task_success,
        "reward": 1.0 if task_success else float(reward),
        "tests_passed": 1 if task_success else 0,
        "tests_failed": 0 if task_success else 1,
        "tests_skipped": 0,
        "tests_total": 1,
        "test_cases": [
            {
                "nodeid": "harbor.verifier",
                "outcome": "passed" if task_success else "failed",
            }
        ],
        "harbor_trial_name": trial.get("trial_name"),
        "harbor_rewards": (verifier or {}).get("rewards"),
        "error": error,
    }


def iter_trial_result_files(jobs_dir: Path) -> Iterator[Path]:
    root = jobs_dir.expanduser().resolve()
    if not root.is_dir():
        return
    # Single job dir (has result.json + trial subdirs) or parent of jobs.
    for path in sorted(root.rglob("result.json")):
        if path.parent == root:
            continue  # job-level result.json
        yield path


def is_finished_trial_result(data: Any) -> bool:
    """True when ``result.json`` is a completed Harbor trial (not a job-level file).

    Harbor writes this file only after a trial ends (verifier and/or exception).
    In-progress trial dirs have no ``result.json``.
    """
    if not isinstance(data, dict):
        return False
    if not data.get("task_name"):
        return False
    # Job-level result.json has ``n_total_trials`` / ``stats`` and no trial_name.
    if not data.get("trial_name"):
        return False
    return True


def load_trial_results(jobs_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in iter_trial_result_files(jobs_dir):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not is_finished_trial_result(data):
            continue
        data["_result_path"] = str(path)
        rows.append(data)
    return rows


def _trial_recency_key(trial: Dict[str, Any]) -> str:
    return str(trial.get("finished_at") or trial.get("started_at") or "")


def latest_finished_trials(jobs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Latest finished Harbor trial per short task id under ``jobs_dir``.

    ``jobs_dir`` may be one job directory or the parent of several jobs.
    """
    latest: Dict[str, Dict[str, Any]] = {}
    for trial in load_trial_results(jobs_dir):
        tid = short_task_id(str(trial.get("task_name") or ""))
        if not tid:
            continue
        prev = latest.get(tid)
        if prev is None or _trial_recency_key(trial) >= _trial_recency_key(prev):
            latest[tid] = trial
    return latest


def completed_task_ids_from_jobs(jobs_dir: Path) -> set[str]:
    """Task ids that already have a finished Harbor ``result.json``."""
    return set(latest_finished_trials(jobs_dir))


def envelope_from_trial(
    trial: Dict[str, Any],
    *,
    model: str,
    jobs_dir: Path,
    dataset_path: Path,
    harbor_env: str,
    experiment_dir: Optional[Path] = None,
    hermes_home: Optional[Path] = None,
    isolate_hermes_home: bool = False,
    hot_pool: Optional[bool] = None,
    hot_pool_persist: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evaluation = evaluation_from_trial(trial)
    task_id = evaluation["task_id"]
    agent_result = trial.get("agent_result") if isinstance(trial.get("agent_result"), dict) else {}
    n_in = agent_result.get("n_input_tokens")
    n_out = agent_result.get("n_output_tokens")
    n_cache = agent_result.get("n_cache_tokens")
    total = None
    if n_in is not None or n_out is not None:
        total = int(n_in or 0) + int(n_out or 0)
    row: Dict[str, Any] = {
        "schema": "terminalbench.hermes_run.v1",
        "benchmark": "terminal-bench",
        "eval_mode": "harbor",
        "skillsbench_task_id": task_id,
        "task_id": task_id,
        "task_name": trial.get("task_name"),
        "model": model,
        "harbor_env": harbor_env,
        "harbor_jobs_dir": str(jobs_dir),
        "harbor_trial_name": trial.get("trial_name"),
        "dataset_path": str(dataset_path),
        "hot_pool_enabled": hot_pool,
        "hot_pool_persist": hot_pool_persist,
        "isolate_hermes_home": isolate_hermes_home,
        "evaluation": evaluation,
        "run_conversation_result": {
            "completed": trial.get("exception_info") is None,
            "input_tokens": n_in,
            "output_tokens": n_out,
            "cache_read_tokens": n_cache,
            "total_tokens": total,
            "estimated_cost_usd": agent_result.get("cost_usd"),
            "api_calls": (agent_result.get("metadata") or {}).get("api_calls")
            if isinstance(agent_result.get("metadata"), dict)
            else None,
        },
        "started_at": trial.get("started_at"),
        "ts_end_iso": trial.get("finished_at"),
        "exception_info": trial.get("exception_info"),
    }
    if experiment_dir is not None:
        row["experiment_dir"] = str(experiment_dir)
    if hermes_home is not None:
        row["hermes_home"] = str(hermes_home)
    if extra:
        row.update(extra)
    return row
