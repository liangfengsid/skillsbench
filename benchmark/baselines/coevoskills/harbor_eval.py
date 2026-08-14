"""Harbor helpers for CoEvoSkills on Terminal-Bench."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import skill_io

COEVO_HARBOR_AGENT = "benchmark.baselines.coevoskills.harbor_agent:CoEvoHarborAgent"

FROZEN_EVAL_PREFIX = (
    "You have a frozen CoEvoSkills library (evo-* packages) in your Hermes "
    "skills directory. Prefer those skills to solve the task. Do not create "
    "new skills. Do not read /tests or /solution."
)

EVOLVE_MATERIALIZE_PREFIX = (
    "You have evolving CoEvoSkills packages (evo-*) in your Hermes skills "
    "directory. Use them to complete this Terminal-Bench task in /app. Do not "
    "create new skill packages. Do not read /tests or /solution."
)


def load_harbor_driver(hermes_root: Path):
    scripts = hermes_root / "benchmark" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    if str(hermes_root) not in sys.path:
        sys.path.insert(0, str(hermes_root))
    path = scripts / "run_terminalbench_with_harbor.py"
    spec = importlib.util.spec_from_file_location("run_terminalbench_with_harbor", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def harbor_env_hint(paths: Dict[str, Path]) -> str:
    return (
        f"Author evo-* skills for a Linux Terminal-Bench sandbox (cwd /app). "
        f"Write skills under {paths['skills']}. Do not finish the task on this host. "
        f"A Harbor trial will execute the skills inside the sandbox and export /app "
        f"to {paths['artifacts']}. Task environment files (reference only) are under "
        f"{paths['env']}."
    )


def harbor_oracle_from_materialize(
    *,
    last_materialize: Optional[Dict[str, Any]] = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Reuse the Harbor verifier from this iteration's materialize trial (Alg. 1 Φ_GT)."""
    ev = (last_materialize or {}).get("evaluation") if last_materialize else None
    if not isinstance(ev, dict):
        return {
            "task_success": False,
            "reward": 0.0,
            "ran": False,
            "error": "Harbor materialize produced no verifier result",
        }
    out = dict(ev)
    out.setdefault("ran", True)
    return out


def run_materialize_trial(
    *,
    hermes_root: Path,
    dataset_path: Path,
    task_id: str,
    paths: Dict[str, Path],
    isolate_home: Path,
    jobs_dir: Path,
    model: str,
    harbor_env: str,
    max_iterations: int,
    iteration: int,
    experiment_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """One Harbor trial that executes current evo-* skills and exports /app."""
    driver = load_harbor_driver(hermes_root)
    isolate_home = Path(isolate_home).expanduser().resolve()
    isolate_home.mkdir(parents=True, exist_ok=True)
    n_ov = skill_io.overlay_skills(paths["skills"], isolate_home / "skills")
    artifacts = paths["artifacts"]
    if artifacts.exists():
        shutil.rmtree(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    jobs_dir = Path(jobs_dir).expanduser().resolve()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_name = datetime.now(timezone.utc).strftime(
        f"coevo-{task_id}-m{iteration}-%Y%m%dT%H%M%SZ"
    )
    extra = driver.envelope_extra_fields(method="coevoskills", phase="evolve_materialize")
    envelope_kwargs: Dict[str, Any] = {
        "model": model,
        "jobs_dir": jobs_dir,
        "dataset_path": Path(dataset_path).expanduser().resolve(),
        "harbor_env": harbor_env,
        "experiment_dir": experiment_dir,
        "hermes_home": isolate_home,
        "isolate_hermes_home": True,
        "hot_pool": False,
        "hot_pool_persist": None,
        "extra": extra,
    }
    cmd = driver.build_harbor_command(
        harbor_bin=driver._which_harbor(),
        dataset_path=Path(dataset_path).expanduser().resolve(),
        agent=COEVO_HARBOR_AGENT,
        model=model,
        harbor_env=harbor_env,
        jobs_dir=jobs_dir,
        job_name=job_name,
        task_ids=[task_id],
        n_concurrent=1,
        max_iterations=max_iterations,
    )
    env = driver.harbor_child_env(
        base=None,
        repo=hermes_root,
        max_iterations=max_iterations,
        hot_pool=False,
        hermes_home=isolate_home,
        instruction_prefix=EVOLVE_MATERIALIZE_PREFIX,
        extra_env={"HERMES_COEVO_APP_EXPORT": str(artifacts.resolve())},
    )
    result = driver.execute_harbor_run(
        cmd=cmd,
        env=env,
        jobs_dir=jobs_dir,
        job_name=job_name,
        envelope_kwargs=envelope_kwargs,
        log_jsonl=None,
        print_summary=False,
    )
    envelopes = result.get("envelopes") or []
    envelope = envelopes[0] if envelopes else None
    evaluation = (envelope or {}).get("evaluation") or {
        "task_success": False,
        "reward": 0.0,
        "ran": False,
        "error": f"no Harbor trial for {task_id} (overlay_n={n_ov})",
    }
    return {
        "evaluation": evaluation,
        "envelope": envelope,
        "job_name": job_name,
        "n_overlay": n_ov,
        "returncode": result.get("returncode"),
    }


def frozen_eval_argv(
    *,
    dataset_path: Path,
    split_file: Path,
    split_part: str,
    model: str,
    harbor_env: str,
    experiment_dir: Path,
    library_dir: Path,
    log_jsonl: Path,
    max_iterations: int,
    n_concurrent: int = 1,
    resume: bool = False,
    print_summary: bool = True,
    job_name: Optional[str] = None,
    per_task: Optional[str] = None,
    exclude_task_names: Optional[List[str]] = None,
) -> List[str]:
    argv: List[str] = [
        "--dataset-path",
        str(dataset_path),
        "--split-file",
        str(split_file),
        "--split-part",
        split_part,
        "--model",
        model,
        "--env",
        harbor_env,
        "--experiment-dir",
        str(experiment_dir),
        "--isolate-hermes-home",
        "--no-hot-pool",
        "--skills-overlay",
        str(library_dir),
        "--instruction-prefix",
        FROZEN_EVAL_PREFIX,
        "--run-method",
        "coevoskills",
        "--run-phase",
        "frozen_eval",
        "--log-jsonl",
        str(log_jsonl),
        "--max-iterations",
        str(max_iterations),
        "--n-concurrent",
        str(n_concurrent),
    ]
    if per_task:
        argv.extend(["--task", per_task])
    else:
        argv.append("--all")
    if print_summary:
        argv.append("--print-summary")
    if resume:
        argv.append("--resume")
    if job_name:
        argv.extend(["--job-name", job_name])
    for name in exclude_task_names or []:
        argv.extend(["--exclude-task-name", name])
    return argv
