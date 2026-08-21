#!/usr/bin/env python3
"""
Official Terminal-Bench eval via Harbor.

Hermes runs as a Harbor agent: the LLM loop stays on the host, but ``terminal``
and file tools exec inside Harbor's task sandbox (``/app``). Harbor then runs
the real ``tests/test.sh`` verifier (Docker by default, Modal optional).

This is the official Terminal-Bench eval path: Hermes ``AIAgent`` solves
inside Harbor sandboxes, then Harbor runs ``tests/test.sh``.

Requires Docker (or a Harbor cloud env) and::

    pip install -e ".[harbor]"

Examples (from Hermes repo root):

  # Oracle sanity check on one task
  python3 benchmark/scripts/run_terminalbench_with_harbor.py \\
      --task cad-model --oracle --env docker

  # Hermes on the stratified train split
  python3 benchmark/scripts/run_terminalbench_with_harbor.py --all \\
      --split-file benchmark/terminalbench_splits/stratified_v1.json \\
      --split-part train \\
      --model Qwen/Qwen3.6-27B --env docker \\
      --experiment-dir benchmark/runs/tb_harbor_train \\
      --isolate-hermes-home --no-hot-pool \\
      --log-jsonl benchmark/runs/tb_harbor_train/runs.jsonl --print-summary --resume

  # After an interrupt: same command. --resume skips tasks that already have
  # a JSONL row *or* a Harbor trial result.json under DIR/jobs, and backfills
  # JSONL for artifacts that never got logged.

  python3 benchmark/scripts/aggregate_skillsbench_runs.py \\
      benchmark/runs/tb_harbor_train/runs.jsonl --print-summary
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT = Path(__file__).resolve()
_REPO = _SCRIPT.parents[2]
_BENCH = _REPO / "benchmark"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))
if str(_SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT.parent))

from harbor_adapter.results import (  # noqa: E402
    envelope_from_trial,
    latest_finished_trials,
    load_trial_results,
)

_HERMES_AGENT = "benchmark.harbor_adapter.hermes_agent:HermesHarborAgent"
_DEFAULT_TASKS = _REPO / "benchmark" / "terminal-bench" / "tasks"


def _is_import_path_agent(agent: str) -> bool:
    """Custom Harbor agents are ``module.path:ClassName``."""
    return ":" in agent


def collect_exclude_task_names(
    named: Optional[List[str]] = None,
    extra_argv: Optional[List[str]] = None,
) -> List[str]:
    """Deduped task ids from ``--exclude-task-name`` and Harbor remainder argv."""
    out: List[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        name = str(raw).strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)

    for item in named or []:
        _add(item)
    tokens = list(extra_argv or [])
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("--exclude-task-name", "-x") and i + 1 < len(tokens):
            _add(tokens[i + 1])
            i += 2
            continue
        if tok.startswith("--exclude-task-name="):
            _add(tok.split("=", 1)[1])
        i += 1
    return out


def agent_cli_flags(agent: str) -> List[str]:
    """Flags that load ``agent`` on Harbor 0.3.x and 0.21+.

    Harbor 0.3.x types ``-a`` as a built-in ``AgentName`` enum, so import
    paths must use ``--agent-import-path``. Harbor 0.21+ accepts
    ``module:Class`` on ``-a`` but still honors ``--agent-import-path``.
    Do not pass ``-a hermes`` with an import path: 0.3.x prefers the
    built-in in-container ``hermes`` agent over ``import_path``.
    """
    if _is_import_path_agent(agent):
        return ["--agent-import-path", agent]
    return ["-a", agent]


def _which_harbor() -> List[str]:
    exe = shutil.which("harbor")
    if exe:
        return [exe]
    return [sys.executable, "-m", "harbor"]


def _harbor_importable() -> bool:
    return importlib.util.find_spec("harbor") is not None


def _load_split_ids(split_file: Path, part: str) -> List[str]:
    from make_skillsbench_splits import resolve_split_task_ids

    return list(resolve_split_task_ids(split_file, part))


def discover_task_ids(tasks_dir: Path) -> List[str]:
    if not tasks_dir.is_dir():
        raise SystemExit(f"Tasks directory not found: {tasks_dir}")
    return sorted(
        p.name
        for p in tasks_dir.iterdir()
        if p.is_dir() and (p / "instruction.md").is_file()
    )


def build_harbor_command(
    *,
    harbor_bin: List[str],
    dataset_path: Path,
    agent: str,
    model: str,
    harbor_env: str,
    jobs_dir: Path,
    job_name: str,
    task_ids: List[str],
    n_concurrent: int,
    max_iterations: Optional[int] = None,
    extra: Optional[List[str]] = None,
) -> List[str]:
    cmd = [
        *harbor_bin,
        "run",
        "-p",
        str(dataset_path),
        *agent_cli_flags(agent),
        "--env",
        harbor_env,
        "--jobs-dir",
        str(jobs_dir),
        "--job-name",
        job_name,
        "-n",
        str(n_concurrent),
    ]
    if model:
        cmd.extend(["-m", model])
    if max_iterations is not None and agent != "oracle":
        cmd.extend(["--ak", f"max_iterations={int(max_iterations)}"])
    for tid in task_ids:
        cmd.extend(["--include-task-name", tid])
    if extra:
        cmd.extend(extra)
    return cmd


_HOT_POOL_ENV_KEYS = (
    "HERMES_HOT_POOL_ENABLED",
    "HERMES_HOT_POOL_PERSIST",
    "HERMES_HOT_POOL_PATH",
)


def harbor_child_env(
    *,
    base: Optional[Dict[str, str]] = None,
    repo: Path = _REPO,
    max_iterations: int = 90,
    hot_pool: Optional[bool] = None,
    hot_pool_persist: Optional[str] = None,
    hermes_home: Optional[Path] = None,
    instruction_prefix: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Env for ``harbor run`` so the host-side Hermes agent matches SkillsBench isolate."""
    env = dict(base if base is not None else os.environ)
    py_path = str(repo)
    if env.get("PYTHONPATH"):
        py_path = py_path + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = py_path
    env["HERMES_HARBOR_MAX_ITERATIONS"] = str(max_iterations)
    for key in _HOT_POOL_ENV_KEYS:
        env.pop(key, None)
    env.pop("HERMES_HARBOR_INSTRUCTION_PREFIX", None)
    if hot_pool is False:
        env["HERMES_HOT_POOL_ENABLED"] = "0"
    elif hot_pool is True:
        env["HERMES_HOT_POOL_ENABLED"] = "1"
        if hot_pool_persist:
            env["HERMES_HOT_POOL_PERSIST"] = "1"
            env["HERMES_HOT_POOL_PATH"] = str(Path(hot_pool_persist).expanduser().resolve())
    elif hot_pool_persist:
        env["HERMES_HOT_POOL_PERSIST"] = "1"
        env["HERMES_HOT_POOL_PATH"] = str(Path(hot_pool_persist).expanduser().resolve())
    if hermes_home is not None:
        env["HERMES_HOME"] = str(Path(hermes_home).expanduser().resolve())
    if instruction_prefix:
        env["HERMES_HARBOR_INSTRUCTION_PREFIX"] = str(instruction_prefix)
    if extra_env:
        for key, value in extra_env.items():
            if value is None:
                env.pop(key, None)
            else:
                env[str(key)] = str(value)
    return env


def prepare_harbor_isolate(
    *,
    experiment_dir: Path,
    dataset_path: Path,
    reset_skills: bool = False,
    source_skills: Optional[Path] = None,
) -> Dict[str, Any]:
    """Seed ``DIR/hermes_home`` like SkillsBench ``--isolate-hermes-home`` (no task copies)."""
    from skillsbench_experiment_workspace import prepare_experiment_workspace

    return prepare_experiment_workspace(
        experiment_dir=experiment_dir,
        source_skillsbench_root=dataset_path.parent,
        task_ids=[],
        isolate_hermes_home=True,
        apply_hermes_home_env=False,
        reset_outputs=reset_skills,
        source_skills=source_skills,
        hermes_root=_REPO,
        dataset_dirname="terminal-bench",
    )


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def resolve_jobs_dir(
    *,
    jobs_dir: Optional[Path],
    experiment_dir: Optional[Path],
    repo: Path = _REPO,
) -> Path:
    if jobs_dir is not None:
        return jobs_dir.expanduser().resolve()
    if experiment_dir is not None:
        return experiment_dir.expanduser().resolve() / "jobs"
    return repo / "benchmark" / "runs" / "harbor_jobs"


def allocate_job_name(jobs_dir: Path, requested: str) -> str:
    """Avoid reusing an existing Harbor job dir when the task list has changed.

    Harbor refuses to resume a job whose ``config.json`` does not match. After
    ``--resume`` filters completed tasks, a subset list would collide — pick a
    fresh name instead.
    """
    if not (jobs_dir / requested).exists():
        return requested
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = f"{requested}-resume-{stamp}"
    if not (jobs_dir / candidate).exists():
        return candidate
    return f"{candidate}-{os.getpid()}"


def _row_task_id(row: Dict[str, Any]) -> str:
    return str(row.get("skillsbench_task_id") or row.get("task_id") or "")


def append_jsonl_missing(path: Path, rows: List[Dict[str, Any]]) -> int:
    """Append rows whose task id is not already a finished (non-error) JSONL row."""
    from skillsbench_batch_progress import load_completed_task_ids

    done = set(load_completed_task_ids(path))
    new_rows: List[Dict[str, Any]] = []
    for row in rows:
        tid = _row_task_id(row)
        if not tid or tid in done:
            continue
        new_rows.append(row)
        done.add(tid)
    if new_rows:
        write_jsonl(path, new_rows)
    return len(new_rows)


def envelopes_from_trials(
    trials: List[Dict[str, Any]],
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    return [envelope_from_trial(trial, **kwargs) for trial in trials]


def apply_harbor_resume(
    run_ids: List[str],
    *,
    log_jsonl: Path,
    jobs_dir: Path,
    envelope_kwargs: Dict[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Skip tasks finished in JSONL or Harbor jobs; backfill JSONL from jobs."""
    from skillsbench_batch_progress import load_completed_task_ids

    jsonl_done = load_completed_task_ids(log_jsonl)
    harbor_trials = latest_finished_trials(jobs_dir)
    harbor_done = set(harbor_trials)
    done = jsonl_done | harbor_done
    skipped = [tid for tid in run_ids if tid in done]
    pending = [tid for tid in run_ids if tid not in done]
    n_from_jsonl = sum(1 for tid in skipped if tid in jsonl_done)
    n_from_harbor = len(skipped) - n_from_jsonl

    backfilled = 0
    if not dry_run:
        missing_trials = [
            harbor_trials[tid] for tid in harbor_done if tid not in jsonl_done
        ]
        backfilled = append_jsonl_missing(
            log_jsonl,
            envelopes_from_trials(missing_trials, **envelope_kwargs),
        )
    return {
        "pending": pending,
        "skipped": skipped,
        "n_from_jsonl": n_from_jsonl,
        "n_from_harbor": n_from_harbor,
        "backfilled": backfilled,
    }


def overlay_skill_packages(src: Path, dest_skills: Path) -> int:
    """Copy ``src/<skill>/SKILL.md`` packages into ``dest_skills`` (overwrite by name)."""
    src = Path(src).expanduser().resolve()
    dest_skills = Path(dest_skills).expanduser().resolve()
    if not src.is_dir():
        return 0
    dest_skills.mkdir(parents=True, exist_ok=True)
    n = 0
    for child in sorted(src.iterdir()):
        if not child.is_dir() or not (child / "SKILL.md").is_file():
            continue
        target = dest_skills / child.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(child, target)
        n += 1
    return n


def envelope_extra_fields(
    *,
    method: Optional[str] = None,
    phase: Optional[str] = None,
    schema: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    extra: Dict[str, Any] = {}
    if method:
        extra["method"] = method
        extra["schema"] = schema or (
            "terminalbench.hermes_run.v1"
            if method == "hermes"
            else "terminalbench.baseline_run.v1"
        )
    elif schema:
        extra["schema"] = schema
    if phase:
        extra["phase"] = phase
    return extra or None


def execute_harbor_run(
    *,
    cmd: List[str],
    env: Dict[str, str],
    jobs_dir: Path,
    job_name: str,
    envelope_kwargs: Dict[str, Any],
    log_jsonl: Optional[Path] = None,
    print_summary: bool = False,
) -> Dict[str, Any]:
    """Run ``harbor`` and parse/salvage trial envelopes. Returns a result dict."""
    print(f"[harbor] {' '.join(cmd)}", flush=True)
    proc_code = 1
    interrupted = False
    try:
        proc = subprocess.run(cmd, env=env, check=False)
        proc_code = proc.returncode
    except FileNotFoundError:
        print(
            "Harbor CLI not found. Install with: pip install -e '.[harbor]'\n"
            "Also require Docker (or pass --env modal / --env daytona).",
            file=sys.stderr,
        )
        return {
            "returncode": 1,
            "envelopes": [],
            "job_name": job_name,
            "interrupted": False,
        }
    except KeyboardInterrupt:
        interrupted = True
        proc_code = 130
        print(
            "[harbor] interrupted; salvaging finished Harbor trials...",
            flush=True,
        )

    job_root = jobs_dir / job_name
    trials = load_trial_results(job_root if job_root.is_dir() else jobs_dir)
    envelopes = envelopes_from_trials(trials, **envelope_kwargs)
    if log_jsonl is not None:
        written = append_jsonl_missing(log_jsonl.expanduser(), envelopes)
        if interrupted and written:
            print(
                f"[harbor] salvaged {written} trial(s) into {log_jsonl}",
                flush=True,
            )

    n = len(envelopes)
    ok = sum(1 for e in envelopes if (e.get("evaluation") or {}).get("task_success"))
    if n:
        print(
            f"[harbor] job={job_name} trials={n} passed={ok} rate={ok / n:.3f}",
            flush=True,
        )
    else:
        print(f"[harbor] job={job_name} no trial results parsed", flush=True)
    if print_summary:
        for e in envelopes:
            ev = e.get("evaluation") or {}
            print(
                f"  {e.get('skillsbench_task_id')}: "
                f"success={ev.get('task_success')!r} reward={ev.get('reward')!r}",
                flush=True,
            )
    return {
        "returncode": 130 if interrupted else (0 if proc_code == 0 else proc_code),
        "envelopes": envelopes,
        "job_name": job_name,
        "interrupted": interrupted,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Official Terminal-Bench eval: Hermes agent inside Harbor sandboxes.",
    )
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--task", metavar="TASK_ID", help="Single task directory name.")
    g.add_argument("--all", action="store_true", help="Run every discovered / split task.")
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument(
        "--dataset-path",
        type=Path,
        default=_DEFAULT_TASKS,
        help=f"Directory of Harbor tasks (default: {_DEFAULT_TASKS}).",
    )
    p.add_argument("--split-file", type=Path, default=None)
    p.add_argument("--split-part", choices=("train", "test", "val", "all"), default="train")
    p.add_argument("--model", type=str, default="")
    p.add_argument(
        "--oracle",
        action="store_true",
        help="Run Harbor's oracle agent instead of Hermes (verifier sanity check).",
    )
    p.add_argument(
        "--env",
        dest="harbor_env",
        default="docker",
        help="Harbor environment type (docker, modal, daytona, …). Default: docker.",
    )
    p.add_argument(
        "--jobs-dir",
        type=Path,
        default=None,
        help="Harbor jobs output directory (default: <experiment-dir>/jobs or benchmark/runs/harbor_jobs).",
    )
    p.add_argument("--job-name", type=str, default=None)
    p.add_argument("--n-concurrent", type=int, default=1)
    p.add_argument("--max-iterations", type=int, default=90)
    p.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Experiment workspace. Defaults --jobs-dir to DIR/jobs. "
            "Does not copy Terminal-Bench tasks (Harbor sandboxes those). "
            "Use with --isolate-hermes-home for a sandboxed HERMES_HOME."
        ),
    )
    p.add_argument(
        "--isolate-hermes-home",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "With --experiment-dir: set HERMES_HOME=DIR/hermes_home for the Harbor "
            "child (writable copy of repo skills/, not ~/.hermes/skills; "
            "config.yaml / .env / SOUL.md symlinked to the real ~/.hermes). "
            "Matches SkillsBench isolate."
        ),
    )
    p.add_argument(
        "--reset-task-workspaces",
        action="store_true",
        help=(
            "With --isolate-hermes-home: re-seed DIR/hermes_home/skills from "
            "the repo skills/ tree (not ~/.hermes/skills)."
        ),
    )
    hot_pool_group = p.add_mutually_exclusive_group()
    hot_pool_group.add_argument(
        "--hot-pool",
        dest="hot_pool",
        action="store_const",
        const=True,
        help="Enable hot skill pool for this run (overrides config.yaml).",
    )
    hot_pool_group.add_argument(
        "--no-hot-pool",
        dest="hot_pool",
        action="store_const",
        const=False,
        help="Disable hot skill pool for this run (overrides config.yaml).",
    )
    p.set_defaults(hot_pool=None)
    p.add_argument(
        "--hot-pool-persist",
        type=str,
        default=None,
        metavar="PATH",
        help="Persist hot skill key points to PATH (requires hot pool enabled).",
    )
    p.add_argument("--log-jsonl", type=Path, default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip tasks that already have a finished (non-error) row in --log-jsonl "
            "or a Harbor trial result.json under --jobs-dir. Backfills JSONL from "
            "Harbor artifacts that were never logged (interrupt mid-job)."
        ),
    )
    p.add_argument(
        "--skills-overlay",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Copy skill packages from DIR into the isolate HERMES_HOME/skills "
            "(requires --isolate-hermes-home). Used by CoEvoSkills frozen eval."
        ),
    )
    p.add_argument(
        "--instruction-prefix",
        type=str,
        default=None,
        help="Prepended to the Harbor agent's user message (e.g. frozen-skill instructions).",
    )
    p.add_argument(
        "--run-method",
        type=str,
        default=None,
        help="JSONL method tag (default: hermes). CoEvoSkills uses coevoskills.",
    )
    p.add_argument(
        "--run-phase",
        type=str,
        default=None,
        help="JSONL phase tag (e.g. frozen_eval).",
    )
    p.add_argument("--print-summary", action="store_true")
    p.add_argument(
        "--exclude-task-name",
        action="append",
        default=None,
        metavar="TASK_ID",
        help=(
            "Skip this task id (repeatable). Applied before Harbor run so GPU "
            "tasks never reach docker. Same names as Harbor -x."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the harbor run command and exit.",
    )
    p.add_argument(
        "harbor_args",
        nargs="*",
        help="Extra args forwarded to `harbor run` after `--`.",
    )
    args = p.parse_args(argv)

    dataset_path = args.dataset_path.expanduser().resolve()
    task_ids = discover_task_ids(dataset_path)
    if args.split_file:
        split_ids = _load_split_ids(args.split_file.expanduser(), args.split_part)
        unknown = [tid for tid in split_ids if tid not in task_ids]
        if unknown:
            preview = ", ".join(unknown[:5])
            p.error(f"Split file lists unknown task ids: {preview}")
        task_ids = split_ids
    excludes = collect_exclude_task_names(args.exclude_task_name, args.harbor_args)
    if excludes:
        skip = set(excludes)
        dropped = [tid for tid in task_ids if tid in skip]
        task_ids = [tid for tid in task_ids if tid not in skip]
        if dropped:
            print(
                f"[harbor] excluding {len(dropped)} task(s): {', '.join(dropped)}",
                flush=True,
            )
    if args.list_tasks:
        for tid in task_ids:
            print(tid)
        return 0
    if not args.task and not args.all:
        p.error("Specify --task TASK_ID, --all, or --list-tasks.")
    if args.task:
        if args.task in set(excludes):
            p.error(f"Task {args.task!r} is excluded by --exclude-task-name")
        if args.task not in task_ids:
            p.error(f"Unknown task {args.task!r}")
        run_ids = [args.task]
    else:
        run_ids = list(task_ids)

    if args.isolate_hermes_home and not args.experiment_dir:
        p.error("--isolate-hermes-home requires --experiment-dir")
    if args.reset_task_workspaces and not args.isolate_hermes_home:
        p.error("--reset-task-workspaces requires --isolate-hermes-home")
    if args.hot_pool is False and args.hot_pool_persist:
        p.error("--hot-pool-persist cannot be used with --no-hot-pool.")
    if args.resume and not args.log_jsonl:
        p.error("--resume requires --log-jsonl")
    if args.skills_overlay and not args.isolate_hermes_home:
        p.error("--skills-overlay requires --isolate-hermes-home")

    experiment_dir = (
        args.experiment_dir.expanduser().resolve() if args.experiment_dir else None
    )
    jobs_dir = resolve_jobs_dir(
        jobs_dir=args.jobs_dir,
        experiment_dir=experiment_dir,
    )
    isolate_home: Optional[Path] = None
    if experiment_dir is not None and args.isolate_hermes_home and not args.oracle:
        isolate_home = experiment_dir / "hermes_home"

    hot_pool_persist = args.hot_pool_persist
    if (
        experiment_dir is not None
        and not hot_pool_persist
        and args.hot_pool is True
    ):
        from skillsbench_experiment_workspace import experiment_hot_pool_path

        hot_pool_persist = str(experiment_hot_pool_path(experiment_dir))

    envelope_kwargs: Dict[str, Any] = {
        "model": args.model,
        "jobs_dir": jobs_dir,
        "dataset_path": dataset_path,
        "harbor_env": args.harbor_env,
        "experiment_dir": experiment_dir,
        "hermes_home": isolate_home,
        "isolate_hermes_home": bool(args.isolate_hermes_home),
        "hot_pool": args.hot_pool,
        "hot_pool_persist": hot_pool_persist,
    }
    extra = envelope_extra_fields(method=args.run_method, phase=args.run_phase)
    if extra:
        envelope_kwargs["extra"] = extra

    if args.resume:
        plan = apply_harbor_resume(
            run_ids,
            log_jsonl=args.log_jsonl.expanduser(),
            jobs_dir=jobs_dir,
            envelope_kwargs=envelope_kwargs,
            dry_run=args.dry_run,
        )
        run_ids = plan["pending"]
        skipped = plan["skipped"]
        if skipped:
            print(
                f"[harbor] resume: skipping {len(skipped)} completed task(s) "
                f"({plan['n_from_jsonl']} from jsonl, "
                f"{plan['n_from_harbor']} from Harbor jobs)",
                flush=True,
            )
        if plan["backfilled"]:
            print(
                f"[harbor] resume: backfilled {plan['backfilled']} row(s) into "
                f"{args.log_jsonl}",
                flush=True,
            )

    if not run_ids:
        print("No tasks to run.", file=sys.stderr)
        return 0

    if experiment_dir is not None and args.isolate_hermes_home and not args.oracle:
        if not args.dry_run:
            manifest = prepare_harbor_isolate(
                experiment_dir=experiment_dir,
                dataset_path=dataset_path,
                reset_skills=bool(args.reset_task_workspaces),
            )
            isolate_home = Path(str(manifest["hermes_home"]))
            envelope_kwargs["hermes_home"] = isolate_home
            seed = manifest.get("skills_seed") or {}
            print(
                f"[harbor] HERMES_HOME → {isolate_home} "
                f"(skills {seed.get('action')}, n={seed.get('n_skills')})",
                flush=True,
            )
        else:
            print(f"[harbor] dry-run isolate HERMES_HOME → {isolate_home}", flush=True)
        if args.skills_overlay and isolate_home is not None:
            overlay_src = args.skills_overlay.expanduser().resolve()
            if not args.dry_run:
                n_ov = overlay_skill_packages(overlay_src, isolate_home / "skills")
                print(
                    f"[harbor] skills overlay {overlay_src} → {isolate_home / 'skills'} "
                    f"(n={n_ov})",
                    flush=True,
                )
            else:
                print(f"[harbor] dry-run skills overlay {overlay_src}", flush=True)

    agent = "oracle" if args.oracle else _HERMES_AGENT
    model = args.model
    job_name = args.job_name or datetime.now(timezone.utc).strftime("tb-harbor-%Y%m%dT%H%M%SZ")
    if args.resume:
        allocated = allocate_job_name(jobs_dir, job_name)
        if allocated != job_name:
            print(
                f"[harbor] resume: job dir {jobs_dir / job_name} exists; "
                f"using {allocated}",
                flush=True,
            )
            job_name = allocated
    jobs_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_harbor_command(
        harbor_bin=_which_harbor(),
        dataset_path=dataset_path,
        agent=agent,
        model=model,
        harbor_env=args.harbor_env,
        jobs_dir=jobs_dir,
        job_name=job_name,
        task_ids=run_ids,
        n_concurrent=args.n_concurrent,
        max_iterations=None if args.oracle else args.max_iterations,
        extra=args.harbor_args or None,
    )
    if args.dry_run:
        print(" ".join(cmd))
        return 0

    if not _harbor_importable() and shutil.which("harbor") is None:
        print(
            "Harbor is not installed. Install with: pip install -e '.[harbor]'\n"
            "Also require Docker (or pass --env modal / --env daytona).",
            file=sys.stderr,
        )
        return 1

    env = harbor_child_env(
        base=os.environ,
        repo=_REPO,
        max_iterations=args.max_iterations,
        hot_pool=None if args.oracle else args.hot_pool,
        hot_pool_persist=None if args.oracle else hot_pool_persist,
        hermes_home=None if args.oracle else isolate_home,
        instruction_prefix=None if args.oracle else args.instruction_prefix,
    )

    result = execute_harbor_run(
        cmd=cmd,
        env=env,
        jobs_dir=jobs_dir,
        job_name=job_name,
        envelope_kwargs=envelope_kwargs,
        log_jsonl=args.log_jsonl,
        print_summary=args.print_summary,
    )
    return int(result["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
