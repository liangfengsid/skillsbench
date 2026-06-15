#!/usr/bin/env python3
"""
Run Hermes AIAgent against SkillsBench tasks (library use of this repo's run_agent).

Requires a Python environment where Hermes dependencies are installed (typically
the Hermes repo venv). This script prepends HERMES_AGENT_ROOT to sys.path so
``from run_agent import AIAgent`` resolves.

Examples (from Hermes repo root):

  python3 benchmark/scripts/run_skillsbench_with_hermes.py --task adaptive-cruise-control \\
      --skill-nudge-interval 5 --memory-nudge-interval 5 \\
      --log-jsonl ./hermes_skillsbench_runs.jsonl

  python3 benchmark/scripts/run_skillsbench_with_hermes.py --all --log-jsonl ./runs.jsonl

  python3 benchmark/scripts/run_skillsbench_with_hermes.py --all --start-task-index 10 \\
      --log-jsonl ./runs.jsonl  # skip first 10 tasks (sorted order), run the rest

Environment / API keys follow Hermes (e.g. OPENROUTER_API_KEY); see Hermes docs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# This file lives at <repo>/benchmark/scripts/<name>.py
_SCRIPT = Path(__file__).resolve()


def _benchmark_dir() -> Path:
    """``<repo>/benchmark`` — sibling of ``skillsbench/`` and ``scripts/``."""
    return _SCRIPT.parents[1]


def _bundled_skillsbench_root() -> Path:
    """Canonical SkillsBench tree shipped in this repo: ``benchmark/skillsbench``."""
    return _benchmark_dir() / "skillsbench"


def _hermes_repo_root_from_script() -> Path:
    """``<repo>`` (parent of ``benchmark/``)."""
    return _SCRIPT.parents[2]

_DEFAULT_HERMES = _hermes_repo_root_from_script()
_DEFAULT_SKILLSBENCH = _bundled_skillsbench_root()
DEFAULT_HERMES_ROOT = _DEFAULT_HERMES
DEFAULT_PROMPT_TASKS_BASE = str((_DEFAULT_SKILLSBENCH / "tasks").resolve())


def _expand(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _ensure_hermes_on_path(hermes_root: Path) -> None:
    root = hermes_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"HERMES_AGENT_ROOT is not a directory: {root}")
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)


def discover_task_ids(tasks_dir: Path) -> List[str]:
    if not tasks_dir.is_dir():
        raise SystemExit(f"Tasks directory not found: {tasks_dir}")
    ids: List[str] = []
    for child in sorted(tasks_dir.iterdir()):
        if child.is_dir() and (child / "instruction.md").is_file():
            ids.append(child.name)
    return ids


def build_user_message(prompt_tasks_base: str, task_id: str) -> str:
    base = prompt_tasks_base.rstrip("/")
    return (
        f"complete task in {base}/{task_id}, following instruction.md and verify the result."
    )


def json_safe(obj: Any) -> Any:
    """Round-trip through JSON with string fallback for non-serializable values."""

    def _default(o: Any) -> Any:
        return str(o)

    return json.loads(json.dumps(obj, default=_default))


def run_one_task(
    *,
    task_id: str,
    hermes_root: Path,
    skillsbench_root: Path,
    prompt_tasks_base: str,
    model: str,
    skill_nudge_interval: Optional[int],
    memory_nudge_interval: Optional[int],
    max_iterations: int,
    quiet_mode: bool,
    skip_context_files: bool,
    skip_memory: bool,
    save_trajectories: bool,
) -> Dict[str, Any]:
    _ensure_hermes_on_path(hermes_root)
    from run_agent import AIAgent  # type: ignore  # after sys.path

    agent = AIAgent(
        model=model,
        quiet_mode=quiet_mode,
        max_iterations=max_iterations,
        skip_context_files=skip_context_files,
        skip_memory=skip_memory,
        save_trajectories=save_trajectories,
        platform="skillsbench-batch",
    )
    if skill_nudge_interval is not None:
        agent._skill_nudge_interval = int(skill_nudge_interval)
    if memory_nudge_interval is not None:
        agent._memory_nudge_interval = int(memory_nudge_interval)

    user_message = build_user_message(prompt_tasks_base, task_id)
    hermes_task_id = f"skillsbench-{task_id}"

    t0 = time.perf_counter()
    result = agent.run_conversation(
        user_message=user_message,
        task_id=hermes_task_id,
    )
    elapsed = time.perf_counter() - t0

    envelope: Dict[str, Any] = {
        "schema": "skillsbench.hermes_run.v1",
        "ts_start_iso": None,  # filled by caller
        "ts_end_iso": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(elapsed, 6),
        "skillsbench_task_id": task_id,
        "skillsbench_root": str(skillsbench_root),
        "hermes_root": str(hermes_root),
        "hermes_conversation_task_id": hermes_task_id,
        "user_message": user_message,
        "nudge_intervals": {
            "skill": getattr(agent, "_skill_nudge_interval", None),
            "memory": getattr(agent, "_memory_nudge_interval", None),
        },
        "model": model,
        "run_conversation_result": result,
    }
    return envelope


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Hermes AIAgent on one or all SkillsBench tasks.",
    )
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "--task",
        metavar="TASK_ID",
        help="Single task directory name under tasks/ (must contain instruction.md).",
    )
    g.add_argument(
        "--all",
        action="store_true",
        help="Run every task under tasks/ that has instruction.md, in sorted order.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print discovered task ids and exit (no Hermes import).",
    )
    parser.add_argument(
        "--skillsbench-root",
        type=str,
        default=str(_bundled_skillsbench_root()),
        help=(
            "SkillsBench checkout root (default: this repo's benchmark/skillsbench, "
            "sibling of benchmark/scripts)."
        ),
    )
    parser.add_argument(
        "--hermes-root",
        type=str,
        default=str(DEFAULT_HERMES_ROOT),
        help=f"Hermes agent repo root (default: {DEFAULT_HERMES_ROOT}).",
    )
    parser.add_argument(
        "--prompt-tasks-base",
        type=str,
        default=DEFAULT_PROMPT_TASKS_BASE,
        help=(
            "Path prefix embedded in the user prompt before /<task_id> "
            "(default: resolved benchmark/skillsbench/tasks in this checkout)."
        ),
    )
    parser.add_argument(
        "--skill-nudge-interval",
        type=int,
        default=None,
        metavar="N",
        help="Set agent._skill_nudge_interval after construction (omit to leave Hermes config).",
    )
    parser.add_argument(
        "--memory-nudge-interval",
        type=int,
        default=None,
        metavar="N",
        help="Set agent._memory_nudge_interval after construction (omit to leave Hermes config).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model id (Hermes/OpenRouter format); empty uses Hermes default from config.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=90,
        help="Max tool-calling iterations per conversation (Hermes default 90).",
    )
    parser.add_argument(
        "--no-quiet",
        action="store_true",
        help="Disable quiet_mode on AIAgent (more console output from Hermes).",
    )
    parser.add_argument(
        "--skip-context-files",
        action="store_true",
        help="Skip AGENTS.md-style context injection (Hermes).",
    )
    parser.add_argument(
        "--skip-memory",
        action="store_true",
        help="Disable persistent memory (Hermes).",
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Append Hermes trajectories to trajectory_samples.jsonl.",
    )
    parser.add_argument(
        "--log-jsonl",
        type=str,
        default=None,
        help="Append one JSON line per task run (full envelope + run_conversation result).",
    )
    parser.add_argument(
        "--log-json-pretty",
        type=str,
        default=None,
        help="Also write a pretty-printed JSON file for the last run only (single-task mode).",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a short stdout summary per task (tokens, cost, duration).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="With --all, exit on first exception instead of continuing.",
    )
    parser.add_argument(
        "--start-task-index",
        type=int,
        default=0,
        metavar="X",
        help=(
            "With --all only: skip the first X tasks (sorted task order, 0-based) "
            "and run the remainder. Default 0 runs from the start. "
            "Combine with --end-task-index for a half-open range [start, end)."
        ),
    )
    parser.add_argument(
        "--end-task-index",
        type=int,
        default=None,
        metavar="Y",
        help=(
            "With --all only: exclusive end index in the same sorted order as "
            "--list-tasks (Python slice: task_ids[start:end]). Omit to run through "
            "the last task. Example: --start-task-index 0 --end-task-index 10 runs "
            "the first 10 tasks."
        ),
    )

    args = parser.parse_args()
    skillsbench_root = _expand(args.skillsbench_root)
    hermes_root = _expand(args.hermes_root)
    tasks_dir = skillsbench_root / "tasks"

    task_ids = discover_task_ids(tasks_dir)
    if not args.list_tasks and not args.all and not args.task:
        parser.error("Specify --task TASK_ID, --all, or --list-tasks.")
    if args.list_tasks:
        for tid in task_ids:
            print(tid)
        return 0

    if args.start_task_index < 0:
        parser.error("--start-task-index must be >= 0.")
    if args.start_task_index != 0 and not args.all:
        parser.error("--start-task-index is only supported with --all.")
    if args.end_task_index is not None:
        if not args.all:
            parser.error("--end-task-index is only supported with --all.")
        if args.end_task_index < 0:
            parser.error("--end-task-index must be >= 0.")
        if args.end_task_index > len(task_ids):
            parser.error(
                f"--end-task-index ({args.end_task_index}) exceeds task count ({len(task_ids)})."
            )
        if args.end_task_index < args.start_task_index:
            parser.error(
                f"--end-task-index ({args.end_task_index}) must be >= --start-task-index ({args.start_task_index})."
            )

    if args.all:
        if not task_ids:
            print("No tasks found (instruction.md under tasks/).", file=sys.stderr)
            return 1
        start = args.start_task_index
        end = args.end_task_index
        run_ids = task_ids[start:] if end is None else task_ids[start:end]
        if not run_ids:
            print(
                f"No tasks in range [--start-task-index={start}, --end-task-index="
                f"{end if end is not None else 'end'}); discovered {len(task_ids)} task(s).",
                file=sys.stderr,
            )
            return 1
    else:
        assert args.task
        if args.task not in task_ids:
            print(
                f"Unknown task {args.task!r}. Known: {', '.join(task_ids[:10])}"
                + (" ..." if len(task_ids) > 10 else ""),
                file=sys.stderr,
            )
            return 1
        run_ids = [args.task]

    log_path = Path(args.log_jsonl).expanduser() if args.log_jsonl else None
    pretty_path = Path(args.log_json_pretty).expanduser() if args.log_json_pretty else None
    if pretty_path and len(run_ids) != 1:
        print("--log-json-pretty requires exactly one task (no --all).", file=sys.stderr)
        return 1

    any_failed = False

    for tid in run_ids:
        ts_start = datetime.now(timezone.utc).isoformat()
        if args.print_summary:
            print(f"\n=== SkillsBench task: {tid} (start {ts_start}) ===", flush=True)
        try:
            envelope = run_one_task(
                task_id=tid,
                hermes_root=hermes_root,
                skillsbench_root=skillsbench_root,
                prompt_tasks_base=args.prompt_tasks_base,
                model=args.model,
                skill_nudge_interval=args.skill_nudge_interval,
                memory_nudge_interval=args.memory_nudge_interval,
                max_iterations=args.max_iterations,
                quiet_mode=not args.no_quiet,
                skip_context_files=args.skip_context_files,
                skip_memory=args.skip_memory,
                save_trajectories=args.save_trajectories,
            )
            envelope["ts_start_iso"] = ts_start

            serializable = json_safe(envelope)
            if log_path:
                append_jsonl(log_path, serializable)

            if pretty_path:
                pretty_path.write_text(
                    json.dumps(serializable, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            res = envelope["run_conversation_result"]
            if args.print_summary:
                print(
                    f"duration_sec={envelope['duration_sec']!r} "
                    f"api_calls={res.get('api_calls')!r} "
                    f"input_tokens={res.get('input_tokens')!r} "
                    f"output_tokens={res.get('output_tokens')!r} "
                    f"total_tokens={res.get('total_tokens')!r} "
                    f"estimated_cost_usd={res.get('estimated_cost_usd')!r} "
                    f"completed={res.get('completed')!r} interrupted={res.get('interrupted')!r}",
                    flush=True,
                )
        except Exception as e:
            any_failed = True
            err = {
                "schema": "skillsbench.hermes_run_error.v1",
                "ts_start_iso": ts_start,
                "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                "skillsbench_task_id": tid,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }
            if log_path:
                append_jsonl(log_path, err)
            print(f"ERROR task={tid}: {e}", file=sys.stderr)
            traceback.print_exc()
            if args.stop_on_error:
                return 1

    if any_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
