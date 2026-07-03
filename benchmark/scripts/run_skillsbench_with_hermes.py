#!/usr/bin/env python3
"""
Run Hermes AIAgent against SkillsBench tasks (library use of this repo's run_agent).

Requires a Python environment where Hermes dependencies are installed (typically
the Hermes repo venv). This script prepends HERMES_AGENT_ROOT to sys.path so
``from run_agent import AIAgent`` resolves.

Examples (from Hermes repo root):

  python3 benchmark/scripts/run_skillsbench_with_hermes.py --task adaptive-cruise-control \\
      --skill-nudge-interval 10 --memory-nudge-interval 10 \\
      --hot-pool --hot-pool-persist benchmark/skillsbench_hot_pool.json \\
      --log-jsonl benchmark/hermes_skillsbench_runs.jsonl --print-summary

  python3 benchmark/scripts/run_skillsbench_with_hermes.py --task adaptive-cruise-control \\
      --no-hot-pool --log-jsonl benchmark/hermes_skillsbench_runs.jsonl --print-summary

  python3 benchmark/scripts/run_skillsbench_with_hermes.py --all --log-jsonl ./runs.jsonl \\
      --hot-pool --hot-pool-persist benchmark/skillsbench_hot_pool.json

  python3 benchmark/scripts/run_skillsbench_with_hermes.py --all --no-hot-pool \\
      --log-jsonl ./runs.jsonl

  python3 benchmark/scripts/run_skillsbench_with_hermes.py --all --start-task-index 10 \\
      --log-jsonl ./runs.jsonl  # skip first 10 tasks (sorted order), run the rest

Environment / API keys follow Hermes (e.g. OPENROUTER_API_KEY); see Hermes docs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Feishu optional deps (lark_oapi) emit setuptools pkg_resources noise on import.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*pkg_resources is deprecated.*",
)

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

SKILLSBENCH_BATCH_SKILL_REVIEW_APPENDIX = (
    "\n\n**SkillsBench batch context (platform=skillsbench-batch):**\n"
    "The conversation above was a benchmark task run, not a casual user chat.\n\n"
    "CREATE or PATCH a Hermes skill when ANY of these occurred:\n"
    "- 3+ tool rounds spent on verification (pytest, docker, path symlinks, "
    "re-reading outputs, or re-running an already-correct answer)\n"
    "- Host vs container path confusion (/root/... vs "
    "benchmark/skillsbench/tasks/<id>/environment/...)\n"
    "- Repeated reads of solution/ or tests/ before solving\n"
    "- Trial-and-error after the core answer was already computed\n"
    "- Wrong tool patterns (e.g. importing execute_code from hermes_tools, "
    "host pytest against /root paths when docker is unavailable)\n\n"
    "Name skills at CLASS level (examples: skillsbench-host-verification, "
    "binary-stl-mass-calc, skillsbench-path-mapping). Do NOT name skills after "
    "a single task id unless the pitfall is truly unique.\n"
    "Survey skills_list first; patch an existing class skill when possible.\n"
    "No user confirmation is required in batch mode — use skill_manage when "
    "criteria match.\n"
    "If verification thrashing or path confusion occurred, saving a workflow "
    "skill is preferred over 'Nothing to save.'"
)

SKILLSBENCH_BATCH_COMBINED_REVIEW_APPENDIX = (
    "\n\n**SkillsBench batch context (platform=skillsbench-batch):**\n"
    "For the skills portion, apply the same batch rules: persist class-level "
    "workflow skills when verification thrashing, host/container path "
    "confusion, or repeated test/solution peeking consumed 3+ tool rounds. "
    "Use skill_manage(create|patch) without user confirmation when criteria "
    "match. Prefer skillsbench-host-verification-style names over task ids."
)


def build_skillsbench_skill_review_prompt(base_prompt: str) -> str:
    return base_prompt + SKILLSBENCH_BATCH_SKILL_REVIEW_APPENDIX


def build_skillsbench_combined_review_prompt(base_prompt: str) -> str:
    return base_prompt + SKILLSBENCH_BATCH_COMBINED_REVIEW_APPENDIX


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


def apply_hot_pool_cli_overrides(
    *,
    hot_pool: Optional[bool],
    hot_pool_persist: Optional[str],
) -> None:
    """Set env vars so AIAgent honors CLI hot-pool on/off before import."""
    for key in ("HERMES_HOT_POOL_ENABLED", "HERMES_HOT_POOL_PERSIST", "HERMES_HOT_POOL_PATH"):
        os.environ.pop(key, None)
    if hot_pool is False:
        os.environ["HERMES_HOT_POOL_ENABLED"] = "0"
        return
    if hot_pool is True:
        os.environ["HERMES_HOT_POOL_ENABLED"] = "1"
    if hot_pool_persist:
        os.environ["HERMES_HOT_POOL_PERSIST"] = "1"
        os.environ["HERMES_HOT_POOL_PATH"] = str(
            Path(hot_pool_persist).expanduser().resolve(),
        )


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
    hot_pool: Optional[bool] = None,
    hot_pool_persist: Optional[str] = None,
    wait_background_review: bool = True,
    background_review_timeout: Optional[float] = 180.0,
    batch_review_prompt: bool = True,
) -> Dict[str, Any]:
    apply_hot_pool_cli_overrides(
        hot_pool=hot_pool,
        hot_pool_persist=hot_pool_persist,
    )
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
    if batch_review_prompt:
        agent.skill_review_prompt_override = build_skillsbench_skill_review_prompt(
            AIAgent._SKILL_REVIEW_PROMPT,
        )
        agent.combined_review_prompt_override = build_skillsbench_combined_review_prompt(
            AIAgent._COMBINED_REVIEW_PROMPT,
        )

    user_message = build_user_message(prompt_tasks_base, task_id)
    hermes_task_id = f"skillsbench-{task_id}"

    t0 = time.perf_counter()
    result = agent.run_conversation(
        user_message=user_message,
        task_id=hermes_task_id,
    )
    background_review: Dict[str, Any] = {"spawned": False, "completed": True, "timeout": False}
    if wait_background_review:
        background_review = agent.wait_for_background_review(
            timeout=background_review_timeout,
        )
        refreshed = agent.export_hot_pool_telemetry()
        if refreshed is not None and isinstance(result, dict):
            result["hot_pool_telemetry"] = refreshed
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
        "batch_review_prompt": batch_review_prompt,
        "hot_pool_enabled": hot_pool,
        "hot_pool_persist": hot_pool_persist,
        "model": model,
        "background_review": background_review,
        "run_conversation_result": result,
    }
    if isinstance(result, dict) and result.get("hot_pool_telemetry"):
        envelope["hot_pool_telemetry"] = result["hot_pool_telemetry"]
    return envelope


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def format_run_summary(envelope: Dict[str, Any]) -> str:
    """One-line stdout summary for --print-summary."""
    res = envelope.get("run_conversation_result") or {}
    br = envelope.get("background_review") or {}
    tel = envelope.get("hot_pool_telemetry") or res.get("hot_pool_telemetry") or {}
    sm = tel.get("skill_manage") if isinstance(tel, dict) else {}
    skill_manage_sync = sm.get("sync") if isinstance(sm, dict) else None
    bg_actions = br.get("actions") or []
    bg_action_count = len(bg_actions)
    bg_action_preview = bg_actions[0] if bg_actions else None
    parts = [
        f"duration_sec={envelope.get('duration_sec')!r}",
        f"api_calls={res.get('api_calls')!r}",
        f"input_tokens={res.get('input_tokens')!r}",
        f"output_tokens={res.get('output_tokens')!r}",
        f"total_tokens={res.get('total_tokens')!r}",
        f"estimated_cost_usd={res.get('estimated_cost_usd')!r}",
        f"completed={res.get('completed')!r}",
        f"interrupted={res.get('interrupted')!r}",
        f"bg_review_spawned={br.get('spawned')!r}",
        f"bg_review_completed={br.get('completed')!r}",
        f"bg_review_timeout={br.get('timeout')!r}",
        f"bg_review_actions={bg_action_count!r}",
        f"bg_review_action_preview={bg_action_preview!r}",
        f"skill_manage_sync={skill_manage_sync!r}",
    ]
    return " ".join(parts)


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
        "--hot-pool-persist",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Persist hot skill key points across tasks to PATH (JSON). "
            "Each task continues the pool like a new user turn. "
            "Requires hot pool to be enabled (default from config, or --hot-pool)."
        ),
    )
    hot_pool_group = parser.add_mutually_exclusive_group()
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
    parser.set_defaults(hot_pool=None)
    parser.add_argument(
        "--no-wait-background-review",
        action="store_true",
        help=(
            "Return immediately after run_conversation without waiting for the "
            "end-of-turn skill/memory review thread (daemon; often killed on exit)."
        ),
    )
    parser.add_argument(
        "--background-review-timeout",
        type=float,
        default=180.0,
        metavar="SEC",
        help=(
            "Max seconds to wait for background review after each task "
            "(default: 180). Ignored with --no-wait-background-review."
        ),
    )
    parser.add_argument(
        "--no-batch-review-prompt",
        action="store_true",
        help=(
            "Use the default Hermes skill/memory review prompts instead of the "
            "SkillsBench batch appendix (host/container pitfalls, verification thrashing)."
        ),
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
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "JSON split from make_skillsbench_splits.py. With --all, run only tasks "
            "in the selected --split-part instead of the full task list."
        ),
    )
    parser.add_argument(
        "--split-part",
        choices=("train", "test", "all"),
        default="train",
        help="Which partition to run when --split-file is set (default: train).",
    )

    args = parser.parse_args()
    skillsbench_root = _expand(args.skillsbench_root)
    hermes_root = _expand(args.hermes_root)
    tasks_dir = skillsbench_root / "tasks"

    task_ids = discover_task_ids(tasks_dir)
    if args.split_file and not args.all:
        parser.error("--split-file requires --all (or use --task for a single id).")
    if args.split_file:
        split_path = Path(args.split_file).expanduser()
        if not split_path.is_file():
            parser.error(f"Split file not found: {split_path}")
        _split_mod_path = _SCRIPT.parent / "make_skillsbench_splits.py"
        import importlib.util

        spec = importlib.util.spec_from_file_location("make_skillsbench_splits", _split_mod_path)
        split_mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(split_mod)
        split_ids = split_mod.resolve_split_task_ids(split_path, args.split_part)
        unknown = [tid for tid in split_ids if tid not in task_ids]
        if unknown:
            preview = ", ".join(unknown[:5]) + (" ..." if len(unknown) > 5 else "")
            parser.error(f"Split file lists unknown task ids: {preview}")
        task_ids = split_ids

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

    if args.hot_pool is False and args.hot_pool_persist:
        parser.error("--hot-pool-persist cannot be used with --no-hot-pool.")

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
                hot_pool=args.hot_pool,
                hot_pool_persist=args.hot_pool_persist,
                wait_background_review=not args.no_wait_background_review,
                background_review_timeout=args.background_review_timeout,
                batch_review_prompt=not args.no_batch_review_prompt,
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
                print(format_run_summary(envelope), flush=True)
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
