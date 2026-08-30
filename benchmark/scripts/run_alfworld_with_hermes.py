#!/usr/bin/env python3
"""
Run Hermes AIAgent on ALFWorld TextWorld games.

ALFWorld is an interactive household simulator, not a file workspace. This
driver owns the env loop (like AppWorld): each turn Hermes replies with one
admissible text command; TextWorld steps; eval is ``won`` / score.

TextWorld-only by default — PDDL ``game.tw-pddl`` files are enough. The
MaskRCNN detector and AI2-THOR are **not** used (those are for embodied
``AlfredThorEnv``). Pretrained seq2seq/DAgger checkpoints are also unused;
Hermes is the policy.

Requires:
  - Hermes venv (``from run_agent import AIAgent``)
  - ``pip install -e ".[alfworld]"`` and the vendored tree on PYTHONPATH
    (this script adds ``benchmark/alfworld`` automatically)
  - Game files via ``ALFWORLD_DATA`` (this machine: ``/data/liangfeng/alfwordData``)

Examples (from Hermes repo root):

  python3 benchmark/scripts/run_alfworld_with_hermes.py --list-tasks --split valid_unseen

  python3 benchmark/scripts/run_alfworld_with_hermes.py \\
      --split valid_unseen --limit 1 --model Qwen/Qwen3.6-27B \\
      --log-jsonl benchmark/runs/alfworld_smoke/runs.jsonl --print-summary

  python3 benchmark/scripts/run_alfworld_with_hermes.py --all --split valid_unseen \\
      --model Qwen/Qwen3.6-27B --max-steps 50 \\
      --experiment-dir benchmark/runs/alfworld_unseen \\
      --log-jsonl benchmark/runs/alfworld_unseen/runs.jsonl --resume --print-summary

  # A-Mem baseline (skill tools on, hot-skill pool off):
  python3 benchmark/scripts/run_alfworld_with_hermes.py --all --split valid_unseen \\
      --model Qwen/Qwen3.6-27B --amem --tools \\
      --experiment-dir benchmark/runs/alfworld_amem_unseen \\
      --isolate-hermes-home --resume --print-summary

  # Dynamic Cheatsheet baseline (skill tools on, hot-skill pool off):
  python3 benchmark/scripts/run_alfworld_with_hermes.py --all --split valid_unseen \\
      --model Qwen/Qwen3.6-27B --dc --tools \\
      --experiment-dir benchmark/runs/alfworld_dc_unseen \\
      --isolate-hermes-home --resume --print-summary
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
from typing import Any, Dict, List, Optional, Sequence, Set

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*pkg_resources is deprecated.*",
)

_SCRIPT = Path(__file__).resolve()
if str(_SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT.parent))
_HERMES_ROOT = _SCRIPT.parents[2]
if str(_HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERMES_ROOT))

from aggregate_skillsbench_runs import aggregate_records, format_summary_text  # noqa: E402
from amem_baseline import (  # noqa: E402
    add_amem_cli_flags,
    amem_telemetry_from_agent,
    apply_amem_argparse_policy,
    apply_amem_cli_overrides,
    resolve_amem_persist,
)
from dc_baseline import (  # noqa: E402
    add_dc_cli_flags,
    apply_dc_argparse_policy,
    apply_dc_cli_overrides,
    dc_telemetry_from_agent,
    resolve_dc_persist,
)
from hermes_hot_pool_outcome import apply_benchmark_hot_pool_outcome_feedback  # noqa: E402
from run_skillsbench_with_hermes import (  # noqa: E402
    apply_hot_pool_cli_overrides,
    resolve_agent_runtime,
    resolve_model_id,
)
from skillsbench_experiment_workspace import (  # noqa: E402
    experiment_hermes_home,
    experiment_hot_pool_path,
    link_shared_hermes_files,
    refresh_skills_dir_caches,
    resolve_source_hermes_home,
    seed_hermes_skills,
)
from skillsbench_metrics import build_envelope_metrics  # noqa: E402

from benchmark.alfworld_adapter.actions import parse_alfworld_action  # noqa: E402
from benchmark.alfworld_adapter.env import (  # noqa: E402
    SPLIT_ALIASES,
    TASK_TYPES,
    apply_alfworld_data_env,
    discover_games,
    ensure_alfworld_on_path,
    format_turn_prompt,
    lookup_game,
    make_text_env,
    resolve_alfworld_data,
)

RUN_SCHEMA = "alfworld.hermes_run.v1"
ERROR_SCHEMA = "alfworld.hermes_run_error.v1"
BATCH_SCHEMA = "alfworld.batch_timing.v1"

_EPHEMERAL_NO_TOOLS = (
    "You are solving an ALFWorld household task in a text simulator. "
    "Each assistant reply must be exactly one admissible command from the "
    "list in the latest user message. Do not use tools. Do not write code."
)
_EPHEMERAL_WITH_TOOLS = (
    "You are solving an ALFWorld household task in a text simulator. "
    "You may use Hermes tools to think, but the final assistant reply for "
    "each environment turn must be exactly one admissible command from the "
    "latest user message."
)


def _ensure_hermes_on_path(hermes_root: Path) -> None:
    root = hermes_root.resolve()
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _finished_task_ids(log_path: Path, *, successes_only: bool = False) -> Set[str]:
    done: Set[str] = set()
    if not log_path.is_file():
        return done
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_id = row.get("task_id") or row.get("skillsbench_task_id")
        if not task_id:
            continue
        if successes_only:
            ev = row.get("evaluation") or {}
            if not ev.get("task_success"):
                continue
        if row.get("schema") in {RUN_SCHEMA, ERROR_SCHEMA}:
            done.add(str(task_id))
    return done


def _accumulate_hermes_stats(acc: Dict[str, Any], result: Optional[Dict[str, Any]]) -> None:
    if not isinstance(result, dict):
        return
    acc["api_calls"] = int(acc.get("api_calls") or 0) + int(result.get("api_calls") or 0)
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_tokens",
        "reasoning_tokens",
    ):
        acc[key] = int(acc.get(key) or 0) + int(result.get(key) or 0)
    cost = result.get("estimated_cost_usd")
    if cost is not None:
        acc["estimated_cost_usd"] = float(acc.get("estimated_cost_usd") or 0) + float(cost)
    if result.get("completed"):
        acc["completed"] = True
    if result.get("interrupted"):
        acc["interrupted"] = True
    if result.get("failed"):
        acc["failed"] = True


def resolve_max_hermes_iterations(
    max_hermes_iterations: Optional[int],
    *,
    no_tools: bool,
) -> int:
    if max_hermes_iterations is not None:
        return max(1, int(max_hermes_iterations))
    return 1 if no_tools else 8


def run_one_game(
    *,
    game: Dict[str, Any],
    hermes_root: Path,
    model: str,
    max_steps: int,
    max_hermes_iterations: int,
    no_tools: bool,
    quiet_mode: bool,
    skip_context_files: bool,
    skip_memory: bool,
    save_trajectories: bool,
    log_steps: bool,
    hot_pool: Optional[bool],
    hot_pool_persist: Optional[str],
    amem: bool = False,
    amem_persist: Optional[str] = None,
    amem_k: int = 5,
    dc: bool = False,
    dc_persist: Optional[str] = None,
    dc_mode: str = "cu",
    dc_k: int = 3,
    runtime: Optional[Dict[str, Any]] = None,
    config_hermes_home: Optional[Path] = None,
) -> Dict[str, Any]:
    apply_hot_pool_cli_overrides(hot_pool=hot_pool, hot_pool_persist=hot_pool_persist)
    _ensure_hermes_on_path(hermes_root)
    ensure_alfworld_on_path()
    from run_agent import AIAgent  # type: ignore

    resolved_model = resolve_model_id(model)
    if not resolved_model:
        raise SystemExit(
            "No model id resolved. Pass --model <id>, or set model.default via "
            "`hermes model` / ~/.hermes/config.yaml."
        )
    agent_runtime = runtime or resolve_agent_runtime(
        model=resolved_model,
        config_hermes_home=config_hermes_home,
    )
    apply_amem_cli_overrides(
        enabled=bool(amem),
        persist_dir=amem_persist,
        k=amem_k,
        llm_model=resolved_model,
        api_key=agent_runtime.get("api_key"),
        base_url=agent_runtime.get("base_url"),
    )
    apply_dc_cli_overrides(
        enabled=bool(dc),
        persist_dir=dc_persist,
        mode=dc_mode,
        k=dc_k,
        llm_model=resolved_model,
        api_key=agent_runtime.get("api_key"),
        base_url=agent_runtime.get("base_url"),
    )
    agent_kwargs: Dict[str, Any] = {
        "model": resolved_model,
        "api_key": agent_runtime.get("api_key"),
        "base_url": agent_runtime.get("base_url"),
        "provider": agent_runtime.get("provider"),
        "api_mode": agent_runtime.get("api_mode"),
        "quiet_mode": quiet_mode,
        "max_iterations": max_hermes_iterations,
        "skip_context_files": skip_context_files,
        "skip_memory": True if amem or dc else skip_memory,
        "save_trajectories": save_trajectories,
        "platform": "alfworld-batch",
        "ephemeral_system_prompt": (
            _EPHEMERAL_NO_TOOLS if no_tools else _EPHEMERAL_WITH_TOOLS
        ),
    }
    if no_tools:
        agent_kwargs["enabled_toolsets"] = []
    agent = AIAgent(**agent_kwargs)
    pool = getattr(agent, "_hot_skill_pool", None)
    if pool is not None and hasattr(pool, "clear_exposed_tips"):
        pool.clear_exposed_tips()

    env = make_text_env(game, max_episode_steps=max_steps)
    steps: List[Dict[str, Any]] = []
    hermes_stats: Dict[str, Any] = {
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cache_read_tokens": 0,
        "reasoning_tokens": 0,
        "estimated_cost_usd": 0.0,
        "completed": False,
        "interrupted": False,
        "failed": False,
    }
    run_error: Optional[str] = None
    conversation: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    try:
        obs = env.reset()
        done = False
        for step_number in range(1, max_steps + 1):
            user_message = format_turn_prompt(
                obs,
                env.admissible,
                first_turn=(step_number == 1),
            )
            result = agent.run_conversation(
                user_message=user_message,
                conversation_history=conversation,
                task_id=f"alfworld-{game['task_id']}-step-{step_number}",
            )
            _accumulate_hermes_stats(hermes_stats, result if isinstance(result, dict) else None)
            messages = (result or {}).get("messages") if isinstance(result, dict) else None
            assistant = ""
            if isinstance(messages, list) and messages:
                conversation = [m for m in messages if isinstance(m, dict)]
                last = conversation[-1] if conversation else {}
                if last.get("role") == "assistant":
                    assistant = str(last.get("content") or "")
            action = parse_alfworld_action(assistant, env.admissible)
            obs, score, done, _info = env.step(action)
            record = {
                "step": step_number,
                "action": action,
                "score": score,
                "won": env.won,
                "done": done,
            }
            if log_steps:
                record["observation"] = obs
                record["admissible_commands"] = list(env.admissible)
                record["assistant_preview"] = assistant[:500]
            steps.append(record)
            if done:
                break
        if not env.won and not done:
            run_error = f"Reached max_steps ({max_steps}) without winning."
        elif done and not env.won:
            run_error = run_error or "Episode ended without winning."
    except Exception as exc:
        run_error = repr(exc)
        steps.append({"error": run_error, "traceback": traceback.format_exc()})
    finally:
        env.close()
        duration_sec = round(time.perf_counter() - t0, 6)

    won = bool(env.won)
    evaluation = {
        "task_success": won,
        "reward": 1.0 if won else 0.0,
        "score": float(env.score),
        "won": won,
        "steps": len(steps),
        "max_steps": max_steps,
        "error": run_error,
    }
    last_assistant = ""
    if conversation:
        last = conversation[-1] if isinstance(conversation[-1], dict) else {}
        if last.get("role") == "assistant":
            last_assistant = str(last.get("content") or "")
    run_conversation_result = {
        "api_calls": hermes_stats.get("api_calls"),
        "input_tokens": hermes_stats.get("input_tokens"),
        "output_tokens": hermes_stats.get("output_tokens"),
        "total_tokens": hermes_stats.get("total_tokens"),
        "cache_read_tokens": hermes_stats.get("cache_read_tokens"),
        "reasoning_tokens": hermes_stats.get("reasoning_tokens"),
        "estimated_cost_usd": hermes_stats.get("estimated_cost_usd"),
        "completed": bool(hermes_stats.get("completed")),
        "interrupted": bool(hermes_stats.get("interrupted")),
        "failed": bool(hermes_stats.get("failed")),
        "messages": conversation,
        "env_actions": [
            s.get("action")
            for s in steps
            if isinstance(s, dict) and s.get("action")
        ],
        "final_response": last_assistant,
    }
    try:
        apply_benchmark_hot_pool_outcome_feedback(
            agent,
            evaluation=evaluation,
            run_result=run_conversation_result,
            duration_sec=duration_sec,
            benchmark="alfworld",
        )
    except Exception:
        pass
    envelope: Dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "benchmark": "alfworld",
        "task_id": game["task_id"],
        "skillsbench_task_id": game["task_id"],
        "task_type": game.get("task_type"),
        "split": game.get("split"),
        "game_file": game.get("game_file"),
        "hermes_root": str(hermes_root),
        "model": resolved_model,
        "duration_sec": duration_sec,
        "max_steps": max_steps,
        "max_hermes_iterations": max_hermes_iterations,
        "hermes_tools_enabled": not no_tools,
        "hot_pool_enabled": False if hot_pool is False else (True if hot_pool else None),
        "hot_pool_persist": hot_pool_persist,
        "amem_enabled": bool(amem),
        "amem_persist": amem_persist,
        "dc_enabled": bool(dc),
        "dc_persist": dc_persist,
        "dc_mode": dc_mode if dc else None,
        "run_error": run_error,
        "evaluation": evaluation,
        "run_conversation_result": run_conversation_result,
        "ts_end_iso": datetime.now(timezone.utc).isoformat(),
    }
    try:
        tel = agent.export_hot_pool_telemetry()
        if tel is not None:
            envelope["hot_pool_telemetry"] = tel
    except Exception:
        pass
    if amem:
        envelope["amem_telemetry"] = amem_telemetry_from_agent(agent)
    if dc:
        envelope["dc_telemetry"] = dc_telemetry_from_agent(agent)
    if log_steps:
        envelope["steps"] = steps
    else:
        envelope["actions"] = [s.get("action") for s in steps if "action" in s]
    envelope["metrics"] = build_envelope_metrics(envelope)
    return envelope


def format_run_summary(envelope: Dict[str, Any]) -> str:
    ev = envelope.get("evaluation") or {}
    res = envelope.get("run_conversation_result") or {}
    ok = "PASS" if ev.get("task_success") else "FAIL"
    return (
        f"{ok} {envelope.get('task_id')} type={envelope.get('task_type')} "
        f"steps={ev.get('steps')}/{ev.get('max_steps')} "
        f"reward={ev.get('reward')} api_calls={res.get('api_calls')}"
    )


def _parse_task_types(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Hermes on ALFWorld TextWorld games (ReAct-style env loop).",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--task", metavar="TASK_ID", help="Single game id (folder/trial).")
    mode.add_argument("--all", action="store_true", help="Run every game in --split.")
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print task ids for --split and exit (no LLM).",
    )
    parser.add_argument(
        "--split",
        default="valid_unseen",
        help="train | valid_seen | valid_unseen (aliases: seen, unseen). Default: valid_unseen.",
    )
    parser.add_argument(
        "--alfworld-data",
        default=None,
        help="ALFWORLD_DATA root (PDDL/game files). Default: $ALFWORLD_DATA or /data/liangfeng/alfwordData.",
    )
    parser.add_argument(
        "--task-types",
        default=None,
        help="Comma-separated ALFWorld type ids 1-6 (default: all six).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of games after filters.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N games (sorted).")
    parser.add_argument(
        "--hermes-root",
        default=str(_HERMES_ROOT),
        help="Hermes repo root (for run_agent import).",
    )
    parser.add_argument("--model", default="", help="Model id (else config default).")
    parser.add_argument("--max-steps", type=int, default=50, help="Env steps per game (paper default 50).")
    parser.add_argument(
        "--max-iterations",
        dest="max_hermes_iterations",
        type=int,
        default=None,
        help="Hermes tool-calling iterations per env step (default: 1 without tools, 8 with --tools).",
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Expose Hermes toolsets (default: text-command only, like a pure ALFWorld agent).",
    )
    hot_pool_group = parser.add_mutually_exclusive_group()
    hot_pool_group.add_argument(
        "--hot-pool",
        dest="hot_pool",
        action="store_const",
        const=True,
        help="Enable hot skill pool (overrides config).",
    )
    hot_pool_group.add_argument(
        "--no-hot-pool",
        dest="hot_pool",
        action="store_const",
        const=False,
        help="Disable hot skill pool (overrides config).",
    )
    parser.set_defaults(hot_pool=None)
    parser.add_argument("--hot-pool-persist", default=None)
    add_amem_cli_flags(parser)
    add_dc_cli_flags(parser)
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Writable run dir (creates hermes_home + default JSONL/hot_pool paths).",
    )
    parser.add_argument(
        "--isolate-hermes-home",
        action="store_true",
        help=(
            "With --experiment-dir: set HERMES_HOME=DIR/hermes_home. Skills are "
            "a writable copy of repo skills/ (not ~/.hermes/skills); "
            "config.yaml / .env / SOUL.md symlink to the real Hermes home."
        ),
    )
    parser.add_argument(
        "--reset-task-workspaces",
        action="store_true",
        help=(
            "With --isolate-hermes-home: re-seed DIR/hermes_home/skills from "
            "the repo skills/ tree (not ~/.hermes/skills)."
        ),
    )
    parser.add_argument("--log-jsonl", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip task ids already in --log-jsonl.")
    parser.add_argument(
        "--resume-successes-only",
        action="store_true",
        help="With --resume, only skip tasks whose last row has task_success=True.",
    )
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument("--log-steps", action="store_true", help="Store per-step obs in JSONL (large).")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-context-files", action="store_true", default=True)
    parser.add_argument("--no-skip-context-files", action="store_false", dest="skip_context_files")
    parser.add_argument("--skip-memory", action="store_true", default=True)
    parser.add_argument("--no-skip-memory", action="store_false", dest="skip_memory")
    parser.add_argument("--save-trajectories", action="store_true")
    args = parser.parse_args(argv)
    apply_amem_argparse_policy(parser, args)
    apply_dc_argparse_policy(parser, args)
    if (args.amem or args.dc) and not args.tools:
        # Fair vs hot-skill: keep Hermes skill tools; A-Mem/DC is the memory path.
        args.tools = True
        label = "amem" if args.amem else "dc"
        print(
            f"[{label}] enabling --tools so skill_view / skill_manage stay available",
            flush=True,
        )

    if args.isolate_hermes_home and not args.experiment_dir:
        parser.error("--isolate-hermes-home requires --experiment-dir")
    if args.reset_task_workspaces and not args.isolate_hermes_home:
        parser.error("--reset-task-workspaces requires --isolate-hermes-home")

    if not args.list_tasks and not args.task and not args.all:
        parser.error("Specify --list-tasks, --task TASK_ID, or --all.")

    split = args.split
    if split not in SPLIT_ALIASES and split not in SPLIT_ALIASES.values():
        parser.error(
            f"Unknown --split {split!r}. Use: {', '.join(sorted(set(SPLIT_ALIASES)))}"
        )

    data_root = resolve_alfworld_data(args.alfworld_data)
    apply_alfworld_data_env(data_root)
    ensure_alfworld_on_path()
    task_types = _parse_task_types(args.task_types)
    games = discover_games(data_root, split, task_types=task_types)

    if args.list_tasks:
        print(f"ALFWORLD_DATA={data_root}")
        print(f"split={SPLIT_ALIASES.get(split, split)} n={len(games)}")
        for g in games:
            print(f"  {g['task_id']}\t{g['task_type']}")
        return 0

    if args.task:
        selected = [lookup_game(games, args.task)]
    else:
        selected = list(games)
    if args.start_index:
        selected = selected[int(args.start_index) :]
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    if not selected:
        print("No ALFWorld games selected.", file=sys.stderr)
        return 1

    hermes_root = Path(args.hermes_root).expanduser().resolve()
    experiment_dir: Optional[Path] = None
    config_hermes_home: Optional[Path] = None
    log_path = Path(args.log_jsonl).expanduser() if args.log_jsonl else None
    hot_pool = args.hot_pool
    hot_pool_persist = args.hot_pool_persist
    if args.experiment_dir:
        experiment_dir = Path(args.experiment_dir).expanduser().resolve()
        experiment_dir.mkdir(parents=True, exist_ok=True)
        if log_path is None:
            log_path = experiment_dir / "runs.jsonl"
        if args.isolate_hermes_home:
            # Capture the real home before rewriting HERMES_HOME.
            source_home = resolve_source_hermes_home()
            home = experiment_hermes_home(experiment_dir)
            seed = seed_hermes_skills(
                home,
                reset_skills=bool(args.reset_task_workspaces),
                hermes_root=hermes_root,
            )
            link_shared_hermes_files(home, source_hermes_home=source_home)
            os.environ["HERMES_HOME"] = str(home)
            refresh_skills_dir_caches(home / "skills")
            print(
                f"[alfworld] HERMES_HOME → {home} "
                f"(skills {seed.get('action')}, n={seed.get('n_skills')} "
                f"from {seed.get('source_skills')})",
                flush=True,
            )
            config_hermes_home = source_home if source_home.is_dir() else None
        if hot_pool and not hot_pool_persist and not args.amem and not args.dc:
            hot_pool_persist = str(experiment_hot_pool_path(experiment_dir))

    amem_persist = resolve_amem_persist(
        enabled=bool(args.amem),
        persist=args.amem_persist,
        experiment_dir=experiment_dir,
    )
    if args.amem:
        print(f"[amem] persist → {amem_persist} k={args.amem_k}", flush=True)
    dc_persist = resolve_dc_persist(
        enabled=bool(args.dc),
        persist=args.dc_persist,
        experiment_dir=experiment_dir,
    )
    if args.dc:
        print(
            f"[dc] persist → {dc_persist} mode={args.dc_mode} k={args.dc_k}",
            flush=True,
        )

    skip_ids: Set[str] = set()
    if args.resume:
        if log_path is None:
            parser.error("--resume requires --log-jsonl or --experiment-dir")
        skip_ids = _finished_task_ids(
            log_path, successes_only=bool(args.resume_successes_only)
        )

    no_tools = not args.tools
    max_hermes_iterations = resolve_max_hermes_iterations(
        args.max_hermes_iterations, no_tools=no_tools
    )
    resolved_model = resolve_model_id(args.model)
    remaining = [g for g in selected if g["task_id"] not in skip_ids]
    runtime = None
    if remaining:
        runtime = resolve_agent_runtime(
            model=resolved_model or args.model,
            config_hermes_home=config_hermes_home,
        )
    ok = 0
    fail = 0
    t_batch = time.perf_counter()
    for i, game in enumerate(selected, start=1):
        task_id = game["task_id"]
        if task_id in skip_ids:
            print(f"[alfworld] [{i}/{len(selected)}] SKIP {task_id}")
            continue
        print(f"[alfworld] [{i}/{len(selected)}] START {task_id}")
        try:
            envelope = run_one_game(
                game=game,
                hermes_root=hermes_root,
                model=args.model,
                max_steps=int(args.max_steps),
                max_hermes_iterations=max_hermes_iterations,
                no_tools=no_tools,
                quiet_mode=bool(args.quiet),
                skip_context_files=bool(args.skip_context_files),
                skip_memory=bool(args.skip_memory),
                save_trajectories=bool(args.save_trajectories),
                log_steps=bool(args.log_steps),
                hot_pool=hot_pool,
                hot_pool_persist=hot_pool_persist,
                amem=bool(args.amem),
                amem_persist=amem_persist,
                amem_k=int(args.amem_k),
                dc=bool(args.dc),
                dc_persist=dc_persist,
                dc_mode=str(args.dc_mode),
                dc_k=int(args.dc_k),
                runtime=runtime,
                config_hermes_home=config_hermes_home,
            )
            if experiment_dir is not None:
                envelope["experiment_dir"] = str(experiment_dir)
                envelope["alfworld_data"] = str(data_root)
            if log_path is not None:
                append_jsonl(log_path, envelope)
            if envelope.get("evaluation", {}).get("task_success"):
                ok += 1
            else:
                fail += 1
            if args.print_summary:
                print("[alfworld]", format_run_summary(envelope))
        except Exception as exc:
            fail += 1
            err = {
                "schema": ERROR_SCHEMA,
                "benchmark": "alfworld",
                "task_id": task_id,
                "skillsbench_task_id": task_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                "evaluation": {"task_success": False, "reward": 0.0, "error": repr(exc)},
            }
            if log_path is not None:
                append_jsonl(log_path, err)
            print(f"[alfworld] [{i}/{len(selected)}] ERROR {task_id}: {exc}", file=sys.stderr)

    elapsed = time.perf_counter() - t_batch
    print(
        f"[alfworld] batch done: ok={ok} fail={fail} skipped_before={len(skip_ids)} "
        f"wall={elapsed:.1f}s split={SPLIT_ALIASES.get(split, split)} data={data_root}"
    )
    if args.print_summary and log_path is not None and log_path.is_file():
        records = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if records:
            summary = aggregate_records(
                records,
                pass_k_values=[1],
                max_user_iterations=args.max_steps,
            )
            print(format_summary_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
