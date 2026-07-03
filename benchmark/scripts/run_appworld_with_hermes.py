#!/usr/bin/env python3
"""
Run Hermes AIAgent on AppWorld tasks with SDK-bridged execution.

The driver owns the AppWorld ReAct loop: Hermes generates Python code blocks;
execution goes through ``AppWorld.execute()``; evaluation uses ``evaluate_task()``.
By default Hermes exposes **all available toolsets** (same as the SkillsBench
driver). Use ``--no-tools`` for the legacy code-only ReAct mode.

Requires:
  - Hermes venv (``from run_agent import AIAgent``)
  - AppWorld installed from ``benchmark/appworld`` (``pip install -e .``)

Examples (from Hermes repo root):

  python3 benchmark/scripts/run_appworld_with_hermes.py --list-tasks
  python3 benchmark/scripts/run_appworld_with_hermes.py --list-tasks --dataset dev

  python3 benchmark/scripts/run_appworld_with_hermes.py \\
      --dataset dev --task 50e1ac9_1 --log-jsonl ./appworld_hermes_runs.jsonl

  python3 benchmark/scripts/run_appworld_with_hermes.py \\
      --dataset dev --all --start-task-index 0 --end-task-index 5 \\
      --experiment-name hermes-dev --log-jsonl ./runs.jsonl

  python3 benchmark/scripts/run_appworld_with_hermes.py \\
      --evaluate-only --dataset dev --task 50e1ac9_1 --experiment-name hermes-dev
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Feishu optional deps (lark_oapi) emit setuptools pkg_resources noise on import.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*pkg_resources is deprecated.*",
)

_SCRIPT = Path(__file__).resolve()
_BENCHMARK_DIR = _SCRIPT.parents[1]
_DEFAULT_HERMES = _SCRIPT.parents[2]
_DEFAULT_APPWORLD = _BENCHMARK_DIR / "appworld"
_DEFAULT_PROMPT = (
    _DEFAULT_APPWORLD / "experiments/prompts/react_code_agent/instructions.txt"
)
DEFAULT_EXPERIMENT_NAME = "hermes-agent"

DATASET_NAMES = ("train", "dev", "test_normal", "test_challenge")

_FULL_CODE_REGEX = re.compile(r"```python\n(.*?)```", re.DOTALL)
_PARTIAL_CODE_REGEX = re.compile(r".*```python\n(.*)", re.DOTALL)
_ROLE_SPLIT_REGEX = re.compile(r"(USER|ASSISTANT|SYSTEM):\n", re.IGNORECASE)

_DEFAULT_MAX_HERMES_ITERATIONS_WITH_TOOLS = 90
_DEFAULT_MAX_HERMES_ITERATIONS_NO_TOOLS = 1

_HERMES_BRIDGE_EPHEMERAL = (
    "You are solving an AppWorld benchmark task in a Python REPL that exposes "
    "an ``apis`` object. You have access to the full Hermes tool suite (terminal, "
    "files, web search, skills, memory, etc.) for research and preparation. "
    "When you are ready to interact with AppWorld APIs, your final reply for "
    "this turn must include exactly one ```python ... ``` code block to run in "
    "the REPL. Follow the conversation format established in the prompt."
)

_HERMES_BRIDGE_EPHEMERAL_NO_TOOLS = (
    "You are solving an AppWorld benchmark task in a Python REPL that exposes "
    "an ``apis`` object. Respond with exactly one ```python ... ``` code block "
    "per turn. Do not use tools. Do not browse the filesystem. Follow the "
    "conversation format established in the prompt."
)


def resolve_max_hermes_iterations(
    max_hermes_iterations: Optional[int],
    *,
    no_tools: bool,
) -> int:
    """Pick per-step Hermes iteration budget (tool calls allowed within one AppWorld step)."""
    if max_hermes_iterations is not None:
        return max(1, int(max_hermes_iterations))
    if no_tools:
        return _DEFAULT_MAX_HERMES_ITERATIONS_NO_TOOLS
    return _DEFAULT_MAX_HERMES_ITERATIONS_WITH_TOOLS


def _expand(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _ensure_hermes_on_path(hermes_root: Path) -> None:
    root = hermes_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"HERMES_AGENT_ROOT is not a directory: {root}")
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)


def _restore_stdio_for_appworld() -> None:
    """Reset stdio before AppWorld/IPython init.

    Hermes wraps stdout/stderr with ``_SafeWriter`` during ``AIAgent`` runs.
    IPython's shell setup assigns to ``sys.stdout.write``, which raises on the
    wrapper. Batch drivers must restore real stdio before each task.
    """
    if sys.__stdout__ is not None:
        sys.stdout = sys.__stdout__
    if sys.__stderr__ is not None:
        sys.stderr = sys.__stderr__


def _configure_appworld(appworld_root: Path) -> None:
    root = str(appworld_root.resolve())
    os.environ["APPWORLD_ROOT"] = root
    try:
        from appworld import update_root

        update_root(root)
    except ImportError as exc:
        raise SystemExit(
            "AppWorld is not installed. From benchmark/appworld run: pip install -e ."
        ) from exc


def _load_task_ids(dataset: str) -> List[str]:
    from appworld import load_task_ids

    return list(load_task_ids(dataset))


def discover_task_ids(
    *,
    dataset: str,
    appworld_root: Path,
) -> List[str]:
    _configure_appworld(appworld_root)
    if dataset not in DATASET_NAMES:
        raise SystemExit(
            f"Unknown dataset {dataset!r}. Choose from: {', '.join(DATASET_NAMES)}"
        )
    return _load_task_ids(dataset)


def list_tasks_report(*, appworld_root: Path, dataset: Optional[str]) -> None:
    _configure_appworld(appworld_root)
    splits = [dataset] if dataset else list(DATASET_NAMES)
    grand_total = 0
    for split in splits:
        ids = _load_task_ids(split)
        grand_total += len(ids)
        print(f"[{split}] {len(ids)} tasks")
        for task_id in ids:
            print(f"  {task_id}")
    if not dataset:
        print(f"total: {grand_total} tasks across {len(DATASET_NAMES)} splits")


def text_to_messages(input_str: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    last_start = 0
    for match in _ROLE_SPLIT_REGEX.finditer(input_str):
        last_end = match.span()[0]
        if messages:
            messages[-1]["content"] = input_str[last_start:last_end]
        else:
            if last_end != 0:
                raise ValueError(
                    f"Start of the prompt has no assigned role: {input_str[:last_end]!r}"
                )
        role = match.group(1).lower()
        messages.append({"role": role, "content": None})
        last_start = match.span()[1]
    if not messages:
        raise ValueError("Prompt template produced no messages.")
    messages[-1]["content"] = input_str[last_start:]
    return messages


def extract_python_code(text: str, *, ignore_multiple_calls: bool = True) -> str:
    original_text = text or ""
    output_code = ""
    match_end = 0
    for re_match in _FULL_CODE_REGEX.finditer(original_text):
        code = re_match.group(1).strip()
        if ignore_multiple_calls:
            return code
        output_code += code + "\n"
        match_end = re_match.end()
    partial_match = _PARTIAL_CODE_REGEX.match(original_text[match_end:])
    if partial_match:
        output_code += partial_match.group(1).strip()
    return output_code.strip()


def render_initial_prompt(
    *,
    prompt_file: Path,
    instruction: str,
    supervisor: Any,
) -> List[Dict[str, Any]]:
    try:
        from jinja2 import Template
    except ImportError as exc:
        raise SystemExit("jinja2 is required (pip install jinja2).") from exc

    template_text = prompt_file.read_text(encoding="utf-8").lstrip()
    rendered = Template(template_text).render(
        instruction=instruction,
        main_user=supervisor,
    )
    return text_to_messages(rendered)


def split_history_for_turn(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    if not messages:
        raise ValueError("Cannot split an empty message list.")
    if messages[-1]["role"] != "user":
        raise ValueError("Expected the last message to be from the user.")
    history = copy.deepcopy(messages[:-1])
    user_message = messages[-1]["content"] or ""
    return history, user_message


def clear_experiment_output(appworld_root: Path, experiment_name: str) -> Optional[Path]:
    target = appworld_root / "experiments" / "outputs" / experiment_name
    if target.exists():
        shutil.rmtree(target)
        return target
    return None


def clear_task_output(appworld_root: Path, experiment_name: str, task_id: str) -> Optional[Path]:
    target = appworld_root / "experiments" / "outputs" / experiment_name / "tasks" / task_id
    if target.exists():
        shutil.rmtree(target)
        return target
    return None


def task_output_dbs_dir(
    appworld_root: Path, experiment_name: str, task_id: str
) -> Path:
    return (
        appworld_root
        / "experiments"
        / "outputs"
        / experiment_name
        / "tasks"
        / task_id
        / "dbs"
    )


def evaluation_skip_reason(
    *,
    appworld_root: Path,
    experiment_name: str,
    task_id: str,
) -> Optional[str]:
    """Return a short skip reason, or None if saved output can be evaluated."""
    dbs_dir = task_output_dbs_dir(appworld_root, experiment_name, task_id)
    if not dbs_dir.is_dir():
        return "not run (no saved output)"
    state_files = list(dbs_dir.glob("*.db")) + list(dbs_dir.glob("*.jsonl"))
    if not state_files:
        return "not run (no saved state)"
    if not any(p.stat().st_size > 0 for p in state_files):
        return "empty saved state — re-run the agent"
    return None


def validate_task_output_for_evaluation(
    *,
    appworld_root: Path,
    experiment_name: str,
    task_id: str,
) -> None:
    skip = evaluation_skip_reason(
        appworld_root=appworld_root,
        experiment_name=experiment_name,
        task_id=task_id,
    )
    if skip == "not run (no saved output)":
        dbs_dir = task_output_dbs_dir(appworld_root, experiment_name, task_id)
        raise FileNotFoundError(
            f"No saved task output at {dbs_dir}. "
            f"Run the agent first with --experiment-name {experiment_name!r}, "
            "or pass the experiment name that matches your prior run "
            f"(default is {DEFAULT_EXPERIMENT_NAME!r})."
        )
    if skip == "not run (no saved state)":
        dbs_dir = task_output_dbs_dir(appworld_root, experiment_name, task_id)
        raise FileNotFoundError(
            f"Task output directory exists but has no .db or .jsonl state files: {dbs_dir}. "
            "The agent run likely failed before AppWorld saved state."
        )
    if skip == "empty saved state — re-run the agent":
        dbs_dir = task_output_dbs_dir(appworld_root, experiment_name, task_id)
        raise FileNotFoundError(
            f"All AppWorld state files under {dbs_dir} are empty (0 bytes). "
            "This usually means AppWorld failed to initialize — e.g. if Hermes "
            "was imported before AppWorld (IPython stdout conflict). Re-run the task."
        )


def json_safe(obj: Any) -> Any:
    def _default(o: Any) -> Any:
        return str(o)

    return json.loads(json.dumps(obj, default=_default))


def load_run_hermes_stats_by_task(
    jsonl_path: Path,
    *,
    experiment_name: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Index ``appworld.hermes_run.v1`` rows by task id (last row wins)."""
    if not jsonl_path.is_file():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("schema") != "appworld.hermes_run.v1":
            continue
        if experiment_name and row.get("experiment_name") != experiment_name:
            continue
        task_id = row.get("appworld_task_id")
        stats = row.get("hermes_stats")
        if task_id and isinstance(stats, dict):
            out[str(task_id)] = stats
    return out


def _pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.1f}%"


def print_evaluate_batch_summary(
    *,
    dataset: str,
    experiment_name: str,
    tasks_total: int,
    evaluated: int,
    skipped: int,
    errors: int,
    task_successes: int,
    pass_count: int,
    num_tests: int,
    evaluated_task_ids: List[str],
    run_stats_by_task: Dict[str, Dict[str, Any]],
) -> None:
    print(f"\n=== AppWorld evaluate summary ({experiment_name} / {dataset}) ===", flush=True)
    print(
        f"tasks: {tasks_total} total | {evaluated} evaluated | "
        f"{skipped} skipped | {errors} errors",
        flush=True,
    )
    if evaluated:
        print(
            f"task success: {task_successes}/{evaluated} ({_pct(task_successes, evaluated)})",
            flush=True,
        )
        print(
            f"test passes: {pass_count}/{num_tests} ({_pct(pass_count, num_tests)})",
            flush=True,
        )
    else:
        print("task success: n/a (no tasks evaluated)", flush=True)
        print("test passes: n/a (no tasks evaluated)", flush=True)

    token_tasks = [tid for tid in evaluated_task_ids if tid in run_stats_by_task]
    total_tokens = sum(
        int((run_stats_by_task.get(tid) or {}).get("total_tokens") or 0) for tid in token_tasks
    )
    if token_tasks:
        avg_tokens = total_tokens / len(token_tasks)
        coverage = (
            f"{len(token_tasks)}/{evaluated} evaluated tasks"
            if evaluated
            else f"{len(token_tasks)} tasks"
        )
        print(
            f"tokens: {total_tokens:,} total | {avg_tokens:,.0f} avg ({coverage} from run log)",
            flush=True,
        )
    elif evaluated:
        print(
            "tokens: unavailable (pass --run-log-jsonl with the agent-run JSONL)",
            flush=True,
        )


def evaluate_one_task(
    *,
    task_id: str,
    experiment_name: str,
    appworld_root: Path,
) -> Dict[str, Any]:
    _configure_appworld(appworld_root)
    validate_task_output_for_evaluation(
        appworld_root=appworld_root,
        experiment_name=experiment_name,
        task_id=task_id,
    )
    from appworld.evaluator import evaluate_task

    tracker = evaluate_task(
        task_id=task_id,
        experiment_name=experiment_name,
        suppress_errors=True,
        save_report=True,
    )
    report_path = (
        appworld_root
        / "experiments"
        / "outputs"
        / experiment_name
        / "tasks"
        / task_id
        / "evaluation"
        / "report.md"
    )
    return {
        "success": tracker.success,
        "pass_count": tracker.pass_count,
        "fail_count": tracker.fail_count,
        "num_tests": tracker.num_tests,
        "pass_percentage": tracker.pass_percentage,
        "difficulty": tracker.difficulty,
        "report_path": str(report_path) if report_path.is_file() else None,
        "evaluation": tracker.to_dict(stats_only=True),
    }


def _accumulate_hermes_stats(total: Dict[str, Any], step_result: Dict[str, Any]) -> None:
    for key in ("api_calls", "input_tokens", "output_tokens", "total_tokens"):
        total[key] = int(total.get(key, 0) or 0) + int(step_result.get(key, 0) or 0)
    cost = step_result.get("estimated_cost_usd")
    if cost is not None:
        total["estimated_cost_usd"] = float(total.get("estimated_cost_usd", 0.0) or 0.0) + float(
            cost
        )


def run_one_task(
    *,
    task_id: str,
    dataset: str,
    hermes_root: Path,
    appworld_root: Path,
    experiment_name: str,
    prompt_file: Path,
    model: str,
    max_steps: int,
    quiet_mode: bool,
    skip_context_files: bool,
    skip_memory: bool,
    save_trajectories: bool,
    evaluate_after_run: bool,
    no_tools: bool = False,
    max_hermes_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    _configure_appworld(appworld_root)

    from appworld import AppWorld
    from appworld.task import Task

    task_meta = Task.load(task_id=task_id, load_ground_truth=False)
    initial_messages = render_initial_prompt(
        prompt_file=prompt_file,
        instruction=task_meta.instruction,
        supervisor=task_meta.supervisor,
    )
    num_instruction_messages = len(initial_messages)

    steps: List[Dict[str, Any]] = []
    hermes_stats: Dict[str, Any] = {
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    run_error: Optional[str] = None
    task_completed = False
    duration_sec = 0.0

    conversation = copy.deepcopy(initial_messages)
    hermes_task_id = f"appworld-{dataset}-{task_id}"
    hermes_iterations = resolve_max_hermes_iterations(
        max_hermes_iterations,
        no_tools=no_tools,
    )

    t0 = time.perf_counter()
    try:
        _restore_stdio_for_appworld()
        # AppWorld must initialize before importing AIAgent: Hermes wraps stdout
        # with _SafeWriter, which breaks IPython's shell setup inside AppWorld.
        with AppWorld(task_id=task_id, experiment_name=experiment_name) as world:
            _ensure_hermes_on_path(hermes_root)
            from run_agent import AIAgent  # type: ignore

            agent_kwargs: Dict[str, Any] = {
                "quiet_mode": quiet_mode,
                "max_iterations": hermes_iterations,
                "skip_context_files": skip_context_files,
                "skip_memory": skip_memory,
                "save_trajectories": save_trajectories,
                "platform": "appworld-batch",
                "ephemeral_system_prompt": (
                    _HERMES_BRIDGE_EPHEMERAL_NO_TOOLS if no_tools else _HERMES_BRIDGE_EPHEMERAL
                ),
            }
            if no_tools:
                agent_kwargs["enabled_toolsets"] = []
            if model:
                agent_kwargs["model"] = model
            agent = AIAgent(**agent_kwargs)

            for step_number in range(1, max_steps + 1):
                history, user_message = split_history_for_turn(conversation)
                step_result = agent.run_conversation(
                    user_message=user_message,
                    conversation_history=history,
                    task_id=f"{hermes_task_id}-step-{step_number}",
                )
                _accumulate_hermes_stats(hermes_stats, step_result)

                messages = step_result.get("messages") or []
                if not messages or messages[-1].get("role") != "assistant":
                    run_error = "Hermes did not return an assistant message."
                    steps.append(
                        {
                            "step": step_number,
                            "error": run_error,
                            "hermes_result": json_safe(step_result),
                        }
                    )
                    break

                assistant_content = messages[-1].get("content") or ""
                code = extract_python_code(assistant_content)
                step_record: Dict[str, Any] = {
                    "step": step_number,
                    "assistant_preview": assistant_content[:500],
                    "code": code,
                }
                if not code:
                    run_error = "No ```python block found in Hermes response."
                    step_record["error"] = run_error
                    steps.append(step_record)
                    break

                execution_output = world.execute(code)
                step_record["execution_output_preview"] = (execution_output or "")[:2000]
                steps.append(step_record)

                conversation = copy.deepcopy(messages)
                maybe_newline = "\n" if not (execution_output or "").endswith("\n") else ""
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            "Output:\n```\n"
                            + (execution_output or "")
                            + maybe_newline
                            + "```\n\n"
                        ),
                    }
                )

                if world.task_completed():
                    task_completed = True
                    break
            else:
                run_error = f"Reached max_steps ({max_steps}) without completing the task."
    except Exception as exc:
        run_error = repr(exc)
        steps.append({"error": run_error, "traceback": traceback.format_exc()})
    finally:
        _restore_stdio_for_appworld()
    duration_sec = round(time.perf_counter() - t0, 6)

    evaluation: Optional[Dict[str, Any]] = None
    if evaluate_after_run:
        try:
            evaluation = evaluate_one_task(
                task_id=task_id,
                experiment_name=experiment_name,
                appworld_root=appworld_root,
            )
        except Exception as exc:
            evaluation = {"error": repr(exc), "traceback": traceback.format_exc()}

    output_dir = (
        appworld_root / "experiments" / "outputs" / experiment_name / "tasks" / task_id
    )

    return {
        "schema": "appworld.hermes_run.v1",
        "appworld_task_id": task_id,
        "dataset": dataset,
        "experiment_name": experiment_name,
        "appworld_root": str(appworld_root),
        "hermes_root": str(hermes_root),
        "prompt_file": str(prompt_file),
        "instruction": task_meta.instruction,
        "num_instruction_messages": num_instruction_messages,
        "duration_sec": duration_sec,
        "max_steps": max_steps,
        "max_hermes_iterations": hermes_iterations,
        "hermes_tools_enabled": not no_tools,
        "steps_taken": len(steps),
        "task_completed": task_completed,
        "run_error": run_error,
        "steps": steps,
        "hermes_stats": hermes_stats,
        "output_directory": str(output_dir),
        "evaluation": evaluation,
    }


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Hermes on AppWorld tasks with SDK-bridged code execution.",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--task", metavar="TASK_ID", help="Single AppWorld task id.")
    mode.add_argument(
        "--all",
        action="store_true",
        help="Run every task in the selected dataset (sorted order).",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print task ids (optionally filtered by --dataset) and exit.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip the agent run; evaluate existing outputs for --task/--all.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_NAMES,
        default="dev",
        help="AppWorld split to use (default: dev).",
    )
    parser.add_argument(
        "--hermes-root",
        type=str,
        default=str(_DEFAULT_HERMES),
        help=f"Hermes repo root (default: {_DEFAULT_HERMES}).",
    )
    parser.add_argument(
        "--appworld-root",
        type=str,
        default=str(_DEFAULT_APPWORLD),
        help=f"AppWorld root / APPWORLD_ROOT (default: {_DEFAULT_APPWORLD}).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"AppWorld experiment output namespace (default: {DEFAULT_EXPERIMENT_NAME}).",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=str(_DEFAULT_PROMPT),
        help="ReAct prompt template (Jinja2) for the first turn.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Hermes model id; empty uses Hermes config default.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=40,
        help="Max AppWorld execute() steps per task (default: 40).",
    )
    parser.add_argument(
        "--max-hermes-iterations",
        type=int,
        default=None,
        help=(
            "Max Hermes tool-calling iterations per AppWorld step. "
            f"Default: {_DEFAULT_MAX_HERMES_ITERATIONS_WITH_TOOLS} with tools, "
            f"{_DEFAULT_MAX_HERMES_ITERATIONS_NO_TOOLS} with --no-tools."
        ),
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help=(
            "Disable Hermes tools (code-only ReAct mode). Default: expose all "
            "available Hermes toolsets, like the SkillsBench driver."
        ),
    )
    parser.add_argument(
        "--no-quiet",
        action="store_true",
        help="Disable Hermes quiet_mode.",
    )
    parser.add_argument(
        "--skip-context-files",
        action="store_true",
        help="Skip AGENTS.md-style context injection in Hermes.",
    )
    parser.add_argument(
        "--skip-memory",
        action="store_true",
        help="Disable Hermes persistent memory.",
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Append Hermes trajectories to trajectory_samples.jsonl.",
    )
    parser.add_argument(
        "--clear-experiment",
        action="store_true",
        help=(
            "Delete experiments/outputs/<experiment-name>/ before the run. "
            "Default: clear only per-task output directories."
        ),
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip AppWorld evaluation after the agent run.",
    )
    parser.add_argument(
        "--log-jsonl",
        type=str,
        default=None,
        help="Append one JSON record per task.",
    )
    parser.add_argument(
        "--run-log-jsonl",
        type=str,
        default=None,
        help=(
            "With --evaluate-only: agent-run JSONL for token stats in the batch "
            "summary (defaults to --log-jsonl when that file already exists)."
        ),
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a short stdout summary per task.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="With --all, exit on the first task exception.",
    )
    parser.add_argument(
        "--start-task-index",
        type=int,
        default=0,
        metavar="X",
        help="With --all: skip the first X tasks (0-based).",
    )
    parser.add_argument(
        "--end-task-index",
        type=int,
        default=None,
        metavar="Y",
        help="With --all: exclusive end index (Python slice end).",
    )

    args = parser.parse_args()
    hermes_root = _expand(args.hermes_root)
    appworld_root = _expand(args.appworld_root)
    prompt_file = _expand(args.prompt_file)

    if args.list_tasks:
        list_tasks_report(appworld_root=appworld_root, dataset=args.dataset)
        return 0

    if not args.task and not args.all:
        parser.error("Specify --task, --all, or --list-tasks.")

    if args.start_task_index < 0:
        parser.error("--start-task-index must be >= 0.")
    if args.end_task_index is not None and args.end_task_index < 0:
        parser.error("--end-task-index must be >= 0.")
    if args.start_task_index != 0 and not args.all:
        parser.error("--start-task-index requires --all.")
    if args.end_task_index is not None and not args.all:
        parser.error("--end-task-index requires --all.")

    task_ids = discover_task_ids(dataset=args.dataset, appworld_root=appworld_root)
    if args.all:
        start = args.start_task_index
        end = args.end_task_index
        if end is not None and end > len(task_ids):
            parser.error(
                f"--end-task-index ({end}) exceeds dataset size ({len(task_ids)})."
            )
        if end is not None and end < start:
            parser.error(
                f"--end-task-index ({end}) must be >= --start-task-index ({start})."
            )
        run_ids = task_ids[start:] if end is None else task_ids[start:end]
        if not run_ids:
            print("No tasks in the requested index range.", file=sys.stderr)
            return 1
    else:
        assert args.task
        if args.task not in task_ids:
            preview = ", ".join(task_ids[:8]) + (" ..." if len(task_ids) > 8 else "")
            print(
                f"Unknown task {args.task!r} in dataset {args.dataset!r}. "
                f"Examples: {preview}",
                file=sys.stderr,
            )
            return 1
        run_ids = [args.task]

    if args.clear_experiment and not args.evaluate_only:
        removed = clear_experiment_output(appworld_root, args.experiment_name)
        if removed and args.print_summary:
            print(f"Cleared experiment output: {removed}", flush=True)

    log_path = Path(args.log_jsonl).expanduser() if args.log_jsonl else None
    any_failed = False

    eval_batch = {
        "tasks_total": len(run_ids),
        "evaluated": 0,
        "skipped": 0,
        "errors": 0,
        "task_successes": 0,
        "pass_count": 0,
        "num_tests": 0,
        "evaluated_task_ids": [],
    }
    run_stats_by_task: Dict[str, Dict[str, Any]] = {}
    if args.evaluate_only:
        run_log = args.run_log_jsonl or args.log_jsonl
        if run_log:
            run_stats_by_task = load_run_hermes_stats_by_task(
                Path(run_log).expanduser(),
                experiment_name=args.experiment_name,
            )

    for tid in run_ids:
        ts_start = datetime.now(timezone.utc).isoformat()
        if args.print_summary:
            mode_label = "evaluate" if args.evaluate_only else "run"
            print(f"\n=== AppWorld [{mode_label}] {args.dataset}/{tid} ===", flush=True)
        try:
            if args.evaluate_only:
                skip_reason = evaluation_skip_reason(
                    appworld_root=appworld_root,
                    experiment_name=args.experiment_name,
                    task_id=tid,
                )
                if skip_reason:
                    eval_batch["skipped"] += 1
                    if args.print_summary:
                        print(f"skipped: {skip_reason}", flush=True)
                    if log_path:
                        append_jsonl(
                            log_path,
                            {
                                "schema": "appworld.hermes_eval_skip.v1",
                                "ts_start_iso": ts_start,
                                "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                                "appworld_task_id": tid,
                                "dataset": args.dataset,
                                "experiment_name": args.experiment_name,
                                "skip_reason": skip_reason,
                            },
                        )
                    continue

                evaluation = evaluate_one_task(
                    task_id=tid,
                    experiment_name=args.experiment_name,
                    appworld_root=appworld_root,
                )
                run_stats = run_stats_by_task.get(tid)
                envelope: Dict[str, Any] = {
                    "schema": "appworld.hermes_eval.v1",
                    "ts_start_iso": ts_start,
                    "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                    "appworld_task_id": tid,
                    "dataset": args.dataset,
                    "experiment_name": args.experiment_name,
                    "evaluation": evaluation,
                }
                if run_stats:
                    envelope["run_hermes_stats"] = run_stats
                eval_batch["evaluated"] += 1
                eval_batch["evaluated_task_ids"].append(tid)
                if evaluation.get("success"):
                    eval_batch["task_successes"] += 1
                eval_batch["pass_count"] += int(evaluation.get("pass_count") or 0)
                eval_batch["num_tests"] += int(evaluation.get("num_tests") or 0)
            else:
                envelope = run_one_task(
                    task_id=tid,
                    dataset=args.dataset,
                    hermes_root=hermes_root,
                    appworld_root=appworld_root,
                    experiment_name=args.experiment_name,
                    prompt_file=prompt_file,
                    model=args.model,
                    max_steps=args.max_steps,
                    max_hermes_iterations=args.max_hermes_iterations,
                    quiet_mode=not args.no_quiet,
                    skip_context_files=args.skip_context_files,
                    skip_memory=args.skip_memory,
                    save_trajectories=args.save_trajectories,
                    evaluate_after_run=not args.no_evaluate,
                    no_tools=args.no_tools,
                )
                envelope["ts_start_iso"] = ts_start
                envelope["ts_end_iso"] = datetime.now(timezone.utc).isoformat()

            serializable = json_safe(envelope)
            if log_path:
                append_jsonl(log_path, serializable)

            if args.print_summary:
                if args.evaluate_only:
                    ev = envelope["evaluation"]
                    print(
                        f"success={ev.get('success')!r} "
                        f"pass={ev.get('pass_count')!r}/{ev.get('num_tests')!r} "
                        f"report={ev.get('report_path')!r}",
                        flush=True,
                    )
                else:
                    ev = envelope.get("evaluation") or {}
                    print(
                        f"duration_sec={envelope.get('duration_sec')!r} "
                        f"steps={envelope.get('steps_taken')!r} "
                        f"task_completed={envelope.get('task_completed')!r} "
                        f"run_error={envelope.get('run_error')!r} "
                        f"eval_success={ev.get('success')!r} "
                        f"tokens={envelope.get('hermes_stats', {}).get('total_tokens')!r}",
                        flush=True,
                    )
        except Exception as exc:
            any_failed = True
            if args.evaluate_only:
                eval_batch["errors"] += 1
            err_schema = (
                "appworld.hermes_eval_error.v1"
                if args.evaluate_only
                else "appworld.hermes_run_error.v1"
            )
            err = {
                "schema": err_schema,
                "ts_start_iso": ts_start,
                "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                "appworld_task_id": tid,
                "dataset": args.dataset,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            if args.evaluate_only:
                err["experiment_name"] = args.experiment_name
            if log_path:
                append_jsonl(log_path, err)
            print(f"ERROR task={tid}: {exc}", file=sys.stderr)
            if not args.evaluate_only:
                traceback.print_exc()
            if args.stop_on_error:
                return 1

    if args.evaluate_only:
        print_evaluate_batch_summary(
            dataset=args.dataset,
            experiment_name=args.experiment_name,
            run_stats_by_task=run_stats_by_task,
            **eval_batch,
        )

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
