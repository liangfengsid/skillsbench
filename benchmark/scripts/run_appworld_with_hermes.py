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

  # 1) Train — generate hot pool
  python3 benchmark/scripts/run_appworld_with_hermes.py \\
      --dataset train --all --model Qwen/Qwen3.6-27B \\
      --experiment-dir benchmark/runs/appworld_hot_train \\
      --isolate-hermes-home --hot-pool --experiment-name hermes-train \\
      --log-jsonl benchmark/runs/appworld_hot_train/runs.jsonl \\
      --resume --print-summary

  # 2) Unseen test_normal — frozen train pool
  python3 benchmark/scripts/run_appworld_with_hermes.py \\
      --dataset test_normal --all --model Qwen/Qwen3.6-27B \\
      --experiment-dir benchmark/runs/appworld_hot_test \\
      --isolate-hermes-home \\
      --hot-pool --hot-pool-persist benchmark/runs/appworld_hot_train/hot_pool.json \\
      --experiment-name hermes-test \\
      --log-jsonl benchmark/runs/appworld_hot_test/runs.jsonl \\
      --resume --print-summary

  # A-Mem baseline (same Hermes + skill tools, no hot-skill pool):
  python3 benchmark/scripts/run_appworld_with_hermes.py \\
      --dataset train --all --model Qwen/Qwen3.6-27B --amem \\
      --experiment-dir benchmark/runs/appworld_amem_train \\
      --isolate-hermes-home --experiment-name hermes-amem-train \\
      --log-jsonl benchmark/runs/appworld_amem_train/runs.jsonl \\
      --resume --print-summary

  # Dynamic Cheatsheet baseline (same Hermes + skill tools, no hot-skill pool):
  python3 benchmark/scripts/run_appworld_with_hermes.py \\
      --dataset train --all --model Qwen/Qwen3.6-27B --dc \\
      --experiment-dir benchmark/runs/appworld_dc_train \\
      --isolate-hermes-home --experiment-name hermes-dc-train \\
      --log-jsonl benchmark/runs/appworld_dc_train/runs.jsonl \\
      --resume --print-summary
"""

from __future__ import annotations

import argparse
import builtins
import copy
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# AppWorld's SafetyGuard patches host I/O process-wide during ``world.execute()``
# and may leak. Bind the real functions now, *before* AppWorld is imported, and
# reinstall them after each task. ``guard.disable()`` is not enough: a leaked
# guard means the *next* ``SafetyGuard()`` snapshots the disabled functions as
# its "originals".
#
# Keep this map in sync with ``DISALLOWED_MODULE_TO_FUNCTION_NAMES`` in
# ``benchmark/appworld/src/appworld/common/safety_guard.py``.
_SAFETY_GUARD_PATCH_TARGETS: Dict[str, List[str]] = {
    "builtins": ["exit", "quit", "open", "breakpoint"],
    "sys": ["exit"],
    "os": [
        "_exit",
        "open",
        "read",
        "write",
        "close",
        "walk",
        "kill",
        "system",
        "putenv",
        "remove",
        "removedirs",
        "rmdir",
        "fchdir",
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "fchmod",
        "fchown",
        "chmod",
        "chown",
        "chroot",
        "lchflags",
        "lchmod",
        "lchown",
        "chdir",
    ],
    "io": ["open", "open_code", "FileIO"],
    "shutil": [
        "rmtree",
        "move",
        "chown",
        "copy",
        "copy2",
        "copyfile",
        "copytree",
        "make_archive",
        "get_archive_formats",
    ],
    "subprocess": [
        "Popen",
        "call",
        "check_call",
        "check_output",
        "run",
        "getoutput",
        "getstatusoutput",
    ],
    "pathlib.Path": [
        "open",
        "write_text",
        "read_bytes",
        "write_bytes",
        "unlink",
        "rmdir",
        "rename",
        "replace",
        "chmod",
        "lchmod",
        "chown",
        "lchown",
        "touch",
        "symlink_to",
        "link_to",
        "mkdir",
        "expanduser",
    ],
    "fileinput": ["input", "filename", "nextfile", "close", "lineno"],
    "glob": ["glob", "iglob"],
    "json": ["dump", "load"],
    "tempfile": [
        "mktemp",
        "mkdtemp",
        "mkstemp",
        "NamedTemporaryFile",
        "TemporaryDirectory",
    ],
    "zipfile": ["ZipFile"],
    "shelve": ["open"],
    "dbm": ["open", "close"],
    "pickle": ["dump", "load"],
    "codecs": ["open"],
    "bz2": ["open"],
    "gzip": ["open"],
    "tarfile": ["open"],
    "csv": ["reader", "writer", "DictReader", "DictWriter"],
    "time": ["sleep"],
    "pdb": ["set_trace"],
    "urllib.request": [
        "urlretrieve",
        "urlcleanup",
        "urlopen",
        "URLopener",
        "FancyURLopener",
    ],
}

_HOST_ATTRS: List[Tuple[Any, str, Any]] = []
_HOST_ATTR_KEYS: Set[Tuple[int, str]] = set()


def _remember_host_attr(obj: Any, name: str) -> None:
    key = (id(obj), name)
    if key in _HOST_ATTR_KEYS:
        return
    fn = getattr(obj, name, None)
    if fn is not None:
        _HOST_ATTRS.append((obj, name, fn))
        _HOST_ATTR_KEYS.add(key)


def _module_by_path(path: str) -> Any:
    """Resolve ``os`` / ``pathlib.Path`` / ``urllib.request`` the way SafetyGuard does."""
    parts = path.split(".")
    obj: Any = None
    rest: List[str] = []
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
            rest = parts[i:]
            break
        except ImportError:
            continue
    if obj is None:
        raise ImportError(path)
    for part in rest:
        obj = getattr(obj, part)
    return obj


def _snapshot_host_safety_guard_targets() -> None:
    """Capture real host functions before AppWorld can patch them."""
    for module_name, names in _SAFETY_GUARD_PATCH_TARGETS.items():
        try:
            module = _module_by_path(module_name)
        except Exception:
            continue
        for name in names:
            _remember_host_attr(module, name)
    # Not in the denylist, but ``os.environ.pop`` / assignment still need them.
    _remember_host_attr(os, "unsetenv")
    _remember_host_attr(builtins, "SystemExit")
    _remember_host_attr(builtins, "open")


_snapshot_host_safety_guard_targets()
_HOST_OPEN = builtins.open
_HOST_PATH_MKDIR = Path.mkdir

# Feishu optional deps (lark_oapi) emit setuptools pkg_resources noise on import.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*pkg_resources is deprecated.*",
)

_SCRIPT = Path(__file__).resolve()
if str(_SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT.parent))
_BENCHMARK_DIR = _SCRIPT.parents[1]
_DEFAULT_HERMES = _SCRIPT.parents[2]
_DEFAULT_APPWORLD = _BENCHMARK_DIR / "appworld"
_DEFAULT_PROMPT = (
    _DEFAULT_APPWORLD / "experiments/prompts/react_code_agent/instructions.txt"
)
DEFAULT_EXPERIMENT_NAME = "hermes-agent"

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
    host_expand_path,
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

DATASET_NAMES = ("train", "dev", "test_normal", "test_challenge")

_FULL_CODE_REGEX = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_PARTIAL_CODE_REGEX = re.compile(
    r".*```(?:python|py)\s*\n(.*)",
    re.DOTALL | re.IGNORECASE,
)
_GENERIC_FENCE_REGEX = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_APPWORLD_CODE_HINT = re.compile(
    r"\b(?:apis\.|print\s*\(|# interact with apis)",
    re.IGNORECASE,
)
_ROLE_SPLIT_REGEX = re.compile(r"(USER|ASSISTANT|SYSTEM):\n", re.IGNORECASE)

_DEFAULT_MAX_HERMES_ITERATIONS_WITH_TOOLS = 90
_DEFAULT_MAX_HERMES_ITERATIONS_NO_TOOLS = 1
_MAX_PYTHON_FORMAT_RETRIES = 2
_PYTHON_FORMAT_NUDGE = (
    "Your last reply did not include a ```python ... ``` code block. "
    "This AppWorld REPL only executes Python cells in that format—plain text "
    "is not run. Reply again with exactly one ```python ... ``` block "
    "(use print() or apis.* calls even if you believe the task is finished)."
)

_HERMES_BRIDGE_EPHEMERAL = (
    "You are solving an AppWorld benchmark task in a Python REPL that exposes "
    "an ``apis`` object. You have access to the full Hermes tool suite (terminal, "
    "files, web search, skills, memory, etc.) for research and preparation. "
    "When you are ready to interact with AppWorld APIs, your final reply for "
    "this turn must include exactly one ```python ... ``` code block to run in "
    "the REPL—even if you are summarizing progress or believe the task is done. "
    "Plain-text-only replies are invalid. Follow the conversation format "
    "established in the prompt."
)

_HERMES_BRIDGE_EPHEMERAL_NO_TOOLS = (
    "You are solving an AppWorld benchmark task in a Python REPL that exposes "
    "an ``apis`` object. Respond with exactly one ```python ... ``` code block "
    "per turn—even when reporting completion (use print() if needed). "
    "Do not use tools. Do not browse the filesystem. Plain-text-only replies "
    "are invalid. Follow the conversation format established in the prompt."
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
    return host_expand_path(p)


def _ensure_hermes_on_path(hermes_root: Path) -> None:
    root = hermes_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"HERMES_AGENT_ROOT is not a directory: {root}")
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)


def _restore_stdio_for_appworld() -> None:
    """Reset stdio before AppWorld/IPython init and before each ``execute()``.

    Hermes wraps stdout/stderr with ``_SafeWriter`` during ``AIAgent`` runs.
    IPython's shell setup assigns to ``sys.stdout.write``, which raises on the
    wrapper. Background review threads may also leave global stdio pointing at
    closed handles. Batch drivers must restore real stdio before each task and
    each AppWorld code cell.
    """
    if sys.__stdout__ is not None:
        sys.stdout = sys.__stdout__
    if sys.__stderr__ is not None:
        sys.stderr = sys.__stderr__


def _restore_host_os_io() -> None:
    """Reinstall host functions captured before AppWorld SafetyGuard patched them.

    ``guard.disable()`` is not enough: if a guard leaked, the *next* AppWorld
    instance snapshots the disabled functions as its "originals".
    """
    for obj, name, fn in _HOST_ATTRS:
        try:
            setattr(obj, name, fn)
        except Exception:
            pass


def _disable_appworld_safety_guard(world: Any = None) -> None:
    """Undo AppWorld's process-wide I/O patches if they leaked past execute()."""
    guard = getattr(world, "safety_guard", None)
    if guard is not None and hasattr(guard, "disable"):
        try:
            guard.disable()
        except Exception:
            pass
    _restore_host_os_io()
    _restore_stdio_for_appworld()


def _ensure_host_dir(path: Path) -> None:
    """Create *path* even when ``Path.mkdir`` is SafetyGuard-disabled."""
    try:
        _HOST_PATH_MKDIR(path, parents=True, exist_ok=True)
    except PermissionError:
        os.makedirs(path, exist_ok=True)


def _configure_appworld(appworld_root: Path) -> None:
    _restore_host_os_io()
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
    if output_code.strip():
        return output_code.strip()

    # Fallback: a single generic fence that looks like AppWorld REPL code (not prose).
    generic_blocks = [m.group(1).strip() for m in _GENERIC_FENCE_REGEX.finditer(original_text)]
    generic_blocks = [b for b in generic_blocks if b and _APPWORLD_CODE_HINT.search(b)]
    if len(generic_blocks) == 1:
        return generic_blocks[0]
    return ""


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
    _restore_host_os_io()
    try:
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
            "success": bool(tracker.success),
            "task_success": bool(tracker.success),
            "reward": 1.0 if tracker.success else 0.0,
            "pass_count": tracker.pass_count,
            "fail_count": tracker.fail_count,
            "num_tests": tracker.num_tests,
            "tests_passed": tracker.pass_count,
            "tests_total": tracker.num_tests,
            "pass_percentage": tracker.pass_percentage,
            "difficulty": tracker.difficulty,
            "report_path": str(report_path) if report_path.is_file() else None,
            "evaluation": tracker.to_dict(stats_only=True),
        }
    finally:
        _restore_host_os_io()


def _finished_task_ids(log_path: Path, *, successes_only: bool = False) -> Set[str]:
    """Task ids already present in JSONL (last row wins)."""
    if not log_path.is_file():
        return set()
    last: Dict[str, Dict[str, Any]] = {}
    with _HOST_OPEN(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = row.get("appworld_task_id") or row.get("task_id") or row.get("skillsbench_task_id")
            if tid:
                last[str(tid)] = row
    out: Set[str] = set()
    for tid, row in last.items():
        if successes_only:
            ev = row.get("evaluation") or {}
            if not (ev.get("task_success") or ev.get("success")):
                continue
        out.add(tid)
    return out


def _accumulate_hermes_stats(total: Dict[str, Any], step_result: Dict[str, Any]) -> None:
    for key in ("api_calls", "input_tokens", "output_tokens", "total_tokens"):
        total[key] = int(total.get(key, 0) or 0) + int(step_result.get(key, 0) or 0)
    cost = step_result.get("estimated_cost_usd")
    if cost is not None:
        total["estimated_cost_usd"] = float(total.get("estimated_cost_usd", 0.0) or 0.0) + float(
            cost
        )


def _wait_background_review(agent: Any, timeout: float = 60.0) -> None:
    """Join Hermes end-of-turn review before AppWorld SafetyGuard is enabled."""
    if agent is None:
        return
    waiter = getattr(agent, "wait_for_background_review", None)
    if not callable(waiter):
        return
    try:
        waiter(timeout=timeout)
    except Exception:
        pass
    finally:
        _restore_stdio_for_appworld()


def _execute_world_code(world: Any, code: str) -> str:
    """Run AppWorld ``execute()`` and turn Python failures into step output.

    A cell that hits AppWorld's 100s ``SIGALRM`` timeout should not abort the
    batch. ``TimeoutError`` is converted to a string observation. A SIGSEGV
    inside IPython still kills the process — that is why ``--all`` isolates
    each task in a subprocess by default.
    """
    _restore_stdio_for_appworld()
    try:
        return world.execute(code)
    except Exception as exc:
        return f"Execution failed. Traceback:\n{type(exc).__name__}: {exc}"
    finally:
        _disable_appworld_safety_guard(world)
        _restore_stdio_for_appworld()


def _jsonable_task_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in kwargs.items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def _task_kwargs_from_json(data: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(data)
    for key in ("hermes_root", "appworld_root", "prompt_file", "config_hermes_home"):
        if out.get(key):
            out[key] = Path(out[key])
    return out


def _task_worker_crash_envelope(
    *,
    task_id: str,
    dataset: str,
    experiment_name: str,
    returncode: Optional[int],
) -> Dict[str, Any]:
    hint = ""
    if returncode == 139 or returncode == -11:
        hint = (
            " (SIGSEGV — often AppWorld's SIGALRM cell timeout interrupting "
            "IPython/SQLite)"
        )
    return {
        "schema": "appworld.hermes_run_error.v1",
        "benchmark": "appworld",
        "appworld_task_id": task_id,
        "task_id": task_id,
        "skillsbench_task_id": task_id,
        "dataset": dataset,
        "experiment_name": experiment_name,
        "error": (
            f"task worker exited with code {returncode}{hint}. "
            "The batch continues; re-run with --resume to retry this task."
        ),
        "worker_returncode": returncode,
        "run_error": f"worker_exit_{returncode}",
        "task_completed": False,
    }


def run_one_task_in_subprocess(**kwargs: Any) -> Dict[str, Any]:
    """Run one task in a child process so a crash cannot kill ``--all``."""
    _restore_host_os_io()
    work = Path(tempfile.mkdtemp(prefix="appworld-hermes-task-"))
    request_path = work / "request.json"
    result_path = work / "result.json"
    request = {
        "result_path": str(result_path),
        "kwargs": _jsonable_task_kwargs(kwargs),
    }
    with _HOST_OPEN(request_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(request, ensure_ascii=False, default=str))
    proc = subprocess.run(
        [sys.executable, "-u", str(_SCRIPT), "--_worker-request", str(request_path)],
        cwd=str(Path.cwd()),
        env=os.environ.copy(),
        check=False,
    )
    if result_path.is_file():
        with _HOST_OPEN(result_path, encoding="utf-8") as fh:
            try:
                payload = json.loads(fh.read())
            except json.JSONDecodeError:
                payload = None
        if isinstance(payload, dict):
            return payload
    return _task_worker_crash_envelope(
        task_id=str(kwargs.get("task_id") or ""),
        dataset=str(kwargs.get("dataset") or ""),
        experiment_name=str(kwargs.get("experiment_name") or ""),
        returncode=proc.returncode,
    )


def _worker_main(request_path: Path) -> int:
    with _HOST_OPEN(request_path, encoding="utf-8") as fh:
        request = json.loads(fh.read())
    kwargs = _task_kwargs_from_json(request["kwargs"])
    envelope = run_one_task(**kwargs)
    result_path = Path(request["result_path"])
    _ensure_host_dir(result_path.parent)
    with _HOST_OPEN(result_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(json_safe(envelope), ensure_ascii=False, default=str))
    return 0


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
    hot_pool: Optional[bool] = None,
    hot_pool_persist: Optional[str] = None,
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
    _restore_host_os_io()
    apply_hot_pool_cli_overrides(hot_pool=hot_pool, hot_pool_persist=hot_pool_persist)
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
    agent = None

    conversation = copy.deepcopy(initial_messages)
    hermes_task_id = f"appworld-{dataset}-{task_id}"
    hermes_iterations = resolve_max_hermes_iterations(
        max_hermes_iterations,
        no_tools=no_tools,
    )
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

    t0 = time.perf_counter()
    world_guard_host = None
    try:
        _restore_stdio_for_appworld()
        # AppWorld must initialize before importing AIAgent: Hermes wraps stdout
        # with _SafeWriter, which breaks IPython's shell setup inside AppWorld.
        with AppWorld(task_id=task_id, experiment_name=experiment_name) as world:
            world_guard_host = world
            _ensure_hermes_on_path(hermes_root)
            from run_agent import AIAgent  # type: ignore

            agent_kwargs: Dict[str, Any] = {
                "model": resolved_model,
                "api_key": agent_runtime.get("api_key"),
                "base_url": agent_runtime.get("base_url"),
                "provider": agent_runtime.get("provider"),
                "api_mode": agent_runtime.get("api_mode"),
                "quiet_mode": quiet_mode,
                "max_iterations": hermes_iterations,
                "skip_context_files": skip_context_files,
                "skip_memory": True if amem or dc else skip_memory,
                "save_trajectories": save_trajectories,
                "platform": "appworld-batch",
                "ephemeral_system_prompt": (
                    _HERMES_BRIDGE_EPHEMERAL_NO_TOOLS if no_tools else _HERMES_BRIDGE_EPHEMERAL
                ),
            }
            if no_tools:
                agent_kwargs["enabled_toolsets"] = []
            agent = AIAgent(**agent_kwargs)
            pool = getattr(agent, "_hot_skill_pool", None)
            if pool is not None and hasattr(pool, "clear_exposed_tips"):
                pool.clear_exposed_tips()

            for step_number in range(1, max_steps + 1):
                history, user_message = split_history_for_turn(conversation)
                format_attempt = 0
                step_result: Dict[str, Any] = {}
                messages: List[Dict[str, Any]] = []
                assistant_content = ""
                code = ""
                while True:
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
                                "format_retries": format_attempt,
                                "hermes_result": json_safe(step_result),
                            }
                        )
                        break

                    assistant_content = messages[-1].get("content") or ""
                    code = extract_python_code(assistant_content)
                    if code:
                        break
                    if format_attempt >= _MAX_PYTHON_FORMAT_RETRIES:
                        break
                    format_attempt += 1
                    conversation_retry = copy.deepcopy(messages)
                    conversation_retry.append(
                        {"role": "user", "content": _PYTHON_FORMAT_NUDGE},
                    )
                    history, user_message = split_history_for_turn(conversation_retry)

                if run_error:
                    break

                step_record: Dict[str, Any] = {
                    "step": step_number,
                    "assistant_preview": assistant_content[:500],
                    "code": code,
                    "format_retries": format_attempt,
                }
                if not code:
                    run_error = "No ```python block found in Hermes response."
                    step_record["error"] = run_error
                    steps.append(step_record)
                    break

                # Review threads use Path.mkdir; AppWorld's SafetyGuard is
                # enabled for the duration of execute() and will kill them.
                _wait_background_review(agent)
                execution_output = _execute_world_code(world, code)
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
        _disable_appworld_safety_guard(world_guard_host)
        _restore_stdio_for_appworld()
        _wait_background_review(agent)
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
        finally:
            _restore_host_os_io()

    if evaluation is not None and isinstance(evaluation, dict):
        evaluation.setdefault("steps", len(steps))
        evaluation.setdefault("task_success", bool(evaluation.get("success")))
        if evaluation.get("reward") is None:
            evaluation["reward"] = 1.0 if evaluation.get("task_success") else 0.0

    if agent is not None and evaluation is not None:
        try:
            apply_benchmark_hot_pool_outcome_feedback(
                agent,
                evaluation=evaluation,
                run_result=hermes_stats,
                duration_sec=duration_sec,
                benchmark="appworld",
                wait_background_review=False,
                post_wait=_restore_stdio_for_appworld,
            )
        except Exception:
            pass

    output_dir = (
        appworld_root / "experiments" / "outputs" / experiment_name / "tasks" / task_id
    )

    envelope: Dict[str, Any] = {
        "schema": "appworld.hermes_run.v1",
        "benchmark": "appworld",
        "appworld_task_id": task_id,
        "task_id": task_id,
        "skillsbench_task_id": task_id,
        "dataset": dataset,
        "experiment_name": experiment_name,
        "appworld_root": str(appworld_root),
        "hermes_root": str(hermes_root),
        "prompt_file": str(prompt_file),
        "model": resolved_model,
        "instruction": task_meta.instruction,
        "num_instruction_messages": num_instruction_messages,
        "duration_sec": duration_sec,
        "max_steps": max_steps,
        "max_hermes_iterations": hermes_iterations,
        "hermes_tools_enabled": not no_tools,
        "hot_pool_enabled": False if hot_pool is False else (True if hot_pool else None),
        "hot_pool_persist": hot_pool_persist,
        "amem_enabled": bool(amem),
        "amem_persist": amem_persist,
        "dc_enabled": bool(dc),
        "dc_persist": dc_persist,
        "dc_mode": dc_mode if dc else None,
        "steps_taken": len(steps),
        "task_completed": task_completed,
        "run_error": run_error,
        "steps": steps,
        "hermes_stats": hermes_stats,
        "run_conversation_result": {
            "api_calls": hermes_stats.get("api_calls"),
            "input_tokens": hermes_stats.get("input_tokens"),
            "output_tokens": hermes_stats.get("output_tokens"),
            "total_tokens": hermes_stats.get("total_tokens"),
            "estimated_cost_usd": hermes_stats.get("estimated_cost_usd"),
        },
        "output_directory": str(output_dir),
        "evaluation": evaluation,
    }
    if agent is not None:
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
    return envelope


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one JSONL row using host I/O captured before AppWorld patches."""
    _ensure_host_dir(path.parent)
    payload = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _HOST_OPEN(path, "a", encoding="utf-8") as f:
        f.write(payload)


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
    isolate = parser.add_mutually_exclusive_group()
    isolate.add_argument(
        "--isolate-tasks",
        dest="isolate_tasks",
        action="store_true",
        default=None,
        help=(
            "Run each task in a subprocess (default with --all). A crash or "
            "AppWorld SIGALRM/segfault then fails only that task."
        ),
    )
    isolate.add_argument(
        "--no-isolate-tasks",
        dest="isolate_tasks",
        action="store_false",
        help="Run tasks in-process (a segfault still kills the whole batch).",
    )
    parser.add_argument(
        "--_worker-request",
        dest="worker_request",
        default=None,
        help=argparse.SUPPRESS,
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
    hot_pool_group = parser.add_mutually_exclusive_group()
    hot_pool_group.add_argument(
        "--hot-pool",
        dest="hot_pool",
        action="store_true",
        help="Force-enable skills.hot_pool for this run.",
    )
    hot_pool_group.add_argument(
        "--no-hot-pool",
        dest="hot_pool",
        action="store_false",
        help="Force-disable skills.hot_pool for this run.",
    )
    parser.set_defaults(hot_pool=None)
    parser.add_argument(
        "--hot-pool-persist",
        default=None,
        help="Persist hot pool JSON across tasks (implies pool enabled).",
    )
    add_amem_cli_flags(parser)
    add_dc_cli_flags(parser)
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help=(
            "Writable run dir (default JSONL + hot_pool paths). "
            "Does not replace --experiment-name (AppWorld output namespace)."
        ),
    )
    parser.add_argument(
        "--isolate-hermes-home",
        action="store_true",
        help=(
            "With --experiment-dir: set HERMES_HOME=DIR/hermes_home. Skills are "
            "seeded from repo skills/; config/.env/SOUL.md link to the real home."
        ),
    )
    parser.add_argument(
        "--reset-task-workspaces",
        action="store_true",
        help=(
            "With --isolate-hermes-home: re-seed DIR/hermes_home/skills from "
            "the repo skills/ tree."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip task ids already present in --log-jsonl / experiment runs.jsonl.",
    )
    parser.add_argument(
        "--resume-successes-only",
        action="store_true",
        help="With --resume, only skip tasks whose last row has task_success/success.",
    )

    args = parser.parse_args()
    if args.worker_request:
        return _worker_main(Path(args.worker_request))

    apply_amem_argparse_policy(parser, args)
    apply_dc_argparse_policy(parser, args)

    hermes_root = _expand(args.hermes_root)
    appworld_root = _expand(args.appworld_root)
    prompt_file = _expand(args.prompt_file)

    if args.isolate_hermes_home and not args.experiment_dir:
        parser.error("--isolate-hermes-home requires --experiment-dir")
    if args.reset_task_workspaces and not args.isolate_hermes_home:
        parser.error("--reset-task-workspaces requires --isolate-hermes-home")

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

    experiment_dir: Optional[Path] = None
    config_hermes_home: Optional[Path] = None
    log_path = host_expand_path(args.log_jsonl) if args.log_jsonl else None
    hot_pool = args.hot_pool
    hot_pool_persist = args.hot_pool_persist
    if args.experiment_dir:
        experiment_dir = host_expand_path(args.experiment_dir)
        _ensure_host_dir(experiment_dir)
        if log_path is None:
            log_path = experiment_dir / "runs.jsonl"
        _ensure_host_dir(log_path.parent)
        if args.isolate_hermes_home:
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
                f"[appworld] HERMES_HOME → {home} "
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

    if hot_pool_persist:
        # Resolve once with host-safe expanduser so per-task re-apply never
        # needs pathlib.Path.expanduser after a leaked SafetyGuard.
        hot_pool_persist = str(host_expand_path(hot_pool_persist))

    isolate_tasks = args.isolate_tasks
    if isolate_tasks is None:
        isolate_tasks = bool(args.all) and not bool(args.evaluate_only)
    if isolate_tasks and args.print_summary:
        print(
            "[appworld] each task runs in a subprocess so a crash cannot stop --all",
            flush=True,
        )

    if log_path is not None:
        _ensure_host_dir(log_path.parent)

    apply_hot_pool_cli_overrides(hot_pool=hot_pool, hot_pool_persist=hot_pool_persist)

    skip_ids: Set[str] = set()
    if args.resume:
        if log_path is None:
            parser.error("--resume requires --log-jsonl or --experiment-dir")
        skip_ids = _finished_task_ids(
            log_path, successes_only=bool(args.resume_successes_only)
        )
        if skip_ids:
            before = len(run_ids)
            run_ids = [tid for tid in run_ids if tid not in skip_ids]
            print(
                f"[appworld] resume: skipping {before - len(run_ids)} finished "
                f"({len(run_ids)} pending)",
                flush=True,
            )

    if args.clear_experiment and not args.evaluate_only:
        removed = clear_experiment_output(appworld_root, args.experiment_name)
        if removed and args.print_summary:
            print(f"Cleared experiment output: {removed}", flush=True)

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
                host_expand_path(run_log),
                experiment_name=args.experiment_name,
            )

    shared_runtime: Optional[Dict[str, Any]] = None
    if run_ids and not args.evaluate_only:
        resolved_model = resolve_model_id(args.model)
        if not resolved_model:
            raise SystemExit(
                "No model id resolved. Pass --model <id>, or set model.default via "
                "`hermes model` / ~/.hermes/config.yaml."
            )
        shared_runtime = resolve_agent_runtime(
            model=resolved_model,
            config_hermes_home=config_hermes_home,
        )
        apply_amem_cli_overrides(
            enabled=bool(args.amem),
            persist_dir=amem_persist,
            k=int(args.amem_k),
            llm_model=resolved_model,
            api_key=shared_runtime.get("api_key"),
            base_url=shared_runtime.get("base_url"),
        )
        apply_dc_cli_overrides(
            enabled=bool(args.dc),
            persist_dir=dc_persist,
            mode=str(args.dc_mode),
            k=int(args.dc_k),
            llm_model=resolved_model,
            api_key=shared_runtime.get("api_key"),
            base_url=shared_runtime.get("base_url"),
        )

    for tid in run_ids:
        _restore_host_os_io()
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
                                "task_id": tid,
                                "skillsbench_task_id": tid,
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
                    "benchmark": "appworld",
                    "ts_start_iso": ts_start,
                    "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                    "appworld_task_id": tid,
                    "task_id": tid,
                    "skillsbench_task_id": tid,
                    "dataset": args.dataset,
                    "experiment_name": args.experiment_name,
                    "evaluation": evaluation,
                }
                if run_stats:
                    envelope["run_hermes_stats"] = run_stats
                if experiment_dir is not None:
                    envelope["experiment_dir"] = str(experiment_dir)
                eval_batch["evaluated"] += 1
                eval_batch["evaluated_task_ids"].append(tid)
                if evaluation.get("success") or evaluation.get("task_success"):
                    eval_batch["task_successes"] += 1
                eval_batch["pass_count"] += int(evaluation.get("pass_count") or 0)
                eval_batch["num_tests"] += int(evaluation.get("num_tests") or 0)
            else:
                task_kwargs = {
                    "task_id": tid,
                    "dataset": args.dataset,
                    "hermes_root": hermes_root,
                    "appworld_root": appworld_root,
                    "experiment_name": args.experiment_name,
                    "prompt_file": prompt_file,
                    "model": args.model,
                    "max_steps": args.max_steps,
                    "max_hermes_iterations": args.max_hermes_iterations,
                    "quiet_mode": not args.no_quiet,
                    "skip_context_files": args.skip_context_files,
                    "skip_memory": args.skip_memory,
                    "save_trajectories": args.save_trajectories,
                    "evaluate_after_run": not args.no_evaluate,
                    "no_tools": args.no_tools,
                    "hot_pool": hot_pool,
                    "hot_pool_persist": hot_pool_persist,
                    "amem": bool(args.amem),
                    "amem_persist": amem_persist,
                    "amem_k": int(args.amem_k),
                    "dc": bool(args.dc),
                    "dc_persist": dc_persist,
                    "dc_mode": str(args.dc_mode),
                    "dc_k": int(args.dc_k),
                    "runtime": shared_runtime,
                    "config_hermes_home": config_hermes_home,
                }
                if isolate_tasks:
                    envelope = run_one_task_in_subprocess(**task_kwargs)
                else:
                    envelope = run_one_task(**task_kwargs)
                envelope["ts_start_iso"] = ts_start
                envelope["ts_end_iso"] = datetime.now(timezone.utc).isoformat()
                if experiment_dir is not None:
                    envelope["experiment_dir"] = str(experiment_dir)
                if envelope.get("schema") == "appworld.hermes_run_error.v1":
                    any_failed = True
                    if args.stop_on_error:
                        if log_path:
                            append_jsonl(log_path, json_safe(envelope))
                        print(
                            f"ERROR task={tid}: {envelope.get('error')}",
                            file=sys.stderr,
                        )
                        return 1

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
                        f"hot_pool={envelope.get('hot_pool_enabled')!r} "
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
                "task_id": tid,
                "skillsbench_task_id": tid,
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
