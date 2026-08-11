#!/usr/bin/env python3
"""
Per-experiment SkillsBench workspaces so Hermes / hot-pool / CoEvo runs do not
pollute the shared ``benchmark/skillsbench/tasks/`` tree or each other.

Layout (default)::

    <experiment_dir>/
      MANIFEST.json
      skillsbench/
        tasks/<task_id>/…     # writable copies used by agent + host eval
      hot_pool.json           # recommended --hot-pool-persist target
      hermes_home/            # optional isolated HERMES_HOME (skills, memory, …)
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Top-level names that are part of the task definition (not agent outputs).
_TASK_DEF_NAMES = frozenset(
    {
        "solution",
        "tests",
        "instruction.md",
        "task.toml",
        "environment",
        ".git",
        "README.md",
        "readme.md",
    }
)


def experiment_skillsbench_root(experiment_dir: Path) -> Path:
    return experiment_dir.resolve() / "skillsbench"


def experiment_tasks_dir(experiment_dir: Path) -> Path:
    return experiment_skillsbench_root(experiment_dir) / "tasks"


def experiment_hot_pool_path(experiment_dir: Path) -> Path:
    return experiment_dir.resolve() / "hot_pool.json"


def experiment_hermes_home(experiment_dir: Path) -> Path:
    return experiment_dir.resolve() / "hermes_home"


def _copy_task(src: Path, dest: Path, *, reset_outputs: bool) -> str:
    """
    Materialize one task into the experiment tree.

    Returns action: ``copied`` | ``refreshed_outputs`` | ``exists``.
    """
    if not src.is_dir():
        raise FileNotFoundError(f"Source task not found: {src}")

    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)
        return "copied"

    if not reset_outputs:
        return "exists"

    # Keep definition files; wipe agent-written top-level outputs and refresh
    # definition from source (environment/tests may be updated upstream).
    for entry in list(dest.iterdir()):
        if entry.name in _TASK_DEF_NAMES or entry.name.startswith("."):
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink(missing_ok=True)
        else:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink(missing_ok=True)

    for entry in src.iterdir():
        if entry.name.startswith(".") and entry.name != ".git":
            continue
        target = dest / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)
    return "refreshed_outputs"


def prepare_experiment_workspace(
    *,
    experiment_dir: Path,
    source_skillsbench_root: Path,
    task_ids: Sequence[str],
    reset_outputs: bool = False,
    isolate_hermes_home: bool = True,
    apply_hermes_home_env: bool = True,
) -> Dict[str, Any]:
    """
    Ensure ``experiment_dir`` has writable copies of the given tasks.

    Does **not** symlink into the shared SkillsBench tree (writes would leak).
    ``task_ids`` may be empty when only isolating ``HERMES_HOME``.
    """
    experiment_dir = experiment_dir.expanduser().resolve()
    source_skillsbench_root = source_skillsbench_root.expanduser().resolve()
    src_tasks = source_skillsbench_root / "tasks"
    if task_ids and not src_tasks.is_dir():
        raise FileNotFoundError(f"Source tasks dir not found: {src_tasks}")

    dest_tasks = experiment_tasks_dir(experiment_dir)
    if task_ids:
        dest_tasks.mkdir(parents=True, exist_ok=True)

    actions: Dict[str, str] = {}
    missing: List[str] = []
    for tid in task_ids:
        src = src_tasks / tid
        if not src.is_dir():
            missing.append(tid)
            continue
        actions[tid] = _copy_task(src, dest_tasks / tid, reset_outputs=reset_outputs)

    if missing:
        preview = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        raise FileNotFoundError(f"Unknown task ids in source SkillsBench: {preview}")

    hermes_home = experiment_hermes_home(experiment_dir)
    if isolate_hermes_home:
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "skills").mkdir(parents=True, exist_ok=True)
        if apply_hermes_home_env:
            os.environ["HERMES_HOME"] = str(hermes_home)

    hot_pool = experiment_hot_pool_path(experiment_dir)
    skillsbench_root = experiment_skillsbench_root(experiment_dir)
    prompt_tasks_base = str(dest_tasks.resolve()) if task_ids else None

    manifest = {
        "schema": "skillsbench.experiment_workspace.v1",
        "experiment_dir": str(experiment_dir),
        "source_skillsbench_root": str(source_skillsbench_root),
        "skillsbench_root": str(skillsbench_root) if task_ids else None,
        "prompt_tasks_base": prompt_tasks_base,
        "hot_pool_path": str(hot_pool),
        "hermes_home": str(hermes_home) if isolate_hermes_home else None,
        "isolate_hermes_home": isolate_hermes_home,
        "reset_outputs": reset_outputs,
        "task_ids": list(task_ids),
        "actions": actions,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (experiment_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def clear_agent_outputs(task_dir: Path) -> List[str]:
    """Remove top-level agent artifacts from a task dir (keep definition files)."""
    removed: List[str] = []
    if not task_dir.is_dir():
        return removed
    for entry in list(task_dir.iterdir()):
        if entry.name in _TASK_DEF_NAMES or entry.name.startswith("."):
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink(missing_ok=True)
        removed.append(entry.name)
    return removed
