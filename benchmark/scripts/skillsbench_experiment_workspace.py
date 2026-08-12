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
      hermes_home/            # only when --isolate-hermes-home (opt-in)
        skills/               # writable copy of source Hermes skills
        config.yaml →         # symlink (or copy) to real ~/.hermes/config.yaml
        .env →                # symlink (or copy) to real ~/.hermes/.env
        SOUL.md →             # symlink (or copy) to real ~/.hermes/SOUL.md
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

# Files kept from the real Hermes home (not sandboxed). Skills stay writable.
_SHARED_HERMES_HOME_FILES: Tuple[str, ...] = ("config.yaml", ".env", "SOUL.md")


def experiment_skillsbench_root(experiment_dir: Path) -> Path:
    return experiment_dir.resolve() / "skillsbench"


def experiment_tasks_dir(experiment_dir: Path) -> Path:
    return experiment_skillsbench_root(experiment_dir) / "tasks"


def experiment_hot_pool_path(experiment_dir: Path) -> Path:
    return experiment_dir.resolve() / "hot_pool.json"


def experiment_hermes_home(experiment_dir: Path) -> Path:
    return experiment_dir.resolve() / "hermes_home"


def resolve_source_hermes_home(explicit: Optional[Path] = None) -> Path:
    """Hermes home to seed from (call before rewriting ``HERMES_HOME``)."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    raw = (os.environ.get("HERMES_HOME") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".hermes").resolve()


def _count_skill_mds(skills_dir: Path) -> int:
    if not skills_dir.is_dir():
        return 0
    return sum(1 for _ in skills_dir.rglob("SKILL.md"))


def _skills_copy_ignore(directory: str, names: List[str]) -> List[str]:
    skip = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    return [n for n in names if n in skip or n.endswith(".pyc")]


def seed_hermes_skills(
    dest_hermes_home: Path,
    *,
    source_hermes_home: Optional[Path] = None,
    reset_skills: bool = False,
) -> Dict[str, Any]:
    """
    Copy ``source/skills`` into ``dest_hermes_home/skills``.

    On first create (or ``reset_skills=True``), replicate the live Hermes skill
    set so the experiment starts with the same skills without sharing writes.
    If dest skills already exist and ``reset_skills`` is false, leave them
    alone (resume / mid-experiment mutations stay intact).
    """
    source_home = resolve_source_hermes_home(source_hermes_home)
    src_skills = source_home / "skills"
    dest_home = dest_hermes_home.expanduser().resolve()
    dest_skills = dest_home / "skills"

    if dest_home == source_home:
        return {
            "action": "skipped_same_home",
            "source_skills": str(src_skills),
            "dest_skills": str(dest_skills),
            "n_skills": _count_skill_mds(dest_skills),
        }

    dest_home.mkdir(parents=True, exist_ok=True)
    dest_nonempty = dest_skills.is_dir() and any(dest_skills.iterdir())
    if dest_nonempty and not reset_skills:
        return {
            "action": "exists",
            "source_skills": str(src_skills),
            "dest_skills": str(dest_skills),
            "n_skills": _count_skill_mds(dest_skills),
        }

    if dest_skills.exists():
        shutil.rmtree(dest_skills)

    if not src_skills.is_dir():
        dest_skills.mkdir(parents=True, exist_ok=True)
        return {
            "action": "empty_source",
            "source_skills": str(src_skills),
            "dest_skills": str(dest_skills),
            "n_skills": 0,
        }

    shutil.copytree(src_skills, dest_skills, ignore=_skills_copy_ignore)
    return {
        "action": "seeded" if not dest_nonempty else "reseeded",
        "source_skills": str(src_skills),
        "dest_skills": str(dest_skills),
        "n_skills": _count_skill_mds(dest_skills),
    }


def link_shared_hermes_files(
    dest_hermes_home: Path,
    *,
    source_hermes_home: Optional[Path] = None,
    names: Sequence[str] = _SHARED_HERMES_HOME_FILES,
    prefer_symlink: bool = True,
) -> Dict[str, Any]:
    """
    Point experiment-home ``config.yaml`` / ``.env`` / ``SOUL.md`` at the real
    Hermes home (symlink by default; copy if symlink is unavailable).

    Skills stay a separate writable tree; these files should keep matching the
    live profile. Note: a symlink write-through would touch the real file —
    SkillsBench batch runs rarely mutate these.
    """
    source_home = resolve_source_hermes_home(source_hermes_home)
    dest_home = dest_hermes_home.expanduser().resolve()
    dest_home.mkdir(parents=True, exist_ok=True)

    if dest_home == source_home:
        return {
            "action": "skipped_same_home",
            "source_hermes_home": str(source_home),
            "files": {},
        }

    files: Dict[str, Dict[str, Any]] = {}
    for name in names:
        src = source_home / name
        dest = dest_home / name
        if not src.exists():
            files[name] = {
                "action": "missing_source",
                "source": str(src),
                "dest": str(dest),
            }
            continue

        src_resolved = src.resolve()
        if dest.is_symlink():
            try:
                if dest.resolve() == src_resolved:
                    files[name] = {
                        "action": "exists",
                        "mode": "symlink",
                        "source": str(src_resolved),
                        "dest": str(dest),
                    }
                    continue
            except OSError:
                pass
        elif dest.is_file() and not prefer_symlink:
            # Prior copy — refresh content from source.
            shutil.copy2(src_resolved, dest)
            files[name] = {
                "action": "refreshed",
                "mode": "copy",
                "source": str(src_resolved),
                "dest": str(dest),
            }
            continue

        if dest.exists() or dest.is_symlink():
            dest.unlink()

        mode = "symlink"
        action = "linked"
        if prefer_symlink:
            try:
                dest.symlink_to(src_resolved)
            except OSError:
                shutil.copy2(src_resolved, dest)
                mode = "copy"
                action = "copied"
        else:
            shutil.copy2(src_resolved, dest)
            mode = "copy"
            action = "copied"

        files[name] = {
            "action": action,
            "mode": mode,
            "source": str(src_resolved),
            "dest": str(dest),
        }

    linked = sum(1 for f in files.values() if f.get("action") in ("linked", "exists", "copied", "refreshed"))
    return {
        "action": "ok",
        "source_hermes_home": str(source_home),
        "n_linked": linked,
        "files": files,
    }


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
    isolate_hermes_home: bool = False,
    apply_hermes_home_env: bool = False,
    source_hermes_home: Optional[Path] = None,
    seed_hermes_skills_from_source: bool = True,
    reset_hermes_skills: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Ensure ``experiment_dir`` has writable copies of the given tasks.

    Does **not** symlink into the shared SkillsBench tree (writes would leak).
    ``task_ids`` may be empty when only isolating ``HERMES_HOME``.

    By default Hermes keeps using the real ``~/.hermes`` (default skills,
    memory, config). Pass ``isolate_hermes_home=True`` for a sandboxed
    ``DIR/hermes_home``. When isolating:

    - ``skills/`` is a writable **copy** of the source Hermes skills tree
      (unless ``seed_hermes_skills_from_source=False``).
    - ``config.yaml``, ``.env``, and ``SOUL.md`` are **symlinked** (or copied)
      to the real Hermes home so provider/config/identity stay shared.
    """
    experiment_dir = experiment_dir.expanduser().resolve()
    source_skillsbench_root = source_skillsbench_root.expanduser().resolve()
    # Capture source home *before* we rewrite HERMES_HOME below.
    resolved_source_home = resolve_source_hermes_home(source_hermes_home)
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
    skills_seed: Optional[Dict[str, Any]] = None
    shared_files: Optional[Dict[str, Any]] = None
    if isolate_hermes_home:
        hermes_home.mkdir(parents=True, exist_ok=True)
        if seed_hermes_skills_from_source:
            do_reset = (
                bool(reset_outputs)
                if reset_hermes_skills is None
                else bool(reset_hermes_skills)
            )
            skills_seed = seed_hermes_skills(
                hermes_home,
                source_hermes_home=resolved_source_home,
                reset_skills=do_reset,
            )
        else:
            (hermes_home / "skills").mkdir(parents=True, exist_ok=True)
            skills_seed = {
                "action": "skipped_no_seed",
                "source_skills": str(resolved_source_home / "skills"),
                "dest_skills": str(hermes_home / "skills"),
                "n_skills": _count_skill_mds(hermes_home / "skills"),
            }
        shared_files = link_shared_hermes_files(
            hermes_home,
            source_hermes_home=resolved_source_home,
        )
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
        "source_hermes_home": str(resolved_source_home) if isolate_hermes_home else None,
        "isolate_hermes_home": isolate_hermes_home,
        "skills_seed": skills_seed,
        "shared_hermes_files": shared_files,
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
