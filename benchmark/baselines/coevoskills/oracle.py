"""Ground-truth oracle Φ_GT — opaque pass/fail only to the generator.

Evaluates workspace artifacts against Harbor-style SkillsBench tests
(``tests/test_outputs.py``) without mutating the permanent task tree
(unless ``sync_into_task=True``).
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def _load_eval_helpers(hermes_root: Path):
    path = hermes_root / "benchmark" / "scripts" / "evaluate_skillsbench_task.py"
    spec = importlib.util.spec_from_file_location("_coevo_eval_skillsbench", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_task_dir(skillsbench_root: Path, task_id: str) -> Path:
    root = skillsbench_root.resolve()
    candidates = (
        root / "tasks" / task_id,
        root / "skillsbench" / "tasks" / task_id,
    )
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def run_ground_truth_oracle(
    *,
    hermes_root: Path,
    skillsbench_root: Path,
    task_id: str,
    artifacts_dir: Path,
    timeout_sec: float = 600.0,
    sync_into_task: bool = False,
) -> Dict[str, Any]:
    """
    Run host verifier on staged (environment + artifacts).

    Returns a full evaluation dict for logging. Callers MUST expose only
    pass/fail (+ scalar reward) to the skill generator (paper §3.2).
    """
    from . import skill_io

    helpers = _load_eval_helpers(hermes_root)
    task_dir = resolve_task_dir(skillsbench_root, task_id)
    if not task_dir.is_dir():
        return {
            "ran": False,
            "task_success": False,
            "reward": 0.0,
            "error": f"Task directory not found: {task_dir}",
        }

    copied: List[str] = []
    if sync_into_task:
        copied = skill_io.sync_artifacts_into_task_dir(artifacts_dir, task_dir)

    with tempfile.TemporaryDirectory(prefix=f"coevo-oracle-{task_id}-") as tmp:
        stage = Path(tmp)
        fake_root = stage / "dataset"
        fake_task = fake_root / "tasks" / task_id
        fake_task.mkdir(parents=True)

        for name in ("instruction.md", "task.toml"):
            src = task_dir / name
            if src.is_file():
                shutil.copy2(src, fake_task / name)
        for dirname in ("tests", "environment"):
            src = task_dir / dirname
            if src.is_dir():
                shutil.copytree(src, fake_task / dirname)

        if artifacts_dir.is_dir():
            for src in artifacts_dir.rglob("*"):
                if not src.is_file():
                    continue
                rel = src.relative_to(artifacts_dir)
                dest = fake_task / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
        if sync_into_task:
            for entry in task_dir.iterdir():
                if entry.name in helpers._SKIP_TOP_LEVEL or entry.name.startswith("."):
                    continue
                dest = fake_task / entry.name
                if entry.is_file():
                    shutil.copy2(entry, dest)
                elif entry.is_dir() and not dest.exists():
                    shutil.copytree(entry, dest)

        evaluation = helpers.evaluate_task_host(
            task_id=task_id,
            skillsbench_root=fake_root,
            timeout_sec=timeout_sec,
        )

    evaluation.setdefault("ran", "error" not in evaluation)
    evaluation.setdefault("passed", evaluation.get("tests_passed"))
    evaluation.setdefault("failed", evaluation.get("tests_failed"))
    evaluation.setdefault("skipped", evaluation.get("tests_skipped"))
    evaluation["_artifacts_copied"] = copied
    return evaluation


def opaque_signal(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Strip everything except the opaque oracle signal."""
    return {
        "task_success": bool(evaluation.get("task_success")),
        "reward": float(evaluation.get("reward") or 0.0),
        "ran": bool(evaluation.get("ran", True)),
        "error": evaluation.get("error"),
    }
