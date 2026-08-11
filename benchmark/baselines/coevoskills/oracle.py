"""Ground-truth oracle Φ_GT — opaque pass/fail only to the generator.

Evaluates workspace artifacts against SkillsBench ``tests/test_outputs.py`` without
mutating the permanent task tree (unless ``sync_into_task=True``).
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PYTEST_SUMMARY_RE = re.compile(
    r"(?P<passed>\d+) passed"
    r"(?:, (?P<failed>\d+) failed)?"
    r"(?:, (?P<skipped>\d+) skipped)?"
)


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
    direct = skillsbench_root.resolve() / "tasks" / task_id
    if direct.is_dir():
        return direct
    alt = skillsbench_root.resolve() / "skillsbench" / "tasks" / task_id
    return alt if alt.is_dir() else direct


def _parse_pytest_stdout(stdout: str) -> Tuple[int, int, int]:
    passed = failed = skipped = 0
    for line in reversed(stdout.splitlines()):
        sm = _PYTEST_SUMMARY_RE.search(line)
        if sm:
            passed = int(sm.group("passed") or 0)
            failed = int(sm.group("failed") or 0)
            skipped = int(sm.group("skipped") or 0)
            break
    return passed, failed, skipped


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
    Run host pytest on staged (environment + artifacts).

    Returns a full evaluation dict for logging. Callers MUST expose only
    pass/fail (+ scalar reward) to the skill generator (paper §3.2).
    """
    from . import skill_io

    helpers = _load_eval_helpers(hermes_root)
    task_dir = resolve_task_dir(skillsbench_root, task_id)
    test_src = task_dir / "tests" / "test_outputs.py"
    if not task_dir.is_dir():
        return {
            "ran": False,
            "task_success": False,
            "reward": 0.0,
            "error": f"Task directory not found: {task_dir}",
        }
    if not test_src.is_file():
        return {
            "ran": False,
            "task_success": False,
            "reward": 0.0,
            "error": f"Missing tests/test_outputs.py under {task_dir}",
        }

    copied: List[str] = []
    if sync_into_task:
        copied = skill_io.sync_artifacts_into_task_dir(artifacts_dir, task_dir)

    with tempfile.TemporaryDirectory(prefix=f"coevo-oracle-{task_id}-") as tmp:
        stage = Path(tmp)
        # Build a fake task dir: env + artifact files at top level like Hermes runs
        fake_task = stage / "task"
        fake_task.mkdir()
        env_src = task_dir / "environment"
        if env_src.is_dir():
            shutil.copytree(env_src, fake_task / "environment")
        # Agent outputs: from artifacts_dir (and any already-synced top-level if sync)
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

        host_root = helpers.stage_task_for_host_eval(fake_task, stage / "stage")
        sim_info = helpers.maybe_run_simulation(host_root)

        tests_dir = stage / "tests"
        tests_dir.mkdir()
        adapted = helpers.adapt_container_paths(
            test_src.read_text(encoding="utf-8", errors="replace"),
            host_root,
        )
        (tests_dir / "test_outputs.py").write_text(adapted, encoding="utf-8")
        # Copy any conftest / helpers beside test_outputs
        src_tests = task_dir / "tests"
        for extra in src_tests.iterdir():
            if extra.name == "test_outputs.py" or not extra.is_file():
                continue
            if extra.suffix in {".py", ".json", ".yaml", ".yml", ".txt"}:
                body = extra.read_text(encoding="utf-8", errors="replace")
                if extra.suffix == ".py":
                    body = helpers.adapt_container_paths(body, host_root)
                (tests_dir / extra.name).write_text(body, encoding="utf-8")

        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(tests_dir / "test_outputs.py"),
                    "-v",
                    "--tb=short",
                ],
                cwd=str(host_root),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return {
                "ran": True,
                "task_success": False,
                "reward": 0.0,
                "error": "oracle pytest timeout",
                "simulation": sim_info,
                "_artifacts_copied": copied,
            }

        passed, failed, skipped = _parse_pytest_stdout(proc.stdout or "")
        total = passed + failed
        task_success = proc.returncode == 0 and failed == 0 and total > 0
        reward = 1.0 if task_success else (round(passed / total, 6) if total else 0.0)
        return {
            "ran": True,
            "task_success": task_success,
            "reward": reward,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "exit_code": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-3000:],
            "stderr_tail": (proc.stderr or "")[-1500:],
            "simulation": sim_info,
            "_artifacts_copied": copied,
        }


def opaque_signal(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Strip everything except the opaque oracle signal."""
    return {
        "task_success": bool(evaluation.get("task_success")),
        "reward": float(evaluation.get("reward") or 0.0),
        "ran": bool(evaluation.get("ran", True)),
        "error": evaluation.get("error"),
    }
