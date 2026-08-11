"""Tests for SkillsBench batch progress / resume helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "scripts"
    / "skillsbench_batch_progress.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("skillsbench_batch_progress", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_resume_skips_completed_retries_errors(tmp_path: Path):
    mod = _load()
    log = tmp_path / "runs.jsonl"
    rows = [
        {
            "schema": "skillsbench.baseline_run.v1",
            "method": "coevoskills",
            "phase": "evolve",
            "skillsbench_task_id": "task-a",
            "ts_end_iso": "2026-01-01T00:00:00Z",
            "evaluation": {"task_success": False},
        },
        {
            "schema": "skillsbench.baseline_run_error.v1",
            "method": "coevoskills",
            "phase": "evolve",
            "skillsbench_task_id": "task-b",
            "ts_end_iso": "2026-01-01T00:01:00Z",
            "error": "boom",
        },
        {
            "schema": "skillsbench.baseline_run.v1",
            "method": "coevoskills",
            "phase": "evolve",
            "skillsbench_task_id": "task-c",
            "ts_end_iso": "2026-01-01T00:02:00Z",
            "evaluation": {"task_success": True},
        },
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    done = mod.load_completed_task_ids(log, phase="evolve", method="coevoskills")
    assert done == {"task-a", "task-c"}
    pending, skipped = mod.filter_pending_tasks(
        ["task-a", "task-b", "task-c", "task-d"], done
    )
    assert pending == ["task-b", "task-d"]
    assert skipped == ["task-a", "task-c"]

    success_only = mod.load_completed_task_ids(
        log, phase="evolve", method="coevoskills", success_only=True
    )
    assert success_only == {"task-c"}
