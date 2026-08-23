"""Tests for benchmark/scripts/repair_appworld_runs.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "benchmark" / "scripts" / "repair_appworld_runs.py"


def _load_repair():
    spec = importlib.util.spec_from_file_location("repair_appworld_runs", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


repair = _load_repair()


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_list_infra_failed_tasks(tmp_path):
    log = tmp_path / "runs.jsonl"
    _write_rows(
        log,
        [
            {"task_id": "a_1", "run_error": None, "evaluation": {"success": True}},
            {"task_id": "b_2", "run_error": "ValueError('I/O operation on closed file')"},
            {"task_id": "a_1", "run_error": "ValueError('I/O operation on closed file')"},
            {"task_id": "c_3", "run_error": None, "evaluation": {"success": False}},
        ],
    )
    assert repair.list_infra_failed_tasks(log) == ["a_1", "b_2"]


def test_strip_tasks_and_compact(tmp_path):
    log = tmp_path / "runs.jsonl"
    _write_rows(
        log,
        [
            {"task_id": "keep_1", "run_error": None, "n": 1},
            {"task_id": "drop_1", "run_error": "infra", "n": 1},
            {"task_id": "keep_1", "run_error": None, "n": 2},
            {"task_id": "drop_1", "run_error": "infra", "n": 2},
        ],
    )
    out = tmp_path / "stripped.jsonl"
    removed = repair.strip_tasks(log, ["drop_1"], out)
    assert removed == 2
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["task_id"] for r in rows] == ["keep_1", "keep_1"]

    compacted = tmp_path / "compact.jsonl"
    total_in, total_out = repair.compact_jsonl(out, compacted)
    assert total_in == 2
    assert total_out == 1
    last = json.loads(compacted.read_text(encoding="utf-8").strip())
    assert last["task_id"] == "keep_1"
    assert last["n"] == 2
