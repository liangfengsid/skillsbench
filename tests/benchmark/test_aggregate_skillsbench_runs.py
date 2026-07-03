"""Tests for benchmark/scripts/aggregate_skillsbench_runs.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "scripts"
    / "aggregate_skillsbench_runs.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("aggregate_skillsbench_runs", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _record(
    task_id: str,
    *,
    pass_at_turn: dict,
    final_success: bool = False,
    final_tokens: int = 1000,
):
    return {
        "schema": "skillsbench.hermes_run.v1",
        "skillsbench_task_id": task_id,
        "ts_end_iso": "2026-01-01T00:00:00Z",
        "pass_at_turn": pass_at_turn,
        "evaluation": {
            "task_success": final_success,
            "tests_passed": 12 if final_success else 0,
            "tests_total": 12,
        },
        "run_conversation_result": {"total_tokens": final_tokens, "api_calls": 29},
    }


def test_pass_at_conversation_turn_across_tasks():
    mod = _load_module()
    records = [
        _record(
            "task-a",
            pass_at_turn={
                "5": {
                    "turn": 5,
                    "evaluation": {"task_success": False, "tests_passed": 6, "tests_total": 12},
                    "total_tokens": 400000,
                },
                "10": {
                    "turn": 10,
                    "evaluation": {"task_success": True, "tests_passed": 12, "tests_total": 12},
                    "total_tokens": 900000,
                },
            },
        ),
        _record(
            "task-b",
            pass_at_turn={
                "5": {
                    "turn": 5,
                    "evaluation": {"task_success": True, "tests_passed": 12, "tests_total": 12},
                    "total_tokens": 300000,
                },
            },
            final_success=True,
        ),
    ]
    summary = mod.aggregate_records(records, pass_k_values=[5, 10])
    assert summary["pass_k"]["5"]["macro_task_pass_rate"] == 0.5
    assert summary["pass_k"]["5"]["tasks_with_turn_data"] == 2
    assert summary["pass_k"]["5"]["micro_test_pass_rate"] == 0.75
    assert summary["pass_k"]["10"]["macro_task_pass_rate"] == 1.0
    assert summary["pass_k"]["10"]["tasks_with_turn_data"] == 1
    assert summary["pass_k"]["5"]["tokens_mean_at_turn"] == 350000.0


def test_final_metrics_from_evaluation():
    mod = _load_module()
    records = [
        _record("t1", pass_at_turn={}, final_success=True, final_tokens=500),
        _record("t2", pass_at_turn={}, final_success=False, final_tokens=700),
    ]
    summary = mod.aggregate_records(records, pass_k_values=[1])
    assert summary["final"]["macro_task_pass_rate"] == 0.5
    assert summary["final"]["tokens_mean"] == 600.0
