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
    assert summary["pass_k"]["5"]["macro_success_rate"] == 0.5
    assert summary["pass_k"]["5"]["tasks_with_turn_data"] == 2
    assert summary["pass_k"]["5"]["micro_test_pass_rate"] == 0.75
    assert summary["pass_k"]["5"]["micro_success_rate"] == 0.75
    # Cumulative: task-b passed at turn 5 counts toward pass@10 even without a
    # turn-10 snapshot; task-a passed at turn 10.
    assert summary["pass_k"]["10"]["macro_task_pass_rate"] == 1.0
    assert summary["pass_k"]["10"]["macro_success_rate"] == 1.0
    assert summary["pass_k"]["10"]["tasks_with_turn_data"] == 2
    assert summary["pass_k"]["10"]["micro_success_rate"] == 1.0
    assert summary["pass_k"]["10"]["cumulative"] is True
    assert summary["pass_k"]["5"]["tokens_mean_at_turn"] == 350000.0


def test_pass_k_cumulative_keeps_early_success_when_later_missing_or_fails():
    mod = _load_module()
    records = [
        _record(
            "early-finish",
            pass_at_turn={
                "10": {
                    "turn": 10,
                    "evaluation": {"task_success": True, "tests_passed": 4, "tests_total": 4},
                    "total_tokens": 100,
                },
            },
            final_success=True,
        ),
        _record(
            "regress",
            pass_at_turn={
                "10": {
                    "turn": 10,
                    "evaluation": {"task_success": True, "tests_passed": 4, "tests_total": 4},
                    "total_tokens": 200,
                },
                "30": {
                    "turn": 30,
                    "evaluation": {"task_success": False, "tests_passed": 1, "tests_total": 4},
                    "total_tokens": 500,
                },
            },
        ),
    ]
    summary = mod.aggregate_records(records, pass_k_values=[10, 30])
    assert summary["pass_k"]["10"]["macro_success_rate"] == 1.0
    # Both tasks still count at 30: early-finish has no @30 key; regress failed
    # at 30 but already passed at 10.
    assert summary["pass_k"]["30"]["macro_success_rate"] == 1.0
    assert summary["pass_k"]["30"]["tasks_with_turn_data"] == 2
    # Micro is terminal status: early-finish final 12/12, regress final 0/12
    # (api_calls=29 ≤ 30), not the best mid-run 4/4.
    assert summary["pass_k"]["30"]["micro_success_rate"] == 0.5
    # Usage prefers in-budget final for both tasks (default tokens=1000).
    assert summary["pass_k"]["30"]["tokens_mean_at_turn"] == 1000.0


def test_pass_k_includes_final_success_between_sparse_checkpoints():
    """Like earthquake-plate-calculation: fail @1/5/10, succeed at final @15."""
    mod = _load_module()
    records = [
        {
            "schema": "skillsbench.hermes_run.v1",
            "skillsbench_task_id": "earthquake-plate-calculation",
            "ts_end_iso": "2026-01-01T00:00:00Z",
            "pass_at_turn": {
                "1": {
                    "turn": 1,
                    "evaluation": {"task_success": False, "tests_passed": 0, "tests_total": 8},
                    "total_tokens": 10,
                },
                "5": {
                    "turn": 5,
                    "evaluation": {"task_success": False, "tests_passed": 0, "tests_total": 8},
                    "total_tokens": 50,
                },
                "10": {
                    "turn": 10,
                    "evaluation": {"task_success": False, "tests_passed": 0, "tests_total": 8},
                    "total_tokens": 100,
                },
            },
            "evaluation": {
                "task_success": True,
                "tests_passed": 8,
                "tests_total": 8,
            },
            "metrics": {
                "final": {
                    "task_success": True,
                    "tests_passed": 8,
                    "tests_total": 8,
                    "api_calls": 15,
                    "tokens": {"total": 200},
                }
            },
            "run_conversation_result": {"total_tokens": 200, "api_calls": 15},
        }
    ]
    summary = mod.aggregate_records(records, pass_k_values=[10, 15, 60])
    # Still failing at the last sparse checkpoint.
    assert summary["pass_k"]["10"]["macro_success_rate"] == 0.0
    assert summary["pass_k"]["10"]["micro_success_rate"] == 0.0
    # Final success at 15 counts once the budget covers that many user iterations.
    assert summary["pass_k"]["15"]["macro_success_rate"] == 1.0
    assert summary["pass_k"]["15"]["micro_success_rate"] == 1.0
    assert summary["pass_k"]["60"]["macro_success_rate"] == 1.0
    assert summary["pass_k"]["60"]["tasks_passed_at_turn"] == 1
    assert summary["final"]["macro_success_rate"] == 1.0


def test_pass_k_micro_prefers_final_over_inflated_later_checkpoint():
    """Like dynamic-object-aware-egomotion: high mid-run micro, worse final."""
    mod = _load_module()
    records = [
        {
            "schema": "skillsbench.hermes_run.v1",
            "skillsbench_task_id": "dynamic-object-aware-egomotion",
            "ts_end_iso": "2026-01-01T00:00:00Z",
            "pass_at_turn": {
                "30": {
                    "turn": 30,
                    "evaluation": {"task_success": False, "tests_passed": 4, "tests_total": 8},
                    "total_tokens": 1000,
                    "api_calls": 31,
                },
                "60": {
                    "turn": 60,
                    "evaluation": {"task_success": False, "tests_passed": 9, "tests_total": 10},
                    "total_tokens": 5000,
                    "api_calls": 60,
                },
            },
            "evaluation": {
                "task_success": False,
                "tests_passed": 0,
                "tests_total": 2,
            },
            "metrics": {
                "final": {
                    "task_success": False,
                    "tests_passed": 0,
                    "tests_total": 2,
                    "api_calls": 56,
                    "tokens": {"total": 4000},
                }
            },
            "run_conversation_result": {"total_tokens": 4000, "api_calls": 56},
        }
    ]
    summary = mod.aggregate_records(records, pass_k_values=[30, 60])
    assert summary["pass_k"]["30"]["micro_success_rate"] == 0.5  # 4/8 checkpoint
    # Prefer final 0/2 over the turn-60 9/10 snapshot.
    assert summary["pass_k"]["60"]["micro_success_rate"] == 0.0
    assert summary["pass_k"]["60"]["tests_passed_sum"] == 0
    assert summary["pass_k"]["60"]["tests_total_sum"] == 2
    assert summary["final"]["micro_success_rate"] == 0.0
    assert summary["pass_k"]["60"]["tokens_mean_at_turn"] == 4000.0


def test_final_metrics_from_evaluation():
    mod = _load_module()
    records = [
        _record("t1", pass_at_turn={}, final_success=True, final_tokens=500),
        _record("t2", pass_at_turn={}, final_success=False, final_tokens=700),
    ]
    summary = mod.aggregate_records(records, pass_k_values=[1])
    assert summary["final"]["macro_task_pass_rate"] == 0.5
    assert summary["final"]["macro_success_rate"] == 0.5
    assert summary["final"]["tokens_mean"] == 600.0
    assert summary["final"]["tokens"]["mean"] == 600.0
