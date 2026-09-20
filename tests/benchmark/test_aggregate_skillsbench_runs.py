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
    summary = mod.aggregate_records(
        records, pass_k_values=[10, 30], include_final_in_pass_k=True
    )
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
    summary = mod.aggregate_records(
        records, pass_k_values=[10, 15, 60], include_final_in_pass_k=True
    )
    # Still failing at the last sparse checkpoint.
    assert summary["pass_k"]["10"]["macro_success_rate"] == 0.0
    assert summary["pass_k"]["10"]["micro_success_rate"] == 0.0
    # Final success at 15 counts once the budget covers that many user iterations.
    assert summary["pass_k"]["15"]["macro_success_rate"] == 1.0
    assert summary["pass_k"]["15"]["micro_success_rate"] == 1.0
    assert summary["pass_k"]["60"]["macro_success_rate"] == 1.0
    assert summary["pass_k"]["60"]["tasks_passed_at_turn"] == 1
    assert summary["final"]["macro_success_rate"] == 1.0


def test_pass_k_checkpoints_only_ignores_final_eval():
    """Between-checkpoint final success must not inflate pass@k when opted in."""
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
                "10": {
                    "turn": 10,
                    "evaluation": {"task_success": False, "tests_passed": 2, "tests_total": 8},
                    "total_tokens": 100,
                },
            },
            "evaluation": {
                "task_success": True,
                "tests_passed": 8,
                "tests_total": 8,
            },
            "run_conversation_result": {"total_tokens": 200, "api_calls": 15},
        }
    ]
    summary = mod.aggregate_records(
        records,
        pass_k_values=[10, 15, 60],
        include_final_in_pass_k=False,
    )
    assert summary["filters"]["include_final_in_pass_k"] is False
    assert summary["pass_k"]["10"]["include_final"] is False
    assert summary["pass_k"]["10"]["macro_success_rate"] == 0.0
    assert summary["pass_k"]["10"]["micro_success_rate"] == 0.25
    assert summary["pass_k"]["15"]["macro_success_rate"] == 0.0
    assert summary["pass_k"]["60"]["macro_success_rate"] == 0.0
    assert summary["pass_k"]["60"]["tasks_passed_at_turn"] == 0
    # Latest checkpoint ≤60 is still the failing @10 snapshot.
    assert summary["pass_k"]["60"]["micro_success_rate"] == 0.25
    assert summary["final"]["macro_success_rate"] == 1.0


def test_pass_k_default_includes_in_budget_final():
    """Default aggregation credits in-budget finals (not checkpoints-only)."""
    mod = _load_module()
    records = [
        {
            "schema": "skillsbench.hermes_run.v1",
            "skillsbench_task_id": "final-only",
            "ts_end_iso": "2026-01-01T00:00:00Z",
            "pass_at_turn": {
                "1": {
                    "turn": 1,
                    "evaluation": {
                        "task_success": False,
                        "tests_passed": 0,
                        "tests_total": 2,
                    },
                }
            },
            "evaluation": {
                "task_success": True,
                "tests_passed": 2,
                "tests_total": 2,
            },
            "run_conversation_result": {"total_tokens": 50, "api_calls": 5},
        }
    ]
    summary = mod.aggregate_records(records, pass_k_values=[5, 60])
    assert summary["filters"]["include_final_in_pass_k"] is True
    assert summary["pass_k"]["5"]["macro_success_rate"] == 1.0
    assert summary["pass_k"]["60"]["macro_success_rate"] == 1.0
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
    summary = mod.aggregate_records(
        records, pass_k_values=[30, 60], include_final_in_pass_k=True
    )
    assert summary["pass_k"]["30"]["micro_success_rate"] == 0.5  # 4/8 checkpoint
    # Prefer final 0/2 over the turn-60 9/10 snapshot.
    assert summary["pass_k"]["60"]["micro_success_rate"] == 0.0
    assert summary["pass_k"]["60"]["tests_passed_sum"] == 0
    assert summary["pass_k"]["60"]["tests_total_sum"] == 2
    assert summary["final"]["micro_success_rate"] == 0.0
    assert summary["pass_k"]["60"]["tokens_mean_at_turn"] == 4000.0


def test_cost_to_succeed_respects_max_user_iterations():
    """Retry / over-budget finals are excluded from cost_to_succeed."""
    mod = _load_module()
    records = [
        _record(
            "in-budget-ckpt",
            pass_at_turn={
                "10": {
                    "turn": 10,
                    "evaluation": {"task_success": True, "tests_passed": 4, "tests_total": 4},
                    "total_tokens": 100,
                },
            },
            final_success=True,
            final_tokens=400,
        ),
        {
            "schema": "skillsbench.hermes_run.v1",
            "skillsbench_task_id": "retry-only",
            "ts_end_iso": "2026-01-01T00:00:01Z",
            "pass_at_turn": {
                "60": {
                    "turn": 60,
                    "evaluation": {"task_success": False, "tests_passed": 0, "tests_total": 4},
                    "total_tokens": 800,
                },
            },
            "evaluation": {
                "task_success": True,
                "tests_passed": 4,
                "tests_total": 4,
            },
            "run_conversation_result": {"total_tokens": 900, "api_calls": 45},
        },
        {
            "schema": "skillsbench.hermes_run.v1",
            "skillsbench_task_id": "over-budget-final",
            "ts_end_iso": "2026-01-01T00:00:02Z",
            "pass_at_turn": {},
            "evaluation": {
                "task_success": True,
                "tests_passed": 4,
                "tests_total": 4,
            },
            "run_conversation_result": {"total_tokens": 2000, "api_calls": 90},
        },
    ]
    records[0]["duration_sec"] = 10.0
    records[1]["duration_sec"] = 80.0
    records[2]["duration_sec"] = 200.0

    capped = mod.aggregate_records(
        records,
        pass_k_values=[10, 60],
        max_user_iterations=60,
        include_final_in_pass_k=True,
    )
    cts = capped["cost_to_succeed"]
    assert cts["max_user_iterations"] == 60
    # in-budget-ckpt (turn 10) + retry-only (final api_calls=45 ≤ 60)
    assert cts["n_succeeded"] == 2
    assert cts["tokens"]["n"] == 2
    assert cts["tokens"]["mean"] == 500.0  # 100 + 900
    assert cts["user_iterations"]["mean"] == 27.5  # 10 + 45
    assert "over-budget-final" not in str(cts["sources"])

    ckpt_only = mod.aggregate_records(
        records,
        pass_k_values=[10, 60],
        max_user_iterations=60,
        include_final_in_pass_k=False,
    )
    cts2 = ckpt_only["cost_to_succeed"]
    assert cts2["n_succeeded"] == 1
    assert cts2["tokens"]["mean"] == 100.0
    assert cts2["user_iterations"]["mean"] == 10.0
    assert cts2["sources"] == {"pass_at_turn": 1}
    assert cts2["duration_sec"]["mean"] == 10.0


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


def test_pass_k_macro_uses_full_suite_denominator():
    """Tasks still running past k (no in-budget obs) count as not passed.

    Matches the A-Mem case with empty pass_at_turn: early finishers must not
    inflate pass@k_macro by shrinking the denominator.
    """
    mod = _load_module()
    records = [
        {
            "schema": "skillsbench.hermes_run.v1",
            "skillsbench_task_id": "early-ok",
            "ts_end_iso": "2026-01-01T00:00:00Z",
            "pass_at_turn": {},
            "evaluation": {
                "task_success": True,
                "tests_passed": 12,
                "tests_total": 12,
            },
            "run_conversation_result": {"total_tokens": 100, "api_calls": 6},
        },
        {
            "schema": "skillsbench.hermes_run.v1",
            "skillsbench_task_id": "long-runner",
            "ts_end_iso": "2026-01-01T00:00:01Z",
            "pass_at_turn": {},
            "evaluation": {
                "task_success": True,
                "tests_passed": 12,
                "tests_total": 12,
            },
            "run_conversation_result": {"total_tokens": 9000, "api_calls": 45},
        },
    ]
    summary = mod.aggregate_records(
        records, pass_k_values=[10, 60], include_final_in_pass_k=True
    )
    # Only early-ok is in-budget at 10; long-runner is still a suite failure@10.
    assert summary["pass_k"]["10"]["tasks_total"] == 2
    assert summary["pass_k"]["10"]["tasks_with_turn_data"] == 1
    assert summary["pass_k"]["10"]["tasks_passed_at_turn"] == 1
    assert summary["pass_k"]["10"]["macro_success_rate"] == 0.5
    # Micro: early-ok 12/12 + long-runner 0/12 (no in-budget obs).
    assert summary["pass_k"]["10"]["tests_passed_sum"] == 12
    assert summary["pass_k"]["10"]["tests_total_sum"] == 24
    assert summary["pass_k"]["10"]["micro_success_rate"] == 0.5
    text = mod.format_metrics_summary_text(summary)
    assert "pass@10_macro=0.5000 (1/2)" in text
    assert "pass@10_micro=0.5000 (12/24)" in text
    # By 60 both finals are in budget → 2/2 and 24/24.
    assert summary["pass_k"]["60"]["macro_success_rate"] == 1.0
    assert summary["pass_k"]["60"]["tasks_passed_at_turn"] == 2
    assert summary["pass_k"]["60"]["tasks_total"] == 2
    assert summary["pass_k"]["60"]["micro_success_rate"] == 1.0
    assert summary["pass_k"]["60"]["tests_total_sum"] == 24


def test_auc_macro_micro_continuous_k():
    """AUC is mean of pass@k rates over continuous k=1..K."""
    mod = _load_module()
    # Passes at turn 3 via final usage (api_calls=3); fails for k=1,2.
    records = [
        {
            "schema": "skillsbench.hermes_run.v1",
            "skillsbench_task_id": "late",
            "ts_end_iso": "2026-01-01T00:00:00Z",
            "pass_at_turn": {
                "1": {
                    "turn": 1,
                    "evaluation": {
                        "task_success": False,
                        "tests_passed": 0,
                        "tests_total": 4,
                    },
                    "total_tokens": 10,
                },
            },
            "evaluation": {
                "task_success": True,
                "tests_passed": 4,
                "tests_total": 4,
            },
            "run_conversation_result": {"total_tokens": 100, "api_calls": 3},
        }
    ]
    summary = mod.aggregate_records(
        records, pass_k_values=[1, 3], auc_max_k=4, include_final_in_pass_k=True
    )
    assert summary["auc"]["k_max"] == 4
    assert summary["auc"]["method"] == "mean_of_rates"
    # macro: k=1,2 → 0; k=3,4 → 1 → mean 0.5
    assert summary["auc"]["macro_success_rate"] == 0.5
    assert summary["auc"]["macro_curve"] == [0.0, 0.0, 1.0, 1.0]
    # micro: k=1 uses @1 snapshot 0/4; k=2 still no better in-budget obs than
    # latest ≤2 which is @1 → 0/4; k=3,4 use final 4/4 → mean 0.5
    assert summary["auc"]["micro_success_rate"] == 0.5
    assert summary["filters"]["auc_max_k"] == 4


def test_auc_disabled_when_max_k_zero():
    mod = _load_module()
    records = [
        _record(
            "a",
            pass_at_turn={
                "1": {
                    "turn": 1,
                    "evaluation": {
                        "task_success": True,
                        "tests_passed": 2,
                        "tests_total": 2,
                    },
                    "total_tokens": 1,
                },
            },
            final_success=True,
        )
    ]
    summary = mod.aggregate_records(records, pass_k_values=[1], auc_max_k=0)
    assert summary["auc"] == {}


def test_summary_order_pass_k_auc_then_final_with_success_turn():
    mod = _load_module()
    records = [
        {
            "schema": "skillsbench.hermes_run.v1",
            "skillsbench_task_id": "verified",
            "ts_end_iso": "2026-01-01T00:00:00Z",
            "pass_at_turn": {
                "5": {
                    "turn": 5,
                    "evaluation": {
                        "task_success": False,
                        "tests_passed": 0,
                        "tests_total": 2,
                    },
                    "total_tokens": 50,
                },
            },
            "evaluation": {
                "task_success": True,
                "tests_passed": 2,
                "tests_total": 2,
            },
            "host_verification_attempts": [
                {
                    "attempt": 1,
                    "api_calls": 12,
                    "evaluation": {
                        "task_success": False,
                        "tests_passed": 0,
                        "tests_total": 2,
                    },
                },
                {
                    "attempt": 2,
                    "api_calls": 18,
                    "evaluation": {
                        "task_success": True,
                        "tests_passed": 2,
                        "tests_total": 2,
                    },
                },
            ],
            "run_conversation_result": {"total_tokens": 200, "api_calls": 18},
        }
    ]
    summary = mod.aggregate_records(
        records, pass_k_values=[5, 60], auc_max_k=60
    )
    sui = summary["final"]["success_user_iterations"]
    assert sui["n"] == 1
    assert sui["mean"] == 18.0
    text = mod.format_metrics_summary_text(summary)
    assert "pass@5_macro=" in text
    assert "pass@60_macro=" in text
    assert "auc_macro@1-60=" in text
    assert "auc_micro@1-60=" in text
    assert "final_macro=" in text
    assert "final_success_turn=18.00±0.00 (n=1, includes_verification)" in text
    # Order: pass@60 before auc before final
    i60 = text.index("pass@60_macro=")
    iauc = text.index("auc_macro@1-60=")
    ifinal = text.index("final_macro=")
    assert i60 < iauc < ifinal
