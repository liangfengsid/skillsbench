"""Success@k / AUC for ALFWorld and AppWorld (env-/execute-step budgets)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "scripts"
    / "aggregate_skillsbench_runs.py"
)


def _load_mod():
    spec = importlib.util.spec_from_file_location("aggregate_skillsbench_runs", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_default_step_pass_k_values():
    mod = _load_mod()
    assert mod.default_step_pass_k_values(50) == [1, 5, 10, 30, 50]
    assert mod.default_step_pass_k_values(40) == [1, 5, 10, 30, 40]
    assert mod.default_step_pass_k_values(10) == [1, 5, 10]


def test_alfworld_success_within_step_budget_and_auc():
    """Success@k uses env steps, not Hermes api_calls, for ALFWorld rows."""
    mod = _load_mod()
    records = [
        {
            "schema": "alfworld.hermes_run.v1",
            "benchmark": "alfworld",
            "task_id": "g1",
            "skillsbench_task_id": "g1",
            "evaluation": {
                "task_success": True,
                "reward": 1.0,
                "steps": 8,
            },
            "run_conversation_result": {
                "api_calls": 40,  # deliberately larger than steps
                "total_tokens": 1000,
            },
            "metrics": {
                "final": {
                    "task_success": True,
                    "reward": 1.0,
                    "api_calls": 40,
                    "tokens": {"total": 1000},
                }
            },
        },
        {
            "schema": "alfworld.hermes_run.v1",
            "benchmark": "alfworld",
            "task_id": "g2",
            "skillsbench_task_id": "g2",
            "evaluation": {
                "task_success": True,
                "reward": 1.0,
                "steps": 25,
            },
            "run_conversation_result": {"api_calls": 25, "total_tokens": 2000},
        },
        {
            "schema": "alfworld.hermes_run.v1",
            "benchmark": "alfworld",
            "task_id": "g3",
            "skillsbench_task_id": "g3",
            "evaluation": {
                "task_success": False,
                "reward": 0.0,
                "steps": 50,
            },
            "run_conversation_result": {"api_calls": 50, "total_tokens": 3000},
        },
    ]
    summary = mod.aggregate_records(
        records,
        pass_k_values=[5, 10, 30, 50],
        max_user_iterations=50,
        auc_max_k=50,
        include_final_in_pass_k=True,
    )
    # Within 5 steps: none
    assert summary["pass_k"]["5"]["macro_success_rate"] == 0.0
    # Within 10: g1 only
    assert summary["pass_k"]["10"]["macro_success_rate"] == 1 / 3
    assert summary["pass_k"]["10"]["tasks_passed_at_turn"] == 1
    # Within 30: g1 + g2
    assert summary["pass_k"]["30"]["macro_success_rate"] == 2 / 3
    # Within 50: same (g3 failed)
    assert summary["pass_k"]["50"]["macro_success_rate"] == 2 / 3
    assert summary["final"]["macro_success_rate"] == 2 / 3
    assert summary["auc"]["macro_success_rate"] is not None
    assert summary["auc"]["k_max"] == 50
    # Cost-to-succeed uses step budget (8 and 25), not inflated api_calls
    cts = summary["cost_to_succeed"]["user_iterations"]
    assert cts["n"] == 2
    assert abs(cts["mean"] - (8 + 25) / 2) < 1e-6


def test_appworld_legacy_success_field_and_steps_taken():
    """Older AppWorld rows use evaluation.success and steps_taken only."""
    mod = _load_mod()
    records = [
        {
            "schema": "appworld.hermes_run.v1",
            "appworld_task_id": "t1",
            "task_id": "t1",
            "skillsbench_task_id": "t1",
            "steps_taken": 12,
            "max_steps": 40,
            "evaluation": {
                "success": True,
                "pass_count": 3,
                "num_tests": 4,
                "tests_passed": 3,
                "tests_total": 4,
            },
            "hermes_stats": {"api_calls": 12, "total_tokens": 500},
        },
        {
            "schema": "appworld.hermes_run.v1",
            "benchmark": "appworld",
            "appworld_task_id": "t2",
            "task_id": "t2",
            "skillsbench_task_id": "t2",
            "steps_taken": 35,
            "evaluation": {
                "success": False,
                "task_success": False,
                "tests_passed": 0,
                "tests_total": 4,
            },
            "run_conversation_result": {"api_calls": 35, "total_tokens": 800},
        },
    ]
    summary = mod.aggregate_records(
        records,
        pass_k_values=[10, 20, 40],
        max_user_iterations=40,
        auc_max_k=40,
    )
    assert summary["pass_k"]["10"]["macro_success_rate"] == 0.0
    assert summary["pass_k"]["20"]["macro_success_rate"] == 0.5
    assert summary["pass_k"]["40"]["macro_success_rate"] == 0.5
    assert summary["pass_k"]["20"]["micro_success_rate"] == 3 / 8
    assert summary["final"]["macro_success_rate"] == 0.5
    text = mod.format_summary_text(summary)
    assert "pass@20_macro=0.5000" in text
    assert "auc_macro@1-40=" in text


def test_appworld_eval_row_merges_steps_from_earlier_run():
    """Later hermes_eval rows must not drop steps_taken from hermes_run."""
    mod = _load_mod()
    records = [
        {
            "schema": "appworld.hermes_run.v1",
            "appworld_task_id": "t1",
            "task_id": "t1",
            "skillsbench_task_id": "t1",
            "ts_end_iso": "2026-01-01T00:00:00+00:00",
            "steps_taken": 16,
            "evaluation": {"success": True, "tests_passed": 2, "tests_total": 2},
            "hermes_stats": {"api_calls": 16, "total_tokens": 100},
        },
        {
            "schema": "appworld.hermes_eval.v1",
            "appworld_task_id": "t1",
            "task_id": "t1",
            "skillsbench_task_id": "t1",
            "ts_end_iso": "2026-01-01T01:00:00+00:00",
            "evaluation": {
                "success": True,
                "task_success": True,
                "tests_passed": 2,
                "tests_total": 2,
            },
        },
    ]
    summary = mod.aggregate_records(
        records,
        pass_k_values=[10, 20, 40],
        max_user_iterations=40,
        auc_max_k=40,
    )
    assert summary["tasks"] == 1
    assert summary["pass_k"]["10"]["macro_success_rate"] == 0.0
    assert summary["pass_k"]["20"]["macro_success_rate"] == 1.0
    assert summary["pass_k"]["40"]["macro_success_rate"] == 1.0
