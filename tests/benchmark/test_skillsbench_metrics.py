"""Tests for benchmark/scripts/skillsbench_metrics.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "scripts"
    / "skillsbench_metrics.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("skillsbench_metrics", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_envelope_metrics_final_and_pass_at_turn():
    mod = _load_module()
    envelope = {
        "duration_sec": 120.5,
        "evaluation": {
            "task_success": True,
            "reward": 1.0,
            "tests_passed": 10,
            "tests_failed": 0,
            "tests_total": 10,
        },
        "pass_at_turn": {
            "5": {
                "turn": 5,
                "evaluation": {
                    "task_success": False,
                    "reward": 0.5,
                    "tests_passed": 5,
                    "tests_total": 10,
                },
                "api_calls": 5,
                "tokens": {"total": 1000, "input": 800, "output": 200},
            },
        },
        "run_conversation_result": {
            "completed": True,
            "interrupted": False,
            "api_calls": 20,
            "input_tokens": 5000,
            "output_tokens": 1000,
            "total_tokens": 6000,
            "cache_read_tokens": 2000,
            "reasoning_tokens": 300,
            "estimated_cost_usd": 0.42,
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "terminal"}},
                        {"function": {"name": "read_file"}},
                    ],
                },
            ],
        },
        "hot_pool_enabled": True,
        "background_review": {"spawned": True, "completed": True, "actions": ["x"]},
    }
    metrics = mod.build_envelope_metrics(envelope)
    assert metrics["final"]["task_success"] is True
    assert metrics["final"]["tool_rounds"] == 2
    assert metrics["final"]["tokens"]["total"] == 6000
    assert metrics["pass_at_turn"]["5"]["micro_pass_rate"] == 0.5
    assert metrics["background_review"]["action_count"] == 1
