"""Tests for benchmark/scripts/run_skillsbench_with_hermes.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "scripts"
    / "run_skillsbench_with_hermes.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_skillsbench_with_hermes", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_skillsbench_skill_review_prompt_includes_batch_rules():
    mod = _load_module()
    prompt = mod.build_skillsbench_skill_review_prompt("BASE")
    assert prompt.startswith("BASE")
    assert "skillsbench-batch" in prompt
    assert "skillsbench-host-verification" in prompt
    assert "3+ tool rounds" in prompt


def test_format_run_summary_includes_background_review_actions():
    mod = _load_module()
    envelope = {
        "duration_sec": 12.5,
        "evaluation": {
            "task_success": True,
            "tests_passed": 10,
            "tests_total": 12,
        },
        "run_conversation_result": {
            "api_calls": 25,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "estimated_cost_usd": 0.01,
            "completed": True,
            "interrupted": False,
        },
        "background_review": {
            "spawned": True,
            "completed": True,
            "timeout": False,
            "actions": ["Skill 'skillsbench-host-verification' created."],
            "telemetry": {
                "tools": [{"tool": "skill_manage", "success": True}],
            },
        },
    }
    summary = mod.format_run_summary(envelope)
    assert "eval_success=True" in summary
    assert "tests_passed=10" in summary
    assert "bg_review_actions=1" in summary
    assert "skillsbench-host-verification" in summary


def test_apply_hot_pool_cli_overrides_disable(monkeypatch):
    mod = _load_module()
    monkeypatch.setenv("HERMES_HOT_POOL_PERSIST", "1")
    monkeypatch.setenv("HERMES_HOT_POOL_PATH", "/tmp/pool.json")
    mod.apply_hot_pool_cli_overrides(hot_pool=False, hot_pool_persist=None)
    import os

    assert os.environ.get("HERMES_HOT_POOL_ENABLED") == "0"
    assert "HERMES_HOT_POOL_PERSIST" not in os.environ
    assert "HERMES_HOT_POOL_PATH" not in os.environ


def test_apply_hot_pool_cli_overrides_enable_with_persist(monkeypatch):
    mod = _load_module()
    mod.apply_hot_pool_cli_overrides(
        hot_pool=True,
        hot_pool_persist="benchmark/skillsbench_hot_pool.json",
    )
    import os

    assert os.environ.get("HERMES_HOT_POOL_ENABLED") == "1"
    assert os.environ.get("HERMES_HOT_POOL_PERSIST") == "1"
    assert os.environ["HERMES_HOT_POOL_PATH"].endswith("skillsbench_hot_pool.json")
