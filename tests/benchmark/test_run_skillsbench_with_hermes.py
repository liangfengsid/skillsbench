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
    assert "skill_manage" in prompt
    assert "prefer saving" in prompt.lower()
    assert "skillsbench-host-verification" not in prompt
    assert "mega-skill" in prompt.lower() or "catch-all" in prompt.lower()
    assert "held-out" in prompt.lower() or "abstract" in prompt.lower()
    assert "pre-existing solution" in prompt.lower() or "solution/" in prompt
    combined = mod.build_skillsbench_combined_review_prompt("BASE")
    assert "skillsbench" in combined.lower()
    assert len(mod.SKILLSBENCH_BATCH_SKILL_REVIEW_APPENDIX) < 1100


def test_build_user_message_includes_host_verification_when_eval_on():
    mod = _load_module()
    msg = mod.build_user_message("/tmp/tasks", "foo", evaluate_after_run=True)
    assert "foo" in msg
    assert "Host verification" in msg
    assert "Do NOT claim" in msg
    assert "solution/" in msg
    short = mod.build_user_message("/tmp/tasks", "foo", evaluate_after_run=False)
    assert "Host verification" not in short


def test_build_host_eval_feedback_message_summarizes_failure():
    mod = _load_module()
    text = mod.build_host_eval_feedback_message(
        {
            "task_success": False,
            "tests_passed": 2,
            "tests_total": 5,
            "tests_failed": 3,
            "test_cases": [
                {"nodeid": "tests/test_outputs.py::test_a", "outcome": "failed"},
            ],
            "stderr_tail": "AssertionError: missing report.json",
        }
    )
    assert "Host verification failed" in text
    assert "2/5" in text
    assert "test_a" in text
    assert "report.json" in text


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


def test_resolve_model_id_prefers_cli_arg():
    mod = _load_module()
    assert mod.resolve_model_id("  Qwen/Qwen3.6-27B  ") == "Qwen/Qwen3.6-27B"


def test_resolve_model_id_falls_back_to_config(monkeypatch):
    mod = _load_module()

    def _fake_load_config():
        return {"model": {"default": "Qwen/Qwen3.6-27B", "provider": "qwen-local"}}

    monkeypatch.setattr("hermes_cli.config.load_config", _fake_load_config)
    assert mod.resolve_model_id("") == "Qwen/Qwen3.6-27B"
    assert mod.resolve_model_id(None) == "Qwen/Qwen3.6-27B"


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
