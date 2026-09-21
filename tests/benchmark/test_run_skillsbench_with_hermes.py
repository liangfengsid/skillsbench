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
    assert "skillsbench-host-grading" in prompt
    assert "mega-skill" in prompt.lower() or "catch-all" in prompt.lower()
    assert "held-out" in prompt.lower() or "abstract" in prompt.lower()
    assert "pre-existing solution" in prompt.lower() or "solution/" in prompt
    combined = mod.build_skillsbench_combined_review_prompt("BASE")
    assert "skillsbench" in combined.lower()
    assert "skillsbench-host-grading" in combined
    assert len(mod.SKILLSBENCH_BATCH_SKILL_REVIEW_APPENDIX) < 1200


def test_build_user_message_includes_host_grading_meta_when_eval_on():
    mod = _load_module()
    msg = mod.build_user_message("/tmp/tasks", "foo", evaluate_after_run=True)
    assert "foo" in msg
    assert "Driver grading" in msg
    assert "discover & encode" in msg
    assert "skill_manage" in msg
    assert "skillsbench-host-grading" in msg
    assert "## Common Pitfalls" in msg
    assert "## Best Practices" in msg
    assert "hot skill pool" in msg.lower() or "hot-pool" in msg.lower()
    assert "short bullet" in msg.lower() or "short bullets" in msg.lower()
    assert "NEVER" in msg and "ALWAYS" in msg
    # Must not spoil the old privileged appendix wording.
    assert "Do NOT claim the task is complete" not in msg
    assert "a host-side pytest verifier runs after your turn" not in msg
    short = mod.build_user_message("/tmp/tasks", "foo", evaluate_after_run=False)
    assert "Driver grading" not in short
    # Alias still points at the meta appendix for older callers.
    assert mod.SKILLSBENCH_HOST_VERIFICATION_APPENDIX is (
        mod.SKILLSBENCH_HOST_GRADING_META_APPENDIX
    )


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


def test_resolve_provider_name_for_model_explicit_and_catalog():
    mod = _load_module()
    cfg = {
        "model": {"default": "Qwen/Qwen3.6-27B", "provider": "qwen-local"},
        "providers": {
            "qwen-31": {
                "base_url": "http://10.50.0.31:8090/v1",
                "default_model": "Qwen/Qwen3.6-27B-n31",
                "models": {"Qwen/Qwen3.6-27B-n31": {"context_length": 131072}},
            },
            "qwen-local": {
                "base_url": "http://127.0.0.1:8090/v1",
                "default_model": "Qwen/Qwen3.6-27B",
                "models": {"Qwen/Qwen3.6-27B": {"context_length": 131072}},
            },
        },
    }
    assert (
        mod.resolve_provider_name_for_model(
            "Qwen/Qwen3.6-27B-n31", provider="qwen-31", config=cfg
        )
        == "qwen-31"
    )
    # Unique catalog match auto-selects without rewriting model.provider.
    assert (
        mod.resolve_provider_name_for_model("Qwen/Qwen3.6-27B-n31", config=cfg)
        == "qwen-31"
    )
    # Default model stays on default provider.
    assert (
        mod.resolve_provider_name_for_model("Qwen/Qwen3.6-27B", config=cfg)
        == "qwen-local"
    )
    # Unknown model falls back to model.provider.
    assert (
        mod.resolve_provider_name_for_model("some/other-model", config=cfg)
        == "qwen-local"
    )


def test_resolve_provider_name_for_model_ambiguous_requires_flag():
    mod = _load_module()
    cfg = {
        "model": {"provider": "primary"},
        "providers": {
            "a": {"models": {"shared/model": {}}},
            "b": {"models": {"shared/model": {}}},
        },
    }
    try:
        mod.resolve_provider_name_for_model("shared/model", config=cfg)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "--provider" in str(exc)


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


def test_apply_hot_pool_cli_overrides_scope_and_attribution(monkeypatch):
    mod = _load_module()
    monkeypatch.setenv("HERMES_HOT_POOL_INJECT_RETRIEVE", "1")
    monkeypatch.setenv("HERMES_HOT_POOL_OUTCOME_FEEDBACK", "1")
    monkeypatch.setenv("HERMES_HOT_POOL_INJECT_FILTER_UTILITIES", "1")
    mod.apply_hot_pool_cli_overrides(
        hot_pool=True,
        hot_pool_persist=None,
        inject_retrieve=False,
        outcome_feedback=False,
        inject_filter_utilities=False,
    )
    import os

    assert os.environ.get("HERMES_HOT_POOL_INJECT_RETRIEVE") == "0"
    assert os.environ.get("HERMES_HOT_POOL_OUTCOME_FEEDBACK") == "0"
    assert os.environ.get("HERMES_HOT_POOL_INJECT_FILTER_UTILITIES") == "0"


def test_build_batch_metrics_summary_matches_aggregator(tmp_path):
    """Batch summary uses full JSONL + same kwargs as aggregate_skillsbench_runs."""
    import json

    mod = _load_module()
    from aggregate_skillsbench_runs import aggregate_records, format_summary_text

    older = {
        "schema": "skillsbench.hermes_run.v1",
        "skillsbench_task_id": "task-a",
        "split_part": "test",
        "ts_end_iso": "2026-01-01T00:00:00Z",
        "pass_at_turn": {
            "1": {
                "turn": 1,
                "evaluation": {
                    "task_success": True,
                    "tests_passed": 2,
                    "tests_total": 2,
                },
                "total_tokens": 10,
                "api_calls": 1,
            }
        },
        "evaluation": {
            "task_success": True,
            "tests_passed": 2,
            "tests_total": 2,
        },
        "run_conversation_result": {"api_calls": 5, "total_tokens": 100},
        "duration_sec": 12.0,
    }
    newer = {
        "schema": "skillsbench.hermes_run.v1",
        "skillsbench_task_id": "task-b",
        "split_part": "test",
        "ts_end_iso": "2026-01-01T01:00:00Z",
        "pass_at_turn": {
            "1": {
                "turn": 1,
                "evaluation": {
                    "task_success": False,
                    "tests_passed": 0,
                    "tests_total": 2,
                },
                "total_tokens": 20,
                "api_calls": 1,
            }
        },
        "evaluation": {
            "task_success": True,
            "tests_passed": 2,
            "tests_total": 2,
        },
        "run_conversation_result": {"api_calls": 8, "total_tokens": 200},
        "duration_sec": 30.0,
    }
    jsonl = tmp_path / "runs.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(r) for r in (older, newer)) + "\n",
        encoding="utf-8",
    )

    # This-run envelopes omit the resume-skipped task-a; full log must still count it.
    summary = mod.build_batch_metrics_summary(
        batch_envelopes=[newer],
        log_path=jsonl,
        pass_k_values=[1, 60],
        split_part="test",
        max_user_iterations=60,
        include_final_in_pass_k=False,
        auc_max_k=60,
    )
    assert summary is not None
    assert summary["tasks"] == 2
    assert summary["pass_k"]["1"]["tasks_passed_at_turn"] == 1
    assert summary["final"]["tasks_passed"] == 2
    assert summary["filters"]["include_final_in_pass_k"] is False
    assert summary["filters"]["auc_max_k"] == 60
    assert summary["auc"]["k_max"] == 60

    expected = aggregate_records(
        [older, newer],
        pass_k_values=[1, 60],
        split_part="test",
        max_user_iterations=60,
        include_final_in_pass_k=False,
        auc_max_k=60,
    )
    text = format_summary_text({k: v for k, v in summary.items() if k != "_summary_source"})
    assert text == format_summary_text(expected)
    assert "pass@1_macro=" in text
    assert "auc_macro@1-60=" in text
    i1 = text.index("pass@1_macro=")
    iauc = text.index("auc_macro@1-60=")
    ifinal = text.index("final_macro=")
    assert i1 < iauc < ifinal
