"""Tests for benchmark hot-pool outcome attribution helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "benchmark" / "scripts"


@pytest.fixture
def pool_cfg():
    return {
        "enabled": True,
        "max_entries": 9,
        "max_chars": 2000,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "eviction_policy": "oldest",
        "inject_on_turn": True,
        "skip_if_in_history": True,
        "history_lookback": 10,
        "hydrate_from_history": True,
        "hydrate_limit": 10,
        "use_hermes_hot_markers": True,
        "extract_sections": True,
        "fallback_extract": True,
    }


def _load_outcome_module():
    path = _SCRIPTS / "hermes_hot_pool_outcome.py"
    spec = importlib.util.spec_from_file_location("hermes_hot_pool_outcome", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_apply_benchmark_hot_pool_outcome_feedback_wires_complete_fn(pool_cfg):
    mod = _load_outcome_module()
    from agent.hot_skills import HotSkillPool

    pool = HotSkillPool(pool_cfg)
    pool.record(name="nav", content="", key_points=["open fridge first"], turn=1)
    pool.build_block(user_message="find apple", turn=1)

    class FakeAgent:
        _hot_pool_llm_judge_inflight = False
        _hot_skill_pool = pool
        calls: list = []

        def _complete_sidechannel_text(self, messages, max_tokens=512, reason=""):
            self.calls.append(
                {"messages": messages, "reason": reason, "max_tokens": max_tokens}
            )
            self._last_sidechannel_meta = {
                "reason": reason,
                "max_tokens": max_tokens,
                "finish_reason": "stop",
                "raw_chars": 20,
                "stripped_chars": 20,
                "raw_preview": '{"labels":{"0":"helpful"}}',
                "stripped_preview": '{"labels":{"0":"helpful"}}',
                "thinking_off": True,
                "retries": 0,
                "recovered_json": False,
            }
            return json.dumps({"labels": {"0": "helpful"}})

        def wait_for_background_review(self, timeout=60.0):
            return {"completed": True}

    agent = FakeAgent()
    summary = mod.apply_benchmark_hot_pool_outcome_feedback(
        agent,
        evaluation={"task_success": True, "reward": 1.0, "steps": 6},
        benchmark="appworld",
    )
    assert summary is not None
    assert summary["applied"] is True
    assert summary["attribution_source"] == "llm"
    assert agent.calls
    assert agent.calls[0]["reason"] == "hot_pool_outcome_feedback"
    assert agent.calls[0]["max_tokens"] == 2048
    of = pool.export_telemetry()["outcome_feedback"]
    assert of["judge_finish_reason"] == "stop"
    assert of["judge_thinking_off"] is True
    assert of["judge_max_tokens"] == 2048


def test_apply_benchmark_hot_pool_outcome_feedback_skips_when_disabled(pool_cfg):
    mod = _load_outcome_module()
    from agent.hot_skills import HotSkillPool

    cfg = dict(pool_cfg)
    cfg["outcome_feedback"] = False
    pool = HotSkillPool(cfg)

    class FakeAgent:
        _hot_pool_llm_judge_inflight = False
        _hot_skill_pool = pool

    summary = mod.apply_benchmark_hot_pool_outcome_feedback(
        FakeAgent(),
        evaluation={"task_success": True},
        benchmark="alfworld",
    )
    assert summary == {"applied": False, "skipped_reason": "disabled"}


def test_appworld_driver_uses_benchmark_outcome_helper():
    path = _SCRIPTS / "run_appworld_with_hermes.py"
    text = path.read_text(encoding="utf-8")
    assert "apply_benchmark_hot_pool_outcome_feedback" in text
    assert "hermes_hot_pool_outcome" in text
