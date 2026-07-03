"""Tests for background review telemetry exposed to batch drivers."""

from __future__ import annotations

import json

import run_agent as run_agent_module
from run_agent import AIAgent


def _collect_telemetry(agent, review_messages, prior_snapshot, **kwargs):
    return AIAgent._collect_background_review_telemetry(
        agent, review_messages, prior_snapshot, **kwargs,
    )


def _tool_msg(tool_call_id, payload):
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(payload),
    }


def _assistant_tool_call(tool_call_id, name, arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


def test_collect_telemetry_skill_manage_create():
    snapshot = [{"role": "user", "content": "solve task"}]
    review_messages = list(snapshot) + [
        {"role": "user", "content": "<review prompt>"},
        _assistant_tool_call(
            "call_new",
            "skill_manage",
            {"action": "create", "name": "skillsbench-host-verification"},
        ),
        _tool_msg(
            "call_new",
            {
                "success": True,
                "message": "Skill 'skillsbench-host-verification' created.",
            },
        ),
        {"role": "assistant", "content": "Saved host verification workflow."},
    ]

    telemetry = _collect_telemetry(
        object.__new__(AIAgent),
        review_messages,
        snapshot,
        review_memory=False,
        review_skills=True,
    )

    assert telemetry["review_skills"] is True
    assert telemetry["review_memory"] is False
    assert telemetry["actions"] == [
        "Skill 'skillsbench-host-verification' created.",
    ]
    assert len(telemetry["tools"]) == 1
    tool = telemetry["tools"][0]
    assert tool["tool"] == "skill_manage"
    assert tool["success"] is True
    assert tool["arguments"]["action"] == "create"
    assert tool["arguments"]["name"] == "skillsbench-host-verification"
    assert telemetry["assistant_texts"] == ["Saved host verification workflow."]
    assert telemetry["nothing_to_save"] is False


def test_collect_telemetry_skips_prior_tool_messages():
    prior_payload = {"success": True, "message": "Cron job 'remind-me' created."}
    snapshot = [
        {"role": "user", "content": "hello"},
        _tool_msg("call_old", prior_payload),
    ]
    review_messages = list(snapshot) + [
        {"role": "user", "content": "<review prompt>"},
        {"role": "assistant", "content": "Nothing to save."},
    ]

    telemetry = _collect_telemetry(
        object.__new__(AIAgent),
        review_messages,
        snapshot,
        review_skills=True,
    )

    assert telemetry["tools"] == []
    assert telemetry["actions"] == []
    assert telemetry["nothing_to_save"] is True


def test_wait_for_background_review_includes_telemetry(monkeypatch):
    joined = {"called": False}

    class FakeThread:
        def __init__(self, *, target, daemon=None, name=None):
            self._target = target
            self._alive = True

        def start(self):
            pass

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            joined["called"] = True
            self._target()
            self._alive = False

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = [
                {"role": "user", "content": "task"},
                {"role": "user", "content": "review"},
                _assistant_tool_call(
                    "call_new",
                    "skill_manage",
                    {"action": "patch", "name": "skillsbench-host-verification"},
                ),
                _tool_msg(
                    "call_new",
                    {
                        "success": True,
                        "message": "Skill 'skillsbench-host-verification' updated.",
                    },
                ),
            ]

        def run_conversation(self, **kwargs):
            pass

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(run_agent_module.threading, "Thread", FakeThread)
    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)

    agent = object.__new__(AIAgent)
    agent.model = "fake-model"
    agent.platform = "skillsbench-batch"
    agent.provider = "openai"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = ""
    agent.session_id = "test-session"
    agent._credential_pool = None
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.skill_review_prompt_override = None
    agent.combined_review_prompt_override = None
    agent.background_review_callback = None
    agent.status_callback = None
    agent._hot_skill_pool = None
    agent._safe_print = lambda *_args, **_kwargs: None
    agent._background_review_result = None
    agent._background_review_thread = None

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "task"}],
        review_skills=True,
    )
    status = AIAgent.wait_for_background_review(agent, timeout=30.0)

    assert joined["called"] is True
    assert status["spawned"] is True
    assert status["completed"] is True
    assert status["timeout"] is False
    assert status["actions"] == [
        "Skill 'skillsbench-host-verification' updated.",
    ]
    assert status["telemetry"]["tools"][0]["tool"] == "skill_manage"


def test_spawn_background_review_uses_prompt_override(monkeypatch):
    captured = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            captured["user_message"] = kwargs.get("user_message")

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    class ImmediateThread:
        def __init__(self, *, target, daemon=None, name=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)

    agent = object.__new__(AIAgent)
    agent.model = "fake-model"
    agent.platform = "skillsbench-batch"
    agent.provider = "openai"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = ""
    agent.session_id = "test-session"
    agent._credential_pool = None
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.skill_review_prompt_override = "batch skill review override"
    agent.combined_review_prompt_override = "batch combined override"
    agent.background_review_callback = None
    agent.status_callback = None
    agent._hot_skill_pool = None
    agent._safe_print = lambda *_args, **_kwargs: None
    agent._background_review_result = None
    agent._background_review_thread = None

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_skills=True,
    )

    assert captured["user_message"] == "batch skill review override"

    captured.clear()
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
        review_skills=True,
    )

    assert captured["user_message"] == "batch combined override"
