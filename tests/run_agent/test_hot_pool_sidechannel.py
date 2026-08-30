"""Side-channel hot-pool LLM eviction uses the agent's existing client."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    with (
        patch(
            "run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


def test_agent_wires_hot_pool_eviction_judge(agent):
    assert agent._hot_skill_pool.eviction_judge == agent._hot_pool_eviction_judge


def test_complete_sidechannel_chat_completions_does_not_pass_tools(agent):
    transport = MagicMock()
    transport.normalize_response.return_value = SimpleNamespace(
        content='{"keep": ["1"]}'
    )
    agent.api_mode = "chat_completions"
    agent.model = "test-model"
    agent._get_transport = MagicMock(return_value=transport)
    agent._ensure_primary_openai_client = MagicMock(return_value=agent.client)
    agent.client.chat.completions.create.return_value = SimpleNamespace()

    text = agent._complete_sidechannel_text(
        [{"role": "user", "content": "pick"}],
        max_tokens=64,
        reason="hot_pool_eviction",
    )
    assert text == '{"keep": ["1"]}'
    kwargs = agent.client.chat.completions.create.call_args.kwargs
    assert "tools" not in kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"][0]["content"] == "pick"
    # Fixture base_url is OpenRouter — thinking-off uses reasoning.enabled.
    assert kwargs["extra_body"]["reasoning"] == {"enabled": False}


def test_hot_pool_eviction_judge_parses_sidechannel_json(agent):
    agent._complete_sidechannel_text = MagicMock(return_value='{"keep": ["1"]}')
    items = [
        {"id": "0", "skill": "alpha", "point": "A"},
        {"id": "1", "skill": "github-auth", "point": "G"},
    ]
    keep = agent._hot_pool_eviction_judge(items, 1, "fix oauth")
    assert keep == ["1"]
    agent._complete_sidechannel_text.assert_called_once()
    msgs = agent._complete_sidechannel_text.call_args.args[0]
    assert msgs[0]["role"] == "system"
    assert "github-auth" in msgs[1]["content"]


def test_hot_pool_eviction_judge_returns_none_on_empty_completion(agent):
    agent._complete_sidechannel_text = MagicMock(return_value="")
    items = [{"id": "0", "skill": "alpha", "point": "A"}]
    assert agent._hot_pool_eviction_judge(items, 1, "") is None


def test_sidechannel_qwen_sends_thinking_off(agent):
    transport = MagicMock()
    transport.normalize_response.return_value = SimpleNamespace(
        content='{"labels": {"0": "helpful"}}',
        finish_reason="stop",
    )
    agent.api_mode = "chat_completions"
    agent.model = "Qwen/Qwen3.6-27B"
    agent.base_url = "http://127.0.0.1:8000/v1"
    agent._base_url_lower = "http://127.0.0.1:8000/v1"
    agent._is_openrouter_url = MagicMock(return_value=False)
    agent.provider = "custom"
    agent._get_transport = MagicMock(return_value=transport)
    agent._ensure_primary_openai_client = MagicMock(return_value=agent.client)
    agent.client.chat.completions.create.return_value = SimpleNamespace()

    text = agent._complete_sidechannel_text(
        [{"role": "user", "content": "label"}],
        max_tokens=2048,
        reason="hot_pool_outcome_feedback",
    )
    assert text == '{"labels": {"0": "helpful"}}'
    extra = agent.client.chat.completions.create.call_args.kwargs["extra_body"]
    assert extra["chat_template_kwargs"]["enable_thinking"] is False
    assert extra["enable_thinking"] is False
    assert extra["think"] is False
    meta = agent._last_sidechannel_meta
    assert meta["finish_reason"] == "stop"
    assert meta["thinking_off"] is True
    assert meta["max_tokens"] == 2048
    assert meta["retries"] == 0


def test_sidechannel_openai_omits_qwen_thinking_kwargs(agent):
    transport = MagicMock()
    transport.normalize_response.return_value = SimpleNamespace(
        content='{"keep": ["0"]}',
        finish_reason="stop",
    )
    agent.api_mode = "chat_completions"
    agent.model = "gpt-4o"
    agent.base_url = "https://api.openai.com/v1"
    agent._base_url_lower = "https://api.openai.com/v1"
    agent._is_openrouter_url = MagicMock(return_value=False)
    agent.provider = "openai"
    agent._get_transport = MagicMock(return_value=transport)
    agent._ensure_primary_openai_client = MagicMock(return_value=agent.client)
    agent.client.chat.completions.create.return_value = SimpleNamespace()

    agent._complete_sidechannel_text(
        [{"role": "user", "content": "pick"}],
        max_tokens=64,
        reason="hot_pool_eviction",
    )
    kwargs = agent.client.chat.completions.create.call_args.kwargs
    assert "extra_body" not in kwargs


def test_sidechannel_think_only_returns_empty_and_retries(agent):
    transport = MagicMock()
    transport.normalize_response.return_value = SimpleNamespace(
        content="<think>only reasoning no json</think>",
        finish_reason="length",
    )
    agent.api_mode = "chat_completions"
    agent.model = "Qwen/Qwen3.6-27B"
    agent._is_openrouter_url = MagicMock(return_value=False)
    agent.provider = "custom"
    agent._get_transport = MagicMock(return_value=transport)
    agent._ensure_primary_openai_client = MagicMock(return_value=agent.client)
    agent.client.chat.completions.create.return_value = SimpleNamespace()

    text = agent._complete_sidechannel_text(
        [{"role": "user", "content": "label"}],
        max_tokens=512,
        reason="hot_pool_outcome_feedback",
    )
    assert text == ""
    assert agent.client.chat.completions.create.call_count == 2
    meta = agent._last_sidechannel_meta
    assert meta["finish_reason"] == "length"
    assert meta["retries"] == 1
    assert meta["stripped_chars"] == 0
    assert meta["raw_chars"] > 0


def test_sidechannel_recovers_json_inside_think(agent):
    transport = MagicMock()
    transport.normalize_response.return_value = SimpleNamespace(
        content='<think>{"labels": {"0": "harmful"}}</think>',
        finish_reason="stop",
    )
    agent.api_mode = "chat_completions"
    agent.model = "Qwen/Qwen3.6-27B"
    agent._is_openrouter_url = MagicMock(return_value=False)
    agent.provider = "custom"
    agent._get_transport = MagicMock(return_value=transport)
    agent._ensure_primary_openai_client = MagicMock(return_value=agent.client)
    agent.client.chat.completions.create.return_value = SimpleNamespace()

    text = agent._complete_sidechannel_text(
        [{"role": "user", "content": "label"}],
        max_tokens=2048,
        reason="hot_pool_outcome_feedback",
    )
    assert text == '{"labels": {"0": "harmful"}}'
    assert agent._last_sidechannel_meta["recovered_json"] is True
    assert agent.client.chat.completions.create.call_count == 1
