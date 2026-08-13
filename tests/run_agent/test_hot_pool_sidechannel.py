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
