"""Regression tests for memory provider selection during AIAgent init."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_blank_memory_provider_does_not_auto_enable_honcho():
    """Blank memory.provider should remain opt-out even if Honcho fallback looks configured."""
    cfg = {"memory": {"provider": ""}, "agent": {}}
    honcho_cfg = SimpleNamespace(enabled=True, api_key="stale-key", base_url=None)

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.save_config") as save_config,
        patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            return_value=honcho_cfg,
        ) as from_global_config,
        patch("plugins.memory.load_memory_provider") as load_memory_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    assert agent._memory_manager is None
    from_global_config.assert_not_called()
    load_memory_provider.assert_not_called()
    save_config.assert_not_called()


def test_skip_memory_does_not_load_provider(monkeypatch):
    monkeypatch.delenv("HERMES_AMEM_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_DC_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_MEMORY_PROVIDER", raising=False)
    cfg = {"memory": {"provider": "honcho"}, "agent": {}}
    load_memory_provider = MagicMock()

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.save_config"),
        patch("plugins.memory.load_memory_provider", load_memory_provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent._memory_manager is None
    load_memory_provider.assert_not_called()


def test_amem_loads_when_skip_memory_and_env(monkeypatch):
    monkeypatch.setenv("HERMES_AMEM_ENABLED", "1")
    monkeypatch.delenv("HERMES_DC_ENABLED", raising=False)
    cfg = {"memory": {"provider": ""}, "agent": {}}
    fake = MagicMock()
    fake.is_available.return_value = True
    fake.get_tool_schemas.return_value = []
    load_memory_provider = MagicMock(return_value=fake)

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.save_config"),
        patch("plugins.memory.load_memory_provider", load_memory_provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    load_memory_provider.assert_called_once_with("amem")
    assert agent._memory_manager is not None
    assert fake in agent._memory_manager.providers


def test_dc_loads_when_skip_memory_and_env(monkeypatch):
    monkeypatch.delenv("HERMES_AMEM_ENABLED", raising=False)
    monkeypatch.setenv("HERMES_DC_ENABLED", "1")
    cfg = {"memory": {"provider": ""}, "agent": {}}
    fake = MagicMock()
    fake.is_available.return_value = True
    fake.get_tool_schemas.return_value = []
    load_memory_provider = MagicMock(return_value=fake)

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.save_config"),
        patch("plugins.memory.load_memory_provider", load_memory_provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    load_memory_provider.assert_called_once_with("dcheatsheet")
    assert agent._memory_manager is not None
    assert fake in agent._memory_manager.providers


