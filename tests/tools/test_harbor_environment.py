"""Harbor trial backend: bind/unbind + exec forwarding (no Docker / no Harbor SDK)."""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from tools.environments.harbor_bridge import (
    bind_harbor_session,
    get_bound_harbor_session,
    harbor_session_bound,
    unbind_harbor_session,
)


class _ExecResult:
    def __init__(self, stdout: str = "", stderr: str = "", return_code: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code


class _FakeHarborEnv:
    def __init__(self):
        self.commands: list[str] = []

    async def exec(self, command, cwd=None, timeout_sec=None, user=None, **kwargs):
        self.commands.append(command)
        return _ExecResult(stdout="ok-from-harbor\n", return_code=0)


@pytest.fixture
def harbor_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


@pytest.fixture(autouse=True)
def _unbind_harbor():
    unbind_harbor_session()
    yield
    unbind_harbor_session()


def test_unbound_session_raises():
    with pytest.raises(RuntimeError, match="active Harbor trial"):
        get_bound_harbor_session()
    assert harbor_session_bound() is False


def test_bind_and_unbind(harbor_loop):
    fake = _FakeHarborEnv()
    bind_harbor_session(environment=fake, loop=harbor_loop)
    assert harbor_session_bound() is True
    env, loop = get_bound_harbor_session()
    assert env is fake
    assert loop is harbor_loop
    unbind_harbor_session()
    assert harbor_session_bound() is False


def test_harbor_environment_execute(harbor_loop):
    from tools.environments.harbor import HarborEnvironment

    fake = _FakeHarborEnv()
    bind_harbor_session(environment=fake, loop=harbor_loop)
    env = HarborEnvironment(cwd="/app", timeout=30)
    result = env.execute("echo hello")
    assert result["returncode"] == 0
    assert "ok-from-harbor" in result["output"]
    assert fake.commands
    env.cleanup()


def test_harbor_environment_requires_bound_session():
    from tools.environments.harbor import HarborEnvironment

    with pytest.raises(RuntimeError, match="active Harbor trial"):
        HarborEnvironment(cwd="/app")


def test_check_terminal_requirements_harbor(monkeypatch, caplog):
    import importlib

    terminal_tool_module = importlib.import_module("tools.terminal_tool")
    monkeypatch.setenv("TERMINAL_ENV", "harbor")
    unbind_harbor_session()
    with caplog.at_level(logging.ERROR):
        assert terminal_tool_module.check_terminal_requirements() is False
    assert any("Harbor trial" in rec.getMessage() for rec in caplog.records)

    fake = _FakeHarborEnv()
    loop = asyncio.new_event_loop()
    bind_harbor_session(environment=fake, loop=loop)
    try:
        assert terminal_tool_module.check_terminal_requirements() is True
    finally:
        unbind_harbor_session()
        loop.close()
