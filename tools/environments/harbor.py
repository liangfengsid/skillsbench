"""Harbor trial execution backend for Hermes terminal/file tools.

Does **not** start its own container. Harbor's orchestrator already created
the task sandbox; this backend forwards ``execute()`` to
``harbor.environments.base.BaseEnvironment.exec`` on the trial event loop.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from typing import Any, Optional

from tools.environments.base import BaseEnvironment, _ThreadedProcessHandle
from tools.environments.harbor_bridge import get_bound_harbor_session

logger = logging.getLogger(__name__)


def _result_output_and_code(result: Any) -> tuple[str, int]:
    stdout = getattr(result, "stdout", None) or ""
    stderr = getattr(result, "stderr", None) or ""
    code = getattr(result, "return_code", None)
    if code is None:
        code = getattr(result, "exit_code", 1)
    if stderr and stderr not in stdout:
        out = f"{stdout}\n{stderr}" if stdout else stderr
    else:
        out = stdout
    try:
        return out, int(code)
    except (TypeError, ValueError):
        return out, 1


class HarborEnvironment(BaseEnvironment):
    """Forward Hermes shell/file ops into the live Harbor trial sandbox."""

    _stdin_mode = "heredoc"

    def __init__(
        self,
        cwd: str = "/app",
        timeout: int = 180,
        harbor_env: Any = None,
        loop: Any = None,
        **kwargs,
    ):
        if cwd in ("~", ""):
            cwd = "/app"
        super().__init__(cwd=cwd, timeout=timeout)
        if harbor_env is None or loop is None:
            harbor_env, loop = get_bound_harbor_session()
        self._harbor = harbor_env
        self._loop = loop
        self.init_session()

    def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int = 120,
        stdin_data: str | None = None,
    ):
        harbor = self._harbor
        loop = self._loop
        shell = "bash -l -c" if login else "bash -c"
        command = f"{shell} {shlex.quote(cmd_string)}"

        def exec_fn() -> tuple[str, int]:
            if loop.is_closed():
                raise RuntimeError("Harbor event loop is closed")
            coro = harbor.exec(
                command,
                cwd=None,
                timeout_sec=int(timeout) if timeout else None,
            )
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            try:
                result = fut.result(timeout=(timeout or 120) + 30)
            except TimeoutError:
                fut.cancel()
                return ("Harbor exec timed out", 124)
            return _result_output_and_code(result)

        return _ThreadedProcessHandle(exec_fn)

    def cleanup(self):
        """Harbor owns the sandbox; only drop local session state."""
        self._snapshot_ready = False
