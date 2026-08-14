"""Bridge a live Harbor ``BaseEnvironment`` into Hermes terminal tools.

Harbor owns sandbox lifecycle (Docker/Modal/Daytona/…). Hermes only execs
into the trial environment that Harbor already started. Bind the session on
the worker thread that runs ``AIAgent`` before any terminal/file tool calls.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, Tuple

_tls = threading.local()


def bind_harbor_session(*, environment: Any, loop: Any) -> None:
    """Attach Harbor's async environment + event loop to this thread."""
    _tls.environment = environment
    _tls.loop = loop


def unbind_harbor_session() -> None:
    _tls.environment = None
    _tls.loop = None


def get_bound_harbor_session() -> Tuple[Any, Any]:
    env = getattr(_tls, "environment", None)
    loop = getattr(_tls, "loop", None)
    if env is None or loop is None:
        raise RuntimeError(
            "TERMINAL_ENV=harbor requires an active Harbor trial. "
            "Run via benchmark/scripts/run_terminalbench_with_harbor.py "
            "(do not set terminal.backend=harbor in config.yaml)."
        )
    return env, loop


def harbor_session_bound() -> bool:
    return (
        getattr(_tls, "environment", None) is not None
        and getattr(_tls, "loop", None) is not None
    )
