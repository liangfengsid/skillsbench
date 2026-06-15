"""In-process Hermes ``AIAgent`` calls for HLE benchmark scripts (no tools, stateless)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def _ensure_hermes_importable() -> None:
    """
    Make ``from run_agent import AIAgent`` work when:

    - hermes-agent is installed in the current environment, or
    - this file lives at ``<repo>/benchmark/scripts/hle_hermes_inprocess.py`` (prepend repo root), or
    - ``HERMES_AGENT_REPO`` points at a checkout, or
    - a sibling repo exists at ``~/git/hermes-agent``.
    """
    try:
        import run_agent  # noqa: F401

        return
    except Exception:
        pass

    repo_from_script = Path(__file__).resolve().parents[2]
    if repo_from_script.is_dir() and (repo_from_script / "run_agent.py").is_file():
        rs = str(repo_from_script)
        if rs not in sys.path:
            sys.path.insert(0, rs)
        try:
            import run_agent  # noqa: F401

            return
        except Exception:
            pass

    candidate = os.environ.get("HERMES_AGENT_REPO", os.path.expanduser("~/git/hermes-agent"))
    if candidate and os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)


def hermes_chat(
    *,
    model: Optional[str] = None,
    user_message: str,
    system_message: Optional[str] = None,
    max_iterations: int = 1,
) -> str:
    _ensure_hermes_importable()
    from run_agent import AIAgent  # type: ignore

    kwargs = dict(
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=[],
        max_iterations=max_iterations,
    )
    if model:
        kwargs["model"] = model
    agent = AIAgent(**kwargs)
    result = agent.run_conversation(
        user_message=user_message,
        system_message=system_message,
    )
    return (result.get("final_response") or "").strip()
