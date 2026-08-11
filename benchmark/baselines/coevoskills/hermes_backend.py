"""Thin Hermes AIAgent wrapper — no modifications to Hermes core."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def ensure_hermes_on_path(hermes_root: Path) -> None:
    root = str(hermes_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_model(model: str, hermes_root: Path) -> str:
    if model and model.strip():
        return model.strip()
    ensure_hermes_on_path(hermes_root)
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        m = (cfg.get("model") or {}).get("default") or ""
        if isinstance(m, str) and m.strip():
            return m.strip()
        if isinstance(cfg.get("model"), str) and cfg["model"].strip():
            return cfg["model"].strip()
    except Exception:
        pass
    return ""


def make_agent(
    *,
    hermes_root: Path,
    model: str,
    max_iterations: int,
    skip_context_files: bool = True,
    skip_memory: bool = True,
    quiet_mode: bool = False,
    enabled_toolsets: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    print_fn=None,
) -> Any:
    ensure_hermes_on_path(hermes_root)
    from run_agent import AIAgent

    kwargs: Dict[str, Any] = {
        "model": resolve_model(model, hermes_root),
        "max_iterations": max_iterations,
        "quiet_mode": quiet_mode,
        "skip_context_files": skip_context_files,
        "skip_memory": skip_memory,
        "platform": "cli",
    }
    if session_id:
        kwargs["session_id"] = session_id
    if enabled_toolsets is not None:
        kwargs["enabled_toolsets"] = enabled_toolsets
    agent = AIAgent(**kwargs)
    if print_fn is not None:
        agent._print_fn = print_fn
    return agent


def run_turn(
    agent: Any,
    *,
    user_message: str,
    system_message: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run one agent turn; returns the run_conversation result dict."""
    return agent.run_conversation(
        user_message=user_message,
        system_message=system_message,
        conversation_history=conversation_history,
    )
