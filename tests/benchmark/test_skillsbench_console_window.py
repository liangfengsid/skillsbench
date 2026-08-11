"""Tests for fixed-height console window."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from io import StringIO

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "scripts"
    / "skillsbench_console_window.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("skillsbench_console_window", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ephemeral_keeps_last_n_lines():
    mod = _load()
    buf = StringIO()
    # Force non-tty path
    win = mod.ConsoleWindow(3, stream=buf, plain_last_only=False)
    win._enabled = False
    for i in range(5):
        win.ephemeral(f"line-{i}")
    assert list(win.lines) == ["line-2", "line-3", "line-4"]


def test_commit_clears_buffer():
    mod = _load()
    buf = StringIO()
    win = mod.ConsoleWindow(4, stream=buf, plain_last_only=False)
    win._enabled = False
    win.ephemeral("a")
    win.ephemeral("b")
    win.commit("permanent")
    assert list(win.lines) == []
    assert "permanent" in buf.getvalue()
