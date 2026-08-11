"""Fixed-height console window: new status lines replace old ones (TTY).

Permanent messages (batch progress, failures) scroll normally.
Ephemeral agent activity is redrawn in a fixed N-line viewport so long runs
do not flood the terminal. Errors should go to stderr (unchanged).
"""

from __future__ import annotations

import os
import re
import sys
from collections import deque
from typing import Any, Deque, List, Optional, TextIO


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _visible_width() -> int:
    try:
        return max(40, os.get_terminal_size(sys.stdout.fileno()).columns)
    except (OSError, AttributeError, ValueError):
        return 100


class ConsoleWindow:
    """
    Maintain a redrawable viewport of ``height`` lines on stdout.

    - ``ephemeral(text)`` — feed agent/status lines into the ring buffer and redraw
    - ``commit(text)`` — clear viewport, print a permanent line, reset buffer
    - Non-TTY: ephemeral falls back to printing (or last-line only if ``plain_last_only``)
    """

    def __init__(
        self,
        height: int = 12,
        *,
        stream: Optional[TextIO] = None,
        plain_last_only: bool = True,
    ):
        self.height = max(1, int(height))
        self.stream = stream or sys.stdout
        self.lines: Deque[str] = deque(maxlen=self.height)
        self._drawn = 0
        self._enabled = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.plain_last_only = plain_last_only

    def print_fn(self, *args: Any, **kwargs: Any) -> None:
        """Drop-in replacement for ``print`` / AIAgent ``_print_fn``."""
        kwargs = dict(kwargs)
        kwargs.pop("file", None)
        end = kwargs.pop("end", "\n")
        sep = kwargs.pop("sep", " ")
        text = sep.join(str(a) for a in args) + (end if end is not None else "")
        self.ephemeral(text)

    def ephemeral(self, text: str) -> None:
        raw = text.replace("\r\n", "\n").replace("\r", "\n")
        for part in raw.split("\n"):
            cleaned = _strip_ansi(part).rstrip()
            if cleaned == "":
                continue
            # Truncate ultra-long single lines for the viewport
            cols = _visible_width() - 1
            if len(cleaned) > cols:
                cleaned = cleaned[: cols - 1] + "…"
            self.lines.append(cleaned)
        self._redraw()

    def commit(self, text: str) -> None:
        """Print a permanent line below the cleared viewport."""
        self._clear_viewport()
        msg = _strip_ansi(text).rstrip()
        try:
            print(msg, file=self.stream, flush=True)
        except (OSError, ValueError):
            pass
        self.lines.clear()
        self._drawn = 0

    def _clear_viewport(self) -> None:
        if not self._enabled or self._drawn <= 0:
            self._drawn = 0
            return
        try:
            # Move cursor to start of viewport, clear each line
            self.stream.write(f"\033[{self._drawn}A")
            for _ in range(self._drawn):
                self.stream.write("\033[2K\033[1B")
            self.stream.write(f"\033[{self._drawn}A")
            self.stream.flush()
        except (OSError, ValueError):
            pass
        self._drawn = 0

    def _redraw(self) -> None:
        if not self._enabled:
            if self.plain_last_only and self.lines:
                # Non-TTY: avoid flooding redirected logs — one line with CR
                try:
                    last = self.lines[-1]
                    self.stream.write("\r\033[2K" + last[: _visible_width() - 1])
                    self.stream.flush()
                except (OSError, ValueError):
                    pass
            else:
                for line in list(self.lines)[-1:]:
                    try:
                        print(line, file=self.stream, flush=True)
                    except (OSError, ValueError):
                        pass
            return

        pad: List[str] = list(self.lines)
        while len(pad) < self.height:
            pad.append("")
        # Keep only the last ``height`` lines for display
        pad = pad[-self.height :]

        try:
            if self._drawn > 0:
                self.stream.write(f"\033[{self._drawn}A")
            for line in pad:
                self.stream.write("\033[2K" + line + "\n")
            self.stream.flush()
            self._drawn = self.height
        except (OSError, ValueError):
            self._drawn = 0

    def close(self) -> None:
        """Leave the viewport contents as normal scrollback."""
        if self._enabled and self._drawn > 0:
            # Cursor already after the viewport; just stop tracking
            self._drawn = 0
        elif not self._enabled:
            try:
                self.stream.write("\n")
                self.stream.flush()
            except (OSError, ValueError):
                pass


# Process-wide window used by batch drivers (optional)
_ACTIVE: Optional[ConsoleWindow] = None


def get_active_window() -> Optional[ConsoleWindow]:
    return _ACTIVE


def set_active_window(window: Optional[ConsoleWindow]) -> None:
    global _ACTIVE
    _ACTIVE = window


def commit(msg: str) -> None:
    w = _ACTIVE
    if w is not None:
        w.commit(msg)
    else:
        print(msg, flush=True)


def ephemeral(msg: str) -> None:
    w = _ACTIVE
    if w is not None:
        w.ephemeral(msg)
    else:
        print(msg, flush=True)
