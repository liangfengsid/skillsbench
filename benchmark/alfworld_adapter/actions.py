"""Parse a single ALFWorld command from a free-form LLM reply."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*\n?(.*?)```", re.DOTALL)
_PREFIX_RE = re.compile(
    r"^\s*(?:action|command|output|>)\s*[:\-]\s*",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _candidate_lines(text: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    fenced = [m.group(1).strip() for m in _FENCE_RE.finditer(raw) if m.group(1).strip()]
    body = fenced[-1] if fenced else raw
    lines: List[str] = []
    for line in body.splitlines():
        cleaned = _PREFIX_RE.sub("", line).strip().strip("`\"'")
        if cleaned:
            lines.append(cleaned)
    return lines


def parse_alfworld_action(
    text: str,
    admissible: Optional[Sequence[str]] = None,
) -> str:
    """Return one command string.

    Prefer an exact (case-insensitive) match against *admissible*, then a
    unique prefix match, then the last non-empty reply line.
    """
    lines = _candidate_lines(text)
    if not lines:
        return "look"

    allowed = list(admissible or [])
    allowed_norm = {_normalize(cmd): cmd for cmd in allowed}

    for line in reversed(lines):
        key = _normalize(line)
        if key in allowed_norm:
            return allowed_norm[key]

    if allowed:
        for line in reversed(lines):
            key = _normalize(line)
            prefixes = [
                cmd
                for cmd_key, cmd in allowed_norm.items()
                if cmd_key.startswith(key) or key.startswith(cmd_key)
            ]
            unique = list(dict.fromkeys(prefixes))
            if len(unique) == 1:
                return unique[0]

    return lines[-1]
