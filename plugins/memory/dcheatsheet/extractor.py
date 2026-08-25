"""Extract ``<cheatsheet>`` blocks from curator output.

Faithful to ``dynamic_cheatsheet.utils.extractor.extract_cheatsheet``
(https://github.com/suzgunmirac/dynamic-cheatsheet): if the tagged block is
missing, keep the previous cheatsheet.
"""

from __future__ import annotations

_OPEN = "<cheatsheet>"
_CLOSE = "</cheatsheet>"


def extract_cheatsheet(response: str, old_cheatsheet: str) -> str:
    text = (response or "").strip()
    if not text:
        return old_cheatsheet
    lower = text.lower()
    start = lower.find(_OPEN)
    if start < 0:
        return old_cheatsheet
    inner_start = start + len(_OPEN)
    end = lower.find(_CLOSE, inner_start)
    if end < 0:
        body = text[inner_start:].strip()
    else:
        body = text[inner_start:end].strip()
    return body or old_cheatsheet
