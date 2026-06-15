#!/usr/bin/env python3
"""
Load JSON Lines written by ``run_skillsbench_with_hermes.py`` into Python dicts.

Each non-empty line must be one JSON object (typically ``schema`` of
``skillsbench.hermes_run.v1`` or ``skillsbench.hermes_run_error.v1``).

Import for downstream analysis::

    from pathlib import Path
    from read_skillsbench_jsonl import load_skillsbench_run_records

    records = load_skillsbench_run_records(Path("benchmark/hermes_skillsbench_runs.jsonl"))
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "iter_skillsbench_run_records",
    "load_skillsbench_run_records",
]


def _default_jsonl_path() -> Path:
    """``benchmark/hermes_skillsbench_runs.jsonl`` next to ``benchmark/scripts/``."""
    return Path(__file__).resolve().parent.parent / "hermes_skillsbench_runs.jsonl"


def iter_skillsbench_run_records(path: Path | str) -> Iterator[dict[str, Any]]:
    """
    Yield one ``dict`` per line from *path* (UTF-8). Skips blank lines.

    Raises ``json.JSONDecodeError`` with context if a line is not valid JSON.
    """

    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Not a file: {p}")

    with p.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"{p}:{lineno}: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"{p}:{lineno}: expected JSON object at top level, got {type(obj).__name__}"
                )
            yield obj


def load_skillsbench_run_records(path: Path | str) -> list[dict[str, Any]]:
    """Read the entire JSONL file into a list of dicts (in-memory structure)."""

    return list(iter_skillsbench_run_records(path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load Hermes SkillsBench JSONL run logs into memory (CLI summary).",
    )
    parser.add_argument(
        "jsonl_file",
        nargs="?",
        default=str(_default_jsonl_path()),
        help=(
            "Path to .jsonl (default: benchmark/hermes_skillsbench_runs.jsonl "
            "relative to this script's benchmark/ parent)."
        ),
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Print pretty-printed JSON array of all records to stdout.",
    )
    args = parser.parse_args()

    try:
        records = load_skillsbench_run_records(args.jsonl_file)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.dump_json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0

    schemas = Counter(str(r.get("schema", "")) for r in records)
    tasks = [r.get("skillsbench_task_id") for r in records]
    print(f"file: {Path(args.jsonl_file).resolve()}")
    print(f"records: {len(records)}")
    print("schemas:")
    for s, n in schemas.most_common():
        print(f"  {n:4d}  {s!r}")
    if tasks and all(t is not None for t in tasks):
        print("task_ids:", ", ".join(str(t) for t in tasks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
