#!/usr/bin/env python3
"""
Repair AppWorld JSONL logs after infra-crash reruns.

Typical workflow when a batch hit ``I/O operation on closed file`` (or related
stdio / background-review races):

1. ``list-infra`` — show task ids whose *last* row matches infra error patterns.
2. ``strip-for-rerun`` — remove all rows for those tasks so ``--resume --all``
   only re-executes the affected tasks.
3. Re-run with the hardened ``run_appworld_with_hermes.py`` driver.
4. ``compact`` — keep one row per task (last wins) for a clean 90-task log.

Examples::

  python3 benchmark/scripts/repair_appworld_runs.py list-infra \\
      benchmark/runs/appworld_no_hot_train/runs.jsonl

  python3 benchmark/scripts/repair_appworld_runs.py strip-for-rerun \\
      benchmark/runs/appworld_no_hot_train/runs.jsonl \\
      --output benchmark/runs/appworld_no_hot_train/runs.prererun.jsonl

  python3 benchmark/scripts/repair_appworld_runs.py compact \\
      benchmark/runs/appworld_no_hot_train/runs.jsonl \\
      --in-place
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

INFRA_ERROR_MARKERS: Tuple[str, ...] = (
    "I/O operation on closed file",
    "'write' is read-only",
    "UnsupportedOperation('fileno'",
    "AttributeError(\"'_SafeWriter' object attribute 'write' is read-only\")",
)


def _task_id(row: Dict[str, Any]) -> Optional[str]:
    tid = row.get("appworld_task_id") or row.get("task_id") or row.get("skillsbench_task_id")
    return str(tid) if tid else None


def _is_infra_run_error(run_error: Any) -> bool:
    if not run_error:
        return False
    text = str(run_error)
    return any(marker in text for marker in INFRA_ERROR_MARKERS)


def load_last_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    last: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = _task_id(row)
            if tid:
                last[tid] = row
    return last


def list_infra_failed_tasks(path: Path) -> List[str]:
    """Task ids whose latest JSONL row has an infra-style ``run_error``."""
    last = load_last_rows(path)
    failed = [tid for tid, row in sorted(last.items()) if _is_infra_run_error(row.get("run_error"))]
    return failed


def strip_tasks(path: Path, task_ids: Iterable[str], output: Path) -> int:
    """Copy *path* to *output* omitting every row for the given task ids."""
    drop = set(task_ids)
    removed = 0
    kept = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with path.open(encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line in src:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                dst.write(line if line.endswith("\n") else line + "\n")
                kept += 1
                continue
            tid = _task_id(row)
            if tid and tid in drop:
                removed += 1
                continue
            dst.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            kept += 1
    return removed


def compact_jsonl(path: Path, output: Path) -> Tuple[int, int]:
    """Write one row per task id (last row in file wins). Returns (in, out)."""
    last: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    total_in = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total_in += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = _task_id(row)
            if not tid:
                continue
            if tid not in last:
                order.append(tid)
            last[tid] = row
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for tid in order:
            fh.write(json.dumps(last[tid], ensure_ascii=False, default=str) + "\n")
    return total_in, len(order)


def _cmd_list_infra(args: argparse.Namespace) -> int:
    path = Path(args.log)
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 1
    tasks = list_infra_failed_tasks(path)
    if args.output:
        Path(args.output).write_text("\n".join(tasks) + ("\n" if tasks else ""), encoding="utf-8")
    for tid in tasks:
        print(tid)
    print(f"# {len(tasks)} infra-failed task(s) in {path}", file=sys.stderr)
    return 0


def _cmd_strip(args: argparse.Namespace) -> int:
    path = Path(args.log)
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 1
    if args.tasks_file:
        task_ids = [
            line.strip()
            for line in Path(args.tasks_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    elif args.auto_infra:
        task_ids = list_infra_failed_tasks(path)
    else:
        print("Specify --auto-infra or --tasks-file", file=sys.stderr)
        return 1
    if not task_ids:
        print("No tasks to strip.", file=sys.stderr)
        return 0
    output = Path(args.output) if args.output else path
    if args.in_place and output == path:
        tmp = path.with_suffix(path.suffix + ".strip.tmp")
        removed = strip_tasks(path, task_ids, tmp)
        shutil.move(str(tmp), str(path))
    else:
        removed = strip_tasks(path, task_ids, output)
    print(
        f"Stripped {removed} row(s) for {len(task_ids)} task(s) -> {output}",
        file=sys.stderr,
    )
    return 0


def _cmd_compact(args: argparse.Namespace) -> int:
    path = Path(args.log)
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 1
    output = path if args.in_place else Path(args.output or path)
    if args.in_place:
        tmp = path.with_suffix(path.suffix + ".compact.tmp")
        total_in, total_out = compact_jsonl(path, tmp)
        shutil.move(str(tmp), str(path))
        out_path = path
    else:
        total_in, total_out = compact_jsonl(path, output)
        out_path = output
    print(
        f"Compacted {total_in} row(s) -> {total_out} task(s) in {out_path}",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair AppWorld runs.jsonl logs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-infra", help="List task ids with infra run_error.")
    p_list.add_argument("log", help="Path to runs.jsonl")
    p_list.add_argument(
        "--output",
        help="Optional file to write one task id per line (for strip-for-rerun).",
    )

    p_strip = sub.add_parser(
        "strip-for-rerun",
        help="Remove rows for tasks that will be re-run with --resume.",
    )
    p_strip.add_argument("log", help="Source runs.jsonl")
    p_strip.add_argument(
        "--output",
        help="Destination JSONL (default: overwrite with --in-place).",
    )
    p_strip.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the source log (creates a .bak if --backup).",
    )
    p_strip.add_argument(
        "--backup",
        action="store_true",
        help="With --in-place, copy log to log.bak first.",
    )
    src = p_strip.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--auto-infra",
        action="store_true",
        help="Strip every task whose latest row has an infra run_error.",
    )
    src.add_argument(
        "--tasks-file",
        help="Text file with one task id per line.",
    )

    p_compact = sub.add_parser("compact", help="Keep last row per task id.")
    p_compact.add_argument("log", help="Source runs.jsonl")
    p_compact.add_argument("--output", help="Destination (required unless --in-place).")
    p_compact.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite source log with compacted version.",
    )

    args = parser.parse_args()
    if args.command == "strip-for-rerun" and args.in_place and args.backup:
        src_path = Path(args.log)
        if src_path.is_file():
            shutil.copy2(src_path, str(src_path) + ".bak")

    if args.command == "list-infra":
        return _cmd_list_infra(args)
    if args.command == "strip-for-rerun":
        return _cmd_strip(args)
    if args.command == "compact":
        return _cmd_compact(args)
    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
