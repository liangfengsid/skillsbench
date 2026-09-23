#!/usr/bin/env python3
"""
Build reproducible SkillsBench train/test (and optional val) splits.

Protocols:
  - difficulty_stratified: match easy/medium/hard mix across train/test (default 75/25)
  - category_stratified: match category mix across train/test (for datasets without difficulty)
  - category_holdout: entire categories (>= min size, or an explicit list) go to test only

Examples (from Hermes repo root):

  python3 benchmark/scripts/make_skillsbench_splits.py --write-defaults

  python3 benchmark/scripts/make_skillsbench_splits.py \\
      --protocol difficulty_stratified --seed 42 \\
      -o benchmark/skillsbench_splits/stratified_v1.json

  python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \\
      --split-file benchmark/skillsbench_splits/stratified_v1.json \\
      --split-part train --log-jsonl benchmark/runs/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tomllib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPT = Path(__file__).resolve()
_DEFAULT_SKILLSBENCH = _SCRIPT.parents[1] / "skillsbench"
_DEFAULT_OUT_DIR = _SCRIPT.parents[1] / "skillsbench_splits"

_DIFFICULTY_ORDER = ("easy", "medium", "hard", "unknown")


def discover_task_metadata(tasks_dir: Path) -> List[Dict[str, Any]]:
    if not tasks_dir.is_dir():
        raise SystemExit(f"Tasks directory not found: {tasks_dir}")
    rows: List[Dict[str, Any]] = []
    for child in sorted(tasks_dir.iterdir()):
        if not child.is_dir() or not (child / "instruction.md").is_file():
            continue
        meta: Dict[str, Any] = {
            "task_id": child.name,
            "category": "unknown",
            "difficulty": "unknown",
        }
        toml_path = child / "task.toml"
        if toml_path.is_file():
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            block = data.get("metadata") or {}
            meta["category"] = str(block.get("category") or "unknown").strip() or "unknown"
            meta["difficulty"] = _normalize_difficulty(block.get("difficulty"))
        rows.append(meta)
    return rows


def _normalize_difficulty(value: Any) -> str:
    raw = str(value or "unknown").strip().lower()
    if raw in ("middle", "med"):
        return "medium"
    if raw in ("easy", "medium", "hard"):
        return raw
    return "unknown"


def _partition_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("train/val/test ratios must sum to > 0")
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total
    if n == 1:
        return 1, 0, 0
    n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0
    n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 and n - n_test > 1 else 0
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        overflow = n_val + n_test - (n - 1)
        while overflow > 0 and n_val > 0:
            n_val -= 1
            overflow -= 1
        while overflow > 0 and n_test > 0:
            n_test -= 1
            overflow -= 1
    return n_train, n_val, n_test


def difficulty_stratified_split(
    tasks: Sequence[Dict[str, Any]],
    *,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, List[str]]:
    by_diff: Dict[str, List[str]] = defaultdict(list)
    for row in tasks:
        by_diff[row["difficulty"]].append(row["task_id"])

    rng = random.Random(seed)
    train: List[str] = []
    val: List[str] = []
    test: List[str] = []

    for difficulty in _DIFFICULTY_ORDER:
        ids = sorted(by_diff.get(difficulty, []))
        if not ids:
            continue
        rng.shuffle(ids)
        n_train, n_val, n_test = _partition_counts(len(ids), train_ratio, val_ratio, test_ratio)
        train.extend(ids[:n_train])
        val.extend(ids[n_train : n_train + n_val])
        test.extend(ids[n_train + n_val : n_train + n_val + n_test])

    for bucket in sorted(set(by_diff) - set(_DIFFICULTY_ORDER)):
        ids = sorted(by_diff[bucket])
        rng.shuffle(ids)
        n_train, n_val, n_test = _partition_counts(len(ids), train_ratio, val_ratio, test_ratio)
        train.extend(ids[:n_train])
        val.extend(ids[n_train : n_train + n_val])
        test.extend(ids[n_train + n_val : n_train + n_val + n_test])

    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }


def category_stratified_split(
    tasks: Sequence[Dict[str, Any]],
    *,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, List[str]]:
    """Same partition math as difficulty_stratified, keyed by ``category``."""
    by_cat: Dict[str, List[str]] = defaultdict(list)
    for row in tasks:
        by_cat[row["category"]].append(row["task_id"])

    rng = random.Random(seed)
    train: List[str] = []
    val: List[str] = []
    test: List[str] = []
    for category in sorted(by_cat):
        ids = sorted(by_cat[category])
        rng.shuffle(ids)
        n_train, n_val, n_test = _partition_counts(len(ids), train_ratio, val_ratio, test_ratio)
        train.extend(ids[:n_train])
        val.extend(ids[n_train : n_train + n_val])
        test.extend(ids[n_train + n_val : n_train + n_val + n_test])
    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }


def category_holdout_split(
    tasks: Sequence[Dict[str, Any]],
    *,
    min_category_tasks: int = 3,
    holdout_categories: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    by_cat: Dict[str, List[str]] = defaultdict(list)
    for row in tasks:
        by_cat[row["category"]].append(row["task_id"])

    if holdout_categories:
        held_out_categories = sorted({str(c) for c in holdout_categories if str(c) in by_cat})
    else:
        held_out_categories = sorted(
            cat for cat, ids in by_cat.items() if len(ids) >= min_category_tasks
        )
    train: List[str] = []
    test: List[str] = []
    for cat, ids in by_cat.items():
        bucket = test if cat in held_out_categories else train
        bucket.extend(sorted(ids))

    return {
        "train": sorted(train),
        "val": [],
        "test": sorted(test),
        "held_out_categories": held_out_categories,
        "min_category_tasks": min_category_tasks,
    }


def _split_stats(tasks: Sequence[Dict[str, Any]], split: Dict[str, List[str]]) -> Dict[str, Any]:
    meta_by_id = {row["task_id"]: row for row in tasks}

    def bucket_stats(ids: Sequence[str]) -> Dict[str, Any]:
        diffs = Counter(meta_by_id[i]["difficulty"] for i in ids)
        cats = Counter(meta_by_id[i]["category"] for i in ids)
        return {
            "count": len(ids),
            "difficulty": dict(sorted(diffs.items())),
            "category": dict(sorted(cats.items())),
            "unique_categories": len(cats),
        }

    return {
        "train": bucket_stats(split["train"]),
        **({"val": bucket_stats(split["val"])} if split.get("val") else {}),
        "test": bucket_stats(split["test"]),
    }


def build_split_document(
    *,
    protocol: str,
    tasks: Sequence[Dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_category_tasks: int,
    holdout_categories: Optional[Sequence[str]] = None,
    schema: str = "skillsbench.split.v1",
) -> Dict[str, Any]:
    if protocol == "difficulty_stratified":
        parts = difficulty_stratified_split(
            tasks,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        extra: Dict[str, Any] = {
            "ratios": (
                {"train": train_ratio, "test": test_ratio}
                if val_ratio <= 0
                else {"train": train_ratio, "val": val_ratio, "test": test_ratio}
            ),
        }
    elif protocol == "category_stratified":
        parts = category_stratified_split(
            tasks,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        extra = {
            "ratios": (
                {"train": train_ratio, "test": test_ratio}
                if val_ratio <= 0
                else {"train": train_ratio, "val": val_ratio, "test": test_ratio}
            ),
        }
    elif protocol == "category_holdout":
        holdout = category_holdout_split(
            tasks,
            min_category_tasks=min_category_tasks,
            holdout_categories=holdout_categories,
        )
        parts = {
            "train": holdout["train"],
            "val": [],
            "test": holdout["test"],
        }
        extra = {
            "min_category_tasks": holdout["min_category_tasks"],
            "held_out_categories": holdout["held_out_categories"],
        }
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    all_ids = {row["task_id"] for row in tasks}
    assigned = set(parts["train"]) | set(parts["val"]) | set(parts["test"])
    if assigned != all_ids:
        missing = sorted(all_ids - assigned)
        extra_ids = sorted(assigned - all_ids)
        raise RuntimeError(f"Split incomplete: missing={missing!r} extra={extra_ids!r}")

    overlap = (set(parts["train"]) & set(parts["test"])) | (set(parts["val"]) & set(parts["test"]))
    overlap |= set(parts["train"]) & set(parts["val"])
    if overlap:
        raise RuntimeError(f"Split partitions overlap: {sorted(overlap)!r}")

    doc: Dict[str, Any] = {
        "schema": schema,
        "protocol": protocol,
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "skillsbench_tasks_dir": None,
        "tasks_dir": None,
        "total_tasks": len(tasks),
        "train": parts["train"],
        "test": parts["test"],
        "stats": _split_stats(tasks, parts),
        **extra,
    }
    if parts.get("val"):
        doc["val"] = parts["val"]
    return doc


def load_split_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    for key in ("train", "test"):
        if key not in data or not isinstance(data[key], list):
            raise ValueError(f"Split file missing list field {key!r}: {path}")
    if "val" not in data:
        data["val"] = []
    return data


def resolve_split_task_ids(path: Path, part: str) -> List[str]:
    data = load_split_file(path)
    if part == "all":
        return list(data["train"]) + list(data.get("val") or []) + list(data["test"])
    if part not in ("train", "val", "test"):
        raise ValueError(f"Invalid split part: {part!r}")
    return list(data[part])


def _display_tasks_dir(tasks_dir: Path) -> str:
    """Repo-relative path so split JSON does not embed a home directory."""
    resolved = tasks_dir.expanduser().resolve()
    repo = _SCRIPT.parents[2]
    try:
        return str(resolved.relative_to(repo))
    except ValueError:
        return resolved.name


def write_defaults(out_dir: Path, tasks_dir: Path, seed: int) -> None:
    tasks = discover_task_metadata(tasks_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("stratified_v1.json", "difficulty_stratified", {}),
        ("category_holdout_v1.json", "category_holdout", {"min_category_tasks": 3}),
    ]
    for filename, protocol, kwargs in specs:
        doc = build_split_document(
            protocol=protocol,
            tasks=tasks,
            seed=seed,
            train_ratio=0.75,
            val_ratio=0.0,
            test_ratio=0.25,
            min_category_tasks=int(kwargs.get("min_category_tasks", 3)),
        )
        doc["skillsbench_tasks_dir"] = _display_tasks_dir(tasks_dir)
        doc["tasks_dir"] = _display_tasks_dir(tasks_dir)
        doc["name"] = filename.replace(".json", "")
        out_path = out_dir / filename
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"Wrote {out_path} ({doc['stats']['train']['count']} train / "
            f"{doc['stats']['test']['count']} test)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SkillsBench train/test splits.")
    parser.add_argument(
        "--skillsbench-root",
        type=str,
        default=str(_DEFAULT_SKILLSBENCH),
        help=f"SkillsBench root (default: {_DEFAULT_SKILLSBENCH}).",
    )
    parser.add_argument(
        "--protocol",
        choices=("difficulty_stratified", "category_stratified", "category_holdout"),
        default="difficulty_stratified",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Optional validation fraction (default 0 — train/test only).",
    )
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument(
        "--min-category-tasks",
        type=int,
        default=3,
        help="category_holdout: categories with this many tasks (or more) go entirely to test.",
    )
    parser.add_argument(
        "--holdout-categories",
        type=str,
        default=None,
        help="category_holdout: comma-separated category names to hold out (overrides --min-category-tasks).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Write split JSON to this path.",
    )
    parser.add_argument(
        "--write-defaults",
        action="store_true",
        help=f"Write stratified_v1.json and category_holdout_v1.json to {_DEFAULT_OUT_DIR}.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_DEFAULT_OUT_DIR),
        help=f"Output directory for --write-defaults (default: {_DEFAULT_OUT_DIR}).",
    )
    args = parser.parse_args()

    skillsbench_root = Path(args.skillsbench_root).expanduser().resolve()
    tasks_dir = skillsbench_root / "tasks"
    tasks = discover_task_metadata(tasks_dir)

    if args.write_defaults:
        write_defaults(Path(args.out_dir).expanduser(), tasks_dir, args.seed)
        return 0

    holdout: Optional[List[str]] = None
    if args.holdout_categories:
        holdout = [p.strip() for p in args.holdout_categories.split(",") if p.strip()]
    doc = build_split_document(
        protocol=args.protocol,
        tasks=tasks,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_category_tasks=args.min_category_tasks,
        holdout_categories=holdout,
    )
    doc["skillsbench_tasks_dir"] = _display_tasks_dir(tasks_dir)
    doc["tasks_dir"] = _display_tasks_dir(tasks_dir)
    doc["name"] = args.output or f"{args.protocol}_seed{args.seed}"

    text = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out.resolve()}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
