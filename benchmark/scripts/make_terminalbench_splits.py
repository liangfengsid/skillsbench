#!/usr/bin/env python3
"""
Build reproducible Terminal-Bench train/test splits.

Terminal-Bench task.toml metadata has category/subcategory but typically no
difficulty, so the default protocol is category-stratified 75/25 (seed 42).

Examples (from Hermes repo root):

  python3 benchmark/scripts/make_terminalbench_splits.py --write-defaults

  python3 benchmark/scripts/make_terminalbench_splits.py \\
      --protocol category_stratified --seed 42 \\
      -o benchmark/terminalbench_splits/stratified_v1.json

  python3 benchmark/scripts/run_terminalbench_with_harbor.py --all \\
      --split-file benchmark/terminalbench_splits/stratified_v1.json \\
      --split-part train --log-jsonl benchmark/runs/tb_train.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT = Path(__file__).resolve()
_SPLIT_HELPER = _SCRIPT.parent / "make_skillsbench_splits.py"
_DEFAULT_ROOT = _SCRIPT.parents[1] / "terminal-bench"
_DEFAULT_OUT_DIR = _SCRIPT.parents[1] / "terminalbench_splits"
_SCHEMA = "terminalbench.split.v1"
# Explicit hold-out: every TB category is large enough that the SkillsBench
# "min size" rule would send *all* tasks to test.
_DEFAULT_HOLDOUT = ("Hardware", "Media", "Security")


def _load_split_mod():
    spec = importlib.util.spec_from_file_location("make_skillsbench_splits", _SPLIT_HELPER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_defaults(out_dir: Path, tasks_dir: Path, seed: int) -> None:
    split_mod = _load_split_mod()
    tasks = split_mod.discover_task_metadata(tasks_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("stratified_v1.json", "category_stratified", {}),
        (
            "category_holdout_v1.json",
            "category_holdout",
            {"holdout_categories": list(_DEFAULT_HOLDOUT)},
        ),
    ]
    for filename, protocol, kwargs in specs:
        doc = split_mod.build_split_document(
            protocol=protocol,
            tasks=tasks,
            seed=seed,
            train_ratio=0.75,
            val_ratio=0.0,
            test_ratio=0.25,
            min_category_tasks=3,
            holdout_categories=kwargs.get("holdout_categories"),
            schema=_SCHEMA,
        )
        resolved = str(tasks_dir.resolve())
        doc["tasks_dir"] = resolved
        doc["skillsbench_tasks_dir"] = resolved
        doc["dataset"] = "terminal-bench"
        doc["name"] = filename.replace(".json", "")
        out_path = out_dir / filename
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"Wrote {out_path} ({doc['stats']['train']['count']} train / "
            f"{doc['stats']['test']['count']} test)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Terminal-Bench train/test splits.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(_DEFAULT_ROOT),
        help=f"Terminal-Bench root containing tasks/ (default: {_DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--protocol",
        choices=("category_stratified", "category_holdout", "difficulty_stratified"),
        default="category_stratified",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--min-category-tasks", type=int, default=3)
    parser.add_argument(
        "--holdout-categories",
        type=str,
        default=None,
        help=(
            "category_holdout: comma-separated categories (default when writing "
            f"defaults: {', '.join(_DEFAULT_HOLDOUT)})."
        ),
    )
    parser.add_argument("-o", "--output", type=str, default=None)
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

    split_mod = _load_split_mod()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    tasks_dir = dataset_root / "tasks"
    tasks = split_mod.discover_task_metadata(tasks_dir)

    if args.write_defaults:
        write_defaults(Path(args.out_dir).expanduser(), tasks_dir, args.seed)
        return 0

    holdout: Optional[List[str]] = None
    if args.holdout_categories:
        holdout = [p.strip() for p in args.holdout_categories.split(",") if p.strip()]
    elif args.protocol == "category_holdout":
        holdout = list(_DEFAULT_HOLDOUT)

    doc: Dict[str, Any] = split_mod.build_split_document(
        protocol=args.protocol,
        tasks=tasks,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_category_tasks=args.min_category_tasks,
        holdout_categories=holdout,
        schema=_SCHEMA,
    )
    resolved = str(tasks_dir)
    doc["tasks_dir"] = resolved
    doc["skillsbench_tasks_dir"] = resolved
    doc["dataset"] = "terminal-bench"
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
