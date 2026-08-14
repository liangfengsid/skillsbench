"""Tests for Terminal-Bench split files (Harbor eval uses these ids)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SPLIT = _REPO / "benchmark" / "scripts" / "make_skillsbench_splits.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_terminalbench_default_splits_exist_and_cover_tasks():
    split_mod = _load(_SPLIT, "make_skillsbench_splits")
    root = _REPO / "benchmark" / "terminalbench_splits"
    tasks_dir = _REPO / "benchmark" / "terminal-bench" / "tasks"
    if not tasks_dir.is_dir():
        return
    discovered = {row["task_id"] for row in split_mod.discover_task_metadata(tasks_dir)}
    for name in ("stratified_v1.json", "category_holdout_v1.json"):
        path = root / name
        assert path.is_file(), name
        data = split_mod.load_split_file(path)
        assert data["schema"] == "terminalbench.split.v1"
        assigned = set(data["train"]) | set(data.get("val") or []) | set(data["test"])
        assert assigned == discovered
        assert not (set(data["train"]) & set(data["test"]))
        assert data["total_tasks"] == len(discovered)
