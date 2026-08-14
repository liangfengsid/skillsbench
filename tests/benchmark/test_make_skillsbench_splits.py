"""Tests for benchmark/scripts/make_skillsbench_splits.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "benchmark" / "scripts" / "make_skillsbench_splits.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("make_skillsbench_splits", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_difficulty_stratified_covers_all_tasks():
    mod = _load_module()
    tasks = [
        {"task_id": "a", "category": "x", "difficulty": "easy"},
        {"task_id": "b", "category": "x", "difficulty": "easy"},
        {"task_id": "c", "category": "y", "difficulty": "medium"},
        {"task_id": "d", "category": "y", "difficulty": "hard"},
    ]
    split = mod.difficulty_stratified_split(tasks, seed=1)
    assert sorted(split["train"] + split["val"] + split["test"]) == ["a", "b", "c", "d"]
    assert not (set(split["train"]) & set(split["test"]))


def test_category_stratified_covers_all_tasks():
    mod = _load_module()
    tasks = [
        {"task_id": "a", "category": "Software", "difficulty": "unknown"},
        {"task_id": "b", "category": "Software", "difficulty": "unknown"},
        {"task_id": "c", "category": "ML", "difficulty": "unknown"},
        {"task_id": "d", "category": "ML", "difficulty": "unknown"},
        {"task_id": "e", "category": "Science", "difficulty": "unknown"},
    ]
    split = mod.category_stratified_split(tasks, seed=42, train_ratio=0.75, val_ratio=0.0, test_ratio=0.25)
    assigned = sorted(split["train"] + split["val"] + split["test"])
    assert assigned == ["a", "b", "c", "d", "e"]
    assert not (set(split["train"]) & set(split["test"]))
    assert split["test"], "expected a non-empty test partition"


def test_category_holdout_explicit_list():
    mod = _load_module()
    tasks = [
        {"task_id": "h1", "category": "Hardware", "difficulty": "unknown"},
        {"task_id": "s1", "category": "Software", "difficulty": "unknown"},
        {"task_id": "m1", "category": "Media", "difficulty": "unknown"},
    ]
    split = mod.category_holdout_split(tasks, holdout_categories=["Hardware", "Media"])
    assert split["test"] == ["h1", "m1"]
    assert split["train"] == ["s1"]
    assert split["held_out_categories"] == ["Hardware", "Media"]


def test_category_holdout_keeps_categories_disjoint():
    mod = _load_module()
    tasks = [
        {"task_id": "s1", "category": "security", "difficulty": "hard"},
        {"task_id": "s2", "category": "security", "difficulty": "medium"},
        {"task_id": "s3", "category": "security", "difficulty": "easy"},
        {"task_id": "u1", "category": "unique-a", "difficulty": "medium"},
    ]
    split = mod.category_holdout_split(tasks, min_category_tasks=3)
    assert split["test"] == ["s1", "s2", "s3"]
    assert split["train"] == ["u1"]
    assert split["held_out_categories"] == ["security"]


def test_default_split_files_exist_and_validate():
    mod = _load_module()
    root = Path(__file__).resolve().parents[2] / "benchmark" / "skillsbench_splits"
    for name in ("stratified_v1.json", "category_holdout_v1.json"):
        path = root / name
        assert path.is_file(), name
        data = mod.load_split_file(path)
        assert data["schema"] == "skillsbench.split.v1"
        val = data.get("val") or []
        all_ids = set(data["train"]) | set(val) | set(data["test"])
        assert len(all_ids) == data["total_tasks"] == 88
        assert not val, f"expected no val split in {name}"


def test_stratified_default_is_train_test_only():
    mod = _load_module()
    path = Path(__file__).resolve().parents[2] / "benchmark" / "skillsbench_splits" / "stratified_v1.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    data = mod.load_split_file(path)
    assert "val" not in raw
    assert len(data["train"]) + len(data["test"]) == 88
    assert raw["ratios"] == {"train": 0.75, "test": 0.25}


def test_category_holdout_no_category_overlap():
    mod = _load_module()
    path = Path(__file__).resolve().parents[2] / "benchmark" / "skillsbench_splits" / "category_holdout_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks_dir = Path(data["skillsbench_tasks_dir"])
    meta = {row["task_id"]: row for row in mod.discover_task_metadata(tasks_dir)}

    train_cats = {meta[tid]["category"] for tid in data["train"]}
    test_cats = {meta[tid]["category"] for tid in data["test"]}
    assert not (train_cats & test_cats)


def test_resolve_split_task_ids_all():
    mod = _load_module()
    path = Path(__file__).resolve().parents[2] / "benchmark" / "skillsbench_splits" / "stratified_v1.json"
    data = mod.load_split_file(path)
    assert len(mod.resolve_split_task_ids(path, "all")) == 88
    assert len(mod.resolve_split_task_ids(path, "test")) == len(data["test"])
