"""Tests for shared SkillsBench metrics aggregation + CoEvoSkills library build."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CORE = _REPO / "benchmark" / "scripts" / "skillsbench_aggregate_core.py"
_AGG = _REPO / "benchmark" / "scripts" / "aggregate_skillsbench_runs.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _record(
    task_id: str,
    *,
    pass_at_turn: dict,
    final_success: bool = False,
    final_tokens: int = 1000,
    final_api_calls: int = 29,
    method: str = "hermes",
    phase: str | None = None,
    split_part: str | None = None,
):
    rec = {
        "schema": "skillsbench.hermes_run.v1",
        "skillsbench_task_id": task_id,
        "ts_end_iso": "2026-01-01T00:00:00Z",
        "method": method,
        "pass_at_turn": pass_at_turn,
        "evaluation": {
            "task_success": final_success,
            "tests_passed": 12 if final_success else 0,
            "tests_total": 12,
        },
        "run_conversation_result": {
            "total_tokens": final_tokens,
            "api_calls": final_api_calls,
            "completed": True,
        },
    }
    if phase:
        rec["phase"] = phase
    if split_part:
        rec["split_part"] = split_part
    return rec


def test_pass_at_conversation_turn_across_tasks():
    mod = _load(_AGG, "aggregate_skillsbench_runs")
    records = [
        _record(
            "task-a",
            pass_at_turn={
                "5": {
                    "turn": 5,
                    "evaluation": {"task_success": False, "tests_passed": 6, "tests_total": 12},
                    "total_tokens": 400000,
                },
                "10": {
                    "turn": 10,
                    "evaluation": {"task_success": True, "tests_passed": 12, "tests_total": 12},
                    "total_tokens": 900000,
                    "api_calls": 10,
                },
            },
            final_success=True,
            final_tokens=900000,
            final_api_calls=12,
        ),
        _record(
            "task-b",
            pass_at_turn={
                "5": {
                    "turn": 5,
                    "evaluation": {"task_success": True, "tests_passed": 12, "tests_total": 12},
                    "total_tokens": 300000,
                    "api_calls": 5,
                },
            },
            final_success=True,
            final_tokens=300000,
            final_api_calls=5,
        ),
    ]
    summary = mod.aggregate_records(records, pass_k_values=[5, 10])
    assert summary["pass_k"]["5"]["macro_success_rate"] == 0.5
    assert summary["pass_k"]["5"]["macro_task_pass_rate"] == 0.5
    assert summary["pass_k"]["5"]["micro_success_rate"] == 0.75
    assert summary["pass_k"]["10"]["macro_success_rate"] == 1.0
    assert summary["pass_k"]["5"]["tokens_mean_at_turn"] == 350000.0

    cts = summary["cost_to_succeed"]
    # task-a first success at turn 10 (900k); task-b at turn 5 (300k)
    assert cts["tokens"]["n"] == 2
    assert cts["tokens"]["mean"] == 600000.0
    assert cts["user_iterations"]["mean"] == 7.5
    assert cts["tokens"]["std"] is not None


def test_final_metrics_mean_std():
    mod = _load(_AGG, "aggregate_skillsbench_runs")
    records = [
        _record("t1", pass_at_turn={}, final_success=True, final_tokens=500, final_api_calls=10),
        _record("t2", pass_at_turn={}, final_success=False, final_tokens=700, final_api_calls=20),
    ]
    records[0]["duration_sec"] = 100.0
    records[1]["duration_sec"] = 200.0
    summary = mod.aggregate_records(
        records, pass_k_values=[1], max_user_iterations=90
    )
    assert summary["final"]["macro_success_rate"] == 0.5
    assert summary["final"]["tokens"]["mean"] == 600.0
    assert summary["final"]["user_iterations"]["mean"] == 15.0
    assert summary["final"]["max_user_iterations"] == 90
    assert summary["final"]["duration_sec"]["mean"] == 150.0
    assert summary["final"]["duration_sec_sum"] == 300.0
    assert summary["cost_to_succeed"]["tokens"]["mean"] == 500.0
    assert summary["cost_to_succeed"]["user_iterations"]["mean"] == 10.0
    assert summary["cost_to_succeed"]["duration_sec"]["mean"] == 100.0


def test_filter_method_phase_split():
    core = _load(_CORE, "skillsbench_aggregate_core")
    records = [
        _record("a", pass_at_turn={}, final_success=True, method="coevoskills", phase="frozen_eval", split_part="test"),
        _record("b", pass_at_turn={}, final_success=False, method="hermes", phase=None, split_part="test"),
        _record("c", pass_at_turn={}, final_success=True, method="coevoskills", phase="evolve", split_part="train"),
    ]
    summary = core.aggregate_skillsbench_metrics(
        records,
        pass_k_values=[1],
        method="coevoskills",
        phase="frozen_eval",
        split_part="test",
    )
    assert summary["tasks"] == 1
    assert summary["final"]["macro_success_rate"] == 1.0


def test_build_frozen_library(tmp_path: Path):
    sys.path.insert(0, str(_REPO))
    from benchmark.baselines.coevoskills.run_split_protocol import build_frozen_library

    work = tmp_path / "work"
    for tid, name in (("task-a", "evo-foo"), ("task-b", "evo-foo"), ("task-b", "evo-bar")):
        skill_dir = work / tid / "skills" / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
        (skill_dir / "scripts").mkdir(exist_ok=True)
        (skill_dir / "scripts" / "utils.py").write_text("x=1\n", encoding="utf-8")

    lib = tmp_path / "lib"
    manifest = build_frozen_library(
        work_root=work,
        source_task_ids=["task-a", "task-b"],
        library_dir=lib,
    )
    assert manifest["n_skills"] == 3
    names = {s["skill"] for s in manifest["skills"]}
    assert "evo-foo" in names
    assert "evo-bar" in names
    assert any(n.startswith("evo-foo__from_") for n in names)
    assert (lib / "MANIFEST.json").is_file()


def test_split_file_train_test_counts():
    split = _REPO / "benchmark" / "skillsbench_splits" / "stratified_v1.json"
    data = json.loads(split.read_text(encoding="utf-8"))
    assert len(data["train"]) == 65
    assert len(data["test"]) == 23
    sys.path.insert(0, str(_REPO))
    from benchmark.baselines.coevoskills.run_split_protocol import _load_split_ids

    train = _load_split_ids(split, "train")
    test = _load_split_ids(split, "test")
    assert len(train) == 65
    assert len(test) == 23
    assert set(train).isdisjoint(set(test))
