"""Tests for benchmark/scripts/evaluate_skillsbench_task.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "scripts"
    / "evaluate_skillsbench_task.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_skillsbench_task", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_adapt_container_paths(tmp_path):
    mod = _load_module()
    host_root = tmp_path / "root"
    host_root.mkdir()
    text = 'assert os.path.exists("/root/out.txt")\n'
    adapted = mod.adapt_container_paths(text, host_root)
    assert f'"{host_root}/out.txt"' in adapted
    assert '"/root/' not in adapted


def test_stage_task_copies_environment_and_outputs(tmp_path):
    mod = _load_module()
    task_dir = tmp_path / "tasks" / "demo"
    env = task_dir / "environment"
    env.mkdir(parents=True)
    (env / "input.txt").write_text("data", encoding="utf-8")
    (task_dir / "out.txt").write_text("done", encoding="utf-8")
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test_outputs.py").write_text("pass\n", encoding="utf-8")

    stage = tmp_path / "stage"
    host_root = mod.stage_task_for_host_eval(task_dir, stage)
    assert (host_root / "input.txt").read_text(encoding="utf-8") == "data"
    assert (host_root / "out.txt").read_text(encoding="utf-8") == "done"


def test_evaluate_task_host_minimal(tmp_path):
    mod = _load_module()
    task_dir = tmp_path / "tasks" / "mini"
    task_dir.mkdir(parents=True)
    (task_dir / "out.txt").write_text("ok", encoding="utf-8")
    tests = task_dir / "tests"
    tests.mkdir()
    (tests / "test_outputs.py").write_text(
        "import os\n\n"
        "def test_out_exists():\n"
        '    assert os.path.exists("/root/out.txt")\n',
        encoding="utf-8",
    )

    result = mod.evaluate_task_host(
        task_id="mini",
        skillsbench_root=tmp_path,
        timeout_sec=60.0,
    )
    assert result["task_success"] is True
    assert result["tests_passed"] == 1
    assert result["tests_total"] == 1
    assert result["reward"] == 1.0


def test_pass_at_turn_tracker_records_completed_turn():
    mod = _load_module()
    calls = []

    def fake_eval(*, task_id, skillsbench_root, timeout_sec=600.0):
        calls.append(task_id)
        return {"task_success": True, "tests_passed": 1, "tests_total": 1}

    class FakeAgent:
        step_callback = None
        session_total_tokens = 123
        session_input_tokens = 100
        session_output_tokens = 23
        _api_call_count = 0

    agent = FakeAgent()
    tracker = mod.PassAtTurnTracker(
        task_id="demo",
        skillsbench_root=Path("/tmp"),
        pass_k_turns=[1, 3],
        evaluate_fn=fake_eval,
    )
    tracker.attach(agent)
    tracker._on_step(2, [])
    tracker._on_step(4, [])
    out = tracker.finalize(3)
    tracker.detach(agent)
    assert set(out.keys()) == {"1", "3"}
    assert out["1"]["tokens"]["total"] == 123
    assert calls == ["demo", "demo"]
