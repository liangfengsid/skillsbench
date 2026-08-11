"""Tests for benchmark/scripts/skillsbench_experiment_workspace.py."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "scripts"
    / "skillsbench_experiment_workspace.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "skillsbench_experiment_workspace", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_source_task(root: Path, task_id: str) -> Path:
    task = root / "tasks" / task_id
    task.mkdir(parents=True)
    (task / "instruction.md").write_text(f"# {task_id}\n", encoding="utf-8")
    (task / "task.toml").write_text('name = "demo"\n', encoding="utf-8")
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (task / "tests").mkdir()
    (task / "tests" / "test_dummy.py").write_text("def test_ok():\n    assert True\n")
    return task


def test_prepare_copies_tasks_and_sets_hermes_home(tmp_path, monkeypatch):
    mod = _load_module()
    source = tmp_path / "skillsbench"
    _make_source_task(source, "task-a")
    _make_source_task(source, "task-b")
    exp = tmp_path / "exp1"
    monkeypatch.delenv("HERMES_HOME", raising=False)

    manifest = mod.prepare_experiment_workspace(
        experiment_dir=exp,
        source_skillsbench_root=source,
        task_ids=["task-a", "task-b"],
        reset_outputs=False,
        isolate_hermes_home=True,
        apply_hermes_home_env=True,
    )

    assert (exp / "skillsbench" / "tasks" / "task-a" / "instruction.md").is_file()
    assert (exp / "skillsbench" / "tasks" / "task-b" / "tests" / "test_dummy.py").is_file()
    assert (exp / "hermes_home" / "skills").is_dir()
    assert os.environ["HERMES_HOME"] == str(exp / "hermes_home")
    assert manifest["actions"]["task-a"] == "copied"
    assert manifest["skillsbench_root"] == str(exp / "skillsbench")
    assert (exp / "MANIFEST.json").is_file()
    loaded = json.loads((exp / "MANIFEST.json").read_text(encoding="utf-8"))
    assert loaded["schema"] == "skillsbench.experiment_workspace.v1"


def test_prepare_does_not_pollute_source_on_agent_write(tmp_path, monkeypatch):
    mod = _load_module()
    source = tmp_path / "skillsbench"
    _make_source_task(source, "task-a")
    exp = tmp_path / "exp2"
    monkeypatch.delenv("HERMES_HOME", raising=False)

    mod.prepare_experiment_workspace(
        experiment_dir=exp,
        source_skillsbench_root=source,
        task_ids=["task-a"],
        isolate_hermes_home=False,
        apply_hermes_home_env=False,
    )
    dest = exp / "skillsbench" / "tasks" / "task-a"
    (dest / "agent_output.txt").write_text("pollute\n", encoding="utf-8")
    (dest / "environment" / "skills").mkdir(parents=True)
    (dest / "environment" / "skills" / "evo").mkdir()
    (dest / "environment" / "skills" / "evo" / "SKILL.md").write_text("# skill\n")

    assert not (source / "tasks" / "task-a" / "agent_output.txt").exists()
    assert not (source / "tasks" / "task-a" / "environment" / "skills").exists()
    assert "HERMES_HOME" not in os.environ


def test_reset_outputs_wipes_agent_files_and_refreshes_defs(tmp_path, monkeypatch):
    mod = _load_module()
    source = tmp_path / "skillsbench"
    src_task = _make_source_task(source, "task-a")
    exp = tmp_path / "exp3"
    monkeypatch.delenv("HERMES_HOME", raising=False)

    mod.prepare_experiment_workspace(
        experiment_dir=exp,
        source_skillsbench_root=source,
        task_ids=["task-a"],
        isolate_hermes_home=True,
        apply_hermes_home_env=False,
    )
    dest = exp / "skillsbench" / "tasks" / "task-a"
    (dest / "junk.json").write_text("{}", encoding="utf-8")
    (src_task / "instruction.md").write_text("# updated\n", encoding="utf-8")

    manifest = mod.prepare_experiment_workspace(
        experiment_dir=exp,
        source_skillsbench_root=source,
        task_ids=["task-a"],
        reset_outputs=True,
        isolate_hermes_home=True,
        apply_hermes_home_env=False,
    )
    assert manifest["actions"]["task-a"] == "refreshed_outputs"
    assert not (dest / "junk.json").exists()
    assert (dest / "instruction.md").read_text(encoding="utf-8") == "# updated\n"


def test_clear_agent_outputs_keeps_definition(tmp_path):
    mod = _load_module()
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("x\n")
    (task / "environment").mkdir()
    (task / "out.bin").write_text("y\n")
    removed = mod.clear_agent_outputs(task)
    assert removed == ["out.bin"]
    assert (task / "instruction.md").is_file()
    assert (task / "environment").is_dir()
