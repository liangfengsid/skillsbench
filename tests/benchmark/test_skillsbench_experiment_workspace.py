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


def test_prepare_default_keeps_real_hermes_home(tmp_path, monkeypatch):
    """--experiment-dir isolates task trees only; Hermes defaults stay in ~/.hermes."""
    mod = _load_module()
    source = tmp_path / "skillsbench"
    _make_source_task(source, "task-a")
    exp = tmp_path / "exp_default"
    monkeypatch.delenv("HERMES_HOME", raising=False)

    manifest = mod.prepare_experiment_workspace(
        experiment_dir=exp,
        source_skillsbench_root=source,
        task_ids=["task-a"],
    )

    assert (exp / "skillsbench" / "tasks" / "task-a" / "instruction.md").is_file()
    assert not (exp / "hermes_home").exists()
    assert "HERMES_HOME" not in os.environ
    assert manifest["isolate_hermes_home"] is False
    assert manifest["hermes_home"] is None


def test_prepare_empty_task_ids_still_creates_experiment_dir(tmp_path, monkeypatch):
    """CoEvo --evolve with --experiment-dir passes task_ids=[] (no task copies)."""
    mod = _load_module()
    source = tmp_path / "skillsbench"
    source.mkdir()
    exp = tmp_path / "coevo_exp1"
    monkeypatch.delenv("HERMES_HOME", raising=False)

    manifest = mod.prepare_experiment_workspace(
        experiment_dir=exp,
        source_skillsbench_root=source,
        task_ids=[],
    )

    assert exp.is_dir()
    assert (exp / "MANIFEST.json").is_file()
    assert manifest["task_ids"] == []
    assert manifest["skillsbench_root"] is None
    assert not (exp / "hermes_home").exists()


def test_prepare_copies_tasks_and_sets_hermes_home(tmp_path, monkeypatch):
    mod = _load_module()
    source = tmp_path / "skillsbench"
    _make_source_task(source, "task-a")
    _make_source_task(source, "task-b")
    real_home = tmp_path / "real_hermes"
    (real_home / "skills" / "demo-skill").mkdir(parents=True)
    (real_home / "skills" / "demo-skill" / "SKILL.md").write_text(
        "---\nname: demo-skill\n---\n# Demo\n", encoding="utf-8"
    )
    (real_home / "config.yaml").write_text("model:\n  default: demo\n", encoding="utf-8")
    (real_home / ".env").write_text("DEMO_KEY=1\n", encoding="utf-8")
    (real_home / "SOUL.md").write_text("# Soul\n", encoding="utf-8")
    exp = tmp_path / "exp1"
    monkeypatch.delenv("HERMES_HOME", raising=False)

    manifest = mod.prepare_experiment_workspace(
        experiment_dir=exp,
        source_skillsbench_root=source,
        task_ids=["task-a", "task-b"],
        reset_outputs=False,
        isolate_hermes_home=True,
        apply_hermes_home_env=True,
        source_hermes_home=real_home,
    )

    assert (exp / "skillsbench" / "tasks" / "task-a" / "instruction.md").is_file()
    assert (exp / "skillsbench" / "tasks" / "task-b" / "tests" / "test_dummy.py").is_file()
    seeded = exp / "hermes_home" / "skills" / "demo-skill" / "SKILL.md"
    assert seeded.is_file()
    assert "Demo" in seeded.read_text(encoding="utf-8")
    assert os.environ["HERMES_HOME"] == str(exp / "hermes_home")
    assert manifest["actions"]["task-a"] == "copied"
    assert manifest["skills_seed"]["action"] == "seeded"
    assert manifest["skills_seed"]["n_skills"] == 1
    assert manifest["skillsbench_root"] == str(exp / "skillsbench")
    for name in ("config.yaml", ".env", "SOUL.md"):
        link = exp / "hermes_home" / name
        assert link.is_symlink()
        assert link.resolve() == (real_home / name).resolve()
        assert manifest["shared_hermes_files"]["files"][name]["mode"] == "symlink"
    assert (exp / "MANIFEST.json").is_file()
    loaded = json.loads((exp / "MANIFEST.json").read_text(encoding="utf-8"))
    assert loaded["schema"] == "skillsbench.experiment_workspace.v1"


def test_shared_files_do_not_sandbox_config_writes_via_symlink(tmp_path, monkeypatch):
    """Symlinked config is the real file — document write-through behavior."""
    mod = _load_module()
    source = tmp_path / "skillsbench"
    _make_source_task(source, "task-a")
    real_home = tmp_path / "real_hermes"
    (real_home / "skills").mkdir(parents=True)
    (real_home / "config.yaml").write_text("a: 1\n", encoding="utf-8")
    exp = tmp_path / "exp_shared"
    monkeypatch.delenv("HERMES_HOME", raising=False)

    mod.prepare_experiment_workspace(
        experiment_dir=exp,
        source_skillsbench_root=source,
        task_ids=["task-a"],
        isolate_hermes_home=True,
        source_hermes_home=real_home,
    )
    (exp / "hermes_home" / "config.yaml").write_text("a: 2\n", encoding="utf-8")
    assert (real_home / "config.yaml").read_text(encoding="utf-8") == "a: 2\n"
    # Skills remain sandboxed.
    (exp / "hermes_home" / "skills" / "x").mkdir()
    (exp / "hermes_home" / "skills" / "x" / "SKILL.md").write_text("# x\n")
    assert not (real_home / "skills" / "x").exists()


def test_isolate_preserves_seeded_skills_across_reruns(tmp_path, monkeypatch):
    mod = _load_module()
    source = tmp_path / "skillsbench"
    _make_source_task(source, "task-a")
    real_home = tmp_path / "real_hermes"
    (real_home / "skills" / "base").mkdir(parents=True)
    (real_home / "skills" / "base" / "SKILL.md").write_text("# base\n")
    exp = tmp_path / "exp_resume"
    monkeypatch.delenv("HERMES_HOME", raising=False)

    mod.prepare_experiment_workspace(
        experiment_dir=exp,
        source_skillsbench_root=source,
        task_ids=["task-a"],
        isolate_hermes_home=True,
        source_hermes_home=real_home,
    )
    # Experiment mutates skills (hot-pool / skill_manage).
    new_skill = exp / "hermes_home" / "skills" / "learned"
    new_skill.mkdir()
    (new_skill / "SKILL.md").write_text("# learned\n")

    manifest = mod.prepare_experiment_workspace(
        experiment_dir=exp,
        source_skillsbench_root=source,
        task_ids=["task-a"],
        isolate_hermes_home=True,
        source_hermes_home=real_home,
    )
    assert manifest["skills_seed"]["action"] == "exists"
    assert (new_skill / "SKILL.md").is_file()


def test_reset_reseeds_hermes_skills(tmp_path, monkeypatch):
    mod = _load_module()
    source = tmp_path / "skillsbench"
    _make_source_task(source, "task-a")
    real_home = tmp_path / "real_hermes"
    (real_home / "skills" / "base").mkdir(parents=True)
    (real_home / "skills" / "base" / "SKILL.md").write_text("# base\n")
    exp = tmp_path / "exp_reset"
    monkeypatch.delenv("HERMES_HOME", raising=False)

    mod.prepare_experiment_workspace(
        experiment_dir=exp,
        source_skillsbench_root=source,
        task_ids=["task-a"],
        isolate_hermes_home=True,
        source_hermes_home=real_home,
    )
    learned = exp / "hermes_home" / "skills" / "learned"
    learned.mkdir()
    (learned / "SKILL.md").write_text("# learned\n")
    (real_home / "skills" / "base" / "SKILL.md").write_text("# base-v2\n")

    manifest = mod.prepare_experiment_workspace(
        experiment_dir=exp,
        source_skillsbench_root=source,
        task_ids=["task-a"],
        reset_outputs=True,
        isolate_hermes_home=True,
        source_hermes_home=real_home,
    )
    assert manifest["skills_seed"]["action"] == "reseeded"
    assert not learned.exists()
    assert (exp / "hermes_home" / "skills" / "base" / "SKILL.md").read_text() == "# base-v2\n"


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
    real_home = tmp_path / "real_hermes"
    (real_home / "skills").mkdir(parents=True)
    exp = tmp_path / "exp3"
    monkeypatch.delenv("HERMES_HOME", raising=False)

    mod.prepare_experiment_workspace(
        experiment_dir=exp,
        source_skillsbench_root=source,
        task_ids=["task-a"],
        isolate_hermes_home=True,
        apply_hermes_home_env=False,
        source_hermes_home=real_home,
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
        source_hermes_home=real_home,
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
