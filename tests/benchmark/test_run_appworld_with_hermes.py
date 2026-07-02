"""Tests for benchmark/scripts/run_appworld_with_hermes.py helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "benchmark" / "scripts" / "run_appworld_with_hermes.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_appworld_with_hermes", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_extract_python_code_single_block():
    mod = _load_module()
    text = "Thought\n```python\nprint(apis.api_docs.show_app_descriptions())\n```\n"
    assert mod.extract_python_code(text) == "print(apis.api_docs.show_app_descriptions())"


def test_text_to_messages_roles():
    mod = _load_module()
    prompt = "USER:\nhello\n\nASSISTANT:\nworld\n"
    messages = mod.text_to_messages(prompt)
    assert messages == [
        {"role": "user", "content": "hello\n\n"},
        {"role": "assistant", "content": "world\n"},
    ]


def test_split_history_for_turn():
    mod = _load_module()
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    history, user_message = mod.split_history_for_turn(messages)
    assert history == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    assert user_message == "c"


def test_list_tasks_dev_subset(tmp_path, monkeypatch):
    mod = _load_module()
    appworld_root = tmp_path / "appworld"
    datasets = appworld_root / "data" / "datasets"
    tasks = appworld_root / "data" / "tasks"
    datasets.mkdir(parents=True)
    tasks.mkdir(parents=True)
    (datasets / "dev.txt").write_text("abc123_1\nabc123_2\n", encoding="utf-8")
    (tasks / "abc123_1").mkdir()
    (tasks / "abc123_2").mkdir()

    monkeypatch.setenv("APPWORLD_ROOT", str(appworld_root))
    ids = mod.discover_task_ids(dataset="dev", appworld_root=appworld_root)
    assert ids == ["abc123_1", "abc123_2"]


def test_validate_task_output_for_evaluation_missing_dir(tmp_path):
    mod = _load_module()
    appworld_root = tmp_path / "appworld"
    with pytest.raises(FileNotFoundError, match="No saved task output"):
        mod.validate_task_output_for_evaluation(
            appworld_root=appworld_root,
            experiment_name="hermes-agent",
            task_id="abc123_1",
        )


def test_evaluation_skip_reason_not_run(tmp_path):
    mod = _load_module()
    appworld_root = tmp_path / "appworld"
    assert (
        mod.evaluation_skip_reason(
            appworld_root=appworld_root,
            experiment_name="hermes-agent",
            task_id="abc123_1",
        )
        == "not run (no saved output)"
    )


def test_evaluation_skip_reason_ok_jsonl(tmp_path):
    mod = _load_module()
    appworld_root = tmp_path / "appworld"
    dbs = appworld_root / "experiments/outputs/hermes-agent/tasks/abc123_1/dbs"
    dbs.mkdir(parents=True)
    (dbs / "supervisor.jsonl").write_text('{"change": 1}\n', encoding="utf-8")
    assert (
        mod.evaluation_skip_reason(
            appworld_root=appworld_root,
            experiment_name="hermes-agent",
            task_id="abc123_1",
        )
        is None
    )


def test_validate_task_output_for_evaluation_empty_dbs(tmp_path):
    mod = _load_module()
    appworld_root = tmp_path / "appworld"
    dbs = appworld_root / "experiments/outputs/hermes-agent/tasks/abc123_1/dbs"
    dbs.mkdir(parents=True)
    (dbs / "supervisor.db").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="empty"):
        mod.validate_task_output_for_evaluation(
            appworld_root=appworld_root,
            experiment_name="hermes-agent",
            task_id="abc123_1",
        )


def test_validate_task_output_for_evaluation_db_ok(tmp_path):
    mod = _load_module()
    appworld_root = tmp_path / "appworld"
    dbs = appworld_root / "experiments/outputs/hermes-agent/tasks/abc123_1/dbs"
    dbs.mkdir(parents=True)
    (dbs / "supervisor.db").write_bytes(b"sqlite")
    mod.validate_task_output_for_evaluation(
        appworld_root=appworld_root,
        experiment_name="hermes-agent",
        task_id="abc123_1",
    )


def test_validate_task_output_for_evaluation_jsonl_ok(tmp_path):
    mod = _load_module()
    appworld_root = tmp_path / "appworld"
    dbs = appworld_root / "experiments/outputs/hermes-agent/tasks/abc123_1/dbs"
    dbs.mkdir(parents=True)
    (dbs / "gmail.jsonl").write_bytes(b"")
    (dbs / "supervisor.jsonl").write_text('{"change": 1}\n', encoding="utf-8")
    mod.validate_task_output_for_evaluation(
        appworld_root=appworld_root,
        experiment_name="hermes-agent",
        task_id="abc123_1",
    )


def test_validate_task_output_for_evaluation_jsonl_all_empty(tmp_path):
    mod = _load_module()
    appworld_root = tmp_path / "appworld"
    dbs = appworld_root / "experiments/outputs/hermes-agent/tasks/abc123_1/dbs"
    dbs.mkdir(parents=True)
    (dbs / "gmail.jsonl").write_bytes(b"")
    (dbs / "supervisor.jsonl").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="empty"):
        mod.validate_task_output_for_evaluation(
            appworld_root=appworld_root,
            experiment_name="hermes-agent",
            task_id="abc123_1",
        )


def test_resolve_max_hermes_iterations_defaults():
    mod = _load_module()
    assert mod.resolve_max_hermes_iterations(None, no_tools=False) == 90
    assert mod.resolve_max_hermes_iterations(None, no_tools=True) == 1
    assert mod.resolve_max_hermes_iterations(5, no_tools=False) == 5
    assert mod.resolve_max_hermes_iterations(0, no_tools=True) == 1


def test_restore_stdio_for_appworld_unwraps_safe_writer():
    mod = _load_module()
    from run_agent import _SafeWriter

    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    try:
        inner = sys.__stdout__ or saved_stdout
        sys.stdout = _SafeWriter(inner)
        mod._restore_stdio_for_appworld()
        assert sys.stdout is sys.__stdout__
        assert not isinstance(sys.stdout, _SafeWriter)
    finally:
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr


def test_load_run_hermes_stats_by_task(tmp_path):
    mod = _load_module()
    log = tmp_path / "runs.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema": "appworld.hermes_run.v1",
                        "appworld_task_id": "t1",
                        "experiment_name": "hermes-dev",
                        "hermes_stats": {"total_tokens": 100},
                    }
                ),
                json.dumps(
                    {
                        "schema": "appworld.hermes_run.v1",
                        "appworld_task_id": "t1",
                        "experiment_name": "hermes-dev",
                        "hermes_stats": {"total_tokens": 200},
                    }
                ),
                json.dumps(
                    {
                        "schema": "appworld.hermes_eval.v1",
                        "appworld_task_id": "t1",
                        "experiment_name": "hermes-dev",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stats = mod.load_run_hermes_stats_by_task(log, experiment_name="hermes-dev")
    assert stats == {"t1": {"total_tokens": 200}}


def test_print_evaluate_batch_summary(capsys):
    mod = _load_module()
    mod.print_evaluate_batch_summary(
        dataset="dev",
        experiment_name="hermes-dev",
        tasks_total=10,
        evaluated=2,
        skipped=8,
        errors=0,
        task_successes=1,
        pass_count=3,
        num_tests=4,
        evaluated_task_ids=["a", "b"],
        run_stats_by_task={
            "a": {"total_tokens": 1000},
            "b": {"total_tokens": 500},
        },
    )
    out = capsys.readouterr().out
    assert "task success: 1/2 (50.0%)" in out
    assert "test passes: 3/4 (75.0%)" in out
    assert "tokens: 1,500 total" in out
    assert "750 avg" in out
