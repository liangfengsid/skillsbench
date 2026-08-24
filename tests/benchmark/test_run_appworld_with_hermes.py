"""Tests for benchmark/scripts/run_appworld_with_hermes.py helpers."""

from __future__ import annotations

import importlib.util
import json
import os
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


def test_finished_task_ids_resume(tmp_path):
    mod = _load_module()
    log = tmp_path / "runs.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "appworld_task_id": "a",
                        "evaluation": {"task_success": False, "success": False},
                    }
                ),
                json.dumps(
                    {
                        "appworld_task_id": "b",
                        "evaluation": {"task_success": True, "success": True},
                    }
                ),
                json.dumps({"appworld_task_id": "a", "evaluation": {"success": True}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert mod._finished_task_ids(log) == {"a", "b"}
    assert mod._finished_task_ids(log, successes_only=True) == {"a", "b"}


def test_script_documents_hot_pool_cli_flags():
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "--hot-pool" in text
    assert "--isolate-hermes-home" in text
    assert "--experiment-dir" in text
    assert "--amem" in text
    assert "apply_hot_pool_outcome_feedback" in text
    assert "task_success" in text


def test_append_jsonl_survives_appworld_safety_guard(tmp_path, monkeypatch):
    """Host JSONL logging must work after AppWorld patches Path.mkdir / open."""
    import builtins

    mod = _load_module()
    log = tmp_path / "exp" / "runs.jsonl"

    def blocked_mkdir(self, *args, **kwargs):
        raise PermissionError(
            "Usage of the following function is not allowed: pathlib.Path.mkdir."
        )

    real_open = builtins.open

    def read_only_open(file, mode="r", *args, **kwargs):
        if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
            raise PermissionError("Writing to OS file system is disabled.")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", blocked_mkdir)
    monkeypatch.setattr(builtins, "open", read_only_open)

    mod.append_jsonl(log, {"schema": "appworld.hermes_run.v1", "ok": True})
    rows = log.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(rows[0])["ok"] is True

    mod.append_jsonl(log, {"schema": "appworld.hermes_run.v1", "ok": False})
    rows = log.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(r)["ok"] for r in rows] == [True, False]


def test_restore_host_os_io_allows_environ_after_putenv_guard(monkeypatch):
    """--no-hot-pool sets os.environ; that must work after SafetyGuard leaks."""
    mod = _load_module()

    def blocked_putenv(*args, **kwargs):
        raise PermissionError(
            "Usage of the following function is not allowed: os.putenv."
        )

    monkeypatch.setattr(os, "putenv", blocked_putenv)
    with pytest.raises(PermissionError, match="os.putenv"):
        os.environ["_HERMES_TEST_HOT_POOL"] = "0"

    mod._restore_host_os_io()
    mod.apply_hot_pool_cli_overrides(hot_pool=False, hot_pool_persist=None)
    assert os.environ.get("HERMES_HOT_POOL_ENABLED") == "0"


def test_restore_host_os_io_allows_hot_pool_enable_environ(tmp_path, monkeypatch):
    """--hot-pool also assigns os.environ (putenv) after a leaked SafetyGuard."""
    mod = _load_module()
    persist = tmp_path / "hot_pool.json"

    def blocked_putenv(*args, **kwargs):
        raise PermissionError(
            "Usage of the following function is not allowed: os.putenv."
        )

    monkeypatch.setattr(os, "putenv", blocked_putenv)
    with pytest.raises(PermissionError, match="os.putenv"):
        os.environ["HERMES_HOT_POOL_ENABLED"] = "1"

    mod._restore_host_os_io()
    mod.apply_hot_pool_cli_overrides(hot_pool=True, hot_pool_persist=str(persist))
    assert os.environ.get("HERMES_HOT_POOL_ENABLED") == "1"
    assert os.environ.get("HERMES_HOT_POOL_PATH") == str(persist.resolve())


def test_apply_hot_pool_persist_avoids_path_expanduser(tmp_path, monkeypatch):
    """--hot-pool persist must not call Path.expanduser (SafetyGuard denylist)."""
    mod = _load_module()
    persist = tmp_path / "hot_pool.json"

    def blocked_expanduser(self):
        raise PermissionError(
            "Usage of the following function is not allowed: pathlib.Path.expanduser."
        )

    monkeypatch.setattr(Path, "expanduser", blocked_expanduser)
    with pytest.raises(PermissionError, match="expanduser"):
        Path(persist).expanduser()

    # No restore — helper uses os.path.expanduser.
    mod.apply_hot_pool_cli_overrides(hot_pool=True, hot_pool_persist=str(persist))
    assert os.environ.get("HERMES_HOT_POOL_ENABLED") == "1"
    assert os.environ.get("HERMES_HOT_POOL_PATH") == str(persist.resolve())


def test_restore_host_os_io_allows_expanduser_after_path_guard(tmp_path, monkeypatch):
    """hot_skills.resolve_hot_pool_persist_path still uses Path.expanduser."""
    mod = _load_module()
    persist = tmp_path / "hot_pool.json"

    def blocked_expanduser(self):
        raise PermissionError(
            "Usage of the following function is not allowed: pathlib.Path.expanduser."
        )

    monkeypatch.setattr(Path, "expanduser", blocked_expanduser)
    with pytest.raises(PermissionError, match="expanduser"):
        Path(persist).expanduser()

    mod._restore_host_os_io()
    resolved = Path(persist).expanduser()
    assert resolved == persist or resolved == persist.resolve()


def test_restore_host_os_io_allows_hot_pool_persist_writes(tmp_path, monkeypatch):
    """Hot-pool persist uses Path.mkdir / write_text / replace after execute()."""
    mod = _load_module()

    def blocked(name):
        def _blocked(self, *args, **kwargs):
            raise PermissionError(
                f"Usage of the following function is not allowed: pathlib.Path.{name}."
            )

        return _blocked

    monkeypatch.setattr(Path, "mkdir", blocked("mkdir"))
    monkeypatch.setattr(Path, "write_text", blocked("write_text"))
    monkeypatch.setattr(Path, "replace", blocked("replace"))
    with pytest.raises(PermissionError, match="mkdir"):
        (tmp_path / "blocked").mkdir()

    mod._restore_host_os_io()
    path = tmp_path / "pool" / "hot_pool.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("{}", encoding="utf-8")
    tmp.replace(path)
    assert path.read_text(encoding="utf-8") == "{}"


def test_restore_host_os_io_allows_shutil_rmtree_and_json_dump(tmp_path, monkeypatch):
    """--clear-experiment and eval reports use shutil.rmtree / json.dump."""
    import json as json_mod
    import shutil as shutil_mod

    mod = _load_module()
    target = tmp_path / "outputs"
    target.mkdir()
    report = tmp_path / "report.json"

    def blocked_rmtree(*args, **kwargs):
        raise PermissionError(
            "Usage of the following function is not allowed: shutil.rmtree."
        )

    def blocked_dump(*args, **kwargs):
        raise PermissionError(
            "Usage of the following function is not allowed: json.dump."
        )

    monkeypatch.setattr(shutil_mod, "rmtree", blocked_rmtree)
    monkeypatch.setattr(json_mod, "dump", blocked_dump)
    with pytest.raises(PermissionError, match="rmtree"):
        shutil_mod.rmtree(target)
    with pytest.raises(PermissionError, match="json.dump"):
        with report.open("w", encoding="utf-8") as f:
            json_mod.dump({"ok": True}, f)

    mod._restore_host_os_io()
    shutil_mod.rmtree(target)
    assert not target.exists()
    with report.open("w", encoding="utf-8") as f:
        json_mod.dump({"ok": True}, f)
    assert json.loads(report.read_text(encoding="utf-8")) == {"ok": True}


def test_host_snapshot_covers_appworld_safety_guard_targets():
    """Driver restore list must be a superset of AppWorld's process-wide patches."""
    safety_guard = pytest.importorskip("appworld.common.safety_guard")
    mod = _load_module()
    snapped = {(id(obj), name) for obj, name, _ in mod._HOST_ATTRS}
    missing = []
    for module_name, names in safety_guard.DISALLOWED_MODULE_TO_FUNCTION_NAMES.items():
        try:
            module = safety_guard.SafetyGuard.module_by_path(module_name)
        except Exception:
            continue
        for name in names:
            if getattr(module, name, None) is None:
                continue
            if (id(module), name) not in snapped:
                missing.append(f"{module_name}.{name}")
    assert missing == []


def test_execute_world_code_turns_timeout_into_step_output():
    mod = _load_module()

    class _World:
        def execute(self, code):
            raise TimeoutError("Function run_cell execution timed out after 100 seconds.")

    out = mod._execute_world_code(_World(), "print(1)")
    assert "TimeoutError" in out
    assert "timed out" in out


def test_execute_world_code_restores_stdio_before_and_after():
    mod = _load_module()

    class _FakeWrapper:
        """Minimal stand-in for _SafeWriter (avoids importing run_agent)."""

        __slots__ = ("_inner",)

        def __init__(self, inner):
            self._inner = inner

        def write(self, data):
            return self._inner.write(data)

    calls: list[str] = []
    real_stdout = sys.__stdout__ or sys.stdout
    real_stderr = sys.__stderr__ or sys.stderr

    class _World:
        def execute(self, code):
            calls.append("execute")
            assert sys.stdout is real_stdout
            assert sys.stderr is real_stderr
            return "ok"

    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = _FakeWrapper(real_stdout)
        sys.stderr = _FakeWrapper(real_stderr)
        out = mod._execute_world_code(_World(), "print(1)")
        assert out == "ok"
        assert calls == ["execute"]
        assert sys.stdout is real_stdout
        assert sys.stderr is real_stderr
    finally:
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr


def test_wait_background_review_joins_agent():
    mod = _load_module()
    calls = []

    class _Agent:
        def wait_for_background_review(self, timeout=60.0):
            calls.append(timeout)

    mod._wait_background_review(_Agent(), timeout=12.0)
    assert calls == [12.0]
    mod._wait_background_review(None)


def test_task_worker_crash_envelope_mentions_segfault():
    mod = _load_module()
    env = mod._task_worker_crash_envelope(
        task_id="2a163ab_3",
        dataset="train",
        experiment_name="exp",
        returncode=139,
    )
    assert env["schema"] == "appworld.hermes_run_error.v1"
    assert env["appworld_task_id"] == "2a163ab_3"
    assert "SIGSEGV" in env["error"]


def test_run_one_task_in_subprocess_survives_child_crash(tmp_path, monkeypatch):
    mod = _load_module()

    class _Proc:
        returncode = 139

    class _FakeSubprocess:
        @staticmethod
        def run(*args, **kwargs):
            return _Proc()

    monkeypatch.setattr(mod, "subprocess", _FakeSubprocess)
    envelope = mod.run_one_task_in_subprocess(
        task_id="2a163ab_3",
        dataset="train",
        experiment_name="exp",
        hermes_root=tmp_path,
        appworld_root=tmp_path,
        prompt_file=tmp_path / "p.txt",
        model="m",
        max_steps=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        evaluate_after_run=False,
    )
    assert envelope["schema"] == "appworld.hermes_run_error.v1"
    assert envelope["worker_returncode"] == 139
