"""Tests for Dynamic Cheatsheet benchmark CLI helpers (no live LLM)."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "benchmark" / "scripts" / "dc_baseline.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("dc_baseline", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_apply_dc_cli_overrides_sets_and_clears(tmp_path, monkeypatch):
    mod = _load_module()
    persist = tmp_path / "dcheatsheet"
    monkeypatch.setenv("HERMES_DC_ENABLED", "stale")
    mod.apply_dc_cli_overrides(
        enabled=True,
        persist_dir=str(persist),
        mode="rs",
        k=4,
        sync_every=5,
        llm_model="Qwen/Qwen3.6-27B",
        api_key="sk-test",
        base_url="http://localhost:8000/v1",
    )
    assert os.environ["HERMES_DC_ENABLED"] == "1"
    assert os.environ["HERMES_MEMORY_PROVIDER"] == "dcheatsheet"
    assert os.environ["HERMES_DC_PATH"] == str(persist.resolve())
    assert persist.is_dir()
    assert os.environ["HERMES_DC_MODE"] == "rs"
    assert os.environ["HERMES_DC_K"] == "4"
    assert os.environ["HERMES_DC_SYNC_EVERY"] == "5"
    assert os.environ["HERMES_DC_LLM_MODEL"] == "Qwen/Qwen3.6-27B"
    assert os.environ["HERMES_DC_API_KEY"] == "sk-test"
    assert os.environ["HERMES_DC_API_BASE"] == "http://localhost:8000/v1"

    mod.apply_dc_cli_overrides(enabled=False)
    assert "HERMES_DC_ENABLED" not in os.environ
    assert "HERMES_MEMORY_PROVIDER" not in os.environ
    assert "HERMES_DC_PATH" not in os.environ
    assert "HERMES_DC_SYNC_EVERY" not in os.environ


def test_apply_dc_cli_overrides_freeze_sets_readonly(tmp_path, monkeypatch):
    mod = _load_module()
    persist = tmp_path / "dcheatsheet"
    mod.apply_dc_cli_overrides(
        enabled=True,
        persist_dir=str(persist),
        k=3,
        sync_every=5,
        freeze=True,
    )
    assert os.environ["HERMES_DC_READONLY"] == "1"
    assert "HERMES_DC_SYNC_EVERY" not in os.environ

    mod.apply_dc_cli_overrides(
        enabled=True,
        persist_dir=str(persist),
        freeze=False,
        sync_every=3,
    )
    assert "HERMES_DC_READONLY" not in os.environ
    assert os.environ["HERMES_DC_SYNC_EVERY"] == "3"


def test_apply_dc_argparse_policy_rejects_freeze_without_dc():
    mod = _load_module()
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(dc=False, dc_freeze=True, amem=False, hot_pool=None)
    with pytest.raises(SystemExit):
        mod.apply_dc_argparse_policy(parser, args)


def test_apply_dc_cli_overrides_does_not_wipe_amem_provider(monkeypatch):
    mod = _load_module()
    monkeypatch.setenv("HERMES_MEMORY_PROVIDER", "amem")
    mod.apply_dc_cli_overrides(enabled=False)
    assert os.environ["HERMES_MEMORY_PROVIDER"] == "amem"


def test_resolve_dc_persist_prefers_explicit_then_experiment(tmp_path):
    mod = _load_module()
    exp = tmp_path / "exp"
    exp.mkdir()
    explicit = tmp_path / "custom"
    assert mod.resolve_dc_persist(
        enabled=False, persist=str(explicit), experiment_dir=exp
    ) is None
    got = mod.resolve_dc_persist(
        enabled=True, persist=str(explicit), experiment_dir=exp
    )
    assert Path(got) == explicit.resolve()
    got = mod.resolve_dc_persist(enabled=True, persist=None, experiment_dir=exp)
    assert Path(got) == (exp / "dcheatsheet").resolve()


def test_apply_dc_argparse_policy_rejects_hot_pool():
    mod = _load_module()
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(dc=True, amem=False, hot_pool=True, skip_memory=False)
    with pytest.raises(SystemExit):
        mod.apply_dc_argparse_policy(parser, args)


def test_apply_dc_argparse_policy_rejects_amem():
    mod = _load_module()
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(dc=True, amem=True, hot_pool=None, skip_memory=False)
    with pytest.raises(SystemExit):
        mod.apply_dc_argparse_policy(parser, args)


def test_apply_dc_argparse_policy_forces_no_hot_pool_and_skip_memory():
    mod = _load_module()
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(dc=True, amem=False, hot_pool=None, skip_memory=False)
    mod.apply_dc_argparse_policy(parser, args)
    assert args.hot_pool is False
    assert args.skip_memory is True


def test_dc_telemetry_from_agent_reads_provider():
    mod = _load_module()
    provider = SimpleNamespace(
        name="dcheatsheet",
        export_telemetry=lambda: {"n_episodes": 4, "n_curations": 2},
    )
    agent = SimpleNamespace(
        _memory_manager=SimpleNamespace(providers=[provider]),
    )
    assert mod.dc_telemetry_from_agent(agent) == {"n_episodes": 4, "n_curations": 2}
    assert mod.dc_telemetry_from_agent(SimpleNamespace()) == {}


def test_add_dc_cli_flags_on_parser():
    mod = _load_module()
    parser = argparse.ArgumentParser()
    mod.add_dc_cli_flags(parser)
    ns = parser.parse_args(
        [
            "--dc",
            "--dc-k",
            "2",
            "--dc-persist",
            "/tmp/x",
            "--dc-mode",
            "rs",
            "--dc-sync-every",
            "5",
            "--dc-freeze",
        ]
    )
    assert ns.dc is True
    assert ns.dc_k == 2
    assert ns.dc_persist == "/tmp/x"
    assert ns.dc_mode == "rs"
    assert ns.dc_sync_every == 5
    assert ns.dc_freeze is True


def test_skillsbench_driver_wires_dc_flags():
    text = (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "scripts"
        / "run_skillsbench_with_hermes.py"
    ).read_text(encoding="utf-8")
    assert "add_dc_cli_flags" in text
    assert "apply_dc_cli_overrides" in text
    assert "dc_telemetry" in text
    assert "dc_freeze" in text
    assert "dc_sync_every" in text


def test_alfworld_and_appworld_drivers_wire_dc_flags():
    root = Path(__file__).resolve().parents[2] / "benchmark" / "scripts"
    for name in ("run_alfworld_with_hermes.py", "run_appworld_with_hermes.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "add_dc_cli_flags" in text
        assert "apply_dc_cli_overrides" in text
        assert "dc_telemetry" in text
        assert "dc=bool" in text or '"dc":' in text
        assert "dc_freeze" in text
        assert "freeze=bool(dc_freeze)" in text or "freeze=bool(args.dc_freeze)" in text
