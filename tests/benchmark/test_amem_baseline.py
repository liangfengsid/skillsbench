"""Tests for A-Mem benchmark CLI helpers (no chromadb)."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "benchmark" / "scripts" / "amem_baseline.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("amem_baseline", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_apply_amem_cli_overrides_sets_and_clears(tmp_path, monkeypatch):
    mod = _load_module()
    persist = tmp_path / "amem"
    monkeypatch.setenv("HERMES_AMEM_ENABLED", "stale")
    mod.apply_amem_cli_overrides(
        enabled=True,
        persist_dir=str(persist),
        k=7,
        sync_every=5,
        llm_model="Qwen/Qwen3.6-27B",
        api_key="sk-test",
        base_url="http://localhost:8000/v1",
    )
    assert os.environ["HERMES_AMEM_ENABLED"] == "1"
    assert os.environ["HERMES_MEMORY_PROVIDER"] == "amem"
    assert os.environ["HERMES_AMEM_PATH"] == str(persist.resolve())
    assert persist.is_dir()
    assert os.environ["HERMES_AMEM_K"] == "7"
    assert os.environ["HERMES_AMEM_SYNC_EVERY"] == "5"
    assert os.environ["HERMES_AMEM_LLM_MODEL"] == "Qwen/Qwen3.6-27B"
    assert os.environ["HERMES_AMEM_API_KEY"] == "sk-test"
    assert os.environ["HERMES_AMEM_API_BASE"] == "http://localhost:8000/v1"

    mod.apply_amem_cli_overrides(enabled=False)
    assert "HERMES_AMEM_ENABLED" not in os.environ
    assert "HERMES_MEMORY_PROVIDER" not in os.environ
    assert "HERMES_AMEM_PATH" not in os.environ


def test_resolve_amem_persist_prefers_explicit_then_experiment(tmp_path):
    mod = _load_module()
    exp = tmp_path / "exp"
    exp.mkdir()
    explicit = tmp_path / "custom"
    assert mod.resolve_amem_persist(
        enabled=False, persist=str(explicit), experiment_dir=exp
    ) is None
    got = mod.resolve_amem_persist(
        enabled=True, persist=str(explicit), experiment_dir=exp
    )
    assert Path(got) == explicit.resolve()
    got = mod.resolve_amem_persist(enabled=True, persist=None, experiment_dir=exp)
    assert Path(got) == (exp / "amem").resolve()


def test_apply_amem_argparse_policy_rejects_hot_pool():
    mod = _load_module()
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(amem=True, hot_pool=True, skip_memory=False)
    with pytest.raises(SystemExit):
        mod.apply_amem_argparse_policy(parser, args)


def test_apply_amem_argparse_policy_rejects_dc():
    mod = _load_module()
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(amem=True, dc=True, hot_pool=None, skip_memory=False)
    with pytest.raises(SystemExit):
        mod.apply_amem_argparse_policy(parser, args)


def test_apply_amem_argparse_policy_forces_no_hot_pool_and_skip_memory():
    mod = _load_module()
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(amem=True, hot_pool=None, skip_memory=False)
    mod.apply_amem_argparse_policy(parser, args)
    assert args.hot_pool is False
    assert args.skip_memory is True


def test_amem_telemetry_from_agent_reads_provider():
    mod = _load_module()
    provider = SimpleNamespace(
        name="amem",
        export_telemetry=lambda: {"n_notes": 4, "n_syncs": 2},
    )
    agent = SimpleNamespace(
        _memory_manager=SimpleNamespace(providers=[provider]),
    )
    assert mod.amem_telemetry_from_agent(agent) == {"n_notes": 4, "n_syncs": 2}
    assert mod.amem_telemetry_from_agent(SimpleNamespace()) == {}


def test_add_amem_cli_flags_on_parser():
    mod = _load_module()
    parser = argparse.ArgumentParser()
    mod.add_amem_cli_flags(parser)
    ns = parser.parse_args(
        ["--amem", "--amem-k", "3", "--amem-persist", "/tmp/x", "--amem-sync-every", "5"]
    )
    assert ns.amem is True
    assert ns.amem_k == 3
    assert ns.amem_persist == "/tmp/x"
    assert ns.amem_sync_every == 5


def test_skillsbench_driver_wires_amem_flags():
    text = (
        Path(__file__).resolve().parents[2]
        / "benchmark"
        / "scripts"
        / "run_skillsbench_with_hermes.py"
    ).read_text(encoding="utf-8")
    assert "add_amem_cli_flags" in text
    assert "apply_amem_cli_overrides" in text
    assert "amem_telemetry" in text


def test_alfworld_and_appworld_drivers_wire_amem_flags():
    root = Path(__file__).resolve().parents[2] / "benchmark" / "scripts"
    for name in ("run_alfworld_with_hermes.py", "run_appworld_with_hermes.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "add_amem_cli_flags" in text
        assert "apply_amem_cli_overrides" in text
        assert '"amem":' in text or "amem=bool" in text or "amem=bool(" in text
        assert "amem_telemetry" in text
