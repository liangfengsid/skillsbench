"""Unit tests for the A-Mem memory provider (no chromadb / agentic-memory)."""

from __future__ import annotations

import pytest

from plugins.memory.amem import AMemMemoryProvider
from plugins.memory.amem import store as amem_store


class _FakeStore:
    def __init__(self, persist_dir="/tmp/amem-fake"):
        self.persist_dir = persist_dir
        self.notes = []
        self.queries = []
        self.persisted = 0

    @property
    def n_notes(self) -> int:
        return len(self.notes)

    def search(self, query, k=None):
        self.queries.append(query)
        if "fridge" in (query or "").lower() or "apple" in (query or "").lower():
            return [
                {
                    "id": "n1",
                    "content": "Prior success: go to fridge then take apple.",
                }
            ]
        return []

    def add_note(self, content, **kwargs):
        self.notes.append({"content": content, **kwargs})
        return f"id{len(self.notes)}"

    def persist(self):
        self.persisted += 1


@pytest.fixture
def fake_store():
    fake = _FakeStore()
    amem_store._OVERRIDE_STORE = fake
    amem_store._SINGLETON = None
    yield fake
    amem_store.reset_store_for_tests()


@pytest.fixture(autouse=True)
def _clean_amem_env(monkeypatch):
    for key in (
        "HERMES_AMEM_ENABLED",
        "HERMES_AMEM_PATH",
        "HERMES_AMEM_K",
        "HERMES_AMEM_SYNC_EVERY",
        "HERMES_AMEM_READONLY",
        "HERMES_MEMORY_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)


def test_get_store_uses_override(fake_store):
    assert amem_store.get_store() is fake_store


def test_prefetch_injects_notes(fake_store, monkeypatch):
    monkeypatch.setenv("HERMES_AMEM_ENABLED", "1")
    provider = AMemMemoryProvider()
    provider.initialize("sess-1")
    text = provider.prefetch("How do I get the apple from the fridge?")
    assert "go to fridge" in text
    assert "Prior success" in text
    assert fake_store.queries
    tel = provider.export_telemetry()
    assert tel["last_prefetch"]["n_retrieved"] == 1
    assert tel["n_notes"] == 0


def test_prefetch_empty_when_no_hits(fake_store):
    provider = AMemMemoryProvider()
    provider.initialize("sess-1")
    assert provider.prefetch("unrelated query xyz") == ""


def test_sync_turn_buffers_until_episode_flush(fake_store, monkeypatch):
    """sync_every=0: no add_note until episode flush."""
    monkeypatch.setenv("HERMES_AMEM_SYNC_EVERY", "0")
    provider = AMemMemoryProvider()
    provider.initialize("sess-1")
    provider.sync_turn("open the fridge", "go to fridge 1")
    provider.sync_turn("take apple", "take apple 1")
    assert fake_store.n_notes == 0
    assert len(provider._pending) == 2
    provider.flush_pending(reason="session_end")
    assert fake_store.n_notes == 1
    body = fake_store.notes[0]["content"]
    assert "open the fridge" in body
    assert "take apple" in body
    assert provider.export_telemetry()["n_syncs"] == 1


def test_default_sync_every_is_five(fake_store):
    provider = AMemMemoryProvider()
    provider.initialize("sess-1")
    assert provider._sync_every == 5
    for i in range(4):
        provider.sync_turn(f"u{i}", f"a{i}")
        assert fake_store.n_notes == 0
    provider.sync_turn("u4", "a4")
    assert fake_store.n_notes == 1
    assert provider.export_telemetry()["sync_every"] == 5


def test_sync_every_n_flushes_on_interval(fake_store, monkeypatch):
    monkeypatch.setenv("HERMES_AMEM_SYNC_EVERY", "2")
    provider = AMemMemoryProvider()
    provider.initialize("sess-1")
    provider.sync_turn("u1", "a1")
    assert fake_store.n_notes == 0
    provider.sync_turn("u2", "a2")
    assert fake_store.n_notes == 1
    provider.sync_turn("u3", "a3")
    assert fake_store.n_notes == 1
    provider.on_session_end([])
    assert fake_store.n_notes == 2
    assert provider.export_telemetry()["n_syncs"] == 2


def test_sync_every_1_is_legacy_per_turn(fake_store, monkeypatch):
    monkeypatch.setenv("HERMES_AMEM_SYNC_EVERY", "1")
    provider = AMemMemoryProvider()
    provider.initialize("sess-1")
    provider.sync_turn("open the fridge", "go to fridge 1")
    assert fake_store.n_notes == 1
    assert "open the fridge" in fake_store.notes[0]["content"]
    assert provider.export_telemetry()["n_syncs"] == 1


def test_export_telemetry_flushes_pending(fake_store, monkeypatch):
    monkeypatch.setenv("HERMES_AMEM_SYNC_EVERY", "0")
    provider = AMemMemoryProvider()
    provider.initialize("sess-1")
    provider.sync_turn("u", "a")
    assert fake_store.n_notes == 0
    tel = provider.export_telemetry()
    assert fake_store.n_notes == 1
    assert tel["n_syncs"] == 1
    assert tel["sync_every"] == 0
    assert tel["n_turns_buffered"] == 1


def test_sync_turn_skips_empty(fake_store):
    provider = AMemMemoryProvider()
    provider.initialize("sess-1")
    provider.sync_turn("  ", "")
    assert fake_store.n_notes == 0
    assert provider._pending == []


def test_readonly_skips_writes_but_prefetch_works(fake_store, monkeypatch):
    monkeypatch.setenv("HERMES_AMEM_ENABLED", "1")
    monkeypatch.setenv("HERMES_AMEM_READONLY", "1")
    provider = AMemMemoryProvider()
    provider.initialize("sess-1")
    assert provider._readonly is True
    provider.sync_turn("user turn", "assistant turn")
    assert provider._pending == []
    assert fake_store.n_notes == 0
    text = provider.prefetch("How do I get the apple from the fridge?")
    assert "go to fridge" in text
    tel = provider.export_telemetry()
    assert tel["readonly"] is True
    assert tel["n_syncs"] == 0
    assert fake_store.n_notes == 0
    provider.shutdown()
    assert fake_store.persisted == 0


def test_no_extra_tools():
    provider = AMemMemoryProvider()
    assert provider.get_tool_schemas() == []
    assert provider.name == "amem"


def test_system_prompt_mentions_skills_and_fence():
    block = AMemMemoryProvider().system_prompt_block()
    assert "memory-context" in block
    assert "skill_view" in block


def test_is_available_with_env(monkeypatch):
    monkeypatch.setenv("HERMES_AMEM_ENABLED", "1")
    assert AMemMemoryProvider().is_available() is True


def test_shutdown_flushes_and_persists(fake_store):
    provider = AMemMemoryProvider()
    provider.initialize("sess-1")
    provider.sync_turn("u", "a")
    provider.shutdown()
    assert fake_store.n_notes == 1
    assert fake_store.persisted == 1
