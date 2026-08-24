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


def test_sync_turn_stores_compact_note(fake_store):
    provider = AMemMemoryProvider()
    provider.initialize("sess-1")
    provider.sync_turn("open the fridge", "go to fridge 1")
    assert fake_store.n_notes == 1
    assert "open the fridge" in fake_store.notes[0]["content"]
    assert provider.export_telemetry()["n_syncs"] == 1


def test_sync_turn_skips_empty(fake_store):
    provider = AMemMemoryProvider()
    provider.initialize("sess-1")
    provider.sync_turn("  ", "")
    assert fake_store.n_notes == 0


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


def test_shutdown_persists(fake_store):
    provider = AMemMemoryProvider()
    provider.initialize("sess-1")
    provider.shutdown()
    assert fake_store.persisted == 1
