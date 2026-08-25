"""Unit tests for Dynamic Cheatsheet memory provider (no live LLM)."""

from __future__ import annotations

import pytest

from plugins.memory.dcheatsheet import DCMemoryProvider
from plugins.memory.dcheatsheet.extractor import extract_cheatsheet
from plugins.memory.dcheatsheet import store as dc_store
from plugins.memory.dcheatsheet.store import HermesDCStore


@pytest.fixture(autouse=True)
def _clean_dc_env(monkeypatch, tmp_path):
    persist = tmp_path / "dcheatsheet"
    persist.mkdir()
    monkeypatch.setenv("HERMES_DC_PATH", str(persist))
    monkeypatch.setenv("HERMES_DC_MODE", "cu")
    monkeypatch.setenv("HERMES_DC_K", "3")
    for key in (
        "HERMES_DC_ENABLED",
        "HERMES_DC_LLM_MODEL",
        "HERMES_DC_API_KEY",
        "HERMES_DC_API_BASE",
        "HERMES_MEMORY_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)
    dc_store.reset_store_for_tests()
    dc_store._OVERRIDE_CURATOR = lambda prompt: (
        "<cheatsheet>\nTip: go to fridge then take apple.\n</cheatsheet>"
    )
    yield
    dc_store.reset_store_for_tests()


def test_extract_cheatsheet_reads_tagged_block():
    old = "keep me"
    new = extract_cheatsheet(
        "noise <cheatsheet>\nupdated tip\n</cheatsheet> trailing",
        old,
    )
    assert new == "updated tip"


def test_extract_cheatsheet_keeps_old_when_tags_missing():
    assert extract_cheatsheet("no tags here", "previous") == "previous"
    assert extract_cheatsheet("", "previous") == "previous"


def test_prefetch_empty_when_sheet_empty():
    provider = DCMemoryProvider()
    provider.initialize("sess-1")
    assert provider.prefetch("How do I get the apple?") == ""


def test_sync_turn_curates_and_prefetch_injects():
    provider = DCMemoryProvider()
    provider.initialize("sess-1")
    provider.sync_turn("get the apple from the fridge", "go to fridge 1")
    text = provider.prefetch("How do I get the apple from the fridge?")
    assert "Dynamic Cheatsheet" in text
    assert "go to fridge" in text
    tel = provider.export_telemetry()
    assert tel["n_episodes"] == 1
    assert tel["n_curations"] == 1
    assert tel["sheet_chars"] > 0


def test_sync_turn_skips_empty():
    provider = DCMemoryProvider()
    provider.initialize("sess-1")
    provider.sync_turn("  ", "")
    assert provider.export_telemetry()["n_episodes"] == 0
    assert provider.export_telemetry()["n_curations"] == 0


def test_failed_curator_keeps_previous_sheet(monkeypatch):
    persist = dc_store.get_store().persist_dir
    store = HermesDCStore(persist)
    store.cheatsheet = "old tip: use microwave"
    store.persist()
    dc_store.reset_store_for_tests()
    dc_store._OVERRIDE_CURATOR = lambda prompt: "I forgot the tags"
    provider = DCMemoryProvider()
    provider.initialize("sess-1")
    provider.sync_turn("heat the mug", "use microwave")
    assert provider._store.cheatsheet == "old tip: use microwave"
    assert provider.export_telemetry()["n_curations"] == 0


def test_persist_reloads_across_singleton_reset():
    provider = DCMemoryProvider()
    provider.initialize("sess-1")
    provider.sync_turn("open fridge", "go to fridge 1")
    persist = provider._store.persist_dir
    dc_store.reset_store_for_tests()
    reloaded = HermesDCStore(persist)
    assert "go to fridge" in reloaded.cheatsheet
    assert reloaded.episodes
    assert "open fridge" in reloaded.episodes[0]["query"]


def test_curetr_prefetch_includes_retrieved_examples(monkeypatch):
    monkeypatch.setenv("HERMES_DC_MODE", "curetr")
    dc_store._EMBED_FAILED = True
    provider = DCMemoryProvider()
    provider.initialize("sess-1")
    provider._store.cheatsheet = "Reusable: go to fridge first."
    provider._store.episodes.append(
        {"query": "get apple", "answer": "go to fridge then take apple"}
    )
    text = provider.prefetch("get apple from fridge")
    assert "Reusable: go to fridge first." in text
    assert "PREVIOUS SOLUTIONS" in text
    assert "take apple" in text


def test_no_extra_tools():
    provider = DCMemoryProvider()
    assert provider.get_tool_schemas() == []
    assert provider.name == "dcheatsheet"


def test_system_prompt_mentions_skills_and_fence():
    block = DCMemoryProvider().system_prompt_block()
    assert "memory-context" in block
    assert "skill_view" in block


def test_official_prompts_have_placeholders():
    from plugins.memory.dcheatsheet.store import _load_prompt

    cu = _load_prompt("curator_cu.txt")
    rs = _load_prompt("curator_rs.txt")
    assert "[[PREVIOUS_CHEATSHEET]]" in cu
    assert "[[QUESTION]]" in cu
    assert "[[MODEL_ANSWER]]" in cu
    assert "[[PREVIOUS_INPUT_OUTPUT_PAIRS]]" in rs
    assert "[[NEXT_INPUT]]" in rs
