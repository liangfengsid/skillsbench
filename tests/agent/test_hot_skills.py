"""Tests for agent/hot_skills.py — hot skill key point pool."""

import json

import pytest

from agent.hot_skills import (
    HotSkillPool,
    build_hot_skills_block,
    build_llm_eviction_messages,
    compute_alignment_hits,
    derive_hot_scope,
    extract_hot_key_points,
    format_hot_skill_section,
    llm_eviction_keep_ids,
    load_hot_skills_config,
    normalize_hot_scope_value,
    parse_hot_pool_inner_meta,
    parse_llm_keep_ids,
    replay_message_tool_stats,
    resolve_hot_pool_persist_path,
)

_SAMPLE_SKILL = """
# Demo skill

Long procedural body that should not appear in the hot pool.

<!-- hermes-hot -->
- NEVER run bare pytest — use scripts/run_tests.sh
- ALWAYS use get_hermes_home() instead of ~/.hermes
<!-- /hermes-hot -->

## Pitfalls
- Do not hardcode profile paths

More steps here that are not guardrails.
"""


@pytest.fixture
def pool_cfg():
    return {
        "enabled": True,
        "max_entries": 9,
        "max_chars": 2000,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "eviction_policy": "oldest",
        "inject_on_turn": True,
        "skip_if_in_history": True,
        "history_lookback": 10,
        "hydrate_from_history": True,
        "hydrate_limit": 10,
        "use_hermes_hot_markers": True,
        "extract_sections": True,
        "fallback_extract": True,
    }


def test_extract_hermes_hot_markers():
    points = extract_hot_key_points(_SAMPLE_SKILL)
    assert any("pytest" in p for p in points)
    assert any("get_hermes_home" in p for p in points)
    assert not any("Long procedural body" in p for p in points)


def test_extract_pitfalls_section():
    content = "## Pitfalls\n- Never commit secrets\n\n## Usage\n- Run the tool"
    points = extract_hot_key_points(content)
    assert "Never commit secrets" in points
    assert not any("Run the tool" in p for p in points)


def test_extract_common_pitfalls_numbered():
    """Bundled/authoring convention: ## Common Pitfalls with 1. items."""
    content = """# Demo
## Overview
Procedural body.

## Common Pitfalls
1. **Using bare pytest** — ALWAYS use scripts/run_tests.sh
2. Hardcoding ~/.hermes — use get_hermes_home()

## Usage
- Run the tool
"""
    points = extract_hot_key_points(content)
    assert any("pytest" in p for p in points)
    assert any("get_hermes_home" in p for p in points)
    assert not any("Run the tool" in p for p in points)


def test_extract_key_points_section():
    content = "## Key Points\n- ALWAYS validate paths\n\n## Steps\n- Do the work"
    points = extract_hot_key_points(content)
    assert any("validate paths" in p for p in points)
    assert not any("Do the work" in p for p in points)


def test_extract_skips_false_positive_headings():
    """'Apple Reminders' / bare 'Notes' must not be treated as hot sections."""
    content = """## Apple Reminders
- Create a reminder for milk

## Notes
- Some procedural note about the API

## Steps
- Continue the workflow
"""
    points = extract_hot_key_points(
        content,
        config={
            "use_hermes_hot_markers": False,
            "extract_sections": True,
            "fallback_extract": False,
            "max_points_per_skill": 8,
            "max_chars_per_point": 240,
        },
    )
    assert points == []


def test_extract_skillsbench_best_practices_and_limitations():
    """SkillsBench task skills use Best Practices / Limitations heavily."""
    content = """# Video frames

## Overview
How to extract frames.

## Best Practices
- Prefer seeking by timestamp over frame index for VFR videos
- ALWAYS write frames under /tmp/out/

## Limitations
- OpenCV may not support all codecs
- Encrypted videos cannot be processed

## Steps
- Run the extractor
"""
    points = extract_hot_key_points(
        content,
        config={
            "use_hermes_hot_markers": False,
            "extract_sections": True,
            "fallback_extract": False,
            "max_points_per_skill": 8,
            "max_chars_per_point": 240,
        },
    )
    assert any("VFR" in p or "timestamp" in p for p in points)
    assert any("codec" in p.lower() or "Encrypted" in p for p in points)
    assert not any("Run the extractor" in p for p in points)


def test_extract_skillsbench_critical_formula_heading():
    content = """## CRITICAL: Use Formulas, Not Hardcoded Values
**Always use Excel formulas instead of calculating values in Python.**

## Usage
- Open the workbook
"""
    points = extract_hot_key_points(
        content,
        config={
            "use_hermes_hot_markers": False,
            "extract_sections": True,
            "fallback_extract": False,
            "max_points_per_skill": 8,
            "max_chars_per_point": 240,
        },
    )
    assert any("formula" in p.lower() for p in points)
    assert not any("Open the workbook" in p for p in points)


def test_skill_has_hot_section():
    from agent.hot_skills import skill_has_hot_section

    assert skill_has_hot_section("## Common Pitfalls\n- x")
    assert skill_has_hot_section("## Pitfalls & Gotchas\n- x")
    assert skill_has_hot_section("## Best Practices\n- x")
    assert skill_has_hot_section("## Limitations\n- x")
    assert not skill_has_hot_section("## Apple Reminders\n- x")


def test_record_skips_when_no_key_points(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(name="empty", content="Just prose with no guardrails at all.", turn=1)
    assert "empty" not in pool._entries


def test_load_hot_skills_config_merges_defaults():
    cfg = load_hot_skills_config({"hot_pool": {"max_entries": 2, "enabled": False}})
    assert cfg["max_entries"] == 2
    assert cfg["enabled"] is False
    assert cfg["max_points_per_skill"] == 8
    assert cfg["eviction_policy"] == "llm"


def test_unknown_eviction_policy_defaults_to_llm():
    cfg = load_hot_skills_config({"hot_pool": {"eviction_policy": "not-a-policy"}})
    assert cfg["eviction_policy"] == "llm"
    cfg_rel = load_hot_skills_config({"hot_pool": {"eviction_policy": "relevance"}})
    assert cfg_rel["eviction_policy"] == "llm"


def test_load_hot_skills_config_env_enabled_override(monkeypatch):
    monkeypatch.setenv("HERMES_HOT_POOL_ENABLED", "0")
    monkeypatch.setenv("HERMES_HOT_POOL_PERSIST", "1")
    cfg = load_hot_skills_config({"hot_pool": {"enabled": True, "persist_across_conversations": True}})
    assert cfg["enabled"] is False
    assert cfg["persist_across_conversations"] is False

    monkeypatch.setenv("HERMES_HOT_POOL_ENABLED", "1")
    cfg_on = load_hot_skills_config({"hot_pool": {"enabled": False}})
    assert cfg_on["enabled"] is True


def test_global_pool_max_entries_is_retain_and_inject():
    cfg = {
        "enabled": True,
        "max_entries": 4,
        "max_chars": 4000,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "eviction_policy": "oldest",
        "inject_on_turn": True,
    }
    pool = HotSkillPool(cfg)
    for i in range(3):
        points = "\n".join(f"- Rule {i}-{j}" for j in range(3))
        pool.record(
            name=f"skill-{i}",
            content=f"<!-- hermes-hot -->\n{points}\n<!-- /hermes-hot -->",
            turn=i + 1,
        )
    total = sum(len(e.key_points) for e in pool._entries.values())
    assert total == 4
    block = pool.build_block(user_message="hello", turn=4)
    assert block.count("- Rule") == 4


def test_oldest_eviction_drops_earliest_skill(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    for i, name in enumerate(("a", "b", "c", "d")):
        pool.record(name=name, content=_SAMPLE_SKILL, turn=i + 1)
    assert "a" not in pool._entries
    assert set(pool._entries) == {"b", "c", "d"}
    block = pool.build_block(user_message="hello", turn=5)
    assert "### a" not in block
    assert "### d" in block


def test_record_from_tool_result_skips_file_views(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    main = json.dumps(
        {"success": True, "name": "foo", "content": _SAMPLE_SKILL, "skill_dir": "/tmp/x"}
    )
    pool.record_from_tool_result(main, turn=1)
    assert "foo" in pool._entries
    assert pool._entries["foo"].key_points

    subfile = json.dumps(
        {"success": True, "name": "foo", "file": "references/x.md", "content": "ref"}
    )
    pool.record_from_tool_result(subfile, turn=2)
    assert any("pytest" in p for p in pool._entries["foo"].key_points)


def test_build_block_injects_key_points_not_full_body(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(name="demo", content=_SAMPLE_SKILL, turn=1)
    block = pool.build_block(user_message="hello", turn=2)
    assert block.startswith("<hot-skills>")
    assert "get_hermes_home" in block
    assert "Long procedural body" not in block
    assert "Use skill_view(name)" in block


def test_build_block_injects_full_retained_points(pool_cfg):
    """Inject dumps the retained point set; no secondary char truncate."""
    cfg = dict(pool_cfg)
    cfg["max_entries"] = 5
    cfg["max_points_per_skill"] = 5
    pool = HotSkillPool(cfg)
    points = [f"Guardrail point number {i} with some extra text" for i in range(5)]
    pool.record(
        name="big",
        content="",
        turn=1,
        key_points=points,
    )
    block = pool.build_block(user_message="hello", turn=2)
    for p in points:
        assert p in block
    assert pool.export_telemetry()["inject"]["point_count"] == 5


def test_skip_if_in_history(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="github",
        content="<!-- hermes-hot -->\n- Use gh cli\n- Prefer gh over raw git push\n<!-- /hermes-hot -->",
        turn=1,
    )
    history = [
        {
            "role": "tool",
            "content": json.dumps({"success": True, "name": "github", "content": "full skill body"}),
        }
    ]
    block = pool.build_block(
        user_message="push to github",
        turn=2,
        exclude_names=pool.skills_in_recent_history(history),
    )
    assert block == ""
    tel = pool.export_telemetry()["inject"]
    assert tel["point_count"] == 0
    assert tel["skills_excluded_in_history"] == ["github"]
    assert tel["points_excluded_in_history"] == 2
    assert "github" in tel["excluded_skills"]


def test_hydrate_from_history(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    messages = [
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "success": True,
                    "name": "pytest-skill",
                    "content": "## Pitfalls\n- Run scripts/run_tests.sh\n",
                    "description": "testing",
                }
            ),
        }
    ]
    added = pool.hydrate_from_history(messages)
    assert added == 1
    assert "pytest-skill" in pool._entries


def test_refresh_stale_entries_reloads_key_points(pool_cfg, monkeypatch):
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="demo-skill",
        content="<!-- hermes-hot -->\n- version one\n<!-- /hermes-hot -->",
        skill_dir="/tmp/demo",
        turn=1,
    )
    pool._entries["demo-skill"].skill_md_mtime = 1.0

    def _fake_mtime(skill_dir_arg):
        return 2.0 if skill_dir_arg else 0.0

    monkeypatch.setattr("agent.hot_skills._skill_md_mtime", _fake_mtime)

    def _fake_reload(name, session_id=None):
        return "<!-- hermes-hot -->\n- version two\n<!-- /hermes-hot -->"

    monkeypatch.setattr("agent.hot_skills._reload_skill_content", _fake_reload)
    refreshed = pool.refresh_stale_entries()
    assert refreshed == 1
    assert pool._entries["demo-skill"].key_points == ["version two"]


def test_build_hot_skills_block_wraps_content():
    out = build_hot_skills_block("### demo\n- rule one")
    assert "<hot-skills>" in out
    assert "Use skill_view(name)" in out
    assert "scope" in out.lower()
    assert "rule one" in out


def test_derive_hot_scope_natural_language():
    # Prefer description as-is
    assert (
        derive_hot_scope(
            "ocr-and-documents",
            description="Extract text from PDFs/scans",
            tags=["PDF", "OCR"],
        )
        == "Extract text from PDFs/scans"
    )
    # No description → join tags
    assert derive_hot_scope("ocr-and-documents", description="", tags=["PDF", "OCR"]) == "PDF, OCR"
    # Neither → skill name
    assert derive_hot_scope("ocr-and-documents", description="", tags=[]) == "ocr-and-documents"


def test_format_and_parse_scope_section():
    section = format_hot_skill_section(
        "demo",
        "Applies to PDF/OCR tasks.",
        ["NEVER skip validation"],
    )
    assert section.startswith("### demo\n")
    assert "Scope: Applies to PDF/OCR tasks." in section
    assert "- NEVER skip validation" in section
    skills, points = parse_hot_pool_inner_meta(
        "### demo\nScope: Applies to PDF/OCR tasks.\n- NEVER skip validation\n### other\n- tip"
    )
    assert skills == ["demo", "other"]
    assert points == ["NEVER skip validation", "tip"]


def test_normalize_legacy_list_scope():
    # Coerce only — no "Applies to tasks involving" re-template at load.
    assert normalize_hot_scope_value(["email", "oauth"]) == "email, oauth"
    assert normalize_hot_scope_value("  PDF / OCR tasks.  ") == "PDF / OCR tasks."
    assert normalize_hot_scope_value(None) == ""
    assert normalize_hot_scope_value(123) == ""


def test_record_stores_scope_and_inject_shows_it(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="ocr-and-documents",
        content="<!-- hermes-hot -->\n- Prefer web_extract for URLs\n<!-- /hermes-hot -->",
        description="Extract text from PDFs",
        tags=["PDF", "OCR"],
        turn=1,
    )
    entry = pool._entries["ocr-and-documents"]
    assert isinstance(entry.scope, str)
    assert entry.scope == "Extract text from PDFs"
    block = pool.build_block(user_message="hello", turn=2)
    assert "### ocr-and-documents" in block
    assert "Scope: Extract text from PDFs" in block
    assert "Prefer web_extract" in block
    assert pool.export_telemetry()["inject"]["skills_injected"] == ["ocr-and-documents"]


def test_persist_roundtrip_preserves_scope(pool_cfg, tmp_path):
    path = tmp_path / "hot_pool.json"
    cfg = dict(pool_cfg)
    cfg["persist_across_conversations"] = True
    cfg["persist_path"] = str(path)
    pool_a = HotSkillPool(cfg)
    pool_a.on_turn_start(1)
    pool_a.record(
        name="gmail-helper",
        content="<!-- hermes-hot -->\n- Check existing tokens first\n<!-- /hermes-hot -->",
        tags=["email", "oauth"],
        turn=1,
    )
    saved = pool_a._entries["gmail-helper"].scope
    pool_b = HotSkillPool(cfg)
    assert pool_b._entries["gmail-helper"].scope == saved
    assert isinstance(json.loads(path.read_text())["entries"]["gmail-helper"]["scope"], str)


def test_skill_manage_create_syncs_hot_pool(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    content = (
        "---\nname: new-skill\ndescription: demo\n---\n\n"
        "<!-- hermes-hot -->\n- Always run scripts/run_tests.sh\n<!-- /hermes-hot -->"
    )
    pool.record_from_skill_manage(
        {"action": "create", "name": "new-skill", "content": content},
        json.dumps({"success": True, "message": "created"}),
        turn=2,
    )
    assert "new-skill" in pool._entries
    assert any("run_tests" in p for p in pool._entries["new-skill"].key_points)
    assert pool._telemetry.skill_manage_sync == 1


def test_skill_manage_delete_evicts(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="editable",
        content="<!-- hermes-hot -->\n- v1\n<!-- /hermes-hot -->",
        turn=1,
    )
    pool.record_from_skill_manage(
        {"action": "delete", "name": "editable"},
        json.dumps({"success": True, "message": "deleted"}),
    )
    assert "editable" not in pool._entries
    assert pool._telemetry.skill_manage_evict == 1


def test_skill_manage_patch_reload_syncs(pool_cfg, monkeypatch):
    pool = HotSkillPool(pool_cfg)
    monkeypatch.setattr(
        "agent.hot_skills._read_skill_md_content",
        lambda name: (
            "<!-- hermes-hot -->\n- patched rule\n<!-- /hermes-hot -->"
            if name == "patched-skill"
            else ""
        ),
    )
    monkeypatch.setattr(
        "agent.hot_skills._resolve_skill_dir_path",
        lambda name: "/tmp/skills/patched-skill" if name == "patched-skill" else None,
    )
    pool.record_from_skill_manage(
        {"action": "patch", "name": "patched-skill", "old_string": "x", "new_string": "y"},
        json.dumps({"success": True, "message": "patched SKILL.md"}),
        turn=3,
    )
    assert "patched-skill" in pool._entries
    assert pool._entries["patched-skill"].key_points == ["patched rule"]
    assert pool._telemetry.skill_manage_sync == 1


def test_skill_manage_supporting_file_patch_ignored(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="with-ref",
        content="<!-- hermes-hot -->\n- keep me\n<!-- /hermes-hot -->",
        turn=1,
    )
    pool.record_from_skill_manage(
        {"action": "patch", "name": "with-ref", "file_path": "references/guide.md"},
        json.dumps({"success": True, "message": "patched"}),
    )
    assert "with-ref" in pool._entries
    assert pool._telemetry.skill_manage_sync == 0


def test_persist_load_save_roundtrip(tmp_path):
    path = tmp_path / "hot_pool.json"
    cfg = {
        "enabled": True,
        "max_entries": 12,
        "max_chars": 2000,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "inject_on_turn": True,
        "persist_across_conversations": True,
        "persist_path": str(path),
    }
    pool_a = HotSkillPool(cfg)
    pool_a.on_turn_start(1)
    pool_a.record(
        name="bench-skill",
        content="<!-- hermes-hot -->\n- Use scripts/run_tests.sh\n<!-- /hermes-hot -->",
        turn=1,
    )
    assert path.is_file()

    pool_b = HotSkillPool(cfg)
    assert "bench-skill" in pool_b._entries
    assert any("run_tests" in p for p in pool_b._entries["bench-skill"].key_points)
    pool_b.on_turn_start(1)
    block = pool_b.build_block(user_message="run tests", turn=pool_b.active_turn)
    assert "run_tests" in block
    assert pool_b.global_turn == pool_a.global_turn + 1


def test_persist_oldest_eviction_uses_global_turn_across_conversations(tmp_path):
    """Session turn resets per AIAgent; persisted global_turn must drive oldest."""
    path = tmp_path / "hot_pool.json"
    cfg = {
        "enabled": True,
        "max_entries": 1,
        "max_chars": 2000,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "eviction_policy": "oldest",
        "inject_on_turn": True,
        "persist_across_conversations": True,
        "persist_path": str(path),
    }
    pool_a = HotSkillPool(cfg)
    pool_a.on_turn_start(1)
    pool_a.record(
        name="from-conv-a",
        content="<!-- hermes-hot -->\n- Rule A\n<!-- /hermes-hot -->",
        turn=1,
    )
    assert pool_a._entries["from-conv-a"].recorded_turn == pool_a.global_turn

    pool_b = HotSkillPool(cfg)
    pool_b.on_turn_start(1)  # new conversation, session turn would be 1 again
    pool_b.record(
        name="from-conv-b",
        content="<!-- hermes-hot -->\n- Rule B\n<!-- /hermes-hot -->",
        turn=1,
    )
    assert "from-conv-a" not in pool_b._entries
    assert "from-conv-b" in pool_b._entries
    assert pool_b._entries["from-conv-b"].recorded_turn == pool_b.global_turn
    assert pool_b.global_turn > pool_a.global_turn


def test_resolve_hot_pool_persist_path_default(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    p = resolve_hot_pool_persist_path({"persist_path": ""})
    assert p == home / "hot_skill_pool.json"


def test_telemetry_inject_and_carryover(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.on_turn_start(1)
    pool.record(
        name="demo",
        content="<!-- hermes-hot -->\n- rule one\n<!-- /hermes-hot -->",
        turn=1,
    )
    block = pool.build_block(user_message="hello", turn=pool.active_turn)
    assert block
    pool.note_api_injection(cache_nonempty=True)
    pool.note_api_injection(cache_nonempty=True)
    pool.note_api_injection(cache_nonempty=False)

    tel = pool.export_telemetry()
    assert tel["inject"]["build_block_applied"] is True
    assert tel["inject"]["point_count"] >= 1
    assert tel["inject"]["injections_nonempty"] == 2
    assert tel["inject"]["injections_attempted"] == 3
    assert tel["carryover"]["new_records_this_task"] == 1


def test_telemetry_skill_view_counters(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.on_turn_start(1)
    pool.record(
        name="foo",
        content="<!-- hermes-hot -->\n- Rule\n<!-- /hermes-hot -->",
        turn=1,
    )
    pool.note_skill_view("foo", in_pool_before=True)
    pool.note_skill_view("foo", in_pool_before=True)
    pool.note_skill_view("bar", in_pool_before=False)

    tel = pool.export_telemetry()
    assert tel["skill_view"]["total"] == 3
    assert tel["skill_view"]["unique"] == 2
    assert tel["skill_view"]["repeat"] == 1
    assert tel["skill_view"]["while_in_pool"] == 2


def test_replay_and_alignment_helpers():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "scripts/run_tests.sh tests/foo"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": '{"success": true}'},
    ]
    replay = replay_message_tool_stats(messages)
    assert replay["tool_errors_total"] == 0
    hits, checks = compute_alignment_hits(
        replay["substantive_tool_calls"],
        ["NEVER run bare pytest — use scripts/run_tests.sh"],
    )
    assert checks == 1
    assert hits == 1


def test_inject_does_not_refresh_eviction_clock(pool_cfg):
    """Dumping the pool into the prompt must not protect a skill from oldest eviction."""
    pool = HotSkillPool(pool_cfg)
    for i, name in enumerate(("a", "b", "c")):
        pool.record(name=name, content=_SAMPLE_SKILL, turn=i + 1)
    pool.build_block(user_message="hello", turn=10)
    pool.record(name="d", content=_SAMPLE_SKILL, turn=11)
    assert "a" not in pool._entries
    assert "d" in pool._entries


def test_llm_eviction_uses_judge_callback():
    cfg = {
        "enabled": True,
        "max_entries": 1,
        "max_chars": 4000,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "eviction_policy": "llm",
        "inject_on_turn": True,
    }

    def keep_github(items, keep_n, context):
        del keep_n, context
        for it in items:
            if it["skill"] == "github-auth":
                return [it["id"]]
        return [items[-1]["id"]]

    pool = HotSkillPool(cfg, eviction_judge=keep_github)
    pool.record(
        name="alpha",
        content="<!-- hermes-hot -->\n- Alpha rule\n<!-- /hermes-hot -->",
        turn=1,
    )
    pool.record(
        name="github-auth",
        content="<!-- hermes-hot -->\n- Validate github oauth tokens\n<!-- /hermes-hot -->",
        turn=2,
    )
    assert list(pool._entries) == ["github-auth"]


def test_llm_eviction_falls_back_to_oldest_without_judge():
    cfg = {
        "enabled": True,
        "max_entries": 1,
        "max_chars": 4000,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "eviction_policy": "llm",
        "inject_on_turn": True,
    }
    pool = HotSkillPool(cfg)
    pool.record(
        name="alpha",
        content="<!-- hermes-hot -->\n- Alpha rule\n<!-- /hermes-hot -->",
        turn=1,
    )
    pool.record(
        name="beta",
        content="<!-- hermes-hot -->\n- Beta rule\n<!-- /hermes-hot -->",
        turn=2,
    )
    assert "alpha" not in pool._entries
    assert list(pool._entries) == ["beta"]


def test_parse_llm_keep_ids_accepts_fenced_json():
    items = [{"id": "0", "skill": "a"}, {"id": "1", "skill": "b"}]
    text = '```json\n{"keep": ["1"]}\n```'
    assert parse_llm_keep_ids(text, items, 1) == ["1"]


def test_build_llm_eviction_messages_broadcast_not_task_local():
    items = [
        {
            "id": "0",
            "skill": "host-verif",
            "index": 0,
            "point": "ALWAYS write outputs under /tmp/foo",
            "recorded_turn": 1,
        },
        {
            "id": "1",
            "skill": "testing",
            "index": 0,
            "point": "NEVER run bare pytest — use scripts/run_tests.sh",
            "recorded_turn": 2,
        },
    ]
    msgs = build_llm_eviction_messages(items, keep_n=1, context="schedule gmail meetings")
    assert len(msgs) == 2
    system = msgs[0]["content"]
    assert "retain equals inject" in system.lower() or "retain = inject" in system.lower()
    assert "broadcast" in system.lower()
    assert "Do NOT treat" in system and "primary keep criterion" in system
    assert "absolute paths" in system.lower() or "instance-specific" in system.lower()
    payload = json.loads(msgs[1]["content"])
    assert payload["keep_n"] == 1
    assert payload["context"] == "schedule gmail meetings"
    assert "broadcast" in payload["selection_goal"].lower()
    assert "tie-breaker" in payload["context_role"].lower()
    assert payload["points"] == items


def test_parse_llm_keep_ids_strips_think_and_ignores_unknown():
    items = [{"id": "0", "skill": "a"}, {"id": "1", "skill": "b"}]
    text = '<think>nope</think>{"keep": ["1", "999"]}'
    assert parse_llm_keep_ids(text, items, 2) == ["1"]


def test_llm_eviction_keep_ids_uses_complete_fn():
    items = [
        {"id": "0", "skill": "alpha", "point": "A"},
        {"id": "1", "skill": "github-auth", "point": "G"},
    ]

    def complete(messages):
        assert messages[0]["role"] == "system"
        assert "broadcast" in messages[0]["content"].lower()
        assert "github-auth" in messages[1]["content"]
        payload = json.loads(messages[1]["content"])
        assert "selection_goal" in payload
        return '{"keep": ["1"]}'

    assert llm_eviction_keep_ids(items, 1, "fix oauth", complete) == ["1"]


def test_llm_eviction_via_complete_fn_judge():
    cfg = {
        "enabled": True,
        "max_entries": 1,
        "max_chars": 4000,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "eviction_policy": "llm",
        "inject_on_turn": True,
    }

    def judge(items, keep_n, context):
        return llm_eviction_keep_ids(
            items,
            keep_n,
            context,
            complete_fn=lambda _msgs: '{"keep": ["1"]}',
        )

    pool = HotSkillPool(cfg, eviction_judge=judge)
    pool.record(
        name="alpha",
        content="<!-- hermes-hot -->\n- Alpha rule\n<!-- /hermes-hot -->",
        turn=1,
    )
    pool.record(
        name="github-auth",
        content="<!-- hermes-hot -->\n- Validate github oauth tokens\n<!-- /hermes-hot -->",
        turn=2,
    )
    assert list(pool._entries) == ["github-auth"]


def test_persist_roundtrip_keeps_recorded_turn(tmp_path):
    path = tmp_path / "hot_pool.json"
    cfg = {
        "enabled": True,
        "eviction_policy": "oldest",
        "max_entries": 12,
        "max_chars": 2000,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "inject_on_turn": True,
        "persist_across_conversations": True,
        "persist_path": str(path),
    }
    pool_a = HotSkillPool(cfg)
    pool_a.on_turn_start(1)
    pool_a.record(
        name="bench-skill",
        content="<!-- hermes-hot -->\n- Use scripts/run_tests.sh\n<!-- /hermes-hot -->",
        turn=pool_a.active_turn,
    )
    recorded = pool_a._entries["bench-skill"].recorded_turn
    pool_a.build_block(user_message="run tests", turn=pool_a.active_turn)
    assert pool_a._entries["bench-skill"].recorded_turn == recorded

    pool_b = HotSkillPool(cfg)
    assert pool_b._entries["bench-skill"].recorded_turn == recorded
