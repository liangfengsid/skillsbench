"""Tests for agent/hot_skills.py — hot skill key point pool."""

import json

import pytest

from agent.hot_skills import (
    HotSkillPool,
    _utility_inject_score,
    abstract_hot_key_point,
    abstract_hot_key_points,
    build_hot_pool_outcome,
    build_hot_skills_block,
    build_llm_eviction_messages,
    build_outcome_attribution_messages,
    compute_alignment_hits,
    derive_hot_scope,
    empty_point_utility,
    extract_hot_key_points,
    filter_junk_key_points,
    format_hot_skill_section,
    heuristic_outcome_attributions,
    is_excluded_hot_skill,
    is_junk_key_point,
    _fenced_char_ranges,
    _position_in_fenced_region,
    llm_eviction_keep_ids,
    load_hot_skills_config,
    normalize_hot_scope_value,
    parse_hot_pool_inner_meta,
    parse_llm_keep_ids,
    parse_outcome_attributions,
    replay_message_tool_stats,
    resolve_hot_pool_persist_path,
    summarize_point_utility,
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
    assert "constraint" in out.lower()
    assert "enumerat" in out.lower()
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
        content="<!-- hermes-hot -->\n- ALWAYS keep the skill body editable\n<!-- /hermes-hot -->",
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
    assert "broadcast" in system.lower() or "unseen" in system.lower()
    assert "cheatsheet" in system.lower() or "replay" in system.lower()
    assert "Do NOT treat" in system and "primary keep criterion" in system
    assert "absolute paths" in system.lower() or "instance-specific" in system.lower()
    assert "recap" in system.lower() or "episode" in system.lower()
    assert "search order" in system.lower() or "enumerat" in system.lower()
    payload = json.loads(msgs[1]["content"])
    assert payload["keep_n"] == 1
    assert payload["context"] == "schedule gmail meetings"
    goal = payload["selection_goal"].lower()
    assert "broadcast" in goal or "unseen" in goal or "held-out" in goal
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
        sys_l = messages[0]["content"].lower()
        assert "broadcast" in sys_l or "unseen" in sys_l
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


def test_outcome_feedback_default_on():
    cfg = load_hot_skills_config({"hot_pool": {}})
    assert cfg.get("outcome_feedback") is True
    assert cfg.get("outcome_feedback_heuristic") is True
    assert cfg.get("abstract_extract") is True
    assert cfg.get("inject_filter_utilities") is True


def test_build_hot_pool_outcome_prefers_env_steps():
    outcome = build_hot_pool_outcome(
        evaluation={"task_success": True, "reward": 1.0, "steps": 7},
        run_result={"api_calls": 40},
        benchmark="alfworld",
    )
    assert outcome["success"] is True
    assert outcome["iterations"] == 7
    assert outcome["iterations_kind"] == "env_steps"


def test_parse_outcome_attributions():
    items = [{"id": "0", "skill": "a", "point": "tip"}, {"id": "1", "skill": "b", "point": "x"}]
    labels = parse_outcome_attributions(
        '{"labels": {"0": "helpful", "1": "harmful", "9": "irrelevant"}}',
        items,
    )
    assert labels == {"0": "helpful", "1": "harmful"}


def test_apply_outcome_feedback_updates_multidim_utility(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="nav",
        content="",
        key_points=["go to fridge first", "never open locked door"],
        turn=1,
    )
    pool.build_block(user_message="find apple", turn=1)
    assert pool._exposed_tips

    def fake_complete(msgs):
        return json.dumps({"labels": {"0": "helpful", "1": "irrelevant"}})

    summary = pool.apply_outcome_feedback(
        {
            "success": True,
            "reward": 1.0,
            "iterations": 12,
            "iterations_kind": "env_steps",
            "benchmark": "alfworld",
        },
        complete_fn=fake_complete,
    )
    assert summary["applied"] is True
    assert summary["points_scored"] == 2
    assert summary["attributions"]["helpful"] == 1
    assert summary["attributions"]["irrelevant"] == 1

    entry = pool._entries["nav"]
    util0 = summarize_point_utility(entry.point_utilities[0])
    assert util0["n_labeled"] == 1
    assert util0["helpful"] == 1
    assert util0["avg_iterations_when_success"] == 12.0
    util1 = summarize_point_utility(entry.point_utilities[1])
    assert util1["irrelevant"] == 1

    items = pool._point_items()
    assert items[0]["utility"]["n_labeled"] == 1
    tel = pool.export_telemetry()
    assert tel["outcome_feedback"]["applied"] is True
    assert tel["outcome_feedback"]["skipped_reason"] == ""
    assert tel["outcome_feedback"]["attribution_source"] == "llm"
    assert tel["config"]["outcome_feedback"] is True


def test_outcome_feedback_disabled_skips(pool_cfg):
    pool_cfg = dict(pool_cfg)
    pool_cfg["outcome_feedback"] = False
    pool = HotSkillPool(pool_cfg)
    pool.record(name="x", content="", key_points=["tip"], turn=1)
    pool.build_block(user_message="hi", turn=1)
    summary = pool.apply_outcome_feedback(
        {"success": True, "reward": 1.0, "iterations": 3},
        complete_fn=lambda _m: '{"labels": {"0": "helpful"}}',
    )
    assert summary["applied"] is False
    assert summary["skipped_reason"] == "disabled"
    tel = pool.export_telemetry()
    assert tel["outcome_feedback"]["skipped_reason"] == "disabled"
    assert tel["outcome_feedback"]["applied"] is False


def test_strongly_harmful_tip_filtered_on_re_admit(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(name="bad", content="", key_points=["poison tip"], turn=1)
    entry = pool._entries["bad"]
    util = empty_point_utility()
    util.update(
        {
            "n_labeled": 4,
            "helpful": 0,
            "harmful": 3,
            "irrelevant": 1,
        }
    )
    entry.point_utilities = [util]
    pool.record(name="bad", content="", key_points=["poison tip", "fresh tip"], turn=2)
    assert pool._entries["bad"].key_points == ["fresh tip"]


def test_outcome_attribution_messages_include_outcome():
    msgs = build_outcome_attribution_messages(
        [{"id": "0", "skill": "s", "point": "p"}],
        {"success": True, "iterations": 5},
    )
    assert msgs[0]["role"] == "system"
    assert "multi-dimensional" in msgs[0]["content"]
    system = msgs[0]["content"].lower()
    assert "transfer" in system
    assert "recap" in system or "episode-local" in system
    assert "enumerat" in system or "extra iterations" in system
    body = json.loads(msgs[1]["content"])
    assert body["outcome"]["iterations"] == 5


def test_abstract_extract_redacts_structural_ids_only():
    content = """## Key Points
- The alarmclock was on desk 1, the desklamp on dresser 1
- NEVER toggle a lamp before the target object is in hand
- ALWAYS write frames under /tmp/out/
- Email admin@example.com citing 550e8400-e29b-41d4-a716-446655440000
"""
    points = extract_hot_key_points(content)
    # English recaps are not regex-dropped; numbered slots stay as text.
    assert any("alarmclock" in p.lower() and "desk 1" in p for p in points)
    assert any("NEVER" in p and "lamp" in p.lower() for p in points)
    assert any("ALWAYS" in p and "<path>" in p for p in points)
    assert not any("/tmp/" in p for p in points)
    assert any("<email>" in p and "<id>" in p for p in points)
    assert not any("admin@example.com" in p for p in points)


def test_abstract_extract_can_disable():
    content = "## Key Points\n- ALWAYS write frames under /tmp/out/"
    points = extract_hot_key_points(
        content,
        config={
            "use_hermes_hot_markers": False,
            "extract_sections": True,
            "fallback_extract": False,
            "abstract_extract": False,
            "max_points_per_skill": 8,
            "max_chars_per_point": 240,
        },
    )
    assert any("/tmp/out/" in p for p in points)


def test_abstract_hot_key_point_keeps_english_and_redacts_paths():
    assert abstract_hot_key_point("poison tip") == "poison tip"
    assert (
        abstract_hot_key_point("The mug was on countertop 1")
        == "The mug was on countertop 1"
    )
    rewritten = abstract_hot_key_point(
        "ALWAYS use get_hermes_home() instead of ~/.hermes"
    )
    assert rewritten is not None
    assert "get_hermes_home" in rewritten
    assert "~/.hermes" not in rewritten


def test_abstract_hot_key_points_dedupes_after_path_redaction():
    points = abstract_hot_key_points(
        [
            "ALWAYS write frames under /tmp/out/",
            "ALWAYS write frames under /home/user/out/",
            "ALWAYS write frames under /tmp/out/",
        ]
    )
    assert points == ["ALWAYS write frames under <path>"]


def test_outcome_feedback_skipped_reason_without_heuristic(pool_cfg):
    cfg = dict(pool_cfg)
    cfg["outcome_feedback_heuristic"] = False
    pool = HotSkillPool(cfg)
    pool.record(name="nav", content="", key_points=["go to fridge first"], turn=1)
    pool.build_block(user_message="find apple", turn=1)
    summary = pool.apply_outcome_feedback(
        {"success": True, "reward": 1.0, "iterations": 8},
        complete_fn=None,
    )
    assert summary["applied"] is False
    assert summary["skipped_reason"] == "no_complete_fn"
    tel = pool.export_telemetry()
    assert tel["outcome_feedback"]["skipped_reason"] == "no_complete_fn"
    assert tel["outcome_feedback"]["applied"] is False
    assert pool._entries["nav"].point_utilities[0]["n_labeled"] == 0


def test_outcome_feedback_heuristic_persists_when_llm_empty(pool_cfg, tmp_path):
    cfg = dict(pool_cfg)
    cfg["persist_across_conversations"] = True
    cfg["persist_path"] = str(tmp_path / "hot_pool.json")
    pool = HotSkillPool(cfg)
    pool.record(
        name="nav",
        content="",
        key_points=["NEVER open a locked door", "check nearby furniture first"],
        turn=1,
    )
    pool.build_block(user_message="find apple", turn=1)
    summary = pool.apply_outcome_feedback(
        {"success": True, "reward": 1.0, "iterations": 8},
        complete_fn=lambda _m: "",
    )
    assert summary["applied"] is True
    assert summary["attribution_source"] == "heuristic"
    assert summary["skipped_reason"] == "unparseable_or_empty"
    assert summary["attributions"]["helpful"] == 2
    entry = pool._entries["nav"]
    assert entry.point_utilities[0]["n_labeled"] == 1
    assert entry.point_utilities[0]["helpful"] == 1
    tel = pool.export_telemetry()
    assert tel["outcome_feedback"]["attribution_source"] == "heuristic"
    assert tel["outcome_feedback"]["skipped_reason"] == "unparseable_or_empty"

    reloaded = HotSkillPool(cfg)
    util = reloaded._entries["nav"].point_utilities[0]
    assert util["n_labeled"] == 1
    assert util["helpful"] == 1


def test_outcome_feedback_heuristic_failure_is_irrelevant(pool_cfg):
    items = [
        {"id": "0", "skill": "a", "point": "NEVER skip examine", "via": "inject"},
        {"id": "1", "skill": "a", "point": "random leftover note", "via": "inject"},
    ]
    labels = heuristic_outcome_attributions(
        items, {"success": False, "iterations": 40}, low_step_threshold=12
    )
    assert labels == {"0": "irrelevant", "1": "irrelevant"}


def test_outcome_feedback_heuristic_ignores_wording():
    items = [{"id": "0", "skill": "a", "point": "NEVER skip examine", "via": "inject"}]
    labels = heuristic_outcome_attributions(
        items, {"success": True, "iterations": 40}, low_step_threshold=12
    )
    assert labels == {"0": "irrelevant"}


def test_heuristic_outcome_only_injected_tips_helpful_on_success():
    items = [
        {"id": "0", "skill": "a", "point": "check fridge", "via": "inject"},
        {"id": "1", "skill": "b", "point": "already in skill_view", "via": "history"},
    ]
    labels = heuristic_outcome_attributions(
        items, {"success": True, "iterations": 8}, low_step_threshold=12
    )
    assert labels == {"0": "helpful", "1": "irrelevant"}


def test_heuristic_outcome_history_skipped_even_on_fast_success(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(name="nav", content="", key_points=["tip from history"], turn=1)
    pool._exposed_tips[("nav", "tip from history")] = {
        "skill": "nav",
        "point": "tip from history",
        "scope": "",
        "via": "history",
    }
    summary = pool.apply_outcome_feedback(
        {"success": True, "reward": 1.0, "iterations": 5},
        complete_fn=lambda _m: "",
    )
    assert summary["attribution_source"] == "heuristic"
    assert summary["attributions"]["irrelevant"] == 1
    assert summary["attributions"].get("helpful", 0) == 0


def test_inject_score_penalizes_slow_successes():
    fast = empty_point_utility()
    fast.update(
        {
            "n_labeled": 2,
            "helpful": 2,
            "n_success": 2,
            "iterations_when_success_sum": 16.0,
        }
    )
    slow = empty_point_utility()
    slow.update(
        {
            "n_labeled": 2,
            "helpful": 2,
            "n_success": 2,
            "iterations_when_success_sum": 60.0,
        }
    )
    assert _utility_inject_score(fast) > _utility_inject_score(slow)


def test_inject_keeps_strongly_irrelevant_and_sorts(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="nav",
        content="",
        key_points=["ALWAYS examine before toggle", "noisy leftover recap text"],
        turn=1,
    )
    entry = pool._entries["nav"]
    good = empty_point_utility()
    good.update({"n_labeled": 2, "helpful": 2, "harmful": 0, "irrelevant": 0})
    noisy = empty_point_utility()
    noisy.update({"n_labeled": 3, "helpful": 0, "harmful": 0, "irrelevant": 3})
    entry.point_utilities = [good, noisy]
    block = pool.build_block(user_message="find apple", turn=2)
    assert "ALWAYS examine before toggle" in block
    assert "noisy leftover recap text" in block
    assert block.index("ALWAYS examine before toggle") < block.index(
        "noisy leftover recap text"
    )
    tel = pool.export_telemetry()
    assert tel["inject"]["points_omitted_utility"] == 0
    assert tel["inject"]["point_count"] == 2


def test_inject_falls_back_when_all_filtered(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(name="nav", content="", key_points=["poison leftover tip text"], turn=1)
    entry = pool._entries["nav"]
    util = empty_point_utility()
    util.update({"n_labeled": 4, "helpful": 0, "harmful": 3, "irrelevant": 1})
    entry.point_utilities = [util]
    block = pool.build_block(user_message="find apple", turn=2)
    assert "poison leftover tip text" in block
    assert pool.export_telemetry()["inject"]["points_omitted_utility"] == 1


def test_inject_filter_utilities_can_disable(pool_cfg):
    cfg = dict(pool_cfg)
    cfg["inject_filter_utilities"] = False
    pool = HotSkillPool(cfg)
    pool.record(
        name="nav",
        content="",
        key_points=["ALWAYS examine before toggle", "noisy leftover recap text"],
        turn=1,
    )
    entry = pool._entries["nav"]
    noisy = empty_point_utility()
    noisy.update({"n_labeled": 3, "helpful": 0, "harmful": 0, "irrelevant": 3})
    entry.point_utilities = [empty_point_utility(), noisy]
    block = pool.build_block(user_message="find apple", turn=2)
    assert "noisy leftover recap text" in block
    assert pool.export_telemetry()["inject"]["points_omitted_utility"] == 0


def test_is_junk_key_point_filters_template_bullets():
    assert is_junk_key_point("Important details")
    assert is_junk_key_point("```")
    assert is_junk_key_point(
        "Trigger conditions — State clearly when the agent should load the skill"
    )
    assert not is_junk_key_point("NEVER use os or sys in the AppWorld REPL")


def test_filter_junk_key_points_preserves_guardrails():
    points = filter_junk_key_points(
        [
            "Important details",
            "NEVER paginate with page_limit=5 defaults",
            "```",
            "ALWAYS check show_api_doc before calling",
        ]
    )
    assert points == [
        "NEVER paginate with page_limit=5 defaults",
        "ALWAYS check show_api_doc before calling",
    ]


def test_extract_hot_key_points_strips_skill_authoring_template():
    content = """
## Key Points

- Important details
- ALWAYS paginate API results
"""
    points = extract_hot_key_points(content)
    assert "Important details" not in points
    assert any("paginate" in p.lower() for p in points)


def test_excluded_meta_skill_not_admitted(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="hermes-agent-skill-authoring",
        content="<!-- hermes-hot -->\n- ALWAYS write numbered steps\n<!-- /hermes-hot -->",
        turn=1,
    )
    assert "hermes-agent-skill-authoring" not in pool._entries
    assert pool._telemetry.records_skipped_excluded_skill == 1


def test_appworld_batch_platform_excludes_mcp_media_skills(pool_cfg):
    assert is_excluded_hot_skill(
        "spotify",
        {"runtime_platform": "appworld-batch", "exclude_skills_from_pool": []},
    )
    assert not is_excluded_hot_skill(
        "spotify",
        {"runtime_platform": "cli", "exclude_skills_from_pool": []},
    )


def test_set_platform_evicts_mcp_media_from_pool(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="spotify",
        content="<!-- hermes-hot -->\n- ALWAYS check Premium before playback\n<!-- /hermes-hot -->",
        turn=1,
    )
    assert "spotify" in pool._entries
    pool.set_platform("appworld-batch")
    assert "spotify" not in pool._entries


def test_extract_section_points_skips_headings_inside_code_fences():
    content = """
## SKILL.md Format

```markdown
---
name: skill-name
---

# Skill Name

## Key Points

- Important details
```

## Common Pitfalls

- ALWAYS paginate API results
"""
    points = extract_hot_key_points(
        content,
        config={
            "use_hermes_hot_markers": False,
            "extract_sections": True,
            "fallback_extract": False,
            "junk_filter": True,
            "max_points_per_skill": 8,
            "max_chars_per_point": 240,
        },
    )
    assert "Important details" not in points
    assert any("paginate" in p.lower() for p in points)


def test_fenced_char_ranges_open_close():
    text = "before\n```markdown\n## Key Points\n- fake\n```\nafter"
    ranges = _fenced_char_ranges(text)
    assert len(ranges) == 1
    start, end = ranges[0]
    assert "## Key Points" in text[start:end]
    assert _position_in_fenced_region(text.index("## Key Points"), ranges)
    assert not _position_in_fenced_region(0, ranges)


def test_reconcile_on_update_runs_llm_under_cap():
    cfg = {
        "enabled": True,
        "max_entries": 12,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "eviction_policy": "llm",
        "reconcile_on_update": True,
        "inject_on_turn": True,
    }
    calls = []

    def keep_transferable(items, keep_n, context):
        del keep_n, context
        calls.append(list(items))
        return [it["id"] for it in items if "transferable" in it["point"].lower()]

    pool = HotSkillPool(cfg, eviction_judge=keep_transferable)
    pool.record(
        name="demo",
        content="",
        key_points=[
            "transferable pitfall: paginate all API pages",
            "episode-local recap of shelf three layout only",
        ],
        turn=1,
    )
    assert calls, "reconcile should run even when pool is under cap"
    assert pool._entries["demo"].key_points == [
        "transferable pitfall: paginate all API pages"
    ]
    assert pool.export_telemetry()["eviction"]["reconcile_ran"] is True


def test_reconcile_on_update_can_be_disabled():
    cfg = {
        "enabled": True,
        "max_entries": 12,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "eviction_policy": "llm",
        "reconcile_on_update": False,
        "inject_on_turn": True,
    }

    def judge(items, keep_n, context):
        del items, keep_n, context
        raise AssertionError("judge should not run under cap when reconcile_on_update is off")

    pool = HotSkillPool(cfg, eviction_judge=judge)
    pool.record(
        name="demo",
        content="<!-- hermes-hot -->\n- ALWAYS keep me\n<!-- /hermes-hot -->",
        turn=1,
    )
    assert "demo" in pool._entries
    assert pool.export_telemetry()["eviction"]["reconcile_ran"] is False

