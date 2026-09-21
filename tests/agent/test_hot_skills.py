"""Tests for agent/hot_skills.py — hot skill key point pool."""

import json

import pytest

from agent.hot_skills import (
    HotSkillPool,
    _utility_inject_score,
    _utility_strongly_harmful,
    abstract_hot_key_point,
    abstract_hot_key_points,
    build_hot_pool_outcome,
    build_hot_skills_block,
    build_llm_eviction_messages,
    build_outcome_attribution_messages,
    clamp_outcome_attributions,
    compute_alignment_hits,
    derive_hot_scope,
    empty_point_utility,
    extract_hot_key_points,
    filter_junk_key_points,
    format_hot_skill_section,
    heuristic_outcome_attributions,
    is_excluded_hot_skill,
    is_junk_key_point,
    is_ritual_key_point,
    is_off_session_domain,
    is_observation_user_message,
    platform_retrieve_tokens,
    recover_json_object,
    retrieve_overlap,
    build_compressed_episode_log,
    collect_outcome_episode_log,
    compress_env_code,
    compress_tool_args,
    sidechannel_thinking_off_extra_body,
    skill_retrieve_tokens,
    tokenize_retrieve,
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
        "junk_min_point_chars": 4,
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
    assert cfg["inject_k"] == 4
    assert cfg["inject_retrieve"] is True
    assert cfg["admit_domain_gate"] is True
    assert cfg["ritual_filter"] is True
    assert cfg["inject_throttle"] is True
    assert cfg["inject_throttle_min_episodes"] == 8
    assert cfg["inject_throttle_margin"] == 0.05
    assert cfg["inject_throttle_probe_every"] == 10
    assert cfg["inject_score_min_labeled"] == 5
    assert cfg["outcome_judge_max_tokens"] == 2048
    assert cfg["outcome_feedback_heuristic"] is False


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


def test_load_hot_skills_config_env_inject_and_outcome_overrides(monkeypatch):
    monkeypatch.setenv("HERMES_HOT_POOL_INJECT_RETRIEVE", "0")
    monkeypatch.setenv("HERMES_HOT_POOL_OUTCOME_FEEDBACK", "0")
    monkeypatch.setenv("HERMES_HOT_POOL_INJECT_FILTER_UTILITIES", "0")
    cfg = load_hot_skills_config(
        {
            "hot_pool": {
                "inject_retrieve": True,
                "outcome_feedback": True,
                "inject_filter_utilities": True,
            }
        }
    )
    assert cfg["inject_retrieve"] is False
    assert cfg["outcome_feedback"] is False
    assert cfg["inject_filter_utilities"] is False


def test_global_pool_max_entries_is_retain_and_inject():
    cfg = {
        "enabled": True,
        "max_entries": 4,
        "max_chars": 4000,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "junk_min_point_chars": 4,
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
    pool_cfg = dict(pool_cfg)
    pool_cfg["inject_k"] = 0
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
    assert "exception handler" in block.lower()


def test_build_block_injects_full_retained_points(pool_cfg):
    """inject_k=0 still serializes the full retained set (no char truncate)."""
    cfg = dict(pool_cfg)
    cfg["max_entries"] = 5
    cfg["max_points_per_skill"] = 5
    cfg["inject_k"] = 0
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
    assert "exception handler" in out.lower()
    assert "first-action" in out.lower() or "checklist" in out.lower()
    assert "scope" in out.lower()
    assert "rule one" in out


def test_hot_skills_note_sanitizer_matches_old_and_new_preface():
    from agent.hot_skills import sanitize_hot_skills_text

    old = (
        "[System note: The following are hot skill key points (guardrails) "
        "from recently used skills, NOT new user input. "
        "Use skill_view(name) for full procedures.]\n\nkept"
    )
    new = build_hot_skills_block("kept-inner")
    assert sanitize_hot_skills_text(old).strip() == "kept"
    assert "kept-inner" not in sanitize_hot_skills_text(new)
    assert "exception handler" not in sanitize_hot_skills_text(new).lower()


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
        "junk_min_point_chars": 4,
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
        "junk_min_point_chars": 4,
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
    assert "keep is the store" in system.lower() or "store-then-retrieve" in system.lower()
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
    assert cfg.get("outcome_feedback_heuristic") is False
    assert cfg.get("ritual_filter") is True
    assert cfg.get("abstract_extract") is True
    assert cfg.get("inject_filter_utilities") is True
    assert cfg.get("inject_retrieve") is True
    assert cfg.get("admit_domain_gate") is True
    assert cfg.get("outcome_judge_max_tokens") == 2048
    assert cfg.get("outcome_judge_log") is True
    assert cfg.get("inject_k") == 4


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
    assert "episode_log" in body
    assert "episode_log" in msgs[0]["content"]


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


def test_outcome_feedback_empty_llm_does_not_persist_utilities(pool_cfg, tmp_path):
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
    assert summary["applied"] is False
    assert summary["attribution_source"] == ""
    assert summary["skipped_reason"] == "unparseable_or_empty"
    assert summary["points_scored"] == 0
    entry = pool._entries["nav"]
    assert entry.point_utilities[0]["n_labeled"] == 0
    tel = pool.export_telemetry()
    assert tel["outcome_feedback"]["applied"] is False
    assert tel["outcome_feedback"]["attribution_source"] == ""
    assert tel["config"]["outcome_feedback_heuristic"] is False

    reloaded = HotSkillPool(cfg)
    assert reloaded._entries["nav"].point_utilities[0]["n_labeled"] == 0


def test_outcome_feedback_heuristic_persists_when_llm_empty(pool_cfg, tmp_path):
    cfg = dict(pool_cfg)
    cfg["outcome_feedback_heuristic"] = True
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


def test_outcome_feedback_heuristic_failure_marks_injected_harmful(pool_cfg):
    items = [
        {"id": "0", "skill": "a", "point": "NEVER skip examine", "via": "inject"},
        {"id": "1", "skill": "a", "point": "random leftover note", "via": "inject"},
    ]
    labels = heuristic_outcome_attributions(
        items, {"success": False, "iterations": 40}, low_step_threshold=12
    )
    assert labels == {"0": "harmful", "1": "harmful"}


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
    pool = HotSkillPool({**pool_cfg, "outcome_feedback_heuristic": True})
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


def test_inject_stays_silent_when_all_filtered(pool_cfg):
    """Negatively evidenced tips must not dump back into the inject block."""
    pool = HotSkillPool({**pool_cfg, "inject_retrieve": False})
    pool.record(name="nav", content="", key_points=["poison leftover tip text"], turn=1)
    entry = pool._entries["nav"]
    util = empty_point_utility()
    util.update({"n_labeled": 4, "helpful": 0, "harmful": 3, "irrelevant": 1})
    entry.point_utilities = [util]
    block = pool.build_block(user_message="find apple", turn=2)
    assert block == ""
    tel = pool.export_telemetry()["inject"]
    assert tel["points_omitted_utility"] == 1
    assert tel["point_count"] == 0


def test_inject_omits_negative_score_tips_when_labeled_enough(pool_cfg):
    pool = HotSkillPool(
        {
            **pool_cfg,
            "inject_retrieve": False,
            "inject_score_min_labeled": 5,
            "inject_throttle": False,
        }
    )
    pool.record(
        name="nav",
        content="",
        key_points=["weak tip with negative score", "fresh unlabeled tip"],
        turn=1,
    )
    entry = pool._entries["nav"]
    weak = empty_point_utility()
    # Not strongly_harmful (harmful < helpful+2) but score < 0 at n=5.
    weak.update(
        {
            "n_labeled": 5,
            "helpful": 3,
            "harmful": 2,
            "irrelevant": 0,
            "success_sum": 2.0,
            "reward_sum": 2.0,
            "iterations_sum": 100.0,
            "n_success": 2,
            "iterations_when_success_sum": 40.0,
        }
    )
    entry.point_utilities = [weak, empty_point_utility()]
    assert _utility_inject_score(weak) < 0.0
    assert not _utility_strongly_harmful(weak)
    block = pool.build_block(user_message="find apple", turn=2)
    assert "fresh unlabeled tip" in block
    assert "weak tip with negative score" not in block
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


def test_is_ritual_key_point_matches_closer_and_short_answer():
    assert is_ritual_key_point("ALWAYS call done() when the task is finished")
    assert is_ritual_key_point("Remember to call complete_task()")
    assert is_ritual_key_point("call the environment's done()")
    assert is_ritual_key_point("complete_task(answer='Payment request created')")
    assert is_ritual_key_point("Return a short answer")
    assert is_ritual_key_point("Keep the answer short")
    assert is_ritual_key_point("never forget to call done()")
    assert is_ritual_key_point("ALWAYS issue the done action last")


def test_is_ritual_key_point_matches_show_api_doc_process_rituals():
    assert is_ritual_key_point("ALWAYS show_api_doc before every API call.")
    assert is_ritual_key_point("ALWAYS call show_api_doc for each endpoint")
    assert is_ritual_key_point(
        "ALWAYS check API docs before calling any endpoint"
    )
    assert is_ritual_key_point("MUST consult the API documentation before use")
    # NEVER about the same action is a constraint, not a ritual.
    assert not is_ritual_key_point(
        "NEVER call show_api_doc for every endpoint before acting"
    )


def test_is_ritual_key_point_keeps_constraints_and_policy():
    assert not is_ritual_key_point(
        "NEVER call complete_task with a status string"
    )
    assert not is_ritual_key_point(
        "DO NOT call done() before verifying the inventory"
    )
    assert not is_ritual_key_point("ALWAYS paginate API results")
    assert not is_ritual_key_point("When the write is done, verify the file hash")
    assert not is_ritual_key_point("NEVER use os or sys in the AppWorld REPL")
    assert not is_ritual_key_point("ALWAYS use SUMPRODUCT instead of an array")
    assert not is_ritual_key_point("Keep formulas short to avoid parse errors")


def test_record_drops_ritual_tips_keeps_policy(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="app",
        content="",
        key_points=[
            "ALWAYS call done() after the last API",
            "ALWAYS paginate long API lists",
            "Return a short answer in complete_task",
        ],
        turn=1,
    )
    assert "app" in pool._entries
    points = pool._entries["app"].key_points
    assert any("paginate" in p.lower() for p in points)
    assert not any(is_ritual_key_point(p) for p in points)
    assert pool.export_telemetry()["carryover"]["records_ritual_filtered"] == 2


def test_extract_drops_ritual_tips():
    content = """
## Key Points

- ALWAYS call done() when finished
- ALWAYS paginate API results
- Return a short answer
"""
    points = extract_hot_key_points(content)
    assert any("paginate" in p.lower() for p in points)
    assert not any(is_ritual_key_point(p) for p in points)


def test_inject_omits_ritual_tips_already_in_store(pool_cfg):
    from agent.hot_skills import HotSkillEntry

    pool = HotSkillPool({**pool_cfg, "inject_retrieve": False, "inject_k": 8})
    pool._entries["app"] = HotSkillEntry(
        name="app",
        key_points=[
            "ALWAYS call done() when finished",
            "ALWAYS paginate long API lists",
        ],
        skill_dir=None,
        skill_md_mtime=0.0,
        recorded_turn=1,
        description="",
        tags=[],
        scope="",
        point_utilities=[empty_point_utility(), empty_point_utility()],
    )
    block = pool.build_block(user_message="send venmo payment", turn=1)
    assert "paginate" in block.lower()
    assert "done()" not in block
    assert pool.export_telemetry()["inject"]["points_omitted_ritual"] == 1


def test_persist_load_strips_ritual_tips(pool_cfg, tmp_path):
    path = tmp_path / "hot_pool.json"
    cfg = {
        **pool_cfg,
        "persist_across_conversations": True,
        "persist_path": str(path),
        "ritual_filter": False,
    }
    writer = HotSkillPool(cfg)
    writer.record(
        name="app",
        content="",
        key_points=[
            "ALWAYS call done() when finished",
            "ALWAYS paginate long API lists",
        ],
        turn=1,
    )
    assert any(is_ritual_key_point(p) for p in writer._entries["app"].key_points)

    reader = HotSkillPool({**cfg, "ritual_filter": True})
    assert "app" in reader._entries
    assert not any(is_ritual_key_point(p) for p in reader._entries["app"].key_points)
    assert any("paginate" in p.lower() for p in reader._entries["app"].key_points)


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


def test_skillsbench_task_runner_not_excluded_by_platform(pool_cfg):
    cfg = {"runtime_platform": "skillsbench-batch", "exclude_skills_from_pool": []}
    assert not is_excluded_hot_skill("skillsbench-task-runner", cfg)


def test_detect_claimed_success_mismatch():
    from agent.hot_skills import build_hot_pool_outcome, detect_claimed_success_mismatch

    assert detect_claimed_success_mismatch(
        {"task_success": False},
        {"final_response": "All tests passed successfully."},
    )
    assert not detect_claimed_success_mismatch(
        {"task_success": True},
        {"final_response": "All tests passed successfully."},
    )
    outcome = build_hot_pool_outcome(
        evaluation={"task_success": False, "tests_passed": 0, "tests_total": 5},
        run_result={"final_response": "Task is complete — all tests passed."},
        benchmark="skillsbench",
    )
    assert outcome["claimed_success_mismatch"] is True


def test_heuristic_claimed_success_mismatch_marks_injected_harmful():
    items = [{"id": "0", "skill": "a", "point": "verify in container", "via": "inject"}]
    labels = heuristic_outcome_attributions(
        items,
        {"success": False, "claimed_success_mismatch": True, "iterations": 20},
        low_step_threshold=12,
    )
    assert labels == {"0": "harmful"}


def test_admit_gate_drops_oracle_tips_deterministically(pool_cfg):
    pool = HotSkillPool({**pool_cfg, "admit_transfer_gate": True})
    pool.record(
        name="skillsbench-task-runner",
        content="",
        key_points=[
            "NEVER claim success without host outputs",
            "Test pre-existing solution before writing a new one",
        ],
        turn=1,
    )
    assert "skillsbench-task-runner" in pool._entries
    assert len(pool._entries["skillsbench-task-runner"].key_points) == 1
    assert "host outputs" in pool._entries["skillsbench-task-runner"].key_points[0]


def test_deterministic_reconcile_drops_oracle_tips(pool_cfg):
    from agent.hot_skills import HotSkillEntry, empty_point_utility

    pool = HotSkillPool(pool_cfg)
    pool._entries["x"] = HotSkillEntry(
        name="x",
        key_points=["Use reference solution from solution/solve.sh"],
        skill_dir=None,
        skill_md_mtime=0.0,
        recorded_turn=1,
        description="",
        tags=[],
        scope="",
        point_utilities=[empty_point_utility()],
    )
    pool.reconcile_pool(material_update=True)
    assert "x" not in pool._entries


def test_apply_outcome_feedback_reconciles_after_scoring(pool_cfg):
    cfg = {**pool_cfg, "max_entries": 2, "eviction_policy": "oldest", "reconcile_on_update": True}
    pool = HotSkillPool(cfg)
    pool._telemetry.reconcile_ran = False
    pool.record(name="good", content="", key_points=["NEVER skip graded outputs"], turn=1)
    pool.record(name="bad", content="", key_points=["ALWAYS paginate long API lists"], turn=2)
    pool.build_block(user_message="task", turn=2)
    pool._telemetry.reconcile_ran = False
    summary = pool.apply_outcome_feedback(
        {"success": False, "benchmark": "skillsbench", "iterations": 30},
        complete_fn=lambda _m: '{"labels": {"0": "harmful", "1": "harmful"}}',
    )
    assert summary["applied"] is True
    assert summary["attributions"]["harmful"] >= 1
    assert pool._entries["good"].point_utilities[0]["harmful"] >= 1


def test_admit_gate_drops_ritual_without_judge(pool_cfg):
    pool = HotSkillPool({**pool_cfg, "admit_transfer_gate": True})
    filtered = pool._gate_admission_points(
        "app",
        ["ALWAYS call done() when finished", "ALWAYS paginate long API lists"],
        scope="apis",
        context="send payment",
    )
    assert filtered == ["ALWAYS paginate long API lists"]


def test_admit_transfer_prompt_rejects_rituals():
    from agent.hot_skills import build_admit_transfer_system_prompt

    text = build_admit_transfer_system_prompt()
    assert "done()" in text
    assert "short" in text.lower() or "minimal" in text.lower()
    assert "show_api_doc" in text


def test_clamp_outcome_overwrites_helpful_on_failure():
    items = [
        {"id": "0", "via": "inject", "skill": "a", "point": "tip"},
        {"id": "1", "via": "history", "skill": "a", "point": "other"},
    ]
    labels, n = clamp_outcome_attributions(
        items, {"0": "helpful", "1": "helpful"}, {"success": False}
    )
    assert labels == {"0": "harmful", "1": "helpful"}
    assert n == 1


def test_clamp_outcome_fills_empty_llm_on_failure():
    items = [
        {"id": "0", "via": "inject", "skill": "a", "point": "tip"},
        {"id": "1", "via": "history", "skill": "b", "point": "seen"},
    ]
    labels, n = clamp_outcome_attributions(items, {}, {"success": False})
    assert labels == {"0": "harmful", "1": "irrelevant"}
    assert n == 1


def test_failure_floor_persists_when_llm_labels_helpful(pool_cfg):
    pool = HotSkillPool(
        {**pool_cfg, "inject_retrieve": False, "outcome_feedback_heuristic": False}
    )
    tip = "NEVER invent required parameter values"
    pool.record(name="app", content="", key_points=[tip], turn=1)
    pool.build_block(user_message="send payment", turn=1)

    def judge(_msgs):
        return json.dumps({"labels": {"0": "helpful"}})

    summary = pool.apply_outcome_feedback(
        {"success": False, "reward": 0.0, "iterations": 40},
        complete_fn=judge,
    )
    assert summary["applied"] is True
    assert summary["attribution_source"] == "llm+failure_floor"
    assert summary["attributions"]["harmful"] == 1
    assert summary.get("labels_clamped", 0) >= 1
    util = pool._entries["app"].point_utilities[0]
    assert util["harmful"] == 1
    assert util["helpful"] == 0
    assert pool.export_telemetry()["outcome_feedback"]["labels_clamped"] >= 1


def test_failure_floor_applies_without_heuristic_on_empty_llm(pool_cfg):
    pool = HotSkillPool(
        {**pool_cfg, "inject_retrieve": False, "outcome_feedback_heuristic": False}
    )
    tip = "ALWAYS paginate long API lists"
    pool.record(name="app", content="", key_points=[tip], turn=1)
    pool.build_block(user_message="list invoices", turn=1)
    summary = pool.apply_outcome_feedback(
        {"success": False, "reward": 0.0, "iterations": 30},
        complete_fn=lambda _m: "",
    )
    assert summary["applied"] is True
    assert summary["attribution_source"] == "failure_floor"
    assert summary["attributions"]["harmful"] == 1
    assert pool._entries["app"].point_utilities[0]["harmful"] == 1


def test_inject_throttle_when_inject_cohort_underperforms(pool_cfg, tmp_path):
    cfg = {
        **pool_cfg,
        "inject_retrieve": False,
        "inject_throttle": True,
        "inject_throttle_min_episodes": 4,
        "inject_throttle_margin": 0.05,
        "inject_throttle_probe_every": 100,
        "inject_filter_utilities": False,  # isolate throttle from score silence
        "persist_across_conversations": True,
        "persist_path": str(tmp_path / "hot_pool.json"),
        "outcome_feedback_heuristic": True,
    }
    pool = HotSkillPool(cfg)
    tip = "NEVER invent required parameter values"
    pool.record(name="app", content="", key_points=[tip], turn=1)

    # 4 inject fails (cohort counters only — avoid utility silence mid-loop).
    for i in range(4):
        pool.clear_exposed_tips()
        block = pool.build_block(user_message="pay someone", turn=i)
        assert block, f"expected inject before throttle, episode {i}"
        pool._note_inject_cohort(injected=True, success=False)

    # 4 no-inject successes → throttle.
    pool._config["inject_on_turn"] = False
    for i in range(4):
        pool.clear_exposed_tips()
        assert pool.build_block(user_message="pay someone", turn=20 + i) == ""
        pool._note_inject_cohort(injected=False, success=True)
    pool._config["inject_on_turn"] = True

    assert pool._inject_throttled is True
    pool._episodes_since_inject = 0
    pool.clear_exposed_tips()
    blocked = pool.build_block(user_message="pay someone", turn=40)
    assert blocked == ""
    tel = pool.export_telemetry()["inject"]
    assert tel["throttled"] is True

    reloaded = HotSkillPool(cfg)
    assert reloaded._inject_throttled is True
    assert reloaded._cohort_inject_n >= 4
    assert reloaded._cohort_no_inject_n >= 4


def test_no_exposed_tips_still_updates_inject_cohort(pool_cfg):
    pool = HotSkillPool({**pool_cfg, "inject_on_turn": False})
    pool.record(
        name="app",
        content="",
        key_points=["NEVER invent required parameter values"],
        turn=1,
    )
    pool.clear_exposed_tips()
    pool.build_block(user_message="pay", turn=1)
    summary = pool.apply_outcome_feedback(
        {"success": True, "reward": 1.0, "iterations": 5},
        complete_fn=lambda _m: "",
    )
    assert summary["skipped_reason"] == "no_exposed_tips"
    assert pool._cohort_no_inject_n == 1
    assert pool._cohort_no_inject_success == 1


def test_inject_throttle_probe_every_n_episodes(pool_cfg):
    cfg = {
        **pool_cfg,
        "inject_retrieve": False,
        "inject_throttle": True,
        "inject_throttle_min_episodes": 2,
        "inject_throttle_margin": 0.0,
        "inject_throttle_probe_every": 2,
        "outcome_feedback_heuristic": True,
    }
    pool = HotSkillPool(cfg)
    pool.record(
        name="app",
        content="",
        key_points=["NEVER invent required parameter values"],
        turn=1,
    )
    pool._cohort_inject_n = 2
    pool._cohort_inject_success = 0
    pool._cohort_no_inject_n = 2
    pool._cohort_no_inject_success = 2
    pool._inject_throttled = True
    pool._episodes_since_inject = 0

    assert pool.build_block(user_message="pay", turn=1) == ""
    pool._note_inject_cohort(injected=False, success=True)
    assert pool.build_block(user_message="pay", turn=2) == ""
    pool._note_inject_cohort(injected=False, success=True)
    # episodes_since_inject == 2 → probe allowed
    probe = pool.build_block(user_message="pay", turn=3)
    assert probe
    assert pool.export_telemetry()["inject"]["throttle_probe"] is True


def test_pre_throttle_holdout_every_n_labeled_episodes(pool_cfg):
    """Even before throttle arms, force periodic no-inject to fill control arm."""
    cfg = {
        **pool_cfg,
        "inject_retrieve": False,
        "inject_throttle": True,
        "inject_throttle_min_episodes": 50,  # do not arm throttle
        "inject_throttle_probe_every": 3,
        "inject_filter_utilities": False,  # isolate holdout from score silence
        "outcome_feedback_heuristic": True,
    }
    pool = HotSkillPool(cfg)
    pool.record(
        name="app",
        content="",
        key_points=["NEVER invent required parameter values"],
        turn=1,
    )
    for i in range(3):
        pool.clear_exposed_tips()
        block = pool.build_block(user_message="pay someone", turn=i)
        assert block, f"expected inject on episode {i}"
        pool._note_inject_cohort(injected=True, success=True)
    assert pool._cohort_inject_n == 3
    assert pool._cohort_no_inject_n == 0
    # labeled % 3 == 0 → next build is a holdout
    pool.clear_exposed_tips()
    holdout = pool.build_block(user_message="pay someone", turn=10)
    assert holdout == ""
    tel = pool.export_telemetry()["inject"]
    assert tel["throttle_probe"] is True
    assert tel["throttled"] is False
    pool._note_inject_cohort(injected=False, success=True)
    assert pool._cohort_no_inject_n == 1
    # After holdout, inject resumes
    pool.clear_exposed_tips()
    resumed = pool.build_block(user_message="pay someone", turn=11)
    assert resumed


def test_record_clears_inject_throttle_for_new_material(pool_cfg):
    pool = HotSkillPool({**pool_cfg, "inject_retrieve": False, "inject_throttle": True})
    pool._inject_throttled = True
    pool.record(
        name="app",
        content="",
        key_points=["ALWAYS paginate long API lists"],
        turn=1,
    )
    assert pool._inject_throttled is False
    assert pool.build_block(user_message="list", turn=1)


def test_admit_transfer_parse():
    from agent.hot_skills import parse_admit_transfer_ids

    items = [{"id": "0", "point": "good"}, {"id": "1", "point": "bad"}]
    assert parse_admit_transfer_ids('{"admit": ["0"]}', items) == ["0"]
    assert parse_admit_transfer_ids('{"admit": []}', items) == []


def test_admit_transfer_gate_with_judge(pool_cfg):
    items = [
        {"id": "0", "point": "NEVER claim pass without host verification"},
        {"id": "1", "point": "Test pre-existing solution before writing a new one"},
    ]

    def judge(candidates, skill_name, scope, context):
        return ["0"]

    pool = HotSkillPool({**pool_cfg, "admit_transfer_gate": True})
    pool.admission_judge = judge
    filtered = pool._gate_admission_points(
        "skillsbench-task-runner",
        [it["point"] for it in items],
        scope="batch tasks",
        context="solve task",
    )
    assert filtered == ["NEVER claim pass without host verification"]


def test_hermes_agent_globally_excluded_from_pool(pool_cfg):
    assert is_excluded_hot_skill("hermes-agent", {"runtime_platform": "cli"})
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="hermes-agent",
        content="<!-- hermes-hot -->\n- ALWAYS use /tools\n<!-- /hermes-hot -->",
        turn=1,
    )
    assert "hermes-agent" not in pool._entries


def test_junk_filter_oracle_shortcut_bullets():
    assert is_junk_key_point("Test pre-existing solution before writing a new one")
    assert is_junk_key_point("Static ground truth in test file — skip heavy build")
    assert not is_junk_key_point("NEVER claim success from in-container pytest alone")


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


def test_tokenize_retrieve_drops_process_stopwords():
    toks = tokenize_retrieve("ALWAYS check the API before you call venmo login")
    assert "venmo" in toks
    assert "login" in toks
    assert "always" not in toks
    assert "check" not in toks
    assert "the" not in toks


def test_observation_user_message_detection():
    assert is_observation_user_message("Output:\n```\nok\n```")
    assert is_observation_user_message("```\nprint(1)\n```")
    assert is_observation_user_message("Observation: You are in the kitchen")
    assert not is_observation_user_message("Reset friends on venmo")


def test_retrieve_overlap_ignores_pool_wide_tokens():
    query = frozenset({"venmo", "never"})
    lex = frozenset({"venmo", "never", "phone"})
    hits = retrieve_overlap(query, lex, ignore=frozenset({"never"}))
    assert hits == frozenset({"venmo"})


def test_inject_retrieve_omits_off_domain_skill(pool_cfg):
    cfg = dict(pool_cfg)
    cfg["inject_k"] = 8
    cfg["runtime_platform"] = "appworld-batch"
    cfg["admit_domain_gate"] = False
    pool = HotSkillPool(cfg)
    pool.record(
        name="appworld-api-interaction",
        content="",
        description="AppWorld sandbox APIs",
        tags=["appworld", "sandbox", "apis"],
        key_points=[
            "ALWAYS paginate fully",
            "Venmo search returns unrelated users — verify email",
        ],
        turn=1,
    )
    pool.record(
        name="dev-config",
        content="",
        description="Configure the local agent install",
        tags=["setup", "configuration", "cli", "gateway"],
        key_points=[
            "Use get_hermes_home for profile-safe paths",
            "Config values go in config.yaml",
        ],
        turn=2,
    )
    block = pool.build_block(user_message="Reset friends on venmo to match my phone", turn=3)
    assert "Venmo search" in block
    assert "get_hermes_home" not in block
    tel = pool.export_telemetry()["inject"]
    assert "dev-config" in tel["skills_omitted_retrieve"]
    assert tel["retrieve_mode"] == "overlap"


def test_inject_retrieve_ranks_overlapping_tip_first(pool_cfg):
    cfg = dict(pool_cfg)
    cfg["inject_k"] = 1
    pool = HotSkillPool(cfg)
    pool.record(
        name="appworld-api-interaction",
        content="",
        tags=["appworld"],
        key_points=[
            "ALWAYS paginate fully — don't stop at the first page",
            "Venmo search returns unrelated users — verify email",
        ],
        turn=1,
    )
    block = pool.build_block(user_message="Request $28 on Venmo from Melissa", turn=2)
    assert "Venmo search" in block
    assert "paginate" not in block
    assert pool.export_telemetry()["inject"]["point_count"] == 1


def test_inject_k_caps_block(pool_cfg):
    cfg = dict(pool_cfg)
    cfg["inject_k"] = 2
    pool = HotSkillPool(cfg)
    points = [
        "Venmo friends must be looked up via search",
        "Phone contacts use phone.login username",
        "ALWAYS paginate venmo transaction lists",
        "Spotify library songs need show_song",
    ]
    pool.record(name="appworld-api-interaction", content="", key_points=points, turn=1)
    block = pool.build_block(user_message="How many venmo friends did I add", turn=2)
    assert block.count("- ") == 2
    assert "venmo" in block.lower()


def test_observation_keeps_frozen_episode_query(pool_cfg):
    cfg = dict(pool_cfg)
    cfg["inject_k"] = 1
    pool = HotSkillPool(cfg)
    pool.record(
        name="appworld-api-interaction",
        content="",
        key_points=[
            "Venmo search returns unrelated users",
            "ALWAYS paginate fully",
        ],
        turn=1,
    )
    pool.build_block(user_message="Reset friends on venmo", turn=1)
    block = pool.build_block(user_message="Output:\n```\n[]\n```", turn=2)
    assert "Venmo search" in block
    assert pool._episode_query.startswith("Reset friends")
    assert pool.export_telemetry()["inject"]["retrieve_mode"] == "overlap"


def test_clear_exposed_tips_resets_episode_query(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.note_episode_query("Reset friends on venmo")
    assert pool._episode_query
    pool.clear_exposed_tips()
    assert pool._episode_query == ""
    assert pool._episode_query_tokens == frozenset()


def test_empty_query_falls_back_to_utility_without_omitting(pool_cfg):
    cfg = dict(pool_cfg)
    cfg["inject_k"] = 4
    pool = HotSkillPool(cfg)
    pool.record(
        name="nav",
        content="",
        key_points=["go to fridge first", "never open locked door"],
        turn=1,
    )
    block = pool.build_block(user_message="hello", turn=2)
    assert "fridge" in block
    assert "locked door" in block
    tel = pool.export_telemetry()["inject"]
    assert tel["retrieve_mode"] == "utility_fallback"
    assert tel["skills_omitted_retrieve"] == []


def test_platform_fallback_keeps_in_domain_when_query_misses_tips(pool_cfg):
    cfg = dict(pool_cfg)
    cfg["inject_k"] = 4
    cfg["runtime_platform"] = "appworld-batch"
    cfg["admit_domain_gate"] = False
    pool = HotSkillPool(cfg)
    pool.record(
        name="appworld-api-interaction",
        content="",
        tags=["appworld", "sandbox", "apis"],
        key_points=["ALWAYS paginate fully"],
        turn=1,
    )
    pool.record(
        name="dev-config",
        content="",
        tags=["setup", "cli", "gateway"],
        key_points=["Use get_hermes_home for paths"],
        turn=2,
    )
    block = pool.build_block(user_message="walk the dog around the block", turn=3)
    assert "paginate" in block
    assert "get_hermes_home" not in block
    assert pool.export_telemetry()["inject"]["retrieve_mode"] == "platform_fallback"


def test_skill_retrieve_tokens_include_tip_nouns():
    toks = skill_retrieve_tokens(
        "appworld-api-interaction",
        tags=["appworld", "sandbox"],
        scope="AppWorld sandbox APIs",
        points=["Venmo search returns unrelated users"],
    )
    assert "appworld" in toks
    assert "venmo" in toks
    assert "sandbox" in toks


def test_platform_retrieve_tokens_include_aliases():
    toks = platform_retrieve_tokens("appworld-batch")
    assert "appworld" in toks
    assert "apis" in toks
    assert "skillsbench" in platform_retrieve_tokens("skillsbench-batch")


def test_admit_domain_gate_omits_distinctive_off_domain_on_appworld(pool_cfg):
    cfg = {**pool_cfg, "runtime_platform": "appworld-batch"}
    pool = HotSkillPool(cfg)
    pool.note_episode_query("Reset friends on venmo to match my phone")
    pool.record(
        name="dev-config",
        content="",
        description="Configure the local agent install",
        tags=["setup", "configuration", "cli", "gateway"],
        key_points=["Use get_hermes_home for profile-safe paths"],
        turn=1,
    )
    pool.record(
        name="appworld-api-interaction",
        content="",
        tags=["appworld", "sandbox", "apis"],
        key_points=["ALWAYS paginate fully"],
        turn=1,
    )
    assert "dev-config" not in pool._entries
    assert "appworld-api-interaction" in pool._entries
    assert pool._telemetry.records_skipped_off_domain == 1


def test_admit_domain_gate_keeps_appworld_skill_when_query_misses(pool_cfg):
    cfg = {**pool_cfg, "runtime_platform": "appworld-batch"}
    pool = HotSkillPool(cfg)
    pool.note_episode_query("walk the dog around the block")
    pool.record(
        name="appworld-api-interaction",
        content="",
        tags=["appworld", "sandbox"],
        key_points=["ALWAYS paginate fully"],
        turn=1,
    )
    assert "appworld-api-interaction" in pool._entries


def test_admit_domain_gate_omits_appworld_skill_on_alfworld(pool_cfg):
    cfg = {**pool_cfg, "runtime_platform": "alfworld-batch"}
    pool = HotSkillPool(cfg)
    pool.note_episode_query("put the apple in the fridge")
    pool.record(
        name="appworld-api-interaction",
        content="",
        tags=["appworld", "sandbox", "apis"],
        key_points=["ALWAYS paginate fully"],
        turn=1,
    )
    pool.record(
        name="nav",
        content="",
        key_points=["go to the fridge first", "never open a locked door"],
        turn=1,
    )
    assert "appworld-api-interaction" not in pool._entries
    assert "nav" in pool._entries
    pool.record(
        name="helper",
        content="",
        key_points=["ALWAYS look around before you move"],
        turn=2,
    )
    assert "helper" in pool._entries


def test_admit_domain_gate_keeps_alfworld_skill_via_tip_overlap(pool_cfg):
    cfg = {**pool_cfg, "runtime_platform": "alfworld-batch"}
    pool = HotSkillPool(cfg)
    pool.note_episode_query("put the apple in the fridge")
    pool.record(
        name="heat-apple",
        content="",
        key_points=["heat the apple before you put it in the fridge"],
        turn=1,
    )
    assert "heat-apple" in pool._entries


def test_admit_domain_gate_skillsbench_fail_open_on_generic_query(pool_cfg):
    cfg = {**pool_cfg, "runtime_platform": "skillsbench-batch"}
    pool = HotSkillPool(cfg)
    pool.note_episode_query("complete the task in the environment")
    pool.record(
        name="xlsx-formula-audit",
        content="",
        tags=["spreadsheet"],
        key_points=["ALWAYS use Excel formulas instead of Python values"],
        turn=1,
    )
    assert "xlsx-formula-audit" in pool._entries
    assert pool._telemetry.records_skipped_off_domain == 0


def test_admit_domain_gate_skillsbench_omits_framework_skill(pool_cfg):
    cfg = {**pool_cfg, "runtime_platform": "skillsbench-batch"}
    pool = HotSkillPool(cfg)
    pool.note_episode_query("complete the task in the environment")
    pool.record(
        name="claude-code",
        content="",
        tags=["claude", "vscode"],
        key_points=["ALWAYS open the workspace folder first"],
        turn=1,
    )
    assert "claude-code" not in pool._entries
    assert pool._telemetry.records_skipped_off_domain == 1


def test_admit_domain_gate_empty_query_no_platform_admits(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="dev-config",
        content="",
        tags=["setup", "cli", "gateway"],
        key_points=["Use get_hermes_home for paths"],
        turn=1,
    )
    assert "dev-config" in pool._entries


def test_admit_domain_gate_observation_does_not_replace_episode(pool_cfg):
    cfg = {**pool_cfg, "runtime_platform": "appworld-batch"}
    pool = HotSkillPool(cfg)
    pool.note_episode_query("Reset friends on venmo")
    pool.record(
        name="appworld-api-interaction",
        content="",
        tags=["appworld"],
        key_points=["Venmo search returns unrelated users"],
        user_message="Output:\n```\n[]\n```",
        turn=2,
    )
    assert "appworld-api-interaction" in pool._entries


def test_admit_domain_gate_disabled_admits_off_domain(pool_cfg):
    cfg = {
        **pool_cfg,
        "runtime_platform": "appworld-batch",
        "admit_domain_gate": False,
    }
    pool = HotSkillPool(cfg)
    pool.note_episode_query("Reset friends on venmo")
    pool.record(
        name="dev-config",
        content="",
        tags=["setup", "cli", "gateway"],
        key_points=["Use get_hermes_home for paths"],
        turn=1,
    )
    assert "dev-config" in pool._entries


def test_set_platform_evicts_foreign_closed_world_not_generic(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="appworld-api-interaction",
        content="",
        tags=["appworld", "sandbox"],
        key_points=["ALWAYS paginate fully"],
        turn=1,
    )
    pool.record(
        name="xlsx-formula-audit",
        content="",
        tags=["spreadsheet"],
        key_points=["ALWAYS use Excel formulas"],
        turn=2,
    )
    pool.set_platform("alfworld-batch")
    assert "appworld-api-interaction" not in pool._entries
    assert "xlsx-formula-audit" in pool._entries


def test_is_off_session_domain_helpers():
    assert is_off_session_domain(
        "github",
        tags=["git"],
        points=["ALWAYS open a pull request"],
        platform="appworld-batch",
        query_tokens=frozenset({"venmo", "friends"}),
    )
    assert not is_off_session_domain(
        "nav",
        points=["go to the fridge first"],
        platform="alfworld-batch",
        query_tokens=frozenset({"apple", "fridge"}),
    )
    assert not is_off_session_domain(
        "xlsx-formula-audit",
        tags=["spreadsheet"],
        points=["ALWAYS use Excel formulas"],
        platform="skillsbench-batch",
        query_tokens=frozenset({"complete", "environment"}),
    )


def test_sidechannel_thinking_off_extra_body_qwen_only():
    qwen = sidechannel_thinking_off_extra_body(model="Qwen/Qwen3.6-27B")
    assert qwen["chat_template_kwargs"]["enable_thinking"] is False
    assert qwen["enable_thinking"] is False
    assert "reasoning" not in qwen
    openai = sidechannel_thinking_off_extra_body(model="gpt-4o")
    assert openai == {}
    router = sidechannel_thinking_off_extra_body(model="gpt-4o", is_openrouter=True)
    assert router == {"reasoning": {"enabled": False}}


def test_recover_json_object_from_think_block():
    raw = '<think>ok</think>\n{"labels": {"0": "helpful"}}'
    assert recover_json_object(raw) == '{"labels": {"0": "helpful"}}'
    inside = '<think>{"labels": {"1": "harmful"}}</think>'
    assert '"1"' in recover_json_object(inside)


def test_note_outcome_judge_meta_in_telemetry(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.note_outcome_judge_meta(
        {
            "finish_reason": "length",
            "raw_chars": 80,
            "stripped_chars": 0,
            "raw_preview": "<think>only reasoning</think>",
            "stripped_preview": "",
            "max_tokens": 2048,
            "thinking_off": True,
            "retries": 1,
            "recovered_json": False,
        }
    )
    of = pool.export_telemetry()["outcome_feedback"]
    assert of["judge_finish_reason"] == "length"
    assert of["judge_raw_chars"] == 80
    assert of["judge_stripped_chars"] == 0
    assert of["judge_thinking_off"] is True
    assert of["judge_retries"] == 1
    assert of["judge_max_tokens"] == 2048
    assert "think" in of["judge_raw_preview"]


def test_compress_tool_args_drops_observation_blobs():
    slim = compress_tool_args(
        json.dumps(
            {
                "command": "ls\ncat huge",
                "stdout": "x" * 5000,
                "content": "full file",
                "path": "/tmp/out.txt",
            }
        ),
        max_chars=40,
    )
    assert slim["command"] == "ls"
    assert "stdout" not in slim
    assert "content" not in slim
    assert slim["path"] == "/tmp/out.txt"


def test_compress_env_code_extracts_appworld_apis():
    snippets = compress_env_code(
        "x = 1\napis.venmo.search_users(query='a')\ncomplete_task()\n",
        max_chars=60,
    )
    assert any("apis.venmo.search_users" in s for s in snippets)
    assert any("complete_task" in s for s in snippets)


def test_compressed_episode_log_from_tool_calls_skips_output_user():
    messages = [
        {"role": "user", "content": "Reset friends on venmo"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps({"command": "python -c 'print(1)'"}),
                    }
                },
                {"function": {"name": "todo", "arguments": "{}"}},
            ],
        },
        {"role": "user", "content": "Output:\n```\n[]\n```"},
        {"role": "assistant", "content": "done"},
    ]
    log = build_compressed_episode_log(messages=messages, episode_query="")
    assert log["query"].startswith("Reset friends")
    assert log["final_response"] == "done"
    names = [ev["name"] for ev in log["events"]]
    assert names == ["terminal"]
    assert "todo" not in names
    assert not any("Output" in json.dumps(ev) for ev in log["events"])


def test_compressed_episode_log_from_alfworld_actions():
    log = build_compressed_episode_log(
        episode_query="put the apple in the fridge",
        env_actions=["go to fridge 1", "take apple 1", "go to fridge 1"],
        run_result={"final_response": "I put it there"},
    )
    assert log["n_events"] == 3
    assert log["events"][0]["via"] == "env"
    assert "fridge" in str(log["events"][0]["args"])
    assert log["tool_counts"]["action"] == 3


def test_compressed_episode_log_from_appworld_steps():
    log = build_compressed_episode_log(
        episode_query="Reset friends on venmo",
        run_result={
            "steps": [
                {"code": "apis.venmo.search_users(query='Ann')\nprint(x)"},
                {"code": "complete_task()"},
            ]
        },
    )
    names = [ev["name"] for ev in log["events"]]
    assert any("venmo.search_users" in n for n in names)
    assert any("complete_task" in n for n in names)


def test_compressed_episode_log_head_tail_truncation():
    actions = [f"go to drawer {i}" for i in range(20)]
    log = build_compressed_episode_log(env_actions=actions, max_events=10)
    assert log["n_events"] == 20
    assert log["n_truncated"] == 10
    assert len(log["events"]) == 10
    assert "drawer 0" in str(log["events"][0]["args"])
    assert "drawer 19" in str(log["events"][-1]["args"])


def test_collect_outcome_episode_log_respects_config_off(pool_cfg):
    pool = HotSkillPool({**pool_cfg, "outcome_judge_log": False})
    log = collect_outcome_episode_log(
        pool=pool,
        run_result={"messages": [{"role": "assistant", "tool_calls": []}]},
    )
    assert log == {}


def test_apply_outcome_feedback_records_episode_log_telemetry(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(name="nav", content="", key_points=["go to fridge first"], turn=1)
    pool.build_block(user_message="put apple in fridge", turn=1)

    def fake_complete(msgs):
        body = json.loads(msgs[1]["content"])
        assert body["episode_log"]["events"]
        assert body["episode_log"]["query"].startswith("put apple")
        return json.dumps({"labels": {"0": "helpful"}})

    summary = pool.apply_outcome_feedback(
        {"success": True, "reward": 1.0, "iterations": 4},
        complete_fn=fake_complete,
        episode_log=build_compressed_episode_log(
            episode_query="put apple in fridge",
            env_actions=["go to fridge 1", "take apple 1"],
        ),
    )
    assert summary["attribution_source"] == "llm"
    tel = pool.export_telemetry()["outcome_feedback"]
    assert tel["judge_log_events"] == 2
    assert "action" in tel["judge_log_preview"]

