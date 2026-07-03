"""Tests for agent/hot_skills.py — hot skill key point pool."""

import json

import pytest

from agent.hot_skills import (
    HotSkillPool,
    build_hot_skills_block,
    compute_alignment_hits,
    extract_hot_key_points,
    load_hot_skills_config,
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
        "entry_schedule": "per_skill",
        "max_skills": 3,
        "max_chars": 2000,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "ttl_turns": 5,
        "inject_on_turn": True,
        "skip_if_in_history": True,
        "history_lookback": 10,
        "hydrate_from_history": True,
        "hydrate_limit": 10,
        "semantic_prefetch": True,
        "semantic_min_score": 1,
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


def test_record_skips_when_no_key_points(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(name="empty", content="Just prose with no guardrails at all.", turn=1)
    assert "empty" not in pool._entries


def test_load_hot_skills_config_merges_defaults():
    cfg = load_hot_skills_config({"hot_pool": {"max_skills": 2, "enabled": False}})
    assert cfg["max_skills"] == 2
    assert cfg["enabled"] is False
    assert cfg["max_points_per_skill"] == 8
    assert cfg["entry_schedule"] == "global_pool"
    assert cfg["max_entries"] == 12


def test_load_hot_skills_config_env_enabled_override(monkeypatch):
    monkeypatch.setenv("HERMES_HOT_POOL_ENABLED", "0")
    monkeypatch.setenv("HERMES_HOT_POOL_PERSIST", "1")
    cfg = load_hot_skills_config({"hot_pool": {"enabled": True, "persist_across_conversations": True}})
    assert cfg["enabled"] is False
    assert cfg["persist_across_conversations"] is False

    monkeypatch.setenv("HERMES_HOT_POOL_ENABLED", "1")
    cfg_on = load_hot_skills_config({"hot_pool": {"enabled": False}})
    assert cfg_on["enabled"] is True


def test_global_pool_max_entries_spreads_across_skills():
    cfg = {
        "enabled": True,
        "entry_schedule": "global_pool",
        "max_entries": 4,
        "max_pool_skills": 10,
        "max_chars": 4000,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "inject_on_turn": True,
    }
    pool = HotSkillPool(cfg)
    for i in range(3):
        points = "\n".join(f"- Rule {i}-{j}" for j in range(3))
        pool.record(
            name=f"skill-{i}",
            content=f"<!-- hermes-hot -->\n{points}\n<!-- /hermes-hot -->",
            turn=1,
        )
    block = pool.build_block(user_message="hello", turn=2)
    assert block.count("- Rule") == 4
    assert block.count("### skill-") >= 2


def test_global_pool_respects_per_skill_inject_cap():
    cfg = {
        "enabled": True,
        "entry_schedule": "global_pool",
        "max_entries": 20,
        "max_points_per_skill_inject": 2,
        "max_pool_skills": 10,
        "max_chars": 4000,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "inject_on_turn": True,
    }
    pool = HotSkillPool(cfg)
    pool.record(
        name="heavy",
        content="<!-- hermes-hot -->\n"
        + "\n".join(f"- Point {i}" for i in range(6))
        + "\n<!-- /hermes-hot -->",
        turn=1,
    )
    block = pool.build_block(user_message="hello", turn=2)
    assert block.count("- Point") == 2


def test_global_pool_lru_uses_max_pool_skills():
    cfg = {
        "enabled": True,
        "entry_schedule": "global_pool",
        "max_pool_skills": 3,
        "max_entries": 12,
        "max_chars": 4000,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "inject_on_turn": True,
    }
    pool = HotSkillPool(cfg)
    for name in ("a", "b", "c", "d"):
        pool.record(name=name, content=_SAMPLE_SKILL, turn=1)
    assert list(pool._entries.keys()) == ["b", "c", "d"]


def test_record_and_lru_eviction(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    for name in ("a", "b", "c", "d"):
        pool.record(name=name, content=_SAMPLE_SKILL, turn=1)
    assert list(pool._entries.keys()) == ["b", "c", "d"]


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


def test_build_block_respects_max_chars(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    many_points = "<!-- hermes-hot -->\n" + "\n".join(
        f"- Guardrail point number {i} with some extra text"
        for i in range(40)
    ) + "\n<!-- /hermes-hot -->"
    pool.record(name="big", content=many_points, turn=1)
    block = pool.build_block(user_message="hello", turn=2)
    assert len(block) <= 2200


def test_skip_if_in_history(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="github",
        content="<!-- hermes-hot -->\n- Use gh cli\n<!-- /hermes-hot -->",
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


def test_semantic_ranking_boosts_matching_skill(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="alpha",
        content="<!-- hermes-hot -->\n- Alpha rule\n<!-- /hermes-hot -->",
        description="unrelated",
        turn=1,
    )
    pool.record(
        name="github-auth",
        content="<!-- hermes-hot -->\n- Validate github oauth tokens\n<!-- /hermes-hot -->",
        description="github tokens",
        turn=1,
    )
    ranked = pool._rank_entries(list(pool._entries.values()), "fix github auth token")
    assert ranked[0].name == "github-auth"


def test_evict_by_ttl(pool_cfg):
    pool = HotSkillPool(pool_cfg)
    pool.record(
        name="old",
        content="<!-- hermes-hot -->\n- Old rule\n<!-- /hermes-hot -->",
        turn=1,
    )
    pool.record(
        name="fresh",
        content="<!-- hermes-hot -->\n- Fresh rule\n<!-- /hermes-hot -->",
        turn=8,
    )
    evicted = pool.evict_by_ttl(7)
    assert evicted == 1
    assert "old" not in pool._entries
    assert "fresh" in pool._entries


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
    assert "rule one" in out


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
        "entry_schedule": "per_skill",
        "max_skills": 5,
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


def test_persist_global_turn_ttl_across_instances(tmp_path):
    path = tmp_path / "hot_pool.json"
    cfg = {
        "enabled": True,
        "entry_schedule": "per_skill",
        "max_skills": 5,
        "max_chars": 2000,
        "max_points_per_skill": 8,
        "max_chars_per_point": 240,
        "ttl_turns": 1,
        "inject_on_turn": True,
        "persist_across_conversations": True,
        "persist_path": str(path),
    }
    pool_a = HotSkillPool(cfg)
    pool_a.on_turn_start(1)
    pool_a.record(
        name="old",
        content="<!-- hermes-hot -->\n- Old rule\n<!-- /hermes-hot -->",
        turn=pool_a.active_turn,
    )

    pool_b = HotSkillPool(cfg)
    pool_b.on_turn_start(1)
    pool_b.on_turn_start(1)
    pool_b.evict_by_ttl(pool_b.active_turn)
    assert "old" not in pool_b._entries


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
