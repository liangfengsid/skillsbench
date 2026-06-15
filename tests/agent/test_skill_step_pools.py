"""Tests for optional per-step skill variant pools (agent/skill_step_pools.py)."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def skills_cfg_on():
    return {"step_pools": {"enabled": True, "default_max_variants_per_step": 5}}


@pytest.fixture
def skills_cfg_off():
    return {"step_pools": {"enabled": False}}


def test_expand_disabled_returns_unchanged(skills_cfg_off, tmp_path):
    from agent.skill_step_pools import expand_skill_content_step_pools

    md = """<!-- hermes-step id="a" -->\nhello\n<!-- /hermes-step -->"""
    (tmp_path / "SKILL.md").write_text(md, encoding="utf-8")
    out = expand_skill_content_step_pools(md, tmp_path, "t", skills_cfg_off)
    assert out == md


def test_expand_enabled_replaces_marker(skills_cfg_on, tmp_path):
    from agent.skill_step_pools import expand_skill_content_step_pools

    md = """<!-- hermes-step id="boot" max_variants="3" -->\nRun `npm ci`\n<!-- /hermes-step -->\n"""
    (tmp_path / "SKILL.md").write_text(md, encoding="utf-8")
    out = expand_skill_content_step_pools(md, tmp_path, "my-skill", skills_cfg_on)
    assert "Step pool `boot`" in out
    assert "Variant `baseline`" in out
    assert "skill_step_variant" in out
    assert "Run `npm ci`" in out
    assert "<!-- hermes-step id=" not in out


def test_variant_rank_score_strategies():
    from agent.skill_step_pools import variant_rank_score

    v = {"id": "x", "success": 1, "fail": 9, "attempts": 10}
    assert abs(variant_rank_score(v, "laplace", pool_total_trials=10) - 2.0 / 12.0) < 1e-9
    assert abs(variant_rank_score(v, "raw", pool_total_trials=10) - 0.1) < 1e-9
    w = variant_rank_score(v, "wilson", pool_total_trials=10)
    assert 0.0 <= w <= 1.0
    u = variant_rank_score(v, "ucb1", pool_total_trials=100, skills_cfg={"step_pools": {"ucb1_c": 1.0}})
    assert u > 0.1


def test_rank_strategy_raw_changes_order_vs_laplace(tmp_path):
    from agent.skill_step_pools import expand_skill_content_step_pools, save_step_pools

    md = """<!-- hermes-step id="x" -->\nBASE\n<!-- /hermes-step -->"""
    save_step_pools(
        tmp_path,
        {
            "version": 1,
            "steps": {
                "x": {
                    "variants": [
                        {
                            "id": "perfect_small_n",
                            "body": "SMALL_N_BODY",
                            "success": 1,
                            "fail": 0,
                            "attempts": 1,
                        },
                        {
                            "id": "strong_large_n",
                            "body": "LARGE_N_BODY",
                            "success": 8,
                            "fail": 2,
                            "attempts": 10,
                        },
                    ]
                }
            },
        },
    )
    cfg_lap = {"step_pools": {"enabled": True, "default_max_variants_per_step": 5, "rank_strategy": "laplace"}}
    out_lap = expand_skill_content_step_pools(md, tmp_path, "s", cfg_lap)
    cfg_raw = {"step_pools": {"enabled": True, "default_max_variants_per_step": 5, "rank_strategy": "raw"}}
    out_raw = expand_skill_content_step_pools(md, tmp_path, "s", cfg_raw)
    # Raw: 100% (1/1) beats 80% (8/10) → small-n variant first
    assert out_raw.index("SMALL_N_BODY") < out_raw.index("LARGE_N_BODY")
    # Laplace: (1+1)/(1+2)=2/3 vs (8+2)/(10+2)=10/12 — second wins → large-n first
    assert out_lap.index("LARGE_N_BODY") < out_lap.index("SMALL_N_BODY")


def test_merge_extra_variant_and_ordering(skills_cfg_on, tmp_path):
    from agent.skill_step_pools import expand_skill_content_step_pools, save_step_pools

    md = """<!-- hermes-step id="x" -->\nBASE\n<!-- /hermes-step -->"""
    save_step_pools(
        tmp_path,
        {
            "version": 1,
            "steps": {
                "x": {
                    "variants": [
                        {
                            "id": "alt",
                            "body": "ALT BODY",
                            "success": 10,
                            "fail": 0,
                            "attempts": 10,
                        }
                    ]
                }
            },
        },
    )
    out = expand_skill_content_step_pools(md, tmp_path, "s", skills_cfg_on)
    # alt scores higher than fresh baseline — should sort before baseline when no manual_order
    assert out.index("ALT BODY") < out.index("BASE")


def test_record_attempt_baseline_creates_row(tmp_path):
    from agent.skill_step_pools import load_step_pools, record_step_attempt

    r = record_step_attempt(tmp_path, "st", "baseline", "success")
    assert r["success"]
    data = load_step_pools(tmp_path)
    v = data["steps"]["st"]["variants"]
    assert any(x["id"] == "baseline" and x["success"] == 1 for x in v)


def test_add_variant_respects_cap(tmp_path):
    from agent.skill_step_pools import add_step_variant, load_step_pools

    (tmp_path / "SKILL.md").write_text(
        '<!-- hermes-step id="s" max_variants="2" -->\nA\n<!-- /hermes-step -->\n',
        encoding="utf-8",
    )
    cfg = {"step_pools": {"enabled": True, "default_max_variants_per_step": 5}}
    assert add_step_variant(tmp_path, "s", "body1", "v1", None, skills_cfg=cfg)["success"]
    assert add_step_variant(tmp_path, "s", "body2", "v2", None, skills_cfg=cfg)["success"]
    # cap 2 total shown = baseline + 1 json; adding third should evict weakest json
    assert add_step_variant(tmp_path, "s", "body3", "v3", None, skills_cfg=cfg)["success"]
    ids = {v["id"] for v in load_step_pools(tmp_path)["steps"]["s"]["variants"]}
    # max_variants=2 ⇒ baseline + one JSON row; third add evicts older JSON variants.
    assert ids == {"v3"}


def test_patch_variant_substring(tmp_path):
    from agent.skill_step_pools import add_step_variant, patch_step_variant_body

    (tmp_path / "SKILL.md").write_text(
        '<!-- hermes-step id="s" -->\nA\n<!-- /hermes-step -->\n',
        encoding="utf-8",
    )
    cfg = {"step_pools": {"enabled": True, "default_max_variants_per_step": 5}}
    add_step_variant(tmp_path, "s", "hello world", "w", None, skills_cfg=cfg)
    assert patch_step_variant_body(tmp_path, "s", "w", "world", "there")["success"]
    data = json.loads((tmp_path / "step_pools.json").read_text(encoding="utf-8"))
    bodies = [v["body"] for v in data["steps"]["s"]["variants"] if v["id"] == "w"]
    assert bodies == ["hello there"]


def test_manual_order_prefix(skills_cfg_on, tmp_path):
    from agent.skill_step_pools import expand_skill_content_step_pools, save_step_pools

    md = """<!-- hermes-step id="z" -->\nB\n<!-- /hermes-step -->"""
    save_step_pools(
        tmp_path,
        {
            "version": 1,
            "steps": {
                "z": {
                    "manual_order": ["baseline", "slow"],
                    "variants": [
                        {
                            "id": "slow",
                            "body": "SLOW",
                            "success": 100,
                            "fail": 0,
                            "attempts": 100,
                        }
                    ],
                }
            },
        },
    )
    out = expand_skill_content_step_pools(md, tmp_path, "s", skills_cfg_on)
    assert out.index("B") < out.index("SLOW")


def test_skill_step_variant_tool_gated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "skills" / "t").mkdir(parents=True)
    (tmp_path / "skills" / "t" / "SKILL.md").write_text(
        '<!-- hermes-step id="a" -->\nx\n<!-- /hermes-step -->', encoding="utf-8"
    )

    from tools.skill_step_variant_tool import skill_step_variant

    raw = skill_step_variant(
        "record_attempt",
        "t",
        step_id="a",
        variant_id="baseline",
        outcome="success",
    )
    assert "enabled" in raw.lower() or "false" in raw.lower()
