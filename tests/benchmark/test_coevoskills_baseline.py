"""Unit tests for isolated CoEvoSkills baseline helpers (no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.baselines.coevoskills import oracle, prompts, skill_io
from benchmark.baselines.coevoskills.verifier import _parse_verifier_json


def test_opaque_signal_strips_details():
    full = {
        "task_success": False,
        "reward": 0.5,
        "ran": True,
        "stdout_tail": "SECRET_TEST_NAME failed because ...",
        "error": None,
    }
    opaque = oracle.opaque_signal(full)
    assert opaque == {
        "task_success": False,
        "reward": 0.5,
        "ran": True,
        "error": None,
    }
    assert "SECRET" not in json.dumps(opaque)


def test_opaque_oracle_feedback_has_no_test_body():
    msg = prompts.opaque_oracle_feedback(passed=False, reward=0.25)
    assert "OPAQUE ORACLE: FAIL" in msg
    assert "test_outputs" not in msg
    assert "assert" not in msg


def test_workspace_and_skills_roundtrip(tmp_path: Path):
    paths = skill_io.ensure_workspace(tmp_path, "demo-task")
    skill_dir = paths["skills"] / "evo-demo"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: evo-demo\ndescription: demo\n---\n# Demo\n",
        encoding="utf-8",
    )
    (skill_dir / "scripts" / "utils.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    packs = skill_io.list_skills(paths["skills"])
    assert len(packs) == 1
    assert packs[0].name == "evo-demo"
    assert "def f" in packs[0].script_files["utils.py"]
    listing = skill_io.skills_listing_text(paths["skills"])
    assert "evo-demo" in listing


def test_parse_verifier_json_extracts_fields():
    text = 'Done.\n{"r_hat": 1.0, "passed": true, "diagnostics": "ok"}\n'
    parsed = _parse_verifier_json(text)
    assert parsed["passed"] is True
    assert parsed["r_hat"] == 1.0
