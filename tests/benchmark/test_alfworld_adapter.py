"""Unit tests for the ALFWorld Hermes adapter (no TextWorld / LLM required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.alfworld_adapter.actions import parse_alfworld_action
from benchmark.alfworld_adapter.env import (
    discover_games,
    format_turn_prompt,
    lookup_game,
    resolve_alfworld_data,
)


def _write_game(root: Path, rel: str, *, solvable: bool = True) -> Path:
    game_dir = root / "json_2.1.1" / "valid_unseen" / rel
    game_dir.mkdir(parents=True)
    payload = {
        "pddl_domain": "(define (domain dummy))",
        "grammar": "",
        "pddl_problem": "(define (problem dummy))",
        "solvable": solvable,
    }
    path = game_dir / "game.tw-pddl"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_action_exact_and_fenced():
    admissible = ["go to fridge 1", "open fridge 1", "look"]
    assert parse_alfworld_action("go to fridge 1", admissible) == "go to fridge 1"
    assert (
        parse_alfworld_action("Action: Open Fridge 1\n", admissible) == "open fridge 1"
    )
    assert parse_alfworld_action("```\ngo to fridge 1\n```", admissible) == "go to fridge 1"


def test_parse_action_fallback_look_and_raw():
    assert parse_alfworld_action("   ", ["look"]) == "look"
    assert parse_alfworld_action("dance wildly", ["look", "inventory"]) == "dance wildly"


def test_discover_skips_unsolvable_movable_and_sliced(tmp_path):
    data = tmp_path / "alfdata"
    keep = "pick_and_place_simple-Mug-None-Desk-1/trial_A"
    _write_game(data, keep, solvable=True)
    _write_game(data, "pick_and_place_simple-Mug-None-Desk-2/trial_B", solvable=False)
    _write_game(
        data,
        "pick_and_place_with_movable_recep-Knife-Bowl-1/trial_C",
        solvable=True,
    )
    _write_game(
        data,
        "pick_heat_then_place_in_recep-TomatoSliced-None-Fridge-1/trial_D",
        solvable=True,
    )
    games = discover_games(data, "valid_unseen")
    assert [g["task_id"] for g in games] == [keep]
    assert games[0]["task_type"] == "pick_and_place_simple"
    found = lookup_game(games, keep)
    assert found["task_id"] == keep


def test_format_turn_prompt_lists_commands():
    text = format_turn_prompt("You see a fridge.", ["go to fridge 1"], first_turn=True)
    assert "ALFWorld" in text
    assert "- go to fridge 1" in text
    later = format_turn_prompt("Nothing happens.", ["look"], first_turn=False)
    assert "ALFWorld" not in later
    assert "- look" in later


def test_resolve_alfworld_data_explicit(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        resolve_alfworld_data(missing)
    data = tmp_path / "data"
    data.mkdir()
    assert resolve_alfworld_data(data) == data.resolve()
