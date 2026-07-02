"""Tests for benchmark/scripts/analyze_hot_pool_runs.py."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "benchmark" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import analyze_hot_pool_runs as ahp  # noqa: E402


def test_enrich_task_with_telemetry_and_messages():
    record = {
        "schema": "skillsbench.hermes_run.v1",
        "skillsbench_task_id": "demo-task",
        "duration_sec": 12.5,
        "hot_pool_telemetry": {
            "carryover": {
                "pool_skills_at_start": 1,
                "pool_points_at_start": 2,
                "pool_skills_at_end": 2,
                "new_records_this_task": 1,
            },
            "inject": {
                "build_block_applied": True,
                "injections_nonempty": 5,
                "point_count": 2,
                "points_injected": ["Use scripts/run_tests.sh", "Never bare pytest"],
                "skills_injected": ["pytest-skill"],
            },
            "skill_view": {"total": 1, "unique": 1, "repeat": 0, "while_in_pool": 0},
        },
        "run_conversation_result": {
            "completed": True,
            "api_calls": 8,
            "total_tokens": 1000,
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "terminal",
                                "arguments": '{"command": "scripts/run_tests.sh"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": '{"success": true}'},
            ],
        },
    }
    row = ahp.enrich_task(record)
    assert row is not None
    assert row["hot_active"] is True
    assert row["procedure_proxies"]["alignment_checks"] == 1
    assert row["procedure_proxies"]["alignment_hits"] == 1


def test_summarize_tasks():
    tasks = [
        {
            "task_id": "a",
            "completed": True,
            "api_calls": 10,
            "total_tokens": 100,
            "hot_active": True,
            "inject": {"injections_nonempty": 10},
            "carryover": {"pool_skills_at_start": 1},
            "skill_view_replay": {"total": 1},
            "procedure_proxies": {
                "alignment_rate": 0.5,
                "tool_errors_total": 0,
                "guardrail_violations": 0,
            },
        }
    ]
    s = ahp.summarize_tasks(tasks, label="test")
    assert s["n_tasks"] == 1
    assert s["mean_api_calls"] == 10.0
    assert s["n_hot_active"] == 1
