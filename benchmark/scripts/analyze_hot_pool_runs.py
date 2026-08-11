#!/usr/bin/env python3
"""
Analyze Hermes SkillsBench JSONL logs with hot_pool_telemetry (Option 3 hybrid).

Agent-side counters live in ``run_conversation_result.hot_pool_telemetry`` (or the
envelope copy). This script adds post-hoc procedure proxies from ``messages[]``:
alignment hits, guardrail rubric, tool errors, recovery cost.

Usage::

  # Single run summary (hot-pool proxies + shared macro/micro/cost metrics)
  python3 benchmark/scripts/analyze_hot_pool_runs.py runs_treatment.jsonl \\
      --pass-k 1,5,10,70 --max-user-iterations 90 \\
      --output benchmark/hot_pool_summary.json

  # Control vs treatment comparison
  python3 benchmark/scripts/analyze_hot_pool_runs.py \\
      runs_control.jsonl runs_treatment.jsonl \\
      --label-a control --label-b treatment \\
      --output benchmark/hot_pool_compare.json

Shared success/cost metrics also live in ``aggregate_skillsbench_runs.py`` (same
``skillsbench_aggregate_core`` schema used by CoEvoSkills and future baselines).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluate_skillsbench_task import parse_pass_k_values  # noqa: E402
from read_skillsbench_jsonl import iter_skillsbench_run_records  # noqa: E402
from skillsbench_aggregate_core import (  # noqa: E402
    aggregate_skillsbench_metrics,
    format_metrics_summary_text,
)

from agent.hot_skills import (  # noqa: E402
    check_guardrail_violations,
    compute_alignment_hits,
    replay_message_tool_stats,
)


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _mean(vals: List[float]) -> Optional[float]:
    return statistics.mean(vals) if vals else None


def _get_telemetry(record: dict) -> dict:
    tel = record.get("hot_pool_telemetry")
    if isinstance(tel, dict):
        return tel
    res = record.get("run_conversation_result")
    if isinstance(res, dict) and isinstance(res.get("hot_pool_telemetry"), dict):
        return res["hot_pool_telemetry"]
    return {}


def _get_messages(record: dict) -> List[dict]:
    res = record.get("run_conversation_result")
    if isinstance(res, dict) and isinstance(res.get("messages"), list):
        return res["messages"]
    return []


def enrich_task(record: dict) -> Optional[dict]:
    tid = record.get("skillsbench_task_id")
    if not tid or not isinstance(tid, str):
        return None
    if record.get("schema") == "skillsbench.hermes_run_error.v1":
        return {
            "task_id": tid,
            "error": str(record.get("error") or "")[:500],
        }

    res = record.get("run_conversation_result")
    if not isinstance(res, dict):
        return {"task_id": tid, "error": "missing run_conversation_result"}

    tel = _get_telemetry(record)
    inject = tel.get("inject") if isinstance(tel.get("inject"), dict) else {}
    carry = tel.get("carryover") if isinstance(tel.get("carryover"), dict) else {}
    skill_tel = tel.get("skill_view") if isinstance(tel.get("skill_view"), dict) else {}

    messages = _get_messages(record)
    replay = replay_message_tool_stats(messages)
    injected_points = inject.get("points_injected") or []
    if not isinstance(injected_points, list):
        injected_points = []
    align_hits, align_checks = compute_alignment_hits(
        replay.get("substantive_tool_calls") or [],
        injected_points,
    )
    guard = check_guardrail_violations(messages)

    api_calls = _to_int(res.get("api_calls"))
    first_err = replay.get("first_tool_error_at_iter")
    api_after_err = None
    if api_calls is not None and first_err is not None:
        api_after_err = max(0, api_calls - int(first_err))

    first_inject = inject.get("first_nonempty_inject_iter")
    errors_after_inject = None
    if first_err is not None and first_inject is not None:
        if int(first_err) >= int(first_inject):
            errors_after_inject = replay.get("tool_errors_total")

    pool_start = _to_int(carry.get("pool_skills_at_start")) or 0
    inject_nonempty = _to_int(inject.get("injections_nonempty")) or 0

    ev = record.get("evaluation") if isinstance(record.get("evaluation"), dict) else {}
    metrics_final = ((record.get("metrics") or {}).get("final") or {}) if isinstance(
        record.get("metrics"), dict
    ) else {}
    task_success = ev.get("task_success")
    if task_success is None:
        task_success = metrics_final.get("task_success")

    row = {
        "task_id": tid,
        "completed": res.get("completed"),
        "task_success": task_success,
        "api_calls": api_calls,
        "total_tokens": _to_int(res.get("total_tokens")),
        "estimated_cost_usd": _to_float(res.get("estimated_cost_usd")),
        "duration_sec": _to_float(record.get("duration_sec")),
        "hot_active": pool_start > 0 and inject_nonempty > 0,
        "carryover": {
            "pool_skills_at_start": pool_start,
            "pool_points_at_start": _to_int(carry.get("pool_points_at_start")) or 0,
            "pool_skills_at_end": _to_int(carry.get("pool_skills_at_end")) or 0,
            "new_records_this_task": _to_int(carry.get("new_records_this_task")) or 0,
            "global_turn_at_start": _to_int(carry.get("global_turn_at_start")),
        },
        "inject": {
            "build_block_applied": bool(inject.get("build_block_applied")),
            "injections_nonempty": inject_nonempty,
            "point_count": _to_int(inject.get("point_count")) or 0,
            "skills_injected": inject.get("skills_injected") or [],
        },
        "skill_view_agent": skill_tel,
        "skill_view_replay": {
            "total": replay.get("skill_view_total"),
            "unique": replay.get("skill_view_unique"),
            "repeat": replay.get("skill_view_repeat"),
        },
        "procedure_proxies": {
            "alignment_checks": align_checks,
            "alignment_hits": align_hits,
            "alignment_rate": (align_hits / align_checks) if align_checks else None,
            "tool_errors_total": replay.get("tool_errors_total"),
            "first_tool_error_at_iter": first_err,
            "api_calls_after_first_error": api_after_err,
            "errors_after_first_inject": errors_after_inject,
            **guard,
        },
    }
    return row


def summarize_tasks(tasks: List[dict], label: str = "run") -> dict:
    ok = [t for t in tasks if t and not t.get("error") and "completed" in t]
    hot_active = [t for t in ok if t.get("hot_active")]
    completed = [t for t in ok if t.get("completed") is True]

    def col(key_path: Tuple[str, ...], subset: List[dict]) -> List[float]:
        out: List[float] = []
        for t in subset:
            cur: Any = t
            for k in key_path:
                if not isinstance(cur, dict):
                    cur = None
                    break
                cur = cur.get(k)
            v = _to_float(cur)
            if v is not None:
                out.append(v)
        return out

    succeeded = [t for t in ok if t.get("task_success") is True]
    success_tokens = col(("total_tokens",), succeeded)
    success_iters = col(("api_calls",), succeeded)
    success_costs = col(("estimated_cost_usd",), succeeded)
    success_durs = col(("duration_sec",), succeeded)
    all_durs = col(("duration_sec",), ok)

    def _mean_std(vals: List[float]) -> Dict[str, Optional[float]]:
        if not vals:
            return {"n": 0, "mean": None, "std": None}
        std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        return {"n": len(vals), "mean": statistics.mean(vals), "std": std}

    summary = {
        "label": label,
        "n_tasks": len(tasks),
        "n_analyzed": len(ok),
        "n_completed": len(completed),
        "completion_rate": (len(completed) / len(ok)) if ok else None,
        "n_task_success": len(succeeded),
        "macro_success_rate": (len(succeeded) / len(ok)) if ok else None,
        "n_hot_active": len(hot_active),
        "mean_api_calls": _mean(col(("api_calls",), ok)),
        "mean_total_tokens": _mean(col(("total_tokens",), ok)),
        "mean_cost_usd": _mean(col(("estimated_cost_usd",), ok)),
        "duration_sec": _mean_std(all_durs),
        "duration_sec_sum": sum(all_durs) if all_durs else None,
        "cost_to_succeed": {
            "tokens": _mean_std(success_tokens),
            "user_iterations": _mean_std(success_iters),
            "estimated_cost_usd": _mean_std(success_costs),
            "duration_sec": _mean_std(success_durs),
            "note": "Among tasks with task_success at final evaluation (hot-pool view).",
        },
        "mean_injections_nonempty": _mean(col(("inject", "injections_nonempty"), ok)),
        "mean_pool_skills_at_start": _mean(
            col(("carryover", "pool_skills_at_start"), ok)
        ),
        "mean_skill_view_total_replay": _mean(
            col(("skill_view_replay", "total"), ok)
        ),
        "mean_alignment_rate": _mean(
            col(("procedure_proxies", "alignment_rate"), hot_active)
        ),
        "mean_tool_errors": _mean(
            col(("procedure_proxies", "tool_errors_total"), ok)
        ),
        "mean_guardrail_violations": _mean(
            col(("procedure_proxies", "guardrail_violations"), ok)
        ),
    }
    return summary


def compare_summaries(a: dict, b: dict, tasks_a: List[dict], tasks_b: List[dict]) -> dict:
    by_a = {t["task_id"]: t for t in tasks_a if t.get("task_id")}
    by_b = {t["task_id"]: t for t in tasks_b if t.get("task_id")}
    common = sorted(set(by_a) & set(by_b))
    paired = [
        tid
        for tid in common
        if by_a[tid].get("completed") and by_b[tid].get("completed")
    ]

    def paired_delta(field: Tuple[str, ...]) -> Optional[float]:
        deltas: List[float] = []
        for tid in paired:
            va = by_b[tid]
            vb = by_a[tid]
            for row in (va, vb):
                if not isinstance(row, dict):
                    continue
            cur_b: Any = va
            cur_a: Any = vb
            for k in field:
                cur_b = cur_b.get(k) if isinstance(cur_b, dict) else None
                cur_a = cur_a.get(k) if isinstance(cur_a, dict) else None
            fb = _to_float(cur_b)
            fa = _to_float(cur_a)
            if fb is not None and fa is not None:
                deltas.append(fb - fa)
        return _mean(deltas)

    return {
        "n_common_tasks": len(common),
        "n_paired_completed": len(paired),
        "mean_delta_api_calls_b_minus_a": paired_delta(("api_calls",)),
        "mean_delta_tokens_b_minus_a": paired_delta(("total_tokens",)),
        "mean_delta_injections_nonempty_b_minus_a": paired_delta(
            ("inject", "injections_nonempty")
        ),
        "mean_delta_skill_view_replay_b_minus_a": paired_delta(
            ("skill_view_replay", "total")
        ),
    }


def load_tasks(path: Path) -> List[dict]:
    rows: List[dict] = []
    for rec in iter_skillsbench_run_records(path):
        row = enrich_task(rec)
        if row:
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze hot pool SkillsBench JSONL logs.")
    parser.add_argument(
        "jsonl",
        nargs="+",
        help="One JSONL (summary) or two JSONL files (control treatment compare).",
    )
    parser.add_argument("--label-a", default="a", help="Label for first log (compare mode).")
    parser.add_argument("--label-b", default="b", help="Label for second log (compare mode).")
    parser.add_argument(
        "--pass-k",
        type=str,
        default="1,5,10,70",
        help="Turns for shared macro/micro success@k (skillsbench_aggregate_core).",
    )
    parser.add_argument(
        "--max-user-iterations",
        type=int,
        default=None,
        help="Annotate final / max-iteration metrics in core summary.",
    )
    parser.add_argument(
        "--split-part",
        type=str,
        default=None,
        choices=("train", "test", "all"),
        help="Filter rows by split_part when aggregating core metrics.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write summary JSON to this path.",
    )
    parser.add_argument(
        "--print-core-summary",
        action="store_true",
        help="Print one-line core metrics summary to stdout.",
    )
    args = parser.parse_args()

    try:
        pass_k_values = parse_pass_k_values(args.pass_k)
    except ValueError as e:
        parser.error(str(e))

    paths = [Path(p).expanduser() for p in args.jsonl]
    for p in paths:
        if not p.is_file():
            print(f"Not a file: {p}", file=sys.stderr)
            return 1

    def _core(path: Path) -> dict:
        records = list(iter_skillsbench_run_records(path))
        return aggregate_skillsbench_metrics(
            records,
            pass_k_values=pass_k_values,
            split_part=args.split_part,
            max_user_iterations=args.max_user_iterations,
        )

    if len(paths) == 1:
        tasks = load_tasks(paths[0])
        core = _core(paths[0])
        report = {
            "source": str(paths[0]),
            "tasks": tasks,
            "summary": summarize_tasks(tasks, label=paths[0].name),
            "core_metrics": core,
        }
        if args.print_core_summary:
            print(format_metrics_summary_text(core))
    elif len(paths) == 2:
        tasks_a = load_tasks(paths[0])
        tasks_b = load_tasks(paths[1])
        core_a = _core(paths[0])
        core_b = _core(paths[1])
        report = {
            "path_a": str(paths[0]),
            "path_b": str(paths[1]),
            "summary_a": summarize_tasks(tasks_a, label=args.label_a),
            "summary_b": summarize_tasks(tasks_b, label=args.label_b),
            "core_metrics_a": core_a,
            "core_metrics_b": core_b,
            "paired": compare_summaries(
                summarize_tasks(tasks_a, args.label_a),
                summarize_tasks(tasks_b, args.label_b),
                tasks_a,
                tasks_b,
            ),
            "tasks_a": tasks_a,
            "tasks_b": tasks_b,
        }
        if args.print_core_summary:
            print(f"[{args.label_a}] {format_metrics_summary_text(core_a)}")
            print(f"[{args.label_b}] {format_metrics_summary_text(core_b)}")
    else:
        parser.error("Provide one or two JSONL files.")
        return 1

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
