#!/usr/bin/env python3
"""
Aggregate Hermes SkillsBench JSONL runs: pass@k at conversation turns, token stats.

**pass@k** here means macro task pass rate and micro test-case pass rate **at agent
conversation turn k within a single run** (logged in ``pass_at_turn`` on each JSONL
row). This is not averaged across k separate conversation reruns.

Usage::

  python3 benchmark/scripts/aggregate_skillsbench_runs.py \\
    benchmark/runs/batch.jsonl \\
    --pass-k 1,5,10,70 \\
    -o benchmark/runs/batch_summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from evaluate_skillsbench_task import parse_pass_k_values  # noqa: E402
from read_skillsbench_jsonl import iter_skillsbench_run_records  # noqa: E402
from skillsbench_metrics import build_envelope_metrics  # noqa: E402


def _latest_record_per_task(records: Sequence[dict]) -> Dict[str, dict]:
    by_task: Dict[str, dict] = {}
    for rec in records:
        if rec.get("schema") == "skillsbench.hermes_run_error.v1":
            continue
        tid = rec.get("skillsbench_task_id")
        if not tid:
            continue
        tid = str(tid)
        prev = by_task.get(tid)
        if prev is None or (rec.get("ts_end_iso") or "") >= (prev.get("ts_end_iso") or ""):
            by_task[tid] = rec
    return by_task


def _turn_block(record: dict, turn: int) -> Optional[dict]:
    pat = record.get("pass_at_turn")
    if isinstance(pat, dict):
        block = pat.get(str(turn))
        if isinstance(block, dict):
            return block
    return None


def _evaluation_from_turn_block(block: dict) -> dict:
    ev = block.get("evaluation")
    return ev if isinstance(ev, dict) else {}


def _mean(vals: List[float]) -> Optional[float]:
    return statistics.mean(vals) if vals else None


def _median(vals: List[float]) -> Optional[float]:
    return statistics.median(vals) if vals else None


def _filter_records(
    records: Sequence[dict],
    *,
    split_part: Optional[str] = None,
) -> List[dict]:
    if not split_part:
        return list(records)
    return [r for r in records if r.get("split_part") == split_part]


def aggregate_records(
    records: Sequence[dict],
    *,
    pass_k_values: List[int],
    split_part: Optional[str] = None,
) -> Dict[str, Any]:
    """Build summary dict from JSONL run records."""
    records = _filter_records(records, split_part=split_part)
    by_task = _latest_record_per_task(records)
    errors = sum(1 for r in records if r.get("schema") == "skillsbench.hermes_run_error.v1")

    if not by_task:
        return {
            "tasks": 0,
            "records": len(records),
            "errors": errors,
            "pass_k": {},
            "note": "No task records found.",
        }

    task_ids = sorted(by_task)
    pass_k_summary: Dict[str, Any] = {}

    for turn in pass_k_values:
        tasks_with_data = 0
        tasks_passed = 0
        tests_passed_sum = 0
        tests_total_sum = 0
        rewards: List[float] = []
        tokens_at_turn: List[float] = []
        input_tokens: List[float] = []
        api_calls_at_turn: List[float] = []
        costs_at_turn: List[float] = []

        for tid in task_ids:
            block = _turn_block(by_task[tid], turn)
            if block is None:
                metrics_pat = (by_task[tid].get("metrics") or {}).get("pass_at_turn") or {}
                block = metrics_pat.get(str(turn))
            if block is None:
                continue
            ev = _evaluation_from_turn_block(block)
            if not ev and isinstance(block, dict) and "task_success" in block:
                ev = block
            tasks_with_data += 1
            if ev.get("task_success"):
                tasks_passed += 1
            p = ev.get("tests_passed")
            t = ev.get("tests_total")
            if isinstance(p, int) and isinstance(t, int) and t > 0:
                tests_passed_sum += p
                tests_total_sum += t
            reward = ev.get("reward")
            if reward is None:
                reward = block.get("reward")
            if reward is not None:
                try:
                    rewards.append(float(reward))
                except (TypeError, ValueError):
                    pass
            tok = block.get("total_tokens")
            if tok is None:
                tok_block = block.get("tokens") if isinstance(block.get("tokens"), dict) else {}
                tok = tok_block.get("total")
            if tok is not None:
                try:
                    tokens_at_turn.append(float(tok))
                except (TypeError, ValueError):
                    pass
            inp = block.get("input_tokens")
            if inp is None and isinstance(block.get("tokens"), dict):
                inp = block["tokens"].get("input")
            if inp is not None:
                try:
                    input_tokens.append(float(inp))
                except (TypeError, ValueError):
                    pass
            calls = block.get("api_calls")
            if calls is not None:
                try:
                    api_calls_at_turn.append(float(calls))
                except (TypeError, ValueError):
                    pass
            cost = block.get("estimated_cost_usd")
            if cost is not None:
                try:
                    costs_at_turn.append(float(cost))
                except (TypeError, ValueError):
                    pass

        macro_rate = (tasks_passed / tasks_with_data) if tasks_with_data else None
        micro_rate = (
            tests_passed_sum / tests_total_sum if tests_total_sum else None
        )
        pass_k_summary[str(turn)] = {
            "macro_task_pass_rate": macro_rate,
            "tasks_with_turn_data": tasks_with_data,
            "tasks_passed_at_turn": tasks_passed,
            "micro_test_pass_rate": micro_rate,
            "tests_passed_sum": tests_passed_sum,
            "tests_total_sum": tests_total_sum,
            "reward_mean": _mean(rewards),
            "tokens_mean_at_turn": _mean(tokens_at_turn),
            "tokens_median_at_turn": _median(tokens_at_turn),
            "input_tokens_mean_at_turn": _mean(input_tokens),
            "api_calls_mean_at_turn": _mean(api_calls_at_turn),
            "estimated_cost_usd_mean_at_turn": _mean(costs_at_turn),
        }

    # Final-state metrics from envelope.evaluation (end of conversation)
    final_passed = 0
    final_with_eval = 0
    final_tests_passed = 0
    final_tests_total = 0
    final_tokens: List[float] = []
    final_input_tokens: List[float] = []
    final_api_calls: List[float] = []
    final_costs: List[float] = []
    final_durations: List[float] = []
    final_rewards: List[float] = []
    completed_count = 0
    interrupted_count = 0
    tool_rounds: List[float] = []

    for tid in task_ids:
        rec = by_task[tid]
        metrics = rec.get("metrics") or {}
        final_metrics = metrics.get("final") if isinstance(metrics, dict) else {}
        ev = rec.get("evaluation")
        if isinstance(final_metrics, dict) and "task_success" in final_metrics:
            ev = final_metrics
        if isinstance(ev, dict) and "task_success" in ev:
            final_with_eval += 1
            if ev.get("task_success"):
                final_passed += 1
            p, t = ev.get("tests_passed"), ev.get("tests_total")
            if isinstance(p, int) and isinstance(t, int) and t > 0:
                final_tests_passed += p
                final_tests_total += t
            reward = ev.get("reward")
            if reward is not None:
                try:
                    final_rewards.append(float(reward))
                except (TypeError, ValueError):
                    pass
        res = rec.get("run_conversation_result") or {}
        if isinstance(final_metrics, dict):
            tok_block = final_metrics.get("tokens") or {}
            tok = tok_block.get("total")
            inp = tok_block.get("input")
            calls = final_metrics.get("api_calls")
            cost = final_metrics.get("estimated_cost_usd")
            tr = final_metrics.get("tool_rounds")
            if final_metrics.get("completed") is True:
                completed_count += 1
            if final_metrics.get("interrupted") is True:
                interrupted_count += 1
        else:
            tok = res.get("total_tokens")
            inp = res.get("input_tokens")
            calls = res.get("api_calls")
            cost = res.get("estimated_cost_usd")
            tr = None
            if res.get("completed") is True:
                completed_count += 1
            if res.get("interrupted") is True:
                interrupted_count += 1
        if tok is not None:
            try:
                final_tokens.append(float(tok))
            except (TypeError, ValueError):
                pass
        if inp is not None:
            try:
                final_input_tokens.append(float(inp))
            except (TypeError, ValueError):
                pass
        if calls is not None:
            try:
                final_api_calls.append(float(calls))
            except (TypeError, ValueError):
                pass
        if cost is not None:
            try:
                final_costs.append(float(cost))
            except (TypeError, ValueError):
                pass
        dur = rec.get("duration_sec")
        if dur is not None:
            try:
                final_durations.append(float(dur))
            except (TypeError, ValueError):
                pass
        if tr is not None:
            try:
                tool_rounds.append(float(tr))
            except (TypeError, ValueError):
                pass

    return {
        "tasks": len(task_ids),
        "records": len(records),
        "errors": errors,
        "split_part_filter": split_part,
        "pass_k": pass_k_summary,
        "final": {
            "macro_task_pass_rate": (
                final_passed / final_with_eval if final_with_eval else None
            ),
            "tasks_with_evaluation": final_with_eval,
            "tasks_passed": final_passed,
            "micro_test_pass_rate": (
                final_tests_passed / final_tests_total if final_tests_total else None
            ),
            "tests_passed_sum": final_tests_passed,
            "tests_total_sum": final_tests_total,
            "reward_mean": _mean(final_rewards),
            "completion_rate": (completed_count / len(task_ids)) if task_ids else None,
            "interrupted_tasks": interrupted_count,
            "tokens_mean": _mean(final_tokens),
            "tokens_median": _median(final_tokens),
            "input_tokens_mean": _mean(final_input_tokens),
            "api_calls_mean": _mean(final_api_calls),
            "tool_rounds_mean": _mean(tool_rounds),
            "estimated_cost_usd_mean": _mean(final_costs),
            "estimated_cost_usd_sum": sum(final_costs) if final_costs else None,
            "duration_sec_mean": _mean(final_durations),
        },
    }


def format_summary_text(summary: Dict[str, Any]) -> str:
    lines = [
        f"tasks={summary.get('tasks')} records={summary.get('records')} errors={summary.get('errors')}",
    ]
    final = summary.get("final") or {}
    if final.get("macro_task_pass_rate") is not None:
        lines.append(
            f"final_task_pass_rate={final['macro_task_pass_rate']:.4f} "
            f"({final.get('tasks_passed')}/{final.get('tasks_with_evaluation')})"
        )
    for turn, block in (summary.get("pass_k") or {}).items():
        if not isinstance(block, dict):
            continue
        macro = block.get("macro_task_pass_rate")
        micro = block.get("micro_test_pass_rate")
        if macro is not None:
            lines.append(
                f"pass@{turn}_task={macro:.4f} "
                f"({block.get('tasks_passed_at_turn')}/{block.get('tasks_with_turn_data')})"
            )
        if micro is not None:
            lines.append(
                f"pass@{turn}_tests={micro:.4f} "
                f"({block.get('tests_passed_sum')}/{block.get('tests_total_sum')})"
            )
        if block.get("tokens_mean_at_turn") is not None:
            lines.append(f"pass@{turn}_tokens={block['tokens_mean_at_turn']:.0f}")
    if final.get("tokens_mean") is not None:
        lines.append(f"final_tokens_mean={final['tokens_mean']:.1f}")
    if final.get("api_calls_mean") is not None:
        lines.append(f"final_api_calls_mean={final['api_calls_mean']:.1f}")
    if final.get("estimated_cost_usd_sum") is not None:
        lines.append(f"final_cost_sum={final['estimated_cost_usd_sum']:.4f}")
    if final.get("completion_rate") is not None:
        lines.append(f"completion_rate={final['completion_rate']:.4f}")
    return " ".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate SkillsBench Hermes JSONL pass@turn metrics.",
    )
    parser.add_argument(
        "jsonl_files",
        nargs="+",
        help="JSONL logs from run_skillsbench_with_hermes.py (with --pass-k)",
    )
    parser.add_argument(
        "--pass-k",
        type=str,
        default="1",
        help="Comma-separated conversation turns (e.g. 1,5,10,70).",
    )
    parser.add_argument(
        "--split-part",
        type=str,
        default=None,
        choices=("train", "test", "all"),
        help="Only aggregate rows logged with this split_part (from --split-file runs).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Write JSON summary to this path.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print one-line human summary to stdout.",
    )
    args = parser.parse_args()

    try:
        pass_k_values = parse_pass_k_values(args.pass_k)
    except ValueError as e:
        parser.error(str(e))

    records: List[dict] = []
    for path_str in args.jsonl_files:
        path = Path(path_str).expanduser()
        records.extend(iter_skillsbench_run_records(path))

    summary = aggregate_records(
        records,
        pass_k_values=pass_k_values,
        split_part=args.split_part,
    )

    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.print_summary or not args.output:
        print(format_summary_text(summary))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
