#!/usr/bin/env python3
"""
Shared SkillsBench run aggregation for Hermes, CoEvoSkills, and future baselines.

Produces a common summary schema so multiple drivers can log compatible JSONL and
share one aggregator.

Headline metrics
----------------
- **macro success rate@k** — fraction of tasks with ``task_success`` at conversation
  turn *k* (and at final / max user iterations).
- **micro success rate@k** — fraction of pytest cases passed / total across tasks
  at turn *k* (and at final).
- **cost to succeed** — among tasks that eventually succeed, mean ± std of
  cumulative tokens and user iterations (API turns) at the *first* successful
  checkpoint (earliest ``pass_at_turn`` with success, else final usage).

A **user iteration** is one completed agent API turn (same as Hermes ``api_calls`` /
``pass_at_turn`` turn index).
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


def _mean(vals: Sequence[float]) -> Optional[float]:
    return statistics.mean(vals) if vals else None


def _stdev(vals: Sequence[float]) -> Optional[float]:
    if len(vals) < 2:
        return 0.0 if len(vals) == 1 else None
    return statistics.stdev(vals)


def _mean_std(vals: List[float]) -> Dict[str, Optional[float]]:
    return {
        "n": len(vals),
        "mean": _mean(vals),
        "std": _stdev(vals),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def latest_record_per_task(records: Sequence[dict]) -> Dict[str, dict]:
    by_task: Dict[str, dict] = {}
    for rec in records:
        if rec.get("schema") in (
            "skillsbench.hermes_run_error.v1",
            "skillsbench.baseline_run_error.v1",
            "terminalbench.hermes_run_error.v1",
            "terminalbench.baseline_run_error.v1",
        ):
            continue
        tid = rec.get("skillsbench_task_id") or rec.get("task_id")
        if not tid:
            continue
        tid = str(tid)
        prev = by_task.get(tid)
        if prev is None or (rec.get("ts_end_iso") or "") >= (prev.get("ts_end_iso") or ""):
            by_task[tid] = rec
    return by_task


def filter_records(
    records: Sequence[dict],
    *,
    split_part: Optional[str] = None,
    method: Optional[str] = None,
    phase: Optional[str] = None,
) -> List[dict]:
    out = list(records)
    if split_part and split_part != "all":
        out = [r for r in out if r.get("split_part") == split_part]
    if method:
        out = [r for r in out if (r.get("method") or "hermes") == method]
    if phase:
        out = [r for r in out if r.get("phase") == phase]
    return out


def _turn_block(record: dict, turn: int) -> Optional[dict]:
    pat = record.get("pass_at_turn")
    if isinstance(pat, dict):
        block = pat.get(str(turn))
        if isinstance(block, dict):
            return block
    metrics_pat = (record.get("metrics") or {}).get("pass_at_turn") or {}
    if isinstance(metrics_pat, dict):
        block = metrics_pat.get(str(turn))
        if isinstance(block, dict):
            return block
    return None


def _eval_from_block(block: dict) -> dict:
    ev = block.get("evaluation")
    if isinstance(ev, dict):
        return ev
    if "task_success" in block:
        return block
    return {}


def _tokens_from_block(block: dict) -> Optional[float]:
    tok = block.get("total_tokens")
    if tok is None:
        tokens = block.get("tokens")
        if isinstance(tokens, dict):
            tok = tokens.get("total")
    return _to_float(tok)


def _api_calls_from_block(block: dict) -> Optional[float]:
    return _to_float(block.get("api_calls"))


def _cost_from_block(block: dict) -> Optional[float]:
    return _to_float(block.get("estimated_cost_usd"))


def _final_usage(record: dict) -> Dict[str, Optional[float]]:
    metrics = record.get("metrics") or {}
    final_m = metrics.get("final") if isinstance(metrics, dict) else None
    res = record.get("run_conversation_result") or {}
    if isinstance(final_m, dict):
        tok_block = final_m.get("tokens") or {}
        return {
            "tokens": _to_float(tok_block.get("total")),
            "user_iterations": _to_float(final_m.get("api_calls")),
            "estimated_cost_usd": _to_float(final_m.get("estimated_cost_usd")),
        }
    return {
        "tokens": _to_float(res.get("total_tokens")),
        "user_iterations": _to_float(res.get("api_calls")),
        "estimated_cost_usd": _to_float(res.get("estimated_cost_usd")),
    }


def _final_eval(record: dict) -> dict:
    metrics = record.get("metrics") or {}
    final_m = metrics.get("final") if isinstance(metrics, dict) else None
    if isinstance(final_m, dict) and "task_success" in final_m:
        return final_m
    ev = record.get("evaluation")
    return ev if isinstance(ev, dict) else {}


def first_success_checkpoint(record: dict) -> Optional[Dict[str, Any]]:
    """
    Earliest successful checkpoint for cost-to-succeed.

    Prefers the smallest ``pass_at_turn`` with ``task_success``, else final eval
    when the task eventually succeeds.
    """
    pat = record.get("pass_at_turn") or {}
    if not isinstance(pat, dict):
        pat = {}
    metrics_pat = ((record.get("metrics") or {}).get("pass_at_turn") or {}) if isinstance(
        record.get("metrics"), dict
    ) else {}
    turns: List[int] = []
    for key in set(list(pat.keys()) + list(metrics_pat.keys())):
        t = _to_int(key)
        if t is not None:
            turns.append(t)
    for turn in sorted(set(turns)):
        block = _turn_block(record, turn)
        if not block:
            continue
        ev = _eval_from_block(block)
        if ev.get("task_success"):
            return {
                "source": "pass_at_turn",
                "user_iterations": float(turn),
                "tokens": _tokens_from_block(block),
                "estimated_cost_usd": _cost_from_block(block),
                "api_calls": _api_calls_from_block(block) or float(turn),
            }

    ev = _final_eval(record)
    if not ev.get("task_success"):
        return None
    usage = _final_usage(record)
    return {
        "source": "final",
        "user_iterations": usage.get("user_iterations"),
        "tokens": usage.get("tokens"),
        "estimated_cost_usd": usage.get("estimated_cost_usd"),
        "api_calls": usage.get("user_iterations"),
    }


def _pass_k_slice(
    by_task: Dict[str, dict],
    turn: int,
) -> Dict[str, Any]:
    tasks_with_data = 0
    tasks_passed = 0
    tests_passed_sum = 0
    tests_total_sum = 0
    rewards: List[float] = []
    tokens_at_turn: List[float] = []
    api_calls_at_turn: List[float] = []
    costs_at_turn: List[float] = []

    for tid in sorted(by_task):
        block = _turn_block(by_task[tid], turn)
        if block is None:
            continue
        ev = _eval_from_block(block)
        tasks_with_data += 1
        if ev.get("task_success"):
            tasks_passed += 1
        p = _to_int(ev.get("tests_passed"))
        t = _to_int(ev.get("tests_total"))
        if p is not None and t is not None and t > 0:
            tests_passed_sum += p
            tests_total_sum += t
        reward = ev.get("reward")
        if reward is None:
            reward = block.get("reward")
        rf = _to_float(reward)
        if rf is not None:
            rewards.append(rf)
        tok = _tokens_from_block(block)
        if tok is not None:
            tokens_at_turn.append(tok)
        calls = _api_calls_from_block(block)
        if calls is not None:
            api_calls_at_turn.append(calls)
        cost = _cost_from_block(block)
        if cost is not None:
            costs_at_turn.append(cost)

    return {
        "k": turn,
        "macro_success_rate": (tasks_passed / tasks_with_data) if tasks_with_data else None,
        "macro_task_pass_rate": (tasks_passed / tasks_with_data) if tasks_with_data else None,
        "tasks_with_turn_data": tasks_with_data,
        "tasks_passed_at_turn": tasks_passed,
        "micro_success_rate": (
            tests_passed_sum / tests_total_sum if tests_total_sum else None
        ),
        "micro_test_pass_rate": (
            tests_passed_sum / tests_total_sum if tests_total_sum else None
        ),
        "tests_passed_sum": tests_passed_sum,
        "tests_total_sum": tests_total_sum,
        "reward_mean": _mean(rewards),
        "tokens": _mean_std(tokens_at_turn),
        "tokens_mean_at_turn": _mean(tokens_at_turn),
        "tokens_median_at_turn": (
            statistics.median(tokens_at_turn) if tokens_at_turn else None
        ),
        "user_iterations": _mean_std(api_calls_at_turn),
        "api_calls_mean_at_turn": _mean(api_calls_at_turn),
        "estimated_cost_usd": _mean_std(costs_at_turn),
        "estimated_cost_usd_mean_at_turn": _mean(costs_at_turn),
    }


def aggregate_skillsbench_metrics(
    records: Sequence[dict],
    *,
    pass_k_values: List[int],
    split_part: Optional[str] = None,
    method: Optional[str] = None,
    phase: Optional[str] = None,
    max_user_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Aggregate compatible SkillsBench JSONL rows into a shared metrics summary.
    """
    records = filter_records(
        records, split_part=split_part, method=method, phase=phase
    )
    by_task = latest_record_per_task(records)
    errors = sum(
        1
        for r in records
        if r.get("schema")
        in (
            "skillsbench.hermes_run_error.v1",
            "skillsbench.baseline_run_error.v1",
            "terminalbench.hermes_run_error.v1",
            "terminalbench.baseline_run_error.v1",
        )
    )

    if not by_task:
        return {
            "schema": "skillsbench.metrics_summary.v1",
            "tasks": 0,
            "records": len(records),
            "errors": errors,
            "pass_k": {},
            "final": {},
            "cost_to_succeed": {},
            "note": "No task records found.",
            "filters": {
                "split_part": split_part,
                "method": method,
                "phase": phase,
                "max_user_iterations": max_user_iterations,
            },
        }

    pass_k_summary: Dict[str, Any] = {}
    for turn in pass_k_values:
        pass_k_summary[str(turn)] = _pass_k_slice(by_task, turn)

    # Final / max-iteration metrics
    final_passed = 0
    final_with_eval = 0
    final_tests_passed = 0
    final_tests_total = 0
    final_tokens: List[float] = []
    final_iters: List[float] = []
    final_costs: List[float] = []
    final_rewards: List[float] = []
    final_durations: List[float] = []
    completed_count = 0
    interrupted_count = 0

    success_tokens: List[float] = []
    success_iters: List[float] = []
    success_costs: List[float] = []
    success_durations: List[float] = []
    success_sources: Dict[str, int] = {}

    for tid in sorted(by_task):
        rec = by_task[tid]
        ev = _final_eval(rec)
        usage = _final_usage(rec)
        if "task_success" in ev:
            final_with_eval += 1
            if ev.get("task_success"):
                final_passed += 1
            p, t = _to_int(ev.get("tests_passed")), _to_int(ev.get("tests_total"))
            if p is not None and t is not None and t > 0:
                final_tests_passed += p
                final_tests_total += t
            rf = _to_float(ev.get("reward"))
            if rf is not None:
                final_rewards.append(rf)

        if usage.get("tokens") is not None:
            final_tokens.append(float(usage["tokens"]))
        if usage.get("user_iterations") is not None:
            final_iters.append(float(usage["user_iterations"]))
        if usage.get("estimated_cost_usd") is not None:
            final_costs.append(float(usage["estimated_cost_usd"]))

        dur = _to_float(rec.get("duration_sec"))
        if dur is None:
            metrics_final_pre = (rec.get("metrics") or {}).get("final") if isinstance(
                rec.get("metrics"), dict
            ) else None
            if isinstance(metrics_final_pre, dict):
                dur = _to_float(metrics_final_pre.get("duration_sec"))
        if dur is not None:
            final_durations.append(dur)

        res = rec.get("run_conversation_result") or {}
        metrics_final = (rec.get("metrics") or {}).get("final") if isinstance(
            rec.get("metrics"), dict
        ) else None
        if isinstance(metrics_final, dict):
            if metrics_final.get("completed") is True:
                completed_count += 1
            if metrics_final.get("interrupted") is True:
                interrupted_count += 1
        else:
            if isinstance(res, dict) and res.get("completed") is True:
                completed_count += 1
            if isinstance(res, dict) and res.get("interrupted") is True:
                interrupted_count += 1

        ckpt = first_success_checkpoint(rec)
        if ckpt:
            success_sources[str(ckpt.get("source"))] = (
                success_sources.get(str(ckpt.get("source")), 0) + 1
            )
            if ckpt.get("tokens") is not None:
                success_tokens.append(float(ckpt["tokens"]))
            if ckpt.get("user_iterations") is not None:
                success_iters.append(float(ckpt["user_iterations"]))
            if ckpt.get("estimated_cost_usd") is not None:
                success_costs.append(float(ckpt["estimated_cost_usd"]))
            # Wall time for the task run (envelope duration) among successes
            if dur is not None:
                success_durations.append(float(dur))

    final_block = {
        "label": "max_user_iterations" if max_user_iterations else "final",
        "max_user_iterations": max_user_iterations,
        "macro_success_rate": (
            final_passed / final_with_eval if final_with_eval else None
        ),
        "macro_task_pass_rate": (
            final_passed / final_with_eval if final_with_eval else None
        ),
        "tasks_with_evaluation": final_with_eval,
        "tasks_passed": final_passed,
        "micro_success_rate": (
            final_tests_passed / final_tests_total if final_tests_total else None
        ),
        "micro_test_pass_rate": (
            final_tests_passed / final_tests_total if final_tests_total else None
        ),
        "tests_passed_sum": final_tests_passed,
        "tests_total_sum": final_tests_total,
        "reward_mean": _mean(final_rewards),
        "completion_rate": (completed_count / len(by_task)) if by_task else None,
        "interrupted_tasks": interrupted_count,
        "tokens": _mean_std(final_tokens),
        "tokens_mean": _mean(final_tokens),
        "tokens_median": statistics.median(final_tokens) if final_tokens else None,
        "user_iterations": _mean_std(final_iters),
        "api_calls_mean": _mean(final_iters),
        "estimated_cost_usd": _mean_std(final_costs),
        "estimated_cost_usd_mean": _mean(final_costs),
        "estimated_cost_usd_sum": sum(final_costs) if final_costs else None,
        "duration_sec": _mean_std(final_durations),
        "duration_sec_mean": _mean(final_durations),
        "duration_sec_sum": sum(final_durations) if final_durations else None,
        "duration_sec_std": _stdev(final_durations),
    }

    cost_to_succeed = {
        "n_succeeded": len(success_iters) or len(success_tokens) or final_passed,
        "sources": success_sources,
        "tokens": _mean_std(success_tokens),
        "user_iterations": _mean_std(success_iters),
        "estimated_cost_usd": _mean_std(success_costs),
        "duration_sec": _mean_std(success_durations),
        "note": (
            "Mean±std among tasks that succeed, measured at first successful "
            "pass@k checkpoint when available, else at final usage. "
            "duration_sec is wall-clock for the full task run among successes."
        ),
    }

    return {
        "schema": "skillsbench.metrics_summary.v1",
        "tasks": len(by_task),
        "records": len(records),
        "errors": errors,
        "filters": {
            "split_part": split_part,
            "method": method,
            "phase": phase,
            "max_user_iterations": max_user_iterations,
        },
        "pass_k": pass_k_summary,
        "final": final_block,
        "cost_to_succeed": cost_to_succeed,
    }


def format_metrics_summary_text(summary: Dict[str, Any]) -> str:
    lines = [
        f"tasks={summary.get('tasks')} records={summary.get('records')} "
        f"errors={summary.get('errors')}",
    ]
    final = summary.get("final") or {}
    if final.get("macro_success_rate") is not None:
        lines.append(
            f"final_macro={final['macro_success_rate']:.4f} "
            f"({final.get('tasks_passed')}/{final.get('tasks_with_evaluation')})"
        )
    if final.get("micro_success_rate") is not None:
        lines.append(
            f"final_micro={final['micro_success_rate']:.4f} "
            f"({final.get('tests_passed_sum')}/{final.get('tests_total_sum')})"
        )
    for turn, block in sorted(
        (summary.get("pass_k") or {}).items(),
        key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0,
    ):
        if not isinstance(block, dict):
            continue
        macro = block.get("macro_success_rate")
        micro = block.get("micro_success_rate")
        if macro is not None:
            lines.append(
                f"pass@{turn}_macro={macro:.4f} "
                f"({block.get('tasks_passed_at_turn')}/{block.get('tasks_with_turn_data')})"
            )
        if micro is not None:
            lines.append(
                f"pass@{turn}_micro={micro:.4f} "
                f"({block.get('tests_passed_sum')}/{block.get('tests_total_sum')})"
            )
    cts = summary.get("cost_to_succeed") or {}
    tok = cts.get("tokens") or {}
    it = cts.get("user_iterations") or {}
    cts_dur = cts.get("duration_sec") or {}
    if tok.get("mean") is not None:
        lines.append(
            f"cost_to_succeed_tokens={tok['mean']:.1f}±{tok.get('std') or 0:.1f} "
            f"(n={tok.get('n')})"
        )
    if it.get("mean") is not None:
        lines.append(
            f"cost_to_succeed_iters={it['mean']:.2f}±{it.get('std') or 0:.2f} "
            f"(n={it.get('n')})"
        )
    if cts_dur.get("mean") is not None:
        lines.append(
            f"cost_to_succeed_sec={cts_dur['mean']:.1f}±{cts_dur.get('std') or 0:.1f} "
            f"(n={cts_dur.get('n')})"
        )
    ui = final.get("user_iterations") or {}
    if ui.get("mean") is not None:
        lines.append(
            f"final_iters={ui['mean']:.2f}±{ui.get('std') or 0:.2f}"
        )
    ft = final.get("tokens") or {}
    if ft.get("mean") is not None:
        lines.append(
            f"final_tokens={ft['mean']:.1f}±{ft.get('std') or 0:.1f}"
        )
    fd = final.get("duration_sec") or {}
    if fd.get("mean") is not None:
        lines.append(
            f"duration_sec={fd['mean']:.1f}±{fd.get('std') or 0:.1f} "
            f"sum={final.get('duration_sec_sum') or 0:.1f}"
        )
    return " ".join(lines)
