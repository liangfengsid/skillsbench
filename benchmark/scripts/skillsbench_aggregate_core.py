#!/usr/bin/env python3
"""
Shared SkillsBench run aggregation for Hermes, CoEvoSkills, and future baselines.

Produces a common summary schema so multiple drivers can log compatible JSONL and
share one aggregator.

Headline metrics
----------------
- **macro success rate@k** — fraction of **all** tasks that succeeded at *any*
  ``pass_at_turn`` checkpoint with turn ≤ *k*, **or** (by default) at final
  evaluation when ``user_iterations ≤ k``. With
  ``include_final_in_pass_k=False`` / ``--pass-k-checkpoints-only``, only
  ``pass_at_turn`` keys count — post-conversation host eval and verification
  retries do not. Tasks with no in-budget observation (still running past
  *k*, or missing checkpoints) count as **not** succeeded — denominator is
  the suite size, not only early finishers. This is cumulative within a
  single trajectory budget, so rates are monotonic in *k*.
  Snapshot-at-exact-*k* is wrong: tasks that finish early omit later keys,
  workspace state can regress after an earlier pass, and success can land
  between sparse checkpoint turns (e.g. final pass at 15 API calls with keys
  only at 1/5/10).
- **micro success rate@k** — fraction of pytest cases passed within budget *k*
  out of **all** known test cases in the suite. For each task, if there is an
  in-budget observation, use that terminal score (final when
  ``user_iterations ≤ k``, else latest ``pass_at_turn`` ≤ *k*). If there is
  no in-budget observation, contribute ``0 / final_tests_total`` (still
  running past *k*, or missing checkpoints). Macro stays cumulative over the
  full task suite; micro does not take the best mid-run pytest score.
- **cost to succeed** — among tasks that succeed **within the turn budget**,
  mean ± std of cumulative tokens and user iterations (API turns) at the
  *first* successful checkpoint (earliest ``pass_at_turn`` with success,
  else in-budget final usage). The budget is ``max_user_iterations`` when
  set, else ``max(pass_k_values)`` under ``include_final_in_pass_k=False``.
  Tasks that only succeed after a verification retry or past the budget
  are excluded.
- **AUC of pass@k** — for continuous integer *k* from 1 to ``auc_max_k``
  (default 60), compute macro and micro success rates at each *k*, then
  report the normalized area under each curve as the mean of those rates:
  ``AUC = (1/K) Σ_{k=1}^{K} rate(k)`` ∈ [0, 1]. Perfect always-pass@1 → 1.0.
- **final** — end-of-run host eval (after any verification retries). Reported
  after sparse pass@k / AUC in the text summary, with mean±std of the API
  turn at which final success was recorded (verification retries included).

Token / iteration stats reported under ``pass_k[k]`` use that same **terminal**
in-budget observation.

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


def _turn_blocks_upto(record: dict, turn: int) -> List[Tuple[int, dict]]:
    """Return ``(t, block)`` for every recorded checkpoint with ``t <= turn``."""
    pat = record.get("pass_at_turn")
    if not isinstance(pat, dict):
        pat = {}
    metrics_pat = ((record.get("metrics") or {}).get("pass_at_turn") or {}) if isinstance(
        record.get("metrics"), dict
    ) else {}
    out: List[Tuple[int, dict]] = []
    for key in set(list(pat.keys()) + list(metrics_pat.keys())):
        t = _to_int(key)
        if t is None or t > turn:
            continue
        block = _turn_block(record, t)
        if block is not None:
            out.append((t, block))
    out.sort(key=lambda item: item[0])
    return out


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


def _suite_tests_total(record: dict) -> Optional[int]:
    """Canonical pytest case count for a task (final, else max checkpoint)."""
    tot = _to_int(_final_eval(record).get("tests_total"))
    if tot is not None and tot > 0:
        return tot
    best = 0
    pat = record.get("pass_at_turn") or {}
    if not isinstance(pat, dict):
        pat = {}
    metrics_pat = (
        ((record.get("metrics") or {}).get("pass_at_turn") or {})
        if isinstance(record.get("metrics"), dict)
        else {}
    )
    for key in set(list(pat.keys()) + list(metrics_pat.keys())):
        turn = _to_int(key)
        if turn is None:
            continue
        block = _turn_block(record, turn)
        if not block:
            continue
        n = _to_int(_eval_from_block(block).get("tests_total"))
        if n is not None and n > best:
            best = n
    return best if best > 0 else None


def first_success_checkpoint(
    record: dict,
    *,
    max_turn: Optional[int] = None,
    include_final: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Earliest successful checkpoint for cost-to-succeed.

    Prefers the smallest ``pass_at_turn`` with ``task_success`` and
    ``turn ≤ max_turn`` (when ``max_turn`` is set). Falls back to final eval
    only when ``include_final`` and that success is also in budget
    (``user_iterations ≤ max_turn``).
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
        if max_turn is not None and turn > max_turn:
            continue
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

    if not include_final:
        return None
    ev = _final_eval(record)
    if not ev.get("task_success"):
        return None
    usage = _final_usage(record)
    ui = usage.get("user_iterations")
    if max_turn is not None and (ui is None or ui > max_turn):
        return None
    return {
        "source": "final",
        "user_iterations": ui,
        "tokens": usage.get("tokens"),
        "estimated_cost_usd": usage.get("estimated_cost_usd"),
        "api_calls": ui,
    }


def _cost_to_succeed_max_turn(
    *,
    max_user_iterations: Optional[int],
    pass_k_values: List[int],
    include_final_in_pass_k: bool,
) -> Optional[int]:
    """Turn budget for cost-to-succeed (None = no cap)."""
    if max_user_iterations is not None:
        return int(max_user_iterations)
    if not include_final_in_pass_k and pass_k_values:
        return max(pass_k_values)
    return None


def _observations_upto(
    record: dict,
    turn: int,
    *,
    include_final: bool = True,
) -> List[Tuple[int, dict, str]]:
    """Checkpoints and (optionally) in-budget final eval as ``(t, block, source)``.

    Final evaluation is included when ``include_final`` and
    ``user_iterations ≤ turn``. That covers tasks that succeed between sparse
    ``pass_at_turn`` keys (e.g. pass only at final after 15 API calls while
    checkpoints exist only at 1/5/10). Set ``include_final=False`` to score
    only mid-run ``pass_at_turn`` snapshots.
    """
    out: List[Tuple[int, dict, str]] = []
    for t, block in _turn_blocks_upto(record, turn):
        out.append((t, block, "pass_at_turn"))

    if include_final:
        final_ev = _final_eval(record)
        usage = _final_usage(record)
        ui = usage.get("user_iterations")
        if final_ev and ui is not None and ui <= turn:
            synthetic = {
                "evaluation": final_ev,
                "total_tokens": usage.get("tokens"),
                "api_calls": ui,
                "estimated_cost_usd": usage.get("estimated_cost_usd"),
                "reward": final_ev.get("reward"),
            }
            out.append((int(ui), synthetic, "final"))

    # Prefer final after a same-turn checkpoint so "latest" reflects end state.
    out.sort(key=lambda item: (item[0], 0 if item[2] == "pass_at_turn" else 1))
    return out


def _terminal_observation(
    obs: List[Tuple[int, dict, str]],
) -> Tuple[int, dict, str]:
    """Prefer in-budget final eval over a later sparse checkpoint.

    Mid-run ``pass_at_turn`` keys can outrank final ``user_iterations`` (e.g.
    a turn-60 snapshot after the agent stopped at 56) and may count pytest
    cases differently than final. Micro / usage should follow the true
    end-of-run eval whenever it is within budget.
    """
    for item in reversed(obs):
        if item[2] == "final":
            return item
    return obs[-1]


def _pass_k_slice(
    by_task: Dict[str, dict],
    turn: int,
    *,
    include_final: bool = True,
) -> Dict[str, Any]:
    """Aggregate pass@k as cumulative success within a turn budget.

    Macro denominator is **all tasks** in ``by_task``. A task counts as
    success if any ``pass_at_turn`` checkpoint ≤ ``turn`` has
    ``task_success``, or (when ``include_final``) final evaluation succeeds
    with ``user_iterations ≤ turn``. Tasks with no in-budget observation
    count as not succeeded (still running past the budget, or missing
    checkpoints).

    Micro denominator is **all known suite test cases**. In-budget tasks
    contribute their terminal pytest score; tasks with no in-budget
    observation contribute ``0 / suite_tests_total`` from final (or max
    checkpoint) totals. Usage stats still only average over in-budget
    observations.
    """
    tasks_total = len(by_task)
    tasks_with_data = 0
    tasks_passed = 0
    tests_passed_sum = 0
    tests_total_sum = 0
    rewards: List[float] = []
    tokens_at_turn: List[float] = []
    api_calls_at_turn: List[float] = []
    costs_at_turn: List[float] = []

    for tid in sorted(by_task):
        rec = by_task[tid]
        obs = _observations_upto(rec, turn, include_final=include_final)
        if not obs:
            suite_n = _suite_tests_total(rec)
            if suite_n is not None:
                tests_total_sum += suite_n
            continue

        tasks_with_data += 1

        any_success = False
        for _t, block, _src in obs:
            if _eval_from_block(block).get("task_success"):
                any_success = True
                break
        if any_success:
            tasks_passed += 1

        _t, terminal, _src = _terminal_observation(obs)
        ev = _eval_from_block(terminal)
        p = _to_int(ev.get("tests_passed"))
        tot = _to_int(ev.get("tests_total"))
        if p is not None and tot is not None and tot > 0:
            tests_passed_sum += p
            tests_total_sum += tot
        else:
            suite_n = _suite_tests_total(rec)
            if suite_n is not None:
                tests_total_sum += suite_n
        reward = ev.get("reward")
        if reward is None:
            reward = terminal.get("reward")
        rf = _to_float(reward)
        if rf is not None:
            rewards.append(rf)

        tok = _tokens_from_block(terminal)
        if tok is not None:
            tokens_at_turn.append(tok)
        calls = _api_calls_from_block(terminal)
        if calls is not None:
            api_calls_at_turn.append(calls)
        cost = _cost_from_block(terminal)
        if cost is not None:
            costs_at_turn.append(cost)

    return {
        "k": turn,
        "cumulative": True,
        "macro_success_rate": (tasks_passed / tasks_total) if tasks_total else None,
        "macro_task_pass_rate": (tasks_passed / tasks_total) if tasks_total else None,
        "tasks_total": tasks_total,
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
        "include_final": include_final,
    }


def _final_success_turn(record: dict) -> Optional[float]:
    """
    API turns when end-of-run host eval first succeeded.

    Prefers ``api_calls`` on the successful ``host_verification_attempts``
    entry (verification retries included in the turn count). Falls back to
    final usage ``user_iterations`` / ``api_calls``.
    """
    ev = _final_eval(record)
    if not ev.get("task_success"):
        return None
    attempts = record.get("host_verification_attempts")
    if isinstance(attempts, list):
        for attempt in reversed(attempts):
            if not isinstance(attempt, dict):
                continue
            aev = attempt.get("evaluation") or {}
            if not isinstance(aev, dict) or not aev.get("task_success"):
                continue
            ac = _to_float(attempt.get("api_calls"))
            if ac is not None:
                return ac
    usage = _final_usage(record)
    return usage.get("user_iterations")


def _auc_pass_k(
    by_task: Dict[str, dict],
    *,
    k_max: int,
    include_final: bool,
) -> Dict[str, Any]:
    """
    Normalized AUC of macro/micro pass@k for continuous k=1..k_max.

    AUC = (1/K) Σ rate(k) — mean of unit-width rectangular areas, so a curve
    that is always 1.0 yields AUC 1.0.
    """
    if k_max < 1 or not by_task:
        return {}
    macro_curve: List[Optional[float]] = []
    micro_curve: List[Optional[float]] = []
    for k in range(1, k_max + 1):
        slice_k = _pass_k_slice(by_task, k, include_final=include_final)
        macro_curve.append(slice_k.get("macro_success_rate"))
        micro_curve.append(slice_k.get("micro_success_rate"))
    macro_vals = [float(v) for v in macro_curve if v is not None]
    micro_vals = [float(v) for v in micro_curve if v is not None]
    return {
        "k_min": 1,
        "k_max": k_max,
        "method": "mean_of_rates",
        "note": (
            "Normalized AUC over continuous integer k: "
            f"AUC = (1/{k_max}) * sum(rate(k) for k in 1..{k_max})."
        ),
        "include_final": include_final,
        "macro_success_rate": _mean(macro_vals),
        "micro_success_rate": _mean(micro_vals),
        "macro_curve": macro_curve,
        "micro_curve": micro_curve,
    }


def aggregate_skillsbench_metrics(
    records: Sequence[dict],
    *,
    pass_k_values: List[int],
    split_part: Optional[str] = None,
    method: Optional[str] = None,
    phase: Optional[str] = None,
    max_user_iterations: Optional[int] = None,
    include_final_in_pass_k: bool = True,
    auc_max_k: int = 60,
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

    empty_filters = {
        "split_part": split_part,
        "method": method,
        "phase": phase,
        "max_user_iterations": max_user_iterations,
        "include_final_in_pass_k": include_final_in_pass_k,
        "auc_max_k": auc_max_k,
    }

    if not by_task:
        return {
            "schema": "skillsbench.metrics_summary.v1",
            "tasks": 0,
            "records": len(records),
            "errors": errors,
            "pass_k": {},
            "auc": {},
            "final": {},
            "cost_to_succeed": {},
            "note": "No task records found.",
            "filters": empty_filters,
        }

    pass_k_summary: Dict[str, Any] = {}
    for turn in pass_k_values:
        pass_k_summary[str(turn)] = _pass_k_slice(
            by_task, turn, include_final=include_final_in_pass_k
        )

    auc_block = (
        _auc_pass_k(
            by_task,
            k_max=int(auc_max_k),
            include_final=include_final_in_pass_k,
        )
        if auc_max_k and int(auc_max_k) > 0
        else {}
    )

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
    final_success_turns: List[float] = []
    completed_count = 0
    interrupted_count = 0

    success_tokens: List[float] = []
    success_iters: List[float] = []
    success_costs: List[float] = []
    success_durations: List[float] = []
    success_sources: Dict[str, int] = {}
    cts_max_turn = _cost_to_succeed_max_turn(
        max_user_iterations=max_user_iterations,
        pass_k_values=pass_k_values,
        include_final_in_pass_k=include_final_in_pass_k,
    )

    for tid in sorted(by_task):
        rec = by_task[tid]
        ev = _final_eval(rec)
        usage = _final_usage(rec)
        if "task_success" in ev:
            final_with_eval += 1
            if ev.get("task_success"):
                final_passed += 1
                st = _final_success_turn(rec)
                if st is not None:
                    final_success_turns.append(float(st))
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

        ckpt = first_success_checkpoint(
            rec,
            max_turn=cts_max_turn,
            include_final=include_final_in_pass_k,
        )
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
        "success_user_iterations": _mean_std(final_success_turns),
        "success_user_iterations_note": (
            "Among finally-passed tasks: API turns when host eval succeeded "
            "(includes host verification retries when logged)."
        ),
        "estimated_cost_usd": _mean_std(final_costs),
        "estimated_cost_usd_mean": _mean(final_costs),
        "estimated_cost_usd_sum": sum(final_costs) if final_costs else None,
        "duration_sec": _mean_std(final_durations),
        "duration_sec_mean": _mean(final_durations),
        "duration_sec_sum": sum(final_durations) if final_durations else None,
        "duration_sec_std": _stdev(final_durations),
    }

    cost_to_succeed = {
        "n_succeeded": sum(success_sources.values()),
        "max_user_iterations": cts_max_turn,
        "sources": success_sources,
        "tokens": _mean_std(success_tokens),
        "user_iterations": _mean_std(success_iters),
        "estimated_cost_usd": _mean_std(success_costs),
        "duration_sec": _mean_std(success_durations),
        "note": (
            "Mean±std among tasks that succeed within the turn budget "
            f"({cts_max_turn if cts_max_turn is not None else 'uncapped'}), "
            "measured at first successful pass@k checkpoint when available"
            + (
                ", else at in-budget final usage. "
                if include_final_in_pass_k
                else " (checkpoints only; post-conversation / retry finals excluded). "
            )
            + "duration_sec is wall-clock for the full task run among those successes."
        ),
    }

    return {
        "schema": "skillsbench.metrics_summary.v1",
        "tasks": len(by_task),
        "records": len(records),
        "errors": errors,
        "filters": empty_filters,
        "pass_k": pass_k_summary,
        "auc": auc_block,
        "final": final_block,
        "cost_to_succeed": cost_to_succeed,
    }


def format_metrics_summary_text(summary: Dict[str, Any]) -> str:
    lines = [
        f"tasks={summary.get('tasks')} records={summary.get('records')} "
        f"errors={summary.get('errors')}",
    ]
    # Sparse pass@k first, then AUC, then final (after pass@60 when present).
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
                f"({block.get('tasks_passed_at_turn')}/"
                f"{block.get('tasks_total', block.get('tasks_with_turn_data'))})"
            )
        if micro is not None:
            lines.append(
                f"pass@{turn}_micro={micro:.4f} "
                f"({block.get('tests_passed_sum')}/{block.get('tests_total_sum')})"
            )
    auc = summary.get("auc") or {}
    if isinstance(auc, dict) and auc.get("macro_success_rate") is not None:
        k_min = auc.get("k_min", 1)
        k_max = auc.get("k_max")
        lines.append(
            f"auc_macro@{k_min}-{k_max}={auc['macro_success_rate']:.4f}"
        )
    if isinstance(auc, dict) and auc.get("micro_success_rate") is not None:
        k_min = auc.get("k_min", 1)
        k_max = auc.get("k_max")
        lines.append(
            f"auc_micro@{k_min}-{k_max}={auc['micro_success_rate']:.4f}"
        )
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
    sui = final.get("success_user_iterations") or {}
    if sui.get("mean") is not None:
        lines.append(
            f"final_success_turn={sui['mean']:.2f}±{sui.get('std') or 0:.2f} "
            f"(n={sui.get('n')}, includes_verification)"
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
