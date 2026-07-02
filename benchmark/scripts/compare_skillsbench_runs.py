#!/usr/bin/env python3
"""
Compare two Hermes SkillsBench JSONL logs (e.g. different skill strategies).

By default, headline metrics use the **same task set**: task ids that appear in **both**
logs **and** completed successfully in **both** runs (``--cohort both-completed``), so
mean tokens / api_calls / cost are not skewed by different benchmark subsets. Use
``--cohort intersection`` to widen the cohort to all overlapping ids (asymmetric
completion pools per side).

Usage::

  python3 benchmark/scripts/compare_skillsbench_runs.py \\
    benchmark/hermes_skillsbench_runs_all_task_qwen_0503.jsonl \\
    benchmark/hermes_skillsbench_runs_all_task_qwen_0508.jsonl \\
    --label-a nudge-0503 --label-b nudge-0508 \\
    --html benchmark/skillsbench_compare_report.html

Metrics use envelope + ``run_conversation_result`` fields present today. See
``FUTURE_METRICS`` in this file (and printed with ``--future-metrics``) for
suggested additions to logging.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse JSONL reader from the same directory when run as ``python3 benchmark/scripts/...``.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from read_skillsbench_jsonl import iter_skillsbench_run_records  # noqa: E402

FUTURE_METRICS = """
Suggested fields for richer strategy comparison (not in current envelopes):

- **Verifier outcome**: task pass/fail from SkillsBench ``reward.txt`` / pytest
  (requires post-hoc runner or hooking Hermes to write ``skillsbench_pass: bool``).
- **Tool usage**: counts per tool name, terminal vs read_file ratio, total tool rounds.
- **Latency breakdown**: wall-clock excluding LLM vs provider latency (needs timestamps per phase).
- **Cache**: ``cache_read_tokens`` / ``cache_write_tokens`` aggregates (already in result but not summarized here — easy to add).
- **Reasoning tokens**: ``reasoning_tokens`` vs output for cost attribution.
- **Skill loads**: which skills were injected / ``skill_view`` count per task (Hermes session metadata).
  Implemented: ``hot_pool_telemetry`` on each JSONL row + ``benchmark/scripts/analyze_hot_pool_runs.py``.
- **Stable run id**: UUID per invocation to join external verifier tables.
"""


@dataclass
class ParsedRow:
    task_id: str
    schema: str
    completed: Optional[bool]  # None if error / no result
    api_calls: Optional[int]
    total_tokens: Optional[int]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    estimated_cost_usd: Optional[float]
    duration_sec: Optional[float]
    failed: Optional[bool]
    interrupted: Optional[bool]
    error: Optional[str]


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


def parse_record(obj: Dict[str, Any]) -> Optional[ParsedRow]:
    tid = obj.get("skillsbench_task_id")
    if not tid or not isinstance(tid, str):
        return None
    sch = str(obj.get("schema") or "")
    if sch == "skillsbench.hermes_run_error.v1":
        return ParsedRow(
            task_id=tid,
            schema=sch,
            completed=None,
            api_calls=None,
            total_tokens=None,
            input_tokens=None,
            output_tokens=None,
            estimated_cost_usd=None,
            duration_sec=_to_float(obj.get("duration_sec")),
            failed=None,
            interrupted=None,
            error=str(obj.get("error") or "")[:500],
        )
    res = obj.get("run_conversation_result")
    if not isinstance(res, dict):
        return ParsedRow(
            task_id=tid,
            schema=str(obj.get("schema") or "unknown"),
            completed=None,
            api_calls=None,
            total_tokens=None,
            input_tokens=None,
            output_tokens=None,
            estimated_cost_usd=None,
            duration_sec=_to_float(obj.get("duration_sec")),
            failed=None,
            interrupted=None,
            error="missing run_conversation_result",
        )
    return ParsedRow(
        task_id=tid,
        schema=sch,
        completed=bool(res.get("completed")),
        api_calls=_to_int(res.get("api_calls")),
        total_tokens=_to_int(res.get("total_tokens")),
        input_tokens=_to_int(res.get("input_tokens")),
        output_tokens=_to_int(res.get("output_tokens")),
        estimated_cost_usd=_to_float(res.get("estimated_cost_usd")),
        duration_sec=_to_float(obj.get("duration_sec")),
        failed=bool(res.get("failed")) if res.get("failed") is not None else None,
        interrupted=bool(res.get("interrupted")) if res.get("interrupted") is not None else None,
        error=None,
    )


def load_last_per_task(path: Path) -> Tuple[Dict[str, ParsedRow], int]:
    """Last record wins per ``task_id``; returns (rows, duplicate_overwrites)."""
    rows: Dict[str, ParsedRow] = {}
    dup = 0
    for obj in iter_skillsbench_run_records(path):
        pr = parse_record(obj)
        if pr is None:
            continue
        if pr.task_id in rows:
            dup += 1
        rows[pr.task_id] = pr
    return rows, dup


def mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, statistics.stdev(xs)


def median_val(xs: List[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None
    return float(statistics.median(xs))


def summarize(
    rows: Dict[str, ParsedRow],
    *,
    completed_only_for_load: bool,
) -> Dict[str, Any]:
    n = len(rows)
    errors = sum(
        1
        for r in rows.values()
        if r.error and r.completed is None and r.api_calls is None
    )
    completed = [r for r in rows.values() if r.completed is True]
    incomplete = [r for r in rows.values() if r.completed is False]
    unknown = max(0, n - len(completed) - len(incomplete) - errors)
    rate = len(completed) / n if n else 0.0

    def take(vals: List[Optional[int]]) -> List[float]:
        out: List[float] = []
        for v in vals:
            if v is None:
                continue
            out.append(float(v))
        return out

    if completed_only_for_load:
        pool = completed
    else:
        pool = [r for r in rows.values() if r.api_calls is not None]

    tok = take([r.total_tokens for r in pool])
    api = take([r.api_calls for r in pool])
    dur = [r.duration_sec for r in pool if r.duration_sec is not None]
    cost = [r.estimated_cost_usd for r in pool if r.estimated_cost_usd is not None]
    inp = take([r.input_tokens for r in pool])
    outp = take([r.output_tokens for r in pool])

    m_tok, s_tok = mean_std(tok)
    m_api, s_api = mean_std(api)
    m_dur, s_dur = mean_std(dur)
    m_cost, s_cost = mean_std(cost)
    m_in, s_in = mean_std(inp)
    m_out, s_out = mean_std(outp)
    med_tok = median_val(tok)
    med_api = median_val(api)

    return {
        "n_tasks": n,
        "n_completed": len(completed),
        "n_incomplete": len(incomplete),
        "n_errors": errors,
        "n_unknown": max(0, unknown),
        "completion_rate": rate,
        "pool_label": "completed_only" if completed_only_for_load else "all_with_metrics",
        "pool_n": len(pool),
        "mean_total_tokens": m_tok,
        "std_total_tokens": s_tok,
        "mean_api_calls": m_api,
        "std_api_calls": s_api,
        "mean_duration_sec": m_dur,
        "std_duration_sec": s_dur,
        "mean_estimated_cost_usd": m_cost,
        "std_estimated_cost_usd": s_cost,
        "mean_input_tokens": m_in,
        "std_input_tokens": s_in,
        "mean_output_tokens": m_out,
        "std_output_tokens": s_out,
        "median_total_tokens": med_tok,
        "median_api_calls": med_api,
    }


def _count_completed(rows: Dict[str, ParsedRow], task_ids: set[str]) -> int:
    return sum(1 for tid in task_ids if tid in rows and rows[tid].completed is True)


def paired_completed(
    a: Dict[str, ParsedRow], b: Dict[str, ParsedRow]
) -> List[Tuple[str, ParsedRow, ParsedRow]]:
    out: List[Tuple[str, ParsedRow, ParsedRow]] = []
    common = set(a) & set(b)
    for tid in sorted(common):
        ra, rb = a[tid], b[tid]
        if ra.completed is True and rb.completed is True:
            out.append((tid, ra, rb))
    return out


def paired_deltas(pairs: List[Tuple[str, ParsedRow, ParsedRow]]) -> Dict[str, Any]:
    """Return mean and stdev of (B - A) for numeric fields plus ``n_pairs``."""
    if not pairs:
        return {"n_pairs": 0}
    dtok = []
    dapi = []
    dcost = []
    ddur = []
    for _tid, ra, rb in pairs:
        if ra.total_tokens is not None and rb.total_tokens is not None:
            dtok.append(float(rb.total_tokens - ra.total_tokens))
        if ra.api_calls is not None and rb.api_calls is not None:
            dapi.append(float(rb.api_calls - ra.api_calls))
        if ra.estimated_cost_usd is not None and rb.estimated_cost_usd is not None:
            dcost.append(float(rb.estimated_cost_usd - ra.estimated_cost_usd))
        if ra.duration_sec is not None and rb.duration_sec is not None:
            ddur.append(float(rb.duration_sec - ra.duration_sec))
    out: Dict[str, Any] = {
        "delta_total_tokens": mean_std(dtok),
        "delta_api_calls": mean_std(dapi),
        "delta_cost_usd": mean_std(dcost),
        "delta_duration_sec": mean_std(ddur),
        "n_pairs": len(pairs),
    }
    return out


def fmt_ms(x: Optional[float], d: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x:.{d}f}"


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{100.0 * x:.1f}%"


def print_report(
    label_a: str,
    label_b: str,
    path_a: Path,
    path_b: Path,
    sum_a: Dict[str, Any],
    sum_b: Dict[str, Any],
    paired: Dict[str, Any],
    *,
    completed_only: bool,
    cohort: Dict[str, Any],
) -> None:
    mode = "completed-only" if completed_only else "all runs with numeric metrics"
    print(f"SkillsBench run comparison ({mode} for load/cost/token means)")
    print(f"  A: {label_a!r}  ({path_a})")
    print(f"  B: {label_b!r}  ({path_b})")
    print()
    print("## Cohort (fair comparison)")
    print(f"  mode:                    {cohort.get('cohort_mode')!r}")
    print(f"  task_ids in both logs:   {cohort.get('n_intersection')}")
    print(f"  completed in BOTH runs:  {cohort.get('n_both_completed')}")
    print(
        f"  completed on intersection — A: {cohort.get('completed_a_intersection')}"
        f"/{cohort.get('n_intersection')} ({fmt_pct(cohort.get('rate_a_intersection'))}), "
        f"B: {cohort.get('completed_b_intersection')}/{cohort.get('n_intersection')} "
        f"({fmt_pct(cohort.get('rate_b_intersection'))})"
    )
    print(
        "  Headline means below:    "
        f"same {cohort.get('n_cohort')} task id(s) in each column "
        f"({'both completed' if cohort.get('cohort_mode') == 'both-completed' else 'intersection cohort'})"
    )
    print()
    for lab, s in ((label_a, sum_a), (label_b, sum_b)):
        print(f"## {lab}")
        print(f"  tasks (unique ids):     {s['n_tasks']}")
        print(f"  completed:              {s['n_completed']}")
        print(f"  incomplete:             {s['n_incomplete']}")
        print(f"  parse/API errors:     {s['n_errors']}")
        print(f"  completion rate:      {fmt_pct(s['completion_rate'])}")
        print(f"  pool size for means:  {s['pool_n']} ({s['pool_label']})")
        print(
            f"  mean ± std total_tokens: {fmt_ms(s['mean_total_tokens'])} ± {fmt_ms(s['std_total_tokens'])}"
        )
        print(f"  median total_tokens:     {fmt_ms(s['median_total_tokens'])}")
        print(f"  mean ± std api_calls:    {fmt_ms(s['mean_api_calls'])} ± {fmt_ms(s['std_api_calls'])}")
        print(f"  median api_calls:        {fmt_ms(s['median_api_calls'])}")
        print(
            f"  mean ± std duration_s:   {fmt_ms(s['mean_duration_sec'])} ± {fmt_ms(s['std_duration_sec'])}"
        )
        print(
            f"  mean ± std cost_usd:     {fmt_ms(s['mean_estimated_cost_usd'], 4)} ± {fmt_ms(s['std_estimated_cost_usd'], 4)}"
        )
        print()
    print("## Paired (tasks completed in BOTH A and B)")
    n_p = paired.get("n_pairs", 0)
    print(f"  n pairs: {n_p}")
    for key, title in [
        ("delta_total_tokens", "mean ± std (B−A) total_tokens"),
        ("delta_api_calls", "mean ± std (B−A) api_calls"),
        ("delta_cost_usd", "mean ± std (B−A) cost_usd"),
        ("delta_duration_sec", "mean ± std (B−A) duration_sec"),
    ]:
        m, sd = paired.get(key, (None, None))
        if m is None:
            print(f"  {title}: —")
        else:
            print(f"  {title}: {fmt_ms(m)} ± {fmt_ms(sd)}")
    print()


def build_html(
    label_a: str,
    label_b: str,
    path_a: Path,
    path_b: Path,
    sum_a: Dict[str, Any],
    sum_b: Dict[str, Any],
    paired: Dict[str, Any],
    *,
    completed_only: bool,
    cohort: Dict[str, Any],
) -> str:
    mode = "completed-only" if completed_only else "all runs with metrics"
    cohort_p = (
        f"Cohort <strong>{html.escape(str(cohort.get('cohort_mode')))}</strong>: "
        f"{cohort.get('n_intersection')} task id(s) in both logs; "
        f"{cohort.get('n_both_completed')} completed in both. "
        f"On intersection — A completed {cohort.get('completed_a_intersection')}/{cohort.get('n_intersection')}, "
        f"B {cohort.get('completed_b_intersection')}/{cohort.get('n_intersection')}. "
        f"Headline table: same {cohort.get('n_cohort')} id(s) per column."
    )
    rows_metric = [
        ("Unique task ids", sum_a["n_tasks"], sum_b["n_tasks"], ""),
        ("Completed", sum_a["n_completed"], sum_b["n_completed"], ""),
        ("Incomplete", sum_a["n_incomplete"], sum_b["n_incomplete"], ""),
        ("Errors / no result", sum_a["n_errors"], sum_b["n_errors"], ""),
        ("Completion rate", fmt_pct(sum_a["completion_rate"]), fmt_pct(sum_b["completion_rate"]), ""),
        (
            "Mean total tokens ± σ",
            f"{fmt_ms(sum_a['mean_total_tokens'])} ± {fmt_ms(sum_a['std_total_tokens'])}",
            f"{fmt_ms(sum_b['mean_total_tokens'])} ± {fmt_ms(sum_b['std_total_tokens'])}",
            f"pool n={sum_a['pool_n']}/{sum_b['pool_n']}",
        ),
        (
            "Median total tokens",
            fmt_ms(sum_a["median_total_tokens"]),
            fmt_ms(sum_b["median_total_tokens"]),
            "",
        ),
        (
            "Mean api_calls ± σ",
            f"{fmt_ms(sum_a['mean_api_calls'])} ± {fmt_ms(sum_a['std_api_calls'])}",
            f"{fmt_ms(sum_b['mean_api_calls'])} ± {fmt_ms(sum_b['std_api_calls'])}",
            "",
        ),
        (
            "Median api_calls",
            fmt_ms(sum_a["median_api_calls"]),
            fmt_ms(sum_b["median_api_calls"]),
            "",
        ),
        (
            "Mean duration (s) ± σ",
            f"{fmt_ms(sum_a['mean_duration_sec'])} ± {fmt_ms(sum_a['std_duration_sec'])}",
            f"{fmt_ms(sum_b['mean_duration_sec'])} ± {fmt_ms(sum_b['std_duration_sec'])}",
            "",
        ),
        (
            "Mean est. cost USD ± σ",
            f"{fmt_ms(sum_a['mean_estimated_cost_usd'], 4)} ± {fmt_ms(sum_a['std_estimated_cost_usd'], 4)}",
            f"{fmt_ms(sum_b['mean_estimated_cost_usd'], 4)} ± {fmt_ms(sum_b['std_estimated_cost_usd'], 4)}",
            "",
        ),
    ]
    trs = []
    for a0, a1, a2, note in rows_metric:
        trs.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(str(a0)),
                html.escape(str(a1)),
                html.escape(str(a2)),
                html.escape(str(note)),
            )
        )
    pair_rows = []
    for title, key in [
        ("Pairs (both completed)", "n_pairs"),
        ("Δ total_tokens mean ± σ", "delta_total_tokens"),
        ("Δ api_calls mean ± σ", "delta_api_calls"),
        ("Δ cost USD mean ± σ", "delta_cost_usd"),
        ("Δ duration s mean ± σ", "delta_duration_sec"),
    ]:
        if key == "n_pairs":
            v = paired.get("n_pairs", 0)
            pair_rows.append(f"<tr><td colspan='2'>{html.escape(title)}</td><td>{v}</td></tr>")
            continue
        m, sd = paired.get(key, (None, None))
        cell = "—" if m is None else f"{fmt_ms(m)} ± {fmt_ms(sd)}"
        pair_rows.append(f"<tr><td colspan='2'>{html.escape(title)}</td><td>{html.escape(cell)}</td></tr>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>SkillsBench strategy comparison</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 960px; }}
    h1 {{ font-size: 1.25rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #f4f4f4; }}
    .muted {{ color: #555; font-size: 0.9rem; }}
    pre {{ background: #f8f8f8; padding: 1rem; overflow: auto; font-size: 0.8rem; }}
  </style>
</head>
<body>
  <h1>SkillsBench run comparison</h1>
  <p class="muted">Load mode: <strong>{html.escape(mode)}</strong>. A = {html.escape(label_a)}, B = {html.escape(label_b)}</p>
  <p class="muted">{cohort_p}</p>
  <p class="muted">{html.escape(str(path_a))}<br/>{html.escape(str(path_b))}</p>
  <h2>Side-by-side</h2>
  <table>
    <thead><tr><th>Metric</th><th>{html.escape(label_a)}</th><th>{html.escape(label_b)}</th><th>Note</th></tr></thead>
    <tbody>{"".join(trs)}</tbody>
  </table>
  <h2>Paired (B − A), tasks completed in both</h2>
  <table><tbody>{"".join(pair_rows)}</tbody></table>
  <h2>Future metrics (not in JSONL today)</h2>
  <pre>{html.escape(FUTURE_METRICS.strip())}</pre>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_a", type=str, help="First JSONL (baseline / strategy A)")
    p.add_argument("run_b", type=str, help="Second JSONL (strategy B)")
    p.add_argument("--label-a", type=str, default="run_a", help="Display label for first file")
    p.add_argument("--label-b", type=str, default="run_b", help="Display label for second file")
    p.add_argument(
        "--cohort",
        choices=("both-completed", "intersection"),
        default="both-completed",
        help=(
            "both-completed (default): headline metrics use task ids present in BOTH logs "
            "with completed=True in A and B — same symmetric set for fair token/API/cost means. "
            "intersection: all overlapping task ids; per-side pools follow --metrics-scope "
            "(means can differ in coverage)."
        ),
    )
    p.add_argument(
        "--metrics-scope",
        choices=("completed-only", "all-with-metrics"),
        default="completed-only",
        help=(
            "completed-only: means/std over completed runs only (default). "
            "all-with-metrics: include incomplete runs that still have api_calls/tokens."
        ),
    )
    p.add_argument("--html", type=str, default=None, help="Write standalone HTML report to this path")
    p.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="PATH",
        help="Write summary dict (JSON) for dashboards / CI",
    )
    p.add_argument(
        "--future-metrics",
        action="store_true",
        help="Print suggested fields for richer logs and exit",
    )
    args = p.parse_args()
    if args.future_metrics:
        print(FUTURE_METRICS.strip())
        return 0

    path_a = Path(args.run_a).expanduser().resolve()
    path_b = Path(args.run_b).expanduser().resolve()
    if not path_a.is_file() or not path_b.is_file():
        print("error: both inputs must be files", file=sys.stderr)
        return 1

    rows_a, dup_a = load_last_per_task(path_a)
    rows_b, dup_b = load_last_per_task(path_b)
    if dup_a or dup_b:
        print(f"note: duplicate task_ids overwritten (last wins): A={dup_a} B={dup_b}", file=sys.stderr)

    intersection = set(rows_a) & set(rows_b)
    both_completed_ids = {
        tid
        for tid in intersection
        if rows_a[tid].completed is True and rows_b[tid].completed is True
    }
    if args.cohort == "both-completed":
        cohort_ids = both_completed_ids
    else:
        cohort_ids = intersection
    if not cohort_ids:
        print(
            "error: empty cohort — no overlapping task_ids between logs, or "
            "(with --cohort both-completed) none completed in both runs. "
            "Try --cohort intersection or broader logs.",
            file=sys.stderr,
        )
        return 1

    n_inter = len(intersection)
    ca_inter = _count_completed(rows_a, intersection)
    cb_inter = _count_completed(rows_b, intersection)
    cohort: Dict[str, Any] = {
        "cohort_mode": args.cohort,
        "n_intersection": n_inter,
        "n_both_completed": len(both_completed_ids),
        "completed_a_intersection": ca_inter,
        "completed_b_intersection": cb_inter,
        "rate_a_intersection": (ca_inter / n_inter) if n_inter else 0.0,
        "rate_b_intersection": (cb_inter / n_inter) if n_inter else 0.0,
        "n_cohort": len(cohort_ids),
    }

    rows_af = {tid: rows_a[tid] for tid in cohort_ids}
    rows_bf = {tid: rows_b[tid] for tid in cohort_ids}

    comp_only = args.metrics_scope == "completed-only"
    sum_a = summarize(rows_af, completed_only_for_load=comp_only)
    sum_b = summarize(rows_bf, completed_only_for_load=comp_only)
    pairs = paired_completed(rows_a, rows_b)
    pdel = paired_deltas(pairs)
    pdel["n_pairs"] = len(pairs)

    print_report(
        args.label_a,
        args.label_b,
        path_a,
        path_b,
        sum_a,
        sum_b,
        pdel,
        completed_only=comp_only,
        cohort=cohort,
    )
    only_a = sorted(set(rows_a) - set(rows_b))
    only_b = sorted(set(rows_b) - set(rows_a))
    if only_a:
        print(f"Tasks only in A ({len(only_a)}): {', '.join(only_a[:20])}" + (" …" if len(only_a) > 20 else ""))
    if only_b:
        print(f"Tasks only in B ({len(only_b)}): {', '.join(only_b[:20])}" + (" …" if len(only_b) > 20 else ""))

    if args.html:
        out = Path(args.html).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            build_html(
                args.label_a,
                args.label_b,
                path_a,
                path_b,
                sum_a,
                sum_b,
                pdel,
                completed_only=comp_only,
                cohort=cohort,
            ),
            encoding="utf-8",
        )
        print(f"Wrote HTML report to {out}")
    if args.json:
        jpath = Path(args.json).expanduser()
        jpath.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "label_a": args.label_a,
            "label_b": args.label_b,
            "path_a": str(path_a),
            "path_b": str(path_b),
            "cohort": cohort,
            "metrics_scope": args.metrics_scope,
            "summary_a": sum_a,
            "summary_b": sum_b,
            "paired_b_minus_a": pdel,
            "tasks_only_in_a": sorted(set(rows_a) - set(rows_b)),
            "tasks_only_in_b": sorted(set(rows_b) - set(rows_a)),
        }

        def _json_safe(x: Any) -> Any:
            if isinstance(x, dict):
                return {k: _json_safe(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_json_safe(v) for v in x]
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return x

        jpath.write_text(json.dumps(_json_safe(blob), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON summary to {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
