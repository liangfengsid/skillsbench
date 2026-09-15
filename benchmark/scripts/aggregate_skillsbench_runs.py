#!/usr/bin/env python3
"""
Aggregate SkillsBench JSONL runs into shared metrics summaries.

Works for Hermes, CoEvoSkills, and other baselines that log compatible envelopes
(``skillsbench.hermes_run.v1`` / ``skillsbench.baseline_run.v1`` with
``evaluation``, ``pass_at_turn``, ``run_conversation_result``).

Metrics (see ``skillsbench_aggregate_core.py``):
  - macro / micro success rate at each pass@k turn budget (cumulative ≤k;
    checkpoints-only by default; ``--no-pass-k-checkpoints-only`` to include finals)
  - AUC of macro/micro pass@k over continuous k=1..60 (default; ``--auc-max-k``)
  - final rates after pass@k / AUC, with mean±std success turn (verification included)
  - cost to succeed: mean ± std of tokens and user iterations among successes

Usage::

  python3 benchmark/scripts/aggregate_skillsbench_runs.py \\
    benchmark/runs/batch.jsonl \\
    --pass-k 1,5,10,60 \\
    --auc-max-k 60 \\
    --max-user-iterations 90 \\
    -o benchmark/runs/batch_summary.json \\
    --print-summary

  # Opt into crediting in-budget final / verification-retry eval for pass@k:
  python3 benchmark/scripts/aggregate_skillsbench_runs.py RUN.jsonl \\
    --pass-k 1,5,10,60 --no-pass-k-checkpoints-only --print-summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from evaluate_skillsbench_task import parse_pass_k_values  # noqa: E402
from read_skillsbench_jsonl import iter_skillsbench_run_records  # noqa: E402
from skillsbench_aggregate_core import (  # noqa: E402
    aggregate_skillsbench_metrics,
    format_metrics_summary_text,
)


def aggregate_records(
    records,
    *,
    pass_k_values: List[int],
    split_part: Optional[str] = None,
    method: Optional[str] = None,
    phase: Optional[str] = None,
    max_user_iterations: Optional[int] = None,
    include_final_in_pass_k: bool = False,
    auc_max_k: int = 60,
) -> Dict[str, Any]:
    """Build summary dict from JSONL run records (shared schema)."""
    return aggregate_skillsbench_metrics(
        records,
        pass_k_values=pass_k_values,
        split_part=split_part,
        method=method,
        phase=phase,
        max_user_iterations=max_user_iterations,
        include_final_in_pass_k=include_final_in_pass_k,
        auc_max_k=auc_max_k,
    )


def format_summary_text(summary: Dict[str, Any]) -> str:
    return format_metrics_summary_text(summary)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate SkillsBench JSONL: macro/micro success@k, AUC@1..K, "
            "final rates (with success turn), cost-to-succeed mean±std."
        ),
    )
    parser.add_argument(
        "jsonl_files",
        nargs="+",
        help="JSONL logs from Hermes / CoEvoSkills / other baseline drivers",
    )
    parser.add_argument(
        "--pass-k",
        type=str,
        default="1",
        help="Comma-separated conversation turns (e.g. 1,5,10,60).",
    )
    parser.add_argument(
        "--auc-max-k",
        type=int,
        default=60,
        help=(
            "Compute normalized AUC of macro/micro pass@k for continuous "
            "integer k from 1 to this value (default 60). Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-user-iterations",
        type=int,
        default=None,
        help=(
            "Turn budget for cost_to_succeed (and a label on final_*). "
            "Only tasks whose first success is at or before this turn are "
            "included in cost_to_succeed_*. With --pass-k-checkpoints-only "
            "and this omitted, the budget is max(--pass-k)."
        ),
    )
    parser.add_argument(
        "--pass-k-checkpoints-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Score pass@k / AUC from pass_at_turn snapshots only (default: on). "
            "Do not credit post-conversation host eval or verification retries "
            "when user_iterations ≤ k. final_* rates are unchanged. "
            "Use --no-pass-k-checkpoints-only to also credit in-budget finals."
        ),
    )
    parser.add_argument(
        "--split-part",
        type=str,
        default=None,
        choices=("train", "test", "all"),
        help="Only aggregate rows logged with this split_part.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Filter by method field (e.g. hermes, coevoskills).",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        help="Filter by phase field (e.g. evolve, frozen_eval).",
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
        method=args.method,
        phase=args.phase,
        max_user_iterations=args.max_user_iterations,
        include_final_in_pass_k=not args.pass_k_checkpoints_only,
        auc_max_k=args.auc_max_k,
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
