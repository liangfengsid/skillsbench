"""Batch progress + resume helpers for long SkillsBench runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


ERROR_SCHEMAS = frozenset(
    {
        "skillsbench.hermes_run_error.v1",
        "skillsbench.baseline_run_error.v1",
        "terminalbench.hermes_run_error.v1",
        "terminalbench.baseline_run_error.v1",
    }
)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def load_completed_task_ids(
    log_jsonl: Path,
    *,
    phase: Optional[str] = None,
    method: Optional[str] = None,
    success_only: bool = False,
) -> Set[str]:
    """
    Tasks that already have a finished (non-error) JSONL row.

    Uses the latest row per task. Error schemas do not count as completed
    (so ``--resume`` will retry them).
    """
    if not log_jsonl.is_file():
        return set()
    latest: Dict[str, dict] = {}
    with log_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            tid = rec.get("skillsbench_task_id") or rec.get("task_id")
            if not tid:
                continue
            tid = str(tid)
            if phase and rec.get("phase") and rec.get("phase") != phase:
                # Rows without phase (Hermes) are ignored when filtering by phase
                continue
            if phase and method == "coevoskills" and not rec.get("phase"):
                continue
            if method and rec.get("method") and rec.get("method") != method:
                continue
            prev = latest.get(tid)
            if prev is None or (rec.get("ts_end_iso") or "") >= (prev.get("ts_end_iso") or ""):
                latest[tid] = rec

    done: Set[str] = set()
    for tid, rec in latest.items():
        if rec.get("schema") in ERROR_SCHEMAS:
            continue
        if success_only:
            ev = rec.get("evaluation") or {}
            ok = ev.get("task_success")
            if ok is None and isinstance(rec.get("evolution"), dict):
                ok = rec["evolution"].get("success")
            if not ok:
                continue
        done.add(tid)
    return done


def filter_pending_tasks(
    task_ids: List[str],
    completed: Set[str],
) -> tuple[List[str], List[str]]:
    """Return (pending, skipped)."""
    pending = [t for t in task_ids if t not in completed]
    skipped = [t for t in task_ids if t in completed]
    return pending, skipped


class BatchProgress:
    """Console progress tracker for multi-task batches."""

    def __init__(
        self,
        *,
        phase: str,
        total: int,
        already_done: int = 0,
    ):
        self.phase = phase
        self.total = total
        self.already_done = already_done
        self.started_at = time.perf_counter()
        self.task_durations: List[float] = []
        self.completed_this_run = 0
        self.failed_this_run = 0

    def banner(self, *, resume: bool, skipped: int, pending: int) -> None:
        print_step(
            f"\n[{self.phase}] batch start: pending={pending}/{self.total} "
            f"skipped_completed={skipped} resume={resume}"
        )

    def task_start(self, index: int, task_id: str) -> float:
        """index is 1-based among pending tasks this run."""
        pending_total = max(1, self.total - self.already_done)
        done = self.completed_this_run + self.failed_this_run
        eta = ""
        if self.task_durations:
            avg = sum(self.task_durations) / len(self.task_durations)
            remain = pending_total - done
            eta = f" eta≈{_fmt_duration(avg * remain)}"
        elapsed = _fmt_duration(time.perf_counter() - self.started_at)
        print_step(
            f"\n[{self.phase}] [{index}/{pending_total}] "
            f"START {task_id}  (batch {self.already_done + done + 1}/{self.total}) "
            f"elapsed={elapsed}{eta}"
        )
        return time.perf_counter()

    def task_end(
        self,
        index: int,
        task_id: str,
        *,
        t0: float,
        ok: bool,
        detail: str = "",
    ) -> None:
        dt = time.perf_counter() - t0
        self.task_durations.append(dt)
        if ok:
            self.completed_this_run += 1
        else:
            self.failed_this_run += 1
        pending_total = max(1, self.total - self.already_done)
        status = "OK" if ok else "FAIL"
        extra = f" {detail}" if detail else ""
        print_step(
            f"[{self.phase}] [{index}/{pending_total}] "
            f"{status} {task_id}  took={_fmt_duration(dt)}{extra}"
        )

    def finish(self) -> Dict[str, Any]:
        wall = time.perf_counter() - self.started_at
        elapsed = _fmt_duration(wall)
        stats = {
            "phase": self.phase,
            "ok": self.completed_this_run,
            "fail": self.failed_this_run,
            "skipped_before": self.already_done,
            "wall_sec": round(wall, 3),
            "task_duration_sec": {
                "n": len(self.task_durations),
                "mean": (
                    sum(self.task_durations) / len(self.task_durations)
                    if self.task_durations
                    else None
                ),
                "sum": sum(self.task_durations) if self.task_durations else 0.0,
                "min": min(self.task_durations) if self.task_durations else None,
                "max": max(self.task_durations) if self.task_durations else None,
            },
        }
        td = stats["task_duration_sec"]
        mean_s = f"{td['mean']:.1f}s" if td["mean"] is not None else "n/a"
        print_step(
            f"\n[{self.phase}] batch done: ok={self.completed_this_run} "
            f"fail={self.failed_this_run} skipped_before={self.already_done} "
            f"wall={elapsed} task_time_mean={mean_s} "
            f"task_time_sum={_fmt_duration(td['sum'] or 0)}"
        )
        return stats


def print_step(msg: str) -> None:
    try:
        from skillsbench_console_window import commit

        commit(msg)
    except Exception:
        print(msg, flush=True)
