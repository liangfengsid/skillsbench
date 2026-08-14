#!/usr/bin/env python3
"""
CoEvoSkills train/freeze/test protocol on Terminal-Bench (official Harbor eval).

Fair counterpart to ``run_terminalbench_with_harbor.py``:

1. **Evolve** skills on a TB split. Generator authors evo-* packages on the host;
   each iteration materializes Φ with ``CoEvoHarborAgent`` (same Hermes loop as
   the official TB agent) and uses Harbor ``tests/test.sh`` as the opaque oracle.
2. **Freeze** the skill library (same merge as SkillsBench).
3. **Frozen eval** on another split: ``HermesHarborAgent`` + overlay frozen
   skills into an isolated ``HERMES_HOME``. No pass@k (one Harbor reward / trial).

Examples
--------
python -m benchmark.baselines.coevoskills.run_terminalbench_protocol \\
  --evolve \\
  --experiment-dir benchmark/runs/coevo_tb_exp1 \\
  --split-file benchmark/terminalbench_splits/stratified_v1.json \\
  --split-part train \\
  --model Qwen/Qwen3.6-27B \\
  --max-iterations 60 \\
  --log-jsonl benchmark/runs/coevo_tb_exp1/evolve_train.jsonl \\
  --exclude-task-name math-eval-grader --exclude-task-name jax-speedrun-gpu

python -m benchmark.baselines.coevoskills.run_terminalbench_protocol \\
  --build-library --frozen-eval \\
  --experiment-dir benchmark/runs/coevo_tb_exp1 \\
  --split-part test \\
  --model Qwen/Qwen3.6-27B \\
  --log-jsonl benchmark/runs/coevo_tb_exp1/frozen_test.jsonl \\
  --aggregate-out benchmark/runs/coevo_tb_exp1/frozen_test_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_path(hermes_root: Path) -> None:
    root = str(hermes_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    scripts = str((hermes_root / "benchmark" / "scripts").resolve())
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _print_task_summary(rows: List[Dict[str, Any]], *, label: str) -> None:
    n = len(rows)
    ok = sum(1 for e in rows if (e.get("evaluation") or {}).get("task_success"))
    if n:
        print(f"[{label}] tasks={n} passed={ok} rate={ok / n:.3f}", flush=True)
    else:
        print(f"[{label}] no task results", flush=True)
    for e in rows:
        ev = e.get("evaluation") or {}
        print(
            f"  {e.get('skillsbench_task_id')}: "
            f"success={ev.get('task_success')!r} reward={ev.get('reward')!r}",
            flush=True,
        )


def _load_split_ids(split_file: Path, part: str) -> List[str]:
    from .run_split_protocol import _load_split_ids as load_ids

    return load_ids(split_file, part)


def main(argv: Optional[List[str]] = None) -> int:
    repo = _repo_root()
    p = argparse.ArgumentParser(
        description=(
            "CoEvoSkills on Terminal-Bench: evolve with Harbor oracle, freeze "
            "skills, evaluate with HermesHarborAgent (no pass@k)."
        )
    )
    p.add_argument("--hermes-root", type=Path, default=repo)
    p.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Harbor tasks directory (default: benchmark/terminal-bench/tasks).",
    )
    p.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Default: benchmark/terminalbench_splits/stratified_v1.json.",
    )
    p.add_argument(
        "--split-part",
        choices=("train", "test", "val", "all"),
        default=None,
        help="Partition for --evolve / --frozen-eval.",
    )
    p.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help=(
            "Experiment workspace. Evolution under DIR/coevoskills/; "
            "HERMES_HOME=DIR/hermes_home; Harbor jobs under DIR/jobs."
        ),
    )
    p.add_argument("--library-dir", type=Path, default=None)
    p.add_argument("--library-source-part", choices=("train", "test", "val", "all"), default="train")
    p.add_argument("--model", type=str, default="")
    p.add_argument("--max-oracle-iters", type=int, default=5)
    p.add_argument("--max-surrogate-iters", type=int, default=15)
    p.add_argument("--max-agent-iterations", type=int, default=60)
    p.add_argument(
        "--max-iterations",
        type=int,
        default=90,
        help="Harbor agent budget for materialize + frozen eval (matches Hermes TB).",
    )
    p.add_argument("--eval-timeout-sec", type=float, default=600.0)
    p.add_argument("--skip-context-files", action="store_true", default=True)
    p.add_argument("--no-skip-context-files", action="store_false", dest="skip_context_files")
    p.add_argument("--skip-memory", action="store_true", default=True)
    p.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--env",
        dest="harbor_env",
        default="docker",
        help="Harbor environment type (default: docker).",
    )
    p.add_argument("--n-concurrent", type=int, default=1)
    p.add_argument("--evolve", action="store_true")
    p.add_argument("--build-library", action="store_true")
    p.add_argument(
        "--frozen-eval",
        action="store_true",
        help="HermesHarborAgent + frozen skills overlay; Harbor verifier. No pass@k.",
    )
    p.add_argument(
        "--per-task-skills",
        action="store_true",
        help="Frozen eval: overlay each task's own evolved skills instead of the pooled library.",
    )
    p.add_argument("--log-jsonl", type=Path, default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip finished tasks (JSONL and, for frozen-eval, Harbor result.json).",
    )
    p.add_argument("--resume-success-only", action="store_true")
    p.add_argument("--aggregate-out", type=Path, default=None)
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument("--start-task-index", type=int, default=0)
    p.add_argument("--end-task-index", type=int, default=None)
    p.add_argument(
        "--print-summary",
        action="store_true",
        help=(
            "After evolve / frozen-eval, print per-task success and reward "
            "(same recap as the Hermes Harbor driver)."
        ),
    )
    p.add_argument(
        "--exclude-task-name",
        action="append",
        default=None,
        metavar="TASK_ID",
        help=(
            "Skip this task id (repeatable). Drops GPU tasks before evolve "
            "materialize and frozen eval. Same as Harbor -x / the Hermes driver "
            "`-- --exclude-task-name`."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print Harbor frozen-eval command (and evolve plan) without running.",
    )
    p.add_argument("--reset-task-workspaces", action="store_true")
    p.add_argument(
        "harbor_args",
        nargs="*",
        help="Extra Harbor CLI args after `--` (scanned for --exclude-task-name).",
    )
    args = p.parse_args(argv)

    hermes_root = args.hermes_root.resolve()
    _ensure_path(hermes_root)
    experiment_dir = args.experiment_dir.expanduser().resolve()
    dataset_path = (
        args.dataset_path.expanduser().resolve()
        if args.dataset_path
        else (hermes_root / "benchmark" / "terminal-bench" / "tasks")
    )
    dataset_root = dataset_path.parent
    split_file = (
        args.split_file.expanduser().resolve()
        if args.split_file
        else (hermes_root / "benchmark" / "terminalbench_splits" / "stratified_v1.json")
    )
    if args.split_part is None and (args.evolve or args.frozen_eval or args.list_tasks):
        p.error("--split-part is required with --evolve / --frozen-eval / --list-tasks")
    if not (args.evolve or args.build_library or args.frozen_eval or args.list_tasks):
        p.error("Provide at least one of --evolve, --build-library, --frozen-eval, --list-tasks")
    if args.resume and not args.log_jsonl:
        p.error("--resume requires --log-jsonl")

    from run_terminalbench_with_harbor import collect_exclude_task_names, discover_task_ids

    task_ids = discover_task_ids(dataset_path)
    excludes = collect_exclude_task_names(args.exclude_task_name, args.harbor_args)
    if args.list_tasks or args.evolve or args.frozen_eval:
        run_ids = _load_split_ids(split_file, args.split_part)
        unknown = [tid for tid in run_ids if tid not in task_ids]
        if unknown:
            p.error(f"Split file lists unknown task ids: {', '.join(unknown[:5])}")
        if excludes:
            skip = set(excludes)
            dropped = [tid for tid in run_ids if tid in skip]
            run_ids = [tid for tid in run_ids if tid not in skip]
            if dropped:
                print(
                    f"[protocol] excluding {len(dropped)} task(s): {', '.join(dropped)}",
                    flush=True,
                )
        run_ids = run_ids[args.start_task_index : args.end_task_index]
    else:
        run_ids = []

    if args.list_tasks:
        for tid in run_ids:
            print(tid)
        return 0

    work_root = experiment_dir / "coevoskills"
    library_dir = (
        args.library_dir.expanduser().resolve()
        if args.library_dir
        else (work_root / "_frozen_library")
    )
    jobs_dir = experiment_dir / "jobs"
    isolate_home = experiment_dir / "hermes_home"
    log_path = args.log_jsonl.expanduser().resolve() if args.log_jsonl else None

    from benchmark.baselines.coevoskills import harbor_eval, hermes_backend
    from benchmark.baselines.coevoskills.algorithm import run_coevo_skills
    from benchmark.baselines.coevoskills.config import CoEvoConfig
    from .run_split_protocol import build_frozen_library
    from skillsbench_batch_progress import (
        BatchProgress,
        filter_pending_tasks,
        load_completed_task_ids,
    )
    from skillsbench_metrics import build_envelope_metrics

    model = hermes_backend.resolve_model(args.model, hermes_root)
    cfg = CoEvoConfig(
        max_oracle_iters=args.max_oracle_iters,
        max_surrogate_iters=args.max_surrogate_iters,
        max_agent_iterations=args.max_agent_iterations,
        eval_timeout_sec=args.eval_timeout_sec,
        skip_context_files=args.skip_context_files,
        skip_memory=args.skip_memory,
        model=model,
        quiet_mode=args.quiet,
    )
    driver = harbor_eval.load_harbor_driver(hermes_root)

    def _pending_for_phase(phase: str, ids: List[str]) -> tuple[List[str], List[str]]:
        if not args.resume or not log_path:
            return ids, []
        done = load_completed_task_ids(
            log_path,
            phase=phase,
            method="coevoskills",
            success_only=args.resume_success_only,
        )
        return filter_pending_tasks(ids, done)

    rc = 0

    if args.evolve:
        if not args.dry_run:
            driver.prepare_harbor_isolate(
                experiment_dir=experiment_dir,
                dataset_path=dataset_path,
                reset_skills=bool(args.reset_task_workspaces),
            )
        pending, skipped = _pending_for_phase("evolve", run_ids)
        prog = BatchProgress(phase="evolve", total=len(run_ids), already_done=len(skipped))
        prog.banner(resume=args.resume, skipped=len(skipped), pending=len(pending))
        if skipped:
            preview = ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else "")
            print(f"[evolve] skipping completed: {preview}", flush=True)
        if args.dry_run:
            print(f"[evolve] dry-run pending={len(pending)} isolate={isolate_home}")
        else:
            evolve_rows: List[Dict[str, Any]] = []
            for i, task_id in enumerate(pending, start=1):
                t0 = prog.task_start(i, task_id)

                def _materialize(task_id=task_id, paths=None, iteration=0, **_k):
                    return harbor_eval.run_materialize_trial(
                        hermes_root=hermes_root,
                        dataset_path=dataset_path,
                        task_id=task_id,
                        paths=paths,
                        isolate_home=isolate_home,
                        jobs_dir=jobs_dir,
                        model=model,
                        harbor_env=args.harbor_env,
                        max_iterations=args.max_iterations,
                        iteration=iteration,
                        experiment_dir=experiment_dir,
                    )
    
                try:
                    from benchmark.baselines.coevoskills.skill_io import ensure_workspace
    
                    hint_paths = ensure_workspace(work_root, task_id)
                    summary = run_coevo_skills(
                        task_id=task_id,
                        cfg=cfg,
                        hermes_root=hermes_root,
                        skillsbench_root=dataset_root,
                        work_root=work_root,
                        progress=True,
                        env_hint=harbor_eval.harbor_env_hint(hint_paths),
                        prompt_mode="harbor",
                        materialize_fn=_materialize,
                        oracle_fn=harbor_eval.harbor_oracle_from_materialize,
                    )
                    envelope: Dict[str, Any] = {
                        "schema": "terminalbench.baseline_run.v1",
                        "method": "coevoskills",
                        "phase": "evolve",
                        "benchmark": "terminal-bench",
                        "eval_mode": "harbor",
                        "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                        "duration_sec": round(time.perf_counter() - t0, 6),
                        "skillsbench_task_id": task_id,
                        "dataset_path": str(dataset_path),
                        "hermes_root": str(hermes_root),
                        "split_file": str(split_file),
                        "split_part": args.split_part,
                        "model": model,
                        "max_user_iterations": args.max_iterations,
                        "evolution": summary,
                        "evaluation": {
                            "eval_mode": "harbor",
                            "task_success": bool(summary.get("success")),
                            "reward": 1.0 if summary.get("success") else float(
                                ((summary.get("final_oracle_opaque") or {}).get("reward") or 0.0)
                            ),
                            "ran": True,
                        },
                        "run_conversation_result": {
                            "completed": True,
                            "api_calls": summary.get("surrogate_iters_used"),
                            "total_tokens": None,
                            "note": "Evolution loop; prefer frozen_eval Harbor rows for transfer metrics.",
                        },
                    }
                    envelope["metrics"] = build_envelope_metrics(envelope)
                    evolve_rows.append(envelope)
                    if log_path:
                        _append_jsonl(log_path, envelope)
                    prog.task_end(
                        i,
                        task_id,
                        t0=t0,
                        ok=True,
                        detail=f"success={summary.get('success')} "
                        f"oracle={summary.get('oracle_iters_used')} "
                        f"surrogate={summary.get('surrogate_iters_used')}",
                    )
                except Exception as exc:
                    rc = 1
                    err = {
                        "schema": "terminalbench.baseline_run_error.v1",
                        "method": "coevoskills",
                        "phase": "evolve",
                        "skillsbench_task_id": task_id,
                        "error": str(exc),
                        "evaluation": {
                            "task_success": False,
                            "reward": 0.0,
                            "ran": False,
                            "error": str(exc),
                        },
                        "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                        "split_file": str(split_file),
                        "split_part": args.split_part,
                    }
                    if log_path:
                        _append_jsonl(log_path, err)
                    evolve_rows.append(err)
                    prog.task_end(i, task_id, t0=t0, ok=False, detail=str(exc)[:200])
                    print(f"evolve FAILED {task_id}: {exc}", file=sys.stderr, flush=True)
        prog.finish()
        if args.print_summary and not args.dry_run:
            _print_task_summary(evolve_rows, label="evolve")

    if args.build_library:
        src_ids = _load_split_ids(split_file, args.library_source_part)
        print(
            f"[build-library] merging skills from {len(src_ids)} "
            f"{args.library_source_part} tasks → {library_dir}",
            flush=True,
        )
        if not args.dry_run:
            manifest = build_frozen_library(
                work_root=work_root,
                source_task_ids=src_ids,
                library_dir=library_dir,
            )
            print(
                f"[build-library] done → {library_dir} ({manifest.get('n_skills')} skills)",
                flush=True,
            )
            if log_path:
                _append_jsonl(
                    log_path,
                    {
                        "schema": "coevoskills.library_build.v1",
                        "method": "coevoskills",
                        "phase": "build_library",
                        "benchmark": "terminal-bench",
                        "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                        "manifest": manifest,
                        "split_file": str(split_file),
                        "library_source_part": args.library_source_part,
                    },
                )

    if args.frozen_eval:
        if not args.per_task_skills and not library_dir.is_dir() and not args.dry_run:
            print(
                f"Frozen library missing: {library_dir}. Run --build-library first "
                f"or pass --per-task-skills.",
                file=sys.stderr,
            )
            return 1
        if args.per_task_skills:
            pending, skipped = _pending_for_phase("frozen_eval", run_ids)
            print(
                f"[frozen_eval] per-task skills pending={len(pending)} skipped={len(skipped)}",
                flush=True,
            )
            for tid in pending:
                overlay = work_root / tid / "skills"
                argv = harbor_eval.frozen_eval_argv(
                    dataset_path=dataset_path,
                    split_file=split_file,
                    split_part=args.split_part,
                    model=model,
                    harbor_env=args.harbor_env,
                    experiment_dir=experiment_dir,
                    library_dir=overlay,
                    log_jsonl=log_path or (experiment_dir / "frozen_eval.jsonl"),
                    max_iterations=args.max_iterations,
                    n_concurrent=1,
                    resume=bool(args.resume),
                    print_summary=args.print_summary,
                    per_task=tid,
                    exclude_task_names=excludes,
                )
                if args.dry_run:
                    print(" ".join(["run_terminalbench_with_harbor.py", *argv]))
                    continue
                code = driver.main(argv)
                if code not in (0, None):
                    rc = int(code) if isinstance(code, int) else 1
        else:
            if log_path is None:
                p.error("--frozen-eval requires --log-jsonl")
            argv = harbor_eval.frozen_eval_argv(
                dataset_path=dataset_path,
                split_file=split_file,
                split_part=args.split_part,
                model=model,
                harbor_env=args.harbor_env,
                experiment_dir=experiment_dir,
                library_dir=library_dir,
                log_jsonl=log_path,
                max_iterations=args.max_iterations,
                n_concurrent=args.n_concurrent,
                resume=bool(args.resume),
                print_summary=args.print_summary,
                exclude_task_names=excludes,
            )
            if args.dry_run:
                print(" ".join(["run_terminalbench_with_harbor.py", *argv]))
            else:
                code = driver.main(argv)
                if code not in (0, None):
                    rc = int(code) if isinstance(code, int) else 1

        if args.aggregate_out and log_path and not args.dry_run:
            from read_skillsbench_jsonl import iter_skillsbench_run_records
            from skillsbench_aggregate_core import (
                aggregate_skillsbench_metrics,
                format_metrics_summary_text,
            )

            records = list(iter_skillsbench_run_records(log_path))
            summary = aggregate_skillsbench_metrics(
                records,
                pass_k_values=[1],
                split_part=args.split_part,
                method="coevoskills",
                phase="frozen_eval",
                max_user_iterations=args.max_iterations,
            )
            out = args.aggregate_out.expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(format_metrics_summary_text(summary), flush=True)
            print(f"Wrote aggregate → {out}", flush=True)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
