#!/usr/bin/env python3
"""
CoEvoSkills train/test protocol on SkillsBench splits.

1. **Evolve** skills on a split (default: ``train`` of ``stratified_v1.json``).
2. **Freeze** the skill library (no further evolution).
3. **Evaluate** a fresh Hermes agent with frozen skills on any split
   (e.g. ``test``), with pass@k checkpoints for shared metrics aggregation.

JSONL rows use ``schema=skillsbench.baseline_run.v1`` with the same metric fields
as Hermes (``evaluation``, ``pass_at_turn``, ``run_conversation_result``) so
``aggregate_skillsbench_runs.py`` / ``analyze_hot_pool_runs.py`` work unchanged.

Examples
--------
# Evolve on stratified train (65 tasks), isolated *task* workspace
# (Hermes still uses ~/.hermes default skills unless --isolate-hermes-home)
python -m benchmark.baselines.coevoskills.run_split_protocol \\
  --evolve \\
  --experiment-dir benchmark/runs/coevo_exp1 \\
  --split-file benchmark/skillsbench_splits/stratified_v1.json \\
  --split-part train \\
  --model qwen/qwen3.6-plus \\
  --log-jsonl benchmark/runs/coevo_exp1/evolve_train.jsonl

# Build frozen library from train workspaces, then eval on test
python -m benchmark.baselines.coevoskills.run_split_protocol \\
  --build-library \\
  --frozen-eval \\
  --experiment-dir benchmark/runs/coevo_exp1 \\
  --split-file benchmark/skillsbench_splits/stratified_v1.json \\
  --split-part test \\
  --pass-k 1,5,10,70 \\
  --max-iterations 90 \\
  --install-skills-into-task \\
  --log-jsonl benchmark/runs/coevo_exp1/frozen_test.jsonl \\
  --aggregate-out benchmark/runs/coevo_exp1/frozen_test_summary.json

# After an interrupt: same command + --resume (skips tasks already in the JSONL)
python -m benchmark.baselines.coevoskills.run_split_protocol \\
  --evolve --split-part train --resume \\
  --log-jsonl benchmark/runs/coevo_evolve_train.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
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


def _load_split_ids(split_file: Path, part: str) -> List[str]:
    path = split_file.expanduser().resolve()
    mod_path = path.parents[1] / "scripts" / "make_skillsbench_splits.py"
    # Prefer repo scripts next to skillsbench_splits/
    candidates = [
        _repo_root() / "benchmark" / "scripts" / "make_skillsbench_splits.py",
        mod_path,
    ]
    for cand in candidates:
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("make_skillsbench_splits", cand)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return list(mod.resolve_split_task_ids(path, part))
    raise FileNotFoundError("make_skillsbench_splits.py not found")


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def build_frozen_library(
    *,
    work_root: Path,
    source_task_ids: List[str],
    library_dir: Path,
) -> Dict[str, Any]:
    """
    Merge evolved ``evo-*`` skills from per-task workspaces into one frozen library.

    Name collisions: keep first occurrence; later duplicates are stored as
    ``<skill>__from_<task_id>``.
    """
    from benchmark.baselines.coevoskills import skill_io

    if library_dir.exists():
        shutil.rmtree(library_dir)
    library_dir.mkdir(parents=True, exist_ok=True)

    installed: List[Dict[str, str]] = []
    seen: Dict[str, str] = {}
    for tid in source_task_ids:
        skills_dir = work_root / tid / "skills"
        for pack in skill_io.list_skills(skills_dir):
            name = pack.name
            dest_name = name
            if name in seen:
                dest_name = f"{name}__from_{tid}"
            dest = library_dir / dest_name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(pack.path, dest)
            seen[dest_name] = tid
            installed.append({"skill": dest_name, "from_task": tid, "src_name": name})

    manifest = {
        "schema": "coevoskills.frozen_library.v1",
        "library_dir": str(library_dir.resolve()),
        "source_tasks": list(source_task_ids),
        "skills": installed,
        "n_skills": len(installed),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (library_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _library_skills_prompt(library_dir: Path, max_chars: int = 24000) -> str:
    from benchmark.baselines.coevoskills import skill_io

    return skill_io.skills_meta_context(library_dir, max_chars=max_chars)


def run_frozen_eval_task(
    *,
    task_id: str,
    hermes_root: Path,
    skillsbench_root: Path,
    library_dir: Path,
    model: str,
    max_iterations: int,
    pass_k_turns: Optional[List[int]],
    eval_timeout_sec: float,
    skip_context_files: bool,
    skip_memory: bool,
    quiet_mode: bool,
    install_into_task: bool,
    per_task_skills_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fresh Hermes solve with frozen skills + optional pass@k host evals."""
    _ensure_path(hermes_root)
    from benchmark.baselines.coevoskills import hermes_backend, oracle, skill_io
    from evaluate_skillsbench_task import PassAtTurnTracker, evaluate_task_host
    from skillsbench_metrics import build_envelope_metrics

    skills_src = per_task_skills_dir if per_task_skills_dir is not None else library_dir
    task_dir = oracle.resolve_task_dir(skillsbench_root, task_id)
    skills_dest = None
    if install_into_task and skills_src.is_dir():
        skills_dest = skill_io.install_skills_into_task(skills_src, task_dir)

    skills_text = _library_skills_prompt(skills_src) if skills_src.is_dir() else "(no skills)"
    prompt_base = str((skillsbench_root / "tasks").resolve())
    user = (
        f"Complete the SkillsBench task at {prompt_base}/{task_id}, "
        f"following instruction.md. Produce outputs in that task directory.\n\n"
        f"Frozen CoEvoSkills library (do not create new skills; reuse these):\n"
        f"Path: {skills_src}\n\n{skills_text}\n\n"
        f"Prefer importing their scripts/ utilities rather than rewriting logic."
    )
    if skills_dest:
        user += f"\nSkills also installed under {skills_dest}."

    print_fn = None
    try:
        from skillsbench_console_window import get_active_window

        win = get_active_window()
        if win is not None:
            print_fn = win.print_fn
    except Exception:
        pass

    agent = hermes_backend.make_agent(
        hermes_root=hermes_root,
        model=model,
        max_iterations=max_iterations,
        skip_context_files=skip_context_files,
        skip_memory=skip_memory,
        quiet_mode=quiet_mode,
        session_id=f"coevo-frozen-{task_id}",
        print_fn=print_fn,
    )

    pass_tracker = None
    if pass_k_turns:
        pass_tracker = PassAtTurnTracker(
            task_id=task_id,
            skillsbench_root=skillsbench_root,
            pass_k_turns=pass_k_turns,
            eval_timeout_sec=eval_timeout_sec,
            evaluate_fn=evaluate_task_host,
        )
        pass_tracker.attach(agent)

    t0 = time.perf_counter()
    pass_at_turn: Dict[str, Any] = {}
    try:
        result = hermes_backend.run_turn(
            agent,
            user_message=user,
            system_message=(
                "You are evaluating frozen CoEvoSkills packages. "
                "Solve the task using the provided skills; do not evolve new skills."
            ),
        )
        if pass_tracker is not None and isinstance(result, dict):
            pass_at_turn = pass_tracker.finalize(int(result.get("api_calls") or 0))
    finally:
        if pass_tracker is not None:
            pass_tracker.detach(agent)

    evaluation = evaluate_task_host(
        task_id=task_id,
        skillsbench_root=skillsbench_root,
        timeout_sec=eval_timeout_sec,
    )
    elapsed = time.perf_counter() - t0

    # Normalize usage fields onto run_conversation_result for aggregators
    if not isinstance(result, dict):
        result = {"final_response": str(result)}
    # Prefer agent session counters if present on the agent object
    try:
        from skillsbench_metrics import snapshot_agent_usage

        snap = snapshot_agent_usage(agent)
        result.setdefault("api_calls", snap.get("api_calls"))
        tokens = snap.get("tokens") or {}
        result.setdefault("total_tokens", tokens.get("total"))
        result.setdefault("input_tokens", tokens.get("input"))
        result.setdefault("output_tokens", tokens.get("output"))
        result.setdefault("estimated_cost_usd", snap.get("estimated_cost_usd"))
    except Exception:
        pass

    envelope: Dict[str, Any] = {
        "schema": "skillsbench.baseline_run.v1",
        "method": "coevoskills",
        "phase": "frozen_eval",
        "ts_end_iso": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(elapsed, 6),
        "skillsbench_task_id": task_id,
        "skillsbench_root": str(skillsbench_root),
        "hermes_root": str(hermes_root),
        "model": hermes_backend.resolve_model(model, hermes_root),
        "max_user_iterations": max_iterations,
        "frozen_skills_dir": str(skills_src.resolve()),
        "skills_installed_to": str(skills_dest) if skills_dest else None,
        "user_message": user[:4000],
        "run_conversation_result": result,
        "evaluation": evaluation,
        "pass_k_turns": list(pass_k_turns or []),
        "pass_at_turn": pass_at_turn,
    }
    envelope["metrics"] = build_envelope_metrics(envelope)
    return envelope


def main(argv: Optional[List[str]] = None) -> int:
    repo = _repo_root()
    p = argparse.ArgumentParser(
        description=(
            "CoEvoSkills: evolve on a SkillsBench split, freeze skills, "
            "evaluate on another split with shared pass@k metrics."
        )
    )
    p.add_argument("--hermes-root", type=Path, default=repo)
    p.add_argument(
        "--skillsbench-root",
        type=Path,
        default=repo / "benchmark" / "skillsbench",
    )
    p.add_argument(
        "--split-file",
        type=Path,
        default=repo / "benchmark" / "skillsbench_splits" / "stratified_v1.json",
        help="Split JSON (default: stratified_v1.json).",
    )
    p.add_argument(
        "--split-part",
        choices=("train", "test", "val", "all"),
        default=None,
        help="Partition to run for --evolve / --frozen-eval (required with those flags).",
    )
    p.add_argument(
        "--library-dir",
        type=Path,
        default=None,
        help=(
            "Frozen skill library path (default: "
            "<experiment-dir>/coevoskills/_frozen_library if --experiment-dir "
            "is set, else benchmark/runs/coevoskills/_frozen_library)."
        ),
    )
    p.add_argument(
        "--library-source-part",
        choices=("train", "test", "val", "all"),
        default="train",
        help="Which split's evolved workspaces feed --build-library (default: train).",
    )
    p.add_argument("--model", type=str, default="")
    p.add_argument("--max-oracle-iters", type=int, default=5)
    p.add_argument("--max-surrogate-iters", type=int, default=15)
    p.add_argument("--max-agent-iterations", type=int, default=60)
    p.add_argument(
        "--max-iterations",
        type=int,
        default=90,
        help="Max user iterations for frozen Hermes eval (also annotated on metrics).",
    )
    p.add_argument(
        "--pass-k",
        type=str,
        default="1,5,10,70",
        help="Conversation turns for pass@k during frozen eval.",
    )
    p.add_argument("--eval-timeout-sec", type=float, default=600.0)
    p.add_argument("--skip-context-files", action="store_true", default=True)
    p.add_argument("--no-skip-context-files", action="store_false", dest="skip_context_files")
    p.add_argument("--skip-memory", action="store_true", default=True)
    p.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Suppress Hermes agent tool/stream console output. "
            "Default is --no-quiet so you can see what the agent is doing."
        ),
    )
    p.add_argument(
        "--console-window",
        type=int,
        default=12,
        metavar="N",
        help=(
            "Fixed N-line viewport: agent activity overwrites in-place "
            "(batch progress / errors stay permanent). "
            "Set 0 for full scrolling output. Default: 12."
        ),
    )
    p.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help=(
            "Isolated experiment workspace. Evolution workspaces go under "
            "DIR/coevoskills/<task_id>/; frozen library defaults to "
            "DIR/coevoskills/_frozen_library. For --frozen-eval, copies selected "
            "tasks under DIR/skillsbench/tasks/. Does NOT rewrite HERMES_HOME by "
            "default — opt in with --isolate-hermes-home."
        ),
    )
    p.add_argument(
        "--isolate-hermes-home",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "With --experiment-dir: also set HERMES_HOME=DIR/hermes_home "
            "(default: off — keep ~/.hermes). When enabled, skills are a "
            "writable copy; config.yaml / .env / SOUL.md symlink to the real "
            "Hermes home. Re-seed skills with --reset-task-workspaces."
        ),
    )
    p.add_argument(
        "--reset-task-workspaces",
        action="store_true",
        help=(
            "With --experiment-dir: refresh task copies and wipe prior agent "
            "outputs. With --isolate-hermes-home, also re-seed hermes_home/skills."
        ),
    )
    p.add_argument("--evolve", action="store_true", help="Run Alg. 1 on --split-part tasks.")
    p.add_argument(
        "--build-library",
        action="store_true",
        help="Merge evolved skills from --library-source-part into --library-dir.",
    )
    p.add_argument(
        "--frozen-eval",
        action="store_true",
        help="Fresh Hermes + frozen skills on --split-part (no further evolution).",
    )
    p.add_argument(
        "--per-task-skills",
        action="store_true",
        help=(
            "For frozen-eval: use each task's own evolved skills under "
            "<experiment-dir>/coevoskills/<task>/skills (or "
            "benchmark/runs/coevoskills/...) instead of the pooled library."
        ),
    )
    p.add_argument(
        "--install-skills-into-task",
        action="store_true",
        help="Copy frozen skills into tasks/<id>/environment/skills/ before eval.",
    )
    p.add_argument("--log-jsonl", type=Path, default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip tasks that already have a finished (non-error) row in --log-jsonl "
            "for this phase. Re-run interrupted/failed tasks. Requires --log-jsonl."
        ),
    )
    p.add_argument(
        "--resume-success-only",
        action="store_true",
        help="With --resume: only skip tasks whose last row has task_success=True.",
    )
    p.add_argument(
        "--aggregate-out",
        type=Path,
        default=None,
        help="After frozen-eval, write aggregate metrics JSON for --log-jsonl.",
    )
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument(
        "--start-task-index",
        type=int,
        default=0,
        help="Slice start within the selected split (for resumable batches).",
    )
    p.add_argument("--end-task-index", type=int, default=None)
    args = p.parse_args(argv)

    if args.reset_task_workspaces and not args.experiment_dir:
        p.error("--reset-task-workspaces requires --experiment-dir")

    hermes_root = args.hermes_root.resolve()
    skillsbench_root = args.skillsbench_root.resolve()
    default_work_root = (hermes_root / "benchmark" / "runs" / "coevoskills").resolve()
    _ensure_path(hermes_root)

    if args.resume or args.resume_success_only:
        if not args.log_jsonl:
            p.error("--resume / --resume-success-only requires --log-jsonl")
        if args.resume_success_only:
            args.resume = True

    if args.split_part is None and (args.evolve or args.frozen_eval or args.list_tasks):
        if args.list_tasks:
            args.split_part = "train"
        else:
            p.error("--split-part is required with --evolve / --frozen-eval")

    task_ids = _load_split_ids(args.split_file.resolve(), args.split_part or "train")
    if args.list_tasks:
        for tid in task_ids:
            print(tid)
        return 0

    if not (args.evolve or args.build_library or args.frozen_eval):
        p.error("Provide at least one of --evolve, --build-library, --frozen-eval")

    start = args.start_task_index
    end = args.end_task_index
    run_ids = task_ids[start:] if end is None else task_ids[start:end]
    if (args.evolve or args.frozen_eval) and not run_ids:
        print("No tasks in selected range.", file=sys.stderr)
        return 1

    source_skillsbench_root = skillsbench_root
    experiment_dir: Optional[Path] = None
    if args.experiment_dir:
        scripts = str((hermes_root / "benchmark" / "scripts").resolve())
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from skillsbench_experiment_workspace import prepare_experiment_workspace
        from hermes_constants import get_hermes_home

        experiment_dir = args.experiment_dir.expanduser().resolve()
        source_hermes_home = Path(get_hermes_home()).resolve()
        work_root = (experiment_dir / "coevoskills").resolve()
        print(f"[experiment] work_root → {work_root}", flush=True)

        copy_tasks = bool(args.frozen_eval)
        print(
            f"[experiment] preparing workspace → {experiment_dir} "
            f"(copy_tasks={copy_tasks}, n={len(run_ids) if copy_tasks else 0})",
            flush=True,
        )
        manifest = prepare_experiment_workspace(
            experiment_dir=experiment_dir,
            source_skillsbench_root=source_skillsbench_root,
            task_ids=run_ids if copy_tasks else [],
            reset_outputs=bool(args.reset_task_workspaces) and copy_tasks,
            isolate_hermes_home=bool(args.isolate_hermes_home),
            apply_hermes_home_env=bool(args.isolate_hermes_home),
            source_hermes_home=source_hermes_home,
            # Even when not copying tasks, --reset-task-workspaces should
            # re-seed isolated Hermes skills for a clean evolve/eval start.
            reset_hermes_skills=(
                bool(args.reset_task_workspaces)
                if args.isolate_hermes_home
                else None
            ),
        )
        if copy_tasks:
            skillsbench_root = Path(manifest["skillsbench_root"])
            print(f"[experiment] skillsbench_root → {skillsbench_root}", flush=True)
        if args.isolate_hermes_home:
            seed = manifest.get("skills_seed") or {}
            shared = manifest.get("shared_hermes_files") or {}
            shared_files = shared.get("files") or {}
            linked = [
                name
                for name, info in shared_files.items()
                if info.get("action") in ("linked", "exists", "copied", "refreshed")
            ]
            print(
                f"[experiment] HERMES_HOME → {manifest.get('hermes_home')} "
                f"(skills {seed.get('action')}, n={seed.get('n_skills')}; "
                f"shared {','.join(linked) or 'none'})",
                flush=True,
            )
        else:
            print(
                "[experiment] HERMES_HOME unchanged (default Hermes skills from ~/.hermes); "
                "pass --isolate-hermes-home to sandbox skills/memory "
                "(seeded copy of current Hermes skills)",
                flush=True,
            )
    else:
        work_root = default_work_root
        if args.isolate_hermes_home or args.reset_task_workspaces:
            p.error("--isolate-hermes-home / --reset-task-workspaces require --experiment-dir")

    library_dir = (
        args.library_dir.resolve()
        if args.library_dir
        else (work_root / "_frozen_library")
    )

    from benchmark.baselines.coevoskills import hermes_backend
    from benchmark.baselines.coevoskills.algorithm import run_coevo_skills
    from benchmark.baselines.coevoskills.config import CoEvoConfig
    from evaluate_skillsbench_task import parse_pass_k_values
    from skillsbench_batch_progress import (
        BatchProgress,
        filter_pending_tasks,
        load_completed_task_ids,
    )
    from skillsbench_console_window import ConsoleWindow, set_active_window
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

    pass_k_turns: Optional[List[int]] = None
    if args.pass_k:
        try:
            pass_k_turns = parse_pass_k_values(args.pass_k)
        except ValueError as e:
            p.error(str(e))

    rc = 0
    log_path = args.log_jsonl.resolve() if args.log_jsonl else None
    console_win: Optional[ConsoleWindow] = None
    if args.console_window and args.console_window > 0:
        console_win = ConsoleWindow(args.console_window)
        set_active_window(console_win)
        print(
            f"[console] activity window={args.console_window} lines "
            f"(batch progress permanent; agent output overwrites; errors→stderr)",
            flush=True,
        )

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

    if args.evolve:
        pending, skipped = _pending_for_phase("evolve", run_ids)
        prog = BatchProgress(
            phase="evolve", total=len(run_ids), already_done=len(skipped)
        )
        prog.banner(resume=args.resume, skipped=len(skipped), pending=len(pending))
        if skipped:
            preview = ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else "")
            print(f"[evolve] skipping completed: {preview}", flush=True)
        for i, task_id in enumerate(pending, start=1):
            t0 = prog.task_start(i, task_id)
            try:
                summary = run_coevo_skills(
                    task_id=task_id,
                    cfg=cfg,
                    hermes_root=hermes_root,
                    skillsbench_root=skillsbench_root,
                    work_root=work_root,
                    progress=True,
                )
                envelope: Dict[str, Any] = {
                    "schema": "skillsbench.baseline_run.v1",
                    "method": "coevoskills",
                    "phase": "evolve",
                    "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                    "duration_sec": round(time.perf_counter() - t0, 6),
                    "skillsbench_task_id": task_id,
                    "skillsbench_root": str(skillsbench_root),
                    "hermes_root": str(hermes_root),
                    "split_file": str(args.split_file.resolve()),
                    "split_part": args.split_part,
                    "model": model,
                    "max_user_iterations": args.max_agent_iterations,
                    "evolution": summary,
                    "evaluation": {
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
                        "note": "Evolution loop; prefer frozen_eval rows for transfer metrics.",
                    },
                }
                envelope["metrics"] = build_envelope_metrics(envelope)
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
                    "schema": "skillsbench.baseline_run_error.v1",
                    "method": "coevoskills",
                    "phase": "evolve",
                    "skillsbench_task_id": task_id,
                    "error": str(exc),
                    "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                    "split_file": str(args.split_file.resolve()),
                    "split_part": args.split_part,
                }
                if log_path:
                    _append_jsonl(log_path, err)
                prog.task_end(i, task_id, t0=t0, ok=False, detail=str(exc)[:200])
                print(f"evolve FAILED {task_id}: {exc}", file=sys.stderr, flush=True)
        batch_stats = prog.finish()
        if log_path:
            _append_jsonl(
                log_path,
                {
                    "schema": "skillsbench.batch_timing.v1",
                    "method": "coevoskills",
                    "phase": "evolve",
                    "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                    "split_file": str(args.split_file.resolve()),
                    "split_part": args.split_part,
                    "timing": batch_stats,
                },
            )

    if args.build_library:
        src_ids = _load_split_ids(args.split_file.resolve(), args.library_source_part)
        print(
            f"[build-library] merging skills from {len(src_ids)} "
            f"{args.library_source_part} tasks → {library_dir}",
            flush=True,
        )
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
                    "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                    "manifest": manifest,
                    "split_file": str(args.split_file.resolve()),
                    "library_source_part": args.library_source_part,
                },
            )

    if args.frozen_eval:
        if not args.per_task_skills and not library_dir.is_dir():
            print(
                f"Frozen library missing: {library_dir}. Run --build-library first "
                f"or pass --per-task-skills.",
                file=sys.stderr,
            )
            return 1
        pending, skipped = _pending_for_phase("frozen_eval", run_ids)
        prog = BatchProgress(
            phase="frozen_eval", total=len(run_ids), already_done=len(skipped)
        )
        prog.banner(resume=args.resume, skipped=len(skipped), pending=len(pending))
        if skipped:
            preview = ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else "")
            print(f"[frozen_eval] skipping completed: {preview}", flush=True)
        for i, task_id in enumerate(pending, start=1):
            t0 = prog.task_start(i, task_id)
            per_task = (work_root / task_id / "skills") if args.per_task_skills else None
            try:
                envelope = run_frozen_eval_task(
                    task_id=task_id,
                    hermes_root=hermes_root,
                    skillsbench_root=skillsbench_root,
                    library_dir=library_dir,
                    model=model,
                    max_iterations=args.max_iterations,
                    pass_k_turns=pass_k_turns,
                    eval_timeout_sec=args.eval_timeout_sec,
                    skip_context_files=args.skip_context_files,
                    skip_memory=args.skip_memory,
                    quiet_mode=args.quiet,
                    install_into_task=args.install_skills_into_task,
                    per_task_skills_dir=per_task,
                )
                envelope["split_file"] = str(args.split_file.resolve())
                envelope["split_part"] = args.split_part
                if experiment_dir is not None:
                    envelope["experiment_dir"] = str(experiment_dir)
                    envelope["skillsbench_root"] = str(skillsbench_root)
                if log_path:
                    _append_jsonl(log_path, envelope)
                ev = envelope.get("evaluation") or {}
                ok = bool(ev.get("task_success"))
                if not ok:
                    rc = 1
                prog.task_end(
                    i,
                    task_id,
                    t0=t0,
                    ok=ok,
                    detail=f"reward={ev.get('reward')}",
                )
            except Exception as exc:
                rc = 1
                err = {
                    "schema": "skillsbench.baseline_run_error.v1",
                    "method": "coevoskills",
                    "phase": "frozen_eval",
                    "skillsbench_task_id": task_id,
                    "error": str(exc),
                    "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                    "split_file": str(args.split_file.resolve()),
                    "split_part": args.split_part,
                }
                if log_path:
                    _append_jsonl(log_path, err)
                prog.task_end(i, task_id, t0=t0, ok=False, detail=str(exc)[:200])
                print(f"frozen-eval FAILED {task_id}: {exc}", file=sys.stderr, flush=True)
        batch_stats = prog.finish()
        if log_path:
            _append_jsonl(
                log_path,
                {
                    "schema": "skillsbench.batch_timing.v1",
                    "method": "coevoskills",
                    "phase": "frozen_eval",
                    "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                    "split_file": str(args.split_file.resolve()),
                    "split_part": args.split_part,
                    "timing": batch_stats,
                },
            )

        if args.aggregate_out and log_path:
            from read_skillsbench_jsonl import iter_skillsbench_run_records
            from skillsbench_aggregate_core import (
                aggregate_skillsbench_metrics,
                format_metrics_summary_text,
            )

            records = list(iter_skillsbench_run_records(log_path))
            summary = aggregate_skillsbench_metrics(
                records,
                pass_k_values=pass_k_turns or [1],
                split_part=args.split_part,
                method="coevoskills",
                phase="frozen_eval",
                max_user_iterations=args.max_iterations,
            )
            out = args.aggregate_out.resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(format_metrics_summary_text(summary), flush=True)
            print(f"Wrote aggregate → {out}", flush=True)

    if console_win is not None:
        console_win.close()
        set_active_window(None)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
