#!/usr/bin/env python3
"""
SkillsBench driver for the isolated CoEvoSkills baseline.

Evolve skills via Alg. 1, optionally install them for a fresh Hermes solve,
and/or score workspace artifacts with the host verifier.

Examples
--------
# Evolve one task (writes under benchmark/runs/coevoskills/<task>/)
python -m benchmark.baselines.coevoskills.run_skillsbench \\
  --task adaptive-cruise-control --evolve \\
  --model qwen/qwen3.6-plus --skip-context-files

# Score artifacts produced during evolution (host pytest, same metrics as Hermes)
python -m benchmark.baselines.coevoskills.run_skillsbench \\
  --task adaptive-cruise-control --oracle-only

# Fresh Hermes agent using evolved skills, then host eval → JSONL
python -m benchmark.baselines.coevoskills.run_skillsbench \\
  --task adaptive-cruise-control --evaluate-with-hermes \\
  --log-jsonl benchmark/runs/coevoskills_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_import_path(hermes_root: Path) -> None:
    root = str(hermes_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _list_tasks(skillsbench_root: Path) -> List[str]:
    tasks_dir = skillsbench_root / "tasks"
    if not tasks_dir.is_dir():
        return []
    return sorted(
        p.name
        for p in tasks_dir.iterdir()
        if p.is_dir() and (p / "instruction.md").is_file()
    )


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _evaluate_with_hermes(
    *,
    task_id: str,
    hermes_root: Path,
    skillsbench_root: Path,
    work_root: Path,
    model: str,
    max_iterations: int,
    skip_context_files: bool,
    skip_memory: bool,
    quiet_mode: bool,
    install_skills_into_task: bool,
) -> Dict[str, Any]:
    """Run a fresh Hermes solve with evolved skills injected, then host-eval."""
    _ensure_import_path(hermes_root)
    from benchmark.baselines.coevoskills import hermes_backend, oracle, skill_io
    from benchmark.scripts.evaluate_skillsbench_task import evaluate_task_host

    paths = skill_io.ensure_workspace(work_root, task_id)
    task_dir = oracle.resolve_task_dir(skillsbench_root, task_id)
    skills_dest = None
    if install_skills_into_task:
        skills_dest = skill_io.install_skills_into_task(paths["skills"], task_dir)

    skills_text = skill_io.skills_meta_context(paths["skills"], max_chars=20000)
    prompt_base = str((skillsbench_root / "tasks").resolve())
    user = (
        f"Complete the SkillsBench task at {prompt_base}/{task_id}, "
        f"following instruction.md. Produce outputs in that task directory.\n\n"
        f"You MUST use the following evolved CoEvoSkills packages "
        f"(already on disk under {paths['skills']}):\n\n{skills_text}\n\n"
        f"Prefer importing their scripts/ utilities rather than rewriting logic."
    )
    if skills_dest:
        user += f"\nSkills are also installed at {skills_dest}."

    agent = hermes_backend.make_agent(
        hermes_root=hermes_root,
        model=model,
        max_iterations=max_iterations,
        skip_context_files=skip_context_files,
        skip_memory=skip_memory,
        quiet_mode=quiet_mode,
        session_id=f"coevo-eval-{task_id}",
    )
    result = hermes_backend.run_turn(
        agent,
        user_message=user,
        system_message=(
            "You are evaluating evolved skills from CoEvoSkills. "
            "Solve the task using the provided evo-* skills."
        ),
    )
    evaluation = evaluate_task_host(
        task_id=task_id,
        skillsbench_root=skillsbench_root,
        timeout_sec=600.0,
    )
    return {
        "final_response": (result.get("final_response") or "")[:4000],
        "evaluation": evaluation,
        "skills_dir": str(paths["skills"]),
        "skills_installed_to": str(skills_dest) if skills_dest else None,
    }


def main(argv: Optional[List[str]] = None) -> int:
    repo = _repo_root()
    p = argparse.ArgumentParser(
        description="CoEvoSkills baseline on SkillsBench (isolated under benchmark/baselines/coevoskills)."
    )
    p.add_argument("--hermes-root", type=Path, default=repo)
    p.add_argument(
        "--skillsbench-root",
        type=Path,
        default=repo / "benchmark" / "skillsbench",
    )
    p.add_argument(
        "--work-root",
        type=Path,
        default=repo / "benchmark" / "runs" / "coevoskills",
        help="Per-task evolution workspaces (skills, artifacts, history).",
    )
    p.add_argument("--task", type=str, default=None, help="Single task id.")
    p.add_argument("--all-tasks", action="store_true")
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument("--model", type=str, default="")
    p.add_argument("--max-oracle-iters", type=int, default=5)
    p.add_argument("--max-surrogate-iters", type=int, default=15)
    p.add_argument("--max-agent-iterations", type=int, default=60)
    p.add_argument("--context-cap", type=float, default=0.7)
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
    p.add_argument("--evolve", action="store_true", help="Run Alg. 1 skill co-evolution.")
    p.add_argument(
        "--oracle-only",
        action="store_true",
        help="Score existing workspace artifacts with the opaque host oracle.",
    )
    p.add_argument(
        "--evaluate-with-hermes",
        action="store_true",
        help="Fresh Hermes solve using evolved skills, then host eval.",
    )
    p.add_argument(
        "--install-skills-into-task",
        action="store_true",
        help="Copy evo-* skills into tasks/<id>/environment/skills/ before Hermes eval.",
    )
    p.add_argument(
        "--log-jsonl",
        type=Path,
        default=None,
        help="Append per-task result envelopes (compatible with aggregate scripts).",
    )
    p.add_argument(
        "--sync-artifacts-into-task",
        action="store_true",
        help="Copy evolved artifacts into the SkillsBench task tree (mutates tasks/).",
    )
    args = p.parse_args(argv)

    hermes_root = args.hermes_root.resolve()
    skillsbench_root = args.skillsbench_root.resolve()
    work_root = args.work_root.resolve()
    _ensure_import_path(hermes_root)

    if args.list_tasks:
        for tid in _list_tasks(skillsbench_root):
            print(tid)
        return 0

    tasks: List[str] = []
    if args.task:
        tasks = [args.task]
    elif args.all_tasks:
        tasks = _list_tasks(skillsbench_root)
    else:
        p.error("Provide --task, --all-tasks, or --list-tasks")

    if not (args.evolve or args.oracle_only or args.evaluate_with_hermes):
        p.error("Provide at least one of --evolve, --oracle-only, --evaluate-with-hermes")

    from benchmark.baselines.coevoskills.config import CoEvoConfig
    from benchmark.baselines.coevoskills.algorithm import run_coevo_skills
    from benchmark.baselines.coevoskills import hermes_backend, oracle, skill_io

    model = hermes_backend.resolve_model(args.model, hermes_root)
    cfg = CoEvoConfig(
        max_oracle_iters=args.max_oracle_iters,
        max_surrogate_iters=args.max_surrogate_iters,
        context_cap=args.context_cap,
        max_agent_iterations=args.max_agent_iterations,
        eval_timeout_sec=args.eval_timeout_sec,
        skip_context_files=args.skip_context_files,
        skip_memory=args.skip_memory,
        model=model,
        quiet_mode=args.quiet,
    )

    rc = 0
    for task_id in tasks:
        envelope: Dict[str, Any] = {
            "benchmark": "skillsbench",
            "method": "coevoskills",
            "paper": "arXiv:2604.01687",
            "task_id": task_id,
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "eval_mode": "host",
        }
        print(f"=== CoEvoSkills {task_id} ===", flush=True)

        if args.evolve:
            summary = run_coevo_skills(
                task_id=task_id,
                cfg=cfg,
                hermes_root=hermes_root,
                skillsbench_root=skillsbench_root,
                work_root=work_root,
            )
            envelope["evolution"] = summary
            envelope["task_success"] = bool(summary.get("success"))
            envelope["reward"] = 1.0 if summary.get("success") else float(
                ((summary.get("final_oracle_opaque") or {}).get("reward") or 0.0)
            )
            print(
                f"evolve done success={summary.get('success')} "
                f"oracle_iters={summary.get('oracle_iters_used')} "
                f"surrogate_iters={summary.get('surrogate_iters_used')}",
                flush=True,
            )

        if args.oracle_only or args.sync_artifacts_into_task:
            paths = skill_io.ensure_workspace(work_root, task_id)
            ev = oracle.run_ground_truth_oracle(
                hermes_root=hermes_root,
                skillsbench_root=skillsbench_root,
                task_id=task_id,
                artifacts_dir=paths["artifacts"],
                timeout_sec=args.eval_timeout_sec,
                sync_into_task=args.sync_artifacts_into_task,
            )
            envelope["evaluation"] = ev
            envelope["task_success"] = bool(ev.get("task_success"))
            envelope["reward"] = float(ev.get("reward") or 0.0)
            print(
                f"oracle task_success={ev.get('task_success')} reward={ev.get('reward')}",
                flush=True,
            )

        if args.evaluate_with_hermes:
            hermes_out = _evaluate_with_hermes(
                task_id=task_id,
                hermes_root=hermes_root,
                skillsbench_root=skillsbench_root,
                work_root=work_root,
                model=model,
                max_iterations=args.max_agent_iterations,
                skip_context_files=args.skip_context_files,
                skip_memory=args.skip_memory,
                quiet_mode=args.quiet,
                install_skills_into_task=args.install_skills_into_task,
            )
            envelope["hermes_eval"] = {
                "skills_dir": hermes_out.get("skills_dir"),
                "skills_installed_to": hermes_out.get("skills_installed_to"),
                "final_response_tail": hermes_out.get("final_response"),
            }
            ev = hermes_out.get("evaluation") or {}
            envelope["evaluation"] = ev
            envelope["task_success"] = bool(ev.get("task_success"))
            envelope["reward"] = float(ev.get("reward") or 0.0)
            print(
                f"hermes-eval task_success={ev.get('task_success')} reward={ev.get('reward')}",
                flush=True,
            )

        if args.log_jsonl:
            _append_jsonl(args.log_jsonl.resolve(), envelope)
            print(f"logged → {args.log_jsonl}", flush=True)

        if not envelope.get("task_success"):
            rc = 1

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
