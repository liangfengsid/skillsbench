"""CoEvoSkills Algorithm 1 — co-evolutionary skill generation + surrogate verification."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CoEvoConfig
from .generator import SkillGenerator
from . import oracle as oracle_mod
from . import prompts, skill_io
from .verifier import SurrogateVerifier


def load_instruction(task_dir: Path) -> str:
    path = task_dir / "instruction.md"
    if not path.is_file():
        raise FileNotFoundError(f"Missing instruction.md under {task_dir}")
    return path.read_text(encoding="utf-8", errors="replace")


def run_coevo_skills(
    *,
    task_id: str,
    cfg: CoEvoConfig,
    hermes_root: Path,
    skillsbench_root: Path,
    work_root: Path,
    progress: bool = True,
    env_hint: Optional[str] = None,
    prompt_mode: str = "host",
    materialize_fn=None,
    oracle_fn=None,
) -> Dict[str, Any]:
    """
    Execute Alg. 1 for one SkillsBench task.

    Returns a run summary including opaque final oracle result and skill paths.
    """
    task_dir = oracle_mod.resolve_task_dir(skillsbench_root, task_id)
    instruction = load_instruction(task_dir)
    paths = skill_io.ensure_workspace(work_root, task_id)
    skill_io.copy_task_environment(task_dir, paths["env"])

    if env_hint is None:
        env_hint = (
            f"Input files are under {paths['env']}. "
            f"If doc/ or data/ exist there, read them fully before authoring skills. "
            f"Write ALL task outputs under {paths['artifacts']} (same relative layout "
            f"the instruction / tests expect under /root or the task directory)."
        )

    def _p(msg: str) -> None:
        if not progress:
            return
        try:
            from skillsbench_console_window import commit

            commit(msg)
        except Exception:
            print(msg, flush=True)

    generator = SkillGenerator(
        cfg=cfg,
        hermes_root=hermes_root,
        instruction=instruction,
        paths=paths,
        env_hint=env_hint,
        prompt_mode=prompt_mode,
    )
    # Fresh verifier session — information isolation
    verifier = SurrogateVerifier(
        cfg=cfg,
        hermes_root=hermes_root,
        instruction=instruction,
        paths=paths,
    )

    n = 0  # oracle interventions
    m = 0  # surrogate iterations
    feedback = ""
    escalate_next = False
    history: List[Dict[str, Any]] = []
    final_oracle: Dict[str, Any] = {}
    t0 = time.time()
    _p(
        f"  [evolve:{task_id}] begin "
        f"(max_oracle={cfg.max_oracle_iters} max_surrogate={cfg.max_surrogate_iters})"
    )

    while n < cfg.max_oracle_iters and m < cfg.max_surrogate_iters:
        m += 1
        _p(
            f"  [evolve:{task_id}] surrogate_iter={m}/{cfg.max_surrogate_iters} "
            f"oracle_used={n}/{cfg.max_oracle_iters} → generator"
        )
        gen_out = generator.propose(iteration=m, feedback=feedback)
        skill_io.append_history(
            paths["history"],
            {"type": "generator", "m": m, "n": n, "out": {
                "skills_listing": gen_out.get("skills_listing"),
                "artifacts_listing": gen_out.get("artifacts_listing"),
                "final_response_tail": (gen_out.get("final_response") or "")[-1500:],
            }},
        )

        last_mat: Optional[Dict[str, Any]] = None
        if materialize_fn is not None:
            _p(f"  [evolve:{task_id}] surrogate_iter={m} → Harbor materialize")
            try:
                mat = materialize_fn(task_id=task_id, paths=paths, iteration=m) or {}
            except Exception as exc:
                mat = {
                    "evaluation": {
                        "task_success": False,
                        "reward": 0.0,
                        "ran": False,
                        "error": str(exc),
                    }
                }
            gen_out["materialize"] = {
                "task_success": (mat.get("evaluation") or {}).get("task_success"),
                "error": (mat.get("evaluation") or {}).get("error"),
            }
            skill_io.append_history(
                paths["history"],
                {
                    "type": "materialize",
                    "m": m,
                    "n": n,
                    "evaluation_opaque": oracle_mod.opaque_signal(mat.get("evaluation") or {}),
                },
            )
            last_mat = mat

        _p(
            f"  [evolve:{task_id}] surrogate_iter={m} → verifier "
            f"(escalate={escalate_next})"
        )
        ver_out = verifier.evaluate(iteration=m, escalate=escalate_next)
        escalate_next = False
        skill_io.append_history(
            paths["history"],
            {"type": "surrogate", "m": m, "n": n, "out": {
                "r_hat": ver_out.get("r_hat"),
                "passed": ver_out.get("passed"),
                "diagnostics": (ver_out.get("diagnostics") or "")[:4000],
            }},
        )
        history.append({"m": m, "n": n, "r_hat": ver_out.get("r_hat"), "surrogate_passed": ver_out.get("passed")})

        r_hat = float(ver_out.get("r_hat") or 0.0)
        surrogate_ok = bool(ver_out.get("passed")) and r_hat >= 1.0
        if not surrogate_ok:
            _p(
                f"  [evolve:{task_id}] surrogate FAIL r_hat={r_hat:.3f} "
                f"(retry generator)"
            )
            # Dense diagnostics back to generator (same session C)
            feedback = (
                "SURROGATE VERIFIER FAILED.\n"
                f"Estimated reward R_hat={r_hat:.4f}.\n\n"
                f"{ver_out.get('diagnostics') or ''}"
            )
            continue

        # Surrogate believes success → call opaque ground-truth oracle
        n += 1
        _p(
            f"  [evolve:{task_id}] surrogate PASS → opaque oracle "
            f"({n}/{cfg.max_oracle_iters})"
        )
        if oracle_fn is not None:
            full_eval = oracle_fn(
                task_id=task_id,
                paths=paths,
                iteration=m,
                last_materialize=last_mat,
            )
        else:
            full_eval = oracle_mod.run_ground_truth_oracle(
                hermes_root=hermes_root,
                skillsbench_root=skillsbench_root,
                task_id=task_id,
                artifacts_dir=paths["artifacts"],
                timeout_sec=cfg.eval_timeout_sec,
                sync_into_task=False,
            )
        opaque = oracle_mod.opaque_signal(full_eval)
        final_oracle = {"opaque": opaque, "full": full_eval}
        skill_io.append_history(
            paths["history"],
            {"type": "oracle", "m": m, "n": n, "opaque": opaque},
        )
        history.append({"m": m, "n": n, "oracle": opaque})

        if opaque.get("task_success"):
            _p(f"  [evolve:{task_id}] oracle PASS — done")
            feedback = prompts.opaque_oracle_feedback(passed=True, reward=1.0)
            break

        _p(
            f"  [evolve:{task_id}] oracle FAIL reward={opaque.get('reward')} "
            f"— escalate verifier + continue"
        )
        # Oracle fail after surrogate pass → escalate V, opaque fail to generator
        escalate_next = True
        feedback = prompts.opaque_oracle_feedback(
            passed=False,
            reward=float(opaque.get("reward") or 0.0),
        )
        # Also give generator a nudge that surrogate was over-optimistic (no GT details)
        feedback += (
            "\nSurrogate suite previously estimated success but disagreed with the oracle. "
            "The verifier will escalate its own tests; you should harden skills and regenerate."
        )

        # Soft context-cap: if history messages are huge, stop (paper β)
        # Approximate via generator history length
        approx_chars = sum(len(str(x)) for x in generator.history)
        if approx_chars > int(cfg.context_cap * 400_000):
            _p(f"  [evolve:{task_id}] context_cap reached — stop")
            skill_io.append_history(
                paths["history"],
                {"type": "stop", "reason": "context_cap", "approx_chars": approx_chars},
            )
            break

    elapsed = time.time() - t0
    success = bool((final_oracle.get("opaque") or {}).get("task_success"))
    _p(
        f"  [evolve:{task_id}] finished success={success} "
        f"surrogate_iters={m} oracle_iters={n} elapsed={elapsed:.1f}s"
    )
    summary: Dict[str, Any] = {
        "method": "coevoskills",
        "paper": "arXiv:2604.01687",
        "task_id": task_id,
        "success": success,
        "oracle_iters_used": n,
        "surrogate_iters_used": m,
        "elapsed_sec": round(elapsed, 2),
        "skills_dir": str(paths["skills"]),
        "artifacts_dir": str(paths["artifacts"]),
        "work_dir": str(paths["base"]),
        "skills_listing": skill_io.skills_listing_text(paths["skills"]),
        "final_oracle_opaque": final_oracle.get("opaque"),
        "history_brief": history,
        "config": {
            "max_oracle_iters": cfg.max_oracle_iters,
            "max_surrogate_iters": cfg.max_surrogate_iters,
            "context_cap": cfg.context_cap,
            "max_agent_iterations": cfg.max_agent_iterations,
            "model": cfg.model,
        },
    }
    skill_io.dump_run_summary(paths["base"], summary)
    return summary
