#!/usr/bin/env python3
"""
Run Hermes AIAgent against SkillsBench tasks (library use of this repo's run_agent).

Requires a Python environment where Hermes dependencies are installed (typically
the Hermes repo venv). This script prepends HERMES_AGENT_ROOT to sys.path so
``from run_agent import AIAgent`` resolves.

Examples (from Hermes repo root):

  python3 benchmark/scripts/run_skillsbench_with_hermes.py --task adaptive-cruise-control \\
      --skill-nudge-interval 10 --memory-nudge-interval 10 \\
      --hot-pool --hot-pool-persist benchmark/skillsbench_hot_pool.json \\
      --log-jsonl benchmark/hermes_skillsbench_runs.jsonl --print-summary

  python3 benchmark/scripts/run_skillsbench_with_hermes.py --task adaptive-cruise-control \\
      --no-hot-pool --log-jsonl benchmark/hermes_skillsbench_runs.jsonl --print-summary

  python3 benchmark/scripts/run_skillsbench_with_hermes.py --all --log-jsonl ./runs.jsonl \\
      --hot-pool --hot-pool-persist benchmark/skillsbench_hot_pool.json

  python3 benchmark/scripts/run_skillsbench_with_hermes.py --all --no-hot-pool \\
      --log-jsonl ./runs.jsonl

  # A-Mem baseline (same Hermes + skill tools, no hot-skill pool):
  python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \\
      --amem --experiment-dir benchmark/runs/exp_amem_train \\
      --isolate-hermes-home \\
      --split-file benchmark/skillsbench_splits/stratified_v1.json --split-part train \\
      --log-jsonl benchmark/runs/exp_amem_train/runs.jsonl --print-summary

  # Dynamic Cheatsheet baseline (same Hermes + skill tools, no hot-skill pool):
  python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \\
      --dc --experiment-dir benchmark/runs/exp_dc_train \\
      --isolate-hermes-home \\
      --split-file benchmark/skillsbench_splits/stratified_v1.json --split-part train \\
      --log-jsonl benchmark/runs/exp_dc_train/runs.jsonl --print-summary

  python3 benchmark/scripts/run_skillsbench_with_hermes.py --all --start-task-index 10 \\
      --log-jsonl ./runs.jsonl  # skip first 10 tasks (sorted order), run the rest

  # After interrupt: re-run same command with --resume (skips finished tasks in JSONL)
  python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \\
      --split-file benchmark/skillsbench_splits/stratified_v1.json --split-part train \\
      --log-jsonl benchmark/runs/hermes_train.jsonl --resume --print-summary

  # Isolated experiment workspace (no pollution of shared skillsbench/tasks):
  python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \\
      --experiment-dir benchmark/runs/exp_hot_train \\
      --split-file benchmark/skillsbench_splits/stratified_v1.json --split-part train \\
      --hot-pool --log-jsonl benchmark/runs/exp_hot_train/runs.jsonl --print-summary

  # Fresh start in the same experiment dir (wipe prior agent outputs):
  python3 benchmark/scripts/run_skillsbench_with_hermes.py --all \\
      --experiment-dir benchmark/runs/exp_hot_train --reset-task-workspaces \\
      --hot-pool --log-jsonl benchmark/runs/exp_hot_train/runs.jsonl

  # pass@k: verifier scores at conversation turns 1,5,10,... within one run (no reruns):
  python3 benchmark/scripts/run_skillsbench_with_hermes.py --task adaptive-cruise-control \\
      --pass-k 1,5,10,70 \\
      --log-jsonl benchmark/hermes_skillsbench_runs.jsonl --print-summary

  # Aggregate pass@k across tasks from JSONL (macro/micro rates at each turn):
  python3 benchmark/scripts/aggregate_skillsbench_runs.py \\
      benchmark/runs/batch.jsonl --pass-k 1,5,10,70 --print-summary

  # Re-score task outputs without re-running the agent:
  python3 benchmark/scripts/run_skillsbench_with_hermes.py \\
      --task adaptive-cruise-control --evaluate-only \\
      --log-jsonl benchmark/hermes_skillsbench_runs.jsonl --print-summary

Environment / API keys follow Hermes (e.g. OPENROUTER_API_KEY); see Hermes docs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Feishu optional deps (lark_oapi) emit setuptools pkg_resources noise on import.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*pkg_resources is deprecated.*",
)

# This file lives at <repo>/benchmark/scripts/<name>.py
_SCRIPT = Path(__file__).resolve()
if str(_SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT.parent))

from aggregate_skillsbench_runs import aggregate_records, format_summary_text  # noqa: E402
from amem_baseline import (  # noqa: E402
    add_amem_cli_flags,
    amem_telemetry_from_agent,
    apply_amem_argparse_policy,
    apply_amem_cli_overrides,
    resolve_amem_persist,
)
from dc_baseline import (  # noqa: E402
    add_dc_cli_flags,
    apply_dc_argparse_policy,
    apply_dc_cli_overrides,
    dc_telemetry_from_agent,
    resolve_dc_persist,
)
from hermes_hot_pool_outcome import apply_benchmark_hot_pool_outcome_feedback  # noqa: E402
from skillsbench_metrics import build_envelope_metrics  # noqa: E402


def _benchmark_dir() -> Path:
    """``<repo>/benchmark`` — sibling of ``skillsbench/`` and ``scripts/``."""
    return _SCRIPT.parents[1]


def _bundled_skillsbench_root() -> Path:
    """Canonical SkillsBench tree shipped in this repo: ``benchmark/skillsbench``."""
    return _benchmark_dir() / "skillsbench"


def _hermes_repo_root_from_script() -> Path:
    """``<repo>`` (parent of ``benchmark/``)."""
    return _SCRIPT.parents[2]

_DEFAULT_HERMES = _hermes_repo_root_from_script()
_DEFAULT_SKILLSBENCH = _bundled_skillsbench_root()
DEFAULT_HERMES_ROOT = _DEFAULT_HERMES
DEFAULT_PROMPT_TASKS_BASE = str((_DEFAULT_SKILLSBENCH / "tasks").resolve())


def benchmark_presets() -> Dict[str, Dict[str, Any]]:
    bench = _benchmark_dir()
    return {
        "skillsbench": {
            "root": bench / "skillsbench",
            "split_file": bench / "skillsbench_splits" / "stratified_v1.json",
            "schema": "skillsbench.hermes_run.v1",
            "error_schema": "skillsbench.hermes_run_error.v1",
            "batch_schema": "skillsbench.batch_timing.v1",
            "label": "SkillsBench",
            "platform": "skillsbench-batch",
            "dataset_dirname": "skillsbench",
        },
    }

SKILLSBENCH_BATCH_SKILL_REVIEW_APPENDIX = (
    "\n\n**SkillsBench batch (platform=skillsbench-batch):** "
    "No user to confirm — call skill_manage yourself when warranted. "
    "Bar is slightly lower than interactive: after non-trivial trial-and-error, "
    "prefer saving a class-level skill over 'Nothing to save.' "
    "Name by task genre or platform grading protocol "
    "(e.g. skillsbench-host-grading), not by task id; avoid catch-all "
    "mega-skills that dump oracle recipes. Include short "
    "'## Common Pitfalls' or '## Best Practices' bullets that stay abstract "
    "(no absolute paths, solution/oracle peeking, running bundled solution/ "
    "scripts, reading ground truth from tests, or temp-dir remaps of graded "
    "outputs) so hot-pool tips transfer to held-out tasks."
)

SKILLSBENCH_BATCH_COMBINED_REVIEW_APPENDIX = (
    "\n\n**SkillsBench batch:** for skills, same rules — no confirmation; "
    "prefer a class-level save after real trial-and-error; platform grading "
    "skills (e.g. skillsbench-host-grading) are encouraged; no catch-all "
    "mega-skills with oracle/path recipes; short '## Common Pitfalls' / "
    "'## Best Practices' bullets only (no 'test pre-existing solution first')."
)

# Meta-prompt: ask the agent to *infer and persist* host grading protocol
# rather than spoiling it as privileged eval knowledge every turn.
# Format constraints match hot-skill extract (short Pitfalls/Best Practices
# bullets) so discovered tips can enter the hot pool.
SKILLSBENCH_HOST_GRADING_META_APPENDIX = (
    "\n\n**Driver grading (discover & encode — do not treat this as the "
    "solution):** After you stop, this batch runs a **host verification** "
    "pass on your task tree and returns pass/fail. Infer the real grading "
    "protocol from those outcomes (and from instruction.md, tests/, "
    "environment/) — including what the verifier actually inspects versus "
    "what only helps local debugging. Persist durable guardrails via "
    "skill_manage so later tasks reuse them.\n"
    "\n"
    "**Required skill shape (hot-pool extractable):**\n"
    "- Create or patch a class-level skill named for platform behavior "
    "(e.g. skillsbench-host-grading), not the task id.\n"
    "- MUST include a ``## Common Pitfalls`` or ``## Best Practices`` section.\n"
    "- Put each guardrail as its own short bullet or numbered item "
    "(one idea, ≤2 sentences, prefer NEVER / DO NOT / ALWAYS / MUST). "
    "Do NOT dump long prose paragraphs — only short bullets are admitted "
    "to the hot skill pool.\n"
    "- Cover at least: (1) where graded deliverables must live in the "
    "**task tree** so the host verifier can see them; (2) what does *not* "
    "count as done (e.g. checks that only pass in a live debugging "
    "session, running bundled scripts under solution/, or workflows that "
    "never show up as host-verification success).\n"
    "- Prefer tips about reading tests/test_outputs.py and test.sh first, "
    "matching expected paths/schemas, and not copying oracles / ground "
    "truth from tests/. Treat local container builds and Dockerfile edits "
    "as hypotheses to validate against host verification — promote a "
    "rule into the skill only after host verification confirms it.\n"
    "- Keep tips abstract and transferable — no oracle answers, no "
    "task-specific paths, no copying from tests/ground truth, no "
    "'test pre-existing solution first'.\n"
    "- Update the same skill when host verification feedback contradicts it "
    "(including when a tip you encoded earlier fails to transfer).\n"
    "Then continue fixing *this* task under that protocol."
)

# Backward-compatible alias for older imports/tests.
SKILLSBENCH_HOST_VERIFICATION_APPENDIX = SKILLSBENCH_HOST_GRADING_META_APPENDIX


def build_skillsbench_skill_review_prompt(base_prompt: str) -> str:
    return base_prompt + SKILLSBENCH_BATCH_SKILL_REVIEW_APPENDIX


def build_skillsbench_combined_review_prompt(base_prompt: str) -> str:
    return base_prompt + SKILLSBENCH_BATCH_COMBINED_REVIEW_APPENDIX


def host_expand_path(p: str | Path) -> Path:
    """Expand ``~`` without ``Path.expanduser``.

    AppWorld's SafetyGuard process-wide-patches ``pathlib.Path.expanduser``.
    ``os.path.expanduser`` is not in that denylist, so host-side path
    resolution (hot-pool persist, experiment dirs, HERMES_HOME) stays usable
    even if a guard leaks between tasks.
    """
    return Path(os.path.expanduser(str(p))).resolve()


def _expand(p: str | Path) -> Path:
    return host_expand_path(p)


def _ensure_hermes_on_path(hermes_root: Path) -> None:
    root = hermes_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"HERMES_AGENT_ROOT is not a directory: {root}")
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)


def discover_task_ids(tasks_dir: Path) -> List[str]:
    if not tasks_dir.is_dir():
        raise SystemExit(f"Tasks directory not found: {tasks_dir}")
    ids: List[str] = []
    for child in sorted(tasks_dir.iterdir()):
        if child.is_dir() and (child / "instruction.md").is_file():
            ids.append(child.name)
    return ids


def build_user_message(
    prompt_tasks_base: str,
    task_id: str,
    *,
    evaluate_after_run: bool = True,
) -> str:
    base = prompt_tasks_base.rstrip("/")
    msg = (
        f"complete task in {base}/{task_id}, following instruction.md and verify the result."
    )
    if evaluate_after_run:
        msg += SKILLSBENCH_HOST_GRADING_META_APPENDIX
    return msg


def build_host_eval_feedback_message(evaluation: Dict[str, Any]) -> str:
    """Follow-up user turn after host verification failed."""
    ev = evaluation or {}
    passed = ev.get("tests_passed", 0)
    total = ev.get("tests_total", 0)
    failed = ev.get("tests_failed", 0)
    lines = [
        "**Host verification failed.** The driver ran the host pytest verifier "
        "on staged task outputs. Do not claim success until this passes.",
        f"Score: {passed}/{total} tests passed"
        + (f" ({failed} failed)" if failed else "")
        + ".",
    ]
    err = ev.get("error")
    if err:
        lines.append(f"Verifier error: {err}")
    cases = ev.get("test_cases") or []
    failed_cases = [
        c for c in cases if str(c.get("outcome", "")).lower() not in ("passed", "skipped")
    ]
    if failed_cases:
        sample = failed_cases[:8]
        lines.append(
            "Failed cases: "
            + "; ".join(str(c.get("nodeid") or c.get("name") or c) for c in sample)
            + (" …" if len(failed_cases) > len(sample) else "")
        )
    stderr = (ev.get("stderr_tail") or "").strip()
    if stderr:
        lines.append(f"Verifier stderr (tail):\n{stderr[-1200:]}")
    stdout = (ev.get("stdout_tail") or "").strip()
    if stdout and not failed_cases:
        lines.append(f"Verifier stdout (tail):\n{stdout[-1200:]}")
    lines.append(
        "Fix deliverables under the task directory (paths the tests expect), "
        "then reply when ready for another host verification attempt."
    )
    return "\n\n".join(lines)


def json_safe(obj: Any) -> Any:
    """Round-trip through JSON with string fallback for non-serializable values."""

    def _default(o: Any) -> Any:
        return str(o)

    return json.loads(json.dumps(obj, default=_default))


def resolve_model_id(model: Optional[str] = None) -> str:
    """Resolve the model id for AIAgent.

    ``AIAgent(model="")`` keeps an empty string and will POST ``\"model\": \"\"``
    to the API. The interactive CLI fills ``model.default`` from config before
    constructing the agent; this driver must do the same when ``--model`` is
    omitted.
    """
    if isinstance(model, str) and model.strip():
        return model.strip()
    try:
        from hermes_cli.config import load_config

        cfg = load_config().get("model") or {}
        if isinstance(cfg, dict):
            return str(cfg.get("default") or cfg.get("model") or "").strip()
        if isinstance(cfg, str):
            return cfg.strip()
    except Exception:
        pass
    return ""


def resolve_provider_name_for_model(
    model: str,
    *,
    provider: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Pick which ``providers.<name>`` to use without rewriting ``model.provider``.

    Precedence:
    1. Explicit ``provider`` (CLI ``--provider``)
    2. Unique match of ``model`` in ``providers.*.models`` or ``default_model``
    3. ``model.provider`` from config (unchanged default)

    If the model is catalogued under multiple providers and the default is not
    among them, raise so the caller can require ``--provider``.
    """
    if isinstance(provider, str) and provider.strip():
        return provider.strip()

    cfg = config
    if cfg is None:
        try:
            from hermes_cli.config import load_config

            cfg = load_config()
        except Exception:
            cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}

    cfg_model = cfg.get("model") or {}
    default_provider: Optional[str] = None
    if isinstance(cfg_model, dict):
        raw = cfg_model.get("provider")
        if isinstance(raw, str) and raw.strip():
            default_provider = raw.strip()

    model_id = (model or "").strip()
    providers = cfg.get("providers") or {}
    if not model_id or not isinstance(providers, dict):
        return default_provider

    matches: List[str] = []
    seen: Set[str] = set()
    for name, pcfg in providers.items():
        if not isinstance(pcfg, dict):
            continue
        hit = False
        models = pcfg.get("models") or {}
        if isinstance(models, dict) and model_id in models:
            hit = True
        elif str(pcfg.get("default_model") or "").strip() == model_id:
            hit = True
        if not hit:
            continue
        key = str(name)
        if key in seen:
            continue
        seen.add(key)
        matches.append(key)

    if len(matches) == 1:
        return matches[0]
    if default_provider and default_provider in matches:
        return default_provider
    if len(matches) > 1:
        raise RuntimeError(
            f"Model {model_id!r} is listed under multiple providers {matches}; "
            "pass --provider <name> to disambiguate (leaves model.provider unchanged)."
        )
    return default_provider


def resolve_agent_runtime(
    *,
    model: str,
    provider: Optional[str] = None,
    config_hermes_home: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve provider/base_url/api_key the same way the interactive CLI does.

    ``AIAgent(model=...)`` alone is not enough for custom/local endpoints: the CLI
    passes ``provider``, ``base_url``, and ``api_key`` from
    ``resolve_runtime_provider()``. Without that, local vLLM (``qwen-local``) and
    ``--experiment-dir`` isolated ``HERMES_HOME`` trees often fail with
    ``No LLM provider configured``.

    ``provider`` (or a unique ``providers.*`` catalog match for ``model``) overrides
    ``model.provider`` for this process only — the config file is not rewritten.

    When ``config_hermes_home`` is set, temporarily point ``HERMES_HOME`` there so
    resolution still reads the user's real ``~/.hermes`` even after the driver
    isolates the experiment home for skills/memory.
    """
    from hermes_cli.runtime_provider import resolve_runtime_provider

    prev_home = os.environ.get("HERMES_HOME")
    if config_hermes_home is not None:
        os.environ["HERMES_HOME"] = str(host_expand_path(config_hermes_home))

    try:
        cfg: Dict[str, Any] = {}
        try:
            from hermes_cli.config import load_config

            loaded = load_config()
            if isinstance(loaded, dict):
                cfg = loaded
        except Exception:
            cfg = {}

        requested = resolve_provider_name_for_model(
            model, provider=provider, config=cfg
        )

        runtime = resolve_runtime_provider(
            requested=requested,
            target_model=model,
        )
    finally:
        if config_hermes_home is not None:
            if prev_home is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = prev_home

    if not isinstance(runtime, dict):
        raise RuntimeError(
            "Failed to resolve LLM runtime. Run `hermes model` / `hermes setup`, "
            "or set model.provider + providers.<name>.base_url in ~/.hermes/config.yaml."
        )
    if not (runtime.get("base_url") or runtime.get("api_key")):
        raise RuntimeError(
            "No LLM provider configured. Run `hermes model` to select a provider, "
            "or set providers.<name>.base_url for a local endpoint "
            "(e.g. http://127.0.0.1:8090/v1)."
        )
    return runtime


def apply_hot_pool_cli_overrides(
    *,
    hot_pool: Optional[bool],
    hot_pool_persist: Optional[str],
) -> None:
    """Set env vars so AIAgent honors CLI hot-pool on/off before import."""
    for key in ("HERMES_HOT_POOL_ENABLED", "HERMES_HOT_POOL_PERSIST", "HERMES_HOT_POOL_PATH"):
        os.environ.pop(key, None)
    if hot_pool is False:
        os.environ["HERMES_HOT_POOL_ENABLED"] = "0"
        return
    if hot_pool is True:
        os.environ["HERMES_HOT_POOL_ENABLED"] = "1"
    if hot_pool_persist:
        os.environ["HERMES_HOT_POOL_PERSIST"] = "1"
        os.environ["HERMES_HOT_POOL_PATH"] = str(host_expand_path(hot_pool_persist))


def run_one_task(
    *,
    task_id: str,
    hermes_root: Path,
    skillsbench_root: Path,
    prompt_tasks_base: str,
    model: str,
    skill_nudge_interval: Optional[int],
    memory_nudge_interval: Optional[int],
    max_iterations: int,
    quiet_mode: bool,
    skip_context_files: bool,
    skip_memory: bool,
    save_trajectories: bool,
    hot_pool: Optional[bool] = None,
    hot_pool_persist: Optional[str] = None,
    amem: bool = False,
    amem_persist: Optional[str] = None,
    amem_k: int = 5,
    amem_sync_every: int = 5,
    amem_freeze: bool = False,
    dc: bool = False,
    dc_persist: Optional[str] = None,
    dc_mode: str = "cu",
    dc_k: int = 3,
    dc_sync_every: int = 5,
    dc_freeze: bool = False,
    wait_background_review: bool = True,
    background_review_timeout: Optional[float] = 180.0,
    batch_review_prompt: bool = True,
    pass_k_turns: Optional[List[int]] = None,
    eval_timeout_sec: float = 600.0,
    evaluate_after_run: bool = False,
    host_verification_retries: int = 1,
    runtime: Optional[Dict[str, Any]] = None,
    config_hermes_home: Optional[Path] = None,
    benchmark_id: str = "skillsbench",
    run_schema: str = "skillsbench.hermes_run.v1",
    platform: str = "skillsbench-batch",
) -> Dict[str, Any]:
    apply_hot_pool_cli_overrides(
        hot_pool=hot_pool,
        hot_pool_persist=hot_pool_persist,
    )
    _ensure_hermes_on_path(hermes_root)
    from run_agent import AIAgent  # type: ignore  # after sys.path

    resolved_model = resolve_model_id(model)
    if not resolved_model:
        raise SystemExit(
            "No model id resolved. Pass --model <id>, or set model.default via "
            "`hermes model` / ~/.hermes/config.yaml."
        )

    agent_runtime = runtime or resolve_agent_runtime(
        model=resolved_model,
        config_hermes_home=config_hermes_home,
    )
    apply_amem_cli_overrides(
        enabled=bool(amem),
        persist_dir=amem_persist,
        k=amem_k,
        sync_every=amem_sync_every,
        freeze=bool(amem_freeze),
        llm_model=resolved_model,
        api_key=agent_runtime.get("api_key"),
        base_url=agent_runtime.get("base_url"),
    )
    apply_dc_cli_overrides(
        enabled=bool(dc),
        persist_dir=dc_persist,
        mode=dc_mode,
        k=dc_k,
        sync_every=dc_sync_every,
        freeze=bool(dc_freeze),
        llm_model=resolved_model,
        api_key=agent_runtime.get("api_key"),
        base_url=agent_runtime.get("base_url"),
    )
    agent = AIAgent(
        model=resolved_model,
        api_key=agent_runtime.get("api_key"),
        base_url=agent_runtime.get("base_url"),
        provider=agent_runtime.get("provider"),
        api_mode=agent_runtime.get("api_mode"),
        quiet_mode=quiet_mode,
        max_iterations=max_iterations,
        skip_context_files=skip_context_files,
        skip_memory=True if amem or dc else skip_memory,
        save_trajectories=save_trajectories,
        platform=platform,
    )
    pool = getattr(agent, "_hot_skill_pool", None)
    if pool is not None and hasattr(pool, "clear_exposed_tips"):
        pool.clear_exposed_tips()
    try:
        from skillsbench_console_window import get_active_window

        win = get_active_window()
        if win is not None:
            agent._print_fn = win.print_fn
    except Exception:
        pass
    if skill_nudge_interval is not None:
        agent._skill_nudge_interval = int(skill_nudge_interval)
    if memory_nudge_interval is not None:
        agent._memory_nudge_interval = int(memory_nudge_interval)
    if batch_review_prompt:
        agent.skill_review_prompt_override = build_skillsbench_skill_review_prompt(
            AIAgent._SKILL_REVIEW_PROMPT,
        )
        agent.combined_review_prompt_override = build_skillsbench_combined_review_prompt(
            AIAgent._COMBINED_REVIEW_PROMPT,
        )

    user_message = build_user_message(
        prompt_tasks_base,
        task_id,
        evaluate_after_run=evaluate_after_run,
    )
    hermes_task_id = f"{benchmark_id}-{task_id}"

    pass_tracker = None
    if pass_k_turns:
        _eval_mod = _load_evaluate_module()
        pass_tracker = _eval_mod.PassAtTurnTracker(
            task_id=task_id,
            skillsbench_root=skillsbench_root,
            pass_k_turns=pass_k_turns,
            eval_timeout_sec=eval_timeout_sec,
            evaluate_fn=_eval_mod.evaluate_task_host,
        )
        pass_tracker.attach(agent)

    t0 = time.perf_counter()
    pass_at_turn: Dict[str, Dict[str, Any]] = {}
    host_verification_attempts: List[Dict[str, Any]] = []
    try:
        result = agent.run_conversation(
            user_message=user_message,
            task_id=hermes_task_id,
        )
        if pass_tracker is not None and isinstance(result, dict):
            final_calls = int(result.get("api_calls") or 0)
            pass_at_turn = pass_tracker.finalize(final_calls)

        if evaluate_after_run:
            retries_left = max(0, int(host_verification_retries or 0))
            while True:
                evaluation = evaluate_skillsbench_task(
                    task_id=task_id,
                    skillsbench_root=skillsbench_root,
                    eval_timeout_sec=eval_timeout_sec,
                )
                attempt = {
                    "attempt": len(host_verification_attempts) + 1,
                    "evaluation": evaluation,
                }
                if isinstance(result, dict):
                    attempt["api_calls"] = result.get("api_calls")
                    attempt["completed"] = result.get("completed")
                host_verification_attempts.append(attempt)
                if evaluation.get("task_success"):
                    break
                if retries_left <= 0 or not isinstance(result, dict):
                    break
                if result.get("interrupted"):
                    break
                feedback = build_host_eval_feedback_message(evaluation)
                result = agent.run_conversation(
                    user_message=feedback,
                    conversation_history=result.get("messages") or [],
                    task_id=hermes_task_id,
                )
                if pass_tracker is not None and isinstance(result, dict):
                    final_calls = int(result.get("api_calls") or 0)
                    pass_at_turn = pass_tracker.finalize(final_calls)
                retries_left -= 1
    finally:
        if pass_tracker is not None:
            pass_tracker.detach(agent)
    background_review: Dict[str, Any] = {"spawned": False, "completed": True, "timeout": False}
    if wait_background_review:
        background_review = agent.wait_for_background_review(
            timeout=background_review_timeout,
        )
        refreshed = agent.export_hot_pool_telemetry()
        if refreshed is not None and isinstance(result, dict):
            result["hot_pool_telemetry"] = refreshed
    elapsed = time.perf_counter() - t0

    evaluation: Optional[Dict[str, Any]] = None
    if evaluate_after_run:
        evaluation = (
            host_verification_attempts[-1]["evaluation"]
            if host_verification_attempts
            else evaluate_skillsbench_task(
                task_id=task_id,
                skillsbench_root=skillsbench_root,
                eval_timeout_sec=eval_timeout_sec,
            )
        )
        try:
            apply_benchmark_hot_pool_outcome_feedback(
                agent,
                evaluation=evaluation,
                run_result=result if isinstance(result, dict) else None,
                duration_sec=elapsed,
                benchmark=benchmark_id,
                wait_background_review=False,
            )
            refreshed = agent.export_hot_pool_telemetry()
            if refreshed is not None and isinstance(result, dict):
                result["hot_pool_telemetry"] = refreshed
        except Exception:
            pass

    envelope: Dict[str, Any] = {
        "schema": run_schema,
        "benchmark": benchmark_id,
        "ts_start_iso": None,  # filled by caller
        "ts_end_iso": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(elapsed, 6),
        "skillsbench_task_id": task_id,
        "task_id": task_id,
        "skillsbench_root": str(skillsbench_root),
        "hermes_root": str(hermes_root),
        "hermes_conversation_task_id": hermes_task_id,
        "user_message": user_message,
        "nudge_intervals": {
            "skill": getattr(agent, "_skill_nudge_interval", None),
            "memory": getattr(agent, "_memory_nudge_interval", None),
        },
        "batch_review_prompt": batch_review_prompt,
        "hot_pool_enabled": hot_pool,
        "hot_pool_persist": hot_pool_persist,
        "amem_enabled": bool(amem),
        "amem_persist": amem_persist,
        "dc_enabled": bool(dc),
        "dc_persist": dc_persist,
        "dc_mode": dc_mode if dc else None,
        "model": resolved_model,
        "background_review": background_review,
        "run_conversation_result": result,
    }
    if evaluation is not None:
        envelope["evaluation"] = evaluation
    if host_verification_attempts:
        envelope["host_verification_attempts"] = host_verification_attempts
    if pass_tracker is not None and pass_at_turn:
        envelope["pass_k_turns"] = list(pass_k_turns or [])
        envelope["pass_at_turn"] = pass_at_turn
    if isinstance(result, dict) and result.get("hot_pool_telemetry"):
        envelope["hot_pool_telemetry"] = result["hot_pool_telemetry"]
    if amem:
        envelope["amem_telemetry"] = amem_telemetry_from_agent(agent)
    if dc:
        envelope["dc_telemetry"] = dc_telemetry_from_agent(agent)
    return envelope


def _load_evaluate_module():
    _eval_path = _SCRIPT.parent / "evaluate_skillsbench_task.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("evaluate_skillsbench_task", _eval_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def evaluate_skillsbench_task(
    *,
    task_id: str,
    skillsbench_root: Path,
    eval_timeout_sec: float = 600.0,
) -> Dict[str, Any]:
    """Host-side SkillsBench pytest verifier (see evaluate_skillsbench_task.py)."""
    mod = _load_evaluate_module()
    return mod.evaluate_task_host(
        task_id=task_id,
        skillsbench_root=skillsbench_root,
        timeout_sec=eval_timeout_sec,
    )


def finalize_envelope(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Attach compact ``metrics`` block derived from evaluation + run result."""
    envelope["metrics"] = build_envelope_metrics(envelope)
    return envelope


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def format_run_summary(envelope: Dict[str, Any]) -> str:
    """One-line stdout summary for --print-summary."""
    res = envelope.get("run_conversation_result") or {}
    br = envelope.get("background_review") or {}
    tel = envelope.get("hot_pool_telemetry") or res.get("hot_pool_telemetry") or {}
    sm = tel.get("skill_manage") if isinstance(tel, dict) else {}
    skill_manage_sync = sm.get("sync") if isinstance(sm, dict) else None
    bg_actions = br.get("actions") or []
    bg_action_count = len(bg_actions)
    bg_action_preview = bg_actions[0] if bg_actions else None
    ev = envelope.get("evaluation") or {}
    metrics = envelope.get("metrics") or {}
    final_m = metrics.get("final") if isinstance(metrics, dict) else {}
    if not isinstance(final_m, dict):
        final_m = {}
    pass_at_turn = envelope.get("pass_at_turn") or {}
    metrics_pat = metrics.get("pass_at_turn") if isinstance(metrics, dict) else {}
    pass_k_bits = []
    turn_keys = sorted(
        set(pass_at_turn) | set(metrics_pat or {}),
        key=lambda x: int(x),
    )
    for key in turn_keys:
        block = pass_at_turn.get(key) or (metrics_pat or {}).get(key) or {}
        pev = (block or {}).get("evaluation") or block or {}
        pass_k_bits.append(
            f"pass@{key}={pev.get('task_success')!r}"
            f"({pev.get('tests_passed')}/{pev.get('tests_total')})"
        )
    parts = [
        f"duration_sec={envelope.get('duration_sec')!r}",
        f"eval_success={ev.get('task_success')!r}",
        f"reward={ev.get('reward')!r}",
        f"tests_passed={ev.get('tests_passed')!r}",
        f"tests_total={ev.get('tests_total')!r}",
        *pass_k_bits,
        f"api_calls={res.get('api_calls')!r}",
        f"input_tokens={res.get('input_tokens')!r}",
        f"output_tokens={res.get('output_tokens')!r}",
        f"total_tokens={res.get('total_tokens')!r}",
        f"cache_read_tokens={res.get('cache_read_tokens')!r}",
        f"reasoning_tokens={res.get('reasoning_tokens')!r}",
        f"estimated_cost_usd={res.get('estimated_cost_usd')!r}",
        f"tool_rounds={final_m.get('tool_rounds')!r}",
        f"completed={res.get('completed')!r}",
        f"interrupted={res.get('interrupted')!r}",
        f"bg_review_spawned={br.get('spawned')!r}",
        f"bg_review_completed={br.get('completed')!r}",
        f"bg_review_timeout={br.get('timeout')!r}",
        f"bg_review_actions={bg_action_count!r}",
        f"bg_review_action_preview={bg_action_preview!r}",
        f"skill_manage_sync={skill_manage_sync!r}",
    ]
    return " ".join(parts)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Hermes AIAgent on SkillsBench tasks (host pytest eval).",
    )
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "--task",
        metavar="TASK_ID",
        help="Single task directory name under tasks/ (must contain instruction.md).",
    )
    g.add_argument(
        "--all",
        action="store_true",
        help="Run every task under tasks/ that has instruction.md, in sorted order.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print discovered task ids and exit (no Hermes import).",
    )
    parser.add_argument(
        "--skillsbench-root",
        type=str,
        default=str(_bundled_skillsbench_root()),
        help=(
            "SkillsBench checkout root (default: this repo's "
            "benchmark/skillsbench)."
        ),
    )
    parser.add_argument(
        "--hermes-root",
        type=str,
        default=str(DEFAULT_HERMES_ROOT),
        help=f"Hermes agent repo root (default: {DEFAULT_HERMES_ROOT}).",
    )
    parser.add_argument(
        "--prompt-tasks-base",
        type=str,
        default=DEFAULT_PROMPT_TASKS_BASE,
        help=(
            "Path prefix embedded in the user prompt before /<task_id> "
            "(default: resolved benchmark/skillsbench/tasks in this checkout)."
        ),
    )
    parser.add_argument(
        "--skill-nudge-interval",
        type=int,
        default=None,
        metavar="N",
        help="Set agent._skill_nudge_interval after construction (omit to leave Hermes config).",
    )
    parser.add_argument(
        "--memory-nudge-interval",
        type=int,
        default=None,
        metavar="N",
        help="Set agent._memory_nudge_interval after construction (omit to leave Hermes config).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help=(
            "Model id sent in chat/completions (e.g. Qwen/Qwen3.6-27B). "
            "Empty uses model.default from ~/.hermes/config.yaml "
            "(after `hermes model`). Required if config has no default."
        ),
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="",
        help=(
            "Named providers.<name> from ~/.hermes/config.yaml (e.g. qwen-31). "
            "Overrides model.provider for this run only. If omitted, auto-selects "
            "when --model uniquely matches a providers.*.models / default_model entry."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=90,
        help="Max tool-calling iterations per conversation (Hermes default 90).",
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Suppress Hermes agent tool/stream console output. "
            "Default is --no-quiet so you can see tool calls and progress. "
            "Use --quiet for silent overnight batches."
        ),
    )
    parser.add_argument(
        "--console-window",
        type=int,
        default=12,
        metavar="N",
        help=(
            "Fixed N-line viewport for agent activity (overwrites in-place). "
            "Batch progress stays permanent; errors go to stderr. "
            "0 = full scrolling output. Default: 12."
        ),
    )
    parser.add_argument(
        "--skip-context-files",
        action="store_true",
        help="Skip AGENTS.md-style context injection (Hermes).",
    )
    parser.add_argument(
        "--skip-memory",
        action="store_true",
        help="Disable persistent memory (Hermes).",
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Append Hermes trajectories to trajectory_samples.jsonl.",
    )
    parser.add_argument(
        "--hot-pool-persist",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Persist hot skill key points across tasks to PATH (JSON). "
            "Each task continues the pool like a new user turn. "
            "Requires hot pool to be enabled (default from config, or --hot-pool)."
        ),
    )
    hot_pool_group = parser.add_mutually_exclusive_group()
    hot_pool_group.add_argument(
        "--hot-pool",
        dest="hot_pool",
        action="store_const",
        const=True,
        help="Enable hot skill pool for this run (overrides config.yaml).",
    )
    hot_pool_group.add_argument(
        "--no-hot-pool",
        dest="hot_pool",
        action="store_const",
        const=False,
        help="Disable hot skill pool for this run (overrides config.yaml).",
    )
    parser.set_defaults(hot_pool=None)
    add_amem_cli_flags(parser)
    add_dc_cli_flags(parser)
    parser.add_argument(
        "--no-wait-background-review",
        action="store_true",
        help=(
            "Return immediately after run_conversation without waiting for the "
            "end-of-turn skill/memory review thread (daemon; often killed on exit)."
        ),
    )
    parser.add_argument(
        "--background-review-timeout",
        type=float,
        default=180.0,
        metavar="SEC",
        help=(
            "Max seconds to wait for background review after each task "
            "(default: 180). Ignored with --no-wait-background-review."
        ),
    )
    parser.add_argument(
        "--no-batch-review-prompt",
        action="store_true",
        help=(
            "Use the default Hermes skill/memory review prompts instead of the "
            "SkillsBench batch appendix (class-level skill saves, abstract "
            "pitfalls, host-grading skill encouragement)."
        ),
    )
    parser.add_argument(
        "--evaluate-after-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run host pytest verifier after each agent run (default: on).",
    )
    parser.add_argument(
        "--host-verification-retries",
        type=int,
        default=1,
        metavar="N",
        help=(
            "When --evaluate-after-run and host verification fails, continue the "
            "conversation with verifier feedback up to N extra times (default: 1)."
        ),
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip the agent; evaluate existing task outputs on disk.",
    )
    parser.add_argument(
        "--eval-timeout-sec",
        type=float,
        default=600.0,
        metavar="SEC",
        help="Max seconds for host pytest evaluation per task (default: 600).",
    )
    parser.add_argument(
        "--pass-k",
        type=str,
        default=None,
        metavar="TURNS",
        help=(
            "Comma-separated agent conversation turns for pass@k metrics "
            "(e.g. 1,5,10,70). Evaluates task outputs during the run at the end "
            "of each listed turn — no reruns or early termination."
        ),
    )
    parser.add_argument(
        "--log-jsonl",
        type=str,
        default=None,
        help="Append one JSON line per task run (full envelope + run_conversation result).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip tasks that already have a finished (non-error) row in --log-jsonl. "
            "Re-runs interrupted/failed tasks. Requires --log-jsonl."
        ),
    )
    parser.add_argument(
        "--resume-success-only",
        action="store_true",
        help="With --resume: only skip tasks whose last row has task_success=True.",
    )
    parser.add_argument(
        "--log-json-pretty",
        type=str,
        default=None,
        help="Also write a pretty-printed JSON file for the last run only (single-task mode).",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a short stdout summary per task (tokens, cost, duration).",
    )
    parser.add_argument(
        "--print-batch-summary",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "After --all, print aggregate pass@k / token / cost metrics across tasks "
            "(default: on when --all)."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="With --all, exit on first exception instead of continuing.",
    )
    parser.add_argument(
        "--start-task-index",
        type=int,
        default=0,
        metavar="X",
        help=(
            "With --all only: skip the first X tasks (sorted task order, 0-based) "
            "and run the remainder. Default 0 runs from the start. "
            "Combine with --end-task-index for a half-open range [start, end)."
        ),
    )
    parser.add_argument(
        "--end-task-index",
        type=int,
        default=None,
        metavar="Y",
        help=(
            "With --all only: exclusive end index in the same sorted order as "
            "--list-tasks (Python slice: task_ids[start:end]). Omit to run through "
            "the last task. Example: --start-task-index 0 --end-task-index 10 runs "
            "the first 10 tasks."
        ),
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "JSON split from make_skillsbench_splits.py. "
            "With --all, run only tasks in the selected --split-part instead of the "
            "full task list."
        ),
    )
    parser.add_argument(
        "--split-part",
        choices=("train", "test", "all"),
        default="train",
        help="Which partition to run when --split-file is set (default: train).",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Isolated experiment workspace for *task* files. Copies selected tasks "
            "under DIR/<dataset>/tasks/, points --skillsbench-root / "
            "--prompt-tasks-base there, and defaults --hot-pool-persist to "
            "DIR/hot_pool.json. Does NOT rewrite HERMES_HOME by default — Hermes "
            "keeps using ~/.hermes (default skills). Opt into Hermes isolation with "
            "--isolate-hermes-home."
        ),
    )
    parser.add_argument(
        "--isolate-hermes-home",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "With --experiment-dir: also set HERMES_HOME=DIR/hermes_home "
            "(default: off — keep ~/.hermes). When enabled, DIR/hermes_home/skills "
            "is a writable copy of the repo skills/ tree (not ~/.hermes/skills), "
            "while config.yaml / .env / SOUL.md are symlinked to the real "
            "~/.hermes files. Re-seed skills with --reset-task-workspaces."
        ),
    )
    parser.add_argument(
        "--reset-task-workspaces",
        action="store_true",
        help=(
            "With --experiment-dir: re-copy task definition files and wipe prior "
            "agent outputs in the experiment task dirs before running. With "
            "--isolate-hermes-home, also re-seed DIR/hermes_home/skills from "
            "the repo skills/ tree (not ~/.hermes/skills)."
        ),
    )

    args = parser.parse_args(argv)
    apply_amem_argparse_policy(parser, args)
    apply_dc_argparse_policy(parser, args)
    preset = benchmark_presets()["skillsbench"]
    if args.reset_task_workspaces and not args.experiment_dir:
        parser.error("--reset-task-workspaces requires --experiment-dir")
    source_skillsbench_root = _expand(args.skillsbench_root)
    hermes_root = _expand(args.hermes_root)
    skillsbench_root = source_skillsbench_root
    tasks_dir = skillsbench_root / "tasks"

    task_ids = discover_task_ids(tasks_dir)
    split_file_path: Optional[Path] = None
    if args.split_file and not args.all:
        parser.error("--split-file requires --all (or use --task for a single id).")
    if args.split_file:
        split_path = Path(args.split_file).expanduser()
        split_file_path = split_path
        if not split_path.is_file():
            parser.error(f"Split file not found: {split_path}")
        _split_mod_path = _SCRIPT.parent / "make_skillsbench_splits.py"
        import importlib.util

        spec = importlib.util.spec_from_file_location("make_skillsbench_splits", _split_mod_path)
        split_mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(split_mod)
        split_ids = split_mod.resolve_split_task_ids(split_path, args.split_part)
        unknown = [tid for tid in split_ids if tid not in task_ids]
        if unknown:
            preview = ", ".join(unknown[:5]) + (" ..." if len(unknown) > 5 else "")
            parser.error(f"Split file lists unknown task ids: {preview}")
        task_ids = split_ids

    if args.evaluate_only and args.all:
        parser.error("--evaluate-only requires --task (not --all).")
    if not args.list_tasks and not args.all and not args.task:
        parser.error("Specify --task TASK_ID, --all, or --list-tasks.")
    if args.list_tasks:
        for tid in task_ids:
            print(tid)
        return 0

    if args.start_task_index < 0:
        parser.error("--start-task-index must be >= 0.")
    if args.start_task_index != 0 and not args.all:
        parser.error("--start-task-index is only supported with --all.")
    if args.end_task_index is not None:
        if not args.all:
            parser.error("--end-task-index is only supported with --all.")
        if args.end_task_index < 0:
            parser.error("--end-task-index must be >= 0.")
        if args.end_task_index > len(task_ids):
            parser.error(
                f"--end-task-index ({args.end_task_index}) exceeds task count ({len(task_ids)})."
            )
        if args.end_task_index < args.start_task_index:
            parser.error(
                f"--end-task-index ({args.end_task_index}) must be >= --start-task-index ({args.start_task_index})."
            )

    if args.all:
        if not task_ids:
            print("No tasks found (instruction.md under tasks/).", file=sys.stderr)
            return 1
        start = args.start_task_index
        end = args.end_task_index
        run_ids = task_ids[start:] if end is None else task_ids[start:end]
        if not run_ids:
            print(
                f"No tasks in range [--start-task-index={start}, --end-task-index="
                f"{end if end is not None else 'end'}); discovered {len(task_ids)} task(s).",
                file=sys.stderr,
            )
            return 1
    else:
        assert args.task
        if args.task not in task_ids:
            print(
                f"Unknown task {args.task!r}. Known: {', '.join(task_ids[:10])}"
                + (" ..." if len(task_ids) > 10 else ""),
                file=sys.stderr,
            )
            return 1
        run_ids = [args.task]

    prompt_tasks_base = args.prompt_tasks_base
    if prompt_tasks_base == DEFAULT_PROMPT_TASKS_BASE:
        prompt_tasks_base = str((skillsbench_root / "tasks").resolve())
    experiment_dir: Optional[Path] = None
    # Resolve LLM credentials against the real ~/.hermes *before* any
    # --experiment-dir isolation rewrites HERMES_HOME (empty experiment homes
    # have no model.provider / providers.* and would fail AIAgent init).
    from hermes_constants import get_hermes_home

    config_hermes_home = Path(get_hermes_home()).resolve()
    resolved_model_for_runtime = resolve_model_id(args.model)
    if not args.list_tasks and not args.evaluate_only:
        if not resolved_model_for_runtime:
            parser.error(
                "No model id resolved. Pass --model <id>, or set model.default via "
                "`hermes model` / ~/.hermes/config.yaml."
            )
        try:
            shared_runtime = resolve_agent_runtime(
                model=resolved_model_for_runtime,
                provider=(args.provider or None),
                config_hermes_home=config_hermes_home,
            )
        except RuntimeError as exc:
            parser.error(str(exc))
        print(
            f"[provider] {shared_runtime.get('provider')!r} "
            f"model={resolved_model_for_runtime!r} "
            f"base_url={shared_runtime.get('base_url')!r}",
            flush=True,
        )
    else:
        shared_runtime = None

    if args.experiment_dir:
        from skillsbench_experiment_workspace import (
            experiment_hot_pool_path,
            prepare_experiment_workspace,
        )

        experiment_dir = Path(args.experiment_dir).expanduser().resolve()
        print(
            f"[experiment] preparing workspace → {experiment_dir} "
            f"({len(run_ids)} tasks, reset_outputs={bool(args.reset_task_workspaces)})",
            flush=True,
        )
        manifest = prepare_experiment_workspace(
            experiment_dir=experiment_dir,
            source_skillsbench_root=source_skillsbench_root,
            task_ids=run_ids,
            reset_outputs=bool(args.reset_task_workspaces),
            isolate_hermes_home=bool(args.isolate_hermes_home),
            apply_hermes_home_env=bool(args.isolate_hermes_home),
            source_hermes_home=config_hermes_home,
            hermes_root=hermes_root,
            dataset_dirname=str(preset["dataset_dirname"]),
        )
        skillsbench_root = Path(manifest["skillsbench_root"])
        prompt_tasks_base = str(manifest["prompt_tasks_base"])
        if args.amem or args.dc:
            args.hot_pool = False
        elif not args.hot_pool_persist and args.hot_pool is not False:
            args.hot_pool_persist = str(experiment_hot_pool_path(experiment_dir))
            if args.hot_pool is None:
                # Auto-persist implies enabling the pool for this experiment.
                args.hot_pool = True
            print(
                f"[experiment] hot_pool_persist → {args.hot_pool_persist}",
                flush=True,
            )
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
                f"shared {','.join(linked) or 'none'} → "
                f"{manifest.get('source_hermes_home')})",
                flush=True,
            )
        else:
            print(
                "[experiment] HERMES_HOME unchanged (default Hermes skills from ~/.hermes); "
                "pass --isolate-hermes-home to sandbox skills/memory "
                "(seeded copy of repo skills/, not ~/.hermes/skills)",
                flush=True,
            )
        print(
            f"[experiment] skillsbench_root → {skillsbench_root}",
            flush=True,
        )

    if args.hot_pool is False and args.hot_pool_persist:
        parser.error("--hot-pool-persist cannot be used with --no-hot-pool.")
    args.amem_persist = resolve_amem_persist(
        enabled=bool(args.amem),
        persist=args.amem_persist,
        experiment_dir=experiment_dir,
    )
    if args.amem:
        print(
            f"[amem] persist → {args.amem_persist} k={args.amem_k} "
            f"sync_every={'frozen' if args.amem_freeze else (args.amem_sync_every if args.amem_sync_every > 0 else 'episode')}"
            f"{' readonly' if args.amem_freeze else ''}",
            flush=True,
        )
    args.dc_persist = resolve_dc_persist(
        enabled=bool(args.dc),
        persist=args.dc_persist,
        experiment_dir=experiment_dir,
    )
    if args.dc:
        print(
            f"[dc] persist → {args.dc_persist} mode={args.dc_mode} k={args.dc_k} "
            f"sync_every={'frozen' if args.dc_freeze else (args.dc_sync_every if args.dc_sync_every > 0 else 'episode')}"
            f"{' readonly' if args.dc_freeze else ''}",
            flush=True,
        )

    if args.resume_success_only:
        args.resume = True
    if args.resume and not args.log_jsonl:
        parser.error("--resume / --resume-success-only requires --log-jsonl")

    pass_k_turns: Optional[List[int]] = None
    if args.pass_k:
        try:
            pass_k_turns = _load_evaluate_module().parse_pass_k_values(args.pass_k)
        except ValueError as exc:
            parser.error(str(exc))

    log_path = Path(args.log_jsonl).expanduser() if args.log_jsonl else None
    pretty_path = Path(args.log_json_pretty).expanduser() if args.log_json_pretty else None
    if pretty_path and len(run_ids) != 1:
        print("--log-json-pretty requires exactly one task (no --all).", file=sys.stderr)
        return 1

    print_batch_summary = args.print_batch_summary
    if print_batch_summary is None:
        print_batch_summary = bool(args.all)

    from skillsbench_batch_progress import (
        BatchProgress,
        filter_pending_tasks,
        load_completed_task_ids,
    )
    from skillsbench_console_window import ConsoleWindow, set_active_window

    original_run_ids = list(run_ids)
    skipped: List[str] = []
    if args.resume and log_path:
        done = load_completed_task_ids(
            log_path,
            success_only=args.resume_success_only,
        )
        run_ids, skipped = filter_pending_tasks(run_ids, done)

    console_win: Optional[ConsoleWindow] = None
    if getattr(args, "console_window", 0) and args.console_window > 0:
        console_win = ConsoleWindow(args.console_window)
        set_active_window(console_win)
        print(
            f"[console] activity window={args.console_window} lines "
            f"(batch progress permanent; agent output overwrites; errors→stderr)",
            flush=True,
        )

    any_failed = False
    batch_envelopes: List[Dict[str, Any]] = []
    prog = BatchProgress(
        phase="hermes",
        total=len(original_run_ids),
        already_done=len(skipped),
    )
    if len(original_run_ids) > 1 or args.resume:
        prog.banner(
            resume=bool(args.resume),
            skipped=len(skipped),
            pending=len(run_ids),
        )
        if skipped:
            preview = ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else "")
            print(f"[hermes] skipping completed: {preview}", flush=True)

    if not run_ids:
        print("[hermes] nothing pending (all tasks already completed).", flush=True)
        prog.finish()
        return 0

    for batch_index, tid in enumerate(run_ids):
        ts_start = datetime.now(timezone.utc).isoformat()
        t0 = prog.task_start(batch_index + 1, tid)
        try:
            if args.evaluate_only:
                evaluation = evaluate_skillsbench_task(
                    task_id=tid,
                    skillsbench_root=skillsbench_root,
                    eval_timeout_sec=args.eval_timeout_sec,
                )
                elapsed = time.perf_counter() - t0
                envelope = {
                    "schema": preset["schema"],
                    "benchmark": "skillsbench",
                    "ts_start_iso": ts_start,
                    "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                    "duration_sec": round(elapsed, 6),
                    "skillsbench_task_id": tid,
                    "task_id": tid,
                    "skillsbench_root": str(skillsbench_root),
                    "hermes_root": str(hermes_root),
                    "evaluate_only": True,
                    "evaluation": evaluation,
                }
            else:
                envelope = run_one_task(
                    task_id=tid,
                    hermes_root=hermes_root,
                    skillsbench_root=skillsbench_root,
                    prompt_tasks_base=prompt_tasks_base,
                    model=args.model,
                    skill_nudge_interval=args.skill_nudge_interval,
                    memory_nudge_interval=args.memory_nudge_interval,
                    max_iterations=args.max_iterations,
                    quiet_mode=args.quiet,
                    skip_context_files=args.skip_context_files,
                    skip_memory=args.skip_memory,
                    save_trajectories=args.save_trajectories,
                    hot_pool=args.hot_pool,
                    hot_pool_persist=args.hot_pool_persist,
                    amem=bool(args.amem),
                    amem_persist=args.amem_persist,
                    amem_k=int(args.amem_k),
                    amem_sync_every=int(args.amem_sync_every),
                    amem_freeze=bool(args.amem_freeze),
                    dc=bool(args.dc),
                    dc_persist=args.dc_persist,
                    dc_mode=str(args.dc_mode),
                    dc_k=int(args.dc_k),
                    dc_sync_every=int(args.dc_sync_every),
                    dc_freeze=bool(args.dc_freeze),
                    wait_background_review=not args.no_wait_background_review,
                    background_review_timeout=args.background_review_timeout,
                    batch_review_prompt=not args.no_batch_review_prompt,
                    pass_k_turns=pass_k_turns,
                    eval_timeout_sec=args.eval_timeout_sec,
                    evaluate_after_run=bool(args.evaluate_after_run),
                    host_verification_retries=int(args.host_verification_retries),
                    runtime=shared_runtime,
                    config_hermes_home=config_hermes_home,
                    benchmark_id="skillsbench",
                    run_schema=str(preset["schema"]),
                    platform=str(preset["platform"]),
                )
                envelope["ts_start_iso"] = ts_start
                # Evaluation (+ hot-pool outcome feedback) runs inside run_one_task
                # when evaluate_after_run is set.

            if split_file_path is not None:
                envelope["split_file"] = str(split_file_path)
                envelope["split_part"] = args.split_part
            if experiment_dir is not None:
                envelope["experiment_dir"] = str(experiment_dir)
                envelope["skillsbench_root"] = str(skillsbench_root)
            if len(original_run_ids) > 1:
                envelope["batch"] = {
                    "task_index": batch_index + len(skipped),
                    "task_count": len(original_run_ids),
                }

            envelope = finalize_envelope(envelope)
            batch_envelopes.append(envelope)

            serializable = json_safe(envelope)
            if log_path:
                append_jsonl(log_path, serializable)

            if pretty_path:
                pretty_path.write_text(
                    json.dumps(serializable, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            ev = envelope.get("evaluation") or {}
            ok = bool(ev.get("task_success")) if "task_success" in ev else True
            detail = ""
            if args.print_summary:
                detail = format_run_summary(envelope)
                print(detail, flush=True)
            elif "task_success" in ev:
                detail = f"success={ev.get('task_success')} reward={ev.get('reward')}"
            prog.task_end(batch_index + 1, tid, t0=t0, ok=ok, detail=detail[:240])
        except Exception as e:
            any_failed = True
            err = {
                "schema": preset["error_schema"],
                "benchmark": "skillsbench",
                "ts_start_iso": ts_start,
                "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                "skillsbench_task_id": tid,
                "task_id": tid,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }
            if log_path:
                append_jsonl(log_path, err)
            prog.task_end(batch_index + 1, tid, t0=t0, ok=False, detail=repr(e)[:200])
            print(f"ERROR task={tid}: {e}", file=sys.stderr)
            traceback.print_exc()
            if args.stop_on_error:
                batch_stats = prog.finish()
                if log_path:
                    append_jsonl(
                        log_path,
                        json_safe(
                            {
                                "schema": preset["batch_schema"],
                                "benchmark": "skillsbench",
                                "method": "hermes",
                                "phase": "hermes",
                                "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                                "timing": batch_stats,
                            }
                        ),
                    )
                return 1

    batch_stats = prog.finish()
    if log_path and (len(original_run_ids) > 1 or args.resume):
        append_jsonl(
            log_path,
            json_safe(
                {
                    "schema": preset["batch_schema"],
                    "benchmark": "skillsbench",
                    "method": "hermes",
                    "phase": "hermes",
                    "ts_end_iso": datetime.now(timezone.utc).isoformat(),
                    "split_file": str(split_file_path) if split_file_path else None,
                    "split_part": args.split_part if split_file_path else None,
                    "timing": batch_stats,
                }
            ),
        )

    if print_batch_summary and len(batch_envelopes) > 1:
        pass_k_for_agg = pass_k_turns or [1]
        summary = aggregate_records(
            batch_envelopes,
            pass_k_values=pass_k_for_agg,
            split_part=args.split_part if split_file_path else None,
        )
        print(
            f"\n=== {preset['label']} batch summary ({len(batch_envelopes)} tasks this run; "
            f"{len(skipped)} skipped via --resume) ===",
            flush=True,
        )
        print(format_summary_text(summary), flush=True)

    if console_win is not None:
        console_win.close()
        set_active_window(None)

    if any_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
