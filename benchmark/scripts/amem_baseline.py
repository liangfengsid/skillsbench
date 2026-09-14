"""A-Mem paper baseline helpers for Hermes benchmark drivers.

A-Mem (Xu et al., arXiv:2502.12110) is an alternative to the hot-skill pool:
same Hermes agent and skill tools, no hot key-points. Retrieved notes inject
via the memory-provider prefetch fence.

Enable with ``--amem`` on SkillsBench / ALFWorld / AppWorld drivers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


def experiment_amem_path(experiment_dir: Path) -> Path:
    return experiment_dir.resolve() / "amem"


def apply_amem_cli_overrides(
    *,
    enabled: bool,
    persist_dir: Optional[str] = None,
    k: Optional[int] = None,
    sync_every: Optional[int] = None,
    freeze: bool = False,
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """Set env vars so AIAgent loads the A-Mem provider (even with skip_memory)."""
    for key in (
        "HERMES_AMEM_ENABLED",
        "HERMES_AMEM_PATH",
        "HERMES_AMEM_K",
        "HERMES_AMEM_SYNC_EVERY",
        "HERMES_AMEM_READONLY",
        "HERMES_AMEM_LLM_MODEL",
        "HERMES_AMEM_API_KEY",
        "HERMES_AMEM_API_BASE",
        "HERMES_MEMORY_PROVIDER",
    ):
        os.environ.pop(key, None)
    if not enabled:
        return
    os.environ["HERMES_AMEM_ENABLED"] = "1"
    os.environ["HERMES_MEMORY_PROVIDER"] = "amem"
    if persist_dir:
        path = Path(persist_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.environ["HERMES_AMEM_PATH"] = str(path)
    if k is not None:
        os.environ["HERMES_AMEM_K"] = str(int(k))
    if freeze:
        os.environ["HERMES_AMEM_READONLY"] = "1"
    elif sync_every is not None:
        # 0 = one note per episode; N = flush every N buffered turns.
        os.environ["HERMES_AMEM_SYNC_EVERY"] = str(max(0, int(sync_every)))
    if llm_model:
        os.environ["HERMES_AMEM_LLM_MODEL"] = str(llm_model)
    if api_key:
        os.environ["HERMES_AMEM_API_KEY"] = str(api_key)
        os.environ.setdefault("OPENAI_API_KEY", str(api_key))
    if base_url:
        os.environ["HERMES_AMEM_API_BASE"] = str(base_url)
        # litellm / OpenAI SDK used by A-Mem's LLMController
        os.environ.setdefault("OPENAI_BASE_URL", str(base_url))
        os.environ.setdefault("OPENAI_API_BASE", str(base_url))


def amem_telemetry_from_agent(agent: Any) -> Dict[str, Any]:
    mgr = getattr(agent, "_memory_manager", None)
    providers = getattr(mgr, "providers", None) or []
    for provider in providers:
        if getattr(provider, "name", "") == "amem" and hasattr(provider, "export_telemetry"):
            try:
                return dict(provider.export_telemetry())
            except Exception:
                return {}
    return {}


def add_amem_cli_flags(parser) -> None:
    """``--amem`` / ``--amem-persist`` / ``--amem-k`` / ``--amem-freeze``."""
    parser.add_argument(
        "--amem",
        action="store_true",
        help=(
            "A-Mem baseline (Xu et al. 2025): same Hermes agent and skill tools, "
            "hot-skill off. Retrieved notes inject via <memory-context>. "
            "Requires pip install -e '.[amem]'."
        ),
    )
    parser.add_argument(
        "--amem-persist",
        default=None,
        metavar="DIR",
        help="A-Mem store directory (default: DIR/amem with --experiment-dir).",
    )
    parser.add_argument(
        "--amem-k",
        type=int,
        default=5,
        metavar="K",
        help="How many A-Mem notes to inject per turn (default: 5).",
    )
    parser.add_argument(
        "--amem-sync-every",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Flush buffered turns into A-Mem every N sync_turn calls "
            "(default: 5). Use 0 for one note per episode/task; 1 for "
            "legacy per-turn writes (expensive: embed + LLM evolve each step). "
            "Ignored with --amem-freeze."
        ),
    )
    parser.add_argument(
        "--amem-freeze",
        action="store_true",
        help=(
            "Frozen A-Mem eval: prefetch/retrieve only — no note writes "
            "(no embed/evolve). Sets HERMES_AMEM_READONLY=1. Use on held-out "
            "splits with --amem-persist pointing at a train snapshot."
        ),
    )


def apply_amem_argparse_policy(parser, args) -> None:
    """``--amem`` cannot share a run with ``--hot-pool`` or ``--dc``."""
    if getattr(args, "amem_freeze", False) and not getattr(args, "amem", False):
        parser.error("--amem-freeze requires --amem")
    if not getattr(args, "amem", False):
        return
    if getattr(args, "dc", False):
        parser.error(
            "--amem cannot be combined with --dc "
            "(pick one paper memory baseline per run)."
        )
    if getattr(args, "hot_pool", None) is True:
        parser.error(
            "--amem cannot be combined with --hot-pool "
            "(A-Mem is the no-hot-skill baseline)."
        )
    args.hot_pool = False
    if hasattr(args, "skip_memory"):
        args.skip_memory = True


def resolve_amem_persist(
    *,
    enabled: bool,
    persist: Optional[str],
    experiment_dir: Optional[Path],
) -> Optional[str]:
    if not enabled:
        return None
    if persist:
        return str(Path(persist).expanduser().resolve())
    if experiment_dir is not None:
        return str(experiment_amem_path(experiment_dir))
    return str((Path.cwd() / "benchmark" / "runs" / "amem_store").resolve())

