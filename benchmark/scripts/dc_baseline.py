"""Dynamic Cheatsheet paper baseline helpers for Hermes benchmark drivers.

Suzgun et al., arXiv:2504.07952. Same Hermes agent and skill tools as hot-skill /
A-Mem; the cheatsheet injects via the memory-provider prefetch fence.

Enable with ``--dc`` on SkillsBench / ALFWorld / AppWorld drivers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


_DC_MODES = ("cu", "rs", "curetr")


def experiment_dc_path(experiment_dir: Path) -> Path:
    return experiment_dir.resolve() / "dcheatsheet"


def apply_dc_cli_overrides(
    *,
    enabled: bool,
    persist_dir: Optional[str] = None,
    mode: Optional[str] = None,
    k: Optional[int] = None,
    sync_every: Optional[int] = None,
    freeze: bool = False,
    llm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """Set env vars so AIAgent loads the DC provider (even with skip_memory).

    Does **not** pop ``HERMES_MEMORY_PROVIDER`` when disabled so a prior
    ``apply_amem_cli_overrides(enabled=True)`` is not wiped. When enabled,
    this function overwrites the provider name to ``dcheatsheet``.
    """
    for key in (
        "HERMES_DC_ENABLED",
        "HERMES_DC_PATH",
        "HERMES_DC_MODE",
        "HERMES_DC_K",
        "HERMES_DC_SYNC_EVERY",
        "HERMES_DC_READONLY",
        "HERMES_DC_LLM_MODEL",
        "HERMES_DC_API_KEY",
        "HERMES_DC_API_BASE",
    ):
        os.environ.pop(key, None)
    if os.getenv("HERMES_MEMORY_PROVIDER") == "dcheatsheet":
        os.environ.pop("HERMES_MEMORY_PROVIDER", None)
    if not enabled:
        return
    os.environ["HERMES_DC_ENABLED"] = "1"
    os.environ["HERMES_MEMORY_PROVIDER"] = "dcheatsheet"
    if persist_dir:
        path = Path(persist_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.environ["HERMES_DC_PATH"] = str(path)
    if mode:
        os.environ["HERMES_DC_MODE"] = str(mode).strip().lower()
    if k is not None:
        os.environ["HERMES_DC_K"] = str(int(k))
    if freeze:
        os.environ["HERMES_DC_READONLY"] = "1"
    elif sync_every is not None:
        # 0 = one curation per episode; N = flush every N buffered turns.
        os.environ["HERMES_DC_SYNC_EVERY"] = str(max(0, int(sync_every)))
    if llm_model:
        os.environ["HERMES_DC_LLM_MODEL"] = str(llm_model)
    if api_key:
        os.environ["HERMES_DC_API_KEY"] = str(api_key)
        os.environ.setdefault("OPENAI_API_KEY", str(api_key))
    if base_url:
        os.environ["HERMES_DC_API_BASE"] = str(base_url)
        os.environ.setdefault("OPENAI_BASE_URL", str(base_url))
        os.environ.setdefault("OPENAI_API_BASE", str(base_url))


def dc_telemetry_from_agent(agent: Any) -> Dict[str, Any]:
    mgr = getattr(agent, "_memory_manager", None)
    providers = getattr(mgr, "providers", None) or []
    for provider in providers:
        if getattr(provider, "name", "") == "dcheatsheet" and hasattr(
            provider, "export_telemetry"
        ):
            try:
                return dict(provider.export_telemetry())
            except Exception:
                return {}
    return {}


def add_dc_cli_flags(parser) -> None:
    """``--dc`` / ``--dc-persist`` / ``--dc-mode`` / ``--dc-k`` / ``--dc-freeze``."""
    parser.add_argument(
        "--dc",
        action="store_true",
        help=(
            "Dynamic Cheatsheet baseline (Suzgun et al. 2025): same Hermes agent "
            "and skill tools, hot-skill off. Cheatsheet injects via <memory-context>. "
            "Incompatible with --hot-pool and --amem."
        ),
    )
    parser.add_argument(
        "--dc-persist",
        default=None,
        metavar="DIR",
        help="DC store directory (default: DIR/dcheatsheet with --experiment-dir).",
    )
    parser.add_argument(
        "--dc-mode",
        choices=_DC_MODES,
        default="cu",
        help=(
            "cu = generate then curate (DC-Cu, default); "
            "rs = retrieve+curate then generate (DC-RS); "
            "curetr = DC-Cu plus retrieved examples at prefetch."
        ),
    )
    parser.add_argument(
        "--dc-k",
        type=int,
        default=3,
        metavar="K",
        help="Retrieved prior episodes for --dc-mode rs/curetr (default: 3).",
    )
    parser.add_argument(
        "--dc-sync-every",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Flush buffered turns into the cheatsheet every N sync_turn calls "
            "(default: 5). Use 0 for one curation per episode/task; 1 for "
            "legacy per-turn curator calls (expensive). Ignored with --dc-freeze."
        ),
    )
    parser.add_argument(
        "--dc-freeze",
        action="store_true",
        help=(
            "Frozen DC eval: prefetch/inject only — no curator writes "
            "(no episode appends). Sets HERMES_DC_READONLY=1. Use on held-out "
            "splits with --dc-persist pointing at a train snapshot."
        ),
    )


def apply_dc_argparse_policy(parser, args) -> None:
    """``--dc`` cannot share a run with ``--hot-pool`` or ``--amem``."""
    if getattr(args, "dc_freeze", False) and not getattr(args, "dc", False):
        parser.error("--dc-freeze requires --dc")
    if not getattr(args, "dc", False):
        return
    if getattr(args, "amem", False):
        parser.error(
            "--dc cannot be combined with --amem "
            "(pick one paper memory baseline per run)."
        )
    if getattr(args, "hot_pool", None) is True:
        parser.error(
            "--dc cannot be combined with --hot-pool "
            "(Dynamic Cheatsheet is the no-hot-skill baseline)."
        )
    args.hot_pool = False
    if hasattr(args, "skip_memory"):
        args.skip_memory = True


def resolve_dc_persist(
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
        return str(experiment_dc_path(experiment_dir))
    return str((Path.cwd() / "benchmark" / "runs" / "dcheatsheet_store").resolve())
