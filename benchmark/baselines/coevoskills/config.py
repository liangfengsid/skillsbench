"""Defaults aligned with CoEvoSkills paper Table A1 (Alg. 1 hyperparameters)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CoEvoConfig:
    """Hyperparameters from Alg. 1 / Table A1."""

    # N = max ground-truth oracle interventions (paper: K=5)
    max_oracle_iters: int = 5
    # M = max surrogate retries across the loop (paper: M=15)
    max_surrogate_iters: int = 15
    # Break if generator context usage exceeds this fraction (paper: β=0.7)
    context_cap: float = 0.7
    # Hermes tool-calling iterations per generator / verifier turn
    max_agent_iterations: int = 60
    # Host pytest timeout for oracle and surrogate local pytest
    eval_timeout_sec: float = 600.0
    # Skip AGENTS.md injection during evolution (recommended for benchmarks)
    skip_context_files: bool = True
    skip_memory: bool = True
    model: str = ""
    hermes_root: Optional[Path] = None
    skillsbench_root: Optional[Path] = None
    work_root: Optional[Path] = None
    quiet_mode: bool = False  # False → show tool/stream activity on console

    extra: dict = field(default_factory=dict)
