"""Skill Generator π_θ — persistent context C = (I, S_meta)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from . import hermes_backend, prompts, skill_io
from .config import CoEvoConfig


class SkillGenerator:
    def __init__(
        self,
        *,
        cfg: CoEvoConfig,
        hermes_root: Path,
        instruction: str,
        paths: Dict[str, Path],
        env_hint: str = "",
    ):
        self.cfg = cfg
        self.hermes_root = hermes_root
        self.instruction = instruction
        self.paths = paths
        self.env_hint = env_hint
        self.history: List[Dict[str, Any]] = []
        self._agent = None

    def _system(self) -> str:
        return prompts.generator_system_prompt(
            instruction=self.instruction,
            skills_dir=str(self.paths["skills"]),
            artifacts_dir=str(self.paths["artifacts"]),
        )

    def _ensure_agent(self):
        if self._agent is None:
            print_fn = None
            try:
                from skillsbench_console_window import get_active_window

                win = get_active_window()
                if win is not None:
                    print_fn = win.print_fn
            except Exception:
                pass
            self._agent = hermes_backend.make_agent(
                hermes_root=self.hermes_root,
                model=self.cfg.model,
                max_iterations=self.cfg.max_agent_iterations,
                skip_context_files=self.cfg.skip_context_files,
                skip_memory=self.cfg.skip_memory,
                quiet_mode=self.cfg.quiet_mode,
                session_id=f"coevo-gen-{self.paths['base'].name}",
                print_fn=print_fn,
            )
        return self._agent

    def propose(
        self,
        *,
        iteration: int,
        feedback: str,
    ) -> Dict[str, Any]:
        """One generator step: update skills + produce artifacts (Φ)."""
        agent = self._ensure_agent()
        meta = skill_io.skills_meta_context(self.paths["skills"])
        listing = skill_io.skills_listing_text(self.paths["skills"])
        user = prompts.generator_turn_prompt(
            feedback=feedback,
            iteration=iteration,
            skills_listing=listing,
        )
        # Persist S_meta in the user turn (paper: C carries skill content across rounds)
        user = (
            f"{user}\n\n"
            f"### Persistent skill contents (S_meta)\n{meta}\n\n"
            f"### Environment location\n{self.paths['env']}\n"
            f"{self.env_hint}"
        )
        result = hermes_backend.run_turn(
            agent,
            user_message=user,
            system_message=self._system(),
            conversation_history=list(self.history) if self.history else None,
        )
        # Keep a compact rolling history for the same generator session
        messages = result.get("messages") or []
        if messages:
            self.history = messages[-40:]
        return {
            "final_response": result.get("final_response") or "",
            "skills_listing": skill_io.skills_listing_text(self.paths["skills"]),
            "artifacts_listing": skill_io.list_artifacts(self.paths["artifacts"]),
        }
