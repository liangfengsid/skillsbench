"""Harbor agent for CoEvoSkills evolve materialize (export /app after solve).

Frozen eval uses ``HermesHarborAgent`` so it matches the Hermes Terminal-Bench
driver. This subclass only adds a sandbox ``/app`` export for the surrogate.
"""

from __future__ import annotations

import os
from pathlib import Path

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from benchmark.harbor_adapter.hermes_agent import HermesHarborAgent


class CoEvoHarborAgent(HermesHarborAgent):
    """Same host-side Hermes loop as the official TB agent, plus /app export."""

    @staticmethod
    def name() -> str:
        return "coevoskills"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await super().run(instruction, environment, context)
        export = (os.getenv("HERMES_COEVO_APP_EXPORT") or "").strip()
        if not export:
            return
        dest = Path(export)
        dest.mkdir(parents=True, exist_ok=True)
        try:
            await environment.download_dir("/app", dest)
        except Exception as exc:
            (dest / "_export_error.txt").write_text(str(exc), encoding="utf-8")
