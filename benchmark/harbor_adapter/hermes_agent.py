"""Harbor ``BaseAgent`` that runs Hermes ``AIAgent`` against the trial sandbox.

Harbor starts the task container (Docker / Modal / …). This agent binds that
environment to Hermes ``TERMINAL_ENV=harbor`` so ``terminal`` / file tools exec
inside ``/app``, then Harbor runs the official ``tests/test.sh`` verifier.

Loaded as::

    --agent benchmark.harbor_adapter.hermes_agent:HermesHarborAgent
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "benchmark" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


class HermesHarborAgent(BaseAgent):
    """External Hermes agent: host-side LLM loop, Harbor sandbox for tools."""

    SUPPORTS_ATIF = False

    def __init__(self, *args, max_iterations: int | None = None, **kwargs):
        raw = max_iterations or os.getenv("HERMES_HARBOR_MAX_ITERATIONS") or 90
        self._max_iterations = int(raw)
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        # Result metadata only. Harbor also ships a built-in installed ``hermes``
        # agent (CLI inside the sandbox). This adapter is the host-side
        # ``AIAgent``; the driver always passes this class via import path.
        return "hermes"

    def version(self) -> str | None:
        try:
            from importlib.metadata import version

            return version("hermes-agent")
        except Exception:
            return None

    async def setup(self, environment: BaseEnvironment) -> None:
        # Hermes stays on the host; tools exec into ``environment``.
        return None

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._run_hermes_sync(instruction, environment, loop),
        )
        self._fill_context(context, result)

    def _run_hermes_sync(
        self,
        instruction: str,
        environment: BaseEnvironment,
        loop: asyncio.AbstractEventLoop,
    ) -> Dict[str, Any]:
        from tools.environments.harbor_bridge import (
            bind_harbor_session,
            unbind_harbor_session,
        )
        from tools.terminal_tool import cleanup_vm

        bind_harbor_session(environment=environment, loop=loop)
        prev_env = os.environ.get("TERMINAL_ENV")
        prev_cwd = os.environ.get("TERMINAL_CWD")
        os.environ["TERMINAL_ENV"] = "harbor"
        os.environ["TERMINAL_CWD"] = "/app"
        task_id = "harbor-trial"
        try:
            return self._invoke_agent(instruction, task_id)
        finally:
            try:
                cleanup_vm(task_id)
            except Exception:
                pass
            unbind_harbor_session()
            if prev_env is None:
                os.environ.pop("TERMINAL_ENV", None)
            else:
                os.environ["TERMINAL_ENV"] = prev_env
            if prev_cwd is None:
                os.environ.pop("TERMINAL_CWD", None)
            else:
                os.environ["TERMINAL_CWD"] = prev_cwd

    def _invoke_agent(self, instruction: str, task_id: str) -> Dict[str, Any]:
        from run_agent import AIAgent
        from run_skillsbench_with_hermes import resolve_agent_runtime, resolve_model_id

        # Honor HERMES_HOME / HERMES_HOT_POOL_* set by the Harbor driver
        # (--isolate-hermes-home, --no-hot-pool). Skip context + memory match
        # SkillsBench isolate runs.
        try:
            from skillsbench_experiment_workspace import refresh_skills_dir_caches

            refresh_skills_dir_caches()
        except Exception:
            pass
        model = resolve_model_id(self.model_name)
        runtime = resolve_agent_runtime(model=model)
        max_iterations = int(self._max_iterations or 90)
        agent = AIAgent(
            model=model,
            api_key=runtime.get("api_key"),
            base_url=runtime.get("base_url"),
            provider=runtime.get("provider"),
            api_mode=runtime.get("api_mode"),
            quiet_mode=True,
            max_iterations=max_iterations,
            skip_context_files=True,
            skip_memory=True,
            platform="terminalbench-harbor",
        )
        user = (
            "You are in a Terminal-Bench Harbor sandbox. The working directory is /app.\n"
            "Use terminal and file tools to complete the task. Do not read /tests or "
            "/solution.\n\n"
            f"{instruction}"
        )
        prefix = (os.getenv("HERMES_HARBOR_INSTRUCTION_PREFIX") or "").strip()
        if prefix:
            user = prefix + "\n\n" + user
        return agent.run_conversation(user_message=user, task_id=task_id) or {}

    def _fill_context(self, context: AgentContext, result: Dict[str, Any]) -> None:
        tokens = result.get("tokens") if isinstance(result.get("tokens"), dict) else {}
        n_in = result.get("input_tokens")
        n_out = result.get("output_tokens")
        if n_in is None:
            n_in = tokens.get("input")
        if n_out is None:
            n_out = tokens.get("output")
        context.n_input_tokens = int(n_in) if n_in is not None else None
        context.n_output_tokens = int(n_out) if n_out is not None else None
        cache = result.get("cache_read_tokens")
        if cache is None:
            cache = (tokens.get("cache_read") if tokens else None)
        context.n_cache_tokens = int(cache) if cache is not None else None
        cost = result.get("estimated_cost_usd")
        context.cost_usd = float(cost) if cost is not None else None
        meta = dict(context.metadata or {})
        meta["api_calls"] = result.get("api_calls")
        meta["completed"] = result.get("completed")
        context.metadata = meta
