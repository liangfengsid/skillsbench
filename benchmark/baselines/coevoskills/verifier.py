"""Surrogate Verifier π_θ^V — info-isolated session; owns test suite V."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import hermes_backend, prompts, skill_io
from .config import CoEvoConfig


_JSON_RE = re.compile(r"\{[^{}]*\"r_hat\"[^{}]*\}", re.DOTALL)


def _parse_verifier_json(text: str) -> Dict[str, Any]:
    if not text:
        return {"r_hat": 0.0, "passed": False, "diagnostics": "(empty verifier response)"}
    m = _JSON_RE.search(text)
    if m:
        try:
            data = json.loads(m.group(0))
            return {
                "r_hat": float(data.get("r_hat", 0.0)),
                "passed": bool(data.get("passed", False)),
                "diagnostics": str(data.get("diagnostics") or ""),
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Fallback: look for last JSON object in text
    try:
        start = text.rfind("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(text[start : end + 1])
            return {
                "r_hat": float(data.get("r_hat", 0.0)),
                "passed": bool(data.get("passed", False)),
                "diagnostics": str(data.get("diagnostics") or text[-2000:]),
            }
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return {"r_hat": 0.0, "passed": False, "diagnostics": text[-4000:]}


def run_surrogate_pytest(tests_dir: Path, artifacts_dir: Path, timeout_sec: float) -> Dict[str, Any]:
    """Execute V locally if test_surrogate.py exists."""
    test_file = tests_dir / "test_surrogate.py"
    if not test_file.is_file():
        return {"ran": False, "passed": False, "reward": 0.0, "stdout": "", "stderr": "no test file"}
    import os

    env = {**os.environ, "COEVO_ARTIFACTS_DIR": str(artifacts_dir)}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            cwd=str(tests_dir),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ran": True, "passed": False, "reward": 0.0, "stdout": "", "stderr": "timeout"}
    out = proc.stdout or ""
    ok = proc.returncode == 0
    # crude reward from summary
    reward = 1.0 if ok else 0.0
    return {
        "ran": True,
        "passed": ok,
        "reward": reward,
        "exit_code": proc.returncode,
        "stdout": out[-4000:],
        "stderr": (proc.stderr or "")[-2000:],
    }


class SurrogateVerifier:
    def __init__(
        self,
        *,
        cfg: CoEvoConfig,
        hermes_root: Path,
        instruction: str,
        paths: Dict[str, Path],
    ):
        self.cfg = cfg
        self.hermes_root = hermes_root
        self.instruction = instruction
        self.paths = paths
        # Fresh agent / history — information isolation from generator
        self.history: List[Dict[str, Any]] = []
        self._agent = None
        self.last_diagnostics: str = ""

    def _system(self) -> str:
        return prompts.verifier_system_prompt(
            instruction=self.instruction,
            artifacts_dir=str(self.paths["artifacts"]),
            tests_dir=str(self.paths["surrogate_tests"]),
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
                session_id=f"coevo-ver-{self.paths['base'].name}",
                print_fn=print_fn,
            )
        return self._agent

    def evaluate(
        self,
        *,
        iteration: int,
        escalate: bool = False,
    ) -> Dict[str, Any]:
        agent = self._ensure_agent()
        user = prompts.verifier_turn_prompt(
            iteration=iteration,
            escalate=escalate,
            artifact_listing=skill_io.list_artifacts(self.paths["artifacts"]),
            prior_diagnostics=self.last_diagnostics,
        )
        result = hermes_backend.run_turn(
            agent,
            user_message=user,
            system_message=self._system(),
            conversation_history=list(self.history) if self.history else None,
        )
        messages = result.get("messages") or []
        if messages:
            self.history = messages[-40:]
        text = result.get("final_response") or ""
        parsed = _parse_verifier_json(text)

        pytest_result = run_surrogate_pytest(
            self.paths["surrogate_tests"],
            self.paths["artifacts"],
            self.cfg.eval_timeout_sec,
        )
        # Prefer executed suite when available
        if pytest_result.get("ran"):
            parsed["r_hat"] = float(pytest_result.get("reward") or 0.0)
            parsed["passed"] = bool(pytest_result.get("passed"))
            if not parsed["passed"]:
                diag = parsed.get("diagnostics") or ""
                parsed["diagnostics"] = (
                    diag
                    + "\n\n### pytest\n"
                    + (pytest_result.get("stdout") or "")
                    + "\n"
                    + (pytest_result.get("stderr") or "")
                ).strip()

        self.last_diagnostics = str(parsed.get("diagnostics") or "")
        return {
            "r_hat": float(parsed.get("r_hat") or 0.0),
            "passed": bool(parsed.get("passed")),
            "diagnostics": self.last_diagnostics,
            "pytest": pytest_result,
            "raw_response": text[-2000:],
        }
