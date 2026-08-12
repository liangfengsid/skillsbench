#!/usr/bin/env python3
"""
Host-side SkillsBench verifier for Hermes batch runs.

Maps container paths (``/root/...``, ``/app/...``, ``/tests/...``, ``/logs/...``)
to a staged task tree, runs ``tests/test_outputs.py`` via pytest, and returns
macro (task) + micro (test-case) metrics.

Official container eval (``bench eval``) is not required; this module is for
repeatable Hermes JSONL logging. Set ``eval_mode: host`` on envelopes.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_SKIP_TOP_LEVEL = frozenset(
    {"solution", "tests", "instruction.md", "task.toml", "environment", ".git"}
)
_PYTEST_SUMMARY_RE = re.compile(
    r"(?P<passed>\d+) passed"
    r"(?:, (?P<failed>\d+) failed)?"
    r"(?:, (?P<skipped>\d+) skipped)?"
    r"(?:, (?P<errors>\d+) errors)?"
)

# Container workspace mounts used by SkillsBench task tests.
_WORKSPACE_PREFIXES: Tuple[str, ...] = ("/root", "/app")
_TESTS_PREFIX = "/tests"
_LOGS_PREFIX = "/logs"


def task_dir_for(skillsbench_root: Path, task_id: str) -> Path:
    return skillsbench_root.resolve() / "tasks" / task_id


def _remap_abs_prefix(text: str, container_prefix: str, host_path: str) -> str:
    """
    Rewrite absolute container path prefixes to a host staging path.

    Handles quoted strings (``"/app/x"``, ``'/app'``), ``Path(...)``, bare
    paths in shell snippets (``cd /app/workspace``), and ``sys.path.insert``.
    Does not match longer pathnames (``/apple`` stays intact when prefix is
    ``/app``).
    """
    host = str(Path(host_path).resolve())
    pref = container_prefix.rstrip("/")
    esc = re.escape(pref)

    # "/app/foo" or '/app/foo' (and Path("/app/foo"))
    text = re.sub(rf'(["\']){esc}/', rf"\1{host}/", text)
    # Exact mount: "/app" or '/app'
    text = re.sub(rf'(["\']){esc}\1', rf"\1{host}\1", text)
    # Bare paths in shell commands: cd /app/workspace && ...
    text = re.sub(rf"(?<![A-Za-z0-9_]){esc}/", f"{host}/", text)
    text = re.sub(rf"(?<![A-Za-z0-9_]){esc}(?![A-Za-z0-9_/])", host, text)
    text = text.replace(f"sys.path.insert(0, '{pref}')", f"sys.path.insert(0, '{host}')")
    text = text.replace(f'sys.path.insert(0, "{pref}")', f'sys.path.insert(0, "{host}")')
    return text


def adapt_container_paths(
    content: str,
    host_root: Path,
    *,
    tests_dir: Optional[Path] = None,
    logs_dir: Optional[Path] = None,
    workspace_prefixes: Sequence[str] = _WORKSPACE_PREFIXES,
) -> str:
    """
    Rewrite container absolute paths in test sources for host staging.

    - ``/root`` and ``/app`` → ``host_root`` (agent workspace)
    - ``/tests`` → ``tests_dir`` (verifier inputs next to pytest)
    - ``/logs`` → ``logs_dir`` (verifier logs; created by the host runner)
    """
    text = content
    for prefix in workspace_prefixes:
        text = _remap_abs_prefix(text, prefix, host_root)
    if tests_dir is not None:
        text = _remap_abs_prefix(text, _TESTS_PREFIX, tests_dir)
    if logs_dir is not None:
        text = _remap_abs_prefix(text, _LOGS_PREFIX, logs_dir)
    return text


def stage_task_for_host_eval(task_dir: Path, stage_dir: Path) -> Path:
    """
    Build ``<stage_dir>/root`` with environment inputs + agent-written task outputs.

    Agent artifacts typically live directly under ``tasks/<id>/``; inputs under
    ``environment/``.
    """
    host_root = stage_dir / "root"
    host_root.mkdir(parents=True, exist_ok=True)

    env_dir = task_dir / "environment"
    if env_dir.is_dir():
        for src in env_dir.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(env_dir)
            dest = host_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    for entry in task_dir.iterdir():
        if entry.name in _SKIP_TOP_LEVEL or entry.name.startswith("."):
            continue
        dest = host_root / entry.name
        if entry.is_file():
            shutil.copy2(entry, dest)
        elif entry.is_dir():
            shutil.copytree(entry, dest, dirs_exist_ok=True)

    return host_root


def stage_tests_for_host_eval(
    task_dir: Path,
    stage_dir: Path,
    host_root: Path,
) -> Tuple[Path, Path]:
    """
    Copy ``tasks/<id>/tests/`` into the stage, rewriting container paths in
    ``*.py`` files. Also create ``logs/verifier`` for tests that write there.

    Returns ``(tests_dir, logs_dir)``.
    """
    tests_dir = stage_dir / "tests"
    logs_dir = stage_dir / "logs"
    (logs_dir / "verifier").mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    src_tests = task_dir / "tests"
    if not src_tests.is_dir():
        return tests_dir, logs_dir

    for src in src_tests.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_tests)
        dest = tests_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".py":
            raw = src.read_text(encoding="utf-8")
            dest.write_text(
                adapt_container_paths(
                    raw,
                    host_root,
                    tests_dir=tests_dir,
                    logs_dir=logs_dir,
                ),
                encoding="utf-8",
            )
        else:
            shutil.copy2(src, dest)

    return tests_dir, logs_dir


def maybe_run_simulation(host_root: Path, timeout_sec: float = 120.0) -> Dict[str, Any]:
    """Run ``simulation.py`` when present (mirrors many ``tests/test.sh`` scripts)."""
    sim = host_root / "simulation.py"
    tuning = host_root / "tuning_results.yaml"
    if not sim.is_file():
        return {"ran": False}
    if not tuning.is_file():
        return {"ran": False, "reason": "tuning_results.yaml missing"}
    proc = subprocess.run(
        [sys.executable, str(sim)],
        cwd=str(host_root),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return {
        "ran": True,
        "exit_code": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def _parse_pytest_stdout(stdout: str) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    passed = failed = skipped = 0
    cases: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        m = re.match(r"^(?P<path>\S+::\S+)\s+(?P<outcome>PASSED|FAILED|SKIPPED|ERROR)", line)
        if m:
            outcome = m.group("outcome").lower()
            cases.append({"nodeid": m.group("path"), "outcome": outcome})
            if outcome == "passed":
                passed += 1
            elif outcome == "skipped":
                skipped += 1
            else:
                failed += 1
    if cases:
        return passed, failed, skipped, cases

    for line in reversed(stdout.splitlines()):
        sm = _PYTEST_SUMMARY_RE.search(line)
        if sm:
            passed = int(sm.group("passed") or 0)
            failed = int(sm.group("failed") or 0)
            skipped = int(sm.group("skipped") or 0)
            break
    return passed, failed, skipped, cases


def _parse_ctrf(ctrf_path: Path) -> Optional[List[Dict[str, Any]]]:
    if not ctrf_path.is_file():
        return None
    try:
        data = json.loads(ctrf_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    tests = (data.get("results") or {}).get("tests")
    if not isinstance(tests, list):
        return None
    cases: List[Dict[str, Any]] = []
    for t in tests:
        if not isinstance(t, dict):
            continue
        name = t.get("name") or t.get("nodeid") or ""
        status = str(t.get("status") or "").lower()
        outcome = "passed" if status in ("passed", "pass") else (
            "skipped" if status in ("skipped", "skip") else "failed"
        )
        cases.append({"nodeid": name, "outcome": outcome})
    return cases or None


def evaluate_task_host(
    *,
    task_id: str,
    skillsbench_root: Path,
    timeout_sec: float = 600.0,
) -> Dict[str, Any]:
    """
    Run SkillsBench pytest verifier on the host for one task directory.

    Returns an ``evaluation`` dict suitable for JSONL envelopes.
    """
    task_dir = task_dir_for(skillsbench_root, task_id)
    test_src = task_dir / "tests" / "test_outputs.py"
    if not task_dir.is_dir():
        return _eval_error(task_id, f"Task directory not found: {task_dir}")
    if not test_src.is_file():
        return _eval_error(task_id, f"Missing tests/test_outputs.py under {task_dir}")

    with tempfile.TemporaryDirectory(prefix=f"skillsbench-eval-{task_id}-") as tmp:
        stage = Path(tmp)
        host_root = stage_task_for_host_eval(task_dir, stage)
        sim_info = maybe_run_simulation(host_root)
        tests_dir, logs_dir = stage_tests_for_host_eval(task_dir, stage, host_root)

        adapted_test = tests_dir / "test_outputs.py"
        if not adapted_test.is_file():
            return _eval_error(
                task_id, f"Failed to stage tests/test_outputs.py for {task_id}"
            )

        ctrf_path = stage / "ctrf.json"
        # Prefer package-style discovery so sibling conftest.py is loaded.
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_outputs.py",
            "-v",
            "--tb=short",
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(stage),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return _eval_error(
                task_id, f"pytest timed out after {timeout_sec}s", eval_mode="host"
            )
        except FileNotFoundError:
            return _eval_error(
                task_id,
                "pytest not available in current interpreter",
                eval_mode="host",
            )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        ctrf_cases = _parse_ctrf(ctrf_path)
        if ctrf_cases is not None:
            cases = ctrf_cases
            passed = sum(1 for c in cases if c.get("outcome") == "passed")
            failed = sum(1 for c in cases if c.get("outcome") == "failed")
            skipped = sum(1 for c in cases if c.get("outcome") == "skipped")
        else:
            passed, failed, skipped, cases = _parse_pytest_stdout(stdout)

        total = passed + failed + skipped
        task_success = proc.returncode == 0 and failed == 0 and total > 0
        reward = round(passed / total, 6) if total else 0.0
        if task_success:
            reward = 1.0

        return {
            "eval_mode": "host",
            "task_id": task_id,
            "task_success": task_success,
            "reward": reward,
            "tests_passed": passed,
            "tests_failed": failed,
            "tests_skipped": skipped,
            "tests_total": total,
            "test_cases": cases,
            "pytest_exit_code": proc.returncode,
            "simulation": sim_info,
            "host_root": str(host_root),
            "tests_dir": str(tests_dir),
            "logs_dir": str(logs_dir),
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-2000:],
        }


def _eval_error(task_id: str, message: str, eval_mode: str = "host") -> Dict[str, Any]:
    return {
        "eval_mode": eval_mode,
        "task_id": task_id,
        "task_success": False,
        "reward": 0.0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0,
        "tests_total": 0,
        "test_cases": [],
        "pytest_exit_code": None,
        "error": message,
    }


def parse_pass_k_values(raw: str) -> List[int]:
    """Parse comma-separated conversation turn indices for pass@k metrics."""
    vals: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k < 1:
            raise ValueError(f"pass-k turn values must be >= 1, got {k}")
        vals.append(k)
    if not vals:
        raise ValueError("pass-k list is empty")
    return sorted(set(vals))


def snapshot_token_usage(agent: Any) -> Dict[str, Any]:
    """Read cumulative token counters from AIAgent at a conversation turn."""
    from skillsbench_metrics import snapshot_agent_usage

    return snapshot_agent_usage(agent)


class PassAtTurnTracker:
    """
    Evaluate task outputs at conversation turn *k* during a single agent run.

    A **turn** is one completed agent API iteration (model response + any tool
    execution). ``step_callback`` fires at the start of API call ``k+1``, so the
    on-disk task state reflects the end of turn ``k``. If the conversation ends
    on turn ``k`` without starting call ``k+1``, ``finalize()`` records turn ``k``.
    """

    def __init__(
        self,
        *,
        task_id: str,
        skillsbench_root: Path,
        pass_k_turns: List[int],
        eval_timeout_sec: float = 600.0,
        evaluate_fn=None,
    ) -> None:
        self.task_id = task_id
        self.skillsbench_root = skillsbench_root
        self.pass_k_turns = set(pass_k_turns)
        self.eval_timeout_sec = eval_timeout_sec
        self.evaluate_fn = evaluate_fn or evaluate_task_host
        self._agent: Any = None
        self._original_step_callback = None
        self._recorded: Dict[int, Dict[str, Any]] = {}

    def attach(self, agent: Any) -> None:
        self._agent = agent
        self._original_step_callback = getattr(agent, "step_callback", None)
        agent.step_callback = self._on_step

    def detach(self, agent: Any) -> None:
        agent.step_callback = self._original_step_callback
        self._agent = None

    def _on_step(self, api_call_count: int, prev_tools: Any) -> None:
        if self._original_step_callback is not None:
            try:
                self._original_step_callback(api_call_count, prev_tools)
            except Exception:
                pass
        completed_turn = api_call_count - 1
        if completed_turn in self.pass_k_turns:
            self._record_turn(completed_turn)

    def _record_turn(self, turn: int) -> None:
        if turn in self._recorded or self._agent is None:
            return
        evaluation = self.evaluate_fn(
            task_id=self.task_id,
            skillsbench_root=self.skillsbench_root,
            timeout_sec=self.eval_timeout_sec,
        )
        self._recorded[turn] = {
            "turn": turn,
            "evaluation": evaluation,
            **snapshot_token_usage(self._agent),
        }

    def finalize(self, final_api_calls: int) -> Dict[str, Dict[str, Any]]:
        """Record any requested turns not yet captured (conversation ended on turn k)."""
        for turn in sorted(self.pass_k_turns):
            if turn <= int(final_api_calls or 0) and turn not in self._recorded:
                self._record_turn(turn)
        return {str(turn): self._recorded[turn] for turn in sorted(self._recorded)}
