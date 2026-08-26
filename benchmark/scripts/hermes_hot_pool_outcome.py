"""Hot-pool outcome attribution after benchmark task evaluation."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

from agent.hot_skills import build_hot_pool_outcome

logger = logging.getLogger(__name__)


def _hot_pool_outcome_complete_fn(agent: Any) -> Callable[[list], str]:
    """Side-channel LLM completion for per-tip outcome labels (not session history)."""

    def complete_fn(messages: list) -> str:
        sidechannel = getattr(agent, "_complete_sidechannel_text", None)
        if not callable(sidechannel):
            return ""
        return sidechannel(
            messages,
            max_tokens=512,
            reason="hot_pool_outcome_feedback",
        )

    return complete_fn


def apply_benchmark_hot_pool_outcome_feedback(
    agent: Any,
    *,
    evaluation: Optional[dict] = None,
    run_result: Optional[dict] = None,
    duration_sec: Optional[float] = None,
    benchmark: str = "",
    wait_background_review: bool = True,
    background_review_timeout: float = 60.0,
    post_wait: Optional[Callable[[], None]] = None,
) -> Optional[dict]:
    """Run LLM outcome attribution on exposed hot-pool tips after a labeled task.

    Waits for end-of-turn background review when requested so eviction judging
    does not race outcome attribution.     Wires ``complete_fn`` on the running agent's transport so benchmark drivers
    do not rely on implicit agent hooks.
    """
    pool = getattr(agent, "_hot_skill_pool", None)
    if pool is None or not pool.enabled:
        return None
    if not pool.config.get("outcome_feedback", True):
        return {"applied": False, "skipped_reason": "disabled"}

    if wait_background_review:
        waiter = getattr(agent, "wait_for_background_review", None)
        if callable(waiter):
            try:
                waiter(timeout=background_review_timeout)
            except Exception:
                logger.debug("wait_for_background_review failed", exc_info=True)
    if post_wait is not None:
        try:
            post_wait()
        except Exception:
            logger.debug("post_wait after background review failed", exc_info=True)

    record = build_hot_pool_outcome(
        evaluation=evaluation,
        run_result=run_result,
        duration_sec=duration_sec,
        benchmark=benchmark,
    )
    complete_fn = _hot_pool_outcome_complete_fn(agent)

    inflight = getattr(agent, "_hot_pool_llm_judge_inflight", False)
    if inflight:
        for _ in range(5):
            time.sleep(0.2)
            if not getattr(agent, "_hot_pool_llm_judge_inflight", False):
                inflight = False
                break
    if getattr(agent, "_hot_pool_llm_judge_inflight", False):
        return {"applied": False, "skipped_reason": "judge_inflight"}

    agent._hot_pool_llm_judge_inflight = True
    try:
        return pool.apply_outcome_feedback(record, complete_fn=complete_fn)
    except Exception:
        logger.debug("benchmark hot pool outcome feedback failed", exc_info=True)
        return {"applied": False, "skipped_reason": "error"}
    finally:
        agent._hot_pool_llm_judge_inflight = False
