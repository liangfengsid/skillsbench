#!/usr/bin/env python3
"""
Extract and summarize SkillsBench Hermes run metrics for JSONL logging and aggregation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def snapshot_agent_usage(agent: Any) -> Dict[str, Any]:
    """Cumulative usage counters from AIAgent at a conversation turn."""
    return {
        "api_calls": int(getattr(agent, "_api_call_count", 0) or 0),
        "tokens": {
            "input": int(getattr(agent, "session_input_tokens", 0) or 0),
            "output": int(getattr(agent, "session_output_tokens", 0) or 0),
            "total": int(getattr(agent, "session_total_tokens", 0) or 0),
            "cache_read": int(getattr(agent, "session_cache_read_tokens", 0) or 0),
            "cache_write": int(getattr(agent, "session_cache_write_tokens", 0) or 0),
            "reasoning": int(getattr(agent, "session_reasoning_tokens", 0) or 0),
        },
        "estimated_cost_usd": _to_float(getattr(agent, "session_estimated_cost_usd", None)),
    }


def count_tool_calls(messages: Any) -> Dict[str, int]:
    """Count tool invocations by name from OpenAI-format messages."""
    if not isinstance(messages, list):
        return {}
    counts: Counter[str] = Counter()
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name") if isinstance(fn, dict) else None
            if name:
                counts[str(name)] += 1
    return dict(counts)


def _eval_block(ev: Optional[dict]) -> Dict[str, Any]:
    ev = ev if isinstance(ev, dict) else {}
    passed = _to_int(ev.get("tests_passed"))
    total = _to_int(ev.get("tests_total"))
    failed = _to_int(ev.get("tests_failed"))
    micro = (passed / total) if passed is not None and total else None
    return {
        "task_success": ev.get("task_success"),
        "reward": _to_float(ev.get("reward")),
        "tests_passed": passed,
        "tests_failed": failed,
        "tests_skipped": _to_int(ev.get("tests_skipped")),
        "tests_total": total,
        "micro_pass_rate": micro,
    }


def _usage_from_result(res: dict) -> Dict[str, Any]:
    return {
        "api_calls": _to_int(res.get("api_calls")),
        "tokens": {
            "input": _to_int(res.get("input_tokens")),
            "output": _to_int(res.get("output_tokens")),
            "total": _to_int(res.get("total_tokens")),
            "cache_read": _to_int(res.get("cache_read_tokens")),
            "cache_write": _to_int(res.get("cache_write_tokens")),
            "reasoning": _to_int(res.get("reasoning_tokens")),
        },
        "estimated_cost_usd": _to_float(res.get("estimated_cost_usd")),
    }


def _usage_from_turn_block(block: dict) -> Dict[str, Any]:
    tokens = {
        "input": _to_int(block.get("input_tokens")),
        "output": _to_int(block.get("output_tokens")),
        "total": _to_int(block.get("total_tokens")),
        "cache_read": _to_int(block.get("cache_read_tokens")),
        "cache_write": _to_int(block.get("cache_write_tokens")),
        "reasoning": _to_int(block.get("reasoning_tokens")),
    }
    if block.get("tokens") and isinstance(block["tokens"], dict):
        tokens = {**tokens, **{k: _to_int(v) for k, v in block["tokens"].items()}}
    return {
        "api_calls": _to_int(block.get("api_calls")),
        "tokens": tokens,
        "estimated_cost_usd": _to_float(block.get("estimated_cost_usd")),
    }


def build_envelope_metrics(envelope: dict) -> Dict[str, Any]:
    """Build a compact ``metrics`` block for JSONL rows."""
    res = envelope.get("run_conversation_result") or {}
    messages = res.get("messages") if isinstance(res, dict) else None
    tool_calls = count_tool_calls(messages)
    tool_rounds = sum(tool_calls.values()) if tool_calls else 0

    final_eval = _eval_block(envelope.get("evaluation"))
    usage = _usage_from_result(res) if isinstance(res, dict) else {
        "api_calls": None,
        "tokens": {},
        "estimated_cost_usd": None,
    }

    pass_at_turn: Dict[str, Any] = {}
    for key, block in sorted(
        (envelope.get("pass_at_turn") or {}).items(),
        key=lambda kv: int(kv[0]),
    ):
        if not isinstance(block, dict):
            continue
        turn_eval = _eval_block(block.get("evaluation"))
        turn_usage = _usage_from_turn_block(block)
        pass_at_turn[str(key)] = {
            "turn": _to_int(block.get("turn")) or int(key),
            **turn_eval,
            **turn_usage,
        }

    br = envelope.get("background_review") or {}
    tel = envelope.get("hot_pool_telemetry") or res.get("hot_pool_telemetry") or {}

    return {
        "final": {
            **final_eval,
            **usage,
            "completed": res.get("completed") if isinstance(res, dict) else None,
            "interrupted": res.get("interrupted") if isinstance(res, dict) else None,
            "failed": res.get("failed") if isinstance(res, dict) else None,
            "duration_sec": _to_float(envelope.get("duration_sec")),
            "tool_calls": tool_calls,
            "tool_rounds": tool_rounds,
        },
        "pass_at_turn": pass_at_turn,
        "background_review": {
            "spawned": br.get("spawned"),
            "completed": br.get("completed"),
            "timeout": br.get("timeout"),
            "action_count": len(br.get("actions") or []),
        },
        "hot_pool": {
            "enabled": envelope.get("hot_pool_enabled"),
            "persist": envelope.get("hot_pool_persist"),
            "inject": (tel.get("inject") if isinstance(tel, dict) else None),
        },
    }
