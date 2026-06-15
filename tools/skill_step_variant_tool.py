#!/usr/bin/env python3
"""Tool: manage per-step skill variant pools (``skills.step_pools``)."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error
from tools.skill_manager_tool import _find_skill, _is_local_skill

logger = logging.getLogger(__name__)


def _step_pools_cfg() -> dict:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        sp = cfg.get("skills", {}).get("step_pools")
        return sp if isinstance(sp, dict) else {}
    except Exception:
        logger.debug("step_pools config load failed", exc_info=True)
        return {}


def _require_enabled() -> Optional[str]:
    if not _step_pools_cfg().get("enabled"):
        return (
            "skills.step_pools.enabled is false — enable it in config.yaml "
            "(skills.step_pools.enabled: true) to use this tool."
        )
    return None


def _maybe_clear_cache() -> None:
    try:
        from agent.prompt_builder import clear_skills_system_prompt_cache

        clear_skills_system_prompt_cache(clear_snapshot=True)
    except Exception:
        pass


def skill_step_variant(
    action: str,
    name: str,
    step_id: str = None,
    variant_id: str = None,
    outcome: str = None,
    body: str = None,
    max_variants: int = None,
    old_string: str = None,
    new_string: str = None,
    ordered_variant_ids: list = None,
) -> str:
    """Dispatch step-pool actions for a local skill (under ~/.hermes/skills/)."""
    gate = _require_enabled()
    if gate:
        return tool_error(gate, success=False)

    found = _find_skill(name)
    if not found:
        return tool_error(
            f"Skill {name!r} not found. Use skills_list / skill_view.", success=False
        )
    skill_dir = found["path"]
    if not _is_local_skill(skill_dir):
        return tool_error(
            "Step pools can only be edited for skills under your local ~/.hermes/skills/ directory.",
            success=False,
        )

    skills_full = {}
    try:
        from hermes_cli.config import load_config

        skills_full = load_config().get("skills") or {}
    except Exception:
        pass

    from agent import skill_step_pools as sp

    action = (action or "").strip().lower()
    try:
        if action == "record_attempt":
            if not step_id or not variant_id or not outcome:
                return tool_error(
                    "record_attempt requires step_id, variant_id, and outcome.",
                    success=False,
                )
            result = sp.record_step_attempt(skill_dir, step_id, variant_id, outcome)
        elif action == "add_variant":
            if not step_id or not body:
                return tool_error(
                    "add_variant requires step_id and body.", success=False
                )
            result = sp.add_step_variant(
                skill_dir,
                step_id,
                body,
                variant_id,
                max_variants,
                skills_cfg=skills_full,
            )
        elif action == "remove_variant":
            if not step_id or not variant_id:
                return tool_error(
                    "remove_variant requires step_id and variant_id.", success=False
                )
            result = sp.remove_step_variant(skill_dir, step_id, variant_id)
        elif action == "patch_variant":
            if not step_id or not variant_id:
                return tool_error(
                    "patch_variant requires step_id and variant_id.", success=False
                )
            if old_string is None or new_string is None:
                return tool_error(
                    "patch_variant requires old_string and new_string.", success=False
                )
            result = sp.patch_step_variant_body(
                skill_dir, step_id, variant_id, old_string, new_string
            )
        elif action == "set_variant_order":
            if not step_id:
                return tool_error("set_variant_order requires step_id.", success=False)
            if not isinstance(ordered_variant_ids, list):
                return tool_error(
                    "set_variant_order requires ordered_variant_ids (list of strings).",
                    success=False,
                )
            result = sp.set_manual_variant_order(
                skill_dir, step_id, [str(x) for x in ordered_variant_ids]
            )
        elif action == "clear_variant_order":
            if not step_id:
                return tool_error("clear_variant_order requires step_id.", success=False)
            result = sp.clear_manual_variant_order(skill_dir, step_id)
        elif action == "list_pools":
            result = sp.list_step_pools(skill_dir)
        else:
            return tool_error(
                f"Unknown action {action!r}. See tool description.", success=False
            )

        if result.get("success"):
            _maybe_clear_cache()
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.warning("skill_step_variant failed: %s", e, exc_info=True)
        return tool_error(str(e), success=False)


SKILL_STEP_VARIANT_SCHEMA: Dict[str, Any] = {
    "name": "skill_step_variant",
    "description": (
        "Manage **per-step variant pools** for skills that use `<!-- hermes-step id=\"...\" -->` "
        "markers in SKILL.md (requires `skills.step_pools.enabled: true` in config). "
        "Variant bodies and statistics live in `step_pools.json` beside SKILL.md. "
        "After trying a variant during execution, call `record_attempt`. "
        "Use `add_variant` / `patch_variant` / `remove_variant` to maintain alternatives; "
        "`set_variant_order` fixes try-order before score-based ranking. "
        "Baseline text lives inside SKILL.md between markers — edit it with skill_manage(patch), "
        "not patch_variant."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "record_attempt",
                    "add_variant",
                    "remove_variant",
                    "patch_variant",
                    "set_variant_order",
                    "clear_variant_order",
                    "list_pools",
                ],
                "description": "Operation to perform.",
            },
            "name": {
                "type": "string",
                "description": "Skill directory name (same as for skill_view / skill_manage).",
            },
            "step_id": {
                "type": "string",
                "description": "Step id from the hermes-step marker (e.g. `bootstrap`).",
            },
            "variant_id": {
                "type": "string",
                "description": (
                    "Variant id: `baseline` for SKILL.md marker body, or an id from add_variant."
                ),
            },
            "outcome": {
                "type": "string",
                "enum": ["success", "fail", "skip"],
                "description": "For record_attempt: result of trying this variant.",
            },
            "body": {
                "type": "string",
                "description": "For add_variant: full markdown/instructions for this variant.",
            },
            "max_variants": {
                "type": "integer",
                "description": (
                    "Optional pool cap for add_variant eviction. "
                    "Omit to use the marker's max_variants or skills.step_pools.default_max_variants_per_step."
                ),
            },
            "old_string": {
                "type": "string",
                "description": "For patch_variant: unique substring of the variant body.",
            },
            "new_string": {
                "type": "string",
                "description": "For patch_variant: replacement text.",
            },
            "ordered_variant_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "For set_variant_order: try-order prefix before score sort.",
            },
        },
        "required": ["action", "name"],
    },
}


registry.register(
    name="skill_step_variant",
    toolset="skills",
    schema=SKILL_STEP_VARIANT_SCHEMA,
    handler=lambda args, **kw: skill_step_variant(
        action=args.get("action", ""),
        name=args.get("name", ""),
        step_id=args.get("step_id"),
        variant_id=args.get("variant_id"),
        outcome=args.get("outcome"),
        body=args.get("body"),
        max_variants=args.get("max_variants"),
        old_string=args.get("old_string"),
        new_string=args.get("new_string"),
        ordered_variant_ids=args.get("ordered_variant_ids"),
    ),
    emoji="🔀",
)
