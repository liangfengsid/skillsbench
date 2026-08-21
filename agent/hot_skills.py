"""Hot skill pool — admission-capped key points, injected in full.

Extracts guardrails from SKILL.md in this order:

1. ``## Common Pitfalls`` / ``## Pitfalls`` / ``## Key Points``-style sections
   (what bundled and agent-authored skills actually use)
2. Optional ``<!-- hermes-hot -->`` markers (rare hand override)
3. Heuristic NEVER/ALWAYS-style bullets elsewhere

The pool cap **is** the inject budget (``max_entries`` points). Eviction runs
only when a new extract would overflow — not every turn. Model "use" of a
point is not observable; policies are ``llm`` (default: admission-time
side-channel judge on the running agent's client) or ``oldest`` (FIFO
fallback / explicit opt-in). Full procedures stay behind ``skill_view``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,63}", re.IGNORECASE)

_HOT_SKILLS_OPEN = "<hot-skills>"
_HOT_SKILLS_CLOSE = "</hot-skills>"
_HOT_SKILLS_BLOCK_RE = re.compile(
    r"<\s*hot-skills\s*>[\s\S]*?</\s*hot-skills\s*>",
    re.IGNORECASE,
)
_HOT_SKILLS_TAG_RE = re.compile(r"</?\s*hot-skills\s*>", re.IGNORECASE)
_HOT_SKILLS_NOTE_RE = re.compile(
    r"\[System note:\s*The following are hot skill key points \(guardrails\)\s*"
    r"from recently used skills,\s*NOT new user input\.\s*"
    r"Use skill_view\(name\) for full procedures\.\]\s*",
    re.IGNORECASE,
)

_HERMES_HOT_OPEN = re.compile(r"<!--\s*hermes-hot\b[^>]*-->", re.IGNORECASE)
_HERMES_HOT_CLOSE = re.compile(r"<!--\s*/hermes-hot\s*-->", re.IGNORECASE)

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_BULLET_RE = re.compile(r"^(\s*(?:[-*•]|\d+\.)\s+)(.+)$", re.MULTILINE)

# Heading titles that match real Hermes + SkillsBench skill conventions.
# Prefer phrase matches over bare tokens like "remember"/"must"/"key" which
# false-positive on procedural headings (Apple Reminders, Key Flags, …).
# SkillsBench task skills (``tasks/*/environment/skills/**/SKILL.md``) heavily
# use Best Practices / Limitations / Error Handling / Important Requirements —
# those must match here or the hot pool stays empty on benchmark runs.
_HOT_SECTION_TITLE_RES = (
    # Hermes authoring + bundled skills
    re.compile(r"\bpitfalls?\b", re.IGNORECASE),
    re.compile(r"\bgotchas?\b", re.IGNORECASE),
    re.compile(r"\bwarnings?\b", re.IGNORECASE),
    re.compile(r"\bcautions?\b", re.IGNORECASE),
    re.compile(r"\bguardrails?\b", re.IGNORECASE),
    re.compile(r"\bkey\s+points?\b", re.IGNORECASE),
    re.compile(r"\bkey\s+rules?\b", re.IGNORECASE),
    re.compile(r"\bkey\s+constraints?\b", re.IGNORECASE),
    re.compile(r"\bcritical\s+rules?\b", re.IGNORECASE),
    re.compile(r"\bcritical\s+(?:implementation\s+)?notes?\b", re.IGNORECASE),
    re.compile(r"\bcritical\s+tips?\b", re.IGNORECASE),
    re.compile(r"\bcritical\s+failure\b", re.IGNORECASE),
    re.compile(r"\bimportant\s+notes?\b", re.IGNORECASE),
    re.compile(r"\bfailure\s+modes?\b", re.IGNORECASE),
    re.compile(r"\bred\s+flags?\b", re.IGNORECASE),
    re.compile(r"\bnever\s+do\b", re.IGNORECASE),
    re.compile(r"\bverification\s+checklist\b", re.IGNORECASE),
    re.compile(r"^critical$", re.IGNORECASE),
    re.compile(r"^warnings?$", re.IGNORECASE),
    # SkillsBench task-skill conventions (very common in environment/skills/)
    re.compile(r"\bbest\s+practices?\b", re.IGNORECASE),
    re.compile(r"\blimitations?\b", re.IGNORECASE),
    re.compile(r"\bcaveats?\b", re.IGNORECASE),
    re.compile(r"\bcommon\s+errors?\b", re.IGNORECASE),
    re.compile(r"\berror\s+handling\b", re.IGNORECASE),
    re.compile(r"\bimportant\s+(?:requirements?|guidelines?)\b", re.IGNORECASE),
    re.compile(r"\bsafety\s+requirements?\b", re.IGNORECASE),
    re.compile(r"\bmust\s+follow\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\b", re.IGNORECASE),
    re.compile(r"\bdon'?t\b", re.IGNORECASE),
    # xlsx / formula skills: "CRITICAL: Use Formulas…", "Zero Formula Errors", …
    re.compile(r"^(?:important|critical)\b", re.IGNORECASE),
    re.compile(r"\bzero\s+formula\s+errors?\b", re.IGNORECASE),
    re.compile(r"\bformula\s+error(?:\s+prevention)?\b", re.IGNORECASE),
    re.compile(r"\brequired\s+format\s+rules?\b", re.IGNORECASE),
    re.compile(r"\bformula\s+construction\s+rules?\b", re.IGNORECASE),
)

_GUARDRAIL_PREFIX_RE = re.compile(
    r"^(?:\*\*)?(?:NEVER|ALWAYS|IMPORTANT|WARNING|CAUTION|DO NOT|MUST NOT|MUST\b|AVOID\b)",
    re.IGNORECASE,
)

_DEFAULT_HOT_POOL = {
    "enabled": True,
    # Retain = inject: this many key points are stored and dumped into
    # <hot-skills> (minus skip_if_in_history). No second per-turn subset.
    "max_entries": 12,
    # Deprecated / ignored: inject size is governed by max_entries (points),
    # not a character budget. Kept in normalize for old configs only.
    "max_chars": 0,
    "max_points_per_skill": 8,
    "max_chars_per_point": 240,
    # Admission-time victim when over max_entries. Model reliance is not
    # observable — do not treat inject as "use".
    # llm (default): side-channel judge — keep a broadcast-worthy subset
    #      (one call per overflow). Falls back to oldest if judge missing/fails.
    # oldest: drop points with the earliest recorded_turn (FIFO of extract).
    #      Deprecated as primary policy; keep for fallback / explicit opt-in.
    "eviction_policy": "llm",
    "inject_on_turn": True,
    "skip_if_in_history": True,
    "history_lookback": 40,
    "hydrate_from_history": True,
    "hydrate_limit": 30,
    "use_hermes_hot_markers": True,
    "extract_sections": True,
    "fallback_extract": True,
    "persist_across_conversations": False,
    "persist_path": "",
    # After a labeled task/episode, LLM-attribute exposed tips and update
    # multi-dimensional utilities (default on). Used at admit/evict, not inject.
    "outcome_feedback": True,
}

_VALID_EVICTION_POLICIES = frozenset({"oldest", "llm"})
_DEFAULT_EVICTION_POLICY = "llm"
_PERSIST_SCHEMA = "hermes.hot_skill_pool.v1"

# Optional test/agent hook: (items, keep_n, context) -> list[id str] to keep.
EvictionJudge = Callable[[List[Dict[str, Any]], int, str], Optional[List[str]]]


def _normalize_hot_pool_cfg(cfg: dict) -> dict:
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["max_entries"] = max(0, int(cfg.get("max_entries", 12) or 0))
    # Legacy key — no longer truncates the inject block.
    cfg["max_chars"] = max(0, int(cfg.get("max_chars", 0) or 0))
    cfg["max_points_per_skill"] = max(1, int(cfg.get("max_points_per_skill", 8) or 8))
    cfg["max_chars_per_point"] = max(32, int(cfg.get("max_chars_per_point", 240) or 240))
    policy = str(
        cfg.get("eviction_policy", _DEFAULT_EVICTION_POLICY) or _DEFAULT_EVICTION_POLICY
    ).strip().lower()
    cfg["eviction_policy"] = (
        policy if policy in _VALID_EVICTION_POLICIES else _DEFAULT_EVICTION_POLICY
    )
    cfg["history_lookback"] = max(1, int(cfg.get("history_lookback", 40) or 40))
    cfg["hydrate_limit"] = max(1, int(cfg.get("hydrate_limit", 30) or 30))
    cfg["use_hermes_hot_markers"] = bool(cfg.get("use_hermes_hot_markers", True))
    cfg["extract_sections"] = bool(cfg.get("extract_sections", True))
    cfg["fallback_extract"] = bool(cfg.get("fallback_extract", True))
    cfg["persist_across_conversations"] = bool(cfg.get("persist_across_conversations", False))
    cfg["persist_path"] = str(cfg.get("persist_path") or "").strip()
    cfg["inject_on_turn"] = bool(cfg.get("inject_on_turn", True))
    cfg["skip_if_in_history"] = bool(cfg.get("skip_if_in_history", True))
    cfg["hydrate_from_history"] = bool(cfg.get("hydrate_from_history", True))
    cfg["outcome_feedback"] = bool(cfg.get("outcome_feedback", True))
    return cfg


def load_hot_skills_config(skills_cfg: Optional[dict] = None) -> dict:
    """Return merged ``skills.hot_pool`` settings."""
    cfg = dict(_DEFAULT_HOT_POOL)
    if not skills_cfg:
        try:
            from hermes_cli.config import load_config

            skills_cfg = (load_config() or {}).get("skills") or {}
        except Exception:
            skills_cfg = {}
    hot = (skills_cfg or {}).get("hot_pool") or {}
    if isinstance(hot, dict):
        for key in cfg:
            if key in hot:
                cfg[key] = hot[key]
    cfg = _normalize_hot_pool_cfg(cfg)
    if os.getenv("HERMES_HOT_POOL_PERSIST", "").strip().lower() in ("1", "true", "yes", "on"):
        cfg["persist_across_conversations"] = True
    _env_path = os.getenv("HERMES_HOT_POOL_PATH", "").strip()
    if _env_path:
        cfg["persist_path"] = _env_path
    _env_enabled = os.getenv("HERMES_HOT_POOL_ENABLED", "").strip().lower()
    if _env_enabled in ("0", "false", "no", "off"):
        cfg["enabled"] = False
        cfg["persist_across_conversations"] = False
    elif _env_enabled in ("1", "true", "yes", "on"):
        cfg["enabled"] = True
    return cfg


def resolve_hot_pool_persist_path(config: Optional[dict] = None) -> Path:
    """Return the JSON file used for cross-conversation hot pool persistence."""
    cfg = config or load_hot_skills_config()
    raw = str(cfg.get("persist_path") or "").strip()
    if raw:
        return Path(raw).expanduser()
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "hot_skill_pool.json"


def extract_hot_key_points(content: str, config: Optional[dict] = None) -> List[str]:
    """Extract decisive key points from skill body text.

    Prefers guardrail-style sections used by Hermes-authored skills
    (``## Common Pitfalls`` / ``## Key Points``) and SkillsBench task skills
    (``## Best Practices`` / ``## Limitations`` / ``## Error Handling`` / …),
    then optional ``<!-- hermes-hot -->`` markers, then heuristic bullets.
    """
    cfg = config or load_hot_skills_config()
    text = (content or "").strip()
    if not text:
        return []

    points: List[str] = []
    # Sections first — matches what ships and what agents are told to write.
    if cfg.get("extract_sections", True):
        points.extend(_extract_section_points(text))
    if cfg.get("use_hermes_hot_markers", True):
        points.extend(_extract_hermes_hot_markers(text))
    if cfg.get("fallback_extract", True):
        points.extend(_extract_heuristic_points(text))

    return _normalize_key_points(points, cfg)


def skill_has_hot_section(content: str) -> bool:
    """True if SKILL.md has a Pitfalls / Best Practices / Key Points-style heading."""
    for match in _HEADING_RE.finditer(content or ""):
        if _is_hot_section_title(match.group(2)):
            return True
    return False


@dataclass
class HotSkillEntry:
    name: str
    key_points: List[str]
    skill_dir: Optional[str] = None
    skill_md_mtime: float = 0.0
    recorded_turn: int = 0  # extract clock; persisted global_turn across conversations
    description: str = ""
    tags: List[str] = field(default_factory=list)
    # Short natural-language applicability hint (shown in inject / eviction).
    # Not used to filter at inject — retain = inject; soft guidance only.
    scope: str = ""
    # Parallel to key_points: multi-dimensional utility stats per tip.
    point_utilities: List[Dict[str, Any]] = field(default_factory=list)


def empty_point_utility() -> Dict[str, Any]:
    """Multi-dimensional tip utility (not success-rate alone)."""
    return {
        "n_labeled": 0,
        "helpful": 0,
        "harmful": 0,
        "irrelevant": 0,
        "success_sum": 0.0,
        "reward_sum": 0.0,
        "iterations_sum": 0.0,
        "n_success": 0,
        "iterations_when_success_sum": 0.0,
    }


def summarize_point_utility(raw: Optional[dict]) -> Dict[str, Any]:
    """Compact utility view for eviction judge / telemetry."""
    u = raw if isinstance(raw, dict) else {}
    n = int(u.get("n_labeled") or 0)
    if n <= 0:
        return {"n_labeled": 0}
    n_success = int(u.get("n_success") or 0)
    out: Dict[str, Any] = {
        "n_labeled": n,
        "helpful": int(u.get("helpful") or 0),
        "harmful": int(u.get("harmful") or 0),
        "irrelevant": int(u.get("irrelevant") or 0),
        "success_rate": round(float(u.get("success_sum") or 0.0) / n, 4),
        "avg_reward": round(float(u.get("reward_sum") or 0.0) / n, 4),
        "avg_iterations": round(float(u.get("iterations_sum") or 0.0) / n, 4),
    }
    if n_success > 0:
        out["avg_iterations_when_success"] = round(
            float(u.get("iterations_when_success_sum") or 0.0) / n_success, 4
        )
    return out


def _utility_strongly_harmful(util: dict, *, min_n: int = 3) -> bool:
    """Conservative admit filter: drop tips with a clear harmful majority."""
    n = int(util.get("n_labeled") or 0)
    if n < min_n:
        return False
    harmful = int(util.get("harmful") or 0)
    helpful = int(util.get("helpful") or 0)
    return harmful >= 2 and harmful >= helpful + 2


def align_point_utilities(
    points: List[str],
    utilities: Optional[List[Any]] = None,
    *,
    previous: Optional["HotSkillEntry"] = None,
) -> List[Dict[str, Any]]:
    """Ensure one utility dict per point; inherit by matching tip text when possible."""
    prev_by_text: Dict[str, Dict[str, Any]] = {}
    if previous is not None:
        for text, util in zip(previous.key_points, previous.point_utilities or []):
            if isinstance(util, dict):
                prev_by_text[str(text)] = dict(util)
    if isinstance(utilities, list):
        for i, text in enumerate(points):
            if i < len(utilities) and isinstance(utilities[i], dict):
                prev_by_text.setdefault(str(text), dict(utilities[i]))
    out: List[Dict[str, Any]] = []
    for text in points:
        inherited = prev_by_text.get(str(text))
        out.append(dict(inherited) if inherited else empty_point_utility())
    return out


def build_hot_pool_outcome(
    *,
    evaluation: Optional[dict] = None,
    run_result: Optional[dict] = None,
    duration_sec: Optional[float] = None,
    benchmark: str = "",
) -> Dict[str, Any]:
    """Normalize harness / run metrics into a multi-dimensional outcome record."""
    ev = evaluation if isinstance(evaluation, dict) else {}
    res = run_result if isinstance(run_result, dict) else {}

    success = bool(ev.get("task_success") if "task_success" in ev else ev.get("success") or ev.get("won"))
    try:
        reward = float(ev["reward"]) if ev.get("reward") is not None else (1.0 if success else 0.0)
    except (TypeError, ValueError):
        reward = 1.0 if success else 0.0

    iterations = None
    iterations_kind = ""
    for key, kind in (
        ("steps", "env_steps"),
        ("env_steps", "env_steps"),
        ("api_calls", "api_calls"),
        ("tool_rounds", "tool_rounds"),
    ):
        if ev.get(key) is not None:
            try:
                iterations = int(ev[key])
                iterations_kind = kind
                break
            except (TypeError, ValueError):
                pass
    if iterations is None and res.get("api_calls") is not None:
        try:
            iterations = int(res["api_calls"])
            iterations_kind = "api_calls"
        except (TypeError, ValueError):
            iterations = None

    dur = duration_sec
    if dur is None and ev.get("duration_sec") is not None:
        try:
            dur = float(ev["duration_sec"])
        except (TypeError, ValueError):
            dur = None

    outcome: Dict[str, Any] = {
        "success": success,
        "reward": reward,
        "iterations": iterations,
        "iterations_kind": iterations_kind,
        "benchmark": (benchmark or "").strip(),
    }
    if dur is not None:
        outcome["duration_sec"] = dur
    if ev.get("tests_passed") is not None:
        try:
            outcome["tests_passed"] = int(ev["tests_passed"])
        except (TypeError, ValueError):
            pass
    if ev.get("tests_total") is not None:
        try:
            outcome["tests_total"] = int(ev["tests_total"])
        except (TypeError, ValueError):
            pass
    return outcome


def derive_hot_scope(
    name: str,
    description: str = "",
    tags: Optional[Iterable[str]] = None,
    *,
    max_chars: int = 200,
) -> str:
    """Seed a short NL scope string at admit time.

    Prefer skill description, else tags joined as prose, else the skill name.
    No regex / stopword matching — semantic judgment of scope belongs to the
    LLM eviction prompt and the agent reading the inject block.
    """
    max_chars = max(0, int(max_chars or 0))
    if max_chars == 0:
        return ""

    desc = (description or "").strip()
    if desc:
        text = desc
    else:
        tag_bits = [str(t).strip() for t in (tags or ()) if str(t).strip()]
        if tag_bits:
            text = ", ".join(tag_bits)
        else:
            text = (name or "").strip()

    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def format_hot_skill_section(name: str, scope: str = "", key_points: Optional[Iterable[str]] = None) -> str:
    """Render one skill section with optional NL ``Scope:`` line + bullets."""
    key = (name or "").strip() or "skill"
    lines = [f"### {key}"]
    scope_text = (scope or "").strip()
    if scope_text:
        lines.append(f"Scope: {scope_text}")
    for pt in key_points or ():
        text = str(pt).strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)


def normalize_hot_scope_value(raw: Any) -> str:
    """Coerce a persisted scope field to a string.

    Scope meaning is authored at admit (``derive_hot_scope``) and judged by the
    LLM eviction prompt / reading agent. Load-time normalize must not rebuild
    or reinterpret scope — only accept what the pool already stored.
    """
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        # Legacy token-list persist: flatten, do not re-template.
        parts = [str(x).strip() for x in raw if str(x).strip()]
        return ", ".join(parts)
    return ""


_TELEMETRY_SCHEMA = "hermes.hot_pool_telemetry.v1"
_HOUSEKEEPING_TOOLS = frozenset({"todo", "memory", "clarify", "cronjob"})


@dataclass
class HotPoolTelemetry:
    """Per-conversation counters for hot pool evaluation (export via export_telemetry)."""

    pool_skills_at_start: int = 0
    pool_points_at_start: int = 0
    global_turn_at_start: int = 0
    loaded_from_persist: bool = False
    pool_skills_at_end: int = 0
    pool_points_at_end: int = 0
    global_turn_at_end: int = 0
    new_records_this_task: int = 0
    records_skipped_no_points: int = 0
    persist_path: str = ""
    build_block_applied: bool = False
    skills_injected: List[str] = field(default_factory=list)
    points_injected: List[str] = field(default_factory=list)
    point_count: int = 0
    chars: int = 0
    excluded_skills: List[str] = field(default_factory=list)
    # Pool skills / points omitted from inject because skip_if_in_history
    # matched a recent skill_view (or similar) — not "hot off".
    skills_excluded_in_history: List[str] = field(default_factory=list)
    points_excluded_in_history: int = 0
    injections_attempted: int = 0
    injections_nonempty: int = 0
    first_nonempty_inject_iter: Optional[int] = None
    skill_view_total: int = 0
    skill_view_repeat: int = 0
    skill_view_while_in_pool: int = 0
    skill_manage_sync: int = 0
    skill_manage_evict: int = 0
    evicted_capacity: int = 0
    outcome_feedback_applied: bool = False
    outcome_feedback_points: int = 0
    outcome_attributions: Dict[str, int] = field(default_factory=dict)
    _skill_view_seen: Set[str] = field(default_factory=set, repr=False)

    @property
    def skill_view_unique(self) -> int:
        return len(self._skill_view_seen)

    def to_dict(self) -> dict:
        return {
            "schema": _TELEMETRY_SCHEMA,
            "carryover": {
                "pool_skills_at_start": self.pool_skills_at_start,
                "pool_points_at_start": self.pool_points_at_start,
                "global_turn_at_start": self.global_turn_at_start,
                "loaded_from_persist": self.loaded_from_persist,
                "pool_skills_at_end": self.pool_skills_at_end,
                "pool_points_at_end": self.pool_points_at_end,
                "global_turn_at_end": self.global_turn_at_end,
                "new_records_this_task": self.new_records_this_task,
                "records_skipped_no_points": self.records_skipped_no_points,
                "persist_path": self.persist_path,
            },
            "inject": {
                "build_block_applied": self.build_block_applied,
                "skills_injected": list(self.skills_injected),
                "points_injected": list(self.points_injected),
                "point_count": self.point_count,
                "chars": self.chars,
                "excluded_skills": list(self.excluded_skills),
                "skills_excluded_in_history": list(self.skills_excluded_in_history),
                "points_excluded_in_history": self.points_excluded_in_history,
                "injections_attempted": self.injections_attempted,
                "injections_nonempty": self.injections_nonempty,
                "first_nonempty_inject_iter": self.first_nonempty_inject_iter,
            },
            "skill_view": {
                "total": self.skill_view_total,
                "unique": self.skill_view_unique,
                "repeat": self.skill_view_repeat,
                "while_in_pool": self.skill_view_while_in_pool,
            },
            "skill_manage": {
                "sync": self.skill_manage_sync,
                "evict": self.skill_manage_evict,
            },
            "eviction": {
                "capacity": self.evicted_capacity,
            },
            "outcome_feedback": {
                "applied": self.outcome_feedback_applied,
                "points_scored": self.outcome_feedback_points,
                "attributions": dict(self.outcome_attributions),
            },
        }


def parse_hot_pool_inner_meta(inner: str) -> tuple[List[str], List[str]]:
    """Extract skill headers and bullet points from a build_block inner body."""
    skills: List[str] = []
    points: List[str] = []
    for line in (inner or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            skills.append(stripped[4:].strip())
        elif stripped.lower().startswith("scope:"):
            continue
        elif stripped.startswith("- "):
            points.append(stripped[2:].strip())
    return skills, points


def alignment_keywords_from_points(points: Iterable[str]) -> Set[str]:
    """Significant tokens from hot points for post-hoc tool alignment checks."""
    out: Set[str] = set()
    for pt in points:
        for tok in _WORD_RE.findall(pt.lower()):
            if len(tok) >= 4:
                out.add(tok)
    return out


def replay_message_tool_stats(messages: List[dict]) -> dict:
    """Post-hoc stats from persisted conversation messages."""
    skill_view_total = 0
    skill_view_seen: Set[str] = set()
    skill_view_repeat = 0
    tool_errors_total = 0
    first_error_iter = None
    api_iter = 0
    substantive_tools: List[tuple[str, str]] = []

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            api_iter += 1
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                name = str(fn.get("name") or "")
                args = str(fn.get("arguments") or "")
                if name == "skill_view":
                    sk = _skill_name_from_tool_args(args)
                    if sk and not _is_subfile_skill_view(args):
                        skill_view_total += 1
                        if sk in skill_view_seen:
                            skill_view_repeat += 1
                        else:
                            skill_view_seen.add(sk)
                if name and name not in _HOUSEKEEPING_TOOLS:
                    substantive_tools.append((name, args))
        elif role == "tool":
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            is_error = False
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    if data.get("success") is False:
                        is_error = True
                    exit_code = data.get("exit_code")
                    if exit_code is not None:
                        try:
                            if int(exit_code) != 0:
                                is_error = True
                        except (TypeError, ValueError):
                            pass
            except (json.JSONDecodeError, TypeError):
                if "error" in content.lower()[:200]:
                    is_error = True
            if is_error:
                tool_errors_total += 1
                if first_error_iter is None:
                    first_error_iter = api_iter

    return {
        "skill_view_total": skill_view_total,
        "skill_view_unique": len(skill_view_seen),
        "skill_view_repeat": skill_view_repeat,
        "tool_errors_total": tool_errors_total,
        "first_tool_error_at_iter": first_error_iter,
        "substantive_tool_calls": substantive_tools,
    }


def _skill_name_from_tool_args(args: str) -> str:
    try:
        data = json.loads(args) if args else {}
        if isinstance(data, dict):
            return str(data.get("name") or "").strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def _is_subfile_skill_view(args: str) -> bool:
    try:
        data = json.loads(args) if args else {}
        if isinstance(data, dict):
            return bool(data.get("file"))
    except (json.JSONDecodeError, TypeError):
        pass
    return False


def compute_alignment_hits(
    substantive_tools: List[tuple[str, str]],
    injected_points: Iterable[str],
) -> tuple[int, int]:
    """Return (hits, checks) for keyword overlap between hot points and tool args."""
    keywords = alignment_keywords_from_points(injected_points)
    if not keywords or not substantive_tools:
        return 0, 0
    hits = 0
    for _name, args in substantive_tools:
        blob = args.lower()
        if any(kw in blob for kw in keywords):
            hits += 1
    return hits, len(substantive_tools)


def check_guardrail_violations(messages: List[dict]) -> dict:
    """Lightweight rubric checks over terminal / write content in messages."""
    checks = 0
    violations = 0
    details: List[str] = []
    blobs: List[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            content = msg.get("content")
            if isinstance(content, str):
                blobs.append(content)
        elif msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                args = str(fn.get("arguments") or "")
                blobs.append(args)

    text = "\n".join(blobs)
    # bare pytest without run_tests.sh wrapper
    if re.search(r"\bpytest\b", text, re.I):
        checks += 1
        if not re.search(r"run_tests\.sh", text, re.I):
            violations += 1
            details.append("bare_pytest")
    if re.search(r"~/.hermes|Path\.home\(\)\s*/\s*[\"']\.hermes", text):
        checks += 1
        if not re.search(r"get_hermes_home|display_hermes_home", text, re.I):
            violations += 1
            details.append("hardcoded_hermes_home")
    return {
        "guardrail_checks_run": checks,
        "guardrail_violations": violations,
        "guardrail_violation_ids": details,
    }


class HotSkillPool:
    """Admission-capped pool of skill key points (session or persisted).

    ``max_entries`` is both retain and inject size. Eviction happens when a
    new extract overflows that cap. Inject dumps the whole pool (optional
    skip of skills already in recent ``skill_view`` history).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        *,
        eviction_judge: Optional[EvictionJudge] = None,
    ) -> None:
        if isinstance(config, dict) and "hot_pool" not in config and "enabled" in config:
            merged = dict(_DEFAULT_HOT_POOL)
            merged.update(config)
            self._config = _normalize_hot_pool_cfg(merged)
        else:
            self._config = load_hot_skills_config(config if isinstance(config, dict) else None)
        self._entries: "OrderedDict[str, HotSkillEntry]" = OrderedDict()
        self._global_turn: int = 0
        self._active_turn: int = 0
        self._admission_context: str = ""
        self.eviction_judge = eviction_judge
        self._telemetry = HotPoolTelemetry()
        self._telemetry_turn_started = False
        self._pool_at_turn_start: Set[str] = set()
        # Tips exposed this episode (injected or skipped via history) for outcome feedback.
        self._exposed_tips: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()
        if self._persist_enabled():
            self._load_persisted()

    def _normalize_config(self) -> None:
        self._config = _normalize_hot_pool_cfg(self._config)

    @property
    def enabled(self) -> bool:
        return bool(self._config.get("enabled"))

    @property
    def config(self) -> dict:
        return dict(self._config)

    @property
    def active_turn(self) -> int:
        """Monotonic extract clock used for ``recorded_turn`` / ``oldest`` eviction.

        With ``persist_across_conversations``, this is the persisted ``global_turn``
        (one tick per ``run_conversation`` / user turn, surviving new agent
        instances). Without persist it is the in-session user-turn count.
        """
        return self._active_turn

    @property
    def global_turn(self) -> int:
        return self._global_turn

    def _persist_enabled(self) -> bool:
        return bool(self._config.get("persist_across_conversations"))

    def _effective_turn(self, session_turn: int) -> int:
        """Stamp for ``recorded_turn``.

        Persisted pools must not use the per-conversation session counter —
        that resets to 1 on each new ``AIAgent``, which would make a new
        conversation's extracts look *older* than prior conversations.
        """
        if self._persist_enabled():
            return self._global_turn or int(session_turn or 0)
        return int(session_turn or 0)

    def on_turn_start(self, session_turn: int) -> None:
        """Begin a user turn / conversation — call before hydrate and build_block."""
        if not self._telemetry_turn_started:
            self._telemetry.pool_skills_at_start = len(self._entries)
            self._telemetry.pool_points_at_start = sum(
                len(e.key_points) for e in self._entries.values()
            )
            self._telemetry.global_turn_at_start = self._global_turn
            self._telemetry.loaded_from_persist = bool(
                self._persist_enabled() and self._entries
            )
            self._pool_at_turn_start = set(self._entries.keys())
            self._telemetry_turn_started = True
        if self._persist_enabled():
            self._global_turn += 1
            self._active_turn = self._global_turn
            self._maybe_persist()
        else:
            self._active_turn = int(session_turn or 0)

    def set_admission_context(self, text: str) -> None:
        """User/task text passed to llm eviction at record time."""
        self._admission_context = (text or "").strip()

    def reset_telemetry(self) -> None:
        """Reset per-conversation telemetry (call at start of run_conversation)."""
        self._telemetry = HotPoolTelemetry()
        self._telemetry_turn_started = False
        self._pool_at_turn_start = set()

    def note_api_injection(self, *, cache_nonempty: bool) -> None:
        """Record one API iteration where hot inject was eligible."""
        if not self.enabled:
            return
        self._telemetry.injections_attempted += 1
        if cache_nonempty:
            self._telemetry.injections_nonempty += 1
            if self._telemetry.first_nonempty_inject_iter is None:
                self._telemetry.first_nonempty_inject_iter = (
                    self._telemetry.injections_attempted
                )

    def note_skill_view(self, name: str, *, in_pool_before: bool) -> None:
        if not self.enabled or not name:
            return
        self._telemetry.skill_view_total += 1
        if name in self._telemetry._skill_view_seen:
            self._telemetry.skill_view_repeat += 1
        else:
            self._telemetry._skill_view_seen.add(name)
        if in_pool_before:
            self._telemetry.skill_view_while_in_pool += 1

    def export_telemetry(self) -> dict:
        """Snapshot telemetry for run_conversation / JSONL envelope."""
        self._telemetry.pool_skills_at_end = len(self._entries)
        self._telemetry.pool_points_at_end = sum(
            len(e.key_points) for e in self._entries.values()
        )
        self._telemetry.global_turn_at_end = self._global_turn
        if self._persist_enabled():
            self._telemetry.persist_path = str(
                resolve_hot_pool_persist_path(self._config)
            )
        out = self._telemetry.to_dict()
        out["config"] = {
            "enabled": self.enabled,
            "max_entries": self._config.get("max_entries"),
            "eviction_policy": self._config.get("eviction_policy"),
            "persist_across_conversations": self._persist_enabled(),
            "skip_if_in_history": bool(self._config.get("skip_if_in_history", True)),
            "outcome_feedback": bool(self._config.get("outcome_feedback", True)),
        }
        return out

    def clear(self) -> None:
        self._entries.clear()
        self._maybe_persist()

    def evict(self, name: str) -> None:
        key = (name or "").strip()
        if key and self._entries.pop(key, None) is not None:
            self._maybe_persist()

    def record(
        self,
        *,
        name: str,
        content: str,
        skill_dir: Optional[str] = None,
        description: str = "",
        tags: Optional[Iterable[str]] = None,
        turn: int = 0,
        file_path: Optional[str] = None,
        key_points: Optional[List[str]] = None,
        user_message: Optional[str] = None,
    ) -> None:
        """Add or refresh key points for a skill (main SKILL.md only)."""
        if not self.enabled:
            return
        if file_path:
            return
        key = (name or "").strip()
        if not key:
            return

        points = list(key_points) if key_points is not None else extract_hot_key_points(content, self._config)
        if not points:
            logger.debug("hot skill pool: no key points extracted for %r — skipping", key)
            self._telemetry.records_skipped_no_points += 1
            return

        previous = self._entries.get(key)
        if previous is not None and self._config.get("outcome_feedback", True):
            prev_utils = {
                str(t): u
                for t, u in zip(previous.key_points, previous.point_utilities or [])
                if isinstance(u, dict)
            }
            filtered: List[str] = []
            for p in points:
                util = prev_utils.get(str(p))
                if util and _utility_strongly_harmful(util):
                    continue
                filtered.append(p)
            points = filtered
            if not points:
                logger.debug(
                    "hot skill pool: all tips for %r filtered by harmful utilities — skipping",
                    key,
                )
                self._telemetry.records_skipped_no_points += 1
                return

        mtime = _skill_md_mtime(skill_dir)
        tag_list = [str(t) for t in (tags or []) if t]
        desc = (description or "").strip()
        scope = derive_hot_scope(key, description=desc, tags=tag_list)
        previous = self._entries.pop(key, None)
        now = self._effective_turn(turn)
        entry = HotSkillEntry(
            name=key,
            key_points=points,
            skill_dir=skill_dir,
            skill_md_mtime=mtime,
            recorded_turn=now,
            description=desc,
            tags=tag_list,
            scope=scope,
            point_utilities=align_point_utilities(points, previous=previous),
        )
        self._entries[key] = entry
        context = user_message if user_message is not None else self._admission_context
        self._trim_to_max_entries(new_name=key, context=context or "")
        self._telemetry.new_records_this_task += 1
        self._maybe_persist()

    def record_from_tool_result(self, result_json: str, turn: int = 0) -> None:
        if not self.enabled or not result_json:
            return
        try:
            data = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(data, dict) or not data.get("success"):
            return
        if data.get("file"):
            return
        self.record(
            name=str(data.get("name") or ""),
            content=str(data.get("content") or ""),
            skill_dir=data.get("skill_dir"),
            description=str(data.get("description") or ""),
            tags=data.get("tags") if isinstance(data.get("tags"), list) else [],
            turn=turn,
        )

    def record_from_loaded_skill(self, loaded_skill: dict, turn: int = 0) -> None:
        if not isinstance(loaded_skill, dict):
            return
        self.record(
            name=str(loaded_skill.get("name") or ""),
            content=str(loaded_skill.get("content") or ""),
            skill_dir=loaded_skill.get("skill_dir"),
            description=str(loaded_skill.get("description") or ""),
            tags=loaded_skill.get("tags") if isinstance(loaded_skill.get("tags"), list) else [],
            turn=turn,
        )

    def record_from_skill_manage(
        self,
        function_args: Optional[dict],
        result_json: str,
        *,
        turn: int = 0,
    ) -> None:
        """Sync hot key points after a successful ``skill_manage`` on SKILL.md.

        Background skill review and foreground ``skill_manage`` calls do not go
        through ``skill_view``; reload SKILL.md from disk (or use create/edit
        content from args) so the pool matches newly written skills.
        """
        if not self.enabled or not result_json:
            return
        try:
            data = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(data, dict) or not data.get("success"):
            return

        args = function_args if isinstance(function_args, dict) else {}
        action = args.get("action")
        name = str(args.get("name") or "").strip()
        if not name:
            return

        if action == "delete":
            if self._entries.pop(name, None) is not None:
                self._telemetry.skill_manage_evict += 1
                self._maybe_persist()
            return

        if action == "patch" and args.get("file_path"):
            return
        if action in ("write_file", "remove_file"):
            return
        if action not in ("create", "edit", "patch"):
            return

        content = ""
        if action in ("create", "edit"):
            content = str(args.get("content") or "").strip()
        if not content:
            content = _read_skill_md_content(name)
        if not content:
            return

        skill_dir = _resolve_skill_dir_path(name)
        points = extract_hot_key_points(content, self._config)
        if not points:
            if self._entries.pop(name, None) is not None:
                self._telemetry.skill_manage_evict += 1
                self._maybe_persist()
            return

        self.record(
            name=name,
            content=content,
            skill_dir=skill_dir,
            turn=turn,
        )
        self._telemetry.skill_manage_sync += 1

    def hydrate_from_history(self, messages: List[dict], *, limit: Optional[int] = None) -> int:
        """Bootstrap pool entries from recent ``skill_view`` tool results."""
        if not self.enabled or not self._config.get("hydrate_from_history", True):
            return 0
        if not messages:
            return 0
        max_scan = int(limit or self._config.get("hydrate_limit", 30) or 30)
        added = 0
        for msg in reversed(messages[-max_scan:]):
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict) or not data.get("success"):
                continue
            if data.get("file"):
                continue
            name = str(data.get("name") or "").strip()
            if not name or name in self._entries:
                continue
            self.record_from_tool_result(content, turn=self._active_turn)
            if name in self._entries:
                added += 1
        return added

    def refresh_stale_entries(self, *, session_id: Optional[str] = None) -> int:
        """Re-extract key points when SKILL.md mtime changed on disk."""
        if not self.enabled:
            return 0
        refreshed = 0
        for key, entry in list(self._entries.items()):
            current_mtime = _skill_md_mtime(entry.skill_dir)
            if current_mtime <= 0 or current_mtime == entry.skill_md_mtime:
                continue
            new_content = _reload_skill_content(key, session_id=session_id)
            if not new_content:
                self.evict(key)
                continue
            new_points = extract_hot_key_points(new_content, self._config)
            if not new_points:
                self.evict(key)
                continue
            entry.key_points = new_points
            entry.skill_md_mtime = current_mtime
            refreshed += 1
        if refreshed:
            self._trim_to_max_entries(new_name="", context=self._admission_context)
            self._maybe_persist()
        return refreshed

    def skills_in_recent_history(
        self,
        messages: List[dict],
        *,
        lookback: Optional[int] = None,
    ) -> Set[str]:
        """Return skill names with a recent main ``skill_view`` tool result."""
        names: Set[str] = set()
        if not messages:
            return names
        n = int(lookback or self._config.get("history_lookback", 40) or 40)
        for msg in messages[-n:]:
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict) or not data.get("success"):
                continue
            if data.get("file"):
                continue
            name = str(data.get("name") or "").strip()
            if name:
                names.add(name)
        return names

    def build_block(
        self,
        *,
        user_message: str = "",
        turn: int = 0,
        exclude_names: Optional[Set[str]] = None,
    ) -> str:
        """Serialize the full retained pool. No per-turn subset ranking."""
        if user_message:
            self._admission_context = user_message
        if not self.enabled or not self._config.get("inject_on_turn", True):
            return ""
        if not self._entries:
            return ""

        exclude = set(exclude_names or ())
        # Pool tips omitted because the skill was already opened in recent
        # history (skip_if_in_history). Empty inject + nonzero here ≠ hot off.
        skipped = [
            entry
            for entry in self._entries.values()
            if entry.name in exclude and entry.key_points
        ]
        self._telemetry.excluded_skills = sorted(exclude)
        self._telemetry.skills_excluded_in_history = sorted(e.name for e in skipped)
        self._telemetry.points_excluded_in_history = sum(
            len(e.key_points) for e in skipped
        )
        # Track exposure for outcome feedback (injected + history-skipped).
        for entry in skipped:
            for text in entry.key_points:
                key = (entry.name, text)
                self._exposed_tips[key] = {
                    "skill": entry.name,
                    "point": text,
                    "scope": entry.scope or "",
                    "via": "history",
                }

        candidates = [
            entry for entry in self._entries.values()
            if entry.name not in exclude and entry.key_points
        ]
        if not candidates:
            self._telemetry.build_block_applied = False
            self._telemetry.skills_injected = []
            self._telemetry.points_injected = []
            self._telemetry.point_count = 0
            self._telemetry.chars = 0
            return ""

        # Full retain = inject dump. Budget is max_entries (points), not chars —
        # providers bill tokens; a second char truncate would silently drop tips
        # the overflow judge already chose to keep.
        inner = self._format_entries(candidates)
        if not inner:
            return ""
        skills, points = parse_hot_pool_inner_meta(inner)
        block = build_hot_skills_block(inner)
        self._telemetry.build_block_applied = True
        self._telemetry.skills_injected = skills
        self._telemetry.points_injected = points
        self._telemetry.point_count = len(points)
        self._telemetry.chars = len(block)
        for entry in candidates:
            for text in entry.key_points:
                key = (entry.name, text)
                self._exposed_tips[key] = {
                    "skill": entry.name,
                    "point": text,
                    "scope": entry.scope or "",
                    "via": "inject",
                }
        return block

    def _format_entries(self, entries: List[HotSkillEntry]) -> str:
        parts = [
            format_hot_skill_section(entry.name, entry.scope, entry.key_points)
            for entry in entries
            if entry.key_points
        ]
        if not parts:
            return ""
        return "\n\n---\n\n".join(parts)

    def _total_points(self) -> int:
        return sum(len(e.key_points) for e in self._entries.values())

    def _point_items(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        n = 0
        for name, entry in self._entries.items():
            utils = list(entry.point_utilities or [])
            while len(utils) < len(entry.key_points):
                utils.append(empty_point_utility())
            for idx, text in enumerate(entry.key_points):
                items.append(
                    {
                        "id": str(n),
                        "skill": name,
                        "index": idx,
                        "point": text,
                        "scope": entry.scope or "",
                        "utility": summarize_point_utility(utils[idx]),
                        "recorded_turn": int(entry.recorded_turn),
                    }
                )
                n += 1
        return items

    def _drop_point(self, skill: str, index: int) -> None:
        entry = self._entries.get(skill)
        if entry is None or index < 0 or index >= len(entry.key_points):
            return
        entry.key_points.pop(index)
        if entry.point_utilities and index < len(entry.point_utilities):
            entry.point_utilities.pop(index)
        if not entry.key_points:
            self._entries.pop(skill, None)

    def _trim_to_max_entries(self, *, new_name: str, context: str) -> None:
        cap = int(self._config.get("max_entries", 12) or 0)
        if cap <= 0:
            self._entries.clear()
            return
        incoming = self._entries.get(new_name)
        if incoming is not None and len(incoming.key_points) > cap:
            incoming.key_points = incoming.key_points[:cap]
            if incoming.point_utilities:
                incoming.point_utilities = incoming.point_utilities[:cap]
        if self._total_points() <= cap:
            return

        policy = str(self._config.get("eviction_policy") or _DEFAULT_EVICTION_POLICY)
        evicted = 0
        if policy == "llm":
            dropped = self._apply_llm_keep(cap, context)
            evicted += dropped

        # oldest fallback (also the full path when eviction_policy == "oldest")
        while self._total_points() > cap:
            items = self._point_items()
            if not items:
                break
            victim = self._oldest_victim(items, new_name=new_name)
            if victim is None:
                break
            self._drop_point(victim[0], victim[1])
            evicted += 1
        if evicted:
            self._telemetry.evicted_capacity += evicted

    def _apply_llm_keep(self, cap: int, context: str) -> int:
        """Ask the judge once for which point ids to keep. Returns how many dropped."""
        items = self._point_items()
        if not items:
            return 0
        keep_ids: Optional[List[str]] = None
        if callable(self.eviction_judge):
            try:
                keep_ids = self.eviction_judge(items, cap, context)
            except Exception:
                logger.debug("hot skill llm eviction_judge failed", exc_info=True)
                keep_ids = None
        if not keep_ids:
            return 0
        keep = {str(i) for i in keep_ids}
        drop = [it for it in items if str(it["id"]) not in keep]
        drop.sort(key=lambda it: (str(it["skill"]), -int(it["index"])))
        dropped = 0
        for it in drop:
            if self._total_points() <= cap:
                break
            self._drop_point(str(it["skill"]), int(it["index"]))
            dropped += 1
        return dropped

    def _oldest_victim(
        self, items: List[Dict[str, Any]], *, new_name: str
    ) -> Optional[Tuple[str, int]]:
        others = [it for it in items if it["skill"] != new_name]
        pool = others or items
        victim = min(
            pool,
            key=lambda it: (
                int(it["recorded_turn"]),
                str(it["skill"]),
                -int(it["index"]),
            ),
        )
        return str(victim["skill"]), int(victim["index"])

    def clear_exposed_tips(self) -> None:
        """Reset episode exposure tracking (call at episode / task start if needed)."""
        self._exposed_tips.clear()

    def apply_outcome_feedback(
        self,
        outcome: dict,
        *,
        complete_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ) -> Dict[str, Any]:
        """LLM-attribute exposed tips after a labeled task; update utilities.

        Uses multi-dimensional outcome fields (success, reward, iterations, …).
        Does not change inject behavior (retain = inject).
        """
        summary = {
            "applied": False,
            "points_scored": 0,
            "attributions": {},
            "skipped_reason": "",
        }
        if not self.enabled or not self._config.get("outcome_feedback", True):
            summary["skipped_reason"] = "disabled"
            return summary
        if not isinstance(outcome, dict) or not outcome:
            summary["skipped_reason"] = "missing_outcome"
            return summary
        if not self._exposed_tips:
            summary["skipped_reason"] = "no_exposed_tips"
            return summary
        if not callable(complete_fn):
            summary["skipped_reason"] = "no_complete_fn"
            return summary

        items = []
        for i, ((_sk, _pt), tip) in enumerate(self._exposed_tips.items()):
            items.append(
                {
                    "id": str(i),
                    "skill": tip.get("skill"),
                    "point": tip.get("point"),
                    "scope": tip.get("scope") or "",
                    "via": tip.get("via") or "",
                }
            )
        try:
            text = complete_fn(
                build_outcome_attribution_messages(items, outcome)
            ) or ""
        except Exception:
            logger.debug("hot pool outcome attribution complete_fn failed", exc_info=True)
            summary["skipped_reason"] = "complete_fn_failed"
            return summary

        labels = parse_outcome_attributions(text, items)
        if not labels:
            summary["skipped_reason"] = "unparseable_or_empty"
            return summary

        success = bool(outcome.get("success"))
        try:
            reward = float(outcome.get("reward") if outcome.get("reward") is not None else (1.0 if success else 0.0))
        except (TypeError, ValueError):
            reward = 1.0 if success else 0.0
        try:
            iterations = float(outcome["iterations"]) if outcome.get("iterations") is not None else None
        except (TypeError, ValueError):
            iterations = None

        attr_counts: Dict[str, int] = {"helpful": 0, "harmful": 0, "irrelevant": 0}
        scored = 0
        for item in items:
            label = labels.get(str(item["id"]))
            if label not in ("helpful", "harmful", "irrelevant"):
                continue
            skill = str(item.get("skill") or "")
            point = str(item.get("point") or "")
            entry = self._entries.get(skill)
            if entry is None:
                continue
            try:
                idx = entry.key_points.index(point)
            except ValueError:
                continue
            while len(entry.point_utilities) < len(entry.key_points):
                entry.point_utilities.append(empty_point_utility())
            util = entry.point_utilities[idx]
            util["n_labeled"] = int(util.get("n_labeled") or 0) + 1
            util[label] = int(util.get(label) or 0) + 1
            util["success_sum"] = float(util.get("success_sum") or 0.0) + (1.0 if success else 0.0)
            util["reward_sum"] = float(util.get("reward_sum") or 0.0) + reward
            if iterations is not None:
                util["iterations_sum"] = float(util.get("iterations_sum") or 0.0) + iterations
                if success:
                    util["n_success"] = int(util.get("n_success") or 0) + 1
                    util["iterations_when_success_sum"] = (
                        float(util.get("iterations_when_success_sum") or 0.0) + iterations
                    )
            attr_counts[label] = attr_counts.get(label, 0) + 1
            scored += 1

        self._telemetry.outcome_feedback_applied = scored > 0
        self._telemetry.outcome_feedback_points = scored
        self._telemetry.outcome_attributions = dict(attr_counts)
        if scored:
            self._maybe_persist()
        summary.update(
            {
                "applied": scored > 0,
                "points_scored": scored,
                "attributions": attr_counts,
            }
        )
        return summary

    def _entry_to_dict(self, entry: HotSkillEntry) -> dict:
        utils = align_point_utilities(entry.key_points, entry.point_utilities)
        return {
            "name": entry.name,
            "key_points": list(entry.key_points),
            "skill_dir": entry.skill_dir,
            "skill_md_mtime": entry.skill_md_mtime,
            "recorded_turn": entry.recorded_turn,
            "description": entry.description,
            "tags": list(entry.tags),
            "scope": entry.scope or "",
            "point_utilities": utils,
        }

    @staticmethod
    def _entry_from_dict(data: dict) -> Optional[HotSkillEntry]:
        if not isinstance(data, dict):
            return None
        name = str(data.get("name") or "").strip()
        points = data.get("key_points")
        if not name or not isinstance(points, list) or not points:
            return None
        clean_points = [str(p).strip() for p in points if str(p).strip()]
        if not clean_points:
            return None
        tags = data.get("tags")
        tag_list = [str(t) for t in tags if t] if isinstance(tags, list) else []
        description = str(data.get("description") or "")
        # Prefer stored scope; only derive when older persist rows lack it.
        scope = normalize_hot_scope_value(data.get("scope"))
        if not scope:
            scope = derive_hot_scope(name, description=description, tags=tag_list)
        recorded = data.get("recorded_turn")
        if recorded is None:
            recorded = data.get("last_used_turn") or 0
        utilities = data.get("point_utilities")
        return HotSkillEntry(
            name=name,
            key_points=clean_points,
            skill_dir=data.get("skill_dir"),
            skill_md_mtime=float(data.get("skill_md_mtime") or 0.0),
            recorded_turn=int(recorded or 0),
            description=description,
            tags=tag_list,
            scope=scope,
            point_utilities=align_point_utilities(clean_points, utilities),
        )

    def _load_persisted(self) -> None:
        path = resolve_hot_pool_persist_path(self._config)
        try:
            if not path.is_file():
                return
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("hot skill pool: could not load %s (%s)", path, exc)
            return
        if not isinstance(raw, dict):
            return
        if raw.get("schema") not in (_PERSIST_SCHEMA, None):
            logger.warning("hot skill pool: unsupported persist schema in %s", path)
            return
        self._global_turn = max(0, int(raw.get("global_turn") or 0))
        order = raw.get("order")
        if not isinstance(order, list):
            order = raw.get("lru_order")
        entries_raw = raw.get("entries")
        if not isinstance(entries_raw, dict):
            return
        loaded: "OrderedDict[str, HotSkillEntry]" = OrderedDict()
        names: List[str] = []
        if isinstance(order, list):
            names.extend(str(n).strip() for n in order if str(n).strip())
        for key in entries_raw:
            if key not in names:
                names.append(str(key))
        for name in names:
            entry = self._entry_from_dict(entries_raw.get(name))
            if entry is not None:
                loaded[name] = entry
        self._entries = loaded
        logger.debug(
            "hot skill pool: loaded %d entries from %s (global_turn=%d)",
            len(self._entries),
            path,
            self._global_turn,
        )

    def _maybe_persist(self) -> None:
        if not self._persist_enabled():
            return
        path = resolve_hot_pool_persist_path(self._config)
        payload = {
            "schema": _PERSIST_SCHEMA,
            "updated_at": time.time(),
            "global_turn": self._global_turn,
            "order": list(self._entries.keys()),
            "entries": {k: self._entry_to_dict(v) for k, v in self._entries.items()},
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            logger.warning("hot skill pool: could not persist to %s (%s)", path, exc)


def _extract_hermes_hot_markers(content: str) -> List[str]:
    points: List[str] = []
    pos = 0
    while True:
        open_m = _HERMES_HOT_OPEN.search(content, pos)
        if not open_m:
            break
        close_m = _HERMES_HOT_CLOSE.search(content, open_m.end())
        if not close_m:
            break
        block = content[open_m.end(): close_m.start()]
        points.extend(_split_block_into_points(block))
        pos = close_m.end()
    return points


def _strip_heading_decorations(title: str) -> str:
    """Drop leading emoji / symbols so '🔴 Critical' → 'Critical'."""
    cleaned = re.sub(r"^[^\w]+", "", (title or "").strip(), flags=re.UNICODE)
    return cleaned.strip()


def _is_hot_section_title(title: str) -> bool:
    """Whether a markdown heading is a guardrail / pitfalls / key-points section."""
    normalized = _strip_heading_decorations(title)
    if not normalized:
        return False
    return any(pat.search(normalized) for pat in _HOT_SECTION_TITLE_RES)


def _extract_section_points(content: str) -> List[str]:
    headings = list(_HEADING_RE.finditer(content))
    if not headings:
        return []

    points: List[str] = []
    for idx, match in enumerate(headings):
        title = match.group(2).strip()
        if not _is_hot_section_title(title):
            continue
        start = match.end()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(content)
        section = content[start:end]
        points.extend(_split_block_into_points(section))
    return points


def _extract_heuristic_points(content: str) -> List[str]:
    points: List[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        bullet_m = _BULLET_RE.match(stripped)
        body = bullet_m.group(2).strip() if bullet_m else stripped
        body = re.sub(r"\*\*", "", body).strip()
        if _GUARDRAIL_PREFIX_RE.match(body):
            points.append(body)
            continue
        if bullet_m and len(body) <= 160 and any(
            kw in body.lower() for kw in ("never", "always", "must not", "do not", "important")
        ):
            points.append(body)
    return points


def _split_block_into_points(block: str) -> List[str]:
    """Split a pitfalls / key-points section into injectable lines.

    Handles ``-`` / ``*`` bullets, ``1.`` numbered pitfalls (common in
    ``## Common Pitfalls``), blockquotes, and short standalone lines.
    Skips checkbox scaffolding like ``- [ ]`` only when the remainder is empty.
    """
    points: List[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Checkbox items: keep the task text after "[ ]" / "[x]"
        checkbox_m = re.match(r"^[-*•]\s*\[(?: |x|X)\]\s*(.+)$", stripped)
        if checkbox_m:
            body = checkbox_m.group(1).strip()
            if body:
                points.append(body)
            continue
        bullet_m = _BULLET_RE.match(stripped)
        if bullet_m:
            points.append(bullet_m.group(2).strip())
            continue
        if stripped.startswith(">"):
            points.append(stripped.lstrip("> ").strip())
            continue
        if len(stripped) <= 240 and not stripped.startswith("#"):
            points.append(stripped)
    return points


def _normalize_key_points(points: List[str], config: dict) -> List[str]:
    max_points = int(config.get("max_points_per_skill", 8) or 8)
    max_len = int(config.get("max_chars_per_point", 240) or 240)
    seen: Set[str] = set()
    out: List[str] = []
    for raw in points:
        pt = re.sub(r"\s+", " ", (raw or "").strip())
        pt = re.sub(r"\*\*", "", pt).strip()
        if not pt:
            continue
        if len(pt) > max_len:
            pt = pt[: max_len - 3].rstrip() + "..."
        key = pt.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(pt)
        if len(out) >= max_points:
            break
    return out


def _resolve_skill_dir_path(name: str) -> Optional[str]:
    try:
        from tools.skill_manager_tool import _find_skill

        found = _find_skill(name)
        if found and found.get("path"):
            return str(found["path"])
    except Exception:
        pass
    return None


def _read_skill_md_content(name: str) -> str:
    skill_dir = _resolve_skill_dir_path(name)
    if not skill_dir:
        return ""
    skill_md = Path(skill_dir) / "SKILL.md"
    try:
        return skill_md.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _skill_md_mtime(skill_dir: Optional[str]) -> float:
    if not skill_dir:
        return 0.0
    try:
        skill_md = Path(skill_dir) / "SKILL.md"
        if skill_md.is_file():
            return skill_md.stat().st_mtime
    except OSError:
        pass
    return 0.0


def _reload_skill_content(name: str, *, session_id: Optional[str] = None) -> str:
    try:
        from tools.skills_tool import skill_view

        raw = skill_view(name, task_id=session_id, preprocess=True)
        data = json.loads(raw)
    except Exception:
        return ""
    if not isinstance(data, dict) or not data.get("success") or data.get("file"):
        return ""
    return str(data.get("content") or "").strip()


def build_llm_eviction_system_prompt(keep_n: int) -> str:
    """System prompt for overflow keep-set judgment (retain = inject)."""
    return (
        "You curate a small hot-skill key-point pool used as short guardrail "
        "reminders.\n"
        "\n"
        "Hard constraint — retain equals inject: every tip you KEEP will be "
        "shown in full on later turns and tasks, not only on the current one. "
        "Optimize for broadcast-worthiness (transferable across future work), "
        "not for maximizing usefulness on the present task alone.\n"
        "\n"
        f"Return JSON only: {{\"keep\": [\"id\", ...]}} with at most {keep_n} "
        "ids drawn from the provided points. Do not invent ids. You may return "
        "fewer than keep_n ids if fewer tips meet the bar.\n"
        "\n"
        "KEEP tips that:\n"
        "- State clear, actionable guardrails that remain meaningful after "
        "removing instance-specific details\n"
        "- Capture transferable pitfalls or procedures (often NEVER/ALWAYS/"
        "MUST-style) likely to help on other similar-class tasks\n"
        "- Add distinct coverage — prefer diversity across skills/topics when "
        "candidates are similar\n"
        "- Use each point's natural-language scope (when present) as a soft hint "
        "about where it applies; prefer tips whose stated scope is broadly "
        "transferable over narrow niche scopes when choosing among equals\n"
        "- When utility stats are present (n_labeled > 0), prefer tips with more "
        "helpful than harmful attributions; also prefer lower "
        "avg_iterations_when_success / avg_iterations when success rates are "
        "similar — efficiency matters when almost all tasks succeed\n"
        "\n"
        "DROP or deprioritize tips that:\n"
        "- Are bound to absolute paths, hostnames, credentials, account names, "
        "one-off filenames, task IDs, or a single product/CLI workflow\n"
        "- Encode one task's I/O or setup recipe rather than a general pitfall\n"
        "- Duplicate a stronger tip already being kept\n"
        "- Would mislead or waste effort if shown on an unrelated later task\n"
        "- Have clearly worse utility than alternatives (high harmful count, or "
        "much higher iteration cost for similar success)\n"
        "\n"
        "About context: it describes the overflow / incoming extract only. Use "
        "it to understand what is being admitted and to break ties. Do NOT "
        "treat \"most relevant to context\" as the primary keep criterion."
    )


def build_outcome_attribution_messages(
    items: List[Dict[str, Any]],
    outcome: dict,
) -> List[Dict[str, str]]:
    """Isolated judge: label each exposed tip helpful / harmful / irrelevant."""
    system = (
        "You attribute credit for task outcome to hot-skill tips that were "
        "exposed during the episode (injected or opened via skill_view).\n"
        "\n"
        "The outcome is multi-dimensional — do NOT use success alone. Consider "
        "reward, iterations/steps (efficiency; lower is better when the task "
        "succeeded), tests_passed/tests_total, and duration when present. On "
        "benchmarks where success is near-ceiling, prefer judging whether a tip "
        "helped or hurt efficiency and reliability.\n"
        "\n"
        "Return JSON only: {\"labels\": {\"<id>\": \"helpful\"|\"harmful\"|"
        "\"irrelevant\", ...}} for the given point ids. Do not invent ids.\n"
        "- helpful: tip plausibly improved outcome or efficiency\n"
        "- harmful: tip plausibly caused waste, wrong paths, or failure modes\n"
        "- irrelevant: tip did not meaningfully affect this episode\n"
    )
    payload = {
        "outcome": outcome,
        "points": items,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_outcome_attributions(
    text: str,
    items: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Parse attribution JSON into id -> label. Empty dict on failure."""
    valid_ids = {str(it["id"]) for it in items}
    allowed = {"helpful", "harmful", "irrelevant"}
    raw = (text or "").strip()
    if not raw:
        return {}
    raw = re.sub(
        r"<think(?:ing)?\b[^>]*>.*?</think(?:ing)?>",
        "",
        raw,
        flags=re.I | re.DOTALL,
    ).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    labels = data.get("labels") if isinstance(data, dict) else None
    if not isinstance(labels, dict):
        return {}
    out: Dict[str, str] = {}
    for key, val in labels.items():
        kid = str(key)
        label = str(val or "").strip().lower()
        if kid in valid_ids and label in allowed:
            out[kid] = label
    return out


def build_llm_eviction_messages(
    items: List[Dict[str, Any]],
    keep_n: int,
    context: str,
) -> List[Dict[str, str]]:
    """Isolated judge prompt — not appended to the conversation."""
    payload = {
        "selection_goal": (
            "Choose a keep-set safe to broadcast on future unrelated tasks "
            "(retain = inject). Prefer transferable abstraction over "
            "context-local relevance."
        ),
        "context_role": (
            "Background for the incoming extract / overflow. Tie-breaker only; "
            "not the primary ranking objective."
        ),
        "context": context or "",
        "keep_n": keep_n,
        "points": items,
    }
    return [
        {
            "role": "system",
            "content": build_llm_eviction_system_prompt(keep_n),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def parse_llm_keep_ids(
    text: str,
    items: List[Dict[str, Any]],
    keep_n: int,
) -> Optional[List[str]]:
    """Parse a judge completion into valid point ids. None → fall back to oldest."""
    if keep_n <= 0 or not items:
        return []
    raw = (text or "").strip()
    if not raw:
        return None
    raw = re.sub(
        r"<think(?:ing)?\b[^>]*>.*?</think(?:ing)?>",
        "",
        raw,
        flags=re.I | re.DOTALL,
    ).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    keep = data.get("keep") if isinstance(data, dict) else data
    if not isinstance(keep, list):
        return None
    valid = {str(it["id"]) for it in items}
    out = [str(i) for i in keep if str(i) in valid]
    return out[:keep_n] if out else None


def llm_eviction_keep_ids(
    items: List[Dict[str, Any]],
    keep_n: int,
    context: str,
    complete_fn: Callable[[List[Dict[str, str]]], str],
) -> Optional[List[str]]:
    """Run the judge via ``complete_fn(messages) -> text``. None on failure."""
    if not callable(complete_fn) or keep_n <= 0 or not items:
        return None
    try:
        text = complete_fn(build_llm_eviction_messages(items, keep_n, context)) or ""
    except Exception:
        logger.debug("hot skill llm eviction complete_fn failed", exc_info=True)
        return None
    return parse_llm_keep_ids(text, items, keep_n)


def sanitize_hot_skills_text(text: str) -> str:
    """Strip hot-skills fence tags and injected blocks from text."""
    text = _HOT_SKILLS_BLOCK_RE.sub("", text)
    text = _HOT_SKILLS_NOTE_RE.sub("", text)
    text = _HOT_SKILLS_TAG_RE.sub("", text)
    return text


def build_hot_skills_block(raw_context: str) -> str:
    """Wrap hot skill key points in a fenced ephemeral block."""
    if not raw_context or not raw_context.strip():
        return ""
    clean = sanitize_hot_skills_text(raw_context)
    if clean != raw_context:
        logger.warning("hot skill pool returned pre-wrapped context; stripped")
    return (
        f"{_HOT_SKILLS_OPEN}\n"
        "[System note: The following are hot skill key points (guardrails) "
        "from recently used skills, NOT new user input. "
        "Each section may include a Scope: line in natural language — apply a "
        "tip when that scope fits the current task; otherwise ignore it. "
        "Use skill_view(name) for full procedures.]\n\n"
        f"{clean}\n"
        f"{_HOT_SKILLS_CLOSE}"
    )
