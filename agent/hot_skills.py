"""Hot skill pool — LRU cache of decisive skill key points for prompt injection.

Extracts guardrails from SKILL.md in this order:

1. ``## Common Pitfalls`` / ``## Pitfalls`` / ``## Key Points``-style sections
   (what bundled and agent-authored skills actually use)
2. Optional ``<!-- hermes-hot -->`` markers (rare hand override)
3. Heuristic NEVER/ALWAYS-style bullets elsewhere

Injects ephemerally into the current turn's user message. Full procedures stay
behind ``skill_view``.
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
from typing import Any, Dict, Iterable, List, Optional, Set

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
    # Budget schedule: "per_skill" (grid: max_skills × max_points_per_skill) or
    # "global_pool" (flat cap on total injected key points — recommended).
    "entry_schedule": "global_pool",
    "max_entries": 12,
    "max_skills": 5,
    "max_pool_skills": 15,
    "max_chars": 4000,
    "max_points_per_skill": 8,
    "max_points_per_skill_inject": 0,
    "max_chars_per_point": 240,
    "ttl_turns": 20,
    "inject_on_turn": True,
    "skip_if_in_history": True,
    "history_lookback": 40,
    "hydrate_from_history": True,
    "hydrate_limit": 30,
    "semantic_prefetch": True,
    "semantic_min_score": 1,
    "use_hermes_hot_markers": True,
    "extract_sections": True,
    "fallback_extract": True,
    # Load/save pool to disk so a new conversation (new AIAgent) continues the
    # same key points as the next user turn in one session — useful for
    # sequential benchmark tasks. Default off.
    "persist_across_conversations": False,
    "persist_path": "",
}

_VALID_ENTRY_SCHEDULES = frozenset({"per_skill", "global_pool"})
_PERSIST_SCHEMA = "hermes.hot_skill_pool.v1"


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
    cfg["enabled"] = bool(cfg.get("enabled", False))
    schedule = str(cfg.get("entry_schedule", "global_pool") or "global_pool").strip().lower()
    cfg["entry_schedule"] = schedule if schedule in _VALID_ENTRY_SCHEDULES else "global_pool"
    cfg["max_entries"] = max(0, int(cfg.get("max_entries", 12) or 0))
    cfg["max_skills"] = max(0, int(cfg.get("max_skills", 5) or 0))
    cfg["max_pool_skills"] = max(0, int(cfg.get("max_pool_skills", 15) or 0))
    cfg["max_chars"] = max(0, int(cfg.get("max_chars", 4000) or 0))
    cfg["max_points_per_skill"] = max(1, int(cfg.get("max_points_per_skill", 8) or 8))
    cfg["max_points_per_skill_inject"] = max(
        0, int(cfg.get("max_points_per_skill_inject", 0) or 0)
    )
    cfg["max_chars_per_point"] = max(32, int(cfg.get("max_chars_per_point", 240) or 240))
    cfg["ttl_turns"] = max(0, int(cfg.get("ttl_turns", 20) or 0))
    cfg["history_lookback"] = max(1, int(cfg.get("history_lookback", 40) or 40))
    cfg["hydrate_limit"] = max(1, int(cfg.get("hydrate_limit", 30) or 30))
    cfg["semantic_min_score"] = max(0, int(cfg.get("semantic_min_score", 1) or 0))
    cfg["use_hermes_hot_markers"] = bool(cfg.get("use_hermes_hot_markers", True))
    cfg["extract_sections"] = bool(cfg.get("extract_sections", True))
    cfg["fallback_extract"] = bool(cfg.get("fallback_extract", True))
    cfg["persist_across_conversations"] = bool(cfg.get("persist_across_conversations", False))
    cfg["persist_path"] = str(cfg.get("persist_path") or "").strip()
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
    last_used_turn: int = 0
    description: str = ""
    tags: List[str] = field(default_factory=list)


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
    injections_attempted: int = 0
    injections_nonempty: int = 0
    first_nonempty_inject_iter: Optional[int] = None
    skill_view_total: int = 0
    skill_view_repeat: int = 0
    skill_view_while_in_pool: int = 0
    skill_manage_sync: int = 0
    skill_manage_evict: int = 0
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
        }


def parse_hot_pool_inner_meta(inner: str) -> tuple[List[str], List[str]]:
    """Extract skill headers and bullet points from a build_block inner body."""
    skills: List[str] = []
    points: List[str] = []
    for line in (inner or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            skills.append(stripped[4:].strip())
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
    """LRU pool of recently used skill key points (session or persisted)."""

    def __init__(self, config: Optional[dict] = None) -> None:
        if isinstance(config, dict) and "hot_pool" not in config and "enabled" in config:
            merged = dict(_DEFAULT_HOT_POOL)
            merged.update(config)
            self._config = merged
            self._normalize_config()
        else:
            self._config = load_hot_skills_config(config if isinstance(config, dict) else None)
        self._entries: "OrderedDict[str, HotSkillEntry]" = OrderedDict()
        self._global_turn: int = 0
        self._active_turn: int = 0
        self._telemetry = HotPoolTelemetry()
        self._telemetry_turn_started = False
        self._pool_at_turn_start: Set[str] = set()
        if self._persist_enabled():
            self._load_persisted()

    def _normalize_config(self) -> None:
        cfg = self._config
        cfg["enabled"] = bool(cfg.get("enabled", False))
        schedule = str(cfg.get("entry_schedule", "global_pool") or "global_pool").strip().lower()
        cfg["entry_schedule"] = schedule if schedule in _VALID_ENTRY_SCHEDULES else "global_pool"
        cfg["max_entries"] = max(0, int(cfg.get("max_entries", 12) or 0))
        cfg["max_skills"] = max(0, int(cfg.get("max_skills", 5) or 0))
        cfg["max_pool_skills"] = max(0, int(cfg.get("max_pool_skills", 15) or 0))
        cfg["max_chars"] = max(0, int(cfg.get("max_chars", 4000) or 0))
        cfg["max_points_per_skill"] = max(1, int(cfg.get("max_points_per_skill", 8) or 8))
        cfg["max_points_per_skill_inject"] = max(
            0, int(cfg.get("max_points_per_skill_inject", 0) or 0)
        )
        cfg["max_chars_per_point"] = max(32, int(cfg.get("max_chars_per_point", 240) or 240))
        cfg["ttl_turns"] = max(0, int(cfg.get("ttl_turns", 20) or 0))
        cfg["history_lookback"] = max(1, int(cfg.get("history_lookback", 40) or 40))
        cfg["hydrate_limit"] = max(1, int(cfg.get("hydrate_limit", 30) or 30))
        cfg["semantic_min_score"] = max(0, int(cfg.get("semantic_min_score", 1) or 0))
        cfg["use_hermes_hot_markers"] = bool(cfg.get("use_hermes_hot_markers", True))
        cfg["extract_sections"] = bool(cfg.get("extract_sections", True))
        cfg["fallback_extract"] = bool(cfg.get("fallback_extract", True))
        cfg["persist_across_conversations"] = bool(cfg.get("persist_across_conversations", False))
        cfg["persist_path"] = str(cfg.get("persist_path") or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self._config.get("enabled"))

    @property
    def config(self) -> dict:
        return dict(self._config)

    @property
    def active_turn(self) -> int:
        """Turn counter used for TTL and last_used (global when persisted)."""
        return self._active_turn

    @property
    def global_turn(self) -> int:
        return self._global_turn

    def _persist_enabled(self) -> bool:
        return bool(self._config.get("persist_across_conversations"))

    def _effective_turn(self, session_turn: int) -> int:
        if self._persist_enabled():
            return self._global_turn
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
            "entry_schedule": self._config.get("entry_schedule"),
            "persist_across_conversations": self._persist_enabled(),
            "skip_if_in_history": bool(self._config.get("skip_if_in_history", True)),
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

        mtime = _skill_md_mtime(skill_dir)
        tag_list = [str(t) for t in (tags or []) if t]
        entry = HotSkillEntry(
            name=key,
            key_points=points,
            skill_dir=skill_dir,
            skill_md_mtime=mtime,
            last_used_turn=self._effective_turn(turn),
            description=(description or "").strip(),
            tags=tag_list,
        )
        self._entries.pop(key, None)
        self._entries[key] = entry
        self._trim_to_max_skills()
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
            before = len(self._entries)
            self.record_from_tool_result(content, turn=0)
            if name in self._entries and len(self._entries) > before:
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
            self._maybe_persist()
        return refreshed

    def evict_by_ttl(self, current_turn: Optional[int] = None) -> int:
        ttl = int(self._config.get("ttl_turns", 0) or 0)
        if ttl <= 0:
            return 0
        turn = int(current_turn if current_turn is not None else self._active_turn)
        evicted = 0
        for key, entry in list(self._entries.items()):
            if entry.last_used_turn <= 0:
                continue
            if turn - entry.last_used_turn > ttl:
                self._entries.pop(key, None)
                evicted += 1
        if evicted:
            self._maybe_persist()
        return evicted

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
        if not self.enabled or not self._config.get("inject_on_turn", True):
            return ""
        if not self._entries:
            return ""

        exclude = set(exclude_names or ())
        if exclude:
            self._telemetry.excluded_skills = sorted(exclude)
        candidates = [
            entry for entry in self._entries.values()
            if entry.name not in exclude and entry.key_points
        ]
        if not candidates:
            return ""

        ranked = self._rank_entries(candidates, user_message)
        max_chars = int(self._config.get("max_chars", 4000) or 0)
        if max_chars <= 0:
            return ""

        schedule = self._config.get("entry_schedule", "global_pool")
        if schedule == "global_pool":
            inner = self._build_inner_global_pool(ranked, max_chars, turn)
        else:
            inner = self._build_inner_per_skill(ranked, max_chars, turn)

        if not inner:
            return ""
        skills, points = parse_hot_pool_inner_meta(inner)
        block = build_hot_skills_block(inner)
        self._telemetry.build_block_applied = True
        self._telemetry.skills_injected = skills
        self._telemetry.points_injected = points
        self._telemetry.point_count = len(points)
        self._telemetry.chars = len(block)
        self._maybe_persist()
        return block

    def _build_inner_per_skill(
        self,
        ranked: List[HotSkillEntry],
        max_chars: int,
        turn: int,
    ) -> str:
        max_skills = int(self._config.get("max_skills", 5) or 0)
        if max_skills <= 0:
            return ""

        parts: List[str] = []
        used_chars = 0
        count = 0
        per_skill_cap = int(self._config.get("max_points_per_skill", 8) or 8)
        for entry in ranked:
            if count >= max_skills:
                break
            points = entry.key_points[:per_skill_cap]
            if not points:
                continue
            header = f"### {entry.name}\n"
            bullets = "\n".join(f"- {pt}" for pt in points)
            chunk = header + bullets
            remaining = max_chars - used_chars
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                if remaining < len(header) + 32:
                    break
                chunk = chunk[: remaining - 20].rstrip() + "\n\n[... truncated ...]"
            parts.append(chunk)
            used_chars += len(chunk)
            count += 1
            entry.last_used_turn = max(entry.last_used_turn, int(turn or 0))

        if not parts:
            return ""
        inner = "\n\n---\n\n".join(parts)
        if max_chars and len(inner) > max_chars:
            inner = inner[: max_chars - 20].rstrip() + "\n\n[... truncated ...]"
        return inner

    def _build_inner_global_pool(
        self,
        ranked: List[HotSkillEntry],
        max_chars: int,
        turn: int,
    ) -> str:
        max_entries = int(self._config.get("max_entries", 12) or 0)
        if max_entries <= 0:
            return ""

        per_skill_inject = int(self._config.get("max_points_per_skill_inject", 0) or 0)
        grouped: "OrderedDict[str, List[str]]" = OrderedDict()
        entries_touched: Set[str] = set()
        used_chars = 0
        entry_count = 0

        for entry in ranked:
            skill_count = 0
            for pt in entry.key_points:
                if entry_count >= max_entries:
                    break
                if per_skill_inject > 0 and skill_count >= per_skill_inject:
                    break

                if entry.name not in grouped:
                    prefix = ("\n\n---\n\n" if grouped else "") + f"### {entry.name}\n"
                    add_cost = len(prefix) + len(f"- {pt}")
                else:
                    add_cost = len(f"\n- {pt}")

                if used_chars + add_cost > max_chars:
                    if not grouped:
                        break
                    continue

                if entry.name not in grouped:
                    grouped[entry.name] = []
                grouped[entry.name].append(pt)
                used_chars += add_cost
                entry_count += 1
                skill_count += 1
                entries_touched.add(entry.name)

            if entry_count >= max_entries:
                break

        for name in entries_touched:
            entry = self._entries.get(name)
            if entry is not None:
                entry.last_used_turn = max(entry.last_used_turn, int(turn or 0))

        if not grouped:
            return ""

        parts = []
        for name, points in grouped.items():
            parts.append(f"### {name}\n" + "\n".join(f"- {pt}" for pt in points))
        inner = "\n\n---\n\n".join(parts)
        if max_chars and len(inner) > max_chars:
            inner = inner[: max_chars - 20].rstrip() + "\n\n[... truncated ...]"
        return inner

    def _rank_entries(self, entries: List[HotSkillEntry], user_message: str) -> List[HotSkillEntry]:
        ordered = list(reversed(entries))
        if not self._config.get("semantic_prefetch", True):
            return ordered

        query_tokens = _tokenize(user_message)
        if not query_tokens:
            return ordered

        scored: List[tuple[int, int, HotSkillEntry]] = []
        for idx, entry in enumerate(ordered):
            score = _semantic_score(query_tokens, entry)
            scored.append((score, idx, entry))

        scored.sort(key=lambda t: (-t[0], t[1]))
        return [entry for _, _, entry in scored]

    def _trim_to_max_skills(self) -> None:
        schedule = self._config.get("entry_schedule", "global_pool")
        if schedule == "global_pool":
            cap = int(self._config.get("max_pool_skills", 15) or 0)
        else:
            cap = int(self._config.get("max_skills", 5) or 0)
        if cap <= 0:
            return
        while len(self._entries) > cap:
            self._entries.popitem(last=False)

    def _entry_to_dict(self, entry: HotSkillEntry) -> dict:
        return {
            "name": entry.name,
            "key_points": list(entry.key_points),
            "skill_dir": entry.skill_dir,
            "skill_md_mtime": entry.skill_md_mtime,
            "last_used_turn": entry.last_used_turn,
            "description": entry.description,
            "tags": list(entry.tags),
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
        return HotSkillEntry(
            name=name,
            key_points=clean_points,
            skill_dir=data.get("skill_dir"),
            skill_md_mtime=float(data.get("skill_md_mtime") or 0.0),
            last_used_turn=int(data.get("last_used_turn") or 0),
            description=str(data.get("description") or ""),
            tags=[str(t) for t in tags if t] if isinstance(tags, list) else [],
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
            "lru_order": list(self._entries.keys()),
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


def _tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def _semantic_score(query_tokens: Set[str], entry: HotSkillEntry) -> int:
    if not query_tokens:
        return 0
    score = 0
    name_tokens = _tokenize(entry.name.replace("/", " ").replace("-", " "))
    desc_tokens = _tokenize(entry.description)
    tag_tokens = _tokenize(" ".join(entry.tags))
    point_tokens = _tokenize(" ".join(entry.key_points))
    for tok in query_tokens:
        if tok in name_tokens:
            score += 4
        if tok in tag_tokens:
            score += 2
        if tok in point_tokens:
            score += 3
        if tok in desc_tokens:
            score += 1
        if len(tok) >= 4 and tok in entry.name.lower():
            score += 2
    return score


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
        "Use skill_view(name) for full procedures.]\n\n"
        f"{clean}\n"
        f"{_HOT_SKILLS_CLOSE}"
    )
