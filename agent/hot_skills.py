"""Hot skill pool — admission-capped key points, injected as a utility-aware subset.

Extracts guardrails from SKILL.md in this order:

1. ``## Common Pitfalls`` / ``## Pitfalls`` / ``## Key Points``-style sections
   (what bundled and agent-authored skills actually use)
2. Optional ``<!-- hermes-hot -->`` markers (rare hand override)
3. Heuristic NEVER/ALWAYS-style bullets elsewhere

Extract rewrites structural identifiers (paths, emails, UUIDs, hex IDs).
Transferability is a side-channel LLM judgment (reconcile / outcome), not a
semantic regex filter. Mechanical extract is followed by deterministic junk
and ritual filters (episode closers / short-answer procedures), then
``reconcile_pool`` (LLM keep-set when configured, else oldest only on
overflow).

The pool cap is the store budget (``max_entries`` points). Under policy
``llm``, reconcile runs on every material pool update (admit / skill sync /
stale refresh), not only when over cap — so a full pool can still drop weak
tips. Admit also applies a lexical domain gate (no LLM): closed-world
platforms (AppWorld / ALFWorld) refuse distinctive off-domain skills;
open-world (SkillsBench / CLI) fail-open except framework identities.
Inject is a subset: ``inject_k`` tips selected by retrieve-lexicon
overlap with a frozen episode query, then utility. Off-domain skills with
empty intersection are omitted. Strongly harmful utilities are omitted when
outcome utilities exist. Irrelevant scores only rank; they do not omit.
Model "use" of a point is not observable; policies are ``llm`` (default:
side-channel reconcile judge) or ``oldest`` (FIFO fallback / explicit
opt-in). Full procedures stay behind ``skill_view``.
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
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,63}", re.IGNORECASE)

_HOT_SKILLS_OPEN = "<hot-skills>"
_HOT_SKILLS_CLOSE = "</hot-skills>"
_HOT_SKILLS_BLOCK_RE = re.compile(
    r"<\s*hot-skills\s*>[\s\S]*?</\s*hot-skills\s*>",
    re.IGNORECASE,
)
_HOT_SKILLS_TAG_RE = re.compile(r"</?\s*hot-skills\s*>", re.IGNORECASE)
# Matches the current inject preface and older short notes in persisted history.
_HOT_SKILLS_NOTE_RE = re.compile(
    r"\[System note:\s*The following are hot skill key points \(guardrails\)"
    r"[\s\S]*?\]\s*",
    re.IGNORECASE,
)
_HOT_SKILLS_SYSTEM_NOTE = (
    "[System note: The following are hot skill key points (guardrails) "
    "from recently used skills, NOT new user input. "
    "ALWAYS/NEVER/MUST bullets are exception handlers for when that "
    "situation arises — not a first-action checklist. Do not tour "
    "locations, IDs, files, or APIs because a tip mentions them; use "
    "the current observation and admissible actions first. Apply a tip "
    "when its Scope: fits this task; otherwise ignore it. "
    "Use skill_view(name) for full procedures.]"
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

# Structural identifier shapes only — not English semantics.
_PATH_RE = re.compile(
    r"(?:"
    r"~(?:/[\w.+\-]+)+/?"
    r"|/(?:tmp|home|var|opt|usr|etc|Users)(?:/[\w.+\-]*)*/?"
    r"|[A-Za-z]:\\[\w.\\+\-]+"
    r")"
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_HEX_ID_RE = re.compile(r"\b[0-9a-f]{16,}\b", re.IGNORECASE)
_PLACEHOLDER_ONLY_RE = re.compile(
    r"(?i)^[\s,.;:]*((?:<(?:path|id|email)>|and|or|the|a|an|to|on|in|at|of)[\s,.;:]*)+$"
)

# Meta / authoring skills — never admit to the broadcast pool.
_DEFAULT_EXCLUDE_SKILLS_FROM_POOL = frozenset(
    {
        "hermes-agent-skill-authoring",
        "hermes-agent",
    }
)

# Bundled ``skills/media/*`` — real MCP / external API integrations (not ``apis.*`` REPL).
_DEFAULT_MCP_MEDIA_SKILLS_FROM_POOL = frozenset(
    {
        "spotify",
        "gif-search",
        "heartmula",
        "youtube-content",
        "songsee",
    }
)

# Sandbox REPL platforms where MCP media tips are wrong-domain (e.g. ``apis.spotify``).
_PLATFORM_DENY_MCP_MEDIA_SKILLS = frozenset({"appworld-batch"})

# Agent final-response phrases that falsely claim pass/completion.
_CLAIMED_PASS_RE = re.compile(
    r"(?i)\b("
    r"all tests passed|all \d+ tests passed|task (?:is )?complete|"
    r"successfully completed|verification passed|fully passed|"
    r"tests pass(?:ed)?(?: successfully)?|macro success"
    r")\b"
)

# Template bullets from SKILL.md examples (not real guardrails).
_JUNK_POINT_EXACT = frozenset(
    {
        "important details",
        "```",
        "---",
        "...",
    }
)

_JUNK_POINT_PREFIX_RES = (
    re.compile(r"^trigger conditions\b", re.IGNORECASE),
    re.compile(r"^numbered steps\b", re.IGNORECASE),
    re.compile(r"^key points section\b", re.IGNORECASE),
    re.compile(r"^keep it short\b", re.IGNORECASE),
    re.compile(r"^test — load with\b", re.IGNORECASE),
    re.compile(r"^test - load with\b", re.IGNORECASE),
    re.compile(r"^don'?t put session progress\b", re.IGNORECASE),
)

# Oracle / harness shortcuts that do not transfer across benchmark tasks.
_JUNK_POINT_SUBSTRING_RES = (
    re.compile(r"pre-existing solution", re.IGNORECASE),
    re.compile(r"reference solution", re.IGNORECASE),
    re.compile(r"\bsolution/solve\.sh\b", re.IGNORECASE),
    re.compile(r"static ground truth", re.IGNORECASE),
    re.compile(r"skip heavy build", re.IGNORECASE),
    re.compile(r"ground truth.*from.*test", re.IGNORECASE),
    re.compile(r"read.*expected.*from.*test", re.IGNORECASE),
)

# Episode / harness procedures — not transferable policy. NEVER/DO NOT about
# the closer is a constraint (keep); ALWAYS/MUST/bare "call done()" is a ritual.
_RITUAL_FORBID_PREFIX_RE = re.compile(
    r"^(?:\*\*)?(?:NEVER|DO\s+NOT|MUST\s+NOT|AVOID)\b(?!\s+forget)",
    re.IGNORECASE,
)
_RITUAL_CLOSER_RES = (
    re.compile(
        r"(?i)\b(?:always|must|remember\s+to|don'?t\s+forget\s+to|"
        r"never\s+forget\s+to|be\s+sure\s+to)\b.{0,48}"
        r"\b(?:call|invoke|run|use|issue|submit|execute)\b.{0,24}"
        r"\b(?:done|complete_task)\s*\("
    ),
    re.compile(
        r"(?i)\b(?:always|must)\b.{0,24}\b(?:done|complete_task)\s*\("
    ),
    re.compile(
        r"(?i)\b(?:call|invoke|run|issue|submit)\s+"
        r"(?:the\s+)?(?:env(?:ironment)?(?:'s)?\s+)?"
        r"(?:done|complete_task)\s*\("
    ),
    re.compile(r"(?i)\bcomplete_task\s*\(\s*answer\s*="),
    re.compile(
        r"(?i)\b(?:always|must)\b.{0,32}\b(?:done\s+action|action\s+done)\b"
    ),
)
_RITUAL_SHORT_ANSWER_RES = (
    re.compile(
        r"(?i)\b(?:return|give|write|put|pass|include)\b.{0,28}"
        r"\b(?:a\s+)?(?:short|brief|minimal|concise)\s+"
        r"(?:answer|response|final(?:\s+answer)?)\b"
    ),
    re.compile(
        r"(?i)\b(?:short|brief|minimal|concise)\s+(?:answer|response)\b.{0,48}"
        r"\b(?:complete_task|done\s*\(|finalizer|harness|evaluator)\b"
    ),
    re.compile(
        r"(?i)\bkeep\s+(?:the\s+|your\s+)?"
        r"(?:answer|response|final\s+(?:answer|response)|output)\s+short\b"
    ),
    re.compile(
        r"(?i)\b(?:answer|response)\s+should\s+be\s+"
        r"(?:short|brief|minimal|concise)\b"
    ),
)

# ALWAYS/MUST process rituals (doc-first / exploratory API tours) — burn steps
# without stating a transferable constraint. NEVER/DO NOT about the same
# action stays a constraint. "ALWAYS paginate" / "ALWAYS use X formula" kept.
_RITUAL_PROCEDURE_RES = (
    re.compile(r"(?i)\b(?:always|must)\b.{0,48}\bshow_api_doc\b"),
    re.compile(r"(?i)\b(?:always|must)\b.{0,48}\bshow_api_descriptions?\b"),
    re.compile(
        r"(?i)\b(?:always|must)\b.{0,48}\b(?:check|read|open|consult|look\s+up)\b"
        r".{0,32}\b(?:api\s+)?(?:docs?|documentation|spec(?:ification)?s?)\b"
        r".{0,48}\bbefore\b"
    ),
    re.compile(
        r"(?i)\b(?:always|must)\b.{0,32}\b(?:call|invoke|run|use)\b.{0,48}"
        r"\bshow_api_(?:doc|descriptions?)\b"
    ),
)

_DEFAULT_HOT_POOL = {
    "enabled": True,
    # Store budget (key points). Inject is a separate top-k subset.
    "max_entries": 12,
    # Per-turn inject budget. 0 = no cap (all tips that pass the skill gate).
    "inject_k": 4,
    # Lexical skill/tip gate against a frozen episode query (no LLM).
    "inject_retrieve": True,
    # Deprecated / ignored: inject size is inject_k, not a character budget.
    "max_chars": 0,
    "max_points_per_skill": 8,
    "max_chars_per_point": 240,
    # Pool curation when over max_entries or on material updates (admit/sync).
    # llm (default): side-channel reconcile judge — keep a broadcast-worthy
    #      subset (runs on overflow and on each material update when
    #      reconcile_on_update is true). Falls back to oldest if missing/fails.
    # oldest: drop earliest recorded_turn only when over max_entries.
    "eviction_policy": "llm",
    # Run LLM reconcile after admit/sync even when the pool is under cap.
    "reconcile_on_update": True,
    # Per-tip side-channel gate before admit (transfer vs episode-local).
    "admit_transfer_gate": True,
    # Lexical skill-vs-session domain gate at admit (no LLM). Closed-world
    # platforms omit distinctive off-domain skills; SkillsBench/CLI fail-open
    # except framework identities. Does not evict persist on episode change.
    "admit_domain_gate": True,
    # Drop template/meta bullets after extract (see filter_junk_key_points).
    "junk_filter": True,
    "junk_min_point_chars": 12,
    # Drop episode-procedure tips (always call done(), short answer,
    # ALWAYS show_api_doc before every call, …).
    "ritual_filter": True,
    # Extra skill names denied admission (lowercased); merged with built-in denylist.
    "exclude_skills_from_pool": [],
    "inject_on_turn": True,
    # Pause inject when inject-cohort success lags no-inject cohort.
    "inject_throttle": True,
    "inject_throttle_min_episodes": 8,
    "inject_throttle_margin": 0.05,
    # Cadence for control-arm holdout (pre-throttle) and inject probes
    # (while throttled), in labeled episodes.
    "inject_throttle_probe_every": 10,
    # Once a tip has this many labels, omit it at inject when score < 0.
    # Unlabeled tips stay eligible; silence when nothing viable remains.
    "inject_score_min_labeled": 5,
    "skip_if_in_history": True,
    "history_lookback": 40,
    "hydrate_from_history": True,
    "hydrate_limit": 30,
    "use_hermes_hot_markers": True,
    "extract_sections": True,
    "fallback_extract": True,
    "persist_across_conversations": False,
    "persist_path": "",
    # After a labeled task/episode, attribute exposed tips and update
    # multi-dimensional utilities (default on). Used at admit/evict and inject.
    "outcome_feedback": True,
    # Opt-in only. Empty/unparseable judge JSON must not write utilities
    # (cloned win/loss is a broken credit-assignment loop).
    "outcome_feedback_heuristic": False,
    # Success with env/API steps at or below this counts as "low-step" helpful.
    "outcome_helpful_max_iterations": 12,
    # Side-channel completion budget for per-tip outcome JSON. Thinking
    # models need headroom after thinking is disabled; 512 was too small.
    "outcome_judge_max_tokens": 2048,
    # Include a compressed tool/env action log in the outcome judge payload
    # (no observations / API dumps).
    "outcome_judge_log": True,
    "outcome_judge_log_max_events": 48,
    "outcome_judge_log_arg_chars": 96,
    # Rewrite structural identifiers (paths, emails, UUIDs, hex IDs).
    "abstract_extract": True,
    # Omit strongly harmful tips at inject (store still retains them).
    # Irrelevant is ranked, not omitted — lock-in would hide transferable tips.
    "inject_filter_utilities": True,
}

_VALID_EVICTION_POLICIES = frozenset({"oldest", "llm"})
_DEFAULT_EVICTION_POLICY = "llm"
_PERSIST_SCHEMA = "hermes.hot_skill_pool.v1"

# Optional test/agent hook: (items, keep_n, context) -> list[id str] to keep.
EvictionJudge = Callable[[List[Dict[str, Any]], int, str], Optional[List[str]]]
# Side-channel gate: (items, skill_name, scope, context) -> ids to admit.
AdmissionJudge = Callable[
    [List[Dict[str, Any]], str, str, str], Optional[List[str]]
]


def _normalize_hot_pool_cfg(cfg: dict) -> dict:
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["max_entries"] = max(0, int(cfg.get("max_entries", 12) or 0))
    cfg["inject_k"] = max(0, int(cfg.get("inject_k", 4) or 0))
    cfg["inject_retrieve"] = bool(cfg.get("inject_retrieve", True))
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
    cfg["inject_throttle"] = bool(cfg.get("inject_throttle", True))
    cfg["inject_throttle_min_episodes"] = max(
        2, int(cfg.get("inject_throttle_min_episodes", 8) or 8)
    )
    cfg["inject_throttle_margin"] = max(
        0.0, float(cfg.get("inject_throttle_margin", 0.05) or 0.05)
    )
    cfg["inject_throttle_probe_every"] = max(
        1, int(cfg.get("inject_throttle_probe_every", 10) or 10)
    )
    cfg["inject_score_min_labeled"] = max(
        1, int(cfg.get("inject_score_min_labeled", 5) or 5)
    )
    cfg["skip_if_in_history"] = bool(cfg.get("skip_if_in_history", True))
    cfg["hydrate_from_history"] = bool(cfg.get("hydrate_from_history", True))
    cfg["outcome_feedback"] = bool(cfg.get("outcome_feedback", True))
    cfg["outcome_feedback_heuristic"] = bool(cfg.get("outcome_feedback_heuristic", False))
    cfg["ritual_filter"] = bool(cfg.get("ritual_filter", True))
    cfg["outcome_helpful_max_iterations"] = max(
        1, int(cfg.get("outcome_helpful_max_iterations", 12) or 12)
    )
    cfg["outcome_judge_max_tokens"] = max(
        256, int(cfg.get("outcome_judge_max_tokens", 2048) or 2048)
    )
    cfg["outcome_judge_log"] = bool(cfg.get("outcome_judge_log", True))
    cfg["outcome_judge_log_max_events"] = max(
        4, int(cfg.get("outcome_judge_log_max_events", 48) or 48)
    )
    cfg["outcome_judge_log_arg_chars"] = max(
        24, int(cfg.get("outcome_judge_log_arg_chars", 96) or 96)
    )
    cfg["abstract_extract"] = bool(cfg.get("abstract_extract", True))
    cfg["inject_filter_utilities"] = bool(cfg.get("inject_filter_utilities", True))
    cfg["reconcile_on_update"] = bool(cfg.get("reconcile_on_update", True))
    cfg["admit_transfer_gate"] = bool(cfg.get("admit_transfer_gate", True))
    cfg["admit_domain_gate"] = bool(cfg.get("admit_domain_gate", True))
    cfg["junk_filter"] = bool(cfg.get("junk_filter", True))
    cfg["junk_min_point_chars"] = max(
        4, int(cfg.get("junk_min_point_chars", 12) or 12)
    )
    raw_exclude = cfg.get("exclude_skills_from_pool")
    if isinstance(raw_exclude, (list, tuple)):
        cfg["exclude_skills_from_pool"] = [
            str(s).strip() for s in raw_exclude if str(s).strip()
        ]
    else:
        cfg["exclude_skills_from_pool"] = []
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

    if cfg.get("abstract_extract", True):
        points = abstract_hot_key_points(points, cfg)
    points = _normalize_key_points(points, cfg)
    points = filter_junk_key_points(points, cfg)
    return filter_ritual_key_points(points, cfg)


def skill_has_hot_section(content: str) -> bool:
    """True if SKILL.md has a Pitfalls / Best Practices / Key Points-style heading."""
    text = content or ""
    fenced = _fenced_char_ranges(text)
    for match in _HEADING_RE.finditer(text):
        if _position_in_fenced_region(match.start(), fenced):
            continue
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
    # Soft guidance; inject also ranks/filters by point_utilities when enabled.
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
    """Conservative admit/inject filter: clear harmful majority or low success."""
    n = int(util.get("n_labeled") or 0)
    if n < min_n:
        return False
    harmful = int(util.get("harmful") or 0)
    helpful = int(util.get("helpful") or 0)
    if harmful >= 2 and harmful >= helpful + 2:
        return True
    # Empirical success rate — independent of (possibly inflated) helpful labels.
    if n >= 5:
        success_rate = float(util.get("success_sum") or 0.0) / n
        if success_rate <= 0.45 and harmful + int(util.get("irrelevant") or 0) >= helpful:
            return True
    return False


def _utility_inject_score(util: dict) -> float:
    """Higher is better. Unlabeled tips stay neutral so they still inject.

    Successful but slow episodes sink a tip slightly so search-policy noise
    does not stay tied with a short-path constraint. Low empirical success
    rate sinks tips even when the judge over-labeled helpful.
    """
    n = int(util.get("n_labeled") or 0)
    if n <= 0:
        return 0.0
    helpful = int(util.get("helpful") or 0)
    harmful = int(util.get("harmful") or 0)
    irrelevant = int(util.get("irrelevant") or 0)
    score = helpful - 1.5 * harmful - 0.75 * irrelevant
    success_rate = float(util.get("success_sum") or 0.0) / n
    score += 3.0 * (success_rate - 0.5)
    n_success = int(util.get("n_success") or 0)
    if n_success > 0:
        avg_it = float(util.get("iterations_when_success_sum") or 0.0) / n_success
        score -= 0.08 * max(0.0, avg_it - 8.0)
    return score


def _redact_structural_ids(text: str) -> str:
    out = _PATH_RE.sub("<path>", text)
    out = _EMAIL_RE.sub("<email>", out)
    out = _UUID_RE.sub("<id>", out)
    out = _HEX_ID_RE.sub("<id>", out)
    return out


def _cleanup_abstracted(text: str) -> str:
    out = re.sub(r"\s+", " ", text or "").strip()
    out = re.sub(r"(?:\s*[,;:]){2,}", ",", out)
    out = re.sub(r"^[\s,;:.]+|[\s,;:]+$", "", out)
    out = re.sub(r"\s+([.,;:])", r"\1", out)
    return out.strip()


def abstract_hot_key_point(raw: str) -> Optional[str]:
    """Redact structural identifiers. Does not judge English meaning."""
    original = re.sub(r"\s+", " ", (raw or "").strip())
    if not original:
        return None
    text = _cleanup_abstracted(_redact_structural_ids(original))
    if not text or _PLACEHOLDER_ONLY_RE.match(text):
        return None
    return text


def abstract_hot_key_points(points: List[str], config: Optional[dict] = None) -> List[str]:
    """Redact structural identifiers and drop empty leftovers."""
    del config  # reserved; redaction is deterministic
    out: List[str] = []
    seen: Set[str] = set()
    for raw in points or []:
        pt = abstract_hot_key_point(raw)
        if not pt:
            continue
        key = pt.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(pt)
    return out


def _platform_exclude_skills(config: dict) -> Set[str]:
    """Skill names denied for the active runtime platform (e.g. MCP media on REPL batch)."""
    platform = str(config.get("runtime_platform") or "").strip().lower()
    if platform in _PLATFORM_DENY_MCP_MEDIA_SKILLS:
        return {s.lower() for s in _DEFAULT_MCP_MEDIA_SKILLS_FROM_POOL}
    return set()


def is_oracle_style_tip(text: str) -> bool:
    """True when a tip encodes oracle/harness shortcuts (non-transferable)."""
    pt = re.sub(r"\s+", " ", (text or "").strip())
    if not pt:
        return False
    for pat in _JUNK_POINT_SUBSTRING_RES:
        if pat.search(pt):
            return True
    return False


def is_excluded_hot_skill(name: str, config: Optional[dict] = None) -> bool:
    """True when a skill must never enter the hot pool (meta/authoring / platform deny)."""
    cfg = config if isinstance(config, dict) else load_hot_skills_config()
    key = (name or "").strip().lower()
    if not key:
        return False
    deny = {s.lower() for s in _DEFAULT_EXCLUDE_SKILLS_FROM_POOL}
    deny.update(_platform_exclude_skills(cfg))
    extra = cfg.get("exclude_skills_from_pool") or []
    if isinstance(extra, (list, tuple)):
        deny.update(str(s).strip().lower() for s in extra if str(s).strip())
    return key in deny


def is_junk_key_point(text: str, config: Optional[dict] = None) -> bool:
    """Deterministic filter for template/scaffold bullets after extract."""
    cfg = config if isinstance(config, dict) else load_hot_skills_config()
    if not cfg.get("junk_filter", True):
        return False
    pt = re.sub(r"\s+", " ", (text or "").strip())
    if not pt:
        return True
    if re.fullmatch(r"`+", pt):
        return True
    key = pt.lower()
    if key in _JUNK_POINT_EXACT:
        return True
    min_len = int(cfg.get("junk_min_point_chars", 12) or 12)
    if len(pt) < min_len and not _GUARDRAIL_PREFIX_RE.match(pt):
        return True
    for pat in _JUNK_POINT_PREFIX_RES:
        if pat.search(pt):
            return True
    for pat in _JUNK_POINT_SUBSTRING_RES:
        if pat.search(pt):
            return True
    return False


def filter_junk_key_points(
    points: List[str], config: Optional[dict] = None
) -> List[str]:
    """Drop template/meta bullets; preserve order."""
    cfg = config if isinstance(config, dict) else load_hot_skills_config()
    if not cfg.get("junk_filter", True):
        return list(points or [])
    out: List[str] = []
    seen: Set[str] = set()
    for raw in points or []:
        pt = re.sub(r"\s+", " ", (raw or "").strip())
        if not pt or is_junk_key_point(pt, cfg):
            continue
        norm = pt.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(pt)
    return out


def is_ritual_key_point(text: str) -> bool:
    """True when a tip is an episode closer / verbosity / process ritual.

    ALWAYS/MUST/bare ``call done()``, ``return a short answer``, and
    ``ALWAYS show_api_doc before…`` do not transfer as broadcast tips.
    NEVER/DO NOT about the same action is a constraint and is kept.
    """
    pt = re.sub(r"\s+", " ", (text or "").strip())
    if not pt:
        return False
    closerish = bool(
        re.search(r"(?i)\b(?:done|complete_task)\s*\(", pt)
        or re.search(r"(?i)\b(?:done\s+action|action\s+done)\b", pt)
        or re.search(r"(?i)\bcomplete_task\b", pt)
    )
    if closerish and _RITUAL_FORBID_PREFIX_RE.match(pt):
        return False
    for pat in _RITUAL_CLOSER_RES:
        if pat.search(pt):
            return True
    for pat in _RITUAL_SHORT_ANSWER_RES:
        if pat.search(pt):
            return True
    # Doc-first / exploratory API tours — forbid-prefix keeps NEVER variants.
    if not _RITUAL_FORBID_PREFIX_RE.match(pt):
        for pat in _RITUAL_PROCEDURE_RES:
            if pat.search(pt):
                return True
    return False


def filter_ritual_key_points(
    points: List[str], config: Optional[dict] = None
) -> List[str]:
    """Drop episode-procedure tips; preserve order."""
    cfg = config if isinstance(config, dict) else load_hot_skills_config()
    if not cfg.get("ritual_filter", True):
        return list(points or [])
    out: List[str] = []
    seen: Set[str] = set()
    for raw in points or []:
        pt = re.sub(r"\s+", " ", (raw or "").strip())
        if not pt or is_ritual_key_point(pt):
            continue
        norm = pt.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(pt)
    return out


def clamp_outcome_attributions(
    items: List[Dict[str, Any]],
    labels: Dict[str, str],
    outcome: dict,
) -> Tuple[Dict[str, str], int]:
    """Hard floor: injected tips cannot be helpful on task failure.

    On ``success=False`` or ``claimed_success_mismatch``, every ``via=inject``
    tip is forced to ``harmful`` (overwriting helpful/irrelevant). History-only
    exposure stays irrelevant when unlabeled. Returns (labels, n_clamped).
    """
    success = bool(outcome.get("success"))
    claimed_mismatch = bool(outcome.get("claimed_success_mismatch"))
    if success and not claimed_mismatch:
        return dict(labels or {}), 0
    out = dict(labels or {})
    clamped = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        iid = str(item.get("id") or "")
        if not iid:
            continue
        via = str(item.get("via") or "")
        if via != "inject":
            out.setdefault(iid, "irrelevant")
            continue
        prev = out.get(iid)
        if prev != "harmful":
            clamped += 1
        out[iid] = "harmful"
    return out, clamped


def heuristic_outcome_attributions(
    items: List[Dict[str, Any]],
    outcome: dict,
    *,
    low_step_threshold: int = 12,
) -> Dict[str, str]:
    """Label exposed tips when the attribution LLM is missing or empty.

    Per-point and exposure-aware: only tips actually injected this turn
    (``via == "inject"``) may be ``helpful`` on success with low steps.
    Injected tips on failure or claimed-success mismatch are ``harmful``.
    History-only exposure (``via == "history"``) is ``irrelevant``.
    """
    success = bool(outcome.get("success"))
    claimed_mismatch = bool(outcome.get("claimed_success_mismatch"))
    iterations = None
    if outcome.get("iterations") is not None:
        try:
            iterations = float(outcome["iterations"])
        except (TypeError, ValueError):
            iterations = None
    low_steps = (
        success and iterations is not None and iterations <= float(low_step_threshold)
    )
    labels: Dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        iid = str(item.get("id") or "")
        if not iid:
            continue
        via = str(item.get("via") or "")
        if via != "inject":
            labels[iid] = "irrelevant"
            continue
        if claimed_mismatch:
            labels[iid] = "harmful"
        elif not success:
            labels[iid] = "harmful"
        elif low_steps:
            labels[iid] = "helpful"
        else:
            labels[iid] = "irrelevant"
    return labels


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
    if detect_claimed_success_mismatch(ev, res):
        outcome["claimed_success_mismatch"] = True
    return outcome


def detect_claimed_success_mismatch(
    evaluation: Optional[dict],
    run_result: Optional[dict],
) -> bool:
    """True when the agent claimed pass/completion but labeled eval failed."""
    ev = evaluation if isinstance(evaluation, dict) else {}
    res = run_result if isinstance(run_result, dict) else {}
    if bool(ev.get("task_success") if "task_success" in ev else ev.get("success")):
        return False
    final = str(res.get("final_response") or "").strip()
    if not final:
        return False
    return bool(_CLAIMED_PASS_RE.search(final))


_RETRIEVE_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_RETRIEVE_SPLIT_RE = re.compile(r"[-_/.:]+")
_RETRIEVE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "from",
        "with",
        "by",
        "as",
        "is",
        "are",
        "be",
        "do",
        "not",
        "no",
        "if",
        "then",
        "this",
        "that",
        "it",
        "its",
        "your",
        "you",
        "we",
        "our",
        "my",
        "always",
        "never",
        "must",
        "should",
        "check",
        "use",
        "using",
        "used",
        "call",
        "called",
        "return",
        "returned",
        "just",
        "only",
        "also",
        "before",
        "after",
        "when",
        "than",
        "into",
        "via",
        "per",
        "task",
        "skill",
        "please",
        "hello",
        "hi",
        "ok",
        "yes",
        "all",
        "any",
        "each",
        "both",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "can",
        "will",
        "may",
        "dont",
        "does",
        "did",
        "has",
        "have",
        "had",
        "was",
        "were",
        "been",
        "being",
        "them",
        "they",
        "their",
    }
)
_OBS_USER_PREFIX_RE = re.compile(
    r"(?is)^\s*(?:output|observation|stdout|stderr)\s*:",
)
_PLATFORM_DOMAIN_ALIASES: Dict[str, FrozenSet[str]] = {
    "appworld-batch": frozenset({"appworld", "sandbox", "apis", "api", "repl"}),
    "alfworld-batch": frozenset({"alfworld", "household", "receptacle"}),
    "skillsbench-batch": frozenset({"skillsbench"}),
}

# One environment / API surface — empty ∩ + distinctive identity ⇒ foreign.
_CLOSED_WORLD_PLATFORMS = frozenset({"appworld-batch", "alfworld-batch"})

# Product / framework tokens. Open-world omits only these when the session
# does not mention them. Do not add task-domain names (github, xlsx, pdf):
# those are legitimate SkillsBench skills.
_FRAMEWORK_IDENTITY_TOKENS = frozenset(
    {
        "hermes",
        "claude",
        "anthropic",
        "opencode",
        "vscode",
        "copilot",
        "gateway",
    }
)

# Identity leftovers that do not name a competing domain. Closed-world
# fail-open when identity ⊆ this set so agent-authored generic skills
# (nav, helper) still admit on ALFWorld / AppWorld.
_GENERIC_IDENTITY_TOKENS = frozenset(
    {
        "nav",
        "helper",
        "helpers",
        "guide",
        "runner",
        "notes",
        "note",
        "general",
        "common",
        "basic",
        "demo",
        "test",
        "tests",
        "bench",
        "batch",
        "tool",
        "tools",
        "env",
        "environment",
        "process",
        "practice",
        "practices",
        "tip",
        "tips",
        "rule",
        "rules",
        "usage",
        "workflow",
        "workflows",
        "utility",
        "utilities",
        "core",
        "main",
        "misc",
        "default",
        "standard",
        "simple",
        "navigation",
        "navigate",
        "explore",
        "exploration",
        "look",
        "agent",
        "assistant",
        "proc",
    }
)


def tokenize_retrieve(text: str, *, min_len: int = 3) -> Set[str]:
    """Lowercased content tokens for retrieve intersection (no stemming)."""
    if not text:
        return set()
    split = _RETRIEVE_SPLIT_RE.sub(" ", str(text).lower())
    out: Set[str] = set()
    min_len = max(1, int(min_len or 1))
    for match in _RETRIEVE_TOKEN_RE.finditer(split):
        tok = match.group(0)
        if len(tok) < min_len or tok in _RETRIEVE_STOPWORDS:
            continue
        out.add(tok)
    return out


def is_observation_user_message(text: str) -> bool:
    """True when *text* is env/tool output, not a new episode instruction."""
    raw = str(text or "").lstrip()
    if not raw:
        return True
    if raw.startswith("```"):
        return True
    return bool(_OBS_USER_PREFIX_RE.match(raw))


def skill_identity_tokens(
    name: str,
    tags: Optional[Iterable[str]] = None,
    scope: str = "",
    description: str = "",
) -> FrozenSet[str]:
    """Name / catalog tags / scope — used for platform-domain fallback."""
    toks = tokenize_retrieve(name, min_len=3)
    for tag in tags or ():
        toks |= tokenize_retrieve(str(tag), min_len=3)
    toks |= tokenize_retrieve(scope, min_len=3)
    toks |= tokenize_retrieve(description, min_len=3)
    return frozenset(toks)


def tip_retrieve_tokens(point: str) -> FrozenSet[str]:
    return frozenset(tokenize_retrieve(point, min_len=3))


def skill_retrieve_tokens(
    name: str,
    tags: Optional[Iterable[str]] = None,
    scope: str = "",
    description: str = "",
    points: Optional[Iterable[str]] = None,
) -> FrozenSet[str]:
    """Admit-time retrieve lexicon: identity tokens plus tip nouns."""
    toks = set(skill_identity_tokens(name, tags, scope, description))
    for point in points or ():
        toks |= tip_retrieve_tokens(str(point))
    return frozenset(toks)


def platform_retrieve_tokens(platform: str) -> FrozenSet[str]:
    key = (platform or "").strip().lower()
    toks = tokenize_retrieve(key, min_len=3)
    toks |= set(_PLATFORM_DOMAIN_ALIASES.get(key) or ())
    return frozenset(toks)


_SIDECANNEL_LOG_PREVIEW_CHARS = 240
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def sidechannel_text_preview(text: str, *, max_chars: int = _SIDECANNEL_LOG_PREVIEW_CHARS) -> str:
    """Single-line preview for JSONL telemetry (not a full transcript)."""
    n = max(0, int(max_chars or 0))
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if n == 0 or len(cleaned) <= n:
        return cleaned
    return cleaned[:n]


def recover_json_object(text: str) -> str:
    """First JSON object in *text*, including inside think blocks."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    match = _JSON_OBJECT_RE.search(raw)
    return match.group(0).strip() if match else ""


def sidechannel_thinking_off_extra_body(
    *,
    model: str = "",
    is_openrouter: bool = False,
    is_nous: bool = False,
    is_kimi: bool = False,
    is_custom_provider: bool = False,
) -> Dict[str, Any]:
    """Provider extras that disable thinking on a tools-free JSON judge.

    Official OpenAI chat.completions gets an empty dict (unknown extra_body
    keys can 400). Qwen/vLLM, OpenRouter, Kimi, and custom/Ollama get the
    flags those stacks actually honor.
    """
    extra: Dict[str, Any] = {}
    model_l = (model or "").lower()
    if "qwen" in model_l:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
        extra["enable_thinking"] = False
    if is_openrouter or is_nous:
        extra["reasoning"] = {"enabled": False}
    if is_kimi:
        extra["thinking"] = {"type": "disabled"}
    if is_custom_provider:
        extra["think"] = False
    return extra


def retrieve_overlap(
    query: FrozenSet[str],
    lexicon: FrozenSet[str],
    *,
    ignore: Optional[FrozenSet[str]] = None,
) -> FrozenSet[str]:
    """Query ∩ lexicon, minus tokens that sit on every skill (IDF)."""
    hits = set(query or ()) & set(lexicon or ())
    if ignore:
        hits -= set(ignore)
    return frozenset(hits)


def is_closed_world_platform(platform: str) -> bool:
    """True for single-environment batch platforms (AppWorld, ALFWorld)."""
    return (platform or "").strip().lower() in _CLOSED_WORLD_PLATFORMS


def session_domain_tokens(
    platform: str,
    query_tokens: Optional[Iterable[str]] = None,
) -> FrozenSet[str]:
    """Platform aliases ∪ frozen episode-query tokens."""
    toks = set(platform_retrieve_tokens(platform))
    for tok in query_tokens or ():
        piece = str(tok).strip().lower()
        if piece:
            toks.add(piece)
    return frozenset(toks)


def foreign_closed_world_tokens(platform: str) -> FrozenSet[str]:
    """Alias tokens that belong to a different closed-world platform."""
    key = (platform or "").strip().lower()
    mine = set(platform_retrieve_tokens(key))
    foreign: Set[str] = set()
    for plat in _CLOSED_WORLD_PLATFORMS:
        if plat == key:
            continue
        foreign |= set(platform_retrieve_tokens(plat))
    return frozenset(foreign - mine)


def is_off_session_domain(
    name: str,
    *,
    tags: Optional[Iterable[str]] = None,
    scope: str = "",
    description: str = "",
    points: Optional[Iterable[str]] = None,
    platform: str = "",
    query_tokens: Optional[Iterable[str]] = None,
) -> bool:
    """True when *name* is a foreign domain for this session.

    Fail-open when the session has no domain signal, when identity/tips
    overlap the session, or when a closed-world skill is generic
    (``nav`` / ``helper``). SkillsBench / CLI omit only framework-shaped
    identities the query did not mention — so a task skill with a weak
    name still admits. No LLM.
    """
    session = session_domain_tokens(platform, query_tokens)
    if not session:
        return False

    identity = skill_identity_tokens(name, tags, scope, description)
    lexicon = skill_retrieve_tokens(name, tags, scope, description, points)
    session_fs = frozenset(session)
    if retrieve_overlap(session_fs, identity):
        return False
    if retrieve_overlap(session_fs, lexicon):
        return False

    if is_closed_world_platform(platform):
        distinctive = identity - _GENERIC_IDENTITY_TOKENS
        return bool(distinctive)

    return bool(identity & _FRAMEWORK_IDENTITY_TOKENS)


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
    records_skipped_excluded_skill: int = 0
    records_skipped_off_domain: int = 0
    records_junk_filtered: int = 0
    records_ritual_filtered: int = 0
    records_admit_gate_dropped: int = 0
    evicted_deterministic: int = 0
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
    points_omitted_utility: int = 0
    points_omitted_retrieve: int = 0
    points_omitted_ritual: int = 0
    skills_omitted_retrieve: List[str] = field(default_factory=list)
    retrieve_mode: str = ""
    episode_query_preview: str = ""
    injections_attempted: int = 0
    injections_nonempty: int = 0
    first_nonempty_inject_iter: Optional[int] = None
    skill_view_total: int = 0
    skill_view_repeat: int = 0
    skill_view_while_in_pool: int = 0
    skill_manage_sync: int = 0
    skill_manage_evict: int = 0
    evicted_capacity: int = 0
    reconcile_ran: bool = False
    outcome_feedback_applied: bool = False
    outcome_feedback_points: int = 0
    outcome_attributions: Dict[str, int] = field(default_factory=dict)
    outcome_feedback_skipped_reason: str = ""
    outcome_feedback_attribution_source: str = ""
    outcome_judge_finish_reason: str = ""
    outcome_judge_raw_chars: int = 0
    outcome_judge_stripped_chars: int = 0
    outcome_judge_raw_preview: str = ""
    outcome_judge_stripped_preview: str = ""
    outcome_judge_max_tokens: int = 0
    outcome_judge_thinking_off: bool = False
    outcome_judge_retries: int = 0
    outcome_judge_recovered_json: bool = False
    outcome_judge_log_events: int = 0
    outcome_judge_log_preview: str = ""
    outcome_labels_clamped: int = 0
    inject_throttled: bool = False
    inject_throttle_probe: bool = False
    inject_cohort_inject_n: int = 0
    inject_cohort_inject_success: int = 0
    inject_cohort_no_inject_n: int = 0
    inject_cohort_no_inject_success: int = 0
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
                "records_skipped_excluded_skill": self.records_skipped_excluded_skill,
                "records_skipped_off_domain": self.records_skipped_off_domain,
                "records_junk_filtered": self.records_junk_filtered,
                "records_ritual_filtered": self.records_ritual_filtered,
                "records_admit_gate_dropped": self.records_admit_gate_dropped,
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
                "points_omitted_utility": self.points_omitted_utility,
                "points_omitted_retrieve": self.points_omitted_retrieve,
                "points_omitted_ritual": self.points_omitted_ritual,
                "skills_omitted_retrieve": list(self.skills_omitted_retrieve),
                "retrieve_mode": self.retrieve_mode,
                "episode_query_preview": self.episode_query_preview,
                "injections_attempted": self.injections_attempted,
                "injections_nonempty": self.injections_nonempty,
                "first_nonempty_inject_iter": self.first_nonempty_inject_iter,
                "throttled": self.inject_throttled,
                "throttle_probe": self.inject_throttle_probe,
                "cohort_inject_n": self.inject_cohort_inject_n,
                "cohort_inject_success": self.inject_cohort_inject_success,
                "cohort_no_inject_n": self.inject_cohort_no_inject_n,
                "cohort_no_inject_success": self.inject_cohort_no_inject_success,
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
                "deterministic": self.evicted_deterministic,
                "reconcile_ran": self.reconcile_ran,
            },
            "outcome_feedback": {
                "applied": self.outcome_feedback_applied,
                "points_scored": self.outcome_feedback_points,
                "attributions": dict(self.outcome_attributions),
                "skipped_reason": self.outcome_feedback_skipped_reason,
                "attribution_source": self.outcome_feedback_attribution_source,
                "labels_clamped": self.outcome_labels_clamped,
                "judge_finish_reason": self.outcome_judge_finish_reason,
                "judge_raw_chars": self.outcome_judge_raw_chars,
                "judge_stripped_chars": self.outcome_judge_stripped_chars,
                "judge_raw_preview": self.outcome_judge_raw_preview,
                "judge_stripped_preview": self.outcome_judge_stripped_preview,
                "judge_max_tokens": self.outcome_judge_max_tokens,
                "judge_thinking_off": self.outcome_judge_thinking_off,
                "judge_retries": self.outcome_judge_retries,
                "judge_recovered_json": self.outcome_judge_recovered_json,
                "judge_log_events": self.outcome_judge_log_events,
                "judge_log_preview": self.outcome_judge_log_preview,
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


_BULKY_ARG_KEYS = frozenset(
    {
        "content",
        "output",
        "stdout",
        "stderr",
        "observation",
        "result",
        "html",
        "text",
        "body",
        "data",
        "messages",
        "transcript",
    }
)
_COMMAND_ARG_KEYS = frozenset({"command", "cmd", "code", "script", "source"})
_API_CALL_RE = re.compile(
    r"(?:apis\.\w[\w.]*|complete_task)\s*\([^)]{0,80}",
    re.IGNORECASE,
)


def _head_tail(items: List[Any], max_n: int, *, head: int = 8) -> Tuple[List[Any], int]:
    if max_n <= 0 or len(items) <= max_n:
        return list(items), 0
    tail = max(1, max_n - min(head, max_n // 2))
    keep_head = max_n - tail
    return list(items[:keep_head]) + list(items[-tail:]), len(items) - max_n


def compress_tool_args(args: Any, *, max_chars: int = 96) -> Any:
    """Keep short structured args; drop observation-sized blobs."""
    max_chars = max(16, int(max_chars or 96))
    if isinstance(args, str):
        raw = args.strip()
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return sidechannel_text_preview(raw, max_chars=max_chars)
        return compress_tool_args(parsed, max_chars=max_chars)
    if isinstance(args, dict):
        slim: Dict[str, Any] = {}
        for key, val in args.items():
            k = str(key)
            if k.lower() in _BULKY_ARG_KEYS:
                continue
            if isinstance(val, str):
                if k.lower() in _COMMAND_ARG_KEYS:
                    first = val.strip().splitlines()[0] if val.strip() else ""
                    slim[k] = first[:max_chars]
                else:
                    slim[k] = val[:max_chars]
            elif isinstance(val, (int, float, bool)) or val is None:
                slim[k] = val
            # skip nested dict/list blobs
        return slim
    if isinstance(args, (int, float, bool)) or args is None:
        return args
    return sidechannel_text_preview(str(args), max_chars=max_chars)


def compress_env_code(code: str, *, max_snippets: int = 3, max_chars: int = 80) -> List[str]:
    """AppWorld-style snippets: apis.* / complete_task, else first code line."""
    max_chars = max(16, int(max_chars or 80))
    snippets: List[str] = []
    for match in _API_CALL_RE.finditer(code or ""):
        snippets.append(match.group(0)[:max_chars])
        if len(snippets) >= max_snippets:
            return snippets
    if snippets:
        return snippets
    for line in str(code or "").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            return [text[:max_chars]]
    return []


def build_compressed_episode_log(
    *,
    messages: Optional[List[dict]] = None,
    run_result: Optional[dict] = None,
    episode_query: str = "",
    env_actions: Optional[Iterable[Any]] = None,
    max_events: int = 48,
    max_arg_chars: int = 96,
    max_final_chars: int = 240,
    max_query_chars: int = 160,
) -> Dict[str, Any]:
    """Compressed tool/env actions for the outcome judge — no observations.

    Sources, when present:
    - assistant ``tool_calls`` in *messages* (SkillsBench / CLI)
    - ``run_result.env_actions`` / ``actions`` / ``steps`` (ALFWorld actions,
      AppWorld ``code`` snippets)
    - frozen *episode_query* and truncated ``final_response``
    """
    max_events = max(1, int(max_events or 48))
    max_arg_chars = max(16, int(max_arg_chars or 96))
    res = run_result if isinstance(run_result, dict) else {}
    events: List[Dict[str, Any]] = []
    tool_counts: Dict[str, int] = {}

    def _add(via: str, name: str, args: Any = "") -> None:
        key = (name or via or "event").strip() or "event"
        tool_counts[key] = tool_counts.get(key, 0) + 1
        events.append({"via": via, "name": key, "args": args})

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "").strip()
            if not name or name in _HOUSEKEEPING_TOOLS:
                continue
            _add("tool", name, compress_tool_args(fn.get("arguments") or "", max_chars=max_arg_chars))

    extra_actions = env_actions
    if extra_actions is None:
        extra_actions = res.get("env_actions")
        if extra_actions is None:
            extra_actions = res.get("actions")
    for raw in extra_actions or ():
        if isinstance(raw, dict):
            action = raw.get("action") or raw.get("name") or ""
            code = raw.get("code") or ""
            if action:
                _add("env", "action", sidechannel_text_preview(str(action), max_chars=max_arg_chars))
            elif code:
                snippets = compress_env_code(str(code), max_chars=max_arg_chars)
                if snippets:
                    _add("code", snippets[0], snippets[1:] if len(snippets) > 1 else "")
                else:
                    _add("code", "execute", "")
            continue
        text = str(raw or "").strip()
        if text:
            _add("env", "action", sidechannel_text_preview(text, max_chars=max_arg_chars))

    for step in res.get("steps") or ():
        if not isinstance(step, dict):
            continue
        if step.get("action"):
            _add(
                "env",
                "action",
                sidechannel_text_preview(str(step.get("action")), max_chars=max_arg_chars),
            )
        elif step.get("code"):
            snippets = compress_env_code(str(step.get("code")), max_chars=max_arg_chars)
            if snippets:
                _add("code", snippets[0], snippets[1:] if len(snippets) > 1 else "")

    query = (episode_query or "").strip()
    if not query:
        for msg in messages or []:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = str(msg.get("content") or "")
            if content and not is_observation_user_message(content):
                query = content.strip()
                break

    final = str(res.get("final_response") or "").strip()
    if not final:
        for msg in reversed(list(messages or [])):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            text = str(msg.get("content") or "").strip()
            if text:
                final = text
                break

    kept, n_truncated = _head_tail(events, max_events)
    numbered = []
    for i, ev in enumerate(kept, start=1):
        numbered.append({"i": i, **ev})

    return {
        "query": sidechannel_text_preview(query, max_chars=max_query_chars),
        "final_response": sidechannel_text_preview(final, max_chars=max_final_chars),
        "events": numbered,
        "tool_counts": dict(sorted(tool_counts.items())),
        "n_events": len(events),
        "n_truncated": n_truncated,
    }


def collect_outcome_episode_log(
    *,
    agent: Any = None,
    run_result: Optional[dict] = None,
    pool: Any = None,
    messages: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Build the judge log from agent / run_result / pool query."""
    cfg = {}
    if pool is not None:
        cfg = getattr(pool, "config", None) or getattr(pool, "_config", None) or {}
        if not isinstance(cfg, dict):
            cfg = {}
    if cfg and not cfg.get("outcome_judge_log", True):
        return {}
    res = run_result if isinstance(run_result, dict) else {}
    msgs = messages
    if msgs is None and isinstance(res.get("messages"), list):
        msgs = res.get("messages")
    if msgs is None and agent is not None:
        candidate = getattr(agent, "messages", None)
        if isinstance(candidate, list):
            msgs = candidate
    query = ""
    if pool is not None:
        query = str(getattr(pool, "_episode_query", "") or "")
    return build_compressed_episode_log(
        messages=msgs,
        run_result=res,
        episode_query=query,
        max_events=int(cfg.get("outcome_judge_log_max_events", 48) or 48),
        max_arg_chars=int(cfg.get("outcome_judge_log_arg_chars", 96) or 96),
    )


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

    ``max_entries`` is the store cap. Eviction happens when a new extract
    overflows that cap. Admit applies a lexical domain gate before the
    transfer tip gate. Inject selects up to ``inject_k`` tips by retrieve
    overlap with a frozen episode query, optionally omitting strongly
    harmful ones (``inject_filter_utilities``) and skipping skills already
    in recent ``skill_view`` history.
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
        self.admission_judge: Optional[AdmissionJudge] = None
        self._telemetry = HotPoolTelemetry()
        self._telemetry_turn_started = False
        self._pool_at_turn_start: Set[str] = set()
        # Tips exposed this episode (injected or skipped via history) for outcome feedback.
        self._exposed_tips: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()
        self._episode_query: str = ""
        self._episode_query_tokens: FrozenSet[str] = frozenset()
        self._lexicon_dirty = True
        self._skill_lexicons: Dict[str, FrozenSet[str]] = {}
        self._tip_lexicons: Dict[Tuple[str, int], FrozenSet[str]] = {}
        self._identity_lexicons: Dict[str, FrozenSet[str]] = {}
        self._idf_ignore: FrozenSet[str] = frozenset()
        # Rolling inject vs no-inject success (persisted when enabled).
        self._cohort_inject_n: int = 0
        self._cohort_inject_success: int = 0
        self._cohort_no_inject_n: int = 0
        self._cohort_no_inject_success: int = 0
        self._inject_throttled: bool = False
        self._episodes_since_inject: int = 0
        self._last_block_had_inject: bool = False
        if self._persist_enabled():
            self._load_persisted()

    def set_platform(self, platform: str) -> None:
        """Set runtime platform for deny rules; evict entries that no longer qualify.

        Evicts name-denylist / MCP-media skills and, when the domain gate is
        on, persist entries whose *identity* is another closed-world or a
        framework. Does not empty-∩ evict (that would wipe ALFWorld
        household persist at every agent init).
        """
        self._config["runtime_platform"] = (platform or "").strip()
        evicted = 0
        plat = str(self._config.get("runtime_platform") or "")
        foreign = foreign_closed_world_tokens(plat)
        domain_gate = bool(self._config.get("admit_domain_gate", True))
        closed = is_closed_world_platform(plat)
        for name in list(self._entries.keys()):
            if is_excluded_hot_skill(name, self._config):
                self.evict(name)
                evicted += 1
                continue
            if not domain_gate:
                continue
            entry = self._entries.get(name)
            if entry is None:
                continue
            ident = skill_identity_tokens(
                entry.name, entry.tags, entry.scope, entry.description
            )
            if ident & foreign:
                self.evict(name)
                evicted += 1
                continue
            if closed and (ident & _FRAMEWORK_IDENTITY_TOKENS):
                self.evict(name)
                evicted += 1
        if evicted:
            self._maybe_persist()

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
        self.note_episode_query(text)

    def note_episode_query(self, text: str) -> None:
        """Freeze retrieve query on the first real instruction; ignore observations."""
        raw = str(text or "").strip()
        if not raw or is_observation_user_message(raw):
            return
        self._episode_query = raw
        self._episode_query_tokens = frozenset(tokenize_retrieve(raw, min_len=3))

    def clear_episode_query(self) -> None:
        self._episode_query = ""
        self._episode_query_tokens = frozenset()

    def _session_query_tokens(
        self, user_message: Optional[str] = None
    ) -> FrozenSet[str]:
        """Frozen episode tokens, else a real instruction — never observations."""
        if self._episode_query_tokens:
            return self._episode_query_tokens
        raw = user_message if user_message is not None else self._admission_context
        if raw and not is_observation_user_message(raw):
            return frozenset(tokenize_retrieve(raw, min_len=3))
        return frozenset()

    def _skill_off_session_domain(
        self,
        name: str,
        *,
        tags: Optional[Iterable[str]] = None,
        scope: str = "",
        description: str = "",
        points: Optional[Iterable[str]] = None,
        user_message: Optional[str] = None,
    ) -> bool:
        if not self._config.get("admit_domain_gate", True):
            return False
        return is_off_session_domain(
            name,
            tags=tags,
            scope=scope,
            description=description,
            points=points,
            platform=str(self._config.get("runtime_platform") or ""),
            query_tokens=self._session_query_tokens(user_message),
        )

    def _mark_lexicons_dirty(self) -> None:
        self._lexicon_dirty = True

    def _ensure_lexicons(self) -> None:
        if not self._lexicon_dirty:
            return
        skill_lex: Dict[str, FrozenSet[str]] = {}
        tip_lex: Dict[Tuple[str, int], FrozenSet[str]] = {}
        ident_lex: Dict[str, FrozenSet[str]] = {}
        for name, entry in self._entries.items():
            ident_lex[name] = skill_identity_tokens(
                entry.name, entry.tags, entry.scope, entry.description
            )
            skill_lex[name] = skill_retrieve_tokens(
                entry.name,
                entry.tags,
                entry.scope,
                entry.description,
                entry.key_points,
            )
            for idx, point in enumerate(entry.key_points):
                tip_lex[(name, idx)] = tip_retrieve_tokens(point)
        ignore: Set[str] = set()
        if len(skill_lex) >= 2:
            common = set.intersection(*(set(v) for v in skill_lex.values()))
            ignore = common
        self._skill_lexicons = skill_lex
        self._tip_lexicons = tip_lex
        self._identity_lexicons = ident_lex
        self._idf_ignore = frozenset(ignore)
        self._lexicon_dirty = False

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
            "outcome_feedback_heuristic": bool(
                self._config.get("outcome_feedback_heuristic", False)
            ),
            "ritual_filter": bool(self._config.get("ritual_filter", True)),
            "abstract_extract": bool(self._config.get("abstract_extract", True)),
            "inject_filter_utilities": bool(
                self._config.get("inject_filter_utilities", True)
            ),
            "inject_k": int(self._config.get("inject_k", 4) or 0),
            "inject_retrieve": bool(self._config.get("inject_retrieve", True)),
            "inject_throttle": bool(self._config.get("inject_throttle", True)),
            "inject_throttle_probe_every": int(
                self._config.get("inject_throttle_probe_every", 10) or 10
            ),
            "inject_score_min_labeled": int(
                self._config.get("inject_score_min_labeled", 5) or 5
            ),
            "outcome_judge_max_tokens": int(
                self._config.get("outcome_judge_max_tokens", 2048) or 2048
            ),
        }
        return out

    def clear(self) -> None:
        self._entries.clear()
        self._mark_lexicons_dirty()
        self._maybe_persist()

    def evict(self, name: str) -> None:
        key = (name or "").strip()
        if key and self._entries.pop(key, None) is not None:
            self._mark_lexicons_dirty()
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
        if is_excluded_hot_skill(key, self._config):
            logger.debug("hot skill pool: skill %r excluded from pool", key)
            self._telemetry.records_skipped_excluded_skill += 1
            return

        points = list(key_points) if key_points is not None else extract_hot_key_points(content, self._config)
        if key_points is not None and self._config.get("abstract_extract", True):
            points = abstract_hot_key_points(points, self._config)
            points = _normalize_key_points(points, self._config)
            before = len(points)
            points = filter_junk_key_points(points, self._config)
            if before > len(points):
                self._telemetry.records_junk_filtered += before - len(points)
        before_ritual = len(points)
        points = filter_ritual_key_points(points, self._config)
        if before_ritual > len(points):
            self._telemetry.records_ritual_filtered += before_ritual - len(points)
        if not points:
            logger.debug("hot skill pool: no key points extracted for %r — skipping", key)
            self._telemetry.records_skipped_no_points += 1
            return

        tag_list = [str(t) for t in (tags or []) if t]
        desc = (description or "").strip()
        scope = derive_hot_scope(key, description=desc, tags=tag_list)
        context = user_message if user_message is not None else self._admission_context
        if self._skill_off_session_domain(
            key,
            tags=tag_list,
            scope=scope,
            description=desc,
            points=points,
            user_message=user_message,
        ):
            logger.debug(
                "hot skill pool: skill %r off session domain — skipping admit",
                key,
            )
            self._telemetry.records_skipped_off_domain += 1
            return
        before_gate = len(points)
        points = self._gate_admission_points(
            key,
            points,
            scope=scope,
            context=context or "",
        )
        if before_gate > len(points):
            self._telemetry.records_admit_gate_dropped += before_gate - len(points)
        if not points:
            logger.debug(
                "hot skill pool: no transferable key points for %r after admit gate — skipping",
                key,
            )
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
        self._mark_lexicons_dirty()
        self.reconcile_pool(new_name=key, context=context or "", material_update=True)
        self._telemetry.new_records_this_task += 1
        if self._inject_throttled:
            # New pool material deserves a fresh inject trial window.
            self._inject_throttled = False
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
            self._mark_lexicons_dirty()
        if refreshed:
            self.reconcile_pool(
                new_name="",
                context=self._admission_context,
                material_update=True,
            )
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
        """Serialize a retrieve + utility subset of retained tips."""
        if user_message:
            self.set_admission_context(user_message)
        self._last_block_had_inject = False
        self._telemetry.inject_throttled = False
        self._telemetry.inject_throttle_probe = False
        self._sync_cohort_telemetry()
        if not self.enabled or not self._config.get("inject_on_turn", True):
            return ""
        if not self._entries:
            return ""

        allow_inject, is_probe = self._inject_allowed_this_turn()
        self._telemetry.inject_throttled = bool(self._inject_throttled and not allow_inject)
        self._telemetry.inject_throttle_probe = bool(is_probe)
        if not allow_inject:
            # Still record history-skipped exposure for outcome feedback.
            exclude = set(exclude_names or ())
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
            for entry in skipped:
                for text in entry.key_points:
                    if not self._is_injectable_point(text):
                        continue
                    key = (entry.name, text)
                    self._exposed_tips[key] = {
                        "skill": entry.name,
                        "point": text,
                        "scope": entry.scope or "",
                        "via": "history",
                    }
            self._telemetry.build_block_applied = False
            self._telemetry.skills_injected = []
            self._telemetry.points_injected = []
            self._telemetry.point_count = 0
            self._telemetry.chars = 0
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
        self._telemetry.points_omitted_utility = 0
        self._telemetry.points_omitted_retrieve = 0
        self._telemetry.points_omitted_ritual = 0
        self._telemetry.skills_omitted_retrieve = []
        self._telemetry.retrieve_mode = ""
        preview = self._episode_query.replace("\n", " ").strip()
        self._telemetry.episode_query_preview = preview[:160]
        # Track exposure for outcome feedback (injected + history-skipped).
        for entry in skipped:
            for text in entry.key_points:
                if not self._is_injectable_point(text):
                    continue
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

        views = self._inject_point_views(candidates)
        # Budget is max_entries (store points), not chars — providers bill
        # tokens; a second char truncate would silently drop tips the overflow
        # judge already chose to keep.
        inner = self._format_entry_views(views)
        if not inner:
            return ""
        skills, points = parse_hot_pool_inner_meta(inner)
        block = build_hot_skills_block(inner)
        self._telemetry.build_block_applied = True
        self._telemetry.skills_injected = skills
        self._telemetry.points_injected = points
        self._telemetry.point_count = len(points)
        self._telemetry.chars = len(block)
        for entry, selected in views:
            for text in selected:
                key = (entry.name, text)
                self._exposed_tips[key] = {
                    "skill": entry.name,
                    "point": text,
                    "scope": entry.scope or "",
                    "via": "inject",
                }
        self._last_block_had_inject = bool(points)
        if self._last_block_had_inject:
            self._episodes_since_inject = 0
        return block

    def _sync_cohort_telemetry(self) -> None:
        tel = self._telemetry
        tel.inject_cohort_inject_n = self._cohort_inject_n
        tel.inject_cohort_inject_success = self._cohort_inject_success
        tel.inject_cohort_no_inject_n = self._cohort_no_inject_n
        tel.inject_cohort_no_inject_success = self._cohort_no_inject_success
        tel.inject_throttled = self._inject_throttled

    def _inject_cohort_rates(self) -> Tuple[Optional[float], Optional[float]]:
        inj_n = self._cohort_inject_n
        no_n = self._cohort_no_inject_n
        inj_rate = (
            self._cohort_inject_success / inj_n if inj_n > 0 else None
        )
        no_rate = (
            self._cohort_no_inject_success / no_n if no_n > 0 else None
        )
        return inj_rate, no_rate

    def _recompute_inject_throttle(self) -> None:
        if not self._config.get("inject_throttle", True):
            self._inject_throttled = False
            return
        min_n = int(self._config.get("inject_throttle_min_episodes", 8) or 8)
        margin = float(self._config.get("inject_throttle_margin", 0.05) or 0.05)
        if self._cohort_inject_n < min_n or self._cohort_no_inject_n < min_n:
            return
        inj_rate, no_rate = self._inject_cohort_rates()
        if inj_rate is None or no_rate is None:
            return
        if inj_rate < no_rate - margin:
            self._inject_throttled = True
        elif inj_rate >= no_rate:
            self._inject_throttled = False

    def _inject_allowed_this_turn(self) -> Tuple[bool, bool]:
        """Return (allow_inject, is_probe).

        While throttled: skip inject except an inject probe every N no-inject
        episodes. While not throttled: mostly inject, but force a no-inject
        holdout every N labeled episodes so the control arm can fill and
        throttling can actually arm.
        """
        if not self._config.get("inject_throttle", True):
            return True, False
        self._recompute_inject_throttle()
        probe_every = int(self._config.get("inject_throttle_probe_every", 10) or 10)
        if self._inject_throttled:
            if self._episodes_since_inject >= probe_every:
                return True, True
            return False, False
        # Pre-throttle holdout: after every N completed labeled episodes,
        # skip inject once to observe the no-inject cohort.
        labeled = self._cohort_inject_n + self._cohort_no_inject_n
        if probe_every > 0 and labeled > 0 and labeled % probe_every == 0:
            return False, True
        return True, False

    def _note_inject_cohort(self, *, injected: bool, success: bool) -> None:
        if injected:
            self._cohort_inject_n += 1
            if success:
                self._cohort_inject_success += 1
            self._episodes_since_inject = 0
        else:
            self._cohort_no_inject_n += 1
            if success:
                self._cohort_no_inject_success += 1
            self._episodes_since_inject += 1
        self._recompute_inject_throttle()
        self._sync_cohort_telemetry()
        self._maybe_persist()

    def _aligned_utils(self, entry: HotSkillEntry) -> List[Dict[str, Any]]:
        utils = list(entry.point_utilities or [])
        while len(utils) < len(entry.key_points):
            utils.append(empty_point_utility())
        return utils

    def _is_injectable_point(self, text: str) -> bool:
        if self._config.get("ritual_filter", True) and is_ritual_key_point(text):
            return False
        return True

    def _filter_inject_points(self, entry: HotSkillEntry) -> Tuple[List[str], int, int]:
        utils = self._aligned_utils(entry)
        scored: List[Tuple[str, float]] = []
        omitted = 0
        ritual_omitted = 0
        min_labeled = int(self._config.get("inject_score_min_labeled", 5) or 5)
        for text, util in zip(entry.key_points, utils):
            if not self._is_injectable_point(text):
                ritual_omitted += 1
                continue
            if _utility_strongly_harmful(util):
                omitted += 1
                continue
            score = _utility_inject_score(util)
            n_labeled = int(util.get("n_labeled") or 0)
            # Silence over broadcast: enough evidence + negative score → omit.
            if n_labeled >= min_labeled and score < 0.0:
                omitted += 1
                continue
            scored.append((text, score))
        scored.sort(key=lambda item: -item[1])
        return [text for text, _ in scored], omitted, ritual_omitted

    def _inject_point_views(
        self, entries: List[HotSkillEntry]
    ) -> List[Tuple[HotSkillEntry, List[str]]]:
        if not self._config.get("inject_filter_utilities", True):
            ranked: List[Tuple[HotSkillEntry, List[str]]] = []
            ritual_omitted = 0
            for entry in entries:
                kept = [t for t in entry.key_points if self._is_injectable_point(t)]
                ritual_omitted += len(entry.key_points) - len(kept)
                if kept:
                    ranked.append((entry, kept))
            self._telemetry.points_omitted_utility = 0
            self._telemetry.points_omitted_ritual = ritual_omitted
            return self._select_retrieve_views(ranked)

        views: List[Tuple[HotSkillEntry, List[str]]] = []
        omitted = 0
        ritual_omitted = 0
        for entry in entries:
            selected, n_omit, n_ritual = self._filter_inject_points(entry)
            omitted += n_omit
            ritual_omitted += n_ritual
            if selected:
                views.append((entry, selected))

        if not views:
            # All labeled tips filtered — still allow unlabeled exploration
            # tips. Do NOT dump negatively scored / strongly harmful tips back
            # into the inject block; empty inject is first-class silence.
            unlabeled: List[Tuple[HotSkillEntry, List[str]]] = []
            for entry in entries:
                utils = self._aligned_utils(entry)
                kept = [
                    text
                    for text, util in zip(entry.key_points, utils)
                    if self._is_injectable_point(text)
                    and int(util.get("n_labeled") or 0) <= 0
                ]
                if kept:
                    unlabeled.append((entry, kept))
            views = unlabeled

        self._telemetry.points_omitted_utility = omitted
        self._telemetry.points_omitted_ritual = ritual_omitted
        return self._select_retrieve_views(views)

    def _select_retrieve_views(
        self, views: List[Tuple[HotSkillEntry, List[str]]]
    ) -> List[Tuple[HotSkillEntry, List[str]]]:
        """Keep in-domain skills and top-k tips. No LLM on this path."""
        if not views:
            return views
        if not self._config.get("inject_retrieve", True):
            self._telemetry.retrieve_mode = "off"
            return self._cap_inject_views(views)

        self._ensure_lexicons()
        query = self._episode_query_tokens
        ignore = self._idf_ignore
        platform = platform_retrieve_tokens(
            str(self._config.get("runtime_platform") or "")
        )

        scored_skills: List[Tuple[HotSkillEntry, List[str], int]] = []
        omitted_skills: List[str] = []
        for entry, points in views:
            lex = self._skill_lexicons.get(entry.name) or skill_retrieve_tokens(
                entry.name, entry.tags, entry.scope, entry.description, points
            )
            hits = retrieve_overlap(query, lex, ignore=ignore)
            if hits:
                scored_skills.append((entry, points, len(hits)))
                continue
            if query:
                ident = self._identity_lexicons.get(entry.name) or skill_identity_tokens(
                    entry.name, entry.tags, entry.scope, entry.description
                )
                if retrieve_overlap(platform, ident):
                    scored_skills.append((entry, points, 0))
                    continue
                omitted_skills.append(entry.name)
                continue
            scored_skills.append((entry, points, 0))

        self._telemetry.skills_omitted_retrieve = omitted_skills
        if query and scored_skills and any(n > 0 for _, _, n in scored_skills):
            self._telemetry.retrieve_mode = "overlap"
        elif query and omitted_skills and scored_skills:
            self._telemetry.retrieve_mode = "platform_fallback"
        elif query and not scored_skills:
            self._telemetry.retrieve_mode = "utility_fallback"
            scored_skills = [(entry, points, 0) for entry, points in views]
            self._telemetry.skills_omitted_retrieve = []
        else:
            self._telemetry.retrieve_mode = "utility_fallback"

        ranked_items: List[Tuple[float, str, int, HotSkillEntry, str]] = []
        for entry, points, _skill_hits in scored_skills:
            utils = self._aligned_utils(entry)
            util_by_text = {text: util for text, util in zip(entry.key_points, utils)}
            for idx, text in enumerate(points):
                orig_idx = next(
                    (i for i, point in enumerate(entry.key_points) if point == text),
                    idx,
                )
                tip_lex = self._tip_lexicons.get((entry.name, orig_idx))
                if tip_lex is None:
                    tip_lex = tip_retrieve_tokens(text)
                overlap_n = len(retrieve_overlap(query, tip_lex, ignore=ignore))
                util_score = _utility_inject_score(util_by_text.get(text) or empty_point_utility())
                score = (10.0 * overlap_n) + util_score
                ranked_items.append((score, entry.name, idx, entry, text))

        ranked_items.sort(key=lambda item: (-item[0], item[1], item[2]))
        inject_k = int(self._config.get("inject_k", 4) or 0)
        chosen = ranked_items if inject_k <= 0 else ranked_items[:inject_k]
        kept_texts = {(entry.name, text) for _, _, _, entry, text in chosen}
        omitted_pts = 0
        grouped: "OrderedDict[str, Tuple[HotSkillEntry, List[str]]]" = OrderedDict()
        for _score, _name, _idx, entry, text in chosen:
            grouped.setdefault(entry.name, (entry, []))
            grouped[entry.name][1].append(text)
        for entry, points, _hits in scored_skills:
            for text in points:
                if (entry.name, text) not in kept_texts:
                    omitted_pts += 1
        self._telemetry.points_omitted_retrieve = omitted_pts
        return [(entry, pts) for entry, pts in grouped.values() if pts]

    def _cap_inject_views(
        self, views: List[Tuple[HotSkillEntry, List[str]]]
    ) -> List[Tuple[HotSkillEntry, List[str]]]:
        inject_k = int(self._config.get("inject_k", 4) or 0)
        if inject_k <= 0:
            return views
        kept: List[Tuple[HotSkillEntry, List[str]]] = []
        remaining = inject_k
        omitted = 0
        for entry, points in views:
            if remaining <= 0:
                omitted += len(points)
                continue
            take = points[:remaining]
            omitted += len(points) - len(take)
            remaining -= len(take)
            if take:
                kept.append((entry, take))
        self._telemetry.points_omitted_retrieve += omitted
        return kept

    def _format_entries(self, entries: List[HotSkillEntry]) -> str:
        return self._format_entry_views(
            [(entry, list(entry.key_points)) for entry in entries if entry.key_points]
        )

    def _format_entry_views(
        self, views: List[Tuple[HotSkillEntry, List[str]]]
    ) -> str:
        parts = [
            format_hot_skill_section(entry.name, entry.scope, points)
            for entry, points in views
            if points
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
        self._mark_lexicons_dirty()

    def _gate_admission_points(
        self,
        skill_name: str,
        points: List[str],
        *,
        scope: str,
        context: str,
    ) -> List[str]:
        """Per-tip transfer gate before admit (LLM when configured, else deterministic)."""
        if not self._config.get("admit_transfer_gate", True) or not points:
            return list(points)
        items = [
            {"id": str(i), "point": p, "scope": scope or ""}
            for i, p in enumerate(points)
        ]
        admit_ids: Optional[List[str]] = None
        if callable(self.admission_judge):
            try:
                admit_ids = self.admission_judge(
                    items,
                    skill_name,
                    scope or "",
                    context or "",
                )
            except Exception:
                logger.debug("hot skill admission_judge failed", exc_info=True)
                admit_ids = None
        drop_ritual = bool(self._config.get("ritual_filter", True))

        def _keep_point(p: str) -> bool:
            if is_oracle_style_tip(p):
                return False
            if drop_ritual and is_ritual_key_point(p):
                return False
            return True

        if admit_ids is not None:
            keep = {str(i) for i in admit_ids}
            filtered = [p for i, p in enumerate(points) if str(i) in keep]
        else:
            filtered = [p for p in points if _keep_point(p)]
        if not filtered and points:
            filtered = [p for p in points if _keep_point(p)]
        if drop_ritual:
            filtered = [p for p in filtered if not is_ritual_key_point(p)]
        return filtered

    def _deterministic_reconcile_drops(self, cap: int) -> int:
        """Drop oracle tips and harmful-heavy utilities before/alongside LLM reconcile."""
        dropped = 0
        for name in list(self._entries.keys()):
            entry = self._entries.get(name)
            if entry is None:
                continue
            for idx in reversed(range(len(entry.key_points))):
                tip = entry.key_points[idx]
                if is_oracle_style_tip(tip) or (
                    self._config.get("ritual_filter", True)
                    and is_ritual_key_point(tip)
                ):
                    self._drop_point(name, idx)
                    dropped += 1

        skill_counts = {
            name: len(entry.key_points)
            for name, entry in self._entries.items()
            if entry.key_points
        }

        while self._total_points() > cap:
            items = self._point_items()
            if not items:
                break

            oracle_items = [
                it
                for it in items
                if is_oracle_style_tip(str(it.get("point") or ""))
            ]
            if oracle_items:
                victim = oracle_items[0]
            else:
                harmful_items = [
                    it
                    for it in items
                    if int((it.get("utility") or {}).get("n_labeled") or 0) > 0
                    and int((it.get("utility") or {}).get("harmful") or 0)
                    > int((it.get("utility") or {}).get("helpful") or 0)
                ]
                if harmful_items:
                    victim = max(
                        harmful_items,
                        key=lambda it: (
                            int((it.get("utility") or {}).get("harmful") or 0),
                            int(it.get("recorded_turn") or 0),
                        ),
                    )
                else:
                    victim = min(
                        items,
                        key=lambda it: (
                            int(it.get("recorded_turn") or 0),
                            str(it.get("skill") or ""),
                            int(it.get("index") or 0),
                        ),
                    )
            skill = str(victim["skill"])
            index = int(victim["index"])
            self._drop_point(skill, index)
            if skill in skill_counts:
                skill_counts[skill] = max(0, skill_counts.get(skill, 1) - 1)
            dropped += 1
        return dropped

    def reconcile_pool(
        self,
        *,
        new_name: str = "",
        context: str = "",
        material_update: bool = True,
    ) -> None:
        """Curate the pool to ``max_entries`` after admit, sync, or refresh.

        Under policy ``llm``, runs the side-channel keep-set judge when the
        pool overflows *or* on each material update (``reconcile_on_update``).
        Falls back to ``oldest`` point drops when still over cap.
        """
        cap = int(self._config.get("max_entries", 12) or 0)
        if cap <= 0:
            self._entries.clear()
            return

        incoming = self._entries.get(new_name) if new_name else None
        if incoming is not None:
            per_skill = int(self._config.get("max_points_per_skill", 8) or 8)
            if len(incoming.key_points) > per_skill:
                incoming.key_points = incoming.key_points[:per_skill]
                if incoming.point_utilities:
                    incoming.point_utilities = incoming.point_utilities[:per_skill]
            if len(incoming.key_points) > cap:
                incoming.key_points = incoming.key_points[:cap]
                if incoming.point_utilities:
                    incoming.point_utilities = incoming.point_utilities[:cap]

        total = self._total_points()
        policy = str(self._config.get("eviction_policy") or _DEFAULT_EVICTION_POLICY)
        reconcile_on_update = bool(self._config.get("reconcile_on_update", True))

        evicted = 0
        if material_update or total > cap:
            det = self._deterministic_reconcile_drops(cap)
            if det:
                self._telemetry.evicted_deterministic += det
                evicted += det
                total = self._total_points()

        should_llm = (
            policy == "llm"
            and callable(self.eviction_judge)
            and self._point_items()
            and (
                total > cap
                or (material_update and reconcile_on_update)
            )
        )
        if should_llm:
            self._telemetry.reconcile_ran = True
            evicted += self._apply_llm_keep(cap, context)
        elif evicted and material_update:
            self._telemetry.reconcile_ran = True

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

    def _trim_to_max_entries(self, *, new_name: str, context: str) -> None:
        """Deprecated alias — use :meth:`reconcile_pool`."""
        self.reconcile_pool(new_name=new_name, context=context, material_update=True)

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
        self.clear_episode_query()

    def apply_outcome_feedback(
        self,
        outcome: dict,
        *,
        complete_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        episode_log: Optional[Dict[str, Any]] = None,
        messages: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Attribute exposed tips after a labeled task; persist utilities.

        Prefers a side-channel LLM (transfer vs episode-local is a prompt
        judgment). Empty or unparseable JSON does not write utilities unless
        ``outcome_feedback_heuristic`` is explicitly enabled.
        """
        summary = {
            "applied": False,
            "points_scored": 0,
            "attributions": {},
            "skipped_reason": "",
            "attribution_source": "",
        }
        if not self.enabled or not self._config.get("outcome_feedback", True):
            summary["skipped_reason"] = "disabled"
            return self._record_outcome_summary(summary)
        if not isinstance(outcome, dict) or not outcome:
            summary["skipped_reason"] = "missing_outcome"
            return self._record_outcome_summary(summary)
        if not self._exposed_tips:
            summary["skipped_reason"] = "no_exposed_tips"
            # No tips to score, but still update inject vs no-inject cohort so
            # throttling / probe cadence learn from empty-inject episodes.
            if isinstance(outcome, dict) and outcome:
                self._note_inject_cohort(
                    injected=self._episode_had_inject(),
                    success=bool(outcome.get("success")),
                )
                self._maybe_persist()
            return self._record_outcome_summary(summary)
        self._reset_outcome_judge_meta()

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

        log = episode_log
        if log is None and self._config.get("outcome_judge_log", True):
            log = build_compressed_episode_log(
                messages=messages,
                episode_query=self._episode_query,
                max_events=int(self._config.get("outcome_judge_log_max_events", 48) or 48),
                max_arg_chars=int(self._config.get("outcome_judge_log_arg_chars", 96) or 96),
            )
        if not isinstance(log, dict):
            log = {}
        events = log.get("events") if isinstance(log.get("events"), list) else []
        self._telemetry.outcome_judge_log_events = int(log.get("n_events") or len(events) or 0)
        names = [str(ev.get("name") or "") for ev in events if isinstance(ev, dict)]
        self._telemetry.outcome_judge_log_preview = sidechannel_text_preview(
            " | ".join(n for n in names if n)
        )

        labels: Dict[str, str] = {}
        if callable(complete_fn):
            try:
                text = complete_fn(
                    build_outcome_attribution_messages(items, outcome, episode_log=log)
                ) or ""
            except Exception:
                logger.debug("hot pool outcome attribution complete_fn failed", exc_info=True)
                summary["skipped_reason"] = "complete_fn_failed"
            else:
                labels = parse_outcome_attributions(text, items)
                if not labels:
                    summary["skipped_reason"] = "unparseable_or_empty"
        else:
            summary["skipped_reason"] = "no_complete_fn"

        if not labels and self._config.get("outcome_feedback_heuristic", False):
            labels = heuristic_outcome_attributions(
                items,
                outcome,
                low_step_threshold=int(
                    self._config.get("outcome_helpful_max_iterations", 12) or 12
                ),
            )
            if labels:
                summary["attribution_source"] = "heuristic"
        elif labels:
            summary["skipped_reason"] = ""
            summary["attribution_source"] = "llm"

        # Honest failure→harmful floor (also fills labels when the judge is empty).
        labels, n_clamped = clamp_outcome_attributions(items, labels, outcome)
        self._telemetry.outcome_labels_clamped = n_clamped
        if n_clamped and not summary.get("attribution_source"):
            summary["attribution_source"] = "failure_floor"
            summary["skipped_reason"] = ""
        elif n_clamped and summary.get("attribution_source") == "llm":
            summary["attribution_source"] = "llm+failure_floor"

        if not labels:
            # Still update inject cohort so throttling can learn from unlabeled fails.
            self._note_inject_cohort(
                injected=self._episode_had_inject(),
                success=bool(outcome.get("success")),
            )
            self._maybe_persist()
            return self._record_outcome_summary(summary)

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

        self._note_inject_cohort(
            injected=self._episode_had_inject(),
            success=success,
        )

        if scored:
            self._maybe_persist()
            if self._config.get("reconcile_on_update", True):
                bench = str(outcome.get("benchmark") or "")
                self.reconcile_pool(context=bench, material_update=True)
        summary.update(
            {
                "applied": scored > 0,
                "points_scored": scored,
                "attributions": attr_counts,
                "labels_clamped": n_clamped,
            }
        )
        if scored and not summary.get("attribution_source"):
            summary["attribution_source"] = "llm"
        return self._record_outcome_summary(summary)

    def _episode_had_inject(self) -> bool:
        if self._last_block_had_inject:
            return True
        for tip in self._exposed_tips.values():
            if str(tip.get("via") or "") == "inject":
                return True
        return False

    def _reset_outcome_judge_meta(self) -> None:
        self._telemetry.outcome_judge_finish_reason = ""
        self._telemetry.outcome_judge_raw_chars = 0
        self._telemetry.outcome_judge_stripped_chars = 0
        self._telemetry.outcome_judge_raw_preview = ""
        self._telemetry.outcome_judge_stripped_preview = ""
        self._telemetry.outcome_judge_max_tokens = 0
        self._telemetry.outcome_judge_thinking_off = False
        self._telemetry.outcome_judge_retries = 0
        self._telemetry.outcome_judge_recovered_json = False
        self._telemetry.outcome_judge_log_events = 0
        self._telemetry.outcome_judge_log_preview = ""

    def note_outcome_judge_meta(self, meta: Optional[Dict[str, Any]]) -> None:
        """Record side-channel decode diagnostics after an outcome judge call."""
        if not isinstance(meta, dict) or not meta:
            return
        tel = self._telemetry
        tel.outcome_judge_finish_reason = str(meta.get("finish_reason") or "")
        try:
            tel.outcome_judge_raw_chars = int(meta.get("raw_chars") or 0)
        except (TypeError, ValueError):
            tel.outcome_judge_raw_chars = 0
        try:
            tel.outcome_judge_stripped_chars = int(meta.get("stripped_chars") or 0)
        except (TypeError, ValueError):
            tel.outcome_judge_stripped_chars = 0
        tel.outcome_judge_raw_preview = sidechannel_text_preview(
            str(meta.get("raw_preview") or meta.get("raw") or "")
        )
        tel.outcome_judge_stripped_preview = sidechannel_text_preview(
            str(meta.get("stripped_preview") or meta.get("stripped") or "")
        )
        try:
            tel.outcome_judge_max_tokens = int(meta.get("max_tokens") or 0)
        except (TypeError, ValueError):
            tel.outcome_judge_max_tokens = 0
        tel.outcome_judge_thinking_off = bool(meta.get("thinking_off"))
        try:
            tel.outcome_judge_retries = int(meta.get("retries") or 0)
        except (TypeError, ValueError):
            tel.outcome_judge_retries = 0
        tel.outcome_judge_recovered_json = bool(meta.get("recovered_json"))
        try:
            tel.outcome_judge_log_events = int(meta.get("log_events") or 0)
        except (TypeError, ValueError):
            tel.outcome_judge_log_events = 0
        preview = str(meta.get("log_preview") or "")
        if preview:
            tel.outcome_judge_log_preview = sidechannel_text_preview(preview)

    def _record_outcome_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        self._telemetry.outcome_feedback_applied = bool(summary.get("applied"))
        self._telemetry.outcome_feedback_points = int(summary.get("points_scored") or 0)
        self._telemetry.outcome_attributions = dict(summary.get("attributions") or {})
        self._telemetry.outcome_feedback_skipped_reason = str(
            summary.get("skipped_reason") or ""
        )
        self._telemetry.outcome_feedback_attribution_source = str(
            summary.get("attribution_source") or ""
        )
        if "labels_clamped" in summary:
            try:
                self._telemetry.outcome_labels_clamped = int(summary.get("labels_clamped") or 0)
            except (TypeError, ValueError):
                pass
        self._sync_cohort_telemetry()
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
        cohort = raw.get("inject_cohort")
        if isinstance(cohort, dict):
            self._cohort_inject_n = max(0, int(cohort.get("inject_n") or 0))
            self._cohort_inject_success = max(
                0, int(cohort.get("inject_success") or 0)
            )
            self._cohort_no_inject_n = max(0, int(cohort.get("no_inject_n") or 0))
            self._cohort_no_inject_success = max(
                0, int(cohort.get("no_inject_success") or 0)
            )
            self._inject_throttled = bool(cohort.get("throttled"))
            self._episodes_since_inject = max(
                0, int(cohort.get("episodes_since_inject") or 0)
            )
            self._recompute_inject_throttle()
            self._sync_cohort_telemetry()
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
        if self._config.get("ritual_filter", True):
            for name in list(loaded.keys()):
                entry = loaded[name]
                kept: List[str] = []
                kept_u: List[Dict[str, Any]] = []
                utils = list(entry.point_utilities or [])
                while len(utils) < len(entry.key_points):
                    utils.append(empty_point_utility())
                for text, util in zip(entry.key_points, utils):
                    if is_ritual_key_point(text):
                        continue
                    kept.append(text)
                    kept_u.append(util)
                if not kept:
                    loaded.pop(name, None)
                else:
                    entry.key_points = kept
                    entry.point_utilities = kept_u
        self._entries = loaded
        self._mark_lexicons_dirty()
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
            "inject_cohort": {
                "inject_n": self._cohort_inject_n,
                "inject_success": self._cohort_inject_success,
                "no_inject_n": self._cohort_no_inject_n,
                "no_inject_success": self._cohort_no_inject_success,
                "throttled": self._inject_throttled,
                "episodes_since_inject": self._episodes_since_inject,
            },
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


def _fenced_char_ranges(content: str) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` spans inside markdown ``` fenced blocks (exclusive of markers)."""
    ranges: List[Tuple[int, int]] = []
    in_fence = False
    fence_start = 0
    pos = 0
    for line in (content or "").splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                fence_start = pos + len(line)
            else:
                ranges.append((fence_start, pos))
                in_fence = False
        pos += len(line)
    if in_fence:
        ranges.append((fence_start, len(content)))
    return ranges


def _position_in_fenced_region(pos: int, fenced_ranges: List[Tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in fenced_ranges)


def _extract_section_points(content: str) -> List[str]:
    headings = list(_HEADING_RE.finditer(content))
    if not headings:
        return []

    fenced = _fenced_char_ranges(content)
    points: List[str] = []
    for idx, match in enumerate(headings):
        if _position_in_fenced_region(match.start(), fenced):
            continue
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
    """System prompt for pool reconcile / overflow keep-set judgment."""
    return (
        "You curate a small hot-skill key-point pool used as short guardrail "
        "reminders.\n"
        "\n"
        "Hard constraint — keep is the store, not a guaranteed inject: every "
        "tip you KEEP may be prepended on later turns and on later tasks, "
        "including held-out tasks that were never seen while the pool was "
        "built. Inject later selects a matching subset. Optimize for "
        "unseen-task transfer and broadcast-safety (still helpful or at least "
        "harmless out of domain). Do NOT optimize for replaying the same "
        "tasks that produced the tips, and do NOT maximize usefulness on the "
        "present pool alone.\n"
        "\n"
        f"Return JSON only: {{\"keep\": [\"id\", ...]}} with at most {keep_n} "
        "ids drawn from the provided points. Do not invent ids. Prefer fewer "
        "strong tips over filling the quota: return an empty or short keep "
        "list when most candidates fail the bar.\n"
        "\n"
        "KEEP tips that:\n"
        "- State a transferable pitfall or procedure that remains meaningful "
        "after stripping instance-specific tokens (paths, hostnames, accounts, "
        "filenames, task IDs, one-off product recipes) and would still apply "
        "to a new task instance in the same environment/genre\n"
        "- Encode evaluator-/environment-agnostic process rules (e.g. write "
        "graded artifacts where the harness reads them; prefer admissible "
        "actions; use the provided API surface; verify before long runs) "
        "rather than a single task's setup recipe or a default env closer\n"
        "- Add distinct failure-mode coverage — prefer diversity of pitfalls "
        "across skills/topics when candidates are similar\n"
        "- Use each point's natural-language scope (when present) as a soft "
        "hint. Prefer tips that are either broadly safe or clearly "
        "conditional; do not keep a narrow niche tip merely because it is "
        "useful on the current overflow\n"
        "- When utility stats are present (n_labeled > 0), prefer tips with "
        "more helpful than harmful attributions; when success rates are "
        "similar, prefer lower avg_iterations_when_success / avg_iterations "
        "(efficiency often separates tips when success is near-ceiling)\n"
        "\n"
        "DROP or deprioritize tips that:\n"
        "- Are bound to absolute paths, hostnames, credentials, account names, "
        "phone numbers, one-off filenames, task IDs, or a single product/CLI/"
        "app workflow (unless the same sentence also states a general rule "
        "that survives removing those tokens)\n"
        "- Recap one episode's entities or layout (which object was where, "
        "which inbox/account/file the last task used, numbered instance slots, "
        "exact names that will not recur). Identifier placeholders such as "
        "<path> or <id> do not make a recap transferable — if the remaining "
        "claim is still a memory of one episode, drop it\n"
        "- Teach oracle/leakage or harness-cheating shortcuts (reading hidden "
        "solvers/oracles, claiming pass without producing graded outputs, "
        "rewriting verifier paths to temporary locations as a default)\n"
        "- Encode one task's I/O layout, auth bootstrap, or dependency install "
        "recipe rather than a reusable pitfall\n"
        "- Act mainly as a cheatsheet for replaying an already-seen task "
        "(memorized steps, exact entities, or solutions that would not help a "
        "new unseen task in the same benchmark)\n"
        "- Duplicate a stronger tip already being kept, or add a second tip "
        "from the same skill that covers the same failure mode\n"
        "- Would mislead, waste steps, or pull work off the graded surface if "
        "shown on an unrelated later task\n"
        "- Prescribe a default search order over instance-indexed slots "
        "(visit every shelf / drawer / file / message N) rather than a "
        "constraint that applies only after the current observation fails. "
        "Keep a legal-action rule if it can stand without the tour\n"
        "- Prescribe a termination ritual (always call done()/complete_task, "
        "always return a short/minimal answer). Those are episode procedures; "
        "the correct closer depends on the task. A NEVER/DO NOT about a bad "
        "closer is a constraint — keep that\n"
        "- Have clearly worse utility than alternatives (high harmful count, "
        "or much higher iteration cost for similar success)\n"
        "\n"
        "Imperative intensity (NEVER/ALWAYS/MUST) is not evidence of quality — "
        "judge the abstract claim, not the wording.\n"
        "\n"
        "About context: it describes the incoming extract / current task only "
        "(often from the training task). Use it to understand what is "
        "being admitted and to break ties. Do NOT treat \"most relevant to "
        "context\", \"will help if this task is repeated\", or \"likely needed "
        "on similar seen tasks\" as the primary keep criterion under "
        "store-then-retrieve inject.\n"
        "\n"
        "When utilities_flat_no_helpful is true in the user payload, no tip "
        "yet has a helpful attribution — prefer dropping oracle/cheatsheet "
        "candidates and episode-local recipes even if they are merely irrelevant."
    )


def build_admit_transfer_system_prompt() -> str:
    """System prompt for per-tip admission gate (transfer vs episode-local)."""
    return (
        "You gate admission of individual hot-skill key points to a broadcast pool.\n"
        "\n"
        "Every admitted tip will be injected on later tasks, including held-out "
        "tasks never seen while the pool was built. Admit only tips that remain "
        "meaningful after stripping instance-specific tokens and would help on an "
        "unseen task in the same benchmark/environment genre.\n"
        "\n"
        "Return JSON only: {\"admit\": [\"id\", ...]} listing ids to admit from "
        "the provided candidates. Omit ids for non-transferable tips. Prefer "
        "admitting fewer strong tips over keeping marginal ones.\n"
        "\n"
        "ADMIT tips that:\n"
        "- State evaluator-/environment-agnostic process rules (write graded "
        "artifacts where the harness reads them; verify on the graded surface; "
        "use admissible APIs/tools). That is not a default env closer call\n"
        "- Describe reusable pitfalls without absolute paths, task IDs, or oracle "
        "shortcuts\n"
        "\n"
        "REJECT tips that:\n"
        "- Tell the agent to run bundled solution/ scripts, read ground truth from "
        "tests, or skip builds using static expected constants\n"
        "- Encode one episode's Docker/Lean/install recipe rather than a reusable rule\n"
        "- Prescribe a termination ritual (always call done()/complete_task, "
        "always return a short/minimal answer). Those are episode procedures\n"
        "- Prescribe a process ritual (ALWAYS show_api_doc / check docs before "
        "every call) rather than a reusable constraint\n"
        "- Would mislead on an unrelated later task\n"
    )


def build_admit_transfer_messages(
    items: List[Dict[str, Any]],
    skill_name: str,
    scope: str,
    context: str,
) -> List[Dict[str, str]]:
    payload = {
        "skill_name": skill_name,
        "scope": scope or "",
        "context": context or "",
        "candidates": items,
    }
    return [
        {"role": "system", "content": build_admit_transfer_system_prompt()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_admit_transfer_ids(
    text: str,
    items: List[Dict[str, Any]],
) -> Optional[List[str]]:
    valid_ids = {str(it["id"]) for it in items}
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
    admit = data.get("admit") if isinstance(data, dict) else None
    if not isinstance(admit, list):
        return None
    out = [str(i) for i in admit if str(i) in valid_ids]
    return out if out else []


def llm_admit_transfer_ids(
    items: List[Dict[str, Any]],
    skill_name: str,
    scope: str,
    context: str,
    complete_fn: Callable[[List[Dict[str, str]]], str],
) -> Optional[List[str]]:
    if not callable(complete_fn) or not items:
        return None
    try:
        text = complete_fn(
            build_admit_transfer_messages(items, skill_name, scope, context)
        ) or ""
    except Exception:
        logger.debug("hot skill llm admit transfer complete_fn failed", exc_info=True)
        return None
    return parse_admit_transfer_ids(text, items)


def build_outcome_attribution_messages(
    items: List[Dict[str, Any]],
    outcome: dict,
    episode_log: Optional[Dict[str, Any]] = None,
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
        "episode_log is a compressed action trace (tool names + short args, "
        "env actions, or code API snippets). It does NOT include observations "
        "or raw API dumps. Use it to see whether a tip was followed, ignored, "
        "or contradicted. A tip whose advice never appears in the log is "
        "irrelevant even on success. A tip that matches a long enumeration of "
        "lookups or extra env steps is harmful when iterations are high. "
        "Missing log fields are not evidence the tip helped.\n"
        "\n"
        "When outcome.success is false OR outcome.claimed_success_mismatch is "
        "true: you MUST NOT label any via=inject tip as helpful. Prefer harmful "
        "for injected tips (they were broadcast into a failed episode). "
        "History-only tips may be irrelevant.\n"
        "\n"
        "When outcome.claimed_success_mismatch is true, the agent asserted "
        "pass/completion in its final response but labeled evaluation failed — "
        "label injected tips that plausibly steered toward wrong-env verification, "
        "oracle/replay shortcuts, or premature success claims as harmful.\n"
        "\n"
        "Return JSON only: {\"labels\": {\"<id>\": \"helpful\"|\"harmful\"|"
        "\"irrelevant\", ...}} for the given point ids. Do not invent ids.\n"
        "- helpful: tip is a transferable process or constraint that plausibly "
        "improved this episode and would still apply to a new task instance\n"
        "- harmful: tip plausibly caused waste, wrong paths, or failure modes, "
        "or would mislead on a different instance (wrong object, path, account). "
        "On task failure, prefer harmful over irrelevant for injected tips that "
        "prescribe shortcuts, oracle replay, or environment-specific cheats\n"
        "- irrelevant: tip did not meaningfully affect this episode, OR it is "
        "an episode-local recap (specific objects/locations/accounts/filenames) "
        "even if the episode succeeded — those do not transfer\n"
        "\n"
        "Judge the abstract claim, not wording. NEVER/ALWAYS/MUST is not "
        "evidence of helpfulness. Prefer irrelevant over helpful when a tip "
        "only restates one episode's entities.\n"
        "If a tip plausibly caused extra iterations — an enumeration tour, "
        "or visiting slots the observation did not suggest — label it "
        "harmful even when the episode eventually succeeded.\n"
    )
    payload = {
        "outcome": outcome,
        "points": items,
        "episode_log": episode_log if isinstance(episode_log, dict) else {},
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
    labeled = [
        it
        for it in items
        if int((it.get("utility") or {}).get("n_labeled") or 0) > 0
    ]
    utilities_flat = bool(labeled) and all(
        int((it.get("utility") or {}).get("helpful") or 0) == 0 for it in labeled
    )
    payload = {
        "selection_goal": (
            "Choose a keep-set that transfers to held-out / unseen tasks "
            "(store; inject later retrieves a subset). Prefer abstract, evaluator-safe pitfalls over "
            "cheatsheets for replaying seen tasks. Prefer fewer strong tips "
            "over filling keep_n."
        ),
        "context_role": (
            "Background for the incoming extract / overflow. Tie-breaker only; "
            "not the primary ranking objective."
        ),
        "context": context or "",
        "keep_n": keep_n,
        "utilities_flat_no_helpful": utilities_flat,
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
        f"{_HOT_SKILLS_SYSTEM_NOTE}\n\n"
        f"{clean}\n"
        f"{_HOT_SKILLS_CLOSE}"
    )
