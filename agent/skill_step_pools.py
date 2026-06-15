"""Step-wise skill variant pools (optional ``skills.step_pools`` feature).

Skills may mark regions in SKILL.md with HTML comments::

    <!-- hermes-step id=\"bootstrap\" max_variants=\"4\" -->
    Run ``npm ci``
    <!-- /hermes-step -->

Per-step variant bodies and success statistics live in ``step_pools.json``
next to SKILL.md. When ``skills.step_pools.enabled`` is true, ``skill_view``
expands markers into ordered variant blocks for the agent.

``max_variants`` on the marker caps how many variants (including baseline)
are shown; additional JSON variants are ranked and truncated.

Ranking after ``manual_order`` is controlled by ``skills.step_pools.rank_strategy``
(``laplace`` | ``raw`` | ``wilson`` | ``ucb1``); see ``variant_rank_score()`` and
``hermes_cli`` defaults.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STEP_POOLS_FILENAME = "step_pools.json"
BASELINE_ID = "baseline"

# ``skills.step_pools.rank_strategy`` — see hermes_cli/config.py DEFAULT_CONFIG.
VALID_RANK_STRATEGIES = frozenset({"laplace", "raw", "wilson", "ucb1"})

# Opening tag: attributes on one line (id required, max_variants optional).
STEP_OPEN_RE = re.compile(
    r"<!--\s*hermes-step\s+([^>]+?)\s*-->",
    re.IGNORECASE,
)
STEP_CLOSE_RE = re.compile(r"<!--\s*/hermes-step\s*-->", re.IGNORECASE)

_STEP_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_VARIANT_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")


def _parse_open_attrs(raw: str) -> Tuple[str, int | None]:
    """Parse id=\"...\" and optional max_variants=\"N\" from the open tag body."""
    id_m = re.search(r'\bid\s*=\s*"([^"]+)"', raw, re.IGNORECASE)
    if not id_m:
        raise ValueError("hermes-step tag requires id=\"...\"")
    step_id = id_m.group(1).strip()
    if not _STEP_ID_RE.match(step_id):
        raise ValueError(f"invalid step id: {step_id!r}")
    mv_m = re.search(r'\bmax_variants\s*=\s*"(\d+)"', raw, re.IGNORECASE)
    max_v = int(mv_m.group(1)) if mv_m else None
    if max_v is not None and (max_v < 1 or max_v > 64):
        raise ValueError("max_variants must be between 1 and 64")
    return step_id, max_v


def _default_max_variants(skills_cfg: dict) -> int:
    sp = skills_cfg.get("step_pools")
    if not isinstance(sp, dict):
        return 5
    v = sp.get("default_max_variants_per_step", 5)
    try:
        n = int(v)
    except (TypeError, ValueError):
        return 5
    return max(1, min(64, n))


def step_pools_enabled(skills_cfg: dict) -> bool:
    sp = skills_cfg.get("step_pools")
    return isinstance(sp, dict) and bool(sp.get("enabled"))


def step_pools_path(skill_dir: Path) -> Path:
    return skill_dir / STEP_POOLS_FILENAME


def _variant_sfa(v: dict[str, Any]) -> Tuple[int, int, int]:
    """Return ``(success, fail, trials)`` with ``trials = success + fail`` (ignores stray attempts)."""
    s = int(v.get("success") or 0)
    f = int(v.get("fail") or 0)
    a = int(v.get("attempts") or 0)
    trials = s + f
    if a != trials:
        trials = s + f
    return s, f, trials


def _rank_strategy(skills_cfg: dict) -> str:
    sp = skills_cfg.get("step_pools")
    if not isinstance(sp, dict):
        return "laplace"
    raw = sp.get("rank_strategy") or "laplace"
    if not isinstance(raw, str):
        return "laplace"
    key = raw.strip().lower()
    if key not in VALID_RANK_STRATEGIES:
        logger.warning("Invalid skills.step_pools.rank_strategy %r; using laplace", raw)
        return "laplace"
    return key


def _ucb1_c(skills_cfg: dict) -> float:
    sp = skills_cfg.get("step_pools")
    if not isinstance(sp, dict):
        return math.sqrt(2.0)
    v = sp.get("ucb1_c", math.sqrt(2.0))
    try:
        c = float(v)
    except (TypeError, ValueError):
        return math.sqrt(2.0)
    return max(1e-6, min(100.0, c))


def _wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """95% Wilson score interval lower bound in [0, 1]."""
    if trials <= 0:
        return 0.5
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    rad = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials)
    lb = (center - rad) / denom
    return max(0.0, min(1.0, lb))


def _pool_total_trials(variants: List[dict[str, Any]]) -> int:
    return max(1, sum(_variant_sfa(v)[2] for v in variants))


def variant_rank_score(
    v: dict[str, Any],
    strategy: str,
    *,
    pool_total_trials: int,
    skills_cfg: Optional[dict] = None,
) -> float:
    """
    Scalar used to sort variants (higher = try earlier after ``manual_order``).

    * **laplace** — Laplace-smoothed success rate ``(s+1)/(n+2)``, ``n = s+f``.
    * **raw** — ``s / max(1, n)``; no prior (volatile when ``n`` is small).
    * **wilson** — conservative lower bound of the 95% Wilson interval for ``s``/``n``.
    * **ucb1** — ``mean + c * sqrt(2 ln(max(2, T)) / max(1, n))`` with ``T = pool_total_trials``
      (sum of ``s+f`` over variants in this step); favors exploration when data is sparse.
    """
    s, _f, n = _variant_sfa(v)
    strat = (strategy or "laplace").strip().lower()
    cfg = skills_cfg if isinstance(skills_cfg, dict) else {}
    t_pool = max(1, int(pool_total_trials))

    if strat == "raw":
        return float(s) / float(max(1, n))

    if strat == "wilson":
        return _wilson_lower_bound(s, n)

    if strat == "ucb1":
        c = _ucb1_c(cfg)
        t = max(2, t_pool)
        if n <= 0:
            return float(c * math.sqrt(2.0 * math.log(t)))
        mean = s / n
        return float(mean + c * math.sqrt(2.0 * math.log(t) / max(1, n)))

    # laplace (default)
    if n < 0:
        return 0.5
    return (s + 1.0) / (n + 2.0)


def read_step_max_variants_from_skill_md(
    skill_dir: Path, step_id: str, skills_cfg: dict
) -> int:
    """Return max_variants from the SKILL.md marker for ``step_id``, else config default."""
    default_cap = _default_max_variants(skills_cfg)
    md = skill_dir / "SKILL.md"
    if not md.exists():
        return default_cap
    try:
        text = md.read_text(encoding="utf-8")
    except OSError:
        return default_cap
    for _start, _end, attrs, _inner, _full in _iter_marked_blocks(text):
        try:
            sid, mv = _parse_open_attrs(attrs)
        except ValueError:
            continue
        if sid != step_id:
            continue
        return mv if mv is not None else default_cap
    return default_cap


def load_step_pools(skill_dir: Path) -> dict[str, Any]:
    path = step_pools_path(skill_dir)
    if not path.exists():
        return {"version": 1, "steps": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return {"version": 1, "steps": {}}
    if not isinstance(data, dict):
        return {"version": 1, "steps": {}}
    steps = data.get("steps")
    if not isinstance(steps, dict):
        data["steps"] = {}
    data.setdefault("version", 1)
    return data


def save_step_pools(skill_dir: Path, data: dict[str, Any]) -> None:
    path = step_pools_path(skill_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _sort_variants(
    variants: List[dict[str, Any]],
    manual_order: Optional[List[str]],
    skills_cfg: dict,
) -> List[dict[str, Any]]:
    """Order variants: manual_order first (known ids), then rest by rank score desc, id."""
    strat = _rank_strategy(skills_cfg)
    t_pool = _pool_total_trials(variants)
    by_id = {str(v["id"]): v for v in variants if isinstance(v, dict) and v.get("id")}
    ordered: List[dict[str, Any]] = []
    seen: set[str] = set()
    if manual_order:
        for vid in manual_order:
            if vid in by_id and vid not in seen:
                ordered.append(by_id[vid])
                seen.add(vid)
    rest = [v for vid, v in by_id.items() if vid not in seen]
    rest.sort(
        key=lambda v: (
            -variant_rank_score(v, strat, pool_total_trials=t_pool, skills_cfg=skills_cfg),
            str(v.get("id", "")),
        )
    )
    ordered.extend(rest)
    return ordered


def _iter_marked_blocks(content: str) -> List[Tuple[int, int, str, str, str]]:
    """Return list of (start, end, attrs, inner, full_match) for each step block."""
    blocks: List[Tuple[int, int, str, str, str]] = []
    pos = 0
    while pos < len(content):
        m = STEP_OPEN_RE.search(content, pos)
        if not m:
            break
        inner_start = m.end()
        mc = STEP_CLOSE_RE.search(content, inner_start)
        if not mc:
            logger.warning(
                "hermes-step at %d missing closing <!-- /hermes-step -->", m.start()
            )
            break
        attrs = m.group(1).strip()
        inner = content[inner_start : mc.start()]
        full = content[m.start() : mc.end()]
        blocks.append((m.start(), mc.end(), attrs, inner, full))
        pos = mc.end()
    return blocks


def _merge_variant_records(
    baseline_body: str,
    step_id: str,
    pool_doc: dict[str, Any],
    max_variants: int,
    skills_cfg: dict,
) -> List[dict[str, Any]]:
    steps = pool_doc.setdefault("steps", {})
    if not isinstance(steps, dict):
        pool_doc["steps"] = {}
        steps = pool_doc["steps"]
    step_entry = steps.get(step_id)
    if not isinstance(step_entry, dict):
        step_entry = {"variants": []}
        steps[step_id] = step_entry

    raw_variants = step_entry.get("variants")
    if not isinstance(raw_variants, list):
        raw_variants = []
    json_by_id: Dict[str, dict[str, Any]] = {}
    for v in raw_variants:
        if isinstance(v, dict) and v.get("id"):
            json_by_id[str(v["id"])] = dict(v)

    baseline_rec = json_by_id.pop(BASELINE_ID, {})
    merged: List[dict[str, Any]] = [
        {
            "id": BASELINE_ID,
            "body": baseline_body.strip(),
            "success": int(baseline_rec.get("success") or 0),
            "fail": int(baseline_rec.get("fail") or 0),
            "attempts": int(baseline_rec.get("attempts") or 0),
        }
    ]
    for vid, rec in json_by_id.items():
        if vid == BASELINE_ID:
            continue
        body = rec.get("body")
        if not isinstance(body, str) or not body.strip():
            continue
        merged.append(
            {
                "id": vid,
                "body": body.strip(),
                "success": int(rec.get("success") or 0),
                "fail": int(rec.get("fail") or 0),
                "attempts": int(rec.get("attempts") or 0),
            }
        )

    manual = step_entry.get("manual_order")
    manual_list: Optional[List[str]] = None
    if isinstance(manual, list):
        manual_list = [str(x) for x in manual if x]

    ranked = _sort_variants(merged, manual_list, skills_cfg)
    cap = max(1, min(64, max_variants))
    if len(ranked) > cap:
        ranked = ranked[:cap]
    return ranked


def _format_expanded_block(
    skill_name: str,
    step_id: str,
    variants: List[dict[str, Any]],
    skills_cfg: dict,
) -> str:
    strat = _rank_strategy(skills_cfg)
    t_pool = _pool_total_trials(variants)
    strat_label = strat if strat != "laplace" else "laplace smoothed"
    lines = [
        f"<!-- expanded hermes-step id={step_id!r} -->",
        "",
        f"### Step pool `{step_id}`",
        "",
        "Try **one variant at a time** in the numbered order below (higher scores are tried first after any manual order). "
        "If the first variant fails or is unsuitable, try the next until one succeeds or all are exhausted.",
        "",
        f"After each attempt, record the outcome with "
        f'`skill_step_variant` action `record_attempt` (skill name `{skill_name}`, '
        f'step_id `{step_id}`, variant_id as shown, outcome `success` or `fail`).',
        "",
    ]
    for i, v in enumerate(variants, start=1):
        vid = v.get("id", "?")
        sc = variant_rank_score(v, strat, pool_total_trials=t_pool, skills_cfg=skills_cfg)
        s, f, a = int(v.get("success", 0)), int(v.get("fail", 0)), int(v.get("attempts", 0))
        lines.append(
            f"{i}. **Variant `{vid}`** ({strat_label} rank score {sc:.3f}; success={s}, fail={f}, attempts={a})"
        )
        lines.append("")
        lines.append(str(v.get("body", "")).strip())
        lines.append("")
    lines.append(f"<!-- /expanded hermes-step id={step_id!r} -->")
    return "\n".join(lines)


def expand_skill_content_step_pools(
    content: str,
    skill_dir: Path | None,
    skill_name: str,
    skills_cfg: dict,
) -> str:
    """Replace hermes-step markers with expanded variant sections when enabled."""
    if not skill_dir or not step_pools_enabled(skills_cfg):
        return content
    if "hermes-step" not in content.lower():
        return content

    pool_doc = load_step_pools(skill_dir)
    default_cap = _default_max_variants(skills_cfg)
    blocks = _iter_marked_blocks(content)
    if not blocks:
        return content

    out_parts: List[str] = []
    cursor = 0
    for start, end, attrs, inner, _full in blocks:
        out_parts.append(content[cursor:start])
        try:
            step_id, marker_max = _parse_open_attrs(attrs)
        except ValueError as e:
            out_parts.append(f"<!-- hermes-step parse error: {e} -->")
            out_parts.append(content[start:end])
            cursor = end
            continue
        max_v = marker_max if marker_max is not None else default_cap
        variants = _merge_variant_records(inner, step_id, pool_doc, max_v, skills_cfg)
        out_parts.append(_format_expanded_block(skill_name, step_id, variants, skills_cfg))
        cursor = end
    out_parts.append(content[cursor:])
    return "".join(out_parts)


def record_step_attempt(
    skill_dir: Path,
    step_id: str,
    variant_id: str,
    outcome: str,
) -> dict[str, Any]:
    """Update success/fail counters. outcome in success, fail, skip."""
    if not _STEP_ID_RE.match(step_id):
        return {"success": False, "error": f"invalid step_id: {step_id!r}"}
    if not _VARIANT_ID_RE.match(variant_id):
        return {"success": False, "error": f"invalid variant_id: {variant_id!r}"}
    o = (outcome or "").strip().lower()
    if o not in ("success", "fail", "skip"):
        return {
            "success": False,
            "error": "outcome must be 'success', 'fail', or 'skip'",
        }

    data = load_step_pools(skill_dir)
    steps = data.setdefault("steps", {})
    step_entry = steps.setdefault(step_id, {})
    if not isinstance(step_entry, dict):
        steps[step_id] = {}
        step_entry = steps[step_id]
    variants = step_entry.setdefault("variants", [])
    if not isinstance(variants, list):
        variants = []
        step_entry["variants"] = variants

    found = None
    for v in variants:
        if isinstance(v, dict) and str(v.get("id")) == variant_id:
            found = v
            break
    if found is None:
        if variant_id == BASELINE_ID:
            found = {"id": BASELINE_ID, "success": 0, "fail": 0, "attempts": 0}
            variants.append(found)
        else:
            return {
                "success": False,
                "error": f"variant_id {variant_id!r} not found for step {step_id!r}; use add_variant first",
            }

    if o == "skip":
        found["attempts"] = int(found.get("attempts") or 0) + 1
    elif o == "success":
        found["success"] = int(found.get("success") or 0) + 1
        found["attempts"] = int(found.get("attempts") or 0) + 1
    else:
        found["fail"] = int(found.get("fail") or 0) + 1
        found["attempts"] = int(found.get("attempts") or 0) + 1

    save_step_pools(skill_dir, data)
    return {"success": True, "message": f"Recorded {o} for {step_id}/{variant_id}"}


def add_step_variant(
    skill_dir: Path,
    step_id: str,
    body: str,
    variant_id: str | None,
    max_variants: int | None,
    skills_cfg: dict | None = None,
) -> dict[str, Any]:
    if not _STEP_ID_RE.match(step_id):
        return {"success": False, "error": f"invalid step_id: {step_id!r}"}
    body = (body or "").strip()
    if not body:
        return {"success": False, "error": "body is required"}

    vid = variant_id.strip() if variant_id else None
    if not vid:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", body[:48]).strip("-").lower() or "variant"
        vid = slug[:64]
    if not _VARIANT_ID_RE.match(vid):
        return {"success": False, "error": f"invalid variant_id: {vid!r}"}
    if vid == BASELINE_ID:
        return {
            "success": False,
            "error": f"reserved variant id '{BASELINE_ID}' — edit SKILL.md marker body instead",
        }

    cfg = skills_cfg if isinstance(skills_cfg, dict) else {}
    cap_eff = max_variants
    if cap_eff is None or cap_eff <= 0:
        cap_eff = read_step_max_variants_from_skill_md(skill_dir, step_id, cfg)

    data = load_step_pools(skill_dir)
    steps = data.setdefault("steps", {})
    step_entry = steps.setdefault(step_id, {})
    if not isinstance(step_entry, dict):
        step_entry = {}
        steps[step_id] = step_entry
    variants = step_entry.setdefault("variants", [])
    if not isinstance(variants, list):
        variants = []
        step_entry["variants"] = variants

    for v in variants:
        if isinstance(v, dict) and str(v.get("id")) == vid:
            return {"success": False, "error": f"variant {vid!r} already exists for this step"}

    # Pool size includes baseline (in SKILL) + JSON variants; enforce cap on JSON rows
    cap = max(1, min(64, int(cap_eff)))
    # baseline occupies one slot in the merged list
    if len(variants) >= max(0, cap - 1):
        # drop lowest-scoring non-baseline variant
        scored = [
            (i, v)
            for i, v in enumerate(variants)
            if isinstance(v, dict) and str(v.get("id")) != BASELINE_ID
        ]
        if not scored:
            return {
                "success": False,
                "error": f"step pool is full (max_variants={cap}); raise max_variants or remove a variant",
            }
        strat = _rank_strategy(cfg)
        t_pool = _pool_total_trials([v for _, v in scored])
        scored.sort(
            key=lambda iv: (
                variant_rank_score(iv[1], strat, pool_total_trials=t_pool, skills_cfg=cfg),
                str(iv[1].get("id", "")),
            )
        )
        drop_idx = scored[0][0]
        variants.pop(drop_idx)

    variants.append(
        {
            "id": vid,
            "body": body,
            "success": 0,
            "fail": 0,
            "attempts": 0,
        }
    )
    save_step_pools(skill_dir, data)
    return {"success": True, "message": f"Added variant {vid!r} to step {step_id!r}"}


def remove_step_variant(skill_dir: Path, step_id: str, variant_id: str) -> dict[str, Any]:
    if variant_id == BASELINE_ID:
        return {
            "success": False,
            "error": "cannot remove baseline variant; edit SKILL.md instead",
        }
    data = load_step_pools(skill_dir)
    steps = data.get("steps") or {}
    step_entry = steps.get(step_id)
    if not isinstance(step_entry, dict):
        return {"success": False, "error": f"unknown step_id {step_id!r}"}
    variants = step_entry.get("variants")
    if not isinstance(variants, list):
        return {"success": False, "error": "no variants"}
    new_list = [v for v in variants if not (isinstance(v, dict) and str(v.get("id")) == variant_id)]
    if len(new_list) == len(variants):
        return {"success": False, "error": f"variant {variant_id!r} not found"}
    step_entry["variants"] = new_list
    save_step_pools(skill_dir, data)
    return {"success": True, "message": f"Removed variant {variant_id!r}"}


def patch_step_variant_body(
    skill_dir: Path,
    step_id: str,
    variant_id: str,
    old_string: str,
    new_string: str,
) -> dict[str, Any]:
    if variant_id == BASELINE_ID:
        return {
            "success": False,
            "error": "patch baseline text with skill_manage(action='patch') on SKILL.md",
        }
    data = load_step_pools(skill_dir)
    steps = data.get("steps") or {}
    step_entry = steps.get(step_id)
    if not isinstance(step_entry, dict):
        return {"success": False, "error": f"unknown step_id {step_id!r}"}
    variants = step_entry.get("variants")
    if not isinstance(variants, list):
        return {"success": False, "error": "no variants"}
    for v in variants:
        if isinstance(v, dict) and str(v.get("id")) == variant_id:
            body = v.get("body", "")
            if not isinstance(body, str):
                return {"success": False, "error": "invalid body"}
            if old_string not in body:
                return {
                    "success": False,
                    "error": "old_string not found in variant body",
                }
            count = body.count(old_string)
            if count > 1:
                return {
                    "success": False,
                    "error": "old_string matches multiple times; include more context",
                }
            v["body"] = body.replace(old_string, new_string, 1)
            save_step_pools(skill_dir, data)
            return {"success": True, "message": f"Patched variant {variant_id!r}"}
    return {"success": False, "error": f"variant {variant_id!r} not found"}


def set_manual_variant_order(
    skill_dir: Path,
    step_id: str,
    ordered_variant_ids: List[str],
) -> dict[str, Any]:
    if not _STEP_ID_RE.match(step_id):
        return {"success": False, "error": f"invalid step_id: {step_id!r}"}
    data = load_step_pools(skill_dir)
    steps = data.setdefault("steps", {})
    step_entry = steps.setdefault(step_id, {})
    if not isinstance(step_entry, dict):
        step_entry = {}
        steps[step_id] = step_entry
    step_entry["manual_order"] = [str(x) for x in ordered_variant_ids]
    save_step_pools(skill_dir, data)
    return {"success": True, "message": f"Set manual_order for step {step_id!r}"}


def clear_manual_variant_order(skill_dir: Path, step_id: str) -> dict[str, Any]:
    data = load_step_pools(skill_dir)
    steps = data.get("steps") or {}
    step_entry = steps.get(step_id)
    if isinstance(step_entry, dict) and "manual_order" in step_entry:
        del step_entry["manual_order"]
        save_step_pools(skill_dir, data)
    return {"success": True, "message": f"Cleared manual_order for step {step_id!r}"}


def list_step_pools(skill_dir: Path) -> dict[str, Any]:
    data = load_step_pools(skill_dir)
    return {"success": True, "step_pools": data}
