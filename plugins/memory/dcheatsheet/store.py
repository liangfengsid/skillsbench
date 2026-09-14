"""Persistent Dynamic Cheatsheet store (DC-Cu / DC-RS / hybrid).

Suzgun et al., arXiv:2504.07952. Official curator prompts live beside this
module. Hermes remains the *generator*; this store only curates and injects.

Disk layout (AppWorld task subprocesses reload this):

    persist_dir/cheatsheet.txt
    persist_dir/episodes.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from plugins.memory.dcheatsheet.extractor import extract_cheatsheet

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SINGLETON: Optional["HermesDCStore"] = None
_OVERRIDE_STORE: Optional["HermesDCStore"] = None
_OVERRIDE_CURATOR: Optional[Callable[[str], str]] = None

_CHEATSHEET_NAME = "cheatsheet.txt"
_EPISODES_NAME = "episodes.jsonl"
_MAX_SHEET_CHARS = 12000
_MAX_TURN_CHARS = 1800
_EMPTY = "(empty)"

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _env_path() -> Path:
    raw = (os.getenv("HERMES_DC_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    from hermes_constants import get_hermes_home

    return (get_hermes_home() / "dcheatsheet").resolve()


def _env_mode(default: str = "cu") -> str:
    raw = (os.getenv("HERMES_DC_MODE") or default).strip().lower()
    if raw in ("cu", "cumulative", "dc-cu", "dc_cu"):
        return "cu"
    if raw in ("rs", "retrieval", "retrievalsynthesis", "dc-rs", "dc_rs"):
        return "rs"
    if raw in ("curetr", "hybrid", "cumulative_retrieval", "cu+rs"):
        return "curetr"
    return default


def _env_k(default: int = 3) -> int:
    raw = (os.getenv("HERMES_DC_K") or "").strip()
    if not raw:
        return default
    try:
        return max(1, min(int(raw), 16))
    except ValueError:
        return default


def _readonly_from_env() -> bool:
    return os.getenv("HERMES_DC_READONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
        "freeze",
        "frozen",
        "readonly",
        "read-only",
    )


def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def _clip(text: str, n: int) -> str:
    t = (text or "").strip()
    if len(t) <= n:
        return t
    return t[: n - 3] + "..."


class HermesDCStore:
    """Disk-backed cheatsheet + episode log."""

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir.expanduser().resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._sheet_path = self.persist_dir / _CHEATSHEET_NAME
        self._episodes_path = self.persist_dir / _EPISODES_NAME
        self._lock = threading.Lock()
        self.cheatsheet = self._read_sheet()
        self.episodes: List[Dict[str, str]] = self._read_episodes()
        self.n_curations = 0
        self.last_prefetch: Dict[str, Any] = {
            "mode": _env_mode(),
            "sheet_chars": len(self.cheatsheet),
            "n_episodes": len(self.episodes),
            "n_retrieved": 0,
        }

    def _read_sheet(self) -> str:
        if not self._sheet_path.is_file():
            return ""
        try:
            return self._sheet_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _read_episodes(self) -> List[Dict[str, str]]:
        if not self._episodes_path.is_file():
            return []
        out: List[Dict[str, str]] = []
        try:
            for line in self._episodes_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and (row.get("query") or row.get("answer")):
                    out.append(
                        {
                            "query": str(row.get("query") or ""),
                            "answer": str(row.get("answer") or ""),
                        }
                    )
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read DC episodes %s", self._episodes_path)
        return out

    def persist(self) -> None:
        tmp = self._sheet_path.with_suffix(".txt.tmp")
        tmp.write_text(self.cheatsheet + ("\n" if self.cheatsheet else ""), encoding="utf-8")
        tmp.replace(self._sheet_path)
        ep_tmp = self._episodes_path.with_suffix(".jsonl.tmp")
        with ep_tmp.open("w", encoding="utf-8") as fh:
            for row in self.episodes:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        ep_tmp.replace(self._episodes_path)

    def inject_text(self, *, query: str = "", readonly: Optional[bool] = None) -> str:
        """Cheatsheet (and optional retrieved examples) for ``<memory-context>``.

        When ``readonly`` (or ``HERMES_DC_READONLY``) is set, DC-RS skips the
        curator LLM and injects the frozen sheet (+ retrieved examples for
        ``curetr``).
        """
        mode = _env_mode()
        frozen = _readonly_from_env() if readonly is None else bool(readonly)
        retrieved = ""
        n_ret = 0
        if mode in ("rs", "curetr") and query.strip() and self.episodes:
            pairs = self._retrieve(query, k=_env_k())
            n_ret = len(pairs)
            retrieved = _format_retrieved(pairs)
        if mode == "rs" and not frozen:
            sheet = self._curate_rs(query, retrieved)
        else:
            sheet = self.cheatsheet
        self.last_prefetch = {
            "mode": mode,
            "sheet_chars": len(sheet),
            "n_episodes": len(self.episodes),
            "n_retrieved": n_ret,
            "readonly": frozen,
        }
        parts = []
        if sheet.strip():
            parts.append(sheet.strip())
        elif mode != "rs":
            parts.append(_EMPTY)
        if mode == "curetr" and retrieved:
            parts.append(retrieved)
        if mode == "rs" and frozen and retrieved and not sheet.strip():
            parts.append(retrieved)
        text = "\n\n".join(p for p in parts if p).strip()
        if len(text) > _MAX_SHEET_CHARS:
            text = text[: _MAX_SHEET_CHARS]
        return text

    def sync_turn(self, query: str, answer: str) -> None:
        """Legacy single-turn append + curate (also used by ``flush_turns``)."""
        self.flush_turns([(query, answer)])

    def flush_turns(self, turns: List[Tuple[str, str]]) -> None:
        """Append one or more turns, then run at most one CU curator update."""
        if _readonly_from_env():
            return
        cleaned: List[Tuple[str, str]] = []
        for query, answer in turns:
            q = _clip(query, _MAX_TURN_CHARS)
            a = _clip(answer, _MAX_TURN_CHARS)
            if q or a:
                cleaned.append((q, a))
        if not cleaned:
            return
        mode = _env_mode()
        with self._lock:
            for q, a in cleaned:
                self.episodes.append({"query": q, "answer": a})
            if mode in ("cu", "curetr"):
                if len(cleaned) == 1:
                    q_cur, a_cur = cleaned[0]
                else:
                    n = len(cleaned)
                    q_parts = [
                        f"Turn {i}/{n}:\n{q}" for i, (q, _) in enumerate(cleaned, start=1)
                    ]
                    a_parts = [
                        f"Turn {i}/{n}:\n{a}" for i, (_, a) in enumerate(cleaned, start=1)
                    ]
                    q_cur = _clip("\n".join(q_parts), _MAX_TURN_CHARS)
                    a_cur = _clip("\n".join(a_parts), _MAX_TURN_CHARS)
                self._curate_cu(q_cur, a_cur)
            try:
                self.persist()
            except OSError:
                logger.debug("DC persist failed", exc_info=True)

    def _curate_cu(self, question: str, model_answer: str) -> None:
        template = _load_prompt("curator_cu.txt")
        prompt = (
            template.replace("[[PREVIOUS_CHEATSHEET]]", self.cheatsheet or _EMPTY)
            .replace("[[QUESTION]]", question or _EMPTY)
            .replace("[[MODEL_ANSWER]]", model_answer or _EMPTY)
        )
        raw = _run_curator(prompt)
        if "<cheatsheet>" not in (raw or "").lower():
            return
        new = extract_cheatsheet(raw, self.cheatsheet or _EMPTY)
        if new and new != _EMPTY:
            self.cheatsheet = new[:_MAX_SHEET_CHARS]
            self.n_curations += 1

    def _curate_rs(self, next_input: str, retrieved_block: str) -> str:
        template = _load_prompt("curator_rs.txt")
        prompt = (
            template.replace("[[PREVIOUS_CHEATSHEET]]", self.cheatsheet or _EMPTY)
            .replace("[[PREVIOUS_INPUT_OUTPUT_PAIRS]]", retrieved_block or _EMPTY)
            .replace("[[NEXT_INPUT]]", next_input or _EMPTY)
        )
        raw = _run_curator(prompt)
        if "<cheatsheet>" not in (raw or "").lower():
            return self.cheatsheet or retrieved_block
        new = extract_cheatsheet(raw, self.cheatsheet or retrieved_block or _EMPTY)
        if new and new != _EMPTY:
            with self._lock:
                self.cheatsheet = new[:_MAX_SHEET_CHARS]
                self.n_curations += 1
                try:
                    self.persist()
                except OSError:
                    logger.debug("DC persist failed", exc_info=True)
            return self.cheatsheet
        return self.cheatsheet or retrieved_block

    def _retrieve(self, query: str, k: int) -> List[Tuple[float, Dict[str, str]]]:
        embedder = _embedder()
        if embedder is None:
            return [(0.0, row) for row in self.episodes[-k:]]
        try:
            qv = embedder([query])[0]
            corpus = [row.get("query") or "" for row in self.episodes]
            ev = embedder(corpus)
        except Exception:
            logger.debug("DC embed failed", exc_info=True)
            return [(0.0, row) for row in self.episodes[-k:]]
        scores = _cosine_scores(qv, ev)
        ranked = sorted(
            zip(scores, self.episodes),
            key=lambda item: float(item[0]),
            reverse=True,
        )
        return [(float(s), row) for s, row in ranked[:k]]


def _format_retrieved(pairs: List[Tuple[float, Dict[str, str]]]) -> str:
    if not pairs:
        return ""
    lines = [
        "### PREVIOUS SOLUTIONS (START)",
        "",
        "Note: The input-output pairs listed below are taken from previous "
        "test cases. Do not copy them blindly.",
        "",
    ]
    for i, (sim, row) in enumerate(reversed(pairs), start=1):
        lines.append(f"#### Previous Input #{i} (Similarity: {sim:.2f}):")
        lines.append("")
        lines.append(row.get("query") or "")
        lines.append("")
        lines.append(f"#### Model Solution to Previous Input #{i}:")
        lines.append("")
        lines.append(row.get("answer") or "")
        lines.append("---")
        lines.append("---")
        lines.append("")
    lines.append("#### PREVIOUS SOLUTIONS (END)")
    return "\n".join(lines)


def _cosine_scores(query_vec, matrix) -> List[float]:
    import math

    def _norm(v):
        return math.sqrt(sum(float(x) * float(x) for x in v)) or 1e-9

    qn = _norm(query_vec)
    q = [float(x) / qn for x in query_vec]
    out = []
    for row in matrix:
        rn = _norm(row)
        r = [float(x) / rn for x in row]
        n = min(len(q), len(r))
        out.append(sum(q[i] * r[i] for i in range(n)))
    return out


_EMBED_FN: Optional[Callable[[List[str]], List[List[float]]]] = None
_EMBED_FAILED = False


def _embedder() -> Optional[Callable[[List[str]], List[List[float]]]]:
    global _EMBED_FN, _EMBED_FAILED
    if _EMBED_FN is not None:
        return _EMBED_FN
    if _EMBED_FAILED:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("HERMES_DC_EMBED_MODEL") or "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)

        def _encode(texts: List[str]) -> List[List[float]]:
            vecs = model.encode(list(texts), show_progress_bar=False)
            return [list(map(float, row)) for row in vecs]

        _EMBED_FN = _encode
        return _EMBED_FN
    except Exception:
        _EMBED_FAILED = True
        logger.info("DC retrieval embeddings unavailable; using recency fallback")
        return None


def _run_curator(prompt: str) -> str:
    if _OVERRIDE_CURATOR is not None:
        return _OVERRIDE_CURATOR(prompt) or ""
    model = (
        os.getenv("HERMES_DC_LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    ).strip()
    api_key = (
        os.getenv("HERMES_DC_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    ).strip() or None
    base_url = (
        os.getenv("HERMES_DC_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or ""
    ).strip() or None
    try:
        from openai import OpenAI

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=int(os.getenv("HERMES_DC_CURATOR_MAX_TOKENS") or 4096),
        )
        return (resp.choices[0].message.content or "") if resp.choices else ""
    except Exception:
        logger.warning("DC curator LLM call failed", exc_info=True)
        return ""


def get_store() -> HermesDCStore:
    if _OVERRIDE_STORE is not None:
        return _OVERRIDE_STORE
    global _SINGLETON
    with _LOCK:
        persist = _env_path()
        if _SINGLETON is not None and _SINGLETON.persist_dir == persist:
            return _SINGLETON
        _SINGLETON = HermesDCStore(persist)
        return _SINGLETON


def reset_store_for_tests() -> None:
    global _SINGLETON, _OVERRIDE_STORE, _OVERRIDE_CURATOR, _EMBED_FN, _EMBED_FAILED
    _SINGLETON = None
    _OVERRIDE_STORE = None
    _OVERRIDE_CURATOR = None
    _EMBED_FN = None
    _EMBED_FAILED = False
