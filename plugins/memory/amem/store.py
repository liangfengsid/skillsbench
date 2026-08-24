"""Persistent A-Mem store for Hermes benchmark runs.

Upstream ``AgenticMemorySystem`` resets an in-memory Chroma client on every
construct, which would wipe cross-task memories. This wrapper:

- Replaces the retriever with a disk-backed Chroma client under ``persist_dir``
- Snapshots notes to ``amem_notes.json`` so AppWorld task subprocesses resume
- Reuses one process-level singleton (SkillsBench / ALFWorld in-process loops)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SINGLETON: Optional["HermesAMemStore"] = None
# Tests inject a fake store without chromadb / sentence-transformers.
_OVERRIDE_STORE: Optional["HermesAMemStore"] = None

_NOTES_NAME = "amem_notes.json"
_CHROMA_SUBDIR = "chroma"


def _env_path() -> Path:
    raw = (os.getenv("HERMES_AMEM_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    from hermes_constants import get_hermes_home

    return (get_hermes_home() / "amem").resolve()


def _env_k(default: int = 5) -> int:
    raw = (os.getenv("HERMES_AMEM_K") or "").strip()
    if not raw:
        return default
    try:
        return max(1, min(int(raw), 32))
    except ValueError:
        return default


def _llm_kwargs() -> Dict[str, Any]:
    model = (os.getenv("HERMES_AMEM_LLM_MODEL") or os.getenv("OPENAI_MODEL") or "").strip()
    api_key = (os.getenv("HERMES_AMEM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip() or None
    kwargs: Dict[str, Any] = {
        "llm_backend": (os.getenv("HERMES_AMEM_LLM_BACKEND") or "openai").strip() or "openai",
        "llm_model": model or "gpt-4o-mini",
        "api_key": api_key,
    }
    return kwargs


class HermesAMemStore:
    """Disk-backed A-Mem notes + search."""

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir.expanduser().resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._notes_path = self.persist_dir / _NOTES_NAME
        self._lock = threading.Lock()
        self._system = self._open_system()
        self._reload_notes_if_needed()

    def _open_system(self):
        try:
            from agentic_memory.memory_system import AgenticMemorySystem
            from agentic_memory.retrievers import ChromaRetriever
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "A-Mem baseline requires the agentic-memory package. "
                'Install with: pip install -e ".[amem]" '
                "(or pip install git+https://github.com/WujiangXu/A-mem-sys.git)"
            ) from exc

        llm = _llm_kwargs()
        system = AgenticMemorySystem(
            model_name=os.getenv("HERMES_AMEM_EMBED_MODEL") or "all-MiniLM-L6-v2",
            **llm,
        )
        chroma_dir = self.persist_dir / _CHROMA_SUBDIR
        chroma_dir.mkdir(parents=True, exist_ok=True)
        retriever = ChromaRetriever.__new__(ChromaRetriever)
        retriever.client = chromadb.PersistentClient(path=str(chroma_dir))
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        retriever.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=system.model_name
        )
        retriever.collection = retriever.client.get_or_create_collection(
            name="hermes_amem",
            embedding_function=retriever.embedding_function,
        )
        system.retriever = retriever
        return system

    def _reload_notes_if_needed(self) -> None:
        if self._system.memories:
            return
        if not self._notes_path.is_file():
            return
        try:
            payload = json.loads(self._notes_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read A-Mem notes snapshot %s", self._notes_path)
            return
        notes = payload.get("notes") if isinstance(payload, dict) else None
        if not isinstance(notes, list):
            return
        from agentic_memory.memory_system import MemoryNote

        for row in notes:
            if not isinstance(row, dict) or not row.get("content"):
                continue
            note = MemoryNote(
                content=str(row["content"]),
                id=row.get("id"),
                keywords=list(row.get("keywords") or []),
                context=str(row.get("context") or "General"),
                tags=list(row.get("tags") or []),
                timestamp=row.get("timestamp"),
            )
            self._system.memories[note.id] = note
            try:
                existing = self._system.retriever.collection.get(ids=[note.id])
                have = (existing or {}).get("ids") or []
                if have:
                    continue
            except Exception:
                pass
            metadata = {
                "id": note.id,
                "content": note.content,
                "keywords": note.keywords,
                "links": note.links,
                "retrieval_count": note.retrieval_count,
                "timestamp": note.timestamp,
                "last_accessed": note.last_accessed,
                "context": note.context,
                "evolution_history": note.evolution_history,
                "category": note.category,
                "tags": note.tags,
            }
            try:
                self._system.retriever.add_document(note.content, metadata, note.id)
            except Exception:
                logger.debug("A-Mem chroma re-add failed for %s", note.id, exc_info=True)

    def persist(self) -> None:
        notes = []
        for note in self._system.memories.values():
            notes.append(
                {
                    "id": note.id,
                    "content": note.content,
                    "keywords": list(note.keywords or []),
                    "context": note.context,
                    "tags": list(note.tags or []),
                    "timestamp": note.timestamp,
                }
            )
        tmp = self._notes_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"schema": "hermes.amem_notes.v1", "notes": notes}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(self._notes_path)

    def add_note(self, content: str, **kwargs: Any) -> str:
        text = (content or "").strip()
        if not text:
            return ""
        with self._lock:
            note_id = self._system.add_note(text, **kwargs)
            try:
                self.persist()
            except OSError:
                logger.debug("A-Mem persist failed", exc_info=True)
            return note_id

    def search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q or not self._system.memories:
            return []
        top_k = _env_k() if k is None else max(1, int(k))
        top_k = min(top_k, len(self._system.memories))
        with self._lock:
            try:
                results = self._system.retriever.search(q, top_k)
            except Exception:
                logger.debug("A-Mem search failed", exc_info=True)
                return []
        out: List[Dict[str, Any]] = []
        ids = (results.get("ids") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        for i, doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            content = str(meta.get("content") or "")
            if not content:
                note = self._system.memories.get(doc_id)
                content = note.content if note is not None else ""
            if not content:
                continue
            out.append(
                {
                    "id": doc_id,
                    "content": content,
                    "context": meta.get("context") or "",
                    "keywords": meta.get("keywords") or [],
                    "tags": meta.get("tags") or [],
                }
            )
        return out

    @property
    def n_notes(self) -> int:
        return len(self._system.memories)


def get_store() -> HermesAMemStore:
    if _OVERRIDE_STORE is not None:
        return _OVERRIDE_STORE
    global _SINGLETON
    with _LOCK:
        persist = _env_path()
        if _SINGLETON is not None and _SINGLETON.persist_dir == persist:
            return _SINGLETON
        _SINGLETON = HermesAMemStore(persist)
        return _SINGLETON


def reset_store_for_tests() -> None:
    global _SINGLETON, _OVERRIDE_STORE
    _SINGLETON = None
    _OVERRIDE_STORE = None
