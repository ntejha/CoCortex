"""
engine/memory_engine.py

FIXED VERSION:
- Applies lifecycle filtering
- Applies reliability threshold
- Respects status
- Limits scoring to bounded candidate pool
- Actually uses compute_reliability()
"""

import json
import logging
from typing import List, Optional

from memory.store import MemoryStore
from memory.schemas import MemoryItem
from memory.embeddings import EmbeddingEngine, FAISSIndex
from memory.scoring import compute_reliability

logger = logging.getLogger(__name__)


_EXCLUDED_LIFECYCLE = {"stale", "deprecated", "archived"}
_MIN_RELIABILITY = 0.35   # tune experimentally
_MAX_CANDIDATES = 30      # prevents latency explosion


class MemoryEngine:
    CONV_PREFIX = "conv::"

    def __init__(self, db_path: str = "cocortex_memory.db", embedding_model: Optional[str] = None):
        self.store = MemoryStore(db_path)
        self._embedding_engine: Optional[EmbeddingEngine] = None
        self._faiss_index: Optional[FAISSIndex] = None
        self._embedding_model = embedding_model

    # ------------------------------------------------------------------
    # Embedding init
    # ------------------------------------------------------------------

    def _ensure_embeddings(self):
        if self._embedding_engine is not None:
            return

        self._embedding_engine = EmbeddingEngine(self._embedding_model)
        dim = self._embedding_engine.dimension
        self._faiss_index = FAISSIndex(dimension=dim)

        pairs = self.store.get_all_embeddings()
        if pairs:
            id_vec_pairs = [
                (mid, EmbeddingEngine.deserialize(blob, dim))
                for mid, blob in pairs
            ]
            self._faiss_index.rebuild(id_vec_pairs)

    # ------------------------------------------------------------------
    # Load full MemoryItems (NOT dict records)
    # ------------------------------------------------------------------

    def _load_memory_items(self, session_id: str) -> List[MemoryItem]:
        return self.store.get_memories_by_session(session_id)

    # ------------------------------------------------------------------
    # Filter pipeline (THIS WAS MISSING)
    # ------------------------------------------------------------------

    def _filter_memories(self, memories: List[MemoryItem]) -> List[MemoryItem]:
        filtered = []

        for mem in memories:
            if mem.status != "active":
                continue

            if mem.lifecycle_state in _EXCLUDED_LIFECYCLE:
                continue

            reliability = compute_reliability(mem)
            if reliability < _MIN_RELIABILITY:
                continue

            filtered.append(mem)

        return filtered

    # ------------------------------------------------------------------
    # HYBRID RETRIEVAL (FIXED)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        session_id: str,
        query: str,
        top_n: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> List[dict]:

        memories = self._load_memory_items(session_id)
        if not memories:
            return []

        # 🚨 FIX: apply lifecycle + reliability filtering
        memories = self._filter_memories(memories)
        if not memories:
            return []

        # 🚨 FIX: Hard candidate cap to prevent O(n) explosion
        memories = memories[:_MAX_CANDIDATES]

        # Prepare content for scoring
        memory_texts = {}
        for mem in memories:
            text = mem.content
            if text.startswith(self.CONV_PREFIX):
                try:
                    payload = json.loads(text[len(self.CONV_PREFIX):])
                    text = f"{payload.get('input', '')} {payload.get('output', '')}"
                except json.JSONDecodeError:
                    pass
            memory_texts[str(mem.id)] = text

        # Semantic scoring
        try:
            self._ensure_embeddings()
            query_vec = self._embedding_engine.encode(query)
            faiss_results = self._faiss_index.search(query_vec, top_k=len(memories))
            faiss_scores = {mid: score for mid, score in faiss_results}
        except Exception as e:
            logger.warning("Semantic retrieval failed: %s", e)
            faiss_scores = {}

        query_tokens = set(query.lower().split())
        scored = []

        for mem in memories:
            mid = str(mem.id)
            text = memory_texts.get(mid, "").lower()

            sem_score = max(0.0, faiss_scores.get(mid, 0.0))

            if query_tokens:
                kw_score = sum(1 for t in query_tokens if t in text) / len(query_tokens)
            else:
                kw_score = 0.0

            final = (semantic_weight * sem_score) + (keyword_weight * kw_score)

            if final > 0:
                scored.append((final, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _, mem in scored[:top_n]:
            results.append({
                "memory_id": str(mem.id),
                "input": mem.content,
                "output": "",
                "confidence": mem.confidence_score,
                "lifecycle_state": mem.lifecycle_state,
                "reliability": compute_reliability(mem),
            })

        return results