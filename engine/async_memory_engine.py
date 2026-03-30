"""
engine/async_memory_engine.py

Corrected asynchronous equivalent of MemoryEngine.

Fixes:
- Applies lifecycle filtering
- Applies reliability threshold
- Caps candidate pool to prevent latency explosion
- Fixes FAISS score mapping
- Ensures quarantined/degraded memories never re-enter retrieval
"""

import json
import logging
import asyncio
from typing import List, Optional

from memory.async_store import AsyncMemoryStore
from memory.schemas import MemoryItem
from memory.embeddings import EmbeddingEngine, FAISSIndex
from memory.scoring import compute_reliability

logger = logging.getLogger(__name__)

_EXCLUDED_LIFECYCLE = {"stale", "deprecated", "archived"}
_MIN_RELIABILITY = 0.35
_MAX_CANDIDATES = 30


class AsyncMemoryEngine:
    CONV_PREFIX = "conv::"

    def __init__(self, db_path: str = "cocortex_memory.db", embedding_model: Optional[str] = None):
        self.store = AsyncMemoryStore(db_path)
        self._embedding_engine: Optional[EmbeddingEngine] = None
        self._faiss_index: Optional[FAISSIndex] = None
        self._embedding_model = embedding_model

    # ------------------------------------------------------------------
    # Embedding init
    # ------------------------------------------------------------------

    async def _ensure_embeddings(self):
        if self._embedding_engine is not None:
            return

        self._embedding_engine = await asyncio.to_thread(
            EmbeddingEngine, self._embedding_model
        )
        dim = self._embedding_engine.dimension
        self._faiss_index = FAISSIndex(dimension=dim)

        # Rebuild FAISS index from stored embeddings
        rows = await self.store.get_all_embeddings()
        if rows:
            id_vec_pairs = []
            for mid, blob in rows:
                vec = await asyncio.to_thread(
                    EmbeddingEngine.deserialize, blob, dim
                )
                id_vec_pairs.append((mid, vec))

            self._faiss_index.rebuild(id_vec_pairs)
            logger.info(
                "Async FAISS index rebuilt with %d vectors", len(id_vec_pairs)
            )

    # ------------------------------------------------------------------
    # LOAD MEMORY ITEMS
    # ------------------------------------------------------------------

    async def _load_memory_items(self, session_id: str) -> List[MemoryItem]:
        all_active = await self.store.get_all_active_memories()
        return [m for m in all_active if session_id in m.task_ids]

    # ------------------------------------------------------------------
    # FILTER PIPELINE
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
    # SAVE
    # ------------------------------------------------------------------

    async def save(self, session_id: str, records: List[dict]) -> None:
        for r in records:
            content = json.dumps(r) if isinstance(r, dict) else str(r)

            if isinstance(r, dict) and "input" in r and "output" in r:
                content = self.CONV_PREFIX + content

            mem = MemoryItem(
                content=content,
                memory_type="episodic",
                source_agent="worker",
            )
            mem.task_ids.append(session_id)

            await self.store.add_memory(mem)
            await self._generate_and_store_embedding(mem)

    async def _generate_and_store_embedding(self, mem: MemoryItem):
        await self._ensure_embeddings()

        text = mem.content
        if text.startswith(self.CONV_PREFIX):
            try:
                payload = json.loads(text[len(self.CONV_PREFIX):])
                text = f"{payload.get('input', '')} {payload.get('output', '')}"
            except json.JSONDecodeError:
                pass

        vec = await asyncio.to_thread(self._embedding_engine.encode, text)

        blob = EmbeddingEngine.serialize(vec)
        await self.store.set_embedding(mem.id, blob)

        self._faiss_index.add(str(mem.id), vec)

    # ------------------------------------------------------------------
    # HYBRID RETRIEVAL (FIXED)
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        session_id: str,
        query: str,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_n: int = 10,
    ) -> List[dict]:

        memories = await self._load_memory_items(session_id)
        if not memories:
            return []

        # 🚨 Apply lifecycle + reliability filtering
        memories = self._filter_memories(memories)
        if not memories:
            return []

        # 🚨 Hard cap to prevent O(n) explosion
        memories = memories[:_MAX_CANDIDATES]

        # Prepare text map
        text_map = {}
        for mem in memories:
            text = mem.content
            if text.startswith(self.CONV_PREFIX):
                try:
                    payload = json.loads(text[len(self.CONV_PREFIX):])
                    text = f"{payload.get('input', '')} {payload.get('output', '')}"
                except json.JSONDecodeError:
                    pass
            text_map[str(mem.id)] = text.lower()

        # Semantic scoring
        await self._ensure_embeddings()
        query_vec = await asyncio.to_thread(
            self._embedding_engine.encode, query
        )

        try:
            # Expect list[(id, score)]
            faiss_results = self._faiss_index.search(query_vec, k=len(memories))
            faiss_scores = {mid: score for mid, score in faiss_results}
        except Exception:
            logger.warning("FAISS search failed, using keyword-only ranking.")
            faiss_scores = {}

        query_tokens = set(query.lower().split())
        scored = []

        for mem in memories:
            mid = str(mem.id)
            text = text_map.get(mid, "")

            sem_score = max(0.0, faiss_scores.get(mid, 0.0))

            if query_tokens:
                kw_score = sum(1 for t in query_tokens if t in text) / len(query_tokens)
            else:
                kw_score = 0.0

            final_score = (semantic_weight * sem_score) + (
                keyword_weight * kw_score
            )

            if final_score > 0:
                scored.append((final_score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "memory_id": str(mem.id),
                "input": mem.content,
                "confidence": mem.confidence_score,
                "lifecycle_state": mem.lifecycle_state,
                "reliability": compute_reliability(mem),
            }
            for _, mem in scored[:top_n]
        ]

    # ------------------------------------------------------------------
    # SAVE TURN (LangChain-compatible)
    # ------------------------------------------------------------------

    async def save_turn(self, session_id: str, human: str, assistant: str) -> None:
        await self.save(session_id, [{"input": human, "output": assistant}])

    # ------------------------------------------------------------------
    # DEDUP REPAIR (Simple)
    # ------------------------------------------------------------------

    async def repair_if_needed(self, session_id: str) -> int:
        memories = await self._load_memory_items(session_id)

        seen = {}
        quarantined = 0

        for mem in memories:
            if mem.content in seen:
                await self.store.update_memory(
                    mem.id,
                    {"status": "quarantined"}
                )
                quarantined += 1
            else:
                seen[mem.content] = mem.id

        return quarantined