"""
engine/memory_engine.py

Production-ready memory facade for CoCortex.

Supports:
1. Task-scoped mode  — session_id maps to a task
2. Conversation mode — Human/Assistant turns (LangChain-compatible)
3. Hybrid retrieval  — FAISS semantic search + keyword overlap
"""

import json
import logging
from typing import List, Optional

from memory.store import MemoryStore
from memory.schemas import MemoryItem
from memory.embeddings import EmbeddingEngine, FAISSIndex

logger = logging.getLogger(__name__)


class MemoryEngine:
    CONV_PREFIX = "conv::"  # marks conversation-mode records in content

    def __init__(self, db_path: str = "cocortex_memory.db", embedding_model: Optional[str] = None):
        self.store = MemoryStore(db_path)
        self._embedding_engine: Optional[EmbeddingEngine] = None
        self._faiss_index: Optional[FAISSIndex] = None
        self._embedding_model = embedding_model

    # ---- Lazy initialisation (avoids loading model if never needed) ----

    def _ensure_embeddings(self):
        """Lazy-load embedding engine and rebuild FAISS index from SQLite."""
        if self._embedding_engine is not None:
            return

        self._embedding_engine = EmbeddingEngine(self._embedding_model)
        dim = self._embedding_engine.dimension
        self._faiss_index = FAISSIndex(dimension=dim)

        # Rebuild FAISS from persisted BLOBs
        pairs = self.store.get_all_embeddings()
        if pairs:
            id_vec_pairs = [
                (mid, EmbeddingEngine.deserialize(blob, dim))
                for mid, blob in pairs
            ]
            self._faiss_index.rebuild(id_vec_pairs)
            logger.info("FAISS index loaded: %d vectors from SQLite", len(pairs))

    # ------------------------------------------------------------------
    # TASK-SCOPED MODE
    # ------------------------------------------------------------------

    def load(self, session_id: str) -> List[dict]:
        memories = self.store.get_memories_by_session(session_id)

        results = []
        for mem in memories:
            if mem.content.startswith(self.CONV_PREFIX):
                try:
                    payload = json.loads(mem.content[len(self.CONV_PREFIX):])
                    payload["memory_id"] = str(mem.id)
                    results.append(payload)
                except (json.JSONDecodeError, KeyError):
                    pass
            else:
                results.append({
                    "input": mem.content,
                    "output": "",
                    "memory_id": str(mem.id),
                })

        return results

    def save(self, session_id: str, records: List[dict]):
        for record in records:
            if record.get("memory_id"):
                continue

            content = self.CONV_PREFIX + json.dumps({
                "input":  record.get("input", ""),
                "output": record.get("output", ""),
            })

            mem = MemoryItem(
                content=content,
                source_agent="memory_manager",
                memory_type="episodic",
                task_ids=[session_id],
                confidence_score=0.8,
            )
            self.store.add_memory(mem)

            # Generate and store embedding for new memory
            self._generate_and_store_embedding(mem)

    def delete_session(self, session_id: str):
        self.store.delete_by_session(session_id)

    # ------------------------------------------------------------------
    # CONVERSATION HISTORY MODE (LangChain-compatible)
    # ------------------------------------------------------------------

    def load_history(self, session_id: str) -> str:
        records = self.load(session_id)
        lines = []
        for r in records:
            user = r.get("input", "").strip()
            assistant = r.get("output", "").strip()
            if user:
                lines.append(f"Human: {user}")
            if assistant:
                lines.append(f"Assistant: {assistant}")
        return "\n".join(lines)

    def save_turn(self, session_id: str, human: str, assistant: str):
        self.save(session_id, [{"input": human, "output": assistant}])

    # ------------------------------------------------------------------
    # HYBRID RETRIEVAL (FAISS semantic + keyword)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        session_id: str,
        query: str,
        top_n: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> List[dict]:
        """
        Hybrid retrieval: semantic similarity (FAISS) + keyword overlap.

        1. Encode query via EmbeddingEngine
        2. FAISS search for top candidates with cosine scores
        3. Compute keyword overlap scores for the same candidates
        4. Blend: final_score = semantic_weight * cos + keyword_weight * keyword
        5. Filter to session_id, return top_n

        Backward compatible: old calls like engine.retrieve(sid, "Python") work.
        """
        records = self.load(session_id)
        if not records:
            return []

        # Try semantic search
        try:
            self._ensure_embeddings()
            query_vec = self._embedding_engine.encode(query)

            # Get FAISS results (more than top_n to allow filtering)
            faiss_results = self._faiss_index.search(query_vec, top_k=top_n * 3)
            faiss_scores = {mid: score for mid, score in faiss_results}
        except Exception as e:
            logger.warning("Semantic search unavailable, falling back to keyword: %s", e)
            faiss_scores = {}

        # Score each record
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        scored = []

        for r in records:
            mid = r.get("memory_id", "")
            text = f"{r.get('input', '')} {r.get('output', '')}".lower()

            # Semantic score (from FAISS, 0-1 range for normalised vectors)
            sem_score = max(0.0, faiss_scores.get(mid, 0.0))

            # Keyword score (fraction of query tokens found in text)
            if query_tokens:
                kw_score = sum(1 for t in query_tokens if t in text) / len(query_tokens)
            else:
                kw_score = 0.0

            # Blend
            final = (semantic_weight * sem_score) + (keyword_weight * kw_score)
            if final > 0:
                scored.append((final, r))

        # Sort by blended score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_n]]

    # ------------------------------------------------------------------
    # EMBEDDING MANAGEMENT
    # ------------------------------------------------------------------

    def _generate_and_store_embedding(self, mem: MemoryItem):
        """Generate embedding for a single memory and persist it."""
        try:
            self._ensure_embeddings()
            # Strip conv:: prefix for cleaner embedding
            text = mem.content
            if text.startswith(self.CONV_PREFIX):
                try:
                    payload = json.loads(text[len(self.CONV_PREFIX):])
                    text = f"{payload.get('input', '')} {payload.get('output', '')}"
                except json.JSONDecodeError:
                    pass

            vec = self._embedding_engine.encode(text)
            blob = EmbeddingEngine.serialize(vec)
            self.store.set_embedding(mem.id, blob)
            self._faiss_index.add(str(mem.id), vec)
        except Exception as e:
            logger.warning("Failed to generate embedding for %s: %s", mem.id, e)

    def backfill_embeddings(self) -> int:
        """
        Generate embeddings for all memories that don't have one yet.
        Call once after migrating to vector search.
        Returns the number of memories backfilled.
        """
        self._ensure_embeddings()

        # Get all memory IDs that already have embeddings
        existing = {mid for mid, _ in self.store.get_all_embeddings()}

        # Get all memories
        all_mems = self.store.get_all_active_memories() + self.store.get_quarantined_memories()
        to_backfill = [m for m in all_mems if str(m.id) not in existing]

        if not to_backfill:
            logger.info("backfill_embeddings: all memories already have embeddings")
            return 0

        # Prepare texts (strip conv:: prefix)
        texts = []
        for mem in to_backfill:
            text = mem.content
            if text.startswith(self.CONV_PREFIX):
                try:
                    payload = json.loads(text[len(self.CONV_PREFIX):])
                    text = f"{payload.get('input', '')} {payload.get('output', '')}"
                except json.JSONDecodeError:
                    pass
            texts.append(text)

        # Batch encode
        vectors = self._embedding_engine.encode_batch(texts)

        # Persist and index
        for mem, vec in zip(to_backfill, vectors):
            blob = EmbeddingEngine.serialize(vec)
            self.store.set_embedding(mem.id, blob)
            self._faiss_index.add(str(mem.id), vec)

        logger.info("backfill_embeddings: generated embeddings for %d memories", len(to_backfill))
        return len(to_backfill)

    # ------------------------------------------------------------------
    # REPAIR HOOK
    # ------------------------------------------------------------------

    def repair_if_needed(self, records: List[dict]) -> List[dict]:
        seen = set()
        cleaned = []
        for r in records:
            key = (r.get("input", ""), r.get("output", ""))
            if key in seen:
                continue
            if not r.get("input") and not r.get("output"):
                continue
            seen.add(key)
            cleaned.append(r)
        return cleaned