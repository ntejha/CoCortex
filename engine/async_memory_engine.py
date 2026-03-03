"""
engine/async_memory_engine.py

Asynchronous equivalent of MemoryEngine using the AsyncMemoryStore.
It performs concurrent tasks (where possible) and avoids blocking the event loop
on SQLite reads/writes by delegating to asyncio.to_thread.
"""

import json
import logging
import sqlite3
from typing import List, Optional

from memory.async_store import AsyncMemoryStore
from memory.schemas import MemoryItem
from memory.embeddings import EmbeddingEngine, FAISSIndex
import asyncio

logger = logging.getLogger(__name__)


class AsyncMemoryEngine:
    CONV_PREFIX = "conv::"

    def __init__(self, db_path: str = "cocortex_memory.db", embedding_model: Optional[str] = None):
        self.store = AsyncMemoryStore(db_path)
        self._embedding_engine: Optional[EmbeddingEngine] = None
        self._faiss_index: Optional[FAISSIndex] = None
        self._embedding_model = embedding_model

    async def _ensure_embeddings(self):
        """Lazy-load embedding engine and rebuild FAISS index from SQLite."""
        if self._embedding_engine is not None:
            return

        self._embedding_engine = await asyncio.to_thread(EmbeddingEngine, self._embedding_model)
        dim = self._embedding_engine.dimension
        self._faiss_index = FAISSIndex(dimension=dim)

        # Rebuild FAISS from persisted BLOBs
        def _load_blobs():
            conn = self.store._get_conn_sync()
            return conn.execute(
                "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
            ).fetchall()

        rows = await asyncio.to_thread(_load_blobs)
        pairs = [(row["id"], row["embedding"]) for row in rows]

        if pairs:
            def deserialize_all():
                return [
                    (mid, EmbeddingEngine.deserialize(blob, dim))
                    for mid, blob in pairs
                ]
            id_vec_pairs = await asyncio.to_thread(deserialize_all)
            self._faiss_index.rebuild(id_vec_pairs)
            logger.info("FAISS index loaded asynchronously: %d vectors from SQLite", len(pairs))

    async def load(self, session_id: str) -> List[dict]:
        all_active = await self.store.get_all_active_memories()

        results = []
        for mem in all_active:
            if session_id in mem.task_ids:
                if mem.content.startswith(self.CONV_PREFIX):
                    try:
                        record = json.loads(mem.content[len(self.CONV_PREFIX):])
                        record["memory_id"] = mem.id
                        results.append(record)
                    except json.JSONDecodeError:
                        pass
                else:
                    results.append({"task_data": mem.content, "memory_id": mem.id})
        return results

    async def _generate_and_store_embedding(self, memory_id: str, content: str):
        await self._ensure_embeddings()

        vector = await asyncio.to_thread(self._embedding_engine.encode, content)
        self._faiss_index.add_vector(memory_id, vector)

        blob = EmbeddingEngine.serialize(vector)

        def _store_blob():
            conn = self.store._get_conn_sync()
            conn.execute(
                "UPDATE memories SET embedding = ? WHERE id = ?",
                (blob, memory_id),
            )
            conn.commit()

        async with self.store._write_lock:
            await asyncio.to_thread(_store_blob)

    async def save(self, session_id: str, records: List[dict]) -> None:
        for r in records:
            content = json.dumps(r) if isinstance(r, dict) else str(r)
            if "input" in r and "output" in r:
                content = self.CONV_PREFIX + content

            mem = MemoryItem(
                content=content,
                memory_type="episodic",
                source_agent="worker",
            )
            mem.task_ids.append(session_id)
            await self.store.add_memory(mem)
            await self._generate_and_store_embedding(str(mem.id), mem.content)

    async def retrieve(
        self,
        session_id: str,
        query: str,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_n: int = 10,
    ) -> List[dict]:
        candidates = await self.load(session_id)
        if not candidates:
            return []

        await self._ensure_embeddings()
        query_vector = await asyncio.to_thread(self._embedding_engine.encode, query)
        faiss_results = self._faiss_index.search(query_vector, k=top_n * 2)

        memory_to_semantic_score = {mid: score for score, mid in zip(*faiss_results)}

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_candidates = []
        for r in candidates:
            if "input" in r and "output" in r:
                text = f"{r['input']} {r['output']}".lower()
            else:
                text = str(r.get("task_data", "")).lower()

            text_words = set(text.split())
            kw_score = 0.0
            if query_words:
                overlap = len(query_words.intersection(text_words))
                kw_score = overlap / len(query_words)

            mid = r["memory_id"]
            sem_score = memory_to_semantic_score.get(mid, 0.0)
            final_score = (semantic_weight * sem_score) + (keyword_weight * kw_score)

            if final_score > 0.01:
                scored_candidates.append((final_score, r))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [r for score, r in scored_candidates[:top_n]]

    async def save_turn(self, session_id: str, human: str, assistant: str) -> None:
        await self.save(session_id, [{"input": human, "output": assistant}])

    async def repair_if_needed(self, session_id: str) -> int:
        memories = await self.store.get_all_active_memories()
        session_mems = [m for m in memories if session_id in m.task_ids]

        content_map = {}
        deleted = 0
        for m in session_mems:
            if m.content in content_map:
                await self.store.update_memory(str(m.id), "status", "quarantined")
                deleted += 1
            else:
                content_map[m.content] = m.id

        return deleted
