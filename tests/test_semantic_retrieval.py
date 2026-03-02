"""
tests/test_semantic_retrieval.py

Tests for the FAISS + sentence-transformers semantic retrieval system.

These tests use the real sentence-transformers model (all-MiniLM-L6-v2)
for embedding accuracy validation — no mocking of the ML model.
"""
import os
import tempfile

import numpy as np
import pytest

from memory.embeddings import EmbeddingEngine, FAISSIndex
from memory.store import MemoryStore
from memory.schemas import MemoryItem
from engine.memory_engine import MemoryEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """EmbeddingEngine backed by all-MiniLM-L6-v2."""
    return EmbeddingEngine("all-MiniLM-L6-v2")


@pytest.fixture
def tmp_db():
    """Temp SQLite database that's cleaned up after each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def store(tmp_db):
    return MemoryStore(tmp_db)


@pytest.fixture
def mem_engine(tmp_db):
    return MemoryEngine(db_path=tmp_db, embedding_model="all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# EmbeddingEngine
# ---------------------------------------------------------------------------

class TestEmbeddingEngine:

    def test_encode_returns_correct_dimension(self, engine):
        vec = engine.encode("Hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert vec.shape == (384,)

    def test_encode_batch(self, engine):
        vecs = engine.encode_batch(["Hello", "World", "Test"])
        assert vecs.shape == (3, 384)
        assert vecs.dtype == np.float32

    def test_cosine_similarity_identical_texts(self, engine):
        v1 = engine.encode("Photosynthesis converts CO2 to glucose")
        v2 = engine.encode("Photosynthesis converts CO2 to glucose")
        sim = float(v1 @ v2)
        assert sim > 0.99, f"Identical texts should have sim ~1.0, got {sim}"

    def test_cosine_similarity_unrelated_texts(self, engine):
        v1 = engine.encode("Photosynthesis requires sunlight and water")
        v2 = engine.encode("The stock market crashed in 2008")
        sim = float(v1 @ v2)
        assert sim < 0.5, f"Unrelated texts should have sim < 0.5, got {sim}"

    def test_cosine_similarity_related_texts(self, engine):
        v1 = engine.encode("Plants use sunlight to make food")
        v2 = engine.encode("Photosynthesis converts CO2 into glucose using light")
        sim = float(v1 @ v2)
        # Moderately related but different vocabulary — threshold at 0.4
        assert sim > 0.4, f"Related texts should have sim > 0.4, got {sim}"

    def test_serialize_roundtrip(self, engine):
        original = engine.encode("Test roundtrip")
        blob = EmbeddingEngine.serialize(original)
        restored = EmbeddingEngine.deserialize(blob)
        np.testing.assert_array_almost_equal(original, restored, decimal=6)


# ---------------------------------------------------------------------------
# FAISSIndex
# ---------------------------------------------------------------------------

class TestFAISSIndex:

    def test_add_and_search(self, engine):
        index = FAISSIndex(dimension=384)
        v1 = engine.encode("Machine learning uses neural networks")
        v2 = engine.encode("Cooking requires fresh ingredients")
        index.add("mem1", v1)
        index.add("mem2", v2)

        query = engine.encode("Deep learning and AI models")
        results = index.search(query, top_k=2)
        assert len(results) == 2
        # Machine learning should be more similar to deep learning than cooking
        assert results[0][0] == "mem1"
        assert results[0][1] > results[1][1]

    def test_search_empty_index(self):
        index = FAISSIndex(dimension=384)
        query = np.random.randn(384).astype(np.float32)
        results = index.search(query, top_k=5)
        assert results == []

    def test_count(self, engine):
        index = FAISSIndex(dimension=384)
        assert index.count() == 0
        index.add("a", engine.encode("Hello"))
        assert index.count() == 1
        index.add("b", engine.encode("World"))
        assert index.count() == 2

    def test_remove(self, engine):
        index = FAISSIndex(dimension=384)
        v = engine.encode("Test removal")
        index.add("mem1", v)
        assert index.count() == 1
        index.remove("mem1")
        assert index.count() == 0
        # Removed items should not appear in search results
        results = index.search(v, top_k=5)
        assert not any(mid == "mem1" for mid, _ in results)

    def test_rebuild(self, engine):
        index = FAISSIndex(dimension=384)
        pairs = [
            ("m1", engine.encode("Alpha")),
            ("m2", engine.encode("Beta")),
            ("m3", engine.encode("Gamma")),
        ]
        index.rebuild(pairs)
        assert index.count() == 3

    def test_duplicate_add_ignored(self, engine):
        index = FAISSIndex(dimension=384)
        v = engine.encode("Same text")
        index.add("mem1", v)
        index.add("mem1", v)  # should be silently ignored
        assert index.count() == 1


# ---------------------------------------------------------------------------
# Hybrid Retrieval (MemoryEngine.retrieve)
# ---------------------------------------------------------------------------

class TestHybridRetrieval:

    def _add_memories(self, store, session_id, contents):
        """Helper: add memories with given contents to a session."""
        for text in contents:
            mem = MemoryItem(
                content=f"conv::{{'input': '{text}', 'output': ''}}",
                source_agent="worker",
                memory_type="episodic",
                task_ids=[session_id],
                confidence_score=0.8,
            )
            store.add_memory(mem)

    def test_semantic_finds_relevant_memory(self, mem_engine):
        """Query with no keyword overlap but same meaning should still find the memory."""
        import json
        sid = "test-session"
        # Store memory about photosynthesis
        content = json.dumps({"input": "Photosynthesis converts carbon dioxide into glucose using sunlight", "output": ""})
        mem = MemoryItem(
            content=f"conv::{content}",
            source_agent="worker",
            memory_type="episodic",
            task_ids=[sid],
            confidence_score=0.8,
        )
        mem_engine.store.add_memory(mem)
        mem_engine._generate_and_store_embedding(mem)

        # Query using completely different words but same meaning
        results = mem_engine.retrieve(sid, "How do plants make food from light?", top_n=5)
        assert len(results) >= 1
        assert "Photosynthesis" in results[0].get("input", "")

    def test_keyword_miss_semantic_hit(self, mem_engine):
        """No keyword overlap but semantically related → still found."""
        import json
        sid = "session-kw-miss"
        content = json.dumps({"input": "Neural networks learn by adjusting weights via backpropagation", "output": ""})
        mem = MemoryItem(
            content=f"conv::{content}",
            source_agent="worker",
            memory_type="episodic",
            task_ids=[sid],
            confidence_score=0.8,
        )
        mem_engine.store.add_memory(mem)
        mem_engine._generate_and_store_embedding(mem)

        # Query has zero keyword overlap
        results = mem_engine.retrieve(sid, "How do AI models train themselves?", top_n=5)
        assert len(results) >= 1

    def test_retrieve_backward_compatible(self, mem_engine):
        """Old-style retrieve(session_id, keyword) call still works."""
        import json
        sid = "compat-session"
        content = json.dumps({"input": "Python is a programming language", "output": ""})
        mem = MemoryItem(
            content=f"conv::{content}",
            source_agent="worker",
            memory_type="episodic",
            task_ids=[sid],
            confidence_score=0.8,
        )
        mem_engine.store.add_memory(mem)
        # Don't generate embedding — test keyword fallback
        results = mem_engine.retrieve(sid, "Python")
        assert len(results) >= 1

    def test_empty_session_returns_empty(self, mem_engine):
        results = mem_engine.retrieve("nonexistent", "anything")
        assert results == []


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

class TestBackfill:

    def test_backfill_generates_embeddings(self, mem_engine):
        """Memories added without embeddings get them via backfill."""
        import json
        sid = "backfill-session"
        content = json.dumps({"input": "Test content for backfill", "output": ""})
        mem = MemoryItem(
            content=f"conv::{content}",
            source_agent="worker",
            memory_type="episodic",
            task_ids=[sid],
            confidence_score=0.8,
        )
        mem_engine.store.add_memory(mem)

        # No embeddings yet
        assert len(mem_engine.store.get_all_embeddings()) == 0

        # Backfill
        count = mem_engine.backfill_embeddings()
        assert count == 1

        # Now has embedding
        assert len(mem_engine.store.get_all_embeddings()) == 1

    def test_backfill_idempotent(self, mem_engine):
        """Running backfill twice doesn't regenerate existing embeddings."""
        import json
        sid = "idem-session"
        content = json.dumps({"input": "Idempotent test", "output": ""})
        mem = MemoryItem(
            content=f"conv::{content}",
            source_agent="worker",
            memory_type="episodic",
            task_ids=[sid],
            confidence_score=0.8,
        )
        mem_engine.store.add_memory(mem)

        first = mem_engine.backfill_embeddings()
        second = mem_engine.backfill_embeddings()
        assert first == 1
        assert second == 0


# ---------------------------------------------------------------------------
# Store embedding methods
# ---------------------------------------------------------------------------

class TestStoreEmbeddings:

    def test_set_and_get_embedding(self, store):
        mem = MemoryItem(
            content="Test embedding storage",
            source_agent="worker",
            memory_type="episodic",
            confidence_score=0.8,
        )
        store.add_memory(mem)

        blob = np.random.randn(384).astype(np.float32).tobytes()
        store.set_embedding(mem.id, blob)

        all_emb = store.get_all_embeddings()
        assert len(all_emb) == 1
        assert all_emb[0][0] == str(mem.id)
        assert all_emb[0][1] == blob

    def test_get_all_embeddings_excludes_null(self, store):
        """Memories without embeddings should not be returned."""
        mem = MemoryItem(
            content="No embedding here",
            source_agent="worker",
            memory_type="episodic",
            confidence_score=0.8,
        )
        store.add_memory(mem)
        assert len(store.get_all_embeddings()) == 0
