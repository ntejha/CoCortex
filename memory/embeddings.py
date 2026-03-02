"""
memory/embeddings.py

Production-grade embedding engine using sentence-transformers + FAISS.

- EmbeddingEngine: generates and caches text embeddings
- FAISSIndex: in-memory vector index for fast cosine similarity search,
  rebuilt from SQLite BLOBs at startup

Architecture:
  SQLite (source of truth) ──startup──▶ FAISS (read-optimised cache)
  New memory → encode → store BLOB in SQLite + add to FAISS
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding Engine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """
    Generates text embeddings using sentence-transformers.

    Model is lazy-loaded on first encode() call to avoid import-time
    slowness (important for tests that never touch embeddings).
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv(
            "COCORTEX_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded: dim=%d", self._dimension)

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dimension

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text into a normalised float32 vector."""
        self._load_model()
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts in one pass. Returns (N, dim) array."""
        self._load_model()
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=64)
        return vecs.astype(np.float32)

    # ---- Serialisation (numpy ↔ SQLite BLOB) ----

    @staticmethod
    def serialize(vec: np.ndarray) -> bytes:
        """Convert a float32 numpy vector to raw bytes for BLOB storage."""
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def deserialize(blob: bytes, dimension: int = 384) -> np.ndarray:
        """Restore a numpy vector from BLOB bytes."""
        return np.frombuffer(blob, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# FAISS Index
# ---------------------------------------------------------------------------

class FAISSIndex:
    """
    In-memory FAISS index for cosine similarity search.

    Uses IndexFlatIP (inner product on normalised vectors = cosine similarity).
    Maintains a mapping between FAISS row positions and memory UUIDs.

    For <100k vectors IndexFlatIP is brute-force but still sub-millisecond.
    Can be swapped to IndexIVFFlat for 1M+ scale without API changes.
    """

    def __init__(self, dimension: int = 384):
        import faiss

        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        # Bidirectional mapping: position ↔ memory_id
        self._id_to_pos: Dict[str, int] = {}
        self._pos_to_id: Dict[int, str] = {}
        self._next_pos = 0

    def add(self, memory_id: str, vector: np.ndarray):
        """Add or update a vector for a memory ID."""
        if memory_id in self._id_to_pos:
            # FAISS IndexFlat doesn't support in-place update.
            # For simplicity, we skip duplicates (rebuild handles correctness).
            return

        vec = vector.astype(np.float32).reshape(1, -1)
        self._index.add(vec)
        pos = self._next_pos
        self._id_to_pos[memory_id] = pos
        self._pos_to_id[pos] = memory_id
        self._next_pos += 1

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for the top_k most similar vectors.
        Returns list of (memory_id, cosine_similarity) tuples, descending.
        """
        if self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        q = query_vec.astype(np.float32).reshape(1, -1)
        scores, indices = self._index.search(q, k)

        results = []
        for i in range(k):
            pos = int(indices[0][i])
            if pos < 0:
                continue
            mid = self._pos_to_id.get(pos)
            if mid:
                results.append((mid, float(scores[0][i])))
        return results

    def remove(self, memory_id: str):
        """
        Mark a memory as removed. Since IndexFlat doesn't support
        deletion, this removes it from the ID mapping so it won't
        appear in results. Full cleanup happens on next rebuild.
        """
        pos = self._id_to_pos.pop(memory_id, None)
        if pos is not None:
            self._pos_to_id.pop(pos, None)

    def count(self) -> int:
        return len(self._id_to_pos)

    def rebuild(self, id_vector_pairs: List[Tuple[str, np.ndarray]]):
        """
        Rebuild the entire index from scratch.
        Called at startup to sync FAISS with SQLite BLOBs.
        """
        import faiss

        self._index = faiss.IndexFlatIP(self.dimension)
        self._id_to_pos.clear()
        self._pos_to_id.clear()
        self._next_pos = 0

        if not id_vector_pairs:
            return

        ids = []
        vecs = []
        for mid, vec in id_vector_pairs:
            ids.append(mid)
            vecs.append(vec.astype(np.float32))

        matrix = np.vstack(vecs)
        self._index.add(matrix)

        for i, mid in enumerate(ids):
            self._id_to_pos[mid] = i
            self._pos_to_id[i] = mid
        self._next_pos = len(ids)

        logger.info("FAISS index rebuilt: %d vectors, dim=%d", len(ids), self.dimension)
