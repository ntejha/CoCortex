"""
memory/views.py
===============
Role-specialized memory views for CoCortex agents.

Two improvements over the original:

1. LIFECYCLE FILTERING
   Views exclude stale, deprecated, and archived memories. Only memories
   in healthy lifecycle states (episodic, semantic) are served to agents.

2. RELEVANCE FILTERING (word-level Jaccard similarity)
   When a query string is provided, memories are ranked by keyword overlap
   using Jaccard similarity: |intersection| / |union| of word sets.
   This is intentionally simple — no embeddings, no external dependencies,
   fully auditable and explainable. Sufficient for this project's scale.

   Note: this is Jaccard similarity, NOT TF-IDF. TF-IDF would additionally
   weight terms by inverse document frequency across the corpus. Jaccard is
   chosen here for simplicity and interpretability.
"""

from typing import List, Optional
from memory.store import MemoryStore
from memory.schemas import MemoryItem

HEALTHY_LIFECYCLE_STATES = {"episodic", "semantic"}
DEFAULT_TOP_N = 10


def _relevance_score(memory_content: str, query: str) -> float:
    """Word-level Jaccard similarity: |intersection| / |union|."""
    if not query:
        return 1.0
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of",
                  "and", "or", "in", "on", "at", "for", "with", "by"}
    mem_words = set(memory_content.lower().split()) - stop_words
    query_words = set(query.lower().split()) - stop_words
    if not query_words:
        return 1.0
    intersection = mem_words & query_words
    union = mem_words | query_words
    return len(intersection) / len(union) if union else 0.0


def _filter_and_rank(
    memories: List[MemoryItem],
    query: Optional[str],
    top_n: int,
) -> List[MemoryItem]:
    """Filter by healthy lifecycle, rank by relevance, return top_n."""
    healthy = [m for m in memories if m.lifecycle_state in HEALTHY_LIFECYCLE_STATES]
    if query:
        scored = [(m, _relevance_score(m.content, query)) for m in healthy]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_n]]
    return healthy[:top_n]


# -------- PLANNER VIEW --------
def get_planner_view(
    store: MemoryStore,
    query: Optional[str] = None,
    top_n: int = DEFAULT_TOP_N,
) -> List[MemoryItem]:
    """
    Planner sees semantic memories only, truncated to 300 chars.
    """
    semantic = store.get_memory_by_type("semantic")
    ranked = _filter_and_rank(semantic, query, top_n)
    return [
        MemoryItem(
            id=mem.id,
            content=mem.content[:300],
            memory_type=mem.memory_type,
            source_agent=mem.source_agent,
            confidence_score=mem.confidence_score,
            status=mem.status,
            influenced_decisions=mem.influenced_decisions,
            lifecycle_state=mem.lifecycle_state,
        )
        for mem in ranked
    ]


# -------- WORKER VIEW --------
def get_worker_view(
    store: MemoryStore,
    query: Optional[str] = None,
    top_n: int = DEFAULT_TOP_N,
) -> List[MemoryItem]:
    """
    Worker sees episodic + semantic memories, full content.
    """
    episodic = store.get_memory_by_type("episodic")
    semantic = store.get_memory_by_type("semantic")
    return _filter_and_rank(episodic + semantic, query, top_n)


# -------- EVALUATOR VIEW --------
def get_evaluator_view(
    store: MemoryStore,
    query: Optional[str] = None,
    top_n: int = DEFAULT_TOP_N,
) -> List[MemoryItem]:
    """
    Evaluator sees high-confidence (>= 0.8) semantic memories only.
    """
    semantic = store.get_memory_by_type("semantic")
    high_conf = [m for m in semantic if m.confidence_score >= 0.8]
    return _filter_and_rank(high_conf, query, top_n)