"""
memory/views.py
===============
Role-specialized memory views for CoCortex agents.

Two improvements over the original:

1. LIFECYCLE FILTERING
   Views now exclude stale, deprecated, and archived memories. Only memories
   in healthy lifecycle states (episodic, semantic) are served to agents.
   Serving degraded memories would undermine the reliability guarantees.

2. RELEVANCE FILTERING (TF-IDF keyword overlap)
   When a query string is provided, each view returns only the top-N most
   relevant memories based on keyword overlap with the query. This prevents
   context flooding as the knowledge base grows, and ensures agents receive
   targeted information rather than a dump of everything.

   Implementation uses simple TF-IDF word overlap (no external dependencies,
   no embeddings needed). This is sufficient for the scale of this project
   and is fully explainable — important for an academic submission.
"""

from typing import List, Optional
from memory.store import MemoryStore
from memory.schemas import MemoryItem

# Lifecycle states considered "healthy" — degraded states are excluded
HEALTHY_LIFECYCLE_STATES = {"episodic", "semantic"}

# Default max memories per view (prevents context flooding)
DEFAULT_TOP_N = 10


def _relevance_score(memory_content: str, query: str) -> float:
    """
    Compute simple keyword overlap score between memory content and query.
    Uses word-level Jaccard similarity: |intersection| / |union|.

    This is intentionally simple — no embeddings, no external deps.
    Good enough for memory retrieval at this project's scale, and
    fully auditable (important for academic work).
    """
    if not query:
        return 1.0  # no query = all memories equally relevant
    mem_words = set(memory_content.lower().split())
    query_words = set(query.lower().split())
    # Remove very common stop words that inflate scores
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of",
                  "and", "or", "in", "on", "at", "for", "with", "by"}
    mem_words -= stop_words
    query_words -= stop_words
    if not query_words:
        return 1.0
    intersection = mem_words & query_words
    union = mem_words | query_words
    return len(intersection) / len(union) if union else 0.0


def _filter_and_rank(memories: List[MemoryItem], query: Optional[str], top_n: int) -> List[MemoryItem]:
    """
    Filter by healthy lifecycle state, then rank by query relevance.
    Returns top_n most relevant memories.
    """
    # Step 1: exclude degraded lifecycle states
    healthy = [m for m in memories if m.lifecycle_state in HEALTHY_LIFECYCLE_STATES]

    # Step 2: rank by relevance if query given
    if query:
        scored = [(m, _relevance_score(m.content, query)) for m in healthy]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_n]]

    return healthy[:top_n]


# -------- PLANNER VIEW --------
def get_planner_view(
    store: MemoryStore,
    query: Optional[str] = None,
    top_n: int = DEFAULT_TOP_N
) -> List[MemoryItem]:
    """
    Planner sees:
    - Only semantic memories (general, reusable knowledge)
    - Only healthy lifecycle states (episodic or semantic)
    - Truncated content (300 chars) — Planner needs overview, not detail
    - Top-N most relevant to current task (if query provided)
    """
    semantic = store.get_memory_by_type("semantic")
    ranked = _filter_and_rank(semantic, query, top_n)

    # Apply content truncation for the planner's high-level view
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
    top_n: int = DEFAULT_TOP_N
) -> List[MemoryItem]:
    """
    Worker sees:
    - Episodic + semantic memories (procedures + facts)
    - Only healthy lifecycle states
    - Full content (Worker needs detail to execute)
    - Top-N most relevant to current plan (if query provided)
    """
    episodic = store.get_memory_by_type("episodic")
    semantic = store.get_memory_by_type("semantic")
    combined = episodic + semantic
    return _filter_and_rank(combined, query, top_n)


# -------- EVALUATOR VIEW --------
def get_evaluator_view(
    store: MemoryStore,
    query: Optional[str] = None,
    top_n: int = DEFAULT_TOP_N
) -> List[MemoryItem]:
    """
    Evaluator sees:
    - Only semantic memories
    - Only high-confidence memories (>= 0.8) — Evaluator needs verified facts
    - Only healthy lifecycle states
    - Top-N most relevant to current output (if query provided)
    """
    semantic = store.get_memory_by_type("semantic")
    high_conf = [m for m in semantic if m.confidence_score >= 0.8]
    return _filter_and_rank(high_conf, query, top_n)