"""
memory/views.py

Role-specialized, lifecycle-filtered memory views with optional
query-based relevance ranking and top-N limiting.
"""
from typing import List, Optional
from memory.store import MemoryStore
from memory.schemas import MemoryItem

# Lifecycle states that are too degraded to be useful in any view
_EXCLUDED_LIFECYCLE = {"stale", "deprecated", "archived"}


def _is_usable(mem: MemoryItem) -> bool:
    """Return True if the memory's lifecycle state is not degraded."""
    return mem.lifecycle_state not in _EXCLUDED_LIFECYCLE


def _rank_by_query(memories: List[MemoryItem], query: str) -> List[MemoryItem]:
    """
    Simple keyword-overlap ranking.
    Memories whose content contains more query words rank higher.
    """
    if not query:
        return memories
    tokens = set(query.lower().split())
    def _score(m: MemoryItem) -> int:
        content_lower = m.content.lower()
        return sum(1 for t in tokens if t in content_lower)
    return sorted(memories, key=_score, reverse=True)


# -------- PLANNER VIEW --------

def get_planner_view(
    store: MemoryStore,
    query: Optional[str] = None,
    top_n: Optional[int] = None,
) -> List[MemoryItem]:
    """
    Planner sees:
    - Only semantic memory
    - Only active memories
    - Only non-degraded lifecycle states (excludes stale/deprecated/archived)
    - Content truncated to 300 chars
    - Optionally ranked by query relevance and/or limited to top_n
    """
    semantic = store.get_memory_by_type("semantic")
    view = []
    for mem in semantic:
        if not _is_usable(mem):
            continue
        view.append(
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
        )

    if query:
        view = _rank_by_query(view, query)
    if top_n is not None:
        view = view[:top_n]
    return view


# -------- WORKER VIEW --------

def get_worker_view(
    store: MemoryStore,
    query: Optional[str] = None,
    top_n: Optional[int] = None,
) -> List[MemoryItem]:
    """
    Worker sees:
    - Episodic + semantic memory
    - Only active memories
    - Excludes archived lifecycle states
    """
    episodic = store.get_memory_by_type("episodic")
    semantic = store.get_memory_by_type("semantic")
    combined = [m for m in (episodic + semantic) if m.lifecycle_state != "archived"]

    if query:
        combined = _rank_by_query(combined, query)
    if top_n is not None:
        combined = combined[:top_n]
    return combined


# -------- EVALUATOR VIEW --------

def get_evaluator_view(
    store: MemoryStore,
    query: Optional[str] = None,
    top_n: Optional[int] = None,
) -> List[MemoryItem]:
    """
    Evaluator sees:
    - Only semantic memory
    - Only high-confidence memories (≥ 0.8)
    - Excludes degraded lifecycle states
    """
    semantic = store.get_memory_by_type("semantic")
    view = [
        mem for mem in semantic
        if mem.confidence_score >= 0.8 and _is_usable(mem)
    ]

    if query:
        view = _rank_by_query(view, query)
    if top_n is not None:
        view = view[:top_n]
    return view