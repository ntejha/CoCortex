from typing import List
from memory.store import MemoryStore
from memory.schemas import MemoryItem


# -------- PLANNER VIEW --------
def get_planner_view(store: MemoryStore) -> List[MemoryItem]:
    """
    Planner sees:
    - Only semantic memory
    - Only active memory
    - High-level (truncated) content
    """
    semantic = store.get_memory_by_type("semantic")
    view = []

    for mem in semantic:
        view.append(
            MemoryItem(
                id=mem.id,
                content=mem.content[:300],  # simple truncation
                memory_type=mem.memory_type,
                source_agent=mem.source_agent,
                confidence_score=mem.confidence_score,
                status=mem.status,
                influenced_decisions=mem.influenced_decisions,
            )
        )

    return view


# -------- WORKER VIEW --------
def get_worker_view(store: MemoryStore) -> List[MemoryItem]:
    """
    Worker sees:
    - Episodic + semantic memory
    - Full content
    - Only active memory
    """
    episodic = store.get_memory_by_type("episodic")
    semantic = store.get_memory_by_type("semantic")

    return episodic + semantic


# -------- EVALUATOR VIEW --------
def get_evaluator_view(store: MemoryStore) -> List[MemoryItem]:
    """
    Evaluator sees:
    - Only semantic memory
    - High-confidence memory only
    - No execution traces
    """
    semantic = store.get_memory_by_type("semantic")
    return [
        mem for mem in semantic
        if mem.confidence_score >= 0.8
    ]
