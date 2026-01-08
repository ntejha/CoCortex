# memory/lifecycle.py
from memory.schemas import MemoryItem
from memory.scoring import compute_reliability

def update_lifecycle(memory: MemoryItem) -> str:
    reliability = compute_reliability(memory)

    if memory.failure_count >= 3:
        return "deprecated"

    if reliability >= 0.8:
        return "semantic"

    if 0.5 <= reliability < 0.8:
        return memory.lifecycle_state

    if 0.3 <= reliability < 0.5:
        return "stale"

    return "archived"
