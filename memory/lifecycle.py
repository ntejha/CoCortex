# memory/lifecycle.py
from memory.schemas import MemoryItem
from memory.scoring import compute_reliability


def update_lifecycle(memory: MemoryItem) -> str:
    """
    Compute the new lifecycle state for a memory.

    Rules (applied in priority order):
    1. 3+ failures  → deprecated (hard rule, cannot be overridden)
    2. quarantined  → retain current lifecycle_state (quarantined memories
                      must NOT be promoted, even if their reliability is high)
    3. reliability ≥ 0.8              → semantic
    4. 0.5 ≤ reliability < 0.8       → unchanged (keep current state)
    5. 0.3 ≤ reliability < 0.5       → stale
    6. reliability < 0.3             → archived
    """
    reliability = compute_reliability(memory)

    if memory.failure_count >= 3:
        return "deprecated"

    # Quarantine guard — do not promote quarantined memories
    if memory.status == "quarantined":
        return memory.lifecycle_state

    if reliability >= 0.8:
        return "semantic"

    if 0.5 <= reliability < 0.8:
        return memory.lifecycle_state

    if 0.3 <= reliability < 0.5:
        return "stale"

    return "archived"