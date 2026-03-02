# memory/lifecycle.py
from memory.schemas import MemoryItem
from memory.scoring import compute_reliability


def update_lifecycle(memory: MemoryItem) -> str:
    """
    Compute the new lifecycle state for a memory based on reliability.

    Priority order:
      1. deprecated  — failure_count >= 3 (overrides everything)
      2. (unchanged) — memory is quarantined (don't promote unhealthy memories)
      3. semantic    — reliability >= 0.8
      4. (unchanged) — 0.5 <= reliability < 0.8
      5. stale       — 0.3 <= reliability < 0.5
      6. archived    — reliability < 0.3

    Quarantine guard: a quarantined memory must not have its lifecycle_state
    promoted to "semantic". That would create a misleading state where
    lifecycle_state reports "healthy" while status reports "quarantined".
    Quarantined memories stay at their current lifecycle_state until
    rehabilitated back to active.
    """
    reliability = compute_reliability(memory)

    if memory.failure_count >= 3:
        return "deprecated"

    # Do not promote quarantined memories — their status already flags them
    # as unsafe. Promoting lifecycle would mislead provenance reports.
    if memory.status == "quarantined":
        return memory.lifecycle_state

    if reliability >= 0.8:
        return "semantic"

    if 0.5 <= reliability < 0.8:
        return memory.lifecycle_state

    if 0.3 <= reliability < 0.5:
        return "stale"

    return "archived"