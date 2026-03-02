"""
memory/repair.py
================
Causal traceback, repair, and rehabilitation for CoCortex memories.

Key design decisions:

1. ATTRIBUTION-GUIDED TRACEBACK
   trace_suspect_memories() only returns memories that were explicitly linked
   to the failed decision via attribution (not all memories in the context).
   This avoids false accusations of innocent memories.

2. REPAIR IS NOT PERMANENT CONDEMNATION
   Quarantined memories can be rehabilitated if subsequent decisions succeed
   WITHOUT them. This prevents the system from permanently degrading due to
   false positives during repair.

3. DETERMINISTIC REPAIR POLICY
   The repair action (quarantine / downrank / none) is deterministic given
   the LLM's verification verdict, confidence, and failure count. This makes
   repair auditable and predictable.
"""

from typing import List
from memory.store import MemoryStore
from memory.schemas import MemoryItem
from memory.verification import MemoryVerifier


# -------------------------------
# Deterministic Repair Policy
# -------------------------------

def decide_repair_action(
    verification: str,
    confidence: float,
    failure_count: int
) -> str:
    """
    Decide repair action deterministically given verification outcome.

    Policy rationale:
    - "incorrect" from LLM verifier → quarantine immediately (strong signal)
    - "uncertain" with low confidence → downrank (weak signal, cautious action)
    - high failure count alone → downrank (usage pattern suggests problem)
    - otherwise → no action (innocent until proven guilty)
    """
    if verification == "incorrect":
        return "quarantine"

    if verification == "uncertain" and confidence < 0.6:
        return "downrank"

    if failure_count >= 2:
        return "downrank"

    return "none"


# --------------------------------
# Causal Traceback
# --------------------------------

def trace_suspect_memories(
    store: MemoryStore,
    failed_decision_id: str
) -> List[MemoryItem]:
    """
    Find memories that were causally attributed to a failed decision.

    Only returns memories that were explicitly linked via attribution-guided
    tracking (i.e. the LLM reported using them). This avoids blaming innocent
    memories that were merely present in the context window.
    """
    suspects = []
    for mem in store.get_all_active_memories() + store.get_quarantined_memories():
        if failed_decision_id in mem.influenced_decisions:
            suspects.append(mem)
    return suspects


# --------------------------------
# Memory Rehabilitation
# --------------------------------

def rehabilitate_memory(store: MemoryStore, memory_id) -> bool:
    """
    Restore a quarantined memory to active status with reduced confidence.

    Called when subsequent decisions SUCCEED without a memory that was
    previously quarantined. This prevents the system from permanently
    losing potentially correct memories due to false-positive repairs.

    Returns True if rehabilitation occurred, False if memory not found
    or already active.
    """
    memory = store.get_memory(memory_id)
    if not memory or memory.status == "active":
        return False

    # Restore to active with halved confidence — it's back but flagged
    rehabilitated_confidence = round(max(0.1, memory.confidence_score * 0.5), 3)
    store.update_status(memory_id, "active")
    store.update_confidence(memory_id, rehabilitated_confidence)
    store.log_repair_event(
        memory_id,
        f"Rehabilitated: restored to active with confidence={rehabilitated_confidence} "
        f"after subsequent task successes (was quarantined with confidence={memory.confidence_score})"
    )
    return True


def check_and_rehabilitate(
    store: MemoryStore,
    successful_decision_id: str
) -> List[MemoryItem]:
    """
    After a successful decision, check if any quarantined memories were
    NOT involved in this decision. If a memory has been quarantined but
    recent decisions succeeded without it, it may have been wrongly blamed.

    Heuristic: if a quarantined memory has failure_count == 1 and the system
    is now succeeding, we rehabilitate it as a potential false positive.
    This is a conservative approach — only clearly borderline cases recover.

    Returns list of rehabilitated memories.
    """
    quarantined = store.get_quarantined_memories()
    rehabilitated = []

    for mem in quarantined:
        # Only rehabilitate memories with exactly 1 failure (borderline cases)
        # and that were NOT involved in the successful decision
        if (mem.failure_count == 1
                and successful_decision_id not in mem.influenced_decisions):
            if rehabilitate_memory(store, mem.id):
                rehabilitated.append(store.get_memory(mem.id))

    return rehabilitated


# --------------------------------
# Memory Repair Orchestrator
# --------------------------------

def repair_memories(
    store: MemoryStore,
    failed_decision_id: str,
    verifier: MemoryVerifier
) -> List[MemoryItem]:
    """
    Full repair pipeline: trace → verify → act.

    Steps:
    1. Find memories causally attributed to the failed decision
    2. LLM-verify each suspect memory for factual correctness
    3. Apply deterministic repair policy
    4. Log all repair events for audit trail
    """
    suspects = trace_suspect_memories(store, failed_decision_id)

    for mem in suspects:
        verification = verifier.verify(mem.content)
        failure_count = mem.failure_count

        action = decide_repair_action(
            verification,
            mem.confidence_score,
            failure_count
        )

        if action == "downrank":
            new_conf = max(0.1, mem.confidence_score - 0.2)
            store.update_confidence(mem.id, new_conf)
            store.mark_memory_failed(mem.id)
            store.log_repair_event(
                mem.id,
                f"Downranked via causal traceback: confidence {mem.confidence_score:.2f} → {new_conf:.2f} "
                f"(decision={failed_decision_id}, verification={verification})"
            )

        elif action == "quarantine":
            store.update_status(mem.id, "quarantined")
            store.mark_memory_failed(mem.id)
            store.log_repair_event(
                mem.id,
                f"Quarantined via causal traceback: LLM verification='{verification}' "
                f"(decision={failed_decision_id})"
            )

        # action == "none" → memory survives, no log needed

    return suspects