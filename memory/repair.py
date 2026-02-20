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
    Decide repair action deterministically.
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
    suspects = []

    for mem in (
        store.get_memory_by_type("semantic") +
        store.get_memory_by_type("episodic")
    ):
        if failed_decision_id in mem.influenced_decisions:
            suspects.append(mem)

    return suspects


# --------------------------------
# Memory Repair Orchestrator
# --------------------------------

def repair_memories(
    store: MemoryStore,
    failed_decision_id: str,
    verifier: MemoryVerifier
):
    suspects = trace_suspect_memories(store, failed_decision_id)

    for mem in suspects:
        verification = verifier.verify(mem.content)
        failure_count = len(mem.influenced_decisions)

        action = decide_repair_action(
            verification,
            mem.confidence_score,
            failure_count
        )

        if action == "downrank":
            new_conf = max(0.1, mem.confidence_score - 0.2)
            store.update_confidence(mem.id, new_conf)
            store.log_repair_event(
                mem.id,
                f"Downranked via causal traceback: confidence reduced "
                f"from {mem.confidence_score} to {new_conf} "
                f"(decision={failed_decision_id}, verification={verification})"
            )

        elif action == "quarantine":
            store.update_status(mem.id, "quarantined")
            store.log_repair_event(
                mem.id,
                f"Quarantined via causal traceback: LLM verification returned "
                f"'{verification}' (decision={failed_decision_id})"
            )

    return suspects