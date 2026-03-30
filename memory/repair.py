"""
memory/repair.py

Causal repair with:
- Rehabilitation path (restore quarantined memories that stabilise)
- repair_on_success() for post-success recovery
"""
import logging
from typing import List
from memory.store import MemoryStore
from memory.schemas import MemoryItem
from memory.verification import MemoryVerifier
from memory.scoring import compute_reliability

logger = logging.getLogger(__name__)

# --------------------------------
# Deterministic Repair Policy
# --------------------------------

def decide_repair_action(
    verification: str,
    confidence: float,
    failure_count: int,
) -> str:
    """
    Rules (priority order):
    1. LLM says incorrect → quarantine
    2. uncertain + low confidence → downrank
    3. too many failures (≥2) → downrank
    4. otherwise → none
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
    failed_decision_id: str,
) -> List[MemoryItem]:
    """
    Find all memories (active OR quarantined) whose influenced_decisions
    list includes failed_decision_id.
    """
    suspects = []
    all_memories = store.get_all_active_memories() + store.get_quarantined_memories()
    for mem in all_memories:
        if failed_decision_id in mem.influenced_decisions:
            suspects.append(mem)
            logger.debug("Suspect: memory_id=%s decision=%s", mem.id, failed_decision_id)

    logger.info(
        "Traceback for decision=%s → %d suspect(s)", failed_decision_id, len(suspects)
    )
    return suspects


# --------------------------------
# Memory Repair Orchestrator
# --------------------------------

def repair_memories(
    store: MemoryStore,
    failed_decision_id: str,
    verifier: MemoryVerifier,
) -> List[MemoryItem]:
    """
    Full causal repair:
    1. Trace suspects
    2. Verify with LLM
    3. Apply deterministic action (quarantine / downrank / none)
    4. Log to repair_history
    """
    suspects = trace_suspect_memories(store, failed_decision_id)

    for mem in suspects:
        verification = verifier.verify(mem.content)
        failure_count = mem.failure_count

        action = decide_repair_action(verification, mem.confidence_score, failure_count)
        logger.info(
            "Repair: memory_id=%s verification=%s action=%s", mem.id, verification, action
        )

        if action == "downrank":
            new_conf = max(0.1, mem.confidence_score - 0.2)
            store.update_confidence(mem.id, new_conf)
            fresh = store.get_memory(mem.id)
            actual_conf = fresh.confidence_score if fresh else new_conf
            store.log_repair_event(
                mem.id,
                f"Downranked via causal traceback: confidence "
                f"from {mem.confidence_score:.3f} to {actual_conf:.3f} "
                f"(decision={failed_decision_id}, verification={verification})",
            )

        elif action == "quarantine":
            store.update_status(mem.id, "quarantined")
            store.log_repair_event(
                mem.id,
                f"Quarantined via causal traceback: LLM verification returned "
                f"'{verification}' (decision={failed_decision_id})",
            )

    return suspects


# --------------------------------
# Rehabilitation
# --------------------------------

# A quarantined memory is eligible for rehabilitation if:
# - failure_count < this threshold (it hasn't failed catastrophically)
_REHAB_MAX_FAILURES = 3


def rehabilitate_memory(store: MemoryStore, memory_id) -> bool:
    """
    Restore a single quarantined memory to 'active' with a reduced confidence.

    Returns True if rehabilitation happened, False if the memory was already
    active or could not be found.
    """
    memory = store.get_memory(memory_id)
    if not memory:
        logger.warning("rehabilitate_memory: memory %s not found", memory_id)
        return False

    if memory.status != "quarantined":
        return False

    # Apply a confidence penalty to reflect that it was quarantined
    new_conf = max(0.1, round(memory.confidence_score * 0.8, 3))
    store.update_confidence(memory_id, new_conf)
    store.update_status(memory_id, "active")
    store.log_repair_event(
        memory_id,
        f"Rehabilitated: confidence adjusted from {memory.confidence_score} "
        f"to {new_conf} (failure_count={memory.failure_count})",
    )
    logger.info(
        "Rehabilitated memory_id=%s confidence %.3f → %.3f",
        memory_id, memory.confidence_score, new_conf,
    )
    return True


def check_and_rehabilitate(
    store: MemoryStore,
    success_decision_id: str,
) -> List[MemoryItem]:
    """
    Review all quarantined memories. Rehabilitate those that:
    - Have fewer than _REHAB_MAX_FAILURES failures
    - Are still reasonably reliable (compute_reliability > 0.3)

    Called after a successful evaluator decision to recover borderline memories.
    """
    candidates = store.get_quarantined_memories()
    rehabilitated = []

    for mem in candidates:
        reliability = compute_reliability(mem)
        if mem.failure_count < _REHAB_MAX_FAILURES and reliability > 0.3:
            rehabilitate_memory(store, mem.id)
            rehabilitated.append(mem)
            logger.info(
                "check_and_rehabilitate: restored memory_id=%s reliability=%.3f",
                mem.id, reliability,
            )

    return rehabilitated


def repair_on_success(
    store: MemoryStore,
    success_decision_id: str,
) -> List[MemoryItem]:
    """
    Convenience wrapper called by the evaluator when output is correct.
    Triggers rehabilitation of borderline quarantined memories.
    """
    logger.info("repair_on_success triggered by decision=%s", success_decision_id)
    return check_and_rehabilitate(store, success_decision_id)