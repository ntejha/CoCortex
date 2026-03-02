"""
memory/repair.py
================
Causal traceback, repair, and rehabilitation for CoCortex memories.
"""

import logging
from typing import List
from memory.store import MemoryStore
from memory.schemas import MemoryItem
from memory.verification import MemoryVerifier

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
    Decide repair action deterministically.

    - "incorrect"                        → quarantine immediately
    - "uncertain" + confidence < 0.6     → downrank
    - failure_count >= 2                 → downrank
    - otherwise                          → no action
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
    Find memories causally attributed to a failed decision.
    Searches both active and quarantined memories.
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
    Restore a quarantined memory to active with halved confidence.

    Called when subsequent decisions succeed without a quarantined memory —
    suggesting it may have been wrongly blamed (false-positive repair).

    Returns True if rehabilitation occurred, False otherwise.
    """
    memory = store.get_memory(memory_id)
    if not memory or memory.status == "active":
        return False

    rehabilitated_confidence = round(max(0.1, memory.confidence_score * 0.5), 3)
    store.update_status(memory_id, "active")
    store.update_confidence(memory_id, rehabilitated_confidence)
    store.log_repair_event(
        memory_id,
        f"Rehabilitated: restored to active with confidence={rehabilitated_confidence} "
        f"(was quarantined at confidence={memory.confidence_score})"
    )
    return True


def check_and_rehabilitate(
    store: MemoryStore,
    successful_decision_id: str,
) -> List[MemoryItem]:
    """
    After a SUCCESSFUL decision, check whether any quarantined memories
    were NOT involved in that success. A memory quarantined after exactly
    1 failure that didn't influence the successful decision may have been
    wrongly blamed — rehabilitate it conservatively.

    Returns list of rehabilitated memories.
    """
    quarantined = store.get_quarantined_memories()
    rehabilitated = []

    for mem in quarantined:
        if (
            mem.failure_count == 1
            and successful_decision_id not in mem.influenced_decisions
        ):
            if rehabilitate_memory(store, mem.id):
                restored = store.get_memory(mem.id)
                if restored:
                    rehabilitated.append(restored)
                    logger.info(
                        f"Rehabilitated memory {mem.id} after successful "
                        f"decision {successful_decision_id}"
                    )

    return rehabilitated


# --------------------------------
# Memory Repair Orchestrator
# --------------------------------

def repair_memories(
    store: MemoryStore,
    failed_decision_id: str,
    verifier: MemoryVerifier,
) -> List[MemoryItem]:
    """
    Full repair pipeline for a FAILED decision: trace → verify → act.

    Steps:
    1. Find memories causally attributed to the failed decision.
    2. LLM-verify each suspect for factual correctness.
    3. Apply deterministic repair policy (quarantine / downrank / none).
    4. Log all repair events for audit trail.

    Returns the list of suspect memories (regardless of action taken).
    """
    suspects = trace_suspect_memories(store, failed_decision_id)

    if not suspects:
        logger.warning(
            f"repair_memories: no memories attributed to decision "
            f"'{failed_decision_id}'. Attribution tracking likely returned [] "
            f"for this decision — self-healing cannot proceed without causal links."
        )
        return suspects

    for mem in suspects:
        verification = verifier.verify(mem.content)

        # If verifier returned 'uncertain' due to LLM outage, take no action.
        # The verification module already handles LLM_UNAVAILABLE → 'uncertain',
        # and decide_repair_action("uncertain", high_confidence) → "none".
        # This guard makes the intent explicit in the log.
        action = decide_repair_action(
            verification,
            mem.confidence_score,
            mem.failure_count,
        )

        if action == "downrank":
            old_conf = mem.confidence_score
            new_conf = round(max(0.1, old_conf - 0.2), 3)
            store.update_confidence(mem.id, new_conf)
            store.mark_memory_failed(mem.id)
            # Re-fetch after writes so the log reflects the actual stored value
            fresh = store.get_memory(mem.id)
            actual_conf = fresh.confidence_score if fresh else new_conf
            store.log_repair_event(
                mem.id,
                f"Downranked via causal traceback: confidence "
                f"{old_conf:.3f} → {actual_conf:.3f} "
                f"(decision={failed_decision_id}, verification={verification})"
            )

        elif action == "quarantine":
            store.update_status(mem.id, "quarantined")
            store.mark_memory_failed(mem.id)
            store.log_repair_event(
                mem.id,
                f"Quarantined via causal traceback: verification='{verification}' "
                f"(decision={failed_decision_id})"
            )

        # action == "none" → memory survives; no log needed

    return suspects


def repair_on_success(
    store: MemoryStore,
    successful_decision_id: str,
) -> List[MemoryItem]:
    """
    Call after a SUCCESSFUL evaluator decision to attempt rehabilitation
    of borderline-quarantined memories.

    This is the counterpart to repair_memories() — together they make the
    repair state machine bidirectional:
      failure  → repair_memories()     → quarantine / downrank
      success  → repair_on_success()   → rehabilitate (if borderline)

    Usage in agent pipeline:
        result, decision_id = evaluator.evaluate(output)
        if "PASS" in result:
            repair_on_success(store, decision_id)
        else:
            repair_memories(store, decision_id, verifier)
    """
    return check_and_rehabilitate(store, successful_decision_id)