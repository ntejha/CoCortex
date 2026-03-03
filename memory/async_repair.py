"""
memory/async_repair.py

Asynchronous memory repair pipeline.
Demonstrates the key benefit of the async conversion:
Verifying multiple suspect memories concurrently via asyncio.gather instead of sequentially.
"""
import asyncio
import logging
from typing import List

from memory.async_store import AsyncMemoryStore
from memory.schemas import MemoryItem
from memory.repair import decide_repair_action

logger = logging.getLogger(__name__)


async def trace_suspect_memories_async(store: AsyncMemoryStore, failed_decision_id: str) -> List[MemoryItem]:
    """Finds all memories whose influenced_decisions contains the failed_decision_id."""
    active = await store.get_all_active_memories()
    active_mems = [m for m in active if m.status == "active"]

    suspects = []
    for mem in active_mems:
        # Check causal list
        if failed_decision_id in mem.influenced_decisions:
            suspects.append(mem)

    return suspects


async def check_and_repair_async(store: AsyncMemoryStore, failed_decision_id: str, verifier_agent) -> None:
    """
    Concurrent repair pipeline.
    Uses asyncio.gather to run verification LLM calls in parallel.
    Writes to SQLite sequentially to prevent lock contention.
    """
    suspects = await trace_suspect_memories_async(store, failed_decision_id)
    if not suspects:
        logger.info(f"No suspect memories traced for failure {failed_decision_id}")
        return

    logger.info(f"Tracing found {len(suspects)} suspect memories for {failed_decision_id}. Verifying concurrently...")

    # Define a helper coroutine to verify one memory using the async agent
    async def verify_one(mem: MemoryItem) -> str:
        # Note: verifier_agent is an EvaluatorAgent (now an AsyncAgent)
        prompt = f"Verify this memory: '{mem.content}'. Respond only with CORRECT, INCORRECT, or UNCERTAIN."
        # Using the rate-limited async _allm method to prevent blowing out API limits
        return await verifier_agent._allm(prompt)

    # 1. RUN VERIFICATIONS CONCURRENTLY
    # This is where the real speedup happens!
    tasks = [verify_one(m) for m in suspects]
    verdicts = await asyncio.gather(*tasks, return_exceptions=True)

    # 2. APPLY ACTIONS FOR EACH
    # Apply one by one using the async lock-protected store methods
    for i, mem in enumerate(suspects):
        verdict = verdicts[i]
        
        # Handle API failure during concurrent run
        if isinstance(verdict, Exception):
            logger.error(f"Concurrent verification failed for memory {mem.id}: {verdict}")
            continue

        verdict_upper = verdict.strip().upper()
        # Parse pass/fail the same way sync repair does
        if "CORRECT" in verdict_upper and "INCORRECT" not in verdict_upper:
            verification_status = "correct"
        elif "INCORRECT" in verdict_upper:
            verification_status = "incorrect"
        else:
            verification_status = "uncertain"

        # Safe sequential state update
        await store.mark_memory_failed(mem.id)
        # Get latest state after increment
        mem_latest = await store.get_memory(mem.id)
        if not mem_latest:
            continue
            
        action = decide_repair_action(
            verification_status, mem_latest.confidence_score, mem_latest.failure_count
        )

        logger.info(f"Async Repair decision for {mem.id}: action={action}")

        new_conf = mem_latest.confidence_score
        new_status = mem_latest.status

        if action == "downrank":
            new_conf = max(0.1, round(mem_latest.confidence_score - 0.2, 3))
        elif action == "quarantine":
            new_status = "quarantined"

        if new_conf != mem_latest.confidence_score:
            await store.update_memory(mem.id, "confidence_score", new_conf)
        if new_status != mem_latest.status:
            await store.update_memory(mem.id, "status", new_status)

        # Log the event
        message = (f"Async repair: decision_id={failed_decision_id}, "
                   f"status={verification_status}, action={action}, "
                   f"conf_change={mem_latest.confidence_score}->{new_conf}, "
                   f"status_change={mem_latest.status}->{new_status}")
        await store.log_repair_event(mem.id, message)
