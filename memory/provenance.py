# memory/provenance.py

import json
import logging
from memory.store import MemoryStore
from memory.scoring import compute_reliability

logger = logging.getLogger(__name__)


class ProvenanceEngine:
    """
    Read-only engine that explains memory behaviour
    and traces failures using stored provenance data.
    """

    def __init__(self, store: MemoryStore):
        self.store = store

    def explain_memory(self, memory_id) -> dict:
        """
        Explains a single memory:
        - who created it
        - how reliable it is
        - how often it failed
        - what repairs happened

        Returns a dict so callers can format output as they like.
        """
        memory = self.store.get_memory(memory_id)
        if not memory:
            logger.warning("explain_memory: memory %s not found", memory_id)
            return {}

        report = {
            "memory_id": str(memory.id),
            "content": memory.content,
            "created_by": memory.source_agent,
            "lifecycle_state": memory.lifecycle_state,
            "reliability": compute_reliability(memory),
            "usage_count": memory.usage_count,
            "failure_count": memory.failure_count,
            "influenced_decisions": memory.influenced_decisions,
            "associated_tasks": memory.task_ids,
            "repair_history": memory.repair_history,
        }

        logger.info(
            "Memory explanation: id=%s source=%s lifecycle=%s reliability=%.3f",
            memory.id,
            memory.source_agent,
            memory.lifecycle_state,
            report["reliability"],
        )

        return report

    def trace_failure(self, task_id: str) -> list:
        """
        Traces which memories contributed to a task failure.
        Returns a list of dicts, one per suspect memory.

        Previously used WHERE task_ids LIKE which caused false positives:
        task-1 would match rows belonging to task-10, task-100, etc.,
        because LIKE does substring matching on the raw JSON string.

        Now loads all rows and filters with an exact Python in check on the
        parsed JSON array — correct at this project scale and consistent with
        how store.get_memories_by_session and store.delete_by_session work.
        """
        all_rows = self.store.conn.execute(
            "SELECT * FROM memories"
        ).fetchall()

        candidates = []
        for row in all_rows:
            try:
                task_ids = json.loads(row["task_ids"] or "[]")
                if task_id in task_ids:
                    candidates.append(self.store._to_memory(row))
            except (json.JSONDecodeError, TypeError):
                pass

        if not candidates:
            logger.info("trace_failure: no memory linked to task_id=%s", task_id)
            return []

        suspects = []
        for memory in candidates:
            rel = compute_reliability(memory)
            suspects.append(
                {
                    "memory_id": str(memory.id),
                    "content": memory.content,
                    "created_by": memory.source_agent,
                    "lifecycle": memory.lifecycle_state,
                    "failure_count": memory.failure_count,
                    "reliability": rel,
                }
            )
            logger.warning(
                "Suspect memory for task=%s: id=%s lifecycle=%s failures=%d reliability=%.3f",
                task_id,
                memory.id,
                memory.lifecycle_state,
                memory.failure_count,
                rel,
            )

        return suspects