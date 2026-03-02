# memory/provenance.py

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
        """
        rows = self.store.conn.execute(
            """
            SELECT * FROM memories
            WHERE task_ids LIKE ?
            """,
            (f"%{task_id}%",),
        ).fetchall()

        if not rows:
            logger.info("trace_failure: no memory linked to task_id=%s", task_id)
            return []

        suspects = []
        for row in rows:
            memory = self.store._to_memory(row)
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