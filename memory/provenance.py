# memory/provenance.py
import json
from memory.store import MemoryStore
from memory.scoring import compute_reliability


class ProvenanceEngine:
    """
    Read-only engine that explains memory behaviour and traces failures
    using stored provenance data.
    """

    def __init__(self, store: MemoryStore):
        self.store = store

    def explain_memory(self, memory_id):
        """
        Explain a single memory: who created it, reliability, repair history.
        """
        memory = self.store.get_memory(memory_id)
        if not memory:
            print("❌ Memory not found")
            return

        print("\nMEMORY EXPLANATION")
        print("-" * 40)
        print(f"Memory ID        : {memory.id}")
        print(f"Content          : {memory.content}")
        print(f"Created By       : {memory.source_agent}")
        print(f"Lifecycle State  : {memory.lifecycle_state}")
        print(f"Reliability      : {compute_reliability(memory)}")
        print(f"Usage Count      : {memory.usage_count}")
        print(f"Failure Count    : {memory.failure_count}")

        print("\nInfluenced Decisions:")
        if memory.influenced_decisions:
            for d in memory.influenced_decisions:
                print(f"- {d}")
        else:
            print("- None")

        print("\nAssociated Tasks:")
        if memory.task_ids:
            for t in memory.task_ids:
                print(f"- {t}")
        else:
            print("- None")

        print("\nRepair History:")
        if memory.repair_history:
            for r in memory.repair_history:
                print(f"- {r}")
        else:
            print("- None")

    def trace_failure(self, task_id: str):
        """
        Trace which memories contributed to a task failure.

        Previously used `WHERE task_ids LIKE '%task_id%'` which caused
        false positives: 'task_001' would match 'task_0012' because it
        contains that substring. Now uses JSON membership check in Python
        for correctness — slower but accurate at this project's scale.
        """
        print("\nFAILURE TRACE REPORT")
        print("-" * 40)
        print(f"Task ID: {task_id}")

        # Load all rows and filter correctly in Python
        all_rows = self.store.conn.execute("SELECT * FROM memories").fetchall()
        candidates = []
        for row in all_rows:
            try:
                task_ids = json.loads(row["task_ids"] or "[]")
                if task_id in task_ids:
                    candidates.append(self.store._to_memory(row))
            except (json.JSONDecodeError, TypeError):
                pass

        if not candidates:
            print("✅ No memory linked to this task.")
            return

        for memory in candidates:
            print("\nRoot Cause Candidate:")
            print(f"- Memory ID   : {memory.id}")
            print(f"- Content     : {memory.content}")
            print(f"- Created By  : {memory.source_agent}")
            print(f"- Lifecycle   : {memory.lifecycle_state}")
            print(f"- Failures    : {memory.failure_count}")
            print(f"- Reliability : {compute_reliability(memory)}")