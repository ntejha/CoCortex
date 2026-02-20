import json
from typing import List

from memory.store import MemoryStore
from memory.schemas import MemoryItem


class MemoryEngine:
    """
    Production-ready memory facade for CoCortex.

    Supports two usage modes:
    1. Task-scoped mode  — session_id maps to a task; memories are stored
                          as MemoryItems linked via task_ids.
    2. Conversation mode — Human/Assistant turns stored as JSON blobs
                          under a session key (LangChain-compatible).

    Both modes persist to SQLite via MemoryStore.
    """

    CONV_PREFIX = "conv::"  # marks conversation-mode records in content

    def __init__(self, db_path: str = "cocortex_memory.db"):
        self.store = MemoryStore(db_path)

    # ------------------------------------------------------------------
    # TASK-SCOPED MODE
    # ------------------------------------------------------------------

    def load(self, session_id: str) -> List[dict]:
        """
        Load all active memory items linked to session_id.
        Returns a list of dicts with 'input', 'output', 'memory_id'.
        """
        rows = self.store.conn.execute(
            "SELECT * FROM memories WHERE task_ids LIKE ? AND status = 'active'",
            (f"%{session_id}%",),
        ).fetchall()

        results = []
        for row in rows:
            mem = self.store._to_memory(row)

            if mem.content.startswith(self.CONV_PREFIX):
                try:
                    payload = json.loads(mem.content[len(self.CONV_PREFIX):])
                    payload["memory_id"] = str(mem.id)
                    results.append(payload)
                except (json.JSONDecodeError, KeyError):
                    pass
            else:
                results.append({
                    "input": mem.content,
                    "output": "",
                    "memory_id": str(mem.id),
                })

        return results

    def save(self, session_id: str, records: List[dict]):
        """
        Save a list of records to the memory store under session_id.
        Skips records that are already persisted (have a 'memory_id').
        """
        for record in records:
            if record.get("memory_id"):
                continue

            content = self.CONV_PREFIX + json.dumps({
                "input":  record.get("input", ""),
                "output": record.get("output", ""),
            })

            mem = MemoryItem(
                content=content,
                source_agent="memory_manager",
                memory_type="episodic",
                task_ids=[session_id],
                confidence_score=0.8,
            )
            self.store.add_memory(mem)

    # ------------------------------------------------------------------
    # CONVERSATION HISTORY MODE (LangChain-compatible)
    # ------------------------------------------------------------------

    def load_history(self, session_id: str) -> str:
        """
        Return conversation history as a formatted Human/Assistant string.
        Inject directly into an LLM prompt.
        """
        records = self.load(session_id)
        lines = []
        for r in records:
            user = r.get("input", "").strip()
            assistant = r.get("output", "").strip()
            if user:
                lines.append(f"Human: {user}")
            if assistant:
                lines.append(f"Assistant: {assistant}")
        return "\n".join(lines)

    def save_turn(self, session_id: str, human: str, assistant: str):
        """
        Save a single Human/Assistant turn.
        Convenience wrapper over save() for LangChain usage.
        """
        self.save(session_id, [{"input": human, "output": assistant}])

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------

    def retrieve(self, session_id: str, query: str) -> List[dict]:
        """
        Basic keyword retrieval within a session.
        (Replace with vector search when Module 2 is built.)
        """
        records = self.load(session_id)
        query_lower = query.lower()
        return [
            r for r in records
            if query_lower in r.get("input", "").lower()
            or query_lower in r.get("output", "").lower()
        ]

    # ------------------------------------------------------------------
    # REPAIR HOOK
    # ------------------------------------------------------------------

    def repair_if_needed(self, records: List[dict]) -> List[dict]:
        """
        Lightweight in-session cleanup: removes duplicates and empty entries.

        Full causal repair (traceback + LLM verification) is handled
        directly via memory.repair.repair_memories() with a
        failed_decision_id and MemoryVerifier.
        """
        seen = set()
        cleaned = []
        for r in records:
            key = (r.get("input", ""), r.get("output", ""))
            if key in seen:
                continue
            if not r.get("input") and not r.get("output"):
                continue
            seen.add(key)
            cleaned.append(r)
        return cleaned