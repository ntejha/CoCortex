"""
memory/async_store.py

Asynchronous MemoryStore implementation using asyncio.to_thread + sqlite3.
Provides the same async API as if using aiosqlite, but avoids the broken
aiosqlite library by offloading blocking sqlite3 calls to threads.
"""
import json
import logging
import asyncio
import sqlite3
from typing import List, Optional

from core.config import DB_PATH
from memory.schemas import MemoryItem

logger = logging.getLogger(__name__)

_ALLOWED_UPDATE_FIELDS = {
    "content",
    "memory_type",
    "source_agent",
    "timestamp",
    "confidence_score",
    "status",
    "influenced_decisions",
    "usage_count",
    "failure_count",
    "last_validated_at",
    "lifecycle_state",
    "repair_history",
    "task_ids",
    "embedding",
}


class AsyncMemoryStore:
    """
    Async-compatible memory store backed by sqlite3.

    All blocking sqlite3 operations are offloaded to a thread via
    asyncio.to_thread, keeping the event loop responsive.
    A single persistent connection is used (with check_same_thread=False)
    and an asyncio.Lock serialises writes.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._write_lock = asyncio.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn_sync(self) -> sqlite3.Connection:
        """Get or create the persistent sqlite3 connection (called inside threads)."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    async def _get_conn(self) -> sqlite3.Connection:
        """Async wrapper — ensures the connection is initialized."""
        if self._conn is None:
            await asyncio.to_thread(self._get_conn_sync)
        return self._conn

    async def close(self) -> None:
        """Explicitly close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def init_schema(self) -> None:
        """Create the memories table if it doesn't exist (used in tests)."""
        def _init():
            conn = self._get_conn_sync()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY, content TEXT, memory_type TEXT, source_agent TEXT,
                    timestamp TEXT, confidence_score REAL, status TEXT, influenced_decisions TEXT,
                    usage_count INTEGER DEFAULT 0, failure_count INTEGER DEFAULT 0,
                    last_validated_at TEXT, lifecycle_state TEXT DEFAULT 'episodic',
                    repair_history TEXT DEFAULT '[]', task_ids TEXT DEFAULT '[]',
                    embedding BLOB
                )
            """)
            conn.commit()
        await asyncio.to_thread(_init)

    # ---- CRUD ----

    async def add_memory(self, item: MemoryItem) -> None:
        def _add():
            conn = self._get_conn_sync()
            conn.execute(
                "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(item.id),
                    item.content,
                    item.memory_type,
                    item.source_agent,
                    item.timestamp.isoformat(),
                    item.confidence_score,
                    item.status,
                    json.dumps([str(d) for d in item.influenced_decisions]),
                    item.usage_count,
                    item.failure_count,
                    item.last_validated_at.isoformat() if item.last_validated_at else None,
                    item.lifecycle_state,
                    json.dumps(item.repair_history),
                    json.dumps(item.task_ids),
                    None,
                ),
            )
            conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_add)

    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        def _get():
            conn = self._get_conn_sync()
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (str(memory_id),)
            ).fetchone()
            return row

        row = await asyncio.to_thread(_get)
        if row:
            return self._to_memory(row)
        return None

    async def get_memory_by_type(self, m_type: str) -> List[MemoryItem]:
        def _get():
            conn = self._get_conn_sync()
            return conn.execute(
                "SELECT * FROM memories WHERE memory_type = ? AND status != 'quarantined'",
                (m_type,),
            ).fetchall()

        rows = await asyncio.to_thread(_get)
        return [self._to_memory(row) for row in rows]

    async def get_all_active_memories(self) -> List[MemoryItem]:
        def _get():
            conn = self._get_conn_sync()
            return conn.execute(
                "SELECT * FROM memories WHERE status = 'active'"
            ).fetchall()

        rows = await asyncio.to_thread(_get)
        return [self._to_memory(row) for row in rows]

    async def update_memory(self, memory_id: str, field: str, new_value) -> None:
        if field not in _ALLOWED_UPDATE_FIELDS:
            logger.warning("Attempted to update restricted or unknown field: %s", field)
            return

        def _update():
            conn = self._get_conn_sync()
            conn.execute(
                f"UPDATE memories SET {field} = ? WHERE id = ?",
                (new_value, str(memory_id)),
            )
            conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_update)

    async def link_memory_to_decision(self, memory_id: str, decision_id: str) -> None:
        def _link():
            conn = self._get_conn_sync()
            row = conn.execute(
                "SELECT influenced_decisions FROM memories WHERE id = ?",
                (str(memory_id),)
            ).fetchone()
            if row:
                current = json.loads(row["influenced_decisions"])
                if decision_id not in current:
                    current.append(decision_id)
                    conn.execute(
                        "UPDATE memories SET influenced_decisions = ? WHERE id = ?",
                        (json.dumps(current), str(memory_id)),
                    )
                    conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_link)

    async def link_memory_to_task(self, memory_id: str, task_id: str) -> None:
        def _link():
            conn = self._get_conn_sync()
            row = conn.execute(
                "SELECT task_ids FROM memories WHERE id = ?", (str(memory_id),)
            ).fetchone()
            if row:
                current = json.loads(row["task_ids"])
                if task_id not in current:
                    current.append(task_id)
                    conn.execute(
                        "UPDATE memories SET task_ids = ? WHERE id = ?",
                        (json.dumps(current), str(memory_id)),
                    )
                    conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_link)

    async def promote_memory(self, memory_id: str, new_type: str) -> None:
        await self.update_memory(memory_id, "memory_type", new_type)

    async def mark_memory_used(self, memory_id: str) -> None:
        def _mark():
            conn = self._get_conn_sync()
            conn.execute(
                "UPDATE memories SET usage_count = usage_count + 1 WHERE id = ?",
                (str(memory_id),)
            )
            conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_mark)

    async def mark_memory_failed(self, memory_id: str) -> None:
        def _mark():
            conn = self._get_conn_sync()
            conn.execute(
                "UPDATE memories SET failure_count = failure_count + 1 WHERE id = ?",
                (str(memory_id),)
            )
            conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_mark)

    async def log_repair_event(self, memory_id: str, message: str) -> None:
        def _log():
            import datetime
            conn = self._get_conn_sync()
            row = conn.execute(
                "SELECT repair_history FROM memories WHERE id = ?", (str(memory_id),)
            ).fetchone()
            if row:
                history = json.loads(row["repair_history"])
                event = f"{datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat()} - {message}"
                history.append(event)
                conn.execute(
                    "UPDATE memories SET repair_history = ? WHERE id = ?",
                    (json.dumps(history), str(memory_id)),
                )
                conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_log)

    async def clear_all_memories(self) -> None:
        def _clear():
            conn = self._get_conn_sync()
            conn.execute("DELETE FROM memories")
            conn.commit()

        async with self._write_lock:
            await asyncio.to_thread(_clear)

    # ---- Helpers ----

    def _to_memory(self, row: sqlite3.Row) -> MemoryItem:
        row_dict = dict(row)
        if "confidence" in row_dict:
            row_dict["confidence_score"] = row_dict.pop("confidence")
        if "embedding" in row_dict:
            row_dict.pop("embedding")
        for field in ["influenced_decisions", "repair_history", "task_ids"]:
            if field in row_dict and isinstance(row_dict[field], str):
                row_dict[field] = json.loads(row_dict[field])
        return MemoryItem(**row_dict)
