# memory/store.py
import sqlite3
import json
import logging
import threading
from uuid import UUID
from typing import List, Optional
from datetime import datetime, timezone

from memory.schemas import MemoryItem
from memory.lifecycle import update_lifecycle

logger = logging.getLogger(__name__)

DB_PATH = "cocortex_memory.db"

# Whitelist of columns that may be updated via update_memory().
# Prevents SQL-injection style attacks via dynamic field names.
_ALLOWED_UPDATE_FIELDS = {
    "content",
    "memory_type",
    "source_agent",
    "timestamp",
    "confidence",
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


class MemoryStore:
    def __init__(self, db_path: str = DB_PATH):
        # Thread-safety: all writes are serialised through this lock.
        self._write_lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._initialize_table()
        self._migrate_schema()

    def _initialize_table(self):
        with self._write_lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    memory_type TEXT,
                    source_agent TEXT,
                    timestamp TEXT,
                    confidence REAL,
                    status TEXT,
                    influenced_decisions TEXT
                )
                """
            )
            self.conn.commit()

    def _migrate_schema(self):
        migrations = {
            "usage_count": "INTEGER DEFAULT 0",
            "failure_count": "INTEGER DEFAULT 0",
            "last_validated_at": "TEXT",
            "lifecycle_state": "TEXT DEFAULT 'episodic'",
            "repair_history": "TEXT DEFAULT '[]'",
            "task_ids": "TEXT DEFAULT '[]'",
            "embedding": "BLOB",
        }
        with self._write_lock:
            for column, definition in migrations.items():
                try:
                    self.conn.execute(
                        f"ALTER TABLE memories ADD COLUMN {column} {definition}"
                    )
                except sqlite3.OperationalError:
                    pass  # column already exists
            self.conn.commit()

    # -------- PUBLIC API --------

    def add_memory(self, memory_item: MemoryItem):
        with self._write_lock:
            self.conn.execute(
                """
                INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(memory_item.id),
                    memory_item.content,
                    memory_item.memory_type,
                    memory_item.source_agent,
                    memory_item.timestamp.isoformat(),
                    memory_item.confidence_score,
                    memory_item.status,
                    json.dumps(memory_item.influenced_decisions),
                    memory_item.usage_count,
                    memory_item.failure_count,
                    memory_item.last_validated_at.isoformat()
                    if memory_item.last_validated_at else None,
                    memory_item.lifecycle_state,
                    json.dumps(memory_item.repair_history),
                    json.dumps(memory_item.task_ids),
                    None,  # embedding BLOB — set later via set_embedding()
                ),
            )
            self.conn.commit()

    def get_memory(self, memory_id: UUID) -> Optional[MemoryItem]:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?",
            (str(memory_id),),
        ).fetchone()
        return self._to_memory(row) if row else None

    def get_memory_by_type(self, memory_type: str) -> List[MemoryItem]:
        rows = self.conn.execute(
            """
            SELECT * FROM memories
            WHERE memory_type = ? AND status = 'active'
            """,
            (memory_type,),
        ).fetchall()
        return [self._to_memory(row) for row in rows]

    def update_memory(self, memory_id: UUID, fields: dict):
        """
        Update arbitrary fields on a memory row.

        Only fields in _ALLOWED_UPDATE_FIELDS are permitted — any unknown
        field name raises ValueError to prevent SQL-injection via dynamic SQL.
        """
        unknown = set(fields) - _ALLOWED_UPDATE_FIELDS
        if unknown:
            raise ValueError(
                f"update_memory: unknown field(s) {unknown}. "
                f"Allowed fields: {_ALLOWED_UPDATE_FIELDS}"
            )
        with self._write_lock:
            for field, value in fields.items():
                self.conn.execute(
                    f"UPDATE memories SET {field} = ? WHERE id = ?",
                    (
                        json.dumps(value) if isinstance(value, list) else value,
                        str(memory_id),
                    ),
                )
            self.conn.commit()

    def mark_memory_used(self, memory_id: UUID):
        memory = self.get_memory(memory_id)
        if not memory:
            return
        memory.usage_count += 1
        memory.lifecycle_state = update_lifecycle(memory)
        self.update_memory(
            memory_id,
            {"usage_count": memory.usage_count, "lifecycle_state": memory.lifecycle_state},
        )

    def mark_memory_failed(self, memory_id: UUID):
        memory = self.get_memory(memory_id)
        if not memory:
            return
        memory.failure_count += 1
        memory.lifecycle_state = update_lifecycle(memory)
        self.update_memory(
            memory_id,
            {"failure_count": memory.failure_count, "lifecycle_state": memory.lifecycle_state},
        )

    def validate_memory(self, memory_id: UUID):
        memory = self.get_memory(memory_id)
        if not memory:
            return
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        memory.last_validated_at = now
        memory.lifecycle_state = update_lifecycle(memory)
        self.update_memory(
            memory_id,
            {
                "last_validated_at": now.isoformat(),
                "lifecycle_state": memory.lifecycle_state,
            },
        )

    def log_repair_event(self, memory_id: UUID, message: str):
        memory = self.get_memory(memory_id)
        if not memory:
            return
        event = f"{datetime.now(timezone.utc).replace(tzinfo=None).isoformat()} - {message}"
        memory.repair_history.append(event)
        self.update_memory(memory_id, {"repair_history": memory.repair_history})

    def link_memory_to_task(self, memory_id: UUID, task_id: str):
        memory = self.get_memory(memory_id)
        if not memory:
            return
        if task_id not in memory.task_ids:
            memory.task_ids.append(task_id)
        self.update_memory(memory_id, {"task_ids": memory.task_ids})

    def update_confidence(self, memory_id: UUID, new_score: float):
        clamped = round(max(0.0, min(float(new_score), 1.0)), 3)
        memory = self.get_memory(memory_id)
        if not memory:
            return
        memory.confidence_score = clamped
        memory.lifecycle_state = update_lifecycle(memory)
        self.update_memory(
            memory_id,
            {"confidence": clamped, "lifecycle_state": memory.lifecycle_state},
        )

    def update_status(self, memory_id: UUID, status: str):
        if status not in ("active", "quarantined"):
            raise ValueError(f"Invalid status '{status}'. Must be 'active' or 'quarantined'.")
        self.update_memory(memory_id, {"status": status})

    def link_memory_to_decision(self, memory_id: UUID, decision_id: str):
        memory = self.get_memory(memory_id)
        if not memory:
            return
        if decision_id not in memory.influenced_decisions:
            memory.influenced_decisions.append(decision_id)
            self.update_memory(
                memory_id,
                {"influenced_decisions": memory.influenced_decisions},
            )

    def promote_memory(self, memory_id: UUID):
        memory = self.get_memory(memory_id)
        if not memory:
            return
        self.update_memory(memory_id, {"memory_type": "semantic"})

    def get_all_active_memories(self) -> List[MemoryItem]:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE status = 'active'"
        ).fetchall()
        return [self._to_memory(row) for row in rows]

    def get_quarantined_memories(self) -> List[MemoryItem]:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE status = 'quarantined'"
        ).fetchall()
        return [self._to_memory(row) for row in rows]

    def get_memories_by_session(self, session_id: str) -> List[MemoryItem]:
        """
        Return all active memories whose task_ids list contains session_id.
        Uses in-Python filtering for exact JSON membership (avoids LIKE false-positives).
        """
        all_active = self.get_all_active_memories()
        return [m for m in all_active if session_id in m.task_ids]

    def set_embedding(self, memory_id, embedding_blob: bytes):
        """Store a precomputed embedding BLOB for a memory."""
        with self._write_lock:
            self.conn.execute(
                "UPDATE memories SET embedding = ? WHERE id = ?",
                (embedding_blob, str(memory_id)),
            )
            self.conn.commit()

    def get_all_embeddings(self):
        """
        Return all (id, embedding_blob) pairs where embedding is not NULL.
        Used to rebuild the FAISS index at startup.
        """
        rows = self.conn.execute(
            "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
        ).fetchall()
        return [(row["id"], row["embedding"]) for row in rows]

    def delete_by_session(self, session_id: str):
        """
        Delete all memory rows whose task_ids list contains exactly session_id.

        Uses JSON exact-match to avoid the LIKE '%session-1%' false-positive
        problem (e.g. 'session-1' matching 'session-10').
        """
        all_memories = (
            self.get_all_active_memories() + self.get_quarantined_memories()
        )
        ids_to_delete = [
            str(m.id)
            for m in all_memories
            if session_id in m.task_ids
        ]
        if ids_to_delete:
            with self._write_lock:
                placeholders = ",".join("?" for _ in ids_to_delete)
                self.conn.execute(
                    f"DELETE FROM memories WHERE id IN ({placeholders})",
                    ids_to_delete,
                )
                self.conn.commit()
        logger.info(
            "delete_by_session: removed %d record(s) for session='%s'",
            len(ids_to_delete), session_id,
        )

    def clear_all_memories(self):
        with self._write_lock:
            self.conn.execute("DELETE FROM memories")
            self.conn.commit()

    # -------- INTERNAL --------

    def _to_memory(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            source_agent=row["source_agent"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            confidence_score=row["confidence"],
            status=row["status"],
            influenced_decisions=json.loads(row["influenced_decisions"]),
            usage_count=row["usage_count"],
            failure_count=row["failure_count"],
            last_validated_at=datetime.fromisoformat(row["last_validated_at"])
            if row["last_validated_at"] else None,
            lifecycle_state=row["lifecycle_state"],
            repair_history=json.loads(row["repair_history"]),
            task_ids=json.loads(row["task_ids"]),
        )