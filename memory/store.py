# memory/store.py
import sqlite3
import json
from uuid import UUID
from typing import List, Optional
from datetime import datetime, timezone

from memory.schemas import MemoryItem
from memory.lifecycle import update_lifecycle

DB_PATH = "cocortex_memory.db"


class MemoryStore:
    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._initialize_table()
        self._migrate_schema()

    def _initialize_table(self):
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

    # 🔹 SAFE MIGRATION (NEW)
    def _migrate_schema(self):
        migrations = {
            "usage_count": "INTEGER DEFAULT 0",
            "failure_count": "INTEGER DEFAULT 0",
            "last_validated_at": "TEXT",
            "lifecycle_state": "TEXT DEFAULT 'episodic'",
            "repair_history": "TEXT DEFAULT '[]'",
            "task_ids": "TEXT DEFAULT '[]'",
        }

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
        self.conn.execute(
            """
            INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            {
                "usage_count": memory.usage_count,
                "lifecycle_state": memory.lifecycle_state,
            },
        )

    def mark_memory_failed(self, memory_id: UUID):
        memory = self.get_memory(memory_id)
        if not memory:
            return

        memory.failure_count += 1
        memory.lifecycle_state = update_lifecycle(memory)

        self.update_memory(
            memory_id,
            {
                "failure_count": memory.failure_count,
                "lifecycle_state": memory.lifecycle_state,
            },
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

        self.update_memory(
            memory_id,
            {"repair_history": memory.repair_history},
        )


    def link_memory_to_task(self, memory_id: UUID, task_id: str):
        memory = self.get_memory(memory_id)
        if not memory:
            return

        if task_id not in memory.task_ids:
            memory.task_ids.append(task_id)

        self.update_memory(
            memory_id,
            {"task_ids": memory.task_ids},
        )


    def update_confidence(self, memory_id: UUID, new_score: float):
        """Update confidence score, clamped to [0.0, 1.0]."""
        clamped = round(max(0.0, min(float(new_score), 1.0)), 3)
        memory = self.get_memory(memory_id)
        if not memory:
            return
        memory.confidence_score = clamped
        memory.lifecycle_state = update_lifecycle(memory)
        self.update_memory(
            memory_id,
            {
                "confidence": clamped,
                "lifecycle_state": memory.lifecycle_state,
            },
        )

    def update_status(self, memory_id: UUID, status: str):
        """Update status — 'active' or 'quarantined'."""
        if status not in ("active", "quarantined"):
            raise ValueError(f"Invalid status '{status}'. Must be 'active' or 'quarantined'.")
        self.update_memory(memory_id, {"status": status})

    def link_memory_to_decision(self, memory_id: UUID, decision_id: str):
        """Append a decision ID to this memory's influenced_decisions list."""
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
        """Promote a memory from episodic → semantic."""
        memory = self.get_memory(memory_id)
        if not memory:
            return
        self.update_memory(memory_id, {"memory_type": "semantic"})

    def get_all_active_memories(self) -> List[MemoryItem]:
        """Return all memories with status = 'active', regardless of type."""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE status = 'active'"
        ).fetchall()
        return [self._to_memory(row) for row in rows]


    def clear_all_memories(self):
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