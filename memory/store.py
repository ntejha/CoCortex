# memory/store.py
import sqlite3
import json
from uuid import UUID
from typing import List, Optional
from datetime import datetime

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
            INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

        now = datetime.utcnow()
        memory.last_validated_at = now
        memory.lifecycle_state = update_lifecycle(memory)

        self.update_memory(
            memory_id,
            {
                "last_validated_at": now.isoformat(),
                "lifecycle_state": memory.lifecycle_state,
            },
        )

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
        )
