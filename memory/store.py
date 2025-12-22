import sqlite3
import json
from uuid import UUID
from typing import List, Optional
from memory.schemas import MemoryItem


DB_PATH = "cocortex_memory.db"


class MemoryStore:
    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._initialize_table()

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

    # -------- PUBLIC API --------

    def add_memory(self, memory_item: MemoryItem):
        self.conn.execute(
            """
            INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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

    def promote_memory(self, memory_id: UUID):
        self.update_memory(memory_id, {"memory_type": "semantic"})

    def update_confidence(self, memory_id: UUID, new_score: float):
        self.update_memory(memory_id, {"confidence": new_score})

    def update_status(self, memory_id: UUID, status: str):
        self.update_memory(memory_id, {"status": status})

    def link_memory_to_decision(self, memory_id: UUID, decision_id: str):
        memory = self.get_memory(memory_id)
        if not memory:
            return
        memory.influenced_decisions.append(decision_id)
        self.update_memory(
            memory_id,
            {"influenced_decisions": memory.influenced_decisions},
        )

    def delete_memory(self, memory_id: UUID):
        self.conn.execute(
            "DELETE FROM memories WHERE id = ?",
            (str(memory_id),),
        )
        self.conn.commit()

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
            timestamp=row["timestamp"],
            confidence_score=row["confidence"],
            status=row["status"],
            influenced_decisions=json.loads(row["influenced_decisions"]),
        )
