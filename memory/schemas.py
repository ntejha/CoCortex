# memory/schemas.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from uuid import UUID, uuid4
from datetime import datetime

LifecycleState = Literal[
    "episodic",
    "semantic",
    "stale",
    "deprecated",
    "archived",
]

class MemoryItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str

    memory_type: Literal["episodic", "semantic"] = "episodic"
    source_agent: Literal["planner", "worker", "evaluator", "memory_manager"]

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    status: Literal["active", "quarantined"] = "active"

    influenced_decisions: List[str] = Field(default_factory=list)

    # 🔹 Persisted intelligence fields
    usage_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    last_validated_at: Optional[datetime] = None
    lifecycle_state: LifecycleState = "episodic"
