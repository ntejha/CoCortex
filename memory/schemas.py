from pydantic import BaseModel, Field
from typing import List, Literal
from uuid import UUID, uuid4
from datetime import datetime


class MemoryItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)

    content: str

    memory_type: Literal["episodic", "semantic"]

    source_agent: Literal[
        "planner",
        "worker",
        "evaluator",
        "memory_manager"
    ]

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    confidence_score: float = 0.5

    status: Literal[
        "active",
        "quarantined",
        "deprecated"
    ] = "active"

    influenced_decisions: List[str] = []
