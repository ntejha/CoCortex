"""
consensus/schemas.py
"""
from pydantic import BaseModel, Field
from typing import Literal, Dict


class MemoryProposal(BaseModel):
    content: str
    source_agent: Literal["planner", "worker", "evaluator", "memory_manager"]
    suggested_type: Literal["episodic", "semantic"]
    context: Dict[str, str]


class Vote(BaseModel):
    approve: bool
    confidence: float = Field(ge=0.0, le=1.0)
    risk: bool
    reason: str