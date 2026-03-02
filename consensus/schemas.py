from pydantic import BaseModel, Field
from typing import Literal, Dict


class MemoryProposal(BaseModel):
    content: str
    # "memory_manager" added to match MemoryItem.source_agent — previously
    # MemoryProposal rejected "memory_manager" while MemoryItem accepted it,
    # causing a ValidationError if the engine tried to create a proposal for
    # a memory sourced from the memory_manager agent.
    source_agent: Literal["planner", "worker", "evaluator", "memory_manager"]
    suggested_type: Literal["episodic", "semantic"]
    context: Dict[str, str]


class Vote(BaseModel):
    approve: bool
    # ge/le enforces [0.0, 1.0] at the Pydantic boundary.
    # LLMs routinely return values like 1.5 or -0.1; without this,
    # out-of-range values corrupt run_consensus() averaging and every
    # downstream confidence score written to the DB.
    confidence: float = Field(ge=0.0, le=1.0)
    risk: bool
    reason: str