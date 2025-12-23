from pydantic import BaseModel
from typing import Literal, Dict

class MemoryProposal(BaseModel):
    content: str
    source_agent: Literal["planner", "worker", "evaluator"]
    suggested_type: Literal["episodic", "semantic"]
    context: Dict[str, str]

class Vote(BaseModel):
    approve: bool
    confidence: float
    risk: bool
    reason: str
