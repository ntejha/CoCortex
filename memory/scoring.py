# memory/scoring.py
from datetime import datetime
from memory.schemas import MemoryItem

def compute_reliability(memory: MemoryItem) -> float:
    score = memory.confidence_score
    score += min(memory.usage_count * 0.02, 0.2)
    score -= memory.failure_count * 0.15

    if memory.last_validated_at:
        days = (datetime.utcnow() - memory.last_validated_at).days
        score -= min(days * 0.01, 0.2)

    return round(max(0.0, min(score, 1.0)), 3)
