from datetime import datetime, timezone
from memory.schemas import MemoryItem


def compute_reliability(memory: MemoryItem) -> float:
    score = memory.confidence_score
    score += min(memory.usage_count * 0.02, 0.2)
    score -= memory.failure_count * 0.15

    # Use last_validated_at if set, otherwise fall back to creation timestamp.
    # This ensures memories that were never validated still decay over time.
    reference_date = memory.last_validated_at or memory.timestamp
    days = (datetime.now(timezone.utc).replace(tzinfo=None) - reference_date).days
    score -= min(days * 0.01, 0.2)

    return round(max(0.0, min(score, 1.0)), 3)