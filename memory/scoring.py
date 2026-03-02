"""
memory/scoring.py

Reliability scoring with configurable parameters via ScoringConfig.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from memory.schemas import MemoryItem


@dataclass
class ScoringConfig:
    """Named constants for the reliability formula — override in tests or experiments."""
    usage_reward: float = 0.02    # added per successful use
    usage_cap: float = 0.20       # maximum bonus from usage
    failure_penalty: float = 0.15 # subtracted per failure
    decay_rate: float = 0.01      # subtracted per day since last validation
    decay_cap: float = 0.20       # maximum staleness penalty


_DEFAULT_CONFIG = ScoringConfig()


def compute_reliability(memory: MemoryItem, config: ScoringConfig = _DEFAULT_CONFIG) -> float:
    score = memory.confidence_score
    score += min(memory.usage_count * config.usage_reward, config.usage_cap)
    score -= memory.failure_count * config.failure_penalty

    # Use last_validated_at if set, otherwise fall back to creation timestamp.
    reference_date = memory.last_validated_at or memory.timestamp
    days = (datetime.now(timezone.utc).replace(tzinfo=None) - reference_date).days
    score -= min(days * config.decay_rate, config.decay_cap)

    return round(max(0.0, min(score, 1.0)), 3)