"""
memory/scoring.py
=================
Reliability scoring for CoCortex memory items.

SCORING MODEL
-------------
The reliability score models three real-world intuitions about knowledge trust:

1. USAGE REWARD: A memory used more often and found helpful becomes more trusted.
   Coefficient: +0.02 per use, capped at +0.20 (10 uses = max reward).
   Rationale: Caps at 0.20 to prevent runaway trust accumulation — no memory
   should become unquestionable just because it's been used a lot.

2. FAILURE PENALTY: A memory associated with a failed decision is less trustworthy.
   Coefficient: -0.15 per failure.
   Rationale: Penalty (0.15) is 7.5× the per-use reward (0.02) because one
   failure is much more informative than one use. A single failure should
   meaningfully reduce trust; it takes ~7 successful uses to offset one failure.

3. TIME DECAY: Unvalidated memories become less trusted over time (knowledge expires).
   Coefficient: -0.01 per day since last validation, capped at -0.20 (20 days max decay).
   Rationale: Caps at 0.20 so even ancient memories don't drop to zero from age
   alone — they need actual failures to reach zero. A 20-day cap reflects that
   most factual knowledge doesn't change on human timescales but should still
   be periodically validated.

TUNING NOTE
-----------
These coefficients are heuristic defaults. For domain-specific deployments,
use ScoringConfig to tune them without modifying logic:

    config = ScoringConfig(failure_penalty=0.20, usage_reward=0.03)
    score = compute_reliability(memory, config=config)
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from memory.schemas import MemoryItem


@dataclass
class ScoringConfig:
    """
    Named constants for scoring coefficients.
    Override to tune the scoring model for your domain without touching logic.
    """
    usage_reward: float = 0.02      # per use
    usage_cap: float = 0.20         # max reward from usage
    failure_penalty: float = 0.15   # per failure
    decay_rate: float = 0.01        # per day since last validation
    decay_cap: float = 0.20         # max penalty from time decay


_DEFAULT_CONFIG = ScoringConfig()


def compute_reliability(memory: MemoryItem, config: ScoringConfig = _DEFAULT_CONFIG) -> float:
    """
    Compute a reliability score in [0.0, 1.0] for a memory item.

    Args:
        memory: The memory item to score.
        config: Scoring coefficients (use defaults or override per deployment).

    Returns:
        Float in [0.0, 1.0] representing current reliability.
    """
    score = memory.confidence_score

    # Usage reward: capped to prevent over-trust accumulation
    score += min(memory.usage_count * config.usage_reward, config.usage_cap)

    # Failure penalty: linear, no cap — many failures should kill a memory
    score -= memory.failure_count * config.failure_penalty

    # Time decay: use last_validated_at if set, else fall back to creation time
    reference_date = memory.last_validated_at or memory.timestamp
    days = (datetime.now(timezone.utc).replace(tzinfo=None) - reference_date).days
    score -= min(days * config.decay_rate, config.decay_cap)

    return round(max(0.0, min(score, 1.0)), 3)