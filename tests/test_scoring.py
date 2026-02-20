"""
Tests for memory/scoring.py and memory/lifecycle.py
Run from project root: python -m pytest tests/test_scoring.py -v
"""
import pytest
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.schemas import MemoryItem
from memory.scoring import compute_reliability
from memory.lifecycle import update_lifecycle


@pytest.fixture
def fresh_memory():
    return MemoryItem(
        content="A factual statement about photosynthesis.",
        source_agent="worker",
        memory_type="episodic",
        confidence_score=0.8,
    )


# --- Scoring ---

def test_reliability_equals_confidence_for_new_memory(fresh_memory):
    # New memory with no usage, no failures, just created — barely any decay
    score = compute_reliability(fresh_memory)
    # Should be close to 0.8 (minor staleness for 0 days)
    assert 0.75 <= score <= 0.80


def test_reliability_increases_with_usage(fresh_memory):
    base = compute_reliability(fresh_memory)
    fresh_memory.usage_count = 5
    boosted = compute_reliability(fresh_memory)
    assert boosted > base


def test_reliability_caps_usage_bonus_at_10_uses(fresh_memory):
    fresh_memory.usage_count = 10
    score_10 = compute_reliability(fresh_memory)
    fresh_memory.usage_count = 100
    score_100 = compute_reliability(fresh_memory)
    # Both should be the same — bonus is capped at +0.2
    assert score_10 == score_100


def test_reliability_decreases_with_failures(fresh_memory):
    base = compute_reliability(fresh_memory)
    fresh_memory.failure_count = 2
    degraded = compute_reliability(fresh_memory)
    assert degraded < base


def test_reliability_never_below_zero(fresh_memory):
    fresh_memory.failure_count = 100
    score = compute_reliability(fresh_memory)
    assert score >= 0.0


def test_reliability_never_above_one(fresh_memory):
    fresh_memory.confidence_score = 1.0
    fresh_memory.usage_count = 100
    score = compute_reliability(fresh_memory)
    assert score <= 1.0


def test_reliability_decays_with_old_timestamp(fresh_memory):
    base = compute_reliability(fresh_memory)
    # Simulate memory created 90 days ago
    fresh_memory.timestamp = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=90)
    stale = compute_reliability(fresh_memory)
    assert stale < base


def test_staleness_uses_timestamp_when_not_validated(fresh_memory):
    """Regression: old memories with no validation should still decay."""
    fresh_memory.timestamp = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30)
    fresh_memory.last_validated_at = None
    score = compute_reliability(fresh_memory)
    # Should have some staleness decay (30 days * 0.01 = 0.3, capped at 0.2)
    assert score < fresh_memory.confidence_score


def test_staleness_uses_validated_at_when_available(fresh_memory):
    """last_validated_at should take priority over timestamp."""
    fresh_memory.timestamp = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=90)
    fresh_memory.last_validated_at = datetime.now(timezone.utc).replace(tzinfo=None)  # validated today
    score = compute_reliability(fresh_memory)
    # Should not have much decay despite old timestamp
    assert score >= 0.75


# --- Lifecycle ---

def test_lifecycle_promotes_to_semantic_at_high_reliability(fresh_memory):
    fresh_memory.confidence_score = 0.9
    fresh_memory.usage_count = 5
    fresh_memory.failure_count = 0
    state = update_lifecycle(fresh_memory)
    assert state == "semantic"


def test_lifecycle_deprecated_after_3_failures(fresh_memory):
    fresh_memory.failure_count = 3
    state = update_lifecycle(fresh_memory)
    assert state == "deprecated"


def test_lifecycle_stale_at_low_reliability(fresh_memory):
    fresh_memory.confidence_score = 0.4
    fresh_memory.failure_count = 1
    fresh_memory.timestamp = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=20)
    state = update_lifecycle(fresh_memory)
    assert state in ("stale", "archived", "deprecated")


def test_lifecycle_archived_at_very_low_reliability(fresh_memory):
    fresh_memory.confidence_score = 0.1
    fresh_memory.failure_count = 2
    fresh_memory.timestamp = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=60)
    state = update_lifecycle(fresh_memory)
    assert state == "archived"