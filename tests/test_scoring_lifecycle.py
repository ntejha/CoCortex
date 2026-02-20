"""
tests/test_scoring_lifecycle.py

Tests for reliability scoring formula and lifecycle state transitions.

Run with: pytest tests/test_scoring_lifecycle.py -v
"""

import pytest
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.schemas import MemoryItem
from memory.scoring import compute_reliability
from memory.lifecycle import update_lifecycle


# ── helpers ───────────────────────────────────────────────────────────────────

def make_memory(**kwargs) -> MemoryItem:
    defaults = {
        "content": "Test memory content about a topic.",
        "source_agent": "worker",
        "confidence_score": 0.8,
    }
    defaults.update(kwargs)
    return MemoryItem(**defaults)


# ── compute_reliability ───────────────────────────────────────────────────────

def test_reliability_equals_confidence_for_fresh_memory():
    """Fresh memory with no usage, failures, or age — reliability ≈ confidence."""
    mem = make_memory(confidence_score=0.8)
    # Slight staleness decay since timestamp is set to now, but very small
    score = compute_reliability(mem)
    assert 0.75 <= score <= 0.81


def test_reliability_increases_with_usage():
    mem = make_memory(confidence_score=0.7, usage_count=5)
    score = compute_reliability(mem)
    # Each use adds 0.02, capped at 0.2
    assert score >= 0.79  # 0.7 + (5 * 0.02) = 0.8, minus tiny staleness


def test_reliability_usage_bonus_caps_at_0_2():
    mem = make_memory(confidence_score=0.7, usage_count=100)
    score = compute_reliability(mem)
    assert score <= 0.91  # 0.7 + 0.2 cap, minus staleness


def test_reliability_decreases_with_failures():
    mem = make_memory(confidence_score=0.9, failure_count=2)
    score = compute_reliability(mem)
    # 0.9 - (2 * 0.15) = 0.6, minus tiny staleness
    assert score <= 0.62


def test_reliability_clamps_to_zero_with_many_failures():
    mem = make_memory(confidence_score=0.5, failure_count=10)
    score = compute_reliability(mem)
    assert score == 0.0


def test_reliability_clamps_to_one_max():
    mem = make_memory(confidence_score=1.0, usage_count=10)
    score = compute_reliability(mem)
    assert score <= 1.0


def test_reliability_decays_for_old_unvalidated_memory():
    """A memory created 30 days ago with no validation should show decay."""
    old_timestamp = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=30)
    mem = make_memory(confidence_score=0.8)
    mem.timestamp = old_timestamp  # simulate old memory
    score = compute_reliability(mem)
    # 30 days * 0.01 = 0.3 decay, capped at 0.2
    assert score <= 0.61  # 0.8 - 0.2 cap


def test_reliability_uses_last_validated_date_when_available():
    """If validated recently, should use that date not the older timestamp."""
    mem = make_memory(confidence_score=0.8)
    mem.timestamp = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=60)
    mem.last_validated_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
    score_with_recent_validation = compute_reliability(mem)

    # Compare to same memory but validated 60 days ago
    mem2 = make_memory(confidence_score=0.8)
    mem2.timestamp = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=60)
    mem2.last_validated_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=60)
    score_with_old_validation = compute_reliability(mem2)

    assert score_with_recent_validation > score_with_old_validation


# ── update_lifecycle ──────────────────────────────────────────────────────────

def test_lifecycle_promotes_to_semantic_at_high_reliability():
    """High confidence + usage → semantic."""
    mem = make_memory(confidence_score=0.9, usage_count=5)
    state = update_lifecycle(mem)
    assert state == "semantic"


def test_lifecycle_deprecated_after_3_failures():
    """3+ failures → deprecated, regardless of reliability."""
    mem = make_memory(confidence_score=0.8, failure_count=3)
    state = update_lifecycle(mem)
    assert state == "deprecated"


def test_lifecycle_deprecated_at_4_failures_too():
    mem = make_memory(confidence_score=0.8, failure_count=4)
    state = update_lifecycle(mem)
    assert state == "deprecated"


def test_lifecycle_stale_at_low_reliability():
    """Low reliability (0.3–0.5) → stale."""
    # confidence=0.5, failure=1 → 0.5 - 0.15 - tiny_staleness ≈ 0.34 → stale range
    mem = make_memory(confidence_score=0.5, failure_count=1)
    state = update_lifecycle(mem)
    assert state == "stale"


def test_lifecycle_archived_at_very_low_reliability():
    """Very low reliability (< 0.3) → archived."""
    mem = make_memory(confidence_score=0.2, failure_count=2)
    state = update_lifecycle(mem)
    assert state == "archived"


def test_lifecycle_stays_current_at_mid_reliability():
    """Mid reliability (0.5–0.8) → stays at current state."""
    mem = make_memory(confidence_score=0.65)
    mem.lifecycle_state = "episodic"
    state = update_lifecycle(mem)
    assert state == "episodic"  # unchanged


def test_deprecated_takes_priority_over_reliability():
    """
    3 failures triggers deprecated even if reliability would say semantic.
    failure_count check happens before reliability check in lifecycle.py.
    """
    mem = make_memory(confidence_score=1.0, usage_count=10, failure_count=3)
    state = update_lifecycle(mem)
    assert state == "deprecated"