"""
Tests for memory/repair.py
Run from project root: python -m pytest tests/test_repair.py -v
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.store import MemoryStore
from memory.schemas import MemoryItem
from memory.repair import decide_repair_action, trace_suspect_memories


@pytest.fixture
def store(tmp_path):
    db = str(tmp_path / "test.db")
    s = MemoryStore(db_path=db)
    yield s
    s.conn.close()


def test_decide_repair_action_quarantines_incorrect():
    action = decide_repair_action("incorrect", confidence=0.9, failure_count=0)
    assert action == "quarantine"


def test_decide_repair_action_downranks_uncertain_low_confidence():
    action = decide_repair_action("uncertain", confidence=0.4, failure_count=0)
    assert action == "downrank"


def test_decide_repair_action_no_action_uncertain_high_confidence():
    action = decide_repair_action("uncertain", confidence=0.8, failure_count=0)
    assert action == "none"


def test_decide_repair_action_downranks_high_failure_count():
    action = decide_repair_action("correct", confidence=0.9, failure_count=3)
    assert action == "downrank"


def test_decide_repair_action_none_when_all_ok():
    action = decide_repair_action("correct", confidence=0.9, failure_count=0)
    assert action == "none"


def test_trace_suspect_memories_finds_linked_memory(store):
    mem = MemoryItem(
        content="Photosynthesis occurs only at night.",
        source_agent="worker",
        memory_type="semantic",
    )
    store.add_memory(mem)
    store.link_memory_to_decision(mem.id, "planner_fail001")

    suspects = trace_suspect_memories(store, "planner_fail001")
    assert len(suspects) == 1
    assert suspects[0].id == mem.id


def test_trace_suspect_memories_ignores_unlinked(store):
    mem = MemoryItem(
        content="Unrelated memory about something else.",
        source_agent="worker",
        memory_type="episodic",
    )
    store.add_memory(mem)

    suspects = trace_suspect_memories(store, "planner_fail001")
    assert len(suspects) == 0


def test_repair_stores_update_confidence_correctly(store):
    """
    This is the regression test for the original bug:
    repair.py called store.update_confidence() which didn't exist.
    Now it does — verify the call works end-to-end.
    """
    mem = MemoryItem(
        content="Some uncertain memory.",
        source_agent="worker",
        memory_type="semantic",
        confidence_score=0.8,
    )
    store.add_memory(mem)

    # Directly call the method that was missing — must not raise
    store.update_confidence(mem.id, 0.6)
    updated = store.get_memory(mem.id)
    assert updated.confidence_score == 0.6


def test_repair_stores_update_status_correctly(store):
    """
    Regression test: repair.py called store.update_status() which didn't exist.
    """
    mem = MemoryItem(
        content="Incorrect memory.",
        source_agent="worker",
        memory_type="semantic",
    )
    store.add_memory(mem)

    store.update_status(mem.id, "quarantined")
    updated = store.get_memory(mem.id)
    assert updated.status == "quarantined"


def test_trace_finds_quarantined_memory(store):
    """
    Regression: trace_suspect_memories was skipping quarantined memories.
    A memory quarantined by a previous repair cycle should still be traceable.
    """
    mem = MemoryItem(
        content="Already quarantined but still suspect.",
        source_agent="worker",
        memory_type="semantic",
    )
    store.add_memory(mem)
    store.link_memory_to_decision(mem.id, "planner_fail999")
    store.update_status(mem.id, "quarantined")  # quarantined before traceback runs

    suspects = trace_suspect_memories(store, "planner_fail999")
    assert len(suspects) == 1
    assert suspects[0].id == mem.id


def test_verifier_not_correct_returns_incorrect():
    """
    Regression: 'not correct' contains 'correct' as substring.
    Old code would return 'correct' — new code returns 'incorrect'.
    """
    from memory.verification import MemoryVerifier

    class MockLLM:
        def generate(self, prompt):
            return "not correct"

    verifier = MemoryVerifier(MockLLM())
    result = verifier.verify("Photosynthesis occurs only at night.")
    assert result == "incorrect"


def test_verifier_incorrect_returns_incorrect():
    from memory.verification import MemoryVerifier

    class MockLLM:
        def generate(self, prompt):
            return "incorrect"

    verifier = MemoryVerifier(MockLLM())
    result = verifier.verify("Photosynthesis occurs only at night.")
    assert result == "incorrect"


def test_verifier_correct_returns_correct():
    from memory.verification import MemoryVerifier

    class MockLLM:
        def generate(self, prompt):
            return "correct"

    verifier = MemoryVerifier(MockLLM())
    result = verifier.verify("Photosynthesis converts CO2 into glucose.")
    assert result == "correct"