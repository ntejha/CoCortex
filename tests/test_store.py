"""
Tests for memory/store.py
Run from project root: python -m pytest tests/test_store.py -v
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.store import MemoryStore
from memory.schemas import MemoryItem


@pytest.fixture
def store(tmp_path):
    """Fresh in-memory store for each test."""
    db = str(tmp_path / "test.db")
    s = MemoryStore(db_path=db)
    yield s
    s.conn.close()


@pytest.fixture
def sample_memory():
    return MemoryItem(
        content="Photosynthesis converts CO2 and water into glucose using sunlight.",
        memory_type="episodic",
        source_agent="worker",
        confidence_score=0.8,
    )


def test_add_and_get_memory(store, sample_memory):
    store.add_memory(sample_memory)
    retrieved = store.get_memory(sample_memory.id)
    assert retrieved is not None
    assert retrieved.content == sample_memory.content
    assert retrieved.source_agent == "worker"


def test_get_memory_by_type(store, sample_memory):
    store.add_memory(sample_memory)
    results = store.get_memory_by_type("episodic")
    assert len(results) == 1
    assert results[0].id == sample_memory.id


def test_get_memory_returns_none_for_missing(store):
    from uuid import uuid4
    assert store.get_memory(uuid4()) is None


def test_update_confidence(store, sample_memory):
    store.add_memory(sample_memory)
    store.update_confidence(sample_memory.id, 0.95)
    updated = store.get_memory(sample_memory.id)
    assert updated.confidence_score == 0.95


def test_update_confidence_clamps_above_1(store, sample_memory):
    store.add_memory(sample_memory)
    store.update_confidence(sample_memory.id, 1.5)
    updated = store.get_memory(sample_memory.id)
    assert updated.confidence_score == 1.0


def test_update_confidence_clamps_below_0(store, sample_memory):
    store.add_memory(sample_memory)
    store.update_confidence(sample_memory.id, -0.3)
    updated = store.get_memory(sample_memory.id)
    assert updated.confidence_score == 0.0


def test_update_status_quarantine(store, sample_memory):
    store.add_memory(sample_memory)
    store.update_status(sample_memory.id, "quarantined")
    updated = store.get_memory(sample_memory.id)
    assert updated.status == "quarantined"


def test_get_memory_by_type_excludes_quarantined(store, sample_memory):
    store.add_memory(sample_memory)
    store.update_status(sample_memory.id, "quarantined")
    results = store.get_memory_by_type("episodic")
    assert len(results) == 0


def test_link_memory_to_decision(store, sample_memory):
    store.add_memory(sample_memory)
    store.link_memory_to_decision(sample_memory.id, "planner_abc123")
    updated = store.get_memory(sample_memory.id)
    assert "planner_abc123" in updated.influenced_decisions


def test_link_memory_to_decision_no_duplicates(store, sample_memory):
    store.add_memory(sample_memory)
    store.link_memory_to_decision(sample_memory.id, "planner_abc123")
    store.link_memory_to_decision(sample_memory.id, "planner_abc123")
    updated = store.get_memory(sample_memory.id)
    assert updated.influenced_decisions.count("planner_abc123") == 1


def test_promote_memory(store, sample_memory):
    store.add_memory(sample_memory)
    assert sample_memory.memory_type == "episodic"
    store.promote_memory(sample_memory.id)
    updated = store.get_memory(sample_memory.id)
    assert updated.memory_type == "semantic"


def test_mark_memory_used_increments_count(store, sample_memory):
    store.add_memory(sample_memory)
    store.mark_memory_used(sample_memory.id)
    store.mark_memory_used(sample_memory.id)
    updated = store.get_memory(sample_memory.id)
    assert updated.usage_count == 2


def test_mark_memory_failed_increments_count(store, sample_memory):
    store.add_memory(sample_memory)
    store.mark_memory_failed(sample_memory.id)
    updated = store.get_memory(sample_memory.id)
    assert updated.failure_count == 1


def test_get_all_active_memories(store):
    m1 = MemoryItem(content="Active memory", source_agent="worker", memory_type="episodic")
    m2 = MemoryItem(content="Quarantined memory", source_agent="worker", memory_type="episodic")
    store.add_memory(m1)
    store.add_memory(m2)
    store.update_status(m2.id, "quarantined")
    active = store.get_all_active_memories()
    ids = [m.id for m in active]
    assert m1.id in ids
    assert m2.id not in ids


def test_clear_all_memories(store, sample_memory):
    store.add_memory(sample_memory)
    store.clear_all_memories()
    assert store.get_memory(sample_memory.id) is None


def test_log_repair_event(store, sample_memory):
    store.add_memory(sample_memory)
    store.log_repair_event(sample_memory.id, "Downranked due to failure")
    updated = store.get_memory(sample_memory.id)
    assert any("Downranked" in event for event in updated.repair_history)


def test_link_memory_to_task(store, sample_memory):
    store.add_memory(sample_memory)
    store.link_memory_to_task(sample_memory.id, "task_001")
    updated = store.get_memory(sample_memory.id)
    assert "task_001" in updated.task_ids