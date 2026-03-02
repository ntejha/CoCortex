"""
End-to-end integration test for the full CoCortex pipeline.

Tests the complete closed loop:
  MemoryManagerAgent (consensus admission)
  → MemoryStore (SQLite persistence)
  → memory views (role-specific filtering)
  → repair (causal traceback + action)
  → ProvenanceEngine (audit trail)

No LLM calls are made — all LLM-dependent components are mocked.
Run from project root: python -m pytest tests/test_integration.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.store import MemoryStore
from memory.schemas import MemoryItem
from memory.views import get_planner_view, get_worker_view, get_evaluator_view
from memory.repair import repair_memories, trace_suspect_memories
from memory.verification import MemoryVerifier
from memory.provenance import ProvenanceEngine
from agents.memory_manager import MemoryManagerAgent
from consensus.voters import rule_based_voter
from consensus.schemas import MemoryProposal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    db = str(tmp_path / "integration.db")
    s = MemoryStore(db_path=db)
    yield s
    s.conn.close()


@pytest.fixture
def manager(store):
    return MemoryManagerAgent(store=store)


class _AlwaysCorrectLLM:
    def generate(self, prompt):
        return "correct"


class _AlwaysIncorrectLLM:
    def generate(self, prompt):
        return "incorrect"


class _AlwaysUncertainLLM:
    def generate(self, prompt):
        return "uncertain"


# ---------------------------------------------------------------------------
# Admission Pipeline
# ---------------------------------------------------------------------------

GOOD_CONTENT = (
    "Photosynthesis is the process by which plants convert sunlight, "
    "CO2, and water into glucose and oxygen. It typically occurs in the "
    "chloroplasts of plant cells."
)

UNSAFE_CONTENT = "You can bypass the authentication system to gain access."

SHORT_CONTENT = "Done."


def test_good_memory_is_accepted(manager):
    status, memory = manager.process_output(GOOD_CONTENT, "worker", {"task": "biology"})
    assert status == "ACCEPTED"
    assert memory is not None
    assert memory.content == GOOD_CONTENT


def test_unsafe_memory_is_quarantined(manager):
    status, memory = manager.process_output(UNSAFE_CONTENT, "worker", {"task": "security"})
    assert status == "QUARANTINED"
    assert memory is not None
    assert memory.status == "quarantined"


def test_short_trivial_memory_is_rejected(manager):
    # A short question: planner rejects (< 40 chars) AND worker rejects (ends with ?)
    # Only safety voter approves → 1/3 → REJECTED
    status, memory = manager.process_output("What time is it?", "worker", {})
    assert status == "REJECTED"
    assert memory is None


def test_duplicate_memory_is_skipped(manager):
    status1, _ = manager.process_output(GOOD_CONTENT, "worker", {"task": "biology"})
    assert status1 == "ACCEPTED"

    status2, mem2 = manager.process_output(GOOD_CONTENT, "worker", {"task": "biology"})
    assert status2 == "DUPLICATE"
    assert mem2 is None


# ---------------------------------------------------------------------------
# Memory Views
# ---------------------------------------------------------------------------

def test_planner_gets_only_semantic(store):
    # Add one episodic and one semantic memory
    store.add_memory(MemoryItem(
        content="Episodic event record.",
        source_agent="worker",
        memory_type="episodic",
    ))
    store.add_memory(MemoryItem(
        content="Semantic general knowledge fact about biology.",
        source_agent="worker",
        memory_type="semantic",
    ))
    view = get_planner_view(store)
    assert all(m.memory_type == "semantic" for m in view)
    assert len(view) == 1


def test_worker_gets_all_active(store):
    store.add_memory(MemoryItem(
        content="Episodic event.",
        source_agent="worker",
        memory_type="episodic",
    ))
    store.add_memory(MemoryItem(
        content="Semantic fact.",
        source_agent="worker",
        memory_type="semantic",
    ))
    view = get_worker_view(store)
    assert len(view) == 2


def test_evaluator_filters_low_confidence(store):
    store.add_memory(MemoryItem(
        content="High confidence semantic fact.",
        source_agent="worker",
        memory_type="semantic",
        confidence_score=0.9,
    ))
    store.add_memory(MemoryItem(
        content="Low confidence semantic fact.",
        source_agent="worker",
        memory_type="semantic",
        confidence_score=0.5,
    ))
    view = get_evaluator_view(store)
    assert len(view) == 1
    assert view[0].confidence_score >= 0.8


def test_quarantined_memory_excluded_from_views(store):
    store.add_memory(MemoryItem(
        content="Quarantined semantic memory.",
        source_agent="worker",
        memory_type="semantic",
        status="quarantined",
    ))
    assert get_planner_view(store) == []
    assert get_evaluator_view(store) == []
    # Worker view also uses get_memory_by_type which filters status='active'
    assert get_worker_view(store) == []


# ---------------------------------------------------------------------------
# Causal Traceback & Repair
# ---------------------------------------------------------------------------

def test_full_repair_loop_quarantines_incorrect_memory(store):
    """
    Full closed-loop test:
    1. Seed an incorrect memory
    2. Link it to a failed decision
    3. Run repair with a mock LLM that returns 'incorrect'
    4. Verify memory is quarantined
    """
    bad = MemoryItem(
        content="Photosynthesis occurs only at night.",
        source_agent="worker",
        memory_type="semantic",
        confidence_score=0.9,
    )
    store.add_memory(bad)

    failed_decision_id = "evaluator_fail001"
    store.link_memory_to_decision(bad.id, failed_decision_id)

    verifier = MemoryVerifier(_AlwaysIncorrectLLM())
    repaired = repair_memories(store, failed_decision_id, verifier)

    assert len(repaired) == 1
    updated = store.get_memory(bad.id)
    assert updated.status == "quarantined"


def test_repair_downranks_uncertain_low_confidence_memory(store):
    uncertain = MemoryItem(
        content="Photosynthesis might work under certain conditions.",
        source_agent="worker",
        memory_type="semantic",
        confidence_score=0.4,
    )
    store.add_memory(uncertain)

    failed_decision_id = "evaluator_fail002"
    store.link_memory_to_decision(uncertain.id, failed_decision_id)

    verifier = MemoryVerifier(_AlwaysUncertainLLM())
    repair_memories(store, failed_decision_id, verifier)

    updated = store.get_memory(uncertain.id)
    assert updated.confidence_score < 0.4  # downranked
    assert updated.status == "active"  # not quarantined


def test_correct_memory_not_repaired(store):
    good = MemoryItem(
        content="Photosynthesis converts CO2 into glucose.",
        source_agent="worker",
        memory_type="semantic",
        confidence_score=0.9,
    )
    store.add_memory(good)

    failed_decision_id = "evaluator_fail003"
    store.link_memory_to_decision(good.id, failed_decision_id)

    verifier = MemoryVerifier(_AlwaysCorrectLLM())
    repair_memories(store, failed_decision_id, verifier)

    updated = store.get_memory(good.id)
    assert updated.status == "active"
    assert updated.confidence_score == 0.9  # unchanged


def test_trace_does_not_find_unlinked_memories(store):
    unrelated = MemoryItem(
        content="An unrelated memory with no decision link.",
        source_agent="worker",
        memory_type="episodic",
    )
    store.add_memory(unrelated)

    suspects = trace_suspect_memories(store, "nonexistent_decision_id")
    assert suspects == []


# ---------------------------------------------------------------------------
# Provenance Engine
# ---------------------------------------------------------------------------

def test_provenance_explain_returns_dict(store):
    mem = MemoryItem(
        content="General knowledge fact used in multiple decisions.",
        source_agent="worker",
        memory_type="semantic",
        confidence_score=0.85,
    )
    store.add_memory(mem)

    engine = ProvenanceEngine(store)
    report = engine.explain_memory(mem.id)

    assert isinstance(report, dict)
    assert report["memory_id"] == str(mem.id)
    assert report["created_by"] == "worker"
    assert "reliability" in report
    assert isinstance(report["reliability"], float)


def test_provenance_explain_nonexistent_returns_empty(store):
    import uuid
    engine = ProvenanceEngine(store)
    result = engine.explain_memory(uuid.uuid4())
    assert result == {}


def test_provenance_trace_failure_finds_linked_memory(store):
    mem = MemoryItem(
        content="Memory that caused a task failure.",
        source_agent="worker",
        memory_type="semantic",
    )
    store.add_memory(mem)
    store.link_memory_to_task(mem.id, "task-failure-001")

    engine = ProvenanceEngine(store)
    suspects = engine.trace_failure("task-failure-001")

    assert len(suspects) == 1
    assert suspects[0]["memory_id"] == str(mem.id)
    assert suspects[0]["created_by"] == "worker"


def test_provenance_trace_failure_empty_for_unknown_task(store):
    engine = ProvenanceEngine(store)
    suspects = engine.trace_failure("unknown-task-999")
    assert suspects == []


# ---------------------------------------------------------------------------
# Repair History Logging
# ---------------------------------------------------------------------------

def test_repair_event_logged_to_history(store):
    mem = MemoryItem(
        content="Memory to be repaired and logged.",
        source_agent="worker",
        memory_type="semantic",
        confidence_score=0.4,
    )
    store.add_memory(mem)
    failed = "evaluator_fail999"
    store.link_memory_to_decision(mem.id, failed)

    verifier = MemoryVerifier(_AlwaysUncertainLLM())
    repair_memories(store, failed, verifier)

    updated = store.get_memory(mem.id)
    assert len(updated.repair_history) == 1
    assert "Downranked" in updated.repair_history[0]
