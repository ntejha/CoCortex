"""
tests/test_fixes.py
===================
Pytest suite verifying all fixes applied to CoCortex.

Each test class maps to one fix. Tests use in-memory SQLite and mock LLMs —
no API key or network required.

Run with:
    python -m pytest tests/test_fixes.py -v
"""

import json
import threading
import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from memory.store import MemoryStore
from memory.schemas import MemoryItem
from memory.scoring import compute_reliability, ScoringConfig
from memory.views import get_planner_view, get_worker_view, get_evaluator_view
from memory.repair import (
    decide_repair_action,
    trace_suspect_memories,
    repair_memories,
    rehabilitate_memory,
    check_and_rehabilitate,
    repair_on_success,
)
from consensus.schemas import MemoryProposal, Vote
from consensus.voters import planner_voter, worker_voter, rule_based_voter
from consensus.engine import run_consensus


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_store() -> MemoryStore:
    return MemoryStore(":memory:")


def make_memory(
    content="Photosynthesis converts CO2 and water into glucose using sunlight.",
    memory_type="semantic",
    source_agent="worker",
    confidence=0.8,
    status="active",
    lifecycle_state="semantic",
    usage_count=0,
    failure_count=0,
) -> MemoryItem:
    return MemoryItem(
        content=content,
        memory_type=memory_type,
        source_agent=source_agent,
        confidence_score=confidence,
        status=status,
        lifecycle_state=lifecycle_state,
        usage_count=usage_count,
        failure_count=failure_count,
    )


def make_proposal(content: str, source_agent="worker") -> MemoryProposal:
    return MemoryProposal(
        content=content,
        source_agent=source_agent,
        suggested_type="episodic",
        context={},
    )


def mock_llm(response: str) -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = response
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# FIX — Vote.confidence Pydantic validation [ge=0.0, le=1.0]
# ─────────────────────────────────────────────────────────────────────────────

class TestVoteConfidenceValidation:
    """Vote.confidence now has ge=0.0, le=1.0 — out-of-range values are rejected."""

    def test_vote_accepts_valid_confidence(self):
        v = Vote(approve=True, confidence=0.85, risk=False, reason="ok")
        assert v.confidence == 0.85

    def test_vote_rejects_confidence_above_1(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Vote(approve=True, confidence=1.5, risk=False, reason="LLM returned 1.5")

    def test_vote_rejects_confidence_below_0(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Vote(approve=True, confidence=-0.3, risk=False, reason="LLM returned -0.3")

    def test_voters_clamp_before_constructing_vote(self):
        """If LLM returns out-of-range confidence, voter clamps it before Vote()."""
        # LLM returns confidence=1.5 — voter must clamp to 1.0 before constructing Vote
        llm = mock_llm('{"approve": true, "confidence": 1.5, "reason": "great"}')
        proposal = make_proposal(
            "Photosynthesis typically requires sunlight to produce glucose in plants."
        )
        # Should not raise ValidationError — _clamp() in voter handles this
        vote = planner_voter(proposal, llm=llm)
        assert 0.0 <= vote.confidence <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# FIX — LLM-Based Voters
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMBasedVoters:

    def test_planner_voter_uses_llm_when_provided(self):
        llm = mock_llm('{"approve": true, "confidence": 0.85, "reason": "General knowledge"}')
        proposal = make_proposal("Photosynthesis requires sunlight to produce glucose.")
        vote = planner_voter(proposal, llm=llm)
        assert vote.approve is True
        assert vote.confidence == 0.85
        assert llm.generate.called

    def test_planner_voter_falls_back_to_heuristic_when_no_llm(self):
        proposal = make_proposal("Photosynthesis typically requires sunlight.")
        vote = planner_voter(proposal, llm=None)
        assert vote.approve is True

    def test_planner_voter_rejects_short_content_before_llm_call(self):
        llm = mock_llm('{"approve": true, "confidence": 0.9, "reason": "ok"}')
        proposal = make_proposal("Too short")
        vote = planner_voter(proposal, llm=llm)
        assert vote.approve is False
        assert not llm.generate.called

    def test_planner_voter_rejects_when_llm_says_no(self):
        llm = mock_llm('{"approve": false, "confidence": 0.2, "reason": "Task-specific"}')
        proposal = make_proposal("I executed step 1 of the task and it worked fine.")
        vote = planner_voter(proposal, llm=llm)
        assert vote.approve is False
        assert vote.confidence == 0.2

    def test_worker_voter_uses_llm_when_provided(self):
        llm = mock_llm('{"approve": true, "confidence": 0.80, "reason": "Concrete result"}')
        proposal = make_proposal("The API returned a 200 status code with JSON payload.")
        vote = worker_voter(proposal, llm=llm)
        assert vote.approve is True
        assert llm.generate.called

    def test_worker_voter_rejects_questions_before_llm(self):
        llm = mock_llm('{"approve": true, "confidence": 0.9, "reason": "ok"}')
        proposal = make_proposal("What is the capital of France?")
        vote = worker_voter(proposal, llm=llm)
        assert vote.approve is False
        assert not llm.generate.called

    def test_worker_voter_falls_back_to_heuristic_when_no_llm(self):
        proposal = make_proposal("The function returned the correct output value.")
        vote = worker_voter(proposal, llm=None)
        assert vote.approve is True

    def test_rule_based_voter_never_calls_llm(self):
        llm = mock_llm("should not be called")
        proposal = make_proposal("This is a safe factual statement about biology.")
        vote = rule_based_voter(proposal, llm=llm)
        assert not llm.generate.called
        assert vote.approve is True
        assert vote.risk is False

    def test_rule_based_voter_catches_unsafe_content(self):
        proposal = make_proposal("How to hack into a system using an exploit.")
        vote = rule_based_voter(proposal)
        assert vote.approve is False
        assert vote.risk is True

    def test_planner_voter_handles_malformed_llm_json_gracefully(self):
        llm = mock_llm("I think this is good content, approve it!")
        proposal = make_proposal("Photosynthesis typically requires sunlight to produce energy.")
        vote = planner_voter(proposal, llm=llm)
        assert isinstance(vote.approve, bool)
        assert 0.0 <= vote.confidence <= 1.0

    def test_planner_voter_handles_llm_unavailable(self):
        from core.llm_client import LLM_UNAVAILABLE
        llm = mock_llm(LLM_UNAVAILABLE)
        proposal = make_proposal("Photosynthesis typically requires sunlight to produce glucose in plants.")
        vote = planner_voter(proposal, llm=llm)
        assert isinstance(vote.approve, bool)
        assert 0.0 <= vote.confidence <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# FIX — Attribution-Guided Causal Tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestAttributionGuidedTracking:

    def test_only_attributed_memories_are_linked(self):
        from agents.planner import PlannerAgent
        store = make_store()
        mem_used = make_memory(content="Photosynthesis requires sunlight to work.")
        mem_unused = make_memory(content="The mitochondria is the powerhouse of the cell.")
        store.add_memory(mem_used)
        store.add_memory(mem_unused)

        llm = MagicMock()
        llm.generate.side_effect = [
            "Step 1: Use sunlight. Step 2: Produce glucose.",
            "[0]",
        ]
        agent = PlannerAgent(llm, store)
        _, decision_id = agent.plan("Explain photosynthesis")

        assert decision_id in store.get_memory(mem_used.id).influenced_decisions
        assert decision_id not in store.get_memory(mem_unused.id).influenced_decisions

    def test_no_memories_linked_when_llm_attributes_none(self):
        from agents.worker import WorkerAgent
        store = make_store()
        mem = make_memory(content="Use a hammer to drive nails into wood.")
        store.add_memory(mem)

        llm = MagicMock()
        llm.generate.side_effect = ["I executed the plan independently.", "[]"]
        agent = WorkerAgent(llm, store)
        _, decision_id = agent.execute("Build a table")

        assert decision_id not in store.get_memory(mem.id).influenced_decisions

    def test_attribution_parse_failure_links_nothing(self):
        from agents.planner import PlannerAgent
        store = make_store()
        mem = make_memory(content="Photosynthesis requires sunlight to work properly.")
        store.add_memory(mem)

        llm = MagicMock()
        llm.generate.side_effect = ["Here is my plan.", "sorry I cannot provide that"]
        agent = PlannerAgent(llm, store)
        _, decision_id = agent.plan("Explain photosynthesis")

        assert decision_id not in store.get_memory(mem.id).influenced_decisions


# ─────────────────────────────────────────────────────────────────────────────
# FIX — Rehabilitation Mechanism
# ─────────────────────────────────────────────────────────────────────────────

class TestRehabilitation:

    def test_quarantined_memory_can_be_rehabilitated(self):
        store = make_store()
        mem = make_memory(status="quarantined", confidence=0.7, failure_count=1)
        store.add_memory(mem)

        result = rehabilitate_memory(store, mem.id)
        assert result is True

        restored = store.get_memory(mem.id)
        assert restored.status == "active"
        assert restored.confidence_score < 0.7
        assert len(restored.repair_history) > 0

    def test_active_memory_not_rehabilitated(self):
        store = make_store()
        mem = make_memory(status="active")
        store.add_memory(mem)
        assert rehabilitate_memory(store, mem.id) is False

    def test_check_and_rehabilitate_restores_borderline_quarantined(self):
        store = make_store()
        borderline = make_memory(
            content="A safe fact about science.", status="quarantined",
            failure_count=1, confidence=0.6
        )
        clearly_bad = make_memory(
            content="A dangerous incorrect statement.", status="quarantined",
            failure_count=5, confidence=0.1
        )
        store.add_memory(borderline)
        store.add_memory(clearly_bad)

        rehabilitated = check_and_rehabilitate(store, "evaluator_success_abc123")

        assert any(m.id == borderline.id for m in rehabilitated)
        assert not any(m.id == clearly_bad.id for m in rehabilitated)

    def test_rehabilitation_logs_event_in_repair_history(self):
        store = make_store()
        mem = make_memory(status="quarantined", failure_count=1, confidence=0.5)
        store.add_memory(mem)
        rehabilitate_memory(store, mem.id)
        restored = store.get_memory(mem.id)
        assert any("Rehabilitated" in event for event in restored.repair_history)

    def test_repair_on_success_triggers_rehabilitation(self):
        """repair_on_success() (called by evaluator on PASS) rehabilitates borderline memories."""
        store = make_store()
        borderline = make_memory(
            content="A borderline fact.", status="quarantined", failure_count=1, confidence=0.6
        )
        store.add_memory(borderline)

        rehabilitated = repair_on_success(store, "evaluator_pass_xyz")
        assert any(m.id == borderline.id for m in rehabilitated)
        assert store.get_memory(borderline.id).status == "active"


# ─────────────────────────────────────────────────────────────────────────────
# FIX — Consensus Engine Validation + Weighted Confidence
# ─────────────────────────────────────────────────────────────────────────────

class TestConsensusEngine:

    def test_run_consensus_raises_with_fewer_than_3_voters(self):
        proposal = make_proposal("Some content about a topic.")
        votes = [Vote(approve=True, confidence=0.8, risk=False, reason="ok")]
        with pytest.raises(ValueError, match="3 voters"):
            run_consensus(votes, proposal)

    def test_two_approvals_accepts(self):
        proposal = make_proposal("Photosynthesis converts CO2 into glucose using sunlight.")
        votes = [
            Vote(approve=True, confidence=0.8, risk=False, reason="ok"),
            Vote(approve=True, confidence=0.7, risk=False, reason="ok"),
            Vote(approve=False, confidence=0.3, risk=False, reason="no"),
        ]
        decision, _, confidence = run_consensus(votes, proposal)
        assert decision == "accept"
        # Weighted average of ALL voters: (0.8 + 0.7 + 0.3) / 3 ≈ 0.6
        assert abs(confidence - round((0.8 + 0.7 + 0.3) / 3, 3)) < 0.001

    def test_risk_flag_quarantines(self):
        proposal = make_proposal("Some content.")
        votes = [
            Vote(approve=True, confidence=0.9, risk=False, reason="ok"),
            Vote(approve=True, confidence=0.8, risk=False, reason="ok"),
            Vote(approve=False, confidence=0.1, risk=True, reason="risky"),
        ]
        decision, _, _ = run_consensus(votes, proposal)
        assert decision == "quarantine"

    def test_no_approvals_rejects(self):
        proposal = make_proposal("Some content.")
        votes = [
            Vote(approve=False, confidence=0.2, risk=False, reason="no"),
            Vote(approve=False, confidence=0.3, risk=False, reason="no"),
            Vote(approve=False, confidence=0.1, risk=False, reason="no"),
        ]
        decision, _, _ = run_consensus(votes, proposal)
        assert decision == "reject"

    def test_single_approval_rejects(self):
        proposal = make_proposal("Some content.")
        votes = [
            Vote(approve=True, confidence=0.9, risk=False, reason="yes"),
            Vote(approve=False, confidence=0.2, risk=False, reason="no"),
            Vote(approve=False, confidence=0.1, risk=False, reason="no"),
        ]
        decision, _, _ = run_consensus(votes, proposal)
        assert decision == "reject"

    def test_dissenting_vote_reduces_accepted_confidence(self):
        """Weighted average means strong dissent lowers accepted confidence."""
        proposal = make_proposal("Some content about a topic for testing confidence.")
        votes_strong_dissent = [
            Vote(approve=True, confidence=0.9, risk=False, reason="yes"),
            Vote(approve=True, confidence=0.9, risk=False, reason="yes"),
            Vote(approve=False, confidence=0.05, risk=False, reason="strongly no"),
        ]
        votes_weak_dissent = [
            Vote(approve=True, confidence=0.9, risk=False, reason="yes"),
            Vote(approve=True, confidence=0.9, risk=False, reason="yes"),
            Vote(approve=False, confidence=0.5, risk=False, reason="mildly no"),
        ]
        _, _, conf_strong = run_consensus(votes_strong_dissent, proposal)
        _, _, conf_weak = run_consensus(votes_weak_dissent, proposal)
        assert conf_strong < conf_weak


# ─────────────────────────────────────────────────────────────────────────────
# FIX — store.update_memory SQL Injection Whitelist
# ─────────────────────────────────────────────────────────────────────────────

class TestStoreSQLWhitelist:

    def test_update_memory_rejects_unknown_field(self):
        store = make_store()
        mem = make_memory()
        store.add_memory(mem)
        with pytest.raises(ValueError, match="unknown field"):
            store.update_memory(mem.id, {"DROP TABLE memories": "injected"})

    def test_update_memory_accepts_valid_field(self):
        store = make_store()
        mem = make_memory()
        store.add_memory(mem)
        store.update_memory(mem.id, {"confidence": 0.5})
        updated = store.get_memory(mem.id)
        assert updated.confidence_score == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# FIX — Lifecycle Quarantine Guard
# ─────────────────────────────────────────────────────────────────────────────

class TestLifecycleQuarantineGuard:

    def test_quarantined_memory_not_promoted_to_semantic(self):
        """A quarantined memory with high reliability must NOT become 'semantic'."""
        from memory.lifecycle import update_lifecycle
        mem = make_memory(
            confidence=0.95, usage_count=10, failure_count=0,
            status="quarantined", lifecycle_state="episodic"
        )
        new_state = update_lifecycle(mem)
        # Must stay at current lifecycle_state, not be promoted
        assert new_state == "episodic"
        assert new_state != "semantic"

    def test_active_memory_with_high_reliability_promoted(self):
        """Active memories with high reliability are still promoted normally."""
        from memory.lifecycle import update_lifecycle
        mem = make_memory(
            confidence=0.95, usage_count=10, failure_count=0,
            status="active", lifecycle_state="episodic"
        )
        new_state = update_lifecycle(mem)
        assert new_state == "semantic"


# ─────────────────────────────────────────────────────────────────────────────
# FIX — Verification LLM_UNAVAILABLE Handling
# ─────────────────────────────────────────────────────────────────────────────

class TestVerificationLLMUnavailable:

    def test_verifier_returns_uncertain_on_llm_unavailable(self):
        from memory.verification import MemoryVerifier
        from core.llm_client import LLM_UNAVAILABLE

        llm = mock_llm(LLM_UNAVAILABLE)
        verifier = MemoryVerifier(llm)
        result = verifier.verify("Photosynthesis occurs only at night.")
        assert result == "uncertain"

    def test_verifier_uncertain_does_not_trigger_repair_on_high_confidence(self):
        """uncertain + high confidence → decide_repair_action returns 'none'."""
        action = decide_repair_action("uncertain", confidence=0.9, failure_count=0)
        assert action == "none"

    def test_verifier_returns_incorrect_correctly(self):
        from memory.verification import MemoryVerifier
        llm = mock_llm("incorrect")
        verifier = MemoryVerifier(llm)
        assert verifier.verify("Photosynthesis occurs only at night.") == "incorrect"

    def test_verifier_not_correct_returns_incorrect(self):
        from memory.verification import MemoryVerifier
        llm = mock_llm("not correct")
        verifier = MemoryVerifier(llm)
        assert verifier.verify("some statement") == "incorrect"


# ─────────────────────────────────────────────────────────────────────────────
# FIX — MemoryManager Auto-Detects Semantic Type
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryManagerTypeInference:

    def test_general_knowledge_inferred_as_semantic(self):
        from agents.memory_manager import _infer_memory_type
        assert _infer_memory_type("Photosynthesis typically requires sunlight.") == "semantic"
        assert _infer_memory_type("Water is always composed of hydrogen and oxygen.") == "semantic"

    def test_execution_trace_inferred_as_episodic(self):
        from agents.memory_manager import _infer_memory_type
        assert _infer_memory_type("I executed step 3 and the process completed.") == "episodic"

    def test_manager_stores_semantic_memory_as_semantic(self):
        from agents.memory_manager import MemoryManagerAgent
        store = make_store()
        # Mock LLM that approves everything
        llm = mock_llm('{"approve": true, "confidence": 0.85, "reason": "General knowledge"}')
        manager = MemoryManagerAgent(llm=llm, store=store)

        decision, memory = manager.process_output(
            content="Photosynthesis typically requires sunlight and CO2 to produce glucose.",
            source_agent="worker",
            context={"domain": "biology"},
        )
        assert decision == "ACCEPTED"
        assert memory.memory_type == "semantic"

    def test_manager_injectable_store(self):
        """MemoryManagerAgent uses the injected store, not its own."""
        from agents.memory_manager import MemoryManagerAgent
        store = make_store()
        manager = MemoryManagerAgent(store=store)
        assert manager.store is store


# ─────────────────────────────────────────────────────────────────────────────
# FIX — store.delete_by_session + CoCortexMemory.clear()
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionDeletion:

    def test_delete_by_session_removes_correct_records(self):
        store = make_store()
        mem_a = MemoryItem(
            content="Memory for session A.", source_agent="worker",
            memory_type="episodic", task_ids=["session-A"]
        )
        mem_b = MemoryItem(
            content="Memory for session B.", source_agent="worker",
            memory_type="episodic", task_ids=["session-B"]
        )
        store.add_memory(mem_a)
        store.add_memory(mem_b)

        store.delete_by_session("session-A")

        assert store.get_memory(mem_a.id) is None
        assert store.get_memory(mem_b.id) is not None

    def test_delete_by_session_no_false_positives(self):
        """'session-1' must NOT delete records belonging to 'session-10'."""
        store = make_store()
        mem1 = MemoryItem(
            content="Memory for session 1.", source_agent="worker",
            memory_type="episodic", task_ids=["session-1"]
        )
        mem10 = MemoryItem(
            content="Memory for session 10.", source_agent="worker",
            memory_type="episodic", task_ids=["session-10"]
        )
        store.add_memory(mem1)
        store.add_memory(mem10)

        store.delete_by_session("session-1")

        assert store.get_memory(mem1.id) is None
        assert store.get_memory(mem10.id) is not None  # must NOT be deleted

    def test_cocortex_memory_clear_actually_deletes(self):
        from engine.memory_engine import MemoryEngine
        from memory.cocortex_memory import CoCortexMemory
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            engine = MemoryEngine(db_path=db_path)
            mem = CoCortexMemory(engine=engine, session_id="test-session")
            engine.save_turn("test-session", "hello", "hi")

            assert len(engine.load("test-session")) == 1

            mem.clear()

            assert len(engine.load("test-session")) == 0
        finally:
            os.unlink(db_path)


# ─────────────────────────────────────────────────────────────────────────────
# FIX — ScoringConfig Named Constants
# ─────────────────────────────────────────────────────────────────────────────

class TestScoringConfig:

    def test_default_config_produces_expected_score(self):
        mem = make_memory(confidence=0.8, usage_count=5, failure_count=0)
        score = compute_reliability(mem)
        assert score >= 0.85

    def test_failure_penalty_dominates_usage_reward(self):
        mem_used = make_memory(confidence=0.5, usage_count=1, failure_count=0)
        mem_failed = make_memory(confidence=0.5, usage_count=0, failure_count=1)
        assert compute_reliability(mem_used) > compute_reliability(mem_failed)

    def test_custom_config_overrides_defaults(self):
        mem = make_memory(confidence=0.5, usage_count=10, failure_count=0)
        default_score = compute_reliability(mem)
        aggressive_config = ScoringConfig(usage_reward=0.05, usage_cap=0.5)
        aggressive_score = compute_reliability(mem, config=aggressive_config)
        assert aggressive_score > default_score

    def test_score_clamped_to_0_1(self):
        high_failure = make_memory(confidence=0.1, failure_count=100)
        high_usage = make_memory(confidence=1.0, usage_count=1000, failure_count=0)
        assert compute_reliability(high_failure) == 0.0
        assert compute_reliability(high_usage) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# FIX — Views Filter by Lifecycle + Relevance Ranking
# ─────────────────────────────────────────────────────────────────────────────

class TestViews:

    def test_planner_view_excludes_stale_memories(self):
        store = make_store()
        healthy = make_memory(content="Photosynthesis requires sunlight.", lifecycle_state="semantic")
        stale = make_memory(content="Old knowledge about photosynthesis.", lifecycle_state="stale")
        store.add_memory(healthy)
        store.add_memory(stale)
        view = get_planner_view(store)
        ids = [m.id for m in view]
        assert healthy.id in ids
        assert stale.id not in ids

    def test_planner_view_excludes_deprecated_memories(self):
        store = make_store()
        deprecated = make_memory(content="Deprecated knowledge.", lifecycle_state="deprecated")
        store.add_memory(deprecated)
        view = get_planner_view(store)
        assert deprecated.id not in [m.id for m in view]

    def test_worker_view_excludes_archived_memories(self):
        store = make_store()
        archived = make_memory(
            content="Old procedure.", memory_type="episodic", lifecycle_state="archived"
        )
        active = make_memory(
            content="Current procedure.", memory_type="episodic", lifecycle_state="episodic"
        )
        store.add_memory(archived)
        store.add_memory(active)
        view = get_worker_view(store)
        ids = [m.id for m in view]
        assert active.id in ids
        assert archived.id not in ids

    def test_planner_view_with_query_returns_relevant_memories(self):
        store = make_store()
        relevant = make_memory(content="Photosynthesis requires sunlight and chlorophyll.")
        irrelevant = make_memory(content="The Roman Empire collapsed in 476 AD.")
        store.add_memory(relevant)
        store.add_memory(irrelevant)
        view = get_planner_view(store, query="photosynthesis sunlight plants")
        assert view[0].id == relevant.id

    def test_view_limits_results_to_top_n(self):
        store = make_store()
        for i in range(20):
            store.add_memory(make_memory(content=f"Fact {i} about biology and science."))
        view = get_planner_view(store, top_n=5)
        assert len(view) <= 5

    def test_evaluator_view_only_returns_high_confidence(self):
        store = make_store()
        high_conf = make_memory(content="Well established fact.", confidence=0.9)
        low_conf = make_memory(content="Uncertain claim.", confidence=0.5)
        store.add_memory(high_conf)
        store.add_memory(low_conf)
        view = get_evaluator_view(store)
        ids = [m.id for m in view]
        assert high_conf.id in ids
        assert low_conf.id not in ids

    def test_planner_view_truncates_content_to_300_chars(self):
        store = make_store()
        mem = make_memory(content="A" * 500)
        store.add_memory(mem)
        view = get_planner_view(store)
        assert all(len(m.content) <= 300 for m in view)


# ─────────────────────────────────────────────────────────────────────────────
# FIX — Thread Safety
# ─────────────────────────────────────────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_writes_do_not_corrupt_store(self):
        store = make_store()
        errors = []

        def write_memory(i):
            try:
                mem = make_memory(content=f"Concurrent fact {i} about science and nature.")
                store.add_memory(mem)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_memory, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert len(store.get_all_active_memories()) == 20

    def test_store_has_write_lock_attribute(self):
        store = make_store()
        assert hasattr(store, "_write_lock")
        assert isinstance(store._write_lock, type(threading.Lock()))


# ─────────────────────────────────────────────────────────────────────────────
# FIX — LLM Retry + Graceful Degradation
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMRetry:

    def test_llm_client_returns_sentinel_after_max_retries(self):
        from core.llm_client import LLMClient, LLM_UNAVAILABLE
        with patch("core.llm_client.Groq") as MockGroq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("Rate limited")
            MockGroq.return_value = mock_client
            with patch.dict("os.environ", {"GROQ_API_KEY": "fake-key"}):
                with patch("core.llm_client.time.sleep"):
                    llm = LLMClient()
                    result = llm.generate("test prompt")
        assert result == LLM_UNAVAILABLE

    def test_llm_client_retries_before_giving_up(self):
        from core.llm_client import LLMClient, MAX_RETRIES
        with patch("core.llm_client.Groq") as MockGroq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("Timeout")
            MockGroq.return_value = mock_client
            with patch.dict("os.environ", {"GROQ_API_KEY": "fake-key"}):
                with patch("core.llm_client.time.sleep"):
                    llm = LLMClient()
                    llm.generate("test prompt")
        assert mock_client.chat.completions.create.call_count == MAX_RETRIES

    def test_llm_client_succeeds_on_retry(self):
        from core.llm_client import LLMClient
        success_response = MagicMock()
        success_response.choices[0].message.content = "correct answer"
        with patch("core.llm_client.Groq") as MockGroq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [
                Exception("First attempt fails"),
                success_response,
            ]
            MockGroq.return_value = mock_client
            with patch.dict("os.environ", {"GROQ_API_KEY": "fake-key"}):
                with patch("core.llm_client.time.sleep"):
                    llm = LLMClient()
                    result = llm.generate("test prompt")
        assert result == "correct answer"

    def test_permanent_error_not_retried(self):
        """A 401 auth error should not be retried — it will never succeed."""
        from core.llm_client import LLMClient, LLM_UNAVAILABLE
        permanent_error = Exception("Unauthorized")
        permanent_error.status_code = 401

        with patch("core.llm_client.Groq") as MockGroq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = permanent_error
            MockGroq.return_value = mock_client
            with patch.dict("os.environ", {"GROQ_API_KEY": "fake-key"}):
                with patch("core.llm_client.time.sleep") as mock_sleep:
                    llm = LLMClient()
                    result = llm.generate("test prompt")

        # Should return sentinel immediately — no sleep/retry
        assert result == LLM_UNAVAILABLE
        mock_sleep.assert_not_called()
        assert mock_client.chat.completions.create.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION — Consensus Pipeline End-to-End
# ─────────────────────────────────────────────────────────────────────────────

class TestConsensusIntegration:

    def test_good_memory_accepted_by_consensus(self):
        llm = mock_llm('{"approve": true, "confidence": 0.85, "reason": "General knowledge"}')
        proposal = make_proposal(
            "Photosynthesis requires sunlight, CO2, and water to produce glucose and oxygen."
        )
        votes = [
            planner_voter(proposal, llm=llm),
            worker_voter(proposal, llm=llm),
            rule_based_voter(proposal),
        ]
        decision, _, confidence = run_consensus(votes, proposal)
        assert decision == "accept"
        assert confidence > 0.0

    def test_unsafe_memory_quarantined_by_safety_voter(self):
        llm = mock_llm('{"approve": true, "confidence": 0.9, "reason": "Looks fine"}')
        proposal = make_proposal(
            "You can hack into systems by exploiting buffer overflow vulnerabilities."
        )
        votes = [
            planner_voter(proposal, llm=llm),
            worker_voter(proposal, llm=llm),
            rule_based_voter(proposal),
        ]
        decision, _, _ = run_consensus(votes, proposal)
        assert decision == "quarantine"

    def test_memory_rejected_when_majority_disapprove(self):
        proposal = make_proposal("Some debatable content about a topic.")
        votes = [
            Vote(approve=False, confidence=0.2, risk=False, reason="No"),
            Vote(approve=False, confidence=0.3, risk=False, reason="No"),
            Vote(approve=True, confidence=0.7, risk=False, reason="Yes"),
        ]
        decision, _, _ = run_consensus(votes, proposal)
        assert decision == "reject"