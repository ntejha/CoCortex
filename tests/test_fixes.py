"""
tests/test_fixes.py
===================
Pytest suite verifying all 8 fixes applied to CoCortex.

Each test class maps to one fix. Tests use in-memory SQLite and
mock LLMs — no API key or network required.

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
)
from consensus.schemas import MemoryProposal, Vote
from consensus.voters import planner_voter, worker_voter, rule_based_voter
from consensus.engine import run_consensus


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_store() -> MemoryStore:
    """Fresh in-memory SQLite store for each test."""
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
    """Returns a mock LLMClient whose generate() always returns `response`."""
    llm = MagicMock()
    llm.generate.return_value = response
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 & 2 — LLM-Based Voters (planner_voter, worker_voter)
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMBasedVoters:
    """
    Fix 1: voters now call the LLM for semantic reasoning instead of keyword matching.
    Fix 2: rule_based_voter remains deterministic (no LLM) for safety.
    """

    def test_planner_voter_uses_llm_when_provided(self):
        """Planner voter sends proposal to LLM and parses JSON response."""
        llm = mock_llm('{"approve": true, "confidence": 0.85, "reason": "General knowledge"}')
        proposal = make_proposal("Photosynthesis requires sunlight to produce glucose.")
        vote = planner_voter(proposal, llm=llm)

        assert vote.approve is True
        assert vote.confidence == 0.85
        assert llm.generate.called

    def test_planner_voter_falls_back_to_heuristic_when_no_llm(self):
        """Planner voter falls back to heuristics when llm=None (offline/test mode)."""
        proposal = make_proposal("Photosynthesis typically requires sunlight.")
        vote = planner_voter(proposal, llm=None)
        # "typically" is a reusable signal in heuristic
        assert vote.approve is True

    def test_planner_voter_rejects_short_content_before_llm_call(self):
        """Pre-filter: content < 40 chars is rejected without calling the LLM."""
        llm = mock_llm('{"approve": true, "confidence": 0.9, "reason": "ok"}')
        proposal = make_proposal("Too short")
        vote = planner_voter(proposal, llm=llm)

        assert vote.approve is False
        assert not llm.generate.called  # LLM should NOT be called for pre-filtered content

    def test_planner_voter_rejects_when_llm_says_no(self):
        """Planner voter respects LLM rejection."""
        llm = mock_llm('{"approve": false, "confidence": 0.2, "reason": "Task-specific trace"}')
        proposal = make_proposal("I executed step 1 of the task and it worked.")
        vote = planner_voter(proposal, llm=llm)

        assert vote.approve is False
        assert vote.confidence == 0.2

    def test_worker_voter_uses_llm_when_provided(self):
        """Worker voter sends proposal to LLM."""
        llm = mock_llm('{"approve": true, "confidence": 0.80, "reason": "Concrete result"}')
        proposal = make_proposal("The API returned a 200 status code with JSON payload.")
        vote = worker_voter(proposal, llm=llm)

        assert vote.approve is True
        assert vote.confidence == 0.80
        assert llm.generate.called

    def test_worker_voter_rejects_questions_before_llm(self):
        """Pre-filter: questions are rejected without calling LLM."""
        llm = mock_llm('{"approve": true, "confidence": 0.9, "reason": "ok"}')
        proposal = make_proposal("What is the capital of France?")
        vote = worker_voter(proposal, llm=llm)

        assert vote.approve is False
        assert not llm.generate.called

    def test_worker_voter_falls_back_to_heuristic_when_no_llm(self):
        """Worker voter heuristic fallback works offline."""
        proposal = make_proposal("The function returned the correct output value.")
        vote = worker_voter(proposal, llm=None)
        assert vote.approve is True

    def test_rule_based_voter_never_calls_llm(self):
        """Safety voter is deterministic — never calls LLM even when provided."""
        llm = mock_llm("should not be called")
        proposal = make_proposal("This is a safe factual statement about biology.")
        vote = rule_based_voter(proposal, llm=llm)

        assert not llm.generate.called
        assert vote.approve is True
        assert vote.risk is False

    def test_rule_based_voter_catches_unsafe_content(self):
        """Safety voter catches hardcoded unsafe keywords."""
        proposal = make_proposal("How to hack into a system using an exploit.")
        vote = rule_based_voter(proposal)

        assert vote.approve is False
        assert vote.risk is True

    def test_planner_voter_handles_malformed_llm_json_gracefully(self):
        """If LLM returns garbage JSON, planner voter falls back to heuristic."""
        llm = mock_llm("I think this is good content, approve it!")  # not JSON
        proposal = make_proposal("Photosynthesis typically requires sunlight to produce energy.")
        vote = planner_voter(proposal, llm=llm)

        # Should not raise — falls back to heuristic
        assert isinstance(vote.approve, bool)
        assert 0.0 <= vote.confidence <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — Attribution-Guided Causal Tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestAttributionGuidedTracking:
    """
    Fix 2: Agents only link memories they report using, not all context memories.
    Tests that un-used memories in context are NOT linked to decisions.
    """

    def test_only_attributed_memories_are_linked(self):
        """
        If the LLM says it used memory index [0] only, only that memory
        gets linked to the decision — not memory at index [1].
        """
        from agents.planner import PlannerAgent

        store = make_store()
        mem_used = make_memory(content="Photosynthesis requires sunlight to work.")
        mem_unused = make_memory(content="The mitochondria is the powerhouse of the cell.")
        store.add_memory(mem_used)
        store.add_memory(mem_unused)

        # First call: plan output. Second call: attribution — LLM says it used [0] only.
        llm = MagicMock()
        llm.generate.side_effect = [
            "Step 1: Use sunlight. Step 2: Produce glucose.",  # plan output
            "[0]",  # attribution: only used memory index 0
        ]

        agent = PlannerAgent(llm, store)
        _, decision_id = agent.plan("Explain photosynthesis")

        mem_used_after = store.get_memory(mem_used.id)
        mem_unused_after = store.get_memory(mem_unused.id)

        assert decision_id in mem_used_after.influenced_decisions
        assert decision_id not in mem_unused_after.influenced_decisions

    def test_no_memories_linked_when_llm_attributes_none(self):
        """If LLM says it used no memories ([]), nothing is linked."""
        from agents.worker import WorkerAgent

        store = make_store()
        mem = make_memory(content="Use a hammer to drive nails into wood.")
        store.add_memory(mem)

        llm = MagicMock()
        llm.generate.side_effect = [
            "I executed the plan independently.",  # worker output
            "[]",  # attribution: used no memories
        ]

        agent = WorkerAgent(llm, store)
        _, decision_id = agent.execute("Build a table")

        mem_after = store.get_memory(mem.id)
        assert decision_id not in mem_after.influenced_decisions

    def test_attribution_parse_failure_links_nothing(self):
        """If attribution call returns garbage, no memories are linked (safe default)."""
        from agents.planner import PlannerAgent

        store = make_store()
        mem = make_memory(content="Photosynthesis requires sunlight to work properly.")
        store.add_memory(mem)

        llm = MagicMock()
        llm.generate.side_effect = [
            "Here is my plan.",
            "sorry I cannot provide that",  # malformed attribution
        ]

        agent = PlannerAgent(llm, store)
        _, decision_id = agent.plan("Explain photosynthesis")

        mem_after = store.get_memory(mem.id)
        # Safe default: nothing linked on parse failure
        assert decision_id not in mem_after.influenced_decisions


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 — Rehabilitation Mechanism
# ─────────────────────────────────────────────────────────────────────────────

class TestRehabilitation:
    """
    Fix 3: Quarantined memories can recover. The state machine is no longer
    one-directional. False positives from repair are reversible.
    """

    def test_quarantined_memory_can_be_rehabilitated(self):
        """rehabilitate_memory() restores a quarantined memory to active."""
        store = make_store()
        mem = make_memory(status="quarantined", confidence=0.7, failure_count=1)
        store.add_memory(mem)

        result = rehabilitate_memory(store, mem.id)
        assert result is True

        restored = store.get_memory(mem.id)
        assert restored.status == "active"
        assert restored.confidence_score < 0.7  # halved confidence
        assert len(restored.repair_history) > 0  # event logged

    def test_active_memory_not_rehabilitated(self):
        """rehabilitate_memory() is a no-op on already-active memories."""
        store = make_store()
        mem = make_memory(status="active")
        store.add_memory(mem)

        result = rehabilitate_memory(store, mem.id)
        assert result is False

    def test_check_and_rehabilitate_restores_borderline_quarantined(self):
        """
        After a successful decision, memories quarantined with only 1 failure
        that weren't involved in the success are candidates for rehabilitation.
        """
        store = make_store()

        # Memory with failure_count=1 — borderline, might have been wrongly blamed
        borderline = make_memory(content="A safe fact about science.", status="quarantined",
                                  failure_count=1, confidence=0.6)
        store.add_memory(borderline)

        # A clearly bad memory with many failures — should NOT be rehabilitated
        clearly_bad = make_memory(content="A dangerous incorrect statement.", status="quarantined",
                                   failure_count=5, confidence=0.1)
        store.add_memory(clearly_bad)

        successful_decision_id = "evaluator_success_abc123"
        rehabilitated = check_and_rehabilitate(store, successful_decision_id)

        assert any(m.id == borderline.id for m in rehabilitated)
        assert not any(m.id == clearly_bad.id for m in rehabilitated)

    def test_rehabilitation_logs_event_in_repair_history(self):
        """Rehabilitation is auditable — it logs an entry in repair_history."""
        store = make_store()
        mem = make_memory(status="quarantined", failure_count=1, confidence=0.5)
        store.add_memory(mem)

        rehabilitate_memory(store, mem.id)

        restored = store.get_memory(mem.id)
        assert any("Rehabilitated" in event for event in restored.repair_history)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 5 — ScoringConfig Named Constants
# ─────────────────────────────────────────────────────────────────────────────

class TestScoringConfig:
    """
    Fix 5: Scoring coefficients are named constants in ScoringConfig,
    not magic numbers. Scores are tunable without touching logic.
    """

    def test_default_config_produces_expected_score(self):
        """Default config: usage reward, failure penalty, and decay all apply."""
        mem = make_memory(confidence=0.8, usage_count=5, failure_count=0)
        score = compute_reliability(mem)
        # Base 0.8 + usage(5*0.02=0.1, capped 0.1) = 0.9 (ignoring small decay)
        assert score >= 0.85  # some decay expected but score should be high

    def test_failure_penalty_dominates_usage_reward(self):
        """One failure (0.15) costs more than one use (0.02) — by design."""
        mem_used = make_memory(confidence=0.5, usage_count=1, failure_count=0)
        mem_failed = make_memory(confidence=0.5, usage_count=0, failure_count=1)

        assert compute_reliability(mem_used) > compute_reliability(mem_failed)

    def test_custom_config_overrides_defaults(self):
        """ScoringConfig allows custom coefficients without modifying logic."""
        mem = make_memory(confidence=0.5, usage_count=10, failure_count=0)

        default_score = compute_reliability(mem)

        aggressive_config = ScoringConfig(usage_reward=0.05, usage_cap=0.5)
        aggressive_score = compute_reliability(mem, config=aggressive_config)

        assert aggressive_score > default_score

    def test_score_clamped_to_0_1(self):
        """Score never goes below 0 or above 1 regardless of inputs."""
        high_failure = make_memory(confidence=0.1, failure_count=100)
        high_usage = make_memory(confidence=1.0, usage_count=1000, failure_count=0)

        assert compute_reliability(high_failure) == 0.0
        assert compute_reliability(high_usage) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# FIX 6 — Views Filter by Lifecycle State + Relevance Ranking
# ─────────────────────────────────────────────────────────────────────────────

class TestViews:
    """
    Fix 6: Views exclude stale/deprecated/archived memories.
    Fix 7: Views support relevance-based filtering via query keyword overlap.
    """

    def test_planner_view_excludes_stale_memories(self):
        """Stale memories are not served to the planner."""
        store = make_store()
        healthy = make_memory(content="Photosynthesis requires sunlight to produce glucose.", lifecycle_state="semantic")
        stale = make_memory(content="Old knowledge about photosynthesis decay process.", lifecycle_state="stale")
        store.add_memory(healthy)
        store.add_memory(stale)

        view = get_planner_view(store)
        ids = [m.id for m in view]

        assert healthy.id in ids
        assert stale.id not in ids

    def test_planner_view_excludes_deprecated_memories(self):
        """Deprecated memories are not served to the planner."""
        store = make_store()
        deprecated = make_memory(content="This knowledge is deprecated and wrong.", lifecycle_state="deprecated")
        store.add_memory(deprecated)

        view = get_planner_view(store)
        assert deprecated.id not in [m.id for m in view]

    def test_worker_view_excludes_archived_memories(self):
        """Archived memories are not served to the worker."""
        store = make_store()
        archived = make_memory(content="Old procedure, archived.", memory_type="episodic", lifecycle_state="archived")
        active = make_memory(content="Current procedure for task execution.", memory_type="episodic", lifecycle_state="episodic")
        store.add_memory(archived)
        store.add_memory(active)

        view = get_worker_view(store)
        ids = [m.id for m in view]

        assert active.id in ids
        assert archived.id not in ids

    def test_planner_view_with_query_returns_relevant_memories(self):
        """When query is given, memories are ranked by keyword relevance."""
        store = make_store()
        relevant = make_memory(content="Photosynthesis requires sunlight and chlorophyll.")
        irrelevant = make_memory(content="The Roman Empire collapsed in 476 AD due to invasions.")
        store.add_memory(relevant)
        store.add_memory(irrelevant)

        view = get_planner_view(store, query="photosynthesis sunlight plants")

        # Relevant memory should appear first
        assert view[0].id == relevant.id

    def test_view_with_query_limits_results_to_top_n(self):
        """Views respect top_n limit to prevent context flooding."""
        store = make_store()
        for i in range(20):
            store.add_memory(make_memory(content=f"Fact number {i} about biology and science."))

        view = get_planner_view(store, top_n=5)
        assert len(view) <= 5

    def test_evaluator_view_only_returns_high_confidence(self):
        """Evaluator only sees memories with confidence >= 0.8."""
        store = make_store()
        high_conf = make_memory(content="Well-established biological fact confirmed many times.", confidence=0.9)
        low_conf = make_memory(content="Uncertain claim about a process, not fully verified yet.", confidence=0.5)
        store.add_memory(high_conf)
        store.add_memory(low_conf)

        view = get_evaluator_view(store)
        ids = [m.id for m in view]

        assert high_conf.id in ids
        assert low_conf.id not in ids

    def test_planner_view_truncates_content_to_300_chars(self):
        """Planner view truncates memory content to 300 characters."""
        store = make_store()
        long_content = "A" * 500
        mem = make_memory(content=long_content)
        store.add_memory(mem)

        view = get_planner_view(store)
        assert all(len(m.content) <= 300 for m in view)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 7 — Thread Safety (MemoryStore)
# ─────────────────────────────────────────────────────────────────────────────

class TestThreadSafety:
    """
    Fix 8: MemoryStore uses WAL mode + threading.Lock for concurrent write safety.
    """

    def test_concurrent_writes_do_not_corrupt_store(self):
        """Multiple threads writing simultaneously should not raise or corrupt data."""
        store = make_store()
        errors = []

        def write_memory(i):
            try:
                mem = make_memory(content=f"Concurrent fact number {i} about science and nature.")
                store.add_memory(mem)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_memory, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

        # All 20 memories should have been written
        all_mems = store.get_all_active_memories()
        assert len(all_mems) == 20

    def test_store_has_write_lock_attribute(self):
        """MemoryStore exposes _write_lock (threading.Lock)."""
        store = make_store()
        assert hasattr(store, "_write_lock")
        assert isinstance(store._write_lock, type(threading.Lock()))


# ─────────────────────────────────────────────────────────────────────────────
# FIX 8 — LLM Retry + Graceful Degradation
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMRetry:
    """
    Fix 9: LLMClient retries on failure and returns LLM_UNAVAILABLE sentinel
    instead of crashing. Callers degrade gracefully.
    """

    def test_llm_client_returns_sentinel_after_max_retries(self):
        """After MAX_RETRIES failures, generate() returns LLM_UNAVAILABLE."""
        from core.llm_client import LLMClient, LLM_UNAVAILABLE

        with patch("core.llm_client.Groq") as MockGroq:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("Rate limited")
            MockGroq.return_value = mock_client

            with patch.dict("os.environ", {"GROQ_API_KEY": "fake-key"}):
                with patch("core.llm_client.time.sleep"):  # skip actual sleep
                    llm = LLMClient()
                    result = llm.generate("test prompt")

        assert result == LLM_UNAVAILABLE

    def test_llm_client_retries_before_giving_up(self):
        """LLMClient retries the correct number of times before returning sentinel."""
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
        """If first call fails but second succeeds, the success is returned."""
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

    def test_planner_voter_degrades_gracefully_when_llm_unavailable(self):
        """
        When LLM returns LLM_UNAVAILABLE, planner_voter falls back to heuristic
        and does not raise an exception.
        """
        from core.llm_client import LLM_UNAVAILABLE

        llm = mock_llm(LLM_UNAVAILABLE)
        proposal = make_proposal("Photosynthesis typically requires sunlight to produce glucose in plants.")
        vote = planner_voter(proposal, llm=llm)

        # Should not raise, should return a valid vote
        assert isinstance(vote.approve, bool)
        assert 0.0 <= vote.confidence <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION — Consensus Pipeline End-to-End
# ─────────────────────────────────────────────────────────────────────────────

class TestConsensusIntegration:
    """
    End-to-end tests of the consensus pipeline with the new LLM-based voters.
    """

    def test_good_memory_accepted_by_consensus(self):
        """A factually sound, general memory is accepted when voters approve."""
        llm = mock_llm('{"approve": true, "confidence": 0.85, "reason": "General knowledge"}')
        proposal = make_proposal(
            "Photosynthesis requires sunlight, CO2, and water to produce glucose and oxygen."
        )

        votes = [
            planner_voter(proposal, llm=llm),
            worker_voter(proposal, llm=llm),
            rule_based_voter(proposal),
        ]
        decision, mem_type, confidence = run_consensus(votes, proposal)

        assert decision == "accept"
        assert confidence > 0.0

    def test_unsafe_memory_quarantined_by_safety_voter(self):
        """A memory with unsafe content is quarantined regardless of other votes."""
        llm = mock_llm('{"approve": true, "confidence": 0.9, "reason": "Looks fine"}')
        proposal = make_proposal(
            "You can hack into systems by exploiting buffer overflow vulnerabilities."
        )

        votes = [
            planner_voter(proposal, llm=llm),   # would approve based on LLM mock
            worker_voter(proposal, llm=llm),    # would approve based on LLM mock
            rule_based_voter(proposal),          # catches "hack" and "exploit"
        ]
        decision, _, _ = run_consensus(votes, proposal)

        assert decision == "quarantine"

    def test_memory_rejected_when_majority_disapprove(self):
        """Memory is rejected when fewer than 2 voters approve."""
        disapprove = Vote(approve=False, confidence=0.2, risk=False, reason="No")
        approve = Vote(approve=True, confidence=0.7, risk=False, reason="Yes")

        proposal = make_proposal("Some content that is debatable.")
        decision, _, _ = run_consensus([disapprove, disapprove, approve], proposal)

        assert decision == "reject"