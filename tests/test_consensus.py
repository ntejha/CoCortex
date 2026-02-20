"""
Tests for consensus/engine.py and consensus/voters.py
Run from project root: python -m pytest tests/test_consensus.py -v
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consensus.engine import run_consensus
from consensus.schemas import Vote, MemoryProposal
from consensus.voters import planner_voter, worker_voter, rule_based_voter


@pytest.fixture
def basic_proposal():
    return MemoryProposal(
        content="Photosynthesis converts CO2 and water into glucose using sunlight energy.",
        source_agent="worker",
        suggested_type="episodic",
        context={"task": "biology explanation"},
    )


@pytest.fixture
def risky_proposal():
    return MemoryProposal(
        content="You can hack the system by bypassing the authentication layer.",
        source_agent="worker",
        suggested_type="episodic",
        context={"task": "security"},
    )


# --- Consensus Engine ---

def test_two_approvals_accepts(basic_proposal):
    votes = [
        Vote(approve=True, confidence=0.8, risk=False, reason="ok"),
        Vote(approve=True, confidence=0.7, risk=False, reason="ok"),
        Vote(approve=False, confidence=0.3, risk=False, reason="no"),
    ]
    decision, mem_type, confidence = run_consensus(votes, basic_proposal)
    assert decision == "accept"
    assert confidence > 0


def test_risk_flag_quarantines(basic_proposal):
    votes = [
        Vote(approve=True, confidence=0.9, risk=False, reason="ok"),
        Vote(approve=True, confidence=0.8, risk=False, reason="ok"),
        Vote(approve=False, confidence=0.1, risk=True, reason="risky"),
    ]
    decision, mem_type, confidence = run_consensus(votes, basic_proposal)
    assert decision == "quarantine"
    assert confidence == 0.1


def test_no_approvals_rejects(basic_proposal):
    votes = [
        Vote(approve=False, confidence=0.2, risk=False, reason="no"),
        Vote(approve=False, confidence=0.3, risk=False, reason="no"),
        Vote(approve=False, confidence=0.1, risk=False, reason="no"),
    ]
    decision, mem_type, confidence = run_consensus(votes, basic_proposal)
    assert decision == "reject"


def test_single_approval_rejects(basic_proposal):
    votes = [
        Vote(approve=True, confidence=0.9, risk=False, reason="yes"),
        Vote(approve=False, confidence=0.2, risk=False, reason="no"),
        Vote(approve=False, confidence=0.1, risk=False, reason="no"),
    ]
    decision, _, _ = run_consensus(votes, basic_proposal)
    assert decision == "reject"


# --- Planner Voter ---

def test_planner_voter_rejects_short_content():
    proposal = MemoryProposal(
        content="Done.",
        source_agent="worker",
        suggested_type="episodic",
        context={},
    )
    vote = planner_voter(proposal)
    assert vote.approve is False


def test_planner_voter_approves_general_knowledge(basic_proposal):
    vote = planner_voter(basic_proposal)
    assert vote.approve is True
    assert vote.risk is False


# --- Worker Voter ---

def test_worker_voter_rejects_questions():
    proposal = MemoryProposal(
        content="What is the correct approach to this problem?",
        source_agent="planner",
        suggested_type="episodic",
        context={},
    )
    vote = worker_voter(proposal)
    assert vote.approve is False


def test_worker_voter_approves_concrete_result(basic_proposal):
    vote = worker_voter(basic_proposal)
    assert vote.risk is False


# --- Rule-Based Safety Voter ---

def test_rule_voter_flags_unsafe_content(risky_proposal):
    vote = rule_based_voter(risky_proposal)
    assert vote.risk is True
    assert vote.approve is False


def test_rule_voter_passes_safe_content(basic_proposal):
    vote = rule_based_voter(basic_proposal)
    assert vote.risk is False
    assert vote.approve is True


def test_risky_memory_quarantined_end_to_end(risky_proposal):
    """Full pipeline: risky content should always be quarantined."""
    votes = [
        planner_voter(risky_proposal),
        worker_voter(risky_proposal),
        rule_based_voter(risky_proposal),
    ]
    decision, _, _ = run_consensus(votes, risky_proposal)
    assert decision == "quarantine"