"""
consensus/engine.py

Consensus engine with:
- Exactly-3-voters validation
- Weighted confidence across ALL voters (including dissenting ones)
"""
from typing import List
from consensus.schemas import Vote, MemoryProposal


def run_consensus(votes: List[Vote], proposal: MemoryProposal):
    """
    Tally votes and return (decision, memory_type, confidence).

    Rules:
    1. Exactly 3 voters are required.
    2. Any risk=True vote → quarantine immediately.
    3. ≥ 2 approvals → accept, confidence = weighted average of ALL votes.
    4. Otherwise → reject.
    """
    if len(votes) != 3:
        raise ValueError(
            f"run_consensus requires exactly 3 voters, got {len(votes)}. "
            "Pass votes from planner_voter, worker_voter, and rule_based_voter."
        )

    if any(v.risk for v in votes):
        return "quarantine", proposal.suggested_type, 0.1

    approvals = [v for v in votes if v.approve]

    if len(approvals) >= 2:
        # Weighted average across ALL voters — dissent lowers confidence
        avg_conf = round(sum(v.confidence for v in votes) / len(votes), 3)
        return "accept", proposal.suggested_type, avg_conf

    return "reject", None, 0.0