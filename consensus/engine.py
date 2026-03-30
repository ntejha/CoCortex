"""
consensus/engine.py

Consensus engine with:
- Exactly-3-voters validation
- Weighted confidence across ALL voters (including dissenting ones)
"""
from typing import List
from consensus.schemas import Vote, MemoryProposal


def run_consensus(votes: List[Vote], proposal: MemoryProposal):

    if len(votes) != 3:
        raise ValueError(
            f"run_consensus requires exactly 3 voters, got {len(votes)}."
        )

    # 🔥 New logic: majority risk required
    risk_votes = sum(1 for v in votes if v.risk)
    if risk_votes >= 2:
        return "quarantine", proposal.suggested_type, 0.1

    approvals = [v for v in votes if v.approve]

    if len(approvals) >= 2:
        avg_conf = sum(v.confidence for v in votes) / len(votes)

        # 🔒 Stricter acceptance rule
        if avg_conf >= 0.75:
            return "accept", proposal.suggested_type, round(avg_conf, 3)
        else:
            return "reject", None, 0.0

    return "reject", None, 0.0