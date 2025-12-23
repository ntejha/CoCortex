from typing import List
from consensus.schemas import Vote, MemoryProposal

def run_consensus(votes: List[Vote], proposal: MemoryProposal):
    if any(v.risk for v in votes):
        return "quarantine", proposal.suggested_type, 0.1

    approvals = [v for v in votes if v.approve]

    if len(approvals) >= 2:
        avg_conf = sum(v.confidence for v in approvals) / len(approvals)
        return "accept", proposal.suggested_type, avg_conf

    return "reject", None, 0.0
