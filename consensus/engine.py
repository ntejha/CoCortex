from typing import List, Tuple
from consensus.schemas import Vote, MemoryProposal


def run_consensus(
    votes: List[Vote],
    proposal: MemoryProposal,
) -> Tuple[str, str | None, float]:
    """
    Determine whether to accept, quarantine, or reject a memory proposal.

    Rules (in priority order):
      1. Any vote with risk=True  → quarantine immediately (safety veto)
      2. Fewer than 3 voters      → reject (misconfiguration guard)
      3. Majority approve (≥ 2)  → accept with weighted confidence
      4. Otherwise               → reject

    Confidence calculation:
      Uses a weighted average across ALL voters (not just approvers).
      Dissenting votes with low confidence reduce the accepted score,
      reflecting genuine uncertainty in the consensus. Previously, only
      approving voters were averaged, which ignored strong opposition.
    """
    if len(votes) < 3:
        # Guard against misconfigured callers — 3 voters are always expected.
        # Silently accepting with fewer voters would bypass intended governance.
        raise ValueError(
            f"run_consensus requires exactly 3 voters, got {len(votes)}. "
            f"Ensure planner_voter, worker_voter, and rule_based_voter are all passed."
        )

    # Safety veto: any risk flag quarantines immediately
    if any(v.risk for v in votes):
        return "quarantine", proposal.suggested_type, 0.1

    approvals = [v for v in votes if v.approve]

    if len(approvals) >= 2:
        # Weighted average across ALL voters — dissenting confidence
        # counts against acceptance, reflecting real uncertainty.
        avg_conf = sum(v.confidence for v in votes) / len(votes)
        return "accept", proposal.suggested_type, round(avg_conf, 3)

    return "reject", None, 0.0