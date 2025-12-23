from consensus.schemas import Vote, MemoryProposal

def planner_voter(proposal: MemoryProposal) -> Vote:
    useful = len(proposal.content) > 30
    return Vote(
        approve=useful,
        confidence=0.7 if useful else 0.3,
        risk=False,
        reason="Planner finds it reusable" if useful else "Too specific"
    )

def worker_voter(proposal: MemoryProposal) -> Vote:
    consistent = "works" in proposal.content.lower()
    return Vote(
        approve=consistent,
        confidence=0.8 if consistent else 0.4,
        risk=False,
        reason="Matches execution" if consistent else "Execution mismatch"
    )

def rule_based_voter(proposal: MemoryProposal) -> Vote:
    risky_words = ["hack", "bypass", "illegal"]
    risk = any(w in proposal.content.lower() for w in risky_words)

    return Vote(
        approve=not risk,
        confidence=0.9 if not risk else 0.1,
        risk=risk,
        reason="Risk detected" if risk else "Passed rules"
    )
