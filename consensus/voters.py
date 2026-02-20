from consensus.schemas import Vote, MemoryProposal


# ---------------------------------------------------------------------------
# Planner Voter
# Asks: "Is this memory general enough to reuse across future tasks?"
# Rejects: single-use observations, task-specific outputs, very short content
# ---------------------------------------------------------------------------

def planner_voter(proposal: MemoryProposal) -> Vote:
    content = proposal.content.strip()

    # Too short to be reusable knowledge
    if len(content) < 40:
        return Vote(
            approve=False,
            confidence=0.2,
            risk=False,
            reason="Too short to contain reusable knowledge",
        )

    # Looks like a task-specific output (contains step numbers, bullet points
    # about a single action, or first-person execution language)
    task_specific_signals = [
        "step 1", "step 2", "first, ", "then, ", "finally, ",
        "i executed", "i ran", "i completed",
    ]
    content_lower = content.lower()
    if any(sig in content_lower for sig in task_specific_signals):
        return Vote(
            approve=False,
            confidence=0.35,
            risk=False,
            reason="Looks like a task-specific execution trace, not general knowledge",
        )

    # Looks like a reusable fact or principle
    reusable_signals = [
        "always", "never", "typically", "generally", "is defined as",
        "works by", "requires", "consists of", "is used for", "can be",
        "should be", "must be", "is a", "are a",
    ]
    if any(sig in content_lower for sig in reusable_signals):
        return Vote(
            approve=True,
            confidence=0.75,
            risk=False,
            reason="Contains general/reusable knowledge signal",
        )

    # Default: approve with moderate confidence if length is reasonable
    return Vote(
        approve=True,
        confidence=0.55,
        risk=False,
        reason="Moderate length, no red flags — provisionally reusable",
    )


# ---------------------------------------------------------------------------
# Worker Voter
# Asks: "Is this memory actionable for execution?"
# Approves: facts, results, procedures, outcomes
# Rejects: vague summaries, meta-commentary, questions
# ---------------------------------------------------------------------------

def worker_voter(proposal: MemoryProposal) -> Vote:
    content = proposal.content.strip()
    content_lower = content.lower()

    # Vague or meta content — not actionable
    vague_signals = [
        "it depends", "might work", "could be", "perhaps",
        "i'm not sure", "unclear", "unknown",
    ]
    if any(sig in content_lower for sig in vague_signals):
        return Vote(
            approve=False,
            confidence=0.25,
            risk=False,
            reason="Content is vague or uncertain — not reliably actionable",
        )

    # Ends as a question — not a memory, it's a prompt
    if content.endswith("?"):
        return Vote(
            approve=False,
            confidence=0.1,
            risk=False,
            reason="Content is a question, not a memory",
        )

    # Concrete action/result signals
    actionable_signals = [
        "completed", "succeeded", "failed", "returned", "output",
        "result", "works", "error", "exception", "value", "response",
        "executed", "produced", "found", "confirmed",
    ]
    if any(sig in content_lower for sig in actionable_signals):
        return Vote(
            approve=True,
            confidence=0.80,
            risk=False,
            reason="Contains concrete execution or result signal — actionable",
        )

    # Default: approve with low-moderate confidence
    return Vote(
        approve=True,
        confidence=0.50,
        risk=False,
        reason="No red flags found — provisionally actionable",
    )


# ---------------------------------------------------------------------------
# Rule-Based Safety Voter
# Asks: "Is this memory dangerous, misleading, or policy-violating?"
# This voter has effective veto power — any risk flag triggers quarantine.
# ---------------------------------------------------------------------------

def rule_based_voter(proposal: MemoryProposal) -> Vote:
    content_lower = proposal.content.strip().lower()

    # Hard safety violations
    unsafe_words = [
        "hack", "bypass", "illegal", "exploit", "jailbreak",
        "malware", "inject", "phishing", "steal", "override safety",
    ]
    if any(w in content_lower for w in unsafe_words):
        return Vote(
            approve=False,
            confidence=0.05,
            risk=True,
            reason=f"Safety violation detected: content contains restricted term",
        )

    # Factual impossibilities (obvious misinformation patterns)
    misinformation_signals = [
        "only at night",        # e.g. "photosynthesis occurs only at night"
        "never requires",
        "always impossible",
        "is completely false",
        "proven to be fake",
    ]
    if any(sig in content_lower for sig in misinformation_signals):
        return Vote(
            approve=False,
            confidence=0.1,
            risk=True,
            reason="Possible misinformation pattern detected",
        )

    # All clear
    return Vote(
        approve=True,
        confidence=0.90,
        risk=False,
        reason="No safety or policy violations found",
    )