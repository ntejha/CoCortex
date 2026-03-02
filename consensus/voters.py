"""
consensus/voters.py
===================
Hybrid voting system for CoCortex memory admission.

Architecture:
  - planner_voter  → LLM-based: reasons about reusability
  - worker_voter   → LLM-based: reasons about actionability
  - rule_based_voter → Deterministic: safety veto (intentionally NOT LLM,
                       because safety decisions must be fast, consistent,
                       and not subject to LLM hallucination)

Why hybrid?
  LLM voters handle nuanced semantic quality judgements that keyword
  matching cannot. The rule-based safety voter intentionally stays
  deterministic — an LLM should never be the sole arbiter of safety.
"""

import json
from consensus.schemas import Vote, MemoryProposal


# ---------------------------------------------------------------------------
# LLM-Based Planner Voter
# Asks: "Is this memory general enough to reuse across future tasks?"
# Uses the LLM to reason semantically, not keyword match.
# ---------------------------------------------------------------------------

def planner_voter(proposal: MemoryProposal, llm=None) -> Vote:
    """
    LLM-based voter that assesses whether a memory is reusable general knowledge.

    Falls back to heuristics if no LLM is provided (e.g. in unit tests).
    """
    content = proposal.content.strip()

    # Hard pre-filter: too short to be knowledge regardless of LLM opinion
    if len(content) < 40:
        return Vote(
            approve=False,
            confidence=0.2,
            risk=False,
            reason="Too short to contain reusable knowledge (pre-filter)",
        )

    if llm is None:
        # Fallback heuristic (used in tests / offline mode)
        return _planner_heuristic(content)

    prompt = f"""You are a memory quality assessor for an AI knowledge base.

Evaluate whether the following memory should be stored as REUSABLE general knowledge
that will benefit future tasks across different contexts.

Memory: "{content}"

Answer in JSON only, no extra text:
{{
  "approve": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "one sentence explanation"
}}

Approve if: the memory states a general fact, principle, or procedure applicable beyond one task.
Reject if: the memory is a single-use execution trace, a task-specific output, or too vague."""

    try:
        raw = llm.generate(prompt).strip()
        # Strip markdown fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        return Vote(
            approve=bool(parsed.get("approve", False)),
            confidence=float(parsed.get("confidence", 0.5)),
            risk=False,
            reason=str(parsed.get("reason", "LLM assessment")),
        )
    except Exception as e:
        # If LLM fails or returns malformed JSON, fall back to heuristic
        return _planner_heuristic(content)


def _planner_heuristic(content: str) -> Vote:
    """Offline fallback heuristic for planner voter."""
    content_lower = content.lower()
    task_specific_signals = ["step 1", "step 2", "i executed", "i ran", "i completed"]
    if any(sig in content_lower for sig in task_specific_signals):
        return Vote(approve=False, confidence=0.35, risk=False,
                    reason="Looks like a task-specific execution trace (heuristic fallback)")
    reusable_signals = ["always", "never", "typically", "generally", "is defined as",
                        "works by", "requires", "consists of", "is used for"]
    if any(sig in content_lower for sig in reusable_signals):
        return Vote(approve=True, confidence=0.70, risk=False,
                    reason="Contains general knowledge signal (heuristic fallback)")
    return Vote(approve=True, confidence=0.50, risk=False,
                reason="No red flags, provisionally reusable (heuristic fallback)")


# ---------------------------------------------------------------------------
# LLM-Based Worker Voter
# Asks: "Is this memory actionable for execution agents?"
# ---------------------------------------------------------------------------

def worker_voter(proposal: MemoryProposal, llm=None) -> Vote:
    """
    LLM-based voter that assesses whether a memory is actionable for execution.

    Falls back to heuristics if no LLM is provided.
    """
    content = proposal.content.strip()

    # Hard pre-filter: questions are never memories
    if content.endswith("?"):
        return Vote(approve=False, confidence=0.1, risk=False,
                    reason="Content is a question, not a memory (pre-filter)")

    if llm is None:
        return _worker_heuristic(content)

    prompt = f"""You are a memory quality assessor for an AI execution agent.

Evaluate whether the following memory is ACTIONABLE — meaning an execution agent
can use it to complete tasks, make decisions, or understand procedures.

Memory: "{content}"

Answer in JSON only, no extra text:
{{
  "approve": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "one sentence explanation"
}}

Approve if: the memory contains a concrete fact, procedure, outcome, or result an agent can act on.
Reject if: the memory is vague, speculative, contradictory, or purely meta-commentary."""

    try:
        raw = llm.generate(prompt).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        return Vote(
            approve=bool(parsed.get("approve", False)),
            confidence=float(parsed.get("confidence", 0.5)),
            risk=False,
            reason=str(parsed.get("reason", "LLM assessment")),
        )
    except Exception:
        return _worker_heuristic(content)


def _worker_heuristic(content: str) -> Vote:
    """Offline fallback heuristic for worker voter."""
    content_lower = content.lower()
    vague_signals = ["it depends", "might work", "could be", "perhaps", "i'm not sure", "unclear"]
    if any(sig in content_lower for sig in vague_signals):
        return Vote(approve=False, confidence=0.25, risk=False,
                    reason="Content is vague or uncertain (heuristic fallback)")
    actionable_signals = ["completed", "succeeded", "failed", "returned", "output",
                          "result", "works", "error", "executed", "produced", "found"]
    if any(sig in content_lower for sig in actionable_signals):
        return Vote(approve=True, confidence=0.75, risk=False,
                    reason="Contains concrete execution signal (heuristic fallback)")
    return Vote(approve=True, confidence=0.50, risk=False,
                reason="No red flags, provisionally actionable (heuristic fallback)")


# ---------------------------------------------------------------------------
# Rule-Based Safety Voter (INTENTIONALLY NOT LLM-BASED)
# Asks: "Is this memory dangerous, misleading, or policy-violating?"
#
# Design decision: This voter is deterministic by design.
# Safety checks must be fast, consistent, and auditable.
# An LLM should NEVER be the sole gatekeeper for safety — it can be
# manipulated, hallucinate, or disagree with itself across calls.
# This voter has effective veto power — any risk=True triggers quarantine.
# ---------------------------------------------------------------------------

def rule_based_voter(proposal: MemoryProposal, llm=None) -> Vote:
    """
    Deterministic safety voter. The llm parameter is accepted for API
    consistency but intentionally ignored — safety is not LLM-delegated.
    """
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
            reason="Safety violation: content contains restricted term",
        )

    # Factual impossibilities / obvious misinformation patterns
    misinformation_signals = [
        "only at night",
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

    return Vote(
        approve=True,
        confidence=0.90,
        risk=False,
        reason="No safety or policy violations found",
    )