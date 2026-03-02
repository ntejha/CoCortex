"""
consensus/voters.py

Three independent voters with optional LLM support.

Each voter accepts an optional `llm` parameter:
- If provided (and content passes pre-filter), the LLM is asked to vote.
  Its JSON response is parsed and used. Malformed / unavailable responses
  fall back to the heuristic path.
- If llm=None (default), the deterministic heuristic is used.

The rule_based_voter is always deterministic — it never calls the LLM.
"""
import json
import logging
from consensus.schemas import Vote, MemoryProposal

logger = logging.getLogger(__name__)

# Sentinel imported lazily to avoid circular imports at module load time.
_LLM_UNAVAILABLE = None

def _get_sentinel():
    global _LLM_UNAVAILABLE
    if _LLM_UNAVAILABLE is None:
        from core.llm_client import LLM_UNAVAILABLE
        _LLM_UNAVAILABLE = LLM_UNAVAILABLE
    return _LLM_UNAVAILABLE


def _clamp(value: float) -> float:
    """Clamp a confidence value to [0.0, 1.0]."""
    return round(max(0.0, min(float(value), 1.0)), 3)


def _parse_llm_vote(raw: str) -> dict | None:
    """
    Try to parse an LLM response as a JSON vote dict.
    Returns None if parsing fails or the LLM is unavailable.
    Expected shape: {"approve": bool, "confidence": float, "reason": str}
    """
    if raw == _get_sentinel() or not raw:
        return None
    try:
        # Strip markdown code fences if present
        cleaned = raw.strip().strip("```json").strip("```").strip()
        data = json.loads(cleaned)
        if not isinstance(data.get("approve"), bool):
            return None
        return data
    except (json.JSONDecodeError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Planner Voter
# ---------------------------------------------------------------------------

_PLANNER_TASK_SPECIFIC = [
    "step 1", "step 2", "first, ", "then, ", "finally, ",
    "i executed", "i ran", "i completed",
]
_PLANNER_REUSABLE = [
    "always", "never", "typically", "generally", "is defined as",
    "works by", "requires", "consists of", "is used for", "can be",
    "should be", "must be", "is a", "are a",
]

_PLANNER_LLM_PROMPT = """\
You are a memory admission voter for an AI planning agent.

Evaluate whether the following content is general, reusable knowledge
suitable for planning future tasks (semantic memory), or a one-time
execution trace that should not be retained.

Content: "{content}"

Respond in JSON only:
{{"approve": true/false, "confidence": 0.0-1.0, "reason": "short reason"}}
"""


def planner_voter(proposal: MemoryProposal, llm=None) -> Vote:
    content = proposal.content.strip()
    content_lower = content.lower()

    # Pre-filter: always reject trivially short content before calling LLM
    if len(content) < 40:
        return Vote(approve=False, confidence=0.2, risk=False,
                    reason="Too short to contain reusable knowledge")

    # Try LLM if provided
    if llm is not None:
        raw = llm.generate(_PLANNER_LLM_PROMPT.format(content=content))
        parsed = _parse_llm_vote(raw)
        if parsed is not None:
            return Vote(
                approve=bool(parsed["approve"]),
                confidence=_clamp(parsed.get("confidence", 0.5)),
                risk=False,
                reason=str(parsed.get("reason", "LLM decision")),
            )
        logger.debug("planner_voter: LLM response unparseable, falling back to heuristic")

    # Heuristic fallback
    if any(sig in content_lower for sig in _PLANNER_TASK_SPECIFIC):
        return Vote(approve=False, confidence=0.35, risk=False,
                    reason="Looks like a task-specific execution trace")

    if any(sig in content_lower for sig in _PLANNER_REUSABLE):
        return Vote(approve=True, confidence=0.75, risk=False,
                    reason="Contains general/reusable knowledge signal")

    return Vote(approve=True, confidence=0.55, risk=False,
                reason="Moderate length, no red flags — provisionally reusable")


# ---------------------------------------------------------------------------
# Worker Voter
# ---------------------------------------------------------------------------

_WORKER_VAGUE = [
    "it depends", "might work", "could be", "perhaps",
    "i'm not sure", "unclear", "unknown",
]
_WORKER_ACTIONABLE = [
    "completed", "succeeded", "failed", "returned", "output",
    "result", "works", "error", "exception", "value", "response",
    "executed", "produced", "found", "confirmed",
]

_WORKER_LLM_PROMPT = """\
You are a memory admission voter for an AI worker agent.

Evaluate whether the following content is a concrete, actionable piece of
knowledge (suitable for guiding future execution), or too vague/uncertain
to be useful.

Content: "{content}"

Respond in JSON only:
{{"approve": true/false, "confidence": 0.0-1.0, "reason": "short reason"}}
"""


def worker_voter(proposal: MemoryProposal, llm=None) -> Vote:
    content = proposal.content.strip()
    content_lower = content.lower()

    # Pre-filter: reject questions immediately before LLM call
    if content.endswith("?"):
        return Vote(approve=False, confidence=0.1, risk=False,
                    reason="Content is a question, not a memory")

    # Try LLM if provided
    if llm is not None:
        raw = llm.generate(_WORKER_LLM_PROMPT.format(content=content))
        parsed = _parse_llm_vote(raw)
        if parsed is not None:
            return Vote(
                approve=bool(parsed["approve"]),
                confidence=_clamp(parsed.get("confidence", 0.5)),
                risk=False,
                reason=str(parsed.get("reason", "LLM decision")),
            )
        logger.debug("worker_voter: LLM response unparseable, falling back to heuristic")

    # Heuristic fallback
    if any(sig in content_lower for sig in _WORKER_VAGUE):
        return Vote(approve=False, confidence=0.25, risk=False,
                    reason="Content is vague or uncertain — not reliably actionable")

    if any(sig in content_lower for sig in _WORKER_ACTIONABLE):
        return Vote(approve=True, confidence=0.80, risk=False,
                    reason="Contains concrete execution or result signal — actionable")

    return Vote(approve=True, confidence=0.50, risk=False,
                reason="No red flags found — provisionally actionable")


# ---------------------------------------------------------------------------
# Rule-Based Safety Voter (always deterministic, never calls LLM)
# ---------------------------------------------------------------------------

_UNSAFE_TERMS = [
    "hack", "hacking", "hacked",
    "exploit", "exploitation",
    "bypass", "circumvent", "evade", "evading",
    "jailbreak", "jailbreaking",
    "malware", "ransomware", "spyware", "rootkit", "trojan", "virus",
    "sql injection", "code injection", "prompt injection",
    "inject malicious",
    "phishing", "spear phishing", "spoofing",
    "steal credentials", "steal data", "exfiltrate",
    "social engineer",
    "override safety", "disable safety", "remove restrictions",
    "ignore instructions", "ignore your instructions",
    "illegal", "unlawful", "criminal activity",
]

_MISINFORMATION_SIGNALS = [
    "only at night",
    "never requires",
    "always impossible",
    "is completely false",
    "proven to be fake",
    "scientists have disproved",
    "universally rejected",
]


def rule_based_voter(proposal: MemoryProposal, llm=None) -> Vote:
    """
    Deterministic safety voter. The llm parameter is accepted but never used —
    safety rules must be deterministic and cannot be overridden by an LLM.
    """
    content_lower = proposal.content.strip().lower()

    for term in _UNSAFE_TERMS:
        if term in content_lower:
            logger.warning("Safety voter flagged term '%s': %.60s...", term, proposal.content)
            return Vote(
                approve=False, confidence=0.05, risk=True,
                reason=f"Safety violation: restricted term '{term}'",
            )

    for sig in _MISINFORMATION_SIGNALS:
        if sig in content_lower:
            logger.warning("Misinformation pattern '%s': %.60s...", sig, proposal.content)
            return Vote(
                approve=False, confidence=0.1, risk=True,
                reason=f"Possible misinformation: '{sig}'",
            )

    return Vote(approve=True, confidence=0.90, risk=False,
                reason="No safety or policy violations found")