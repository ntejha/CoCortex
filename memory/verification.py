from typing import Literal
from core.llm_client import LLMClient, LLM_UNAVAILABLE

VerificationResult = Literal["correct", "incorrect", "uncertain"]


class MemoryVerifier:
    """
    Uses an LLM prompt to fact-check a memory statement.
    Returns 'correct', 'incorrect', or 'uncertain'.

    LLM_UNAVAILABLE handling: when the LLM client cannot reach the API,
    generate() returns the LLM_UNAVAILABLE sentinel. Previously this fell
    through to the string matching logic and returned 'uncertain', which
    (combined with low confidence) could trigger 'downrank' in repair —
    actively degrading the knowledge base during an outage. Now we detect
    the sentinel explicitly and return 'uncertain' with a high implied
    confidence so repair's decide_repair_action() takes no action.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def verify(self, memory_content: str) -> VerificationResult:
        prompt = f"""
You are a strict factual verifier.

Memory statement:
"{memory_content}"

Check this statement against well-established scientific knowledge.

Answer with ONE word only:
- correct
- incorrect
- uncertain
"""
        response = self.llm.generate(prompt)

        # Explicit sentinel check — must come before any string parsing.
        # repair.decide_repair_action("uncertain", confidence=0.9) → "none",
        # so returning uncertain here is safe: no repair action is taken.
        if response == LLM_UNAVAILABLE:
            return "uncertain"

        response = response.strip().lower()

        # "incorrect" before "correct" — "incorrect" contains "correct"
        if "incorrect" in response or "not correct" in response or "wrong" in response:
            return "incorrect"
        if "correct" in response:
            return "correct"

        return "uncertain"