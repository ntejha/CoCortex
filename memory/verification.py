"""
memory/verification.py

LLM-based memory verifier with LLM_UNAVAILABLE handling.
"""
from typing import Literal
from core.llm_client import LLMClient, LLM_UNAVAILABLE

VerificationResult = Literal["correct", "incorrect", "uncertain"]


class MemoryVerifier:
    """
    Uses an LLM prompt to verify memory correctness.
    Returns 'uncertain' if the LLM is unavailable.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def verify(self, memory_content: str) -> VerificationResult:
        prompt = f"""You are a strict factual verifier.

Memory statement:
"{memory_content}"

Check this statement against well-established scientific knowledge.

Answer with ONE word only:
- correct
- incorrect
- uncertain
"""
        response = self.llm.generate(prompt)

        # Handle LLM unavailability gracefully
        if response == LLM_UNAVAILABLE:
            return "uncertain"

        response = response.strip().lower()

        # Check "incorrect" before "correct" — "incorrect" contains "correct"
        if "incorrect" in response or "not correct" in response or "wrong" in response:
            return "incorrect"
        if "correct" in response:
            return "correct"

        return "uncertain"