from typing import Literal
from core.llm_client import LLMClient

VerificationResult = Literal["correct", "incorrect", "uncertain"]

class MemoryVerifier:
    """
    Uses an evaluator-style LLM prompt to verify memory correctness.
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
        response = self.llm.generate(prompt).strip().lower()

        if "incorrect" in response:
            return "incorrect"
        if "correct" in response:
            return "correct"

        return "uncertain"
