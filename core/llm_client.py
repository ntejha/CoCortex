"""
core/llm_client.py
==================
LLM client with retry logic and graceful error handling.

Design decisions:
- Exponential backoff on failure (handles rate limits and transient errors)
- Max 3 retries before returning a safe fallback string
- Callers check for LLM_UNAVAILABLE to handle gracefully
- temperature=0.3 for deterministic-enough outputs across voters and verifiers
"""

import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Sentinel value returned when LLM is completely unavailable after retries
LLM_UNAVAILABLE = "LLM_UNAVAILABLE"

MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0


class LLMClient:
    def __init__(self, model="llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the response string.

        Retries up to MAX_RETRIES times with exponential backoff on failure.
        Returns LLM_UNAVAILABLE if all retries are exhausted — callers should
        check for this sentinel and degrade gracefully (e.g. voters fall back
        to heuristics, verifiers return "uncertain").
        """
        backoff = INITIAL_BACKOFF_SECONDS

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                return response.choices[0].message.content

            except Exception as e:
                if attempt == MAX_RETRIES:
                    # All retries exhausted — return sentinel
                    print(f"[LLMClient] All {MAX_RETRIES} retries failed: {e}")
                    return LLM_UNAVAILABLE

                print(f"[LLMClient] Attempt {attempt} failed ({e}), retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= 2  # exponential backoff