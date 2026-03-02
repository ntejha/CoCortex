"""
core/llm_client.py
==================
LLM client with retry logic and graceful error handling.

Design decisions:
- Only retries TRANSIENT errors (rate limits, server errors, timeouts).
- Does NOT retry PERMANENT errors (auth failures, model not found, bad requests)
  — these will never succeed and retrying wastes time and quota.
- Returns LLM_UNAVAILABLE sentinel after exhausting retries so callers can
  degrade gracefully without crashing the system.
- temperature=0.3 for deterministic-enough outputs across voters and verifiers.
"""

import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Sentinel returned when LLM is unavailable after retries
LLM_UNAVAILABLE = "LLM_UNAVAILABLE"

MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0

# HTTP status codes that are worth retrying (transient server-side issues)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    """
    Return True only for transient errors that may succeed on retry.

    Permanent errors (401 auth, 404 model not found, 400 bad request)
    are identified by their HTTP status code and skipped immediately —
    retrying them wastes backoff time and API quota.

    Falls back to retrying unknown exception types (safe default).
    """
    # Groq SDK wraps HTTP errors with a .status_code attribute
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in _RETRYABLE_STATUS_CODES
    # No status code — could be a network timeout or connection error; retry
    return True


class LLMClient:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Send a prompt and return the response string.

        Retries transient failures up to MAX_RETRIES times with exponential
        backoff. Permanent failures (auth, model not found) are not retried.
        Returns LLM_UNAVAILABLE if all retries are exhausted.
        """
        backoff = INITIAL_BACKOFF_SECONDS

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                return response.choices[0].message.content

            except Exception as e:
                if not _is_retryable(e):
                    # Permanent error — no point retrying
                    print(f"[LLMClient] Permanent error (not retrying): {e}")
                    return LLM_UNAVAILABLE

                if attempt == MAX_RETRIES:
                    print(f"[LLMClient] All {MAX_RETRIES} retries exhausted: {e}")
                    return LLM_UNAVAILABLE

                print(f"[LLMClient] Attempt {attempt} failed ({e}), retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= 2