"""
core/llm_client.py

LLM client with:
- Retry with exponential back-off (transient errors)
- Immediate bail-out on permanent errors (401 / auth)
- LLM_UNAVAILABLE sentinel so callers can degrade gracefully
- MAX_RETRIES constant accessible for tests
"""
import os
import time
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Sentinel returned when the LLM is unreachable after all retries.
LLM_UNAVAILABLE = "__LLM_UNAVAILABLE__"

# Number of attempts before giving up on a transient error.
MAX_RETRIES = 3

# Seconds to wait between retries (doubles each attempt).
_RETRY_BASE_DELAY = 1.0

# HTTP status codes that should NOT be retried (will never succeed).
_PERMANENT_ERROR_CODES = {401, 403}


class LLMClient:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the response text.

        Retries up to MAX_RETRIES times on transient errors.
        Returns LLM_UNAVAILABLE if all attempts fail or a permanent error occurs.
        """
        delay = _RETRY_BASE_DELAY

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                return response.choices[0].message.content

            except Exception as exc:
                status = getattr(exc, "status_code", None)

                if status in _PERMANENT_ERROR_CODES:
                    logger.error(
                        "Permanent LLM error (status=%s): %s — not retrying.",
                        status,
                        exc,
                    )
                    return LLM_UNAVAILABLE

                logger.warning(
                    "LLM call failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc
                )

                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                    delay *= 2

        logger.error("LLM unavailable after %d retries.", MAX_RETRIES)
        return LLM_UNAVAILABLE