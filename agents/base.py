"""
agents/base.py

Shared infrastructure for asynchronous LLM agents.
Provides a shared semaphore to control maximum concurrent Groq API calls,
preventing rate limits when running multiple agents in parallel.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)


class AsyncAgent:
    """
    Mixin/Base class providing rate-limited async LLM execution via thread offloading.
    The Groq Python SDK is synchronous, so we offload calls to a thread pool
    while awaiting them asynchronously.
    """
    # Shared across ALL agents to enforce a global concurrency limit for the API.
    # Default is 5 to respect typical free-tier or standard tier rate limits.
    # Initialized lazily to ensure it is created within the running asyncio event loop.
    _semaphore = None
    _default_concurrency = 5

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        if cls._semaphore is None:
            cls._semaphore = asyncio.Semaphore(cls._default_concurrency)
        return cls._semaphore

    @classmethod
    def configure_concurrency(cls, max_concurrent: int):
        """Update the global concurrency limit for LLM calls."""
        cls._default_concurrency = max_concurrent
        cls._semaphore = asyncio.Semaphore(max_concurrent)

    async def _allm(self, prompt: str) -> str:
        """
        Execute a prompt asynchronously, respecting the global concurrency limit.
        Offloads the blocking synchronous call to a worker thread.
        """
        if not hasattr(self, "llm"):
            raise ValueError("AsyncAgent requires self.llm to be set by the subclass")

        async with self._get_semaphore():
            # We use asyncio.to_thread to run the blocking LLM call without blocking the event loop
            try:
                response = await asyncio.to_thread(self.llm.generate, prompt)
                return response
            except Exception as e:
                logger.error(f"Async LLM call failed: {e}")
                # We raise so the pipeline orchestrator can handle it
                raise
