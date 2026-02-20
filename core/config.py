"""
Centralized configuration for CoCortex.
All env vars and defaults live here — import from this file, not os.getenv directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL:    str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# Storage
DB_PATH: str = os.getenv("COCORTEX_DB_PATH", "cocortex_memory.db")


def validate_config() -> None:
    """
    Call at application startup.
    Raises EnvironmentError if required variables are missing.
    """
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if missing:
        raise EnvironmentError(
            f"CoCortex: missing required environment variables: {missing}\n"
            f"Copy .env.example to .env and fill in the values."
        )