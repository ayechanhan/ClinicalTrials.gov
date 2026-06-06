"""Centralized configuration.

Loads ``.env`` once on import and exposes typed, cached settings. The
ClinicalTrials.gov client reads its own ``CT_*`` env vars (it is a self-contained,
independently usable unit); this module owns the app/LLM-level config and ensures
``.env`` is loaded for the whole process regardless of entry point.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # populate os.environ from .env once, at import time


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    log_level: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached process settings sourced from the environment."""
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
