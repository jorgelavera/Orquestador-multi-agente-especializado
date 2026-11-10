"""Runtime configuration for the optional local Ollama supervisor."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    provider: str = "auto"
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen3.5:4b-q4_K_M"
    timeout_seconds: float = 30.0
    max_steps: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("ORCHESTRATOR_PROVIDER", "auto").strip().lower()
        if provider not in {"auto", "ollama", "offline"}:
            raise ValueError("ORCHESTRATOR_PROVIDER must be auto, ollama, or offline")
        return cls(
            provider=provider,
            base_url=os.getenv("ORCHESTRATOR_BASE_URL", "http://127.0.0.1:11434/v1"),
            model=os.getenv("ORCHESTRATOR_MODEL", "qwen3.5:4b-q4_K_M"),
            timeout_seconds=float(os.getenv("ORCHESTRATOR_TIMEOUT_SECONDS", "30")),
            max_steps=max(1, int(os.getenv("ORCHESTRATOR_MAX_STEPS", "8"))),
        )
