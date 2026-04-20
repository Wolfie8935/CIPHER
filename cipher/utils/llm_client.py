"""
LLM client for CIPHER.

Provides a unified interface to LLM backends (NVIDIA NIM, HuggingFace, OpenAI).
In Phase 1, this client is imported but NOT called — all agents use stub logic.
Phase 4 will activate actual API calls through this client.

Owns: LLM API routing, retry logic, and response parsing.
Does NOT own: prompt construction (that is each agent's responsibility) or
configuration loading (uses config singleton).
"""
from __future__ import annotations

import logging
from typing import Any

from cipher.utils.config import config
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """
    Unified LLM client that routes to the correct backend based on config.

    In Phase 1, all methods are callable but the client is not invoked by agents.
    Agents use stub random-action logic instead.
    """

    def __init__(self) -> None:
        self.backend: str = config.llm_backend
        self._client: Any = None
        logger.debug(f"LLMClient initialized with backend: {self.backend}")

    def _ensure_client(self) -> None:
        """Lazily initialize the underlying API client."""
        if self._client is not None:
            return

        if self.backend == "nvidia":
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    base_url=config.nvidia_base_url,
                    api_key=config.nvidia_api_key,
                )
                logger.debug("NVIDIA NIM client initialized")
            except ImportError as exc:
                logger.warning(
                    f"OpenAI package not available: {exc}. "
                    "LLM calls will fail until installed."
                )
        elif self.backend == "huggingface":
            logger.info("HuggingFace backend selected — not implemented until Phase 14")
        else:
            logger.warning(f"Unknown LLM backend: {self.backend}")

    def complete(
        self,
        model_env_key: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
    ) -> str:
        """
        Send a chat completion request to the configured LLM backend.

        Args:
            model_env_key: The config attribute name for the model
                           (e.g. 'nvidia_model_red_planner').
            messages: List of message dicts with 'role' and 'content' keys.
            max_tokens: Maximum tokens in the response.

        Returns:
            The assistant's response text.

        Raises:
            RuntimeError: If the backend is not initialized or the call fails.
        """
        self._ensure_client()

        if self._client is None:
            raise RuntimeError(
                f"LLM client not initialized for backend '{self.backend}'. "
                "Check your .env configuration and installed packages."
            )

        model: str = getattr(config, model_env_key, "")
        if not model:
            raise ValueError(
                f"No model configured for key '{model_env_key}'. "
                "Check your .env file."
            )

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.error(f"LLM call failed for model '{model}': {exc}")
            raise RuntimeError(f"LLM call failed: {exc}") from exc


# ── Singleton ────────────────────────────────────────────────────
llm_client = LLMClient()
