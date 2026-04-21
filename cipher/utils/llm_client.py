"""
cipher/utils/llm_client.py

Unified LLM client for CIPHER. Routes to NVIDIA NIM or HuggingFace
based on LLM_BACKEND in .env. Currently NVIDIA only — HuggingFace
integration is Phase 14 and gated behind explicit instruction.

This is the ONLY file that imports openai or makes HTTP calls to any LLM.
All 8 agents call this. The Oversight agent calls this. Nothing else does.
"""
from __future__ import annotations

import time
import json
from typing import Any
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from cipher.utils.config import config
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """
    Wraps the NVIDIA NIM OpenAI-compatible API.

    Handles:
    - Model routing per agent role (each agent uses the model specified in .env)
    - Retry logic with exponential backoff (3 attempts, 2s/4s/8s delays)
    - Rate limit handling (waits and retries on 429)
    - Malformed response handling (returns fallback WAIT action on parse failure)
    - Token counting and logging
    - Response time logging

    Does NOT handle:
    - Prompt construction (each agent does its own)
    - Action validation (base_agent does this)
    - HuggingFace routing (Phase 14)
    """

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0   # seconds, doubles each retry
    REQUEST_TIMEOUT = 30.0   # seconds

    def __init__(self) -> None:
        self.backend = config.llm_backend
        if self.backend == "nvidia":
            self.client = OpenAI(
                base_url=config.nvidia_base_url,
                api_key=config.nvidia_api_key,
                timeout=self.REQUEST_TIMEOUT,
            )
        else:
            # HuggingFace — Phase 14 only, raise clearly
            raise NotImplementedError(
                f"LLM backend '{self.backend}' is not implemented. "
                "Set LLM_BACKEND=nvidia in .env. "
                "HuggingFace support is Phase 14."
            )
        logger.debug(f"LLMClient initialized with backend: {self.backend}")

    def complete(
        self,
        model_env_key: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        expect_json: bool = True,
    ) -> str:
        """
        Make a completion request.

        Args:
            model_env_key: The .env key for the model name, e.g. "nvidia_model_red_planner"
                          This is looked up in config, NOT passed as the model name directly.
            messages: OpenAI-format message list [{"role": "system", "content": "..."}, ...]
            max_tokens: Max tokens in response (default 512)
            temperature: 0.7 default — enough creativity for strategic reasoning
            expect_json: If True, adds JSON reminder and validates response.

        Returns:
            The model's response as a string.
            On repeated failure, returns a safe fallback JSON string (WAIT action).

        Raises:
            Never raises — all exceptions are caught, logged, and a fallback is returned.
        """
        model_name = getattr(config, model_env_key.lower(), None)
        if model_name is None:
            logger.error(f"Model key '{model_env_key}' not found in config. Using fallback.")
            return self._fallback_action()

        if expect_json:
            messages = [dict(m) for m in messages]  # shallow copy each dict
            messages[-1] = dict(messages[-1])
            messages[-1]["content"] = (
                messages[-1]["content"]
                + "\n\nRespond ONLY with valid JSON. No markdown, no explanation outside the JSON."
            )

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                elapsed = time.time() - start
                content = response.choices[0].message.content.strip()

                logger.debug(
                    f"LLM response: model={model_name}, "
                    f"tokens={response.usage.total_tokens if response.usage else '?'}, "
                    f"time={elapsed:.2f}s"
                )

                if expect_json:
                    # Validate JSON — strip markdown fences if model added them
                    content = self._strip_json_fences(content)
                    try:
                        json.loads(content)  # validate only, return string
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Model returned invalid JSON (attempt {attempt}): {e}. "
                            f"Raw: {content[:200]}"
                        )
                        if attempt == self.MAX_RETRIES:
                            return self._fallback_action()
                        time.sleep(self.RETRY_BASE_DELAY * attempt)
                        continue

                return content

            except RateLimitError:
                wait = self.RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"Rate limit hit (attempt {attempt}/{self.MAX_RETRIES}). Waiting {wait}s."
                )
                time.sleep(wait)

            except APITimeoutError:
                logger.warning(f"API timeout (attempt {attempt}/{self.MAX_RETRIES}).")
                time.sleep(self.RETRY_BASE_DELAY * attempt)

            except APIConnectionError as e:
                logger.error(f"API connection error: {e}")
                time.sleep(self.RETRY_BASE_DELAY * attempt)

            except Exception as e:
                logger.error(
                    f"Unexpected LLM error (attempt {attempt}): {type(e).__name__}: {e}"
                )
                if attempt == self.MAX_RETRIES:
                    return self._fallback_action()
                time.sleep(self.RETRY_BASE_DELAY * attempt)

        return self._fallback_action()

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        """Remove ```json ... ``` or ``` ... ``` wrapping if present."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        return text

    @staticmethod
    def _fallback_action() -> str:
        """
        Returns a safe WAIT action JSON when the API fails after all retries.
        """
        return json.dumps({
            "action_type": "wait",
            "target_node": None,
            "target_file": None,
            "reasoning": "API unavailable — defaulting to WAIT for safety."
        })


# Singleton — one client shared across all agents
_client_instance: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Returns the singleton LLMClient. Creates it on first call."""
    global _client_instance
    if _client_instance is None:
        _client_instance = LLMClient()
    return _client_instance
