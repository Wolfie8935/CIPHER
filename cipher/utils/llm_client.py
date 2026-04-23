"""
cipher/utils/llm_client.py

Unified LLM client for CIPHER.

Supported backends (set LLM_BACKEND in .env):
  nvidia  — NVIDIA NIM API (default, all agents)
  local   — OpenAI-compatible local server (all agents use LOCAL_MODEL_URL)
  hybrid  — RED Planner uses LOCAL_MODEL_URL, all other agents use NVIDIA NIM
             This is the live-competition mode after RunPod training.

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

# Model env keys that route to the local model in hybrid mode
_LOCAL_KEYS_IN_HYBRID = {"nvidia_model_red_planner"}


class LLMClient:
    """
    Routes LLM calls to NVIDIA NIM, a local server, or both (hybrid).

    Handles:
    - Backend routing per agent role
    - Retry logic with exponential backoff (3 attempts)
    - Rate limit handling (waits and retries on 429)
    - Malformed response handling (safe WAIT fallback)
    - Token counting and logging
    """

    MAX_RETRIES = 3          # up to 3 attempts before fallback
    RETRY_BASE_DELAY = 1.0
    REQUEST_TIMEOUT = 12.0   # 12s per call max; fallback on timeout

    def __init__(self) -> None:
        self.backend = config.llm_backend

        if self.backend == "nvidia":
            _extra_headers = {}
            if "openrouter" in config.nvidia_base_url:
                # OpenRouter requires these headers to identify the calling app.
                _extra_headers = {
                    "HTTP-Referer": "https://github.com/Wolfie8935/CIPHER",
                    "X-Title": "CIPHER-MARL",
                }
            self._nvidia = OpenAI(
                base_url=config.nvidia_base_url,
                api_key=config.nvidia_api_key,
                timeout=self.REQUEST_TIMEOUT,
                default_headers=_extra_headers if _extra_headers else None,
            )
            self._local = None

        elif self.backend == "local":
            self._nvidia = None
            self._local = OpenAI(
                base_url=config.local_model_url,
                api_key="local",  # most local servers accept any non-empty key
                timeout=self.REQUEST_TIMEOUT,
            )

        elif self.backend == "hybrid":
            # RED Planner uses the trained local model; everyone else uses NVIDIA NIM.
            _extra_headers = {}
            if "openrouter" in config.nvidia_base_url:
                _extra_headers = {
                    "HTTP-Referer": "https://github.com/Wolfie8935/CIPHER",
                    "X-Title": "CIPHER-MARL",
                }
            self._nvidia = OpenAI(
                base_url=config.nvidia_base_url,
                api_key=config.nvidia_api_key,
                timeout=self.REQUEST_TIMEOUT,
                default_headers=_extra_headers if _extra_headers else None,
            )
            self._local = OpenAI(
                base_url=config.local_model_url,
                api_key="local",
                timeout=self.REQUEST_TIMEOUT,
            )

        else:
            raise NotImplementedError(
                f"LLM backend '{self.backend}' is not supported. "
                "Valid options: nvidia | local | hybrid"
            )

        logger.debug(f"LLMClient initialized: backend={self.backend}")

    def complete(
        self,
        model_env_key: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        expect_json: bool = True,
        team: str = "red",
    ) -> str:
        """
        Make a completion request, routed to the correct backend.

        Args:
            model_env_key: Config attribute name, e.g. "nvidia_model_red_planner".
                In nvidia/hybrid mode this maps to a model name via config lookup.
                In local/hybrid-local mode the local_model_name is used instead.
            messages: OpenAI-format message list.
            max_tokens: Max tokens in response.
            temperature: Sampling temperature.
            expect_json: If True, appends JSON reminder and validates response.

        Returns:
            Model response as string. Never raises — returns fallback WAIT/STAND_DOWN on failure.
        """
        client, model_name = self._resolve(model_env_key)
        if client is None or model_name is None:
            logger.error(f"Cannot resolve client/model for key '{model_env_key}'.")
            return self._fallback_action(team)

        if expect_json:
            messages = [dict(m) for m in messages]
            messages[-1] = dict(messages[-1])
            messages[-1]["content"] = (
                messages[-1]["content"]
                + "\n\nRespond ONLY with valid JSON. No markdown, no explanation outside the JSON."
            )

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                start = time.time()
                response = client.chat.completions.create(
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
                    content = self._strip_json_fences(content)
                    try:
                        json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Invalid JSON (attempt {attempt}): {e}. Raw: {content[:200]}"
                        )
                        if attempt == self.MAX_RETRIES:
                            return self._fallback_action(team)
                        time.sleep(self.RETRY_BASE_DELAY * attempt)
                        continue

                return content

            except RateLimitError:
                wait = self.RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"Rate limit (attempt {attempt}). Waiting {wait}s.")
                if attempt < self.MAX_RETRIES:
                    time.sleep(wait)

            except APITimeoutError:
                logger.warning(f"API timeout (attempt {attempt}).")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_BASE_DELAY * attempt)

            except APIConnectionError as e:
                logger.error(f"API connection error: {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_BASE_DELAY * attempt)

            except Exception as e:
                err_str = str(e)
                # Fast-fail on 404: bad model name will never succeed — don't waste retries
                if "404" in err_str or "NotFoundError" in type(e).__name__:
                    logger.error(
                        f"Model not found (404) for '{model_name}'. "
                        f"Check NVIDIA_MODEL_* values in .env. Falling back immediately."
                    )
                    return self._fallback_action(team)
                logger.error(f"Unexpected LLM error (attempt {attempt}): {type(e).__name__}: {e}")
                if attempt == self.MAX_RETRIES:
                    return self._fallback_action(team)
                time.sleep(self.RETRY_BASE_DELAY * attempt)

        return self._fallback_action(team)


    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _resolve(self, model_env_key: str) -> tuple[OpenAI | None, str | None]:
        """Return (client, model_name) for the given model key and current backend."""
        key = model_env_key.lower()

        if self.backend == "nvidia":
            model_name = getattr(config, key, None)
            return self._nvidia, model_name

        elif self.backend == "local":
            return self._local, config.local_model_name

        elif self.backend == "hybrid":
            if key in _LOCAL_KEYS_IN_HYBRID:
                return self._local, config.local_model_name
            model_name = getattr(config, key, None)
            return self._nvidia, model_name

        return None, None

    @staticmethod
    def _fallback_action(team: str = "red") -> str:
        """Return a safe WAIT/STAND_DOWN action JSON as a fallback when the LLM fails."""
        # BLUE team does NOT have a 'wait' action — use stand_down instead
        action = "wait" if team == "red" else "stand_down"
        return json.dumps({
            "action_type": action,
            "target_node": None,
            "target_file": None,
            "reasoning": f"LLM unavailable — defaulting to {action.upper()}.",
        })

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        """Strip markdown code fences from LLM response."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()
        return text


_llm_client_instance: "LLMClient | None" = None


def get_llm_client() -> "LLMClient":
    """Return a singleton LLMClient instance."""
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance