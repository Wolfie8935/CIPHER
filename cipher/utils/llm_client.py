"""
cipher/utils/llm_client.py

Unified LLM client for CIPHER.

Supported backends (set LLM_BACKEND in .env):
  hf      — HuggingFace Inference API (default, OpenAI-compatible)
  local   — OpenAI-compatible local server (LM Studio / vllm)
  hybrid  — RED Planner uses LoRA from disk, all other agents use HF

This is the ONLY file that imports openai or makes HTTP calls to any LLM.
"""
from __future__ import annotations

import time
import json
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from cipher.utils.config import config
from cipher.utils.logger import get_logger
from cipher.utils.llm_mode import get_llm_mode

logger = get_logger(__name__)

# Model keys that route to the LoRA model in hybrid mode
_LOCAL_KEYS_IN_HYBRID = {"hf_model_red_planner"}


class LLMClient:
    """
    Routes LLM calls to HuggingFace Inference API, a local server, or both (hybrid).

    Handles:
    - Backend routing per agent role
    - Retry logic with exponential backoff (3 attempts)
    - Rate limit handling (waits and retries on 429)
    - Malformed response handling (safe WAIT fallback)
    """

    MAX_RETRIES       = 3
    RETRY_BASE_DELAY  = 1.0
    REQUEST_TIMEOUT   = 25.0

    def __init__(self) -> None:
        configured_backend = str(config.llm_backend).strip().lower()
        runtime_mode = get_llm_mode()

        if runtime_mode == "hybrid":
            self.backend = "hybrid"
        elif configured_backend in {"hf", "local", "hybrid"}:
            self.backend = configured_backend
        else:
            self.backend = "hf"

        if self.backend in {"hf", "hybrid"}:
            self._hf = OpenAI(
                base_url=config.hf_base_url,
                api_key=config.hf_token or "hf_placeholder",
                timeout=self.REQUEST_TIMEOUT,
            )
            self._local = None

        elif self.backend == "local":
            self._hf = None
            self._local = OpenAI(
                base_url=config.local_model_url,
                api_key="local",
                timeout=self.REQUEST_TIMEOUT,
            )

        if self.backend == "hybrid":
            self._lora_adapter_path = config.red_planner_lora_path

        logger.info(f"LLMClient initialized: backend={self.backend}")
        if self.backend == "hybrid":
            logger.info(f"Hybrid: RED Planner -> LoRA ({self._lora_adapter_path}), others -> HF.")

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
        Make a completion request routed to the correct backend.

        Args:
            model_env_key: Config attribute name, e.g. "hf_model_red_planner".
            messages: OpenAI-format message list.
            max_tokens: Max tokens in response.
            temperature: Sampling temperature.
            expect_json: Appends JSON reminder and validates response.

        Returns:
            Model response string. Never raises — returns fallback on failure.
        """
        if self.backend == "hybrid" and model_env_key.lower() in _LOCAL_KEYS_IN_HYBRID:
            return self._lora_complete(messages, team=team, max_tokens=max_tokens, temperature=temperature)

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
                    f"LLM: model={model_name}, "
                    f"tokens={response.usage.total_tokens if response.usage else '?'}, "
                    f"time={elapsed:.2f}s"
                )

                if expect_json:
                    content = self._strip_json_fences(content)
                    try:
                        json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON (attempt {attempt}): {e}. Raw: {content[:200]}")
                        if attempt == self.MAX_RETRIES:
                            return self._fallback_action(team)
                        time.sleep(self.RETRY_BASE_DELAY * attempt)
                        continue

                return content

            except RateLimitError:
                wait = self.RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"Rate limit (attempt {attempt}). Waiting {wait:.1f}s.")
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
                if "404" in err_str or "NotFoundError" in type(e).__name__:
                    logger.error(
                        f"Model not found (404) for '{model_name}'. "
                        "Check HF_MODEL_* values in .env. Falling back immediately."
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
        """Return (client, model_name) for the given key and current backend."""
        key = model_env_key.lower()

        if self.backend in {"hf", "hybrid"}:
            model_name = getattr(config, key, None)
            return self._hf, model_name

        elif self.backend == "local":
            return self._local, config.local_model_name

        return None, None

    def _lora_complete(self, messages: list[dict[str, str]], team: str = "red",
                       max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Run inference via LoRAClient (loads adapter from disk, no server required)."""
        try:
            from cipher.utils.lora_client import LoRAClient
            lora = LoRAClient()
            return lora.complete(
                messages,
                adapter_path=self._lora_adapter_path,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"LoRA inference failed: {e}")
            return self._fallback_action(team)

    @staticmethod
    def _fallback_action(team: str = "red") -> str:
        action = "wait" if team == "red" else "stand_down"
        return json.dumps({
            "action_type": action,
            "target_node": None,
            "target_file": None,
            "reasoning": f"LLM unavailable — defaulting to {action.upper()}.",
        })

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        """Strip markdown code fences and repair bare newlines inside JSON strings."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
        result: list[str] = []
        in_string = False
        escape_next = False
        for ch in text:
            if escape_next:
                result.append(ch); escape_next = False
            elif ch == "\\":
                result.append(ch); escape_next = True
            elif ch == '"':
                result.append(ch); in_string = not in_string
            elif in_string and ch in ("\n", "\r"):
                result.append(" ")
            else:
                result.append(ch)
        final = "".join(result).strip()
        if in_string:
            final += '"'
        if final.startswith("{") and not final.endswith("}"):
            final += "}"
        return final


_llm_client_instance: "LLMClient | None" = None


def get_llm_client() -> "LLMClient":
    """Return a singleton LLMClient instance."""
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance
