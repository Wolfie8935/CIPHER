"""
cipher/utils/lora_client.py

Singleton LoRA inference client for fine-tuned CIPHER specialist agents.
Loads base model (fp16/fp32) + LoRA adapter on first call per adapter path,
then reuses cached models for subsequent calls.

Used when LLM_MODE=hybrid for any agent that has a trained adapter.
All other agents use HuggingFace Inference API via llm_client.py.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Optional

from cipher.utils.logger import get_logger

logger = get_logger(__name__)

# Base model — normal unsloth instruction model (NOT bnb-4bit quantized)
_BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"

# Default adapter path (legacy fallback)
_DEFAULT_ADAPTER_PATH = os.path.join("red trained", "cipher-red-planner-v1")

# Global lock — prevents parallel threads from racing on the first HF import
# (transformers internal state is not thread-safe during initial import)
_load_lock = threading.Lock()


class LoRAClient:
    """
    Thread-safe singleton LoRA inference client.

    Maintains a per-adapter-path model cache so each specialist loads once
    and subsequent calls reuse the cached (model, tokenizer) pair.

    Fixes vs previous version:
    - _load_lock prevents AutoModelForCausalLM import race in parallel threads
    - _models dict caches per adapter_path (no shared model across specialists)
    - complete() accepts team= arg so _text_to_action_json uses valid actions
    - dtype= replaces deprecated torch_dtype=
    - generation_config.max_length cleared to suppress HF warning
    """

    _instance: Optional["LoRAClient"] = None
    # adapter_path → (model, tokenizer)
    _models: dict[str, tuple] = {}

    def __new__(cls) -> "LoRAClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ── Model loading ───────────────────────────────────────────────────────

    def _load(self, adapter_path: str) -> None:
        """
        Lazy-load base model + LoRA adapter for the given adapter_path.
        No-op if this adapter is already in the cache.
        Thread-safe: uses _load_lock so parallel agent calls don't race.
        """
        if adapter_path in self._models:
            return

        with _load_lock:
            # Double-check after acquiring lock (another thread may have loaded meanwhile)
            if adapter_path in self._models:
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            resolved = str(Path(adapter_path).resolve())

            if not Path(resolved).exists():
                raise FileNotFoundError(
                    f"LoRA adapter not found: {resolved}\n"
                    f"Check that the adapter folder exists."
                )

            # ── Tokenizer ──────────────────────────────────────────────────
            logger.info(f"[LoRA] Loading tokenizer from: {resolved}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(resolved)
            except Exception as e:
                logger.warning(f"[LoRA] Adapter tokenizer failed ({e}), using base tokenizer.")
                tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL_ID)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # ── Base model ─────────────────────────────────────────────────
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

            logger.info(f"[LoRA] Loading base model {_BASE_MODEL_ID} on {device} ({dtype})")
            print(f"\n  🔴 [LoRA] Loading specialist: {Path(resolved).name}")
            print(f"  🔴 [LoRA] Device: {device.upper()} | dtype: {str(dtype).split('.')[-1]}")

            base_model = AutoModelForCausalLM.from_pretrained(
                _BASE_MODEL_ID,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            # ── LoRA adapter ───────────────────────────────────────────────
            logger.info(f"[LoRA] Attaching adapter: {resolved}")
            model = PeftModel.from_pretrained(
                base_model,
                resolved,
                low_cpu_mem_usage=True,
            )
            model.eval()

            # Clear conflicting max_length so max_new_tokens is the sole limit
            if hasattr(model, "generation_config") and hasattr(model.generation_config, "max_length"):
                model.generation_config.max_length = None

            self._models[adapter_path] = (model, tokenizer)
            print(f"  🔴 [LoRA] {Path(resolved).name} READY ✓\n")

    # ── Inference ───────────────────────────────────────────────────────────

    def complete(
        self,
        messages: list[dict[str, str]],
        adapter_path: str = _DEFAULT_ADAPTER_PATH,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        team: str = "red",
    ) -> str:
        """
        Run inference with the specialist LoRA for the given adapter_path.

        Args:
            messages:       OpenAI-format message list.
            adapter_path:   Path to the adapter folder (one per specialist).
            max_new_tokens: Max tokens to generate.
            temperature:    Sampling temperature.
            team:           "red" or "blue" — controls action fallback type.

        Returns:
            Always a syntactically valid JSON string with an action_type field.
        """
        import torch

        self._load(adapter_path)
        model, tokenizer = self._models[adapter_path]

        # Build prompt from chat template if available
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_msg   = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens (strip the prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        logger.info(f"[LoRA] {Path(adapter_path).name} raw output: {raw[:100]!r}")

        # Coerce to valid JSON — extract embedded JSON, then keyword fallback
        json_str = self._extract_json(raw)
        if json_str is None:
            json_str = self._text_to_action_json(raw, team=team)
        return json_str

    # ── JSON helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """
        Bracket-count scan — returns the first syntactically valid JSON object
        found anywhere in text (handles prose wrapping, markdown fences, etc).
        Returns None if nothing valid is found.
        """
        # Strip markdown fences first
        stripped = re.sub(r"```(?:json)?", "", text).strip()
        for candidate_text in (stripped, text):
            start = candidate_text.find("{")
            if start == -1:
                continue
            depth = 0
            in_str = False
            esc = False
            for i, ch in enumerate(candidate_text[start:], start):
                if esc:
                    esc = False
                    continue
                if ch == "\\" and in_str:
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                if not in_str:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = candidate_text[start: i + 1]
                            try:
                                json.loads(candidate)
                                return candidate
                            except json.JSONDecodeError:
                                # Try next { after this one
                                nxt = candidate_text.find("{", i + 1)
                                if nxt == -1:
                                    break
                                start = nxt
                                depth = 0
                                in_str = False
                                esc = False
        return None

    @staticmethod
    def _text_to_action_json(text: str, team: str = "red") -> str:
        """
        Keyword-based fallback for when the model outputs plain text.
        Returns team-appropriate actions only — RED models get RED actions,
        BLUE models get BLUE actions (prevents 'move'/'wait' invalid for BLUE).
        """
        t = text.lower()

        if team == "blue":
            # BLUE valid actions: analyze_anomaly, place_honeypot, raise_alert,
            # reconstruct_path, tamper_dead_drop, trigger_false_escalation, stand_down
            if "honeypot" in t or "trap" in t or "deploy" in t:
                m = re.search(r"node[_\s]?(\d+)", t)
                return json.dumps({
                    "action_type": "place_honeypot",
                    "target_node": int(m.group(1)) if m else None,
                    "target_file": None,
                    "reasoning": text[:200],
                })
            if "alert" in t or "raise" in t:
                return json.dumps({
                    "action_type": "raise_alert",
                    "target_node": None, "target_file": None,
                    "reasoning": text[:200],
                })
            if "analyze" in t or "anomaly" in t or "scan" in t or "monitor" in t or "investigate" in t:
                return json.dumps({
                    "action_type": "analyze_anomaly",
                    "target_node": None, "target_file": None,
                    "reasoning": text[:200],
                })
            if "reconstruct" in t or "path" in t or "trace" in t:
                return json.dumps({
                    "action_type": "reconstruct_path",
                    "target_node": None, "target_file": None,
                    "reasoning": text[:200],
                })
            if "tamper" in t or "drop" in t:
                return json.dumps({
                    "action_type": "tamper_dead_drop",
                    "target_node": None, "target_file": None,
                    "reasoning": text[:200],
                })
            # Default BLUE safe action
            return json.dumps({
                "action_type": "stand_down",
                "target_node": None, "target_file": None,
                "reasoning": text[:200],
            })

        else:
            # RED valid actions: move, scan, escalate_privileges, exfiltrate,
            # read_dead_drop, write_dead_drop, plant_false_trail, wait, abort
            if "exfiltrat" in t or "steal" in t:
                m = re.search(r"(?:file|target)[:\s]+([A-Za-z0-9_\-\.]+)", text, re.IGNORECASE)
                return json.dumps({
                    "action_type": "exfiltrate",
                    "target_node": None,
                    "target_file": m.group(1) if m else None,
                    "reasoning": text[:200],
                })
            if "move" in t or "lateral" in t or "pivot" in t or "navigate" in t or "travel" in t:
                m = re.search(r"node[_\s]?(\d+)", t)
                return json.dumps({
                    "action_type": "move",
                    "target_node": int(m.group(1)) if m else None,
                    "target_file": None,
                    "reasoning": text[:200],
                })
            if "scan" in t or "recon" in t or "survey" in t or "discover" in t:
                return json.dumps({
                    "action_type": "scan",
                    "target_node": None, "target_file": None,
                    "reasoning": text[:200],
                })
            if "escalat" in t or "privilege" in t or "root" in t:
                return json.dumps({
                    "action_type": "escalate_privileges",
                    "target_node": None, "target_file": None,
                    "reasoning": text[:200],
                })
            if "read" in t and "drop" in t:
                return json.dumps({
                    "action_type": "read_dead_drop",
                    "target_node": None, "target_file": None,
                    "reasoning": text[:200],
                })
            if "write" in t and "drop" in t:
                return json.dumps({
                    "action_type": "write_dead_drop",
                    "target_node": None, "target_file": None,
                    "reasoning": text[:200],
                })
            # Default RED safe action
            return json.dumps({
                "action_type": "wait",
                "target_node": None, "target_file": None,
                "reasoning": text[:200],
            })
