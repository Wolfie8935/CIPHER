"""
cipher/utils/lora_client.py

Singleton inference client for the fine-tuned RED Planner LoRA specialist.
Loads base model (fp16) + LoRA adapter on first call, then reuses for all steps.

Used ONLY when LLM_MODE=hybrid AND agent is red planner.
All other agents continue to use the NVIDIA NIM API via llm_client.py.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from cipher.utils.logger import get_logger

logger = get_logger(__name__)

# Base model to load from HuggingFace (fp16, no 4-bit needed for inference)
_BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"

# Default adapter path relative to project root
_DEFAULT_ADAPTER_PATH = os.path.join("red trained", "cipher-red-planner")


class LoRAClient:
    """
    Singleton LoRA inference client.

    Loads the base model + LoRA adapter lazily on first call.
    Subsequent calls reuse the loaded model (no re-loading overhead).
    """

    _instance: Optional["LoRAClient"] = None
    _model = None
    _tokenizer = None
    _adapter_path: Optional[str] = None

    def __new__(cls) -> "LoRAClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load(self, adapter_path: str) -> None:
        """Lazy-load base model + LoRA adapter. No-op if already loaded."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        adapter_path = str(Path(adapter_path).resolve())
        self._adapter_path = adapter_path

        if not Path(adapter_path).exists():
            raise FileNotFoundError(
                f"LoRA adapter not found at: {adapter_path}\n"
                f"Make sure the 'red trained/cipher-red-planner' folder exists."
            )

        logger.info(f"[LoRA] Loading tokenizer from adapter: {adapter_path}")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        except Exception as e:
            logger.warning(f"[LoRA] Failed to load tokenizer from adapter ({e}). Falling back to base tokenizer.")
            self._tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL_ID)
            
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info(f"[LoRA] Loading base model: {_BASE_MODEL_ID} on {device} ({dtype})")
        print(f"\n  🔴 [LoRA] Loading RED Planner specialist ({_BASE_MODEL_ID})...")
        print(f"  🔴 [LoRA] Device: {device.upper()} | dtype: {str(dtype).split('.')[-1]}")
        print(f"  🔴 [LoRA] (Note: First run may take 2-5 mins to download ~2.5GB)")

        base_model = AutoModelForCausalLM.from_pretrained(
            _BASE_MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        logger.info(f"[LoRA] Loading LoRA adapter: {adapter_path}")
        print(f"  🔴 [LoRA] Attaching LoRA adapter (r=16)...")
        self._model = PeftModel.from_pretrained(
            base_model, 
            adapter_path,
            low_cpu_mem_usage=True
        )
        self._model.eval()
        print(f"  🔴 [LoRA] RED Planner specialist READY ✓\n")

    def complete(
        self,
        messages: list[dict[str, str]],
        adapter_path: str = _DEFAULT_ADAPTER_PATH,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Run inference using the fine-tuned RED Planner LoRA model.

        Args:
            messages: OpenAI-format messages (system + history + user).
            adapter_path: Path to the LoRA adapter folder.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text string (to be parsed as JSON action).
        """
        import torch

        self._load(adapter_path)

        # Build prompt using the tokenizer's chat template
        if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: system + last user message
            system_msg = next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            )
            user_msg = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
            )
            prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens (strip the prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
