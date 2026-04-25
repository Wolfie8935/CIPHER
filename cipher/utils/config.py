"""
Configuration loader for CIPHER.

``.env`` is intentionally minimal: only ``HF_TOKEN`` and ``HF_BASE_URL`` are
expected there for Hugging Face Inference. Everything else (LLM models, LoRA
paths, subagent caps, ``LLM_MODE`` / ``LLM_BACKEND`` defaults, etc.) lives as
``Field`` defaults on ``CipherConfig`` below.

You can still override any field via process environment (Pydantic naming:
``LLM_MODE``, ``HF_MODEL_RED_PLANNER``, …) without editing this file.

Owns: typed configuration and optional ``.env`` loading.
Does NOT own: domain logic or API calls.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class CipherConfig(BaseSettings):
    """
    Centralized configuration for the entire CIPHER system.

    All values are read from the .env file at project root.
    Missing required values cause an immediate, descriptive validation error
    at startup — not a cryptic KeyError deep in execution.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Backend ──────────────────────────────────────────────
    llm_backend: str = Field(
        "hf",
        description="LLM provider: hf | local | hybrid",
    )
    llm_mode: str = Field(
        "stub",
        description="stub | live | hybrid (runtime may set LLM_MODE in os.environ)",
    )

    # ── HuggingFace Inference API (credentials: .env only HF_TOKEN + HF_BASE_URL) ─
    hf_token: str = Field("", description="HuggingFace API token")
    hf_base_url: str = Field(
        "https://router.huggingface.co/v1",
        description="HF Inference / router OpenAI-compatible base URL",
    )

    # ── Model assignments — RED ───────────────────────────────────
    hf_model_red_planner: str = Field(
        "meta-llama/Llama-3.1-8B-Instruct",
        description="HF model for RED Planner agent",
    )
    hf_model_red_analyst: str = Field(
        "meta-llama/Llama-3.1-8B-Instruct",
        description="HF model for RED Analyst agent",
    )
    hf_model_red_operative: str = Field(
        "meta-llama/Llama-3.1-8B-Instruct",
        description="HF model for RED Operative agent",
    )
    hf_model_red_exfil: str = Field(
        "meta-llama/Llama-3.1-8B-Instruct",
        description="HF model for RED Exfiltrator agent",
    )

    # ── Model assignments — BLUE ──────────────────────────────────
    hf_model_blue_surv: str = Field(
        "meta-llama/Llama-3.1-8B-Instruct",
        description="HF model for BLUE Surveillance agent",
    )
    hf_model_blue_hunter: str = Field(
        "meta-llama/Llama-3.1-8B-Instruct",
        description="HF model for BLUE Threat Hunter agent",
    )
    hf_model_blue_deceiver: str = Field(
        "meta-llama/Llama-3.1-8B-Instruct",
        description="HF model for BLUE Deception Architect agent",
    )
    hf_model_blue_forensics: str = Field(
        "meta-llama/Llama-3.1-8B-Instruct",
        description="HF model for BLUE Forensics agent",
    )

    # ── Oversight Auditor ─────────────────────────────────────────
    hf_model_oversight: str = Field(
        "meta-llama/Llama-3.1-8B-Instruct",
        description="HF model for Oversight Auditor agent",
    )

    # ── Your fine-tuned model ─────────────────────────────────────
    hf_model_red_custom: str = Field(
        "wolfie8935/cipher-red-planner-grpo",
        description="Fine-tuned RED Planner repo on HuggingFace",
    )

    # ── Local model server (hybrid mode) ─────────────────────────
    local_model_url: str = Field(
        "http://localhost:1234/v1",
        description="OpenAI-compatible local model server URL",
    )
    local_model_name: str = Field(
        "cipher-red-planner",
        description="Model name as registered in local server",
    )

    # ── Hugging Face repos (uploads / demo traces) ────────────────
    hf_repo_id: str = Field(
        "wolfie8935/cipher-specialists",
        description="Default HF model dataset repo for specialist uploads",
    )
    hf_traces_repo: str = Field(
        "wolfie8935/cipher-traces",
        description="HF dataset repo for optional demo episode traces",
    )

    # ── LoRA adapter paths (hybrid / local inference) ──────────────
    red_commander_lora_path: str = Field(
        "red trained/cipher-red-commander-v1",
        description="RED commander LoRA directory",
    )
    blue_commander_lora_path: str = Field(
        "blue trained/cipher-blue-commander-v1",
        description="BLUE commander LoRA directory",
    )
    red_planner_lora_path: str = Field(
        "red trained/cipher-red-planner-v1",
        description="RED planner LoRA directory",
    )
    red_analyst_lora_path: str = Field(
        "red trained/cipher-red-analyst-v1",
        description="RED analyst LoRA directory",
    )
    blue_surveillance_lora_path: str = Field(
        "blue trained/cipher-blue-surveillance-v1",
        description="BLUE surveillance LoRA directory",
    )
    blue_threat_hunter_lora_path: str = Field(
        "blue trained/cipher-blue-threat-hunter-v1",
        description="BLUE threat hunter LoRA directory",
    )

    # ── Storyteller (optional narrative TTS path) ───────────────────
    storyteller_hf_model: str = Field(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        description="HF model id for storyteller completions",
    )

    # ── Environment parameters ────────────────────────────────────
    env_graph_size: int = Field(50, description="Number of nodes in enterprise network")
    env_max_steps: int = Field(200, description="Max steps per episode")
    env_context_reset_interval: int = Field(
        40, description="Steps between RED context resets"
    )
    env_honeypot_density: float = Field(
        0.15, description="Fraction of assets that are honeypots"
    )
    env_anomaly_feed_noise: float = Field(
        0.10, description="BLUE false positive rate in raw feed (reduced from 0.2 to give BLUE usable signal)"
    )
    env_dead_drop_max_tokens: int = Field(
        512, description="Max size of each dead drop file"
    )
    env_trap_budget_red: int = Field(
        3, description="Max traps RED can place per episode"
    )
    env_trap_budget_blue: int = Field(
        5, description="Max honeypots BLUE can maintain"
    )

    # ── Reward weights ────────────────────────────────────────────
    reward_red_exfil_weight: float = Field(0.5)
    reward_red_stealth_weight: float = Field(0.3)
    reward_red_memory_efficiency_weight: float = Field(0.2)
    reward_blue_detection_weight: float = Field(0.4)
    reward_blue_speed_weight: float = Field(0.3)
    reward_blue_honeypot_weight: float = Field(0.3)

    # ── Training ──────────────────────────────────────────────────
    training_episodes: int = Field(1000)
    training_log_interval: int = Field(10)
    training_checkpoint_interval: int = Field(100)

    # ── Dashboard ─────────────────────────────────────────────────
    dashboard_port: int = Field(8050)
    dashboard_live_port: int = Field(8051)
    dashboard_live_update_interval: int = Field(2000, description="Milliseconds")

    # ── Commander / Subagent architecture (v2) ───────────────────
    cipher_agent_arch: str = Field(
        "v2",
        description="Agent architecture: 'v1' = legacy 4+4 fixed roster, 'v2' = commander+subagents",
    )
    env_max_subagents_red: int = Field(
        6, description="Max concurrent RED subagents alive at once"
    )
    env_max_subagents_blue: int = Field(
        6, description="Max concurrent BLUE subagents alive at once"
    )
    env_subagent_spawn_budget_red: int = Field(
        12, description="Max RED subagent spawns per episode"
    )
    env_subagent_spawn_budget_blue: int = Field(
        12, description="Max BLUE subagent spawns per episode"
    )
    env_subagent_default_lifespan: int = Field(
        5, description="Default steps before a subagent is auto-dismissed"
    )
    env_reward_delegation_enabled: bool = Field(
        False,
        description="When true, adds delegation_efficiency bonus and spawn_cost penalty to rewards",
    )
    hf_model_red_commander: str = Field(
        "meta-llama/Llama-3.1-8B-Instruct",
        description="HF model for the RED Commander (top-level RED agent)",
    )
    hf_model_blue_commander: str = Field(
        "meta-llama/Llama-3.1-8B-Instruct",
        description="HF model for the BLUE Commander (top-level BLUE agent)",
    )

    def model_post_init(self, __context) -> None:
        if self.dashboard_live_update_interval <= 10:
            self.dashboard_live_update_interval *= 1000

    # ── Derived paths ─────────────────────────────────────────────
    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def drop_vault_dir(self) -> Path:
        return _PROJECT_ROOT / "cipher" / "memory" / "drop_vault"

    @property
    def episode_traces_dir(self) -> Path:
        return _PROJECT_ROOT / "episode_traces"


config = CipherConfig()

# Legacy env var names (role_profiles / BaseAgent) → CipherConfig field names.
_LORA_ENV_TO_FIELD: dict[str, str] = {
    "RED_COMMANDER_LORA_PATH": "red_commander_lora_path",
    "BLUE_COMMANDER_LORA_PATH": "blue_commander_lora_path",
    "RED_PLANNER_LORA_PATH": "red_planner_lora_path",
    "RED_ANALYST_LORA_PATH": "red_analyst_lora_path",
    "BLUE_SURVEILLANCE_LORA_PATH": "blue_surveillance_lora_path",
    "BLUE_THREAT_HUNTER_LORA_PATH": "blue_threat_hunter_lora_path",
}


def resolve_lora_adapter_path(env_key: str, default_path: str) -> str:
    """Resolve a LoRA path from ``CipherConfig`` (no ``os.getenv`` for these keys)."""
    field_name = _LORA_ENV_TO_FIELD.get(env_key)
    if not field_name:
        return default_path
    val = getattr(config, field_name, None)
    s = str(val).strip() if val is not None else ""
    return s or default_path
