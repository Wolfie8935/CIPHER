"""
Configuration loader for CIPHER.

Reads environment variables from .env (only HF_TOKEN + API_BASE_URL are required
there). All other settings have production-ready defaults here.
Every other module imports from config, never from os directly.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class CipherConfig(BaseSettings):
    """
    Centralized configuration for CIPHER.

    Only HF_TOKEN and API_BASE_URL are expected in .env.
    Everything else defaults to sensible values for both local and HF Spaces.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── HuggingFace credentials (the only things that MUST be in .env) ───
    hf_token: str = Field("", description="HuggingFace API token")
    # API_BASE_URL in .env → api_base_url here.
    # Also accepts legacy HF_BASE_URL so old .env files still work.
    api_base_url: str = Field(
        "https://router.huggingface.co/v1",
        description="LLM inference API base URL",
    )
    # Legacy alias kept so existing code that reads config.hf_base_url still works.
    hf_base_url: str = Field(
        "https://router.huggingface.co/v1",
        description="Alias for api_base_url (legacy)",
    )

    # ── LLM Backend (defaults — override via env var if needed) ──────────
    llm_backend: str = Field("hf", description="LLM provider: hf | local | hybrid")
    llm_mode: str = Field("stub", description="stub | live | hybrid")

    # ── Agent architecture ────────────────────────────────────────────────
    cipher_agent_arch: str = Field(
        "v2",
        description="v2 = commander+subagents (default), v1 = legacy 4+4",
    )

    # ── Commander models ──────────────────────────────────────────────────
    hf_model_red_commander: str = Field(
        "meta-llama/Llama-3.1-8B-Instruct",
        description="HF model for RED Commander",
    )
    hf_model_blue_commander: str = Field(
        "meta-llama/Llama-3.1-8B-Instruct",
        description="HF model for BLUE Commander",
    )

    # ── Subagent specialist models ────────────────────────────────────────
    hf_model_red_planner: str = Field("meta-llama/Llama-3.1-8B-Instruct")
    hf_model_red_analyst: str = Field("meta-llama/Llama-3.1-8B-Instruct")
    hf_model_red_operative: str = Field("meta-llama/Llama-3.1-8B-Instruct")
    hf_model_red_exfil: str = Field("meta-llama/Llama-3.1-8B-Instruct")
    hf_model_blue_surv: str = Field("meta-llama/Llama-3.1-8B-Instruct")
    hf_model_blue_hunter: str = Field("meta-llama/Llama-3.1-8B-Instruct")
    hf_model_blue_deceiver: str = Field("meta-llama/Llama-3.1-8B-Instruct")
    hf_model_blue_forensics: str = Field("meta-llama/Llama-3.1-8B-Instruct")
    hf_model_oversight: str = Field("meta-llama/Llama-3.1-8B-Instruct")
    hf_model_red_custom: str = Field(
        "wolfie8935/cipher-red-planner-grpo",
        description="Fine-tuned RED Planner on HuggingFace Hub",
    )

    # ── Local model server (optional, hybrid mode) ────────────────────────
    local_model_url: str = Field("http://localhost:1234/v1")
    local_model_name: str = Field("cipher-red-planner")

    # ── Subagent caps ─────────────────────────────────────────────────────
    env_max_subagents_red: int = Field(6)
    env_max_subagents_blue: int = Field(6)
    env_subagent_spawn_budget_red: int = Field(12)
    env_subagent_spawn_budget_blue: int = Field(12)
    env_subagent_default_lifespan: int = Field(5)
    env_reward_delegation_enabled: bool = Field(False)

    # ── LoRA adapter paths (absolute, resolved from project root) ────────
    # These are hardcoded here — do NOT put them in .env.
    red_commander_lora_path: str = Field(
        str(_PROJECT_ROOT / "red trained" / "cipher-red-commander-v1"),
    )
    blue_commander_lora_path: str = Field(
        str(_PROJECT_ROOT / "blue trained" / "cipher-blue-commander-v1"),
    )
    red_planner_lora_path: str = Field(
        str(_PROJECT_ROOT / "red trained" / "cipher-red-planner-v1"),
    )
    red_analyst_lora_path: str = Field(
        str(_PROJECT_ROOT / "red trained" / "cipher-red-analyst-v1"),
    )
    blue_surveillance_lora_path: str = Field(
        str(_PROJECT_ROOT / "blue trained" / "cipher-blue-surveillance-v1"),
    )
    blue_threat_hunter_lora_path: str = Field(
        str(_PROJECT_ROOT / "blue trained" / "cipher-blue-threat-hunter-v1"),
    )

    # ── Environment parameters ────────────────────────────────────────────
    env_graph_size: int = Field(50)
    env_max_steps: int = Field(200)
    env_context_reset_interval: int = Field(40)
    env_honeypot_density: float = Field(0.15)
    env_anomaly_feed_noise: float = Field(0.10)
    env_dead_drop_max_tokens: int = Field(512)
    env_trap_budget_red: int = Field(3)
    env_trap_budget_blue: int = Field(5)

    # ── Reward weights ────────────────────────────────────────────────────
    reward_red_exfil_weight: float = Field(0.5)
    reward_red_stealth_weight: float = Field(0.3)
    reward_red_memory_efficiency_weight: float = Field(0.2)
    reward_blue_detection_weight: float = Field(0.4)
    reward_blue_speed_weight: float = Field(0.3)
    reward_blue_honeypot_weight: float = Field(0.3)

    # ── Training ──────────────────────────────────────────────────────────
    training_episodes: int = Field(1000)
    training_log_interval: int = Field(10)
    training_checkpoint_interval: int = Field(100)

    # ── Dashboard ─────────────────────────────────────────────────────────
    dashboard_port: int = Field(8050)
    dashboard_live_port: int = Field(8051)
    dashboard_live_update_interval: int = Field(2000)

    @model_validator(mode="after")
    def _sync_url_aliases(self) -> "CipherConfig":
        # Prefer api_base_url if it was set from .env; keep hf_base_url in sync.
        if self.api_base_url and self.api_base_url != "https://router.huggingface.co/v1":
            object.__setattr__(self, "hf_base_url", self.api_base_url)
        elif self.hf_base_url and self.hf_base_url != "https://router.huggingface.co/v1":
            object.__setattr__(self, "api_base_url", self.hf_base_url)
        if self.dashboard_live_update_interval <= 10:
            object.__setattr__(
                self, "dashboard_live_update_interval",
                self.dashboard_live_update_interval * 1000
            )
        return self

    # ── Derived paths ─────────────────────────────────────────────────────
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
