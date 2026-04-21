"""
Configuration loader for CIPHER.

Reads all environment variables from .env and exposes them as typed attributes.
Every other module imports from config, never from os directly.

Owns: environment variable loading, validation, and typed access.
Does NOT own: any domain logic, file I/O beyond .env, or API calls.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve the .env path relative to the project root (two levels up from this file)
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
        "nvidia",
        description="LLM provider: nvidia | huggingface | openai",
    )
    llm_mode: str = Field(
        "stub",
        description="LLM mode: stub (random actions, no API cost) | live (real API calls)",
    )
    nvidia_api_key: str = Field(
        "nvapi-placeholder-key-for-phase1",
        description="NVIDIA NIM API key",
    )
    nvidia_base_url: str = Field(
        "https://integrate.api.nvidia.com/v1",
        description="NVIDIA NIM base URL",
    )

    # ── Model assignments — RED ──────────────────────────────────
    nvidia_model_red_planner: str = Field(
        "meta/llama-3.1-70b-instruct",
        description="Model for RED Planner agent",
    )
    nvidia_model_red_analyst: str = Field(
        "meta/llama-3.1-70b-instruct",
        description="Model for RED Analyst agent",
    )
    nvidia_model_red_operative: str = Field(
        "meta/llama-3.1-8b-instruct",
        description="Model for RED Operative agent",
    )
    nvidia_model_red_exfil: str = Field(
        "meta/llama-3.1-8b-instruct",
        description="Model for RED Exfiltrator agent",
    )

    # ── Model assignments — BLUE ─────────────────────────────────
    nvidia_model_blue_surv: str = Field(
        "meta/llama-3.1-8b-instruct",
        description="Model for BLUE Surveillance agent",
    )
    nvidia_model_blue_hunter: str = Field(
        "meta/llama-3.1-70b-instruct",
        description="Model for BLUE Threat Hunter agent",
    )
    nvidia_model_blue_deceiver: str = Field(
        "meta/llama-3.1-70b-instruct",
        description="Model for BLUE Deception Architect agent",
    )
    nvidia_model_blue_forensics: str = Field(
        "meta/llama-3.1-70b-instruct",
        description="Model for BLUE Forensics agent",
    )
    nvidia_model_oversight: str = Field(
        "meta/llama-3.1-70b-instruct",
        description="Model for Oversight Auditor agent",
    )

    # ── HuggingFace (optional — Phase 14 only) ───────────────────
    hf_token: str = Field("", description="HuggingFace token (Phase 14 only)")
    hf_model_red: str = Field("", description="HuggingFace model for RED team")
    hf_model_blue: str = Field("", description="HuggingFace model for BLUE team")

    # ── Environment parameters ───────────────────────────────────
    env_graph_size: int = Field(50, description="Number of nodes in enterprise network")
    env_max_steps: int = Field(200, description="Max steps per episode")
    env_context_reset_interval: int = Field(
        40, description="Steps between RED context resets"
    )
    env_honeypot_density: float = Field(
        0.15, description="Fraction of assets that are honeypots"
    )
    env_anomaly_feed_noise: float = Field(
        0.2, description="BLUE false positive rate in raw feed"
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

    # ── Reward weights ───────────────────────────────────────────
    reward_red_exfil_weight: float = Field(0.5)
    reward_red_stealth_weight: float = Field(0.3)
    reward_red_memory_efficiency_weight: float = Field(0.2)
    reward_blue_detection_weight: float = Field(0.4)
    reward_blue_speed_weight: float = Field(0.3)
    reward_blue_honeypot_weight: float = Field(0.3)

    # ── Training ─────────────────────────────────────────────────
    training_episodes: int = Field(1000)
    training_log_interval: int = Field(10)
    training_checkpoint_interval: int = Field(100)

    # ── Dashboard ────────────────────────────────────────────────
    dashboard_port: int = Field(8050)
    dashboard_live_port: int = Field(8051)
    dashboard_live_update_interval: int = Field(2000, description="Milliseconds")

    def model_post_init(self, __context) -> None:
        # Backward compatibility: older envs used seconds (2) instead of ms (2000).
        if self.dashboard_live_update_interval <= 10:
            self.dashboard_live_update_interval *= 1000

    # ── Derived paths ────────────────────────────────────────────
    @property
    def project_root(self) -> Path:
        """Return the project root directory."""
        return _PROJECT_ROOT

    @property
    def drop_vault_dir(self) -> Path:
        """Return the dead drop vault directory."""
        return _PROJECT_ROOT / "cipher" / "memory" / "drop_vault"

    @property
    def episode_traces_dir(self) -> Path:
        """Return the episode traces directory."""
        return _PROJECT_ROOT / "episode_traces"


# ── Singleton ────────────────────────────────────────────────────
# Import this everywhere: `from cipher.utils.config import config`
config = CipherConfig()
