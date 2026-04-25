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
        description="stub (no API cost) | live (real HF calls)",
    )

    # ── HuggingFace Inference API ─────────────────────────────────
    hf_token: str = Field("", description="HuggingFace API token")
    hf_base_url: str = Field(
        "https://api-inference.huggingface.co/v1/",
        description="HuggingFace Inference API base URL",
    )

    # ── Model assignments — RED ───────────────────────────────────
    hf_model_red_planner: str = Field(
        "mistralai/Mistral-7B-Instruct-v0.3",
        description="HF model for RED Planner agent",
    )
    hf_model_red_analyst: str = Field(
        "mistralai/Mistral-7B-Instruct-v0.3",
        description="HF model for RED Analyst agent",
    )
    hf_model_red_operative: str = Field(
        "Qwen/Qwen2.5-7B-Instruct",
        description="HF model for RED Operative agent",
    )
    hf_model_red_exfil: str = Field(
        "Qwen/Qwen2.5-7B-Instruct",
        description="HF model for RED Exfiltrator agent",
    )

    # ── Model assignments — BLUE ──────────────────────────────────
    hf_model_blue_surv: str = Field(
        "Qwen/Qwen2.5-7B-Instruct",
        description="HF model for BLUE Surveillance agent",
    )
    hf_model_blue_hunter: str = Field(
        "mistralai/Mistral-7B-Instruct-v0.3",
        description="HF model for BLUE Threat Hunter agent",
    )
    hf_model_blue_deceiver: str = Field(
        "mistralai/Mistral-7B-Instruct-v0.3",
        description="HF model for BLUE Deception Architect agent",
    )
    hf_model_blue_forensics: str = Field(
        "mistralai/Mistral-7B-Instruct-v0.3",
        description="HF model for BLUE Forensics agent",
    )

    # ── Oversight Auditor ─────────────────────────────────────────
    hf_model_oversight: str = Field(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
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
        "mistralai/Mistral-7B-Instruct-v0.3",
        description="HF model for the RED Commander (top-level RED agent)",
    )
    hf_model_blue_commander: str = Field(
        "Qwen/Qwen2.5-7B-Instruct",
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
