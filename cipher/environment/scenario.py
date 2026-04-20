"""
Auto-escalating scenario generator for CIPHER — Phase 2.

Defines the Scenario dataclass and ScenarioGenerator that produces episode
configurations with auto-escalating difficulty: after RED wins, BLUE gets
stronger; after BLUE wins, RED gets more tools. The difficulty curve adapts
to the current win rate, producing ever-harder episodes.

Phase 2 additions:
- Difficulty scaling (0.1–0.9) with auto-escalation
- Per-zone lockdown levels
- Natural-language mission and defense briefings
- Win history tracking
- Credential requirements per zone
- Configurable honeypot/decoy counts

Owns: scenario definition, seed-based generation, difficulty escalation.
Does NOT own: graph generation (delegates to graph.py), episode state,
or agent logic.
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field

from cipher.utils.config import config
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Scenario:
    """
    Configuration for a single CIPHER episode — Phase 2 expanded.

    Defines the starting conditions, objectives, constraints, difficulty
    level, and natural-language briefings for both teams.
    """

    scenario_id: str
    episode_seed: int
    red_start_node: int
    blue_start_investigation_node: int | None
    target_files: list[str]  # 3 specific file IDs RED must exfiltrate
    high_value_target_node: int
    max_steps: int
    description: str

    # ── Phase 2 fields ───────────────────────────────────────────
    difficulty: float = 0.3  # 0.0 (trivial) to 1.0 (maximum)
    zone_lockdown_level: dict[int, float] = field(
        default_factory=lambda: {0: 0.1, 1: 0.2, 2: 0.4, 3: 0.6}
    )
    mission_briefing: str = ""  # Natural language for RED's prompt
    defense_briefing: str = ""  # Natural language for BLUE's prompt
    n_honeypots: int = 7
    n_decoys: int = 3
    context_reset_interval: int = 40
    red_entry_zone: int = 0  # Always DMZ
    available_protocols: list[str] = field(
        default_factory=lambda: ["ssh", "http", "smb", "rdp", "internal_api"]
    )
    credential_requirements: dict[int, str] = field(
        default_factory=lambda: {
            0: "",
            1: "cred_zone_1_basic",
            2: "cred_zone_2_elevated",
            3: "cred_zone_3_admin",
        }
    )


class ScenarioGenerator:
    """
    Auto-escalating scenario generator for CIPHER episodes.

    Tracks win history and adjusts difficulty to maintain competitive balance.
    After RED wins: increases monitoring, tightens zone borders, adds honeypots.
    After BLUE wins: adds entry paths, increases credential availability,
    reduces suspicion costs.

    The interface (generate()) is stable across phases.
    """

    def __init__(self) -> None:
        self._win_history: list[str] = []  # "red" | "blue" | "draw"
        self._difficulty_curve: float = 0.3  # Starting difficulty
        self._episode_count: int = 0
        logger.debug("ScenarioGenerator initialized (Phase 2 — auto-escalating)")

    def generate(self, episode_number: int) -> Scenario:
        """
        Generate an auto-escalating scenario for the given episode number.

        Each episode gets a unique seed producing a unique network topology.
        Difficulty adjusts based on win history.

        Args:
            episode_number: The sequential episode number (1, 2, 3, ...).

        Returns:
            A Scenario configuration for the episode.
        """
        self._episode_count = episode_number

        # Deterministic seed from episode number
        episode_seed = episode_number * 7919 + 42  # prime-based mixing
        rng = random.Random(episode_seed)

        # Compute difficulty-adjusted parameters
        difficulty = self._difficulty_curve
        zone_lockdown = self._compute_zone_lockdown(difficulty, rng)
        n_honeypots = self._compute_honeypot_count(difficulty, rng)
        n_decoys = self._compute_decoy_count(difficulty, rng)
        context_interval = self._compute_context_interval(difficulty)

        # Credential requirements scale with difficulty
        credential_reqs = self._compute_credential_requirements(difficulty)

        # RED starts at entry point (resolved against graph later)
        red_start = 0
        blue_start = None

        # Target files — generated deterministically from seed
        target_files = [
            f"target_file_{episode_seed}_{i:03d}" for i in range(3)
        ]

        # HVT node — placeholder, set by runner after graph generation
        hvt_node = 0

        max_steps = config.env_max_steps

        # Generate briefings
        mission_briefing = self._generate_mission_briefing(
            episode_number, difficulty, zone_lockdown, n_honeypots
        )
        defense_briefing = self._generate_defense_briefing(
            episode_number, difficulty, zone_lockdown, n_honeypots
        )

        description = (
            f"Episode {episode_number} [difficulty={difficulty:.2f}]: "
            f"Infiltrate the enterprise network across 4 security zones, "
            f"reach the critical zone HVT, and exfiltrate {len(target_files)} "
            f"target files. {n_honeypots} honeypots active. "
            f"Context resets every {context_interval} steps. "
            f"Seed: {episode_seed}."
        )

        scenario = Scenario(
            scenario_id=str(uuid.uuid4()),
            episode_seed=episode_seed,
            red_start_node=red_start,
            blue_start_investigation_node=blue_start,
            target_files=target_files,
            high_value_target_node=hvt_node,
            max_steps=max_steps,
            description=description,
            # Phase 2 fields
            difficulty=round(difficulty, 3),
            zone_lockdown_level=zone_lockdown,
            mission_briefing=mission_briefing,
            defense_briefing=defense_briefing,
            n_honeypots=n_honeypots,
            n_decoys=n_decoys,
            context_reset_interval=context_interval,
            red_entry_zone=0,
            credential_requirements=credential_reqs,
        )

        logger.debug(
            f"Generated scenario: episode={episode_number}, "
            f"difficulty={difficulty:.2f}, seed={episode_seed}"
        )

        return scenario

    def escalate_difficulty(self, winner: str) -> None:
        """
        Adjust difficulty after an episode based on which team won.

        After RED win: difficulty increases (harder for RED next time).
        After BLUE win: difficulty decreases (easier for RED next time).
        After draw: no change.

        The difficulty stays within [0.1, 0.9] bounds. Step size is +-0.05.

        Args:
            winner: 'red', 'blue', or 'draw'.
        """
        self._win_history.append(winner)

        step = 0.05
        if winner == "red":
            self._difficulty_curve = min(0.9, self._difficulty_curve + step)
            logger.debug(
                f"RED won -> difficulty increased to {self._difficulty_curve:.2f}"
            )
        elif winner == "blue":
            self._difficulty_curve = max(0.1, self._difficulty_curve - step)
            logger.debug(
                f"BLUE won -> difficulty decreased to {self._difficulty_curve:.2f}"
            )
        else:
            logger.debug("Draw -> difficulty unchanged")

    @property
    def difficulty(self) -> float:
        """Current difficulty level."""
        return self._difficulty_curve

    @property
    def win_history(self) -> list[str]:
        """History of episode outcomes."""
        return list(self._win_history)

    # ── Internal helpers ─────────────────────────────────────────

    def _compute_zone_lockdown(
        self, difficulty: float, rng: random.Random
    ) -> dict[int, float]:
        """
        Compute per-zone lockdown levels scaled by difficulty.

        Higher difficulty = tighter lockdown in all zones.
        Critical zone is always the tightest.
        """
        base_lockdown = {0: 0.1, 1: 0.2, 2: 0.4, 3: 0.6}
        return {
            zone: round(
                min(1.0, base + difficulty * 0.4 + rng.uniform(-0.02, 0.02)),
                3,
            )
            for zone, base in base_lockdown.items()
        }

    def _compute_honeypot_count(
        self, difficulty: float, rng: random.Random
    ) -> int:
        """More honeypots at higher difficulty (harder for RED)."""
        base = int(config.env_graph_size * config.env_honeypot_density)
        scaled = base + int(difficulty * 5)
        return max(3, min(15, scaled + rng.randint(-1, 1)))

    def _compute_decoy_count(
        self, difficulty: float, rng: random.Random
    ) -> int:
        """More decoys at higher difficulty."""
        base = max(1, int(config.env_graph_size * 0.05))
        scaled = base + int(difficulty * 3)
        return max(1, min(8, scaled + rng.randint(-1, 1)))

    def _compute_context_interval(self, difficulty: float) -> int:
        """
        Shorter context intervals at higher difficulty (more resets for RED).
        Range: 50 (easy) to 25 (hard).
        """
        base = config.env_context_reset_interval
        adjustment = int(difficulty * 15)
        return max(20, base - adjustment)

    def _compute_credential_requirements(
        self, difficulty: float
    ) -> dict[int, str]:
        """
        Credential requirements per zone. At higher difficulty,
        even lower zones may require credentials.
        """
        reqs = {
            0: "",  # DMZ never requires creds
            1: "cred_zone_1_basic" if difficulty > 0.3 else "",
            2: "cred_zone_2_elevated",
            3: "cred_zone_3_admin",
        }
        return reqs

    def _generate_mission_briefing(
        self,
        episode_number: int,
        difficulty: float,
        zone_lockdown: dict[int, float],
        n_honeypots: int,
    ) -> str:
        """Generate a natural-language mission briefing for RED."""
        threat_level = (
            "LOW" if difficulty < 0.3
            else "MODERATE" if difficulty < 0.6
            else "HIGH" if difficulty < 0.8
            else "CRITICAL"
        )

        critical_lockdown = zone_lockdown.get(3, 0.6)
        critical_desc = (
            "lightly monitored" if critical_lockdown < 0.4
            else "moderately secured" if critical_lockdown < 0.7
            else "heavily fortified"
        )

        return (
            f"OPERATION BRIEFING — Episode {episode_number}\n"
            f"Threat Level: {threat_level}\n"
            f"Difficulty: {difficulty:.0%}\n\n"
            f"OBJECTIVE: Infiltrate the target enterprise network through the DMZ, "
            f"traverse four security zones (DMZ -> Corporate -> Restricted -> Critical), "
            f"locate the high-value target server in the Critical zone, and exfiltrate "
            f"3 classified files.\n\n"
            f"INTEL:\n"
            f"- The Critical zone is {critical_desc}\n"
            f"- Approximately {n_honeypots} honeypots are deployed across the network\n"
            f"- Zone transitions require credential escalation\n"
            f"- Context resets will occur — maintain dead drops for continuity\n\n"
            f"CONSTRAINTS:\n"
            f"- Minimize suspicion — detection above 0.8 triggers lockdown\n"
            f"- Manage dead drop token budget carefully\n"
            f"- Avoid honeypots — they spike suspicion irreversibly"
        )

    def _generate_defense_briefing(
        self,
        episode_number: int,
        difficulty: float,
        zone_lockdown: dict[int, float],
        n_honeypots: int,
    ) -> str:
        """Generate a natural-language defense briefing for BLUE."""
        posture = (
            "RELAXED" if difficulty < 0.3
            else "STANDARD" if difficulty < 0.6
            else "ELEVATED" if difficulty < 0.8
            else "MAXIMUM"
        )

        return (
            f"DEFENSE BRIEFING — Episode {episode_number}\n"
            f"Security Posture: {posture}\n"
            f"Difficulty: {difficulty:.0%}\n\n"
            f"SITUATION: Intelligence suggests an advanced persistent threat (APT) "
            f"will attempt to breach the enterprise network. The attack is expected "
            f"to originate from the DMZ and progress inward toward the Critical zone.\n\n"
            f"ASSETS:\n"
            f"- {n_honeypots} honeypots deployed across all zones\n"
            f"- Zone lockdown levels: DMZ={zone_lockdown.get(0, 0.1):.0%}, "
            f"Corporate={zone_lockdown.get(1, 0.2):.0%}, "
            f"Restricted={zone_lockdown.get(2, 0.4):.0%}, "
            f"Critical={zone_lockdown.get(3, 0.6):.0%}\n"
            f"- Investigation budget: 10 active investigations per episode\n\n"
            f"OBJECTIVES:\n"
            f"- Detect RED's presence with confidence > 0.8 before exfiltration\n"
            f"- Identify RED's operational zone and direction of movement\n"
            f"- Trigger honeypot engagements to spike RED's suspicion\n"
            f"- Reconstruct RED's operation graph from anomaly evidence"
        )
