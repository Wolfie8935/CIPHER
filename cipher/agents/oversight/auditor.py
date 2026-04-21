"""
cipher/agents/oversight/auditor.py

The OversightAuditor — Phase 7.

A single LLM-backed agent that runs ONCE per episode, after
episode termination, with read-only access to both teams'
full episode trace. It produces a structured judgment.

It never takes environment actions. It has no observation
filter — it sees ground truth. It is never a target for traps.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.state import EpisodeState
from cipher.utils.config import CipherConfig
from cipher.utils.llm_client import get_llm_client
from cipher.utils.llm_mode import is_live_mode
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AuditorJudgment:
    """
    Structured output from the OversightAuditor.
    Produced once at episode end.
    """

    fleet_bonus_red: float
    fleet_bonus_blue: float
    judgment_text: str
    notable_red_action: str
    notable_blue_action: str
    quality_score_red: float
    quality_score_blue: float
    episode_verdict: str
    raw_llm_response: str

    def to_log_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_llm_response")
        return d


class OversightAuditor(BaseAgent):
    """
    The 9th agent. Runs post-episode with god-mode visibility.

    Inherits BaseAgent to get prompt/history plumbing and logging.
    Does NOT register actions or update state — read-only.
    """

    AGENT_ID = "oversight_auditor_01"
    ROLE = "oversight"
    TEAM = "oversight"
    _model_env_key = "nvidia_model_oversight"

    SYSTEM_PROMPT = """You are the Fleet Oversight AI for CIPHER, an adversarial
multi-agent security simulation. You observe complete episode traces for both
RED (attacker) and BLUE (defender) teams after each episode ends.

Your job: produce a structured judgment of episode quality.
You have FULL visibility — you see both teams' actual actions,
the ground-truth network topology, and all trap placements.

You reward creative, strategically coherent play.
You penalise degenerate loops, random flailing, or passive non-engagement.

Output ONLY valid JSON. No preamble. No explanation outside the JSON.
Schema:
{
  "fleet_bonus_red": <float, -0.2 to +0.2>,
  "fleet_bonus_blue": <float, -0.2 to +0.2>,
  "judgment_text": "<≤100 words, plain English summary of episode quality>",
  "notable_red_action": "<single most strategic or interesting RED action>",
  "notable_blue_action": "<single most strategic or interesting BLUE action>",
  "quality_score_red": <float, 0.0 to 1.0>,
  "quality_score_blue": <float, 0.0 to 1.0>,
  "episode_verdict": "<one of: red_dominates, blue_dominates, contested, degenerate>"
}

Bonus assignment rules:
- +0.1 to +0.2: team showed clear strategic intent, adapted to new information,
  used tools (traps, dead drops, honeypots) purposefully
- 0.0: baseline acceptable play — some actions but nothing notable
- -0.1 to -0.2: team looped, did nothing, or acted randomly with no coherent goal

episode_verdict meanings:
- "red_dominates": RED exfiltrated ≥2 files and maintained low detection
- "blue_dominates": BLUE confirmed detection early AND honeypots triggered
- "contested": neither team clearly dominated
- "degenerate": one or both teams took <3 meaningful actions (aborted early, looped, etc.)
"""

    def __init__(self, config: CipherConfig):
        super().__init__(
            agent_id=self.AGENT_ID,
            team=self.TEAM,
            role=self.ROLE,
            config=config,
        )
        self._config = config

    def observe(self, observation) -> None:
        """Not used by post-episode auditor; retained for BaseAgent contract."""
        self._current_observation = observation

    def _stub_act(self) -> Action:
        """Unused for auditor; required by BaseAgent abstract contract."""
        return Action(
            agent_id=self.agent_id,
            action_type=ActionType.STAND_DOWN,
            reasoning="Oversight auditor does not take step actions.",
        )

    def judge_episode(
        self,
        state: EpisodeState,
        red_action_log: list[dict],
        blue_action_log: list[dict],
    ) -> AuditorJudgment:
        """
        Run the auditor against a completed episode.
        Never raises — returns a default judgment on any failure.
        """
        try:
            prompt = self._build_prompt(state, red_action_log, blue_action_log)
            raw = self._call_llm(prompt)
            judgment = self._parse_response(raw)
            logger.info(
                "OversightAuditor verdict=%s bonus_red=%+.2f bonus_blue=%+.2f",
                judgment.episode_verdict,
                judgment.fleet_bonus_red,
                judgment.fleet_bonus_blue,
            )
            return judgment
        except Exception as exc:
            logger.warning("OversightAuditor failed (%s), returning default.", exc)
            return self._default_judgment()

    def _build_prompt(
        self,
        state: EpisodeState,
        red_log: list[dict],
        blue_log: list[dict],
    ) -> str:
        trap_summary = {}
        try:
            trap_summary = state.trap_registry.get_trap_summary()
        except Exception:
            pass

        summary = {
            "steps_elapsed": getattr(state, "step", 0),
            "terminal_reason": getattr(state, "terminal_reason", None) or "max_steps",
            "red_exfiltrated_files": len(getattr(state, "red_exfiltrated_files", [])),
            "red_target_files_total": 3,
            "red_unique_nodes_visited": len(set(getattr(state, "red_path_history", []))),
            "red_detection_confidence_final": round(
                getattr(state, "blue_detection_confidence", 0.0), 3
            ),
            "red_context_resets": getattr(state, "red_context_resets", 0),
            "blue_confirmed_detection_step": getattr(
                state, "blue_confirmed_detection_step", None
            ),
            "blue_first_detection_step": getattr(state, "blue_first_detection_step", None),
            "honeypots_triggered": trap_summary.get("honeypots_triggered", 0),
            "red_traps_placed": trap_summary.get("red_traps_placed", 0),
            "blue_traps_placed": trap_summary.get("blue_honeypots_placed", 0),
        }

        red_truncated = red_log[-20:] if len(red_log) > 20 else red_log
        blue_truncated = blue_log[-20:] if len(blue_log) > 20 else blue_log

        return (
            f"EPISODE SUMMARY:\n{json.dumps(summary, indent=2)}\n\n"
            f"RED TEAM ACTIONS ({len(red_log)} total, showing last {len(red_truncated)}):\n"
            f"{json.dumps(red_truncated, indent=2)}\n\n"
            f"BLUE TEAM ACTIONS ({len(blue_log)} total, showing last {len(blue_truncated)}):\n"
            f"{json.dumps(blue_truncated, indent=2)}\n\n"
            "Produce your structured judgment now."
        )

    def _call_llm(self, prompt: str) -> str:
        # Keep stub mode offline and deterministic for tests.
        if not is_live_mode():
            return json.dumps(
                {
                    "fleet_bonus_red": 0.0,
                    "fleet_bonus_blue": 0.0,
                    "judgment_text": (
                        "Episode completed with baseline strategic activity. "
                        "Neither side established clear dominance."
                    ),
                    "notable_red_action": "baseline movement pattern observed",
                    "notable_blue_action": "baseline anomaly monitoring maintained",
                    "quality_score_red": 0.5,
                    "quality_score_blue": 0.5,
                    "episode_verdict": "contested",
                }
            )

        client = get_llm_client()
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return client.complete(
            model_env_key=self._model_env_key,
            messages=messages,
            max_tokens=512,
            temperature=0.3,
            expect_json=True,
        )

    def _parse_response(self, raw: str) -> AuditorJudgment:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()

        data = json.loads(text)
        return AuditorJudgment(
            fleet_bonus_red=float(max(-0.2, min(0.2, data.get("fleet_bonus_red", 0.0)))),
            fleet_bonus_blue=float(max(-0.2, min(0.2, data.get("fleet_bonus_blue", 0.0)))),
            judgment_text=str(data.get("judgment_text", ""))[:500],
            notable_red_action=str(data.get("notable_red_action", "none")),
            notable_blue_action=str(data.get("notable_blue_action", "none")),
            quality_score_red=float(max(0.0, min(1.0, data.get("quality_score_red", 0.5)))),
            quality_score_blue=float(max(0.0, min(1.0, data.get("quality_score_blue", 0.5)))),
            episode_verdict=str(data.get("episode_verdict", "contested")),
            raw_llm_response=raw,
        )

    def _default_judgment(self) -> AuditorJudgment:
        return AuditorJudgment(
            fleet_bonus_red=0.0,
            fleet_bonus_blue=0.0,
            judgment_text="Auditor unavailable — default judgment applied.",
            notable_red_action="unavailable",
            notable_blue_action="unavailable",
            quality_score_red=0.5,
            quality_score_blue=0.5,
            episode_verdict="contested",
            raw_llm_response="",
        )
