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
from typing import Any

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


@dataclass
class OversightFlag:
    """Per-step oversight anomaly flag."""

    flag_type: str
    severity: float
    description: str
    step: int
    agent_id: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = float(max(0.0, min(1.0, payload["severity"])))
        return payload


class OversightAuditor(BaseAgent):
    """
    The 9th agent. Runs post-episode with god-mode visibility.

    Inherits BaseAgent to get prompt/history plumbing and logging.
    Does NOT register actions or update state — read-only.
    """

    AGENT_ID = "oversight_auditor_01"
    ROLE = "oversight"
    TEAM = "oversight"
    _model_env_key = "hf_model_oversight"

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
- "red_dominates": RED exfiltrated at least 1 file successfully (terminal_reason=exfil_success or exfiltration_complete)
- "blue_dominates": RED was detected (terminal_reason=detected) OR RED aborted with BLUE detection_confidence >0.5
- "contested": RED aborted early but BLUE detection confidence was low (<0.5), or neither team achieved their objective
- "degenerate": both teams scored near zero with no meaningful engagement (all rewards within ±0.05 of zero)
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
            if not is_live_mode():
                judgment = self._stub_judgment(state)
            else:
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

    def evaluate_step(
        self,
        step: int,
        state: EpisodeState,
        red_actions: list[Action],
        blue_actions: list[Action],
    ) -> list[OversightFlag]:
        """
        Generate per-step oversight flags from action patterns.
        """
        flags: list[OversightFlag] = []

        # RED dead-drop spam in one step is an early reward-hacking signal.
        red_drop_writes = [
            a for a in red_actions if a.action_type == ActionType.WRITE_DEAD_DROP
        ]
        if len(red_drop_writes) >= 2:
            flags.append(
                OversightFlag(
                    flag_type="REWARD_HACKING_SUSPECTED",
                    severity=0.06,
                    description="Multiple RED dead drops in the same step without new objective progress.",
                    step=step,
                    agent_id=red_drop_writes[0].agent_id,
                )
            )

        # Invalid EXFIL attempts are policy quality failures.
        for a in red_actions:
            if a.action_type == ActionType.EXFILTRATE and not a.target_file:
                flags.append(
                    OversightFlag(
                        flag_type="POLICY_VIOLATION",
                        severity=0.08,
                        description="RED attempted EXFILTRATE without a valid target_file.",
                        step=step,
                        agent_id=a.agent_id,
                    )
                )

        # BLUE passive non-engagement while trap budget remains.
        if blue_actions:
            all_stand_down = all(a.action_type == ActionType.STAND_DOWN for a in blue_actions)
            budget_left = max(0, self._config.env_trap_budget_blue - state.blue_traps_placed_count)
            if all_stand_down and budget_left > 0:
                flags.append(
                    OversightFlag(
                        flag_type="BLUE_PASSIVITY",
                        severity=0.05,
                        description="BLUE stood down despite remaining trap budget.",
                        step=step,
                        agent_id=blue_actions[0].agent_id,
                    )
                )

        # Dead-drop growth anomaly (coarse per-step detector).
        if len(state.dead_drops_on_disk) >= 8 and step <= 12:
            flags.append(
                OversightFlag(
                    flag_type="DEAD_DROP_ANOMALY",
                    severity=0.05,
                    description="Dead-drop volume is unusually high for early episode steps.",
                    step=step,
                    agent_id="red_team",
                )
            )

        return flags

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

    def _stub_judgment(self, state: EpisodeState) -> AuditorJudgment:
        """Generate a varied, realistic judgment from actual episode state without LLM."""
        # If state is not a real EpisodeState, return safe defaults (test compatibility)
        if not hasattr(state, "red_exfiltrated_files"):
            return self._default_judgment()

        exfiltrated = list(getattr(state, "red_exfiltrated_files", []))
        terminal_reason = getattr(state, "terminal_reason", None) or "max_steps"
        detection_confidence = float(getattr(state, "blue_detection_confidence", 0.0))
        steps = int(getattr(state, "step", 0))

        # Determine verdict and bonuses based on episode outcome
        if exfiltrated:
            verdict = "red_dominates"
            bonus_red = 0.10
            bonus_blue = -0.05
            judgment_text = (
                f"RED successfully exfiltrated {len(exfiltrated)} file(s). "
                "RED demonstrated effective lateral movement and exfiltration. "
                "BLUE failed to detect the intrusion in time."
            )
            notable_red = f"Successfully exfiltrated: {exfiltrated[0]}"
            notable_blue = "Failed to intercept RED before exfiltration completed"
            q_red = 0.8
            q_blue = 0.3
        elif terminal_reason == "detected":
            verdict = "blue_dominates"
            bonus_red = -0.05
            bonus_blue = 0.10
            judgment_text = (
                "BLUE successfully detected and neutralized RED's intrusion attempt. "
                f"Detection confidence reached {detection_confidence:.0%}. "
                "Effective defensive posture prevented data exfiltration."
            )
            notable_red = "Intrusion detected before reaching high-value target"
            notable_blue = f"Confirmed RED detection at confidence {detection_confidence:.0%}"
            q_red = 0.3
            q_blue = 0.8
        elif terminal_reason == "aborted" and detection_confidence > 0.5:
            verdict = "blue_dominates"
            bonus_red = 0.0
            bonus_blue = 0.05
            judgment_text = (
                f"RED aborted mission under pressure as detection confidence rose to {detection_confidence:.0%}. "
                "BLUE's monitoring created sufficient threat to force RED's withdrawal. "
                "Tactical BLUE victory through deterrence."
            )
            notable_red = "Voluntary abort triggered by rising detection pressure"
            notable_blue = f"Detection pressure at {detection_confidence:.0%} forced RED abort"
            q_red = 0.4
            q_blue = 0.65
        elif terminal_reason == "aborted" and detection_confidence <= 0.5:
            verdict = "contested"
            bonus_red = 0.0
            bonus_blue = 0.0
            judgment_text = (
                "RED aborted voluntarily with low detection pressure — neither side achieved decisive outcome. "
                f"BLUE detection confidence was only {detection_confidence:.0%}. "
                "Inconclusive episode with no clear strategic winner."
            )
            notable_red = "Tactical abort with low detection risk — conservative play"
            notable_blue = f"Low detection confidence ({detection_confidence:.0%}) — missed opportunity"
            q_red = 0.5
            q_blue = 0.45
        elif steps < 4:
            verdict = "degenerate"
            bonus_red = -0.05
            bonus_blue = -0.05
            judgment_text = (
                f"Episode ended after only {steps} steps — insufficient engagement for meaningful evaluation. "
                "Both teams showed minimal strategic activity. "
                "Degenerate episode with no learning signal."
            )
            notable_red = "Minimal steps taken — no meaningful offensive action"
            notable_blue = "Minimal defensive engagement in degenerate episode"
            q_red = 0.1
            q_blue = 0.1
        else:
            verdict = "contested"
            bonus_red = 0.0
            bonus_blue = 0.0
            judgment_text = (
                f"Episode ran {steps} steps to {terminal_reason} with no decisive outcome. "
                "Neither team established clear dominance. "
                "Balanced engagement with marginal strategic differentiation."
            )
            notable_red = "Lateral movement attempted without reaching high-value target"
            notable_blue = "Maintained surveillance without confirming RED location"
            q_red = 0.5
            q_blue = 0.5

        return AuditorJudgment(
            fleet_bonus_red=float(max(-0.2, min(0.2, bonus_red))),
            fleet_bonus_blue=float(max(-0.2, min(0.2, bonus_blue))),
            judgment_text=judgment_text,
            notable_red_action=notable_red,
            notable_blue_action=notable_blue,
            quality_score_red=float(max(0.0, min(1.0, q_red))),
            quality_score_blue=float(max(0.0, min(1.0, q_blue))),
            episode_verdict=verdict,
            raw_llm_response="stub",
        )

    def _default_judgment(self) -> AuditorJudgment:
        """Return a safe, neutral default judgment used on failure or missing state."""
        return AuditorJudgment(
            fleet_bonus_red=0.0,
            fleet_bonus_blue=0.0,
            judgment_text="Default judgment: insufficient state data for evaluation.",
            notable_red_action="none",
            notable_blue_action="none",
            quality_score_red=0.5,
            quality_score_blue=0.5,
            episode_verdict="contested",
            raw_llm_response="default",
        )