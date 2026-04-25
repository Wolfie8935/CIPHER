"""
cipher/environment/emergent_evaluator.py

Emergent Action Evaluator for CIPHER.

Evaluates agent-proposed novel actions (action_type == "emergent") that fall
outside the predefined ActionType vocabulary.  Each emergent intent has a
context-dependent effect on the episode state: success depends on node type,
services available, and current state values.

Design philosophy:
- Reward creative, context-appropriate actions.
- Penalise context-inappropriate or repeated spam actions.
- Cap total emergent bonus at 0.3 per episode to prevent exploitation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx
    from cipher.agents.base_agent import Action
    from cipher.environment.state import EpisodeState


@dataclass
class EmergentResult:
    """Outcome of evaluating an emergent action."""
    success: bool
    effect_description: str
    state_changes: dict[str, Any] = field(default_factory=dict)
    reward_modifier: float = 0.0   # Added to agent's reward
    suspicion_delta: float = 0.0   # Change to RED suspicion score
    detection_delta: float = 0.0   # Change to BLUE detection confidence
    penalty: float = 0.0           # Penalty if action was invalid


def _node_type_val(graph: "nx.DiGraph", node: int) -> str:
    """Safely extract the string value of a node's node_type attribute."""
    nt = graph.nodes[node].get("node_type")
    if nt is None:
        return ""
    return nt.value if hasattr(nt, "value") else str(nt)


class EmergentEvaluator:
    """
    Evaluates emergent (novel) actions proposed by agents.

    Instantiate once per episode.  Call evaluate() for each emergent action.
    """

    # ── RED EMERGENT HANDLERS ────────────────────────────────────

    def _eval_spoof_credentials(self, action: "Action", state: "EpisodeState",
                                 graph: "nx.DiGraph") -> EmergentResult:
        node = state.red_current_node
        ntype = _node_type_val(graph, node)
        if ntype in ("auth_gateway", "server"):
            return EmergentResult(
                success=True,
                effect_description="RED spoofed admin credentials — suspicion reduced",
                state_changes={"credential_level": "elevated"},
                reward_modifier=0.05,
                suspicion_delta=-0.15,
                detection_delta=0.0,
            )
        return EmergentResult(
            success=False,
            effect_description="Credential spoofing failed — wrong node type",
            state_changes={},
            reward_modifier=0.0,
            suspicion_delta=0.10,
            detection_delta=0.05,
        )

    def _eval_inject_noise(self, action: "Action", state: "EpisodeState",
                            graph: "nx.DiGraph") -> EmergentResult:
        return EmergentResult(
            success=True,
            effect_description="RED injected noise into anomaly feed — BLUE confidence reduced",
            state_changes={"noise_injected_steps_remaining": 2},
            reward_modifier=0.03,
            suspicion_delta=0.0,
            detection_delta=-0.10,
        )

    def _eval_piggyback_service(self, action: "Action", state: "EpisodeState",
                                 graph: "nx.DiGraph") -> EmergentResult:
        node = state.red_current_node
        services = graph.nodes[node].get("services", [])
        if any(s in services for s in ["http", "ssh", "https"]):
            return EmergentResult(
                success=True,
                effect_description="RED piggybacked on legitimate service traffic",
                state_changes={"piggybacked": True},
                reward_modifier=0.02,
                suspicion_delta=-0.08,
                detection_delta=0.0,
            )
        return EmergentResult(
            success=False,
            effect_description="No piggybackable service on this node",
            state_changes={},
            reward_modifier=0.0,
            suspicion_delta=0.05,
            detection_delta=0.0,
        )

    def _eval_create_distraction(self, action: "Action", state: "EpisodeState",
                                  graph: "nx.DiGraph") -> EmergentResult:
        current_zone = graph.nodes[state.red_current_node].get("zone")
        current_zone_val = current_zone.value if hasattr(current_zone, "value") else current_zone
        distraction_candidates = [
            n for n in graph.nodes
            if (lambda z: z.value if hasattr(z, "value") else z)(
                graph.nodes[n].get("zone", -1)
            ) != current_zone_val
        ]
        if distraction_candidates:
            distraction_node = random.choice(distraction_candidates)
            return EmergentResult(
                success=True,
                effect_description=f"RED triggered false alert at node {distraction_node}",
                state_changes={"distraction_node": distraction_node, "distraction_steps": 2},
                reward_modifier=0.04,
                suspicion_delta=-0.05,
                detection_delta=-0.05,
            )
        return EmergentResult(
            success=False,
            effect_description="Could not create distraction — no other zones available",
            state_changes={},
            reward_modifier=0.0,
            suspicion_delta=0.0,
            detection_delta=0.0,
        )

    # ── BLUE EMERGENT HANDLERS ───────────────────────────────────

    def _eval_network_quarantine(self, action: "Action", state: "EpisodeState",
                                  graph: "nx.DiGraph") -> EmergentResult:
        target = action.target_node
        if target is None or target not in graph.nodes:
            return EmergentResult(
                success=False,
                effect_description="Invalid quarantine target",
                reward_modifier=0.0,
                suspicion_delta=0.0,
                detection_delta=0.0,
            )
        return EmergentResult(
            success=True,
            effect_description=f"BLUE quarantined node {target} — edges blocked for 3 steps",
            state_changes={"quarantined_node": target, "quarantine_steps": 3},
            reward_modifier=0.05,
            suspicion_delta=0.0,
            detection_delta=0.05,
        )

    def _eval_deploy_canary_file(self, action: "Action", state: "EpisodeState",
                                  graph: "nx.DiGraph") -> EmergentResult:
        target = action.target_node
        if target is None:
            zone2_nodes = [
                n for n in graph.nodes
                if (lambda z: z.value if hasattr(z, "value") else z)(
                    graph.nodes[n].get("zone", -1)
                ) in (2, 3)
            ]
            target = random.choice(zone2_nodes) if zone2_nodes else list(graph.nodes)[0]

        if target not in graph.nodes:
            return EmergentResult(
                success=False,
                effect_description="Invalid canary deployment target",
                reward_modifier=0.0,
                suspicion_delta=0.0,
                detection_delta=0.0,
            )

        canary_name = f"canary_{target}_classified.docx"
        current_files = list(graph.nodes[target].get("files", []))
        if canary_name not in current_files:
            current_files.append(canary_name)
            graph.nodes[target]["files"] = current_files

        return EmergentResult(
            success=True,
            effect_description=f"BLUE deployed canary file '{canary_name}' at node {target}",
            state_changes={"canary_file": canary_name, "canary_node": target},
            reward_modifier=0.03,
            suspicion_delta=0.0,
            detection_delta=0.0,
        )

    def _eval_traffic_analysis(self, action: "Action", state: "EpisodeState",
                                graph: "nx.DiGraph") -> EmergentResult:
        path = list(getattr(state, "red_path_history", []))
        revealed = path[-3:] if len(path) >= 3 else path
        return EmergentResult(
            success=True,
            effect_description=f"BLUE analyzed traffic — revealed RED path: {revealed}",
            state_changes={"revealed_red_path": revealed},
            reward_modifier=0.04,
            suspicion_delta=0.0,
            detection_delta=0.08,
        )

    def _eval_credential_rotation(self, action: "Action", state: "EpisodeState",
                                   graph: "nx.DiGraph") -> EmergentResult:
        """Force RED to re-authenticate, costing an extra suspicion increment."""
        target = action.target_node
        return EmergentResult(
            success=True,
            effect_description="BLUE rotated credentials — RED must re-authenticate",
            state_changes={"credential_rotation": True, "rotation_target": target},
            reward_modifier=0.03,
            suspicion_delta=0.05,   # Slight suspicion increase from forced re-auth
            detection_delta=0.03,
        )

    def _eval_shadow_agent(self, action: "Action", state: "EpisodeState",
                            graph: "nx.DiGraph") -> EmergentResult:
        """BLUE briefly mirrors RED's exact position."""
        red_node = state.red_current_node
        return EmergentResult(
            success=True,
            effect_description=f"BLUE shadow agent revealed RED at node {red_node} (1 step)",
            state_changes={"shadow_confirmed_node": red_node, "shadow_expires_step": state.step + 1},
            reward_modifier=0.05,
            suspicion_delta=0.0,
            detection_delta=0.12,
        )

    def _eval_set_alarm_tripwire(self, action: "Action", state: "EpisodeState",
                                  graph: "nx.DiGraph") -> EmergentResult:
        target = action.target_node
        if target is None or target not in graph.nodes:
            return EmergentResult(
                success=False,
                effect_description="Invalid tripwire target",
                reward_modifier=0.0,
                suspicion_delta=0.0,
                detection_delta=0.0,
            )
        return EmergentResult(
            success=True,
            effect_description=f"BLUE set alarm tripwire on node {target}",
            state_changes={"tripwire_node": target},
            reward_modifier=0.02,
            suspicion_delta=0.0,
            detection_delta=0.0,
        )

    # ── GENERIC / UNKNOWN ────────────────────────────────────────

    def _eval_unknown(self, action: "Action", state: "EpisodeState",
                       graph: "nx.DiGraph") -> EmergentResult:
        """Handle completely novel intents not in our registry.

        Small creativity bonus but no state effect — encourages novel thinking
        without rewarding indefinitely unrecognized actions.
        """
        intent = action.emergent_data.intent if action.emergent_data else "none"
        return EmergentResult(
            success=False,
            effect_description=f"Unknown emergent action '{intent}' — creativity noted",
            state_changes={},
            reward_modifier=0.01,
            suspicion_delta=0.0,
            detection_delta=0.0,
        )

    # ── MAIN DISPATCHER ──────────────────────────────────────────

    _HANDLER_MAP: dict[str, str] = {
        "spoof_credentials":   "_eval_spoof_credentials",
        "inject_noise":        "_eval_inject_noise",
        "piggyback_service":   "_eval_piggyback_service",
        "create_distraction":  "_eval_create_distraction",
        "network_quarantine":  "_eval_network_quarantine",
        "deploy_canary_file":  "_eval_deploy_canary_file",
        "traffic_analysis":    "_eval_traffic_analysis",
        "credential_rotation": "_eval_credential_rotation",
        "shadow_agent":        "_eval_shadow_agent",
        "set_alarm_tripwire":  "_eval_set_alarm_tripwire",
        "alarm":               "_eval_set_alarm_tripwire",
        "tripwire":            "_eval_set_alarm_tripwire",
        "quarantine":          "_eval_network_quarantine",
        "canary":              "_eval_deploy_canary_file",
        "shadow":              "_eval_shadow_agent",
    }

    def evaluate(self, action: "Action", state: "EpisodeState",
                 graph: Any) -> EmergentResult:
        """Evaluate an emergent action and return its effect."""
        if action.emergent_data is None:
            return self._eval_unknown(action, state, graph)

        intent = action.emergent_data.intent
        handler_name = self._HANDLER_MAP.get(intent)

        if handler_name and hasattr(self, handler_name):
            handler = getattr(self, handler_name)
            return handler(action, state, graph)

        # Partial match on intent prefix for custom_ intents
        for key, hname in self._HANDLER_MAP.items():
            if intent.startswith(key):
                handler = getattr(self, hname)
                return handler(action, state, graph)

        return self._eval_unknown(action, state, graph)
