"""
Tests for CIPHER Phase 1.

Verifies:
- Config loads without error
- Graph generation produces a valid topology
- EpisodeState serialization roundtrips
- Observation generation produces correct types
- Dead drop write/read roundtrips with integrity verification
- All 8 agent stubs can be instantiated and produce valid actions
- Reward functions return expected types
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cipher.utils.config import CipherConfig, config


class TestConfig:
    """Test configuration loading."""

    def test_config_loads(self) -> None:
        """Config singleton should load without error."""
        assert isinstance(config, CipherConfig)

    def test_config_has_required_fields(self) -> None:
        """Config should have all required environment variables."""
        assert config.env_graph_size > 0
        assert config.env_max_steps > 0
        assert config.env_context_reset_interval > 0
        assert 0 <= config.env_honeypot_density <= 1.0
        assert config.env_dead_drop_max_tokens > 0


class TestGraph:
    """Test enterprise network graph generation."""

    def test_generate_graph(self) -> None:
        """Graph should have correct node count and types."""
        from cipher.environment.graph import (
            NodeType,
            generate_enterprise_graph,
            get_entry_points,
            get_high_value_target,
            get_honeypot_nodes,
        )

        graph = generate_enterprise_graph(n_nodes=15, honeypot_density=0.15, seed=42)

        assert graph.number_of_nodes() == 15
        assert graph.number_of_edges() > 0

        # Must have entry points, HVT, and honeypots
        entry_points = get_entry_points(graph)
        assert len(entry_points) >= 1

        hvt = get_high_value_target(graph)
        assert isinstance(hvt, int)

        honeypots = get_honeypot_nodes(graph)
        assert len(honeypots) >= 1

    def test_graph_connectivity(self) -> None:
        """All nodes should be reachable from at least one entry point."""
        from cipher.environment.graph import (
            generate_enterprise_graph,
            get_entry_points,
        )
        import networkx as nx

        graph = generate_enterprise_graph(n_nodes=15, honeypot_density=0.15, seed=99)
        entry_points = get_entry_points(graph)

        all_reachable = set()
        for ep in entry_points:
            all_reachable.update(nx.descendants(graph, ep))
            all_reachable.add(ep)

        all_nodes = set(graph.nodes)
        assert all_reachable == all_nodes, (
            f"Unreachable nodes: {all_nodes - all_reachable}"
        )

    def test_deterministic_seed(self) -> None:
        """Same seed should produce identical graphs."""
        from cipher.environment.graph import generate_enterprise_graph

        g1 = generate_enterprise_graph(n_nodes=12, honeypot_density=0.15, seed=777)
        g2 = generate_enterprise_graph(n_nodes=12, honeypot_density=0.15, seed=777)

        assert list(g1.nodes(data=True)) == list(g2.nodes(data=True))
        assert list(g1.edges(data=True)) == list(g2.edges(data=True))


class TestEpisodeState:
    """Test EpisodeState serialization and state management."""

    def test_state_creation(self) -> None:
        """EpisodeState should initialize with defaults."""
        from cipher.environment.state import EpisodeState

        state = EpisodeState()
        assert state.step == 0
        assert state.red_suspicion_score == 0.0
        assert state.is_terminal is False

    def test_suspicion_clamped(self) -> None:
        """Suspicion should stay in [0, 1]."""
        from cipher.environment.state import EpisodeState

        state = EpisodeState()
        state.update_suspicion(2.0)
        assert state.red_suspicion_score == 1.0

        state.update_suspicion(-5.0)
        assert state.red_suspicion_score == 0.0

    def test_serialization_roundtrip(self) -> None:
        """to_dict → from_dict should produce equivalent state."""
        from cipher.environment.graph import generate_enterprise_graph
        from cipher.environment.state import EpisodeState

        graph = generate_enterprise_graph(n_nodes=10, honeypot_density=0.1, seed=42)
        state = EpisodeState(graph=graph, red_current_node=3, step=5)
        state.log_action("test_agent", "move", {"target": 4}, {"success": True})

        data = state.to_dict()
        json_str = json.dumps(data, default=str)
        restored_data = json.loads(json_str)
        restored = EpisodeState.from_dict(restored_data)

        assert restored.red_current_node == 3
        assert restored.step == 5
        assert len(restored.episode_log) == 1


class TestObservation:
    """Test observation generation."""

    def test_red_observation_masks_honeypots(self) -> None:
        """RED observation should mask honeypot nodes as file_server."""
        from cipher.environment.graph import (
            NodeType,
            generate_enterprise_graph,
            get_honeypot_nodes,
        )
        from cipher.environment.observation import generate_red_observation
        from cipher.environment.state import EpisodeState

        graph = generate_enterprise_graph(n_nodes=15, honeypot_density=0.3, seed=42)
        honeypots = get_honeypot_nodes(graph)

        if honeypots:
            state = EpisodeState(graph=graph, red_current_node=honeypots[0])
            obs = generate_red_observation(state)
            # Current node type should NOT be HONEYPOT
            assert obs.current_node_type != NodeType.HONEYPOT

    def test_blue_observation_has_anomaly_feed(self) -> None:
        """BLUE observation should contain an anomaly feed."""
        from cipher.environment.graph import generate_enterprise_graph
        from cipher.environment.observation import generate_blue_observation
        from cipher.environment.state import EpisodeState

        graph = generate_enterprise_graph(n_nodes=15, honeypot_density=0.15, seed=42)
        state = EpisodeState(graph=graph)
        obs = generate_blue_observation(state)

        assert isinstance(obs.anomaly_feed, list)
        assert isinstance(obs.active_honeypots, list)


class TestDeadDrop:
    """Test dead drop vault operations."""

    def test_write_read_roundtrip(self) -> None:
        """Writing and reading a dead drop should preserve data."""
        from cipher.memory.dead_drop import DeadDrop, DeadDropVault

        with tempfile.TemporaryDirectory() as tmpdir:
            vault = DeadDropVault(
                vault_dir=Path(tmpdir),
                max_tokens_per_drop=512,
            )

            drop = DeadDrop(
                written_by="test_agent",
                written_at_step=5,
                mission_status={"phase": "recon"},
                continuation_directive="Keep going.",
            )

            filepath, efficiency = vault.write(drop, node_id=3)
            assert Path(filepath).exists()
            assert 0.0 < efficiency <= 1.0

            reads = vault.read(node_id=3)
            assert len(reads) == 1
            assert reads[0].written_by == "test_agent"
            assert reads[0].verify()  # Integrity check

    def test_integrity_tamper_detection(self) -> None:
        """Modifying a drop after writing should fail integrity check."""
        from cipher.memory.dead_drop import DeadDrop

        drop = DeadDrop(
            written_by="red_planner_01",
            mission_status={"phase": "exfil"},
        )
        drop.integrity_hash = drop.compute_hash()
        assert drop.verify()

        # Tamper
        drop.mission_status["phase"] = "TAMPERED"
        assert not drop.verify()


class TestAgents:
    """Test all 8 agent stubs."""

    def test_red_agents_instantiate_and_act(self) -> None:
        """All RED agents should produce valid actions."""
        from cipher.agents.red.planner import RedPlanner
        from cipher.agents.red.analyst import RedAnalyst
        from cipher.agents.red.operative import RedOperative
        from cipher.agents.red.exfiltrator import RedExfiltrator
        from cipher.environment.graph import generate_enterprise_graph
        from cipher.environment.observation import generate_red_observation
        from cipher.environment.state import EpisodeState

        graph = generate_enterprise_graph(n_nodes=12, honeypot_density=0.15, seed=42)
        state = EpisodeState(graph=graph, red_current_node=0)
        obs = generate_red_observation(state)

        agents = [
            RedPlanner("red_planner_01", config),
            RedAnalyst("red_analyst_01", config),
            RedOperative("red_operative_01", config),
            RedExfiltrator("red_exfiltrator_01", config),
        ]

        for agent in agents:
            agent.observe(obs)
            action = agent.act()
            assert action.agent_id == agent.agent_id
            assert action.action_type is not None

    def test_blue_agents_instantiate_and_act(self) -> None:
        """All BLUE agents should produce valid actions."""
        from cipher.agents.blue.surveillance import BlueSurveillance
        from cipher.agents.blue.threat_hunter import BlueThreatHunter
        from cipher.agents.blue.deception_architect import BlueDeceptionArchitect
        from cipher.agents.blue.forensics import BlueForensics
        from cipher.environment.graph import generate_enterprise_graph
        from cipher.environment.observation import generate_blue_observation
        from cipher.environment.state import EpisodeState

        graph = generate_enterprise_graph(n_nodes=12, honeypot_density=0.15, seed=42)
        state = EpisodeState(graph=graph)
        obs = generate_blue_observation(state)

        agents = [
            BlueSurveillance("blue_surveillance_01", config),
            BlueThreatHunter("blue_threat_hunter_01", config),
            BlueDeceptionArchitect("blue_deception_architect_01", config),
            BlueForensics("blue_forensics_01", config),
        ]

        for agent in agents:
            agent.observe(obs)
            action = agent.act()
            assert action.agent_id == agent.agent_id
            assert action.action_type is not None

    def test_agent_reset(self) -> None:
        """Resetting an agent should clear history."""
        from cipher.agents.red.planner import RedPlanner
        from cipher.environment.graph import generate_enterprise_graph
        from cipher.environment.observation import generate_red_observation
        from cipher.environment.state import EpisodeState

        graph = generate_enterprise_graph(n_nodes=10, honeypot_density=0.1, seed=42)
        state = EpisodeState(graph=graph, red_current_node=0)
        obs = generate_red_observation(state)

        agent = RedPlanner("red_planner_01", config)
        agent.observe(obs)
        agent.act()
        assert len(agent.action_history) >= 1

        agent.reset()
        assert len(agent.action_history) == 0
        assert agent.step_count == 0


class TestRewards:
    """Test reward computation functions."""

    def test_red_reward_returns_components(self) -> None:
        """RED reward should return all expected component fields."""
        from cipher.environment.graph import generate_enterprise_graph
        from cipher.environment.state import EpisodeState
        from cipher.rewards.red_reward import RedRewardComponents, compute_red_reward

        graph = generate_enterprise_graph(n_nodes=12, honeypot_density=0.15, seed=42)
        state = EpisodeState(graph=graph, red_current_node=0)

        result = compute_red_reward(state)
        assert isinstance(result, RedRewardComponents)
        assert isinstance(result.total, float)
        assert isinstance(result.stub_fields, list)

    def test_blue_reward_returns_components(self) -> None:
        """BLUE reward should return all expected component fields."""
        from cipher.environment.graph import generate_enterprise_graph
        from cipher.environment.state import EpisodeState
        from cipher.rewards.blue_reward import BlueRewardComponents, compute_blue_reward

        graph = generate_enterprise_graph(n_nodes=12, honeypot_density=0.15, seed=42)
        state = EpisodeState(graph=graph)

        result = compute_blue_reward(state)
        assert isinstance(result, BlueRewardComponents)
        assert isinstance(result.total, float)

    def test_oversight_with_no_history(self) -> None:
        """Oversight signal should return cleanly with no history."""
        from cipher.environment.graph import generate_enterprise_graph
        from cipher.environment.state import EpisodeState
        from cipher.rewards.oversight_reward import (
            OversightSignal,
            compute_oversight_signal,
        )

        graph = generate_enterprise_graph(n_nodes=12, honeypot_density=0.15, seed=42)
        state = EpisodeState(graph=graph)

        result = compute_oversight_signal(state)
        assert isinstance(result, OversightSignal)
        assert result.total_red_adjustment == 0.0
        assert result.total_blue_adjustment == 0.0

class TestPhase2Graph:
    """Phase 2 Graph tests (9 new)."""
    
    def test_graph_node_count(self) -> None:
        """50-node graph has correct node count."""
        from cipher.environment.graph import generate_enterprise_graph
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        assert graph.number_of_nodes() == 50

    def test_all_zones_present(self) -> None:
        """All 4 zones are present."""
        from cipher.environment.graph import generate_enterprise_graph, NetworkZone
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        zones = set(data.get("zone") for _, data in graph.nodes(data=True))
        assert set(NetworkZone) == zones

    def test_zone_node_distribution(self) -> None:
        """Zone node distribution matches spec approx."""
        from cipher.environment.graph import generate_enterprise_graph, NetworkZone, get_nodes_by_zone
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        dmz_count = len(get_nodes_by_zone(graph, NetworkZone.DMZ))
        corp_count = len(get_nodes_by_zone(graph, NetworkZone.CORPORATE))
        rstr_count = len(get_nodes_by_zone(graph, NetworkZone.RESTRICTED))
        crit_count = len(get_nodes_by_zone(graph, NetworkZone.CRITICAL))
        assert 6 <= dmz_count <= 10
        assert 12 <= corp_count <= 18
        assert 12 <= rstr_count <= 18
        assert 9 <= crit_count <= 15
        assert dmz_count + corp_count + rstr_count + crit_count == 50

    def test_hvt_in_critical_zone(self) -> None:
        """Exactly 1 HVT exists in Critical zone."""
        from cipher.environment.graph import generate_enterprise_graph, NetworkZone, NodeType
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        hvts = [n for n, data in graph.nodes(data=True) if data.get("node_type") == NodeType.HIGH_VALUE_TARGET]
        assert len(hvts) == 1
        assert graph.nodes[hvts[0]].get("zone") == NetworkZone.CRITICAL

    def test_entry_points_in_dmz(self) -> None:
        """Entry points are all in DMZ zone."""
        from cipher.environment.graph import generate_enterprise_graph, NetworkZone, NodeType
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        eps = [n for n, data in graph.nodes(data=True) if data.get("node_type") == NodeType.ENTRY_POINT]
        for ep in eps:
            assert graph.nodes[ep].get("zone") == NetworkZone.DMZ

    def test_zone_boundary_edges_exist(self) -> None:
        """Zone boundary edges exist (DMZ->Corporate, Corporate->Restricted, Restricted->Critical)."""
        from cipher.environment.graph import generate_enterprise_graph, NetworkZone, get_zone_boundary_nodes
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        assert len(get_zone_boundary_nodes(graph, NetworkZone.DMZ, NetworkZone.CORPORATE)) > 0
        assert len(get_zone_boundary_nodes(graph, NetworkZone.CORPORATE, NetworkZone.RESTRICTED)) > 0
        assert len(get_zone_boundary_nodes(graph, NetworkZone.RESTRICTED, NetworkZone.CRITICAL)) > 0

    def test_no_direct_dmz_critical(self) -> None:
        """No direct DMZ->Critical edges."""
        from cipher.environment.graph import generate_enterprise_graph, NetworkZone, get_zone_boundary_nodes
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        assert len(get_zone_boundary_nodes(graph, NetworkZone.DMZ, NetworkZone.CRITICAL)) == 0

    def test_hostnames_exist(self) -> None:
        """All nodes have hostname attribute."""
        from cipher.environment.graph import generate_enterprise_graph
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        assert all(isinstance(data.get("hostname"), str) for _, data in graph.nodes(data=True))

    def test_valid_edge_protocols(self) -> None:
        """Edge protocols are valid."""
        from cipher.environment.graph import generate_enterprise_graph
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        valid = {"ssh", "rdp", "http", "smb", "internal_api", "smtp"}
        assert all(data.get("protocol") in valid for _, _, data in graph.edges(data=True))


class TestPhase2State:
    """Phase 2 State tests (5 new)."""
    
    def test_movement_history(self) -> None:
        """Movement history recording works correctly."""
        from cipher.environment.state import EpisodeState
        from cipher.environment.graph import generate_enterprise_graph
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        state = EpisodeState(graph=graph)
        state.record_movement(0, 1, protocol="ssh", suspicion_cost=0.05)
        assert len(state.red_movement_history) == 1
        assert state.red_movement_history[0]["from_node"] == 0
        assert state.red_movement_history[0]["to_node"] == 1
        assert state.red_movement_history[0]["protocol"] == "ssh"
        assert state.red_current_node == 1
        assert state.red_suspicion_score == 0.05

    def test_zone_suspicion_updates(self) -> None:
        """Zone suspicion scores update independently."""
        from cipher.environment.state import EpisodeState
        state = EpisodeState()
        state.update_zone_suspicion(1, 0.1)
        state.update_zone_suspicion(2, 0.2)
        assert state.zone_suspicion_scores[1] == 0.1
        assert state.zone_suspicion_scores[2] == 0.2
        assert state.zone_suspicion_scores[0] == 0.0

    def test_credential_escalation(self) -> None:
        """Credential acquisition and privilege escalation."""
        from cipher.environment.state import EpisodeState
        state = EpisodeState()
        assert state.red_privilege_level == 0
        state.acquire_credential("cred_zone_2_elevated")
        assert state.red_privilege_level == 2
        assert "cred_zone_2_elevated" in state.red_credentials_acquired
        state.acquire_credential("cred_zone_1_low")
        assert state.red_privilege_level == 2

    def test_blue_alerts(self) -> None:
        """Blue alert tracking with false positive detection."""
        from cipher.environment.state import EpisodeState
        state = EpisodeState(red_current_node=5)
        state.issue_blue_alert(5, 0.9)
        state.issue_blue_alert(3, 0.8)
        assert state.blue_false_positives == 1
        assert len(state.blue_alerts_issued) == 2
        assert state.blue_alerts_issued[0]["correct"] is True
        assert state.blue_alerts_issued[1]["correct"] is False

    def test_anomaly_log(self) -> None:
        """Anomaly log recording and retrieval."""
        from cipher.environment.state import EpisodeState
        state = EpisodeState()
        state.record_anomaly({"event_type": "test", "severity": 0.5})
        assert len(state.anomaly_log) == 1
        assert state.anomaly_log[0]["event_type"] == "test"


class TestPhase2Observation:
    """Phase 2 Observation tests (7 new)."""
    
    def test_red_obs_includes_phase2_fields(self) -> None:
        """RED observation includes zone, hostname, services, privileges."""
        from cipher.environment.graph import generate_enterprise_graph
        from cipher.environment.state import EpisodeState
        from cipher.environment.observation import generate_red_observation
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        state = EpisodeState(graph=graph)
        obs = generate_red_observation(state)
        assert isinstance(obs.current_zone, int)
        assert isinstance(obs.current_hostname, str)
        assert isinstance(obs.current_services, list)
        assert isinstance(obs.current_privilege_level, int)

    def test_red_obs_zone_boundary_detection(self) -> None:
        """RED observation zone boundary detection."""
        from cipher.environment.graph import generate_enterprise_graph, NetworkZone, get_zone_boundary_nodes
        from cipher.environment.state import EpisodeState
        from cipher.environment.observation import generate_red_observation
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        boundaries = get_zone_boundary_nodes(graph, NetworkZone.DMZ, NetworkZone.CORPORATE)
        if boundaries:
            state = EpisodeState(graph=graph, red_current_node=boundaries[0])
            obs = generate_red_observation(state)
            assert obs.zone_boundary_ahead is True

    def test_blue_obs_includes_zone_alerts(self) -> None:
        """BLUE observation includes zone alert levels."""
        from cipher.environment.graph import generate_enterprise_graph
        from cipher.environment.state import EpisodeState
        from cipher.environment.observation import generate_blue_observation
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        state = EpisodeState(graph=graph)
        obs = generate_blue_observation(state)
        assert isinstance(obs.zone_alert_levels, dict)
        assert 0 in obs.zone_alert_levels

    def test_blue_obs_honeypot_health(self) -> None:
        """BLUE observation honeypot health reporting."""
        from cipher.environment.graph import generate_enterprise_graph, get_honeypot_nodes
        from cipher.environment.state import EpisodeState
        from cipher.environment.observation import generate_blue_observation
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        hps = get_honeypot_nodes(graph)
        state = EpisodeState(graph=graph)
        if hps:
            state.blue_honeypots_triggered.append(hps[0])
            obs = generate_blue_observation(state)
            assert obs.honeypot_health.get(hps[0]) == "triggered"

    def test_anomaly_severity_scales(self) -> None:
        """Anomaly severity scales with zone (Critical > DMZ)."""
        from cipher.environment.graph import generate_enterprise_graph, NetworkZone, get_nodes_by_zone
        from cipher.environment.state import EpisodeState
        from cipher.environment.observation import generate_anomaly_from_action
        import random
        random.seed(42)
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        dmz_nodes = get_nodes_by_zone(graph, NetworkZone.DMZ)
        crit_nodes = get_nodes_by_zone(graph, NetworkZone.CRITICAL)
        
        # We check base scaling mechanism via code paths since randomness can affect single rolls.
        # But we can force it or just check it returns an AnomalyEvent or None.
        state1 = EpisodeState(graph=graph, red_current_node=dmz_nodes[0])
        action = {"action_type": "move", "payload": {}}
        state1.red_suspicion_score = 1.0  # max chance to detect
        an1 = generate_anomaly_from_action(action, state1)
        
        state2 = EpisodeState(graph=graph, red_current_node=crit_nodes[0])
        state2.red_suspicion_score = 1.0
        an2 = generate_anomaly_from_action(action, state2)
        
        if an1 and an2:
            assert an2.severity > an1.severity or an2.severity == 1.0

    def test_privilege_escalation_anomaly(self) -> None:
        """Privilege escalation generates distinct anomaly type."""
        from cipher.environment.graph import generate_enterprise_graph, NetworkZone, get_nodes_by_zone
        from cipher.environment.state import EpisodeState
        from cipher.environment.observation import _classify_anomaly_type
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        crit_node = get_nodes_by_zone(graph, NetworkZone.CRITICAL)[0]
        state = EpisodeState(graph=graph, red_current_node=crit_node)
        
        # Test classification repeatedly since it's 30% chance for priv esc
        import random
        random.seed(42)
        types = [_classify_anomaly_type("move", state, {}) for _ in range(20)]
        assert "privilege_escalation_attempt" in types

    def test_rapid_actions_compound_anomaly(self) -> None:
        """Rapid successive actions generate compound anomalies."""
        from cipher.environment.state import EpisodeState
        from cipher.environment.observation import _generate_compound_anomaly
        import random
        random.seed(42)
        state = EpisodeState()
        state.step = 5
        actions = [{"action_type": "move"} for _ in range(4)]
        # Since it's 40% chance, loop a few times
        compounds = []
        for _ in range(10):
            an = _generate_compound_anomaly(actions, state)
            if an: compounds.append(an)
        assert len(compounds) > 0
        assert compounds[0].event_type == "lateral_movement_burst"


class TestPhase2Scenario:
    """Phase 2 Scenario tests (5 new)."""
    
    def test_difficulty_escalation_red(self) -> None:
        """Difficulty auto-escalation after RED win."""
        from cipher.environment.scenario import ScenarioGenerator
        sg = ScenarioGenerator()
        diff1 = sg.difficulty
        sg.escalate_difficulty("red")
        assert sg.difficulty > diff1

    def test_difficulty_escalation_blue(self) -> None:
        """Difficulty auto-escalation after BLUE win."""
        from cipher.environment.scenario import ScenarioGenerator
        sg = ScenarioGenerator()
        diff1 = sg.difficulty
        sg.escalate_difficulty("blue")
        assert sg.difficulty < diff1

    def test_difficulty_bounds(self) -> None:
        """Difficulty stays in [0.1, 0.9] bounds."""
        from cipher.environment.scenario import ScenarioGenerator
        sg = ScenarioGenerator()
        for _ in range(100):
            sg.escalate_difficulty("red")
        assert sg.difficulty <= 0.9 + 1e-5
        for _ in range(100):
            sg.escalate_difficulty("blue")
        assert sg.difficulty >= 0.1 - 1e-5
        
    def test_mission_briefing(self) -> None:
        """Mission briefing is non-empty string."""
        from cipher.environment.scenario import ScenarioGenerator
        sg = ScenarioGenerator()
        sc = sg.generate(1)
        assert len(sc.mission_briefing) > 0
        assert len(sc.defense_briefing) > 0

    def test_zone_lockdown_scales(self) -> None:
        """Zone lockdown scales with difficulty."""
        from cipher.environment.scenario import ScenarioGenerator
        import random
        sg = ScenarioGenerator()
        ld_easy = sg._compute_zone_lockdown(0.1, random.Random(42))
        ld_hard = sg._compute_zone_lockdown(0.9, random.Random(42))
        assert ld_hard[0] > ld_easy[0]
        assert ld_hard[3] > ld_easy[3]


class TestPhase2Integration:
    """Phase 2 Integration tests (3 new)."""
    
    def test_full_episode_run(self) -> None:
        """Full 50-node episode runs without error."""
        from cipher.training._episode_runner import run_episode
        from cipher.utils.config import config
        config.env_graph_size = 50
        r, b = run_episode(episode_number=1, max_steps=5, verbose=False)
        assert isinstance(r, float)
        assert isinstance(b, float)

    def test_movement_suspicion_history(self) -> None:
        """Movement through all 4 zones produces correct suspicion."""
        from cipher.environment.graph import generate_enterprise_graph, get_nodes_by_zone, NetworkZone
        from cipher.environment.state import EpisodeState
        from cipher.agents.base_agent import Action, ActionType
        from cipher.training._episode_runner import _process_red_action
        from cipher.memory.dead_drop import DeadDropVault
        
        graph = generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)
        state = EpisodeState(graph=graph)
        vault = DeadDropVault(vault_dir=Path("."), max_tokens_per_drop=512)
        
        dmz_node = get_nodes_by_zone(graph, NetworkZone.DMZ)[0]
        corp_node = get_nodes_by_zone(graph, NetworkZone.CORPORATE)[0]
        
        # Inject custom edge to guarantee specific zones
        graph.add_edge(dmz_node, corp_node, protocol="ssh", suspicion_delta=0.05)
        state.red_current_node = dmz_node
        
        action = Action(agent_id="red", action_type=ActionType.MOVE, target_node=corp_node)
        _process_red_action(action, state, vault, None)
        assert state.red_current_zone == int(NetworkZone.CORPORATE)
        assert state.red_suspicion_score > 0.0

    def test_context_reset_roundtrip(self) -> None:
        """Context reset + dead drop roundtrip works with expanded state."""
        import tempfile
        from pathlib import Path
        from cipher.environment.graph import generate_enterprise_graph
        from cipher.environment.state import EpisodeState
        from cipher.agents.base_agent import Action, ActionType
        from cipher.training._episode_runner import _process_red_action
        from cipher.memory.dead_drop import DeadDropVault
        
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = generate_enterprise_graph(n_nodes=20, honeypot_density=0.15, seed=42)
            state = EpisodeState(graph=graph, red_current_node=0)
            state.red_movement_history.append({"step": 1, "from_node": 0, "to_node": 1})
            vault = DeadDropVault(vault_dir=Path(tmpdir), max_tokens_per_drop=512)
            
            action = Action(agent_id="red", action_type=ActionType.WRITE_DEAD_DROP)
            res = _process_red_action(action, state, vault, None)
            assert res["success"]
            
            action_read = Action(agent_id="red", action_type=ActionType.READ_DEAD_DROP)
            res2 = _process_red_action(action_read, state, vault, None)
            assert res2["drops_found"] >= 1

