import networkx, rich, pydantic
print('deps OK')

print('\n--- Graph Smoke Test ---')
from cipher.environment.graph import generate_enterprise_graph, print_graph_summary, get_high_value_target, get_honeypot_nodes
g = generate_enterprise_graph(n_nodes=50, seed=42)
print_graph_summary(g)
hvt = get_high_value_target(g)
hp = get_honeypot_nodes(g)
print(f"HVT node: {hvt}, zone: {g.nodes[hvt]['zone']}")
print(f"Honeypots: {len(hp)} nodes")
assert g.nodes[hvt]['zone'] == 3, "HVT must be in Zone 3"
assert len(hp) >= 5, "Need at least 5 honeypots"
print('graph smoke test PASSED')

print('\n--- Scenario Smoke Test ---')
from cipher.environment.scenario import ScenarioGenerator
gen = ScenarioGenerator()
scenario = gen.generate(episode_number=1)
graph = g # Use the graph from the previous test
print(f"Scenario: {scenario.description}")
print(f"Target files: {scenario.target_files}")
print(f"Difficulty: {scenario.difficulty}")
print("scenario smoke test PASSED")

print('\n--- Observation Smoke Test ---')
from cipher.environment.state import EpisodeState
from cipher.environment.observation import generate_red_observation
state = EpisodeState.create_from_scenario(scenario, graph)
obs = generate_red_observation(state, graph, scenario)
print(f"RED starts at: {obs.current_node}, Zone: {obs.current_zone}")
print(f"Suspicion: {obs.estimated_suspicion}")
print("observation smoke test PASSED")

print('\n--- Suspicion Smoke Test ---')
start_suspicion = state.red_suspicion_score
hp_node = get_honeypot_nodes(graph)[0]
delta_hp = state.update_suspicion_from_action('move', hp_node, graph)
assert delta_hp >= 0.4, f"Honeypot took too small suspicion step: {delta_hp}"
print("suspicion smoke test PASSED")
