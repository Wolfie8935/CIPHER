import json
from cipher.utils.config import CipherConfig
from cipher.environment.state import EpisodeState
from cipher.rewards.red_reward import compute_red_reward
from cipher.rewards.blue_reward import compute_blue_reward
import networkx as nx
from enum import Enum

class Zone(Enum):
    GENERAL = 0
    PERIMETER = 1
    RESTRICTED = 2
    SECURE = 3

cfg = CipherConfig()
graph = nx.Graph()
graph.add_node(0, zone=Zone.GENERAL)
graph.add_node(5, zone=Zone.PERIMETER)

state = EpisodeState(
    graph=graph,
    red_current_node=5,
    red_path_history=[0, 5],
    red_exfiltrated_files=[],
    blue_detection_confidence=0.5,
    terminal_reason="max_steps",
    step=30
)

class DummyScenario:
    target_files = ["file1", "file2", "file3"]

scenario = DummyScenario()

class DummyVault:
    def list_all_drop_paths(self):
        return []

vault = DummyVault()

class DummyForensics:
    pass

forensics = DummyForensics()

try:
    red_r = compute_red_reward(state, scenario, vault, cfg)
    print("RED REWARD:", red_r)
except Exception as e:
    print("RED EXCEPTION:", e)

try:
    blue_r = compute_blue_reward(state, graph, forensics, cfg)
    print("BLUE REWARD:", blue_r)
except Exception as e:
    print("BLUE EXCEPTION:", e)
