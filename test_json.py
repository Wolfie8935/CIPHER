import json

text5 = """{
  "action_type": "RECONSTRUCT_PATH",
  "target_node": null,
  "target_file": null,
  "reasoning": "New anomaly at node 35 in zone 2, updating operation graph with possible RED movement"
"""
try:
    json.loads(text5)
except Exception as e:
    print("Test 5:", e)

text6 = """{
  "action_type": "INVESTIGATE_NODE",
  "target_node": 23,
  "reasoning": "Investigating node 23 due to repeated auth attempt in zone 2"
"""
try:
    json.loads(text6)
except Exception as e:
    print("Test 6:", e)
