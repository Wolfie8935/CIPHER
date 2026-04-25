import { useState, useEffect, useRef, useCallback } from 'react';

const RED_THOUGHT_POOL = [
  { agent_id: 'red_planner_01',    reasoning: 'Entry point secured at node_0. Perimeter defenses appear minimal — suspicion threshold not yet triggered. Routing through node_3 to access general zone infrastructure.', action_type: 'move', target_node: 3 },
  { agent_id: 'red_analyst_01',    reasoning: 'File system enumeration complete. Detected 2 potential honeypot signatures at nodes 8 and 14. Recommending alternate routing via nodes 11→19 to avoid detection.', action_type: 'read_file', target_node: 7 },
  { agent_id: 'red_planner_01',    reasoning: 'BLUE likely placed honeypot on node_32 — access pattern anomalies suggest a lure. Routing through node_28 instead. Suspicion at 0.42, within acceptable bounds.', action_type: 'move', target_node: 28 },
  { agent_id: 'red_operative_01',  reasoning: 'Planting temporal decoy at node_12. False timestamp metadata suggests RED activity 48h prior, masking current session from forensics reconstruction.', action_type: 'plant_temporal_decoy', target_node: 12 },
  { agent_id: 'red_analyst_01',    reasoning: 'BLUE alert appears to be a false positive triggered by my dead drop at node_15. Detection confidence dropped 0.12. Window open — advancing to sensitive zone.', action_type: 'move', target_node: 20 },
  { agent_id: 'red_exfiltrator_01',reasoning: 'High-value target located in zone 3. Preparing exfiltration protocol for classified_ops_directive.pdf. Dead drop at node_35 confirmed active and uncompromised.', action_type: 'move', target_node: 38 },
  { agent_id: 'red_planner_01',    reasoning: 'Suspicion spike to 0.71 — BLUE may have detected movement pattern. Deploying false trail to node_20 to redirect investigation. Maintaining primary route via node_31.', action_type: 'plant_false_trail', target_node: 20 },
  { agent_id: 'red_operative_01',  reasoning: 'Dead drop written successfully at node_24. Exfiltrator briefed on extraction route. Waiting for analyst confirmation before final approach to HVT node.', action_type: 'write_dead_drop', target_node: 24 },
];

const BLUE_THOUGHT_POOL = [
  { agent_id: 'blue_surveillance_01',       reasoning: 'Network baseline nominal. Anomaly spike detected near zone 2 boundary — traffic pattern inconsistency at 3 nodes. Elevating detection threshold to 0.35.', action_type: 'scan_network', target_node: null },
  { agent_id: 'blue_threat_hunter_01',      reasoning: 'Investigating node_24. Access frequency 3.2× baseline — temporal metadata inconsistency with normal user pattern. This is an active intrusion. Requesting investigation escalation.', action_type: 'investigate_node', target_node: 24 },
  { agent_id: 'blue_deception_architect_01',reasoning: 'Deploying honeypot at node_32 with realistic lure file: classified_ops_directive_DECOY.pdf. Canary token embedded. RED access will push detection to 0.90 automatically.', action_type: 'place_honeypot', target_node: 32 },
  { agent_id: 'blue_forensics_01',          reasoning: 'Path reconstruction complete. RED visited: n0→n3→n7→n12. Temporal decoy detected and de-obfuscated — actual path confirmed. Reporting to threat hunter for interception.', action_type: 'reconstruct_path', target_node: null },
  { agent_id: 'blue_surveillance_01',       reasoning: 'False positive alert cleared. RED used social engineering pattern to trigger escalation. Adjusting parameters. Honeypots at nodes 19 and 23 remain armed and operational.', action_type: 'analyze_anomaly', target_node: null },
  { agent_id: 'blue_threat_hunter_01',      reasoning: 'Canary file accessed at node_19. This is not a false positive — confirmed hostile access. Issuing network-wide alert. RED agent is currently in general zone.', action_type: 'trigger_alert', target_node: 19 },
  { agent_id: 'blue_deception_architect_01',reasoning: 'Planting breadcrumb trail toward node_40 (dead end). If RED follows, they lose 4-6 steps before detecting the trap. Coordination with forensics for path prediction active.', action_type: 'plant_breadcrumb', target_node: 40 },
  { agent_id: 'blue_forensics_01',          reasoning: 'Dead drop at node_15 tampered — RED wrote false intelligence. Counter-tampered with corrupted payload. If RED reads this drop, their operative will be misdirected from the HVT.', action_type: 'tamper_dead_drop', target_node: 15 },
];

function makeDemoThoughts() {
  const now = Date.now();
  const out = [];
  for (let i = 7; i >= 0; i--) {
    const isRed = i % 2 === 0;
    const pool  = isRed ? RED_THOUGHT_POOL : BLUE_THOUGHT_POOL;
    const t     = pool[Math.floor(i / 2) % pool.length];
    out.push({ ...t, step: Math.max(1, 10 - i), team: isRed ? 'red' : 'blue', timestamp: new Date(now - i * 3500).toISOString() });
  }
  return out;
}

export function useThoughts(interval = 2000) {
  const [thoughts, setThoughts] = useState(makeDemoThoughts);
  const [isLive,   setIsLive]   = useState(false);
  const demoTimer  = useRef(null);
  const pollTimer  = useRef(null);
  const demoPoolIdx = useRef(0);

  const startDemoCycle = useCallback(() => {
    if (demoTimer.current) return;
    demoTimer.current = setInterval(() => {
      const totalPool = RED_THOUGHT_POOL.length + BLUE_THOUGHT_POOL.length;
      const gi = demoPoolIdx.current % totalPool;
      demoPoolIdx.current++;
      const isRed = gi % 2 === 0;
      const pool  = isRed ? RED_THOUGHT_POOL : BLUE_THOUGHT_POOL;
      const t     = pool[Math.floor(gi / 2) % pool.length];
      const entry = { ...t, step: (gi + 10), team: isRed ? 'red' : 'blue', timestamp: new Date().toISOString() };
      setThoughts(prev => [...prev.slice(-9), entry]);
    }, 3200);
  }, []);

  useEffect(() => {
    let mounted = true;
    const poll = async () => {
      try {
        const res = await fetch('/api/thoughts', { signal: AbortSignal.timeout(1500) });
        if (!res.ok) throw new Error();
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) {
          if (mounted) { setThoughts(data); setIsLive(true); if (demoTimer.current) { clearInterval(demoTimer.current); demoTimer.current = null; } }
          return;
        }
      } catch { /* API not ready */ }
      if (mounted && !demoTimer.current) startDemoCycle();
    };
    poll();
    pollTimer.current = setInterval(poll, interval);
    return () => {
      mounted = false;
      clearInterval(pollTimer.current);
      if (demoTimer.current) clearInterval(demoTimer.current);
    };
  }, [interval, startDemoCycle]);

  const redThoughts  = thoughts.filter(t => t.team === 'red');
  const blueThoughts = thoughts.filter(t => t.team === 'blue');
  return { thoughts, redThoughts, blueThoughts, isLive };
}
