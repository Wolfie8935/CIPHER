import { useState, useEffect, useRef, useCallback } from 'react';

export function useEpisodeReplay(filename, speed = 1) {
  const [allSteps,   setAllSteps]   = useState([]);
  const [steps,      setSteps]      = useState([]);
  const [graph,      setGraph]      = useState({ nodes: [], edges: [] });
  const [outcome,    setOutcome]    = useState(null);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const [isPlaying,  setIsPlaying]  = useState(false);

  const intervalRef = useRef(null);
  const idxRef      = useRef(0);
  const stepsRef    = useRef([]);

  useEffect(() => {
    if (!filename || filename === 'live') return;
    setSteps([]);
    setAllSteps([]);
    setCurrentIdx(0);
    setIsComplete(false);
    setIsPlaying(false);
    setOutcome(null);
    idxRef.current = 0;
    stepsRef.current = [];
    setGraph({ nodes: [], edges: [] });

    fetch(`/api/episode/${encodeURIComponent(filename)}`)
      .then(r => r.json())
      .then(data => {
        // ── Parse graph ──────────────────────────────────────────────
        const g = data.graph ?? {};
        const rawEdges = g.links ?? g.edges ?? [];
        const nodes = (g.nodes ?? []).map(n => {
          const zoneRaw = n.zone;
          const zone    = zoneRaw?._value_ ?? zoneRaw ?? 0;
          const typeRaw = n.type ?? n.node_type ?? 'server';
          const type    = typeRaw?._value_ ?? String(typeRaw);
          return {
            id:          n.id ?? n.node_id,
            hostname:    n.hostname ?? `node_${n.id}`,
            zone:        Number(zone),
            type,
            is_honeypot: n.is_honeypot ?? false,
            is_entry:    n.is_entry    ?? false,
            is_hvt:      n.is_hvt      ?? false,
            files:       n.files       ?? [],
            services:    n.services    ?? [],
          };
        });
        const edges = rawEdges.map(l => ({
          source: l.source ?? l.src,
          target: l.target ?? l.tgt,
        }));
        setGraph({ nodes, edges });

        // Build node→zone lookup for step state building
        const nodeZoneMap = {};
        for (const n of nodes) nodeZoneMap[n.id] = n.zone;

        // ── Parse episode log ────────────────────────────────────────
        const log = data.episode_log ?? data.steps ?? [];
        const totalStepsNum = typeof data.steps === 'number'
          ? data.steps
          : typeof data.step === 'number'
            ? data.step
            : 30;
        const episodeNum    = data.episode_number ?? data.episode ?? 1;
        const finalSuspicion = data.red_suspicion_score ?? 0;
        const finalDetection = data.blue_detection_confidence ?? 0;
        const exfilFiles    = data.red_exfiltrated_files ?? [];
        const terminalReason = data.terminal_reason ?? '';
        const term = String(terminalReason || '').toLowerCase();
        let winner = 'BLUE';
        if (['exfil_success', 'exfiltration_complete', 'exfil_complete'].includes(term)) winner = 'RED';
        else if (term === 'aborted') winner = 'DRAW';
        setOutcome({ winner, terminalReason: terminalReason || 'max_steps' });

        // Group entries by step number (each step has 8 agent entries)
        const stepGroups = {};
        for (const entry of log) {
          const s = entry.step;
          if (s == null) continue;
          if (!stepGroups[s]) stepGroups[s] = [];
          stepGroups[s].push(entry);
        }

        const ZONE_LABELS = ['Perimeter', 'General', 'Sensitive', 'Critical/HVT'];
        const stepNums = Object.keys(stepGroups).map(Number).sort((a, b) => a - b);

        const extracted = stepNums.map((stepNum, idx) => {
          const entries = stepGroups[stepNum];

          // Red planner's intended target node — prefer planner subagent,
          // fall back to commander, then any RED entry with a target.
          const plannerEntry =
            entries.find(e => e.agent_id?.includes('red_planner') && e.payload?.target_node != null)
            ?? entries.find(e => e.agent_id?.includes('red_commander') && e.payload?.target_node != null)
            ?? entries.find(e => e.agent_id?.startsWith('red_') && e.payload?.target_node != null);
          const redNode = plannerEntry?.payload?.target_node ?? null;

          // Zone derived from graph node metadata
          const zoneNum  = (redNode != null ? nodeZoneMap[redNode] : null) ?? 0;
          const zone     = ZONE_LABELS[Math.min(3, zoneNum)] ?? 'Perimeter';

          // Linear interpolation of suspicion/detection across the episode
          const progress = totalStepsNum > 0 ? stepNum / totalStepsNum : 0;
          const isLast   = idx === stepNums.length - 1;

          return {
            step:        stepNum,
            max_steps:   totalStepsNum,
            episode:     episodeNum,
            red_node:    redNode,
            red_action:  redNode != null ? `move → n${redNode}` : 'wait',
            zone,
            suspicion:   finalSuspicion * progress,
            detection:   finalDetection * progress,
            exfil_count: isLast && terminalReason === 'exfiltration_complete'
              ? exfilFiles.length : 0,
            exfil_files: isLast && terminalReason === 'exfiltration_complete'
              ? exfilFiles : [],
          };
        });

        setAllSteps(extracted);
        stepsRef.current = extracted;
        idxRef.current   = 0;
      })
      .catch(() => {});
  }, [filename]);

  // Playback timer — only runs when isPlaying
  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (!isPlaying || !allSteps.length) return;

    const delay = Math.max(60, 1200 / Math.max(0.1, speed));

    intervalRef.current = setInterval(() => {
      const idx = idxRef.current;
      if (idx >= stepsRef.current.length) {
        clearInterval(intervalRef.current);
        setIsComplete(true);
        setIsPlaying(false);
        return;
      }
      const step = stepsRef.current[idx];
      setSteps(prev => [...prev, step]);
      setCurrentIdx(idx + 1);
      idxRef.current = idx + 1;
    }, delay);

    return () => clearInterval(intervalRef.current);
  }, [isPlaying, allSteps, speed]);

  const play = useCallback(() => {
    if (isComplete) {
      setSteps([]);
      setCurrentIdx(0);
      setIsComplete(false);
      idxRef.current = 0;
    }
    setIsPlaying(true);
  }, [isComplete]);

  const pause  = useCallback(() => setIsPlaying(false), []);

  const seekTo = useCallback((idx) => {
    const clamped = Math.max(0, Math.min(idx, stepsRef.current.length));
    setSteps(stepsRef.current.slice(0, clamped));
    setCurrentIdx(clamped);
    idxRef.current = clamped;
    setIsComplete(clamped >= stepsRef.current.length && stepsRef.current.length > 0);
  }, []);

  const latest = steps[steps.length - 1] ?? null;
  return {
    steps, latest, graph,
    outcome,
    isComplete, isPlaying,
    totalSteps: allSteps.length, currentIdx,
    play, pause, seekTo,
    hasData: allSteps.length > 0,
  };
}
