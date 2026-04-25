import { useState, useEffect, useRef, useCallback } from 'react';

const ZONE_SEQ = ['Perimeter','Perimeter','Perimeter','Perimeter','General','General','General','General','General','Sensitive','Sensitive','Sensitive','Sensitive','Critical/HVT','Critical/HVT'];
const RED_ACTS  = ['move → n3','read_file → n7','move → n12','write_dead_drop → n15','move → n18','plant_false_trail → n20','move → n24','move → n31','exfil_file → n38','abort'];
const BLUE_ACTS = ['scan_network×2 investigate×1','place_honeypot×1 scan×1','analyze_anomaly×2','trigger_alert×1 investigate×1','reconstruct_path×1 scan×2','tamper_dead_drop×1'];

function buildDemoSteps(n = 30) {
  return Array.from({ length: n }, (_, i) => {
    const step = i + 1;
    const susp = Math.min(0.96, 0.06 + step * 0.035 + Math.sin(step * 0.7) * 0.06);
    const det  = Math.min(0.88, 0.02 + step * 0.025 + Math.cos(step * 0.5) * 0.04);
    return {
      step,
      suspicion: Math.round(susp * 1000) / 1000,
      detection: Math.round(det  * 1000) / 1000,
      zone:        ZONE_SEQ[Math.min(step - 1, ZONE_SEQ.length - 1)],
      red_action:  RED_ACTS[step % RED_ACTS.length],
      blue_actions:BLUE_ACTS[step % BLUE_ACTS.length],
      exfil_count: step >= 26 ? 1 : 0,
      exfil_files: step >= 26 ? ['classified_ops_directive.pdf'] : [],
      episode: 1,
      max_steps: 30,
      elapsed: step * 2.1,
      timestamp: new Date(Date.now() - (30 - step) * 2100).toISOString(),
      run_id: 'demo_warroom',
    };
  });
}

const DEMO_STEPS = buildDemoSteps(30);

export function useLivePolling(interval = 2000) {
  const [steps, setSteps] = useState([]);
  const [isLive, setIsLive] = useState(false);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const demoIdxRef   = useRef(0);
  const demoTimer    = useRef(null);
  const pollTimer    = useRef(null);
  const liveConfirmed = useRef(false);

  const stopDemo = useCallback(() => {
    if (demoTimer.current) { clearInterval(demoTimer.current); demoTimer.current = null; }
  }, []);

  const startDemo = useCallback(() => {
    if (demoTimer.current) return;
    setIsDemoMode(true);
    setSteps(DEMO_STEPS.slice(0, 8));
    demoIdxRef.current = 8;
    demoTimer.current = setInterval(() => {
      const idx = demoIdxRef.current;
      if (idx >= DEMO_STEPS.length) { demoIdxRef.current = 0; return; }
      setSteps(prev => [...prev.slice(-29), DEMO_STEPS[idx]]);
      demoIdxRef.current = idx + 1;
    }, 1600);
  }, []);

  const fetchLive = useCallback(async () => {
    try {
      const res = await fetch('/api/live-steps', { signal: AbortSignal.timeout(1500) });
      if (!res.ok) return false;
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) {
        setSteps(data);
        setIsLive(true);
        setIsDemoMode(false);
        liveConfirmed.current = true;
        stopDemo();
        return true;
      }
    } catch { /* network/timeout */ }
    return false;
  }, [stopDemo]);

  useEffect(() => {
    let mounted = true;
    const init = async () => {
      const ok = await fetchLive();
      if (!ok && mounted) startDemo();
    };
    init();
    pollTimer.current = setInterval(async () => {
      if (!mounted) return;
      const ok = await fetchLive();
      if (!ok && !liveConfirmed.current && !demoTimer.current) startDemo();
    }, interval);
    return () => {
      mounted = false;
      clearInterval(pollTimer.current);
      stopDemo();
    };
  }, [fetchLive, startDemo, stopDemo, interval]);

  const latest = steps[steps.length - 1] ?? null;
  return { steps, latest, isLive, isDemoMode };
}
