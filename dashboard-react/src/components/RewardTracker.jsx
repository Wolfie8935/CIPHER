import { useState, useEffect, useRef } from 'react';

function AnimatedValue({ value, decimals = 2 }) {
  const [display, setDisplay] = useState(value);
  const animRef = useRef(null);
  const prevRef = useRef(value);

  useEffect(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    const from = prevRef.current;
    const to   = value;
    const dur  = 700;
    const start = performance.now();
    const step  = (now) => {
      const t    = Math.min((now - start) / dur, 1);
      const ease = 1 - Math.pow(1 - t, 3);
      setDisplay(from + (to - from) * ease);
      if (t < 1) animRef.current = requestAnimationFrame(step);
      else prevRef.current = to;
    };
    animRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animRef.current);
  }, [value]);

  const sign = display >= 0 ? '+' : '';
  return <span>{sign}{display.toFixed(decimals)}</span>;
}

function computeRewards(steps) {
  if (!steps || steps.length === 0) return { redTotal: 0.62, blueTotal: 0.44 };
  const latest = steps[steps.length - 1];
  const stepCount = latest.step || 1;
  const susp   = latest.suspicion || 0;
  const det    = latest.detection || 0;
  const exfil  = latest.exfil_count || 0;
  const redTotal  = Math.min(3, exfil * 0.8 + (1 - susp) * 0.5 + stepCount * 0.02);
  const blueTotal = Math.min(3, det * 0.6 + (1 - exfil * 0.4) * 0.3 + stepCount * 0.015);
  return { redTotal, blueTotal };
}

export default function RewardTracker({ steps }) {
  const { redTotal, blueTotal } = computeRewards(steps);
  const maxReward = 3;
  const redPct  = Math.min(100, (redTotal / maxReward) * 100);
  const bluePct = Math.min(100, (blueTotal / maxReward) * 100);

  return (
    <div className="card card-gold" style={{ flexShrink:0 }}>
      <div className="section-label">
        <div className="label-dot gold" />
        <span>REWARD TRACKER</span>
      </div>
      <div style={{ padding:'10px 4px 10px' }}>
        {/* RED */}
        <div className="reward-row">
          <div className="reward-label" style={{ color:'var(--red-team)' }}>RED</div>
          <div className="reward-bar-track">
            <div className="reward-bar-fill red" style={{ width:`${redPct}%` }} />
          </div>
          <div className="reward-val" style={{ color:'var(--red-team)' }}>
            <AnimatedValue value={redTotal} decimals={2} />
          </div>
        </div>
        {/* BLUE */}
        <div className="reward-row">
          <div className="reward-label" style={{ color:'var(--blue-team)' }}>BLUE</div>
          <div className="reward-bar-track">
            <div className="reward-bar-fill blue" style={{ width:`${bluePct}%` }} />
          </div>
          <div className="reward-val" style={{ color:'var(--blue-team)' }}>
            <AnimatedValue value={blueTotal} decimals={2} />
          </div>
        </div>
      </div>
    </div>
  );
}
