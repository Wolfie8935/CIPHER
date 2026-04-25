import { useState, useEffect } from 'react';

const VERDICTS = {
  red_dominates:  { color:'var(--red-team)',  label:'RED DOMINATES',  icon:'🔴' },
  blue_dominates: { color:'var(--blue-team)', label:'BLUE DOMINATES', icon:'🔵' },
  contested:      { color:'var(--yellow)',    label:'CONTESTED',       icon:'⚡' },
  degenerate:     { color:'#555',            label:'DEGENERATE',      icon:'⚪' },
};

const DEMO_COMMENTARY = [
  'Balanced engagement. Both teams displaying advanced strategic depth.',
  'RED demonstrating sophisticated evasion — temporal decoy technique noted.',
  'BLUE adaptive response notable. Honeypot placement shows predictive intelligence.',
  'Suspicion oscillation pattern suggests deliberate RED misdirection strategy.',
  'Both teams showing emergent coordination behaviors. Fleet bonus under review.',
  'RED path complexity elevated — exfiltration route non-trivial.',
  'BLUE forensics reconstruction successful. Minimal false positive rate.',
];

export default function OversightPanel({ steps }) {
  const [commentIdx, setCommentIdx] = useState(0);
  const [verdict,    setVerdict]    = useState('contested');
  const [flags,      setFlags]      = useState([]);

  useEffect(() => {
    const t = setInterval(() => setCommentIdx(i => (i + 1) % DEMO_COMMENTARY.length), 6000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!steps || steps.length === 0) return;
    const latest = steps[steps.length - 1];
    const susp = latest.suspicion ?? 0;
    const det  = latest.detection ?? 0;
    const exfil = latest.exfil_count ?? 0;
    if      (exfil > 0)   setVerdict('red_dominates');
    else if (det  > 0.75) setVerdict('blue_dominates');
    else if (susp > 0.65) setVerdict('contested');
    else                  setVerdict('contested');

    const newFlags = [];
    if (susp > 0.85 && det < 0.3)  newFlags.push('REWARD_HACKING');
    if (susp < 0.2 && det < 0.15)  newFlags.push('COLLUSION');
    setFlags(newFlags);
  }, [steps]);

  const v = VERDICTS[verdict] || VERDICTS.contested;
  const commentary = DEMO_COMMENTARY[commentIdx];

  return (
    <div className="card" style={{ flexShrink:0 }}>
      <div className="section-label">
        <div className="label-dot gold" />
        <span>🟡 OVERSIGHT AUDITOR</span>
      </div>
      <div style={{ padding:'8px 12px 10px', display:'flex', flexDirection:'column', gap:8 }}>
        {/* Verdict */}
        <div style={{ display:'flex', alignItems:'center', gap:10 }}>
          <div className={`verdict-badge ${verdict}`}>
            {v.icon} {v.label}
          </div>
          {flags.map(f => (
            <div key={f} className="flag-badge">⚑ {f}</div>
          ))}
        </div>
        {/* Commentary */}
        <div style={{
          fontFamily:'var(--font-mono)', fontSize:11, color:'var(--text-dim)',
          lineHeight:1.5, minHeight:32,
          borderLeft:'2px solid var(--border-gold)',
          paddingLeft:8,
          animation:'fadeSlideIn 0.4s ease-out',
        }}>
          "{commentary}"
        </div>
      </div>
    </div>
  );
}
