import { useState, useEffect, useRef } from 'react';

const ZONE_COLORS = { Perimeter: '#7a859f', General: '#00bcd4', Sensitive: '#ffc107', 'Critical/HVT': '#ff1744' };

export default function EpisodeHeader({ latest, isDemoMode }) {
  const [elapsed, setElapsed]     = useState(0);
  const startRef   = useRef(Date.now());

  useEffect(() => {
    const t = setInterval(() => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)), 500);
    return () => clearInterval(t);
  }, []);

  const step      = latest?.step      ?? 0;
  const maxSteps  = latest?.max_steps ?? 30;
  const episode   = latest?.episode   ?? 1;
  const zone      = latest?.zone      ?? 'Perimeter';
  const suspicion = latest?.suspicion ?? 0;
  const exfil     = latest?.exfil_count ?? 0;
  const modeRaw   = latest?.run_id?.split('_')[0] ?? (isDemoMode ? 'demo' : 'stub');

  const modeMap = {
    live:   { label: 'LIVE',   dotCls: 'live',   color: '#ff1744' },
    hybrid: { label: 'HYBRID', dotCls: 'hybrid',  color: '#ff9800' },
    stub:   { label: 'STUB',   dotCls: 'stub',    color: '#555' },
    demo:   { label: 'DEMO',   dotCls: 'demo',    color: '#ffd740' },
  };
  const mode = modeMap[modeRaw] || modeMap.demo;

  const mm  = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const ss  = String(elapsed % 60).padStart(2, '0');
  const pct = maxSteps > 0 ? (step / maxSteps) * 100 : 0;
  const zoneColor = ZONE_COLORS[zone] || '#7a859f';

  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 14px', height: 48,
      background: 'rgba(8,8,22,0.97)',
      borderBottom: '1px solid rgba(255,255,255,0.07)',
      flexShrink: 0, gap: 20,
      boxShadow: '0 2px 20px rgba(0,0,0,0.5)',
      animation: 'headerGlow 4s ease-in-out infinite',
    }}>
      {/* Brand */}
      <div style={{ display:'flex', alignItems:'center', gap:12 }}>
        <span style={{
          fontFamily:'var(--font-mono)', fontWeight:700, fontSize:14,
          letterSpacing:'0.2em', color:'var(--text-primary)',
        }}>
          <span style={{ color:'var(--red-team)' }}>C</span>IPHER{' '}
          <span style={{ color:'var(--text-muted)', fontWeight:400, fontSize:11 }}>WAR ROOM</span>
        </span>
        {/* live mode badge */}
        <div className="live-indicator">
          <div className={`live-pulse ${mode.dotCls}`} style={{ color: mode.color }} />
          <span style={{ color: mode.color, fontSize:9, letterSpacing:'0.15em' }}>{mode.label}</span>
        </div>
      </div>

      {/* Episode + step */}
      <div style={{ display:'flex', alignItems:'center', gap:20 }}>
        <div style={{ textAlign:'center' }}>
          <div style={{ fontFamily:'var(--font-mono)', fontSize:9, color:'var(--text-muted)', letterSpacing:'0.12em' }}>EPISODE</div>
          <div style={{ fontFamily:'var(--font-mono)', fontSize:16, fontWeight:700 }}>{episode}</div>
        </div>

        {/* Step progress bar */}
        <div style={{ width:140 }}>
          <div style={{ display:'flex', justifyContent:'space-between', fontFamily:'var(--font-mono)', fontSize:9, color:'var(--text-muted)', marginBottom:4 }}>
            <span>STEP {step}</span><span>/{maxSteps}</span>
          </div>
          <div style={{ height:4, background:'rgba(255,255,255,0.07)', borderRadius:2, overflow:'hidden' }}>
            <div style={{ height:'100%', width:`${pct}%`, background:'linear-gradient(90deg,#448aff,#ff1744)', borderRadius:2, transition:'width 0.8s ease', boxShadow:`0 0 8px rgba(255,23,68,0.4)` }} />
          </div>
        </div>
      </div>

      {/* Zone */}
      <div style={{ display:'flex', alignItems:'center', gap:8 }}>
        <div style={{ width:8, height:8, borderRadius:'50%', background:zoneColor, boxShadow:`0 0 8px ${zoneColor}` }} />
        <div style={{ fontFamily:'var(--font-mono)', fontSize:11, fontWeight:600, color:zoneColor, letterSpacing:'0.08em' }}>{zone}</div>
      </div>

      {/* Suspicion mini */}
      <div style={{ display:'flex', alignItems:'center', gap:8 }}>
        <span style={{ fontFamily:'var(--font-mono)', fontSize:9, color:'var(--text-muted)', letterSpacing:'0.1em' }}>SUSPICION</span>
        <div style={{ width:80, height:5, background:'rgba(255,255,255,0.07)', borderRadius:3, overflow:'hidden' }}>
          <div style={{
            height:'100%', width:`${suspicion*100}%`, borderRadius:3, transition:'width 0.8s ease',
            background: suspicion > 0.7 ? '#ff1744' : suspicion > 0.4 ? '#ff9800' : '#69f0ae',
            boxShadow: suspicion > 0.7 ? '0 0 8px rgba(255,23,68,0.7)' : 'none',
          }} />
        </div>
        <span style={{ fontFamily:'var(--font-mono)', fontSize:11, fontWeight:700, color: suspicion>0.7?'var(--red-team)':suspicion>0.4?'var(--yellow)':'var(--green)', minWidth:34, textAlign:'right' }}>
          {(suspicion * 100).toFixed(0)}%
        </span>
      </div>

      {/* Timer */}
      <div style={{ display:'flex', flexDirection:'column', alignItems:'center' }}>
        <div style={{ fontFamily:'var(--font-mono)', fontSize:9, color:'var(--text-muted)', letterSpacing:'0.12em' }}>ELAPSED</div>
        <div style={{ fontFamily:'var(--font-mono)', fontSize:16, fontWeight:700, letterSpacing:'0.08em' }}>{mm}:{ss}</div>
      </div>

      {/* Exfil counter */}
      {exfil > 0 && (
        <div style={{
          display:'flex', alignItems:'center', gap:6, padding:'3px 10px',
          background:'rgba(255,23,68,0.15)', border:'1px solid rgba(255,23,68,0.4)',
          borderRadius:5, animation:'fadeSlideIn 0.3s ease-out',
        }}>
          <span style={{ fontSize:12 }}>💀</span>
          <span style={{ fontFamily:'var(--font-mono)', fontSize:10, fontWeight:700, color:'var(--red-team)', letterSpacing:'0.1em' }}>
            {exfil} FILE{exfil>1?'S':''} EXFILTRATED
          </span>
        </div>
      )}
    </div>
  );
}
