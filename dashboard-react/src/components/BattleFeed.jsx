import { useState, useRef, useEffect } from 'react';

const FILTER_OPTIONS = ['ALL', 'RED', 'BLUE', 'TRAPS', 'CRITICAL'];

const EVENT_ICONS = {
  move:                 { icon: '→',  cls: 'red' },
  read_file:            { icon: '📄', cls: 'red' },
  write_dead_drop:      { icon: '📦', cls: 'red' },
  plant_false_trail:    { icon: '👻', cls: 'red' },
  plant_temporal_decoy: { icon: '⏰', cls: 'red' },
  exfil_file:           { icon: '💀', cls: 'red' },
  abort:                { icon: '⛔', cls: 'dim' },
  scan_network:         { icon: '📡', cls: 'blue' },
  investigate_node:     { icon: '🔍', cls: 'blue' },
  place_honeypot:       { icon: '🪤', cls: 'blue' },
  plant_breadcrumb:     { icon: '🍞', cls: 'blue' },
  analyze_anomaly:      { icon: '📊', cls: 'blue' },
  trigger_alert:        { icon: '🚨', cls: 'gold' },
  reconstruct_path:     { icon: '🗺', cls: 'blue' },
  tamper_dead_drop:     { icon: '✂',  cls: 'blue' },
};

function stepsToEvents(steps) {
  const events = [];
  for (const s of steps) {
    const ts = s.timestamp ? new Date(s.timestamp).toLocaleTimeString('en',{ hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit' }) : '--:--:--';
    if (s.red_action && s.red_action !== 'waiting') {
      const parts = s.red_action.split(' → ');
      const atype = parts[0];
      const node  = parts[1] || null;
      const cfg   = EVENT_ICONS[atype] || { icon: '•', cls: 'red' };
      const isExfil = atype.includes('exfil');
      events.push({ id:`r-${s.step}`, team:'red', ts, atype, node, icon:cfg.icon, dotCls:cfg.cls, step:s.step, isExfil, isBreach: s.zone === 'Critical/HVT' });
    }
    if (s.blue_actions && s.blue_actions !== '—') {
      const parts = s.blue_actions.split(' ');
      for (const p of parts) {
        const m = p.match(/^(.+?)×\d+$/);
        const atype = m ? m[1] : p;
        const cfg   = EVENT_ICONS[atype] || { icon: '•', cls: 'blue' };
        const isAlert = atype === 'trigger_alert';
        events.push({ id:`b-${s.step}-${atype}`, team:'blue', ts, atype, node:null, icon:cfg.icon, dotCls:cfg.cls, step:s.step, isAlert });
      }
    }
    if (s.suspicion >= 0.85) {
      events.push({ id:`susp-${s.step}`, team:'alert', ts, atype:'HIGH_SUSPICION', icon:'⚠', dotCls:'gold', step:s.step, isAlert:true, text:`Suspicion critical: ${(s.suspicion*100).toFixed(0)}%` });
    }
  }
  return events.reverse().slice(0, 60);
}

export default function BattleFeed({ steps }) {
  const [filter, setFilter] = useState('ALL');
  const scrollRef = useRef(null);
  const prevLen   = useRef(0);

  const events = stepsToEvents(steps);

  useEffect(() => {
    if (events.length !== prevLen.current && scrollRef.current) {
      scrollRef.current.scrollTop = 0;
      prevLen.current = events.length;
    }
  }, [events.length]);

  const filtered = events.filter(e => {
    if (filter === 'ALL')      return true;
    if (filter === 'RED')      return e.team === 'red';
    if (filter === 'BLUE')     return e.team === 'blue';
    if (filter === 'TRAPS')    return ['place_honeypot','plant_breadcrumb','tamper_dead_drop','plant_false_trail','plant_temporal_decoy'].includes(e.atype);
    if (filter === 'CRITICAL') return e.isExfil || e.isBreach || e.isAlert;
    return true;
  });

  return (
    <div className="card" style={{ flex:1, minHeight:0, display:'flex', flexDirection:'column' }}>
      <div className="section-label">
        <div className="label-dot dim" />
        <span>LIVE BATTLE FEED</span>
        <span style={{ marginLeft:'auto', fontFamily:'var(--font-mono)', fontSize:9, color:'var(--text-muted)' }}>
          {events.length} events
        </span>
      </div>

      {/* Filter buttons */}
      <div className="feed-filters">
        {FILTER_OPTIONS.map(f => (
          <button
            key={f}
            className={`feed-filter-btn${filter === f ? ' active' : ''}${filter === f && f === 'RED' ? ' red' : filter === f && f === 'BLUE' ? ' blue' : ''}`}
            onClick={() => setFilter(f)}
          >
            {f}
          </button>
        ))}
      </div>

      {/* Events */}
      <div className="feed-scroll" ref={scrollRef}>
        {filtered.length === 0 && (
          <div style={{ padding:'12px 6px', fontFamily:'var(--font-mono)', fontSize:11, color:'var(--text-muted)', fontStyle:'italic' }}>
            Awaiting events…
          </div>
        )}
        {filtered.map((e, i) => (
          <div
            key={e.id}
            className={`feed-item${e.isExfil ? ' exfil' : e.isBreach ? ' breach' : e.isAlert ? ' critical' : ''}`}
            style={{ opacity: 1 - i * 0.018 }}
          >
            <div className={`feed-dot ${e.dotCls}`} />
            <span className="feed-time">{e.ts}</span>
            <span className="feed-text">
              <span style={{ marginRight:5 }}>{e.icon}</span>
              {e.text || (
                <>
                  <span style={{ color: e.team === 'red' ? '#ff6b87' : e.team === 'blue' ? '#7eb3ff' : 'var(--yellow)', fontWeight:600 }}>
                    {e.atype.replace(/_/g, ' ')}
                  </span>
                  {e.node && <span className="node-ref"> {e.node}</span>}
                </>
              )}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
