const ZONE_COLORS = {
  Perimeter:       '#00e5ff',
  General:         '#69f0ae',
  Sensitive:       '#ffd740',
  'Critical/HVT':  '#ff6b6b',
};

function GaugeBar({ value, label, color }) {
  const pct = Math.round(value * 100);
  const barColor = value > 0.75 ? '#bb0033' : value > 0.45 ? '#c05000' : color;
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{
          fontFamily: 'var(--mono)', fontSize: 9, letterSpacing: '0.12em',
          textTransform: 'uppercase', color: 'var(--text-mute)',
        }}>
          {label}
        </span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 700, color: barColor }}>
          {pct}%
        </span>
      </div>
      <div className="mini-gauge-track">
        <div
          className="mini-gauge-fill"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}cc, ${barColor})`,
            boxShadow: value > 0.75 ? `0 0 6px ${barColor}80` : 'none',
          }}
        />
      </div>
    </div>
  );
}

export default function StatsHUD({ latest, redThoughts, blueThoughts }) {
  const suspicion = latest?.suspicion ?? 0;
  const detection = latest?.detection ?? 0;
  const zone      = latest?.zone      ?? 'Perimeter';
  const step      = latest?.step      ?? 0;
  const maxSteps  = latest?.max_steps ?? 30;
  const exfil     = latest?.exfil_count ?? 0;
  const zoneColor = ZONE_COLORS[zone] ?? '#0091a8';

  const redCount  = redThoughts.length;
  const blueCount = blueThoughts.length;

  return (
    <div className="stats-hud">

      {/* Suspicion + Detection */}
      <div className="hud-card">
        <GaugeBar value={suspicion} label="SUSPICION" color="#ff4444" />
        <GaugeBar value={detection} label="DETECTION" color="#4488ff" />
      </div>

      {/* Zone + step progress */}
      <div className="hud-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 9, letterSpacing: '0.12em', color: 'var(--text-mute)', textTransform: 'uppercase' }}>
            Zone
          </span>
          <span
            className="zone-badge"
            style={{
              background: `${zoneColor}14`,
              border: `1px solid ${zoneColor}45`,
              color: zoneColor,
            }}
          >
            {zone}
          </span>
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text-mute)' }}>Step</span>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text-dim)' }}>
              {step} / {maxSteps}
            </span>
          </div>
          <div style={{ height: 5, background: 'rgba(140,160,210,0.10)', borderRadius: 3, overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              width: `${maxSteps > 0 ? (step / maxSteps) * 100 : 0}%`,
              background: 'linear-gradient(90deg, #4488ff, #ff4444)',
              borderRadius: 3, transition: 'width 0.8s ease',
            }} />
          </div>
        </div>

        {exfil > 0 && (
          <div style={{
            marginTop: 8, padding: '3px 8px',
            background: 'rgba(211,47,47,0.08)',
            border: '1px solid rgba(211,47,47,0.25)',
            borderRadius: 5,
            fontFamily: 'var(--mono)', fontSize: 9.5, fontWeight: 700,
            color: '#ff4444', letterSpacing: '0.1em', textAlign: 'center',
          }}>
            💀 {exfil} FILE{exfil > 1 ? 'S' : ''} EXFILTRATED
          </div>
        )}
      </div>

      {/* Thought counts */}
      <div className="hud-card" style={{ display: 'flex', gap: 12 }}>
        <div style={{ flex: 1, textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 20, fontWeight: 700, color: '#ff4444' }}>
            {redCount}
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8, letterSpacing: '0.1em', color: 'var(--text-mute)', textTransform: 'uppercase' }}>
            RED thoughts
          </div>
        </div>
        <div style={{ width: 1, background: 'rgba(140,160,210,0.10)' }} />
        <div style={{ flex: 1, textAlign: 'center' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 20, fontWeight: 700, color: '#4488ff' }}>
            {blueCount}
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8, letterSpacing: '0.1em', color: 'var(--text-mute)', textTransform: 'uppercase' }}>
            BLUE thoughts
          </div>
        </div>
      </div>
    </div>
  );
}
