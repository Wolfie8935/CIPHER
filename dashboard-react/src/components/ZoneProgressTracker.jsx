const ZONES = [
  { key: 0, name: 'Perimeter', label: '◌', color: '#7a859f' },
  { key: 1, name: 'General',   label: '◉', color: '#00bcd4' },
  { key: 2, name: 'Sensitive', label: '◈', color: '#ffc107' },
  { key: 3, name: 'Critical',  label: '◆', color: '#ff1744' },
];

const ZONE_NAME_TO_IDX = { 'Perimeter': 0, 'General': 1, 'Sensitive': 2, 'Critical/HVT': 3, 'Critical': 3 };

export default function ZoneProgressTracker({ zone = 'Perimeter' }) {
  const activeIdx = ZONE_NAME_TO_IDX[zone] ?? 0;

  return (
    <div style={{ padding: '8px 16px 16px' }}>
      <div className="section-label" style={{ padding:'0 0 6px', border:'none', marginBottom:8 }}>
        <div className="label-dot red" />
        <span style={{ color:'var(--text-muted)' }}>ZONE PROGRESS</span>
      </div>
      <div style={{ display:'flex', alignItems:'center', position:'relative' }}>
        {ZONES.map((z, i) => (
          <div key={z.key} className={`zone-step${activeIdx === i ? ' active' : ''}`} style={{ flex:1, display:'flex', alignItems:'center' }}>
            <div style={{ position:'relative', display:'flex', flexDirection:'column', alignItems:'center' }}>
              <div
                className={`zone-node${activeIdx === i ? ' active' : ''}${activeIdx > i ? ' visited' : ''}`}
                style={{
                  borderColor: activeIdx >= i ? z.color : 'rgba(255,255,255,0.12)',
                  background:  activeIdx === i ? `rgba(${z.key === 3 ? '255,23,68' : z.key === 2 ? '255,193,7' : z.key === 1 ? '0,188,212' : '120,133,159'},0.2)` : activeIdx > i ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.03)',
                  boxShadow:   activeIdx === i ? `0 0 14px ${z.color}, 0 0 28px ${z.color}44` : 'none',
                }}
              >
                <span style={{ fontSize:10, color: activeIdx >= i ? z.color : 'rgba(255,255,255,0.2)' }}>{z.label}</span>
              </div>
              <div style={{
                position:'absolute', top:22,
                fontFamily:'var(--font-mono)', fontSize:8.5, fontWeight:600,
                letterSpacing:'0.06em', whiteSpace:'nowrap',
                color: activeIdx === i ? z.color : activeIdx > i ? 'rgba(255,255,255,0.3)' : 'var(--text-muted)',
                textShadow: activeIdx === i ? `0 0 8px ${z.color}` : 'none',
              }}>
                {z.name}
              </div>
            </div>
            {i < ZONES.length - 1 && (
              <div className={`zone-connector${activeIdx > i ? ' active' : ''}`} style={{ flex:1, height:2, background: activeIdx > i ? z.color : 'rgba(255,255,255,0.07)', boxShadow: activeIdx > i ? `0 0 6px ${z.color}88` : 'none', transition:'all 0.6s ease' }} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
