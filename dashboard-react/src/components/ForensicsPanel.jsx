import { useState, useEffect } from 'react';

const S = {
  root: { padding: '18px 20px', fontFamily: 'var(--mono)', color: 'var(--text)', height: '100%', overflowY: 'auto' },
  title: { fontSize: 13, fontWeight: 700, letterSpacing: '0.16em', color: '#e6edf3', marginBottom: 14, borderBottom: '1px solid rgba(68,136,255,0.25)', paddingBottom: 8 },
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 14 },
  card: { background: 'rgba(22,27,34,0.85)', border: '1px solid rgba(68,136,255,0.18)', borderRadius: 8, padding: '12px 14px' },
  cardTitle: { fontSize: 9, letterSpacing: '0.14em', color: '#8b949e', marginBottom: 6 },
  cardValue: { fontSize: 22, fontWeight: 700 },
  badge: (grade) => {
    const colors = { A: '#2ecc71', B: '#4488ff', C: '#f0c040', D: '#ff8800', F: '#ff4444' };
    return { display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 40, height: 40, borderRadius: 8, background: `${colors[grade] ?? '#8b949e'}22`, border: `2px solid ${colors[grade] ?? '#8b949e'}`, color: colors[grade] ?? '#8b949e', fontSize: 20, fontWeight: 800 };
  },
  barWrap: { width: '100%', height: 8, background: 'rgba(255,255,255,0.07)', borderRadius: 4, overflow: 'hidden', marginTop: 4 },
  bar: (pct, color) => ({ width: `${Math.max(0, Math.min(100, pct * 100))}%`, height: '100%', background: color, borderRadius: 4, transition: 'width 0.5s ease' }),
  section: { marginBottom: 14 },
  sectionTitle: { fontSize: 9, letterSpacing: '0.14em', color: '#8b949e', marginBottom: 6 },
  nodeList: { display: 'flex', flexWrap: 'wrap', gap: 4, fontSize: 11 },
  nodeChip: (color) => ({ padding: '2px 7px', borderRadius: 4, background: `${color}22`, border: `1px solid ${color}55`, color, fontWeight: 600, fontSize: 10 }),
  timelineItem: { display: 'flex', gap: 10, alignItems: 'flex-start', marginBottom: 6, paddingBottom: 6, borderBottom: '1px solid rgba(255,255,255,0.04)' },
  timelineStep: { minWidth: 28, fontSize: 9, color: '#8b949e', paddingTop: 1 },
  timelineText: { fontSize: 11, color: '#c9d1d9', lineHeight: 1.5 },
  empty: { color: '#8b949e', fontSize: 12, textAlign: 'center', padding: '40px 0' },
  stat: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '5px 0', borderBottom: '1px solid rgba(255,255,255,0.04)', fontSize: 11 },
  statLabel: { color: '#8b949e' },
  statValue: { color: '#e6edf3', fontWeight: 600 },
};

function GradeCard({ grade, accuracy }) {
  const gradeColor = { A: '#2ecc71', B: '#4488ff', C: '#f0c040', D: '#ff8800', F: '#ff4444' }[grade] ?? '#8b949e';
  return (
    <div style={S.card}>
      <div style={S.cardTitle}>INVESTIGATION GRADE</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={S.badge(grade)}>{grade ?? '?'}</div>
        <div>
          <div style={{ ...S.cardValue, fontSize: 16, color: '#e6edf3' }}>
            {accuracy != null ? `${(accuracy * 100).toFixed(0)}%` : '—'} accuracy
          </div>
          <div style={{ fontSize: 10, color: '#8b949e', marginTop: 2 }}>
            {grade === 'A' ? 'Excellent reconstruction' : grade === 'B' ? 'Good reconstruction' : grade === 'C' ? 'Partial reconstruction' : grade === 'D' ? 'Minimal reconstruction' : grade === 'F' ? 'Failed reconstruction' : 'Pending'}
          </div>
        </div>
      </div>
      <div style={S.barWrap}><div style={S.bar(accuracy ?? 0, gradeColor)} /></div>
    </div>
  );
}

function TrapCard({ efficiency, triggered, wasted }) {
  const total = (triggered ?? 0) + (wasted ?? 0);
  const color = efficiency > 0.6 ? '#2ecc71' : efficiency > 0.3 ? '#f0c040' : '#ff4444';
  return (
    <div style={S.card}>
      <div style={S.cardTitle}>TRAP EFFICIENCY</div>
      <div style={{ ...S.cardValue, color }}>{efficiency != null ? `${(efficiency * 100).toFixed(0)}%` : '—'}</div>
      <div style={{ fontSize: 10, color: '#8b949e', margin: '2px 0 4px' }}>{triggered ?? 0}/{total} traps triggered</div>
      <div style={S.barWrap}><div style={S.bar(efficiency ?? 0, '#f0c040')} /></div>
    </div>
  );
}

function NodeSection({ title, nodes, color }) {
  if (!nodes?.length) return null;
  return (
    <div style={S.section}>
      <div style={S.sectionTitle}>{title}</div>
      <div style={S.nodeList}>
        {nodes.map((n, i) => <span key={i} style={S.nodeChip(color)}>node {n}</span>)}
      </div>
    </div>
  );
}

function Timeline({ timeline }) {
  if (!timeline?.length) return null;
  const recent = [...timeline].slice(-20).reverse();
  return (
    <div style={S.section}>
      <div style={S.sectionTitle}>OPERATION TIMELINE (last {recent.length} events)</div>
      <div style={{ maxHeight: 220, overflowY: 'auto' }}>
        {recent.map((evt, i) => (
          <div key={i} style={S.timelineItem}>
            <div style={S.timelineStep}>S{evt.step ?? i}</div>
            <div style={S.timelineText}>{evt.event ?? evt.description ?? JSON.stringify(evt)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ForensicsPanel({ steps, selectedEpisode }) {
  const [forensics, setForensics] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    const loadForensics = async () => {
      // Priority 1: load from selected episode trace
      if (selectedEpisode && selectedEpisode !== 'live') {
        try {
          const r = await fetch(`/api/episode/${encodeURIComponent(selectedEpisode)}`);
          if (r.ok && !cancelled) {
            const data = await r.json();
            const recon = data?.forensics_reconstruction;
            if (recon) { setForensics(recon); setLoading(false); return; }
          }
        } catch { /* fall through */ }
      }

      // Priority 2: latest trace from api
      if (!cancelled) {
        try {
          const r = await fetch('/api/forensics');
          if (r.ok && !cancelled) {
            const data = await r.json();
            if (data) { setForensics(data); setLoading(false); return; }
          }
        } catch { /* fall through */ }
      }

      if (!cancelled) { setForensics(null); setLoading(false); }
    };

    loadForensics();
    return () => { cancelled = true; };
  }, [selectedEpisode]);

  // Also poll for live updates
  useEffect(() => {
    if (selectedEpisode && selectedEpisode !== 'live') return;
    const id = setInterval(async () => {
      try {
        const r = await fetch('/api/forensics');
        if (r.ok) { const d = await r.json(); if (d) setForensics(d); }
      } catch { /* ignore */ }
    }, 5000);
    return () => clearInterval(id);
  }, [selectedEpisode]);

  // Derive from steps as fallback
  const derivedForensics = (() => {
    if (forensics) return forensics;
    if (!steps?.length) return null;
    const last = steps[steps.length - 1];
    return last?.forensics_reconstruction ?? null;
  })();

  if (loading && !derivedForensics) {
    return <div style={S.root}><div style={S.title}>🔍 FORENSICS RECONSTRUCTION</div><div style={S.empty}>Loading…</div></div>;
  }

  if (!derivedForensics) {
    return (
      <div style={S.root}>
        <div style={S.title}>🔍 FORENSICS RECONSTRUCTION</div>
        <div style={S.empty}>
          No forensics data for this episode.<br />
          <span style={{ fontSize: 10, color: '#555e6b' }}>
            Forensics runs after each episode. Run <code>python main.py</code> to generate data.
          </span>
        </div>
      </div>
    );
  }

  const f = derivedForensics;
  const trapsTriggered = f.traps_triggered?.length ?? 0;
  const trapsWasted = f.traps_wasted?.length ?? 0;

  return (
    <div style={S.root}>
      <div style={S.title}>🔍 FORENSICS RECONSTRUCTION</div>

      <div style={S.grid}>
        <GradeCard grade={f.investigation_grade} accuracy={f.path_accuracy} />
        <TrapCard efficiency={f.trap_efficiency} triggered={trapsTriggered} wasted={trapsWasted} />
      </div>

      {f.summary_text && (
        <div style={{ ...S.card, marginBottom: 14, fontSize: 11, lineHeight: 1.6, color: '#c9d1d9' }}>{f.summary_text}</div>
      )}

      <div style={{ ...S.card, marginBottom: 14 }}>
        <div style={S.cardTitle}>PATH ANALYSIS</div>
        <div style={S.stat}><span style={S.statLabel}>RED path length</span><span style={S.statValue}>{f.actual_red_path?.length ?? 0} nodes</span></div>
        <div style={S.stat}><span style={S.statLabel}>Suspected path</span><span style={S.statValue}>{f.suspected_red_path?.length ?? 0} nodes</span></div>
        <div style={S.stat}><span style={S.statLabel}>Correctly identified</span><span style={{ ...S.statValue, color: '#2ecc71' }}>{f.correctly_identified_nodes?.length ?? 0}</span></div>
        <div style={S.stat}><span style={S.statLabel}>Missed nodes</span><span style={{ ...S.statValue, color: '#ff4444' }}>{f.missed_nodes?.length ?? 0}</span></div>
        <div style={S.stat}><span style={S.statLabel}>False positives</span><span style={{ ...S.statValue, color: '#ff8800' }}>{f.false_positive_nodes?.length ?? 0}</span></div>
        <div style={S.stat}>
          <span style={S.statLabel}>Dead drop integrity</span>
          <span style={{ ...S.statValue, color: f.drop_integrity_rate > 0.7 ? '#2ecc71' : '#ff8800' }}>
            {f.drop_integrity_rate != null ? `${(f.drop_integrity_rate * 100).toFixed(0)}%` : '—'}
          </span>
        </div>
      </div>

      <NodeSection title="CORRECTLY IDENTIFIED NODES" nodes={f.correctly_identified_nodes} color="#2ecc71" />
      <NodeSection title="MISSED NODES (RED escaped undetected)" nodes={f.missed_nodes} color="#ff4444" />
      <NodeSection title="FALSE POSITIVES (BLUE flagged incorrectly)" nodes={f.false_positive_nodes} color="#ff8800" />
      <Timeline timeline={f.timeline} />
    </div>
  );
}
