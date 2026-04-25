import { useEffect, useState } from 'react';

/** In-universe wire copy when /api/lore has no storyteller reports yet (client-only). */
const PLACEHOLDER_DISPATCHES = [
  {
    date: '2046-04-22',
    headline: 'AMBER NODE STILL WHISPERS AFTER CURFEW',
    blurb:
      'Grid auditors logged a twelve-second echo on subcarrier 7B—no payload, only rhythm. Blue teams call it equipment fatigue; the breach desk files it as “something learning the cadence of our ACKs.”',
  },
  {
    date: '2046-04-19',
    headline: 'RED HANDLER USES CAFETERIA WIFI AS COVER',
    blurb:
      'Oversight flagged a spike in DHCP churn matching a training sim that never shipped. Correlation is not causation, unless your causation is bored and wearing a stolen badge.',
  },
  {
    date: '2046-04-16',
    headline: 'ARCHIVE WORM ASKS FOR FILES IT ALREADY OWNS',
    blurb:
      'Vault integrity checks passed, yet checksum probes requested the same shard sixteen times. Either a stuck crawler or politeness so aggressive it borders on threat.',
  },
  {
    date: '2046-04-12',
    headline: 'DAILY BREACH CIRCULATION HITS RECORD LOWS',
    blurb:
      'Readers blame doomscrolling; editors blame classification. Either way, the silence is loud—like a channel that forgot to hang up.',
  },
];

export default function LorePanel() {
  const [reports, setReports] = useState([]);
  const [loreFetchReady, setLoreFetchReady] = useState(false);

  useEffect(() => {
    let mounted = true;
    const fetchLore = async () => {
      try {
        const res = await fetch('/api/lore', { signal: AbortSignal.timeout(3000) });
        if (!res.ok) return;
        const data = await res.json();
        if (mounted && Array.isArray(data)) setReports(data);
      } catch {
        // ignore
      } finally {
        if (mounted) setLoreFetchReady(true);
      }
    };
    fetchLore();
    const t = setInterval(fetchLore, 10000);
    return () => {
      mounted = false;
      clearInterval(t);
    };
  }, []);

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ padding: '8px 10px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.14em', color: 'var(--text-mute)' }}>
          THE DAILY BREACH (LORE)
        </div>
      </div>
      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '8px', fontFamily: 'var(--mono)' }}>
        {!loreFetchReady && (
          <div style={{ fontSize: 9, color: 'rgba(160,180,220,0.45)', padding: 8, letterSpacing: '0.06em' }}>
            Establishing breach uplink…
          </div>
        )}
        {loreFetchReady && reports.length === 0 && (
          <>
            {PLACEHOLDER_DISPATCHES.map((d, idx) => (
              <div
                key={`placeholder-${d.date}-${idx}`}
                style={{
                  border: '1px solid rgba(251,191,36,0.18)',
                  borderLeft: '2px solid rgba(255,215,64,0.55)',
                  borderRadius: 8,
                  padding: '8px 9px',
                  marginBottom: 8,
                  background: 'linear-gradient(180deg, rgba(18,24,36,0.72), rgba(14,18,30,0.85))',
                }}
              >
                <div style={{ color: 'rgba(255,215,64,0.75)', fontSize: 7.5, fontWeight: 700, letterSpacing: '0.12em' }}>
                  {d.date} · ARCHIVE DISPATCH
                </div>
                <div style={{ color: '#ffd740', fontSize: 8.5, fontWeight: 700, letterSpacing: '0.06em', marginTop: 4, lineHeight: 1.35 }}>
                  {d.headline}
                </div>
                <div style={{ color: 'rgba(210,225,245,0.78)', fontSize: 9, marginTop: 6, whiteSpace: 'pre-wrap', lineHeight: 1.45 }}>
                  {d.blurb}
                </div>
              </div>
            ))}
            <div
              style={{
                fontSize: 7.5,
                color: 'rgba(160,180,220,0.42)',
                padding: '4px 8px 8px',
                letterSpacing: '0.04em',
                lineHeight: 1.5,
              }}
            >
              Live episode briefs replace this feed when the storyteller pipeline runs.
            </div>
          </>
        )}
        {reports.map((r, idx) => (
          <div
            key={`${r.filename || 'report'}-${idx}`}
            style={{
              border: '1px solid rgba(251,191,36,0.25)',
              borderLeft: '2px solid #ffd740',
              borderRadius: 8,
              padding: '8px 9px',
              marginBottom: 8,
              background: 'linear-gradient(180deg, rgba(18,24,36,0.88), rgba(14,18,30,0.92))',
            }}
          >
            <div style={{ color: '#ffd740', fontSize: 8, fontWeight: 700, letterSpacing: '0.10em' }}>
              EP {r.episode ?? '?'} REPORT
            </div>
            <div style={{ color: 'rgba(160,180,220,0.6)', fontSize: 7.5, marginTop: 2 }}>
              {r.filename || ''}
            </div>
            <div style={{ color: 'rgba(210,225,245,0.8)', fontSize: 9, marginTop: 6, whiteSpace: 'pre-wrap', lineHeight: 1.45 }}>
              {String(r.text || '').slice(0, 420)}
              {String(r.text || '').length > 420 ? '…' : ''}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
