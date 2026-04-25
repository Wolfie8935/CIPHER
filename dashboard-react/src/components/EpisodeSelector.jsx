import { useState, useEffect } from 'react';

export default function EpisodeSelector({ onSelect }) {
  const [episodes, setEpisodes] = useState([]);
  const [loading,  setLoading]  = useState(true);
  const [hovering, setHovering] = useState(null);

  useEffect(() => {
    const fetchEpisodes = () => {
      fetch('/api/episodes')
        .then(r => r.json())
        .then(data => { setEpisodes(Array.isArray(data) ? data : []); setLoading(false); })
        .catch(() => setLoading(false));
    };
    
    fetchEpisodes();
    const intervalId = setInterval(fetchEpisodes, 3000);
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="episode-selector" style={{ animation: 'selectorFadeIn 0.5s ease-out' }}>

      {/* ── Brand ── */}
      <div style={{ textAlign: 'center' }}>
        <div style={{
          fontFamily: 'var(--mono)', fontWeight: 700, fontSize: 52,
          letterSpacing: '0.24em', lineHeight: 1,
          textShadow: '0 0 60px rgba(255,68,68,0.25)',
        }}>
          <span style={{ color: '#ff4444' }}>C</span>
          <span style={{ color: 'var(--text)' }}>IPHER</span>
        </div>
        <div style={{
          fontFamily: 'var(--mono)', fontWeight: 400, fontSize: 10.5,
          letterSpacing: '0.40em', color: 'var(--text-mute)',
          marginTop: 8, textTransform: 'uppercase',
        }}>
          Adversarial Multi-Agent War Room
        </div>
      </div>

      {/* ── Dividers with color strips ── */}
      <div style={{ display: 'flex', gap: 8, opacity: 0.6 }}>
        {['#00e5ff','#69f0ae','#ffd740','#ff6b6b'].map(c => (
          <div key={c} style={{ width: 36, height: 3, borderRadius: 2, background: c }} />
        ))}
      </div>

      {/* ── Live button ── */}
      <button
        onClick={() => onSelect('live')}
        style={{
          padding: '14px 52px',
          background: 'linear-gradient(135deg, rgba(255,68,68,0.2), rgba(180,20,20,0.25))',
          border: '1.5px solid rgba(255,68,68,0.45)',
          borderRadius: 12,
          fontFamily: 'var(--mono)',
          fontSize: 13, fontWeight: 700,
          letterSpacing: '0.18em',
          color: '#ff6666',
          cursor: 'pointer',
          boxShadow: '0 6px 28px rgba(255,68,68,0.2), inset 0 1px 0 rgba(255,68,68,0.1)',
          transition: 'all 0.2s',
          textTransform: 'uppercase',
        }}
        onMouseEnter={e => {
          e.currentTarget.style.borderColor = 'rgba(255,68,68,0.75)';
          e.currentTarget.style.boxShadow = '0 8px 36px rgba(255,68,68,0.32)';
          e.currentTarget.style.transform = 'translateY(-2px)';
        }}
        onMouseLeave={e => {
          e.currentTarget.style.borderColor = 'rgba(255,68,68,0.45)';
          e.currentTarget.style.boxShadow = '0 6px 28px rgba(255,68,68,0.2)';
          e.currentTarget.style.transform = '';
        }}
      >
        ⬤ &nbsp;Connect to Live Feed
      </button>

      {/* ── Divider ── */}
      {(episodes.length > 0 || !loading) && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, width: '90%', maxWidth: 900 }}>
          <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
          <span style={{ fontFamily: 'var(--mono)', fontSize: 9, letterSpacing: '0.24em', color: 'var(--text-mute)' }}>
            OR REPLAY AN EPISODE
          </span>
          <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
        </div>
      )}

      {/* ── Episode grid ── */}
      {episodes.length > 0 && (
        <div className="episode-grid">
          {episodes.map((ep, i) => {
            const parts  = ep.replace('.json', '').split('_');
            const epNum  = parts.find(p => /^\d+$/.test(p)) ?? String(i + 1);
            const isHov  = hovering === ep;
            const zoneColor = ['#00e5ff','#69f0ae','#ffd740','#ff6b6b'][i % 4];

            let mode = 'N/A';
            let timestamp = 'Legacy Trace';
            if (parts.length >= 4) {
              mode = parts[parts.length - 1];
              const timePart = parts[parts.length - 2];
              const datePart = parts[parts.length - 3];
              if (/^\d{6}$/.test(timePart) && /^\d{8}$/.test(datePart)) {
                 timestamp = `${datePart.slice(4,6)}/${datePart.slice(6,8)} ${timePart.slice(0,2)}:${timePart.slice(2,4)}`;
              } else {
                 timestamp = parts.slice(2, -1).join('_');
              }
            }

            return (
              <div
                key={ep}
                className="episode-card"
                onClick={() => onSelect(ep)}
                onMouseEnter={() => setHovering(ep)}
                onMouseLeave={() => setHovering(null)}
                style={{
                  borderColor: isHov ? zoneColor : undefined,
                  boxShadow: isHov ? `0 6px 24px ${zoneColor}22` : undefined,
                }}
              >
                <div style={{
                  fontFamily: 'var(--mono)', fontSize: 8,
                  letterSpacing: '0.20em', color: 'var(--text-mute)',
                  marginBottom: 4, textTransform: 'uppercase',
                }}>
                  {timestamp}
                </div>
                <div style={{
                  fontFamily: 'var(--mono)', fontSize: 26, fontWeight: 700,
                  color: isHov ? zoneColor : 'var(--text)',
                  transition: 'color 0.15s',
                }}>
                  #{epNum.padStart(3, '0')}
                </div>
                <div style={{
                  fontFamily: 'var(--mono)', fontSize: 9.5,
                  color: zoneColor, marginTop: 5,
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  opacity: 0.8
                }}>
                  MODE: {mode}
                </div>
                {isHov && (
                  <div style={{
                    marginTop: 9, display: 'flex', alignItems: 'center', gap: 5,
                    fontFamily: 'var(--mono)', fontSize: 9,
                    color: zoneColor, letterSpacing: '0.1em', fontWeight: 700,
                  }}>
                    <span style={{ fontSize: 12 }}>▶</span> Replay
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── States ── */}
      {loading && (
        <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--text-mute)', letterSpacing: '0.14em' }}>
          Loading episodes…
        </div>
      )}

      {!loading && episodes.length === 0 && (
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 10.5, color: 'var(--text-mute)',
          letterSpacing: '0.1em', textAlign: 'center', lineHeight: 2.2,
        }}>
          No episode traces found.
          <br />
          Run{' '}
          <code style={{
            background: 'rgba(140,160,210,0.10)',
            color: 'var(--z0)',
            padding: '2px 9px', borderRadius: 5,
          }}>python main.py</code>
          {' '}to generate episodes.
        </div>
      )}

      {/* ── Footer ── */}
      <div style={{
        fontFamily: 'var(--mono)', fontSize: 8.5, color: 'rgba(140,160,210,0.22)',
        letterSpacing: '0.1em', textAlign: 'center',
      }}>
        CIPHER · Multi-Agent Reinforcement Learning · OpenEnv 2026
      </div>
    </div>
  );
}
