import { useMemo, useState } from 'react';

const AGENT_SHORT = {
  red_planner_01: 'PLANNER',
  red_analyst_01: 'ANALYST',
  red_operative_01: 'OPERATIVE',
  red_exfiltrator_01: 'EXFIL',
  blue_surveillance_01: 'SURVEILLANCE',
  blue_threat_hunter_01: 'THREAT HUNTER',
  blue_deception_architect_01: 'DECEPTION',
  blue_forensics_01: 'FORENSICS',
};

function shortName(id = '') {
  if (AGENT_SHORT[id]) return AGENT_SHORT[id];
  return String(id).replace(/_\d+$/, '').replace(/_/g, ' ').toUpperCase();
}

export default function MindsPanel({ thoughts = [] }) {
  const [filter, setFilter] = useState('all');

  const redCount = useMemo(() => thoughts.filter((t) => t.team === 'red').length, [thoughts]);
  const blueCount = useMemo(() => thoughts.filter((t) => t.team === 'blue').length, [thoughts]);

  const rows = useMemo(() => {
    const filtered = thoughts.filter((t) => filter === 'all' || t.team === filter);
    return [...filtered].reverse().slice(0, 40);
  }, [thoughts, filter]);

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ padding: '7px 10px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.14em', color: 'var(--text-mute)' }}>
            MINDS
          </span>
          <span style={{ marginLeft: 'auto', fontFamily: 'var(--mono)', fontSize: 8, color: 'rgba(170,190,225,0.55)' }}>
            {thoughts.length} thoughts
          </span>
        </div>
        <div style={{ display: 'flex', gap: 5, marginTop: 6, fontFamily: 'var(--mono)', fontSize: 8 }}>
          <span style={{ padding: '2px 6px', borderRadius: 999, color: 'rgba(190,210,240,0.9)', background: 'rgba(130,160,220,0.16)' }}>ALL {thoughts.length}</span>
          <span style={{ padding: '2px 6px', borderRadius: 999, color: '#ff8a8a', background: 'rgba(255,68,68,0.16)' }}>RED {redCount}</span>
          <span style={{ padding: '2px 6px', borderRadius: 999, color: '#9ecfff', background: 'rgba(68,136,255,0.16)' }}>BLUE {blueCount}</span>
        </div>
        <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
          {[
            ['all', 'ALL'],
            ['red', 'RED'],
            ['blue', 'BLUE'],
          ].map(([id, label]) => (
            <button
              key={id}
              onClick={() => setFilter(id)}
              style={{
                flex: 1,
                border: `1px solid ${filter === id ? 'rgba(0,229,255,0.45)' : 'rgba(140,160,210,0.18)'}`,
                background: filter === id ? 'rgba(0,229,255,0.12)' : 'rgba(30,36,52,0.5)',
                color: filter === id ? 'var(--text)' : 'var(--text-mute)',
                borderRadius: 6,
                padding: '4px 0',
                cursor: 'pointer',
                fontFamily: 'var(--mono)',
                fontSize: 8.5,
                fontWeight: 700,
                letterSpacing: '0.1em',
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '8px', display: 'flex', flexDirection: 'column', gap: 7 }}>
        {rows.length === 0 && (
          <div style={{ padding: 8, fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-mute)', fontStyle: 'italic' }}>
            Awaiting agent reasoning…
          </div>
        )}

        {rows.map((t, i) => (
          <div
            key={`${t.agent_id}-${t.step}-${i}`}
            style={{
              border: `1px solid ${t.team === 'red' ? 'rgba(255,68,68,0.24)' : 'rgba(68,136,255,0.24)'}`,
              background: t.team === 'red'
                ? 'linear-gradient(180deg, rgba(255,68,68,0.09), rgba(255,68,68,0.05))'
                : 'linear-gradient(180deg, rgba(68,136,255,0.09), rgba(68,136,255,0.05))',
              borderRadius: 9,
              padding: '8px 10px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
              <span
                style={{
                  borderRadius: 999,
                  padding: '1px 6px',
                  fontFamily: 'var(--mono)',
                  fontSize: 7.5,
                  fontWeight: 700,
                  letterSpacing: '0.09em',
                  color: t.team === 'red' ? '#ff9d9d' : '#9ecfff',
                  background: t.team === 'red' ? 'rgba(255,68,68,0.18)' : 'rgba(68,136,255,0.18)',
                  border: `1px solid ${t.team === 'red' ? 'rgba(255,68,68,0.35)' : 'rgba(68,136,255,0.35)'}`,
                }}
              >
                {String(t.team || 'agent').toUpperCase()}
              </span>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 8.5, fontWeight: 700, letterSpacing: '0.11em', color: t.team === 'red' ? '#ff6b87' : '#7eb3ff' }}>
                {shortName(t.agent_id)}
              </span>
              <span style={{ marginLeft: 'auto', fontFamily: 'var(--mono)', fontSize: 8, color: 'rgba(170,190,225,0.55)' }}>
                STEP {t.step}
              </span>
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10.5, lineHeight: 1.45, color: 'rgba(212,224,244,0.84)' }}>
              {String(t.reasoning || '').slice(0, 210)}
              {String(t.reasoning || '').length > 210 ? '…' : ''}
            </div>
            {t.action_type && (
              <div style={{ marginTop: 6, display: 'inline-flex', padding: '3px 8px', borderRadius: 5, fontFamily: 'var(--mono)', fontSize: 8.5, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: t.team === 'red' ? '#ff9d9d' : '#9ecfff', background: t.team === 'red' ? 'rgba(255,68,68,0.12)' : 'rgba(68,136,255,0.12)' }}>
                {String(t.action_type).replace(/_/g, ' ')}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
