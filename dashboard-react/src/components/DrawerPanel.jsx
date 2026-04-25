import { useState } from 'react';

const AGENT_SHORT = {
  red_planner_01: 'PLANNER', red_analyst_01: 'ANALYST',
  red_operative_01: 'OPERATIVE', red_exfiltrator_01: 'EXFIL',
  blue_surveillance_01: 'SURVEILLANCE', blue_threat_hunter_01: 'THREAT HUNTER',
  blue_deception_architect_01: 'DECEPTION', blue_forensics_01: 'FORENSICS',
};

const ACT_ICONS = {
  move:'→', read_file:'📄', write_dead_drop:'📦', plant_false_trail:'👻',
  exfil_file:'💀', scan_network:'📡', investigate_node:'🔍', place_honeypot:'🪤',
  analyze_anomaly:'📊', trigger_alert:'🚨', reconstruct_path:'🗺', plant_breadcrumb:'🍞',
  tamper_dead_drop:'✂', plant_temporal_decoy:'⏰', abort:'⛔', stand_down:'🛑',
};

function stepsToEvents(steps) {
  const evts = [];
  for (const s of steps) {
    const ts = s.timestamp ? new Date(s.timestamp).toLocaleTimeString('en', { hour12: false }) : '--';
    if (s.red_action && s.red_action !== 'waiting') {
      evts.push({ id: `r${s.step}`, team: 'red', ts, text: s.red_action.replace('_', ' '), critical: s.red_action.includes('exfil') });
    }
    if (s.blue_actions && s.blue_actions !== '—') {
      evts.push({ id: `b${s.step}`, team: 'blue', ts, text: s.blue_actions.replace(/_/g, ' '), critical: s.blue_actions.includes('alert') });
    }
  }
  return evts.reverse().slice(0, 80);
}

export default function DrawerPanel({ thoughts, steps, isOpen, onToggle }) {
  const [tab, setTab] = useState('thoughts');
  const [teamFilter, setTeamFilter] = useState('all');
  const events = stepsToEvents(steps);
  const redT   = thoughts.filter(t => t.team === 'red');
  const blueT  = thoughts.filter(t => t.team === 'blue');
  const recentThoughts = [...thoughts]
    .reverse()
    .filter(t => teamFilter === 'all' || t.team === teamFilter)
    .slice(0, 24);

  return (
    <div className={`side-drawer${isOpen ? '' : ' closed'}`}>
      {/* Toggle handle */}
      <button className="drawer-toggle" onClick={onToggle}>
        {isOpen ? '◀' : '▶'}
      </button>

      {/* Tab bar */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        {[['thoughts', '💭 MINDS'], ['feed', '📡 EVENTS']].map(([id, label]) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            style={{
              flex: 1, padding: '9px 0',
              fontFamily: 'var(--mono)', fontSize: 9.5, fontWeight: 700,
              letterSpacing: '0.12em', textTransform: 'uppercase',
              background: 'none', border: 'none', cursor: 'pointer',
              color: tab === id ? 'var(--text)' : 'var(--text-mute)',
              borderBottom: tab === id ? '2px solid var(--z0)' : '2px solid transparent',
              transition: 'all 0.2s',
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Thoughts */}
      {tab === 'thoughts' && (
        <div className="thought-list">
          <div className="thought-toolbar">
            <div className="thought-counts">
              <span className="count-pill all">ALL {thoughts.length}</span>
              <span className="count-pill red">RED {redT.length}</span>
              <span className="count-pill blue">BLUE {blueT.length}</span>
            </div>
            <div className="thought-filters">
              {[
                ['all', 'All'],
                ['red', 'Red'],
                ['blue', 'Blue'],
              ].map(([id, label]) => (
                <button
                  key={id}
                  className={`thought-filter-btn${teamFilter === id ? ' active' : ''}`}
                  onClick={() => setTeamFilter(id)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {thoughts.length === 0 && (
            <div style={{ padding: 12, fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-mute)', fontStyle: 'italic' }}>
              Awaiting agent reasoning…
            </div>
          )}
          {recentThoughts.map((t, i) => (
            <div key={`${t.agent_id}-${t.step}-${i}`} className={`thought-item ${t.team}`}>
              <div className="thought-meta">
                <span className={`thought-team-badge ${t.team}`}>
                  {t.team?.toUpperCase() ?? 'AGENT'}
                </span>
                <span className="thought-who">
                  {AGENT_SHORT[t.agent_id] ?? t.agent_id?.replace(/_\d+$/, '').toUpperCase()}
                </span>
                <span className="thought-step">STEP {t.step}</span>
              </div>
              <div className="thought-body">
                {t.reasoning?.slice(0, 180)}
                {(t.reasoning?.length ?? 0) > 180 ? '…' : ''}
              </div>
              {t.action_type && (
                <div className="thought-action">
                  {ACT_ICONS[t.action_type] ?? '•'} {t.action_type.replace(/_/g, ' ')}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Feed */}
      {tab === 'feed' && (
        <div className="feed-list">
          <div className="feed-terminal-head">
            <span className="term-dot red" />
            <span className="term-dot amber" />
            <span className="term-dot green" />
            <span className="term-title">live_event_stream.log</span>
            <span className="term-state">STREAMING</span>
          </div>
          {events.length === 0 && (
            <div className="feed-empty">
              [waiting] no events yet...
            </div>
          )}
          {events.map((e, i) => (
            <div key={e.id} className={`feed-row terminal${e.critical ? ' alert' : ''}`} style={{ opacity: 1 - i * 0.012 }}>
              <span className="feed-gutter">{'>'}</span>
              <span className="feed-time">[{e.ts}]</span>
              <span className={`feed-team ${e.team}`}>{e.team === 'red' ? 'RED' : 'BLUE'}</span>
              <span className={`feed-desc ${e.team}`}>{e.text}</span>
            </div>
          ))}
          {events.length > 0 && (
            <div className="feed-cursor-row">
              <span className="feed-gutter">{'>'}</span>
              <span className="feed-cursor">_</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
