import { useState, useRef, useEffect } from 'react';
import { useBattleLog } from '../hooks/useBattleLog';

const FILTERS = ['ALL', 'RED', 'BLUE', 'SPAWNS', 'TRAPS', 'DROPS', 'CRITICAL'];

const ACTION_ICONS = {
  move:                   { icon: '→',  label: 'MOVE'        },
  read_file:              { icon: '📄', label: 'READ FILE'   },
  write_dead_drop:        { icon: '📦', label: 'DEAD DROP'   },
  read_dead_drop:         { icon: '📬', label: 'READ DROP'   },
  tamper_dead_drop:       { icon: '✂️', label: 'TAMPER DROP'  },
  plant_false_trail:      { icon: '👻', label: 'FALSE TRAIL' },
  plant_temporal_decoy:   { icon: '⏰', label: 'DECOY'       },
  plant_honeypot_poison:  { icon: '☠️', label: 'HP POISON'   },
  exfiltrate:             { icon: '💀', label: 'EXFILTRATE'  },
  exfil_file:             { icon: '💀', label: 'EXFIL FILE'  },
  abort:                  { icon: '⛔', label: 'ABORT'       },
  stand_down:             { icon: '🛑', label: 'STAND DOWN'  },
  wait:                   { icon: '⏳', label: 'WAIT'        },
  spawn_subagent:         { icon: '🤖', label: 'SPAWN AGENT' },
  scan_network:           { icon: '📡', label: 'SCAN'        },
  investigate_node:       { icon: '🔍', label: 'INVESTIGATE' },
  place_honeypot:         { icon: '🪤', label: 'HONEYPOT'    },
  plant_breadcrumb:       { icon: '🍞', label: 'BREADCRUMB'  },
  analyze_anomaly:        { icon: '📊', label: 'ANALYZE'     },
  trigger_alert:          { icon: '🚨', label: 'ALERT'       },
  reconstruct_path:       { icon: '🗺️', label: 'RECONSTRUCT' },
  trigger_false_escalation:{ icon: '📢', label: 'FALSE ESC'  },
  dead_drop_tamper:       { icon: '✂️', label: 'TAMPER'      },
  write_corrupted_drop:   { icon: '💢', label: 'CORRUPT DROP'},
  key_event:              { icon: '⚡', label: 'KEY EVENT'   },
};

function getIcon(action_type) {
  return ACTION_ICONS[action_type]?.icon ?? '•';
}

function getLabel(action_type) {
  return ACTION_ICONS[action_type]?.label ?? action_type?.replace(/_/g, ' ').toUpperCase() ?? '—';
}

function teamColor(team) {
  if (team === 'red')    return '#ff6b87';
  if (team === 'blue')   return '#7eb3ff';
  if (team === 'system') return '#ffd740';
  return 'rgba(160,180,220,0.6)';
}

function EventRow({ event, idx }) {
  const [expanded, setExpanded] = useState(false);

  const ts = event.timestamp
    ? new Date(event.timestamp).toLocaleTimeString('en', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : '--:--:--';

  const isExfil   = event.type === 'exfil' || event.action_type?.includes('exfil');
  const isTrap    = event.type === 'trap';
  const isDrop    = event.type === 'dead_drop';
  const isCrit    = event.critical || isExfil;
  const isSystem  = event.type === 'system';
  const isSpawn   = event.action_type === 'spawn_subagent';

  const rowBg = isSpawn   ? 'rgba(255,215,0,0.08)'
    : isExfil   ? 'rgba(255,68,68,0.10)'
    : isCrit    ? 'rgba(255,170,0,0.07)'
    : isTrap    ? 'rgba(255,200,0,0.05)'
    : isDrop    ? 'rgba(120,80,255,0.06)'
    : isSystem  ? 'rgba(0,229,255,0.05)'
    : 'transparent';

  const dotColor = isExfil ? '#ff4444'
    : event.team === 'red'    ? 'var(--red-agent)'
    : event.team === 'blue'   ? 'var(--blue-agent)'
    : event.team === 'system' ? 'var(--gold)'
    : '#555';

  const roleStr = event.role && event.role !== 'trap' && event.role !== 'drop' && event.role !== 'system'
    ? event.role.toUpperCase()
    : null;

  return (
    <div
      className={`feed-item${isCrit ? ' critical' : ''}${isSpawn ? ' spawn-event' : ''}`}
      style={{
        background: rowBg,
        padding: '5px 8px 4px',
        borderBottom: '1px solid rgba(140,160,210,0.04)',
        cursor: event.reasoning ? 'pointer' : 'default',
        opacity: Math.max(0.55, 1 - idx * 0.012),
        borderLeft: isSpawn  ? '2px solid rgba(255,215,0,0.7)'
          : isCrit ? `2px solid ${dotColor}` : '2px solid transparent',
        boxShadow: isSpawn ? 'inset 0 0 12px rgba(255,215,0,0.04)' : 'none',
      }}
      onClick={() => event.reasoning && setExpanded(e => !e)}
    >
      {/* Top row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
        <span style={{ width: 8, height: 8, borderRadius: '50%', background: dotColor, flexShrink: 0, display: 'inline-block', boxShadow: isCrit ? `0 0 6px ${dotColor}` : 'none' }} />
        <span style={{ fontFamily: 'var(--mono)', fontSize: 8.5, color: 'rgba(140,160,210,0.5)', flexShrink: 0, width: 56 }}>{ts}</span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 8.5, fontWeight: 700, color: teamColor(event.team), flexShrink: 0, minWidth: 30 }}>
          {event.team?.toUpperCase()}
        </span>
        {roleStr && (
          <span style={{ fontFamily: 'var(--mono)', fontSize: 7.5, color: 'rgba(160,180,220,0.45)', letterSpacing: '0.08em', flexShrink: 0 }}>
            [{roleStr}]
          </span>
        )}
        <span style={{ fontFamily: 'var(--mono)', fontSize: 8.5, marginRight: 2 }}>{getIcon(event.action_type)}</span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 9, fontWeight: isSpawn ? 700 : 600, color: isSpawn ? '#ffd740' : isCrit ? '#ffd740' : 'var(--text-dim)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {event.detail || getLabel(event.action_type)}
        </span>
        {event.target_node != null && (
          <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: 'rgba(0,229,255,0.7)', flexShrink: 0 }}>n{event.target_node}</span>
        )}
        {event.step > 0 && (
          <span style={{ fontFamily: 'var(--mono)', fontSize: 7.5, color: 'rgba(140,160,210,0.35)', flexShrink: 0, marginLeft: 4 }}>s{event.step}</span>
        )}
        {event.reasoning && (
          <span style={{ fontSize: 8, color: 'rgba(140,160,210,0.4)', marginLeft: 2, flexShrink: 0 }}>{expanded ? '▲' : '▼'}</span>
        )}
      </div>

      {/* Context bar */}
      {(event.suspicion != null || event.zone) && (
        <div style={{ display: 'flex', gap: 8, marginTop: 2, paddingLeft: 18 }}>
          {event.zone && (
            <span style={{ fontFamily: 'var(--mono)', fontSize: 7.5, color: zoneColor(event.zone), letterSpacing: '0.08em' }}>{event.zone}</span>
          )}
          {event.suspicion != null && (
            <span style={{ fontFamily: 'var(--mono)', fontSize: 7.5, color: 'rgba(255,68,68,0.55)' }}>
              susp {Math.round(event.suspicion * 100)}%
            </span>
          )}
          {event.detection != null && (
            <span style={{ fontFamily: 'var(--mono)', fontSize: 7.5, color: 'rgba(68,136,255,0.55)' }}>
              det {Math.round(event.detection * 100)}%
            </span>
          )}
        </div>
      )}

      {/* Expanded reasoning */}
      {expanded && event.reasoning && (
        <div style={{
          marginTop: 5, paddingLeft: 18,
          fontFamily: 'var(--mono)', fontSize: 9.5, color: 'var(--text-dim)',
          lineHeight: 1.55, borderLeft: '2px solid rgba(140,160,210,0.15)',
          paddingRight: 6, maxHeight: 100, overflowY: 'auto',
          background: 'rgba(0,0,0,0.2)', borderRadius: 4, padding: '5px 8px',
        }}>
          {event.reasoning}
        </div>
      )}
    </div>
  );
}

function zoneColor(zone) {
  if (!zone) return 'rgba(140,160,210,0.4)';
  if (zone.includes('Critical') || zone.includes('HVT')) return '#ff6b6b';
  if (zone.includes('Sensitive')) return '#ffd740';
  if (zone.includes('General')) return '#69f0ae';
  return '#00e5ff';
}

function filterEvents(events, filter) {
  if (filter === 'ALL')      return events;
  if (filter === 'RED')      return events.filter(e => e.team === 'red');
  if (filter === 'BLUE')     return events.filter(e => e.team === 'blue');
  if (filter === 'SPAWNS')   return events.filter(e => e.action_type === 'spawn_subagent');
  if (filter === 'TRAPS')    return events.filter(e => e.type === 'trap' || ['place_honeypot','plant_breadcrumb','plant_false_trail','plant_temporal_decoy','trigger_false_escalation'].includes(e.action_type));
  if (filter === 'DROPS')    return events.filter(e => e.type === 'dead_drop' || ['write_dead_drop','read_dead_drop','tamper_dead_drop','write_corrupted_drop'].includes(e.action_type));
  if (filter === 'CRITICAL') return events.filter(e => e.critical || e.type === 'exfil' || e.type === 'system');
  return events;
}

export default function BattleFeed({ steps }) {
  const [filter, setFilter] = useState('ALL');
  const scrollRef = useRef(null);
  const prevLen   = useRef(0);

  const battleEvents = useBattleLog(2000);

  // Also synthesize events from live steps when no battle-log data
  const stepEvents = (() => {
    if (!steps || steps.length === 0) return [];
    const evts = [];
    for (const s of steps) {
      const ts = s.timestamp || '';
      if (s.red_action && s.red_action !== 'waiting') {
        const parts = s.red_action.split(' → ');
        const atype = parts[0];
        const node  = parts[1] ? parseInt(parts[1].replace('n', '')) : null;
        evts.push({
          id: `step-r-${s.step}`, type: 'action', step: s.step,
          episode: s.episode, timestamp: ts, team: 'red',
          agent_id: 'red_planner_01', role: 'planner',
          action_type: atype, target_node: node,
          detail: s.red_action, critical: atype.includes('exfil'),
          suspicion: s.suspicion, detection: s.detection, zone: s.zone, reasoning: '',
        });
      }
      for (const agent of (s.all_agents || [])) {
        if (agent.team === 'blue') {
          evts.push({
            id: `step-b-${s.step}-${agent.agent_id}`, type: 'action', step: s.step,
            episode: s.episode, timestamp: ts, team: 'blue',
            agent_id: agent.agent_id,
            role: agent.agent_id.replace('blue_', '').replace('_01', '').replace('_', ' '),
            action_type: agent.action_type,
            target_node: agent.target_node,
            detail: `${agent.action_type?.replace(/_/g,' ')}${agent.target_node != null ? ` → n${agent.target_node}` : ''}`,
            critical: agent.action_type === 'trigger_alert',
            suspicion: s.suspicion, detection: s.detection, zone: s.zone,
            reasoning: agent.reasoning || '',
          });
        }
      }
      if (s.suspicion >= 0.85) {
        evts.push({
          id: `susp-${s.step}`, type: 'system', step: s.step,
          episode: s.episode, timestamp: ts, team: 'system',
          agent_id: 'system', role: 'system', action_type: 'key_event',
          detail: `⚠ CRITICAL SUSPICION: ${Math.round(s.suspicion * 100)}%`,
          critical: true, suspicion: s.suspicion, detection: s.detection, zone: s.zone, reasoning: '',
        });
      }
    }
    return evts;
  })();

  // Use battle-log events if available, fall back to step-derived
  const allEvents = battleEvents.length > 0 ? battleEvents : stepEvents;
  // Sort newest-first
  const sorted = [...allEvents].sort((a, b) => {
    if (b.episode !== a.episode) return b.episode - a.episode;
    return b.step - a.step;
  });

  const filtered = filterEvents(sorted, filter);

  useEffect(() => {
    if (filtered.length !== prevLen.current && scrollRef.current) {
      scrollRef.current.scrollTop = 0;
      prevLen.current = filtered.length;
    }
  }, [filtered.length]);

  const counts = {
    red:      allEvents.filter(e => e.team === 'red').length,
    blue:     allEvents.filter(e => e.team === 'blue').length,
    spawns:   allEvents.filter(e => e.action_type === 'spawn_subagent').length,
    traps:    allEvents.filter(e => e.type === 'trap').length,
    drops:    allEvents.filter(e => e.type === 'dead_drop').length,
    critical: allEvents.filter(e => e.critical).length,
  };

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ padding: '6px 10px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
        <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#ff4444', animation: 'livePulse 1.5s ease-in-out infinite' }} />
        <span style={{ fontFamily: 'var(--mono)', fontSize: 9, fontWeight: 700, letterSpacing: '0.14em', color: 'var(--text-dim)' }}>BATTLE LOG</span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 7.5, color: 'rgba(140,160,210,0.4)', marginLeft: 'auto' }}>{allEvents.length} events</span>
      </div>

      {/* Filter bar */}
      <div style={{ display: 'flex', gap: 2, padding: '4px 8px', borderBottom: '1px solid var(--border)', flexShrink: 0, flexWrap: 'wrap' }}>
        {FILTERS.map(f => {
          const cnt = f === 'RED' ? counts.red : f === 'BLUE' ? counts.blue : f === 'SPAWNS' ? counts.spawns : f === 'TRAPS' ? counts.traps : f === 'DROPS' ? counts.drops : f === 'CRITICAL' ? counts.critical : allEvents.length;
          const active = filter === f;
          const accent = f === 'RED' ? '#ff4444' : f === 'BLUE' ? '#4488ff' : f === 'SPAWNS' ? '#ffd740' : f === 'CRITICAL' ? '#ffd740' : f === 'TRAPS' ? '#ff8844' : f === 'DROPS' ? '#a78bfa' : 'rgba(140,160,210,0.6)';
          return (
            <button
              key={f}
              onClick={() => setFilter(f)}
              style={{
                fontFamily: 'var(--mono)', fontSize: 7.5, fontWeight: 700, letterSpacing: '0.10em',
                border: `1px solid ${active ? accent : 'rgba(140,160,210,0.18)'}`,
                background: active ? `${accent}18` : 'transparent',
                color: active ? accent : 'rgba(140,160,210,0.5)',
                borderRadius: 4, padding: '2px 6px', cursor: 'pointer',
                transition: 'all 0.15s',
              }}
            >
              {f} {cnt > 0 && <span style={{ opacity: 0.7 }}>{cnt}</span>}
            </button>
          );
        })}
      </div>

      {/* Event list */}
      <div
        ref={scrollRef}
        style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}
      >
        {filtered.length === 0 && (
          <div style={{ padding: '14px 10px', fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-mute)', fontStyle: 'italic' }}>
            Awaiting events…
          </div>
        )}
        {filtered.map((event, i) => (
          <EventRow key={event.id} event={event} idx={i} />
        ))}
      </div>
    </div>
  );
}
