import { useEffect, useRef, useState } from 'react';

const ACTION_ICONS = {
  move:                '→',
  read_file:           '📄',
  write_dead_drop:     '📦',
  plant_false_trail:   '👻',
  plant_temporal_decoy:'⏰',
  plant_honeypot_poison:'☠',
  exfil_file:          '💀',
  scan_network:        '📡',
  investigate_node:    '🔍',
  place_honeypot:      '🪤',
  plant_breadcrumb:    '🍞',
  analyze_anomaly:     '📊',
  trigger_alert:       '🚨',
  reconstruct_path:    '🗺',
  tamper_dead_drop:    '✂',
  abort:               '⛔',
};

const AGENT_SHORT = {
  red_planner_01:             'PLANNER',
  red_analyst_01:             'ANALYST',
  red_operative_01:           'OPERATIVE',
  red_exfiltrator_01:         'EXFILTRATOR',
  blue_surveillance_01:       'SURVEILLANCE',
  blue_threat_hunter_01:      'THREAT HUNTER',
  blue_deception_architect_01:'DECEPTION',
  blue_forensics_01:          'FORENSICS',
};

function TypewriterText({ text, speed = 18 }) {
  const [shown, setShown] = useState('');
  const idxRef  = useRef(0);
  const timerRef = useRef(null);

  useEffect(() => {
    idxRef.current = 0;
    setShown('');
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      idxRef.current++;
      setShown(text.slice(0, idxRef.current));
      if (idxRef.current >= text.length) clearInterval(timerRef.current);
    }, speed);
    return () => clearInterval(timerRef.current);
  }, [text, speed]);

  return <span>{shown}{shown.length < text.length && <span style={{ opacity: 0.6, animation: 'dot-pulse 0.8s step-end infinite' }}>▋</span>}</span>;
}

function ThoughtBubble({ thought, isNewest, team }) {
  const agentShort = AGENT_SHORT[thought.agent_id] || thought.agent_id;
  const icon       = ACTION_ICONS[thought.action_type] || '•';
  const teamColor  = team === 'red' ? 'var(--red-team)' : 'var(--blue-team)';

  return (
    <div className={`thought-bubble ${team}${isNewest ? ' newest' : ''}`}>
      <div className="thought-agent">
        <span style={{ color: teamColor }}>{agentShort}</span>
        {thought.target_node != null && (
          <span style={{ color:'var(--text-muted)', fontWeight:400, marginLeft:6 }}>→ n{thought.target_node}</span>
        )}
        <span style={{ color:'var(--text-muted)', fontWeight:400, float:'right' }}>
          s{thought.step}
        </span>
      </div>
      <div className="thought-text">
        {isNewest
          ? <TypewriterText text={thought.reasoning} speed={16} />
          : <span>{thought.reasoning}</span>
        }
      </div>
      <div style={{ marginTop:5 }}>
        <span className={`action-badge ${team}`}>
          {icon} {thought.action_type?.replace(/_/g, ' ')}
        </span>
      </div>
    </div>
  );
}

export default function AgentMindPanel({ team, thoughts }) {
  const scrollRef = useRef(null);
  const prevLenRef = useRef(0);

  useEffect(() => {
    if (thoughts.length !== prevLenRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      prevLenRef.current = thoughts.length;
    }
  }, [thoughts]);

  const label       = team === 'red' ? '🔴 RED TEAM MIND' : '🔵 BLUE TEAM MIND';
  const dotCls      = team === 'red' ? 'red' : 'blue';
  const cardCls     = team === 'red' ? 'card card-red' : 'card card-blue';
  const shown       = thoughts.slice(-5);

  return (
    <div className={cardCls} style={{ flex: 1, minHeight: 0, display:'flex', flexDirection:'column' }}>
      <div className="section-label">
        <div className={`label-dot ${dotCls}`} />
        <span>{label}</span>
        <span style={{ marginLeft:'auto', fontFamily:'var(--font-mono)', fontSize:9, color:'var(--text-muted)' }}>
          {thoughts.length} thoughts
        </span>
      </div>
      <div className="thought-scroll" ref={scrollRef} style={{ flex:1, minHeight:0, overflowY:'auto' }}>
        {shown.length === 0 && (
          <div style={{ padding:'12px', fontFamily:'var(--font-mono)', fontSize:11, color:'var(--text-muted)', fontStyle:'italic' }}>
            Awaiting agent reasoning…
          </div>
        )}
        {shown.map((t, i) => (
          <ThoughtBubble
            key={`${t.agent_id}-${t.step}-${t.timestamp}`}
            thought={t}
            isNewest={i === shown.length - 1}
            team={team}
          />
        ))}
      </div>
    </div>
  );
}
