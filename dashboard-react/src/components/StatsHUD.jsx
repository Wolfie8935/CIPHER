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

function extractLiveAgentNames(steps, team) {
  if (!steps?.length) return [];
  const latest = steps[steps.length - 1];
  const agents = Array.isArray(latest?.all_agents) ? latest.all_agents : [];
  const commanderPrefix = `${team}_commander`;
  const names = [];
  for (const a of agents) {
    const id = String(a?.agent_id ?? '').trim().toLowerCase();
    if (!id.startsWith(`${team}_`) || id.startsWith(commanderPrefix)) continue;
    const role = id.replace(/^(red|blue)_/, '').replace(/_\d+$/, '').replace(/_/g, ' ').toUpperCase().slice(0, 6);
    if (role && !names.includes(role)) names.push(role);
  }
  return names.slice(0, 6);
}

export default function StatsHUD({ latest, winnerCard, completionSignals, commanderLifecycle, steps, rightOffset = 14 }) {
  const suspicion = latest?.suspicion ?? 0;
  const detection = latest?.detection ?? 0;
  const zone      = latest?.zone      ?? 'Perimeter';
  const step      = latest?.step      ?? 0;
  const maxSteps  = latest?.max_steps ?? 30;
  const exfil     = latest?.exfil_count ?? 0;
  const zoneColor = ZONE_COLORS[zone] ?? '#0091a8';

  const hasCompletionEvidence = Boolean(
    completionSignals?.replayComplete || completionSignals?.liveTerminalEvidence
  );
  const showWinnerCard = winnerCard?.status === 'FINAL' && hasCompletionEvidence;
  const winnerState = winnerCard?.winner ?? 'PENDING';
  const winnerText = winnerState === 'RED'
    ? 'WINNER RED'
    : winnerState === 'BLUE'
      ? 'WINNER BLUE'
      : winnerState === 'DRAW'
        ? 'DRAW'
        : '';
  const winnerStyle = winnerState === 'RED'
    ? { color: '#ff6b6b', bg: 'rgba(255,68,68,0.14)', border: 'rgba(255,68,68,0.35)' }
    : winnerState === 'BLUE'
      ? { color: '#7eb3ff', bg: 'rgba(68,136,255,0.14)', border: 'rgba(68,136,255,0.35)' }
      : winnerState === 'DRAW'
        ? { color: '#d6e1f3', bg: 'rgba(190,205,235,0.12)', border: 'rgba(190,205,235,0.30)' }
        : { color: 'var(--text-mute)', bg: 'rgba(140,160,210,0.08)', border: 'rgba(140,160,210,0.20)' };
  const hasCommanderTelemetry = Boolean(
    commanderLifecycle?.red?.hasData || commanderLifecycle?.blue?.hasData
  );

  return (
    <div className="stats-hud" style={{ right: rightOffset, transition: 'right 0.35s cubic-bezier(0.4,0,0.2,1)' }}>

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

      {/* Commander lifecycle (anchored below zone card) */}
      <div className="hud-card" style={{ paddingTop: 8 }}>
        <div style={{ fontSize: 8, letterSpacing: '0.14em', color: 'rgba(170,190,225,0.62)', marginBottom: 7 }}>
          COMMANDER AGENT LIFECYCLE
        </div>
        {!hasCommanderTelemetry && (
          <div style={{ fontSize: 9, color: 'rgba(170,190,225,0.58)', fontStyle: 'italic', padding: '2px 0 6px' }}>
            Commander telemetry unavailable for this trace.
          </div>
        )}
        {[
          { key: 'red', label: 'RED COMMANDER', color: '#ff6f6f', stats: commanderLifecycle?.red ?? { spawned: 0, live: 0, expired: 0 } },
          { key: 'blue', label: 'BLUE COMMANDER', color: '#79adff', stats: commanderLifecycle?.blue ?? { spawned: 0, live: 0, expired: 0 } },
        ].map((row) => {
          const agentNames = extractLiveAgentNames(steps, row.key);
          return (
            <div
              key={row.key}
              style={{
                border: `1px solid ${row.color}44`,
                borderRadius: 7,
                padding: '6px 7px',
                marginBottom: row.key === 'red' ? 6 : 0,
                background: `${row.color}12`,
              }}
            >
              <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: '0.10em', color: row.color, marginBottom: 5 }}>
                {row.label}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 4, marginBottom: agentNames.length ? 6 : 0 }}>
                <div style={{ fontSize: 8.5, color: 'rgba(226,236,255,0.84)' }}>
                  <span style={{ color: 'rgba(150,170,205,0.66)' }}>spawned</span><br />
                  <span style={{ fontSize: 11, fontWeight: 700 }}>{row.stats.spawned}</span>
                </div>
                <div style={{ fontSize: 8.5, color: 'rgba(226,236,255,0.84)' }}>
                  <span style={{ color: 'rgba(150,170,205,0.66)' }}>live</span><br />
                  <span style={{ fontSize: 11, fontWeight: 700, color: '#86f4bd' }}>{row.stats.live}</span>
                </div>
                <div style={{ fontSize: 8.5, color: 'rgba(226,236,255,0.84)' }}>
                  <span style={{ color: 'rgba(150,170,205,0.66)' }}>expired</span><br />
                  <span style={{ fontSize: 11, fontWeight: 700, color: '#ffb38f' }}>{row.stats.expired}</span>
                </div>
              </div>
              {agentNames.length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                  {agentNames.map(name => (
                    <span key={name} style={{
                      fontSize: 7.5, fontWeight: 700, letterSpacing: '0.06em',
                      padding: '1px 5px', borderRadius: 3,
                      background: `${row.color}20`, border: `1px solid ${row.color}50`,
                      color: row.color, animation: 'agentPulse 2s ease-in-out infinite',
                    }}>{name}</span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Winner card (only after episode completes) */}
      {showWinnerCard && (
        <div
          className="hud-card"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: winnerStyle.bg,
            border: `1px solid ${winnerStyle.border}`,
          }}
        >
          <div
            style={{
              fontFamily: 'var(--mono)',
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              color: winnerStyle.color,
              textAlign: 'center',
            }}
          >
            {winnerText}
          </div>
        </div>
      )}
    </div>
  );
}
