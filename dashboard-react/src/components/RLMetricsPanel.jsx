import { useRLStats } from '../hooks/useRLStats';

function formatSigned(value) {
  return `${value >= 0 ? '+' : ''}${value.toFixed(3)}`;
}

function WinRatePie({ red, blue }) {
  const draw = Math.max(0, 1 - red - blue);
  const r = 28, cx = 32, cy = 32, circumference = 2 * Math.PI * r;
  const redArc   = circumference * red;
  const blueArc  = circumference * blue;
  const drawArc  = circumference * draw;
  const redOff   = 0;
  const blueOff  = -(redArc);
  const drawOff  = -(redArc + blueArc);
  return (
    <svg width={64} height={64} style={{ flexShrink: 0 }}>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(140,160,210,0.08)" strokeWidth={10} />
      {red > 0.01 && <circle cx={cx} cy={cy} r={r} fill="none" stroke="#ff4444" strokeWidth={10} strokeDasharray={`${redArc} ${circumference}`} strokeDashoffset={redOff} strokeLinecap="butt" transform={`rotate(-90 ${cx} ${cy})`} />}
      {blue > 0.01 && <circle cx={cx} cy={cy} r={r} fill="none" stroke="#4488ff" strokeWidth={10} strokeDasharray={`${blueArc} ${circumference}`} strokeDashoffset={blueOff} strokeLinecap="butt" transform={`rotate(-90 ${cx} ${cy})`} />}
      {draw > 0.01 && <circle cx={cx} cy={cy} r={r} fill="none" stroke="#555" strokeWidth={10} strokeDasharray={`${drawArc} ${circumference}`} strokeDashoffset={drawOff} strokeLinecap="butt" transform={`rotate(-90 ${cx} ${cy})`} />}
    </svg>
  );
}

function SparkLine({ data, color, height = 28 }) {
  if (!data || data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const w = 140, h = height;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * (h - 4) - 2;
    return `${x},${y}`;
  }).join(' ');
  return (
    <svg width={w} height={h} style={{ overflow: 'visible' }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" opacity={0.8} />
      <circle cx={parseFloat(pts.split(' ').pop().split(',')[0])} cy={parseFloat(pts.split(' ').pop().split(',')[1])} r={2.5} fill={color} />
    </svg>
  );
}

const COMPONENT_LABELS = {
  red_exfil:           { label: 'Exfil',        color: '#ff4444' },
  red_stealth:         { label: 'Stealth',       color: '#ff8888' },
  red_complexity:      { label: 'Complexity',    color: '#ffaa44' },
  red_memory:          { label: 'Memory',        color: '#ffcc88' },
  blue_detection:      { label: 'Detection',     color: '#4488ff' },
  blue_speed:          { label: 'Speed',         color: '#88aaff' },
  blue_fp_penalty:     { label: 'FP Penalty',    color: '#ff6644' },
  blue_honeypot_rate:  { label: 'Honeypots',     color: '#44ccff' },
  blue_graph_reconstruction: { label: 'Graph Recon', color: '#66ddff' },
};

const TERMINAL_LABELS = {
  exfiltration_complete: { label: 'RED Win',   color: '#ff4444' },
  exfil_success:         { label: 'RED Win',   color: '#ff4444' },
  exfil_complete:        { label: 'RED Win',   color: '#ff4444' },
  detected:              { label: 'BLUE Win',  color: '#4488ff' },
  aborted:               { label: 'Abort',     color: '#888' },
  max_steps:             { label: 'Max Steps', color: '#555' },
};

export default function RLMetricsPanel() {
  const stats = useRLStats(4000);

  if (!stats) {
    return (
      <div style={{ padding: 16, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-mute)', fontStyle: 'italic' }}>
        Loading RL metrics…
      </div>
    );
  }

  const { red_totals = [], blue_totals = [], episodes = [], win_rate_red = 0, win_rate_blue = 0,
          win_rate_red_10 = 0, win_rate_blue_10 = 0, total_episodes = 0,
          red_avg = 0, blue_avg = 0, best_red = 0, best_blue = 0,
          terminal_counts = {}, component_avgs = {}, episode_table = [], evo_events = [] } = stats;

  const componentRows = Object.entries(COMPONENT_LABELS).map(([key, cfg]) => ({
    key,
    ...cfg,
    value: component_avgs[key] ?? 0,
  }));
  const componentAbsMax = Math.max(0.25, ...componentRows.map((row) => Math.abs(row.value)));

  return (
    <div style={{ flex: 1, overflowY: 'auto', minHeight: 0, padding: '6px 0' }}>

      {/* Summary row */}
      <div style={{ padding: '6px 10px 8px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.14em', color: 'var(--text-mute)', marginBottom: 6 }}>TRAINING SUMMARY</div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <WinRatePie red={win_rate_red} blue={win_rate_blue} />
          <div style={{ flex: 1 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3px 10px' }}>
              {[
                { label: 'Total Episodes', val: total_episodes, color: 'var(--text-dim)' },
                { label: 'RED Win Rate', val: `${Math.round(win_rate_red * 100)}%`, color: '#ff4444' },
                { label: 'RED Win (last 10)', val: `${Math.round(win_rate_red_10 * 100)}%`, color: '#ff8888' },
                { label: 'BLUE Win Rate', val: `${Math.round(win_rate_blue * 100)}%`, color: '#4488ff' },
                { label: 'Avg RED Reward', val: red_avg.toFixed(3), color: '#ff6b87' },
                { label: 'Avg BLUE Reward', val: blue_avg.toFixed(3), color: '#7eb3ff' },
                { label: 'Best RED', val: best_red.toFixed(3), color: '#ff4444' },
                { label: 'Best BLUE', val: best_blue.toFixed(3), color: '#4488ff' },
              ].map(({ label, val, color }) => (
                <div key={label}>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 7, color: 'rgba(140,160,210,0.4)', letterSpacing: '0.08em' }}>{label}</div>
                  <div style={{ fontFamily: 'var(--mono)', fontSize: 10, fontWeight: 700, color }}>{val}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Reward curves sparklines */}
      {red_totals.length > 1 && (
        <div style={{ padding: '6px 10px 8px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--text-mute)', marginBottom: 6 }}>REWARD CURVES ({red_totals.length} eps)</div>
          <div style={{ marginBottom: 6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: '#ff6b87' }}>RED</span>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: '#ff4444' }}>{(red_totals[red_totals.length - 1] ?? 0).toFixed(3)}</span>
            </div>
            <SparkLine data={red_totals} color="#ff4444" />
          </div>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 1 }}>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: '#7eb3ff' }}>BLUE</span>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: '#4488ff' }}>{(blue_totals[blue_totals.length - 1] ?? 0).toFixed(3)}</span>
            </div>
            <SparkLine data={blue_totals} color="#4488ff" />
          </div>
        </div>
      )}

      {/* Outcome breakdown */}
      {Object.keys(terminal_counts).length > 0 && (
        <div style={{ padding: '6px 10px 8px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--text-mute)', marginBottom: 6 }}>TERMINAL OUTCOMES</div>
          {Object.entries(terminal_counts).map(([term, cnt]) => {
            const cfg = TERMINAL_LABELS[term] ?? { label: term, color: '#555' };
            const pct = Math.round((cnt / Math.max(1, total_episodes)) * 100);
            return (
              <div key={term} style={{ marginBottom: 4 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: cfg.color }}>{cfg.label}</span>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: 'var(--text-dim)' }}>{cnt} ({pct}%)</span>
                </div>
                <div style={{ height: 4, background: 'rgba(140,160,210,0.08)', borderRadius: 2 }}>
                  <div style={{ height: '100%', width: `${pct}%`, background: cfg.color, borderRadius: 2, opacity: 0.7 }} />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Component averages */}
      {Object.keys(component_avgs).length > 0 && (
        <div style={{ padding: '8px 10px 10px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--text-mute)', marginBottom: 8 }}>REWARD COMPONENTS (avg)</div>
          <div className="rl-reward-card">
            <div className="rl-reward-totals">
              <div className="rl-total-chip red">
                <span>RED TOTAL</span>
                <strong>{formatSigned(red_avg)}</strong>
              </div>
              <div className="rl-total-chip blue">
                <span>BLUE TOTAL</span>
                <strong>{formatSigned(blue_avg)}</strong>
              </div>
            </div>
            <div className="rl-reward-axis">
              <span>-{componentAbsMax.toFixed(2)}</span>
              <span>0</span>
              <span>{componentAbsMax.toFixed(2)}</span>
            </div>
            <div className="rl-reward-chart">
              {componentRows.map(({ key, label, color, value }) => {
                const pct = Math.min(100, (Math.abs(value) / componentAbsMax) * 100);
                return (
                  <div key={key} className="rl-reward-row">
                    <span className="rl-reward-label">{label}</span>
                    <div className="rl-reward-track-wrap">
                      <div className="rl-reward-track">
                        <div
                          className="rl-reward-fill"
                          style={{
                            width: `${pct}%`,
                            background: `linear-gradient(90deg, ${color}cc, ${color})`,
                            boxShadow: `0 0 10px ${color}33`,
                          }}
                        />
                      </div>
                    </div>
                    <span className="rl-reward-value" style={{ color }}>
                      {formatSigned(value)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Recent episodes table */}
      {episode_table.length > 0 && (
        <div style={{ padding: '6px 10px 8px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--text-mute)', marginBottom: 6 }}>RECENT EPISODES</div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--mono)', fontSize: 8.5 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(140,160,210,0.12)' }}>
                  {['EP', 'STEPS', 'TERMINAL', 'RED', 'BLUE', 'VERDICT'].map(h => (
                    <th key={h} style={{ padding: '2px 4px', textAlign: 'left', color: 'rgba(140,160,210,0.45)', fontWeight: 700, letterSpacing: '0.08em', fontSize: 7 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[...episode_table].reverse().map((row, i) => {
                  const term = TERMINAL_LABELS[row.terminal] ?? { color: '#555' };
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid rgba(140,160,210,0.04)' }}>
                      <td style={{ padding: '2px 4px', color: 'var(--text-dim)' }}>{row.episode}</td>
                      <td style={{ padding: '2px 4px', color: 'var(--text-dim)' }}>{row.steps}</td>
                      <td style={{ padding: '2px 4px', color: term.color, fontSize: 7.5 }}>{row.terminal?.replace(/_/g,' ').toUpperCase()}</td>
                      <td style={{ padding: '2px 4px', color: row.red_total >= 0 ? '#ff8888' : '#ff4444', fontWeight: 700 }}>{row.red_total >= 0 ? '+' : ''}{row.red_total?.toFixed(3)}</td>
                      <td style={{ padding: '2px 4px', color: row.blue_total >= 0 ? '#88aaff' : '#ff6644', fontWeight: 700 }}>{row.blue_total >= 0 ? '+' : ''}{row.blue_total?.toFixed(3)}</td>
                      <td style={{ padding: '2px 4px', color: row.verdict === 'red_dominates' ? '#ff4444' : row.verdict === 'blue_dominates' ? '#4488ff' : '#ffd740', fontSize: 7 }}>{row.verdict?.replace(/_/g,' ')}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Prompt evolution events */}
      {evo_events.length > 0 && (
        <div style={{ padding: '6px 10px 8px' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--text-mute)', marginBottom: 6 }}>PROMPT EVOLUTION</div>
          {evo_events.slice(-5).map((ev, i) => (
            <div key={i} style={{ marginBottom: 4, padding: '4px 6px', background: 'rgba(255,215,64,0.04)', borderLeft: '2px solid rgba(255,215,64,0.3)', borderRadius: 3 }}>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 7.5, color: '#ffd740' }}>
                EP {ev.episode ?? '?'} — {ev.rules_added ?? '?'} rules added
              </div>
              {ev.summary && (
                <div style={{ fontFamily: 'var(--mono)', fontSize: 7.5, color: 'var(--text-dim)', marginTop: 2 }}>
                  {String(ev.summary).slice(0, 80)}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
