import { useMemo } from 'react';
import { useRLStats } from '../hooks/useRLStats';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

const verdictLabelShort = (key) => {
  const m = {
    red_dominates: 'RED↑',
    blue_dominates: 'BLUE↑',
    contested: 'CONT',
    degenerate: 'DEG',
    none: 'PEND',
  };
  return m[key] ?? key?.slice(0, 6) ?? '?';
};

const chartTooltip = {
  contentStyle: { background: '#111827', border: '1px solid rgba(140,160,210,0.3)', fontFamily: 'var(--mono)', fontSize: 10 },
  labelStyle: { color: '#c8d4e8' },
};

const VERDICTS = {
  red_dominates:  { color: '#ff4444', label: 'RED DOMINATES',  icon: '🔴', bg: 'rgba(255,68,68,0.10)' },
  blue_dominates: { color: '#4488ff', label: 'BLUE DOMINATES', icon: '🔵', bg: 'rgba(68,136,255,0.10)' },
  contested:      { color: '#ffd740', label: 'CONTESTED',      icon: '⚡', bg: 'rgba(255,215,64,0.08)' },
  degenerate:     { color: '#555',    label: 'DEGENERATE',     icon: '⚪', bg: 'rgba(100,100,100,0.08)' },
  none:           { color: '#555',    label: 'PENDING',        icon: '…',  bg: 'rgba(100,100,100,0.05)' },
};

const FLAG_COLORS = {
  reward_hacking: '#ff4444',
  collusion: '#ffd740',
  none: '#555',
};

function VerdictBadge({ verdict }) {
  const v = VERDICTS[verdict] ?? VERDICTS.none;
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: '3px 10px',
      background: v.bg,
      border: `1px solid ${v.color}45`,
      borderRadius: 6,
      fontFamily: 'var(--mono)', fontSize: 9, fontWeight: 700,
      letterSpacing: '0.12em', color: v.color,
    }}>
      <span>{v.icon}</span> {v.label}
    </div>
  );
}

const STACK_KEYS = ['red_dominates', 'blue_dominates', 'contested', 'degenerate', 'none'];

export default function OversightPanel({ steps }) {
  const stats = useRLStats(5000);

  const {
    recentEp,
    verdict,
    flags,
    judgment,
    verdictCounts,
    total,
    episodeScoreSeries,
    verdictPieData,
    verdictBarData,
    verdictCumulativeStack,
  } = useMemo(() => {
    const table = stats?.episode_table ?? [];
    const recent = table.length > 0 ? table[table.length - 1] : null;

    const liveVerdict = (() => {
      if (!steps || steps.length === 0) return 'contested';
      const latest = steps[steps.length - 1];
      const exfil = latest.exfil_count ?? 0;
      const det = latest.detection ?? 0;
      const susp = latest.suspicion ?? 0;
      if (exfil > 0) return 'red_dominates';
      if (det > 0.8) return 'blue_dominates';
      if (susp > 0.65) return 'contested';
      return 'contested';
    })();

    const verdictCounts = {};
    for (const ep of table) {
      const v = ep.verdict ?? 'none';
      verdictCounts[v] = (verdictCounts[v] ?? 0) + 1;
    }
    const sum = Object.values(verdictCounts).reduce((a, b) => a + b, 0);
    const total_ = Math.max(1, sum);

    const sorted = [...table].sort((a, b) => Number(a.episode) - Number(b.episode));
    const episodeScoreSeries_ = sorted.map((ep) => ({
      episode: Number(ep.episode) || 0,
      red: ep.red_total ?? 0,
      blue: ep.blue_total ?? 0,
    }));

    const verdictPieData_ = Object.entries(verdictCounts)
      .filter(([, cnt]) => cnt > 0)
      .map(([key, value]) => {
        const cfg = VERDICTS[key] ?? { ...VERDICTS.none, label: String(key).toUpperCase() };
        return { name: cfg.label, value, color: cfg.color, key };
      });

    const verdictBarData_ = verdictPieData_.map((d) => ({
      ...d,
      short: verdictLabelShort(d.key),
    }));

    const running = { red_dominates: 0, blue_dominates: 0, contested: 0, degenerate: 0, none: 0 };
    const verdictCumulativeStack_ = sorted.map((ep) => {
      const raw = ep.verdict ?? 'none';
      const k = STACK_KEYS.includes(raw) ? raw : 'none';
      running[k] += 1;
      return {
        episode: Number(ep.episode) || 0,
        red_dominates: running.red_dominates,
        blue_dominates: running.blue_dominates,
        contested: running.contested,
        degenerate: running.degenerate,
        none: running.none,
      };
    });

    return {
      recentEp: recent,
      verdict: recent?.verdict ?? liveVerdict,
      flags: recent?.flags ? recent.flags.split('|').filter((f) => f && f !== 'none') : [],
      judgment: recent?.judgment ?? '',
      verdictCounts,
      total: total_,
      episodeScoreSeries: episodeScoreSeries_,
      verdictPieData: verdictPieData_,
      verdictBarData: verdictBarData_,
      verdictCumulativeStack: verdictCumulativeStack_,
    };
  }, [stats?.episode_table, steps]);

  return (
    <div style={{ flex: 1, overflowY: 'auto', minHeight: 0, padding: '6px 0' }}>

      {/* Current verdict */}
      <div style={{ padding: '6px 10px 8px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.14em', color: 'var(--text-mute)', marginBottom: 6 }}>LATEST VERDICT</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
          <VerdictBadge verdict={verdict} />
          {flags.map(f => (
            <div key={f} style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              padding: '2px 7px',
              background: `${FLAG_COLORS[f] ?? '#555'}18`,
              border: `1px solid ${FLAG_COLORS[f] ?? '#555'}45`,
              borderRadius: 4,
              fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700,
              color: FLAG_COLORS[f] ?? '#888', letterSpacing: '0.1em',
            }}>
              ⚑ {f.toUpperCase().replace(/_/g, ' ')}
            </div>
          ))}
        </div>
        {judgment && (
          <div style={{
            marginTop: 8, padding: '6px 8px',
            background: 'rgba(0,0,0,0.2)',
            border: '1px solid rgba(255,215,64,0.15)',
            borderLeft: '2px solid rgba(255,215,64,0.35)',
            borderRadius: 4,
            fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text-dim)', lineHeight: 1.55,
            fontStyle: 'italic',
          }}>
            "{judgment}"
          </div>
        )}
      </div>

      {/* RED / BLUE totals by episode */}
      {episodeScoreSeries.length > 0 && (
        <div style={{ padding: '6px 10px 8px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--text-mute)', marginBottom: 6 }}>
            TOTALS BY EPISODE
          </div>
          <div style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={episodeScoreSeries} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
                <CartesianGrid stroke="rgba(140,160,210,0.14)" strokeDasharray="3 3" />
                <XAxis dataKey="episode" tick={{ fill: 'rgba(170,190,225,0.65)', fontSize: 9 }} />
                <YAxis tick={{ fill: 'rgba(170,190,225,0.65)', fontSize: 9 }} />
                <Tooltip {...chartTooltip} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line type="monotone" dataKey="red" name="RED total" stroke="#ff4444" strokeWidth={2} dot={{ r: episodeScoreSeries.length <= 4 ? 3 : 0 }} activeDot={{ r: 4 }} />
                <Line type="monotone" dataKey="blue" name="BLUE total" stroke="#4488ff" strokeWidth={2} dot={{ r: episodeScoreSeries.length <= 4 ? 3 : 0 }} activeDot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Verdict distribution — pie + bars + strip chart */}
      {Object.keys(verdictCounts).length > 0 && (
        <div style={{ padding: '6px 10px 8px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--text-mute)', marginBottom: 8 }}>VERDICT DISTRIBUTION</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'stretch' }}>
            <div style={{ width: 180, height: 200, flexShrink: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={verdictPieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={52}
                    outerRadius={78}
                    paddingAngle={2}
                    stroke="rgba(0,0,0,0.35)"
                    strokeWidth={1}
                  >
                    {verdictPieData.map((e) => (
                      <Cell key={e.key} fill={e.color} />
                    ))}
                  </Pie>
                  <Tooltip {...chartTooltip} />
                  <Legend wrapperStyle={{ fontFamily: 'var(--mono)', fontSize: 8 }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div style={{ flex: '1 1 160px', minWidth: 140, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 0 }}>
              {[...Object.keys(VERDICTS), ...Object.keys(verdictCounts).filter((k) => !VERDICTS[k])].map((key) => {
                const cnt = verdictCounts[key] ?? 0;
                if (cnt === 0) return null;
                const cfg = VERDICTS[key] ?? { color: '#8899aa', label: String(key).toUpperCase(), icon: '◆' };
                const pct = Math.round((cnt / total) * 100);
                return (
                  <div key={key} style={{ marginBottom: 5 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                      <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: cfg.color }}>{cfg.icon} {cfg.label}</span>
                      <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: 'var(--text-dim)' }}>{cnt} ({pct}%)</span>
                    </div>
                    <div style={{ height: 4, background: 'rgba(140,160,210,0.08)', borderRadius: 2 }}>
                      <div style={{ height: '100%', width: `${pct}%`, background: cfg.color, borderRadius: 2, opacity: 0.7 }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          {verdictBarData.length > 0 && (
            <div style={{ height: 180, marginTop: 8 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={verdictBarData} margin={{ top: 4, right: 8, left: -8, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(140,160,210,0.14)" vertical={false} />
                  <XAxis dataKey="short" tick={{ fill: 'rgba(170,190,225,0.65)', fontSize: 9, fontFamily: 'var(--mono)' }} />
                  <YAxis tick={{ fill: 'rgba(170,190,225,0.65)', fontSize: 9 }} allowDecimals={false} />
                  <Tooltip {...chartTooltip} />
                  <Bar dataKey="value" radius={[3, 3, 0, 0]} name="Count">
                    {verdictBarData.map((e) => (
                      <Cell key={e.key} fill={e.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Cumulative verdict counts (stacked) */}
      {verdictCumulativeStack.length >= 2 && (
        <div style={{ padding: '6px 10px 8px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--text-mute)', marginBottom: 6 }}>
            CUMULATIVE VERDICT COUNTS
          </div>
          <div style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={verdictCumulativeStack} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
                <CartesianGrid stroke="rgba(140,160,210,0.14)" strokeDasharray="3 3" />
                <XAxis dataKey="episode" tick={{ fill: 'rgba(170,190,225,0.65)', fontSize: 9 }} />
                <YAxis tick={{ fill: 'rgba(170,190,225,0.65)', fontSize: 9 }} allowDecimals={false} />
                <Tooltip {...chartTooltip} />
                <Legend wrapperStyle={{ fontSize: 9 }} />
                <Area type="stepAfter" dataKey="red_dominates" name="RED dom" stackId="cv" stroke={VERDICTS.red_dominates.color} fill={VERDICTS.red_dominates.color} fillOpacity={0.35} />
                <Area type="stepAfter" dataKey="blue_dominates" name="BLUE dom" stackId="cv" stroke={VERDICTS.blue_dominates.color} fill={VERDICTS.blue_dominates.color} fillOpacity={0.35} />
                <Area type="stepAfter" dataKey="contested" name="Contested" stackId="cv" stroke={VERDICTS.contested.color} fill={VERDICTS.contested.color} fillOpacity={0.38} />
                <Area type="stepAfter" dataKey="degenerate" name="Degenerate" stackId="cv" stroke="#666" fill="#555" fillOpacity={0.4} />
                <Area type="stepAfter" dataKey="none" name="Pending" stackId="cv" stroke="#777" fill="#555" fillOpacity={0.25} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Episode verdict history */}
      {(stats?.episode_table ?? []).length > 0 && (
        <div style={{ padding: '6px 10px 8px' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--text-mute)', marginBottom: 6 }}>EPISODE VERDICTS</div>
          {[...stats.episode_table].reverse().slice(0, 12).map((ep, i) => {
            const vcfg = VERDICTS[ep.verdict] ?? VERDICTS.none;
            const epFlags = ep.flags ? ep.flags.split('|').filter(f => f && f !== 'none') : [];
            return (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '3px 0', borderBottom: '1px solid rgba(140,160,210,0.04)',
              }}>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: 'rgba(140,160,210,0.4)', width: 24 }}>#{ep.episode}</span>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: vcfg.color }}>{vcfg.icon}</span>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 7.5, color: 'var(--text-dim)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {ep.terminal?.replace(/_/g, ' ')}
                </span>
                {epFlags.map(f => (
                  <span key={f} style={{ fontFamily: 'var(--mono)', fontSize: 6.5, color: FLAG_COLORS[f] ?? '#888', background: `${FLAG_COLORS[f] ?? '#555'}15`, padding: '1px 3px', borderRadius: 2 }}>
                    {f.slice(0, 6)}
                  </span>
                ))}
                <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: ep.red_total >= 0 ? '#ff8888' : '#ff4444' }}>{ep.red_total >= 0 ? '+' : ''}{ep.red_total?.toFixed(2)}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
