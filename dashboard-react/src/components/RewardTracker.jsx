import { useRLStats } from '../hooks/useRLStats';
import {
  ResponsiveContainer, LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend,
  BarChart, Bar, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
} from 'recharts';

const chartTooltip = {
  contentStyle: { background: '#111827', border: '1px solid rgba(140,160,210,0.3)', fontFamily: 'var(--mono)', fontSize: 10 },
  labelStyle: { color: '#c8d4e8' },
};

const RED_COMPS = [
  { key: 'red_exfil',       label: 'Exfil',        color: '#ff4444' },
  { key: 'red_stealth',     label: 'Stealth',       color: '#ff7777' },
  { key: 'red_complexity',  label: 'Complexity ×',  color: '#ffaa44' },
  { key: 'red_memory',      label: 'Memory',        color: '#ffcc88' },
];
const BLUE_COMPS = [
  { key: 'blue_detection',     label: 'Detection',  color: '#4488ff' },
  { key: 'blue_speed',         label: 'Speed',      color: '#6699ff' },
  { key: 'blue_fp_penalty',    label: 'FP Penalty', color: '#ff6644', neg: true },
  { key: 'blue_honeypot_rate', label: 'Honeypots',  color: '#44ccff' },
  { key: 'blue_graph_reconstruction', label: 'Graph Recon', color: '#55ddff' },
];

export default function RewardTracker({ steps }) {
  const stats = useRLStats(4000);

  const recentEp = stats?.episode_table?.length > 0
    ? stats.episode_table[stats.episode_table.length - 1]
    : null;

  // Live estimate from steps when no historical data
  const liveRed  = (() => {
    if (!steps || steps.length === 0) return 0;
    const s = steps[steps.length - 1];
    return Math.min(2, (s.exfil_count ?? 0) * 0.8 + (1 - (s.suspicion ?? 0)) * 0.35);
  })();
  const liveBlue = (() => {
    if (!steps || steps.length === 0) return 0;
    const s = steps[steps.length - 1];
    return Math.min(1.5, (s.detection ?? 0) * 0.6);
  })();

  const redTotal  = recentEp ? (recentEp.red_total ?? liveRed) : liveRed;
  const blueTotal = recentEp ? (recentEp.blue_total ?? liveBlue) : liveBlue;
  const compAvg   = stats?.component_avgs ?? {};

  const maxR = Math.max(0.5, Math.abs(redTotal), Math.abs(blueTotal));

  const stepRewardSeries = (() => {
    if (!Array.isArray(steps) || steps.length === 0) return [];
    let prevExfil = 0;
    let redCum = 0;
    let blueCum = 0;

    return steps.map((s, idx) => {
      // Prefer explicit per-step reward deltas if present in telemetry.
      const explicitRedDelta =
        Number(s?.red_reward_delta ?? s?.red_step_reward ?? s?.red_reward ?? NaN);
      const explicitBlueDelta =
        Number(s?.blue_reward_delta ?? s?.blue_step_reward ?? s?.blue_reward ?? NaN);

      let redDelta;
      let blueDelta;

      if (Number.isFinite(explicitRedDelta) && Number.isFinite(explicitBlueDelta)) {
        redDelta = explicitRedDelta;
        blueDelta = explicitBlueDelta;
      } else {
        // Fallback proxy from live step signals when explicit deltas are absent.
        const susp = Number(s?.suspicion ?? 0);
        const det = Number(s?.detection ?? 0);
        const exfil = Number(s?.exfil_count ?? 0);
        const exfilGain = Math.max(0, exfil - prevExfil) * 0.8;
        prevExfil = exfil;

        redDelta = Math.max(-0.05, 0.08 + (1 - susp) * 0.06 + exfilGain - det * 0.02);
        blueDelta = Math.max(-0.05, 0.06 + det * 0.08 + susp * 0.02 - exfilGain * 0.35);
      }

      redCum += redDelta;
      blueCum += blueDelta;
      return {
        step: Number(s?.step ?? idx + 1),
        red: Number(redCum.toFixed(3)),
        blue: Number(blueCum.toFixed(3)),
      };
    });
  })();

  const episodeTrendData = (() => {
    const eps = stats?.episodes ?? [];
    const rt = stats?.red_totals ?? [];
    const bt = stats?.blue_totals ?? [];
    if (!eps.length) return [];
    return eps.map((ep, i) => ({
      episode: Number(ep) || i + 1,
      red: rt[i] ?? 0,
      blue: bt[i] ?? 0,
    }));
  })();

  const redBarData = RED_COMPS.map(({ key, label, color }) => ({
    name: label, value: compAvg[key] ?? 0, color,
  }));
  const blueBarData = BLUE_COMPS.map(({ key, label, color }) => ({
    name: label, value: compAvg[key] ?? 0, color,
  }));

  const compVals = [
    ...redBarData.map((d) => d.value),
    ...blueBarData.map((d) => d.value),
  ];
  const radarMax = Math.max(0.6, ...compVals.map((v) => Math.abs(v)), 0.001);

  const redRadarData = RED_COMPS.map(({ key, label }) => ({
    metric: label.length > 10 ? `${label.slice(0, 9)}…` : label,
    value: Math.max(0, compAvg[key] ?? 0),
  }));
  const blueRadarData = BLUE_COMPS.map(({ key, label }) => ({
    metric: label.length > 10 ? `${label.slice(0, 9)}…` : label,
    value: Math.max(0, compAvg[key] ?? 0),
  }));

  return (
    <div style={{ flex: 1, overflowY: 'auto', minHeight: 0, padding: '6px 0' }}>

      {/* Total rewards */}
      <div style={{ padding: '6px 10px 8px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.14em', color: 'var(--text-mute)', marginBottom: 8 }}>EPISODE REWARDS</div>
        {[
          { label: 'RED', val: redTotal, color: '#ff4444' },
          { label: 'BLUE', val: blueTotal, color: '#4488ff' },
        ].map(({ label, val, color }) => (
          <div key={label} style={{ marginBottom: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 3 }}>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 9, fontWeight: 700, color }}>{label}</span>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 14, fontWeight: 700, color }}>
                {val >= 0 ? '+' : ''}{val.toFixed(3)}
              </span>
            </div>
            <div style={{ height: 6, background: 'rgba(140,160,210,0.08)', borderRadius: 3 }}>
              <div style={{
                height: '100%',
                width: `${Math.min(100, (Math.max(0, val) / Math.max(0.001, maxR)) * 100)}%`,
                background: color, borderRadius: 3, transition: 'width 0.6s ease',
                boxShadow: val > 0.5 ? `0 0 8px ${color}60` : 'none',
              }} />
            </div>
          </div>
        ))}
      </div>

      {/* Episode history from rewards_log (last N rows) */}
      {episodeTrendData.length > 0 && (
        <div style={{ padding: '6px 10px 8px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--text-mute)', marginBottom: 6 }}>
            REWARD HISTORY (RECENT EPISODES)
          </div>
          <div style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={episodeTrendData} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
                <CartesianGrid stroke="rgba(140,160,210,0.14)" strokeDasharray="3 3" />
                <XAxis dataKey="episode" tick={{ fill: 'rgba(170,190,225,0.65)', fontSize: 9 }} />
                <YAxis tick={{ fill: 'rgba(170,190,225,0.65)', fontSize: 9 }} />
                <Tooltip contentStyle={chartTooltip.contentStyle} labelStyle={chartTooltip.labelStyle} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line type="monotone" dataKey="red" name="RED total" stroke="#ff4444" strokeWidth={2} dot={{ r: episodeTrendData.length <= 3 ? 3 : 0 }} activeDot={{ r: 4 }} />
                <Line type="monotone" dataKey="blue" name="BLUE total" stroke="#4488ff" strokeWidth={2} dot={{ r: episodeTrendData.length <= 3 ? 3 : 0 }} activeDot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Per-step progression graph */}
      {stepRewardSeries.length > 1 && (
        <div style={{ padding: '6px 10px 8px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--text-mute)', marginBottom: 6 }}>
            PER-STEP REWARD PROGRESSION
          </div>
          <div style={{ height: 210 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={stepRewardSeries} margin={{ top: 8, right: 8, left: 0, bottom: 6 }}>
                <CartesianGrid stroke="rgba(140,160,210,0.14)" strokeDasharray="3 3" />
                <XAxis dataKey="step" tick={{ fill: 'rgba(170,190,225,0.65)', fontSize: 9 }} />
                <YAxis tick={{ fill: 'rgba(170,190,225,0.65)', fontSize: 9 }} />
                <Tooltip contentStyle={chartTooltip.contentStyle} labelStyle={chartTooltip.labelStyle} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line type="monotone" dataKey="red" name="RED (cumulative)" stroke="#ff4444" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="blue" name="BLUE (cumulative)" stroke="#4488ff" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* RED / BLUE component averages — radar + horizontal bars */}
      {Object.keys(compAvg).length > 0 && (
        <div style={{ padding: '6px 10px 10px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--text-mute)', marginBottom: 8 }}>
            COMPONENT AVERAGES (ROLLING WINDOW)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 10 }}>
            <div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 7.5, fontWeight: 700, color: '#ff8888', marginBottom: 4, textAlign: 'center' }}>RED</div>
              <div style={{ height: 160 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="72%" data={redRadarData}>
                    <PolarGrid stroke="rgba(255,100,100,0.15)" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: 'rgba(255,180,160,0.85)', fontSize: 7 }} />
                    <PolarRadiusAxis angle={45} domain={[0, radarMax]} tick={{ fill: 'rgba(170,190,225,0.5)', fontSize: 7 }} />
                    <Radar name="RED" dataKey="value" stroke="#ff4444" fill="#ff4444" fillOpacity={0.28} />
                    <Tooltip contentStyle={chartTooltip.contentStyle} labelStyle={chartTooltip.labelStyle} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 7.5, fontWeight: 700, color: '#88aaff', marginBottom: 4, textAlign: 'center' }}>BLUE</div>
              <div style={{ height: 160 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="72%" data={blueRadarData}>
                    <PolarGrid stroke="rgba(100,150,255,0.15)" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: 'rgba(160,190,255,0.85)', fontSize: 7 }} />
                    <PolarRadiusAxis angle={45} domain={[0, radarMax]} tick={{ fill: 'rgba(170,190,225,0.5)', fontSize: 7 }} />
                    <Radar name="BLUE" dataKey="value" stroke="#4488ff" fill="#4488ff" fillOpacity={0.28} />
                    <Tooltip contentStyle={chartTooltip.contentStyle} labelStyle={chartTooltip.labelStyle} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 7.5, color: '#ff8888', marginBottom: 4 }}>RED (avg)</div>
              <div style={{ height: 132 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={redBarData} layout="vertical" margin={{ left: 2, right: 8, top: 2, bottom: 2 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(140,160,210,0.1)" horizontal={false} />
                    <XAxis type="number" domain={[0, 'auto']} tick={{ fill: 'rgba(170,190,225,0.65)', fontSize: 8 }} />
                    <YAxis type="category" dataKey="name" width={78} tick={{ fill: 'rgba(255,200,200,0.8)', fontSize: 7.5 }} />
                    <Tooltip contentStyle={chartTooltip.contentStyle} labelStyle={chartTooltip.labelStyle} formatter={(v) => [typeof v === 'number' ? v.toFixed(3) : v, 'avg']} />
                    <Bar dataKey="value" radius={[0, 3, 3, 0]}>
                      {redBarData.map((e) => (
                        <Cell key={e.name} fill={e.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 7.5, color: '#88aaff', marginBottom: 4 }}>BLUE (avg)</div>
              <div style={{ height: 154 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={blueBarData} layout="vertical" margin={{ left: 2, right: 8, top: 2, bottom: 2 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(140,160,210,0.1)" horizontal={false} />
                    <XAxis type="number" domain={[0, 'auto']} tick={{ fill: 'rgba(170,190,225,0.65)', fontSize: 8 }} />
                    <YAxis type="category" dataKey="name" width={78} tick={{ fill: 'rgba(190,210,255,0.85)', fontSize: 7.5 }} />
                    <Tooltip contentStyle={chartTooltip.contentStyle} labelStyle={chartTooltip.labelStyle} formatter={(v) => [typeof v === 'number' ? v.toFixed(3) : v, 'avg']} />
                    <Bar dataKey="value" radius={[0, 3, 3, 0]}>
                      {blueBarData.map((e) => (
                        <Cell key={e.name} fill={e.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recent ep detail */}
      {recentEp && (
        <div style={{ padding: '6px 10px 8px' }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8, fontWeight: 700, letterSpacing: '0.12em', color: 'var(--text-mute)', marginBottom: 6 }}>LAST EPISODE DETAIL</div>
          {[
            { label: 'Episode', val: recentEp.episode },
            { label: 'Steps', val: recentEp.steps },
            { label: 'Terminal', val: recentEp.terminal?.replace(/_/g, ' ') ?? '—' },
            { label: 'Verdict', val: recentEp.verdict?.replace(/_/g, ' ') ?? '—' },
            { label: 'Exfil completeness', val: recentEp.red_exfil?.toFixed(3) },
            { label: 'Detection accuracy', val: recentEp.blue_detection?.toFixed(3) },
          ].map(({ label, val }) => val != null && (
            <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0', borderBottom: '1px solid rgba(140,160,210,0.04)' }}>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: 'rgba(140,160,210,0.45)' }}>{label}</span>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 8.5, color: 'var(--text-dim)' }}>{val}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
