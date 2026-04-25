import { useEffect, useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  Cell,
  LabelList,
} from 'recharts';

const chartTooltip = {
  contentStyle: {
    background: 'linear-gradient(165deg, #131a25 0%, #0b0f16 100%)',
    border: '1px solid rgba(100, 140, 220, 0.38)',
    fontFamily: 'var(--mono)',
    fontSize: 10,
    borderRadius: 4,
    boxShadow: '0 6px 24px rgba(0,0,0,0.5), 0 0 0 1px rgba(100, 130, 200, 0.08) inset',
  },
  labelStyle: { color: '#a8b8d8', fontSize: 9, fontWeight: 600, marginBottom: 2 },
  itemStyle: { color: '#e4eaf5' },
};

const TERMINAL_CHART_TOP_N = 8;
const TERMINAL_TICK_MAX = 11;

/** Map raw API keys to stable bucket id + short axis label + full tooltip label. */
function classifyTerminalRaw(raw) {
  const rawStr = String(raw ?? '').trim();
  if (!rawStr) {
    return { id: 'empty', short: 'Empty', full: '(no terminal reason)' };
  }
  if (/^\d+$/.test(rawStr)) {
    return {
      id: 'misc_numeric',
      short: 'Misc',
      full: 'Unmapped numeric / episode-index keys',
    };
  }
  const t = rawStr.toLowerCase().replace(/-/g, '_');
  if (/exfil|exfiltrat/.test(t)) {
    return { id: 'exfil', short: 'Exfil', full: 'Exfiltration / objective reached' };
  }
  if (/detect|intercept|caught|bust/.test(t)) {
    return { id: 'detected', short: 'Detected', full: 'Detected or intercepted' };
  }
  if (/max_step|max_steps|timeout|time_?out|time_limit|step_limit/.test(t)) {
    return { id: 'timeout', short: 'Timeout', full: 'Max steps / time limit' };
  }
  if (/abort|cancel/.test(t)) {
    return { id: 'aborted', short: 'Aborted', full: 'Run aborted' };
  }
  if (/unknown|unk$|unclassified/.test(t)) {
    return { id: 'unknown', short: 'Unknown', full: 'Unknown outcome' };
  }
  if (/fail|error|loss|defeat/.test(t)) {
    return { id: 'failed', short: 'Failed', full: 'Failure / error' };
  }
  if (/success|win|complete|done/.test(t) && !/exfil/.test(t)) {
    return { id: 'success', short: 'Success', full: rawStr.replace(/_/g, ' ') };
  }
  const human = rawStr.replace(/_/g, ' ');
  return { id: `raw:${t}`, short: human, full: human };
}

function prepareTerminalOutcomeRows(counts, topN = TERMINAL_CHART_TOP_N) {
  const agg = new Map();
  for (const [raw, v] of Object.entries(counts ?? {})) {
    const c = Number(v);
    if (!Number.isFinite(c) || c <= 0) continue;
    const cl = classifyTerminalRaw(raw);
    const prev = agg.get(cl.id);
    if (prev) {
      prev.count += c;
      prev.raws.add(raw);
    } else {
      agg.set(cl.id, {
        id: cl.id,
        count: c,
        short: cl.short,
        full: cl.full,
        raws: new Set([raw]),
      });
    }
  }
  const sorted = [...agg.values()].sort((a, b) => b.count - a.count);
  const head = sorted.slice(0, topN);
  const tail = sorted.slice(topN);
  if (tail.length === 0) {
    return head.map((row, j) => ({
      ...row,
      rank: j + 1,
      axisTick: row.short,
      fullLabel: row.full,
    }));
  }
  const otherCount = tail.reduce((s, r) => s + r.count, 0);
  const typeCount = tail.length;
  const merged = [
    ...head,
    {
      id: 'other_tail',
      count: otherCount,
      short: 'Other',
      full: `Other outcomes (${typeCount} type${typeCount === 1 ? '' : 's'}, ${otherCount} total)`,
      raws: new Set(tail.flatMap((r) => [...r.raws])),
    },
  ];
  return merged.map((row, j) => ({
    ...row,
    rank: j + 1,
    axisTick: row.short,
    fullLabel: row.full,
  }));
}

/** Per stable `id` from classifyTerminalRaw — vivid fills, war-room legible. */
const TERMINAL_COLOR_MAP = {
  exfil: 'url(#gTermExfil)',
  success: 'url(#gTermSuccess)',
  detected: 'url(#gTermDetected)',
  timeout: 'url(#gTermTimeout)',
  aborted: 'url(#gTermAborted)',
  failed: 'url(#gTermFailed)',
  unknown: 'url(#gTermUnknown)',
  empty: 'url(#gTermEmpty)',
  other_tail: 'url(#gTermOther)',
  misc_numeric: 'url(#gTermMisc)',
};

const TERMINAL_STROKE_MAP = {
  exfil: 'rgba(110, 255, 195, 0.55)',
  success: 'rgba(74, 222, 128, 0.55)',
  detected: 'rgba(120, 200, 255, 0.5)',
  timeout: 'rgba(255, 200, 100, 0.55)',
  aborted: 'rgba(255, 130, 120, 0.55)',
  failed: 'rgba(255, 100, 110, 0.5)',
  unknown: 'rgba(200, 170, 255, 0.5)',
  empty: 'rgba(180, 190, 210, 0.45)',
  other_tail: 'rgba(200, 205, 255, 0.5)',
  misc_numeric: 'rgba(200, 210, 230, 0.5)',
};

/** Distinct saturated hues for `raw:*` and any unlisted id. */
const TERMINAL_FALLBACK_FILLS = [
  'url(#gTermFb0)', 'url(#gTermFb1)', 'url(#gTermFb2)', 'url(#gTermFb3)',
  'url(#gTermFb4)', 'url(#gTermFb5)', 'url(#gTermFb6)', 'url(#gTermFb7)',
];

function getTerminalBarStroke(id) {
  const t = String(id);
  if (t.startsWith('raw:')) return 'rgba(255, 255, 255, 0.4)';
  if (TERMINAL_STROKE_MAP[t] != null) return TERMINAL_STROKE_MAP[t];
  return 'rgba(255, 255, 255, 0.38)';
}

function termBarFill(categoryId, i) {
  const t = String(categoryId);
  if (t.startsWith('raw:')) return TERMINAL_FALLBACK_FILLS[i % TERMINAL_FALLBACK_FILLS.length];
  if (TERMINAL_COLOR_MAP[t] != null) return TERMINAL_COLOR_MAP[t];
  return TERMINAL_FALLBACK_FILLS[i % TERMINAL_FALLBACK_FILLS.length];
}

function terminalAxisTickFormatter(value) {
  if (value == null) return '';
  const s = String(value).replace(/_/g, ' ');
  if (s.length <= TERMINAL_TICK_MAX) return s;
  return `${s.slice(0, Math.max(1, TERMINAL_TICK_MAX - 1))}…`;
}

function sectionTitleStyle() {
  return { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, flexShrink: 0 };
}

function SectionTitle({ children }) {
  return (
    <div style={sectionTitleStyle()}>
      <div
        style={{
          width: 2,
          height: 12,
          borderRadius: 1,
          background: 'linear-gradient(180deg, #7a9eff 0%, #4a62c4 100%)',
          boxShadow: '0 0 8px rgba(100, 140, 255, 0.45)',
        }}
      />
      <div
        style={{
          fontFamily: 'var(--mono)',
          fontSize: 8,
          letterSpacing: '0.12em',
          color: 'rgba(180, 200, 240, 0.92)',
          fontWeight: 600,
        }}
      >
        {children}
      </div>
    </div>
  );
}

function chartMargins() {
  return { top: 6, right: 10, left: 2, bottom: 2 };
}

function trapBarFill(i, n) {
  const hue = 265 - i * 9;
  const sat = Math.max(38, 90 - i * 9);
  const light = Math.max(42, 64 - i * 2.5);
  return `hsl(${hue} ${sat}% ${light}%)`;
}

function TerminalOutcomeDefs() {
  const pair = (top, bottom) => (
    <>
      <stop offset="0%" stopColor={top} stopOpacity={1} />
      <stop offset="100%" stopColor={bottom} stopOpacity={1} />
    </>
  );
  return (
    <defs>
      <linearGradient id="gTermExfil" x1="0" y1="0" x2="0" y2="1">
        {pair('#2ef0a0', '#0a9f5c')}
      </linearGradient>
      <linearGradient id="gTermSuccess" x1="0" y1="0" x2="0" y2="1">
        {pair('#4ade80', '#15803d')}
      </linearGradient>
      <linearGradient id="gTermDetected" x1="0" y1="0" x2="0" y2="1">
        {pair('#4eb8ff', '#1d4ed8')}
      </linearGradient>
      <linearGradient id="gTermTimeout" x1="0" y1="0" x2="0" y2="1">
        {pair('#fbbf24', '#b45309')}
      </linearGradient>
      <linearGradient id="gTermAborted" x1="0" y1="0" x2="0" y2="1">
        {pair('#ff7a6a', '#b91c1c')}
      </linearGradient>
      <linearGradient id="gTermFailed" x1="0" y1="0" x2="0" y2="1">
        {pair('#fb7185', '#991b1b')}
      </linearGradient>
      <linearGradient id="gTermUnknown" x1="0" y1="0" x2="0" y2="1">
        {pair('#c4b5fd', '#5b21b6')}
      </linearGradient>
      <linearGradient id="gTermEmpty" x1="0" y1="0" x2="0" y2="1">
        {pair('#9ca3af', '#3f3f46')}
      </linearGradient>
      <linearGradient id="gTermOther" x1="0" y1="0" x2="0" y2="1">
        {pair('#a5b4fc', '#4f46e5')}
      </linearGradient>
      <linearGradient id="gTermMisc" x1="0" y1="0" x2="0" y2="1">
        {pair('#94a3b8', '#475569')}
      </linearGradient>
      {[
        ['gTermFb0', '#a78bfa', '#5b21b6'],
        ['gTermFb1', '#f472b6', '#9d174d'],
        ['gTermFb2', '#2dd4bf', '#0f766e'],
        ['gTermFb3', '#fbbf24', '#92400e'],
        ['gTermFb4', '#818cf8', '#3730a3'],
        ['gTermFb5', '#c084fc', '#6b21a8'],
        ['gTermFb6', '#38bdf8', '#0369a1'],
        ['gTermFb7', '#4ade80', '#166534'],
      ].map(([id, a, b]) => (
        <linearGradient key={id} id={id} x1="0" y1="0" x2="0" y2="1">
          {pair(a, b)}
        </linearGradient>
      ))}
    </defs>
  );
}

function eloLabelFormatter(label, payload) {
  const p = payload && payload[0] && payload[0].payload;
  if (p) {
    const { rawEp, idx } = p;
    const epStr = rawEp != null && rawEp !== '' ? ` · log ep ${rawEp}` : '';
    return `Run #${idx ?? label}${epStr}`;
  }
  return `Run #${label}`;
}

export default function AnalyticsPanel() {
  const [data, setData] = useState(null);

  useEffect(() => {
    let mounted = true;
    const fetchData = async () => {
      try {
        const res = await fetch('/api/analytics', { signal: AbortSignal.timeout(3000) });
        if (!res.ok) return;
        const json = await res.json();
        if (mounted) setData(json);
      } catch {
        // ignore network errors
      }
    };
    fetchData();
    const t = setInterval(fetchData, 5000);
    return () => {
      mounted = false;
      clearInterval(t);
    };
  }, []);

  const elo = useMemo(() => {
    const eps = data?.episodes ?? [];
    const red = data?.red_elo ?? [];
    const blue = data?.blue_elo ?? [];
    return eps.map((ep, i) => ({
      idx: i + 1,
      rawEp: ep,
      red: Number(red[i] ?? 1500),
      blue: Number(blue[i] ?? 1500),
    }));
  }, [data]);

  const eloYBase = useMemo(() => {
    if (elo.length === 0) return 1300;
    const vals = elo.flatMap((d) => [d.red, d.blue]);
    return Math.min(...vals) - 32;
  }, [elo]);

  const terminals = useMemo(
    () => prepareTerminalOutcomeRows(data?.terminal_counts ?? {}, TERMINAL_CHART_TOP_N),
    [data]
  );

  const terminalLabelTopK = 3;

  const traps = useMemo(
    () => (data?.trap_hot_nodes ?? []).slice(0, 8).map((r) => ({
      node: `n${r.node}`,
      count: Number(r.count ?? 0),
    })),
    [data]
  );

  const m = chartMargins();

  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
      <div style={{ padding: '8px 10px', borderBottom: '1px solid var(--border)' }}>
        <div
          style={{
            fontFamily: 'var(--mono)',
            fontSize: 8,
            fontWeight: 700,
            letterSpacing: '0.14em',
            color: 'var(--text-mute)',
          }}
        >
          ANALYTICS
        </div>
      </div>

      <div style={{ height: 180, padding: '10px 10px 4px', borderBottom: '1px solid var(--border)' }}>
        <SectionTitle>ELO TREND</SectionTitle>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={elo} margin={m}>
            <defs>
              <linearGradient id="redEloArea" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ff6b6b" stopOpacity={0.5} />
                <stop offset="100%" stopColor="#ff2d4d" stopOpacity={0.04} />
              </linearGradient>
              <linearGradient id="blueEloArea" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#5ab0ff" stopOpacity={0.45} />
                <stop offset="100%" stopColor="#2563eb" stopOpacity={0.04} />
              </linearGradient>
              <filter id="eloGlowRed" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="1.4" result="b" />
                <feMerge>
                  <feMergeNode in="b" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <filter id="eloGlowBlue" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="1.4" result="b" />
                <feMerge>
                  <feMergeNode in="b" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
            <CartesianGrid
              vertical={false}
              stroke="rgba(130,150,205,0.12)"
              strokeDasharray="3 3"
            />
            <XAxis
              dataKey="idx"
              tick={{ fill: 'rgba(175,198,230,0.8)', fontSize: 8.5, fontFamily: 'var(--mono)' }}
              tickLine={{ stroke: 'rgba(120,140,200,0.3)' }}
              axisLine={{ stroke: 'rgba(100,120,180,0.2)' }}
              minTickGap={8}
              interval="preserveStartEnd"
              label={{
                value: 'Training row',
                position: 'insideBottom',
                offset: -1,
                style: { fill: 'rgba(150,170,210,0.5)', fontSize: 7, fontFamily: 'var(--mono)' },
              }}
            />
            <YAxis
              tick={{ fill: 'rgba(175,198,230,0.8)', fontSize: 8.5, fontFamily: 'var(--mono)' }}
              tickLine={false}
              axisLine={{ stroke: 'rgba(100,120,180,0.2)' }}
              width={40}
              domain={elo.length ? [eloYBase, 'auto'] : [1400, 1600]}
            />
            <Tooltip
              {...chartTooltip}
              labelFormatter={eloLabelFormatter}
              formatter={(v, name) => [Number(v).toFixed(1), name]}
            />
            <Legend
              verticalAlign="top"
              align="right"
              height={18}
              wrapperStyle={{ fontFamily: 'var(--mono)', fontSize: 9, paddingTop: 0 }}
              iconType="plainline"
              formatter={(s) => (
                <span style={{ color: 'rgba(200, 215, 240, 0.95)' }}>{s}</span>
              )}
            />
            <Area
              type="natural"
              dataKey="red"
              name="RED Elo"
              fill="url(#redEloArea)"
              stroke="none"
              baseLine={eloYBase}
              legendType="none"
            />
            <Area
              type="natural"
              dataKey="blue"
              name="BLUE Elo"
              fill="url(#blueEloArea)"
              stroke="none"
              baseLine={eloYBase}
              legendType="none"
            />
            <Line
              type="natural"
              dataKey="red"
              name="RED Elo"
              stroke="#ff3d5c"
              strokeWidth={4}
              dot={false}
              strokeOpacity={0.2}
              isAnimationActive
              legendType="none"
            />
            <Line
              type="natural"
              dataKey="blue"
              name="BLUE Elo"
              stroke="#2d7cff"
              strokeWidth={4}
              dot={false}
              strokeOpacity={0.18}
              isAnimationActive
              legendType="none"
            />
            <Line
              type="natural"
              dataKey="red"
              name="RED Elo"
              stroke="#ff4d6a"
              strokeWidth={2.2}
              dot={false}
              connectNulls
              filter="url(#eloGlowRed)"
            />
            <Line
              type="natural"
              dataKey="blue"
              name="BLUE Elo"
              stroke="#4da3ff"
              strokeWidth={2.2}
              dot={false}
              connectNulls
              filter="url(#eloGlowBlue)"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div style={{ minHeight: 190, height: 190, padding: '10px 10px 6px', borderBottom: '1px solid var(--border)' }}>
        <SectionTitle>TERMINAL OUTCOMES</SectionTitle>
        <ResponsiveContainer width="100%" minHeight={188} height="100%">
          <BarChart
            data={terminals}
            margin={{ top: 8, right: 10, left: 2, bottom: 28 }}
            barCategoryGap="18%"
          >
            <TerminalOutcomeDefs />
            <CartesianGrid
              stroke="rgba(130,150,205,0.12)"
              strokeDasharray="2 3"
              vertical
            />
            <XAxis
              dataKey="axisTick"
              tick={{
                fill: 'rgba(190, 210, 240, 0.92)',
                fontSize: 8,
                fontFamily: 'var(--mono)',
              }}
              tickFormatter={terminalAxisTickFormatter}
              height={52}
              interval={0}
              angle={-15}
              textAnchor="end"
              tickLine={false}
              axisLine={{ stroke: 'rgba(100,120,180,0.25)' }}
            />
            <YAxis
              tick={{ fill: 'rgba(170,198,230,0.88)', fontSize: 8.5, fontFamily: 'var(--mono)' }}
              allowDecimals={false}
              width={34}
              tickLine={false}
              axisLine={{ stroke: 'rgba(100,120,180,0.2)' }}
            />
            <Tooltip
              {...chartTooltip}
              formatter={(v) => [v, 'count']}
              labelFormatter={(_, payload) => {
                const row = payload?.[0]?.payload;
                return row?.fullLabel ?? row?.axisTick ?? '';
              }}
            />
            <Bar dataKey="count" maxBarSize={48} radius={[8, 8, 2, 2]}>
              {terminals.map((row, i) => (
                <Cell
                  key={row.id}
                  fill={termBarFill(row.id, i)}
                  stroke={getTerminalBarStroke(row.id)}
                  strokeWidth={1.1}
                />
              ))}
              <LabelList
                dataKey="count"
                content={(props) => {
                  const { x, y, width, value, index } = props;
                  if (value == null || x == null) return null;
                  const row = terminals[index];
                  if (!row || row.rank > terminalLabelTopK || Number(value) <= 0) return null;
                  return (
                    <text
                      x={x + width / 2}
                      y={y - 4}
                      textAnchor="middle"
                      style={{
                        fontFamily: 'var(--mono)',
                        fontSize: 7.5,
                        fill: 'rgba(220, 230, 250, 0.96)',
                        textShadow: '0 0 6px rgba(0,0,0,0.65)',
                      }}
                    >
                      {value}
                    </text>
                  );
                }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={{ height: 160, padding: '10px 10px 8px' }}>
        <SectionTitle>TRAP HOT NODES</SectionTitle>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={traps} margin={{ ...m, left: 0 }}>
            <defs>
              {traps.map((_, i) => {
                const top = trapBarFill(i, traps.length);
                const deep = trapBarFill(Math.min(i + 2, traps.length - 1), traps.length);
                return (
                  <linearGradient key={i} id={`trapG${i}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={top} stopOpacity={0.95} />
                    <stop offset="100%" stopColor={deep} stopOpacity={0.7} />
                  </linearGradient>
                );
              })}
            </defs>
            <CartesianGrid
              stroke="rgba(130,150,205,0.1)"
              strokeDasharray="2 3"
            />
            <XAxis
              dataKey="node"
              tick={{ fill: 'rgba(170,198,230,0.85)', fontSize: 8, fontFamily: 'var(--mono)' }}
              tickLine={false}
              axisLine={{ stroke: 'rgba(100,120,180,0.25)' }}
            />
            <YAxis
              tick={{ fill: 'rgba(170,198,230,0.85)', fontSize: 8.5, fontFamily: 'var(--mono)' }}
              allowDecimals={false}
              width={32}
              tickLine={false}
              axisLine={{ stroke: 'rgba(100,120,180,0.2)' }}
            />
            <Tooltip
              {...chartTooltip}
              formatter={(v) => [v, 'traps']}
            />
            <Bar dataKey="count" maxBarSize={36} radius={[8, 8, 0, 0]}>
              {traps.map((_, i) => (
                <Cell
                  key={`t-${i}`}
                  fill={`url(#trapG${i})`}
                  stroke="rgba(255,255,255,0.08)"
                  strokeWidth={0.5}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
