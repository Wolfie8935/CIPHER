import { useEffect, useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line,
} from 'recharts';

const chartTooltip = {
  contentStyle: {
    background: '#111827',
    border: '1px solid rgba(140,160,210,0.3)',
    fontFamily: 'var(--mono)',
    fontSize: 10,
  },
  labelStyle: { color: '#c8d4e8' },
};

const sectionLabel = {
  fontFamily: 'var(--mono)',
  fontSize: 8,
  fontWeight: 700,
  letterSpacing: '0.12em',
  color: 'var(--text-mute)',
  marginBottom: 6,
};

function winnerFromTerminal(term) {
  const t = String(term ?? '').toLowerCase();
  if (['exfiltration_complete', 'exfil_success', 'exfil_complete'].includes(t)) return 'RED';
  if (t === 'aborted') return 'DRAW';
  if (!t) return '?';
  return 'BLUE';
}

function OutcomeTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload;
  if (!row) return null;
  const w =
    row.redWin > 0 ? 'RED' : row.blueWin > 0 ? 'BLUE' : row.draw > 0 ? 'DRAW' : '?';
  return (
    <div
      style={{
        ...chartTooltip.contentStyle,
        padding: '8px 10px',
      }}
    >
      <div style={{ ...chartTooltip.labelStyle, marginBottom: 4 }}>
        ep{row.episode} · #{row.seq}
      </div>
      <div style={{ color: 'var(--text-dim)', fontSize: 9, marginBottom: 4 }}>{row.term}</div>
      <div style={{ fontFamily: 'var(--mono)', fontSize: 9, fontWeight: 700 }}>
        <span style={{ color: '#ff6b87' }}>{w}</span>
        <span style={{ color: 'rgba(170,190,225,0.5)', margin: '0 6px' }}>|</span>
        <span style={{ color: 'rgba(170,190,225,0.75)' }}>{row.steps} st</span>
      </div>
    </div>
  );
}

export default function HistoryPanel() {
  const [rows, setRows] = useState([]);

  useEffect(() => {
    let mounted = true;
    const fetchData = async () => {
      try {
        const res = await fetch('/api/history', { signal: AbortSignal.timeout(3000) });
        if (!res.ok) return;
        const data = await res.json();
        if (mounted && Array.isArray(data)) setRows(data);
      } catch {
        // ignore
      }
    };
    fetchData();
    const t = setInterval(fetchData, 5000);
    return () => {
      mounted = false;
      clearInterval(t);
    };
  }, []);

  const counts = useMemo(() => {
    const c = { RED: 0, BLUE: 0, DRAW: 0 };
    for (const r of rows) {
      c[winnerFromTerminal(r.terminal_reason)] += 1;
    }
    return c;
  }, [rows]);

  const WINDOW = 80;
  const chartPack = useMemo(() => {
    const slice = rows.slice(-WINDOW);
    return slice.map((r, i) => {
      const winner = winnerFromTerminal(r.terminal_reason);
      return {
        seq: i + 1,
        episode: r.episode,
        steps: Number(r.steps ?? 0),
        red_total: Number(r.red_total ?? 0),
        blue_total: Number(r.blue_total ?? 0),
        redWin: winner === 'RED' ? 1 : 0,
        blueWin: winner === 'BLUE' ? 1 : 0,
        draw: winner === 'DRAW' ? 1 : 0,
        other: winner === '?' ? 1 : 0,
        term: String(r.terminal_reason ?? 'unknown').replace(/_/g, ' '),
        run_id: String(r.run_id ?? ''),
      };
    });
  }, [rows]);

  const latestTable = useMemo(() => rows.slice(-10).reverse(), [rows]);

  return (
    <div
      style={{
        flex: 1,
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      <div style={{ padding: '8px 10px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        <div
          style={{
            fontFamily: 'var(--mono)',
            fontSize: 8,
            fontWeight: 700,
            letterSpacing: '0.14em',
            color: 'var(--text-mute)',
          }}
        >
          CROSS-RUN HISTORY
        </div>
        <div
          style={{
            marginTop: 6,
            display: 'flex',
            gap: 10,
            fontFamily: 'var(--mono)',
            fontSize: 8.5,
          }}
        >
          <span style={{ color: '#ff6b87' }}>RED {counts.RED}</span>
          <span style={{ color: '#7eb3ff' }}>BLUE {counts.BLUE}</span>
          <span style={{ color: 'rgba(205,220,245,0.8)' }}>DRAW {counts.DRAW}</span>
          <span style={{ color: 'rgba(140,160,210,0.45)', marginLeft: 'auto' }}>
            n={rows.length}
          </span>
        </div>
      </div>

      <div
        style={{
          flex: 1,
          minHeight: 0,
          overflowY: 'auto',
          paddingBottom: 8,
        }}
      >
        {chartPack.length === 0 ? (
          <div
            style={{
              padding: '24px 12px',
              fontFamily: 'var(--mono)',
              fontSize: 9,
              color: 'rgba(140,160,210,0.5)',
              textAlign: 'center',
            }}
          >
            No rewards_log episodes yet.
          </div>
        ) : (
          <>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 8,
                padding: '8px 10px 4px',
                borderBottom: '1px solid rgba(140,160,210,0.06)',
              }}
            >
              <div style={{ minWidth: 0 }}>
                <div style={sectionLabel}>OUTCOME (LAST {chartPack.length})</div>
                <div style={{ height: 150 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={chartPack}
                      margin={{ top: 4, right: 4, left: -18, bottom: 0 }}
                    >
                      <CartesianGrid stroke="rgba(140,160,210,0.14)" strokeDasharray="3 3" />
                      <XAxis
                        dataKey="seq"
                        tick={{ fill: 'rgba(170,190,225,0.55)', fontSize: 8 }}
                        minTickGap={24}
                      />
                      <YAxis
                        domain={[0, 1]}
                        ticks={[0, 1]}
                        tick={{ fill: 'rgba(170,190,225,0.55)', fontSize: 8 }}
                        width={28}
                      />
                      <Tooltip content={<OutcomeTooltip />} />
                      <Legend
                        wrapperStyle={{ fontSize: 9, paddingTop: 4 }}
                        formatter={(value) => (
                          <span style={{ color: 'rgba(190,210,240,0.85)' }}>{value}</span>
                        )}
                      />
                      <Bar
                        dataKey="redWin"
                        name="RED"
                        stackId="o"
                        fill="#ff4444"
                        radius={[0, 0, 0, 0]}
                      />
                      <Bar
                        dataKey="blueWin"
                        name="BLUE"
                        stackId="o"
                        fill="#4488ff"
                        radius={[0, 0, 0, 0]}
                      />
                      <Bar
                        dataKey="draw"
                        name="DRAW"
                        stackId="o"
                        fill="rgba(205,220,245,0.55)"
                        radius={[0, 0, 0, 0]}
                      />
                      <Bar
                        dataKey="other"
                        name="?"
                        stackId="o"
                        fill="rgba(120,130,160,0.5)"
                        radius={[2, 2, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div style={{ minWidth: 0 }}>
                <div style={sectionLabel}>STEPS / EPISODE</div>
                <div style={{ height: 150 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartPack} margin={{ top: 4, right: 4, left: -18, bottom: 0 }}>
                      <CartesianGrid stroke="rgba(140,160,210,0.14)" strokeDasharray="3 3" />
                      <XAxis
                        dataKey="seq"
                        tick={{ fill: 'rgba(170,190,225,0.55)', fontSize: 8 }}
                        minTickGap={24}
                      />
                      <YAxis
                        tick={{ fill: 'rgba(170,190,225,0.55)', fontSize: 8 }}
                        width={32}
                      />
                      <Tooltip
                        contentStyle={chartTooltip.contentStyle}
                        labelStyle={chartTooltip.labelStyle}
                        formatter={(v) => [v, 'steps']}
                        labelFormatter={(_, p) => {
                          const x = p?.[0]?.payload;
                          return x ? `ep${x.episode} · #${x.seq}` : '';
                        }}
                      />
                      <Bar dataKey="steps" fill="rgba(136,153,204,0.85)" radius={[2, 2, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div style={{ padding: '8px 10px 4px', borderBottom: '1px solid rgba(140,160,210,0.06)' }}>
              <div style={sectionLabel}>RED vs BLUE TOTAL REWARD</div>
              <div style={{ height: 190 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartPack} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
                    <CartesianGrid stroke="rgba(140,160,210,0.14)" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="seq"
                      tick={{ fill: 'rgba(170,190,225,0.65)', fontSize: 9 }}
                      minTickGap={28}
                    />
                    <YAxis tick={{ fill: 'rgba(170,190,225,0.65)', fontSize: 9 }} />
                    <Tooltip contentStyle={chartTooltip.contentStyle} labelStyle={chartTooltip.labelStyle} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Line
                      type="monotone"
                      dataKey="red_total"
                      name="RED total"
                      stroke="#ff4444"
                      strokeWidth={2}
                      dot={{ r: chartPack.length <= 4 ? 3 : 0 }}
                      activeDot={{ r: 4 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="blue_total"
                      name="BLUE total"
                      stroke="#4488ff"
                      strokeWidth={2}
                      dot={{ r: chartPack.length <= 4 ? 3 : 0 }}
                      activeDot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {latestTable.length > 0 && (
              <div style={{ padding: '8px 10px 0' }}>
                <div style={sectionLabel}>LATEST EPISODES</div>
                <div
                  style={{
                    fontFamily: 'var(--mono)',
                    fontSize: 8,
                    color: 'rgba(140,160,210,0.45)',
                    display: 'grid',
                    gridTemplateColumns: '32px 36px 44px 1fr 52px 52px',
                    gap: 4,
                    padding: '4px 6px',
                    borderBottom: '1px solid rgba(140,160,210,0.08)',
                  }}
                >
                  <span>ep</span>
                  <span>win</span>
                  <span>st</span>
                  <span>terminal</span>
                  <span style={{ textAlign: 'right' }}>R</span>
                  <span style={{ textAlign: 'right' }}>B</span>
                </div>
                {latestTable.map((r, idx) => {
                  const winner = winnerFromTerminal(r.terminal_reason);
                  const wColor =
                    winner === 'RED' ? '#ff6b87' : winner === 'BLUE' ? '#7eb3ff' : 'rgba(205,220,245,0.75)';
                  return (
                    <div
                      key={`${r.run_id || 'run'}-${r.episode}-${idx}`}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: '32px 36px 44px 1fr 52px 52px',
                        gap: 4,
                        alignItems: 'center',
                        padding: '4px 6px',
                        borderBottom: '1px solid rgba(140,160,210,0.05)',
                        fontFamily: 'var(--mono)',
                        fontSize: 8.5,
                      }}
                    >
                      <span style={{ color: 'rgba(170,190,225,0.55)' }}>{r.episode}</span>
                      <span style={{ color: wColor, fontWeight: 700 }}>{winner}</span>
                      <span style={{ color: 'rgba(170,190,225,0.5)' }}>{r.steps}</span>
                      <span
                        style={{
                          color: 'rgba(190,210,240,0.7)',
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                        }}
                      >
                        {String(r.terminal_reason ?? '').replace(/_/g, ' ')}
                      </span>
                      <span
                        style={{
                          color: Number(r.red_total ?? 0) >= 0 ? '#ff9dab' : '#ff5555',
                          textAlign: 'right',
                        }}
                      >
                        {Number(r.red_total ?? 0) >= 0 ? '+' : ''}
                        {Number(r.red_total ?? 0).toFixed(2)}
                      </span>
                      <span style={{ color: '#9ec5ff', textAlign: 'right' }}>
                        {Number(r.blue_total ?? 0) >= 0 ? '+' : ''}
                        {Number(r.blue_total ?? 0).toFixed(2)}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
