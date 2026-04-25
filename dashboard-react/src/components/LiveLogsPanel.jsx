import { useState, useEffect, useRef, useCallback } from 'react';

const POLL_MS = 1500;
const BOTTOM_THRESHOLD = 72;
const DEFAULT_LIMIT = 200;

function sortKey(entry) {
  const typeOrder = { step: 0, thought: 1, training: 2 };
  const t = String(entry.type || '');
  return [
    Number(entry.episode) || 0,
    Number(entry.step) || 0,
    String(entry.timestamp || ''),
    typeOrder[t] ?? 9,
    String(entry.agent_id || ''),
    String(entry.id || ''),
  ];
}

/** Mirrors `api_server.live_logs` when `/api/live-logs` is unavailable (404 / Flask off). */
function mergeClientLiveLogs(steps, thoughts, tevents, limit) {
  const entries = [];

  for (const s of steps || []) {
    const ep = Number(s.episode) || 0;
    const st = Number(s.step) || 0;
    const ts = String(s.timestamp || '');
    const parts = [];
    if (s.run_id) parts.push(`run=${s.run_id}`);
    if (s.red_action) parts.push(`RED ${s.red_action}`);
    if (s.blue_actions) parts.push(`BLUE ${s.blue_actions}`);
    if (s.zone) parts.push(`zone=${s.zone}`);
    if (s.suspicion != null) parts.push(`sus=${s.suspicion}`);
    if (s.detection != null) parts.push(`det=${s.detection}`);
    if (Number(s.exfil_count) > 0) {
      const ex = Array.isArray(s.exfil_files) ? s.exfil_files : [];
      parts.push(`EXFIL×${s.exfil_count} ${ex.slice(0, 3).join(',')}`);
    }
    if (s.key_event) parts.push(`KEY:${s.key_event}`);
    const msg = parts.length ? parts.join(' │ ') : '(empty step)';
    const rawStep = JSON.stringify(s);
    entries.push({
      id: `step-${ep}-${st}`,
      type: 'step',
      side: 'SYSTEM',
      severity: 'info',
      episode: ep,
      step: st,
      timestamp: ts,
      message: msg,
      snippet: rawStep.length > 420 ? `${rawStep.slice(0, 420)}…` : rawStep,
    });

    (s.trap_events || []).forEach((te, i) => {
      const rawTe = JSON.stringify(te);
      entries.push({
        id: `trap-${ep}-${st}-${i}`,
        type: 'training',
        side: 'SYSTEM',
        severity: 'warn',
        episode: ep,
        step: st,
        timestamp: ts,
        message: `TRAP ${te.action_type || ''} team=${te.team || ''} @n${te.node ?? '?'}`,
        snippet: rawTe.length > 320 ? `${rawTe.slice(0, 320)}…` : rawTe,
      });
    });
    (s.drop_events || []).forEach((de, i) => {
      const rawDe = JSON.stringify(de);
      entries.push({
        id: `drop-${ep}-${st}-${i}`,
        type: 'training',
        side: 'RED',
        severity: 'info',
        episode: ep,
        step: st,
        timestamp: ts,
        message: `DROP ${de.action_type || ''} @n${de.node ?? '?'}`,
        snippet: rawDe.length > 320 ? `${rawDe.slice(0, 320)}…` : rawDe,
      });
    });
  }

  for (const t of thoughts || []) {
    const team = String(t.team || '').toLowerCase();
    const side = team === 'red' ? 'RED' : team === 'blue' ? 'BLUE' : 'SYSTEM';
    const ep = Number(t.episode) || 0;
    const st = Number(t.step) || 0;
    const ts = String(t.timestamp || '');
    const aid = String(t.agent_id || '');
    const atype = String(t.action_type || '');
    const tail = [];
    if (t.target_node != null) tail.push(`→n${t.target_node}`);
    if (t.target_file) tail.push(String(t.target_file));
    const tailS = tail.join(' ');
    const reason = String(t.reasoning || '');
    const reasonShort = reason.length <= 240 ? reason : `${reason.slice(0, 237)}…`;
    const msg = `${aid} ${atype}${tailS ? ` ${tailS}` : ''} — ${reasonShort}`;
    const atLower = atype.toLowerCase();
    const sev = atLower.includes('exfil') || atype === 'trigger_alert' || atype === 'abort' ? 'critical' : 'info';
    const rawTh = JSON.stringify(t);
    entries.push({
      id: `thought-${aid}-${st}-${ts}`,
      type: 'thought',
      side,
      severity: sev,
      episode: ep,
      step: st,
      timestamp: ts,
      agent_id: aid,
      message: msg,
      snippet: rawTh.length > 400 ? `${rawTh.slice(0, 400)}…` : rawTh,
    });
  }

  (tevents || []).forEach((te, i) => {
    const etype = String(te.event_type || '');
    if (!['trap', 'drop', 'dead_drop'].some((k) => etype.includes(k))) return;
    const ep = Number(te.episode) || 0;
    const st = Number(te.step) || 0;
    const ts = String(te.timestamp || '');
    const tm = String(te.team || '').toLowerCase();
    const tside = tm === 'red' ? 'RED' : tm === 'blue' ? 'BLUE' : 'SYSTEM';
    const rawEv = JSON.stringify(te);
    const sev = etype.includes('fired') || etype.includes('tamper') ? 'warn' : 'info';
    entries.push({
      id: `tevent-${ep}-${st}-${i}-${etype}`,
      type: 'training',
      side: tside,
      severity: sev,
      episode: ep,
      step: st,
      timestamp: ts,
      message: String(te.detail || etype.replace(/_/g, ' ')),
      snippet: rawEv.length > 360 ? `${rawEv.slice(0, 360)}…` : rawEv,
    });
  });

  entries.sort((a, b) => {
    const ka = sortKey(a);
    const kb = sortKey(b);
    for (let j = 0; j < ka.length; j += 1) {
      if (ka[j] < kb[j]) return -1;
      if (ka[j] > kb[j]) return 1;
    }
    return 0;
  });
  const lim = Math.max(50, Math.min(500, limit || DEFAULT_LIMIT));
  return entries.slice(-lim);
}

async function fetchJsonSafe(url) {
  const res = await fetch(url, { signal: AbortSignal.timeout(4000) });
  if (!res.ok) return { ok: false, status: res.status, data: null };
  const data = await res.json();
  return { ok: true, status: res.status, data };
}

function formatTime(ts) {
  if (!ts) return '—';
  const s = String(ts);
  if (s.length >= 19 && s.includes('T')) return s.slice(11, 23);
  try {
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) return s.slice(11, 23) || '—';
    return d.toLocaleTimeString('en-GB', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return '—';
  }
}

function sideStyle(side) {
  const s = String(side || '').toUpperCase();
  if (s === 'RED') return { bg: 'rgba(255,68,68,0.14)', border: 'rgba(255,107,135,0.45)', color: '#ff8a8a' };
  if (s === 'BLUE') return { bg: 'rgba(68,136,255,0.12)', border: 'rgba(126,179,255,0.42)', color: '#9ec5ff' };
  return { bg: 'rgba(0,229,255,0.08)', border: 'rgba(255,215,64,0.35)', color: '#ffd740' };
}

function typeStyle(type) {
  const t = String(type || '');
  if (t === 'step') return { label: 'STEP', mute: 'rgba(160,180,220,0.55)' };
  if (t === 'thought') return { label: 'THOUGHT', mute: 'rgba(160,180,220,0.55)' };
  return { label: 'EVENT', mute: 'rgba(160,180,220,0.55)' };
}

export default function LiveLogsPanel() {
  const [lines, setLines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [usingFallback, setUsingFallback] = useState(false);
  const [expanded, setExpanded] = useState({});
  const [stickBottom, setStickBottom] = useState(true);

  const scrollRef = useRef(null);
  const stickRef = useRef(true);

  useEffect(() => {
    stickRef.current = stickBottom;
  }, [stickBottom]);

  const fetchLogs = useCallback(async () => {
    try {
      const res = await fetch(`/api/live-logs?limit=${DEFAULT_LIMIT}`, { signal: AbortSignal.timeout(4000) });
      if (res.ok) {
        const data = await res.json();
        const next = Array.isArray(data?.lines) ? data.lines : [];
        setLines(next);
        setError(null);
        setUsingFallback(false);
        return;
      }

      // Flask missing route or proxy → build same feed from endpoints that usually exist.
      const [stepsR, thoughtsR, eventsR] = await Promise.all([
        fetchJsonSafe('/api/live-steps'),
        fetchJsonSafe('/api/thoughts'),
        fetchJsonSafe('/api/training-events'),
      ]);

      const steps = stepsR.ok && Array.isArray(stepsR.data) ? stepsR.data.slice(-120) : [];
      const thoughts = thoughtsR.ok && Array.isArray(thoughtsR.data) ? thoughtsR.data.slice(-500) : [];
      const tevents = eventsR.ok && Array.isArray(eventsR.data) ? eventsR.data.slice(-300) : [];

      if (steps.length || thoughts.length || tevents.length) {
        const merged = mergeClientLiveLogs(steps, thoughts, tevents, DEFAULT_LIMIT);
        setLines(merged);
        setUsingFallback(true);
        setError(null);
        return;
      }

      throw new Error(
        stepsR.ok || thoughtsR.ok || eventsR.ok
          ? `HTTP ${res.status} (merge sources empty)`
          : `HTTP ${res.status} — start API: cd dashboard-react && python api_server.py`,
      );
    } catch (e) {
      setError(e?.message || 'fetch failed');
      setUsingFallback(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchLogs();
    const id = setInterval(fetchLogs, POLL_MS);
    return () => clearInterval(id);
  }, [fetchLogs]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !stickRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [lines]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < BOTTOM_THRESHOLD;
    stickRef.current = nearBottom;
    setStickBottom(nearBottom);
  };

  const panelBorder = '1px solid var(--border)';
  const header = (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 12,
        padding: '10px 12px',
        borderBottom: panelBorder,
        fontFamily: 'var(--mono)',
        flexShrink: 0,
      }}
    >
      <div>
        <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: '0.14em', color: 'var(--text)' }}>
          LIVE LOGS
        </div>
        <div style={{ fontSize: 9, color: 'var(--text-mute)', marginTop: 3, letterSpacing: '0.06em' }}>
          Merged step summaries, agent thoughts, trap/drop signals · GET /api/live-logs
          {usingFallback && (
            <span style={{ display: 'block', marginTop: 4, color: 'rgba(255, 213, 79, 0.85)' }}>
              · client merge (live-steps + thoughts) — run Flask for full /api/live-logs
            </span>
          )}
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
        <span style={{ fontSize: 9, color: 'var(--text-mute)', letterSpacing: '0.08em' }}>
          {lines.length} lines
        </span>
        <button
          type="button"
          onClick={() => {
            setStickBottom(true);
            stickRef.current = true;
            requestAnimationFrame(() => {
              const el = scrollRef.current;
              if (el) el.scrollTop = el.scrollHeight;
            });
          }}
          style={{
            fontFamily: 'var(--mono)',
            fontSize: 9,
            fontWeight: 700,
            letterSpacing: '0.1em',
            padding: '4px 10px',
            borderRadius: 6,
            border: panelBorder,
            background: stickBottom ? 'rgba(0,229,255,0.12)' : 'rgba(20,28,40,0.9)',
            color: 'var(--text)',
            cursor: 'pointer',
          }}
        >
          AUTO-SCROLL {stickBottom ? 'ON' : 'OFF'}
        </button>
        <button
          type="button"
          onClick={() => fetchLogs()}
          style={{
            fontFamily: 'var(--mono)',
            fontSize: 9,
            fontWeight: 700,
            letterSpacing: '0.1em',
            padding: '4px 10px',
            borderRadius: 6,
            border: panelBorder,
            background: 'rgba(20,28,40,0.9)',
            color: 'var(--text)',
            cursor: 'pointer',
          }}
        >
          REFRESH
        </button>
      </div>
    </div>
  );

  if (loading && lines.length === 0) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 280 }}>
        {header}
        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontFamily: 'var(--mono)',
            fontSize: 11,
            color: 'var(--text-mute)',
            letterSpacing: '0.12em',
          }}
        >
          LOADING LIVE LOGS…
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 280 }}>
      {header}
      {error && (
        <div
          style={{
            margin: '8px 12px 0',
            padding: '8px 10px',
            borderRadius: 8,
            border: '1px solid rgba(255,107,107,0.35)',
            background: 'rgba(255,68,68,0.08)',
            fontFamily: 'var(--mono)',
            fontSize: 10,
            color: '#ff9a9a',
          }}
        >
          {error}
        </div>
      )}
      {usingFallback && !error && lines.length > 0 && (
        <div
          style={{
            margin: '8px 12px 0',
            padding: '8px 10px',
            borderRadius: 8,
            border: '1px solid rgba(255, 213, 79, 0.35)',
            background: 'rgba(255, 193, 7, 0.08)',
            fontFamily: 'var(--mono)',
            fontSize: 9,
            color: 'rgba(255, 224, 130, 0.95)',
            letterSpacing: '0.06em',
          }}
        >
          /api/live-logs unavailable — merged from /api/live-steps + /api/thoughts (+ training-events when present).
          For the single merged API, run: <span style={{ color: '#fff' }}>python api_server.py</span> in dashboard-react (port 5001).
        </div>
      )}
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="live-logs-feed"
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '8px 10px 12px',
          fontFamily: 'var(--mono)',
          fontSize: 11,
          lineHeight: 1.5,
          color: 'var(--text)',
        }}
      >
        {lines.length === 0 && !loading && (
          <div style={{ color: 'var(--text-mute)', letterSpacing: '0.1em', padding: 16, textAlign: 'center' }}>
            NO LOG LINES YET — run training / live episode to populate live_steps.jsonl and logs/agent_thoughts.jsonl
          </div>
        )}
        {lines.map((row, idx) => {
          const ss = sideStyle(row.side);
          const ts = typeStyle(row.type);
          const timeStr = formatTime(row.timestamp);
          const isOpen = expanded[row.id];
          return (
            <div
              key={`${row.id}-${idx}`}
              style={{
                borderLeft: `3px solid ${ss.border}`,
                background: 'rgba(14,18,28,0.55)',
                marginBottom: 6,
                padding: '6px 8px 6px 10px',
                borderRadius: 4,
                border: '1px solid rgba(140,160,210,0.08)',
              }}
            >
              <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', gap: 8, rowGap: 4 }}>
                <span style={{ fontSize: 9, color: 'var(--text-mute)', minWidth: 86 }}>{timeStr}</span>
                <span style={{ fontSize: 9, color: 'var(--text-mute)' }}>
                  ep{row.episode ?? 0}·s{row.step ?? 0}
                </span>
                <span
                  style={{
                    fontSize: 8,
                    fontWeight: 800,
                    letterSpacing: '0.12em',
                    padding: '2px 6px',
                    borderRadius: 4,
                    background: ss.bg,
                    color: ss.color,
                    border: `1px solid ${ss.border}`,
                  }}
                >
                  {String(row.side || '—').toUpperCase()}
                </span>
                <span style={{ fontSize: 8, fontWeight: 700, color: ts.mute, letterSpacing: '0.14em' }}>
                  {ts.label}
                </span>
                {row.severity === 'critical' && (
                  <span style={{ fontSize: 8, fontWeight: 800, color: '#ff6b6b', letterSpacing: '0.1em' }}>CRIT</span>
                )}
              </div>
              <div style={{ marginTop: 4, whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: 'rgba(210,220,240,0.92)' }}>
                {row.message}
              </div>
              {row.snippet && (
                <button
                  type="button"
                  onClick={() => setExpanded((prev) => ({ ...prev, [row.id]: !prev[row.id] }))}
                  style={{
                    marginTop: 6,
                    fontFamily: 'var(--mono)',
                    fontSize: 8,
                    letterSpacing: '0.12em',
                    border: 'none',
                    background: 'transparent',
                    color: 'var(--text-mute)',
                    cursor: 'pointer',
                    padding: 0,
                  }}
                >
                  {isOpen ? '▼ hide raw' : '▸ raw json'}
                </button>
              )}
              {isOpen && row.snippet && (
                <pre
                  style={{
                    marginTop: 6,
                    padding: 8,
                    borderRadius: 6,
                    background: 'rgba(0,0,0,0.35)',
                    border: '1px solid var(--border)',
                    fontSize: 9,
                    lineHeight: 1.4,
                    overflowX: 'auto',
                    color: 'rgba(180,200,230,0.85)',
                  }}
                >
                  {row.snippet}
                </pre>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
