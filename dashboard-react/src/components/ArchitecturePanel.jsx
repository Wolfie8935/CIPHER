import { useCallback, useEffect, useState } from 'react';

// Build-time snapshot of repo-root architecture doc (kept in sync with Flask `/api/architecture-doc`)
import changedMdRaw from '../../../changed.md?raw';
import architectureV2Annotated from '../../../assets/architecture_v2_annotated.png';

const FALLBACK_MARKDOWN =
  typeof changedMdRaw === 'string' && changedMdRaw.trim()
    ? changedMdRaw
    : `# CIPHER system — what changed (architecture snapshot)

## War room (React + Vite)

- **\`dashboard-react/\`** — CIPHER War Room UI: live map, agent thoughts, analytics, and replay of \`episode_traces/\`.
- **Dev** — Vite on port \`5173\` with **\`/api\` → \`http://localhost:5001\`** (Flask). **Prod** — build to \`dist/\`; Flask serves static assets and APIs from the same origin.

## Flask API

- **\`dashboard-react/api_server.py\`** — reads project-root JSON/JSONL (\`live_steps.jsonl\`, \`logs/agent_*.jsonl\`, traces) and exposes REST endpoints for the dashboard.

## Python environment (CIPHER)

- **\`main.py\`**, **\`cipher/\`** — multi-agent cyber-ops RL environment: RED/BLUE commanders, subagents, dead drops, zones, and oversight. Training hooks write logs the war room consumes.

## Data flow (high level)

- Training/simulation steps → \`live_steps.jsonl\` / traces → **Flask** → **React** charts & map. Agent telemetry → \`logs/\`.
`;

const cardBase = {
  background: 'var(--bg-card)',
  border: '1px solid var(--border-hi)',
  borderRadius: 10,
  padding: '14px 16px',
  marginBottom: 12,
  boxShadow: '0 4px 24px rgba(0,0,0,0.28)',
};

const headingStyle = {
  fontFamily: 'var(--mono)',
  fontSize: 11,
  fontWeight: 700,
  letterSpacing: '0.12em',
  color: 'var(--z0)',
  marginBottom: 10,
  textTransform: 'uppercase',
};

const pStyle = { margin: '0 0 8px', lineHeight: 1.55, color: 'var(--text)', fontSize: 12 };
const liStyle = { marginBottom: 4, lineHeight: 1.5, color: 'rgba(210, 222, 240, 0.9)', fontSize: 12 };

function parseSections(markdown) {
  if (!markdown || !String(markdown).trim()) {
    return [{ title: 'Empty', body: '_No content._' }];
  }
  const text = String(markdown).trim();
  const parts = text.split(/\n(?=##\s)/);
  const out = [];
  for (const p of parts) {
    if (p.startsWith('## ')) {
      const rest = p.slice(3);
      const nl = rest.indexOf('\n');
      const title = (nl === -1 ? rest : rest.slice(0, nl)).trim();
      const body = (nl === -1 ? '' : rest.slice(nl + 1)).trim() || '_';
      out.push({ title: title || 'Section', body });
    } else {
      const cleaned = p.replace(/^#[^#\n][^\n]*/m, '').trim();
      if (cleaned) out.push({ title: 'Overview', body: cleaned });
    }
  }
  if (out.length === 0) return [{ title: 'Document', body: text }];
  return out;
}

function renderInline(s) {
  if (!s) return null;
  const boldSplit = s.split(/\*\*([^*]+)\*\*/g);
  if (boldSplit.length === 1) return s;
  return boldSplit.map((bit, j) => (j % 2 ? <strong key={j} style={{ color: '#a8c8ff' }}>{bit}</strong> : bit));
}

function CodeFence({ text }) {
  return (
    <pre className="arch-md-code">
      <code>{text.replace(/\n$/, '')}</code>
    </pre>
  );
}

function SectionBody({ body }) {
  const bits = String(body).split(/```/);
  const out = [];
  for (let i = 0; i < bits.length; i += 1) {
    if (i % 2 === 0) {
      const chunk = bits[i].trim();
      if (!chunk) continue;
      const lines = chunk.split('\n');
      let j = 0;
      while (j < lines.length) {
        const line = lines[j];
        const sub = line.match(/^\s*###\s+(.+)/);
        if (sub) {
          out.push(
            <div key={`h3-${i}-${j}`} className="arch-md-subhead">
              {renderInline(sub[1].trim())}
            </div>,
          );
          j += 1;
          continue;
        }
        if (/^\s*[-*]\s/.test(line)) {
          const listLines = [];
          while (j < lines.length && /^\s*[-*]\s/.test(lines[j])) {
            listLines.push(lines[j]);
            j += 1;
          }
          out.push(
            <ul key={`ul-${i}-${out.length}`} className="arch-md-ul">
              {listLines.map((ln, k) => (
                <li key={k} style={liStyle}>
                  {renderInline(ln.replace(/^\s*[-*]\s+/, ''))}
                </li>
              ))}
            </ul>,
          );
        } else {
          if (line.trim() === '') {
            out.push(<div key={`sp-${i}-${j}`} style={{ height: 4 }} />);
          } else {
            out.push(
              <p key={`p-${i}-${j}`} style={pStyle}>
                {renderInline(line)}
              </p>,
            );
          }
          j += 1;
        }
      }
    } else {
      const code = bits[i].replace(/^[a-zA-Z0-9_-]*\n/, '');
      out.push(<CodeFence key={`c-${i}`} text={code} />);
    }
  }
  return <div className="arch-md-body">{out}</div>;
}

function ArchitectureFlowDiagram() {
  return (
    <div className="arch-flow" aria-label="System architecture layers">
      <div className="arch-flow-glow" aria-hidden />
      <div className="arch-layer arch-layer--hero">
        <div className="arch-card-wrap arch-card-wrap--cyan">
          <div className="arch-card">
            <span className="arch-card-icon" aria-hidden>
              ⚛
            </span>
            <div className="arch-card-text">
              <div className="arch-card-title">War room</div>
              <div className="arch-card-sub">React · Vite · /api → proxy</div>
            </div>
          </div>
        </div>
      </div>

      <div className="arch-connector" aria-hidden>
        <span className="arch-connector-line" />
        <span className="arch-connector-label">HTTP /api/*</span>
        <span className="arch-connector-line" />
      </div>

      <div className="arch-layer arch-grid-2">
        <div className="arch-card-wrap arch-card-wrap--blue">
          <div className="arch-card arch-card--compact">
            <span className="arch-card-icon" aria-hidden>
              ◆
            </span>
            <div className="arch-card-text">
              <div className="arch-card-title">Flask API</div>
              <div className="arch-card-sub">api_server.py · :5001</div>
            </div>
          </div>
        </div>
        <div className="arch-card-wrap arch-card-wrap--amber">
          <div className="arch-card arch-card--compact">
            <span className="arch-card-icon" aria-hidden>
              ⧉
            </span>
            <div className="arch-card-text">
              <div className="arch-card-title">Data plane</div>
              <div className="arch-card-sub">JSONL · traces · logs/</div>
            </div>
          </div>
        </div>
      </div>

      <div className="arch-connector arch-connector--short" aria-hidden>
        <span className="arch-connector-line" />
      </div>

      <div className="arch-layer">
        <div className="arch-card-wrap arch-card-wrap--green">
          <div className="arch-card">
            <span className="arch-card-icon" aria-hidden>
              ⬡
            </span>
            <div className="arch-card-text">
              <div className="arch-card-title">Python CIPHER</div>
              <div className="arch-card-sub">cipher/ · main.py · training · agents</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ArchitecturePanel() {
  const [markdown, setMarkdown] = useState(FALLBACK_MARKDOWN);
  const [loading, setLoading] = useState(true);
  /** Shown only after a completed fetch that did not use live API body (avoids banner before first try). */
  const [showOfflineHint, setShowOfflineHint] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    let live = false;
    try {
      const res = await fetch('/api/architecture-doc');
      if (res.ok) {
        const data = await res.json();
        const md = data && typeof data.markdown === 'string' ? data.markdown.trim() : '';
        if (md) {
          setMarkdown(md);
          live = true;
        }
      }
    } catch {
      /* use fallback */
    }
    if (!live) {
      setMarkdown(FALLBACK_MARKDOWN);
      setShowOfflineHint(true);
    } else {
      setShowOfflineHint(false);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const sections = parseSections(markdown);

  return (
    <div className="arch-panel-root">
      <div className="arch-panel-header">
        <h2 className="arch-panel-title">Architecture</h2>
        <button type="button" className="arch-panel-refresh" onClick={load}>
          REFRESH
        </button>
      </div>

      {showOfflineHint && !loading && (
        <div className="arch-offline-banner" role="status">
          Offline — showing cached copy
        </div>
      )}

      <figure className="arch-diagram-figure">
        <img
          src={architectureV2Annotated}
          alt="CIPHER system architecture diagram v2, annotated layers and data flow"
          className="arch-diagram-img"
          loading="lazy"
          decoding="async"
        />
        <figcaption className="arch-diagram-caption">Repo asset: assets/architecture_v2_annotated.png</figcaption>
      </figure>

      <div className="arch-flow-wrap">
        <div className="arch-flow-wrap-label" aria-hidden>
          Stack overview
        </div>
        <ArchitectureFlowDiagram />
      </div>

      {loading && <p className="arch-loading">Loading documentation…</p>}

      <div className="arch-sections">
        {sections.map((sec) => (
          <div key={sec.title} style={cardBase} className="arch-md-card">
            <div style={headingStyle}>{sec.title}</div>
            <SectionBody body={sec.body} />
          </div>
        ))}
      </div>
    </div>
  );
}
