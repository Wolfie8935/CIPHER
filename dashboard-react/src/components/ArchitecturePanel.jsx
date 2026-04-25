import { useCallback, useEffect, useState } from 'react';
import architectureV2Annotated from '../../../assets/architecture_v2_annotated.png';

const FALLBACK_MARKDOWN = `# CIPHER v2 Architecture — Commander + Subagent Model

## Overview

CIPHER puts two LLM-powered commanders in direct adversarial conflict inside a 50-node
enterprise network. Each commander dynamically spawns specialist subagents on demand.

## RED Team

- **RED Commander** — top-level attacker brain (Llama-3.1-8B). Decides strategy, spawns subagents.
- **Spawned subagents** — Planner (long-horizon), Analyst (belief mapping), Operative (stealth movement), Exfiltrator (file extraction). Spawned on demand; dismissed when task complete.
- Goal: traverse Zones 0→1→2→3 and exfiltrate the classified file without detection.

## BLUE Team

- **BLUE Commander** — top-level defender brain. Decides which specialists to deploy.
- **Spawned subagents** — Surveillance (anomaly feed), Threat Hunter (active investigation), Deception Architect (honeypots), Forensics (path reconstruction). Auto-spawned on escalating detection signals.
- Goal: detect RED, reconstruct RED's path, and trigger the correct alert.

## Oversight Auditor

- Independent 9th LLM that watches both teams every episode.
- Issues verdicts: \`red_dominates\` | \`blue_dominates\` | \`contested\` | \`degenerate\`.
- Detects reward hacking and collusion. Applies fleet bonuses (±0.15).

## Data Flow

\`\`\`
Training / main.py run
  → live_steps.jsonl + episode_traces/ + logs/
    → API Server (:5001)
      → React War Room (:5173 / dist/)
\`\`\`

## Key Mechanics

- **Asymmetric observations** — RED sees the network map (honeypots masked). BLUE sees only an anomaly feed (not RED's position).
- **Dead-drop vault** — SHA-256 encrypted inter-agent memory with token budgets. BLUE can tamper with drops.
- **12 trap types** — FalseTrail, HoneypotPoison, DeadDropTamper, Breadcrumb, TemporalDecoy, ...
- **Dynamic difficulty** — 6-axis curriculum driven by rolling RED win rate (openenv.yaml curriculum block).
- **Self-play pipeline** — failure_cases.jsonl + success_cases.jsonl mined each episode for LoRA fine-tuning.

## Quick Start

\`\`\`bash
python main.py                    # single episode, stub (no API)
python main.py --live             # live HF inference
python main.py --live --episodes 5
python main.py --train            # training loop
python verify_openenv.py          # compliance check
\`\`\`
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

// ── v2 Commander + Subagent flow diagram ──────────────────────────────────────
function ArchitectureFlowDiagram() {
  const teamBox = (color, border, title, sub, agents) => ({
    background: color,
    border: `1.5px solid ${border}`,
    borderRadius: 10,
    padding: '12px 16px',
    minWidth: 200,
    maxWidth: 240,
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0 }}>
      {/* Top row: Oversight */}
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 8 }}>
        <div style={{
          background: 'rgba(40,80,20,0.55)', border: '1.5px solid #4caf50',
          borderRadius: 10, padding: '10px 20px', textAlign: 'center',
        }}>
          <div style={{ color: '#4caf50', fontWeight: 700, fontSize: 13 }}>Oversight Auditor</div>
          <div style={{ color: '#8bc34a', fontSize: 11, marginTop: 2 }}>9th independent LLM · fleet verdicts · reward-hacking detection</div>
        </div>
      </div>

      {/* Oversight arrows down */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: 260, color: '#4caf50', fontSize: 18 }}>
        <span>↓</span><span>↓</span>
      </div>

      {/* Middle row: RED Commander | Episode Runner | BLUE Commander */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 20, marginTop: 4 }}>

        {/* RED side */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
          <div style={{
            background: 'rgba(80,10,10,0.65)', border: '1.5px solid #ff4444',
            borderRadius: 10, padding: '12px 16px', minWidth: 220, textAlign: 'center',
          }}>
            <div style={{ color: '#ff4444', fontWeight: 700, fontSize: 14 }}>RED Commander</div>
            <div style={{ color: '#ff8888', fontSize: 11, marginTop: 2 }}>commander.py · Llama-3.1-8B</div>
          </div>
          <div style={{ color: '#ff4444', fontSize: 16 }}>↓ spawns</div>
          <div style={{
            background: 'rgba(60,10,10,0.55)', border: '1px solid #cc3333',
            borderRadius: 8, padding: '8px 14px', minWidth: 220,
          }}>
            <div style={{ color: '#ffaaaa', fontSize: 11, fontWeight: 600, marginBottom: 4 }}>Dynamic subagents</div>
            {['Planner', 'Analyst', 'Operative', 'Exfiltrator'].map(r => (
              <div key={r} style={{ color: '#ff7777', fontSize: 11, marginBottom: 2 }}>· {r}</div>
            ))}
          </div>
        </div>

        {/* Episode Runner center */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, paddingTop: 8 }}>
          <div style={{ color: '#ff4444', fontSize: 18 }}>→</div>
          <div style={{
            background: 'rgba(20,30,60,0.85)', border: '1.5px solid #5588ff',
            borderRadius: 10, padding: '12px 18px', textAlign: 'center', minWidth: 180,
          }}>
            <div style={{ color: '#88aaff', fontWeight: 700, fontSize: 13 }}>Episode Runner</div>
            <div style={{ color: '#6688cc', fontSize: 11, marginTop: 2 }}>_episode_runner.py</div>
            <div style={{ color: '#6688cc', fontSize: 10, marginTop: 4 }}>EpisodeState · ScenarioGenerator</div>
            <div style={{ color: '#6688cc', fontSize: 10, marginTop: 2 }}>50-node network · 4 zones</div>
          </div>
          <div style={{ color: '#4488ff', fontSize: 18 }}>←</div>
        </div>

        {/* BLUE side */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
          <div style={{
            background: 'rgba(10,10,80,0.65)', border: '1.5px solid #4488ff',
            borderRadius: 10, padding: '12px 16px', minWidth: 220, textAlign: 'center',
          }}>
            <div style={{ color: '#4488ff', fontWeight: 700, fontSize: 14 }}>BLUE Commander</div>
            <div style={{ color: '#8888ff', fontSize: 11, marginTop: 2 }}>commander.py · Llama-3.1-8B</div>
          </div>
          <div style={{ color: '#4488ff', fontSize: 16 }}>↓ spawns</div>
          <div style={{
            background: 'rgba(10,10,60,0.55)', border: '1px solid #3366cc',
            borderRadius: 8, padding: '8px 14px', minWidth: 220,
          }}>
            <div style={{ color: '#aaaaff', fontSize: 11, fontWeight: 600, marginBottom: 4 }}>Dynamic subagents</div>
            {['Surveillance', 'Threat Hunter', 'Deception Architect', 'Forensics'].map(r => (
              <div key={r} style={{ color: '#7799ff', fontSize: 11, marginBottom: 2 }}>· {r}</div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom: data flow */}
      <div style={{ color: '#555', fontSize: 18, marginTop: 10 }}>↓</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 4 }}>
        <div style={{
          background: 'rgba(20,24,30,0.8)', border: '1px solid #334',
          borderRadius: 8, padding: '8px 16px', textAlign: 'center',
        }}>
          <div style={{ color: '#8b949e', fontSize: 12, fontWeight: 600 }}>live_steps.jsonl · episode_traces/ · rewards_log.csv</div>
          <div style={{ color: '#555', fontSize: 11, marginTop: 2 }}>→ API Server (:5001) → React War Room (:5173)</div>
        </div>
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 20, marginTop: 16, flexWrap: 'wrap', justifyContent: 'center' }}>
        {[
          ['#ff4444', 'RED team (attacker)'],
          ['#4488ff', 'BLUE team (defender)'],
          ['#4caf50', 'Oversight Auditor'],
          ['#8b949e', 'Data / logs'],
        ].map(([col, label]) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 12, height: 12, borderRadius: 2, background: col }} />
            <span style={{ color: '#8b949e', fontSize: 11 }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ArchitecturePanel() {
  const [markdown, setMarkdown] = useState(FALLBACK_MARKDOWN);
  const [loading, setLoading] = useState(true);
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
        <h2 className="arch-panel-title">Architecture — CIPHER v2</h2>
        <button type="button" className="arch-panel-refresh" onClick={load}>
          REFRESH
        </button>
      </div>

      {showOfflineHint && !loading && (
        <div className="arch-offline-banner" role="status">
          API offline — showing built-in v2 architecture
        </div>
      )}

      <div className="arch-flow-wrap" style={{ marginBottom: 20 }}>
        <div className="arch-flow-wrap-label" aria-hidden>
          CIPHER v2 — Commander + Dynamic Subagent Model
        </div>
        <ArchitectureFlowDiagram />
      </div>

      <figure className="arch-diagram-figure">
        <img
          src={architectureV2Annotated}
          alt="CIPHER v2 system architecture — commander and subagent model"
          className="arch-diagram-img"
          loading="lazy"
          decoding="async"
        />
        <figcaption className="arch-diagram-caption">
          CIPHER v2: RED/BLUE commanders each spawn specialist subagents on demand.
          Oversight Auditor judges both teams independently.
        </figcaption>
      </figure>

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
