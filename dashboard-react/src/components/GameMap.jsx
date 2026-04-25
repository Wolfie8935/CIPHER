import { useState, useRef, useEffect, useMemo, useCallback } from 'react';

// ─── Canvas dimensions ─────────────────────────────────────────────
const W = 2600, H = 1800, CX = W / 2, CY = H / 2;

// ─── Zone config — cluster positions (no boxes, network graph style) ─
const Z = [
  { label: 'PERIMETER', short: 'P', color: '#00e5ff', cx: 850,  cy: 680,  rx: 340, ry: 270 },
  { label: 'GENERAL',   short: 'G', color: '#69f0ae', cx: 1600, cy: 520,  rx: 400, ry: 270 },
  { label: 'SENSITIVE', short: 'S', color: '#ffd740', cx: 800,  cy: 1320, rx: 440, ry: 195 },
  { label: 'CRITICAL',  short: 'C', color: '#ff6b6b', cx: 2000, cy: 1250, rx: 220, ry: 185 },
];

// ─── Scatter nodes within elliptical zone cluster ─────────────────
function buildPositions(nodes) {
  const byZone = [[], [], [], []];
  for (const n of nodes) byZone[Math.min(3, Number(n.zone ?? 0))].push({ ...n, id: Number(n.id) });

  const pos = {};
  byZone.forEach((zn, z) => {
    const { cx, cy, rx, ry } = Z[z];
    const cnt = Math.max(1, zn.length);

    zn.forEach((n, i) => {
      const h1 = ((n.id * 1103515245 + z * 214013 + 12345) & 0x7fffffff) / 0x7fffffff;
      const h2 = ((n.id * 22695477  + z * 6364136 + 1)    & 0x7fffffff) / 0x7fffffff;

      // Sunflower spiral distributes nodes evenly inside the ellipse
      const angle = i * 2.39996323 + h1 * 0.8;
      const r     = Math.sqrt((i + 0.5) / cnt);
      const jx    = (h1 - 0.5) * rx * 0.22;
      const jy    = (h2 - 0.5) * ry * 0.22;

      pos[n.id] = {
        x: cx + rx * r * Math.cos(angle) + jx,
        y: cy + ry * r * Math.sin(angle) + jy,
      };
    });
  });
  return pos;
}

// ─── Easing ───────────────────────────────────────────────────────
function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

// ─── Smooth agent animation hook ─────────────────────────────────
function useSmooth(targetId, positions, speed) {
  const [pos, setPos]  = useState({ x: CX, y: CY });
  const rafRef  = useRef(null);
  const currRef = useRef({ x: CX, y: CY });

  useEffect(() => {
    const target = positions[targetId];
    if (!target) return;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    const from  = { ...currRef.current };
    const start = performance.now();
    const dur   = Math.max(160, 1000 / Math.max(0.25, speed || 1));

    const tick = (now) => {
      const t    = Math.min((now - start) / dur, 1);
      const ease = easeInOutCubic(t);
      const next = { x: from.x + (target.x - from.x) * ease, y: from.y + (target.y - from.y) * ease };
      currRef.current = next;
      setPos(next);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [targetId, speed]);  // eslint-disable-line react-hooks/exhaustive-deps

  return pos;
}

// ─── Extract node id from step — handles both replay and live formats ──
function extractRedNode(step) {
  if (!step) return null;
  if (step.red_node != null) return Number(step.red_node);
  const m = (step.red_action ?? '').match(/→\s*n?(\d+)/);
  return m ? Number(m[1]) : null;
}

// ─── Agent dot component ─────────────────────────────────────────
function AgentDot({ x, y, color, radius, label, pulseFast, filter }) {
  if (!x || !y) return null;
  return (
    <g filter={filter}>
      <circle cx={x} cy={y} r={radius + 8 + pulseFast * 10}
        fill="none" stroke={color}
        strokeWidth={1} strokeOpacity={0.25 - pulseFast * 0.1}
      />
      <circle cx={x} cy={y} r={radius + 4} fill={`${color}22`} />
      <circle cx={x} cy={y} r={radius}
        fill={color}
        stroke="rgba(255,255,255,0.5)"
        strokeWidth={1.2}
      />
      {label && (
        <text x={x} y={y - radius - 7}
          textAnchor="middle" fontSize={8}
          fontFamily="'JetBrains Mono', monospace"
          fontWeight="700" letterSpacing="0.08em"
          fill={color} opacity={0.85}
        >{label}</text>
      )}
    </g>
  );
}

// ─── Main component ───────────────────────────────────────────────
export default function GameMap({ graph, steps, agentStatus, redThoughts, blueThoughts, speed = 1 }) {
  const svgRef       = useRef(null);
  const [pan, setPan]   = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(0.42);
  const isDragging   = useRef(false);
  const lastMouse    = useRef({ x: 0, y: 0 });
  const [hovered,  setHovered]  = useState(null);
  const [selected, setSelected] = useState(null);

  // Pulse timer
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 40);
    return () => clearInterval(id);
  }, []);
  const pulseFast = 0.5 + 0.5 * Math.sin(tick * 0.18);
  const pulseSlow = 0.5 + 0.5 * Math.sin(tick * 0.08);

  const nodes    = graph?.nodes ?? [];
  const edges    = graph?.edges ?? [];
  const nodeMap  = useMemo(() => Object.fromEntries(nodes.map(n => [Number(n.id), n])), [nodes]);
  const positions = useMemo(() => buildPositions(nodes), [nodes]);

  // RED trail from step history
  const trail = useMemo(() => {
    const path = [];
    for (const s of steps) {
      const n = extractRedNode(s);
      if (n !== null && (path.length === 0 || path[path.length - 1] !== n)) path.push(n);
    }
    return path.slice(-20);
  }, [steps]);

  const visited = useMemo(() => new Set(trail), [trail]);

  // 4 RED agents — staggered along trail
  const RED_LABELS  = ['PLNR', 'ANLT', 'OPRT', 'EXFL'];
  const RED_OFFSETS = [0, 2, 4, 7];
  const RED_RADII   = [13, 11, 10, 9];

  const redNodeIds = useMemo(() =>
    RED_OFFSETS.map(o => trail[Math.max(0, trail.length - 1 - o)] ?? null),
    [trail]  // eslint-disable-line react-hooks/exhaustive-deps
  );

  const r0 = useSmooth(redNodeIds[0], positions, speed);
  const r1 = useSmooth(redNodeIds[1], positions, speed * 0.85);
  const r2 = useSmooth(redNodeIds[2], positions, speed * 0.70);
  const r3 = useSmooth(redNodeIds[3], positions, speed * 0.55);
  const redPositions = [r0, r1, r2, r3];

  // 4 BLUE agents — trailing at larger offsets
  const BLUE_LABELS  = ['SURV', 'HUNT', 'DCVR', 'FRNS'];
  const BLUE_OFFSETS = [3, 6, 9, 13];
  const BLUE_RADII   = [12, 11, 10, 9];

  const blueNodeIds = useMemo(() =>
    BLUE_OFFSETS.map(o => trail[Math.max(0, trail.length - 1 - o)] ?? null),
    [trail]  // eslint-disable-line react-hooks/exhaustive-deps
  );

  const b0 = useSmooth(blueNodeIds[0], positions, speed * 0.7);
  const b1 = useSmooth(blueNodeIds[1], positions, speed * 0.6);
  const b2 = useSmooth(blueNodeIds[2], positions, speed * 0.5);
  const b3 = useSmooth(blueNodeIds[3], positions, speed * 0.4);
  const bluePositions = [b0, b1, b2, b3];

  const currentRedNode = redNodeIds[0] ?? (nodes.find(n => n.is_entry)?.id ?? nodes[0]?.id ?? 0);

  const latestRed  = redThoughts[redThoughts.length - 1];
  const latestBlue = blueThoughts[blueThoughts.length - 1];

  // ── Pan / zoom ───────────────────────────────────────────────────
  const onWheel = useCallback((e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.11;
    setZoom(z => Math.min(3.0, Math.max(0.18, z * delta)));
  }, []);

  const onMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    isDragging.current = true;
    lastMouse.current  = { x: e.clientX, y: e.clientY };
  }, []);

  const onMouseMove = useCallback((e) => {
    if (!isDragging.current) return;
    const dx = e.clientX - lastMouse.current.x;
    const dy = e.clientY - lastMouse.current.y;
    lastMouse.current = { x: e.clientX, y: e.clientY };
    setPan(p => ({ x: p.x + dx, y: p.y + dy }));
  }, []);

  const onMouseUp = useCallback(() => { isDragging.current = false; }, []);

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [onWheel]);

  const resetView = () => { setZoom(0.42); setPan({ x: 0, y: 0 }); };

  const svgW = svgRef.current?.clientWidth  ?? 1200;
  const svgH = svgRef.current?.clientHeight ?? 700;

  const isCenter = (p) => !p || (Math.abs(p.x - CX) < 5 && Math.abs(p.y - CY) < 5);

  return (
    <div
      style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden' }}
      className={isDragging.current ? 'game-area dragging' : 'game-area'}
    >
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        style={{ display: 'block' }}
      >
        <defs>
          <filter id="redGlow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="12" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="blueGlow" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="9" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="goldGlow" x="-70%" y="-70%" width="240%" height="240%">
            <feGaussianBlur stdDeviation="10" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="nodeGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="zoneBlob" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="28" />
          </filter>
        </defs>

        {/* ── World transform ── */}
        <g transform={`translate(${pan.x + svgW / 2}, ${pan.y + svgH / 2}) scale(${zoom}) translate(${-CX}, ${-CY})`}>

          {/* Map background */}
          <rect x={0} y={0} width={W} height={H} fill="#080a10" />

          {/* Subtle grid */}
          <g opacity={0.04}>
            {Array.from({ length: 27 }, (_, i) => (
              <line key={`gv${i}`} x1={i * 100} y1={0} x2={i * 100} y2={H}
                stroke="#6080c0" strokeWidth={0.5} />
            ))}
            {Array.from({ length: 19 }, (_, i) => (
              <line key={`gh${i}`} x1={0} y1={i * 100} x2={W} y2={i * 100}
                stroke="#6080c0" strokeWidth={0.5} />
            ))}
          </g>

          {/* ── Subtle zone cluster blobs (no boxes) ── */}
          {Z.map((z, i) => (
            <ellipse key={`blob-${i}`}
              cx={z.cx} cy={z.cy}
              rx={z.rx + 80} ry={z.ry + 70}
              fill={`${z.color}08`}
              filter="url(#zoneBlob)"
            />
          ))}

          {/* Zone labels at cluster center-top */}
          {Z.map((z, i) => (
            <text key={`zlabel-${i}`}
              x={z.cx}
              y={z.cy - z.ry - 18}
              textAnchor="middle"
              fontSize={10}
              fontFamily="'JetBrains Mono', monospace"
              fontWeight="700"
              letterSpacing="0.22em"
              fill={z.color}
              opacity={0.42}
            >{z.label}</text>
          ))}

          {/* ── Network edges ── */}
          {edges.map((e, i) => {
            const src = Number(e.source ?? e.src);
            const tgt = Number(e.target ?? e.tgt);
            const s   = positions[src], t = positions[tgt];
            if (!s || !t) return null;
            const isTraversed = trail.includes(src) && trail.includes(tgt) &&
              Math.abs(trail.indexOf(src) - trail.indexOf(tgt)) === 1;
            return (
              <line key={i}
                x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                stroke={isTraversed ? 'rgba(255,68,68,0.55)' : 'rgba(80,110,180,0.14)'}
                strokeWidth={isTraversed ? 2.5 : 0.9}
                strokeLinecap="round"
              />
            );
          })}

          {/* ── RED trail heat ── */}
          {trail.slice(0, -1).map((nodeId, i) => {
            const p = positions[nodeId];
            if (!p) return null;
            const opacity = (i + 1) / trail.length * 0.18;
            return (
              <circle key={`trail-${nodeId}-${i}`}
                cx={p.x} cy={p.y}
                r={18 + (trail.length - i) * 1.5}
                fill={`rgba(255,68,68,${opacity})`}
              />
            );
          })}

          {/* ── Nodes — colored by zone, semi-transparent ── */}
          {nodes.map((node) => {
            const id   = Number(node.id);
            const p    = positions[id];
            if (!p) return null;
            const z          = Math.min(3, Number(node.zone ?? 0));
            const cfg        = Z[z];
            const isHVT      = node.is_hvt;
            const isEntry    = node.is_entry;
            const isHoneypot = node.is_honeypot;
            const isVisited  = visited.has(id);
            const isCurrent  = id === currentRedNode;
            const isHov      = id === hovered;
            const isSel      = id === selected;

            const r = isHVT ? 22 : isEntry ? 18 : 14;

            return (
              <g
                key={id}
                style={{ cursor: 'pointer' }}
                onClick={() => setSelected(isSel ? null : id)}
                onMouseEnter={() => setHovered(id)}
                onMouseLeave={() => setHovered(null)}
              >
                {(isHov || isSel) && (
                  <circle cx={p.x} cy={p.y} r={r + 10} fill="none"
                    stroke={cfg.color} strokeWidth={1}
                    strokeDasharray="4,6"
                    strokeOpacity={0.55}
                  />
                )}

                {isHVT && (
                  <>
                    <circle cx={p.x} cy={p.y}
                      r={r + 18 + pulseSlow * 12}
                      fill="rgba(255,193,7,0.05)"
                      filter="url(#goldGlow)"
                    />
                    <circle cx={p.x} cy={p.y}
                      r={r + 8 + pulseSlow * 6}
                      fill="rgba(255,193,7,0.08)"
                    />
                  </>
                )}

                {isHoneypot && (
                  <circle cx={p.x} cy={p.y} r={r + 8} fill="none"
                    stroke="rgba(68,136,255,0.45)" strokeWidth={1.5}
                    strokeDasharray={`${3 + pulseFast * 3},5`}
                  />
                )}

                <circle cx={p.x} cy={p.y} r={r}
                  fill={
                    isHVT     ? 'rgba(255,193,7,0.22)' :
                    isVisited ? `${cfg.color}30` :
                    `${cfg.color}14`
                  }
                  stroke={cfg.color}
                  strokeWidth={isHVT ? 2 : isCurrent ? 2.5 : isVisited ? 1.8 : 1.2}
                  strokeOpacity={isHVT ? 1 : isCurrent ? 1 : isVisited ? 0.8 : 0.5}
                  filter={(isVisited || isCurrent) ? 'url(#nodeGlow)' : undefined}
                />

                <text x={p.x} y={p.y + 4} textAnchor="middle" fontSize={isHVT ? 12 : 9}>
                  {isHVT ? '⭐' : isEntry ? '🚪' : isHoneypot ? '🪤' : ''}
                </text>

                <text
                  x={p.x} y={p.y + r + 13}
                  textAnchor="middle"
                  fontSize={7}
                  fontFamily="'JetBrains Mono', monospace"
                  fill={isVisited || isCurrent || isHVT ? cfg.color : 'rgba(140,170,220,0.35)'}
                  fontWeight={isVisited || isHVT ? '700' : '400'}
                  opacity={isHov || isSel || isVisited || isHVT ? 1 : 0.7}
                >
                  {String(node.hostname ?? `n${id}`).slice(0, 10)}
                </text>
              </g>
            );
          })}

          {/* ── 4 BLUE agents ── */}
          {bluePositions.map((bp, i) => {
            if (isCenter(bp)) return null;
            return (
              <AgentDot
                key={`blue-${i}`}
                x={bp.x} y={bp.y}
                color="#4488ff"
                radius={BLUE_RADII[i]}
                label={BLUE_LABELS[i]}
                pulseFast={pulseFast * (1 - i * 0.15)}
                filter="url(#blueGlow)"
              />
            );
          })}

          {/* ── 4 RED agents ── */}
          {redPositions.map((rp, i) => {
            if (isCenter(rp)) return null;
            return (
              <AgentDot
                key={`red-${i}`}
                x={rp.x} y={rp.y}
                color={i === 0 ? '#ff2222' : '#ff4444'}
                radius={RED_RADII[i]}
                label={RED_LABELS[i]}
                pulseFast={i === 0 ? pulseFast : pulseFast * (1 - i * 0.18)}
                filter="url(#redGlow)"
              />
            );
          })}

          {/* ── RED thought bubble ── */}
          {latestRed && !isCenter(r0) && (
            <foreignObject
              x={r0.x + 28} y={r0.y - 95}
              width={255} height={95}
              style={{ overflow: 'visible', pointerEvents: 'none' }}
            >
              <div style={{
                background: 'rgba(20,8,8,0.94)',
                border: '1.5px solid rgba(255,68,68,0.55)',
                borderRadius: 10, padding: '6px 10px',
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 10, color: '#e8c8c8', lineHeight: 1.5,
                boxShadow: '0 4px 20px rgba(255,68,68,0.2)',
              }}>
                <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: '0.14em',
                  textTransform: 'uppercase', color: '#ff4444', marginBottom: 3 }}>
                  {latestRed.agent_id?.replace(/_\d+$/, '').replace('red_', '').toUpperCase()}
                </div>
                {latestRed.reasoning?.slice(0, 105)}{latestRed.reasoning?.length > 105 ? '…' : ''}
              </div>
            </foreignObject>
          )}

          {/* ── BLUE thought bubble ── */}
          {latestBlue && !isCenter(b0) && (
            <foreignObject
              x={b0.x + 24} y={b0.y - 85}
              width={235} height={85}
              style={{ overflow: 'visible', pointerEvents: 'none' }}
            >
              <div style={{
                background: 'rgba(6,12,24,0.94)',
                border: '1.5px solid rgba(68,136,255,0.50)',
                borderRadius: 10, padding: '6px 10px',
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 10, color: '#c0d0f0', lineHeight: 1.5,
                boxShadow: '0 4px 20px rgba(68,136,255,0.18)',
              }}>
                <div style={{ fontSize: 8, fontWeight: 700, letterSpacing: '0.14em',
                  textTransform: 'uppercase', color: '#4488ff', marginBottom: 3 }}>
                  {latestBlue.agent_id?.replace(/_\d+$/, '').replace('blue_', '').toUpperCase()}
                </div>
                {latestBlue.reasoning?.slice(0, 95)}{latestBlue.reasoning?.length > 95 ? '…' : ''}
              </div>
            </foreignObject>
          )}

          {/* ── Selected node tooltip ── */}
          {selected !== null && positions[selected] && (
            (() => {
              const n  = nodeMap[selected];
              const p  = positions[selected];
              const z  = Math.min(3, Number(n?.zone ?? 0));
              const zc = Z[z];
              return (
                <foreignObject x={p.x + 24} y={p.y - 55} width={215} height={150}
                  style={{ overflow: 'visible', pointerEvents: 'all' }}
                  onClick={e => e.stopPropagation()}
                >
                  <div style={{
                    background: 'rgba(14,18,28,0.97)',
                    border: `1.5px solid ${zc.color}55`,
                    borderRadius: 10, padding: '9px 12px',
                    fontFamily: "'JetBrains Mono', monospace", fontSize: 11,
                    color: '#c8d4e8',
                    boxShadow: `0 4px 24px rgba(0,0,0,0.6), 0 0 0 1px ${zc.color}20`,
                  }}>
                    <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 6, color: zc.color }}>
                      {n?.hostname ?? `node-${selected}`}
                    </div>
                    <div style={{ color: 'rgba(180,200,230,0.55)', display: 'flex', flexDirection: 'column', gap: 3 }}>
                      <span>ID: <b style={{ color: '#c8d4e8' }}>{selected}</b></span>
                      <span>Zone: <b style={{ color: zc.color }}>{zc.label}</b></span>
                      {n?.is_hvt      && <span style={{ color: '#ffc107' }}>⭐ High-Value Target</span>}
                      {n?.is_entry    && <span style={{ color: '#69f0ae' }}>🚪 Entry Point</span>}
                      {n?.is_honeypot && <span style={{ color: '#4488ff' }}>🪤 Honeypot</span>}
                      {visited.has(selected) && <span style={{ color: '#ff4444' }}>⬤ Visited by RED</span>}
                    </div>
                    <div style={{ marginTop: 7, fontSize: 9, color: 'rgba(140,160,200,0.4)', cursor: 'pointer' }}
                      onClick={() => setSelected(null)}>✕ close</div>
                  </div>
                </foreignObject>
              );
            })()
          )}
        </g>

        {/* ── Zone legend (fixed, top-left) ── */}
        <g>
          <rect x={10} y={8} width={130} height={88} rx={8}
            fill="rgba(10,12,18,0.82)"
            stroke="rgba(140,160,210,0.10)"
            strokeWidth={1}
          />
          {Z.map((z, i) => (
            <g key={i}>
              <circle cx={26} cy={24 + i * 18} r={5} fill={z.color} opacity={0.85} />
              <text x={36} y={29 + i * 18}
                fontSize={9.5} fontFamily="'JetBrains Mono', monospace"
                fontWeight="700" letterSpacing="0.1em"
                fill={z.color} opacity={0.80}
              >{z.label}</text>
            </g>
          ))}
        </g>

        {/* ── Agent legend (top-right) ── */}
        <g transform={`translate(${svgW - 155}, 8)`}>
          <rect x={0} y={0} width={145} height={64} rx={8}
            fill="rgba(10,12,18,0.82)"
            stroke="rgba(140,160,210,0.10)"
            strokeWidth={1}
          />
          <circle cx={14} cy={18} r={5} fill="#ff4444" />
          <text x={24} y={22} fontSize={8.5} fontFamily="'JetBrains Mono', monospace" fontWeight="700" fill="#ff4444" opacity={0.8}>
            RED  PLNR·ANLT·OPRT·EXFL
          </text>
          <circle cx={14} cy={38} r={5} fill="#4488ff" />
          <text x={24} y={42} fontSize={8.5} fontFamily="'JetBrains Mono', monospace" fontWeight="700" fill="#4488ff" opacity={0.8}>
            BLUE SURV·HUNT·DCVR·FRNS
          </text>
          <text x={14} y={60} fontSize={7.5} fontFamily="'JetBrains Mono', monospace" fill="rgba(140,160,210,0.30)" letterSpacing="0.06em">
            scroll to zoom · drag to pan
          </text>
        </g>
      </svg>

      {/* ── Viewport controls ── */}
      <div style={{ position: 'absolute', bottom: 12, right: 12, display: 'flex', gap: 5, zIndex: 50 }}>
        {[
          { label: '⌂', action: resetView },
          { label: '+', action: () => setZoom(z => Math.min(3.0, z * 1.25)) },
          { label: '−', action: () => setZoom(z => Math.max(0.18, z * 0.8)) },
        ].map(({ label, action }) => (
          <button key={label} onClick={action} style={{
            padding: '4px 11px',
            background: 'rgba(14,16,21,0.90)',
            border: '1px solid rgba(140,160,210,0.15)',
            borderRadius: 7,
            color: 'rgba(200,215,240,0.6)',
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: label === '+' || label === '−' ? 16 : 12,
            cursor: 'pointer',
            boxShadow: '0 2px 10px rgba(0,0,0,0.4)',
            lineHeight: 1,
          }}>{label}</button>
        ))}
      </div>
    </div>
  );
}
