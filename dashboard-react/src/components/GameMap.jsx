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

function getNodeFiles(node) {
  const files = Array.isArray(node?.files) ? node.files : [];
  return files
    .map((f) => {
      if (typeof f === 'string') return f.trim();
      if (f && typeof f === 'object') return String(f.name ?? f.path ?? f.file ?? '').trim();
      return '';
    })
    .filter(Boolean);
}

function normalizeFileName(name) {
  return String(name ?? '').trim().toLowerCase();
}

function extractFileNames(value) {
  if (value == null) return [];
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed ? [trimmed] : [];
  }
  if (Array.isArray(value)) {
    return value.flatMap((item) => extractFileNames(item));
  }
  if (typeof value === 'object') {
    const candidate = value.name ?? value.file ?? value.path ?? value.filename ?? value.target_file;
    if (typeof candidate === 'string' && candidate.trim()) return [candidate.trim()];
  }
  return [];
}

function collectStepFileNamesByPaths(steps, paths) {
  const names = new Set();
  for (const step of steps ?? []) {
    for (const path of paths) {
      let value = step;
      for (const key of path) {
        if (value == null) break;
        value = value[key];
      }
      for (const fileName of extractFileNames(value)) {
        const normalized = normalizeFileName(fileName);
        if (normalized) names.add(normalized);
      }
    }
  }
  return names;
}

function buildDefinitiveObjectiveFileSet(steps) {
  const explicitObjectivePaths = [
    ['objective_files'],
    ['target_files'],
    ['red_target_files'],
    ['red_objective_files'],
    ['objective', 'files'],
    ['objectives', 'files'],
    ['red_objective', 'files'],
    ['red_objectives', 'files'],
    ['payload', 'objective_files'],
    ['payload', 'target_files'],
    ['payload', 'red_target_files'],
    ['payload', 'red_objective_files'],
    ['payload', 'objective', 'files'],
    ['payload', 'red_objective', 'files'],
    ['live', 'objective_files'],
    ['live', 'target_files'],
    ['replay', 'objective_files'],
    ['replay', 'target_files'],
  ];
  const explicitObjectives = collectStepFileNamesByPaths(steps, explicitObjectivePaths);
  if (explicitObjectives.size > 0) return explicitObjectives;

  const confirmedExfilTargetPaths = [
    ['confirmed_exfil_targets'],
    ['red_confirmed_targets'],
    ['red_exfil_targets'],
    ['exfil_target_files'],
    ['payload', 'confirmed_exfil_targets'],
    ['payload', 'red_confirmed_targets'],
    ['payload', 'red_exfil_targets'],
    ['payload', 'exfil_target_files'],
    ['metadata', 'confirmed_exfil_targets'],
    ['metadata', 'red_exfil_targets'],
    ['exfil_files'],
  ];
  return collectStepFileNamesByPaths(steps, confirmedExfilTargetPaths);
}

const BLUE_TRAP_ACTIONS = new Set([
  'place_honeypot',
  'plant_breadcrumb',
  'plant_temporal_decoy',
  'tamper_dead_drop',
  'plant_honeypot_poison',
]);

const RED_DEAD_DROP_ACTIONS = new Set([
  'write_dead_drop',
  'tamper_dead_drop',
]);

function actionToken(raw) {
  return String(raw ?? '')
    .toLowerCase()
    .trim()
    .replace(/[^\w\s]/g, ' ')
    .replace(/\s+/g, '_');
}

function parseNodeId(value) {
  if (value == null) return null;
  const direct = Number(value);
  if (Number.isFinite(direct)) return direct;
  const m = String(value).match(/(?:→\s*)?n(?:ode)?[_\s-]?(\d+)/i);
  return m ? Number(m[1]) : null;
}

function parseActionList(text) {
  if (!text) return [];
  return String(text)
    .toLowerCase()
    .split(/\s+/)
    .map((chunk) => chunk.replace(/×\d+$/i, '').replace(/[^\w]/g, ''))
    .filter(Boolean);
}

function shortAgentName(agentId, fallback = 'Agent') {
  const raw = String(agentId ?? '').trim();
  if (!raw) return fallback;
  return raw
    .replace(/^(red|blue|system)_/i, '')
    .replace(/_\d+$/i, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function inferTeam(agentId, fallback = 'red') {
  const raw = String(agentId ?? '').toLowerCase();
  if (raw.startsWith('blue_')) return 'blue';
  if (raw.startsWith('red_')) return 'red';
  return fallback;
}

function extractSpawnEventsFromStep(step) {
  if (!step) return [];
  const stepNo = Number(step.step ?? 0);
  const events = [];
  const sourceEntries = Array.isArray(step.all_agents) ? step.all_agents : [];

  for (const entry of sourceEntries) {
    const action = actionToken(entry?.action_type);
    if (action !== 'spawn_subagent') continue;
    const payload = entry?.payload ?? {};
    const agentId = String(entry?.agent_id ?? payload?.spawner_id ?? payload?.parent_agent_id ?? '');
    const team = inferTeam(agentId, 'red');
    const spawnedId = String(
      payload?.spawned_agent_id
      ?? payload?.subagent_id
      ?? payload?.child_agent_id
      ?? payload?.agent_id
      ?? `agent_${stepNo}`,
    );
    const work = String(
      payload?.task
      ?? payload?.work
      ?? payload?.objective
      ?? payload?.goal
      ?? entry?.detail
      ?? 'assigned task',
    ).trim();
    const fromNode = parseNodeId(
      payload?.source_node
      ?? payload?.from_node
      ?? payload?.origin_node
      ?? payload?.spawner_node
      ?? entry?.source_node
      ?? entry?.from_node
      ?? entry?.target_node,
    );
    const toNode = parseNodeId(
      payload?.target_node
      ?? payload?.node
      ?? payload?.spawn_node
      ?? payload?.to_node
      ?? entry?.target_node
      ?? entry?.node,
    );
    events.push({
      key: `spawn-${stepNo}-${agentId || 'agent'}-${spawnedId}`,
      step: stepNo,
      team,
      spawner: shortAgentName(agentId, team.toUpperCase()),
      spawned: shortAgentName(spawnedId, 'Agent'),
      work,
      fromNode,
      toNode,
    });
  }

  // Fallback for compact replay/live rows where spawn appears in red_action.
  const redAct = actionToken(step.red_action);
  if (redAct === 'spawn_subagent') {
    const node = parseNodeId(step.red_node ?? step.target_node ?? step.red_action);
    events.push({
      key: `spawn-${stepNo}-red-fallback-${node ?? 'x'}`,
      step: stepNo,
      team: 'red',
      spawner: 'Commander',
      spawned: 'Subagent',
      work: 'forward operation',
      fromNode: node,
      toNode: node,
    });
  }

  return events;
}

function extractOperationalMarkers(steps, redThoughts, blueThoughts) {
  const traps = [];
  const deadDrops = [];
  const seen = new Set();

  for (const s of steps ?? []) {
    const stepNo = Number(s?.step ?? 0);

    const redAct = actionToken(s?.red_action);
    const redNode = parseNodeId(s?.red_node ?? s?.target_node ?? s?.red_action);
    if (RED_DEAD_DROP_ACTIONS.has(redAct) && redNode != null) {
      const key = `r-${stepNo}-${redNode}-${redAct}`;
      if (!seen.has(key)) {
        seen.add(key);
        deadDrops.push({ nodeId: redNode, step: stepNo, action: redAct });
      }
    }

    const blueTokens = parseActionList(s?.blue_actions);
    const hasTrapAction = blueTokens.some((tok) => BLUE_TRAP_ACTIONS.has(tok));
    const blueNode = parseNodeId(s?.blue_node ?? s?.target_node);
    if (hasTrapAction && blueNode != null) {
      const key = `b-${stepNo}-${blueNode}`;
      if (!seen.has(key)) {
        seen.add(key);
        traps.push({ nodeId: blueNode, step: stepNo, action: blueTokens.find((tok) => BLUE_TRAP_ACTIONS.has(tok)) });
      }
    }
  }

  for (const t of blueThoughts ?? []) {
    const act = actionToken(t?.action_type);
    const node = parseNodeId(t?.target_node);
    if (!BLUE_TRAP_ACTIONS.has(act) || node == null) continue;
    const key = `tb-${t?.step ?? 'x'}-${node}-${act}`;
    if (seen.has(key)) continue;
    seen.add(key);
    traps.push({ nodeId: node, step: Number(t?.step ?? 0), action: act });
  }

  for (const t of redThoughts ?? []) {
    const act = actionToken(t?.action_type);
    const node = parseNodeId(t?.target_node);
    if (!RED_DEAD_DROP_ACTIONS.has(act) || node == null) continue;
    const key = `tr-${t?.step ?? 'x'}-${node}-${act}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deadDrops.push({ nodeId: node, step: Number(t?.step ?? 0), action: act });
  }

  traps.sort((a, b) => a.step - b.step);
  deadDrops.sort((a, b) => a.step - b.step);
  return {
    trapMarkers: traps.slice(-8),
    deadDropMarkers: deadDrops.slice(-8),
  };
}

// ─── Agent dot component ─────────────────────────────────────────
function AgentDot({ x, y, color, radius, label, pulseFast, filter, team = 'neutral', roleIndex = 0 }) {
  if (!x || !y) return null;
  const ringOpacity = team === 'red' ? 0.34 - pulseFast * 0.11 : 0.28 - pulseFast * 0.1;
  const ringRadius = radius + 9 + pulseFast * (team === 'red' ? 12 : 8);
  const accentSize = radius + 3 + roleIndex * 0.2;
  return (
    <g filter={filter}>
      <circle cx={x} cy={y} r={ringRadius}
        fill="none" stroke={color}
        strokeWidth={1.1} strokeOpacity={ringOpacity}
      />
      <circle cx={x} cy={y} r={radius + 4} fill={`${color}26`} />
      {team === 'blue' ? (
        <polygon
          points={[
            `${x},${y - accentSize}`,
            `${x + accentSize * 0.72},${y - accentSize * 0.38}`,
            `${x + accentSize * 0.72},${y + accentSize * 0.38}`,
            `${x},${y + accentSize}`,
            `${x - accentSize * 0.72},${y + accentSize * 0.38}`,
            `${x - accentSize * 0.72},${y - accentSize * 0.38}`,
          ].join(' ')}
          fill="none"
          stroke="rgba(120,180,255,0.75)"
          strokeWidth={1}
          strokeDasharray="3 4"
        />
      ) : (
        <rect
          x={x - accentSize * 0.72}
          y={y - accentSize * 0.72}
          width={accentSize * 1.44}
          height={accentSize * 1.44}
          fill="none"
          stroke="rgba(255,120,120,0.78)"
          strokeWidth={1}
          transform={`rotate(45 ${x} ${y})`}
        />
      )}
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
  const [hoveredAgent, setHoveredAgent] = useState(null);
  const [spawnBursts, setSpawnBursts] = useState([]);
  const [spawnNotices, setSpawnNotices] = useState([]);
  const seenSpawnRef = useRef(new Set());

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
  const visibleTrail = useMemo(() => trail.slice(-10), [trail]);
  const trailSegments = useMemo(() => {
    const segs = [];
    for (let i = 1; i < visibleTrail.length; i += 1) {
      const a = positions[visibleTrail[i - 1]];
      const b = positions[visibleTrail[i]];
      if (a && b) segs.push({ a, b, idx: i - 1 });
    }
    return segs;
  }, [visibleTrail, positions]);

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
  const infoNodeId = selected ?? hovered;
  const { trapMarkers, deadDropMarkers } = useMemo(
    () => extractOperationalMarkers(steps, redThoughts, blueThoughts),
    [steps, redThoughts, blueThoughts],
  );
  const definitiveObjectiveFiles = useMemo(
    () => buildDefinitiveObjectiveFileSet(steps),
    [steps],
  );
  const nowMs = performance.now();

  useEffect(() => {
    if (!steps?.length) return;
    const latestStep = steps[steps.length - 1];
    const spawns = extractSpawnEventsFromStep(latestStep);
    if (spawns.length === 0) return;

    const fresh = spawns.filter((evt) => {
      if (seenSpawnRef.current.has(evt.key)) return false;
      seenSpawnRef.current.add(evt.key);
      return true;
    });
    if (fresh.length === 0) return;

    const bornAt = performance.now();
    setSpawnBursts((prev) => {
      const keep = prev.filter((evt) => bornAt - evt.bornAt < 2600);
      return [
        ...keep,
        ...fresh.map((evt) => ({ ...evt, id: `${evt.key}-${bornAt}`, bornAt, duration: 1650 })),
      ].slice(-18);
    });

    setSpawnNotices((prev) => {
      const keep = prev.filter((evt) => bornAt <= evt.expiresAt);
      const additions = fresh.map((evt, idx) => ({
        id: `${evt.key}-notice-${bornAt}-${idx}`,
        team: evt.team,
        text: `${evt.team.toUpperCase()} spawned ${evt.spawned} for ${evt.work}`,
        bornAt,
        expiresAt: bornAt + 3600,
      }));
      return [...additions, ...keep].slice(0, 4);
    });
  }, [steps]);

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
          {Z.map((z, i) => (
            <radialGradient key={`zone-grad-${i}`} id={`zoneGrad${i}`} cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor={z.color} stopOpacity="0.22" />
              <stop offset="45%" stopColor={z.color} stopOpacity="0.10" />
              <stop offset="75%" stopColor={z.color} stopOpacity="0.04" />
              <stop offset="100%" stopColor="#080c16" stopOpacity="0" />
            </radialGradient>
          ))}
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
          {/* Match app shell color so battle view feels seamless */}
          <rect x={0} y={0} width={W} height={H} fill="#131a24" />

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

          {/* ── Distinct zone gradients for better separation ── */}
          {Z.map((z, i) => (
            <ellipse
              key={`zone-sep-${i}`}
              cx={z.cx}
              cy={z.cy}
              rx={z.rx + 120}
              ry={z.ry + 95}
              fill={`url(#zoneGrad${i})`}
              opacity={0.95}
            />
          ))}

          {/* Soft zone boundary rings */}
          {Z.map((z, i) => (
            <ellipse
              key={`zone-ring-${i}`}
              cx={z.cx}
              cy={z.cy}
              rx={z.rx + 88}
              ry={z.ry + 72}
              fill="none"
              stroke={z.color}
              strokeOpacity={0.14}
              strokeWidth={1.2}
              strokeDasharray="7 10"
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

          {/* ── RED movement path (generated progressively as it moves) ── */}
          {trailSegments.map((seg, i) => {
            const newest = i === trailSegments.length - 1;
            const recency = (i + 1) / Math.max(1, trailSegments.length);
            return (
              <g key={`trail-seg-${seg.idx}`}>
                <line
                  x1={seg.a.x}
                  y1={seg.a.y}
                  x2={seg.b.x}
                  y2={seg.b.y}
                  stroke="rgba(255,68,68,0.28)"
                  strokeWidth={6}
                  strokeLinecap="round"
                  opacity={0.35 + recency * 0.35}
                />
                <line
                  x1={seg.a.x}
                  y1={seg.a.y}
                  x2={seg.b.x}
                  y2={seg.b.y}
                  stroke={newest ? 'rgba(255,120,120,0.98)' : 'rgba(255,92,92,0.72)'}
                  strokeWidth={newest ? 3.2 : 2.2}
                  strokeLinecap="round"
                  strokeDasharray={newest ? '4 4' : '6 7'}
                  opacity={0.5 + recency * 0.45}
                  filter={newest ? 'url(#redGlow)' : undefined}
                />
              </g>
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
            const nodeFileNames = getNodeFiles(node).map(normalizeFileName).filter(Boolean);
            const hasTargetFiles = definitiveObjectiveFiles.size > 0
              && nodeFileNames.some((fileName) => definitiveObjectiveFiles.has(fileName));

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

                {hasTargetFiles && (
                  <>
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r={r + 5}
                      fill="none"
                      stroke="rgba(255, 215, 64, 0.62)"
                      strokeWidth={1}
                      strokeDasharray="2 4"
                    />
                    <g>
                      <circle
                        cx={p.x + r - 2}
                        cy={p.y - r + 2}
                        r={6}
                        fill="rgba(20,26,42,0.95)"
                        stroke="rgba(255,215,64,0.8)"
                        strokeWidth={1}
                      />
                      <text
                        x={p.x + r - 2}
                        y={p.y - r + 4}
                        textAnchor="middle"
                        fontSize={7}
                        fontFamily="'JetBrains Mono', monospace"
                        fill="#ffd740"
                      >
                        F
                      </text>
                    </g>
                  </>
                )}

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

          {/* ── BLUE trap markers (recent only) ── */}
          {trapMarkers.map((mk, i) => {
            const p = positions[mk.nodeId];
            if (!p) return null;
            const recency = (i + 1) / Math.max(1, trapMarkers.length);
            const opacity = 0.22 + recency * 0.62;
            const rr = 18 + (1 - recency) * 7;
            return (
              <g key={`trap-${mk.nodeId}-${mk.step}-${i}`} style={{ pointerEvents: 'none' }}>
                <circle
                  cx={p.x}
                  cy={p.y}
                  r={rr + pulseSlow * 3}
                  fill="none"
                  stroke={`rgba(96,170,255,${Math.max(0.14, opacity * 0.42)})`}
                  strokeWidth={1.2}
                  strokeDasharray="3 5"
                />
                <polygon
                  points={`${p.x},${p.y - 8} ${p.x + 8},${p.y} ${p.x},${p.y + 8} ${p.x - 8},${p.y}`}
                  fill={`rgba(88,158,255,${Math.max(0.2, opacity * 0.44)})`}
                  stroke={`rgba(130,198,255,${Math.min(0.95, opacity)})`}
                  strokeWidth={1.2}
                />
                <text
                  x={p.x}
                  y={p.y + 2.8}
                  textAnchor="middle"
                  fontSize={7}
                  fontWeight="700"
                  fontFamily="'JetBrains Mono', monospace"
                  fill={`rgba(220,240,255,${Math.min(0.95, opacity)})`}
                >
                  T
                </text>
              </g>
            );
          })}

          {/* ── RED dead-drop markers (recent only) ── */}
          {deadDropMarkers.map((mk, i) => {
            const p = positions[mk.nodeId];
            if (!p) return null;
            const recency = (i + 1) / Math.max(1, deadDropMarkers.length);
            const opacity = 0.24 + recency * 0.64;
            const sz = 9 + recency * 1.8;
            return (
              <g key={`dd-${mk.nodeId}-${mk.step}-${i}`} style={{ pointerEvents: 'none' }}>
                <circle
                  cx={p.x}
                  cy={p.y}
                  r={17 + pulseFast * 2.4}
                  fill="none"
                  stroke={`rgba(255,94,94,${Math.max(0.16, opacity * 0.48)})`}
                  strokeWidth={1.15}
                  strokeDasharray="2 4"
                />
                <rect
                  x={p.x - sz}
                  y={p.y - sz}
                  width={sz * 2}
                  height={sz * 2}
                  fill={`rgba(255,64,64,${Math.max(0.2, opacity * 0.38)})`}
                  stroke={`rgba(255,142,142,${Math.min(0.95, opacity)})`}
                  strokeWidth={1.2}
                  transform={`rotate(45 ${p.x} ${p.y})`}
                />
                <text
                  x={p.x}
                  y={p.y + 3}
                  textAnchor="middle"
                  fontSize={6.8}
                  fontWeight="700"
                  fontFamily="'JetBrains Mono', monospace"
                  fill={`rgba(255,238,238,${Math.min(0.96, opacity)})`}
                >
                  DD
                </text>
              </g>
            );
          })}

          {/* ── Spawn burst FX (spawner -> spawned) ── */}
          {spawnBursts.map((evt) => {
            const age = nowMs - evt.bornAt;
            if (age < 0 || age > evt.duration) return null;
            const from = positions[evt.fromNode];
            const to = positions[evt.toNode];
            if (!from || !to) return null;
            const t = Math.min(1, Math.max(0, age / evt.duration));
            const accent = evt.team === 'blue' ? '88,166,255' : '255,92,92';
            const glowOpacity = (1 - t) * 0.85;
            const popScale = 0.35 + easeInOutCubic(t);
            const traceX = from.x + (to.x - from.x) * t;
            const traceY = from.y + (to.y - from.y) * t;
            return (
              <g key={evt.id} style={{ pointerEvents: 'none' }}>
                <line
                  x1={from.x}
                  y1={from.y}
                  x2={to.x}
                  y2={to.y}
                  stroke={`rgba(${accent},${0.12 + (1 - t) * 0.36})`}
                  strokeWidth={1.6}
                  strokeDasharray="4 6"
                />
                <circle
                  cx={traceX}
                  cy={traceY}
                  r={4.2}
                  fill={`rgba(${accent},${0.86 - t * 0.52})`}
                  filter={evt.team === 'blue' ? 'url(#blueGlow)' : 'url(#redGlow)'}
                />
                <circle
                  cx={to.x}
                  cy={to.y}
                  r={10 + popScale * 24}
                  fill={`rgba(${accent},${glowOpacity * 0.22})`}
                  stroke={`rgba(${accent},${glowOpacity * 0.95})`}
                  strokeWidth={1.5}
                />
                <text
                  x={to.x}
                  y={to.y - 22 - (1 - t) * 12}
                  textAnchor="middle"
                  fontSize={8}
                  fontFamily="'JetBrains Mono', monospace"
                  letterSpacing="0.06em"
                  fill={`rgba(230,240,255,${0.42 + glowOpacity * 0.56})`}
                >
                  + {evt.spawned}
                </text>
              </g>
            );
          })}

          {/* ── 4 BLUE agents ── */}
          {bluePositions.map((bp, i) => {
            if (isCenter(bp)) return null;
            const agentKey = `blue-${i}`;
            const agentId = `BLUE ${BLUE_LABELS[i]}`;
            return (
              <g
                key={agentKey}
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoveredAgent(agentKey)}
                onMouseLeave={() => setHoveredAgent(null)}
              >
                <AgentDot
                  x={bp.x} y={bp.y}
                  color="#4488ff"
                  radius={BLUE_RADII[i]}
                  label={BLUE_LABELS[i]}
                  pulseFast={pulseFast * (1 - i * 0.15)}
                  filter="url(#blueGlow)"
                  team="blue"
                  roleIndex={i}
                />
                {hoveredAgent === agentKey && (
                  <foreignObject
                    x={bp.x + 16}
                    y={bp.y - 30}
                    width={120}
                    height={30}
                    style={{ overflow: 'visible', pointerEvents: 'none' }}
                  >
                    <div style={{
                      background: 'rgba(10,16,28,0.92)',
                      border: '1px solid rgba(68,136,255,0.55)',
                      borderRadius: 7,
                      padding: '3px 8px',
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 10,
                      fontWeight: 700,
                      letterSpacing: '0.08em',
                      color: '#77a7ff',
                      boxShadow: '0 4px 16px rgba(0,0,0,0.45)',
                      display: 'inline-block',
                      whiteSpace: 'nowrap',
                    }}>
                      {agentId}
                    </div>
                  </foreignObject>
                )}
              </g>
            );
          })}

          {/* ── 4 RED agents ── */}
          {redPositions.map((rp, i) => {
            if (isCenter(rp)) return null;
            const agentKey = `red-${i}`;
            const agentId = `RED ${RED_LABELS[i]}`;
            return (
              <g
                key={agentKey}
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoveredAgent(agentKey)}
                onMouseLeave={() => setHoveredAgent(null)}
              >
                <AgentDot
                  x={rp.x} y={rp.y}
                  color={i === 0 ? '#ff2222' : '#ff4444'}
                  radius={RED_RADII[i]}
                  label={RED_LABELS[i]}
                  pulseFast={i === 0 ? pulseFast : pulseFast * (1 - i * 0.18)}
                  filter="url(#redGlow)"
                  team="red"
                  roleIndex={i}
                />
                {hoveredAgent === agentKey && (
                  <foreignObject
                    x={rp.x + 16}
                    y={rp.y - 30}
                    width={118}
                    height={30}
                    style={{ overflow: 'visible', pointerEvents: 'none' }}
                  >
                    <div style={{
                      background: 'rgba(24,10,10,0.92)',
                      border: '1px solid rgba(255,68,68,0.55)',
                      borderRadius: 7,
                      padding: '3px 8px',
                      fontFamily: "'JetBrains Mono', monospace",
                      fontSize: 10,
                      fontWeight: 700,
                      letterSpacing: '0.08em',
                      color: '#ff7575',
                      boxShadow: '0 4px 16px rgba(0,0,0,0.45)',
                      display: 'inline-block',
                      whiteSpace: 'nowrap',
                    }}>
                      {agentId}
                    </div>
                  </foreignObject>
                )}
              </g>
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
          {infoNodeId !== null && positions[infoNodeId] && (
            (() => {
              const n = nodeMap[infoNodeId];
              const p = positions[infoNodeId];
              const files = getNodeFiles(n);
              if (!n || files.length === 0) return null;
              const preview = files.slice(0, 4);
              const more = Math.max(0, files.length - preview.length);
              return (
                <foreignObject
                  x={p.x + 18}
                  y={p.y - 112}
                  width={230}
                  height={106}
                  style={{ overflow: 'visible', pointerEvents: 'none' }}
                >
                  <div
                    style={{
                      background: 'rgba(13,18,30,0.94)',
                      border: '1px solid rgba(255,215,64,0.45)',
                      borderRadius: 9,
                      padding: '7px 10px',
                      fontFamily: "'JetBrains Mono', monospace",
                      boxShadow: '0 4px 18px rgba(0,0,0,0.45)',
                    }}
                  >
                    <div
                      style={{
                        fontSize: 9,
                        fontWeight: 700,
                        letterSpacing: '0.08em',
                        textTransform: 'uppercase',
                        color: '#ffd740',
                        marginBottom: 5,
                      }}
                    >
                      Exfil files ({files.length})
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      {preview.map((file) => (
                        <div
                          key={file}
                          style={{
                            fontSize: 9.5,
                            color: 'rgba(230,236,246,0.86)',
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                          }}
                        >
                          - {file}
                        </div>
                      ))}
                      {more > 0 && (
                        <div style={{ fontSize: 8.5, color: 'rgba(255,215,64,0.75)' }}>
                          +{more} more files
                        </div>
                      )}
                    </div>
                  </div>
                </foreignObject>
              );
            })()
          )}

          {selected !== null && positions[selected] && (
            (() => {
              const n  = nodeMap[selected];
              const p  = positions[selected];
              const z  = Math.min(3, Number(n?.zone ?? 0));
              const zc = Z[z];
              const files = getNodeFiles(n);
              return (
                <foreignObject x={p.x + 24} y={p.y - 55} width={240} height={210}
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
                      {files.length > 0 && (
                        <span style={{ color: '#ffd740' }}>F Exfil Files: {files.length}</span>
                      )}
                    </div>
                    {files.length > 0 && (
                      <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 3 }}>
                        {files.slice(0, 6).map((file) => (
                          <div
                            key={file}
                            style={{
                              fontSize: 10,
                              color: 'rgba(226,232,242,0.86)',
                              whiteSpace: 'nowrap',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                            }}
                            title={file}
                          >
                            - {file}
                          </div>
                        ))}
                        {files.length > 6 && (
                          <div style={{ fontSize: 9, color: 'rgba(255,215,64,0.72)' }}>
                            +{files.length - 6} additional files
                          </div>
                        )}
                      </div>
                    )}
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

      {/* ── Spawn notifications ── */}
      <div style={{ position: 'absolute', top: 12, right: 14, display: 'flex', flexDirection: 'column', gap: 6, pointerEvents: 'none', zIndex: 55 }}>
        {spawnNotices
          .filter((note) => nowMs <= note.expiresAt)
          .map((note) => {
            const age = nowMs - note.bornAt;
            const life = Math.max(0, (note.expiresAt - nowMs) / Math.max(1, note.expiresAt - note.bornAt));
            const entering = Math.min(1, age / 260);
            const accent = note.team === 'blue' ? '#66a6ff' : '#ff6d6d';
            return (
              <div
                key={note.id}
                style={{
                  transform: `translateY(${(1 - entering) * -8}px) scale(${0.95 + entering * 0.05})`,
                  opacity: Math.min(1, entering) * (0.35 + life * 0.65),
                  background: 'rgba(8,14,26,0.92)',
                  border: `1px solid ${accent}66`,
                  borderRadius: 7,
                  padding: '4px 9px',
                  boxShadow: `0 2px 14px ${accent}30`,
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 9,
                  letterSpacing: '0.05em',
                  color: accent,
                  whiteSpace: 'nowrap',
                  maxWidth: 340,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
              >
                {note.text}
              </div>
            );
          })}
      </div>
    </div>
  );
}
