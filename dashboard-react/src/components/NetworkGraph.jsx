import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

const ZONE_COLORS = { 0:'#6b7fa3', 1:'#00bcd4', 2:'#ffc107', 3:'#ff1744' };
const ZONE_GLOWS  = { 0:'rgba(107,127,163,0.4)', 1:'rgba(0,188,212,0.5)', 2:'rgba(255,193,7,0.5)', 3:'rgba(255,23,68,0.6)' };

function normalize(graph) {
  const baseNodes = (graph?.nodes ?? []).map(n => ({
    ...n,
    id:   Number(n.id   ?? n.node_id),
    zone: Number(n.zone ?? 0),
  }));

  const byZone = { 0:[], 1:[], 2:[], 3:[] };
  for (const n of baseNodes) {
    (byZone[n.zone] ?? byZone[0]).push(n);
  }
  const colX = { 0:-320, 1:-107, 2:107, 3:320 };
  for (const [z, nodes] of Object.entries(byZone)) {
    const cnt = nodes.length;
    nodes.forEach((n, i) => {
      const spread = cnt > 1 ? i / (cnt - 1) : 0.5;
      n.x = colX[z] ?? 0;
      n.y = -230 + spread * 460;
      n.fx = n.x;
      n.fy = n.y;
    });
  }

  const links = (graph?.edges ?? []).map(e =>
    Array.isArray(e)
      ? { source: Number(e[0]), target: Number(e[1]) }
      : { source: Number(e.source ?? e.src ?? e.from ?? e.u), target: Number(e.target ?? e.dst ?? e.to ?? e.v) }
  );
  return { nodes: baseNodes, links };
}

export default function LiveNetworkGraph({ graph, steps, agentStatus }) {
  const graphRef   = useRef(null);
  const [hoverNode, setHoverNode] = useState(null);
  const pulseRef   = useRef(0);
  const frameRef   = useRef(null);

  // Animate the canvas continuously for pulse effects
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let id;
    const loop = () => { setTick(t => t + 1); id = requestAnimationFrame(loop); };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, []);

  // Derive RED position from steps
  const currentNode = useMemo(() => {
    if (agentStatus?.agents) {
      for (const [aid, info] of Object.entries(agentStatus.agents)) {
        if (aid.startsWith('red_planner') && info.node != null) return Number(info.node);
      }
    }
    if (steps?.length > 0) {
      const latest = steps[steps.length - 1];
      const m = (latest.red_action ?? '').match(/→\s*n(\d+)/);
      if (m) return Number(m[1]);
    }
    return null;
  }, [agentStatus, steps, tick]);

  // Collect visited path
  const visitedNodes = useMemo(() => {
    const set = new Set();
    for (const s of (steps ?? [])) {
      const m = (s.red_action ?? '').match(/→\s*n(\d+)/);
      if (m) set.add(Number(m[1]));
    }
    return set;
  }, [steps]);

  // Trap node set (BLUE placements)
  const trapNodes = useMemo(() => {
    const set = new Set();
    for (const s of (steps ?? [])) {
      if ((s.blue_actions ?? '').includes('place_honeypot') && s.step != null) {
        // We don't know exact node from aggregated data; mark recent step nodes as potential traps
      }
    }
    return set;
  }, [steps]);

  // Recent exfil glow
  const hasExfil = steps?.some(s => (s.exfil_count ?? 0) > 0);

  const graphData = useMemo(() => normalize(graph), [graph]);

  const pt = (performance.now() / 1000);
  const pulse = 0.55 + Math.abs(Math.sin(pt * 2.2)) * 0.45;

  useEffect(() => {
    const timer = setTimeout(() => {
      if (!graphRef.current) return;
      graphRef.current.centerAt(0, 0, 0);
      graphRef.current.zoomToFit(0, 200);
      graphRef.current.zoom(0.82, 0);
    }, 120);
    return () => clearTimeout(timer);
  }, [graphData]);

  const drawNode = useCallback((node, ctx, globalScale) => {
    const id        = Number(node.id);
    const isHVT     = node.is_hvt;
    const isEntry   = node.is_entry;
    const isCurrent = id === currentNode;
    const isVisited = visitedNodes.has(id);
    const isTrap    = node.is_honeypot || trapNodes.has(id);
    const zoneColor = ZONE_COLORS[node.zone] ?? '#6b7fa3';
    const zoneGlow  = ZONE_GLOWS[node.zone]  ?? 'rgba(107,127,163,0.4)';

    const pt2   = performance.now() / 1000;
    const pulse2 = 0.55 + Math.abs(Math.sin(pt2 * 2.2)) * 0.45;

    const baseR = isCurrent ? 9.5 : isHVT ? 8 : isVisited ? 7 : 5;

    // HVT aura
    if (isHVT) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, baseR + 14, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,215,64,${0.08 + 0.12 * pulse2})`;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(node.x, node.y, baseR + 7, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,215,64,${0.35 + 0.3 * pulse2})`;
      ctx.lineWidth = 2 / globalScale;
      ctx.stroke();
    }

    // Current node pulse rings
    if (isCurrent) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, baseR + 16, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,23,68,${0.1 + 0.12 * pulse2})`;
      ctx.lineWidth = 3 / globalScale;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(node.x, node.y, baseR + 9, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,23,68,${0.35 + 0.3 * pulse2})`;
      ctx.lineWidth = 2.5 / globalScale;
      ctx.stroke();
    }

    // Trap/honeypot aura
    if (isTrap) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, baseR + 8, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(68,138,255,${0.12 + 0.1 * pulse2})`;
      ctx.fill();
    }

    // Node fill
    ctx.beginPath();
    ctx.arc(node.x, node.y, baseR, 0, Math.PI * 2);
    if (isCurrent) {
      ctx.fillStyle = '#ff2255';
    } else if (isHVT) {
      ctx.fillStyle = '#ffd740';
    } else if (isEntry) {
      ctx.fillStyle = '#69f0ae';
    } else if (isTrap) {
      ctx.fillStyle = '#1565c0';
    } else if (isVisited) {
      ctx.fillStyle = zoneColor;
    } else {
      ctx.fillStyle = `rgba(${node.zone === 3 ? '80,20,30' : node.zone === 2 ? '60,50,0' : node.zone === 1 ? '0,40,50' : '30,36,50'},0.8)`;
    }
    ctx.fill();

    // Zone border
    if (isVisited || isCurrent || isHVT) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, baseR + 1, 0, Math.PI * 2);
      ctx.strokeStyle = isCurrent ? 'rgba(255,90,110,0.9)' : isHVT ? 'rgba(255,215,64,0.8)' : zoneColor + '99';
      ctx.lineWidth = 1.5 / globalScale;
      ctx.stroke();
    }

    // Icons
    const fs = 12 / globalScale;
    if (isHVT)   { ctx.font = `${fs}px serif`; ctx.fillText('⭐', node.x - fs/2, node.y + fs/2 - 1); }
    if (isEntry) { ctx.font = `${fs}px serif`; ctx.fillText('🚪', node.x - fs/2, node.y + fs/2 - 1); }
    if (isTrap)  { ctx.font = `${fs * 0.9}px serif`; ctx.fillText('🪤', node.x - fs/2, node.y + fs/2 - 1); }

    // Label
    const labelFs = Math.max(7, 10 / globalScale);
    ctx.font      = `${labelFs}px 'JetBrains Mono', monospace`;
    ctx.fillStyle = isCurrent ? '#fff' : isVisited || isHVT ? '#cfd8f5' : 'rgba(180,195,220,0.4)';
    const label   = String(node.hostname ?? `n${id}`).substring(0, 12);
    ctx.fillText(label, node.x + baseR + 2, node.y + labelFs / 3);
  }, [currentNode, visitedNodes, trapNodes, tick]);

  return (
    <div style={{ position:'relative', flex:1, minHeight:0, borderRadius:10, overflow:'hidden', background:'#040d1c' }}>
      {/* Zone column headers */}
      {[['◌ PERIMETER','#6b7fa3'],['◉ GENERAL','#00bcd4'],['◈ SENSITIVE','#ffc107'],['◆ CRITICAL','#ff1744']].map(([lbl, col], i) => (
        <div key={i} style={{
          position:'absolute', top:8,
          left: `${10 + i * 24.5}%`,
          transform:'translateX(-50%)',
          fontFamily:'var(--font-mono)', fontSize:8.5, fontWeight:700,
          letterSpacing:'0.1em', color:col, opacity:0.75,
          pointerEvents:'none', zIndex:10,
          textShadow:`0 0 8px ${col}88`,
        }}>{lbl}</div>
      ))}

      <ForceGraph2D
        ref={graphRef}
        graphData={graphData}
        nodeLabel=""
        nodeRelSize={5}
        minZoom={0.28}
        maxZoom={4}
        d3VelocityDecay={0.95}
        cooldownTicks={3}
        backgroundColor="transparent"
        onEngineStop={() => graphRef.current?.zoomToFit(0, 200)}
        onNodeHover={n => setHoverNode(n ?? null)}
        linkWidth={l => l._visited ? 2.5 : 0.8}
        linkColor={l => l._visited ? 'rgba(255,80,100,0.55)' : 'rgba(100,120,160,0.2)'}
        nodeCanvasObject={drawNode}
        nodePointerAreaPaint={(node, color, ctx) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(node.x, node.y, 12, 0, Math.PI * 2);
          ctx.fill();
        }}
      />

      {/* Hover tooltip */}
      {hoverNode && (
        <div style={{
          position:'absolute', top:40, right:10,
          background:'rgba(8,12,28,0.95)',
          border:'1px solid rgba(255,255,255,0.1)',
          borderRadius:8, padding:'10px 14px',
          fontFamily:'var(--font-mono)', fontSize:11,
          color:'var(--text-primary)', zIndex:20, minWidth:180,
          backdropFilter:'blur(12px)',
        }}>
          <div style={{ fontWeight:700, marginBottom:6, color:'#dde3f5' }}>
            {hoverNode.hostname || `node-${hoverNode.id}`}
          </div>
          <div style={{ color:'var(--text-dim)', display:'flex', flexDirection:'column', gap:3 }}>
            <span>ID: <b style={{ color:'var(--text-primary)' }}>{hoverNode.id}</b></span>
            <span>Zone: <b style={{ color: ZONE_COLORS[hoverNode.zone] }}>{['Perimeter','General','Sensitive','Critical'][hoverNode.zone]}</b></span>
            {hoverNode.is_hvt     && <span style={{ color:'var(--gold)' }}>⭐ High-Value Target</span>}
            {hoverNode.is_entry   && <span style={{ color:'var(--green)' }}>🚪 Entry Point</span>}
            {hoverNode.is_honeypot&& <span style={{ color:'var(--blue-team)' }}>🪤 Honeypot</span>}
          </div>
        </div>
      )}

      {/* Legend */}
      <div style={{
        position:'absolute', bottom:8, left:10,
        background:'rgba(4,8,20,0.8)', borderRadius:6,
        border:'1px solid rgba(255,255,255,0.06)',
        padding:'5px 10px',
        fontFamily:'var(--font-mono)', fontSize:9.5,
        color:'rgba(180,195,220,0.55)',
        display:'flex', gap:14, pointerEvents:'none',
      }}>
        <span><span style={{ color:'#ff2255' }}>⬤</span> RED current</span>
        <span><span style={{ color:'#ffd740' }}>⭐</span> HVT</span>
        <span><span style={{ color:'#69f0ae' }}>🚪</span> Entry</span>
        <span>🪤 Honeypot</span>
      </div>

      {/* Exfil glow overlay */}
      {hasExfil && (
        <div style={{
          position:'absolute', inset:0, pointerEvents:'none',
          background:'radial-gradient(ellipse at center, rgba(255,23,68,0.08) 0%, transparent 70%)',
        }} />
      )}
    </div>
  );
}
