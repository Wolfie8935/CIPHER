import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

const zoneColors = {
  0: "#7a859f",
  1: "#4f9cff",
  2: "#f2b24c",
  3: "#ff5f6b"
};

const normalizeGraph = (graph) => {
  const baseNodes = (graph?.nodes || []).map((node) => ({
    ...node,
    id: Number(node.id ?? node.node_id),
    zone: Number(node.zone ?? 0)
  }));
  const zoneGroups = { 0: [], 1: [], 2: [], 3: [] };
  for (const node of baseNodes) {
    if (!zoneGroups[node.zone]) zoneGroups[node.zone] = [];
    zoneGroups[node.zone].push(node);
  }
  const columnX = { 0: -320, 1: -110, 2: 110, 3: 320 };
  for (const [zoneKey, nodes] of Object.entries(zoneGroups)) {
    const zone = Number(zoneKey);
    const count = nodes.length;
    nodes.forEach((node, idx) => {
      const spread = count > 1 ? idx / (count - 1) : 0.5;
      const y = -210 + spread * 420;
      node.x = columnX[zone] ?? 0;
      node.y = y;
      node.fx = node.x;
      node.fy = node.y;
    });
  }

  const links = (graph?.edges || []).map((edge) => {
    if (Array.isArray(edge)) return { source: Number(edge[0]), target: Number(edge[1]), type: "base" };
    return {
      source: Number(edge.source ?? edge.src ?? edge.from ?? edge.u),
      target: Number(edge.target ?? edge.dst ?? edge.to ?? edge.v),
      type: "base"
    };
  });
  return { nodes: baseNodes, links };
};

const toPathSegments = (path, type) => {
  if (!path || path.length < 2) return [];
  return path.slice(0, -1).map((source, index) => ({
    source,
    target: path[index + 1],
    type,
    index
  }));
};

function NetworkGraph({
  graph,
  currentStep,
  activePath,
  fullPath,
  trapEvents,
  deadDrops,
  honeypotNodes,
  honeypotEvents,
  forensicsPath
}) {
  const graphRef = useRef(null);
  const [hoverNode, setHoverNode] = useState(null);
  const [hoverLink, setHoverLink] = useState(null);

  const graphData = useMemo(() => {
    const base = normalizeGraph(graph);
    const redPathSegments = toPathSegments(activePath, "redPath");
    const forensicsSegments = toPathSegments(forensicsPath, "forensics");

    const falseTrailSegments = (trapEvents || [])
      .filter((event) => event.step <= currentStep && Number.isFinite(Number(event.node)))
      .map((event) => ({
        source: activePath[Math.min(Math.max(event.step - 1, 0), Math.max(activePath.length - 1, 0))],
        target: Number(event.node),
        type: "falseTrail"
      }))
      .filter((segment) => Number.isFinite(segment.source) && Number.isFinite(segment.target));

    return {
      nodes: base.nodes,
      links: [...base.links, ...redPathSegments, ...forensicsSegments, ...falseTrailSegments]
    };
  }, [graph, activePath, trapEvents, currentStep, forensicsPath]);

  const deadDropNodeSet = useMemo(
    () => new Set((deadDrops || []).filter((drop) => drop.step <= currentStep).map((drop) => Number(drop.node))),
    [deadDrops, currentStep]
  );

  const trapNodeSet = useMemo(
    () => new Set((trapEvents || []).filter((event) => event.step <= currentStep).map((event) => Number(event.node))),
    [trapEvents, currentStep]
  );

  const honeypotNodeSet = useMemo(() => new Set((honeypotNodes || []).map(Number)), [honeypotNodes]);
  const currentTrapNodes = useMemo(
    () => new Set((trapEvents || []).filter((event) => event.step === currentStep).map((event) => Number(event.node))),
    [trapEvents, currentStep]
  );
  const currentDropNodes = useMemo(
    () => new Set((deadDrops || []).filter((drop) => drop.step === currentStep).map((drop) => Number(drop.node))),
    [deadDrops, currentStep]
  );
  const currentHoneypotNodes = useMemo(
    () =>
      new Set((honeypotEvents || []).filter((event) => event.step === currentStep).map((event) => Number(event.node))),
    [honeypotEvents, currentStep]
  );
  const activeNode = activePath[activePath.length - 1];
  const pulseStrength = 0.6 + Math.abs(Math.sin(Date.now() / 260)) * 0.4;

  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (!graphRef.current) return;
      graphRef.current.centerAt(0, 0, 0);
      graphRef.current.zoomToFit(0, 220);
      graphRef.current.zoom(0.86, 0);
    }, 80);
    return () => window.clearTimeout(timer);
  }, [graphData]);

  return (
    <div className="relative h-[620px] w-full overflow-hidden rounded-lg bg-[#060d20]">
      <ForceGraph2D
        ref={graphRef}
        graphData={graphData}
        nodeLabel=""
        nodeRelSize={5}
        minZoom={0.32}
        maxZoom={3}
        d3VelocityDecay={0.9}
        cooldownTicks={2}
        onEngineStop={() => graphRef.current?.zoomToFit(0, 220)}
        onNodeHover={(node) => setHoverNode(node || null)}
        onLinkHover={(link) => setHoverLink(link || null)}
        linkWidth={(link) => {
          if (link.type === "redPath") return 3;
          if (link.type === "forensics") return 2;
          if (link.type === "falseTrail") return 1.5;
          return 1;
        }}
        linkColor={(link) => {
          if (link.type === "redPath") {
            const hue = 10 + Math.min(40, (link.index || 0) * 5);
            return `hsl(${hue}, 90%, 58%)`;
          }
          if (link.type === "forensics") return "#5aa3ff";
          if (link.type === "falseTrail") return "#ffd36e";
          return "rgba(138, 155, 190, 0.32)";
        }}
        linkLineDash={(link) => (link.type === "forensics" || link.type === "falseTrail" ? [6, 4] : null)}
        linkDirectionalArrowLength={(link) => (link.type === "falseTrail" ? 6 : 0)}
        linkDirectionalArrowRelPos={1}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const label = String(node.hostname || `N-${node.id}`);
          const fontSize = 12 / globalScale;
          const isVisited = activePath.includes(Number(node.id));
          const isFuture = (fullPath || []).includes(Number(node.id)) && !isVisited;
          const isCurrent = activeNode === Number(node.id);
          const radius = isCurrent ? 10 : isVisited ? 7 : 5.2;
          const nodeColor = zoneColors[node.zone] || "#6f7c95";

          if (honeypotNodeSet.has(Number(node.id))) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius + 5, 0, 2 * Math.PI, false);
            ctx.fillStyle = "rgba(255,70,84,0.2)";
            ctx.fill();
          }
          if (currentHoneypotNodes.has(Number(node.id))) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius + 10, 0, 2 * Math.PI, false);
            ctx.fillStyle = `rgba(255,80,94,${0.18 + 0.22 * pulseStrength})`;
            ctx.fill();
          }
          if (currentTrapNodes.has(Number(node.id)) || currentDropNodes.has(Number(node.id))) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius + 12, 0, 2 * Math.PI, false);
            ctx.fillStyle = `rgba(248,184,78,${0.16 + 0.22 * pulseStrength})`;
            ctx.fill();
          }

          ctx.beginPath();
          ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
          if (isCurrent) {
            ctx.fillStyle = "#ff5a66";
          } else if (isVisited) {
            ctx.fillStyle = nodeColor;
          } else if (isFuture) {
            ctx.fillStyle = "rgba(128,147,183,0.45)";
          } else {
            ctx.fillStyle = "rgba(92,108,140,0.22)";
          }
          ctx.fill();

          if (isCurrent) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius + 6, 0, 2 * Math.PI, false);
            ctx.strokeStyle = `rgba(255,100,112,${0.45 + 0.45 * pulseStrength})`;
            ctx.lineWidth = 3.5 / globalScale;
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius + 11, 0, 2 * Math.PI, false);
            ctx.strokeStyle = `rgba(255,100,112,${0.16 + 0.25 * pulseStrength})`;
            ctx.lineWidth = 4 / globalScale;
            ctx.stroke();
          } else if (isFuture) {
            ctx.setLineDash([4 / globalScale, 4 / globalScale]);
            ctx.lineWidth = 1.2 / globalScale;
            ctx.strokeStyle = "rgba(162,178,209,0.6)";
            ctx.stroke();
            ctx.setLineDash([]);
          }

          if (deadDropNodeSet.has(Number(node.id))) {
            ctx.font = `${14 / globalScale}px Sans-Serif`;
            ctx.fillText("📦", node.x + 5, node.y - 6);
          }
          if (trapNodeSet.has(Number(node.id))) {
            ctx.font = `${14 / globalScale}px Sans-Serif`;
            ctx.fillText("⚡", node.x + 5, node.y + 8);
          }

          ctx.font = `${fontSize}px Sans-Serif`;
          ctx.fillStyle = isVisited || isCurrent ? "#d9e3ff" : "rgba(188,205,235,0.55)";
          ctx.fillText(label, node.x + 8, node.y + 3);
        }}
      />
      <div className="pointer-events-none absolute top-3 left-3 rounded-md border border-cipher-border bg-slate-950/80 px-2 py-1 text-[11px] text-slate-300">
        Zone layout: Perimeter → General → Sensitive → Critical
      </div>
      {hoverNode ? (
        <div className="absolute right-3 top-3 w-56 rounded-md border border-cipher-border bg-slate-950/85 p-2 text-xs text-slate-200">
          <div className="font-semibold text-slate-100">Node Detail</div>
          <div className="mt-1">Host: {hoverNode.hostname || `node-${hoverNode.id}`}</div>
          <div>ID: {hoverNode.id}</div>
          <div>Zone: {hoverNode.zone}</div>
          <div className="mt-1 text-slate-400">Hover any node/edge to inspect context.</div>
        </div>
      ) : null}
      {hoverLink ? (
        <div className="absolute right-3 top-36 w-56 rounded-md border border-cipher-border bg-slate-950/85 p-2 text-xs text-slate-200">
          <div className="font-semibold text-slate-100">Link Detail</div>
          <div>
            {String(hoverLink.source?.id ?? hoverLink.source)} → {String(hoverLink.target?.id ?? hoverLink.target)}
          </div>
          <div className="capitalize text-slate-300">
            {hoverLink.type === "base" ? "Network edge" : hoverLink.type}
          </div>
        </div>
      ) : null}
      <div className="pointer-events-none absolute bottom-3 left-3 rounded-md border border-cipher-border bg-slate-950/70 px-2 py-1 text-[11px] text-slate-300">
        <span className="mr-3">🔴 active path</span>
        <span className="mr-3">⬤ current node</span>
        <span className="mr-3">⚡ trap</span>
        <span className="mr-3">📦 dead drop</span>
        <span>🔵 forensics (dashed)</span>
      </div>
    </div>
  );
}

export default NetworkGraph;
