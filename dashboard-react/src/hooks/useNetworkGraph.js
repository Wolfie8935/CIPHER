import { useState, useEffect } from 'react';

function buildDemoGraph() {
  const ZONE_SIZES = [14, 16, 12, 8];
  const PREFIXES   = ['gw','rtr','fw','prx','ldap','dns','mail','web','api','db','fs','bkp','mon','vpn','ntp','syslog'];
  const nodes = [], edges = [];
  let id = 0;
  const byZone = {};
  for (let z = 0; z < 4; z++) {
    byZone[z] = [];
    for (let i = 0; i < ZONE_SIZES[z]; i++) {
      const pfx = PREFIXES[(id) % PREFIXES.length];
      nodes.push({ id, hostname: `${pfx}-z${z}-${String(i).padStart(2,'0')}`, zone: z, is_entry: z === 0 && i === 0, is_hvt: z === 3 && i === 0, is_honeypot: z === 2 && i === 3 });
      byZone[z].push(id);
      id++;
    }
  }
  for (let z = 0; z < 4; z++) {
    const zn = byZone[z];
    for (let i = 0; i < zn.length - 1; i++) edges.push({ source: zn[i], target: zn[i+1] });
    if (zn.length > 2) edges.push({ source: zn[0], target: zn[zn.length - 1] });
  }
  for (let z = 0; z < 3; z++) {
    const from = byZone[z], to = byZone[z + 1];
    for (let i = 0; i < Math.min(4, from.length); i++) {
      edges.push({ source: from[Math.floor(i * from.length / 4)], target: to[Math.floor(i * to.length / 4)] });
    }
  }
  return { nodes, edges };
}

const DEMO_GRAPH = buildDemoGraph();

export function useNetworkGraph() {
  const [graph,  setGraph]  = useState(DEMO_GRAPH);
  const [isLive, setIsLive] = useState(false);

  useEffect(() => {
    const tryFetch = async () => {
      try {
        const res = await fetch('/api/network-graph', { signal: AbortSignal.timeout(2000) });
        if (!res.ok) return;
        const data = await res.json();
        if (data?.nodes?.length > 0) { setGraph(data); setIsLive(true); }
      } catch { /* use demo */ }
    };
    tryFetch();
    const t = setInterval(tryFetch, 30000);
    return () => clearInterval(t);
  }, []);

  return { graph, isLive };
}
