import { useState, useEffect } from 'react';

export function useAgentStatus(interval = 2000) {
  const [status, setStatus] = useState(null);
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch('/api/agent-status', { signal: AbortSignal.timeout(1500) });
        if (res.ok) { const d = await res.json(); if (d && Object.keys(d).length > 0) setStatus(d); }
      } catch { /* ignore */ }
    };
    poll();
    const t = setInterval(poll, interval);
    return () => clearInterval(t);
  }, [interval]);
  return { status };
}
