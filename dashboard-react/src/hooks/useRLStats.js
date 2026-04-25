import { useState, useEffect, useRef } from 'react';

export function useRLStats(interval = 5000) {
  const [stats, setStats] = useState(null);
  const timerRef = useRef(null);

  useEffect(() => {
    let mounted = true;

    const fetch_ = async () => {
      try {
        const res = await fetch('/api/rl-stats', { signal: AbortSignal.timeout(3000) });
        if (!res.ok) return;
        const data = await res.json();
        if (mounted) setStats(data);
      } catch { /* network */ }
    };

    fetch_();
    timerRef.current = setInterval(fetch_, interval);
    return () => {
      mounted = false;
      clearInterval(timerRef.current);
    };
  }, [interval]);

  return stats;
}
