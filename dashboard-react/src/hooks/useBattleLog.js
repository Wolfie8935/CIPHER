import { useState, useEffect, useRef } from 'react';

export function useBattleLog(interval = 2000) {
  const [events, setEvents] = useState([]);
  const timerRef = useRef(null);

  useEffect(() => {
    let mounted = true;

    const fetch_ = async () => {
      try {
        const res = await fetch('/api/battle-log', { signal: AbortSignal.timeout(2000) });
        if (!res.ok) return;
        const data = await res.json();
        if (mounted && Array.isArray(data)) {
          setEvents(data);
        }
      } catch { /* network */ }
    };

    fetch_();
    timerRef.current = setInterval(fetch_, interval);
    return () => {
      mounted = false;
      clearInterval(timerRef.current);
    };
  }, [interval]);

  return events;
}
