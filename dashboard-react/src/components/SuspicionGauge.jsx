import { useEffect, useRef, useState } from 'react';

function polarToXY(cx, cy, r, angleDeg) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function arcPath(cx, cy, r, startDeg, endDeg) {
  const s = polarToXY(cx, cy, r, startDeg);
  const e = polarToXY(cx, cy, r, endDeg);
  const large = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`;
}

export default function SuspicionGauge({ label, value = 0, color = 'red' }) {
  const [display, setDisplay] = useState(value);
  const animRef = useRef(null);
  const prevRef = useRef(value);

  useEffect(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    const from = prevRef.current;
    const to   = value;
    const dur  = 600;
    const start = performance.now();
    const step  = (now) => {
      const t   = Math.min((now - start) / dur, 1);
      const ease = 1 - Math.pow(1 - t, 3);
      setDisplay(from + (to - from) * ease);
      if (t < 1) animRef.current = requestAnimationFrame(step);
      else prevRef.current = to;
    };
    animRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animRef.current);
  }, [value]);

  const isCritical = display > 0.8;
  const isWarn     = display > 0.4;

  const teamColor = color === 'red'
    ? (isCritical ? '#ff1744' : isWarn ? '#ff9800' : '#69f0ae')
    : (display > 0.7 ? '#ff1744' : display > 0.4 ? '#ff9800' : '#448aff');

  const teamGlow  = `rgba(${color === 'red'
    ? (isCritical ? '255,23,68' : isWarn ? '255,152,0' : '105,240,174')
    : (display > 0.7 ? '255,23,68' : display > 0.4 ? '255,152,0' : '68,138,255')
  },0.5)`;

  const cx = 90, cy = 90, r = 68;
  const startAngle = -220, sweepTotal = 260;
  const arcEnd = startAngle + sweepTotal * display;

  const needleAngle = startAngle + sweepTotal * display;
  const nStart = polarToXY(cx, cy, 12, needleAngle);
  const nEnd   = polarToXY(cx, cy, r - 8, needleAngle);

  return (
    <div style={{
      display:'flex', flexDirection:'column', alignItems:'center', padding:'10px 8px 4px',
      ...(isCritical ? { animation:'suspicionCritical 1.2s ease-in-out infinite' } : {}),
    }}>
      <svg width={180} height={110} viewBox="0 0 180 110" style={{ overflow:'visible' }}>
        <defs>
          <linearGradient id={`arc-grad-${color}`} gradientUnits="userSpaceOnUse" x1="20" y1="90" x2="160" y2="90">
            <stop offset="0%"   stopColor="#69f0ae" />
            <stop offset="40%"  stopColor="#ffca28" />
            <stop offset="75%"  stopColor="#ff9800" />
            <stop offset="100%" stopColor="#ff1744" />
          </linearGradient>
          <filter id={`glow-${color}`}>
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* Track */}
        <path
          d={arcPath(cx, cy, r, startAngle, startAngle + sweepTotal)}
          fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={10} strokeLinecap="round"
        />

        {/* Active arc */}
        {display > 0.005 && (
          <path
            d={arcPath(cx, cy, r, startAngle, arcEnd)}
            fill="none"
            stroke={`url(#arc-grad-${color})`}
            strokeWidth={10}
            strokeLinecap="round"
            filter={`url(#glow-${color})`}
            style={{ transition: 'none' }}
          />
        )}

        {/* Needle */}
        <line
          x1={nStart.x} y1={nStart.y}
          x2={nEnd.x}   y2={nEnd.y}
          stroke={teamColor}
          strokeWidth={2.5}
          strokeLinecap="round"
          filter={`url(#glow-${color})`}
          style={{ transition: 'none' }}
        />
        <circle cx={cx} cy={cy} r={5} fill={teamColor} style={{ filter:`drop-shadow(0 0 6px ${teamColor})` }} />

        {/* Center value */}
        <text x={cx} y={cy + 26}
          textAnchor="middle"
          fill={teamColor}
          fontSize={20}
          fontFamily="var(--font-mono)"
          fontWeight="700"
          style={{ filter:`drop-shadow(0 0 8px ${teamColor})` }}
        >
          {(display * 100).toFixed(1)}%
        </text>
      </svg>

      <div style={{
        fontFamily: 'var(--font-mono)', fontSize: 9, fontWeight: 700,
        letterSpacing: '0.16em', textTransform: 'uppercase',
        color: teamColor, marginTop: -4,
        textShadow: isCritical ? `0 0 10px ${teamColor}` : 'none',
      }}>
        {isCritical ? '⚠ CRITICAL' : label}
      </div>
    </div>
  );
}
