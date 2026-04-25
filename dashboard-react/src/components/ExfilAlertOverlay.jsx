import { useEffect, useState } from 'react';

export default function ExfilAlertOverlay({ files = [], onDismiss }) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const t = setTimeout(() => { setVisible(false); onDismiss?.(); }, 3200);
    return () => clearTimeout(t);
  }, [onDismiss]);

  if (!visible) return null;

  return (
    <div className="exfil-overlay" onClick={() => { setVisible(false); onDismiss?.(); }}>
      {/* Glitch lines */}
      <div style={{
        position:'absolute', inset:0, pointerEvents:'none',
        background:'repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(255,23,68,0.04) 3px,rgba(255,23,68,0.04) 4px)',
      }} />
      <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:18, zIndex:1 }}>
        <div className="exfil-skull">💀</div>
        <div className="exfil-title">FILE EXFILTRATED</div>
        <div className="exfil-file">
          Target acquired: <span>{files[0] ?? 'classified_ops_directive.pdf'}</span>
        </div>
        <div style={{
          fontFamily:'var(--font-mono)', fontSize:11, color:'rgba(255,255,255,0.25)',
          marginTop:8, letterSpacing:'0.1em',
        }}>
          CLICK TO DISMISS
        </div>
      </div>
      {/* Corner decorations */}
      {['top-left','top-right','bottom-left','bottom-right'].map(pos => (
        <div key={pos} style={{
          position:'absolute',
          ...(pos.includes('top')    ? { top:24 }    : { bottom:24 }),
          ...(pos.includes('left')   ? { left:24 }   : { right:24 }),
          width:32, height:32,
          borderTop:    pos.includes('top')    ? '2px solid rgba(255,23,68,0.6)' : 'none',
          borderBottom: pos.includes('bottom') ? '2px solid rgba(255,23,68,0.6)' : 'none',
          borderLeft:   pos.includes('left')   ? '2px solid rgba(255,23,68,0.6)' : 'none',
          borderRight:  pos.includes('right')  ? '2px solid rgba(255,23,68,0.6)' : 'none',
        }} />
      ))}
    </div>
  );
}
