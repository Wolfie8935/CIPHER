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
      <div className="exfil-popup" onClick={(e) => e.stopPropagation()}>
        <div className="exfil-skull">💀</div>
        <div className="exfil-title">FILE EXFILTRATED</div>
        <div className="exfil-file">
          File extracted: <span>{files[0] ?? 'classified_ops_directive.pdf'}</span>
        </div>
      </div>
    </div>
  );
}
