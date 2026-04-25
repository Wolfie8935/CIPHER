export default function OutcomeScreen({ result, onDismiss }) {
  if (!result) return null;

  const termReason = result.terminal_reason ?? 'max_steps';
  let winner, winColor, winLabel, msg;
  if (['exfil_success','exfiltration_complete','exfil_complete'].includes(termReason)) {
    winner = 'RED';  winColor = 'red';  winLabel = '🔴 RED TEAM';
    msg = 'Classified file successfully exfiltrated. Mission accomplished.';
  } else if (termReason === 'detected') {
    winner = 'BLUE'; winColor = 'blue'; winLabel = '🔵 BLUE TEAM';
    msg = 'RED agent detected and neutralized. Network secured.';
  } else if (termReason === 'aborted') {
    winner = 'DRAW'; winColor = 'blue'; winLabel = '⚪ TACTICAL WITHDRAWAL';
    msg = 'RED aborted voluntarily due to rising suspicion.';
  } else {
    winner = 'BLUE'; winColor = 'blue'; winLabel = '🔵 BLUE TEAM';
    msg = 'RED failed to complete exfiltration within mission time.';
  }

  const stats = [
    { label:'STEPS TAKEN',   val: result.step ?? '--' },
    { label:'SUSPICION',     val: `${((result.red_suspicion_score ?? 0) * 100).toFixed(0)}%` },
    { label:'DETECTION',     val: `${((result.blue_detection_confidence ?? 0) * 100).toFixed(0)}%` },
    { label:'FILES EXFIL',   val: (result.red_exfiltrated_files ?? []).length },
  ];

  return (
    <div className="outcome-overlay">
      <div style={{ fontFamily:'var(--font-mono)', fontSize:10, letterSpacing:'0.2em', color:'var(--text-muted)' }}>
        EPISODE CONCLUDED
      </div>
      <div className={`outcome-winner ${winColor}`}>{winLabel}</div>
      <div style={{ width:1, height:40, background:'rgba(255,255,255,0.1)' }} />
      <div className="outcome-msg">{msg}</div>
      <div className="outcome-stats">
        {stats.map(s => (
          <div key={s.label} className="outcome-stat">
            <div className="outcome-stat-val">{s.val}</div>
            <div className="outcome-stat-label">{s.label}</div>
          </div>
        ))}
      </div>
      <button className="outcome-btn" onClick={onDismiss} style={{ marginTop:16 }}>
        ↩ CONTINUE WATCHING
      </button>
    </div>
  );
}
