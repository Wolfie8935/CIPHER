import { useState, useEffect, useRef } from 'react';
import GameMap          from './components/GameMap';
import SpeedControl     from './components/SpeedControl';
import StatsHUD         from './components/StatsHUD';
import DrawerPanel      from './components/DrawerPanel';
import EpisodeSelector  from './components/EpisodeSelector';
import ExfilAlertOverlay from './components/ExfilAlertOverlay';
import { useLivePolling }   from './hooks/useLivePolling';
import { useThoughts }      from './hooks/useThoughts';
import { useNetworkGraph }  from './hooks/useNetworkGraph';
import { useAgentStatus }   from './hooks/useAgentStatus';
import { useEpisodeReplay } from './hooks/useEpisodeReplay';

const MODE_LABELS = {
  live:   { text: 'LIVE',   color: '#ff4444' },
  hybrid: { text: 'HYBRID', color: '#fa8c16' },
  stub:   { text: 'STUB',   color: 'rgba(160,180,220,0.5)' },
  demo:   { text: 'DEMO',   color: '#ffd740' },
  replay: { text: 'REPLAY', color: '#4488ff' },
};

const ZONE_COLORS = {
  Perimeter: '#00e5ff', General: '#69f0ae',
  Sensitive: '#ffd740', 'Critical/HVT': '#ff6b6b',
};

export default function App() {
  const [selectedEpisode, setSelectedEpisode] = useState(null);
  const [speed,      setSpeed]      = useState(1);
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [showExfil,  setShowExfil]  = useState(false);
  const [zoneBreach, setZoneBreach] = useState(null);

  // ── Live mode hooks ─────────────────────────────────────────────
  const live    = useLivePolling(2000);
  const { thoughts, redThoughts, blueThoughts } = useThoughts(2000);
  const { graph: liveGraph }  = useNetworkGraph();
  const { status: agentStatus } = useAgentStatus(2000);

  // ── Episode replay hook ─────────────────────────────────────────
  const isEpisode = selectedEpisode && selectedEpisode !== 'live';
  const replay = useEpisodeReplay(isEpisode ? selectedEpisode : null, speed);

  // ── Derive active data ──────────────────────────────────────────
  const steps   = isEpisode ? replay.steps   : live.steps;
  const latest  = isEpisode ? replay.latest  : live.latest;
  const graph   = isEpisode ? replay.graph   : liveGraph;

  const prevExfilRef = useRef(0);
  const prevZoneRef  = useRef(null);

  // Zone breach banner
  useEffect(() => {
    if (!latest) return;
    const RANK = { Perimeter: 0, General: 1, Sensitive: 2, 'Critical/HVT': 3 };
    const curr = latest.zone;
    const prev = prevZoneRef.current;
    if (prev && RANK[curr] > RANK[prev]) {
      setZoneBreach(`${curr.toUpperCase()} ZONE BREACH`);
      setTimeout(() => setZoneBreach(null), 4000);
    }
    prevZoneRef.current = curr;
  }, [latest?.zone]);

  // Exfil overlay
  useEffect(() => {
    const exfil = latest?.exfil_count ?? 0;
    if (exfil > prevExfilRef.current) { setShowExfil(true); prevExfilRef.current = exfil; }
  }, [latest?.exfil_count]);

  // Mode indicator
  const modeRaw = isEpisode ? 'replay'
    : (latest?.run_id?.split('_')[0] ?? (live.isDemoMode ? 'demo' : 'stub'));
  const modeConf = MODE_LABELS[modeRaw] ?? MODE_LABELS.demo;

  // ── Show selector ────────────────────────────────────────────────
  if (selectedEpisode === null) {
    return <EpisodeSelector onSelect={setSelectedEpisode} />;
  }

  // Progress percent for replay
  const progressPct = replay.totalSteps > 0
    ? (replay.currentIdx / replay.totalSteps) * 100
    : 0;

  const topBarStyle = {
    fontFamily: 'var(--mono)',
    fontSize: 10,
    color: 'var(--text-mute)',
    letterSpacing: '0.08em',
  };

  return (
    <div className="war-room">

      {/* ── Top bar ── */}
      <div className="top-bar">

        {/* Back */}
        <button
          onClick={() => { setSelectedEpisode(null); prevZoneRef.current = null; prevExfilRef.current = 0; }}
          style={{
            padding: '4px 11px',
            background: 'transparent',
            border: '1px solid var(--border)',
            borderRadius: 7,
            fontFamily: 'var(--mono)', fontSize: 9.5, fontWeight: 700,
            letterSpacing: '0.1em', color: 'var(--text-mute)',
            cursor: 'pointer', transition: 'all 0.15s', whiteSpace: 'nowrap',
          }}
          onMouseEnter={e => { e.currentTarget.style.color = 'var(--text)'; e.currentTarget.style.borderColor = 'var(--z0)'; }}
          onMouseLeave={e => { e.currentTarget.style.color = 'var(--text-mute)'; e.currentTarget.style.borderColor = 'var(--border)'; }}
        >
          ← Episodes
        </button>

        {/* Brand */}
        <div style={{ fontFamily: 'var(--mono)', fontWeight: 700, fontSize: 14, letterSpacing: '0.20em', whiteSpace: 'nowrap' }}>
          <span style={{ color: '#ff4444' }}>C</span>
          <span style={{ color: 'var(--text)' }}>IPHER</span>
          <span style={{ color: 'var(--text-mute)', fontWeight: 400, fontSize: 9, marginLeft: 7 }}>WAR ROOM</span>
        </div>

        {/* Mode badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{
            width: 7, height: 7, borderRadius: '50%',
            background: modeConf.color,
            boxShadow: `0 0 8px ${modeConf.color}`,
            animation: modeRaw === 'live' ? 'livePulse 1.5s ease-in-out infinite' : 'none',
          }} />
          <span style={{ fontFamily: 'var(--mono)', fontSize: 9, fontWeight: 700, letterSpacing: '0.15em', color: modeConf.color }}>
            {modeConf.text}
          </span>
        </div>

        {/* Episode / step */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={topBarStyle}>
            EP <b style={{ color: 'var(--text)', fontSize: 13 }}>{latest?.episode ?? 1}</b>
          </span>
          <span style={topBarStyle}>
            STEP <b style={{ color: 'var(--text)', fontSize: 13 }}>{latest?.step ?? 0}</b>
            <span style={{ color: 'var(--text-mute)' }}>/{latest?.max_steps ?? 30}</span>
          </span>
        </div>

        {/* Replay counter */}
        {isEpisode && (
          <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text-mute)' }}>
            {replay.currentIdx} / {replay.totalSteps}
            {replay.isComplete && (
              <span style={{ color: '#69f0ae', marginLeft: 8, fontWeight: 700 }}>✓ COMPLETE</span>
            )}
          </div>
        )}

        <div style={{ flex: 1 }} />

        {/* Zone */}
        {latest?.zone && (
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 9.5, letterSpacing: '0.14em', fontWeight: 700,
            color: ZONE_COLORS[latest.zone] ?? 'var(--z0)',
            padding: '2px 9px',
            background: `${ZONE_COLORS[latest.zone] ?? '#00e5ff'}12`,
            borderRadius: 5,
            border: `1px solid ${ZONE_COLORS[latest.zone] ?? '#00e5ff'}30`,
          }}>
            {latest.zone.toUpperCase()}
          </div>
        )}

        {/* Red action */}
        {latest?.red_action && latest.red_action !== 'waiting' && (
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 9.5, color: 'rgba(255,68,68,0.75)',
            letterSpacing: '0.06em', maxWidth: 190,
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            🔴 {latest.red_action}
          </div>
        )}

        {/* Demo tip */}
        {!isEpisode && live.isDemoMode && (
          <div style={{ fontFamily: 'var(--mono)', fontSize: 8.5, color: 'rgba(255,215,64,0.45)', letterSpacing: '0.1em' }}>
            ◉ DEMO — run python main.py --live
          </div>
        )}
      </div>

      {/* ── Side drawer ── */}
      <DrawerPanel
        thoughts={thoughts}
        steps={steps}
        isOpen={drawerOpen}
        onToggle={() => setDrawerOpen(o => !o)}
      />

      {/* ── Game map ── */}
      <div style={{ position: 'absolute', top: 46, bottom: 54, left: 0, right: 0 }}>
        <GameMap
          graph={graph}
          steps={steps}
          agentStatus={isEpisode ? null : agentStatus}
          redThoughts={redThoughts}
          blueThoughts={blueThoughts}
          speed={speed}
        />
      </div>

      {/* ── Stats HUD ── */}
      <StatsHUD latest={latest} redThoughts={redThoughts} blueThoughts={blueThoughts} />

      {/* ── Bottom bar ── */}
      <div className="bottom-bar">
        <span style={{ fontFamily: 'var(--mono)', fontSize: 8.5, color: 'var(--text-mute)', letterSpacing: '0.12em', flexShrink: 0 }}>
          CIPHER WAR ROOM
        </span>

        <SpeedControl
          speed={speed}
          onChange={setSpeed}
          isPlaying={replay.isPlaying}
          onPlay={replay.play}
          onPause={replay.pause}
          hasReplay={isEpisode && replay.hasData}
        />

        {/* Progress scrubber */}
        {isEpisode && replay.hasData && (
          <div
            className="progress-bar"
            onClick={e => {
              const rect = e.currentTarget.getBoundingClientRect();
              const frac = (e.clientX - rect.left) / rect.width;
              replay.seekTo(Math.round(frac * replay.totalSteps));
            }}
            title="Click to seek"
          >
            <div
              className="progress-fill"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        )}

        <div style={{ flex: 1 }} />

        <div style={{ fontFamily: 'var(--mono)', fontSize: 8.5, color: 'var(--text-mute)', letterSpacing: '0.06em', flexShrink: 0 }}>
          {steps.length} steps · {thoughts.length} thoughts
        </div>
      </div>

      {/* ── Zone breach banner ── */}
      {zoneBreach && (
        <div style={{
          position: 'fixed', top: 58, left: '50%', transform: 'translateX(-50%)',
          padding: '8px 40px',
          background: 'rgba(16,6,6,0.92)',
          border: '1px solid rgba(255,68,68,0.4)',
          borderRadius: 8,
          fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 700,
          letterSpacing: '0.22em', color: '#ff4444',
          zIndex: 800, whiteSpace: 'nowrap',
          animation: 'glitchIn 0.3s ease-out',
          boxShadow: '0 4px 24px rgba(255,68,68,0.25)',
        }}>
          🚨 {zoneBreach}
        </div>
      )}

      {/* ── Exfil overlay ── */}
      {showExfil && (
        <ExfilAlertOverlay
          files={latest?.exfil_files ?? []}
          onDismiss={() => setShowExfil(false)}
        />
      )}
    </div>
  );
}
