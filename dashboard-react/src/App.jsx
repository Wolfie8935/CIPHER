import { useState, useEffect, useRef } from 'react';
import GameMap          from './components/GameMap';
import SpeedControl     from './components/SpeedControl';
import StatsHUD         from './components/StatsHUD';
import DrawerPanel      from './components/DrawerPanel';
import ExfilAlertOverlay from './components/ExfilAlertOverlay';
import { useLivePolling }   from './hooks/useLivePolling';
import { useThoughts }      from './hooks/useThoughts';
import { useNetworkGraph }  from './hooks/useNetworkGraph';
import { useAgentStatus }   from './hooks/useAgentStatus';
import { useEpisodeReplay } from './hooks/useEpisodeReplay';

const MODE_LABELS = {
  idle:   { text: 'IDLE',   color: 'rgba(160,180,220,0.5)' },
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
  const [episodes, setEpisodes] = useState([]);
  const [episodesLoading, setEpisodesLoading] = useState(true);
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
  const hasSource = selectedEpisode !== null;
  const isEpisode = hasSource && selectedEpisode !== 'live';
  const replay = useEpisodeReplay(isEpisode ? selectedEpisode : null, speed);
  const selectedEpisodeMeta = episodes.find(ep => ep.name === selectedEpisode) ?? null;

  // ── Derive active data ──────────────────────────────────────────
  const steps   = !hasSource ? [] : (isEpisode ? replay.steps   : live.steps);
  const latest  = !hasSource ? null : (isEpisode ? replay.latest  : live.latest);
  const graph   = isEpisode ? replay.graph : liveGraph;
  const visibleThoughts = hasSource ? thoughts : [];
  const visibleRedThoughts = hasSource ? redThoughts : [];
  const visibleBlueThoughts = hasSource ? blueThoughts : [];

  const prevExfilRef = useRef(0);
  const prevZoneRef  = useRef(null);

  useEffect(() => {
    const fetchEpisodes = () => {
      fetch('/api/episodes')
        .then(r => r.json())
        .then(data => {
          const normalized = Array.isArray(data)
            ? data.map(item => (
              typeof item === 'string'
                ? { name: item, winner: 'UNKNOWN', terminal_reason: 'unknown' }
                : {
                    name: item?.name ?? '',
                    winner: item?.winner ?? 'UNKNOWN',
                    terminal_reason: item?.terminal_reason ?? 'unknown',
                  }
            )).filter(item => item.name)
            : [];
          setEpisodes(normalized);
          setEpisodesLoading(false);
        })
        .catch(() => setEpisodesLoading(false));
    };

    fetchEpisodes();
    const intervalId = setInterval(fetchEpisodes, 3000);
    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    if (selectedEpisode && selectedEpisode !== 'live' && !episodes.some(ep => ep.name === selectedEpisode)) {
      setSelectedEpisode('live');
    }
  }, [episodes, selectedEpisode]);

  const winnerConf = (() => {
    if (!isEpisode) return null;
    const winner = replay.outcome?.winner ?? selectedEpisodeMeta?.winner ?? 'UNKNOWN';
    if (winner === 'RED') return { text: 'WINNER RED', color: '#ff6b6b', bg: 'rgba(255,68,68,0.14)' };
    if (winner === 'BLUE') return { text: 'WINNER BLUE', color: '#7eb3ff', bg: 'rgba(68,136,255,0.14)' };
    if (winner === 'DRAW') return { text: 'DRAW', color: '#d6e1f3', bg: 'rgba(190,205,235,0.12)' };
    return { text: 'WINNER ?', color: 'rgba(180,195,220,0.9)', bg: 'rgba(160,180,220,0.10)' };
  })();

  const winnerCard = (() => {
    const winnerFromTerminalReason = (reason) => {
      if (!reason) return null;
      const normalized = String(reason).toLowerCase().replace(/[\s-]+/g, '_');
      if (['exfil_success', 'exfiltration_complete', 'exfil_complete'].includes(normalized)) return 'RED';
      if (normalized === 'aborted') return 'DRAW';
      if (
        normalized.includes('detected')
        || normalized.includes('timeout')
        || normalized.includes('failure')
        || normalized === 'max_steps'
      ) return 'BLUE';
      return null;
    };

    const terminalReason = isEpisode
      ? (replay.outcome?.terminalReason ?? selectedEpisodeMeta?.terminal_reason ?? '')
      : (
          latest?.terminal_reason
          ?? latest?.terminalReason
          ?? latest?.outcome?.terminal_reason
          ?? latest?.outcome?.terminalReason
          ?? ''
        );

    const liveStatus = String(latest?.status ?? latest?.run_status ?? '').toLowerCase();
    const hasLiveTerminalEvidence = Boolean(
      terminalReason
      || latest?.is_terminal
      || latest?.terminal
      || latest?.done
      || ['terminated', 'complete', 'completed', 'finished', 'stopped'].includes(liveStatus)
    );
    const hasReplayTerminalEvidence = Boolean(replay.isComplete);
    const isTerminated = isEpisode ? hasReplayTerminalEvidence : hasLiveTerminalEvidence;

    if (!isTerminated) return { status: 'PENDING', winner: 'PENDING' };

    const derivedWinner = winnerFromTerminalReason(terminalReason)
      ?? (isEpisode ? (replay.outcome?.winner ?? selectedEpisodeMeta?.winner ?? null) : null);
    const normalizedWinner = String(derivedWinner ?? '').toUpperCase();
    if (normalizedWinner === 'RED') return { status: 'FINAL', winner: 'RED' };
    if (normalizedWinner === 'BLUE') return { status: 'FINAL', winner: 'BLUE' };
    if (normalizedWinner === 'DRAW') return { status: 'FINAL', winner: 'DRAW' };
    return { status: 'PENDING', winner: 'PENDING' };
  })();

  const completionSignals = {
    replayComplete: Boolean(isEpisode && replay.isComplete),
    liveTerminalEvidence: Boolean(
      !isEpisode && (
        latest?.terminal_reason
        || latest?.terminalReason
        || latest?.outcome?.terminal_reason
        || latest?.outcome?.terminalReason
        || latest?.is_terminal
        || latest?.terminal
        || latest?.done
        || ['terminated', 'complete', 'completed', 'finished', 'stopped'].includes(
          String(latest?.status ?? latest?.run_status ?? '').toLowerCase()
        )
      )
    ),
  };

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
  const modeRaw = !hasSource ? 'idle' : isEpisode ? 'replay'
    : (latest?.run_id?.split('_')[0] ?? (live.isDemoMode ? 'demo' : 'stub'));
  const modeConf = MODE_LABELS[modeRaw] ?? MODE_LABELS.demo;

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

        {/* Episode selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
          <span style={{
            fontFamily: 'var(--mono)', fontSize: 8.5, fontWeight: 700,
            letterSpacing: '0.16em', color: 'var(--text-mute)', textTransform: 'uppercase',
          }}>
            Source ({episodes.length})
          </span>
          <select
            value={selectedEpisode ?? ''}
            onChange={e => {
              setSelectedEpisode(e.target.value === '' ? null : e.target.value);
              prevZoneRef.current = null;
              prevExfilRef.current = 0;
            }}
            style={{
              background: 'rgba(20, 28, 40, 0.9)',
              border: '1px solid var(--border)',
              borderRadius: 7,
              color: 'var(--text)',
              padding: '4px 9px',
              fontFamily: 'var(--mono)',
              fontSize: 9.5,
              fontWeight: 700,
              letterSpacing: '0.08em',
              minWidth: 128,
              cursor: 'pointer',
            }}
            title="Choose live mode or an episode replay"
          >
            <option value="">— SELECT SOURCE —</option>
            <option value="live">● LIVE FEED</option>
            {episodes.length === 0 && !episodesLoading && (
              <option value="" disabled>
                no episode traces found
              </option>
            )}
            {episodes.map((ep, idx) => (
              <option key={ep.name} value={ep.name}>
                {`EP ${String(idx + 1).padStart(2, '0')} · ${ep.winner} · ${ep.name.replace('.json', '')}`}
              </option>
            ))}
          </select>
          {episodesLoading && (
            <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: 'var(--text-mute)', letterSpacing: '0.08em' }}>
              loading…
            </span>
          )}
        </div>

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

        {winnerConf && (
          <div style={{
            fontFamily: 'var(--mono)',
            fontSize: 8.5,
            fontWeight: 700,
            letterSpacing: '0.12em',
            color: winnerConf.color,
            background: winnerConf.bg,
            border: `1px solid ${winnerConf.color}45`,
            borderRadius: 5,
            padding: '2px 8px',
          }}>
            {winnerConf.text}
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

      </div>

      {/* ── Side drawer ── */}
      <DrawerPanel
        thoughts={visibleThoughts}
        steps={steps}
        isOpen={drawerOpen}
        onToggle={() => setDrawerOpen(o => !o)}
      />

      {/* ── Game map ── */}
      <div style={{ position: 'absolute', top: 46, bottom: 54, left: 0, right: 0 }}>
        <GameMap
          graph={graph}
          steps={steps}
          agentStatus={!hasSource || isEpisode ? null : agentStatus}
          redThoughts={visibleRedThoughts}
          blueThoughts={visibleBlueThoughts}
          speed={speed}
        />
      </div>

      {/* ── Stats HUD ── */}
      <StatsHUD
        latest={latest}
        winnerCard={winnerCard}
        completionSignals={completionSignals}
      />

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
          {steps.length} steps · {visibleThoughts.length} thoughts
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
