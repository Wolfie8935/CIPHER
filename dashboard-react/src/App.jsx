import { useState, useEffect, useRef } from 'react';
import GameMap             from './components/GameMap';
import SpeedControl        from './components/SpeedControl';
import StatsHUD            from './components/StatsHUD';
import MindsPanel          from './components/MindsPanel';
import OversightPanel      from './components/OversightPanel';
import RewardTracker       from './components/RewardTracker';
import RLMetricsPanel      from './components/RLMetricsPanel';
import HistoryPanel        from './components/HistoryPanel';
import AnalyticsPanel      from './components/AnalyticsPanel';
import LorePanel           from './components/LorePanel';
import ArchitecturePanel   from './components/ArchitecturePanel';
import ForensicsPanel      from './components/ForensicsPanel';
import LiveLogsPanel         from './components/LiveLogsPanel';
import { useLivePolling }  from './hooks/useLivePolling';
import { useThoughts }     from './hooks/useThoughts';
import { useNetworkGraph } from './hooks/useNetworkGraph';
import { useAgentStatus }  from './hooks/useAgentStatus';
import { useEpisodeReplay }from './hooks/useEpisodeReplay';

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

const RIGHT_TABS = [
  { id: 'minds',     label: '💭 MINDS' },
  { id: 'battle',    label: '🔥 BATTLE' },
  { id: 'live-logs', label: '🧾 LIVE LOGS' },
  { id: 'rewards',   label: '💰 REWARDS' },
  { id: 'learning',  label: '📈 LEARNING' },
  { id: 'oversight', label: '👁 OVERSIGHT' },
  { id: 'history',   label: '📚 HISTORY' },
  { id: 'analytics', label: '⭐ ANALYTICS' },
  { id: 'lore',      label: '📰 LORE' },
  { id: 'architecture', label: '🏛 ARCHITECTURE' },
  { id: 'forensics', label: '🔍 FORENSICS' },
];

const RIGHT_TAB_IDS = new Set(RIGHT_TABS.map((t) => t.id));

function actionToken(raw) {
  return String(raw ?? '')
    .toLowerCase()
    .trim()
    .replace(/[^\w\s]/g, ' ')
    .replace(/\s+/g, '_');
}

const IDLE_ACTIONS = ['stand_down', 'wait', 'idle'];

function isIdleAction(act) {
  return IDLE_ACTIONS.includes(actionToken(act));
}

/** v2 runs often emit only `{team}_commander_*` in all_agents — count as live fleet when acting. */
function commanderLiveFromAgentsList(agents, team) {
  const prefix = `${team}_commander`;
  if (!Array.isArray(agents)) return 0;
  for (const agent of agents) {
    const id = String(agent?.agent_id ?? '').trim().toLowerCase();
    if (!id.startsWith(prefix)) continue;
    if (!isIdleAction(agent?.action_type)) return 1;
  }
  return 0;
}

function computeCommanderAgentLifecycle(steps, team, thoughts = []) {
  const commanderPrefix = `${team}_commander`;
  const spawnedIds = new Set();
  const latestAgents = Array.isArray(steps?.[steps.length - 1]?.all_agents)
    ? steps[steps.length - 1].all_agents
    : [];
  const latestLiveIds = new Set();
  let explicitSpawnEvents = 0;

  for (const step of steps ?? []) {
    const agents = Array.isArray(step?.all_agents) ? step.all_agents : [];
    for (const agent of agents) {
      const id = String(agent?.agent_id ?? '').trim();
      const act = actionToken(agent?.action_type);
      if (!id || !id.startsWith(`${team}_`)) continue;

      // Commander spawn actions should count immediately even before subagent emits telemetry.
      if (id.startsWith(commanderPrefix) && act === 'spawn_subagent') {
        explicitSpawnEvents += 1;
        continue;
      }

      if (id.startsWith(commanderPrefix)) continue;
      spawnedIds.add(id);
      if (act === 'spawn_subagent') explicitSpawnEvents += 1;
    }

    // Additional real-time fallback from summarized step actions.
    const stepActionRaw = team === 'red' ? step?.red_action : step?.blue_actions;
    const stepAction = actionToken(stepActionRaw);
    if (stepAction.includes('spawn_subagent')) explicitSpawnEvents += 1;
  }

  // Fallback path: derive lifecycle from thought/action telemetry when all_agents is unavailable.
  const latestActionByAgent = new Map();
  for (const t of thoughts ?? []) {
    const id = String(t?.agent_id ?? '').trim();
    const act = actionToken(t?.action_type);
    if (!id || !id.startsWith(`${team}_`)) continue;

    // Count commander spawn actions instead of dropping them.
    if (id.startsWith(commanderPrefix) && act === 'spawn_subagent') {
      explicitSpawnEvents += 1;
      continue;
    }

    if (id.startsWith(commanderPrefix)) continue;
    spawnedIds.add(id);
    if (act === 'spawn_subagent') explicitSpawnEvents += 1;
    latestActionByAgent.set(id, act);
  }

  for (const agent of latestAgents) {
    const id = String(agent?.agent_id ?? '').trim();
    if (!id || !id.startsWith(`${team}_`) || id.startsWith(commanderPrefix)) continue;
    const act = actionToken(agent?.action_type);
    if (!isIdleAction(act)) latestLiveIds.add(id);
  }

  // If latest step has no all_agents payload, infer live agents from most recent thought/action.
  if (latestAgents.length === 0 && latestActionByAgent.size > 0) {
    for (const [id, act] of latestActionByAgent.entries()) {
      if (!isIdleAction(act)) latestLiveIds.add(id);
    }
  }

  const cmdLive = commanderLiveFromAgentsList(latestAgents, team);

  const spawned = Math.max(spawnedIds.size, explicitSpawnEvents, cmdLive);
  const live = Math.max(
    cmdLive,
    latestLiveIds.size,
    Math.min(
      spawnedIds.size,
      latestAgents.filter((agent) => {
        const id = String(agent?.agent_id ?? '');
        return id.startsWith(`${team}_`) && !id.startsWith(commanderPrefix);
      }).length,
    ),
  );
  const spawnedOut = Math.max(spawned, live);
  const expired = Math.max(0, spawnedOut - live);
  return {
    spawned: spawnedOut,
    live,
    expired,
    hasData: spawnedOut > 0 || live > 0 || explicitSpawnEvents > 0 || cmdLive > 0,
  };
}

/** Reconcile HUD with `logs/agent_status.json` (live). Subagent keys are team-prefixed; commanders count when active. */
function mergeLifecycleWithAgentStatus(derived, agents, team) {
  if (!derived || !agents || typeof agents !== 'object') return derived;
  const commanderPrefix = `${team}_commander`;
  let liveFromStatus = 0;
  let commanderLiveFromStatus = 0;
  let sawSubagent = false;
  for (const [id, info] of Object.entries(agents)) {
    if (!id.startsWith(`${team}_`)) continue;
    const act = actionToken(info?.action);
    if (id.startsWith(commanderPrefix)) {
      if (!isIdleAction(act)) commanderLiveFromStatus = 1;
      continue;
    }
    sawSubagent = true;
    if (!isIdleAction(act)) liveFromStatus += 1;
  }
  const statusLive = liveFromStatus + commanderLiveFromStatus;
  if (!sawSubagent && statusLive === 0) return derived;
  const live = Math.max(derived.live, statusLive);
  const spawned = Math.max(derived.spawned, live);
  const expired = Math.max(0, spawned - live);
  return {
    ...derived,
    spawned,
    live,
    expired,
    hasData: true,
  };
}

export default function App() {
  const [selectedEpisode, setSelectedEpisode] = useState(null);
  const [episodes, setEpisodes]       = useState([]);
  const [episodesLoading, setEpisodesLoading] = useState(true);
  const [speed,      setSpeed]        = useState(1);
  const [rightTab,   setRightTab]     = useState('battle');
  const [zoneBreach, setZoneBreach]   = useState(null);
  const [bannerDismissed, setBannerDismissed] = useState(false);

  useEffect(() => {
    if (!RIGHT_TAB_IDS.has(rightTab)) {
      setRightTab('battle');
    }
  }, [rightTab]);

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

  // ── Active data ─────────────────────────────────────────────────
  const steps   = !hasSource ? [] : (isEpisode ? replay.steps   : live.steps);
  const latest  = !hasSource ? null : (isEpisode ? replay.latest  : live.latest);
  const graph   = isEpisode ? replay.graph : liveGraph;
  const visibleThoughts     = hasSource ? thoughts     : [];
  const visibleRedThoughts  = hasSource ? redThoughts  : [];
  const visibleBlueThoughts = hasSource ? blueThoughts : [];

  const prevZoneRef  = useRef(null);

  // ── Fetch episode list ──────────────────────────────────────────
  useEffect(() => {
    const fetchEpisodes = () => {
      fetch('/api/episodes')
        .then(r => r.json())
        .then(data => {
          const normalized = Array.isArray(data)
            ? data.map(item => typeof item === 'string'
              ? { name: item, winner: 'UNKNOWN', terminal_reason: 'unknown' }
              : { name: item?.name ?? '', winner: item?.winner ?? 'UNKNOWN', terminal_reason: item?.terminal_reason ?? 'unknown' }
            ).filter(item => item.name)
            : [];
          setEpisodes(normalized);
          setEpisodesLoading(false);
        })
        .catch(() => setEpisodesLoading(false));
    };
    fetchEpisodes();
    const id = setInterval(fetchEpisodes, 3000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (selectedEpisode && selectedEpisode !== 'live' && !episodes.some(ep => ep.name === selectedEpisode)) {
      setSelectedEpisode('live');
    }
  }, [episodes, selectedEpisode]);

  // ── Winner helpers ──────────────────────────────────────────────
  const winnerConf = (() => {
    if (!isEpisode) return null;
    const winner = replay.outcome?.winner ?? selectedEpisodeMeta?.winner ?? 'UNKNOWN';
    if (winner === 'RED')  return { text: 'WINNER RED',  color: '#ff6b6b', bg: 'rgba(255,68,68,0.14)' };
    if (winner === 'BLUE') return { text: 'WINNER BLUE', color: '#7eb3ff', bg: 'rgba(68,136,255,0.14)' };
    if (winner === 'DRAW') return { text: 'DRAW',        color: '#d6e1f3', bg: 'rgba(190,205,235,0.12)' };
    return { text: 'WINNER ?', color: 'rgba(180,195,220,0.9)', bg: 'rgba(160,180,220,0.10)' };
  })();

  const winnerCard = (() => {
    const winnerFromTerminalReason = (reason) => {
      if (!reason) return null;
      const n = String(reason).toLowerCase().replace(/[\s-]+/g, '_');
      if (['exfil_success', 'exfiltration_complete', 'exfil_complete'].includes(n)) return 'RED';
      if (n === 'aborted') return 'DRAW';
      if (n.includes('detected') || n.includes('timeout') || n.includes('failure') || n === 'max_steps') return 'BLUE';
      return null;
    };
    const terminalReason = isEpisode
      ? (replay.outcome?.terminalReason ?? selectedEpisodeMeta?.terminal_reason ?? '')
      : (latest?.terminal_reason ?? latest?.terminalReason ?? '');
    const liveStatus = String(latest?.status ?? latest?.run_status ?? '').toLowerCase();
    const isTerminated = isEpisode
      ? Boolean(replay.isComplete)
      : Boolean(terminalReason || latest?.is_terminal || latest?.terminal || latest?.done || ['terminated','complete','completed','finished','stopped'].includes(liveStatus));
    if (!isTerminated) return { status: 'PENDING', winner: 'PENDING' };
    const derivedWinner = winnerFromTerminalReason(terminalReason) ?? (isEpisode ? (replay.outcome?.winner ?? selectedEpisodeMeta?.winner ?? null) : null);
    const w = String(derivedWinner ?? '').toUpperCase();
    if (w === 'RED')  return { status: 'FINAL', winner: 'RED' };
    if (w === 'BLUE') return { status: 'FINAL', winner: 'BLUE' };
    if (w === 'DRAW') return { status: 'FINAL', winner: 'DRAW' };
    return { status: 'PENDING', winner: 'PENDING' };
  })();

  const completionSignals = {
    replayComplete:     Boolean(isEpisode && replay.isComplete),
    liveTerminalEvidence: Boolean(!isEpisode && (latest?.terminal_reason || latest?.terminalReason || latest?.is_terminal || latest?.terminal || latest?.done || ['terminated','complete','completed','finished','stopped'].includes(String(latest?.status ?? latest?.run_status ?? '').toLowerCase()))),
  };
  const showWinnerBannerBase = winnerCard?.status === 'FINAL'
    && (completionSignals.replayComplete || completionSignals.liveTerminalEvidence);
  const showWinnerBanner = showWinnerBannerBase && !bannerDismissed;
  const thoughtStreamRed = isEpisode ? [] : visibleRedThoughts;
  const thoughtStreamBlue = isEpisode ? [] : visibleBlueThoughts;
  const agentsSnapshot = !isEpisode && agentStatus && typeof agentStatus === 'object'
    ? agentStatus.agents
    : null;
  const commanderLifecycle = {
    red: mergeLifecycleWithAgentStatus(
      computeCommanderAgentLifecycle(steps, 'red', thoughtStreamRed),
      agentsSnapshot,
      'red',
    ),
    blue: mergeLifecycleWithAgentStatus(
      computeCommanderAgentLifecycle(steps, 'blue', thoughtStreamBlue),
      agentsSnapshot,
      'blue',
    ),
  };
  const winnerBanner = winnerCard?.winner === 'RED'
    ? { text: 'RED WINS', icon: '🔴', color: '#ff6b6b', bg: 'rgba(32,10,10,0.94)', border: 'rgba(255,98,98,0.52)' }
    : winnerCard?.winner === 'BLUE'
      ? { text: 'BLUE WINS', icon: '🔵', color: '#7eb3ff', bg: 'rgba(8,16,36,0.94)', border: 'rgba(126,179,255,0.52)' }
      : { text: 'DRAW', icon: '⚪', color: '#d6e1f3', bg: 'rgba(18,22,34,0.94)', border: 'rgba(214,225,243,0.40)' };

  // Auto-dismiss winner banner after 5s; reset when episode changes
  useEffect(() => { setBannerDismissed(false); }, [selectedEpisode]);
  useEffect(() => {
    if (!showWinnerBannerBase) return;
    const t = setTimeout(() => setBannerDismissed(true), 5000);
    return () => clearTimeout(t);
  }, [showWinnerBannerBase]);

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

  // Mode indicator
  const modeRaw  = !hasSource ? 'idle' : isEpisode ? 'replay' : (latest?.run_id?.split('_')[0] ?? (live.isDemoMode ? 'demo' : 'stub'));
  const modeConf = MODE_LABELS[modeRaw] ?? MODE_LABELS.demo;

  const progressPct = replay.totalSteps > 0 ? (replay.currentIdx / replay.totalSteps) * 100 : 0;

  const isBattleView = rightTab === 'battle';

  const selectedSourceFullLabel = (() => {
    if (selectedEpisode == null) return '— SELECT SOURCE —';
    if (selectedEpisode === 'live') return '● LIVE FEED';
    const idx = episodes.findIndex((ep) => ep.name === selectedEpisode);
    if (idx >= 0) {
      const ep = episodes[idx];
      return `EP ${String(idx + 1).padStart(2, '0')} · ${ep.winner} · ${String(ep.name).replace(/\.json$/, '')}`;
    }
    return String(selectedEpisode);
  })();

  return (
    <div className="war-room">

      {/* ── Top bar ── */}
      <div className="top-bar">
        <div className="top-bar-left">
          <div className="top-bar-source">
            <span className="top-bar-source-label">Source ({episodes.length})</span>
            <select
              className="top-bar-select"
              title={selectedSourceFullLabel}
              value={selectedEpisode ?? ''}
              onChange={(e) => {
                setSelectedEpisode(e.target.value === '' ? null : e.target.value);
                prevZoneRef.current = null;
              }}
            >
              <option value="">— SELECT SOURCE —</option>
              <option value="live">● LIVE FEED</option>
              {episodes.length === 0 && !episodesLoading && (
                <option value="" disabled>no episode traces found</option>
              )}
              {episodes.map((ep, idx) => (
                <option key={ep.name} value={ep.name}>
                  {`EP ${String(idx + 1).padStart(2, '0')} · ${ep.winner} · ${ep.name.replace('.json', '')}`}
                </option>
              ))}
            </select>
            {episodesLoading && <span className="top-bar-source-meta">loading…</span>}
          </div>
        </div>

        <div className="top-bar-brand">
          <span style={{ color: '#ff4444' }}>C</span>
          <span style={{ color: 'var(--text)' }}>IPHER</span>
          <span className="top-bar-brand-sub">WAR ROOM</span>
        </div>

        <div className="top-bar-right">
          <div className="top-bar-status">
            <div className="top-bar-mode" style={{ color: modeConf.color }}>
              <span
                className="top-bar-mode-dot"
                style={{
                  background: modeConf.color,
                  boxShadow: `0 0 8px ${modeConf.color}`,
                  animation: modeRaw === 'live' ? 'livePulse 1.5s ease-in-out infinite' : 'none',
                }}
                aria-hidden={true}
              />
              <span>{modeConf.text}</span>
            </div>
            <div className="top-bar-metrics">
              <span className="top-bar-metric-label">
                EP <b className="top-bar-metric-num">{latest?.episode ?? 1}</b>
              </span>
              <span className="top-bar-metric-label">
                STEP{' '}
                <b className="top-bar-metric-num">{latest?.step ?? 0}</b>
                <span className="top-bar-metric-denom">/{latest?.max_steps ?? 30}</span>
              </span>
            </div>
          </div>

          {isEpisode && (
            <div className="top-bar-replay-meta">
              {replay.currentIdx} / {replay.totalSteps}
              {replay.isComplete && (
                <span className="top-bar-replay-complete">✓ COMPLETE</span>
              )}
            </div>
          )}

          {winnerConf && (
            <div
              className="top-bar-pill"
              style={{
                color: winnerConf.color,
                background: winnerConf.bg,
                border: `1px solid ${winnerConf.color}45`,
              }}
            >
              {winnerConf.text}
            </div>
          )}

          {latest?.zone && (
            <div
              className="top-bar-pill"
              style={{
                letterSpacing: '0.14em',
                color: ZONE_COLORS[latest.zone] ?? 'var(--z0)',
                background: `${ZONE_COLORS[latest.zone] ?? '#00e5ff'}12`,
                border: `1px solid ${ZONE_COLORS[latest.zone] ?? '#00e5ff'}30`,
              }}
            >
              {latest.zone.toUpperCase()}
            </div>
          )}

          {latest?.red_action && latest.red_action !== 'waiting' && (
            <div
              className="top-bar-inline"
              style={{ color: 'rgba(255,68,68,0.75)', letterSpacing: '0.06em' }}
              title={latest.red_action}
            >
              🔴 {latest.red_action}
            </div>
          )}

          {(latest?.exfil_count ?? 0) > 0 && (
            <div
              className="top-bar-chip"
              style={{
                color: '#ff6b6b',
                background: 'rgba(255,68,68,0.12)',
                border: '1px solid rgba(255,68,68,0.32)',
              }}
              title={`Exfiltrated: ${(latest?.exfil_files ?? []).join(', ') || 'classified file'}`}
            >
              💀 EXFIL: {(latest?.exfil_files ?? [])[0] || 'classified file'}
            </div>
          )}

          {rightTab === 'battle' && (
            <div
              className="top-bar-legend"
              title="RED: PLNR·ANLT·OPRT·EXFL | BLUE: SURV·HUNT·DCVR·FRNS"
            >
              🔴 RED PLNR·ANLT·OPRT·EXFL  |  🔵 BLUE SURV·HUNT·DCVR·FRNS
            </div>
          )}
        </div>
      </div>

      {/* ── Global feature navbar ── */}
      <div className="global-tabs">
        {RIGHT_TABS.map((tab) => (
          <button
            key={tab.id}
            className={`global-tab${rightTab === tab.id ? ' active' : ''}`}
            onClick={() => setRightTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ── Main view area ── */}
      {isBattleView ? (
        <>
          {/* Battle: open battlefield map */}
          <div
            style={{
              position: 'absolute',
              top: 'var(--chrome-top)',
              bottom: 54,
              left: 0,
              right: 0,
              background: '#131a24',
            }}
          >
            <GameMap
              graph={graph}
              steps={steps}
              agentStatus={!hasSource || isEpisode ? null : agentStatus}
              redThoughts={visibleRedThoughts}
              blueThoughts={visibleBlueThoughts}
              speed={speed}
            />
          </div>
          <StatsHUD
            latest={latest}
            winnerCard={winnerCard}
            completionSignals={completionSignals}
            commanderLifecycle={commanderLifecycle}
            steps={steps}
            rightOffset={14}
          />
        </>
      ) : (
        <>
          {/* Non-battle tabs: render directly on main page (no right sidebar) */}
          <div
            style={{
              position: 'absolute',
              top: 'var(--chrome-top)',
              bottom: 54,
              left: 0,
              right: 0,
              overflowY: 'auto',
              background: 'rgba(12, 18, 32, 0.78)',
              borderTop: '1px solid var(--border)',
            }}
          >
            <div style={{ maxWidth: 1100, margin: '0 auto', minHeight: '100%', padding: '10px 12px' }}>
              <div
                style={{
                  minHeight: 'calc(100vh - 170px)',
                  border: '1px solid var(--border)',
                  borderRadius: 10,
                  background: 'linear-gradient(180deg, rgba(18, 24, 36, 0.96), rgba(14, 18, 30, 0.96))',
                  boxShadow: '0 10px 28px rgba(0,0,0,0.35)',
                  overflow: 'hidden',
                  display: 'flex',
                  flexDirection: 'column',
                }}
              >
                {rightTab === 'minds' && <MindsPanel thoughts={visibleThoughts} />}
                {rightTab === 'live-logs' && <LiveLogsPanel />}
                {rightTab === 'rewards' && <RewardTracker steps={steps} />}
                {rightTab === 'learning' && <RLMetricsPanel />}
                {rightTab === 'oversight' && <OversightPanel steps={steps} />}
                {rightTab === 'history' && <HistoryPanel />}
                {rightTab === 'analytics' && <AnalyticsPanel />}
                {rightTab === 'lore' && <LorePanel />}
                {rightTab === 'architecture' && <ArchitecturePanel />}
                {rightTab === 'forensics' && <ForensicsPanel steps={steps} selectedEpisode={selectedEpisode} />}
              </div>
            </div>
          </div>
        </>
      )}

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

        {isEpisode && replay.hasData && (
          <div className="replay-slider-wrap">
            <span className="replay-slider-label">{replay.currentIdx}</span>
            <input
              id="step-slider"
              className="replay-step-slider"
              type="range"
              min={0}
              max={replay.totalSteps}
              step={1}
              value={replay.currentIdx}
              onChange={(e) => replay.seekTo(Number(e.target.value))}
              aria-label="Replay step slider"
              title={`Step ${replay.currentIdx} of ${replay.totalSteps}`}
              style={{ '--slider-progress': `${progressPct}%` }}
            />
            <span className="replay-slider-label">{replay.totalSteps}</span>
          </div>
        )}

        <div style={{ flex: 1 }} />

        <div style={{ fontFamily: 'var(--mono)', fontSize: 8.5, color: 'var(--text-mute)', letterSpacing: '0.06em', flexShrink: 0 }}>
          {steps.length} steps · {visibleThoughts.length} thoughts
        </div>
      </div>

      {/* ── Zone breach banner ── */}
      {zoneBreach && (
        <div style={{ position: 'fixed', top: 'calc(var(--chrome-top) + 12px)', left: '50%', transform: 'translateX(-50%)', padding: '8px 40px', background: 'rgba(16,6,6,0.92)', border: '1px solid rgba(255,68,68,0.4)', borderRadius: 8, fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 700, letterSpacing: '0.22em', color: '#ff4444', zIndex: 800, whiteSpace: 'nowrap', animation: 'glitchIn 0.3s ease-out', boxShadow: '0 4px 24px rgba(255,68,68,0.25)' }}>
          🚨 {zoneBreach}
        </div>
      )}

      {/* ── Episode winner banner ── */}
      {showWinnerBanner && (
        <div
          style={{
            position: 'fixed',
            top: 'calc(var(--chrome-top) + 12px)',
            left: '50%',
            transform: 'translateX(-50%)',
            padding: '10px 30px',
            background: winnerBanner.bg,
            border: `1px solid ${winnerBanner.border}`,
            borderRadius: 9,
            fontFamily: 'var(--mono)',
            fontSize: 12,
            fontWeight: 700,
            letterSpacing: '0.22em',
            color: winnerBanner.color,
            zIndex: 820,
            whiteSpace: 'nowrap',
            boxShadow: `0 6px 26px ${winnerBanner.border}`,
            animation: 'glitchIn 0.35s ease-out',
          }}
        >
          {winnerBanner.icon} {winnerBanner.text}
        </div>
      )}

    </div>
  );
}
