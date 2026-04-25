const PRESETS = [0.5, 1, 2, 4];
const PRESET_LABELS = ['0.5×', '1×', '2×', '4×'];

export default function SpeedControl({ speed, onChange, isPlaying, onPlay, onPause, hasReplay }) {
  const canPlay = hasReplay;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>

      {/* Play / Pause button */}
      {canPlay && (
        <button
          className={`play-btn ${isPlaying ? 'playing' : 'paused'}`}
          onClick={isPlaying ? onPause : onPlay}
          title={isPlaying ? 'Pause replay' : 'Play replay'}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
      )}

      {/* Label */}
      <span style={{
        fontFamily: 'var(--mono)', fontSize: 9, fontWeight: 700,
        letterSpacing: '0.14em', color: 'var(--text-mute)',
        textTransform: 'uppercase',
      }}>
        SPEED
      </span>

      {/* Preset buttons */}
      <div style={{ display: 'flex', gap: 4 }}>
        {PRESETS.map((p, i) => (
          <button
            key={p}
            className={`speed-preset${Math.abs(speed - p) < 0.01 ? ' active' : ''}`}
            onClick={() => onChange(p)}
          >
            {PRESET_LABELS[i]}
          </button>
        ))}
      </div>

      {/* Slider */}
      <input
        type="range"
        className="speed-slider"
        min={0.1}
        max={5}
        step={0.05}
        value={speed}
        onChange={e => onChange(Number(e.target.value))}
      />

      <span style={{
        fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 700,
        color: 'var(--z0)', minWidth: 36,
      }}>
        {speed.toFixed(1)}×
      </span>
    </div>
  );
}
