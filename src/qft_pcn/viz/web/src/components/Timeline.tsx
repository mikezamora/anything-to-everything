/** Bottom scrubber bound to the active run's cursor + live flag. */

import { useVizStore } from '../store';

export function Timeline() {
  const active = useVizStore((s) => s.activeRunId);
  const run = useVizStore((s) => (active ? s.runs.get(active) : undefined));
  const setCursor = useVizStore((s) => s.setCursor);
  const setLive = useVizStore((s) => s.setLive);

  if (!active || !run) return <div className="timeline empty">no active run</div>;
  const max = Math.max(0, run.frames.length - 1);
  return (
    <div className="timeline">
      <input type="range" aria-label="timeline scrubber"
        min={0} max={max} value={run.cursor}
        disabled={run.frames.length === 0}
        onChange={(e) => setCursor(active, Number(e.target.value))} />
      <span className="timeline-label">
        step {run.frames[run.cursor]?.step ?? '-'} (
        {run.cursor + (run.frames.length ? 1 : 0)}/{run.frames.length})
      </span>
      <label className="timeline-live">
        <input type="checkbox" checked={run.live}
          onChange={(e) => setLive(active, e.target.checked)} />
        live
      </label>
    </div>
  );
}
