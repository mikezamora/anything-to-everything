/** Bottom scrubber: a range input bound to the store cursor. */

import { useVizStore } from '../store';

export function Timeline() {
  const frames = useVizStore((s) => s.frames);
  const cursor = useVizStore((s) => s.cursor);
  const live = useVizStore((s) => s.live);
  const setCursor = useVizStore((s) => s.setCursor);
  const setLive = useVizStore((s) => s.setLive);

  const max = Math.max(0, frames.length - 1);

  return (
    <div className="timeline">
      <input
        type="range"
        aria-label="timeline scrubber"
        min={0}
        max={max}
        value={cursor}
        disabled={frames.length === 0}
        onChange={(e) => {
          // Scrubbing detaches from the live tail.
          setLive(false);
          setCursor(Number(e.target.value));
        }}
      />
      <span className="timeline-label">
        step {frames[cursor]?.step ?? '-'} ({cursor + (frames.length ? 1 : 0)}/
        {frames.length})
      </span>
      <label className="timeline-live">
        <input
          type="checkbox"
          checked={live}
          onChange={(e) => setLive(e.target.checked)}
        />
        live
      </label>
    </div>
  );
}
