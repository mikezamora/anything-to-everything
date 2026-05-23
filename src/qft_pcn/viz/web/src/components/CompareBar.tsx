/** Shown when a baseline run is pinned: lists both run ids and lets the user swap or unpin. */

import { useVizStore } from '../store';

export function CompareBar() {
  const active = useVizStore((s) => s.activeRunId);
  const baseline = useVizStore((s) => s.baselineRunId);
  const setActive = useVizStore((s) => s.setActiveRun);
  const unpin = useVizStore((s) => s.unpinBaseline);
  const pin = useVizStore((s) => s.pinBaseline);

  if (!baseline) return null;
  return (
    <div className="compare-bar">
      <span>active: <code>{active ?? '—'}</code></span>
      <span>baseline: <code>{baseline}</code></span>
      <button type="button" onClick={() => {
        if (!active) return;
        const a = active, b = baseline;
        setActive(b); pin(a);
      }}>swap</button>
      <button type="button" onClick={unpin}>unpin</button>
    </div>
  );
}
