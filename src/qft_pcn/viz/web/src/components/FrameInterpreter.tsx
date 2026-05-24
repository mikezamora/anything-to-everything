// src/qft_pcn/viz/web/src/components/FrameInterpreter.tsx
/**
 * Render a 1-2 sentence interpretation of the active frame for one layer.
 * Subscribes via Zustand; mounts as a small overlay on each panel.
 */

import { useVizStore } from '../store';
import { INTERPRETERS } from '../lib/interpreters';

interface Props {
  layer: string;
}

export function FrameInterpreter({ layer }: Props) {
  const activeRunId = useVizStore((s) => s.activeRunId);
  const frame = useVizStore((s) => {
    if (!activeRunId) return undefined;
    const run = s.runs.get(activeRunId);
    return run?.frames[run.cursor];
  });

  if (!frame) return null;
  const interpret = INTERPRETERS[layer];
  if (!interpret) return null;
  const ls = (frame.layer_states[layer] ?? {}) as Record<string, unknown>;
  const out = interpret(ls, frame.step);
  if (!out) return null;

  return (
    <div className="frame-interpreter" role="note">
      <span className="frame-interpreter-marker">🔬</span>
      <span className="frame-interpreter-text">{out.text}</span>
      {out.citation && (
        <a
          className="frame-interpreter-cite"
          href={`#/learn/${out.citation}`}
        >
          → open article
        </a>
      )}
    </div>
  );
}
