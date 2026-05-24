/**
 * mera_relax step-replay annotator.
 *
 * Diffs `frame.layer_states.mera_relax.residuals` between consecutive frames
 * and surfaces the per-term residual *drops*. The §10.10 induction-theorem
 * demo emits one residual row per active rule×site; whenever a rule fires
 * in the imag-time loop, the corresponding row's residual collapses toward
 * zero. By picking up those collapses we get a frame-by-frame replay of
 * which rules fired — without the substrate having to emit per-step rule
 * activation events.
 *
 * Resolves the `step-replay-mera-relax` deferred extension
 * (src/qft_pcn/viz/EXTENSIONS.md#step-replay-mera-relax).
 */

import type { Frame } from './types';

/** Drop threshold (absolute residual change) before we emit a label. */
export const FIRING_THRESHOLD = 0.05;

export interface MeraRelaxResidual {
  rule_id: string;
  site: number;
  value: number;
}

export interface StepLabel {
  term: { rule_id: string; site: number };
  /** Positive: how much the residual decreased (prev - curr). */
  magnitude: number;
  description: string;
}

function readResiduals(frame: Frame): MeraRelaxResidual[] {
  const layer = frame.layer_states?.mera_relax as
    | Record<string, unknown>
    | undefined;
  if (!layer) return [];
  const rows = layer.residuals as MeraRelaxResidual[] | undefined | null;
  if (!Array.isArray(rows)) return [];
  return rows.filter(
    (r): r is MeraRelaxResidual =>
      r != null &&
      typeof r === 'object' &&
      typeof r.rule_id === 'string' &&
      typeof r.site === 'number' &&
      typeof r.value === 'number' &&
      Number.isFinite(r.value),
  );
}

function keyOf(r: { rule_id: string; site: number }): string {
  return `${r.rule_id}@${r.site}`;
}

/**
 * Walks `frames` in order; for each i>=1, compares residuals against
 * frames[i-1]. Any (rule_id, site) whose absolute residual dropped by
 * more than {@link FIRING_THRESHOLD} produces a {@link StepLabel}. The
 * returned map is keyed by the frame index in `frames` (NOT by `frame.step`).
 * Labels per frame are sorted by descending magnitude so the strongest
 * firing reads first.
 */
export function annotate(frames: Frame[]): Map<number, StepLabel[]> {
  const out = new Map<number, StepLabel[]>();
  if (!Array.isArray(frames) || frames.length < 2) return out;
  let prevByKey = new Map<string, MeraRelaxResidual>();
  for (const r of readResiduals(frames[0])) prevByKey.set(keyOf(r), r);
  for (let i = 1; i < frames.length; i++) {
    const currRows = readResiduals(frames[i]);
    const labels: StepLabel[] = [];
    for (const cur of currRows) {
      const prev = prevByKey.get(keyOf(cur));
      if (!prev) continue;
      const drop = Math.abs(prev.value) - Math.abs(cur.value);
      if (drop > FIRING_THRESHOLD) {
        labels.push({
          term: { rule_id: cur.rule_id, site: cur.site },
          magnitude: drop,
          description: `${cur.rule_id} fired at site ${cur.site}`,
        });
      }
    }
    if (labels.length > 0) {
      labels.sort((a, b) => b.magnitude - a.magnitude);
      out.set(i, labels);
    }
    prevByKey = new Map<string, MeraRelaxResidual>();
    for (const r of currRows) prevByKey.set(keyOf(r), r);
  }
  return out;
}
