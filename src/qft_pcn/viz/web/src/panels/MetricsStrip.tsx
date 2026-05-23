/**
 * A compact multi-line sparkline showing client-side scalar history.
 *
 * The caller passes the layer key + a list of `{key, label, color}` series;
 * we read the active run's frame history from the store and project each
 * series via the supplied `select` function.
 */

import { useMemo } from 'react';
import { useVizStore } from '../store';

export interface MetricSpec {
  key: string;
  label: string;
  color: string;
  /** Pull the scalar value out of one frame's layer_states[layer] dict. */
  select: (layerState: Record<string, unknown>) => number | null | undefined;
}

interface Props { layer: string; metrics: MetricSpec[]; }

export function MetricsStrip({ layer, metrics }: Props) {
  const runId = useVizStore((s) => s.activeRunId);
  const run = useVizStore((s) => (runId ? s.runs.get(runId) : undefined));
  const baselineId = useVizStore((s) => s.baselineRunId);
  const baselineRun = useVizStore((s) =>
    baselineId ? s.runs.get(baselineId) : undefined,
  );
  const frames = run?.frames ?? [];
  const baselineFrames = baselineRun?.frames ?? [];
  const hasBaseline = !!baselineId && baselineId !== runId
    && baselineFrames.length >= 2;

  const series = useMemo(() => metrics.map((m) => ({
    ...m,
    points: frames.map((f) => m.select(
      (f.layer_states[layer] ?? {}) as Record<string, unknown>)),
  })), [frames, metrics, layer]);

  const baselineSeries = useMemo(() => (
    hasBaseline
      ? metrics.map((m) => ({
          ...m,
          points: baselineFrames.map((f) => m.select(
            (f.layer_states[layer] ?? {}) as Record<string, unknown>)),
        }))
      : []
  ), [baselineFrames, metrics, layer, hasBaseline]);

  if (frames.length < 2) {
    return <div className="metrics-strip empty">collecting…</div>;
  }

  const W = 240, H = 56, PAD = 4;
  const xs = (i: number, n: number) =>
    PAD + (n <= 1 ? 0 : (i / (n - 1)) * (W - 2 * PAD));
  const allFinite: number[] = [];
  for (const s of series) {
    for (const p of s.points) {
      if (typeof p === 'number' && Number.isFinite(p)) allFinite.push(p);
    }
  }
  for (const s of baselineSeries) {
    for (const p of s.points) {
      if (typeof p === 'number' && Number.isFinite(p)) allFinite.push(p);
    }
  }
  const lo = allFinite.length ? Math.min(...allFinite) : 0;
  const hi = allFinite.length ? Math.max(...allFinite) : 1;
  const span = hi - lo || 1;
  const ys = (v: number) => H - PAD - ((v - lo) / span) * (H - 2 * PAD);

  const pathD = (points: Array<number | null | undefined>, n: number) =>
    points
      .map((p, i) => (typeof p === 'number' && Number.isFinite(p)
        ? `${i === 0 ? 'M' : 'L'} ${xs(i, n)} ${ys(p)}`
        : ''))
      .filter(Boolean)
      .join(' ');

  return (
    <div className="metrics-strip">
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H}>
        {/* Baseline series rendered first (under the active series) — dashed
            stroke at 60% alpha so the baseline reads as background context. */}
        {baselineSeries.map((s) => (
          <path
            key={`b:${s.key}`}
            d={pathD(s.points, baselineFrames.length)}
            fill="none"
            stroke={s.color}
            strokeOpacity={0.6}
            strokeDasharray="3 2"
            strokeWidth={1.2}
          />
        ))}
        {series.map((s) => (
          <path
            key={s.key}
            d={pathD(s.points, frames.length)}
            fill="none"
            stroke={s.color}
            strokeWidth={1.4}
          />
        ))}
      </svg>
      <div className="metrics-strip-legend">
        {series.map((s) => (
          <span key={s.key}>
            <span className="swatch" style={{ background: s.color }} />
            {s.label}
            {hasBaseline && <small style={{ opacity: 0.6 }}> (vs baseline)</small>}
          </span>
        ))}
      </div>
    </div>
  );
}
