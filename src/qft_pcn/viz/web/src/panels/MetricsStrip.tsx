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

  // Y-axis tick labels show the data span so the trace magnitudes are
  // legible (otherwise the sparkline reads as relative shape only). We
  // place them inside a small reserved gutter so the rendered series
  // pixels don't move when the axis is added.
  const Y_GUTTER = 24;
  const fmtTick = (v: number) => {
    if (!Number.isFinite(v)) return '—';
    if (v === 0) return '0';
    const a = Math.abs(v);
    if (a >= 1000 || a < 1e-3) return v.toExponential(1);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    return v.toFixed(3);
  };

  return (
    <div className="metrics-strip">
      <svg
        viewBox={`0 0 ${W + Y_GUTTER} ${H}`}
        width={W + Y_GUTTER}
        height={H}
      >
        <g transform={`translate(${Y_GUTTER},0)`}>
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
        </g>
        {/* Y-axis: a thin line + min/max tick labels inside the gutter. */}
        <g data-testid="metrics-strip-yaxis">
          <line
            x1={Y_GUTTER - 0.5}
            y1={PAD}
            x2={Y_GUTTER - 0.5}
            y2={H - PAD}
            stroke="#3a4660"
          />
          <text
            x={Y_GUTTER - 3}
            y={PAD + 7}
            fontSize={8}
            fill="#7f8bb0"
            textAnchor="end"
          >
            {fmtTick(hi)}
          </text>
          <text
            x={Y_GUTTER - 3}
            y={H - PAD - 1}
            fontSize={8}
            fill="#7f8bb0"
            textAnchor="end"
          >
            {fmtTick(lo)}
          </text>
        </g>
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
