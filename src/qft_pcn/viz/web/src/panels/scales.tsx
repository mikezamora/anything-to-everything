/**
 * Small reusable scale / axis / legend components shared across panels.
 *
 * All components render plain SVG (or plain DOM with a tiny SVG ramp) so
 * they work in jsdom tests AND in the real browser. None use measure
 * hooks — they accept explicit pixel sizes from the caller.
 */

import { diverging, sequential } from './common';

/** Format a number compactly for axis ticks / legends. */
function fmt(v: number): string {
  if (!Number.isFinite(v)) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1000 || a < 1e-3) return v.toExponential(1);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

/**
 * Horizontal diverging color-ramp legend with min / centre / max labels.
 * Width defaults to 160px so it fits comfortably below a heatmap.
 */
export function ColorRampLegend({
  min,
  max,
  ramp = 'diverging',
  width = 160,
  height = 10,
  label,
}: {
  min: number;
  max: number;
  ramp?: 'diverging' | 'sequential';
  width?: number;
  height?: number;
  label?: string;
}) {
  // Sample N evenly-spaced stops so the gradient reads correctly even when
  // the diverging ramp is asymmetric (negative-heavy vs positive-heavy).
  const N = 32;
  const stops: string[] = [];
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    if (ramp === 'sequential') {
      stops.push(sequential(t));
    } else {
      // Map [0, 1] -> [-1, 1] for diverging.
      stops.push(diverging(t * 2 - 1));
    }
  }
  return (
    <div
      data-testid="color-ramp-legend"
      style={{ display: 'inline-flex', flexDirection: 'column', gap: 2 }}
    >
      {label && (
        <span style={{ fontSize: 9, color: '#7f8bb0' }}>{label}</span>
      )}
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {stops.map((c, i) => (
          <rect
            key={i}
            x={(i / N) * width}
            y={0}
            width={width / N + 0.5}
            height={height}
            fill={c}
          />
        ))}
      </svg>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 9,
          color: '#7f8bb0',
          width,
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        <span>{fmt(min)}</span>
        {ramp === 'diverging' && <span>0</span>}
        <span>{fmt(max)}</span>
      </div>
    </div>
  );
}

/**
 * X+Y axis ticks for a chart drawn into an SVG with `margin` reserved on
 * the left / bottom. The chart's data span is given by [xMin, xMax] /
 * [yMin, yMax]; the helper places `nTicks` evenly-spaced tick marks and
 * labels on each axis. The caller is responsible for drawing the actual
 * series geometry.
 */
export function ChartAxes({
  width,
  height,
  margin,
  xMin,
  xMax,
  yMin,
  yMax,
  xLabel,
  yLabel,
  nTicks = 4,
}: {
  width: number;
  height: number;
  margin: { l: number; r: number; t: number; b: number };
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  xLabel?: string;
  yLabel?: string;
  nTicks?: number;
}) {
  const plotL = margin.l;
  const plotR = width - margin.r;
  const plotT = margin.t;
  const plotB = height - margin.b;
  const xs = (v: number) =>
    plotL + ((v - xMin) / Math.max(1e-12, xMax - xMin)) * (plotR - plotL);
  const ys = (v: number) =>
    plotB - ((v - yMin) / Math.max(1e-12, yMax - yMin)) * (plotB - plotT);

  const xTicks: number[] = [];
  const yTicks: number[] = [];
  for (let i = 0; i < nTicks; i++) {
    const t = nTicks === 1 ? 0.5 : i / (nTicks - 1);
    xTicks.push(xMin + t * (xMax - xMin));
    yTicks.push(yMin + t * (yMax - yMin));
  }

  return (
    <g data-testid="chart-axes" pointerEvents="none">
      {/* axis lines */}
      <line
        x1={plotL}
        y1={plotB}
        x2={plotR}
        y2={plotB}
        stroke="#3a4660"
        strokeWidth={1}
      />
      <line
        x1={plotL}
        y1={plotT}
        x2={plotL}
        y2={plotB}
        stroke="#3a4660"
        strokeWidth={1}
      />
      {/* x ticks + labels */}
      {xTicks.map((t, i) => (
        <g key={`x${i}`}>
          <line
            x1={xs(t)}
            y1={plotB}
            x2={xs(t)}
            y2={plotB + 3}
            stroke="#5f6b86"
          />
          <text
            x={xs(t)}
            y={plotB + 12}
            textAnchor="middle"
            fontSize={9}
            fill="#7f8bb0"
          >
            {fmt(t)}
          </text>
        </g>
      ))}
      {/* y ticks + labels */}
      {yTicks.map((t, i) => (
        <g key={`y${i}`}>
          <line
            x1={plotL - 3}
            y1={ys(t)}
            x2={plotL}
            y2={ys(t)}
            stroke="#5f6b86"
          />
          <text
            x={plotL - 4}
            y={ys(t) + 3}
            textAnchor="end"
            fontSize={9}
            fill="#7f8bb0"
          >
            {fmt(t)}
          </text>
        </g>
      ))}
      {xLabel && (
        <text
          x={(plotL + plotR) / 2}
          y={height - 2}
          textAnchor="middle"
          fontSize={9}
          fill="#9aa3bb"
        >
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text
          x={4}
          y={(plotT + plotB) / 2}
          textAnchor="start"
          fontSize={9}
          fill="#9aa3bb"
          transform={`rotate(-90 ${10} ${(plotT + plotB) / 2})`}
        >
          {yLabel}
        </text>
      )}
    </g>
  );
}
