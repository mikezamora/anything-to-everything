/**
 * Shared helpers for per-layer panels: a resize-observing size hook, a
 * KaTeX label renderer, and a small diverging colour ramp used by the
 * grid / heatmap panels.
 */

import { useEffect, useRef, useState } from 'react';
import katex from 'katex';

export interface Size {
  width: number;
  height: number;
}

/**
 * Track an element's pixel size via `ResizeObserver`. Returns a ref to attach
 * and the current size. Falls back to a sane default before first measure
 * (jsdom reports 0x0).
 */
export function useSize<T extends HTMLElement>(): [
  React.RefObject<T>,
  Size,
] {
  const ref = useRef<T>(null);
  const [size, setSize] = useState<Size>({ width: 640, height: 420 });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => {
      const w = el.clientWidth || 640;
      const h = el.clientHeight || 420;
      setSize({ width: w, height: h });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return [ref, size];
}

/** Render a TeX string to an HTML string for use via `dangerouslySetInnerHTML`. */
export function tex(src: string): string {
  try {
    return katex.renderToString(src, { throwOnError: false });
  } catch {
    return src;
  }
}

/**
 * Map a scalar in [-1, 1] (clamped) to a blue-white-red diverging colour.
 * Used for metric / curvature / energy fields.
 */
export function diverging(t: number): string {
  const x = Math.max(-1, Math.min(1, t));
  if (x < 0) {
    const k = 1 + x; // 0..1
    return `rgb(${Math.round(60 + 195 * k)}, ${Math.round(
      90 + 165 * k,
    )}, 255)`;
  }
  const k = 1 - x;
  return `rgb(255, ${Math.round(90 + 165 * k)}, ${Math.round(60 + 195 * k)})`;
}

/** Map a scalar in [0, 1] to a dark-to-cyan sequential ramp. */
export function sequential(t: number): string {
  const x = Math.max(0, Math.min(1, t));
  return `rgb(${Math.round(20 + 40 * x)}, ${Math.round(
    30 + 200 * x,
  )}, ${Math.round(60 + 195 * x)})`;
}

/** Normalize a numeric grid (2-D array) to [-1, 1] by its peak abs value. */
export function normGrid(grid: number[][]): {
  norm: number[][];
  peak: number;
} {
  let peak = 0;
  for (const row of grid) {
    for (const v of row) {
      const a = Math.abs(v);
      if (Number.isFinite(a) && a > peak) peak = a;
    }
  }
  const scale = peak > 0 ? peak : 1;
  return {
    norm: grid.map((row) => row.map((v) => (Number.isFinite(v) ? v / scale : 0))),
    peak,
  };
}
