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
 *
 * Returns `React.RefObject<T>`: under the installed React 18 types,
 * `useRef<T>(null)` resolves to `RefObject<T>` (whose `.current` is already
 * `T | null`), and that type is what JSX `ref=` props accept.
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

/**
 * Deterministic colour for a field species, keyed on the species' index in a
 * stable name-sorted ordering. Both the 3-D surfaces and the coupling graph
 * use this so a species reads the same colour in both views.
 */
export function speciesColor(name: string, names: string[]): string {
  const ordered = [...names].sort();
  const i = ordered.indexOf(name);
  const denom = Math.max(1, ordered.length - 1);
  return sequential((i < 0 ? 0 : i) / denom);
}

/**
 * Collapse a phi/E/Pi grid from `snapshot_network` / `snapshot_multifield` /
 * `snapshot_pcn_fields` to a 2D `(Nx, Ny)` grid suitable for surface / heatmap
 * rendering. `Field.values` is shape `(C, Nx, Ny)` in numpy and `_grid` just
 * calls `.tolist()`, so the wire shape is 3D `(channels, Nx, Ny)`. Three.js
 * surface + d3 heatmap consumers need 2D, so pick a channel (default 0).
 * Legacy fixtures and tests sometimes pass 2D grids directly — those pass
 * through unchanged.
 */
export type Grid2D = number[][];
export type PhiLike = number[][] | number[][][] | null | undefined;
export function as2DGrid(phi: PhiLike, channel: number = 0): Grid2D | null {
  if (!phi || !phi.length) return null;
  const first = phi[0] as number | number[] | number[][] | undefined;
  if (Array.isArray(first) && first.length && Array.isArray(first[0])) {
    // phi is (C, Nx, Ny) -> pick requested channel (fallback to 0 if OOB).
    const c = phi as number[][][];
    const idx = channel >= 0 && channel < c.length ? channel : 0;
    return c[idx] as Grid2D;
  }
  return phi as Grid2D;
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
