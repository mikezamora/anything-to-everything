/**
 * A horizontal strip of labelled scalar cells.
 *
 * Each cell has an optional `highlightId`; when ExplainerPane fires the
 * `viz:highlight-readout` window event with a matching id, the cell
 * flashes for 800ms.
 */

import { useEffect, useState } from 'react';

export interface ReadoutCell {
  label: string;
  value: string | number | null | undefined;
  unit?: string;
  highlightId?: string;
  /**
   * Optional baseline value for compare mode. When present and both `value`
   * and `baselineValue` resolve to finite numbers, the cell renders
   * `current (±Δ)` with a sign-coloured Δ (green for negative — i.e. lower
   * energy / error — and red for positive). When absent or non-numeric the
   * cell falls back to the plain current-value rendering.
   */
  baselineValue?: number | null;
}

/** Parse a possibly-string value into a finite number for Δ computation. */
function toNumeric(v: string | number | null | undefined): number | null {
  if (v == null) return null;
  const n = typeof v === 'number' ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

export function PanelReadouts({ cells }: { cells: ReadoutCell[] }) {
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    const handler = (ev: Event) => {
      const id = (ev as CustomEvent).detail?.id ?? null;
      setActive(id);
      if (id) {
        const t = setTimeout(() => setActive(null), 800);
        return () => clearTimeout(t);
      }
    };
    window.addEventListener('viz:highlight-readout', handler);
    return () => window.removeEventListener('viz:highlight-readout', handler);
  }, []);

  return (
    <div className="panel-readouts">
      {cells.map((c, i) => {
        const curN = toNumeric(c.value);
        const baseN =
          c.baselineValue != null && Number.isFinite(c.baselineValue)
            ? c.baselineValue
            : null;
        const showDelta = curN != null && baseN != null;
        const delta = showDelta ? curN! - baseN! : null;
        // Match QpcnPanel: green for negative (improvement), red for positive.
        const deltaColor =
          delta == null
            ? undefined
            : delta < 0
              ? '#9aedc1'
              : delta > 0
                ? '#ef9090'
                : '#7f8bb0';
        const deltaSign = delta != null && delta >= 0 ? '+' : '';
        return (
          <div
            key={i}
            className={`panel-readout${
              c.highlightId && c.highlightId === active ? ' highlight' : ''
            }`}
          >
            <span className="panel-readout-label">{c.label}</span>
            <span className="panel-readout-value">
              {c.value == null ? '—' : c.value}
              {c.unit && <small> {c.unit}</small>}
              {showDelta && (
                <small
                  className="panel-readout-delta"
                  style={{ color: deltaColor, marginLeft: 4 }}
                >
                  ({deltaSign}
                  {delta!.toFixed(3)})
                </small>
              )}
            </span>
          </div>
        );
      })}
    </div>
  );
}
