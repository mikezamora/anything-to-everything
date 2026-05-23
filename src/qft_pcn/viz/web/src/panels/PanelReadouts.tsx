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
      {cells.map((c, i) => (
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
          </span>
        </div>
      ))}
    </div>
  );
}
