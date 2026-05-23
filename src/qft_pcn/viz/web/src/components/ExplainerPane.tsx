// src/qft_pcn/viz/web/src/components/ExplainerPane.tsx
/**
 * Right-rail explainer for the active layer.
 *
 * Pulls `EXPLAINERS[layer]` and renders four sections: What (prose),
 * Elements (visual ↦ substrate mapping), Math (KaTeX), Watch (live-watch
 * hints — hovering a hint flashes the matching <PanelReadouts> cell via
 * a custom event the panel listens to).
 */

import { useState } from 'react';
import { EXPLAINERS } from '../lib/explainer';
import { tex } from '../panels/common';

interface Props { layer: string }

export function ExplainerPane({ layer }: Props) {
  const spec = EXPLAINERS[layer];
  const [collapsed, setCollapsed] = useState(false);

  if (!spec) {
    return (
      <aside className="explainer">
        <div className="explainer-header">
          <span>No explainer for "{layer}"</span>
        </div>
      </aside>
    );
  }

  return (
    <aside className={`explainer${collapsed ? ' collapsed' : ''}`}>
      <div className="explainer-header">
        <strong>{spec.title}</strong>
        <button
          type="button"
          className="explainer-toggle"
          onClick={() => setCollapsed((c) => !c)}
          aria-label={collapsed ? 'Expand explainer' : 'Collapse explainer'}
        >
          {collapsed ? '◀' : '▶'}
        </button>
      </div>
      {!collapsed && (
        <div className="explainer-body">
          <p className="explainer-oneline">{spec.oneLine}</p>

          <section>
            <h4>What</h4>
            {spec.what.map((p, i) => <p key={i}>{p}</p>)}
          </section>

          <section>
            <h4>Elements</h4>
            <ul>
              {spec.elements.map((e, i) => (
                <li key={i}>
                  <strong>{e.name}</strong> — {e.meaning}
                  {e.code && <code className="explainer-code"> {e.code}</code>}
                </li>
              ))}
            </ul>
          </section>

          <section>
            <h4>Math</h4>
            {spec.math.map((m, i) => (
              <div key={i} className="explainer-math">
                <div dangerouslySetInnerHTML={{ __html: tex(m.tex) }} />
                <small>{m.caption}</small>
              </div>
            ))}
          </section>

          <section>
            <h4>Watch</h4>
            <ul>
              {spec.watch.map((w, i) => (
                <li
                  key={i}
                  onMouseEnter={() => w.readout && dispatchHighlight(w.readout)}
                  onMouseLeave={() => w.readout && dispatchHighlight(null)}
                >
                  {w.label}
                </li>
              ))}
            </ul>
          </section>

          {spec.references && (
            <section>
              <h4>References</h4>
              <ul>
                {spec.references.map((r, i) => (
                  <li key={i}><a href={r.href}>{r.label}</a></li>
                ))}
              </ul>
            </section>
          )}
        </div>
      )}
    </aside>
  );
}

/** Fire a window event the PanelReadouts component subscribes to. */
function dispatchHighlight(readoutId: string | null) {
  window.dispatchEvent(new CustomEvent('viz:highlight-readout',
    { detail: { id: readoutId } }));
}
