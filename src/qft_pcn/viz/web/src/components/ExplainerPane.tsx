// src/qft_pcn/viz/web/src/components/ExplainerPane.tsx
/**
 * Right-rail explainer for the active layer (tabbed).
 *
 * Pulls `EXPLAINERS[layer]` and renders five tabs: Overview (What + Elements),
 * Math (AnnotatedEquation refs), Worked Example, Training Dynamics
 * (update rule + Expect + If-you-see), and Watch (live-watch hints —
 * hovering a hint flashes the matching <PanelReadouts> cell via a custom
 * event the panel listens to).
 */

import { useState } from 'react';
import { EXPLAINERS, type ExplainerSpec } from '../lib/explainer';
import { AnnotatedEquation } from './AnnotatedEquation';

type TabId = 'overview' | 'math' | 'workedExample' | 'trainingDynamics' | 'watch';

const TAB_ORDER: { id: TabId; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'math', label: 'Math' },
  { id: 'workedExample', label: 'Worked Example' },
  { id: 'trainingDynamics', label: 'Dynamics' },
  { id: 'watch', label: 'Watch' },
];

interface Props { layer: string }

export function ExplainerPane({ layer }: Props) {
  const spec = EXPLAINERS[layer];
  const [collapsed, setCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>('overview');

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
        <>
          <p className="explainer-oneline">{spec.oneLine}</p>
          <nav className="explainer-tabs">
            {TAB_ORDER.map((t) => (
              <button
                key={t.id}
                type="button"
                className={`explainer-tab${
                  activeTab === t.id ? ' active' : ''
                }`}
                onClick={() => setActiveTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
          <div className="explainer-body">
            {renderTab(spec, activeTab)}
            {spec.references && activeTab === 'overview' && (
              <section className="explainer-refs">
                <h4>References</h4>
                <ul>
                  {spec.references.map((r, i) => (
                    <li key={i}>
                      <a href={r.href}>{r.label}</a>
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </div>
        </>
      )}
    </aside>
  );
}

function renderTab(spec: ExplainerSpec, tab: TabId) {
  if (tab === 'overview') {
    return (
      <>
        <section>
          <h4>What</h4>
          {spec.tabs.overview.what.map((p, i) => <p key={i}>{p}</p>)}
        </section>
        <section>
          <h4>Elements</h4>
          <ul>
            {spec.tabs.overview.elements.map((e, i) => (
              <li key={i}>
                <strong>{e.name}</strong> — {e.meaning}
                {e.code && <code className="explainer-code"> {e.code}</code>}
              </li>
            ))}
          </ul>
        </section>
      </>
    );
  }
  if (tab === 'math') {
    return (
      <>
        {spec.tabs.math.equationIds.map((id) => (
          <AnnotatedEquation key={id} id={id} />
        ))}
      </>
    );
  }
  if (tab === 'workedExample') {
    const ex = spec.tabs.workedExample;
    return (
      <section className="explainer-worked-example">
        <h4>{ex.title}</h4>
        <p><strong>Setup.</strong> {ex.setup}</p>
        <ol>
          {ex.steps.map((s, i) => (
            <li key={i}>
              {s.description}
              {s.equationId && (
                <div className="explainer-worked-eq">
                  <AnnotatedEquation id={s.equationId} />
                </div>
              )}
              <div className="explainer-worked-result"><em>⇒ {s.result}</em></div>
            </li>
          ))}
        </ol>
        <p><strong>Takeaway.</strong> {ex.takeaway}</p>
      </section>
    );
  }
  if (tab === 'trainingDynamics') {
    const td = spec.tabs.trainingDynamics;
    return (
      <section>
        {td.updateRuleId && (
          <>
            <h4>Update rule</h4>
            <AnnotatedEquation id={td.updateRuleId} />
          </>
        )}
        <h4>Expect</h4>
        <ul>{td.expect.map((b, i) => <li key={i}>{b}</li>)}</ul>
        <h4>If you see…</h4>
        <ul>
          {td.pathologies.map((p, i) => (
            <li key={i}>
              <strong>{p.signal}</strong> — {p.cause}
            </li>
          ))}
        </ul>
      </section>
    );
  }
  // watch
  return (
    <section>
      <h4>Watch</h4>
      <ul>
        {spec.tabs.watch.map((w, i) => (
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
  );
}

/** Fire a window event the PanelReadouts component subscribes to. */
function dispatchHighlight(readoutId: string | null) {
  window.dispatchEvent(
    new CustomEvent('viz:highlight-readout', { detail: { id: readoutId } }),
  );
}
