/**
 * App shell: 4 regions stacked vertically — RunControls, CompareBar,
 * body grid (LayerSelector | PanelArea | ExplainerPane), Timeline.
 */

import { SectionedLayerSelector } from './components/SectionedLayerSelector';
import { Timeline } from './components/Timeline';
import { RunControls } from './components/RunControls';
import { CompareBar } from './components/CompareBar';
import { ExplainerPane } from './components/ExplainerPane';
import { RouteSwitcher } from './components/RouteSwitcher';
import { DslRoute } from './routes/DslRoute';
import { LearnRoute } from './routes/LearnRoute';
import { panelFor } from './panels';
import { useVizStore } from './store';
import './App.css';

function PanelArea() {
  const selectedLayer = useVizStore((s) => s.selectedLayer);
  const active = useVizStore((s) => s.activeRunId);
  const frame = useVizStore((s) =>
    active ? s.runs.get(active)?.frames[s.runs.get(active)!.cursor] : undefined);
  const baselineFrame = useVizStore((s) => s.baselineFrame());
  const Panel = panelFor(selectedLayer);

  return (
    <main className="panel-area">
      {!frame ? (
        <p className="empty">No frames yet — start a run to stream simulation data.</p>
      ) : !Panel ? (
        <p className="empty">No panel registered for "{selectedLayer}".</p>
      ) : (
        <Panel frame={frame} baselineFrame={baselineFrame} />
      )}
    </main>
  );
}

export default function App() {
  const error = useVizStore((s) => s.error);
  const route = useVizStore((s) => s.route);
  const selectedLayer = useVizStore((s) => s.selectedLayer);
  return (
    <div className="app">
      <RouteSwitcher />
      {error && <div className="error-bar" role="alert">{error}</div>}
      {route === 'viz' ? (
        <>
          <RunControls />
          <CompareBar />
          <div className="body">
            <SectionedLayerSelector />
            <PanelArea />
            <ExplainerPane layer={selectedLayer} />
          </div>
          <Timeline />
        </>
      ) : route === 'dsl' ? (
        <DslRoute />
      ) : route === 'learn' ? (
        <LearnRoute />
      ) : (
        // training branch — placeholder until T14
        <div className="route-placeholder">Training route not yet implemented.</div>
      )}
    </div>
  );
}
