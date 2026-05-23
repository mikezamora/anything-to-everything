/**
 * App shell: 4 regions stacked vertically — RunControls, CompareBar,
 * body grid (LayerSelector | PanelArea | ExplainerPane), Timeline.
 */

import { LayerSelector } from './components/LayerSelector';
import { Timeline } from './components/Timeline';
import { RunControls } from './components/RunControls';
import { CompareBar } from './components/CompareBar';
import { ExplainerPane } from './components/ExplainerPane';
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
        <Panel frame={frame} baselineFrame={baselineFrame as any} />
      )}
    </main>
  );
}

export default function App() {
  const error = useVizStore((s) => s.error);
  const selectedLayer = useVizStore((s) => s.selectedLayer);
  return (
    <div className="app">
      <RunControls />
      <CompareBar />
      {error && <div className="error-bar" role="alert">{error}</div>}
      <div className="body">
        <LayerSelector />
        <PanelArea />
        <ExplainerPane layer={selectedLayer} />
      </div>
      <Timeline />
    </div>
  );
}
