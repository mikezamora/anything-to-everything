/**
 * App shell: a run-control bar, a left layer selector, a central panel area
 * that renders the per-layer panel (`panelFor`) for the selected layer
 * against the current streamed frame, and a bottom timeline scrubber.
 */

import { LayerSelector } from './components/LayerSelector';
import { Timeline } from './components/Timeline';
import { RunControls } from './components/RunControls';
import { panelFor } from './panels';
import { useVizStore } from './store';
import './App.css';

function PanelArea() {
  const selectedLayer = useVizStore((s) => s.selectedLayer);
  const frame = useVizStore((s) => s.currentFrame());
  const Panel = panelFor(selectedLayer);

  return (
    <main className="panel-area">
      {!frame ? (
        <p className="empty">
          No frames yet — start a run to stream simulation data.
        </p>
      ) : !Panel ? (
        <p className="empty">No panel registered for “{selectedLayer}”.</p>
      ) : (
        <Panel frame={frame} />
      )}
    </main>
  );
}

export default function App() {
  const error = useVizStore((s) => s.error);

  return (
    <div className="app">
      <RunControls />
      {error && (
        <div className="error-bar" role="alert">
          {error}
        </div>
      )}
      <div className="body">
        <LayerSelector />
        <PanelArea />
      </div>
      <Timeline />
    </div>
  );
}
