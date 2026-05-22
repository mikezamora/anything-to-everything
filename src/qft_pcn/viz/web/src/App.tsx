/**
 * App shell: a run-control bar, a left layer selector, a central panel area
 * (placeholder until Task 4 wires real per-layer panels), and a bottom
 * timeline scrubber.
 */

import { useEffect, useRef, useState } from 'react';
import { LayerSelector } from './components/LayerSelector';
import { Timeline } from './components/Timeline';
import { connectRun } from './lib/ws';
import type { RunHandle } from './lib/ws';
import { LAYER_KEYS } from './lib/types';
import { useVizStore } from './store';
import './App.css';

function RunControls() {
  const [layers, setLayers] = useState<string[]>(['manifold']);
  const [steps, setSteps] = useState(20);
  const [grid, setGrid] = useState(12);
  const [busy, setBusy] = useState(false);
  const handleRef = useRef<RunHandle | null>(null);

  const toggleLayer = (layer: string) => {
    setLayers((cur) =>
      cur.includes(layer) ? cur.filter((l) => l !== layer) : [...cur, layer],
    );
  };

  // Close any open socket when the controls unmount.
  useEffect(() => {
    return () => {
      handleRef.current?.close();
      handleRef.current = null;
    };
  }, []);

  const run = async () => {
    setBusy(true);
    // Tear down a still-streaming previous run before starting a new one.
    handleRef.current?.close();
    handleRef.current = null;
    try {
      handleRef.current = await connectRun({
        layers: layers.length ? layers : ['manifold'],
        steps,
        grid,
      });
    } catch (err) {
      useVizStore.getState().setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <header className="run-controls">
      <span className="title">QFT-PCN Visualizer</span>
      <div className="run-layers">
        {LAYER_KEYS.map((layer) => (
          <label key={layer}>
            <input
              type="checkbox"
              checked={layers.includes(layer)}
              onChange={() => toggleLayer(layer)}
            />
            {layer}
          </label>
        ))}
      </div>
      <label>
        steps
        <input
          type="number"
          min={1}
          value={steps}
          onChange={(e) => setSteps(Number(e.target.value))}
        />
      </label>
      <label>
        grid
        <input
          type="number"
          min={1}
          value={grid}
          onChange={(e) => setGrid(Number(e.target.value))}
        />
      </label>
      <button type="button" onClick={run} disabled={busy}>
        {busy ? 'starting…' : 'Run'}
      </button>
    </header>
  );
}

function PanelArea() {
  const selectedLayer = useVizStore((s) => s.selectedLayer);
  const frame = useVizStore((s) => s.currentFrame());
  const layerState = frame?.layer_states[selectedLayer];

  return (
    <main className="panel-area">
      <h2>{selectedLayer}</h2>
      {layerState ? (
        <pre>{JSON.stringify(layerState, null, 2)}</pre>
      ) : (
        <p className="empty">No data for this layer at the current step.</p>
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
