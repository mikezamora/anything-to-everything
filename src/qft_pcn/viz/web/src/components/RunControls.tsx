/**
 * Top run/transport bar: preset dropdown, Advanced expander, layer
 * checkboxes, transport (▶/⏸/⏭), playback speed slider, baseline pin,
 * Export menu.
 */

import { useEffect, useRef, useState } from 'react';
import { connectRun, type RunHandle } from '../lib/ws';
import { transport } from '../lib/transport';
import {
  loadPresets, loadParamSchema,
  exportRunJsonl, exportRunMp4,
} from '../lib/presets';
import type { Preset, ParamSchema, RunSpec } from '../lib/types';
import { LAYER_KEYS } from '../lib/types';
import { useVizStore } from '../store';

export function RunControls() {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [schema, setSchema] = useState<ParamSchema | null>(null);
  const [presetId, setPresetId] = useState<string>('');
  const [layers, setLayers] = useState<string[]>(['manifold']);
  const [steps, setSteps] = useState(20);
  const [grid, setGrid] = useState(12);
  const [seed, setSeed] = useState<number | ''>('');
  const [params, setParams] = useState<Record<string, Record<string, unknown>>>({});
  const [advanced, setAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const handleRef = useRef<RunHandle | null>(null);

  const active = useVizStore((s) => s.activeRunId);
  const baseline = useVizStore((s) => s.baselineRunId);
  const paused = useVizStore((s) => s.paused);
  const playbackSpeed = useVizStore((s) => s.playbackSpeed);
  const setPaused = useVizStore((s) => s.setPaused);
  const setSpeed = useVizStore((s) => s.setPlaybackSpeed);
  const pinBaseline = useVizStore((s) => s.pinBaseline);
  const unpinBaseline = useVizStore((s) => s.unpinBaseline);

  useEffect(() => {
    loadPresets().then(setPresets).catch((e) =>
      useVizStore.getState().setError(String(e)));
    loadParamSchema().then(setSchema).catch(() => { /* optional */ });
  }, []);

  useEffect(() => () => { handleRef.current?.close(); }, []);

  const applyPreset = (id: string) => {
    setPresetId(id);
    const p = presets.find((x) => x.id === id);
    if (!p) return;
    if (p.spec_overrides.layers) setLayers(p.spec_overrides.layers);
    else setLayers([p.layer]);
    if (p.spec_overrides.steps) setSteps(p.spec_overrides.steps);
    if (p.spec_overrides.grid) setGrid(p.spec_overrides.grid);
    if (p.spec_overrides.seed !== undefined && p.spec_overrides.seed !== null)
      setSeed(p.spec_overrides.seed);
    if (p.spec_overrides.params) setParams(p.spec_overrides.params);
  };

  const run = async () => {
    setBusy(true);
    handleRef.current?.close();
    const spec: RunSpec = {
      layers: layers.length ? layers : ['manifold'],
      steps, grid,
      seed: seed === '' ? null : Number(seed),
      params,
    };
    try {
      handleRef.current = await connectRun(spec);
    } catch (err) {
      useVizStore.getState().setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  const togglePause = async () => {
    if (!active) return;
    if (paused) { await transport.resume(active); setPaused(false); }
    else { await transport.pause(active); setPaused(true); }
  };
  const step = async () => { if (active) await transport.step(active); };

  return (
    <header className="run-controls">
      <span className="title">QFT-PCN Visualizer</span>

      <label>
        preset
        <select value={presetId} onChange={(e) => applyPreset(e.target.value)}>
          <option value="">(custom)</option>
          {presets.map((p) => (
            <option key={p.id} value={p.id}>{p.label}</option>
          ))}
        </select>
      </label>

      <div className="run-layers">
        {LAYER_KEYS.map((layer) => (
          <label key={layer}>
            <input type="checkbox" checked={layers.includes(layer)}
              onChange={() => setLayers((cur) =>
                cur.includes(layer)
                  ? cur.filter((l) => l !== layer)
                  : [...cur, layer])} />
            {layer}
          </label>
        ))}
      </div>

      <label>steps <input type="number" min={1} value={steps}
        onChange={(e) => setSteps(Number(e.target.value))} /></label>
      <label>grid <input type="number" min={1} value={grid}
        onChange={(e) => setGrid(Number(e.target.value))} /></label>
      <label>seed <input type="number" value={seed}
        onChange={(e) => setSeed(e.target.value === ''
          ? '' : Number(e.target.value))} /></label>

      <button type="button" onClick={() => setAdvanced((a) => !a)}>
        {advanced ? '▾ advanced' : '▸ advanced'}
      </button>

      <button type="button" onClick={run} disabled={busy}>
        {busy ? 'starting…' : 'Run'}
      </button>

      {/* Transport */}
      <div className="transport">
        <button type="button" onClick={togglePause} disabled={!active}>
          {paused ? '▶' : '⏸'}
        </button>
        <button type="button" onClick={step} disabled={!active || !paused}>⏭</button>
        <label>speed
          <input type="range" min={0.25} max={4} step={0.25}
            value={playbackSpeed}
            onChange={(e) => setSpeed(Number(e.target.value))} />
          <small>{playbackSpeed.toFixed(2)}×</small>
        </label>
      </div>

      {/* Baseline */}
      <button type="button" disabled={!active}
        onClick={() => active && (baseline === active
          ? unpinBaseline() : pinBaseline(active))}>
        {baseline === active ? 'unpin baseline' : 'pin as baseline'}
      </button>

      {/* Export */}
      <div className="export-menu">
        <button type="button" disabled={!active}
          onClick={() => active && exportRunJsonl(active)
            .catch((e) => useVizStore.getState().setError(String(e)))}>
          ⬇ JSONL
        </button>
        <button type="button" disabled={!active}
          onClick={() => active && exportRunMp4(active, layers[0] ?? 'manifold')
            .catch((e) => useVizStore.getState().setError(String(e)))}>
          🎞 MP4
        </button>
      </div>

      {advanced && schema && (
        <AdvancedForm schema={schema} value={params} onChange={setParams} />
      )}
    </header>
  );
}

function AdvancedForm({
  schema, value, onChange,
}: {
  schema: ParamSchema;
  value: Record<string, Record<string, unknown>>;
  onChange: (v: Record<string, Record<string, unknown>>) => void;
}) {
  return (
    <div className="advanced-form">
      {Object.entries(schema).map(([layer, layerSchema]) => (
        <fieldset key={layer}>
          <legend>{layer}</legend>
          {Object.entries(layerSchema.properties).map(([key, prop]) => {
            const current = value[layer]?.[key] ?? prop.default ?? '';
            const update = (raw: unknown) => onChange({
              ...value,
              [layer]: { ...(value[layer] ?? {}), [key]: raw },
            });
            const t = Array.isArray(prop.type) ? prop.type[0] : prop.type;
            if (prop.enum) {
              return (
                <label key={key}>{key}
                  <select value={String(current)}
                    onChange={(e) => update(e.target.value)}>
                    {prop.enum.map((o) =>
                      <option key={String(o)} value={String(o)}>{String(o)}</option>)}
                  </select>
                </label>
              );
            }
            if (t === 'boolean') {
              return (
                <label key={key}>{key}
                  <input type="checkbox" checked={Boolean(current)}
                    onChange={(e) => update(e.target.checked)} />
                </label>
              );
            }
            if (t === 'number' || t === 'integer') {
              return (
                <label key={key}>{key}
                  <input type="number"
                    value={current === null ? '' : Number(current)}
                    min={prop.minimum} max={prop.maximum}
                    onChange={(e) => update(e.target.value === ''
                      ? null : Number(e.target.value))} />
                </label>
              );
            }
            return (
              <label key={key}>{key}
                <input type="text" value={String(current)}
                  onChange={(e) => update(e.target.value)} />
              </label>
            );
          })}
        </fieldset>
      ))}
    </div>
  );
}
