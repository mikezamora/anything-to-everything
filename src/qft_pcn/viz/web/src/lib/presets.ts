import type { Preset, ParamSchema } from './types';

let _presets: Preset[] | null = null;
let _schema: ParamSchema | null = null;

export async function loadPresets(): Promise<Preset[]> {
  if (_presets) return _presets;
  const res = await fetch('/presets');
  if (!res.ok) throw new Error(`/presets: ${res.status}`);
  _presets = await res.json();
  return _presets!;
}

export async function loadParamSchema(): Promise<ParamSchema> {
  if (_schema) return _schema;
  const res = await fetch('/params/schema');
  if (!res.ok) throw new Error(`/params/schema: ${res.status}`);
  _schema = await res.json();
  return _schema!;
}

/** Reset module-level caches — test helper. */
export function _resetPresetCache() { _presets = null; _schema = null; }

export async function exportRunJsonl(runId: string): Promise<void> {
  const res = await fetch(`/export/run/${runId}`);
  if (!res.ok) throw new Error(`export: ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `run-${runId}.jsonl`;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

export async function exportRunMp4(runId: string, layer: string)
    : Promise<{ job_id: string }> {
  const res = await fetch('/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: runId, layer }),
  });
  if (!res.ok) throw new Error(`/export: ${res.status}`);
  return res.json();
}
