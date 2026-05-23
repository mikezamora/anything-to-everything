/** Client for /dsl/models with a tiny cache. */

import type { LlmModel } from './types';

let _models: LlmModel[] | null = null;

export async function loadModels(): Promise<LlmModel[]> {
  if (_models) return _models;
  const r = await fetch('/dsl/models');
  if (r.status === 503) throw new Error('Ollama unreachable');
  if (!r.ok) throw new Error(`/dsl/models: ${r.status}`);
  _models = await r.json() as LlmModel[];
  return _models!;
}

/** Default-pick the first gemma-prefixed model, else first model, else null. */
export function defaultModel(models: LlmModel[]): string | null {
  if (!models.length) return null;
  const gemma = models.find((m) => m.name.toLowerCase().startsWith('gemma'));
  return (gemma ?? models[0]).name;
}

/** Reset cache — test helper. */
export function _resetLlmCache() { _models = null; }
