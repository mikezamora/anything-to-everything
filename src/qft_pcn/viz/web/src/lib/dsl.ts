/**
 * Client for the /dsl/* backend endpoints.
 */

import type { DslSpec } from './types';

export async function fetchSchema(): Promise<unknown> {
  const r = await fetch('/dsl/schema');
  if (!r.ok) throw new Error(`/dsl/schema: ${r.status}`);
  return r.json();
}

export async function translate(prompt: string, model: string):
    Promise<{ dsl?: DslSpec; error?: string; raw?: string;
              validation_errors?: string[] }> {
  const r = await fetch('/dsl/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, model }),
  });
  if (!r.ok) throw new Error(`/dsl/translate: ${r.status}`);
  return r.json();
}

export async function runDsl(dsl: DslSpec): Promise<{ run_id: string }> {
  const r = await fetch('/dsl/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dsl }),
  });
  if (!r.ok) {
    const detail = await r.text();
    throw new Error(`/dsl/run: ${r.status} — ${detail}`);
  }
  return r.json();
}

export async function verbalize(observations: unknown, model: string,
                                originalPrompt: string):
    Promise<{ text: string }> {
  const r = await fetch('/dsl/verbalize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ observations, model,
                            original_prompt: originalPrompt }),
  });
  if (!r.ok) throw new Error(`/dsl/verbalize: ${r.status}`);
  return r.json();
}

export async function exportRunAsDsl(runId: string): Promise<DslSpec> {
  const r = await fetch(`/dsl/export/${runId}`);
  if (!r.ok) throw new Error(`/dsl/export: ${r.status}`);
  return r.json();
}
