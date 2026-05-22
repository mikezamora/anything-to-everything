/**
 * Frontend mirror of `src/qft_pcn/viz/schema.py`.
 *
 * A `Frame` is one captured simulation step: a step index plus a mapping from
 * layer name to a plain dict of JSON-ready diagnostic values.
 */

export interface Frame {
  step: number;
  layer_states: Record<string, Record<string, unknown>>;
}

/** Canonical per-layer keys — must stay in sync with `LAYER_KEYS` in schema.py. */
export const LAYER_KEYS = [
  'manifold',
  'multifield',
  'mps',
  'hamiltonian',
  'qpcn',
  'mera',
  'vqc',
  'logic',
] as const;

export type LayerKey = (typeof LAYER_KEYS)[number];

/** Request body for `POST /run`. */
export interface RunSpec {
  layers: string[];
  steps: number;
  grid: number;
}
