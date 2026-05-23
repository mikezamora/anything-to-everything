/** Frontend mirror of `src/qft_pcn/viz/schema.py`. */

export interface Frame {
  step: number;
  layer_states: Record<string, Record<string, unknown>>;
}

export const LAYER_KEYS = [
  'manifold', 'multifield', 'mps', 'hamiltonian',
  'qpcn', 'mera', 'vqc', 'logic',
] as const;
export type LayerKey = (typeof LAYER_KEYS)[number];

export interface RunSpec {
  layers: string[];
  steps: number;
  grid: number;
  seed?: number | null;
  params?: Record<string, Record<string, unknown>>;
}

export interface Preset {
  id: string;
  layer: string;
  label: string;
  description: string;
  spec_overrides: Partial<RunSpec> & {
    params?: Record<string, Record<string, unknown>>;
  };
}

export type ParamSchema = Record<string, {
  type: 'object';
  properties: Record<string, {
    type: string | string[];
    default?: unknown;
    enum?: unknown[];
    minimum?: number;
    maximum?: number;
    description?: string;
    items?: { type: string };
  }>;
}>;
