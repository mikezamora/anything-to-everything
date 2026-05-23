/** Frontend mirror of `src/qft_pcn/viz/schema.py`. */

export interface Frame {
  step: number;
  layer_states: Record<string, Record<string, unknown>>;
}

export const LAYER_KEYS = [
  'manifold', 'multifield', 'mps', 'hamiltonian',
  'qpcn', 'mera', 'vqc', 'logic',
  'pcn-fields', 'pcn-dynamics', 'pcn-coupling',
  'mera_relax', 'bridge',
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

export type Route = 'viz' | 'dsl';

export interface LlmModel {
  name: string;
  size?: number;
  modified_at?: string;
}

export interface ChatTurn {
  role: 'user' | 'assistant';
  text: string;
  /** Optional artifact attached to an assistant turn (e.g. emitted DSL). */
  artifact?: { kind: 'dsl' | 'run' | 'verbalize'; payload: unknown };
}

export interface DslSpec {
  fields: Array<{ name: string; cutoff: number;
                  bare_mass?: number; kinetic?: number }>;
  hamiltonian: { terms: Array<{ kind: string; species?: string;
                                 site?: number; coefficient?: number }> };
  observables: Array<{ operator: string; site?: number;
                       species?: string; target?: number | null }>;
  run?: { steps?: number; chi_max?: number; seed?: number | null };
}
