/**
 * Hand-written sample `Frame`s, one per layer, representative of the dict
 * shapes that the `snapshot_*` functions in `src/qft_pcn/viz/snapshots.py`
 * produce. The overall structure is accurate, but some param/species key
 * names are illustrative rather than exact (e.g. `qpcnFrame.params` uses
 * `mass/coupling/hopping` whereas live keys look like `A.mass`).
 * Used by panel smoke tests and as a manual reference for panel authors.
 */

import type { Frame } from '../../lib/types';

/** Small helper: a g x g grid filled by `fn(x, y)`. */
function grid(g: number, fn: (x: number, y: number) => number): number[][] {
  return Array.from({ length: g }, (_, y) =>
    Array.from({ length: g }, (_, x) => fn(x, y)),
  );
}

const ripple = (x: number, y: number) =>
  Math.sin(x * 0.6) * Math.cos(y * 0.6);

// --- manifold -> snapshot_network -------------------------------------------
export const manifoldFrame: Frame = {
  step: 7,
  layer_states: {
    manifold: {
      metric_h: {
        h_xx: grid(8, (x, y) => 0.2 * ripple(x, y)),
        h_xy: grid(8, (x, y) => 0.05 * ripple(x + 1, y)),
        h_yy: grid(8, (x, y) => 0.2 * ripple(x, y + 1)),
      },
      ricci: grid(8, (x, y) => ripple(x, y)),
      fields: [
        {
          phi: grid(8, ripple),
          E: grid(8, (x, y) => 0.4 * ripple(x + 2, y)),
          Pi: grid(8, () => 1.0),
          channels: 1,
        },
      ],
      mean_abs_ricci: 0.42,
      step: 7,
    },
  },
};

// --- multifield -> snapshot_multifield --------------------------------------
export const multifieldFrame: Frame = {
  step: 3,
  layer_states: {
    multifield: {
      fields: {
        alpha: {
          phi: grid(8, ripple),
          E: grid(8, (x, y) => 0.3 * ripple(x, y)),
          Pi: grid(8, () => 1.0),
        },
        beta: {
          phi: grid(8, (x, y) => ripple(x + 3, y)),
          E: grid(8, (x, y) => 0.3 * ripple(x + 3, y)),
          Pi: grid(8, () => 0.8),
        },
        gamma: {
          phi: grid(8, (x, y) => ripple(x, y + 3)),
          E: grid(8, (x, y) => 0.3 * ripple(x, y + 3)),
          Pi: grid(8, () => 1.2),
        },
      },
      couplings: {
        'alpha|beta': 0.6,
        'beta|gamma': -0.3,
        'alpha|gamma': 0.15,
      },
      mean_abs_coupling: 0.35,
      step: 3,
    },
  },
};

// --- mps -> snapshot_mps -----------------------------------------------------
export const mpsFrame: Frame = {
  step: 12,
  layer_states: {
    mps: {
      bond_dims: [1, 2, 4, 4, 2, 1],
      entropies: [0.0, 0.31, 0.62, 0.45, 0.12],
      n_sites: 6,
      d_local: 2,
    },
  },
};

// --- hamiltonian -> snapshot_hamiltonian ------------------------------------
export const hamiltonianFrame: Frame = {
  step: 5,
  layer_states: {
    hamiltonian: {
      n_sites: 4,
      d_local: 4,
      species_dims: [2, 2],
      species: ['scalar', 'gauge'],
      per_species: {
        scalar: { bare_mass: 1.2, kinetic: 0.5, quartic: 0.1, source: 0.05 },
        gauge: { bare_mass: 0.8, kinetic: 0.3, quartic: 0.0, source: 0.0 },
      },
      density_couplings: { 'gauge|scalar': 0.15 },
      yukawa_couplings: { 'gauge|scalar': -0.07 },
      curvature_xi: 0.4,
      // 1D per-site R(x_k) — NOT a 2D matrix per §3.3.4.
      curvature: [0.0, 0.2, -0.1, 0.05],
    },
  },
};

// --- qpcn -> snapshot_qpcn ---------------------------------------------------
export const qpcnFrame: Frame = {
  step: 9,
  layer_states: {
    qpcn: {
      energy: -1.732,
      pred_errors: { phi: 0.21, E: 0.08, Pi: 0.03 },
      params: { mass: 1.04, coupling: 0.37, hopping: -0.51 },
      bond_dims: [1, 2, 4, 2, 1],
      entropies: [0.0, 0.4, 0.55, 0.18],
      occupations_n: { phi: [0.12, 0.31, 0.74, 0.42, 0.18] },
      tensor_norms: [1.0, 0.97, 1.02, 0.99, 1.0],
      step: 9,
    },
  },
};

// --- mera -> snapshot_mera ---------------------------------------------------
export const meraFrame: Frame = {
  step: 4,
  layer_states: {
    mera: {
      n_leaves: 8,
      layer_dims: [2, 4, 4],
      bond_dims: [2, 3, 4],
      entropies: [0.1, 0.3, 0.5, 0.62, 0.5, 0.3, 0.1],
      iso_residuals: [1.2e-7, 3.4e-6, 8.9e-5],
    },
  },
};

// --- vqc -> snapshot_vqc -----------------------------------------------------
export const vqcFrame: Frame = {
  step: 6,
  layer_states: {
    vqc: {
      theta: [
        [
          [0.1, 0.2],
          [0.3, 0.4],
          [0.5, 0.6],
        ],
        [
          [0.7, 0.8],
          [0.9, 1.0],
          [1.1, 1.2],
        ],
      ],
      n_qubits: 3,
      n_layers: 2,
      input_scale: 1.0,
      bias: [0.05, -0.05, 0.1],
    },
  },
};

// --- logic -> snapshot_logic -------------------------------------------------
export const logicFrame: Frame = {
  step: 2,
  layer_states: {
    logic: {
      n_sites: 7,
      term_count: 12,
      terms: [
        { rule_id: 'R-Beta', site: 0, arity: 2 },
        { rule_id: 'R-Beta', site: 3, arity: 2 },
        { rule_id: 'R-Arith-Pre', site: 1, arity: 2 },
        { rule_id: 'R-Arith-Post', site: 2, arity: 2 },
        { rule_id: 'R-Cmp-Pre', site: 4, arity: 2 },
        { rule_id: 'R-If', site: 5, arity: 2 },
      ],
      lambda_beta: 1.0,
      lambda_arith: 0.5,
      lambda_if: 0.75,
    },
  },
};

/** All fixtures keyed by layer name — handy for the registry test. */
export const fixtureFrames: Record<string, Frame> = {
  manifold: manifoldFrame,
  multifield: multifieldFrame,
  mps: mpsFrame,
  hamiltonian: hamiltonianFrame,
  qpcn: qpcnFrame,
  mera: meraFrame,
  vqc: vqcFrame,
  logic: logicFrame,
};

/** A Frame whose layer states are all empty — exercises empty-state paths. */
export const emptyFrame: Frame = {
  step: 0,
  layer_states: {
    manifold: {},
    multifield: {},
    mps: {},
    hamiltonian: {},
    qpcn: {},
    mera: {},
    vqc: {},
    logic: {},
  },
};
