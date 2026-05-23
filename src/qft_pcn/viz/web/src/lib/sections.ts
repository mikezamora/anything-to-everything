// src/qft_pcn/viz/web/src/lib/sections.ts
/**
 * Three top-level narrative sections grouping the 11 layers into:
 * QFT Substrate, PCN Substrate, QPCN Fusion.
 *
 * Content sourced from QFT_PCN_ARCHITECTURE.md sections §2.1 / §2.2 / §2.3.
 * Used by SectionedLayerSelector + Intro panels.
 */

export interface SectionSpec {
  key: 'qft' | 'pcn' | 'qpcn';
  title: string;
  tagline: string;
  story: string[];
  layers: string[];
  layerSummaries: { layer: string; oneLine: string }[];
  references?: { label: string; href: string }[];
}

export const SECTIONS: SectionSpec[] = [
  {
    key: 'qft',
    title: 'QFT Substrate',
    tagline: 'Operator-valued fields on a tensor-network state.',
    story: [
      'The QFT side carries the system\'s quantum content. Each spatial point holds an operator-valued bosonic field with a truncated Fock space; the many-body state is a Matrix Product State (MPS) with controllable bond dimension.',
      'The Hamiltonian is built locally from one-site and two-site terms. Evolution uses second-order Suzuki-Trotter splitting with SVD-based bond truncation — real-time for unitary dynamics, imaginary-time for relaxation.',
      'MERA layers add multi-scale structure; the VQC panel shows a variational quantum circuit that can drop into any PCN layer as a generative map.',
    ],
    layers: ['mps', 'mera', 'vqc', 'hamiltonian'],
    layerSummaries: [
      { layer: 'mps', oneLine: 'The entanglement carrier — 1D chain of low-rank tensors.' },
      { layer: 'mera', oneLine: 'Multi-scale tree of disentanglers + isometries.' },
      { layer: 'vqc', oneLine: 'Parameterized quantum circuit trained via parameter-shift gradients.' },
      { layer: 'hamiltonian', oneLine: 'Local many-body operator — the QPCN\'s generative model.' },
    ],
    references: [{ label: 'Architecture §2.3', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },
  {
    key: 'pcn',
    title: 'PCN Substrate',
    tagline: 'Hierarchical predictive coding on a dynamic Riemannian manifold.',
    story: [
      'The PCN side is the classical Friston/Bogacz-style predictive-coding network. Belief fields Φ, error fields E, and precision fields Π live on a 2D manifold whose metric is itself dynamic.',
      'Each hierarchical layer compares its downward prediction to the layer below; errors propagate up, predictions propagate down. The variational free energy F = ½ Π E² − ½ log Π + κ R minimizes jointly with respect to beliefs, precisions, and the metric.',
      'Multi-field setups let several field types share one manifold, coupled via learnable Yukawa-style couplings g_ij — correlated fields grow their coupling; uncorrelated fields don\'t.',
    ],
    layers: ['pcn-fields', 'pcn-dynamics', 'multifield'],
    layerSummaries: [
      { layer: 'pcn-fields', oneLine: 'Φ / E / Π surfaces stacked across the full layer hierarchy.' },
      { layer: 'pcn-dynamics', oneLine: 'Free energy F, per-layer trajectories, learning rates.' },
      { layer: 'multifield', oneLine: 'Multiple field species coupled via learnable g_ij.' },
    ],
    references: [{ label: 'Architecture §2.1, §2.2', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },
  {
    key: 'qpcn',
    title: 'QPCN Fusion',
    tagline: 'How QFT + PCN compose into one learner.',
    story: [
      'The QPCN is the fusion layer: its belief state is the MPS, its generative model is the Hamiltonian, observations are target expectation values of local operators, and prediction errors drive gradient updates on Hamiltonian parameters.',
      'Stress-energy expectations from the QFT side feed back into the classical 2D manifold so geometry and quantum content are bidirectionally coupled — that\'s the load-bearing structural claim of the architecture.',
      'The Logic panel demonstrates the same machinery applied to symbolic reasoning: rule terms encoded as Hamiltonian costs whose ground state corresponds to a well-typed / fully-reduced program.',
    ],
    layers: ['pcn-coupling', 'qpcn', 'manifold', 'logic'],
    layerSummaries: [
      { layer: 'pcn-coupling', oneLine: 'The bidirectional bridge — stress-energy ↑, expectations ↓.' },
      { layer: 'qpcn', oneLine: 'Belief = MPS; generative model = H; errors drive parameter updates.' },
      { layer: 'manifold', oneLine: '2D Riemannian manifold whose metric is sourced by prediction error.' },
      { layer: 'logic', oneLine: 'Symbolic-reasoning Hamiltonian — relaxation = reduction.' },
    ],
    references: [{ label: 'Architecture §1.3, §2.3', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },
];

export function sectionForLayer(layer: string): SectionSpec | undefined {
  return SECTIONS.find((s) => s.layers.includes(layer));
}
