// src/qft_pcn/viz/web/src/lib/explainer.ts
/**
 * Static per-layer explainer content.
 *
 * Sourced from QFT_PCN_ARCHITECTURE.md (anchors noted in `references`).
 * `watch` items may carry a `readout` id; the matching `<PanelReadouts>`
 * cell will flash on hover.
 */

export interface ExplainerSpec {
  title: string;
  oneLine: string;
  what: string[];
  elements: { name: string; meaning: string; code?: string }[];
  math: { tex: string; caption: string }[];
  watch: { label: string; readout?: string }[];
  references?: { label: string; href: string }[];
}

export const EXPLAINERS: Record<string, ExplainerSpec> = {
  manifold: {
    title: 'Manifold — Dynamic Riemannian Geometry',
    oneLine: 'The 2D base manifold whose metric is sourced by prediction error.',
    what: [
      'The classical predictive-coding layer sits on a 2D Riemannian manifold whose metric g_μν is not fixed: it is sourced by the stress-energy tensor of the prediction-error field.',
      'Belief diffusion uses the Laplace-Beltrami operator built from that metric — so geometry follows what the system is uncertain about.',
    ],
    elements: [
      { name: 'Surface height', meaning: 'selected metric-perturbation component: h_xx (default), h_xy (shear), h_yy, or tr(h) = h_xx + h_yy. The full h_μν is a rank-2 symmetric tensor (§3.2); a single scalar height cannot represent it, so the toolbar exposes all three plus the mean-curvature proxy.', code: 'manifold.py:h_xx / h_xy / h_yy' },
      { name: 'Surface colour', meaning: 'Ricci scalar curvature R', code: 'manifold.py:ricci_scalar' },
      { name: 'Overlay (Φ / E / Π)', meaning: 'Belief / error / precision fields of the first PCN layer' },
    ],
    math: [
      { tex: 'g_{\\mu\\nu}(x) = \\eta_{\\mu\\nu} + h_{\\mu\\nu}(x)', caption: 'Metric = flat + learned perturbation.' },
      { tex: 'R = g^{\\mu\\nu} R_{\\mu\\nu}', caption: 'Ricci scalar drives the colour map.' },
      { tex: '\\Delta_g \\Phi = \\frac{1}{\\sqrt{|g|}} \\partial_\\mu (\\sqrt{|g|}\\, g^{\\mu\\nu} \\partial_\\nu \\Phi)', caption: 'Belief diffuses via Laplace-Beltrami on g.' },
    ],
    watch: [
      { label: 'Curvature concentrates where error spikes', readout: 'mean_abs_ricci' },
      { label: 'Mean |R| should stabilise after error decays' },
      { label: 'Overlay Φ — peaks track regions the model is confidently predicting' },
    ],
    references: [{ label: 'Architecture §2.1, §3.1', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  multifield: {
    title: 'Multi-field — Coupled Field Species',
    oneLine: 'Multiple field species sharing one manifold, coupled via learnable g_ij.',
    what: [
      'Multiple field species (analogous to electron / photon / Higgs, or shape / motion / colour) live on the same manifold and interact through a Yukawa-style Lagrangian.',
      'Coupling constants g_ij are themselves learnable: correlated fields grow their coupling; uncorrelated fields stay decoupled.',
    ],
    elements: [
      { name: '3D surface per field', meaning: 'belief Φ for that species' },
      { name: 'Coupling graph / matrix', meaning: 'live g_ij entries (≥3 fields ⇒ matrix view); edge stroke uses a diverging ramp so positive and negative couplings are visually distinct', code: 'multifield.py:couplings' },
      { name: 'Per-pair g[(a,b)] cells', meaning: 'signed current value per pair — the architectural diagnostic §2.2 / §4.5; correlated pairs grow, uncorrelated stay near zero', code: 'snapshot_multifield:couplings' },
      { name: 'Per-pair MetricsStrip traces', meaning: 'time series of each g_{ij} so the user can see WHICH pairs are converging' },
      { name: 'Mean |g|', meaning: 'aggregate coupling strength (legacy scalar — hides per-pair structure)', code: 'snapshot_multifield:mean_abs_coupling' },
    ],
    math: [
      { tex: 'L_\\text{int} = \\sum_{i<j} g_{ij}(x)\\, \\Phi_i(x)\\, \\Phi_j(x)', caption: 'Yukawa interaction Lagrangian.' },
      { tex: '\\dot g_{ij} = -\\eta \\frac{\\partial F}{\\partial g_{ij}}', caption: 'Couplings descend the joint free energy.' },
    ],
    watch: [
      { label: 'Mean |g| rises when fields co-vary', readout: 'mean_abs_coupling' },
      { label: 'Surfaces with no shared structure stay near-flat in coupling' },
    ],
    references: [{ label: 'Architecture §2.2', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  mps: {
    title: 'MPS — Matrix Product State',
    oneLine: 'The entanglement carrier: a 1D chain of low-rank tensors.',
    what: [
      'The QPCN\'s belief state is represented as a Matrix Product State with controllable bond dimension χ_max — this is the entanglement carrier.',
      'Bond dimension caps the entanglement entropy across each cut; SVD truncation enforces it after every Trotter step.',
    ],
    elements: [
      { name: 'Bond-dim bar', meaning: 'χ for each bond between sites', code: 'mps.py:bond_dimensions' },
      { name: 'Entropy line', meaning: 'per-cut von Neumann entanglement entropy' },
      { name: 'Page-curve reference', meaning: 'random-state entropy ceiling (visual guide)' },
    ],
    math: [
      { tex: '|\\psi\\rangle = \\sum_{\\{s\\}} A^{s_1} A^{s_2} \\cdots A^{s_N} |s_1 \\ldots s_N\\rangle', caption: 'MPS ansatz.' },
      { tex: 'S(\\rho_A) = -\\mathrm{Tr}\\, \\rho_A \\log \\rho_A', caption: 'Cut entropy follows from the bipartition.' },
    ],
    watch: [
      { label: 'Bonds saturate to χ_max under entangling dynamics' },
      { label: 'Entropy clusters in the middle for short-range Hamiltonians' },
    ],
    references: [{ label: 'Architecture §2.3', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  hamiltonian: {
    title: 'Hamiltonian — Local Many-Body Operator',
    oneLine: 'Sum of one- and two-site terms over species; the QPCN\'s generative model.',
    what: [
      'The Hamiltonian H is the QPCN\'s generative model. It is built locally from one-site terms (mass, source, quartic, curvature coupling) and two-site terms (kinetic hopping, cross-species interaction).',
      'Each species has a Fock cutoff d_local. The curvature field is consumed by the manifold-coupling term.',
    ],
    elements: [
      { name: 'Per-species table', meaning: 'ω (bare_mass), t (kinetic), μ (quartic), J (source) per field species', code: 'hamiltonian.py:FieldSpecies' },
      { name: 'Coupling matrix', meaning: 'g_{ab} (density-density) or λ_{ab} (Yukawa-like field-field), toggleable', code: 'HamiltonianConfig.density_couplings / yukawa_couplings' },
      { name: 'Curvature strip', meaning: '1D R(x_k) per MPS site, modulating ω via ξ', code: 'Hamiltonian.curvature' },
    ],
    math: [
      { tex: 'H = \\sum_i \\bigl[\\omega_i n_i + J_i \\phi_i + \\mu_i n_i^2 + g_{ab} n_a n_b + \\lambda_{ab} \\phi_a \\phi_b\\bigr] - \\sum_{\\langle i,j \\rangle} t\\, (a_i^\\dagger a_j + \\text{h.c.})', caption: 'Site/bond decomposition with all §3.3.4 terms.' },
      { tex: '\\omega_i \\to \\omega_i (1 + \\xi R(x_i))', caption: 'Curvature couples to the local mass via ξ.' },
    ],
    watch: [
      { label: 'd_local equals ∏ species cutoffs — larger d means richer dynamics' },
      { label: 'Curvature peaks should correlate with the manifold panel curvature' },
      { label: 'Live param edits in QPCN show up in this panel\'s ω / t row' },
    ],
    references: [{ label: 'Architecture §2.3, §3', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  qpcn: {
    title: 'QPCN — Quantum Predictive Coder',
    oneLine: 'Belief = MPS; generative model = H; errors drive parameter updates.',
    what: [
      'Imaginary-time evolution relaxes the MPS toward the ground state of the current Hamiltonian; observations are target expectation values of local operators.',
      'Prediction errors (target − ⟨obs⟩) drive gradient updates on learnable Hamiltonian parameters.',
    ],
    elements: [
      { name: 'Energy', meaning: '⟨ψ|H|ψ⟩, the variational energy' },
      { name: 'Pred-errors table', meaning: 'per-observable (target, current, Δ)', code: 'qpcn.py:_last_errors' },
      { name: 'Learnable params', meaning: 'mass / kinetic / coupling values that descend' },
      { name: '⟨n_k⟩ per site', meaning: 'real particle occupation per species per site (state.local_expectation(k, H.n(s)))', code: 'snapshot_qpcn:occupations_n' },
    ],
    math: [
      { tex: '|\\psi(\\tau+d\\tau)\\rangle = e^{-H\\, d\\tau} |\\psi(\\tau)\\rangle', caption: 'Imaginary-time relaxation.' },
      { tex: '\\Delta \\theta = -\\eta\\, \\partial_\\theta \\sum_o (\\langle O \\rangle - t_o)^2', caption: 'Parameter update from observation errors.' },
    ],
    watch: [
      { label: 'Energy decreases monotonically under imaginary time', readout: 'energy' },
      { label: 'Pred-errors shrink as parameters adapt', readout: 'pred_errors' },
    ],
    references: [{ label: 'Architecture §2.3', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  mera: {
    title: 'MERA — Multi-Scale Tree',
    oneLine: 'Tree of disentanglers + isometries; encodes scale structure.',
    what: [
      'A MERA represents a quantum state as a renormalisation-group tree: each layer applies disentanglers (to remove short-range entanglement) AND a 2->1 isometry (to coarse-grain) WITHIN the same RG step; the panel surfaces one isometry-glyph per coarse layer, matching what `snapshot_mera` exposes.',
      'Used here as a substrate for hierarchical reasoning; the panel shows tree shape, per-layer χ, per-cut entropy, and the per-layer isometry-violation residual.',
    ],
    elements: [
      { name: 'Tree nodes', meaning: 'leaves on the boundary circle; every coarse node is a 2->1 isometry (disentanglers exist on the substrate but are not drawn as separate nodes — see D-6)' },
      { name: 'Per-layer χ', meaning: 'bond dimension at each level' },
      { name: 'Entropy line', meaning: 'entanglement at each leaf-cut' },
      { name: 'iso err sparkline', meaning: 'per-layer mean ‖W W† − I‖_F (isometry-condition residual)' },
    ],
    math: [
      { tex: '|\\psi\\rangle = \\prod_{\\ell=1}^{L} W_\\ell\\, U_\\ell\\, |0\\rangle', caption: 'Each RG layer ℓ contains BOTH a disentangler U_ℓ AND an isometry W_ℓ.' },
    ],
    watch: [
      { label: 'Logarithmic entropy scaling on critical states' },
      { label: 'Per-layer χ caps the captured entanglement' },
    ],
    references: [{ label: 'Architecture: MERA section', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  vqc: {
    title: 'VQC — Variational Quantum Circuit',
    oneLine: 'Parameterised circuit trained via parameter-shift gradients.',
    what: [
      'A `QuantumConvMap` wraps a parameterised circuit as a translation-invariant quantum convolution that drops into any PCN layer.',
      'Gradients are exact via the parameter-shift rule. The same code targets real IBM hardware.',
    ],
    elements: [
      { name: 'Theta heatmap', meaning: 'rotation angles per (layer, qubit, axis)' },
      { name: 'Bias', meaning: 'classical bias on the conv output' },
    ],
    math: [
      { tex: '\\partial_\\theta \\langle O \\rangle = \\tfrac{1}{2}[\\langle O \\rangle_{\\theta + \\pi/2} - \\langle O \\rangle_{\\theta - \\pi/2}]', caption: 'Parameter-shift rule.' },
    ],
    watch: [{ label: 'Live training is currently a fixture — see EXTENSIONS.md' }],
    references: [{ label: 'Architecture §2.4', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  logic: {
    title: 'Logic — Evaluation Hamiltonian',
    oneLine: 'Rule terms encoded as Hamiltonian costs; relaxation = reduction.',
    what: [
      'Each rule of the target calculus contributes a local Hamiltonian term; the ground state corresponds to a well-typed / fully-reduced program.',
      'Imaginary-time relaxation drives a superposition state toward zero residual energy under all rules simultaneously.',
    ],
    elements: [
      { name: 'Term nodes', meaning: 'per-rule (rule_id, site, arity); fill opacity tracks per-term residual energy so a relaxed (satisfied) term reads pale, a high-residual term reads saturated', code: 'snapshot_logic:terms / residuals' },
      { name: 'Binder bond entropy chart', meaning: 'per-bond von Neumann entropy on the logic-encoded MPS — the load-bearing §1.1 "variable binding = entanglement" signal; a binder live on a bond contributes entropy across the use→declaration path', code: 'snapshot_logic:bond_entropies' },
      { name: 'λ legend', meaning: 'global term-weight scalars (λ_β / λ_arith / λ_if). These describe relative term weights only, NOT binder geometry.' },
      { name: 'Total energy', meaning: 'sum of all residual term energies' },
    ],
    math: [
      { tex: 'H_\\text{eval} = \\sum_r \\lambda_r \\sum_i H_r^{(i)}', caption: 'Sum of per-rule terms with weights λ.' },
      { tex: 'S(\\rho_A) = -\\mathrm{Tr}\\, \\rho_A \\log \\rho_A', caption: 'Bond entropy = real entanglement across the cut; non-zero where a binder is live.' },
    ],
    watch: [
      { label: 'Term node opacity drops as residual energy decays (relaxation)' },
      { label: 'Binder bond entropy spikes mark live binders — the §1.1 invariant' },
    ],
    references: [{ label: 'Architecture: logic section', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },
};
