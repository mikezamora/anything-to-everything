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

  mera_relax: {
    title: 'MERA imag-time relax — §10.10 induction-theorem demo',
    oneLine:
      'Live ∀-protected MERA relaxation: residuals decay while bound leaves stay frozen.',
    what: [
      'Encode an extended-calculus AST (including Forall / Eq / Nat / Cons / Nil) into a MERA via `encode_mera`. The encoder annotates every leaf used as a bound-variable witness as ∀-protected — those leaves form `meta.forall_protected_leaves`.',
      'Imaginary-time evolution under `MeraEvalHamiltonian` (`mera_trotter_step(..., frozen_leaves=meta.forall_protected_leaves)`) drops any gate whose target intersects the protected set, so the ∀-bound leaves stay bitwise stable across the whole anneal. Residuals on R-AddZero + R-Eq-Refl decay toward zero, witnessing the induction theorem ∀x:Nat. x + 0 = x.',
    ],
    elements: [
      {
        name: 'total energy',
        meaning:
          '⟨H_eval⟩ summed across all 360 R-rule terms; should decay monotonically under imag-time relaxation (the relaxation invariant).',
        code: 'snapshot_mera_relax:total_energy',
      },
      {
        name: 'per-term residuals',
        meaning:
          'one row per (rule_id, site) with the local residual energy; rows are sorted hot-to-cold so active rules float to the top.',
        code: 'snapshot_mera_relax:residuals',
      },
      {
        name: 'ast_text (round-trip)',
        meaning:
          'pretty-printed AST decoded from the live MERA leaves each step; for the default preset the text contains "forall" throughout the run since the ∀-protected leaves are clamped.',
        code: 'mera_decoder.decode_mera',
      },
      {
        name: '∀-protected leaves',
        meaning:
          'leaf indices held bitwise stable by the trotter step (`frozen_leaves=`); the load-bearing §1.1 / §10.10 invariant. Must stay non-empty for any `forall …` proposition.',
        code: 'mera_encoder:forall_protected_leaves',
      },
    ],
    math: [
      {
        tex: 'H_\\text{eval} = \\sum_r \\lambda_r \\sum_i H_r^{(i)}',
        caption: 'Per-rule term sum, ground state = a well-typed program.',
      },
      {
        tex: '|\\psi(\\tau+d\\tau)\\rangle = e^{-H\\,d\\tau}\\,|\\psi(\\tau)\\rangle\\,\\Big|_{\\text{leaves} \\notin \\text{frozen}}',
        caption:
          'Imag-time evolution restricted to non-frozen leaves: gates touching ∀-protected leaves are dropped at dispatch.',
      },
    ],
    watch: [
      {
        label:
          'Total energy decreases monotonically under imag-time (the relaxation invariant)',
        readout: 'total_energy',
      },
      {
        label:
          'The ∀-protected leaf set stays non-empty and bitwise stable across the whole run',
      },
      {
        label:
          'AST text round-trips and continues to contain "forall" — the binder survives every Trotter step',
      },
    ],
    references: [
      { label: 'Architecture §10.10', href: '../../QFT_PCN_ARCHITECTURE.md' },
    ],
  },

  bridge: {
    title: 'Bridge — RunResult inspector',
    oneLine:
      'One-shot physics-DSL resolver; surfaces the resolved MPS, Hamiltonian and convergence.',
    what: [
      'The bridge runtime (`run_problem(dsl)`) validates a physics-DSL problem, compiles it into a Hamiltonian + initial MPS state, evolves it under imag-time with clamps, and measures observables — a one-shot resolver.',
      'This panel snapshots the resulting `RunResult` every frame: trotter step count, final energy, convergence flag, the ground-state MPS bond profile, and the composed Hamiltonian summary. `solved_ast` stays None until a MERA-based runner populates it.',
    ],
    elements: [
      {
        name: 'trotter steps',
        meaning: 'imag-time step count actually run by the resolver.',
        code: 'snapshot_run_result:trotter_steps',
      },
      {
        name: '⟨H⟩',
        meaning:
          'final variational energy under the composed Hamiltonian (after evolution).',
        code: 'snapshot_run_result:energy',
      },
      {
        name: 'ground-state MPS miniature',
        meaning:
          'compact view of `result.ground_state`: N, d_local, χ_max, and the per-bond χ profile.',
        code: 'snapshot_mps(result.ground_state)',
      },
      {
        name: 'Hamiltonian miniature',
        meaning:
          'compact view of `result.hamiltonian`: N, d_local, species list.',
        code: 'snapshot_hamiltonian(result.hamiltonian)',
      },
    ],
    math: [
      {
        tex: '|\\psi_\\infty\\rangle = \\lim_{\\tau\\to\\infty} \\frac{e^{-H\\tau}|\\psi_0\\rangle}{\\|e^{-H\\tau}|\\psi_0\\rangle\\|}',
        caption: 'Imag-time relaxation toward the resolver\'s ground state.',
      },
    ],
    watch: [
      { label: 'converged = yes once the energy-per-step settles within tol' },
      {
        label:
          'solved_ast is None on this MPS path — a MERA-based runner is required to populate it',
      },
    ],
    references: [
      { label: 'Bridge spec §5', href: '../../../bridge/' },
    ],
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

  'pcn-fields': {
    title: 'PCN Fields — Hierarchical Φ / E / Π Stack',
    oneLine: 'One card per PCN layer: belief Φ, prediction-error E, and precision Π fields.',
    what: [
      'A QFT-PCN network is a stack of predictive-coding layers; each layer carries three coupled 2D fields on the same manifold: belief Φ_l, prediction error E_l = Φ_{l-1} − g_l(Φ_l), and precision Π_l (inverse variance).',
      'This panel renders all layers at once so the user can see hierarchical structure forming: top layers carry coarse / abstract beliefs, bottom layers carry fast-changing error against the data, and precision concentrates where the model trusts itself.',
    ],
    elements: [
      { name: 'Per-layer heatmap', meaning: 'selected field (Φ / E / Π) for layer l, rendered as a diverging-colour-mapped grid; click to expand.', code: 'snapshot_pcn_fields:layers[l].{phi,E,Pi}' },
      { name: 'Field toggle (Φ / E / Π)', meaning: 'switches which field is plotted across all layer cards.' },
      { name: 'depth readout', meaning: 'number of PCN layers currently active.' },
      { name: '‖Φ‖₂ / ‖E‖₂', meaning: 'Frobenius norms summed across layers — global belief / error magnitude.' },
      { name: 'mean Π', meaning: 'average precision across all layers; rises as the network becomes more confident.' },
    ],
    math: [
      { tex: 'F[\\Phi,E,\\Pi] = \\int_M \\bigl[\\tfrac12 \\Pi(x) E(x)^2 - \\tfrac12 \\log \\Pi(x)\\bigr] \\sqrt{|g|}\\, d^2x', caption: 'Variational free energy minimised by the PCN (§3.1).' },
      { tex: 'E_l = \\Phi_{l-1} - g_l(\\Phi_l)', caption: 'Layer-l prediction error: bottom-up signal minus top-down prediction.' },
      { tex: '\\dot\\Phi_l = (J_{g_l})^\\top (\\Pi_l E_l) + D\\, \\Delta_g \\Phi_l + \\text{top-down}', caption: 'Belief flow: error-driven update + Laplace-Beltrami diffusion on the dynamic metric.' },
      { tex: '\\dot\\Pi_l = \\tfrac{1}{2\\Pi_l} - \\tfrac12 E_l^2 \\;\\;(\\text{fixed pt: } \\Pi_l = 1/E_l^2)', caption: 'Precision tracks inverse error variance.' },
    ],
    watch: [
      { label: 'depth readout matches the configured PCN stack height', readout: 'depth' },
      { label: '‖E‖₂ should decrease across the run as predictions improve' },
      { label: 'mean Π should grow in regions where E shrinks (high confidence = high precision)' },
      { label: 'Top-layer Φ stays smoother / coarser than bottom-layer Φ — the hierarchical signature' },
    ],
    references: [{ label: 'Architecture §2.1, §3.1', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  'pcn-dynamics': {
    title: 'PCN Dynamics — Free Energy + Per-Layer Trajectories',
    oneLine: 'Total variational free energy F and its per-layer contributions over time.',
    what: [
      'A QFT-PCN network minimises a single scalar objective: the variational free energy F summed across all PCN layers. This panel surfaces the live total F plus its per-layer decomposition so the user can see WHICH layer is dominating the cost at any moment.',
      'Each layer contributes a precision-weighted squared error (½ Π E²), an entropy correction (−½ log Π), and a curvature-coupling term (κR). The time-series strip on total F is the master "is the network learning?" readout.',
    ],
    elements: [
      { name: 'total F readout', meaning: 'sum of per-layer free energies; should decrease monotonically under successful learning.', code: 'snapshot_pcn_dynamics:total_free_energy' },
      { name: 'per-layer F row', meaning: 'free-energy contribution of layer l; isolates which layer is hot.', code: 'snapshot_pcn_dynamics:per_layer_free_energy[l]' },
      { name: 'per-layer ‖E‖₂', meaning: 'Frobenius norm of the layer-l prediction-error field — drives the ½ Π E² term.', code: 'snapshot_pcn_dynamics:per_layer_e_norm[l]' },
      { name: 'per-layer mean Π', meaning: 'average precision in layer l; rises as the layer becomes confident.', code: 'snapshot_pcn_dynamics:per_layer_pi_mean[l]' },
      { name: 'depth readout', meaning: 'number of PCN layers currently contributing.' },
      { name: 'MetricsStrip (total F)', meaning: 'time series of total F across the run — the primary convergence diagnostic.' },
    ],
    math: [
      { tex: 'F = \\sum_l \\int_M \\bigl[\\tfrac12 \\Pi_l(x) E_l(x)^2 - \\tfrac12 \\log \\Pi_l(x) + \\kappa R(x)\\bigr] \\sqrt{|g|}\\, d^2x', caption: 'Variational free energy: per-layer precision-weighted error + entropy correction + curvature-coupling (§3.1).' },
      { tex: 'F_l = \\tfrac12 \\langle \\Pi_l E_l^2 \\rangle - \\tfrac12 \\langle \\log \\Pi_l \\rangle + \\kappa \\langle R \\rangle', caption: 'Per-layer free-energy decomposition surfaced as a row in the panel table.' },
      { tex: '\\dot F = \\sum_l \\bigl(\\partial_{\\Phi_l} F\\, \\dot\\Phi_l + \\partial_{\\Pi_l} F\\, \\dot\\Pi_l\\bigr) \\le 0', caption: 'Belief/precision flow is gradient descent on F — the relaxation invariant.' },
    ],
    watch: [
      { label: 'total F decreases monotonically under successful learning', readout: 'total_free_energy' },
      { label: 'Per-layer F rows reveal which layer is the current bottleneck (largest contribution)' },
      { label: '‖E‖₂ should decay as predictions improve; mean Π should rise where E shrinks' },
      { label: 'depth readout matches the configured PCN stack height', readout: 'depth' },
    ],
    references: [{ label: 'Architecture §3.1', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  'pcn-coupling': {
    title: 'PCN ↔ QFT — Bidirectional Coupling',
    oneLine: 'The bridge: PCN error sources QFT metric (T_μν → h_μν); QFT operator expectations feed PCN observation targets (⟨Ô⟩).',
    what: [
      'A QFT-PCN system is not two independent stacks: the PCN\'s prediction-error field sources a stress-energy tensor T_μν that perturbs the QFT-side metric (g_μν = η_μν + h_μν), and the QFT-side operator expectations ⟨Ô⟩ flow back as observation targets the PCN must explain.',
      'This panel surfaces the live magnitudes on both arrows of that bridge — top arrow (PCN → QFT) widths track mean |T|; bottom arrow (QFT → PCN) widths track ⟨H⟩, a proxy for "how hard QFT is pulling the PCN\'s observation targets." κ_R is the coupling constant that gates how strongly error sources curvature.',
    ],
    elements: [
      { name: 'PCN ↔ QFT bridge diagram', meaning: 'inline SVG with PCN and QFT boxes; the two arrows between them animate with live coupling magnitudes.' },
      { name: 'PCN → QFT arrow (T_μν)', meaning: 'top arrow; stroke-width scales with mean |stress-energy|. The §3 source term for the metric perturbation.', code: 'snapshot_pcn_coupling:mean_abs_stress_energy' },
      { name: 'QFT → PCN arrow (⟨Ô⟩)', meaning: 'bottom arrow; stroke-width scales with the QPCN variational energy — proxy for the operator-expectation feedback into PCN observation targets.', code: 'snapshot_pcn_coupling:qpcn_observable_energy' },
      { name: 'κ_R readout', meaning: 'curvature-coupling constant gating how strongly PCN error sources the QFT-side curvature.', code: 'snapshot_pcn_coupling:kappa_R' },
      { name: 'mean |T| readout', meaning: 'mean absolute stress-energy derived from the PCN error field; the source magnitude of the metric perturbation.', code: 'snapshot_pcn_coupling:mean_abs_stress_energy' },
      { name: 'mean |R| readout', meaning: 'mean absolute Ricci scalar on the QFT-side metric — the geometric response to T_μν.', code: 'snapshot_pcn_coupling:mean_abs_ricci' },
      { name: '⟨H⟩ readout', meaning: 'QPCN variational energy ⟨ψ|H|ψ⟩; stands in for ⟨Ô⟩ — the operator-expectation feedback that PCN tries to match.', code: 'snapshot_pcn_coupling:qpcn_observable_energy' },
    ],
    math: [
      { tex: 'g_{\\mu\\nu}(x) = \\eta_{\\mu\\nu} + h_{\\mu\\nu}(x)', caption: 'QFT-side metric = flat background + perturbation sourced by PCN error.' },
      { tex: 'h_{\\mu\\nu} \\propto \\kappa_R\\, T_{\\mu\\nu}[E_l]', caption: 'PCN → QFT: stress-energy of the prediction-error field sources the metric perturbation (κ_R gates the coupling).' },
      { tex: 't_o = \\langle \\hat O \\rangle_{|\\psi\\rangle}', caption: 'QFT → PCN: operator expectations on the QPCN ground state become observation targets the PCN must explain (§3, bidirectional bridge).' },
    ],
    watch: [
      { label: 'Top arrow widens as PCN error grows (large mean |T| ⇒ strong QFT source)', readout: 'mean_abs_stress_energy' },
      { label: 'mean |R| should track mean |T| × κ_R — the geometric response to the source' },
      { label: 'Bottom arrow widens as ⟨H⟩ grows: QFT is pulling harder on PCN observation targets' },
      { label: 'Both arrows should shrink together as the joint system relaxes (error decays AND QPCN converges)' },
    ],
    references: [{ label: 'Architecture §3', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },
};
