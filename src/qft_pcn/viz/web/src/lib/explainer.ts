// src/qft_pcn/viz/web/src/lib/explainer.ts
/**
 * Static per-layer explainer content (tabbed).
 *
 * Sourced from QFT_PCN_ARCHITECTURE.md (anchors noted in `references`).
 * Each layer carries five tabs (Overview / Math / Worked Example /
 * Training Dynamics / Watch) consumed by `ExplainerPane`.
 * `watch` items may carry a `readout` id; the matching `<PanelReadouts>`
 * cell will flash on hover.
 *
 * `tabs.math.equationIds` is empty until Task 8 registers core equations
 * in `equations.ts` — the renderer shows a graceful "Equations not yet
 * registered" message in that case.
 */

import type { WorkedExample, Pathology } from './article-types';

export interface ExplainerSpec {
  title: string;
  oneLine: string;
  tabs: {
    overview: {
      what: string[];
      elements: { name: string; meaning: string; code?: string }[];
    };
    math: { equationIds: string[] };
    workedExample: WorkedExample;
    trainingDynamics: {
      updateRuleId: string;
      expect: string[];
      pathologies: Pathology[];
    };
    watch: { label: string; readout?: string }[];
  };
  references?: { label: string; href: string }[];
}

export const EXPLAINERS: Record<string, ExplainerSpec> = {
  manifold: {
    title: 'Manifold — Dynamic Riemannian Geometry',
    oneLine: 'The 2D base manifold whose metric is sourced by prediction error.',
    tabs: {
      overview: {
        what: [
          'The classical predictive-coding layer sits on a 2D Riemannian manifold whose metric g_μν is not fixed: it is sourced by the stress-energy tensor of the prediction-error field.',
          'Belief diffusion uses the Laplace-Beltrami operator built from that metric — so geometry follows what the system is uncertain about.',
        ],
        elements: [
          { name: 'Surface height', meaning: 'selected metric-perturbation component: h_xx (default), h_xy (shear), h_yy, or tr(h) = h_xx + h_yy. The full h_μν is a rank-2 symmetric tensor (§3.2); a single scalar height cannot represent it, so the toolbar exposes all three plus the mean-curvature proxy.', code: 'manifold.py:h_xx / h_xy / h_yy' },
          { name: 'Surface colour', meaning: 'Ricci scalar curvature R', code: 'manifold.py:ricci_scalar' },
          { name: 'Overlay (Φ / E / Π)', meaning: 'Belief / error / precision fields of the first PCN layer' },
        ],
      },
      math: { equationIds: [] },
      workedExample: {
        title: 'Flat patch with a single error spike',
        setup: 'Start with h_μν = 0 (g = η, R = 0). Inject a Gaussian bump in E of magnitude 1 at the origin; T_xx ≈ (∂_x E)² peaks on the bump\'s flanks.',
        steps: [
          { description: 'Source the trace-h equation h_xx += κ_R · T_xx · dt with κ_R = 0.1, dt = 0.1.', result: 'h_xx acquires a localised dimple on the bump flanks (≈ 0.01 amplitude).' },
          { description: 'Compute R ≈ −∂²h from finite differences of the resulting h_xx.', result: 'R becomes non-zero in a ring around the error spike — curvature concentrates where uncertainty is sharpest.' },
        ],
        takeaway: 'When E flattens (no prediction error), T_μν → 0 and h_μν relaxes back to zero: geometry literally tracks uncertainty.',
      },
      trainingDynamics: {
        updateRuleId: '',
        expect: [
          'Curvature concentrates where error spikes; mean |R| stabilises after error decays.',
          'Overlay Φ peaks track regions the model is confidently predicting.',
        ],
        pathologies: [
          { signal: 'mean |R| oscillating', cause: 'κ_R too large; reduce coupling strength so geometry can follow rather than overshoot error.' },
          { signal: 'h_xx growing without bound', cause: 'no relaxation term; check that the metric update includes a decay toward flat background between integration steps.' },
        ],
      },
      watch: [
        { label: 'Curvature concentrates where error spikes', readout: 'mean_abs_ricci' },
        { label: 'Mean |R| should stabilise after error decays' },
        { label: 'Overlay Φ — peaks track regions the model is confidently predicting' },
      ],
    },
    references: [{ label: 'Architecture §2.1, §3.1', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  multifield: {
    title: 'Multi-field — Coupled Field Species',
    oneLine: 'Multiple field species sharing one manifold, coupled via learnable g_ij.',
    tabs: {
      overview: {
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
      },
      math: { equationIds: [] },
      workedExample: {
        title: 'Two correlated fields drive g_12 up',
        setup: 'Two species Φ_1, Φ_2 with identical Gaussian-bump beliefs (correlation ≈ 1) and g_12 = 0 at t = 0; learning rate η = 0.05.',
        steps: [
          { description: 'Compute ∂F/∂g_12 = ⟨Φ_1 Φ_2⟩ ≈ −0.4 (large in magnitude because the fields co-vary).', result: 'Gradient is large and negative.' },
          { description: 'Apply ġ_12 = −η · ∂F/∂g_12 = 0.05 · 0.4 = 0.02 per step.', result: 'After 50 steps g_12 ≈ 1.0; the pair MetricsStrip trace rises steadily.' },
        ],
        takeaway: 'A pair with no shared structure has ⟨Φ_i Φ_j⟩ ≈ 0; its g_ij stays near zero. Co-varying pairs grow couplings; uncorrelated pairs decouple — the §4.5 selection signal.',
      },
      trainingDynamics: {
        updateRuleId: '',
        expect: [
          'Mean |g| rises when fields co-vary; pair traces let you see WHICH pairs are converging.',
          'Surfaces with no shared structure stay near-flat in coupling.',
        ],
        pathologies: [
          { signal: 'all g_ij saturate uniformly', cause: 'learning rate η too high or no regulariser — couplings can no longer discriminate co-varying from uncorrelated pairs.' },
          { signal: 'mean |g| oscillates', cause: 'fields have opposite-sign correlations across regions; check that per-pair traces are not cancelling in the aggregate scalar.' },
        ],
      },
      watch: [
        { label: 'Mean |g| rises when fields co-vary', readout: 'mean_abs_coupling' },
        { label: 'Surfaces with no shared structure stay near-flat in coupling' },
      ],
    },
    references: [{ label: 'Architecture §2.2', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  mps: {
    title: 'MPS — Matrix Product State',
    oneLine: 'The entanglement carrier: a 1D chain of low-rank tensors.',
    tabs: {
      overview: {
        what: [
          'The QPCN\'s belief state is represented as a Matrix Product State with controllable bond dimension χ_max — this is the entanglement carrier.',
          'Bond dimension caps the entanglement entropy across each cut; SVD truncation enforces it after every Trotter step.',
        ],
        elements: [
          { name: 'Bond-dim bar', meaning: 'χ for each bond between sites', code: 'mps.py:bond_dimensions' },
          { name: 'Entropy line', meaning: 'per-cut von Neumann entanglement entropy' },
          { name: 'Page-curve reference', meaning: 'random-state entropy ceiling (visual guide)' },
        ],
      },
      math: { equationIds: [] },
      workedExample: {
        title: 'Bell pair on a 2-site MPS',
        setup: 'N = 2 sites, d_local = 2, target state |ψ⟩ = (|00⟩ + |11⟩)/√2.',
        steps: [
          { description: 'Encode as MPS: A^0 = [1, 0], A^1 = [0, 1] (each a 1×2 row); the bond between sites carries χ = 2.', result: 'Bond-dim bar reads χ_1 = 2.' },
          { description: 'SVD the bipartition: singular values σ = (1/√2, 1/√2); cut entropy S = −2 · ½ log(½) = log 2.', result: 'Entropy line at the single cut equals log 2 ≈ 0.693.' },
        ],
        takeaway: 'A maximally entangled cut saturates at log(χ); reducing χ_max forces truncation and lowers achievable S — the bond-dim bar is a literal capacity ceiling.',
      },
      trainingDynamics: {
        updateRuleId: '',
        expect: [
          'Bonds saturate to χ_max under entangling dynamics.',
          'Entropy clusters in the middle for short-range Hamiltonians (volume-law-like in the bulk, area-law at the edges).',
        ],
        pathologies: [
          { signal: 'all bonds pinned at χ_max with stagnant energy', cause: 'χ_max too small; the ansatz cannot represent the ground state — raise χ_max and re-run.' },
          { signal: 'entropy collapses to ≈ 0 everywhere', cause: 'state has decohered to a product state; check that the Trotter step is actually applying entangling gates.' },
        ],
      },
      watch: [
        { label: 'Bonds saturate to χ_max under entangling dynamics' },
        { label: 'Entropy clusters in the middle for short-range Hamiltonians' },
      ],
    },
    references: [{ label: 'Architecture §2.3', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  hamiltonian: {
    title: 'Hamiltonian — Local Many-Body Operator',
    oneLine: 'Sum of one- and two-site terms over species; the QPCN\'s generative model.',
    tabs: {
      overview: {
        what: [
          'The Hamiltonian H is the QPCN\'s generative model. It is built locally from one-site terms (mass, source, quartic, curvature coupling) and two-site terms (kinetic hopping, cross-species interaction).',
          'Each species has a Fock cutoff d_local. The curvature field is consumed by the manifold-coupling term.',
        ],
        elements: [
          { name: 'Per-species table', meaning: 'ω (bare_mass), t (kinetic), μ (quartic), J (source) per field species', code: 'hamiltonian.py:FieldSpecies' },
          { name: 'Coupling matrix', meaning: 'g_{ab} (density-density) or λ_{ab} (Yukawa-like field-field), toggleable', code: 'HamiltonianConfig.density_couplings / yukawa_couplings' },
          { name: 'Curvature strip', meaning: '1D R(x_k) per MPS site, modulating ω via ξ', code: 'Hamiltonian.curvature' },
        ],
      },
      math: { equationIds: [] },
      workedExample: {
        title: 'Single-site mass term under curvature',
        setup: 'One species, one site, ω = 1.0, ξ = 0.2, R(x) = 0.5 at that site; n is the number operator.',
        steps: [
          { description: 'Apply the curvature-coupling rule ω → ω · (1 + ξ · R) = 1.0 · (1 + 0.2 · 0.5) = 1.1.', result: 'Effective on-site mass becomes 1.1 — the site sees an inflated cost of being occupied.' },
          { description: 'Compute the contribution ω_eff · ⟨n⟩ for ⟨n⟩ = 0.5.', result: 'Energy contribution = 0.55 (vs. 0.50 with flat metric); the curvature strip shows the local boost.' },
        ],
        takeaway: 'Curvature is a multiplicative modulation of bare mass; live param edits in the QPCN panel show up here as updated ω rows.',
      },
      trainingDynamics: {
        updateRuleId: '',
        expect: [
          'd_local equals ∏ species cutoffs — larger d means richer dynamics.',
          'Curvature peaks should correlate with the manifold-panel curvature; live param edits in QPCN show up in this panel\'s ω / t rows.',
        ],
        pathologies: [
          { signal: 'curvature strip and manifold-panel R disagree', cause: 'sampling pipeline drift between the manifold projection and the per-site R(x_k) values — re-check the manifold→1D restriction map.' },
          { signal: 'ω blows up under curvature', cause: 'ξ · R near −1 inverts the sign of the effective mass; clamp ξ · R or reduce ξ.' },
        ],
      },
      watch: [
        { label: 'd_local equals ∏ species cutoffs — larger d means richer dynamics' },
        { label: 'Curvature peaks should correlate with the manifold panel curvature' },
        { label: 'Live param edits in QPCN show up in this panel\'s ω / t row' },
      ],
    },
    references: [{ label: 'Architecture §2.3, §3', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  qpcn: {
    title: 'QPCN — Quantum Predictive Coder',
    oneLine: 'Belief = MPS; generative model = H; errors drive parameter updates.',
    tabs: {
      overview: {
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
      },
      math: { equationIds: [] },
      workedExample: {
        title: 'One imag-time step lowers ⟨H⟩',
        setup: 'Start with |ψ_0⟩ such that ⟨H⟩ = 1.20 with ground-state energy E_0 = 0.50; step dτ = 0.1; one target observable o with t_o = 0.30 and ⟨O⟩ = 0.50.',
        steps: [
          { description: 'Apply |ψ\'⟩ = e^{−H dτ}|ψ_0⟩ / ‖·‖; the relaxation projects out high-energy components.', result: '⟨H⟩ drops from 1.20 to ≈ 1.13.' },
          { description: 'Compute the parameter gradient Δθ = −η · 2 · (⟨O⟩ − t_o) · ∂_θ⟨O⟩ for η = 0.05 and ∂_θ⟨O⟩ ≈ 1.', result: 'Δθ = −0.05 · 2 · 0.20 · 1 = −0.02; the learnable-param row drops by 0.02 this step.' },
        ],
        takeaway: 'Two intertwined loops: imag-time relaxation lowers ⟨H⟩ at fixed θ; gradient steps adjust θ to shrink pred-errors. Both must converge for the model to "fit".',
      },
      trainingDynamics: {
        updateRuleId: '',
        expect: [
          'Energy decreases monotonically under imag-time (the relaxation invariant).',
          'Pred-errors shrink as parameters adapt; ⟨n_k⟩ stabilises once the ground state is found.',
        ],
        pathologies: [
          { signal: 'energy not monotone decreasing', cause: 'Trotter step dτ too large — the second-order error is dominating; halve dτ.' },
          { signal: 'pred-errors plateau above zero', cause: 'model is mis-specified (observable is not in the span of the current Hamiltonian); revisit the species list or coupling structure.' },
        ],
      },
      watch: [
        { label: 'Energy decreases monotonically under imaginary time', readout: 'energy' },
        { label: 'Pred-errors shrink as parameters adapt', readout: 'pred_errors' },
      ],
    },
    references: [{ label: 'Architecture §2.3', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  mera: {
    title: 'MERA — Multi-Scale Tree',
    oneLine: 'Tree of disentanglers + isometries; encodes scale structure.',
    tabs: {
      overview: {
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
      },
      math: { equationIds: [] },
      workedExample: {
        title: 'A 4-leaf MERA with two RG layers',
        setup: 'N = 4 leaves on the boundary; one disentangler U_1 (acts on the middle pair) and two 2→1 isometries W_1; one coarse isometry W_2 at the top.',
        steps: [
          { description: 'Apply U_1 to remove short-range entanglement between sites 2 and 3 before coarse-graining.', result: 'Entropy across the middle cut drops, freeing the W_1 isometries to compress.' },
          { description: 'Apply the two W_1 isometries (each 2→1) and then W_2 (2→1) at the apex.', result: 'A 4-leaf state is now described by one apex index; the entropy line should grow logarithmically across cuts on a critical-state input.' },
        ],
        takeaway: 'Each RG layer pairs a disentangler with an isometry; the iso-err sparkline must stay near zero, otherwise the tree is no longer a valid coarse-grainer.',
      },
      trainingDynamics: {
        updateRuleId: '',
        expect: [
          'Logarithmic entropy scaling on critical states; per-layer χ caps captured entanglement.',
          'iso-err sparkline stays near zero — the W_ℓ are valid isometries.',
        ],
        pathologies: [
          { signal: 'iso-err sparkline drifting upward', cause: 'gradient updates not being projected onto the Stiefel manifold; re-orthogonalise W_ℓ after each step.' },
          { signal: 'flat entropy across cuts on a critical state', cause: 'per-layer χ too small to capture the log-scaling — raise χ at the upper layers.' },
        ],
      },
      watch: [
        { label: 'Logarithmic entropy scaling on critical states' },
        { label: 'Per-layer χ caps the captured entanglement' },
      ],
    },
    references: [{ label: 'Architecture: MERA section', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  vqc: {
    title: 'VQC — Variational Quantum Circuit',
    oneLine: 'Parameterised circuit trained via parameter-shift gradients.',
    tabs: {
      overview: {
        what: [
          'A `QuantumConvMap` wraps a parameterised circuit as a translation-invariant quantum convolution that drops into any PCN layer.',
          'Gradients are exact via the parameter-shift rule. The same code targets real IBM hardware.',
        ],
        elements: [
          { name: 'Theta heatmap', meaning: 'rotation angles per (layer, qubit, axis)' },
          { name: 'Bias', meaning: 'classical bias on the conv output' },
        ],
      },
      math: { equationIds: [] },
      workedExample: {
        title: 'Single-θ gradient via parameter shift',
        setup: 'A single rotation R_y(θ) on one qubit, observable O = Z; current θ = π/4, target = 0.6, current ⟨O⟩ = cos(π/4) ≈ 0.707.',
        steps: [
          { description: 'Evaluate ⟨O⟩_{θ + π/2} = cos(3π/4) ≈ −0.707 and ⟨O⟩_{θ − π/2} = cos(−π/4) ≈ 0.707.', result: 'Two forward passes give the shifted expectations.' },
          { description: 'Apply the parameter-shift rule ∂_θ⟨O⟩ = ½ (−0.707 − 0.707) = −0.707.', result: 'Exact analytic gradient at θ = π/4 (matches −sin(π/4)); θ ← θ − η · 2 · (⟨O⟩ − t) · ∂_θ⟨O⟩ steps θ toward the target.' },
        ],
        takeaway: 'No finite differences and no autodiff through the simulator — the shift rule is exact and works identically on real hardware.',
      },
      trainingDynamics: {
        updateRuleId: '',
        expect: [
          'Theta heatmap evolves smoothly under parameter-shift gradient descent.',
          'Bias shifts the mean of the conv output and adapts to data offset.',
        ],
        pathologies: [
          { signal: 'theta updates dominated by noise', cause: 'too few shots for hardware-style expectation estimates; increase shot count or switch the backend to exact simulation.' },
          { signal: 'live training shows no movement', cause: 'live training is currently a fixture — see EXTENSIONS.md for the wiring task.' },
        ],
      },
      watch: [{ label: 'Live training is currently a fixture — see EXTENSIONS.md' }],
    },
    references: [{ label: 'Architecture §2.4', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  logic: {
    title: 'Logic — Evaluation Hamiltonian',
    oneLine: 'Rule terms encoded as Hamiltonian costs; relaxation = reduction.',
    tabs: {
      overview: {
        what: [
          'Each rule of the target calculus contributes a local Hamiltonian term; the ground state corresponds to a well-typed / fully-reduced program.',
          'Imaginary-time relaxation drives a superposition state toward zero residual energy under all rules simultaneously.',
        ],
        elements: [
          { name: 'Term nodes', meaning: 'per-rule (rule_id, site, arity); fill opacity tracks per-term residual energy so a relaxed (satisfied) term reads pale, a high-residual term reads saturated', code: 'snapshot_logic:terms / residuals' },
          { name: 'Binder bond entropy chart', meaning: 'per-bond von Neumann entropy on the logic-encoded MPS — the load-bearing §8.1 "variable binding = entanglement" signal; a binder live on a bond contributes entropy across the use→declaration path', code: 'snapshot_logic:bond_entropies' },
          { name: 'λ legend', meaning: 'global term-weight scalars (λ_β / λ_arith / λ_if). These describe relative term weights only, NOT binder geometry.' },
          { name: 'Total energy', meaning: 'sum of all residual term energies' },
        ],
      },
      math: { equationIds: [] },
      workedExample: {
        title: 'β-reduction lowers H_eval',
        setup: 'Encode (λx. x) y as a logic-MPS with one β-redex; λ_β = 1.0; initial residual on the β-rule term H_β^{(i)} = 1.',
        steps: [
          { description: 'Imag-time evolve under H_eval = λ_β · H_β^{(i)}; the β-redex relaxes toward its reduced form y.', result: 'Per-term residual on H_β decays toward 0; total energy drops by ≈ 1.0.' },
          { description: 'Observe the binder bond entropy: the bond carrying the x↔y binding contracts as the binder is consumed.', result: 'Bond entropy on the binder edge drops from log 2 to ≈ 0 — the §8.1 "binding = entanglement" signal in reverse.' },
        ],
        takeaway: 'Reduction = relaxation: a satisfied rule term reads pale; a live binder reads as a bond-entropy spike. Both fade together as the program reduces.',
      },
      trainingDynamics: {
        updateRuleId: '',
        expect: [
          'Term node opacity drops as residual energy decays (relaxation).',
          'Binder bond entropy spikes mark live binders — the §8.1 invariant.',
        ],
        pathologies: [
          { signal: 'binder entropy stuck at log(2) but total energy decays', cause: 'binder is being treated as a classical lookup, not an entangled pair; check the §8.1 invariant.' },
          { signal: 'total energy plateaus above zero', cause: 'conflicting rule terms (e.g. λ_arith too large vs. λ_β) — re-balance the global weights or check for ill-formed terms.' },
        ],
      },
      watch: [
        { label: 'Term node opacity drops as residual energy decays (relaxation)' },
        { label: 'Binder bond entropy spikes mark live binders — the §8.1 invariant' },
      ],
    },
    references: [{ label: 'Architecture: logic section', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  mera_relax: {
    title: 'MERA imag-time relax — §10.10 induction-theorem demo',
    oneLine:
      'Live ∀-protected MERA relaxation: residuals decay while bound leaves stay frozen.',
    tabs: {
      overview: {
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
              'leaf indices held bitwise stable by the trotter step (`frozen_leaves=`); the load-bearing §8.1 / §10.10 invariant. Must stay non-empty for any `forall …` proposition.',
            code: 'mera_encoder:forall_protected_leaves',
          },
        ],
      },
      math: { equationIds: [] },
      workedExample: {
        title: 'One Trotter step preserves a ∀-bound witness',
        setup: 'AST ∀x:Nat. x + 0 = x encoded into a MERA; encoder marks leaf 7 (the x-witness) as ∀-protected. Initial total energy ⟨H_eval⟩ = 2.4 (R-AddZero + R-Eq-Refl residuals).',
        steps: [
          { description: 'Call `mera_trotter_step(state, H_eval, dτ=0.05, frozen_leaves={7})`; gates touching leaf 7 are dropped at dispatch.', result: 'Leaf 7 is bitwise identical before and after the step.' },
          { description: 'Re-measure total energy and the per-term residuals on R-AddZero / R-Eq-Refl.', result: 'Total energy drops to ≈ 2.28 (≈ 5% per step); residuals on the two relevant rules decay; the AST round-trip still contains "forall".' },
        ],
        takeaway: 'The §8.1 / §10.10 invariant in action: relaxation reduces residual energy WITHOUT mutating the bound variable. Drop the freeze and the theorem witness disintegrates within a few steps.',
      },
      trainingDynamics: {
        updateRuleId: '',
        expect: [
          'Total energy decreases monotonically under imag-time (the relaxation invariant).',
          'The ∀-protected leaf set stays non-empty and bitwise stable across the run; AST round-trip continues to contain "forall".',
        ],
        pathologies: [
          { signal: '∀-protected leaf set becomes empty', cause: 'encoder failed to annotate the binder — re-check `encode_mera` returned a non-empty `meta.forall_protected_leaves` for the input AST.' },
          { signal: 'AST text loses "forall"', cause: 'frozen_leaves was not threaded through to `mera_trotter_step`; a gate mutated the witness leaf and the binder is gone.' },
        ],
      },
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
    },
    references: [
      { label: 'Architecture §10.10', href: '../../QFT_PCN_ARCHITECTURE.md' },
    ],
  },

  bridge: {
    title: 'Bridge — RunResult inspector',
    oneLine:
      'One-shot physics-DSL resolver; surfaces the resolved MPS, Hamiltonian and convergence.',
    tabs: {
      overview: {
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
      },
      math: { equationIds: [] },
      workedExample: {
        title: 'Solve a single-species DSL problem',
        setup: 'DSL: 1 species, N = 4 sites, ω = 1.0, t = 0.5, target tolerance tol = 1e-4, max_steps = 200.',
        steps: [
          { description: 'Call `run_problem(dsl)`; the runtime compiles H, initialises |ψ_0⟩ as a random product state, and runs imag-time with clamps.', result: 'After ≈ 80 Trotter steps, energy-per-step settles within tol; converged = yes.' },
          { description: 'Inspect `result.ground_state`: per-bond χ profile = [2, 4, 2] (χ_max = 4 at the middle cut).', result: 'Hamiltonian miniature reads N = 4, d_local = 2, species = [φ]; trotter steps = 80; ⟨H⟩ = −1.732.' },
        ],
        takeaway: 'A one-shot resolve: the bridge swallows a DSL, returns a fully measured `RunResult`. `solved_ast` is None on this MPS path — only a MERA-based runner populates it.',
      },
      trainingDynamics: {
        updateRuleId: '',
        expect: [
          'converged = yes once the energy-per-step settles within tol.',
          'solved_ast is None on this MPS path — a MERA-based runner is required to populate it.',
        ],
        pathologies: [
          { signal: 'converged = no after max_steps', cause: 'either tol too tight or χ_max too small to represent the ground state; raise χ_max first, then loosen tol if needed.' },
          { signal: 'trotter steps = 0', cause: 'DSL validation failed before evolution started — check the per-species cutoffs and coupling-matrix shapes for a compile error in `run_problem`.' },
        ],
      },
      watch: [
        { label: 'converged = yes once the energy-per-step settles within tol' },
        {
          label:
            'solved_ast is None on this MPS path — a MERA-based runner is required to populate it',
        },
      ],
    },
    references: [
      { label: 'Bridge spec §5', href: '../../../bridge/' },
    ],
  },

  'pcn-fields': {
    title: 'PCN Fields — Hierarchical Φ / E / Π Stack',
    oneLine: 'One card per PCN layer: belief Φ, prediction-error E, and precision Π fields.',
    tabs: {
      overview: {
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
      },
      math: { equationIds: [] },
      workedExample: {
        title: 'Two-layer stack with a single data spike',
        setup: 'L = 2 PCN layers; data Φ_0 has a single bump of amplitude 1 at one pixel; both Φ_1, Φ_2 start at 0; Π_1 = Π_2 = 1; D = 0.1.',
        steps: [
          { description: 'Compute E_1 = Φ_0 − g_1(Φ_1) = bump (since Φ_1 = 0); fold E_1 into Φ̇_1.', result: 'Φ_1 grows toward a smoothed bump; ‖E_1‖₂ drops from 1.0 to ≈ 0.4 in a few steps.' },
          { description: 'Now E_2 = Φ_1 − g_2(Φ_2) becomes non-zero only AFTER Φ_1 changes; the spike propagates upward.', result: 'Top-layer Φ_2 ends up smoother and more spread out than Φ_1 — the hierarchical signature.' },
        ],
        takeaway: 'Top layers carry coarse / abstract beliefs; bottom layers carry fast-changing error. Precision concentrates where E shrinks — the per-card heatmaps make this visible at a glance.',
      },
      trainingDynamics: {
        updateRuleId: '',
        expect: [
          '‖E‖₂ should decrease across the run as predictions improve.',
          'mean Π should grow in regions where E shrinks (high confidence = high precision).',
          'Top-layer Φ stays smoother / coarser than bottom-layer Φ — the hierarchical signature.',
        ],
        pathologies: [
          { signal: '‖E‖₂ stays flat across the run', cause: 'g_l prediction maps are not adapting — check that g_l parameters are on the gradient path.' },
          { signal: 'top-layer Φ is noisier than bottom-layer Φ', cause: 'precision Π is inverted across layers; verify Π_top < Π_bottom so the upper layer averages over more pixels.' },
        ],
      },
      watch: [
        { label: 'depth readout matches the configured PCN stack height', readout: 'depth' },
        { label: '‖E‖₂ should decrease across the run as predictions improve' },
        { label: 'mean Π should grow in regions where E shrinks (high confidence = high precision)' },
        { label: 'Top-layer Φ stays smoother / coarser than bottom-layer Φ — the hierarchical signature' },
      ],
    },
    references: [{ label: 'Architecture §2.1, §3.1', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  'pcn-dynamics': {
    title: 'PCN Dynamics — Free Energy + Per-Layer Trajectories',
    oneLine: 'Total variational free energy F and its per-layer contributions over time.',
    tabs: {
      overview: {
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
      },
      math: { equationIds: [] },
      workedExample: {
        title: 'Per-layer F decomposition on a 2-layer stack',
        setup: 'L = 2; layer 1: ‖E_1‖₂ = 2, mean Π_1 = 0.5; layer 2: ‖E_2‖₂ = 0.5, mean Π_2 = 1.0; κ = 0, R = 0 (ignore curvature term for clarity).',
        steps: [
          { description: 'Compute F_1 ≈ ½ · 0.5 · 4 − ½ · log(0.5) = 1.0 + 0.35 = 1.35 (using ⟨E²⟩ ≈ ‖E‖₂² and a constant Π).', result: 'F_1 ≈ 1.35.' },
          { description: 'Compute F_2 ≈ ½ · 1.0 · 0.25 − ½ · log(1.0) = 0.125; total F = F_1 + F_2 ≈ 1.475.', result: 'Per-layer F table reads F_1 = 1.35, F_2 = 0.125; layer 1 is the hot bottleneck.' },
        ],
        takeaway: 'The per-layer F row tells you WHICH layer is dominating the cost; total F is the master "is the network learning?" trace.',
      },
      trainingDynamics: {
        updateRuleId: '',
        expect: [
          'total F decreases monotonically under successful learning.',
          'Per-layer F rows reveal which layer is the current bottleneck (largest contribution).',
          '‖E‖₂ should decay as predictions improve; mean Π should rise where E shrinks.',
        ],
        pathologies: [
          { signal: 'total F decreases but per-layer F oscillates', cause: 'layers are cancelling each other — check the cross-layer message passing; one layer may be overcorrecting.' },
          { signal: 'total F increases', cause: 'gradient step too large for at least one layer; halve η or add a per-layer adaptive learning rate.' },
        ],
      },
      watch: [
        { label: 'total F decreases monotonically under successful learning', readout: 'total_free_energy' },
        { label: 'Per-layer F rows reveal which layer is the current bottleneck (largest contribution)' },
        { label: '‖E‖₂ should decay as predictions improve; mean Π should rise where E shrinks' },
        { label: 'depth readout matches the configured PCN stack height', readout: 'depth' },
      ],
    },
    references: [{ label: 'Architecture §3.1', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  'pcn-coupling': {
    title: 'PCN ↔ QFT — Bidirectional Coupling',
    oneLine: 'The bridge: PCN error sources QFT metric (T_μν → h_μν); QFT operator expectations feed PCN observation targets (⟨Ô⟩).',
    tabs: {
      overview: {
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
      },
      math: { equationIds: [] },
      workedExample: {
        title: 'PCN error spike sources QFT curvature',
        setup: 'κ_R = 0.1; mean |T| = 0.0 (network at rest); a single PCN error spike pushes mean |T| to 0.5; QPCN ⟨H⟩ = 1.0.',
        steps: [
          { description: 'Apply h_μν ∝ κ_R · T_μν: mean |h_μν| ≈ 0.05 builds up over a few steps; mean |R| ≈ κ_R · mean|T| = 0.05 in the linearised regime.', result: 'Top arrow widens (track mean |T|); the manifold-panel R grows in lockstep.' },
          { description: 'QPCN ⟨H⟩ = 1.0 feeds the bottom arrow; PCN tries to match the operator-expectation target.', result: 'Bottom-arrow width settles proportional to ⟨H⟩; as the joint system relaxes (error decays AND ⟨H⟩ converges), BOTH arrows should shrink together.' },
        ],
        takeaway: 'The two arrows are the load-bearing §3 bridge: if one widens forever, the system is one-way coupled (a bug). At convergence both arrows are quiet.',
      },
      trainingDynamics: {
        updateRuleId: '',
        expect: [
          'Top arrow widens as PCN error grows; mean |R| tracks mean |T| × κ_R.',
          'Both arrows should shrink together as the joint system relaxes.',
        ],
        pathologies: [
          { signal: 'top arrow widens but bottom arrow stays narrow', cause: 'one-way coupling — QFT is not feeding ⟨Ô⟩ back into PCN observation targets; check the bridge wiring (§3 bidirectional bridge).' },
          { signal: 'mean |R| explodes', cause: 'κ_R · mean |T| has left the linearised regime — clamp κ_R or add a metric-update saturation term.' },
        ],
      },
      watch: [
        { label: 'Top arrow widens as PCN error grows (large mean |T| ⇒ strong QFT source)', readout: 'mean_abs_stress_energy' },
        { label: 'mean |R| should track mean |T| × κ_R — the geometric response to the source' },
        { label: 'Bottom arrow widens as ⟨H⟩ grows: QFT is pulling harder on PCN observation targets' },
        { label: 'Both arrows should shrink together as the joint system relaxes (error decays AND QPCN converges)' },
      ],
    },
    references: [{ label: 'Architecture §3', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },
};
