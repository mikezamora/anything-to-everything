// src/qft_pcn/viz/web/src/routes/learn/articles/qft-vqc.tsx
/**
 * §2.4 QFT side — Variational Quantum Circuits and the parameter-shift rule.
 *
 * Covers: parameterised single-qubit rotations, the analytic gradient
 * formula, why the shift is π/2 specifically, and how the substrate's
 * VQC layer (`src/qft_pcn/quantum.py`) uses these gradients to descend
 * observation residuals.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'qft-vqc',
  title: 'Variational quantum circuits and the parameter-shift rule',
  sectionPath: ['§2 QFT side', '2.4 VQC'],
  prerequisites: ['foundations-hilbert-operators'],
  sections: [
    {
      kind: 'prose',
      body: [
        'A **Variational Quantum Circuit** (VQC) is a parameterised unitary `U(theta) = U_L(theta_L) ... U_1(theta_1)` built from a small library of elementary gates whose parameters are learnable real numbers. Apply it to an initial state — usually `|0...0>` — and measure an observable `O`. The resulting expectation `⟨O⟩(theta) = ⟨0| U^†(theta) O U(theta) |0⟩` is a real-valued function of the parameter vector. Training is gradient descent on a loss built from this expectation (typically a sum of squared residuals against observation targets). The whole pipeline is the QFT-side analogue of a feed-forward network: the gates play the role of layers, the parameters play the role of weights, and the expectation plays the role of the output.',
        'The QPCN uses VQCs as a lightweight bridge between the dense Hamiltonian side and the manifold / PCN side. Where the MPS / MERA path is dominant, the VQC supplements it: a short rotation circuit produces target expectations that can be compared against PCN-side observations and used to derive Hamiltonian parameter updates. The substrate\'s VQC support lives in `src/qft_pcn/quantum.py`; the visualiser surfaces a parameter-by-parameter trace as each angle descends.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'The canonical elementary gate is a single-qubit Pauli rotation: `R_P(theta) = exp(-i (theta/2) P)` for `P ∈ {X, Y, Z}`. Expanding: `R_P(theta) = cos(theta/2) I - i sin(theta/2) P`. The half-angle factor traces back to the fact that `P^2 = I`, so the exponential reduces to a sum of two terms. Two practical consequences: the gate is `2π`-periodic in `theta` for some observables but `4π`-periodic in general (the famous "spinor under rotation"), and the derivative of `R_P` with respect to `theta` is itself a half-shifted rotation, which is the structural input the parameter-shift rule exploits.',
        'Stacking rotations on multiple qubits, possibly interleaved with non-parameterised entangling gates (CNOT, CZ, etc.), yields a generic ansatz. The expressivity of a VQC depends on its depth and entangling structure; for the QPCN the circuits are typically shallow because they are diagnostic / coupling layers rather than the dominant compute. A typical VQC layer in the substrate is `R_Y(theta_1) ⊗ R_Y(theta_2) ⊗ ... → CNOT chain → R_Y(theta_{N+1}) ⊗ ...`, which has enough expressivity for low-dimensional target spaces while remaining cheap to differentiate.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'parameter-shift-rule',
      caption: 'The headline result: the gradient of a Pauli-rotation expectation is an exact difference of expectations at shifted parameters. No finite-difference noise; no approximation.',
    },
    {
      kind: 'prose',
      body: [
        'Why exactly `π/2`? The derivation is short and worth following. Let `U(theta) = R_P(theta) V` (the parameterised gate, with everything else absorbed into `V`). The expectation is `f(theta) = ⟨psi_0| V^† R_P^†(theta) O R_P(theta) V |psi_0⟩`. Differentiate using `dR_P/dtheta = -i (P/2) R_P`. After collecting terms, `df/dtheta = -i/2 · ⟨psi(theta)| [P, O] |psi(theta)⟩` where `|psi(theta)⟩ = R_P(theta) V |psi_0⟩`. The commutator `[P, O]` is, generically, not directly measurable — but here is the trick: when `P^2 = I` (as for any single-qubit Pauli), the identity `R_P(theta + π/2) - R_P(theta - π/2) = -2i sin(π/4) (P R_P(theta) - R_P(theta) P) / something` (worked out properly) collapses the commutator into a difference of two expectations evaluated at the same `theta` shifted by `±π/2`. That collapse is exact, not perturbative, and it is the *only* shift value at which it works for Pauli generators.',
        'The practical consequence is that gradients on quantum hardware are accessible without backpropagation through a simulator. Each parameter\'s derivative costs exactly two circuit evaluations: one at `theta + π/2`, one at `theta - π/2`, halve the difference. The cost is `O(num_parameters)` circuit runs per gradient step — the quantum analogue of forward-mode autodiff, and the standard for variational quantum algorithms in the NISQ era.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'param-update',
      caption: 'Once the gradient is in hand, gradient descent proceeds as usual. The QPCN folds the VQC\'s shifted-parameter gradient into the same residual-minimisation loop used for the Hamiltonian parameters.',
    },
    {
      kind: 'prose',
      body: [
        'A subtlety: the parameter-shift rule as stated holds for generators with at most two distinct eigenvalues (the `P^2 = I` condition). Pauli rotations satisfy this trivially. For more general generators — multi-qubit interactions, hardware-specific gates with three or more distinct eigenvalues — the rule generalises but requires either more shift evaluations (one per pair of eigenvalues) or a decomposition of the gate into Pauli-rotation primitives. The substrate sticks to single-qubit Pauli rotations for parameterised gates, which keeps the gradient cost at exactly `2N` circuit evaluations for `N` parameters.',
        'A second subtlety: the shift rule gives an *exact* gradient of the expectation value, but the expectation itself is estimated from a finite number of measurement shots. On simulator runs the shot-noise term is zero by default (`⟨O⟩` is computed analytically from the statevector); on hardware it is `O(1/sqrt(shots))`. The training-dynamics traces in the viz reflect simulator behaviour and so look noise-free; a hardware run would show a stochastic envelope around the same descent.',
      ],
    },
    {
      kind: 'prose',
      body: [
        '**Connection to PCN\'s generative map.** In the classical PCN a layer carries a `ClassicalConvMap`: a parameterised function from latent activity to a prediction of the layer below. The error between that prediction and the actual lower-layer activity is the local prediction-error signal that drives both upward inference and downward parameter updates. The QPCN replaces this slot with a `QuantumConvMap`: the same interface — latents in, prediction out — but the prediction is generated by a short VQC whose parameters are the variational angles `theta`. Concretely the latent vector seeds the initial state preparation (a bank of `R_Y` angles loading the input amplitudes); the body of the VQC is the learnable rotation-plus-entangling ansatz; the measured expectations are the prediction passed back to the lower layer.',
        'What changes vs the classical map is the gradient route. The classical map is differentiated by ordinary backprop through the parameterised function. The quantum map is differentiated by the parameter-shift rule: each angle\'s gradient costs two extra circuit evaluations (one at `+pi/2`, one at `-pi/2`), folded into the same residual `(prediction - target)` that the classical PCN uses. The error signal\'s *shape* and *role* are identical to classical PCN — error-driven update, local to a layer — but the gradient mechanism is hardware-compatible and exact in expectation. This is the wiring that makes the substrate\'s VQC interchangeable with a classical conv map at the PCN-layer boundary: the layer above does not know (or care) which mechanism produced the gradient.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'imag-time-evolution',
      caption: 'When a VQC is used inside a relax-style PCN layer, the angle updates are themselves driven by a short imaginary-time-like descent on the prediction residual: the gate parameters move along the residual gradient at each inner step, exactly as the MPS state moves along the energy gradient in a TEBD sweep.',
    },
    {
      kind: 'prose',
      body: [
        'Reading the VQC panel in the visualiser. The parameter strip shows each `theta_k` as a coloured tick on a circle (the value lives on `S^1`). After each gradient step the ticks move; the colour deepens with the magnitude of the most recent update. A tick that stops moving while the residual is still high signals a barren-plateau-like situation: the gradient with respect to that parameter is vanishingly small. The classic cure is to re-initialise that parameter to a random value or to add an entangling gate adjacent to it that increases its effective Hessian. The QPCN\'s short circuits rarely hit a true barren plateau, but the pattern is worth recognising for deeper architectures.',
      ],
    },
    {
      kind: 'workedExample',
      example: {
        title: 'Parameter-shift gradient of ⟨Z⟩ for a single R_Y(theta) on |0⟩',
        setup:
          'The simplest non-trivial VQC: initial state |0⟩, apply R_Y(theta) = exp(-i (theta/2) Y), measure ⟨Z⟩. Compute the expectation as a function of theta, then verify the parameter-shift formula reproduces dE/dtheta exactly.',
        steps: [
          {
            description:
              'Compute R_Y(theta)|0⟩ directly. R_Y(theta) = cos(theta/2) I - i sin(theta/2) Y = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]. Acting on |0⟩ = (1, 0)^T gives (cos(theta/2), sin(theta/2))^T.',
            result: '|psi(theta)⟩ = cos(theta/2)|0⟩ + sin(theta/2)|1⟩.',
          },
          {
            description:
              'Compute ⟨Z⟩(theta) = ⟨psi(theta)| Z |psi(theta)⟩. With Z = diag(+1, -1), we have ⟨Z⟩ = cos²(theta/2) - sin²(theta/2) = cos(theta).',
            result: '⟨Z⟩(theta) = cos(theta). Analytic derivative: d⟨Z⟩/dtheta = -sin(theta).',
          },
          {
            description:
              'Apply the parameter-shift rule. Evaluate ⟨Z⟩(theta + π/2) = cos(theta + π/2) = -sin(theta). Evaluate ⟨Z⟩(theta - π/2) = cos(theta - π/2) = +sin(theta). The shift-rule gradient is ½ · ((-sin(theta)) - (+sin(theta))) = ½ · (-2 sin(theta)) = -sin(theta).',
            result: 'Shift-rule gradient: -sin(theta).',
            equationId: 'parameter-shift-rule',
          },
          {
            description:
              'Compare. Analytic derivative: -sin(theta). Shift-rule derivative: -sin(theta). They agree exactly, for every theta, with no approximation and no shift-size tuning required.',
            result: 'Parameter-shift rule is exact. This is the property that makes it the standard gradient method for VQCs: cheap (two circuit evals per parameter), noise-free in expectation, hardware-native.',
          },
        ],
        takeaway:
          'For Pauli-rotation gates the parameter-shift rule is not a finite-difference approximation — it is an algebraic identity. The QPCN exploits this to compute VQC gradients with the same fidelity as automatic differentiation on a classical simulator, and the same formula would survive transcription to real quantum hardware.',
      },
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'parameter-shift-rule',
      expect: [
        'Each theta_k descends roughly along -dL/dtheta_k = 2 · (⟨O⟩ - t_o) · d⟨O⟩/dtheta_k, where the inner derivative is supplied exactly by the shift rule.',
        'On a simulator the per-step gradient is deterministic; the descent traces look smooth and the loss decreases monotonically (modulo line-search overshoot).',
        'Effective rotations modulo 2π mean the parameter trace can wrap; the visualiser collapses the wrapped angle to [0, 2π) for display while the optimiser sees the unwrapped trajectory.',
      ],
      pathologies: [
        {
          signal: 'A single theta_k stuck while the residual remains high',
          cause: 'Local barren plateau: ⟨O⟩ has near-zero gradient with respect to that parameter at the current configuration. Re-initialise that theta_k to a random value or insert a nearby entangling gate.',
        },
        {
          signal: 'Loss decreasing then suddenly jumping up',
          cause: 'Learning rate η too large near a curvature spike; halve η and re-run from the pre-jump checkpoint.',
        },
        {
          signal: 'Hardware run\'s loss noisy around a flat mean',
          cause: 'Shot noise dominates the true gradient; increase shots-per-evaluation or apply gradient averaging.',
        },
        {
          signal: 'Gradient sign reversing every step',
          cause: 'Optimiser oscillating across a narrow minimum; switch to a momentum-based update or shrink η.',
        },
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §2.4 (VQC bridge)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/quantum.py (VQC primitives and shift-rule gradients)', href: '../../src/qft_pcn/quantum.py' },
  ],
};
