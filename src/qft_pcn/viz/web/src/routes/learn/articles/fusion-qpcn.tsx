// src/qft_pcn/viz/web/src/routes/learn/articles/fusion-qpcn.tsx
/**
 * §4.3 Fusion — the QPCN. The MPS is the belief, the Hamiltonian is the
 * generative model, observation errors drive Hamiltonian parameter
 * updates. This is the article that explains the quantum-side learner.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'fusion-qpcn',
  title: 'Fusion: the QPCN — MPS belief, Hamiltonian model, parameter learning',
  sectionPath: ['§4 QPCN Fusion', '4.3 QPCN'],
  sections: [
    {
      kind: 'prose',
      body: [
        'The QPCN is the quantum-side analogue of the predictive-coding network. It maintains a **belief state** (a many-body wavefunction represented as a Matrix Product State, MPS), a **generative model** (a parameterised Hamiltonian H whose ground state is what the belief is trying to be), and a **learning rule** (a gradient on the Hamiltonian\'s parameters driven by the gap between the MPS\'s operator expectations and the observation targets passed in from the PCN coupling bridge). All three pieces are needed; together they are the smallest closed-loop variational learner that lives entirely on the quantum side.',
        'The article in §1.2 introduced MPSs as a low-rank ansatz for a high-dimensional state vector. Here we treat that ansatz as a Bayesian belief: at each frame, the MPS encodes the QPCN\'s current best estimate of the world\'s quantum state. The Hamiltonian H is the prior — its ground state is what the QPCN thinks the world should look like in the absence of evidence. Imaginary-time evolution `|psi> -> exp(-tau H) |psi> / norm` is the relaxation that pulls the belief toward the ground state. When evidence arrives (via prediction errors from the PCN side), the Hamiltonian itself is updated, shifting where the ground state lives, which the next imag-time step then pulls the belief toward.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'mps-ansatz',
      caption: 'The MPS factorisation that represents the QPCN belief state — bond dimension chi controls how much entanglement the belief can carry.',
    },
    {
      kind: 'prose',
      body: [
        'The Hamiltonian is decomposed into a sum of local terms with learnable coefficients: `H = sum_k c_k * H_k`, where each `H_k` is a fixed operator (a Pauli string, a kinetic term, a Yukawa coupling) and `c_k` is the corresponding tunable real scalar. This is the form that makes parameter learning tractable: the gradient `partial<H>/partial c_k = <H_k>` is just the expectation of the k-th term operator, which the MPS gives us cheaply.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'hamiltonian-decomp',
      caption: 'Hamiltonian decomposed into learnable-coefficient local terms — the structure that makes parameter-shift gradients trivially cheap to evaluate.',
    },
    {
      kind: 'prose',
      body: [
        'The imag-time update is the relaxation step. Concretely, we Trotter-decompose `exp(-tau H) = prod_k exp(-tau H_k) + O(tau^2)` and apply each `exp(-tau H_k)` to the MPS in sequence, re-orthogonalising and truncating after every gate to keep the bond dimension at the target `chi`. This is the standard TEBD-style algorithm; the only QPCN-specific twist is that the trotter_steps and tau used per frame are pulled from the RunSpec rather than being globally fixed, so the panel can show how different relaxation budgets affect convergence.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'imag-time-evolution',
      caption: 'The imaginary-time evolution that pulls the MPS belief toward the current Hamiltonian\'s ground state — applied each frame.',
    },
    {
      kind: 'prose',
      body: [
        'The parameter-update rule closes the loop. After the MPS relaxation, we read out the targeted operator expectations `<O_j>` for each observable in the RunSpec, compare against the targets passed in by the PCN bridge, and form a per-observable prediction error `delta_j = <O_j> - target_j`. The Hamiltonian coefficients are then updated by gradient descent on the squared prediction error, with the gradient computed via the parameter-shift rule (for parameters that appear as operator coefficients, the shift rule reduces to a single extra expectation; we lean on this).',
        'The §4.3 panel plots `<H>` over time — under correct dynamics it decreases monotonically — together with the per-observable prediction errors and the parameter trajectories. A healthy QPCN run shows `<H>` falling smoothly toward a plateau, prediction errors shrinking, and parameters drifting then stabilising. Pathologies show up as `<H>` oscillating (too-large tau), prediction errors stuck above zero (no Hamiltonian configuration can match the target), or runaway parameters (learning rate too high).',
      ],
    },
    {
      kind: 'equation',
      equationId: 'parameter-shift-rule',
      caption: 'The parameter-shift rule we use to evaluate Hamiltonian-coefficient gradients — exact, hardware-friendly, and cheap on an MPS.',
    },
    {
      kind: 'prose',
      body: [
        '**MPS-as-belief vs MPS-as-state.** In the standard tensor-network literature an MPS represents a *physical* state — the literal wavefunction of an actual quantum system. The QPCN repurposes the same data structure to play a different role: the MPS is the QPCN\'s *belief* about a state, a Bayesian posterior over many-body configurations, not the system itself. The bond dimension `chi` is then a hyperparameter of belief richness, not a property of physics; the bond indices carry *epistemic* correlation (how much joint information the model holds across sites) rather than *ontic* entanglement (how much the underlying system is actually entangled).',
        'The conceptual shift matters because it changes what convergence means. For an MPS-as-state, convergence is faithfulness — the MPS matches the true state. For an MPS-as-belief, convergence is *posterior consistency* — the MPS reaches a fixed point under the combined action of imag-time relaxation (prior pull) and prediction-error updates (evidence pull). Two QPCNs trained on the same data with different `chi` can converge to different beliefs; both are "right" relative to their own representational capacity, exactly as two Bayesian models with different prior families would be. The §4.3 panel\'s `<H>` trace is the energy of the *belief* under the *current model*, not the energy of any external physical system.',
      ],
    },
    {
      kind: 'prose',
      body: [
        '**The role of imag-time in inference.** Imaginary-time evolution `|psi> -> exp(-tau H) |psi> / norm` is best known as the standard recipe for finding a Hamiltonian\'s ground state: it monotonically suppresses high-energy components and leaves the lowest-energy survivor. In the QPCN it does double duty as a *Bayesian posterior projection*. Reading `H` as `-log p(state | model)` (Boltzmann form), `exp(-tau H)` is a partial Bayesian update — it multiplies the current belief by a fractional power of the prior likelihood and renormalises. As `tau -> infinity`, this collapses onto the maximum-likelihood configuration; as `tau -> 0`, the belief barely moves.',
        'This dual reading is why the same Trotter-TEBD machinery used for ground-state finding works unmodified for inference. The Hamiltonian *is* the negative log-prior; the MPS *is* the belief; the imag-time step *is* the soft Bayes update. Hard observation conditioning (a true posterior given evidence) is recovered as the `tau -> infinity` limit of imag-time evolution under an evidence-augmented Hamiltonian; partial updating, which is what the QPCN actually wants frame to frame, is just the finite-`tau` version of the same procedure. The PCN-side bridge does not need to know any of this — it just hands targets to the QPCN and reads expectations back — but the conceptual unification is what makes the architecture coherent rather than a hack.',
      ],
    },
    {
      kind: 'workedExample',
      example: {
        title: 'One Trotter step of imag-time on a single-site state',
        setup:
          'Take a 1-site Hilbert space (a single qubit; the smallest non-trivial MPS). Hamiltonian H = X (Pauli-X). Initial state |psi> = |0> = [1, 0]. Use tau = 0.1. Apply one imag-time step and verify the result is closer to the ground state of X.',
        steps: [
          {
            description:
              'Ground state of X is |-> = (|0> - |1>)/sqrt(2) with eigenvalue -1. We expect imag-time to pull |0> toward |->.',
            result: 'Target: |-> = [0.707, -0.707]; expected to grow the -|1> component from zero.',
          },
          {
            description:
              'exp(-tau X) = cosh(tau) I - sinh(tau) X. With tau = 0.1: cosh(0.1) ~ 1.00500, sinh(0.1) ~ 0.10017. So exp(-tau X) = [[1.005, -0.1002], [-0.1002, 1.005]] (real, symmetric).',
            result: 'exp(-tau X) ~ [[1.005, -0.100], [-0.100, 1.005]].',
            equationId: 'imag-time-evolution',
          },
          {
            description:
              'Apply to |0> = [1, 0]: result = [1.005, -0.100]. This is unnormalised — norm = sqrt(1.005^2 + 0.100^2) = sqrt(1.020) ~ 1.010.',
            result: 'Unnormalised state: [1.005, -0.100]; norm ~ 1.010.',
          },
          {
            description:
              'Normalise: |psi_new> = [1.005, -0.100] / 1.010 = [0.995, -0.099]. The |1> amplitude has grown from 0 to -0.099 — the state has rotated toward |->.',
            result: 'Normalised: |psi_new> ~ [0.995, -0.099]; overlap with |-> is (0.995 + 0.099)/sqrt(2) ~ 0.774, up from <0|-> = 0.707.',
          },
          {
            description:
              'Check <H> = <psi_new|X|psi_new> = 2 * 0.995 * (-0.099) = -0.197. Started at <0|X|0> = 0; now -0.197, moving toward ground-state eigenvalue -1. After ~30 such steps the state is numerically indistinguishable from |->.',
            result: '<H> went from 0 to -0.197 in one step — monotone decrease confirmed.',
            equationId: 'hamiltonian-decomp',
          },
        ],
        takeaway:
          'One Trotter step of exp(-tau H) with tau = 0.1 rotated |0> measurably toward the X ground state |->, decreasing <H> from 0 to -0.197. The MPS version of this is the same calculation per site (or per pair, for two-site gates) with a re-orthogonalisation and bond-dimension truncation between gates. The §4.3 panel plots <H>(t) for the full multi-site system; you should see exactly this monotone-decrease pattern.',
      },
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'imag-time-evolution',
      expect: [
        '<H> decreases monotonically under imag-time (this is a theorem, not a tuning).',
        'Prediction errors shrink as Hamiltonian parameters adapt.',
        'Parameter trajectories drift smoothly then stabilise.',
        'Bond-dimension growth saturates at the configured chi_max; entanglement entropy plateaus.',
      ],
      pathologies: [
        { signal: '<H> oscillates', cause: 'Trotter step tau too large; the linearisation breaks down. Halve tau.' },
        { signal: '<H> stalls above the true ground state energy', cause: 'chi_max too small; entanglement is being clipped. Increase chi_max.' },
        { signal: 'Prediction errors freeze nonzero', cause: 'Hamiltonian decomposition cannot represent the target; no parameter setting works. Add a term or re-examine targets.' },
        { signal: 'Parameters diverge', cause: 'Parameter-update learning rate too high. Reduce by 10x.' },
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §2.3 + §4.7.5 (QPCN — quantum predictive coder)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/qft/qpcn.py', href: '../../../qft/qpcn.py' },
  ],
};
