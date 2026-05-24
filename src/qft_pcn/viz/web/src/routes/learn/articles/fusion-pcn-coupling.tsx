// src/qft_pcn/viz/web/src/routes/learn/articles/fusion-pcn-coupling.tsx
/**
 * §4.2 Fusion — the bidirectional PCN <-> manifold <-> QFT coupling
 * bridge. Explains how stress-energy flows from classical errors to the
 * geometry and how operator expectations flow back to the predictions.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'fusion-pcn-coupling',
  title: 'Fusion: the bidirectional PCN <-> QFT coupling',
  sectionPath: ['§4 QPCN Fusion', '4.2 PCN Coupling'],
  sections: [
    {
      kind: 'prose',
      body: [
        'The §4.1 article showed one direction of the loop: PCN errors source the metric. This article fills in the other direction and makes the bridge fully bidirectional. The QFT side computes operator expectations `<O>` on its current quantum state; those expectations are fed back into the PCN as targets / priors / observation models, depending on the panel. When prediction errors are large, the manifold curves more; when the QFT\'s expectations match the data well, the prediction errors shrink and the manifold relaxes back toward flat.',
        'The whole point of the QPCN architecture is that this loop is closed. Neither side has an "external" target — they target each other through the geometric substrate. The §4.2 panel\'s headline visualisation is the two-arrow diagram: one arrow flowing PCN -> QFT (stress-energy upward into the metric, expectations downward into observations), the other QFT -> PCN (operator readouts updating prediction error). The widths of these arrows in the panel are proportional to the instantaneous rate of information flow in each direction. When they balance, the system is at a coupled fixed point.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'The PCN side maintains a hierarchy of belief layers, each with its own prediction error against the layer above. The total free-energy functional `F` is the sum, across layers, of weighted squared errors plus a regularisation term. Descending `F` is the network\'s only learning signal. On every frame, the layer-wise error fields are summed into a single grid-resolved `E(x)`, and that `E` is what feeds the stress-energy construction described in §4.1.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'free-energy-functional',
      caption: 'The free-energy functional minimised by the PCN. Its layer-wise residuals are the source of the stress-energy that drives the manifold.',
    },
    {
      kind: 'prose',
      body: [
        'The QFT side runs as the §4.3 article describes: an MPS-represented belief state evolved under imaginary time toward the current Hamiltonian\'s ground state, with operator expectations `<O>` read out by standard tensor-network contraction. The bridge reads these expectations and writes them into the PCN\'s observation model — concretely, the top-layer prior in the PCN hierarchy is replaced (or weighted-averaged) with the QFT\'s `<O>` at the matching grid resolution.',
        'The bidirectional structure means the two sides cannot ignore each other. If the PCN converges to a belief that the QFT cannot match (the Hamiltonian has no parameters that produce that `<O>`), then prediction errors stay high, the manifold stays curved, and the QFT keeps getting pulled. If the QFT converges to a low-energy state with `<O>` that the data cannot support (no PCN evidence), the prediction errors point the other way and the parameter-update rule (see §4.3) eventually steers the Hamiltonian.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'param-update',
      caption: 'The Hamiltonian parameter update — descending prediction error in the operator readouts. This is the QFT-side response to the PCN-side coupling.',
    },
    {
      kind: 'prose',
      body: [
        'In practice, the coupling can become asymmetric. If the QFT is much faster to relax than the PCN (small Hilbert space, simple Hamiltonian), the upward arrow in the panel is much wider than the downward — the QFT immediately follows the PCN\'s priors. If the PCN learns much faster than the QFT (large kappa_R, slow Hamiltonian updates), the downward arrow dominates — the QFT looks like a slowly-moving target. The interesting regime is when both arrows are visible and the system spends real time in transient curvature configurations rather than rushing to a flat steady state.',
        'The most common failure mode is "one-way" coupling: the PCN drives the QFT but the QFT readouts barely move the PCN priors. This usually means the observation noise is set very high, so `<O>` is essentially ignored. Lower it (carefully) until the downward arrow becomes visible.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'coupling-descent',
      caption: 'The shared coupling-descent rule that both sides obey — the upstairs/downstairs symmetry that makes the bridge truly bidirectional.',
    },
    {
      kind: 'prose',
      body: [
        '**Why bidirectional and not just one-way.** A naive architecture would treat the PCN as the "model" and the QFT as the "data" (or vice versa): one side has parameters, the other side has targets, and learning is the projection of one onto the other. The §4 architecture rejects this asymmetry deliberately. Neither side is privileged. The PCN supplies priors and reports errors; the QFT supplies operator readouts and updates its Hamiltonian. Each side learns from the other, and the bridge\'s job is to keep the two languages in sync, not to translate one into the other.',
        'The architectural reason for this insistence is in §1.1 of the architecture: *belief* and *evidence* are not different kinds of thing in the QPCN — they are two views of the same underlying entanglement structure. A unidirectional bridge would silently pick one view as canonical and demote the other to "input." The bidirectional bridge refuses to make that choice. In practice, this means you can run the §4.2 panel "data-first" (load observations, let the PCN drive) or "theory-first" (set a Hamiltonian, let the QFT drive) and both reach valid coupled fixed points — the system has no preferred direction of fit.',
      ],
    },
    {
      kind: 'prose',
      body: [
        '**Reading the bridge diagram.** The two arrows in the §4.2 panel are not decoration — they are an instantaneous diagnostic of where the system sits in its trajectory. Arrow width is proportional to the magnitude of the source term in each direction: the upward arrow scales with the integrated stress-energy `kappa_R * integral T_{mu nu}[E] dx`, the downward arrow with the gradient-update magnitude `||d<O>/d theta||` driving the Hamiltonian. Three characteristic patterns appear over a training run. (i) **Early — both arrows fat and roughly equal:** the system is far from any fixed point; errors are large on both sides, the coupling is doing real work, and the geometry is actively reorganising. (ii) **Mid-training — one arrow dominates briefly:** typically the upward arrow leads, meaning the PCN has found structure the QFT has not yet absorbed; the downward arrow swells a few frames later as the Hamiltonian update catches up. This "lag handoff" is the visible signature of the bidirectional loop closing. (iii) **Late — both arrows thin and matched:** the coupled system is near a fixed point; residual flow is just stochastic fluctuation around equilibrium.',
        'Persistent asymmetries are diagnostic. A permanently fat upward arrow means the QFT is undersized for the structure the PCN is finding (raise bond dimension or enrich the Hamiltonian ansatz). A permanently fat downward arrow means the PCN is treating QFT readouts as gospel (lower the observation noise or raise prior strength). A permanently thin pair with high free energy means the system is stuck — neither side has the capacity to move the other, and you need to perturb one of them.',
      ],
    },
    {
      kind: 'workedExample',
      example: {
        title: 'Rate equation for h_xx given an E-field spike',
        setup:
          'Suppose at frame `t` the PCN error field at a chosen point (x*, y*) jumps from 0 to amplitude A. Write the per-frame rate equation for the metric perturbation component h_xx at (x*, y*), including the diffusion term, and find the steady-state h_xx for a sustained spike.',
        steps: [
          {
            description:
              'From §4.1, the per-frame source contribution at a point with gradient (E_x, E_y) is h_{mu nu} += kappa_R * (partial_mu E * partial_nu E - (1/2) eta_{mu nu} (partial E)^2). For h_xx specifically: dh_xx/dt|source = kappa_R * (E_x^2 - (1/2)(E_x^2 + E_y^2)) = kappa_R * (1/2)(E_x^2 - E_y^2).',
            result: 'dh_xx/dt|source = (kappa_R / 2) * (E_x^2 - E_y^2).',
            equationId: 'metric-perturbation',
          },
          {
            description:
              'For a localised spike at (x*, y*), choose coordinates so the spike is x-aligned: E_x = A / sigma, E_y = 0 (taking the gradient just outside the peak in the +x direction). Then dh_xx/dt|source = kappa_R * A^2 / (2 sigma^2).',
            result: 'Per-frame source rate scales as A^2 — a doubling of the spike amplitude quadruples the rate.',
          },
          {
            description:
              'The diffusion term comes from the Laplace–Beltrami operator acting on the h field itself. To first order (h small, metric near flat), this reduces to a standard heat-equation term: dh_xx/dt|diff = D * Delta h_xx, where D is the diffusion coefficient. For a localised perturbation of spatial width sigma, Delta h_xx ~ -h_xx / sigma^2, so dh_xx/dt|diff = -D * h_xx / sigma^2.',
            result: 'Per-frame diffusion drains h_xx with rate D / sigma^2.',
            equationId: 'laplace-beltrami',
          },
          {
            description:
              'Combining: dh_xx/dt = kappa_R * A^2 / (2 sigma^2) - D * h_xx / sigma^2. At steady state, dh_xx/dt = 0, giving h_xx*(steady) = kappa_R * A^2 / (2 D).',
            result: 'h_xx*(steady) = kappa_R * A^2 / (2 D) — independent of sigma, linearly in kappa_R, quadratically in A.',
            equationId: 'coupling-descent',
          },
          {
            description:
              'Plug in defaults: kappa_R = 0.1, A = 1.0, D = 0.05. Then h_xx*(steady) = 0.1 * 1 / 0.1 = 1.0 — the metric perturbation saturates at order unity, which is a strong-coupling regime where the linearisation breaks down. This is why the default panel settings have kappa_R closer to 0.01: keep the linear regime valid so the picture in §4.1 stays accurate.',
            result: 'Default kappa_R = 0.01 gives h_xx* ~ 0.1, comfortably in the linear regime.',
          },
        ],
        takeaway:
          'The steady-state metric perturbation under a sustained PCN error spike is set by the ratio kappa_R / D, with quadratic dependence on the spike amplitude. The arrow widths in the §4.2 panel scale with the source term in this equation: high error amplitude makes the upward arrow fat, low diffusion makes the steady-state perturbation tall. Realistic runs are non-steady — the spike fluctuates — but this is the equilibrium picture that frames the dynamics.',
      },
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'coupling-descent',
      expect: [
        'Both arrows visible — neither dominates by more than a factor of ~3.',
        'Free energy F decreases monotonically when the prior is well-matched.',
        '<O> updates lag E updates by 1-2 frames (QFT relaxation timescale).',
        'Arrow widths fluctuate but mean values stabilise within ~50 frames.',
      ],
      pathologies: [
        { signal: 'Upward arrow much wider than downward', cause: 'QFT relaxation faster than PCN; the QFT slavishly tracks PCN priors. Slow down the QFT (smaller imag-time step) or speed up PCN.' },
        { signal: 'Downward arrow invisible', cause: 'Observation noise too high; the PCN ignores <O>. Reduce observation variance until <O> registers.' },
        { signal: 'F oscillates', cause: 'PCN learning rate or kappa_R too large; coupling overshoots. Reduce one or both by factor of 3.' },
        { signal: 'Both arrows stuck at zero', cause: 'System already at fixed point or substrate disconnected. Check whether E and <O> are truly zero or the bridge is broken.' },
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §4 (Coupling)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/network.py', href: '../../../network.py' },
  ],
};
