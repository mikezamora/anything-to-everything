// src/qft_pcn/viz/web/src/routes/learn/articles/qft-mera.tsx
/**
 * §2.2 QFT side — Multi-scale Entanglement Renormalisation Ansatz (MERA).
 *
 * Covers: disentanglers + isometries, the multi-scale tree, log-scaling
 * entropy at criticality, and how the substrate's per-layer χ caps express
 * a renormalisation-group budget.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'qft-mera',
  title: 'MERA — multi-scale entanglement at criticality',
  sectionPath: ['§2 QFT side', '2.2 MERA'],
  prerequisites: ['qft-mps'],
  sections: [
    {
      kind: 'prose',
      body: [
        'MERA — the Multi-scale Entanglement Renormalisation Ansatz — extends the MPS idea to states whose entanglement does *not* satisfy a flat area law. The motivating case is **criticality**: at a quantum phase transition, the entanglement entropy across a length-`L` block scales as `S ~ (c/3) log L` rather than `S ~ const`. An MPS with finite bond dimension caps `S` at `log(chi)` so it cannot represent critical states without unbounded `chi`. MERA fixes this by building entanglement layer-by-layer across length scales: every doubling of the block size adds another constant chunk of entropy, and the geometric sum reproduces the logarithm exactly.',
        'The architecture is a tree of two tensor types. **Disentanglers** are unitaries acting on pairs of adjacent sites; they "factor out" short-range entanglement before coarse-graining. **Isometries** are rectangular tensors that take two neighbouring coarse-grained sites and produce one site at the next layer up. Stacking `O(log N)` such layers — the renormalisation-group flow, made manifest — yields a state whose entanglement is composed of equal contributions from each scale. This is the structural fingerprint of a critical / scale-invariant state, and the substrate (`src/qft_pcn/qft/mera.py` and `mera_evolution.py`) lays it out exactly this way.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'entanglement-entropy',
      caption: 'The entropy quantity MERA was designed to model. The MERA replaces an MPS\'s area-law cap (S ≤ log χ) with a budget that grows additively across log N layers.',
    },
    {
      kind: 'prose',
      body: [
        'A disentangler `U` is a square unitary of dimension `chi^2 × chi^2` acting on two adjacent sites at some level of the tree. The constraint `U^H U = I` is enforced exactly in the substrate by parameterising `U` on its Stiefel manifold (or, equivalently, by polar-projecting after each gradient update). The intuition: at scale `s` the chain has some entanglement structure. Before we coarse-grain by an isometry, we want to *first* remove correlations that live entirely within an adjacent pair, because keeping them would force the isometry to spend its precious bond budget on intra-pair redundancy instead of on genuinely longer-range correlations. The disentangler is the operator that does that local "tidy-up" step before the rough coarse-graining.',
        'An isometry `W` is a `chi × chi^2` rectangular tensor with the constraint `W W^H = I` (a partial isometry; it preserves inner products on its domain). It maps two `chi`-dim sites at level `s` to one `chi`-dim site at level `s + 1`. The constraint guarantees that contracting a MERA with itself (the calculation of a norm, an expectation, or an entropy) collapses neatly via the "causal cone" property: only `O(log N)` tensors enter any local observable\'s computation. That is the algorithmic reason MERA is computationally tractable despite encoding states an MPS cannot.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'The multi-scale structure is the whole game. Read the tree from leaves to root: layer 0 holds the physical sites (the "ultraviolet" of the field), layer 1 holds pairs coarse-grained once, layer `s` holds blocks of `2^s` original sites. The isometric structure makes this a genuine renormalisation-group flow encoded in tensor form: each layer\'s coarse-grained Hamiltonian is exactly the original Hamiltonian conjugated and projected by the disentanglers and isometries above. Fixed-point MERA — where the disentanglers and isometries are the same at every layer — corresponds to a scale-invariant state, which is exactly the algebraic content of a conformal fixed point.',
        'For QPCN purposes this matters in a specific way. If the generative model implied by the QPCN\'s Hamiltonian is gapped, MPS is fine and MERA is overkill. If it is gapless or near-critical (e.g. when the observation targets demand long-range correlations between fields), MERA captures structure MPS cannot, at the cost of more tensors to optimise and stricter isometric constraints to maintain.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'mps-ansatz',
      caption: 'Reference: the MPS factorisation MERA generalises. Where MPS is a 1D chain of rank-3 tensors, MERA is a 2D tree of disentanglers and isometries.',
    },
    {
      kind: 'prose',
      body: [
        'Counting parameters quickly is a useful sanity-check skill. A disentangler at bond dimension `chi` has `chi^4` complex entries minus the unitary constraints (which subtract `chi^4 - chi^2` real degrees of freedom, leaving `chi^2` real free parameters per disentangler in the standard counting). An isometry from two `chi`-sites to one `chi`-site has `chi^3` complex entries minus its isometry constraints (`chi^2`). The full MERA on `N = 2^L` sites has `O(L)` layers, each with `O(N / 2^s)` tensors at depth `s`. Doubling `chi` roughly multiplies the per-tensor parameter count by 8-to-16, which is why per-layer `chi` caps in the substrate are a meaningful knob: they trade representable entanglement budget at each scale for optimisation cost and memory.',
        'The substrate exposes the per-layer cap directly (`mera.py` accepts a `chi` schedule, one entry per layer). A natural default holds `chi` constant across layers; a more sophisticated schedule lets the bottom layer use a smaller `chi` (less long-range entanglement to capture at the UV) and grow modestly into the higher layers. The viz surfaces both the configured cap and the *used* dimension per layer, so a saturated layer in the middle of the tree is visible at a glance.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'The **causal-cone property** is the single algebraic fact that makes MERA computationally tractable, and it deserves a dedicated stare. Pick any single-site operator `O_i` at the bottom (physical) layer. To compute `⟨psi| O_i |psi⟩` you in principle have to contract the entire tree against itself. Because every disentangler is unitary (`U^† U = I`) and every isometry preserves inner products on its domain (`W W^† = I`), most of the tree contracts trivially to identity: tensors that lie outside the past light-cone of `O_i` collapse pairwise with their conjugates and vanish from the computation entirely. What survives is a narrow "causal cone" — a constant-width strip of tensors that flares upward from the support of `O_i` and that has width bounded by a small constant (typically 2 or 3 sites per layer) regardless of `N`. Counting layers gives `O(log N)` tensors that actually enter the contraction.',
        'This is exponentially better than naïve dense evaluation, but the prefactor matters. A two-site observable\'s causal cone has width 4 to 6 sites per layer, and each surviving tensor must be contracted against its conjugate, which scales as some high power of `chi` (`chi^9` is the textbook figure for a two-site observable on a 1D binary MERA). So MERA is asymptotically cheap (poly-log in `N`) but absolutely expensive (a large constant in `chi`). The substrate respects this by keeping per-layer `chi` modest and by caching causal-cone contractions: a single observable\'s cone changes by only a few tensors when a neighbouring observable is evaluated, and the substrate reuses the rest. The viz panel for a per-site expectation highlights exactly the cone of tensors that contributed to it.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'entanglement-entropy',
      caption: 'Why we are willing to pay the causal-cone cost: critical systems exhibit `S ~ (c/3) log L` (the Cardy–Calabrese formula), which a flat-bond MPS structurally cannot reach. MERA realises this log-scaling exactly because each layer contributes an equal entropy chunk and there are log_2 L layers spanning a block of length L.',
    },
    {
      kind: 'prose',
      body: [
        'Optimising a MERA is harder than optimising an MPS for two reasons. First, the isometric constraints must be maintained: an unconstrained gradient step will throw a disentangler off the unitary manifold, and naive renormalisation does not recover it. Standard practice (and the substrate\'s approach) is the **environment / SVD update**: compute the tensor\'s linear environment, SVD it, and replace the tensor by `U V^H` (the polar factor). This is the analytic minimum of the linearised objective subject to the unitary constraint and converges very fast for well-behaved problems. Second, the causal-cone property gives `O(log N)` cost per observable — exponentially better than dense — but the prefactor is large (`chi^9` or worse for two-site observables), so MERA is most useful when you genuinely need scale-invariant representation.',
      ],
    },
    {
      kind: 'workedExample',
      example: {
        title: 'Count the parameter dimension of a vacuum MERA on 4 leaves',
        setup:
          'Build the smallest non-trivial MERA: 4 physical sites (each of dim d = 2), one layer of disentanglers, one layer of isometries, bond dimension χ = 2 throughout. Count the real free parameters and note how the count would scale with χ and depth.',
        steps: [
          {
            description:
              'Identify the tensors. Layer 0 → layer 1: with N = 4 sites, the standard MERA has one disentangler across the middle bond ({site 2, site 3}) and one isometry per coarse-grained pair → 2 isometries (covering sites {1,2} and {3,4}). One layer total here.',
            result: 'Tensors: 1 disentangler U (acts on χ²=4-dim space → 4×4 unitary), 2 isometries W (each 2 × 4).',
          },
          {
            description:
              'Count disentangler parameters. A 4×4 complex unitary lives on U(4), a real manifold of dimension 4² = 16. So U contributes 16 real parameters.',
            result: 'Disentangler contribution: 16 real DOF.',
          },
          {
            description:
              'Count isometry parameters. A 2×4 partial isometry W satisfies W W^H = I_2; this is the (complex) Stiefel manifold V_2(C^4) of real dimension 2 · (2 · 4) - 2² = 16 - 4 = 12. Two isometries → 24 real DOF.',
            result: 'Isometry contribution: 24 real DOF.',
          },
          {
            description:
              'Add a top-of-tree state vector on the single remaining χ-dim site (the "vacuum" of the coarse-grained chain). A normalised complex 2-vector has 2·2 - 1 = 3 real DOF, modding out the global phase removes 1 more → 2 real DOF.',
            result: 'Top-tensor contribution: 2 real DOF.',
          },
          {
            description:
              'Total: 16 + 24 + 2 = 42 real parameters for a 4-site MERA at χ = 2. For comparison, the full 4-qubit pure-state manifold has 2^5 - 2 = 30 real DOF — so this MERA *over-parameterises* the 4-site Hilbert space (the redundancy is gauge from the disentangler / isometry constraints not being fully tight). The scaling law is what matters as χ grows: a disentangler at χ acts on a χ²×χ² Hilbert space so its parameter cost is O(χ⁴), while a two-site → one-site isometry living on the Stiefel manifold V_χ(C^{χ²}) costs O(χ³). Doubling χ therefore multiplies per-tensor parameter count by 8-to-16, which is the principled way to spend parameters where the entanglement actually lives (architecture §2.3).',
            result: '42 real parameters at χ = 2. Per-layer χ controls a power-of-χ scaling of the parameter count.',
          },
        ],
        takeaway:
          'Even the smallest MERA has more knobs than the smallest MPS for the same N. The pay-off is that those extra knobs are precisely the disentangling and multi-scale degrees of freedom required to model critical states. The per-layer χ schedule in the substrate is the principled way to spend parameters where the entanglement actually lives.',
      },
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'imag-time-evolution',
      expect: [
        'During imaginary-time evolution of a MERA, per-layer "captured entropy" climbs until it saturates the per-layer χ cap or matches the true sub-tree entanglement of the target ground state.',
        'A scale-invariant ground state (e.g. a conformal fixed point) shows roughly equal entropy contributions across layers, which is the algebraic signature MERA was designed to express.',
        'Disentanglers updated via the environment/SVD scheme converge in a small number of sweeps; gradient noise is suppressed because each update is the analytic optimum of a linearised objective.',
      ],
      pathologies: [
        {
          signal: 'Captured entropy concentrated in the bottom layer only',
          cause: 'The substrate is using MERA where MPS would suffice — the target is gapped / area-law. Either accept the inefficiency or fall back to MPS.',
        },
        {
          signal: 'Captured entropy pinned at the cap in a single mid-tree layer',
          cause: 'That layer\'s χ schedule is too tight for the scale of correlations it must encode. Bump that layer\'s χ rather than uniformly inflating all layers.',
        },
        {
          signal: 'Disentangler norm drifting from 1 over a sweep',
          cause: 'Polar / SVD re-projection step skipped or numerically unstable; insert an explicit re-orthogonalisation between updates.',
        },
        {
          signal: 'Per-layer entropy oscillates instead of monotonically rising',
          cause: 'Coarse-graining order is inconsistent across sweeps, breaking the causal cone. Confirm the disentanglers and isometries are applied in the documented bottom-up order.',
        },
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §2.3 (MERA option)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/qft/mera.py (MERA construction)', href: '../../src/qft_pcn/qft/mera.py' },
    { label: 'src/qft_pcn/qft/mera_evolution.py (imag-time sweep)', href: '../../src/qft_pcn/qft/mera_evolution.py' },
  ],
};
