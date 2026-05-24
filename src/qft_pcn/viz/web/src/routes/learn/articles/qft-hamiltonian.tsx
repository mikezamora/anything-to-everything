// src/qft_pcn/viz/web/src/routes/learn/articles/qft-hamiltonian.tsx
/**
 * §2.3 QFT side — the QPCN Hamiltonian as a generative model.
 *
 * Covers: one-site and two-site decomposition, the substrate's
 * `Hamiltonian.local_op` / `bond_op` machinery, parameter updates via
 * `update_param`, and how observation targets drive variational descent
 * of ⟨H⟩.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'qft-hamiltonian',
  title: 'The QPCN Hamiltonian — a generative model in operator form',
  sectionPath: ['§2 QFT side', '2.3 Hamiltonian'],
  prerequisites: ['foundations-hilbert-operators', 'qft-mps'],
  sections: [
    {
      kind: 'prose',
      body: [
        'In ordinary machine learning the "generative model" is the distribution your network parameterises; training mutates the parameters to make that distribution explain the data. In the QPCN the role of the generative model is played by the **Hamiltonian** `H`, a Hermitian operator on the many-body Hilbert space. Its ground state — the lowest-eigenvalue eigenvector — is the state the system *would* settle into in the absence of constraints, and its parameters control which state that is. Observation targets are not labels in the usual sense; they are expectation values that the trained `H`\'s ground state is expected to reproduce. Learning is gradient descent on a mismatch between current `⟨O⟩` and target `t_o`, modulating Hamiltonian parameters. The Hamiltonian is, quite literally, the model.',
        'Because the Hilbert space is exponential in the number of sites, the only Hamiltonians we can work with are those with **local** structure. The substrate (`src/qft_pcn/qft/hamiltonian.py`) hard-codes this assumption: every `H` is decomposed as a sum of one-site (`h_i`) and nearest-neighbour two-site (`h_{ij}`) terms. The `Hamiltonian` class exposes `local_op(site)` returning the on-site operator and `bond_op(site)` returning the bond operator that couples sites `site` and `site+1`. These two methods are how every downstream consumer — gates, energy estimators, imaginary-time sweeps — assembles `H` without ever materialising the full `d^N × d^N` matrix.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'hamiltonian-decomp',
      caption: 'The substrate\'s contract. Every Hamiltonian in the QPCN factors into per-site and per-bond terms; nothing else fits in memory and nothing else is needed for the locally-coupled models the architecture targets.',
    },
    {
      kind: 'prose',
      body: [
        'The one-site terms are the easy half. For each site `i` the substrate builds `h_i` from the registered field species: a mass term `m * a^† a` for a bosonic field, a chemical-potential term, a local source term, and so on. These are tiny matrices — for a single bosonic species with `cutoff = K` the on-site operator is `K × K`. They are built once, mutated as their coefficients change, and consumed by the MPS gate machinery directly. The viz panel rendering "on-site Hamiltonian" shows exactly this matrix for the selected site, with role-coloured entries indicating which physical parameter each value tracks.',
        'The two-site (bond) terms carry the dynamics. A typical bond operator is a hopping term `t (a_i^† a_{i+1} + a_{i+1}^† a_i)`, a density–density interaction `V n_i n_{i+1}`, or a multi-species cross term coming from the multifield Lagrangian. Each `bond_op(i)` is a `(d^2) × (d^2)` matrix that the TEBD / two-site-gate routines exponentiate (`exp(-i h_{ij} dt)` for real time, `exp(-h_{ij} d\\tau)` for imaginary time) to produce the small unitary or contracting gate applied to the MPS. The bond-by-bond decomposition is what makes the cost linear in chain length and polynomial in `chi`.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'Parameters in this scheme are the coefficients of the per-term operators: the mass `m`, hopping strength `t`, interaction `V`, and so on. The substrate\'s `update_param(name, value)` method is the sole authorised mutator. It writes the new coefficient into the internal parameter dictionary and invalidates the cached `local_op` / `bond_op` matrices so the next consumer rebuilds them with the new value. Critically the mutation is *not* applied directly to the MPS; the MPS evolves only via the consumer\'s next gate application. This separation is what lets the QPCN treat the Hamiltonian as a true model: the parameter is the thing learned; the state is the thing computed.',
        'How does a target push a parameter? An observation target says "the ground state should have `⟨O⟩ = t_o`." The training loop measures the current `⟨O⟩` (on the current MPS, which is approximately the ground state of the current `H`), forms the residual `r = ⟨O⟩ - t_o`, and updates each parameter `theta` via the gradient of `r²` with respect to `theta`. That gradient is computed either by finite-difference around `update_param` calls or, for Pauli-rotation circuits, by the analytic parameter-shift rule (see the VQC article). Multiple targets sum into a multi-observable residual; the substrate\'s update loop accumulates these into one descent step.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'param-update',
      caption: 'The learning rule on the Hamiltonian parameters. The "model" `H` is mutated to make its ground state\'s expectations match the observation targets.',
    },
    {
      kind: 'equation',
      equationId: 'imag-time-evolution',
      caption: 'How the state is re-aligned after a parameter update: a short imaginary-time projection drives the MPS back toward the new ground state of the updated `H`.',
    },
    {
      kind: 'prose',
      body: [
        '**Fock-space truncation** is the silent assumption underneath every "cutoff = K" parameter in the substrate. A bosonic field at one site has, in principle, an infinite-dimensional Hilbert space — the Fock space spanned by `{|0⟩, |1⟩, |2⟩, ...}` with arbitrarily many quanta. Numerically we cannot store an infinite-dim vector, so we *truncate*: keep only `{|0⟩, ..., |K-1⟩}` and treat creation past `|K-1⟩` as if it left the space. The default `cutoff = 2` in the substrate is the most aggressive choice: each site is a qubit, holding zero or one quanta. This makes every on-site operator a 2×2 matrix and every bond operator a 4×4 matrix, which is precisely the regime where MPS / TEBD costs are tiny.',
        'The price is that any physics that genuinely populates `|2⟩` or higher is lost. For a dilute / low-density field this is harmless — a vacuum or one-particle sector is exactly representable. For a dense or driven field (large `⟨n⟩` per site) the truncation introduces a hard ceiling on representable states: an attempted application of `a^†` to `|K-1⟩` is dropped silently, biasing the dynamics. The substrate exposes the cutoff as a configurable parameter (and the viz reports per-site `⟨n⟩`) precisely so this assumption can be audited. A useful diagnostic: if `⟨n⟩` on any site approaches `K - 1` during a run, bump `K` and re-run; if the trajectory changes meaningfully, the truncation was biting.',
      ],
    },
    {
      kind: 'prose',
      body: [
        '**Locality and the Lieb–Robinson bound** justify the substrate\'s "one-site + nearest-neighbour two-site" structural cap. The Lieb–Robinson theorem states that for any Hamiltonian built from bounded local terms, the effective propagation speed of information / correlations is bounded by a constant `v_LR` set by the operator norms of those local terms: outside the light-cone `|x - y| > v_LR · t`, the commutator `[O_x(t), O_y(0)]` is exponentially suppressed. Crucially, this is true *without* relativity — it is purely a statement about how local couplings cap the rate at which entanglement can spread along the chain.',
        'For MPS-evolvable physics this is exactly the property we need. An imaginary-time sweep of duration `tau` can only entangle sites within `v_LR · tau` of each other; the bond entropy at any cut grows at most linearly in `tau` with slope set by `v_LR`. That is what makes a bounded `chi_max` sufficient for short evolutions — the entanglement budget needed scales with how far light has had time to travel, not with system size. Adding three-site or longer-range terms to the Hamiltonian raises `v_LR` and accelerates entanglement growth, eating the `chi` budget faster. The two-site cap in `bond_op` is therefore not a casual convenience but an explicit commitment to a regime in which MPS evolution remains controlled.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'A non-obvious consequence of this design is that the QPCN\'s Hamiltonian acts as a **bias-and-prior** simultaneously. Adding a term to `H` is a hard prior: states whose energy under that term is large will be exponentially suppressed in the ground state. Tuning a coefficient is a soft prior: lowering `V` weakens an interaction without removing it. The space of representable generative models is constrained by what local Hermitian operators you allow in `local_op` and `bond_op`, and that is a design choice — adding a new field species or a new bond operator is the substrate\'s analogue of adding a new architectural block in a deep network.',
        'A second consequence: because the model is the Hamiltonian, the loss surface inherits the structure of an energy landscape. Convexity is not guaranteed; the same `H` can have several near-degenerate ground states differing by a symmetry. Symmetry-broken minima are real features of the model, not training artefacts — the viz\'s `<H>` trace plateauing at slightly different levels across reruns can reflect genuinely distinct ground sectors rather than optimiser noise.',
      ],
    },
    {
      kind: 'workedExample',
      example: {
        title: 'Write out the Hamiltonian matrix for a single-site mass term, cutoff = 2',
        setup:
          'Take a single bosonic site with truncated Fock cutoff K = 2 (so the on-site Hilbert space has dimension 2 with basis {|0>, |1>}). The mass term is h = m a^† a. Construct the operator matrix explicitly, then read off ⟨h⟩ on the basis states.',
        steps: [
          {
            description:
              'Write the bosonic ladder operators truncated at K = 2. The annihilation operator a is the 2×2 matrix that sends |1> → |0> and |0> → 0. In matrix form a = [[0, 1], [0, 0]] (rows index the output basis state, columns the input). The creation operator a^† is its conjugate-transpose: a^† = [[0, 0], [1, 0]].',
            result: 'a and a^† built as 2×2 matrices.',
          },
          {
            description:
              'Form the number operator n = a^† a. Multiply: a^† a = [[0,0],[1,0]] @ [[0,1],[0,0]] = [[0,0],[0,1]]. This is diag(0, 1) — exactly the operator that returns 0 on |0> and 1 on |1>, as required.',
            result: 'n = diag(0, 1).',
          },
          {
            description:
              'Multiply by the mass coefficient. h = m * n = diag(0, m). The full 2×2 matrix representation of the on-site Hamiltonian is h = [[0, 0], [0, m]].',
            result: 'h = [[0, 0], [0, m]].',
            equationId: 'hamiltonian-decomp',
          },
          {
            description:
              'Compute expectations on the two basis states. ⟨0| h |0⟩ = h_{00} = 0; ⟨1| h |1⟩ = h_{11} = m. The vacuum |0⟩ has zero mass-energy; the one-particle state |1⟩ has mass-energy m. Exactly the physical content of "mass per particle."',
            result: '⟨h⟩_{|0⟩} = 0, ⟨h⟩_{|1⟩} = m. Mutating m via update_param(\'m\', new_value) directly shifts the |1⟩ eigenvalue.',
          },
        ],
        takeaway:
          'The Hamiltonian is a matrix you can write down. Every parameter in update_param has a precise local effect on a small per-site matrix; the global ⟨H⟩ is a sum of such local contractions against the MPS. The QPCN\'s "learn the generative model" reduces, mechanically, to "learn the coefficients of these local matrices so their ground-state expectations match the targets."',
      },
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'param-update',
      expect: [
        'After each `update_param` call, a short imaginary-time evolution of the MPS re-aligns the state with the new ground manifold of H. The variational energy ⟨H⟩ drops monotonically across the inner sweep.',
        'Across outer training steps, the observation residual Σ_o (⟨O⟩ - t_o)² decreases roughly geometrically while the parameter trajectory is in the linear regime of the loss.',
        'Parameters with strong overlap with the targets move first; weakly-coupled parameters drift slowly and may not move at all — a useful clue when reading the per-parameter trace.',
      ],
      pathologies: [
        {
          signal: 'Inner ⟨H⟩ sweep plateauing well above the previous step\'s minimum',
          cause: 'Imaginary-time step size dτ too small or chi_max too tight for the new ground state; bump dτ or the bond cap.',
        },
        {
          signal: 'Observation residual oscillating across outer steps',
          cause: 'Outer learning rate η too large for the local curvature; halve η and reset the optimiser state.',
        },
        {
          signal: 'Energy ⟨H⟩ drifting upward across outer steps',
          cause: 'A parameter update outside the safe region invalidated the cached operators inconsistently; verify update_param invalidation paths and re-run from a checkpoint.',
        },
        {
          signal: 'Two reruns from the same seed plateauing at different ⟨H⟩',
          cause: 'Likely genuinely distinct symmetry-broken ground states rather than a bug; confirm by inspecting per-site observables and comparing to the symmetry generators of H.',
        },
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §3.3 (Hamiltonian as generative model)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/qft/hamiltonian.py (local_op / bond_op / update_param)', href: '../../src/qft_pcn/qft/hamiltonian.py' },
    { label: 'src/qft_pcn/qft/qpcn.py (training loop wiring)', href: '../../src/qft_pcn/qft/qpcn.py' },
  ],
};
