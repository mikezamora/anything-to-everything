// src/qft_pcn/viz/web/src/routes/learn/articles/qft-mps.tsx
/**
 * §2.1 QFT side — Matrix Product States (MPS) as the QPCN's belief carrier.
 *
 * Covers: tensor-network ansatz, bond dimension as entanglement budget,
 * canonical form, SVD truncation, and how the bond cap shows up in the
 * substrate's `apply_two_site_gate` / `entanglement_entropy` paths.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'qft-mps',
  title: 'Matrix Product States — the belief carrier',
  sectionPath: ['§2 QFT side', '2.1 MPS'],
  prerequisites: ['foundations-hilbert-operators', 'foundations-vectors-tensors'],
  sections: [
    {
      kind: 'prose',
      body: [
        'A Matrix Product State (MPS) is the structural compromise that makes many-body quantum simulation tractable. Recall the brutal fact from the Hilbert chapter: a state on `N` sites lives in a vector space of dimension `d^N` (with `d` the per-site Hilbert dim, e.g. `d = 2` for qubits or `d = cutoff` for a truncated bosonic Fock space). Storing such a vector for `N = 60` is already a thousand exabytes. We need a parameterisation that costs polynomial memory yet still expresses the physically relevant states. The MPS does exactly that: it writes the global amplitude `<s_1 s_2 ... s_N | psi>` as a contraction of `N` small per-site tensors `A^{s_k}` connected by virtual indices ("bonds") of dimension `chi`. The cost goes from `d^N` to roughly `N d chi^2`. The trade is that not every state is representable — only those whose entanglement across any bipartition is bounded.',
        'This bounded-entanglement constraint is not a bug. It is the entire point. The states we actually want to model — ground states of local Hamiltonians, low-energy thermal states, the QPCN\'s belief over a quantised field — provably have entanglement that obeys an "area law" rather than scaling with system volume. The MPS is the smallest ansatz that captures exactly this class. In the QPCN it carries the quantum belief: a finite-dimensional summary of "what the system thinks the field state is right now." Every observation, every gradient, every imaginary-time projection lives on this object.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'mps-ansatz',
      caption: 'The MPS factorisation. The site tensors A^{s_k} are the learnable parameters; the bond indices are summed over and never directly observed.',
    },
    {
      kind: 'prose',
      body: [
        'The single most important hyperparameter of an MPS is the **bond dimension** `chi`. It sits between adjacent site tensors as the shared dimension of the bond index. Geometrically, cut the chain between sites `k` and `k+1`; the singular-value spectrum of the state, viewed as a matrix from the left half to the right half, has at most `chi` non-zero entries. That spectrum is exactly what determines the entanglement entropy across the cut. So `chi` is, very literally, an **entanglement budget**: the maximum von-Neumann entropy across any bond is `log(chi)`. Doubling `chi` adds one bit of entanglement headroom and quadruples the memory of the affected tensor (the cost is `O(chi^2)` per bond).',
        'In the substrate (`src/qft_pcn/qft/mps.py`) you will see a `chi_max` parameter passed to `apply_two_site_gate`. That cap is enforced after every entangling operation: the gate inflates the bond, an SVD is taken, and only the top `chi_max` singular values survive. The discarded weight is renormalised away. Choose `chi_max` too small and the MPS cannot represent the true post-gate state; choose it too large and you waste memory and time on singular values too small to matter.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'The same state has many equivalent MPS representations because each bond admits a `GL(chi)` gauge: insert `M M^{-1}` between two tensors and absorb `M` into one neighbour, `M^{-1}` into the other. The state is unchanged. To pin down the freedom we work in a **canonical form**. Left-canonical at site `k` means every tensor to the left of `k` is an isometry when reshaped as a matrix from `(left-bond, physical)` to `right-bond`; right-canonical means the mirror condition. In **mixed canonical form** the chain is left-canonical up to site `k`, right-canonical from site `k+1`, with a diagonal singular-value matrix `S` sitting on the central bond. This is the form in which (a) the reduced density matrix on either half is just `S^2`, (b) entanglement entropy is `-sum_i s_i^2 log s_i^2`, and (c) the optimal truncation is *literally* "drop the smallest singular values."',
        'Why this matters operationally: every non-trivial MPS routine in the substrate either lives in or moves through canonical form. `apply_two_site_gate` SVD-truncates and re-canonicalises on the spot. Energy estimators sweep canonical centres along the chain. Imaginary-time evolution (a TEBD-style Trotter sweep) is a sequence of two-site gates each of which assumes you arrived in canonical form. If you ever inspect bond dimensions in the viz and one bond is huge while neighbours are tiny, the chain is out of canonical form — the "bond dim" is reflecting gauge garbage rather than real entanglement.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'entanglement-entropy',
      caption: 'In canonical form the entropy across a bond is the Shannon entropy of the squared singular values on that bond — a direct, cheap diagnostic.',
    },
    {
      kind: 'prose',
      body: [
        'SVD truncation is the mechanism. After an entangling two-site gate the bond between the two affected sites can grow to `chi * d`. We reshape the merged tensor into a matrix from `(left-bond, left-physical)` to `(right-physical, right-bond)`, take its SVD `M = U S V^H`, and keep the largest `chi_max` singular values (and the corresponding columns of `U`, rows of `V^H`). The truncation error is the sum of the squares of the discarded singular values; the substrate renormalises so the kept state has unit norm. The choice of "largest" is the optimal one in the Frobenius / `L^2` sense — no other rank-`chi_max` approximation has lower error, which is the content of the Eckart–Young theorem applied to states.',
        'Two failure modes are worth burning into your intuition. First, **bond-cap saturation**: if the truncation always retains `chi_max` non-negligible singular values, the cap is biting and the represented state is a controlled approximation rather than an exact gate application. The viz surfaces this by showing the bond-dimension trace pinned at `chi_max` along several bonds. Second, **normalisation drift**: the renormalisation after truncation is exact for a single gate but compounds across a Trotter sweep. Long evolutions accumulate small phase / norm errors; periodic re-orthogonalisation is cheap and worth doing.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'Moving the **orthogonality centre** is the workhorse operation of every practical MPS routine. Given a chain in mixed canonical form with its singular-value matrix `S` on bond `k`, "sweeping right" means: contract `S` into the right neighbour to make site `k+1` non-canonical, factor the result via QR (or SVD) into an isometric left-tensor and a residual `S\'`, leave the isometric piece on site `k+1`, and place `S\'` on bond `k+1`. The centre has moved one site to the right at cost `O(d chi^3)`, with no truncation involved — purely a basis change. Sweeping the centre across the whole chain costs `O(N d chi^3)` and gives you, as a free by-product, every bond\'s up-to-date Schmidt spectrum (and therefore every bond\'s entropy).',
        'Why bother? Because every operation that is cheap *only* when applied at the orthogonality centre — local observable expectation, two-site SVD truncation, single-site DMRG eigenproblem — becomes expensive (and numerically dirty) anywhere else. The standard pattern in `src/qft_pcn/qft/mps.py` is therefore: park the centre on the bond you are about to touch, do the work, then sweep the centre to the next bond before the next touch. A full TEBD step is just this pattern repeated across a Trotterised bond schedule. Recognising this pattern in the substrate code (look for the alternating `right_sweep` / `left_sweep` calls bracketing every gate application) is the difference between reading MPS code as a black box and understanding why each line is in the order it is.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'imag-time-evolution',
      caption: 'Imaginary-time evolution is the dominant consumer of canonical-form sweeps in the QPCN: each Trotter step is a two-site gate application sandwiched between sweeps that park the orthogonality centre on the active bond.',
    },
    {
      kind: 'prose',
      body: [
        'In the QPCN specifically the MPS is the quantum side of the Fusion bridge. Its bonds carry the entanglement structure of the belief; its site tensors are mutated indirectly through Hamiltonian-parameter updates that drive a short imaginary-time projection. The fact that the bond dimension caps representable entanglement is itself a piece of inductive bias: the QPCN cannot encode arbitrarily long-range quantum correlations between distant sites because the bond budget would forbid it. For the kinds of locally-coupled generative models the architecture document targets, this is a feature: it sharply restricts the hypothesis class to physically plausible states, and the resulting variational problem is well-posed instead of being a search over an exponentially big space of nonsense.',
      ],
    },
    {
      kind: 'workedExample',
      example: {
        title: 'Build a 3-site product state by hand and verify entanglement = 0',
        setup:
          'Construct the MPS for the product state |0> ⊗ |0> ⊗ |0> with d=2, then read off the bond singular values and confirm the entanglement entropy across every bond is zero.',
        steps: [
          {
            description:
              'A product state requires bond dimension chi = 1. Each site tensor A^{s_k} has shape (1, d, 1). For |0> the physical component s=0 holds 1 and s=1 holds 0. Concretely: A^{0} = [[1]] and A^{1} = [[0]] on every site.',
            result: 'Three site tensors of shape (1, 2, 1) initialised so only the s=0 slice is non-zero.',
          },
          {
            description:
              'Contract two adjacent tensors across their shared bond: A^{s_1} A^{s_2} = A^{s_1}[0,0] * A^{s_2}[0,0]. This is the (1×1) matrix [[1]] when s_1 = s_2 = 0 and [[0]] otherwise. Extending to three sites the global amplitude <s_1 s_2 s_3|psi> = 1 only when (s_1, s_2, s_3) = (0, 0, 0).',
            result: 'The resulting wavefunction has a single non-zero amplitude — it really is |000>.',
          },
          {
            description:
              'Inspect the bond between sites 1 and 2. The state restricted to "left = site 1" vs "right = sites 2,3" is a rank-1 outer product: |0>_1 ⊗ |00>_{23}. The SVD has exactly one singular value, s_0 = 1.',
            result: 'Singular spectrum on bond 1–2: {1}. Same on bond 2–3.',
            equationId: 'entanglement-entropy',
          },
          {
            description:
              'Apply the von-Neumann entropy formula S = -Σ s_i^2 log s_i^2 = -(1 * log 1) = 0. Zero across every bond.',
            result: 'S(bond 1–2) = 0, S(bond 2–3) = 0. A product state has zero entanglement, by definition.',
          },
        ],
        takeaway:
          'Bond dimension 1 is the algebraic signature of a product state. Any entanglement at all requires chi >= 2; doubling chi adds at most one extra bit of cross-bond entropy. The MPS\'s expressivity is graded continuously by chi, and that is the knob the QPCN turns to control the QFT side\'s capacity.',
      },
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'imag-time-evolution',
      expect: [
        'During an entangling imaginary-time sweep, bond dimensions monotonically grow from their initial values until they hit chi_max (or until the state\'s natural entanglement is reached, whichever is smaller).',
        'In canonical form the largest singular value on each bond is order 1; the spectrum has a heavy head and a light tail. As long as the tail stays orders of magnitude below the head, the chi_max cap is not biting.',
        'Per-bond entropies plateau when the state reaches the Hamiltonian\'s ground manifold. A flat entropy trace alongside a flat <H> trace is a strong "converged" signal.',
      ],
      pathologies: [
        {
          signal: 'Every bond pinned at chi_max with a heavy tail of singular values',
          cause: 'chi_max is too small for the target state. Raise the cap or accept that the substrate is reporting a controlled approximation, not the exact gate.',
        },
        {
          signal: 'A single bond enormous while neighbours are tiny',
          cause: 'Chain is out of canonical form. Re-canonicalise via a sweep of QR/SVD; the gauge garbage will collapse.',
        },
        {
          signal: 'State norm drifting away from 1 over a long sweep',
          cause: 'Accumulated truncation renormalisation; insert a periodic explicit re-normalisation after every M Trotter steps.',
        },
        {
          signal: 'Entropy diverging instead of plateauing',
          cause: 'The Hamiltonian being projected against is gapless or critical; expect log-scaling entropy and budget chi_max accordingly (MERA, not MPS, is the right ansatz at criticality).',
        },
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §2.3 (MPS as belief)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/qft/mps.py (apply_two_site_gate, entanglement_entropy)', href: '../../src/qft_pcn/qft/mps.py' },
  ],
};
