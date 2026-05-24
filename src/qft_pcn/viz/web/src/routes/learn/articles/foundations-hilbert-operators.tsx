// src/qft_pcn/viz/web/src/routes/learn/articles/foundations-hilbert-operators.tsx
/**
 * §1.2 Foundations — Hilbert space, operators, expectation values.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'foundations-hilbert-operators',
  title: 'Hilbert space, operators, and expectation values',
  sectionPath: ['§1 Foundations', '1.2 Hilbert & Operators'],
  sections: [
    {
      kind: 'prose',
      body: [
        'A Hilbert space is, for our engineering purposes, a complex vector space equipped with an inner product. That is it. If you can picture `C^d` — vectors of `d` complex numbers — together with the rule that two such vectors `psi` and `phi` have inner product `<psi, phi> = sum_k conj(psi[k]) * phi[k]`, you have a Hilbert space. Quantum states live in such a space. The "ket" notation `|psi>` is just a typographic convention for a column vector; the "bra" `<psi|` is the conjugate-transposed row vector. `<psi|phi>` is a single complex number, the inner product. `|psi><phi|` is the outer product, a matrix.',
        'Why complex? Because we need to represent interference: amplitudes can cancel as well as add, and that requires phases, which require complex numbers. Why the inner product? Because it gives us probability: for a normalised state (`<psi|psi> = 1`), the squared modulus of a coefficient is the probability of observing the corresponding basis outcome. Everything else in elementary quantum mechanics — operators, measurement, expectation — is built from these two pieces.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'An **operator** on a Hilbert space is a linear map from the space to itself. In `C^d` an operator is a `d × d` complex matrix. Operators model "things you do to a state" — time evolution, measurements, parameter shifts. The most important class for our purposes is the **Hermitian** operators: those that equal their own conjugate-transpose, `O = O†`. Hermitian matrices have real eigenvalues and orthogonal eigenvectors, which is exactly the structure you need to interpret them as observables: the eigenvalues are the possible measurement outcomes; the eigenvectors are the states that produce a definite outcome with probability one.',
        'The three Pauli matrices `X`, `Y`, `Z` are the canonical examples for a single qubit (2-dim Hilbert space). `Z = diag(+1, -1)` has eigenvalues `±1`; measuring `Z` on the computational basis state `|0> = (1, 0)` gives `+1` with certainty, and on `|1> = (0, 1)` gives `-1`. `X` swaps the two basis components (it has eigenvectors `(|0> ± |1>)/sqrt(2)` with eigenvalues `±1`). `Y` is `X` with an `i` phase. These three together span the space of 2×2 Hermitian matrices, so any one-qubit Hamiltonian or observable can be written as a real linear combination of `I, X, Y, Z`.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'The single most-used quantity in this codebase is an **expectation value**. If the system is in state `|psi>` and you want to know "what is the average value of observable `O` if I measured it many times on identical copies of `|psi>`," the answer is `<O> = <psi|O|psi>`. As a NumPy expression: `np.vdot(psi, O @ psi)`. That is a single complex scalar; if `O` is Hermitian and `psi` is normalised, the imaginary part is zero and the real part is a real number in the spectrum of `O`.',
        'Why this formula? Decompose `|psi>` in the eigenbasis of `O`: `|psi> = sum_k c_k |o_k>` where `O |o_k> = lambda_k |o_k>`. Then `O|psi> = sum_k c_k lambda_k |o_k>`, and `<psi|O|psi> = sum_k |c_k|^2 lambda_k`. That is precisely the probability-weighted average of the eigenvalues — the definition of "average over many measurements." The Born rule (`|c_k|^2` is the probability of outcome `lambda_k`) is built into the formula.',
        'In the QPCN, observables drive learning. The target of training is a set of "observation targets" — desired expectation values for chosen operators. The Hamiltonian\'s learnable parameters descend the squared mismatch between current expectations `<O>` and target values, which is why every Learn article about the QFT side and the Fusion side comes back to expectation values.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'For many sites — the regime tensor networks live in — the Hilbert space is the tensor product of per-site spaces, so its dimension explodes exponentially in the number of sites `N`. A state in this space is, in principle, a `2^N`-entry vector; an operator is `2^N × 2^N`. You cannot store either as a dense ndarray once `N` is much above 30. Tensor-network ansätze (§1.1) circumvent the state side; for operators, the saving grace is that the Hamiltonians we care about are **local** — sums of one- and two-site terms that act non-trivially only on a few sites at a time. That locality is what makes MPS-based variational algorithms tractable. The next chapter formalises this for the QPCN\'s actual Hamiltonian.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'mps-ansatz',
      caption: 'The MPS is how we will store a state |psi> in this exponentially large Hilbert space without storing 2^N numbers.',
    },
    {
      kind: 'equation',
      equationId: 'entanglement-entropy',
      caption: 'And the von-Neumann entropy of a reduced density matrix quantifies how much information about subsystem A is entangled with its complement — the property the MPS\'s bond dimension explicitly limits.',
    },
    {
      kind: 'workedExample',
      example: {
        title: 'Compute <Z> on the computational basis state |0>',
        setup:
          'Take the single-qubit state |0> = (1, 0)^T and the Pauli-Z operator Z = diag(+1, -1). Compute <psi|Z|psi> by hand and confirm it equals +1.',
        steps: [
          {
            description:
              'Write Z as a 2x2 matrix and |0> as a column. Z = [[1, 0], [0, -1]]. |0> = [[1], [0]].',
            result: 'Both written out explicitly.',
          },
          {
            description:
              'Compute Z |0> by matrix-vector multiplication: row 0 dot |0> = 1*1 + 0*0 = 1; row 1 dot |0> = 0*1 + (-1)*0 = 0. So Z|0> = (1, 0)^T = |0>.',
            result: 'Z|0> = |0> — confirming |0> is an eigenvector of Z with eigenvalue +1.',
          },
          {
            description:
              'Compute the inner product <0|Z|0> = <0|0> = conj(1)*1 + conj(0)*0 = 1 + 0 = 1.',
            result: '<Z> = +1.',
          },
          {
            description:
              'Sanity check with the spectral decomposition argument: |0> has c_+ = 1 (component along the +1 eigenvector |0>) and c_- = 0 (component along |1>), so <Z> = |c_+|^2 * (+1) + |c_-|^2 * (-1) = 1*1 + 0*(-1) = 1.',
            result: 'Same answer, two different ways. The expectation of Z on |0> is +1 with zero variance — every measurement returns +1.',
          },
        ],
        takeaway:
          'Expectation values are matrix-vector multiplies followed by an inner product — three lines of NumPy. The whole QFT side of the QPCN is built on this primitive: state in, operator, scalar out. Targets and gradients then chase these scalars.',
      },
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §2', href: '../../QFT_PCN_ARCHITECTURE.md' },
  ],
};
