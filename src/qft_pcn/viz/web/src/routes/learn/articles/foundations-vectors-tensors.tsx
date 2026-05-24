// src/qft_pcn/viz/web/src/routes/learn/articles/foundations-vectors-tensors.tsx
/**
 * §1.1 Foundations — Vectors & Tensors, presented from a software-engineer
 * baseline: ndarrays, rank, contraction, tensor networks.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'foundations-vectors-tensors',
  title: 'Vectors, tensors, and contraction',
  sectionPath: ['§1 Foundations', '1.1 Vectors & Tensors'],
  sections: [
    {
      kind: 'prose',
      body: [
        'For everything that follows in the QFT side of the architecture, you need a comfortable working model of tensors. The good news: if you have ever written `import numpy as np`, you already have one. A tensor, in our context, is just an `np.ndarray`. A vector is a 1-D ndarray. A matrix is a 2-D ndarray. A tensor of "rank N" — also called an N-leg tensor — is an N-dimensional ndarray. The word "rank" is overloaded in math (it can also mean the rank of a matrix, i.e. the dimension of its image), so in tensor-network land people increasingly say "number of legs" or "order". We will use "leg".',
        'The mental picture worth keeping is that each leg of a tensor has a name and a size. A site tensor in an MPS has three legs: a left bond leg, a right bond leg, and a physical leg. The bond legs have size `chi` (the bond dimension), the physical leg has size `d` (the local Hilbert dimension, typically 2 for qubits). So one MPS site tensor in NumPy is just `A.shape == (chi_left, d, chi_right)`. Nothing exotic — three nested for-loops would index it.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'The operation that turns a pile of tensors into useful state is **contraction**. Contraction means: pair up two legs (one from each of two tensors, or two legs of the same tensor) that have the same size, sum over the shared index. That is exactly what `np.einsum` does. Matrix multiplication is the simplest case: `C[i,k] = sum_j A[i,j] * B[j,k]`, i.e. `np.einsum("ij,jk->ik", A, B)`. The contracted index `j` disappears from the output; the free indices (`i` and `k`) remain.',
        'Tensor networks generalise this. Take dozens of small tensors, draw them as nodes, draw a line between every pair of legs that should be summed over, and the whole diagram is one giant scalar (if no legs are left free) or a smaller tensor (if some legs are free). The full wavefunction of a system of `N` qubits is a rank-N tensor of shape `(2,2,...,2)` — that is `2^N` entries, untractable for N > ~40. A matrix product state replaces it with `N` rank-3 tensors of size `(chi, 2, chi)` strung in a chain, total parameter count `~ N * chi^2 * d`, linear in N. The contraction of the chain reproduces the wavefunction; the size of `chi` bounds how much entanglement that wavefunction can carry.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'Two practical points worth internalising before §2. First, **leg order matters for performance but not for math**. You can transpose an ndarray\'s legs at will (`A.transpose(2, 0, 1)`), and the abstract tensor is unchanged; but the cost of contracting a network is sensitive to the order in which you do pairwise contractions, sometimes by many orders of magnitude. Libraries like `opt_einsum` (or the `ncon` convention used in tensor-network code) exist exactly to pick a good order. Second, **contraction is associative but not commutative for the picture**. You can contract A·B first and then with C, or B·C first and then with A — same answer, possibly very different intermediate sizes. This is why MERA — which has a tree-like contraction structure — gives different numerics than MPS even when they encode the same state.',
        'In our codebase the convention is: every tensor leg has a documented role (bond / physical / virtual / coupling), and the contractor (`mps.contract`, the MERA evaluator) is responsible for ordering. You will rarely write a raw einsum yourself; you will manipulate site tensors and let the substrate machinery do the heavy lifting.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'A final word on terminology so the next chapter does not surprise you. A "tensor product" of vector spaces, written `V ⊗ W`, is the space whose elements are all pairs `(v, w)` treated as one composite, with dimension `dim V · dim W`. If `V` is the state space of one qubit (dim 2), then the joint space of N qubits is `V ⊗ V ⊗ ... ⊗ V`, dim `2^N`. The MPS ansatz is exactly the statement "do not try to store an arbitrary vector in this huge product space; only store the ones that have low entanglement, parametrised by a chain of small tensors." Tensor networks are, in this sense, a structured low-rank decomposition for very-high-dimensional state vectors — directly analogous to how SVD-truncation compresses matrices, but generalised to many "axes" at once.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'mps-ansatz',
      caption: 'A glimpse ahead: the MPS we will build in §2.1 is exactly this contraction of N rank-3 tensors. Each A^{s_k} is one ndarray of shape (chi, d, chi).',
    },
    {
      kind: 'equation',
      equationId: 'entanglement-entropy',
      caption: 'And the von-Neumann entropy across any cut of the chain is bounded by log(chi) — bond dimension literally caps entanglement.',
    },
    {
      kind: 'workedExample',
      example: {
        title: 'Contract a 3-leg tensor with a vector',
        setup:
          'Let T be a (2, 2, 2) tensor with T[i,j,k] = i + 2*j + 4*k, and let v be the length-2 vector [1, 10]. Contract T\'s middle leg (j) with v to produce a (2, 2) matrix M, then compute the single entry M[0, 1] by hand.',
        steps: [
          {
            description:
              'Write out the full T as 8 numbers, indexed (i, j, k): T[0,0,0]=0, T[0,0,1]=4, T[0,1,0]=2, T[0,1,1]=6, T[1,0,0]=1, T[1,0,1]=5, T[1,1,0]=3, T[1,1,1]=7.',
            result: '8 scalar entries, one per (i, j, k) triple.',
          },
          {
            description:
              'The contraction M[i, k] = sum over j of T[i, j, k] * v[j]. So for each (i, k) we sum two terms: j=0 weighted by v[0]=1, j=1 weighted by v[1]=10.',
            result: 'M[i, k] = T[i, 0, k] * 1 + T[i, 1, k] * 10.',
          },
          {
            description:
              'Compute M[0, 1] specifically. We need T[0, 0, 1] and T[0, 1, 1]. From step 1: T[0, 0, 1] = 4, T[0, 1, 1] = 6.',
            result: 'M[0, 1] = 4 * 1 + 6 * 10 = 4 + 60 = 64.',
          },
          {
            description:
              'Sanity check via einsum mental model: np.einsum("ijk,j->ik", T, v)[0, 1] would give the same 64. The j leg has been "contracted away," reducing rank from 3 to 2.',
            result: 'Confirmed: contraction reduces leg count by exactly 2 per pair summed (one from each side).',
          },
        ],
        takeaway:
          'Contraction is just "for the shared index, multiply elementwise and sum." Once you see this, MPS, MERA, and even attention layers all become small variations on the same elementwise-multiply-then-sum theme — the cleverness is entirely in choosing which legs to share and in what order to contract.',
      },
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §2.3', href: '../../QFT_PCN_ARCHITECTURE.md' },
  ],
};
