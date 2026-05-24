"""Matrix Product Operator representation.

An MPO encodes an operator `O` acting on the same N-site Hilbert space
as the MPS, as a chain of rank-4 tensors `W[k]` of shape
`(chi_left, d, d, chi_right)`. The two physical dimensions are
(ket_out, ket_in) — i.e. `W[k][a, s_out, s_in, b]` is the matrix
element of the local operator block between the outgoing and incoming
physical configurations at site k, with `a`/`b` carrying the operator
bond.

This module is the operator-algebraic dual of `mps.py` and is the
prerequisite substrate for §12.9 (self-modification meta-Hamiltonian),
which encodes the lower-level Hamiltonian as a STATE of a meta-QPCN
living in the operator Hilbert space. It also enables a (partial)
§12.17 QCA-index reading over MPO forms.

Conventions:

- Index order per tensor: ``(chi_l, d_out, d_in, chi_r)``.
- Action on an MPS site tensor ``A[k]`` of shape
  ``(b_l, d, b_r)`` yields a new tensor of shape
  ``(chi_l * b_l, d, chi_r * b_r)`` — operator and state bonds are
  merged. (See `apply_to_mps`.)
- Composition `MPO_a @ MPO_b` (state-vector convention: `a` acts after
  `b`) contracts via the inner physical legs: the *bra* of `a` meets
  the *ket* of `b`, yielding `(chi_l_a * chi_l_b, d, d, chi_r_a * chi_r_b)`.
- Boundary operator bonds are dimension 1 (closed at left and right).

Prototype scope:

- ``apply_to_mps`` grows bond dimensions multiplicatively; no SVD
  truncation/compression is performed (deferred — flagged in
  `EXTENSIONS.md`). For meta-Hamiltonian prototyping at small N this
  is the honest behaviour and avoids hiding bond growth.
- `from_hamiltonian_sum` uses the standard Schollwock-review
  bond-dimension-2 boundary construction for `H = sum_i h_i` with
  per-site single-site terms (the most common starting point for
  meta-Hamiltonian assembly). Two-site or longer-range terms are not
  built by this convenience constructor (an explicit-tensor MPO can
  still be assembled by the caller using the same conventions).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .mps import MPS


@dataclass
class MPO:
    """List-of-tensors MPO. Each tensor: ``(chi_l, d, d, chi_r)``.

    Physical indices are ordered ``(d_out, d_in)``: ``W[a, s, t, b]``
    contributes to ``<s| O |t>`` along operator-bond labels ``(a, b)``.
    """

    tensors: list[np.ndarray]

    def __post_init__(self) -> None:
        for k, w in enumerate(self.tensors):
            if w.ndim != 4:
                raise ValueError(
                    f"site {k} MPO tensor must be rank-4, got shape {w.shape}"
                )
            if w.shape[1] != w.shape[2]:
                raise ValueError(
                    f"site {k} MPO tensor must be square on physical legs, "
                    f"got d_out={w.shape[1]}, d_in={w.shape[2]}"
                )
        if self.tensors:
            if self.tensors[0].shape[0] != 1:
                raise ValueError(
                    "left boundary operator bond must have dimension 1"
                )
            if self.tensors[-1].shape[-1] != 1:
                raise ValueError(
                    "right boundary operator bond must have dimension 1"
                )
            for k in range(len(self.tensors) - 1):
                if self.tensors[k].shape[-1] != self.tensors[k + 1].shape[0]:
                    raise ValueError(
                        f"operator bond mismatch between sites {k} and "
                        f"{k + 1}: {self.tensors[k].shape[-1]} vs "
                        f"{self.tensors[k + 1].shape[0]}"
                    )

    @property
    def N(self) -> int:
        return len(self.tensors)

    @property
    def d(self) -> int:
        return self.tensors[0].shape[1]

    def copy(self) -> "MPO":
        return MPO(tensors=[w.copy() for w in self.tensors])

    # ---- construction ------------------------------------------------------

    @classmethod
    def from_local_operators(cls, ops: list[np.ndarray]) -> "MPO":
        """Product MPO ``O = O_0 ⊗ O_1 ⊗ ... ⊗ O_{N-1}``.

        Each `ops[k]` is a `(d, d)` matrix. Resulting MPO has trivial
        operator bond dimension 1 everywhere.
        """
        tensors: list[np.ndarray] = []
        for k, op in enumerate(ops):
            if op.ndim != 2 or op.shape[0] != op.shape[1]:
                raise ValueError(
                    f"ops[{k}] must be a square matrix, got shape {op.shape}"
                )
            d = op.shape[0]
            tensors.append(op.reshape(1, d, d, 1).astype(complex))
        return cls(tensors=tensors)

    @classmethod
    def identity(cls, N: int, d: int) -> "MPO":
        eye = np.eye(d, dtype=complex)
        return cls.from_local_operators([eye for _ in range(N)])

    @classmethod
    def from_hamiltonian_sum(
        cls,
        local_terms: list[tuple[int, np.ndarray]],
        N: int,
        d: int,
    ) -> "MPO":
        """Build the MPO for ``H = sum_i h_i`` where each ``h_i`` is a
        single-site operator on site ``i``.

        Uses the standard Schollwock bond-dim-2 sum-of-local-terms
        construction. At each site `k` the bulk operator-block (in the
        operator basis ``[done, not_yet_started]``) reads:

            W[k] = [[ I  ,  h_k ],
                    [ 0  ,  I   ]]

        where each cell is a ``(d, d)`` matrix. Left boundary picks
        the bottom row (state "not yet started") and right boundary
        picks the left column (state "done"), giving exactly one ``h``
        insertion per term in the expansion.

        ``local_terms`` is a list of ``(site_index, (d, d)-matrix)``
        pairs; multiple terms on the same site are summed pointwise.
        Sites not mentioned receive the zero operator (so the running
        site identity is the only contribution there).
        """
        if N <= 0:
            raise ValueError("N must be positive")

        # Aggregate per-site operators.
        h_per_site: list[np.ndarray] = [
            np.zeros((d, d), dtype=complex) for _ in range(N)
        ]
        for site, op in local_terms:
            if not 0 <= site < N:
                raise ValueError(f"local term site {site} out of range [0, {N})")
            if op.shape != (d, d):
                raise ValueError(
                    f"local term at site {site} has shape {op.shape}, "
                    f"expected ({d}, {d})"
                )
            h_per_site[site] = h_per_site[site] + op.astype(complex)

        I = np.eye(d, dtype=complex)
        Z = np.zeros((d, d), dtype=complex)

        # Bulk site tensor: chi_l = chi_r = 2, basis (done, not_yet).
        # Row index a = operator-bond-left, column index b = operator-bond-right.
        # W[a, :, :, b]:
        #   W[0, :, :, 0] = I    (already finished; stay finished)
        #   W[0, :, :, 1] = 0    (cannot go from done to not-yet)
        #   W[1, :, :, 0] = h_k  (insert local term here, transition to done)
        #   W[1, :, :, 1] = I    (still propagating; have not inserted yet)
        def bulk(h: np.ndarray) -> np.ndarray:
            w = np.zeros((2, d, d, 2), dtype=complex)
            w[0, :, :, 0] = I
            w[0, :, :, 1] = Z
            w[1, :, :, 0] = h
            w[1, :, :, 1] = I
            return w

        if N == 1:
            # Trivially: H = h_per_site[0]; bond dim 1.
            return cls(tensors=[h_per_site[0].reshape(1, d, d, 1)])

        tensors: list[np.ndarray] = []
        # Left boundary: select the "not yet started" row only (a = 1).
        left = bulk(h_per_site[0])[1:2, :, :, :]  # shape (1, d, d, 2)
        tensors.append(left)
        # Bulk.
        for k in range(1, N - 1):
            tensors.append(bulk(h_per_site[k]))
        # Right boundary: select the "done" column only (b = 0).
        right = bulk(h_per_site[N - 1])[:, :, :, 0:1]  # shape (2, d, d, 1)
        tensors.append(right)
        return cls(tensors=tensors)

    # ---- action and contraction ------------------------------------------

    def apply_to_mps(self, mps: MPS) -> MPS:
        """Return a new MPS encoding ``O|psi>``.

        The resulting MPS has bond dimension equal to the product of
        the MPO bond and the input MPS bond at every cut. No SVD
        truncation is performed (prototype — see module docstring).
        """
        if mps.N != self.N:
            raise ValueError(
                f"length mismatch: MPS has {mps.N} sites, MPO has {self.N}"
            )
        new_tensors: list[np.ndarray] = []
        for k in range(self.N):
            A = mps.tensors[k]                                # (b_l, d, b_r)
            W = self.tensors[k]                               # (chi_l, d, d, chi_r)
            if A.shape[1] != W.shape[1]:
                raise ValueError(
                    f"site {k} physical-dim mismatch: MPS d={A.shape[1]}, "
                    f"MPO d={W.shape[1]}"
                )
            # T[chi_l, b_l, s_out, chi_r, b_r] = sum_{s_in}
            #     W[chi_l, s_out, s_in, chi_r] * A[b_l, s_in, b_r]
            T = np.einsum('aSsB,lsr->alSBr', W, A)
            chi_l, b_l, d, chi_r, b_r = T.shape
            new_tensors.append(
                T.reshape(chi_l * b_l, d, chi_r * b_r).astype(complex)
            )
        return MPS(tensors=new_tensors)

    def compose(self, other: "MPO") -> "MPO":
        """Return the MPO for ``self ∘ other`` (operator product).

        Acts as ``(self ∘ other) |psi> = self ( other |psi> )``. On
        each site, the ket index of `self` is contracted with the bra
        index of `other`, and operator bonds are merged.
        """
        if other.N != self.N:
            raise ValueError(
                f"length mismatch: self has {self.N} sites, other has {other.N}"
            )
        if other.d != self.d:
            raise ValueError(
                f"physical-dim mismatch: self d={self.d}, other d={other.d}"
            )
        new_tensors: list[np.ndarray] = []
        for k in range(self.N):
            A = self.tensors[k]              # (aL, s_out, s_mid, aR)
            B = other.tensors[k]             # (bL, s_mid, s_in, bR)
            # M[aL, bL, s_out, s_in, aR, bR] =
            #     sum_{s_mid} A[aL, s_out, s_mid, aR] * B[bL, s_mid, s_in, bR]
            M = np.einsum('aSmA,bmtB->abStAB', A, B)
            aL, bL, s_out, s_in, aR, bR = M.shape
            new_tensors.append(
                M.reshape(aL * bL, s_out, s_in, aR * bR).astype(complex)
            )
        return MPO(tensors=new_tensors)

    def expectation(self, mps: MPS) -> complex:
        """Compute ``<psi| O |psi>`` via the standard sandwich contraction.

        Environment carries indices ``(chi_op, b_bra, b_ket)``. At each
        site we contract the MPS ket, the MPO tensor, and the MPS bra
        (conjugated).
        """
        if mps.N != self.N:
            raise ValueError(
                f"length mismatch: MPS has {mps.N} sites, MPO has {self.N}"
            )
        # Boundary: chi_op_left = b_bra_left = b_ket_left = 1.
        env = np.ones((1, 1, 1), dtype=complex)
        for k in range(self.N):
            A = mps.tensors[k]            # (b_l, d, b_r)         ket
            W = self.tensors[k]           # (chi_l, d, d, chi_r)
            Ac = A.conj()                 # (b_l, d, b_r)         bra
            if A.shape[1] != W.shape[1]:
                raise ValueError(
                    f"site {k} physical-dim mismatch: MPS d={A.shape[1]}, "
                    f"MPO d={W.shape[1]}"
                )
            # env[c, p, q] * A[q, t, q'] * W[c, s, t, c'] * Ac[p, s, p']
            #   -> env'[c', p', q']
            env = np.einsum(
                'cpq,qtr,cstd,psu->dur', env, A, W, Ac
            )
        # All right-boundary bonds are dim 1.
        return complex(env[0, 0, 0])

    # ---- diagnostics ------------------------------------------------------

    def bond_dimensions(self) -> list[int]:
        """Operator-bond dimensions between consecutive sites."""
        return [w.shape[-1] for w in self.tensors[:-1]]

    def to_dense(self) -> np.ndarray:
        """Materialize the full ``d**N × d**N`` operator matrix.

        Intended for tests and tiny systems only — exponential in N.
        """
        d = self.d
        # Carry a tensor with axes (chi, out_0, ..., out_k, in_0, ..., in_k).
        # Start as W[0] reshaped so chi sits at axis 0.
        W0 = self.tensors[0]                  # (1, d, d, chi)
        # (chi_r, d_out, d_in)
        acc = W0.reshape(d, d, W0.shape[-1]).transpose(2, 0, 1)
        # acc shape: (chi, d_out_total, d_in_total) with totals = d^(k+1)
        for k in range(1, self.N):
            W = self.tensors[k]               # (chi_l, d_o, d_i, chi_r)
            # acc: (cL, O, I); W: (cL, p, q, cR).
            # tmp[O, p, I, q, cR] = sum_cL acc[cL, O, I] * W[cL, p, q, cR]
            tmp = np.einsum('aOI,apqb->OpIqb', acc, W)
            O_dim, p_dim, I_dim, q_dim, cR = tmp.shape
            acc = (
                tmp.transpose(4, 0, 1, 2, 3)             # (cR, O, p, I, q)
                .reshape(cR, O_dim * p_dim, I_dim * q_dim)
            )
        # Right boundary chi = 1.
        return acc[0]
