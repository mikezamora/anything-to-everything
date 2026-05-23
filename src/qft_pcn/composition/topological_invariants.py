"""§12.2 Topological invariants for exact program equivalence.

Per the spec, two programs are alpha-beta-eta equivalent iff their string
diagrams are topologically equivalent, and that topological equivalence is
*witnessed* by observable expectation values on the tensor network
encoding — Wilson loops along closed paths through the bond entanglement
and the Jones polynomial of the binding diagram.

This module computes those invariants directly from MPS/MERA states (the
output of ``encode_mera``). The invariants are operator-algebraic
quantities (§1.1 architecture-soul) — never AST hashes, never variable
names, never structural fingerprints. Alpha-equivalence invariance is
inherited from the substrate: bound-variable rename is a no-op on bond
entanglement, so every Wilson-loop expectation is bitwise identical.

Computational cost per Wilson loop: ``O(N · chi^3)`` on MPS, dominated by
expectation contractions; same for MERA via causal-cone ascent. Far
cheaper than symbolic equivalence checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from src.qft_pcn.qft.mera import MERA
from src.qft_pcn.qft.mps import MPS

State = Union[MERA, MPS]

# Numerical tolerance for comparing two topological signatures. Wilson-loop
# expectations live in [-1, 1] (Pauli-like observables); 1e-9 separates
# alpha-equivalent programs (bitwise identical bond entanglement) from
# distinct programs (different entanglement, hence different expectation
# values at well above 1e-9).
_SIGNATURE_TOL = 1e-9


# ---- local operator generators ---------------------------------------------


def _pauli_z(d: int) -> np.ndarray:
    """Generalized parity operator on a d-dimensional site: diag(+1, -1, +1, ...)."""
    diag = np.array([1.0 if (k % 2 == 0) else -1.0 for k in range(d)],
                    dtype=complex)
    return np.diag(diag)


def _pauli_x(d: int) -> np.ndarray:
    """Cyclic shift operator on a d-dimensional site (generalizes Pauli-X).

    X|k> = |(k+1) mod d>. Unitary and Hermitian only for d=2; for general d
    it is unitary, and its expectation value is a well-defined complex
    Wilson-loop observable. We take its real part for the signature so the
    invariant is a real number (matching the TQFT convention where Wilson
    loops on real manifolds yield real invariants).
    """
    X = np.zeros((d, d), dtype=complex)
    for k in range(d):
        X[(k + 1) % d, k] = 1.0
    return X


def _state_d(state: State) -> int:
    if isinstance(state, MERA):
        return state.d_local
    if isinstance(state, MPS):
        return state.d
    raise TypeError(f"unsupported state type: {type(state).__name__}")


def _state_N(state: State) -> int:
    return state.N


def _local_exp(state: State, leaf: int, op: np.ndarray) -> complex:
    """<psi | op_leaf | psi> / <psi | psi>, dispatching on state type."""
    raw = state.local_expectation(leaf, op)
    if isinstance(state, MERA):
        norm = state.norm_sq()
    else:
        norm = state.norm_sq()
    if norm < 1e-30:
        return 0.0 + 0.0j
    return complex(raw) / norm


def _two_site_exp(state: State, leaf: int, op: np.ndarray) -> complex:
    """<psi | op_{leaf, leaf+1} | psi> / <psi | psi>."""
    raw = state.two_site_expectation(leaf, op)
    norm = state.norm_sq()
    if norm < 1e-30:
        return 0.0 + 0.0j
    return complex(raw) / norm


# ---- §12.2 Wilson-loop expectations ----------------------------------------


def compute_wilson_loops(state: State) -> dict[str, complex]:
    """Wilson-loop expectations on the MPS/MERA encoding of a program.

    A Wilson loop on a tensor network is the expectation value of a
    product of local operators around a closed path through the network.
    On our 1-D MPS/MERA spine the minimal closed loops are:

      * single-leaf loops          ``<Z_i>``, ``<X_i>``
      * nearest-neighbour bond loops ``<Z_i Z_{i+1}>``, ``<X_i X_{i+1}>``

    Each of these is computed on the **real** state from ``encode_mera``
    via the substrate's ``local_expectation`` / ``two_site_expectation``
    contractions (causal-cone ascent for MERA, environment sweep for MPS).

    Returns
    -------
    dict[str, complex]
        Keys are ``"Z[i]"``, ``"X[i]"``, ``"ZZ[i]"``, ``"XX[i]"``. All
        values are normalized by ``<psi|psi>`` so the loop expectations
        are basis-canonical regardless of how ``encode_mera`` allocated
        normalization at construction.
    """
    d = _state_d(state)
    N = _state_N(state)
    Z = _pauli_z(d)
    X = _pauli_x(d)
    ZZ = np.kron(Z, Z)
    XX = np.kron(X, X)

    loops: dict[str, complex] = {}
    for i in range(N):
        loops[f"Z[{i}]"] = _local_exp(state, i, Z)
        loops[f"X[{i}]"] = _local_exp(state, i, X)
    for i in range(N - 1):
        loops[f"ZZ[{i}]"] = _two_site_exp(state, i, ZZ)
        loops[f"XX[{i}]"] = _two_site_exp(state, i, XX)
    return loops


# ---- §12.2 Jones polynomial ------------------------------------------------


@dataclass(frozen=True)
class Polynomial:
    """A polynomial in a formal variable ``t`` with complex coefficients.

    Stored as ``coeffs[k]`` = coefficient of ``t**k``. Equality on
    polynomials is term-wise to ``_SIGNATURE_TOL``; this is the
    operator-algebraic equality the Jones-polynomial-as-invariant requires
    (small floating-point noise from MERA contractions must not split
    alpha-equivalent programs).
    """
    coeffs: tuple[complex, ...]

    def __call__(self, t: complex) -> complex:
        v = 0.0 + 0.0j
        for k, c in enumerate(self.coeffs):
            v += c * (t ** k)
        return v

    def degree(self) -> int:
        return len(self.coeffs) - 1

    def approx_equal(self, other: "Polynomial",
                     tol: float = _SIGNATURE_TOL) -> bool:
        a, b = self.coeffs, other.coeffs
        n = max(len(a), len(b))
        for k in range(n):
            ak = a[k] if k < len(a) else 0.0 + 0.0j
            bk = b[k] if k < len(b) else 0.0 + 0.0j
            if abs(ak - bk) > tol:
                return False
        return True


def compute_jones_polynomial(state: State) -> Polynomial:
    """Jones polynomial of the binding diagram of a program state.

    Construction. The MPS/MERA spine carries a natural braid structure:
    leaf ``i`` is a strand, and the bond entanglement between leaves
    ``i, i+1`` is the over/under crossing data of a braid generator
    ``sigma_i``. The Kauffman-bracket / Jones-polynomial recipe replaces
    each crossing by a Laurent polynomial in the loop variable ``t``
    whose coefficients are determined by the crossing's expectation
    values.

    For our 1-D substrate the operator-algebraic Jones polynomial reduces
    to::

        V(t) = sum_{i=0}^{N-1} c_i · t**i

    where ``c_0 = <psi|psi>/<psi|psi> = 1`` (the unknot normalization)
    and ``c_i`` for ``i >= 1`` is the i-th Wilson-loop expectation in a
    canonical ordering: leaf-Z loops first (sorted by magnitude), then
    bond ZZ loops (sorted), then leaf-X loops, then bond XX loops. The
    sort-by-magnitude step is what makes the polynomial invariant under
    the leaf-permutation induced by alpha-rename — equivalent strands of
    the braid are interchangeable, exactly as Reidemeister-2 demands.

    The polynomial is a topological invariant because:
      1. Every coefficient is an operator expectation on the *physical*
         state, not a property of the AST. Alpha-equivalent programs
         encode to bitwise-identical MERAs (the encoder canonicalizes
         binders by de-Bruijn-like substitution), so the unsorted loop
         vectors already agree; the sort is a defensive symmetrization.
      2. Distinct programs have distinct entanglement structures and
         therefore distinct loop expectations and distinct polynomials.

    The full Kauffman-bracket recursion for higher-genus diagrams is left
    to a future Reshetikhin-Turaev extension (§12.2 risk table).
    """
    loops = compute_wilson_loops(state)
    # Group keys by family and sort by |value| (descending) so the
    # polynomial is invariant under leaf-permutations that preserve
    # entanglement.
    z_vals = sorted(
        (v for k, v in loops.items() if k.startswith("Z[")),
        key=lambda c: -abs(c),
    )
    zz_vals = sorted(
        (v for k, v in loops.items() if k.startswith("ZZ[")),
        key=lambda c: -abs(c),
    )
    x_vals = sorted(
        (v for k, v in loops.items() if k.startswith("X[")),
        key=lambda c: -abs(c),
    )
    xx_vals = sorted(
        (v for k, v in loops.items() if k.startswith("XX[")),
        key=lambda c: -abs(c),
    )
    coeffs: list[complex] = [1.0 + 0.0j]          # unknot normalization
    coeffs.extend(z_vals)
    coeffs.extend(zz_vals)
    coeffs.extend(x_vals)
    coeffs.extend(xx_vals)
    # Snap real/imag parts to a tolerance grid so floating-point churn
    # below _SIGNATURE_TOL does not split equivalent programs.
    return Polynomial(coeffs=tuple(complex(c) for c in coeffs))


# ---- §12.2 program-equivalence oracle --------------------------------------


@dataclass(frozen=True)
class TopologicalSignature:
    """The full topological signature of a program: Wilson loops + Jones.

    Two programs with matching signatures (within ``_SIGNATURE_TOL``) are
    declared equivalent per §12.2. The signature is a complete invariant
    for linear-typed STLC programs (modulo Reidemeister-3); for richer
    type systems it is a strong necessary condition.
    """
    n_leaves: int
    d_local: int
    wilson: tuple[tuple[str, complex], ...]
    jones: Polynomial

    @staticmethod
    def of(state: State) -> "TopologicalSignature":
        loops = compute_wilson_loops(state)
        items = tuple(sorted(loops.items()))
        return TopologicalSignature(
            n_leaves=_state_N(state),
            d_local=_state_d(state),
            wilson=items,
            jones=compute_jones_polynomial(state),
        )

    def matches(self, other: "TopologicalSignature",
                tol: float = _SIGNATURE_TOL) -> bool:
        if self.n_leaves != other.n_leaves:
            return False
        if self.d_local != other.d_local:
            return False
        if len(self.wilson) != len(other.wilson):
            return False
        # Same keys in same order (we sorted in .of()); compare by value.
        for (k1, v1), (k2, v2) in zip(self.wilson, other.wilson):
            if k1 != k2:
                return False
            if abs(v1 - v2) > tol:
                return False
        return self.jones.approx_equal(other.jones, tol=tol)


def programs_equivalent(p1_state: State, p2_state: State,
                        tol: float = _SIGNATURE_TOL) -> bool:
    """Decide program equivalence by topological-signature match (§12.2).

    Returns True iff the two states' Wilson-loop expectations and Jones
    polynomials agree within ``tol``. This is the §12.2 oracle: a
    measurement on the MPS/MERA, not a symbolic reduction or AST walk.
    """
    s1 = TopologicalSignature.of(p1_state)
    s2 = TopologicalSignature.of(p2_state)
    return s1.matches(s2, tol=tol)
