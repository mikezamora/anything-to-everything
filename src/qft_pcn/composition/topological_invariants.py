"""§12.2 Wilson-loop fingerprint for program equivalence (partial).

Per the spec, two programs are alpha-beta-eta equivalent iff their string
diagrams are topologically equivalent. The spec's witness is the *Jones
polynomial* of the binding diagram — a Laurent polynomial in ``t**(1/4)``
obtained by extracting the braid word from the closed loops of the
binding diagram, evaluating the Kauffman bracket recursion
``<L> = A<L_0> + A^{-1}<L_oo>``, and writhe-normalizing.

This module ships only the **Wilson-loop fingerprint** half of that
program. We compute Wilson-loop expectation values directly from MPS/MERA
states (the output of ``encode_mera``); these are operator-algebraic
quantities (§1.1 architecture-soul) — never AST hashes, never variable
names, never structural fingerprints. Alpha-equivalence invariance is
inherited from the substrate: bound-variable rename is a no-op on bond
entanglement, so every Wilson-loop expectation is bitwise identical.

What this module is **not**: it is not the §12.2 Jones polynomial. The
Wilson-loop signature is a **necessary** but not **sufficient**
fingerprint — two structurally distinct programs that happen to share
all Wilson loops would collide. The real Kauffman-bracket evaluation
(braid-word extraction, recursive bracket, writhe normalization) is
tracked in ``EXTENSIONS.md`` under "Real Jones polynomial /
Kauffman-bracket evaluation". Per ``memory/no-placeholders.md`` we do
not dress the Wilson signature up as a polynomial; we ship the honest
fingerprint and log the deferred dependency.

Computational cost per Wilson loop: ``O(N * chi^3)`` on MPS, dominated
by expectation contractions; same for MERA via causal-cone ascent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from src.qft_pcn.qft.mera import MERA
from src.qft_pcn.qft.mps import MPS

State = Union[MERA, MPS]

# Numerical tolerance for comparing two Wilson-loop signatures. Wilson-loop
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
    Wilson-loop observable.
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


# ---- §12.2 Wilson-loop signature -------------------------------------------


@dataclass(frozen=True)
class LoopSignature:
    """Canonical Wilson-loop fingerprint of a program state.

    Stores all Wilson loops as a tuple of ``(loop_name, complex_value)``
    pairs in **lexicographic key order** (no sort-by-magnitude — if bond
    entanglement is faithful per §1.1, alpha-rename leaves the unsorted
    vector bitwise identical, so no "symmetrization" is needed).

    This is **not** the §12.2 Jones polynomial. It is a necessary but
    not sufficient fingerprint; see module docstring and ``EXTENSIONS.md``
    entry "Real Jones polynomial / Kauffman-bracket evaluation".
    """
    loops: tuple[tuple[str, complex], ...]

    def approx_equal(self, other: "LoopSignature",
                     tol: float = _SIGNATURE_TOL) -> bool:
        if len(self.loops) != len(other.loops):
            return False
        for (k1, v1), (k2, v2) in zip(self.loops, other.loops):
            if k1 != k2:
                return False
            if abs(v1 - v2) > tol:
                return False
        return True


def compute_wilson_loop_signature(state: State) -> LoopSignature:
    """Canonical Wilson-loop fingerprint of a program state.

    Construction. Compute every Wilson loop via ``compute_wilson_loops``
    and pack into a frozen, lex-sorted tuple. **No** magnitude-sort: the
    spec's §1.1 binding-as-entanglement principle guarantees that
    alpha-equivalent programs encode to bitwise-identical bond
    entanglement, so the raw unsorted loop dict already agrees pointwise.
    If a test fails without sort-by-magnitude, that exposes a substrate
    bug worth surfacing — not a reason to launder it through a sort.

    The signature is a necessary but not sufficient invariant for §12.2
    program equivalence. Two programs whose Wilson loops disagree are
    definitely inequivalent; two programs whose Wilson loops agree are
    *probably* equivalent but the real Jones polynomial (Kauffman-bracket
    on the braid word from the binding diagram) is required to certify
    equivalence on beta/eta-equivalent pairs. See ``EXTENSIONS.md``.
    """
    loops = compute_wilson_loops(state)
    items = tuple(sorted(loops.items()))  # lex order on keys, not magnitudes
    return LoopSignature(loops=items)


# ---- §12.2 program-equivalence oracle (partial) ----------------------------


@dataclass(frozen=True)
class TopologicalSignature:
    """Wilson-loop fingerprint of a program state, with shape metadata.

    Two programs with matching signatures (within ``_SIGNATURE_TOL``) share
    the Wilson-loop half of the §12.2 invariant. This is **necessary** but
    not **sufficient** for program equivalence — the real Jones polynomial
    (deferred per ``EXTENSIONS.md``) is needed to certify beta/eta-equivalent
    pairs.
    """
    n_leaves: int
    d_local: int
    signature: LoopSignature

    @staticmethod
    def of(state: State) -> "TopologicalSignature":
        return TopologicalSignature(
            n_leaves=_state_N(state),
            d_local=_state_d(state),
            signature=compute_wilson_loop_signature(state),
        )

    def matches(self, other: "TopologicalSignature",
                tol: float = _SIGNATURE_TOL) -> bool:
        if self.n_leaves != other.n_leaves:
            return False
        if self.d_local != other.d_local:
            return False
        return self.signature.approx_equal(other.signature, tol=tol)


def programs_equivalent(p1_state: State, p2_state: State,
                        tol: float = _SIGNATURE_TOL) -> bool:
    """Decide Wilson-loop-fingerprint match for two program states (§12.2 partial).

    Returns True iff the two states' Wilson-loop expectations agree within
    ``tol``. This is a **necessary** condition for §12.2 program
    equivalence; the **sufficient** condition (real Jones polynomial via
    Kauffman-bracket evaluation) is deferred per ``EXTENSIONS.md``.

    A True return does not certify equivalence; a False return does
    certify inequivalence.
    """
    s1 = TopologicalSignature.of(p1_state)
    s2 = TopologicalSignature.of(p2_state)
    return s1.matches(s2, tol=tol)
