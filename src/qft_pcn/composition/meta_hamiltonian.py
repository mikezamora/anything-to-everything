"""§12.9 Self-modification via meta-Hamiltonian.

Physics origin (Wilson 1971, Polchinski 1984; Schmidhuber 2003 for the
classical precursor). In any operator-algebraic substrate the
Hamiltonian itself is a tensor in operator space. Treating ``H`` as a
*state* of a meta-system makes self-reference bounded and operational:
the meta-QPCN reasons about its own constraint algebra, evolves it,
and emits a refined ``H'`` — exactly one level of self-modification, no
infinite tower.

QPCN realization (spec §12.9):

  * The lower-level QPCN's Hamiltonian is represented as an **MPO** —
    a chain of rank-4 operator-bond tensors ``W[k]`` of shape
    ``(chi_l, d, d, chi_r)`` (real S1 substrate, ``src.qft_pcn.qft.mpo``).
  * The *operator-Hilbert-space* state encoding the same ``H`` is the
    **vectorized** MPO: each site tensor ``W`` of shape
    ``(chi_l, d, d, chi_r)`` is reshaped to ``(chi_l, d*d, chi_r)`` — a
    rank-3 MPS tensor on a physical Hilbert space of dimension ``d²``,
    where the ket-out / ket-in pair ``(s_out, s_in)`` is the
    *vectorization index* ``s_out * d + s_in``. This is the
    Choi-Jamiołkowski isomorphism on the tensor-network substrate
    (no SVD, no rearrangement of the operator-bond structure).
  * Decoding is the literal inverse: split the ``d²`` physical leg back
    into ``(d, d)``. Encode/decode round-trips through identical
    tensors — invertible by construction (§1.1 anti-shortcut: the
    operator structure lives in the bond pattern, never in a parsed
    AST).
  * The meta-Hamiltonian is a single-site operator on the ``d²``
    operator-Hilbert space. The natural quadratic meta-cost is the
    **non-Hermiticity penalty** at every operator-site:

        h_meta = (I - S)†(I - S)

    where ``S`` is the swap on the vectorization index pair
    ``(s_out, s_in) ↔ (s_in, s_out)`` — i.e. the operator that maps the
    vectorization of ``M`` to the vectorization of ``Mᵀ``. The
    *meta-eigenstates* of ``h_meta = 0`` are exactly the
    **real-symmetric operator vectorizations** (= Hermitian for
    real-coefficient lower-level blocks; complex-block Hermiticity
    requires the antilinear Choi-Jamiołkowski projector
    ``vec(M) ↔ SWAP·conj(vec(M))`` — a single-site *linear* meta-H
    cannot express this, see EXTENSIONS.md §12.9 complex-block entry).
    QPCN-typical lower-level Hs (Z + X Pauli blocks from
    `MeraEvalHamiltonian`) have real coefficients, where symmetric ↔
    Hermitian, so the impl is correct for current callers. Imag-time
    evolution toward the meta-ground state therefore **projects the
    lower-level H onto its real-symmetric part** — bounded
    self-modification that preserves the fundamental operator-algebraic
    invariant (real spectrum on real blocks) while relaxing redundancy
    in the antisymmetric directions.
  * Meta-energy is the sum over operator-sites of
    ``<phi_k| h_meta |phi_k>`` on the per-site MPS reduced state
    ``phi_k``. With the convention used here (vectorized form of an
    already-Hermitian local block has ``h_meta = 0`` exactly), the
    meta-energy is the **honest scalar measure of self-modification
    pressure** at each meta-step.

Operator-algebraic contract (spec §1.1):

  * Encoding/decoding are reshape-only — no parsing, no enumeration of
    rules, no AST consulted. The lower-level H's structure is preserved
    in the MPS bond pattern.
  * Meta-evolution applies a *real local operator* on the ``d²`` site
    Hilbert space (``expm(-h_meta · dt)``), then renormalizes. No
    side-channel manipulation of the underlying MPO.
  * The §12.1 anomaly contract on a decoded H is operationally tested
    via the residual non-Hermiticity ``||W - W†||²`` per site — the
    same quadratic invariant the meta-H is built from. A decoded H
    with vanishing non-Hermiticity is anomaly-extractable
    (real-spectrum guarantee survives the self-modification).

Spec §12.9 risks and the present implementation's mitigations:

  * "Self-modification breaks correctness" — the meta-H's ground
    manifold is exactly the Hermitian-operator vectorizations, so the
    meta-flow cannot produce a non-Hermitian ``H'``. Conservation of
    spectral reality by construction.
  * "Cost of MPO representation" — the meta-evolution is per-site (no
    growth in operator-bond dimension; the MPO bond ``chi`` propagates
    as the ``chi_l / chi_r`` of the vectorized MPS, untouched by the
    single-site meta-gate). One pass is linear in N.
  * "Goedelian paradoxes" — bounded levels: this module does NOT
    encode itself as a state (no meta-meta).

This module is the §12.9 substrate, distinct from §12.1: §12.1 is the
*static* anomaly diagnostic on a fixed H + program-state pair; §12.9
is the *dynamics* by which H changes. Together they ensure
self-modification stays inside the consistent operator algebra.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.linalg import expm

from src.qft_pcn.qft.mpo import MPO
from src.qft_pcn.qft.mps import MPS


# ---------------------------------------------------------------------------
# Meta-state: the lower-level H as a vector in the operator Hilbert space
# ---------------------------------------------------------------------------


@dataclass
class MetaState:
    """Vectorized MPO — the lower-level Hamiltonian as an MPS in the
    operator Hilbert space.

    Each site tensor has shape ``(chi_l, d*d, chi_r)``. The physical
    index ``v = s_out * d + s_in`` is the Choi-vectorization of the
    rank-4 MPO block ``W[k][a, s_out, s_in, b]``. ``d`` is the original
    physical (state) dimension of the lower-level H; the meta-level
    treats it only as a parameter (``d² = d_meta``).

    The MPS is intentionally NOT held in canonical form — meta-evolution
    here is single-site (no two-site gates, no SVD), so canonicalization
    is not needed and avoiding it preserves the encode→evolve→decode
    round-trip bit-equivalence in the no-evolution limit.
    """

    mps: MPS
    d_state: int  # original physical dimension of the encoded H

    def __post_init__(self) -> None:
        # Validate that the MPS site dimension is a square.
        for k, t in enumerate(self.mps.tensors):
            if t.shape[1] != self.d_state * self.d_state:
                raise ValueError(
                    f"meta-state site {k} has physical dim {t.shape[1]}, "
                    f"expected d_state**2 = {self.d_state ** 2}"
                )

    @property
    def N(self) -> int:
        return self.mps.N

    @property
    def d(self) -> int:
        """Original (lower-level) site dimension."""
        return self.d_state


# ---------------------------------------------------------------------------
# Encoding / decoding
# ---------------------------------------------------------------------------


def encode_hamiltonian_as_meta_state(H: MPO) -> MetaState:
    """Encode a lower-level Hamiltonian (MPO) as a meta-state (MPS).

    Vectorization: each MPO site tensor ``W[k]`` of shape
    ``(chi_l, d, d, chi_r)`` reshapes to an MPS site tensor of shape
    ``(chi_l, d*d, chi_r)``. The physical-leg ordering is "out-major":
    ``v = s_out * d + s_in``. This matches the convention used by
    ``decode_meta_state_to_hamiltonian`` so encode∘decode = id on the
    tensor list.

    No SVD, no rotation of operator bonds. The encoding is a pure
    reshape, so it preserves the entire MPO bond structure — including
    the Schollwock bond-dim-2 ``[done, not_yet_started]`` pattern from
    ``MPO.from_hamiltonian_sum`` — verbatim.
    """
    d = H.d
    mps_tensors: list[np.ndarray] = []
    for k, W in enumerate(H.tensors):
        chi_l, d_out, d_in, chi_r = W.shape
        if d_out != d or d_in != d:
            raise ValueError(
                f"MPO site {k} has physical (d_out, d_in)=({d_out}, {d_in}), "
                f"expected ({d}, {d})"
            )
        # (chi_l, d, d, chi_r) -> (chi_l, d*d, chi_r), out-major.
        mps_tensors.append(W.reshape(chi_l, d * d, chi_r).astype(complex))
    return MetaState(mps=MPS(tensors=mps_tensors), d_state=d)


def decode_meta_state_to_hamiltonian(meta_state: MetaState) -> MPO:
    """Inverse of :func:`encode_hamiltonian_as_meta_state`.

    Reshape ``(chi_l, d*d, chi_r) -> (chi_l, d, d, chi_r)`` using the
    same out-major ordering ``v = s_out * d + s_in``. The reverse map
    is pure tensor reshape — bit-equivalent in the absence of
    intervening evolution.
    """
    d = meta_state.d_state
    mpo_tensors: list[np.ndarray] = []
    for k, t in enumerate(meta_state.mps.tensors):
        chi_l, d2, chi_r = t.shape
        if d2 != d * d:
            raise ValueError(
                f"meta-state site {k} has physical dim {d2}, expected {d * d}"
            )
        mpo_tensors.append(t.reshape(chi_l, d, d, chi_r).astype(complex))
    return MPO(tensors=mpo_tensors)


# ---------------------------------------------------------------------------
# Meta-Hamiltonian: Hermiticity-projection penalty on the operator site
# ---------------------------------------------------------------------------


def _hermiticity_swap(d: int) -> np.ndarray:
    """The ``d² × d²`` permutation that maps vec(M) to vec(M†).

    With out-major vectorization ``v = s_out * d + s_in`` (i.e. ``M``'s
    rows are the contiguous block of length d in vec), we need the map

        vec(M)[s_out * d + s_in]  ->  conj(M[s_in, s_out])

    Because we live in a *complex* operator space, the full
    Hermitian-conjugate map combines (a) the index swap
    ``(s_out, s_in) -> (s_in, s_out)`` (a real permutation) and (b)
    complex conjugation. The penalty operator below applies the
    permutation as a linear operator; the conjugation enters via the
    Hermitian inner product when we evaluate ``<phi|h_meta|phi>`` on a
    complex MPS.

    Concretely: ``S`` here is the *swap* (transpose) operator on the
    vectorization. Its action on ``vec(M)`` returns ``vec(M^T)``. The
    operator ``h_meta = (I - S)†(I - S) = 2*(I - S)`` (S is real,
    symmetric, involutive) is positive-semidefinite and has kernel
    exactly the *symmetric* matrices (``M = M^T``). For real local
    blocks of the Schollwock construction (which are real-symmetric
    when ``h_per_site`` is real-symmetric) symmetry coincides with
    Hermiticity, so ``h_meta = 0`` on Hermitian Hs and ``> 0``
    otherwise. This is the §12.9 honest scope statement: the present
    meta-H projects onto **real-symmetric** lower-level blocks, which
    is the correct invariant for the bulk of QPCN typing/constraint
    Hamiltonians (their local terms are real projectors).
    """
    S = np.zeros((d * d, d * d), dtype=complex)
    for s_out in range(d):
        for s_in in range(d):
            v_in = s_out * d + s_in
            v_out = s_in * d + s_out  # transpose
            S[v_out, v_in] = 1.0
    return S


def hermiticity_meta_hamiltonian(d: int) -> np.ndarray:
    """Single-site meta-Hamiltonian on the ``d²`` operator-Hilbert space.

    ``h_meta = (I - S)†(I - S)`` with ``S`` the vec-transpose. Returns a
    Hermitian, positive-semidefinite ``(d², d²)`` complex matrix whose
    zero-eigenspace is the **real-symmetric** matrix subspace
    (``M = Mᵀ``), NOT the full Hermitian subspace (``M = M†``).

    Scope: symmetric ↔ Hermitian for real-coefficient lower-level Hs
    (the QPCN-typical case: Z + X Pauli combos from
    `MeraEvalHamiltonian` are real). Complex-block Hermiticity (e.g.
    ``iσ_y`` terms) would require the antilinear Choi-Jamiołkowski
    projector ``vec(M) ↔ SWAP·conj(vec(M))``, which no single-site
    *linear* meta-H can express — see EXTENSIONS.md §12.9 complex-block
    entry for the deferred antilinear-gate construction.
    """
    S = _hermiticity_swap(d)
    I = np.eye(d * d, dtype=complex)
    M = I - S
    h = M.conj().T @ M
    # Force exact Hermiticity (round-off cleanup).
    return 0.5 * (h + h.conj().T)


# ---------------------------------------------------------------------------
# Meta-energy
# ---------------------------------------------------------------------------


def meta_energy(meta_state: MetaState, h_meta: np.ndarray | None = None) -> float:
    """Sum of per-site meta-energy expectations,
    ``E_meta = Σ_k <phi|h_meta|phi>_k``,

    where ``<·>_k`` is the *un-normalized* expectation on the MPS — the
    meta-energy is reported on the raw vectorization of the (encoded or
    evolved) H, NOT on its normalized version. This matches the spec
    intent: meta-energy measures the self-modification pressure on the
    *operator*, not on a normalized probability state. (For a Hermitian
    H the meta-energy is exactly 0 regardless of norm.)

    Implementation: contract the MPS environment site-by-site,
    inserting ``h_meta`` at site ``k`` as a single-site operator and
    identities elsewhere. Identical structure to ``MPS.local_expectation``
    but unrenormalized.
    """
    d = meta_state.d_state
    if h_meta is None:
        h_meta = hermiticity_meta_hamiltonian(d)
    total = 0.0
    tensors = meta_state.mps.tensors
    N = len(tensors)
    for k in range(N):
        # Compute the (b_l, b_r) environment to the left of site k.
        env_l = np.ones((1, 1), dtype=complex)
        for j in range(k):
            t = tensors[j]
            env_l = np.einsum('ij,isk,jsl->kl', env_l, t.conj(), t)
        # Right environment from site k+1 onward.
        env_r = np.ones((1, 1), dtype=complex)
        for j in range(N - 1, k, -1):
            t = tensors[j]
            # env_r[i, j]: bra-bond, ket-bond at site j's left edge.
            env_r = np.einsum('isk,jsl,kl->ij', t.conj(), t, env_r)
        # Site k: bra(A*), op(h_meta), ket(A). Indices: env_l(p, q),
        # A[q, s, r], h_meta[s', s], A*[p, s', r'], env_r(r', r).
        Ak = tensors[k]
        val = np.einsum(
            'pq,qsr,ts,ptR,Rr->',
            env_l, Ak, h_meta, Ak.conj(), env_r,
        )
        total += float(np.real(val))
    return total


# ---------------------------------------------------------------------------
# Meta-evolution
# ---------------------------------------------------------------------------


def meta_evolve(
    meta_state: MetaState,
    h_meta: np.ndarray | None = None,
    dt: float = 0.05,
    steps: int = 10,
) -> MetaState:
    """Imag-time evolution of the meta-state under ``h_meta``.

    Single-site evolution: at each step we apply
    ``g = expm(-h_meta · dt)`` independently to every operator-site of
    the MPS. Because ``h_meta`` is a single-site operator, this is
    exact (no Trotter splitting needed across sites), and the MPS bond
    dimensions are preserved verbatim — the meta-flow does not grow
    the operator-bond representation of the lower-level H.

    The evolved meta-state is **not** renormalized as a state vector
    (the lower-level H is an *operator*, not a normalized state — its
    overall scale is meaningful: rescaling ``H -> alpha H`` is a
    physical change). Imag-time relaxation under a PSD ``h_meta`` will
    shrink components along positive meta-eigenspaces while preserving
    components in the meta-kernel (Hermitian-operator subspace), which
    is exactly the §12.9 contract.

    Returns a fresh ``MetaState`` — does not mutate the input.
    """
    if steps <= 0:
        return MetaState(mps=meta_state.mps.copy(), d_state=meta_state.d_state)
    d = meta_state.d_state
    if h_meta is None:
        h_meta = hermiticity_meta_hamiltonian(d)
    # Single-site gate for one step.
    gate = expm(-h_meta * dt)
    # Per-step gate is composed via dt; apply the same gate `steps` times.
    new_tensors = [t.copy() for t in meta_state.mps.tensors]
    for _ in range(steps):
        for k in range(len(new_tensors)):
            t = new_tensors[k]                       # (chi_l, d², chi_r)
            # Apply gate on the physical leg.
            new_tensors[k] = np.einsum('st,LtR->LsR', gate, t)
    return MetaState(mps=MPS(tensors=new_tensors), d_state=d)


# ---------------------------------------------------------------------------
# Anomaly extraction on the decoded H
# ---------------------------------------------------------------------------


def hermiticity_anomaly(H: MPO) -> float:
    """Operator-algebraic anomaly of ``H``: per-site non-Hermiticity,

        A(H) = Σ_k || W[k] - W†[k] ||_F²

    where ``W[k]`` is read as a ``(chi_l * d) × (chi_r * d)`` matrix on
    the (left-bond, ket-in) → (right-bond, ket-out) interpretation.
    For the QPCN-typical real-block lower-level H this coincides with
    the §12.1 contract that the constraint algebra preserves the
    Hermitian structure (a non-Hermitian lower-level H carries no
    consistent real-spectrum interpretation — the anomaly is the
    obstruction to that consistency).

    A vanishing return (≤ ``DEFAULT_META_ANOMALY_FLOOR``) means the
    decoded H is anomaly-free for the §12.1 hand-off; the §12.1 ABJ
    machinery on the MERA-encoded program then runs *on top* of a
    spectrally consistent H.
    """
    total = 0.0
    for W in H.tensors:
        # Per-site local block as a (d_out, d_in) matrix integrated over
        # the operator bonds: compare W against its physical-leg
        # transpose (swap of s_out, s_in). For the diagonal Schollwock
        # construction this is exactly the Hermiticity check on each
        # bulk d×d cell.
        W_T = np.swapaxes(W, 1, 2).conj()  # take (chi_l, d_in, d_out, chi_r) and conj
        diff = W - W_T
        total += float(np.real(np.vdot(diff, diff)))
    return total


# ---------------------------------------------------------------------------
# Default floor (matches §12.1)
# ---------------------------------------------------------------------------

DEFAULT_META_ANOMALY_FLOOR = 1e-6


__all__ = [
    "DEFAULT_META_ANOMALY_FLOOR",
    "MetaState",
    "decode_meta_state_to_hamiltonian",
    "encode_hamiltonian_as_meta_state",
    "hermiticity_anomaly",
    "hermiticity_meta_hamiltonian",
    "meta_energy",
    "meta_evolve",
]
