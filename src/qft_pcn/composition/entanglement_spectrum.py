"""§12.11 Modular Hamiltonian and entanglement-spectrum classifier.

The entanglement spectrum at a bond is the set of squared Schmidt
coefficients ``{p_i = lambda_i^2}`` (equivalently, eigenvalues of the
reduced density matrix ``rho_A``). The modular Hamiltonian is
``K = -log rho_A``; its spectrum is ``{K_i = -log p_i}``.

Per the spec, the entanglement spectrum is a *topological* fingerprint of
the proof state, strictly finer than the energy expectation value. Two
proofs with the same ``<H>`` can have qualitatively different spectra,
indicating qualitatively different proof structures.

The classifier is entanglement-based, never AST-based — this is the
§1.1 architecture-soul constraint: variable binding is bond entanglement,
not classical lookup, so equivalence is read off the bond spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union

import numpy as np

from src.qft_pcn.qft.mera import MERA
from src.qft_pcn.qft.mps import MPS

State = Union[MERA, MPS]

# Schmidt coefficients below this threshold are treated as numerical noise
# and dropped. Matches the entropy-computation convention in mps.py /
# mera.py (1e-15 there); we use a slightly looser cutoff for the spectrum
# itself because chi can be larger and the long tail is what we classify.
_SPEC_TOL = 1e-12

# Hermiticity tolerance for the modular Hamiltonian.
_HERMITICITY_TOL = 1e-9


# ---- core computations -----------------------------------------------------


def compute_entanglement_spectrum(state: State, cut: int) -> np.ndarray:
    """Squared Schmidt coefficients across the bond after leaf/site ``cut``.

    Returns a 1-D array of probabilities ``{p_i}`` with ``sum(p_i) == 1``
    (up to floating-point), sorted in descending order. The length equals
    the resolved bond rank; numerical-noise eigenvalues below ``_SPEC_TOL``
    are dropped (§12.11 risks: "truncate the spectrum at a clean cutoff").
    """
    p = _raw_spectrum(state, cut)
    p = p[p > _SPEC_TOL]
    if p.size == 0:
        return np.array([1.0])
    p = p / p.sum()
    # Descending order: lambda_1^2 >= lambda_2^2 >= ...
    p = np.sort(p)[::-1]
    return p


def compute_modular_hamiltonian(state: State, cut: int) -> dict:
    """The modular Hamiltonian ``K = -log rho_A`` across ``cut``.

    ``rho_A`` is diagonal in the Schmidt basis with eigenvalues ``p_i``.
    Therefore ``K`` is diagonal in the same basis with eigenvalues
    ``-log p_i``. We return the spectrum and the (diagonal) operator
    matrix; ``K`` is self-adjoint by construction because ``rho_A`` is
    positive Hermitian.

    Returns
    -------
    dict
        - ``spectrum`` (np.ndarray): ``{p_i}``, descending.
        - ``K_eigenvalues`` (np.ndarray): ``{-log p_i}``, ascending.
        - ``K_matrix`` (np.ndarray): diagonal matrix ``diag(K_eigenvalues)``.
        - ``rho_A`` (np.ndarray): diagonal ``diag(p_i)``.
    """
    p = compute_entanglement_spectrum(state, cut)
    # K_i = -log p_i. Since p_i is descending, K_i is ascending and the
    # smallest K_i corresponds to the dominant Schmidt mode (the "ground
    # state" of the modular Hamiltonian — Li-Haldane convention).
    K = -np.log(p)
    K_matrix = np.diag(K).astype(complex)
    rho_A = np.diag(p).astype(complex)
    return {
        "spectrum": p,
        "K_eigenvalues": K,
        "K_matrix": K_matrix,
        "rho_A": rho_A,
    }


# ---- classification --------------------------------------------------------


class ProofTopology(Enum):
    """Topological class of a proof state, read off its entanglement spectrum.

    Per §12.11:
      - ``TRIVIAL``       — nearly degenerate spectrum, low entropy, sharp gap.
      - ``INDUCTIVE``     — power-law spectrum (scale-invariant structure).
      - ``CRITICAL``      — CFT-like tower (broad spectrum, slow decay, high S).
      - ``TOPOLOGICAL``   — degenerate spectrum protected by symmetry.
    """
    TRIVIAL = "trivial"
    INDUCTIVE = "inductive"
    CRITICAL = "critical"
    TOPOLOGICAL = "topological"


@dataclass(frozen=True)
class ProofClassification:
    """An entanglement-equivalence class for a proof state (§12.11).

    Two proofs are in the same class iff their spectra produce the same
    ``ProofTopology`` *and* their entropy + power-law exponent agree to
    within tolerance. This is strictly an entanglement-based equivalence,
    never AST-based.
    """
    topology: ProofTopology
    entropy: float                  # von Neumann S = -sum p_i log p_i
    spectrum: tuple[float, ...]     # full p_i tuple (descending)
    gap: float                      # p_1 - p_2 (or p_1 if rank-1)
    power_law_exponent: float       # fit of log p_i vs log i; nan if rank<3
    degeneracy_top: int             # multiplicity of the largest eigenvalue

    def signature(self, decimals: int = 3) -> tuple:
        """A coarse, comparable signature for clustering proofs.

        We round the spectrum and exponent to a small number of decimals
        so floating-point noise does not split entanglement-equivalent
        proofs into different clusters.
        """
        return (
            self.topology.value,
            round(self.entropy, decimals),
            round(self.power_law_exponent, decimals)
                if np.isfinite(self.power_law_exponent) else None,
            self.degeneracy_top,
            tuple(round(p, decimals) for p in self.spectrum),
        )


def classify_proof_by_spectrum(
    state: State,
    cut: int | None = None,
) -> ProofClassification:
    """Classify a proof state by its entanglement spectrum (§12.11).

    Parameters
    ----------
    state
        An MPS or MERA proof state.
    cut
        Bond/leaf index for the bipartition. If ``None``, use the central
        cut (most informative for typical proof states).
    """
    if cut is None:
        cut = _default_cut(state)
    p = compute_entanglement_spectrum(state, cut)
    entropy = float(-(p * np.log(p)).sum()) if p.size > 1 else 0.0
    gap = float(p[0] - p[1]) if p.size >= 2 else float(p[0])

    # Top-eigenvalue degeneracy: how many p_i are within tol of p_1?
    deg_tol = max(1e-6, 1e-3 * float(p[0]))
    degeneracy_top = int(np.sum(np.abs(p - p[0]) <= deg_tol))

    # Power-law fit log p_i = -alpha log i + c, i >= 1. Only meaningful
    # when the spectrum has at least three resolved values.
    if p.size >= 3:
        idx = np.arange(1, p.size + 1, dtype=float)
        slope, _ = np.polyfit(np.log(idx), np.log(p), 1)
        power_law_exponent = float(-slope)
    else:
        power_law_exponent = float("nan")

    topology = _topology_from_spectrum(
        p=p, entropy=entropy, gap=gap,
        degeneracy_top=degeneracy_top,
        power_law_exponent=power_law_exponent,
    )

    return ProofClassification(
        topology=topology,
        entropy=entropy,
        spectrum=tuple(float(x) for x in p),
        gap=gap,
        power_law_exponent=power_law_exponent,
        degeneracy_top=degeneracy_top,
    )


# ---- internal helpers ------------------------------------------------------


def _topology_from_spectrum(
    p: np.ndarray,
    entropy: float,
    gap: float,
    degeneracy_top: int,
    power_law_exponent: float,
) -> ProofTopology:
    """Map a numerical spectrum to a ``ProofTopology`` class.

    Heuristics derived directly from §12.11:
      - Rank 1 or very small entropy + large gap        -> TRIVIAL
      - Top eigenvalue is degenerate (mult >= 2)        -> TOPOLOGICAL
      - Clean power-law decay (exponent in [0.3, 3.0])  -> INDUCTIVE
      - Broad spectrum with high entropy, no power-law  -> CRITICAL
    The thresholds are intentionally conservative; refinement is a
    wake-sleep abstraction-discovery target (§12.11 risks).
    """
    if p.size == 1 or (entropy < 1e-6 and gap > 0.9):
        return ProofTopology.TRIVIAL
    if degeneracy_top >= 2:
        return ProofTopology.TOPOLOGICAL
    if (np.isfinite(power_law_exponent)
            and 0.3 <= power_law_exponent <= 3.0
            and entropy < np.log(p.size)):
        return ProofTopology.INDUCTIVE
    return ProofTopology.CRITICAL


def _default_cut(state: State) -> int:
    """Centre cut for a bipartite split."""
    if isinstance(state, MERA):
        return (state.N // 2) - 1
    if isinstance(state, MPS):
        return (state.N // 2) - 1
    raise TypeError(f"unsupported state type: {type(state).__name__}")


def _raw_spectrum(state: State, cut: int) -> np.ndarray:
    """Raw (unnormalized, unsorted, unfiltered) eigenvalues of ``rho_A``.

    Routes to the appropriate substrate-specific computation. The MERA
    paths mirror :meth:`MERA.entanglement_entropy` but stop one step
    short: instead of contracting the eigenvalues into ``-sum p log p``
    we return them.
    """
    if isinstance(state, MERA):
        return _mera_spectrum(state, cut)
    if isinstance(state, MPS):
        return _mps_spectrum(state, cut)
    raise TypeError(f"unsupported state type: {type(state).__name__}")


def _mera_spectrum(state: MERA, cut: int) -> np.ndarray:
    """Reduced-density spectrum across ``cut`` for a MERA state."""
    if not 0 <= cut < state.N - 1:
        raise ValueError(f"cut {cut} out of range [0, {state.N - 1})")

    # Fast path: state built from an explicit branch superposition. Mirror
    # the eigenvalue extraction from MERA._entropy_from_terms (the only
    # path that handles structurally entangled MERAs in the small-N
    # composition regime).
    if state._superposition_terms is not None:
        return _mera_spectrum_from_terms(state, cut)

    # Fast path: product MERA → reduced density is rank-1.
    if state._is_product():
        return np.array([1.0])

    # General path: materialize, SVD, square. Same cost guard as
    # MERA.entanglement_entropy — only the small acceptance-test sizes.
    psi = state._materialize()
    N = state.N
    d = state.d_local
    left_size = cut + 1
    right_size = N - left_size
    psi_mat = psi.reshape(d ** left_size, d ** right_size)
    sv = np.linalg.svd(psi_mat, compute_uv=False)
    return sv * sv


def _mera_spectrum_from_terms(state: MERA, cut: int) -> np.ndarray:
    """Spectrum extraction for a superposition-MERA. Mirrors
    :meth:`MERA._entropy_from_terms` up to the eigenvalue step.
    """
    terms = state._superposition_terms
    k = len(terms)
    coeffs = np.array([c for c, _ in terms], dtype=complex)
    left_idx = list(range(cut + 1))
    right_idx = list(range(cut + 1, state.N))

    def gram(idx_set: list[int]) -> np.ndarray:
        G = np.ones((k, k), dtype=complex)
        for kk in idx_set:
            col = np.array([
                terms[t][1][kk].astype(complex) for t in range(k)])
            ov = col.conj() @ col.T
            G = G * ov
        return G

    GL = gram(left_idx)
    GR = gram(right_idx)
    cc = np.outer(coeffs.conj(), coeffs)
    norm_sq = float(np.real(np.sum(cc * GL * GR)))
    if norm_sq < 1e-30:
        return np.array([1.0])
    B = np.outer(coeffs, coeffs.conj()) * GR.T
    K = (GL @ B) / norm_sq
    ev = np.linalg.eigvals(K)
    return np.real(ev)


def _mps_spectrum(state: MPS, cut: int) -> np.ndarray:
    """Reduced-density spectrum across ``cut`` for an MPS.

    Mirrors :meth:`MPS.entanglement_entropy`: mixed-canonicalize by
    sweeping QR from both ends, then diagonalize the small bond density
    matrix. No full statevector is materialized.
    """
    if not 0 <= cut < state.N - 1:
        raise ValueError(f"cut {cut} out of range [0, {state.N - 1})")
    ts = [t.copy() for t in state.tensors]
    for k in range(cut + 1):
        chi_l, d, chi_r = ts[k].shape
        mat = ts[k].reshape(chi_l * d, chi_r)
        Q, R = np.linalg.qr(mat)
        ts[k] = Q.reshape(chi_l, d, Q.shape[1])
        if k + 1 < state.N:
            ts[k + 1] = np.einsum('rs,sdt->rdt', R, ts[k + 1])
    for k in range(state.N - 1, cut, -1):
        chi_l, d, chi_r = ts[k].shape
        mat = ts[k].reshape(chi_l, d * chi_r)
        Q, R = np.linalg.qr(mat.conj().T)
        Q = Q.conj().T
        R = R.conj().T
        ts[k] = Q.reshape(Q.shape[0], d, chi_r)
        if k - 1 >= 0:
            ts[k - 1] = np.einsum('rds,st->rdt', ts[k - 1], R)
    R_t = ts[cut + 1]
    rho_bond = np.einsum('msj,nsj->mn', R_t, R_t.conj())
    eigs = np.linalg.eigvalsh(rho_bond)
    return np.real(eigs)
