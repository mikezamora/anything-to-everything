"""Factored-expectation primitives for the typing Hamiltonian.

Local Hilbert space dim D_LOCAL = 65536. A dense (65536, 65536) operator
is ~64 GiB; materializing one per Hamiltonian term is forbidden by
manifesto Temptation 3. Typing-rule operators factor naturally as
tensor products O_kind ⊗ O_type ⊗ O_bid ⊗ O_value ⊗ O_tobl of per-
species small matrices (each cutoff × cutoff). This module contracts
them species-by-species against the MPS site tensor, never building
the dense product.

Cost per site: O(d_local · chi^2) — comparable to MPS.local_expectation
on a dense operator, but the per-term construction is O(species cost)
instead of O(d_local²).
"""

from __future__ import annotations

import numpy as np

from src.qft_pcn.qft.mps import MPS
from .encoding import (
    KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF,
    D_LOCAL,
)

_SPECIES_NAMES = ("kind", "type", "bid", "value", "tobl")
_SPECIES_DIMS = (KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF)


def _normalize_factors(op_factors: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """Return a 5-tuple of per-species matrices; identity for unspecified species."""
    out: list[np.ndarray] = []
    for name, d in zip(_SPECIES_NAMES, _SPECIES_DIMS):
        if name in op_factors:
            op = op_factors[name]
            if op.shape != (d, d):
                raise ValueError(
                    f"op_factors[{name!r}] has shape {op.shape}; expected ({d}, {d})"
                )
            out.append(np.asarray(op, dtype=complex))
        else:
            out.append(np.eye(d, dtype=complex))
    return tuple(out)


def _is_identity(op: np.ndarray) -> bool:
    """Cheap check: is op the identity?"""
    n = op.shape[0]
    if op.shape != (n, n):
        return False
    # Sentinel: diagonal == 1 and trace == n is necessary; check off-diag fast.
    if abs(op.trace() - n) > 1e-12:
        return False
    return np.allclose(op, np.eye(n, dtype=op.dtype), atol=1e-12)


def _apply_factors_to_site(t: np.ndarray,
                           factors: tuple[np.ndarray, ...]) -> np.ndarray:
    """Apply per-species operators to a site tensor.

    t: (chi_l, D_LOCAL, chi_r). Skips species whose operator is the
    identity — this is critical because typing-rule operators usually
    touch only 1-3 of the 5 species; sweeping the others would multiply
    by an identity and waste 4x the work.
    """
    chi_l, d, chi_r = t.shape
    if d != D_LOCAL:
        raise ValueError(f"expected physical dim {D_LOCAL}, got {d}")
    # Quick path: skip all identity species.
    active = [(sp_idx, op) for sp_idx, op in enumerate(factors)
              if not _is_identity(op)]
    if not active:
        return t
    B = t.reshape(chi_l, *_SPECIES_DIMS, chi_r)
    for sp_idx, op in active:
        target_axis = sp_idx + 1
        B_moved = np.moveaxis(B, target_axis, 1)
        shape = B_moved.shape
        flat = B_moved.reshape(shape[0], shape[1], -1)
        contracted = np.einsum('ji,bir->bjr', op, flat, optimize='greedy')
        B = np.moveaxis(contracted.reshape(shape), 1, target_axis)
    return B.reshape(chi_l, d, chi_r)


def factored_local_expectation(state: MPS, site: int,
                               op_factors: dict[str, np.ndarray]) -> float:
    """<state | O_site | state> where O = ⊗_species op_factors[species].

    Species not in op_factors are treated as identity. Returns a real
    number (Hamiltonian terms are Hermitian).
    """
    factors = _normalize_factors(op_factors)
    if not 0 <= site < state.N:
        raise ValueError(f"site {site} out of range [0, {state.N})")
    env = np.ones((1, 1), dtype=complex)
    for k in range(state.N):
        t = state.tensors[k]   # (chi_l, D_LOCAL, chi_r)
        if k == site:
            t_op = _apply_factors_to_site(t, factors)
            env = np.einsum('ij,isk,jsl->kl',
                            env, t_op, t.conj(), optimize='greedy')
        else:
            env = np.einsum('ij,isk,jsl->kl',
                            env, t, t.conj(), optimize='greedy')
    return float(np.real(env[0, 0]))


def factored_two_site_expectation(
    state: MPS, site: int,
    op_factors_left: dict[str, np.ndarray],
    op_factors_right: dict[str, np.ndarray],
) -> float:
    """<state | O_site ⊗ O_{site+1} | state> for a separable two-site op.

    For non-separable two-site operators, decompose into a sum of
    separable terms and sum the expectations. Returns a real number.
    """
    if not 0 <= site < state.N - 1:
        raise ValueError(f"site {site} invalid for two-site op (N={state.N})")
    factors_l = _normalize_factors(op_factors_left)
    factors_r = _normalize_factors(op_factors_right)
    env = np.ones((1, 1), dtype=complex)
    for k in range(state.N):
        t = state.tensors[k]
        if k == site:
            t_op = _apply_factors_to_site(t, factors_l)
            env = np.einsum('ij,isk,jsl->kl', env, t_op, t.conj(),
                            optimize='greedy')
        elif k == site + 1:
            t_op = _apply_factors_to_site(t, factors_r)
            env = np.einsum('ij,isk,jsl->kl', env, t_op, t.conj(),
                            optimize='greedy')
        else:
            env = np.einsum('ij,isk,jsl->kl', env, t, t.conj(),
                            optimize='greedy')
    return float(np.real(env[0, 0]))


def build_envs(state: MPS) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Precompute left and right identity-operator environments at every bond.

    Returns (lefts, rights) where:
      lefts[k]  is the (chi_l_at_k, chi_l_at_k) environment for sites [0..k)
      rights[k] is the (chi_r_at_k, chi_r_at_k) environment for sites (k..N-1]

    Each per-site contraction is the s-summed identity transfer:
      T[i, j, k, l] = sum_s t[i, s, k] · t.conj()[j, s, l],
    computed once and reused by repeated calls at the same site.
    """
    N = state.N
    # lefts[0] = (1, 1) trivial env.
    lefts: list[np.ndarray] = [np.ones((1, 1), dtype=complex)]
    env = lefts[0]
    for k in range(N - 1):
        t = state.tensors[k]
        env = np.einsum('ij,isk,jsl->kl',
                        env, t, t.conj(), optimize='greedy')
        lefts.append(env)
    rights: list[np.ndarray] = [None] * N  # type: ignore
    rights[N - 1] = np.ones((1, 1), dtype=complex)
    env = rights[N - 1]
    for k in range(N - 2, -1, -1):
        t = state.tensors[k + 1]
        env = np.einsum('isk,jsl,kl->ij',
                        t, t.conj(), env, optimize='greedy')
        rights[k] = env
    return lefts, rights


def factored_local_expectation_cached(state: MPS, site: int,
                                      op_factors: dict[str, np.ndarray],
                                      lefts: list[np.ndarray],
                                      rights: list[np.ndarray]) -> float:
    """factored_local_expectation reusing precomputed environments.

    Use this when evaluating MANY operators at the SAME site (e.g.,
    T-Obligation's sum over 7 candidate types) — amortizes the
    environment construction.
    """
    factors = _normalize_factors(op_factors)
    if not 0 <= site < state.N:
        raise ValueError(f"site {site} out of range [0, {state.N})")
    t = state.tensors[site]
    t_op = _apply_factors_to_site(t, factors)
    # <psi|...|psi> = trace(left_env · contract(t_op vs t.conj()) · right_env).
    # Build the on-site transfer: M[k, l] = sum_{i, j, s} left[i, j] · t_op[i, s, k] · t.conj()[j, s, l].
    M = np.einsum('ij,isk,jsl->kl',
                  lefts[site], t_op, t.conj(), optimize='greedy')
    val = np.einsum('kl,kl->', M, rights[site], optimize='greedy')
    return float(np.real(val))


def factored_left_bond_bid_expectation_cached(
    state: MPS, site: int,
    site_op_factors: dict[str, np.ndarray],
    bid_bond_projector: np.ndarray,
    lefts: list[np.ndarray],
    rights: list[np.ndarray],
) -> float:
    """Cached variant of factored_left_bond_bid_expectation."""
    if site <= 0:
        raise ValueError(f"site must be >= 1 for left-bond projector; got {site}")
    factors = _normalize_factors(site_op_factors)
    chi_l = state.tensors[site].shape[0]
    if bid_bond_projector.shape != (chi_l, chi_l):
        raise ValueError(
            f"bid_bond_projector shape {bid_bond_projector.shape} != ({chi_l}, {chi_l})")
    t = state.tensors[site]
    t_op = _apply_factors_to_site(t, factors)
    t_op = np.einsum('xy,ysk->xsk',
                     bid_bond_projector, t_op, optimize='greedy')
    M = np.einsum('ij,isk,jsl->kl',
                  lefts[site], t_op, t.conj(), optimize='greedy')
    val = np.einsum('kl,kl->', M, rights[site], optimize='greedy')
    return float(np.real(val))


def factored_right_bond_bid_expectation_cached(
    state: MPS, site: int,
    site_op_factors: dict[str, np.ndarray],
    bid_bond_projector: np.ndarray,
    lefts: list[np.ndarray],
    rights: list[np.ndarray],
) -> float:
    """Cached variant of factored_right_bond_bid_expectation."""
    if site >= state.N - 1:
        raise ValueError(
            f"site must be < N-1 for right-bond projector; got {site}")
    factors = _normalize_factors(site_op_factors)
    chi_r = state.tensors[site].shape[2]
    if bid_bond_projector.shape != (chi_r, chi_r):
        raise ValueError(
            f"bid_bond_projector shape {bid_bond_projector.shape} != ({chi_r}, {chi_r})")
    t = state.tensors[site]
    t_op = _apply_factors_to_site(t, factors)
    t_op = np.einsum('isk,xk->isx',
                     t_op, bid_bond_projector, optimize='greedy')
    M = np.einsum('ij,isk,jsl->kl',
                  lefts[site], t_op, t.conj(), optimize='greedy')
    val = np.einsum('kl,kl->', M, rights[site], optimize='greedy')
    return float(np.real(val))


def factored_two_site_expectation_cached(
    state: MPS, site: int,
    op_factors_left: dict[str, np.ndarray],
    op_factors_right: dict[str, np.ndarray],
    lefts: list[np.ndarray],
    rights: list[np.ndarray],
) -> float:
    """Cached variant of factored_two_site_expectation."""
    if not 0 <= site < state.N - 1:
        raise ValueError(f"site {site} invalid for two-site op (N={state.N})")
    factors_l = _normalize_factors(op_factors_left)
    factors_r = _normalize_factors(op_factors_right)
    t_l = state.tensors[site]
    t_r = state.tensors[site + 1]
    t_l_op = _apply_factors_to_site(t_l, factors_l)
    t_r_op = _apply_factors_to_site(t_r, factors_r)
    # M_l[i, j] -> bond: left @ t_l_op @ t_l.conj.
    mid = np.einsum('ij,isk,jsl->kl',
                    lefts[site], t_l_op, t_l.conj(), optimize='greedy')
    # Apply t_r at site+1.
    mid = np.einsum('kl,ksm,lsn->mn',
                    mid, t_r_op, t_r.conj(), optimize='greedy')
    val = np.einsum('mn,mn->', mid, rights[site + 1], optimize='greedy')
    return float(np.real(val))


def factored_left_bond_bid_expectation(
    state: MPS, site: int,
    site_op_factors: dict[str, np.ndarray],
    bid_bond_projector: np.ndarray,
) -> float:
    """<psi | P_bond_left(site) · O_site | psi> for a bond-side projector.

    The bond projector acts on the bond INDEX axis (size = state.tensors[
    site].shape[0]) between sites (site-1, site). This lets us encode
    rules like T-Var that read which channel/param_ty the bond carries.
    """
    if site <= 0:
        raise ValueError(f"site must be >= 1 for left-bond projector; got {site}")
    factors = _normalize_factors(site_op_factors)
    chi_l = state.tensors[site].shape[0]
    if bid_bond_projector.shape != (chi_l, chi_l):
        raise ValueError(
            f"bid_bond_projector shape {bid_bond_projector.shape} != "
            f"({chi_l}, {chi_l})")
    env = np.ones((1, 1), dtype=complex)
    for k in range(state.N):
        t = state.tensors[k]
        if k == site:
            t_op = _apply_factors_to_site(t, factors)
            # Project the LEFT bond axis (axis 0) on the ket side.
            t_op = np.einsum('xy,ysk->xsk',
                             bid_bond_projector, t_op, optimize='greedy')
            env = np.einsum('ij,isk,jsl->kl',
                            env, t_op, t.conj(), optimize='greedy')
        else:
            env = np.einsum('ij,isk,jsl->kl',
                            env, t, t.conj(), optimize='greedy')
    return float(np.real(env[0, 0]))


def factored_right_bond_bid_expectation(
    state: MPS, site: int,
    site_op_factors: dict[str, np.ndarray],
    bid_bond_projector: np.ndarray,
) -> float:
    """Dual of factored_left_bond_bid_expectation acting on the RIGHT bond.

    The bond projector has shape (chi_r, chi_r) where chi_r = state.tensors[
    site].shape[2].
    """
    if site >= state.N - 1:
        raise ValueError(
            f"site must be < N-1 for right-bond projector; got {site}")
    factors = _normalize_factors(site_op_factors)
    chi_r = state.tensors[site].shape[2]
    if bid_bond_projector.shape != (chi_r, chi_r):
        raise ValueError(
            f"bid_bond_projector shape {bid_bond_projector.shape} != "
            f"({chi_r}, {chi_r})")
    env = np.ones((1, 1), dtype=complex)
    for k in range(state.N):
        t = state.tensors[k]
        if k == site:
            t_op = _apply_factors_to_site(t, factors)
            # Project the RIGHT bond axis (axis 2) on the ket side.
            t_op = np.einsum('isk,xk->isx',
                             t_op, bid_bond_projector, optimize='greedy')
            env = np.einsum('ij,isk,jsl->kl',
                            env, t_op, t.conj(), optimize='greedy')
        else:
            env = np.einsum('ij,isk,jsl->kl',
                            env, t, t.conj(), optimize='greedy')
    return float(np.real(env[0, 0]))


def embed_factored_to_dense(op_factors: dict[str, np.ndarray]) -> np.ndarray:
    """Materialize the dense (D_LOCAL × D_LOCAL) operator from its factors.

    SLOW PATH — only for diagnostics/cross-checks. Allocates a
    (65536, 65536) complex matrix (~64 GiB). Do NOT call in production.
    """
    factors = _normalize_factors(op_factors)
    op = factors[0]
    for f in factors[1:]:
        op = np.kron(op, f)
    return op
