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


def _apply_factors_to_site(t: np.ndarray,
                           factors: tuple[np.ndarray, ...]) -> np.ndarray:
    """Apply per-species operators to a site tensor by species-by-species einsum.

    t: (chi_l, D_LOCAL, chi_r). For each species, move its axis to
    position 1, contract with op[s_out, s_in], move back. This never
    materializes the dense (D_LOCAL, D_LOCAL) tensor product.
    """
    chi_l, d, chi_r = t.shape
    if d != D_LOCAL:
        raise ValueError(f"expected physical dim {D_LOCAL}, got {d}")
    # Reshape physical axis into per-species axes.
    B = t.reshape(chi_l, *_SPECIES_DIMS, chi_r)
    # Axis layout: (chi_l=0, kind=1, type=2, bid=3, value=4, tobl=5, chi_r=6).
    for sp_idx, op in enumerate(factors):
        target_axis = sp_idx + 1
        # Move the target axis to position 1 so we can flatten the rest.
        B_moved = np.moveaxis(B, target_axis, 1)
        shape = B_moved.shape
        flat = B_moved.reshape(shape[0], shape[1], -1)
        # out[chi_l, s_out, rest] = sum_s_in op[s_out, s_in] * flat[chi_l, s_in, rest].
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
