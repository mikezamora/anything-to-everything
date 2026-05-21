"""Factored per-redex term builders for the evaluation Hamiltonian
(sub-project C, Phase 1).

Per controller resolutions:
  - Translate the spec onto B's factored-expectation pattern.
  - Per-species cutoff x cutoff projectors only; no dense d_local^2 op
    is ever materialized.
  - No AST inspection here. The Hamiltonian is STRUCTURAL: depends only
    on the lattice. The static guard test in
    test_logic_evaluation_hamiltonian.py enforces this.

Phase 1 ships the **structural-collapse penalty terms** — the diagonal,
PSD projectors onto unreduced redex configurations. Future phases add
the off-diagonal transition couplings (controller resolution #3) that
DRIVE reductions via factored evolution.
"""

from __future__ import annotations

import numpy as np

from .encoding import (
    KIND_CUTOFF, VALUE_CUTOFF,
    KIND_PAD, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN,
    VALUE_PLUS, VALUE_MINUS, VALUE_TIMES, VALUE_LT, VALUE_EQ,
)

# Default coupling strengths (spec §3.1, §5.2). λ_arith ≥ λ_beta so
# arithmetic redices reduce first (call-by-value).
DEFAULT_LAMBDA_BETA = 1.0
DEFAULT_LAMBDA_ARITH = 1.0
DEFAULT_LAMBDA_IF = 1.0


def _project(d: int, idx: int) -> np.ndarray:
    p = np.zeros((d, d), dtype=complex)
    p[idx, idx] = 1.0
    return p


def _project_set(d: int, idxs) -> np.ndarray:
    p = np.zeros((d, d), dtype=complex)
    for i in idxs:
        p[i, i] = 1.0
    return p


# ---- Per-rule factored operators ----------------------------------------
#
# Each builder returns a (op_factors_left, op_factors_right) pair of
# dicts suitable for factored_two_site_expectation_cached, OR a single
# op_factors dict for one-site rules. Diagonal-only here — fast path.

def beta_factors(lam_scale: float = DEFAULT_LAMBDA_BETA):
    """H_beta(k) = λ · P_APP(k) ⊗ P_LAM(k+1)  (spec §3.2).

    Returns (left, right) per-species factor dicts; the scalar λ is
    absorbed into the LEFT kind factor so the returned op is λ·(P⊗P).
    """
    left = {"kind": lam_scale * _project(KIND_CUTOFF, KIND_APP)}
    right = {"kind": _project(KIND_CUTOFF, KIND_LAM)}
    return left, right


_ARITH_VALUE_OPS = (VALUE_PLUS, VALUE_MINUS, VALUE_TIMES)
_CMP_VALUE_OPS = (VALUE_LT, VALUE_EQ)


def arith_pre_factors(lam_scale: float = DEFAULT_LAMBDA_ARITH):
    """H_arith_pre(k) = λ · (P_BIN_arith @ k) ⊗ (P_INT @ k+1)  (spec §3.3).

    Bin must carry an arithmetic op (+, -, *); cmp ops handled separately.
    """
    left = {
        "kind":  lam_scale * _project(KIND_CUTOFF, KIND_BIN),
        "value": _project_set(VALUE_CUTOFF, _ARITH_VALUE_OPS),
    }
    right = {"kind": _project(KIND_CUTOFF, KIND_INT)}
    return left, right


def arith_post_factors(lam_scale: float = DEFAULT_LAMBDA_ARITH):
    """H_arith_post(k+1) = λ · (P_INT @ k+1) ⊗ (P_INT @ k+2)  (spec §3.3).

    Alone this would penalize ANY pair of adjacent IntLits, but the
    pair (pre, post) on overlapping bonds gives total penalty +2λ
    precisely on the unreduced (BIN_arith, INT, INT) tower. After
    reduction the BIN site becomes IntLit and adjacent IntLits are
    PAD, so this term vanishes.
    """
    left = {"kind": lam_scale * _project(KIND_CUTOFF, KIND_INT)}
    right = {"kind": _project(KIND_CUTOFF, KIND_INT)}
    return left, right


def cmp_pre_factors(lam_scale: float = DEFAULT_LAMBDA_ARITH):
    """H_cmp_pre(k) = λ · (P_BIN_cmp @ k) ⊗ (P_INT @ k+1)  (spec §3.4)."""
    left = {
        "kind":  lam_scale * _project(KIND_CUTOFF, KIND_BIN),
        "value": _project_set(VALUE_CUTOFF, _CMP_VALUE_OPS),
    }
    right = {"kind": _project(KIND_CUTOFF, KIND_INT)}
    return left, right


# cmp_post is identical to arith_post — same (INT, INT) two-site projector
# fires under cmp's (BIN_cmp, INT, INT) too. We rely on the same post
# term doing double duty (additive penalty when BOTH arith and cmp
# pre-terms fire on the same bond is impossible because a BIN can't
# carry both an arith and cmp op simultaneously).


def if_redex_factors(lam_scale: float = DEFAULT_LAMBDA_IF):
    """H_if(k) = λ · (P_IF @ k) ⊗ (P_BOOL @ k+1)  (spec §3.5)."""
    left = {"kind": lam_scale * _project(KIND_CUTOFF, KIND_IF)}
    right = {"kind": _project(KIND_CUTOFF, KIND_BOOL)}
    return left, right
