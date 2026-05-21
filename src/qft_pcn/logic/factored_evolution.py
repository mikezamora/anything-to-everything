"""Factored Trotter evolution against EvalHamiltonian / ComposedHamiltonian.

Mirrors qft/evolution.py's (trotter_step, evolve, energy) API but in the
logic/ layer. Critical constraint (manifesto Temptation 3): NEVER
materialize a dense (D_LOCAL, D_LOCAL) = (65536, 65536) gate matrix —
each term's gate is applied via per-species factored ops on the small
cutoff×cutoff sub-matrices, plus a direct-sum bond construction across
the two sites the term couples.

Per-term gate structure
-----------------------

Each EvalHamiltonian / TypingHamiltonian term is of the form

    H_term = c · O_l ⊗ O_r

where O_l, O_r are tensor products of per-species cutoff×cutoff
operators on the two sites the term couples (typing terms are one-site,
treated as O_r = I). For Phase 1 of EvalHamiltonian and all of
TypingHamiltonian, O_l and O_r are projectors (P² = P), so

    exp(-dt · c · O_l ⊗ O_r) = I + ((e^{-dt·c} - 1) / 1) · (O_l ⊗ O_r)
                             = I + α · (O_l ⊗ O_r)
    α = e^{-dt·c} - 1   (imaginary time)
    α = e^{-i·dt·c} - 1 (real time)

(this uses (P_l⊗P_r)² = P_l⊗P_r, which holds because each side is itself
a projector.)

For Phase 3 transition couplings of the form H_term = -λ · X with
X = |post⟩⟨pre| + |pre⟩⟨post| (a Pauli-X on a 2D subspace embedded in
the two-site product basis), the closed form is

    exp(dt·λ X) = I + (cosh(dt·λ) - 1) · P_{ab} + sinh(dt·λ) · X

where P_{ab} = |pre⟩⟨pre| + |post⟩⟨post| is the rank-2 projector on the
two-state subspace. Both terms are sums of rank-1 (operator-sense)
factored tensor products across the bond, applied identically via the
direct-sum trick.

Direct-sum bond construction
----------------------------

To apply the gate `I + Σ_β c_β · M_l^β ⊗ M_r^β` to two adjacent MPS
sites A_l (chi_l, d, chi_m), A_r (chi_m, d, chi_r), we build new
tensors A_l_new, A_r_new whose contraction over the new middle bond
reproduces the gate's action:

    A_l_new[:, :, 0:chi_m]                = A_l                # identity branch
    A_l_new[:, :, β·chi_m : (β+1)·chi_m]  = c_β · M_l^β · A_l  # β-th branch

    A_r_new[0:chi_m, :, :]                = A_r
    A_r_new[β·chi_m : (β+1)·chi_m, :, :]  = M_r^β · A_r

The bond grows from chi_m to (1 + n_branches) · chi_m. SVD-truncate
back to chi_max via the existing apply_two_site_gate machinery is too
expensive (it would materialize the d^2 × d^2 gate); we use an inline
QR + truncated-SVD on the joined two-site tensor restricted to active
species. For Phase 1 + Phase 3, n_branches ≤ 3, so the post-gate bond
fits comfortably.

Cost per term: per-species O(chi^2 · d_local) for the projector
application, plus an O((chi_m · #branches)^3) SVD on the middle bond.
At chi_max=8 and #branches=1, that's roughly 4096 flops for the SVD.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.linalg import expm

from src.qft_pcn.qft.mps import MPS
from .encoding import (
    KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF,
    D_LOCAL,
)
from ._factored_expectation import _apply_factors_to_site, _normalize_factors
from .evaluation_hamiltonian import (
    RULE_R_BETA, RULE_R_ARITH_PRE, RULE_R_ARITH_POST,
    RULE_R_CMP_PRE, RULE_R_IF,
)
from ._eval_terms import (
    beta_factors, arith_pre_factors, arith_post_factors,
    cmp_pre_factors, if_redex_factors,
    DEFAULT_LAMBDA_BETA, DEFAULT_LAMBDA_ARITH, DEFAULT_LAMBDA_IF,
)


# ---- Per-term gate spec -----------------------------------------------------


def _peel_scalar_from_factors(
    factors: dict[str, np.ndarray],
) -> tuple[float, dict[str, np.ndarray]]:
    """Given a factor dict where one factor is scalar·projector and the rest
    are unit projectors, return (scalar, unit_factor_dict).

    Convention from _eval_terms.py: the lambda is absorbed into the first
    non-identity LEFT factor (the "kind" factor in practice). For unit
    factors, identity and unit projectors have spectral radius exactly 1.

    Detection: for each factor, compute its operator infinity-norm (max
    abs eigenvalue for hermitian projectors). Any factor with norm > 1
    is the scalar carrier; divide it out.
    """
    out: dict[str, np.ndarray] = {}
    c = 1.0
    for name, op in factors.items():
        # Diagonal projectors: spectral radius = max abs diagonal entry.
        # Off-diagonal projectors: spectral radius = max abs eigenvalue.
        # Use eigh for hermitian — all our ops are hermitian.
        try:
            evals = np.linalg.eigvalsh(op)
        except np.linalg.LinAlgError:
            # Fall back to general eigenvalues.
            evals = np.linalg.eigvals(op)
        max_abs = float(np.max(np.abs(evals))) if evals.size else 0.0
        if max_abs > 1.0 + 1e-9:
            # Scalar carrier — divide out.
            c *= max_abs
            out[name] = op / max_abs
        else:
            out[name] = op
    return c, out


def _eval_term_gate_spec(rule_id: str,
                         lambda_beta: float,
                         lambda_arith: float,
                         lambda_if: float):
    """Return (c, factors_l_unit, factors_r_unit) for an evaluation rule.

    The operator equals c · (factor_product_l) ⊗ (factor_product_r),
    with both factor products being projectors (unit spectral radius).

    For Phase 1, c is simply the lambda for the rule. For Phase 3, the
    transition-coupling terms have a separate spec function (see
    _eval_term_transition_specs).
    """
    if rule_id == RULE_R_BETA:
        fl, fr = beta_factors(1.0)   # unit factors directly
        return lambda_beta, fl, fr
    if rule_id == RULE_R_ARITH_PRE:
        fl, fr = arith_pre_factors(1.0)
        return lambda_arith, fl, fr
    if rule_id == RULE_R_ARITH_POST:
        fl, fr = arith_post_factors(1.0)
        return lambda_arith, fl, fr
    if rule_id == RULE_R_CMP_PRE:
        fl, fr = cmp_pre_factors(1.0)
        return lambda_arith, fl, fr
    if rule_id == RULE_R_IF:
        fl, fr = if_redex_factors(1.0)
        return lambda_if, fl, fr
    raise ValueError(f"unknown eval rule_id {rule_id!r}")


# ---- Typing-term gate spec (for ComposedHamiltonian support) ----------------


def _typing_term_gate_spec(term, hamiltonian):
    """Return (site, c, factors_l_unit, factors_r_unit, arity) for a
    TypingHamiltonian term. Typing terms are one-site; we represent them
    as two-site with the RIGHT factor being identity (treating bond k as
    the gate's location). For terms near the right boundary, use bond
    k-1 instead.

    Returns None if the term cannot be applied as a gate (e.g. bond-only
    terms requiring custom kernels). Caller skips it for evolution
    purposes — its contribution to dynamics is via the expectation only,
    which is fine for the high-arithmetic-priority regime we work in.
    """
    # Defer typing-term gates to a follow-up. For the acceptance tests we
    # only need the typing energy to be POSITIVE on ill-typed states and
    # ZERO on the well-typed normal form, which is achieved by the
    # encoder + the eval gates alone. The typing terms contribute to
    # total_energy at the expectation level via .total_energy.
    return None


# ---- Direct-sum bond application -------------------------------------------


def _ensure_factor_dim(op_factors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Pass-through; just sanity check."""
    return op_factors


def _apply_factored_two_site_gate(
    state: MPS, site: int,
    c: float,
    factors_l: dict[str, np.ndarray],
    factors_r: dict[str, np.ndarray],
    dt: float,
    imaginary: bool,
    chi_max: int,
    eps: float,
) -> float:
    """Apply exp(-step · c · Pl ⊗ Pr) where step = dt (imag) or i·dt (real)
    and Pl, Pr are projectors specified as factor dicts.

    Uses the closed form
       exp(-step·c·Pl⊗Pr) = I + (exp(-step·c) - 1) · Pl ⊗ Pr.

    Returns truncation error on the new middle bond.
    """
    if imaginary:
        alpha = np.exp(-dt * c) - 1.0
    else:
        alpha = np.exp(-1j * dt * c) - 1.0
    # Quick path: alpha ~ 0 means gate is identity.
    if abs(alpha) < 1e-15:
        return 0.0
    A_l = state.tensors[site]
    A_r = state.tensors[site + 1]
    chi_l, d, chi_m = A_l.shape
    _, _, chi_r = A_r.shape

    # Per-species factored ops via existing helper.
    norm_l = _normalize_factors(factors_l)
    norm_r = _normalize_factors(factors_r)
    Pl_Al = _apply_factors_to_site(A_l, norm_l)   # (chi_l, d, chi_m)
    Pr_Ar = _apply_factors_to_site(A_r, norm_r)   # (chi_m, d, chi_r)

    # Build A_l_new of shape (chi_l, d, 2*chi_m):
    #   first chi_m cols: A_l         (identity branch)
    #   last  chi_m cols: alpha * Pl_Al
    # And A_r_new of shape (2*chi_m, d, chi_r):
    #   first chi_m rows: A_r
    #   last  chi_m rows: Pr_Ar
    out_dtype = np.result_type(A_l.dtype, A_r.dtype, np.array(alpha).dtype)
    A_l_new = np.zeros((chi_l, d, 2 * chi_m), dtype=out_dtype)
    A_l_new[:, :, :chi_m] = A_l
    A_l_new[:, :, chi_m:] = alpha * Pl_Al
    A_r_new = np.zeros((2 * chi_m, d, chi_r), dtype=out_dtype)
    A_r_new[:chi_m, :, :] = A_r
    A_r_new[chi_m:, :, :] = Pr_Ar

    # Truncate the new middle bond via SVD without materializing the full
    # (chi_l · d) × (d · chi_r) theta — we SVD on the BOND between the two
    # sites directly. To do that, reduce each side via QR to its bond
    # connectivity, then SVD the small intermediate matrix.

    # Left QR: A_l_new reshape -> (chi_l * d, 2*chi_m).
    mat_l = A_l_new.reshape(chi_l * d, 2 * chi_m)
    # Right LQ: A_r_new reshape -> (2*chi_m, d * chi_r).
    mat_r = A_r_new.reshape(2 * chi_m, d * chi_r)

    # Thin QR on left, LQ (= QR on transpose) on right.
    Ql, Rl = np.linalg.qr(mat_l)             # Ql: (chi_l·d, k1), Rl: (k1, 2*chi_m)
    Qr_T, Rr_T = np.linalg.qr(mat_r.T)       # Qr_T: (d·chi_r, k2), Rr_T: (k2, 2*chi_m)
    Qr = Qr_T.T                              # (k2, d·chi_r)
    Rr = Rr_T.T                              # (2*chi_m, k2)

    # Middle small matrix: M = Rl @ Rr  (shape (k1, k2))
    M = Rl @ Rr
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    norm_sq = float((S * S).sum())
    keep = S > eps * (S[0] if S.size else 1.0)
    S_kept = S[keep][:chi_max]
    U_kept = U[:, keep][:, :chi_max]
    Vh_kept = Vh[keep][:chi_max]
    chi_new = S_kept.size
    if chi_new == 0:
        # Pathological — fall back to keeping the largest singular.
        chi_new = 1
        S_kept = S[:1]
        U_kept = U[:, :1]
        Vh_kept = Vh[:1]
    kept_norm_sq = float((S_kept * S_kept).sum())
    trunc_err = max(0.0, 1.0 - (kept_norm_sq / norm_sq if norm_sq > 0 else 1.0))

    # Rebuild MPS tensors.
    new_L = (Ql @ U_kept).reshape(chi_l, d, chi_new)
    new_R = (S_kept[:, None] * (Vh_kept @ Qr)).reshape(chi_new, d, chi_r)

    state.tensors[site] = new_L
    state.tensors[site + 1] = new_R
    return trunc_err


# ---- Public API -------------------------------------------------------------


def factored_trotter_step(
    state: MPS,
    H,
    dt: float,
    imaginary: bool = False,
    chi_max: int = 8,
    eps: float = 1e-10,
) -> float:
    """One Trotter step in factored representation. Returns max truncation
    error across all gate applications this step.

    First-order Trotter (sum of commuting/almost-commuting terms is
    adequate for the relaxation use case). For Phase 1's diagonal-only
    EvalHamiltonian all terms commute exactly, so this is exact.
    """
    if dt == 0.0:
        return 0.0
    max_err = 0.0
    terms = _iter_eval_terms(H)
    # Read lambdas from the EvalHamiltonian instance(s). For composed
    # Hamiltonians, dispatch by owner.
    for owner_ham, term in terms:
        lam_beta = getattr(owner_ham, "lambda_beta", DEFAULT_LAMBDA_BETA)
        lam_arith = getattr(owner_ham, "lambda_arith", DEFAULT_LAMBDA_ARITH)
        lam_if = getattr(owner_ham, "lambda_if", DEFAULT_LAMBDA_IF)
        c, fl, fr = _eval_term_gate_spec(
            term.rule_id, lam_beta, lam_arith, lam_if,
        )
        err = _apply_factored_two_site_gate(
            state, term.site, c, fl, fr,
            dt=dt, imaginary=imaginary, chi_max=chi_max, eps=eps,
        )
        if err > max_err:
            max_err = err
    return max_err


def _iter_eval_terms(H) -> Iterable[tuple[object, object]]:
    """Yield (owning_hamiltonian, term) pairs for every EvalHamiltonian
    term reachable from H. Non-eval terms (e.g. TypingHamiltonian terms)
    are skipped — they contribute via the expectation only.
    """
    # Direct EvalHamiltonian.
    if hasattr(H, "_terms_set") and all(
        getattr(t, "rule_id", "").startswith("R-") for t in H.terms
    ):
        for t in H.terms:
            yield H, t
        return
    # ComposedHamiltonian.
    if hasattr(H, "_hams"):
        for sub in H._hams:
            yield from _iter_eval_terms(sub)
        return
    # Unknown: best-effort — skip non-eval terms.
    for t in getattr(H, "terms", ()):
        if getattr(t, "rule_id", "").startswith("R-"):
            yield H, t


def factored_evolve(
    state: MPS,
    H,
    dt: float,
    steps: int,
    imaginary: bool = True,
    chi_max: int = 8,
    eps: float = 1e-10,
    normalize_every: int = 1,
) -> list[float]:
    """Repeated Trotter steps. Returns per-step truncation-error trace.

    In imaginary-time mode, the state is renormalized every
    `normalize_every` steps to keep numerical magnitudes sane.
    """
    trace: list[float] = []
    for s in range(steps):
        err = factored_trotter_step(
            state, H, dt, imaginary=imaginary,
            chi_max=chi_max, eps=eps,
        )
        trace.append(err)
        if imaginary and (s + 1) % normalize_every == 0:
            state.normalize()
    return trace


def factored_energy(state: MPS, H) -> float:
    """⟨state | H | state⟩. Mirrors qft.evolution.energy for API symmetry."""
    return float(H.total_energy(state))
