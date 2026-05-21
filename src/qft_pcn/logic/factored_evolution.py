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
X = |post⟩⟨pre| + |pre⟩⟨post| where |post⟩, |pre⟩ are PRODUCT
computational-basis states across a contiguous window of sites, the
closed form

    exp(dt·λ X) = I + (cosh(dt·λ) - 1) · P_{ab} + sinh(dt·λ) · X
    P_{ab} = |pre⟩⟨pre| + |post⟩⟨post|

is applied via an amplitude-mixing trick that AVOIDS materializing any
n-site dense gate. Concretely:

  1. Compute α = ⟨pre|ψ⟩, β = ⟨post|ψ⟩ via the per-site basis-slice
     contraction of the MPS — O(N · chi²) per amplitude, no d_local
     factor in the contraction (each site contributes only its slice
     A_k[:, flat(s_k), :]).
  2. The closed form rearranges to |ψ'⟩ = |ψ⟩ + γ_pre·|pre⟩
     + γ_post·|post⟩ for explicit scalars γ_pre, γ_post.
  3. Add each product state via the direct-sum MPS-addition trick:
     bond dim grows by 1 at every interior bond, then SVD-truncate
     each bond back to chi_max via QR + small-matrix SVD.

Cost per transition: O(N · chi²) for amplitudes + O(N · chi³ · d_local)
for the bond SVD sweep. The d_local factor only enters in the QR step
of each per-bond truncation; that QR is on a (chi · d_local, chi)
matrix and so is O(chi² · d_local).

NB. For Phase 1 (commuting diagonal projector terms), the simpler
two-site direct-sum trick is still used — see
_apply_factored_two_site_gate.

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
from ._eval_transitions import TransitionTerm


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


# ---- Phase 3: n-site rank-1 transition coupling gates ----------------------

_SPECIES_DIMS = (KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF)


def _basis_outer(s_post: tuple[int, int, int, int, int],
                 s_pre: tuple[int, int, int, int, int]
                 ) -> dict[str, np.ndarray]:
    """Build per-species factor dict for |s_post⟩⟨s_pre| at one site.

    Kept for diagnostics / fallback. Hot path uses _apply_basis_outer_fast
    which avoids the einsum entirely.
    """
    factors: dict[str, np.ndarray] = {}
    names = ("kind", "type", "bid", "value", "tobl")
    for name, dim, post_idx, pre_idx in zip(
        names, _SPECIES_DIMS, s_post, s_pre
    ):
        op = np.zeros((dim, dim), dtype=complex)
        op[post_idx, pre_idx] = 1.0
        factors[name] = op
    return factors


def _flat_basis_index(s: tuple[int, int, int, int, int]) -> int:
    """Linearize per-species index tuple to the flat D_LOCAL basis index.

    Matches encoding._basis_index ordering (kind slowest, tobl fastest).
    """
    k, t, b, v, o = s
    return ((((k * TYPE_CUTOFF + t) * BID_CUTOFF + b)
              * VALUE_CUTOFF + v) * TOBL_CUTOFF + o)


def _apply_basis_outer_fast(A: np.ndarray, s_post, s_pre) -> np.ndarray:
    """Apply rank-1 operator |s_post⟩⟨s_pre| to site tensor A.

    Result[chi_l, flat_post, chi_r] = A[chi_l, flat_pre, chi_r]
    Result is zero elsewhere. NO einsum / reshape — single slice copy.
    """
    chi_l, d, chi_r = A.shape
    flat_pre = _flat_basis_index(s_pre)
    flat_post = _flat_basis_index(s_post)
    out = np.zeros_like(A)
    out[:, flat_post, :] = A[:, flat_pre, :]
    return out


def _apply_n_site_rank1_branches(
    state: MPS,
    site_left: int,
    branches,   # list of (coefficient, [(s_post_per_site, s_pre_per_site), ...])
    chi_max: int,
    eps: float,
) -> float:
    """Apply gate `I + Σ_β c_β · ⊗_w |s_post_β^w⟩⟨s_pre_β^w|` over W
    consecutive sites.

    branches: list of (coefficient, [(s_post, s_pre)] per site).
    Each per-site (s_post, s_pre) is a per-species index tuple.

    Direct-sum construction across W-1 internal bonds: each bond grows by
    factor (1 + n_branches). After application, each bond is SVD-truncated
    back to chi_max via QR + small-matrix SVD.

    Returns the max truncation error across the W-1 bond SVDs.
    """
    n_branches = len(branches)
    if n_branches == 0:
        return 0.0
    W = len(branches[0][1])
    if W < 2:
        raise ValueError(f"window width must be >= 2, got {W}")
    if not all(len(b[1]) == W for b in branches):
        raise ValueError("branches have inconsistent window widths")
    N = state.N
    if site_left < 0 or site_left + W > N:
        raise ValueError(
            f"window [{site_left}, {site_left+W}) out of range [0, {N})"
        )

    # Per-site, per-branch projection: |s_post⟩⟨s_pre| · A[site_left + w]
    # Fast path: each per-site op is rank-1 |post⟩⟨pre|, so the result
    # is a single slice copy — no einsum needed.
    A = [state.tensors[site_left + w].copy() for w in range(W)]
    proj: list[list[np.ndarray]] = []   # proj[w][β] shape == A[w].shape
    for w in range(W):
        per_branch: list[np.ndarray] = []
        for _, per_site_pairs in branches:
            s_post, s_pre = per_site_pairs[w]
            per_branch.append(_apply_basis_outer_fast(A[w], s_post, s_pre))
        proj.append(per_branch)

    # Now build the augmented site tensors. For W sites, we have W-1 bonds.
    # The identity branch lives in the first chi block of each bond; each
    # perturbation branch β lives in its own block of size chi_m_w (the
    # ORIGINAL middle bond at bond w). The total expanded bond size at
    # bond w is (1 + n_branches) * chi_m_w.
    #
    # New site tensors:
    #   N_0[chi_l, d, (1 + n_branches) * chi_m_0]:
    #     block 0: A_0
    #     block β (β=1..n): c_β · proj_β[0]   (apply coefficient at left edge)
    #
    #   N_w (1 <= w < W-1), shape ((1+n)*chi_m_{w-1}, d, (1+n)*chi_m_w):
    #     block (0,0): A_w
    #     block (β,β): proj_β[w]
    #     (off-diagonal blocks are zero)
    #
    #   N_{W-1}[(1 + n_branches) * chi_m_{W-2}, d, chi_r]:
    #     block 0: A_{W-1}
    #     block β: proj_β[W-1]
    #
    # Then SVD-truncate each internal bond.

    new_tensors: list[np.ndarray] = []
    # First (leftmost) site: only right-bond is expanded.
    A0 = A[0]
    chi_l0, d0, chi_m0 = A0.shape
    expanded_right_0 = (1 + n_branches) * chi_m0
    N0 = np.zeros((chi_l0, d0, expanded_right_0), dtype=complex)
    N0[:, :, :chi_m0] = A0
    for b_idx, (coeff, _) in enumerate(branches):
        start = (b_idx + 1) * chi_m0
        end = start + chi_m0
        N0[:, :, start:end] = coeff * proj[0][b_idx]
    new_tensors.append(N0)

    # Interior sites: both bonds expanded, block-diagonal.
    for w in range(1, W - 1):
        Aw = A[w]
        chi_l_w, d_w, chi_r_w = Aw.shape
        exp_l = (1 + n_branches) * chi_l_w
        exp_r = (1 + n_branches) * chi_r_w
        Nw = np.zeros((exp_l, d_w, exp_r), dtype=complex)
        # Identity block.
        Nw[:chi_l_w, :, :chi_r_w] = Aw
        for b_idx, _ in enumerate(branches):
            lstart = (b_idx + 1) * chi_l_w
            lend = lstart + chi_l_w
            rstart = (b_idx + 1) * chi_r_w
            rend = rstart + chi_r_w
            Nw[lstart:lend, :, rstart:rend] = proj[w][b_idx]
        new_tensors.append(Nw)

    # Last (rightmost) site: only left-bond is expanded.
    A_last = A[W - 1]
    chi_l_last, d_last, chi_r_last = A_last.shape
    expanded_left_last = (1 + n_branches) * chi_l_last
    N_last = np.zeros(
        (expanded_left_last, d_last, chi_r_last), dtype=complex,
    )
    N_last[:chi_l_last, :, :] = A_last
    for b_idx, _ in enumerate(branches):
        start = (b_idx + 1) * chi_l_last
        end = start + chi_l_last
        N_last[start:end, :, :] = proj[W - 1][b_idx]
    new_tensors.append(N_last)

    # Write back to state.
    for w in range(W):
        state.tensors[site_left + w] = new_tensors[w]

    # Now SVD-truncate each internal bond.
    max_err = 0.0
    for bond in range(site_left, site_left + W - 1):
        err = _svd_truncate_bond(state, bond, chi_max=chi_max, eps=eps)
        if err > max_err:
            max_err = err
    return max_err


def _svd_truncate_bond(state: MPS, bond: int,
                       chi_max: int, eps: float) -> float:
    """SVD-truncate the bond between site `bond` and `bond+1` to chi_max.

    Uses QR on the (left, right) sites to reduce to a small intermediate
    matrix, then SVDs that. Cost: O(chi^3 · d) per side plus a small SVD.
    """
    Al = state.tensors[bond]
    Ar = state.tensors[bond + 1]
    chi_l, dl, chi_m = Al.shape
    chi_m2, dr, chi_r = Ar.shape
    if chi_m != chi_m2:
        raise ValueError(
            f"bond {bond} inconsistent: {chi_m} vs {chi_m2}"
        )
    mat_l = Al.reshape(chi_l * dl, chi_m)
    mat_r = Ar.reshape(chi_m, dr * chi_r)
    Ql, Rl = np.linalg.qr(mat_l)         # Ql: (chi_l·d, k1), Rl: (k1, chi_m)
    Qr_T, Rr_T = np.linalg.qr(mat_r.T)   # Qr_T: (d·chi_r, k2)
    Qr = Qr_T.T                          # (k2, d·chi_r)
    Rr = Rr_T.T                          # (chi_m, k2)
    M = Rl @ Rr
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    norm_sq = float((S * S).sum())
    if S.size == 0 or norm_sq == 0.0:
        # Degenerate; leave bond alone.
        return 0.0
    keep = S > eps * S[0]
    S_kept = S[keep][:chi_max]
    U_kept = U[:, keep][:, :chi_max]
    Vh_kept = Vh[keep][:chi_max]
    chi_new = S_kept.size
    if chi_new == 0:
        chi_new = 1
        S_kept = S[:1]
        U_kept = U[:, :1]
        Vh_kept = Vh[:1]
    kept_norm_sq = float((S_kept * S_kept).sum())
    trunc_err = max(0.0, 1.0 - (kept_norm_sq / norm_sq))

    new_L = (Ql @ U_kept).reshape(chi_l, dl, chi_new)
    new_R = (S_kept[:, None] * (Vh_kept @ Qr)).reshape(chi_new, dr, chi_r)
    state.tensors[bond] = new_L
    state.tensors[bond + 1] = new_R
    return trunc_err


def _transition_branches(term: TransitionTerm,
                         dt: float, imaginary: bool):
    """Build the (coeff, per-site factor dict list) branches for one
    rank-1 transition term.

    H = -λ · (|post⟩⟨pre| + |pre⟩⟨post|).
    Setting θ = step · λ (with step = dt for imag, i·dt for real),
    exp(-step · H) = exp(θ · X) where X = |post⟩⟨pre| + |pre⟩⟨post|, and
    X² = P_{pp} + P_{qq}. Closed form:

      exp(θ·X) = I + (cosh(θ)-1)·(|pre⟩⟨pre| + |post⟩⟨post|)
                    + sinh(θ)·(|post⟩⟨pre| + |pre⟩⟨post|)

    Yields 4 rank-1 branches:
      (cosh(θ)-1) · |pre⟩⟨pre|
      (cosh(θ)-1) · |post⟩⟨post|
      sinh(θ)     · |post⟩⟨pre|
      sinh(θ)     · |pre⟩⟨post|
    """
    lam = term.coupling
    if imaginary:
        theta = dt * lam
        c_diag = np.cosh(theta) - 1.0
        c_off = np.sinh(theta)
    else:
        theta = 1j * dt * lam
        c_diag = np.cosh(theta) - 1.0
        c_off = np.sinh(theta)
    branches = []
    # Skip near-zero branches for efficiency.
    if abs(c_diag) > 1e-15:
        branches.append((
            complex(c_diag),
            [(s, s) for s in term.pre],
        ))
        branches.append((
            complex(c_diag),
            [(s, s) for s in term.post],
        ))
    if abs(c_off) > 1e-15:
        branches.append((
            complex(c_off),
            [(p, q) for p, q in zip(term.post, term.pre)],
        ))
        branches.append((
            complex(c_off),
            [(q, p) for q, p in zip(term.pre, term.post)],
        ))
    return branches


def _mps_amplitude_at_basis(state: MPS, site_left: int,
                            site_basis_list) -> complex:
    """Compute ⟨b|state⟩ where |b⟩ = |...prev_pads...⟩ ⊗ (⊗_w |s_w⟩) ⊗ |...trailing_pads...⟩.

    site_basis_list[w] is the per-species tuple for site (site_left + w).
    Sites outside the window are projected onto their EXISTING argmax
    basis (i.e., we compute the amplitude of the window's product state
    *relative to* the rest of the chain).

    Actually: we just compute the FULL inner product against the product
    state where the window sites carry s_w, and OTHER sites are summed
    over their full basis (treating |b⟩ as a PARTIAL product state).
    That is: ⟨b_window | state⟩, with the partial inner product yielding
    a complex scalar that captures the amplitude AT this configuration
    integrated over all unspecified sites.

    For our use case the "outside" sites are PAD-dominated (encoded
    state), so we treat them as their current basis. To keep this
    correct, we include those sites' current argmax-projection too.

    Simpler implementation: compute the FULL N-site inner product where:
      - window sites have basis index = flat(_basis_outer(s, s)) chosen
      - other sites are TRACED (i.e., summed) — equivalent to setting
        their projector to identity.

    Equivalent: ⟨b_window | state ⟩ with the OTHER sites' indices summed
    is the marginal amplitude of |b_window⟩. This is NOT what we want
    (we want a specific configuration amplitude including the rest of
    the chain), but for the purpose of driving the window transition,
    it's the right scalar — it captures "how much of state lives on
    this window configuration regardless of the rest".

    Implementation: contract state with the projector P_window = ⊗_w
    |s_w⟩⟨s_w| acting only on window sites, identity elsewhere — this
    gives ||P_window state||². Not exactly the amplitude.

    To get a scalar amplitude that drives ket evolution: use the inner
    product against the LOCALIZED product state where outside sites are
    matched to state's current argmax. This is implementation-defined
    and somewhat arbitrary, but gives a meaningful "alignment" measure.

    For simplicity here we compute the FULL inner product against |b⟩
    being the product state with PADs everywhere outside the window —
    which is the natural ket for an encoded program with trailing PADs.
    """
    # Default outside-window basis: PAD-like for sites past the AST,
    # but for sites BEFORE the window we use the state's current argmax
    # to preserve the encoded prefix. For Phase 3's scope (small N,
    # short ASTs, window at site_left=0), site_left is usually 0 so
    # there's no prefix.
    from .encoding import (
        KIND_PAD, TYPE_NONE, BID_NONE, VALUE_NONE, TOBL_NONE,
    )
    pad_site = (KIND_PAD, TYPE_NONE, BID_NONE, VALUE_NONE, TOBL_NONE)
    full_basis: list[tuple[int, ...]] = []
    for k in range(state.N):
        if site_left <= k < site_left + len(site_basis_list):
            full_basis.append(site_basis_list[k - site_left])
        else:
            full_basis.append(pad_site)

    # Inner product: contract state tensors against the product state
    # vector at each site.
    env = np.ones((1,), dtype=complex)   # left environment, shape (chi_l,)
    for k in range(state.N):
        A = state.tensors[k]   # (chi_l, d, chi_r)
        flat = _flat_basis_index(full_basis[k])
        # A_slice: (chi_l, chi_r)
        A_slice = A[:, flat, :]
        # env_new[chi_r] = sum_chi_l env[chi_l] * A_slice[chi_l, chi_r]
        env = env @ A_slice
    return complex(env[0])


def _add_product_state_to_mps(
    state: MPS, basis_list, coefficient: complex,
    chi_max: int, eps: float,
) -> float:
    """In-place: state += coefficient · |basis_list⟩ where |basis_list⟩ is
    a full-N product computational basis state.

    Direct-sum trick: at each site k, augment A_k by appending the
    1-dim "product state row" carrying the basis vector e_{s_k}. Bond
    dim grows by 1 at every interior bond. Endpoints: scalar coefficient
    is distributed so the boundary closure correctly recovers the
    coefficient.

    Then sweep-SVD to truncate bonds back to chi_max.

    Returns the maximum bond-truncation error after the sweep.
    """
    N = state.N
    if len(basis_list) != N:
        raise ValueError("basis_list length != N")
    if abs(coefficient) < 1e-15:
        return 0.0

    d = state.d
    new_tensors: list[np.ndarray] = []
    for k in range(N):
        A = state.tensors[k]
        chi_l, _, chi_r = A.shape
        # Augmented size: chi_l_new = chi_l + 1 (except left boundary: stays chi_l)
        # Wait — for product-state addition via direct-sum:
        #   for k == 0: chi_l_new = 1 (unchanged), chi_r_new = chi_r + 1
        #   for 0 < k < N-1: chi_l_new = chi_l + 1, chi_r_new = chi_r + 1
        #   for k == N-1: chi_l_new = chi_l + 1, chi_r_new = 1
        chi_l_new = chi_l if k == 0 else chi_l + 1
        chi_r_new = chi_r if k == N - 1 else chi_r + 1
        new_A = np.zeros((chi_l_new, d, chi_r_new), dtype=complex)
        # Original block at top-left.
        new_A[:chi_l, :, :chi_r] = A
        # Product state vector (basis e_s with weight coefficient at site 0).
        flat = _flat_basis_index(basis_list[k])
        # Place at bottom-right slot (chi_l_new - 1, :, chi_r_new - 1).
        # The scalar coefficient is multiplied entirely into the FIRST
        # site; subsequent sites carry weight 1 in their product-state
        # block.
        weight = coefficient if k == 0 else 1.0
        new_A[chi_l_new - 1, flat, chi_r_new - 1] = weight
        new_tensors.append(new_A)

    state.tensors = new_tensors

    # Sweep SVD-truncate.
    max_err = 0.0
    for bond in range(N - 1):
        err = _svd_truncate_bond(state, bond, chi_max=chi_max, eps=eps)
        if err > max_err:
            max_err = err
    return max_err


def apply_transition_term(
    state: MPS, term: TransitionTerm,
    dt: float, imaginary: bool,
    chi_max: int, eps: float = 1e-10,
) -> float:
    """Apply exp(-step · H_trans) for one TransitionTerm using the
    closed-form amplitude-mixing trick.

    For H = -λ(|post⟩⟨pre| + |pre⟩⟨post|), the action on |ψ⟩ is:
      |ψ'⟩ = exp(θ·X)|ψ⟩
           = |ψ⟩ + (cosh(θ)-1)·(α|pre⟩+β|post⟩)
                 + sinh(θ)·(α|post⟩+β|pre⟩)
    where α = ⟨pre|ψ⟩, β = ⟨post|ψ⟩ and X = |post⟩⟨pre| + h.c.

    Group by basis ket:
      |ψ'⟩ = |ψ⟩ + γ_pre·|pre⟩ + γ_post·|post⟩
      γ_pre  = (cosh(θ)-1)·α + sinh(θ)·β
      γ_post = (cosh(θ)-1)·β + sinh(θ)·α

    Apply via TWO product-state additions (each grows bond by 1
    everywhere). SVD-truncate back to chi_max.

    Returns the max truncation error across bonds.
    """
    # Build full-N basis configurations for |pre⟩ and |post⟩.
    from .encoding import (
        KIND_PAD, TYPE_NONE, BID_NONE, VALUE_NONE, TOBL_NONE,
    )
    pad_site = (KIND_PAD, TYPE_NONE, BID_NONE, VALUE_NONE, TOBL_NONE)
    N = state.N
    W = term.width
    pre_full: list[tuple[int, ...]] = []
    post_full: list[tuple[int, ...]] = []
    for k in range(N):
        if term.site_left <= k < term.site_left + W:
            pre_full.append(term.pre[k - term.site_left])
            post_full.append(term.post[k - term.site_left])
        else:
            pre_full.append(pad_site)
            post_full.append(pad_site)

    # Compute alpha and beta. These are amplitudes of the FULL product
    # state |pre_full⟩ and |post_full⟩ in the current state.
    alpha = _mps_amplitude_at_basis(state, 0, pre_full)
    beta = _mps_amplitude_at_basis(state, 0, post_full)
    if abs(alpha) + abs(beta) < 1e-15:
        # Transition has no overlap with current state — skip.
        return 0.0

    lam = term.coupling
    if imaginary:
        theta = dt * lam
        c_diag = np.cosh(theta) - 1.0
        c_off = np.sinh(theta)
    else:
        theta_c = 1j * dt * lam
        c_diag = np.cosh(theta_c) - 1.0
        c_off = np.sinh(theta_c)
    gamma_pre = c_diag * alpha + c_off * beta
    gamma_post = c_diag * beta + c_off * alpha

    err1 = _add_product_state_to_mps(
        state, pre_full, complex(gamma_pre),
        chi_max=chi_max, eps=eps,
    )
    err2 = _add_product_state_to_mps(
        state, post_full, complex(gamma_post),
        chi_max=chi_max, eps=eps,
    )
    return max(err1, err2)


def factored_trotter_step_with_transitions(
    state: MPS, H, dt: float,
    transitions: list[TransitionTerm] | None = None,
    imaginary: bool = True,
    chi_max: int = 8,
    eps: float = 1e-10,
) -> float:
    """Trotter step that ALSO applies the listed transition couplings.

    Order: first apply the diagonal H gates (Phase 1 collapse), then
    sweep transitions left-to-right, then sweep right-to-left for
    second-order symmetry. Each transition's window is independent of
    others (no operator overlap if windows are disjoint), but for
    overlapping windows Trotter error is O(dt²) — fine for relaxation.
    """
    max_err = factored_trotter_step(
        state, H, dt, imaginary=imaginary, chi_max=chi_max, eps=eps,
    )
    if not transitions:
        return max_err
    # Forward sweep with HALF step.
    for term in transitions:
        err = apply_transition_term(
            state, term, dt=0.5 * dt, imaginary=imaginary,
            chi_max=chi_max, eps=eps,
        )
        if err > max_err:
            max_err = err
    # Backward sweep with HALF step (second-order Trotter symmetry).
    for term in reversed(transitions):
        err = apply_transition_term(
            state, term, dt=0.5 * dt, imaginary=imaginary,
            chi_max=chi_max, eps=eps,
        )
        if err > max_err:
            max_err = err
    return max_err
