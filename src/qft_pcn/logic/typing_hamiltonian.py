"""Typing-rule Hamiltonian for STLC + ints/bools (sub-project B).

See docs/superpowers/specs/2026-05-21-typing-hamiltonian-design.md.

Builds H_typing = Σ_rules Σ_sites H_rule(site), where each term is a
one-site or two-site Hermitian projector encoding a single STLC typing
rule. <psi|H_typing|psi> = 0 exactly when the AST encoded by psi is
well-typed; per-term energies localize violations (consumed by D).

Per spec §1.1, the Hamiltonian is STRUCTURAL — depends only on N (and
the implicit species/basis constants). There is NO AST data anywhere
in this module. Operators are kept in factored (per-species) form;
expectations go through _factored_expectation.py. The dense 65536²
operator is NEVER materialized — manifesto Temptation 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.qft_pcn.qft.mps import MPS
from .encoding import (
    SPECIES,
    KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF,
    BID_0,
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN,
    TYPE_NONE, TYPE_INT, TYPE_BOOL,
    TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI, TYPE_ARR_BB, TYPE_ARR_NESTED,
    VALUE_PLUS, VALUE_MINUS, VALUE_TIMES, VALUE_LT, VALUE_EQ,
)
from ._factored_expectation import (
    factored_local_expectation, factored_two_site_expectation,
    factored_left_bond_bid_expectation, factored_right_bond_bid_expectation,
    build_envs, build_marginals, factored_local_expectation_cached,
    factored_left_bond_bid_expectation_cached,
    factored_right_bond_bid_expectation_cached,
    factored_two_site_expectation_cached,
    _diag_factored_op, _is_diagonal, _normalize_factors,
)


# ---- Rule names -----------------------------------------------------------

RULE_T_LIT_INT = "T-Lit-Int"
RULE_T_LIT_BOOL = "T-Lit-Bool"
RULE_T_BIN_ARITH = "T-Bin-Arith"
RULE_T_BIN_CMP = "T-Bin-Cmp"
RULE_T_OBLIGATION = "T-Obligation"
RULE_T_VAR = "T-Var"
RULE_T_ABS = "T-Abs"
RULE_T_APP_ARROW = "T-App-Arrow"

_ONE_SITE_RULES = (
    RULE_T_LIT_INT, RULE_T_LIT_BOOL, RULE_T_BIN_ARITH,
    RULE_T_BIN_CMP, RULE_T_OBLIGATION,
)


# ---- Exceptions -----------------------------------------------------------


class TypingHamiltonianError(Exception):
    """Base error for TypingHamiltonian operations."""


class TermNotFound(TypingHamiltonianError):
    def __init__(self, term):
        super().__init__(f"term {term} not in this Hamiltonian's term list")


# ---- TypingTerm -----------------------------------------------------------


@dataclass(frozen=True)
class TypingTerm:
    """A single typing-rule term, addressable by (rule_id, site, arity)."""
    rule_id: str
    site: int
    arity: int


# ---- Small projector helpers ----------------------------------------------


def _project(d: int, idx: int) -> np.ndarray:
    """|idx><idx| on a d-dim Hilbert space."""
    p = np.zeros((d, d), dtype=complex)
    p[idx, idx] = 1.0
    return p


def _project_one_minus(d: int, idx: int) -> np.ndarray:
    """I - |idx><idx|."""
    return np.eye(d, dtype=complex) - _project(d, idx)


def _project_set(d: int, idxs) -> np.ndarray:
    """Sum of |i><i| for i in idxs."""
    out = np.zeros((d, d), dtype=complex)
    for i in idxs:
        out[i, i] = 1.0
    return out


# ---- bid-bond projector ---------------------------------------------------


def _bid_bond_proj_param_ty(d_bond: int, target_t: int) -> np.ndarray:
    """Projector on the bid bond (dim 1 + 8L for L channels) selecting
    channels whose param_ty register equals target_t.

    Bond slot layout (sub-project B Task 6):
      slot 0:                  |no_info>
      slot 1 + 8(i-1) + t:     |channel i, param_ty = t>  (i in [1, L], t in [0, 8))

    For target_t in [0, 8), this hits slots {1 + 8(c-1) + target_t : c in
    [1, L]}, i.e., the projector contains L diagonal 1s spaced 8 apart.
    """
    proj = np.zeros((d_bond, d_bond), dtype=complex)
    if d_bond == 1:
        return proj
    L = (d_bond - 1) // 8
    if 1 + 8 * L != d_bond:
        # Bond doesn't follow the (1 + 8L) layout — likely a chi_max-truncated
        # bond where channel-slot semantics don't apply. Return zero
        # projector (T-Var will contribute 0).
        return proj
    if not 0 <= target_t < 8:
        raise ValueError(
            f"target_t must be in [0, 8); got {target_t}")
    for c in range(1, L + 1):
        slot = 1 + 8 * (c - 1) + target_t
        proj[slot, slot] = 1.0
    return proj


# ---- Arrow-type tables (spec §3.5, §3.7) ---------------------------------


_ARROW_SRC = {
    TYPE_ARR_II: TYPE_INT,
    TYPE_ARR_IB: TYPE_INT,
    TYPE_ARR_BI: TYPE_BOOL,
    TYPE_ARR_BB: TYPE_BOOL,
}

_APP_FN_ALLOWED_BY_DST = {
    TYPE_INT:        [TYPE_ARR_II, TYPE_ARR_BI, TYPE_ARR_NESTED],
    TYPE_BOOL:       [TYPE_ARR_IB, TYPE_ARR_BB, TYPE_ARR_NESTED],
    TYPE_ARR_II:     [TYPE_ARR_NESTED],
    TYPE_ARR_IB:     [TYPE_ARR_NESTED],
    TYPE_ARR_BI:     [TYPE_ARR_NESTED],
    TYPE_ARR_BB:     [TYPE_ARR_NESTED],
    TYPE_ARR_NESTED: [TYPE_ARR_NESTED],
    TYPE_NONE:       [],
}


# ---- Per-rule energy implementations --------------------------------------


def _local_expect(state, site, op_factors, envs):
    """Dispatch: use cached env (and marginals) if available, else full sweep.

    When the op factors are all diagonal AND marginals are precomputed,
    short-circuits to O(d_local) dot product.
    """
    if envs is not None:
        lefts, rights = envs[0], envs[1]
        marginals = envs[2] if len(envs) > 2 else None
        if marginals is not None:
            factors = _normalize_factors(op_factors)
            if all(_is_diagonal(op) for op in factors):
                diag = _diag_factored_op(factors)
                return float(np.dot(marginals[site], diag))
        return factored_local_expectation_cached(
            state, site, op_factors, lefts, rights)
    return factored_local_expectation(state, site, op_factors)


def _energy_t_lit_int(state: MPS, site: int, envs=None) -> float:
    """Spec §3.1: H_lit_int(k) = P_kind=INT · (I - P_type=INT)."""
    return _local_expect(state, site, {
        "kind": _project(KIND_CUTOFF, KIND_INT),
        "type": _project_one_minus(TYPE_CUTOFF, TYPE_INT),
    }, envs)


def _energy_t_lit_bool(state: MPS, site: int, envs=None) -> float:
    """Spec §3.2: H_lit_bool(k) = P_kind=BOOL · (I - P_type=BOOL)."""
    return _local_expect(state, site, {
        "kind": _project(KIND_CUTOFF, KIND_BOOL),
        "type": _project_one_minus(TYPE_CUTOFF, TYPE_BOOL),
    }, envs)


def _energy_t_bin_arith(state: MPS, site: int, envs=None) -> float:
    """Spec §3.3a."""
    return _local_expect(state, site, {
        "kind":  _project(KIND_CUTOFF, KIND_BIN),
        "value": _project_set(VALUE_CUTOFF,
                              [VALUE_PLUS, VALUE_MINUS, VALUE_TIMES]),
        "type":  _project_one_minus(TYPE_CUTOFF, TYPE_INT),
    }, envs)


def _energy_t_bin_cmp(state: MPS, site: int, envs=None) -> float:
    """Spec §3.3b."""
    return _local_expect(state, site, {
        "kind":  _project(KIND_CUTOFF, KIND_BIN),
        "value": _project_set(VALUE_CUTOFF, [VALUE_LT, VALUE_EQ]),
        "type":  _project_one_minus(TYPE_CUTOFF, TYPE_BOOL),
    }, envs)


def _energy_t_obligation(state: MPS, site: int, envs=None) -> float:
    """Spec §3.6: Σ_{t≠TOBL_NONE} P_tobl=t · (I - P_type=t)."""
    total = 0.0
    for t in range(1, TOBL_CUTOFF):
        total += _local_expect(state, site, {
            "tobl": _project(TOBL_CUTOFF, t),
            "type": _project_one_minus(TYPE_CUTOFF, t),
        }, envs)
    return total


def _left_bond_expect(state, site, site_op, bond_proj, envs):
    if envs is not None:
        return factored_left_bond_bid_expectation_cached(
            state, site, site_op, bond_proj, envs[0], envs[1])
    return factored_left_bond_bid_expectation(
        state, site, site_op, bond_proj)


def _right_bond_expect(state, site, site_op, bond_proj, envs):
    if envs is not None:
        return factored_right_bond_bid_expectation_cached(
            state, site, site_op, bond_proj, envs[0], envs[1])
    return factored_right_bond_bid_expectation(
        state, site, site_op, bond_proj)


def _two_site_expect(state, site, op_l, op_r, envs):
    if envs is not None:
        return factored_two_site_expectation_cached(
            state, site, op_l, op_r, envs[0], envs[1])
    return factored_two_site_expectation(state, site, op_l, op_r)


def _bid_bond_proj_channel_pt(d_bond: int, channel: int,
                              target_t: int) -> np.ndarray:
    """Projector on bid bond selecting EXACTLY channel `channel` with
    param_ty = target_t. Single slot 1+8(channel-1)+target_t.
    """
    proj = np.zeros((d_bond, d_bond), dtype=complex)
    if d_bond == 1:
        return proj
    L = (d_bond - 1) // 8
    if 1 + 8 * L != d_bond or not (1 <= channel <= L):
        return proj
    if not 0 <= target_t < 8:
        return proj
    slot = 1 + 8 * (channel - 1) + target_t
    proj[slot, slot] = 1.0
    return proj


def _energy_t_var(state: MPS, site: int, envs=None) -> float:
    """Spec §3.4: VAR's type must equal the binder's param_ty for the
    binder VAR is reading from.

    Convention: bid_local = BID_{depth_from_innermost}. For a bond with
    L live binders, the binder at depth d (from innermost) lives on
    channel c = L - d. Sums over (channel c, t):
      H_var(v) += P_kind=VAR · P_bid_local=BID_{L-c} · (I - P_type=t)
                · P_left_bond_channel_c_pt_t
    summed over c in [1, L] and t in [0, 8).
    """
    if site == 0:
        return 0.0
    d_bond = state.tensors[site].shape[0]
    if d_bond == 1:
        return 0.0
    L = (d_bond - 1) // 8
    if 1 + 8 * L != d_bond:
        return 0.0
    total = 0.0
    for c in range(1, L + 1):
        depth = L - c
        if depth >= BID_CUTOFF - 1:
            continue
        bid_local_idx = BID_0 + depth   # BID_0 = 1, BID_1 = 2, ...
        for t in range(TOBL_CUTOFF):
            bond_proj = _bid_bond_proj_channel_pt(d_bond, c, t)
            if np.allclose(bond_proj, 0):
                continue
            site_op = {
                "kind": _project(KIND_CUTOFF, KIND_VAR),
                "bid":  _project(BID_CUTOFF, bid_local_idx),
                "type": _project_one_minus(TYPE_CUTOFF, t),
            }
            total += _left_bond_expect(state, site, site_op, bond_proj, envs)
    return total


def _energy_t_abs(state: MPS, site: int, envs=None) -> float:
    """Spec §3.5: LAM's NEW outgoing channel must carry param_ty = src(arrow).

    Convention: LAM creates the newest channel (number L after the LAM's
    binder is added). Its bid_local on the LAM site is BID_0 (depth=0
    from itself, before it pushes into the bond stack). The violating
    projector targets ONLY the LAM's own channel — channel L — with
    pt != src(a). Earlier channels (created by outer LAMs) are NOT
    affected by this LAM's choice.
    """
    if site >= state.N - 1:
        return 0.0
    d_bond = state.tensors[site].shape[2]
    if d_bond == 1:
        return 0.0
    L = (d_bond - 1) // 8
    if 1 + 8 * L != d_bond or L < 1:
        return 0.0
    # The LAM's new channel is the one just added — the last in declaration
    # order on the OUT bond. That's channel L.
    new_channel = L
    total = 0.0
    for a_tag, src_tag in _ARROW_SRC.items():
        # Bond proj: violating slots for the LAM's own channel only.
        violating = np.zeros((d_bond, d_bond), dtype=complex)
        for t_other in range(TOBL_CUTOFF):
            if t_other == src_tag:
                continue
            violating = violating + _bid_bond_proj_channel_pt(
                d_bond, new_channel, t_other)
        if np.allclose(violating, 0):
            continue
        site_op = {
            "kind": _project(KIND_CUTOFF, KIND_LAM),
            "type": _project(TYPE_CUTOFF, a_tag),
        }
        total += _right_bond_expect(state, site, site_op, violating, envs)
    return total


def _energy_t_app_arrow(state: MPS, site: int, envs=None) -> float:
    """Spec §3.7."""
    if site >= state.N - 1:
        return 0.0
    total = 0.0
    for y in range(TYPE_CUTOFF):
        allowed = _APP_FN_ALLOWED_BY_DST[y]
        op_left = {
            "kind": _project(KIND_CUTOFF, KIND_APP),
            "type": _project(TYPE_CUTOFF, y),
        }
        op_right = {
            "type": np.eye(TYPE_CUTOFF, dtype=complex)
                    - _project_set(TYPE_CUTOFF, allowed),
        }
        total += _two_site_expect(state, site, op_left, op_right, envs)
    return total


# ---- The Hamiltonian class -----------------------------------------------


class TypingHamiltonian:
    """Sum-of-terms typing Hamiltonian for an N-site lattice.

    Per spec §1.1, depends only on N (and the implicit species/basis
    constants). The SAME instance evaluates correctly on any MPS
    produced by the typing-aware encoder for an AST of size <= N.
    """

    SPECIES = SPECIES   # 5-species tuple

    def __init__(self, N: int):
        self.N = int(N)
        self.terms: list[TypingTerm] = self._enumerate_terms()
        self._terms_set = frozenset(self.terms)

    def _enumerate_terms(self) -> list[TypingTerm]:
        """Deterministic term enumeration per spec §6.2.

        One-site rules cover every site k in [0, N).
        T-Var: VAR has a left bond → sites k in [1, N) (arity 2).
        T-Abs, T-App-Arrow: anchored at sites with a right bond → k in [0, N-1).
        Total: 5N + 3(N-1).
        """
        terms: list[TypingTerm] = []
        for rule in _ONE_SITE_RULES:
            for k in range(self.N):
                terms.append(TypingTerm(rule_id=rule, site=k, arity=1))
        for k in range(1, self.N):
            terms.append(TypingTerm(rule_id=RULE_T_VAR, site=k, arity=2))
        for k in range(self.N - 1):
            terms.append(TypingTerm(rule_id=RULE_T_ABS, site=k, arity=2))
        for k in range(self.N - 1):
            terms.append(TypingTerm(rule_id=RULE_T_APP_ARROW, site=k, arity=2))
        return terms

    def term_energy(self, state: MPS, term: TypingTerm,
                    envs=None) -> float:
        """<psi|H_term|psi> for a single named term.

        If `envs` is supplied (output of build_envs(state)), one-site rules
        reuse the precomputed left/right environments for speed.
        """
        if term not in self._terms_set:
            raise TermNotFound(term)
        rule = term.rule_id
        site = term.site
        if rule == RULE_T_LIT_INT:
            return _energy_t_lit_int(state, site, envs)
        if rule == RULE_T_LIT_BOOL:
            return _energy_t_lit_bool(state, site, envs)
        if rule == RULE_T_BIN_ARITH:
            return _energy_t_bin_arith(state, site, envs)
        if rule == RULE_T_BIN_CMP:
            return _energy_t_bin_cmp(state, site, envs)
        if rule == RULE_T_OBLIGATION:
            return _energy_t_obligation(state, site, envs)
        if rule == RULE_T_VAR:
            return _energy_t_var(state, site, envs)
        if rule == RULE_T_ABS:
            return _energy_t_abs(state, site, envs)
        if rule == RULE_T_APP_ARROW:
            return _energy_t_app_arrow(state, site, envs)
        raise TermNotFound(term)

    def total_energy(self, state: MPS) -> float:
        """Σ over all terms of term_energy(state, term). Precomputes envs +
        per-site marginals once for use across all one-site rules.
        """
        lefts, rights = build_envs(state)
        marginals = build_marginals(state, lefts, rights)
        envs = (lefts, rights, marginals)
        return sum(self.term_energy(state, t, envs=envs) for t in self.terms)

    def residuals(self, state: MPS) -> dict:
        """All per-term energies keyed by (rule_id, site)."""
        lefts, rights = build_envs(state)
        marginals = build_marginals(state, lefts, rights)
        envs = (lefts, rights, marginals)
        return {(t.rule_id, t.site): self.term_energy(state, t, envs=envs)
                for t in self.terms}
