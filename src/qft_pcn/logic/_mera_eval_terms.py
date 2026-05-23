"""Factored redex-penalty + transition-gate builders for the MERA
evaluation Hamiltonian (spec §7.1, §7.4). Constraint-based: each builder
returns a factored operator (dict[leaf -> (16,16)]) projecting onto an
UNREDUCED redex, or a factored transition gate connecting the unreduced
configuration to its reduced one.

This module contains NO classical interpreter: no substitution routine,
no beta-reduction rewriter, no AST import for walking. Reduction is
driven by the spectrum via imaginary-time evolution (spec §1.6). The
arithmetic / branch-selection result *tables* baked into the transition
gates are operator content (the same role C's U_arith plays in
factored_evolution.py) — they parametrize a Hermitian operator, they are
not a Python interpreter executing the program.
"""
from __future__ import annotations

import numpy as np

from .mera_encoding import (
    MERA_LEAF_DIM, KIND_PAD,
    KIND_APP, KIND_LAM, KIND_BIN, KIND_INT, KIND_IF, KIND_BOOL,
    KIND_SUCC, KIND_NATLIT, KIND_VAR, KIND_FIX, KIND_ZERO, KIND_EQ,
)
from .encoding import (
    VALUE_PLUS, VALUE_MINUS, VALUE_TIMES, VALUE_LT, VALUE_EQ,
    VALUE_FALSE, VALUE_TRUE, INT_LIT_OFFSET, INT_LIT_MIN, INT_LIT_MAX,
)
from ._mera_typing_rules import leaf_proj, leaf_proj_set

DEFAULT_LAMBDA_BETA = 1.0
DEFAULT_LAMBDA_ARITH = 1.0
DEFAULT_LAMBDA_IF = 1.0
DEFAULT_LAMBDA_FIX = 1.0


def _scaled(proj: np.ndarray, lam: float) -> np.ndarray:
    """Scale a projector by lam (applied to one leaf of the product so the
    factored expectation yields lam * <projector product>)."""
    return lam * proj


# ---- diagonal redex-presence penalties (spec §7.1) ----------------------


def beta_penalty_ops(app_kind_leaf: int, fn_kind_leaf: int,
                     lam: float) -> dict:
    """P[kind=APP](app) (x) P[kind=LAM](fn), scaled by lam (spec §7.1)."""
    return {
        app_kind_leaf: _scaled(leaf_proj(KIND_APP), lam),
        fn_kind_leaf:  leaf_proj(KIND_LAM),
    }


def arith_penalty_ops(bin_kind_leaf: int, bin_value_leaf: int,
                      lhs_kind_leaf: int, rhs_kind_leaf: int,
                      lam: float) -> dict:
    """P[kind=BIN] . P[value in {+,-,*}] (bin) (x) P[kind=INT](lhs)
    (x) P[kind=INT](rhs), scaled by lam (spec §7.1)."""
    return {
        bin_kind_leaf:  _scaled(leaf_proj(KIND_BIN), lam),
        bin_value_leaf: leaf_proj_set([VALUE_PLUS, VALUE_MINUS,
                                       VALUE_TIMES]),
        lhs_kind_leaf:  leaf_proj(KIND_INT),
        rhs_kind_leaf:  leaf_proj(KIND_INT),
    }


def cmp_penalty_ops(bin_kind_leaf: int, bin_value_leaf: int,
                    lhs_kind_leaf: int, rhs_kind_leaf: int,
                    lam: float) -> dict:
    """Comparison redex: BIN value in {<,==}, operands INT (spec §7.1)."""
    return {
        bin_kind_leaf:  _scaled(leaf_proj(KIND_BIN), lam),
        bin_value_leaf: leaf_proj_set([VALUE_LT, VALUE_EQ]),
        lhs_kind_leaf:  leaf_proj(KIND_INT),
        rhs_kind_leaf:  leaf_proj(KIND_INT),
    }


def if_penalty_ops(if_kind_leaf: int, cond_kind_leaf: int,
                   lam: float) -> dict:
    """P[kind=IF](if) (x) P[kind=BOOL](cond), scaled by lam (spec §7.1)."""
    return {
        if_kind_leaf:   _scaled(leaf_proj(KIND_IF), lam),
        cond_kind_leaf: leaf_proj(KIND_BOOL),
    }


def succ_penalty_ops(succ_kind_leaf: int, arg_kind_leaf: int,
                     lam: float) -> dict:
    """Succ(NatLit n) redex: P[kind=SUCC] (x) P[kind=NATLIT] (spec §7.1)."""
    return {
        succ_kind_leaf: _scaled(leaf_proj(KIND_SUCC), lam),
        arg_kind_leaf:  leaf_proj(KIND_NATLIT),
    }


# Value-leaf basis index that encodes NatLit(0) (per _tensors.py: NatLit
# sites place `int_val=node.val` directly into the value leaf — slot 0 is
# NatLit(0)). Defined here so add_zero_penalty_ops keeps its operator
# content self-contained.
NATLIT_VALUE_ZERO = 0


def add_zero_penalty_ops(bin_kind_leaf: int, bin_value_leaf: int,
                         lhs_kind_leaf: int, lhs_value_leaf: int,
                         rhs_kind_leaf: int, rhs_value_leaf: int,
                         lam: float) -> list[dict]:
    """Inclusion-exclusion sum of three factored projectors onto the
    R-AddZero redex configuration (spec §7.1 extended; plan blocker #3):
    a BIN(+) node with EXACTLY ONE of {lhs, rhs} carrying a zero (either
    KIND_ZERO or KIND_NATLIT with value-slot 0).

    Returns a LIST of three `dict[leaf -> (16,16)]` factored ops whose
    expectation values sum to <P_exactly_one_zero> on the state. Each
    factored op stays a small dict — never materialized as a 16**k
    operator (spec §1.3).

    Decomposition (P[zero](op) := P[kind=KIND_ZERO](op_kind)
                              + P[kind=KIND_NATLIT](op_kind) . P[value=0](op_value),
    a sum-of-two-factored-products that we split into TWO separate dict
    terms; combined with T1/T2/T3 of the inclusion-exclusion that gives
    six factored ops total — every one a dict[leaf -> (16,16)]):

        T1 = P[BIN](bin_kind) . P[+](bin_value) . P[zero](lhs)
        T2 = P[BIN](bin_kind) . P[+](bin_value) . P[zero](rhs)
        T3 = -P[BIN](bin_kind) . P[+](bin_value) . P[zero](lhs) . P[zero](rhs)

    so T1 + T2 + T3 projects onto "exactly one zero". The lam scaling is
    applied once on `bin_kind_leaf` (matching `arith_penalty_ops`).

    A `P[zero]` on one operand expands as a sum of two factored two-leaf
    products (KIND_ZERO on the kind leaf alone, OR KIND_NATLIT joined with
    value=0). Each expansion is a separate dict term; the cross product
    T3 contains FOUR such combinations. The total list is therefore 1*2 +
    1*2 + 1*4 = 8 factored ops — bounded, no 16**k operator.
    """
    # P[BIN](bin_kind) base, scaled by lam (the scaling rides on
    # bin_kind_leaf so the factored expectation yields lam * <product>).
    p_bin_scaled = _scaled(leaf_proj(KIND_BIN), lam)
    p_plus = leaf_proj(VALUE_PLUS)
    p_kind_zero = leaf_proj(KIND_ZERO)
    p_kind_natlit = leaf_proj(KIND_NATLIT)
    p_val_zero = leaf_proj(NATLIT_VALUE_ZERO)

    def _p_zero_terms(op_kind_leaf, op_value_leaf, sign):
        """Two factored ops summing to sign * P[zero](operand)."""
        return [
            # sign * P[KIND_ZERO] on the kind leaf alone (value leaf
            # unconstrained — Zero's value is encoder-default).
            {op_kind_leaf: sign * p_kind_zero},
            # sign * P[KIND_NATLIT] . P[value=0]
            {op_kind_leaf: sign * p_kind_natlit,
             op_value_leaf: p_val_zero},
        ]

    ops: list[dict] = []

    def _attach_bin_factors(term_ops, scale):
        """Multiply each operand-only dict by the BIN+plus factor; the
        lam scaling is carried on the BIN kind leaf for T1 and T2, and
        spread onto bin_kind_leaf for T3 with the inclusion-exclusion
        sign embedded in `scale`."""
        for op in term_ops:
            merged = dict(op)
            # The lam scaling and the inclusion-exclusion sign both ride
            # on the BIN kind leaf; the operand-side factor in `op`
            # already carries any per-zero-form sub-sign (passed in via
            # _p_zero_terms's sign). For T1/T2 scale == lam, for T3
            # scale == -lam.
            merged[bin_kind_leaf] = scale * leaf_proj(KIND_BIN)
            merged[bin_value_leaf] = p_plus
            ops.append(merged)

    # T1: lhs is zero, rhs unconstrained.
    _attach_bin_factors(_p_zero_terms(lhs_kind_leaf, lhs_value_leaf, 1.0),
                        scale=lam)
    # T2: rhs is zero, lhs unconstrained.
    _attach_bin_factors(_p_zero_terms(rhs_kind_leaf, rhs_value_leaf, 1.0),
                        scale=lam)
    # T3: -2 * P[zero](lhs) . P[zero](rhs). This implements
    # P_exactly_one = P_lhs + P_rhs - 2 * P_lhs * P_rhs, the algebraic
    # form of (P_lhs XOR P_rhs) for orthogonal projectors. The outer
    # product of the two P[zero] sums gives four factored cross terms,
    # each scaled by -2.
    for lhs_op in _p_zero_terms(lhs_kind_leaf, lhs_value_leaf, 1.0):
        for rhs_op in _p_zero_terms(rhs_kind_leaf, rhs_value_leaf, 1.0):
            # Merge the lhs and rhs operand dicts; leaves are disjoint by
            # construction (different nodes), so no overwrite conflict.
            merged = dict(lhs_op)
            for k, v in rhs_op.items():
                merged[k] = v
            merged[bin_kind_leaf] = (-2.0 * lam) * leaf_proj(KIND_BIN)
            merged[bin_value_leaf] = p_plus
            ops.append(merged)

    return ops


def eqrefl_penalty_ops(eq_kind_leaf: int, lhs_leaf: int, rhs_leaf: int,
                       lam: float) -> list[dict]:
    """Diagonal penalty for one (species, lhs/rhs leaf pair) of an Eq
    redex (spec §7.1 extended; plan blocker #4):

        lam * P[kind=KIND_EQ](eq_kind_leaf) * (I - P_equal_pair)

    where P_equal_pair on a (lhs_leaf, rhs_leaf) species pair is
    sum_i P_i(lhs) * P_i(rhs) (the diagonal of |i,i><i,i|), so

        I - P_equal_pair = I - sum_i P_i(lhs) * P_i(rhs)
                        = sum_{i!=j} P_i(lhs) * P_j(rhs)   (240 terms),

    the probability that lhs and rhs carry DIFFERENT basis indices on
    this species. The diagonal expectation is computed as a sum of
    single-leaf-marginal products (no 16^2 operator, spec §1.3): one
    constant `lam * <P[KIND_EQ]>` term, MINUS 16 factored
    `lam * <P[KIND_EQ]> * <P_i(lhs)> * <P_i(rhs)>` terms (one per
    basis index i).

    Returns 17 factored ops summing to the desired penalty:
        +lam * P[KIND_EQ](eq_kind_leaf)                         (1 op)
        -lam * P[KIND_EQ](eq_kind_leaf) * P_i(lhs) * P_i(rhs)   (16 ops)

    Each op is a dict[leaf -> (16,16)] (3 leaves max — eq_kind_leaf,
    lhs_leaf, rhs_leaf), well within the §1.3 budget.

    <H>_state = 0 on this (species, leaf-pair) iff either the Eq node
    is not KIND_EQ (the redex is not present) OR the two leaves carry
    the same basis index (this species matches). Summed over species
    and paired sub-tree nodes by `MeraEvalHamiltonian`, the total
    energy is 0 iff every species on every paired (lhs, rhs) sub-tree
    leaf agrees — i.e. lhs and rhs are leaf-for-leaf identical.
    """
    p_eq_scaled = _scaled(leaf_proj(KIND_EQ), lam)
    ops: list[dict] = []
    # Constant +lam * P[KIND_EQ] term. lhs and rhs leaves identity.
    ops.append({eq_kind_leaf: p_eq_scaled})
    # -lam * P[KIND_EQ] * P_i(lhs) * P_i(rhs) for each basis index i.
    # The lam scaling rides on eq_kind_leaf for every term so the
    # factored expectation yields lam * <product>. The minus sign also
    # rides on eq_kind_leaf via -p_eq_scaled.
    neg_p_eq_scaled = -p_eq_scaled
    for i in range(MERA_LEAF_DIM):
        ops.append({
            eq_kind_leaf: neg_p_eq_scaled,
            lhs_leaf:     leaf_proj(i),
            rhs_leaf:     leaf_proj(i),
        })
    return ops


def fix_penalty_ops(fix_kind_leaf: int, use_kind_leaf: int,
                    lam: float) -> dict:
    """Fix unfold redex: P[kind=FIX] (x) P[kind=VAR](recursion-use)
    (spec §7.2). Penalizes a FIX node whose recursion-use Var has not yet
    been contracted."""
    return {
        fix_kind_leaf: _scaled(leaf_proj(KIND_FIX), lam),
        use_kind_leaf: leaf_proj(KIND_VAR),
    }


# ---- transition-gate construction (spec §7.4) ---------------------------
#
# A transition gate moves amplitude from the UNREDUCED basis configuration
# of a leaf toward its REDUCED basis configuration. For a concrete
# (product) MERA every redex configuration is definite, so the reduced
# leaf values are a fixed function of the unreduced ones.
#
# Mechanism (lifted from the MPS substrate's factored_evolution.py).
# Reduction is a one-way rewrite: the gate must drive |u> -> |r> FULLY and
# leave |r> fixed, so that imaginary-time evolution relaxes to <H>=0. The
# proven MPS path applies projector-boost / amplitude-mixing closed forms
# whose UNREDUCED weight strictly decays and whose REDUCED configuration is
# the exact zero-energy fixed point. A *Hermitian* per-term generator with a
# u<->r off-diagonal coupling CANNOT have |r> as an eigenvector (the
# off-diagonal forces <u|H|r>=0 there), so exp(-dt H) relaxes to a MIXED
# ground state that keeps residual weight on |u> -- <H> floors above 0.
#
# The correct generator is the NON-HERMITIAN one-way drive on span{|u>,|r>}
#
#     H = lam * ( |u><u| - |r><u| )            (M := (|u> - |r>)<u|)
#
# M is idempotent (M^2 = M, since <u|u>=1, <u|r>=0), so the imaginary-time
# gate has the SAME closed form factored_evolution.py uses for its
# projector gates, exp(-dt c P) = I + (e^{-dt c} - 1) P:
#
#     g = exp(-dt lam H) = I + (e^{-dt lam} - 1) * M
#       g|u> = s|u> + (1-s)|r>      g|r> = |r>      s = e^{-dt lam}
#
# H|r> = 0: |r> is the exact zero-energy ground state -- the gate's only
# fixed point on the subspace -- and the unreduced weight decays by the
# factor s < 1 every application, so accumulated over Trotter steps the
# leaf is driven FULLY to |r>. The diagonal redex penalty P_unreduced =
# |u><u| therefore decreases strictly monotonically (architecture §13.1).
# Identity on every other basis state; a single-leaf 16x16 operator, well
# within the no-dense-operator budget (spec §1.3). The redex-presence
# projector is enforced by the evaluation Hamiltonian only applying these
# gates when the redex term's diagonal penalty is non-zero on the current
# state (the eval driver checks this); off-redex leaves are never touched.


def single_leaf_transition_gate(u_idx: int, r_idx: int, dt: float,
                                lam: float) -> np.ndarray:
    """A 16x16 imaginary-time gate that drives basis state `u_idx`
    (unreduced) fully toward `r_idx` (reduced).

    Built from the non-Hermitian one-way generator on span{|u>, |r>}:
        H = lam * ( |u><u| - |r><u| )
    whose idempotent operator M = |u><u| - |r><u| gives the projector-boost
    closed form (the same form factored_evolution.py uses):
        g = I + (e^{-dt.lam} - 1) * M
    so g|u> = s|u> + (1-s)|r> with s = e^{-dt.lam} < 1 and g|r> = |r>.
    |r> is the exact zero-energy fixed point; the unreduced weight decays
    every application, so accumulated Trotter steps drive the leaf fully to
    |r> and the redex penalty falls monotonically to 0. Identity on every
    other basis state.
    """
    if u_idx == r_idx:
        return np.eye(MERA_LEAF_DIM, dtype=complex)
    g = np.eye(MERA_LEAF_DIM, dtype=complex)
    s = np.exp(-dt * lam)
    # g = I + (s - 1) * M, with M = |u><u| - |r><u|.
    g[u_idx, u_idx] = s
    g[r_idx, u_idx] = 1.0 - s
    return g


def fix_transition_gate(u_fix: int, r_fix: int, u_use: int, r_use: int,
                        dt: float, lam: float) -> np.ndarray:
    """A 256x256 FACTORED two-leaf imaginary-time gate propagating one
    layer of a FIX body's head structure into a recursion-use Var node
    (spec §7.2, §7.4).

    The gate spans two leaves — the FIX node's `kind` leaf and a
    recursion-use Var node's `kind` leaf — as a single 256x256 operator,
    so the FIX/recursion-use coupling is one bounded two-leaf object (the
    spec §7.4 "256x256 two-leaf gate" form; never a 16**k, k>2, operator,
    spec §1.3). It is the *factored* form:

        fix_transition_gate = G_fix (x) G_use

    where G_fix swings the FIX node's kind leaf from its current basis
    index `u_fix` toward the body head `r_fix`, and G_use swings the
    recursion-use Var's kind leaf from `u_use` toward the SAME body head
    `r_use`. Both factors are single_leaf_transition_gate's — the
    well-tested per-leaf imaginary-time primitive.

    Why factored, not entangling: the §1.1 binding-as-entanglement of the
    FIX node and its recursion-use already RIDES the encoded MERA tree
    (the use's bid leaf entangled with the FIX's bid leaf, M1 §5); the
    transition gate's job is only to give imaginary-time evolution a
    matrix element to relax through. The genuineness of the coupling is
    that BOTH factors target the SAME `body head` index — the body head
    is recovered by tree addressing (use_to_binder, spec §1.2), so the
    recursion-use Var receives the body structure, not an independent
    classical value. A factored (separable) gate also keeps the product
    MERA a product under the MERA two-leaf-gate primitive's rank-1
    re-split, so the unfold does not spuriously inflate bond dimension —
    which is exactly the O(log N) tree-depth claim (spec §9.4).
    """
    g_fix = single_leaf_transition_gate(u_fix, r_fix, dt, lam)
    g_use = single_leaf_transition_gate(u_use, r_use, dt, lam)
    # 256x256 factored two-leaf gate: G_fix (x) G_use. einsum keeps the
    # tensor-product contraction explicit and within the two-leaf budget.
    dim = MERA_LEAF_DIM
    gate = np.einsum('ab,cd->acbd', g_fix, g_use,
                     optimize='greedy').reshape(dim * dim, dim * dim)
    return gate.astype(complex)


# Arithmetic / comparison result tables (operator content, spec §7.4).
_ARITH_FN = {
    VALUE_PLUS:  lambda a, b: a + b,
    VALUE_MINUS: lambda a, b: a - b,
    VALUE_TIMES: lambda a, b: a * b,
}
_CMP_FN = {
    VALUE_LT: lambda a, b: a < b,
    VALUE_EQ: lambda a, b: a == b,
}


def arith_result_value_idx(op_value: int, a_value_idx: int,
                           b_value_idx: int):
    """The reduced value-leaf index for an arithmetic redex, or None if
    the operands / result fall outside the int-literal range.

    a_value_idx, b_value_idx are value-leaf basis indices of INT literals
    (literal n stored at slot n + INT_LIT_OFFSET). Returns the reduced
    BIN-node value-leaf index (the result int literal's slot)."""
    fn = _ARITH_FN.get(op_value)
    if fn is None:
        return None
    a = a_value_idx - INT_LIT_OFFSET
    b = b_value_idx - INT_LIT_OFFSET
    res = fn(a, b)
    if not INT_LIT_MIN <= res <= INT_LIT_MAX:
        return None
    return res + INT_LIT_OFFSET


def cmp_result_value_idx(op_value: int, a_value_idx: int,
                         b_value_idx: int):
    """The reduced value-leaf index (VALUE_TRUE / VALUE_FALSE) for a
    comparison redex."""
    fn = _CMP_FN.get(op_value)
    if fn is None:
        return None
    a = a_value_idx - INT_LIT_OFFSET
    b = b_value_idx - INT_LIT_OFFSET
    return VALUE_TRUE if fn(a, b) else VALUE_FALSE
