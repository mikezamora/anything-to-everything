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
    KIND_SUCC, KIND_NATLIT, KIND_VAR, KIND_FIX,
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
# Each per-leaf transition is realized as a single-leaf Hermitian
# generator h = |r><u| + |u><r| - |u><u| - |r><r| restricted to the
# (unreduced, reduced) basis pair of that leaf, and the imaginary-time
# gate exp(-dt . lam . (P_pen - h_offdiag)) swings amplitude u -> r while
# damping the unreduced (penalized) configuration. Because the gate is a
# single-leaf 16x16 operator it is trivially within the no-dense-operator
# budget (spec §1.3). The redex-presence projector is enforced by the
# evaluation Hamiltonian only applying these gates when the redex term's
# diagonal penalty is non-zero on the current state (the eval driver
# checks this); off-redex leaves are never touched.


def single_leaf_transition_gate(u_idx: int, r_idx: int, dt: float,
                                lam: float) -> np.ndarray:
    """A 16x16 imaginary-time gate that swings amplitude from basis state
    `u_idx` (unreduced) toward `r_idx` (reduced).

    Built from the Hermitian generator H restricted to span{|u>, |r>}:
        H = lam * ( |u><u| - |u><r| - |r><u| )
    Its ground state on that 2-dim subspace is the symmetric/relaxed
    combination biased to |r> (|u> carries the +lam penalty, |r> carries
    0). exp(-dt.H) therefore moves amplitude u -> r and decays |u>.
    Identity on every other basis state.
    """
    if u_idx == r_idx:
        return np.eye(MERA_LEAF_DIM, dtype=complex)
    H = np.zeros((MERA_LEAF_DIM, MERA_LEAF_DIM), dtype=complex)
    H[u_idx, u_idx] = lam
    H[u_idx, r_idx] = -lam
    H[r_idx, u_idx] = -lam
    # Hermitian 2x2 block -> matrix exponential of -dt H.
    from scipy.linalg import expm
    g = expm(-dt * H)
    return g.astype(complex)


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
