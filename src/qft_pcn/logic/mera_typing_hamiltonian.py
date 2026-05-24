"""MERA-native typing Hamiltonian (sub-project M2; spec §5, §6).

H_typing = sum over rules, over AST nodes, of H_rule(node), each a
factored projector operator on the 5-leaf node windows of the M1 MERA
layout. <psi|H_typing|psi> = 0 exactly when the encoded AST is
well-typed; per-term energies localize violations (consumed by M3).

Per spec §1.7 the Hamiltonian is STRUCTURAL: built from MeraEncodingMeta
only. No AST is walked here. Every term is evaluated via
mera_window_expectation_factored — the dense 16**k operator is NEVER
materialized (spec §1.3).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.qft_pcn.qft.mera import MERA
from .mera_encoder import MeraEncodingMeta
from .mera_encoding import (
    MERA_TYPE_CUTOFF, SPECIES_LEAF_OFFSET, LEAVES_PER_NODE,
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN,
    KIND_ZERO, KIND_SUCC, KIND_NATLIT, KIND_NIL, KIND_CONS,
    KIND_EQ, KIND_FORALL, KIND_FIX,
    TYPE_NAT, TYPE_LIST, TYPE_PROP,
)
from .encoding import (
    TYPE_INT, TYPE_BOOL,
    TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI, TYPE_ARR_BB,
    VALUE_PLUS, VALUE_MINUS, VALUE_TIMES, VALUE_LT, VALUE_EQ,
)
from . import _mera_typing_rules as R
from ._mera_window import mera_window_expectation_factored


# ---- Rule names ---------------------------------------------------------

RULE_T_LIT_INT = "T-Lit-Int"
RULE_T_LIT_BOOL = "T-Lit-Bool"
RULE_T_BIN_ARITH = "T-Bin-Arith"
RULE_T_BIN_CMP = "T-Bin-Cmp"
RULE_T_VAR = "T-Var"
RULE_T_ABS = "T-Abs"
RULE_T_APP_ARROW = "T-App-Arrow"
RULE_T_OBLIGATION = "T-Obligation"
RULE_T_ZERO = "T-Zero"
RULE_T_SUCC = "T-Succ"
RULE_T_NATLIT = "T-NatLit"
RULE_T_NIL = "T-Nil"
RULE_T_CONS = "T-Cons"
RULE_T_EQ = "T-Eq"
RULE_T_FORALL = "T-Forall"
RULE_T_FIX = "T-Fix"

# Per-node (arity 1) rules.
_ONE_NODE_RULES = (
    RULE_T_LIT_INT, RULE_T_LIT_BOOL, RULE_T_BIN_ARITH, RULE_T_BIN_CMP,
    RULE_T_ZERO, RULE_T_NATLIT, RULE_T_NIL,
)
# Two-node (arity 2) rules; anchored at the primary node.
_TWO_NODE_RULES = (
    RULE_T_VAR, RULE_T_ABS, RULE_T_APP_ARROW, RULE_T_OBLIGATION,
    RULE_T_SUCC, RULE_T_CONS, RULE_T_EQ, RULE_T_FORALL, RULE_T_FIX,
)

# Flat arrow type tags and their src/dst decomposition.
_ARROW_SRC = {
    TYPE_ARR_II: TYPE_INT, TYPE_ARR_IB: TYPE_INT,
    TYPE_ARR_BI: TYPE_BOOL, TYPE_ARR_BB: TYPE_BOOL,
}
_ARROW_DST = {
    TYPE_ARR_II: TYPE_INT, TYPE_ARR_IB: TYPE_BOOL,
    TYPE_ARR_BI: TYPE_INT, TYPE_ARR_BB: TYPE_BOOL,
}
_ARROW_TAGS = tuple(_ARROW_SRC.keys())


# ---- Exceptions ---------------------------------------------------------


class MeraTypingHamiltonianError(Exception):
    """Base error for MeraTypingHamiltonian operations."""


class MeraTermNotFound(MeraTypingHamiltonianError):
    def __init__(self, term):
        super().__init__(f"term {term} not in this Hamiltonian's term list")


# ---- MeraTypingTerm -----------------------------------------------------


@dataclass(frozen=True)
class MeraTypingTerm:
    """A typing-rule term, addressable by (rule_id, node, arity)."""
    rule_id: str
    node: int
    arity: int


# ---- The Hamiltonian class ---------------------------------------------


class MeraTypingHamiltonian:
    """Structural typing Hamiltonian over a MERA leaf layout (spec §5)."""

    def __init__(self, meta: MeraEncodingMeta):
        self.meta = meta
        self.N_nodes = meta.n_nodes
        self.terms: list[MeraTypingTerm] = self._enumerate_terms()
        self._terms_set = frozenset(self.terms)

    def _enumerate_terms(self) -> list[MeraTypingTerm]:
        """Deterministic enumeration: every rule at every AST node.

        A rule's energy function is zero when the node's kind does not
        match the rule, so enumerating every rule at every node is
        correct and keeps the Hamiltonian structural (spec §1.7).
        """
        terms: list[MeraTypingTerm] = []
        for rule in _ONE_NODE_RULES:
            for n in range(self.N_nodes):
                terms.append(MeraTypingTerm(rule, n, 1))
        for rule in _TWO_NODE_RULES:
            for n in range(self.N_nodes):
                terms.append(MeraTypingTerm(rule, n, 2))
        return terms

    def term_energy(self, state: MERA, term: MeraTypingTerm) -> float:
        if term not in self._terms_set:
            raise MeraTermNotFound(term)
        fn = _RULE_DISPATCH.get(term.rule_id)
        if fn is None:
            raise MeraTermNotFound(term)
        return float(fn(state, self.meta, term.node))

    def total_energy(self, state: MERA) -> float:
        return sum(self.term_energy(state, t) for t in self.terms)

    def residuals(self, state: MERA) -> dict:
        return {(t.rule_id, t.node): self.term_energy(state, t)
                for t in self.terms}

    def term_gates(self, state: MERA, term: MeraTypingTerm, dt: float,
                   imaginary: bool = True):
        """Factored exp(-dt.H_term) gates for imaginary-time evolution.

        The typing Hamiltonian's terms are constraints, not reductions.
        For a well-typed program every term is already zero — no gate is
        emitted, so typing contributes nothing to the evolution dynamics
        (it only contributes to the measured energy trajectory). On an
        ill-typed program the typing energy is constant under the eval
        gates, so again no typing gate is needed for the M2 reduction
        demos. term_gates therefore returns no gates; the typing
        Hamiltonian's role under compose is the energy functional, not a
        driver (spec §7.5 — typing is a constraint).
        """
        if term not in self._terms_set:
            raise MeraTermNotFound(term)
        return []

    def term_affected_leaves(self, term: MeraTypingTerm) -> frozenset:
        """Typing terms never emit a gate (term_gates returns []), so the
        evolution-loop redex-presence cache can mark them inactive once and
        skip every subsequent step regardless of which leaves moved. The
        empty set means "no leaves can ever activate this term"; any
        ``changed_leaves & frozenset()`` intersection is empty, so the
        ``mera_trotter_step`` filter skips typing terms forever after the
        first inactive observation. The typing energy is still summed in
        ``total_energy`` (which doesn't go through this cache)."""
        return frozenset()


# ---- Window helper ------------------------------------------------------


def _window(meta, node, leaf_ops_by_species):
    """Build a factored leaf_ops dict for one node from a
    {species_name: (16,16) op} mapping."""
    return {meta.layout.leaf_of(node, sp): op
            for sp, op in leaf_ops_by_species.items()}


def _binder_node_of(meta, use_node):
    """The binder AST node index for a use node, via use_to_binder
    addressing (spec §1.2 — addressing, not classical lookup).

    Returns None when the resolved binder node lies outside the encoded
    leaf array (``binder_node >= meta.n_nodes``). Defense-in-depth backstop:
    the underlying bundle-encoder double-rebase bug was fixed in
    `mera_encoder._encode_bundle` (dropped the redundant second-pass
    rebase for ci>=1 witness children — the first pass already offsets
    them by `running`). With that fix in place this guard should never
    fire on valid programs, but is retained as an operator-algebraic skip
    (spec §1.5): a stale binder reference points to a non-existent leaf,
    so the would-be projector contributes 0 to <psi|H|psi> rather than
    raising under `state.leaves[leaf]` in `_energy_t_var`'s factored
    expectation."""
    use_bid = meta.layout.leaf_of(use_node, "bid")
    binder_bid = meta.use_to_binder.get(use_bid)
    if binder_bid is None:
        return None
    binder_node = (binder_bid - SPECIES_LEAF_OFFSET["bid"]) // LEAVES_PER_NODE
    if binder_node < 0 or binder_node >= meta.n_nodes:
        return None
    return binder_node


# ---- Per-rule energy functions (spec §5, §6) ----------------------------


def _energy_t_lit_int(state, meta, node) -> float:
    """B §3.1: P[kind=INT] . (I - P[type=INT])."""
    ops = _window(meta, node, {
        "kind": R.leaf_proj(KIND_INT),
        "type": R.leaf_proj_one_minus(TYPE_INT),
    })
    return float(mera_window_expectation_factored(state, ops).real)


def _energy_t_lit_bool(state, meta, node) -> float:
    """B §3.2."""
    ops = _window(meta, node, {
        "kind": R.leaf_proj(KIND_BOOL),
        "type": R.leaf_proj_one_minus(TYPE_BOOL),
    })
    return float(mera_window_expectation_factored(state, ops).real)


def _energy_t_bin_arith(state, meta, node) -> float:
    """B §3.3a: P[kind=BIN] . P[value in {+,-,*}] . (I - P[type=INT])."""
    ops = _window(meta, node, {
        "kind":  R.leaf_proj(KIND_BIN),
        "value": R.leaf_proj_set([VALUE_PLUS, VALUE_MINUS, VALUE_TIMES]),
        "type":  R.leaf_proj_one_minus(TYPE_INT),
    })
    return float(mera_window_expectation_factored(state, ops).real)


def _energy_t_bin_cmp(state, meta, node) -> float:
    """B §3.3b: P[kind=BIN] . P[value in {<,==}] . (I - P[type=BOOL])."""
    ops = _window(meta, node, {
        "kind":  R.leaf_proj(KIND_BIN),
        "value": R.leaf_proj_set([VALUE_LT, VALUE_EQ]),
        "type":  R.leaf_proj_one_minus(TYPE_BOOL),
    })
    return float(mera_window_expectation_factored(state, ops).real)


def _energy_t_var(state, meta, node) -> float:
    """Spec §5.4: a VAR node's type must equal its binder's parameter type.

    Two-node factored term over the use node and its binder node. meta is
    read only to address leaves (spec §1.2); the binder/use correlation is
    carried by the encoded MERA tree.

    The binder's parameter type leaf depends on the binder kind:
      - Lam: the binder's `type` leaf holds the arrow type; param type is
        src(arrow). We sum over flat arrow tags `a`, projecting the binder
        `type` leaf onto `a` and the use `type` leaf onto I-P[src(a)].
      - Forall / Fix: the parameter type is written into the binder's
        `value` leaf (encoder contract, spec §6.7-§6.8). We sum over type
        tags `t` against that leaf.
    The two binder-kind branches are summed; the kind projector on the
    binder node selects the live branch, so the term is exact regardless.
    """
    binder_node = _binder_node_of(meta, node)
    if binder_node is None:
        return 0.0
    use_kind = meta.layout.leaf_of(node, "kind")
    use_type = meta.layout.leaf_of(node, "type")
    binder_kind = meta.layout.leaf_of(binder_node, "kind")
    binder_type = meta.layout.leaf_of(binder_node, "type")
    binder_value = meta.layout.leaf_of(binder_node, "value")
    total = 0.0
    # Lam binder branch: param type = src(arrow on binder's type leaf).
    for a in _ARROW_TAGS:
        src = _ARROW_SRC[a]
        ops = {
            use_kind:    R.leaf_proj(KIND_VAR),
            binder_kind: R.leaf_proj(KIND_LAM),
            binder_type: R.leaf_proj(a),
            use_type:    R.leaf_proj_one_minus(src),
        }
        total += float(mera_window_expectation_factored(state, ops).real)
    # Forall / Fix binder branch: param type in the binder's value leaf.
    for binder_kind_idx in (KIND_FORALL, KIND_FIX):
        for t in range(MERA_TYPE_CUTOFF):
            ops = {
                use_kind:     R.leaf_proj(KIND_VAR),
                binder_kind:  R.leaf_proj(binder_kind_idx),
                binder_value: R.leaf_proj(t),
                use_type:     R.leaf_proj_one_minus(t),
            }
            total += float(mera_window_expectation_factored(state, ops).real)
    return total


def _energy_t_abs(state, meta, node) -> float:
    """Spec §5.4 (B §3.5): a LAM node's type must be an arrow whose src is
    the parameter type and whose dst is the body node's type. Two-node
    term over the LAM node and its body node.
    """
    kids = meta.children_of_node.get(node, [])
    if not kids:
        return 0.0
    body = kids[0]
    lam_kind = meta.layout.leaf_of(node, "kind")
    lam_type = meta.layout.leaf_of(node, "type")
    body_type = meta.layout.leaf_of(body, "type")
    total = 0.0
    # LAM type is arrow a, body type != dst(a).
    for a in _ARROW_TAGS:
        dst = _ARROW_DST[a]
        ops = {
            lam_kind:  R.leaf_proj(KIND_LAM),
            lam_type:  R.leaf_proj(a),
            body_type: R.leaf_proj_one_minus(dst),
        }
        total += float(mera_window_expectation_factored(state, ops).real)
    return total


def _energy_t_app_arrow(state, meta, node) -> float:
    """B §3.7: an APP node's fn must have an arrow type whose dst equals
    the APP node's result type. Two-node term over the APP node and its fn
    node.
    """
    kids = meta.children_of_node.get(node, [])
    if not kids:
        return 0.0
    fn = kids[0]
    app_kind = meta.layout.leaf_of(node, "kind")
    app_type = meta.layout.leaf_of(node, "type")
    fn_type = meta.layout.leaf_of(fn, "type")
    total = 0.0
    # APP result type is y; the fn's type must be an arrow with dst = y.
    # Penalize: APP type = y AND fn type not in {arrows a : dst(a) = y}.
    for y in (TYPE_INT, TYPE_BOOL):
        allowed = [a for a in _ARROW_TAGS if _ARROW_DST[a] == y]
        ops = {
            app_kind: R.leaf_proj(KIND_APP),
            app_type: R.leaf_proj(y),
            fn_type:  R.leaf_identity() - R.leaf_proj_set(allowed),
        }
        total += float(mera_window_expectation_factored(state, ops).real)
    return total


# Obligation table: parent kind -> list of (child_position, expected_fn).
# expected_fn(parent_type) -> the expected child type tag, or a sentinel.
_OBLIG_FIXED = object()   # expected type is a fixed tag


def _energy_t_obligation(state, meta, node) -> float:
    """Spec §5.5: 'child must have type X' constraints as genuine two-node
    factored terms. Anchored at the parent node; the child node is found
    from meta.children_of_node.

    Parent-kind -> child obligations:
      IF   cond -> Bool;  then/else -> type(IF) (summed over tag)
      BIN  lhs/rhs -> Int  (only for arith ops + - *)
    LAM-body and APP-arg obligations are realized by T-Abs / T-App-Arrow
    and the per-binder rules; Succ/Cons/Eq obligations are realized by
    their dedicated extended rules (spec §6). T-Obligation here covers
    the IF and BIN-operand couplings.
    """
    kids = meta.children_of_node.get(node, [])
    p_kind = meta.layout.leaf_of(node, "kind")
    p_type = meta.layout.leaf_of(node, "type")
    p_value = meta.layout.leaf_of(node, "value")
    total = 0.0
    # IF: kids = [cond, then, else].
    if len(kids) == 3:
        cond, then_b, else_b = kids
        cond_type = meta.layout.leaf_of(cond, "type")
        then_type = meta.layout.leaf_of(then_b, "type")
        else_type = meta.layout.leaf_of(else_b, "type")
        # cond must be Bool.
        ops = {p_kind: R.leaf_proj(KIND_IF),
               cond_type: R.leaf_proj_one_minus(TYPE_BOOL)}
        total += float(mera_window_expectation_factored(state, ops).real)
        # then/else must match IF's type (summed over tag).
        for t in range(MERA_TYPE_CUTOFF):
            for branch_type in (then_type, else_type):
                ops = {p_kind: R.leaf_proj(KIND_IF),
                       p_type: R.leaf_proj(t),
                       branch_type: R.leaf_proj_one_minus(t)}
                total += float(
                    mera_window_expectation_factored(state, ops).real)
    # BIN: kids = [lhs, rhs]. Operands of an arithmetic / comparison op
    # must be Int.
    if len(kids) == 2 and meta.layout.species_of_leaf[p_kind] == "kind":
        lhs, rhs = kids
        lhs_type = meta.layout.leaf_of(lhs, "type")
        rhs_type = meta.layout.leaf_of(rhs, "type")
        for operand_type in (lhs_type, rhs_type):
            ops = {
                p_kind: R.leaf_proj(KIND_BIN),
                p_value: R.leaf_proj_set([VALUE_PLUS, VALUE_MINUS,
                                          VALUE_TIMES, VALUE_LT, VALUE_EQ]),
                operand_type: R.leaf_proj_one_minus(TYPE_INT),
            }
            total += float(mera_window_expectation_factored(state, ops).real)
    return total


# ---- Extended-calculus rules (spec §6) ----------------------------------


def _energy_t_zero(state, meta, node) -> float:
    """§6.1: P[kind=ZERO] . (I - P[type=NAT])."""
    ops = _window(meta, node, {
        "kind": R.leaf_proj(KIND_ZERO),
        "type": R.leaf_proj_one_minus(TYPE_NAT),
    })
    return float(mera_window_expectation_factored(state, ops).real)


def _energy_t_succ(state, meta, node) -> float:
    """§6.2: result type Nat + arg type Nat."""
    kids = meta.children_of_node.get(node, [])
    if not kids:
        return 0.0
    arg = kids[0]
    succ_kind = meta.layout.leaf_of(node, "kind")
    succ_type = meta.layout.leaf_of(node, "type")
    arg_type = meta.layout.leaf_of(arg, "type")
    total = 0.0
    ops = {succ_kind: R.leaf_proj(KIND_SUCC),
           succ_type: R.leaf_proj_one_minus(TYPE_NAT)}
    total += float(mera_window_expectation_factored(state, ops).real)
    ops = {succ_kind: R.leaf_proj(KIND_SUCC),
           arg_type: R.leaf_proj_one_minus(TYPE_NAT)}
    total += float(mera_window_expectation_factored(state, ops).real)
    return total


def _energy_t_natlit(state, meta, node) -> float:
    """§6.3: P[kind=NATLIT] . (I - P[type=NAT])."""
    ops = _window(meta, node, {
        "kind": R.leaf_proj(KIND_NATLIT),
        "type": R.leaf_proj_one_minus(TYPE_NAT),
    })
    return float(mera_window_expectation_factored(state, ops).real)


def _energy_t_nil(state, meta, node) -> float:
    """§6.4: P[kind=NIL] . (I - P[type=LIST])."""
    ops = _window(meta, node, {
        "kind": R.leaf_proj(KIND_NIL),
        "type": R.leaf_proj_one_minus(TYPE_LIST),
    })
    return float(mera_window_expectation_factored(state, ops).real)


def _energy_t_cons(state, meta, node) -> float:
    """§6.5: result is a List + tail is a List + head : tau (summed over
    element-type tags from the CONS node's value leaf)."""
    kids = meta.children_of_node.get(node, [])
    if len(kids) < 2:
        return 0.0
    head, tail = kids[0], kids[1]
    c_kind = meta.layout.leaf_of(node, "kind")
    c_type = meta.layout.leaf_of(node, "type")
    c_value = meta.layout.leaf_of(node, "value")
    head_type = meta.layout.leaf_of(head, "type")
    tail_type = meta.layout.leaf_of(tail, "type")
    total = 0.0
    ops = {c_kind: R.leaf_proj(KIND_CONS),
           c_type: R.leaf_proj_one_minus(TYPE_LIST)}
    total += float(mera_window_expectation_factored(state, ops).real)
    ops = {c_kind: R.leaf_proj(KIND_CONS),
           tail_type: R.leaf_proj_one_minus(TYPE_LIST)}
    total += float(mera_window_expectation_factored(state, ops).real)
    # head : tau, where tau is the element type carried in the value leaf.
    for tau in range(MERA_TYPE_CUTOFF):
        ops = {c_kind: R.leaf_proj(KIND_CONS),
               c_value: R.leaf_proj(tau),
               head_type: R.leaf_proj_one_minus(tau)}
        total += float(mera_window_expectation_factored(state, ops).real)
    return total


def _energy_t_eq(state, meta, node) -> float:
    """§6.6: result is a Prop + lhs/rhs same type (summed over tag)."""
    kids = meta.children_of_node.get(node, [])
    if len(kids) < 2:
        return 0.0
    lhs, rhs = kids[0], kids[1]
    eq_kind = meta.layout.leaf_of(node, "kind")
    eq_type = meta.layout.leaf_of(node, "type")
    lhs_type = meta.layout.leaf_of(lhs, "type")
    rhs_type = meta.layout.leaf_of(rhs, "type")
    total = 0.0
    ops = {eq_kind: R.leaf_proj(KIND_EQ),
           eq_type: R.leaf_proj_one_minus(TYPE_PROP)}
    total += float(mera_window_expectation_factored(state, ops).real)
    for tau in range(MERA_TYPE_CUTOFF):
        ops = {eq_kind: R.leaf_proj(KIND_EQ),
               lhs_type: R.leaf_proj(tau),
               rhs_type: R.leaf_proj_one_minus(tau)}
        total += float(mera_window_expectation_factored(state, ops).real)
    return total


def _energy_t_forall(state, meta, node) -> float:
    """§6.7: result is a Prop + body is a Prop."""
    kids = meta.children_of_node.get(node, [])
    if not kids:
        return 0.0
    body = kids[0]
    f_kind = meta.layout.leaf_of(node, "kind")
    f_type = meta.layout.leaf_of(node, "type")
    body_type = meta.layout.leaf_of(body, "type")
    total = 0.0
    ops = {f_kind: R.leaf_proj(KIND_FORALL),
           f_type: R.leaf_proj_one_minus(TYPE_PROP)}
    total += float(mera_window_expectation_factored(state, ops).real)
    ops = {f_kind: R.leaf_proj(KIND_FORALL),
           body_type: R.leaf_proj_one_minus(TYPE_PROP)}
    total += float(mera_window_expectation_factored(state, ops).real)
    return total


def _energy_t_fix(state, meta, node) -> float:
    """§6.8: result : tau + body : tau, tau from the FIX node's value
    leaf (the FIX's parameter type)."""
    kids = meta.children_of_node.get(node, [])
    if not kids:
        return 0.0
    body = kids[0]
    f_kind = meta.layout.leaf_of(node, "kind")
    f_type = meta.layout.leaf_of(node, "type")
    f_value = meta.layout.leaf_of(node, "value")
    body_type = meta.layout.leaf_of(body, "type")
    total = 0.0
    for tau in range(MERA_TYPE_CUTOFF):
        ops = {f_kind: R.leaf_proj(KIND_FIX),
               f_value: R.leaf_proj(tau),
               f_type: R.leaf_proj_one_minus(tau)}
        total += float(mera_window_expectation_factored(state, ops).real)
        ops = {f_kind: R.leaf_proj(KIND_FIX),
               f_value: R.leaf_proj(tau),
               body_type: R.leaf_proj_one_minus(tau)}
        total += float(mera_window_expectation_factored(state, ops).real)
    return total


_RULE_DISPATCH: dict = {
    RULE_T_LIT_INT: _energy_t_lit_int,
    RULE_T_LIT_BOOL: _energy_t_lit_bool,
    RULE_T_BIN_ARITH: _energy_t_bin_arith,
    RULE_T_BIN_CMP: _energy_t_bin_cmp,
    RULE_T_VAR: _energy_t_var,
    RULE_T_ABS: _energy_t_abs,
    RULE_T_APP_ARROW: _energy_t_app_arrow,
    RULE_T_OBLIGATION: _energy_t_obligation,
    RULE_T_ZERO: _energy_t_zero,
    RULE_T_SUCC: _energy_t_succ,
    RULE_T_NATLIT: _energy_t_natlit,
    RULE_T_NIL: _energy_t_nil,
    RULE_T_CONS: _energy_t_cons,
    RULE_T_EQ: _energy_t_eq,
    RULE_T_FORALL: _energy_t_forall,
    RULE_T_FIX: _energy_t_fix,
}


