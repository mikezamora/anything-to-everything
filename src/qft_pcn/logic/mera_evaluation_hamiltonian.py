"""MERA-native evaluation Hamiltonian (sub-project M2; spec §7).

H_eval = sum over redex rules, over AST nodes, of lambda_rule . H_rule.
Each H_rule is a non-negative factored projector onto an UNREDUCED redex
configuration. <psi|H_eval|psi> = 0 iff the encoded program is in normal
form. Reduction is ground-state finding (spec §1.6): imaginary-time
evolution under H_typing + H_eval drives the MERA toward the normal form.
The transition gates that GIVE evolution a matrix element to relax
through are supplied by term_gates (spec §7.4).

NO classical interpreter: this module never substitutes, never
beta-reduces, never imports the AST module for walking.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.qft_pcn.qft.mera import MERA
from .mera_encoder import MeraEncodingMeta
from .mera_encoding import (
    MERA_LEAF_DIM, SPECIES_LEAF_OFFSET, LEAVES_PER_NODE,
    KIND_PAD, KIND_APP, KIND_BIN, KIND_IF, KIND_SUCC, KIND_FIX,
    KIND_INT, KIND_NATLIT,
)
from .encoding import (
    VALUE_PLUS, VALUE_MINUS, VALUE_TIMES, VALUE_LT, VALUE_EQ,
    VALUE_FALSE, VALUE_TRUE, INT_LIT_OFFSET,
)
from ._mera_window import mera_window_expectation_factored
from ._mera_eval_terms import (
    beta_penalty_ops, arith_penalty_ops, cmp_penalty_ops,
    if_penalty_ops, succ_penalty_ops, fix_penalty_ops,
    single_leaf_transition_gate, fix_transition_gate,
    arith_result_value_idx, cmp_result_value_idx,
    DEFAULT_LAMBDA_BETA, DEFAULT_LAMBDA_ARITH, DEFAULT_LAMBDA_IF,
    DEFAULT_LAMBDA_FIX,
)

RULE_R_BETA = "R-Beta"
RULE_R_ARITH = "R-Arith"
RULE_R_CMP = "R-Cmp"
RULE_R_IF = "R-If"
RULE_R_SUCC = "R-Succ"
RULE_R_FIX = "R-Fix"

_ALL_RULES = (RULE_R_BETA, RULE_R_ARITH, RULE_R_CMP, RULE_R_IF,
              RULE_R_SUCC, RULE_R_FIX)


class MeraEvalError(Exception):
    """Base error for MeraEvalHamiltonian operations."""


class MeraEvalBudgetExceeded(MeraEvalError):
    """A Fix unfold's normal form exceeds the encoder's node budget."""


class MeraEvalTermNotFound(MeraEvalError):
    def __init__(self, term):
        super().__init__(f"term {term} not in this Hamiltonian's term list")


@dataclass(frozen=True)
class MeraEvalTerm:
    rule_id: str
    node: int
    arity: int


def _kind_leaf(meta, node):
    return meta.layout.leaf_of(node, "kind")


def _value_leaf(meta, node):
    return meta.layout.leaf_of(node, "value")


def _leaf_argmax(state: MERA, leaf: int) -> int:
    """Argmax basis index of a leaf's marginal (concrete product MERA).

    For a product MERA the leaf marginal is exactly the per-component
    weight |v_leaf|^2 of the leaf's state vector: the double-network
    ascent telescopes (every isometry W satisfies W^dag W = I) and the
    other leaves' overlaps cancel in the argmax. This reads the leaf
    tensor directly — the same data decode/sample already consume
    (spec §5.4) — replacing 16 full causal-cone ascents per call, the
    dominant cost of term_gates during imaginary-time evolution. The
    argmax is identical to the projector-expectation form (verified
    against state.local_expectation for every leaf).

    A genuine term-superposition state has no single definite leaf
    value, so fall back to the exact projector form there.
    """
    if state._superposition_terms is None:
        v = state.leaves[leaf][0, :, 0]
        return int(np.argmax(np.abs(v) ** 2))
    p = np.empty(MERA_LEAF_DIM, dtype=float)
    for b in range(MERA_LEAF_DIM):
        proj = np.zeros((MERA_LEAF_DIM, MERA_LEAF_DIM), dtype=complex)
        proj[b, b] = 1.0
        p[b] = float(np.real(state.local_expectation(leaf, proj)))
    return int(np.argmax(p))


def _leaf_argmax_in(state: MERA, leaf: int, allowed) -> int:
    """Dominant basis index of a leaf RESTRICTED to `allowed` indices.

    During imaginary-time evolution a redex leaf is a genuine {unreduced,
    reduced} superposition; a plain argmax flips to the reduced index once
    the rotation passes 50%, which would corrupt the operand/operator
    values the transition gate is built from. Restricting the argmax to
    the set of indices a leaf can carry IN THE UNREDUCED redex (e.g. the
    operator set {+,-,*} for a BIN value leaf, or the non-PAD literal
    slots for an operand value leaf) reads the unreduced configuration
    stably for the whole reduction, so the gate keeps targeting the right
    result. Pure addressing of the leaf's own weights (spec §1.2)."""
    w = _leaf_weights(state, leaf)
    allowed = list(allowed)
    best = allowed[0]
    best_w = -1.0
    for idx in allowed:
        if 0 <= idx < w.shape[0] and w[idx] > best_w:
            best_w = w[idx]
            best = idx
    return int(best)


def _leaf_weights(state: MERA, leaf: int) -> np.ndarray:
    """Per-basis weight vector |v_leaf|^2 of a leaf (concrete product MERA)
    or the projector-expectation marginal for a superposition state."""
    if state._superposition_terms is None:
        v = state.leaves[leaf][0, :, 0]
        return np.abs(v) ** 2
    p = np.empty(MERA_LEAF_DIM, dtype=float)
    for b in range(MERA_LEAF_DIM):
        proj = np.zeros((MERA_LEAF_DIM, MERA_LEAF_DIM), dtype=complex)
        proj[b, b] = 1.0
        p[b] = float(np.real(state.local_expectation(leaf, proj)))
    return p


class MeraEvalHamiltonian:
    """Structural evaluation Hamiltonian over a MERA leaf layout (spec §7)."""

    def __init__(self, meta: MeraEncodingMeta,
                 lambda_beta: float = DEFAULT_LAMBDA_BETA,
                 lambda_arith: float = DEFAULT_LAMBDA_ARITH,
                 lambda_if: float = DEFAULT_LAMBDA_IF,
                 lambda_fix: float = DEFAULT_LAMBDA_FIX):
        self.meta = meta
        self.N_nodes = meta.n_nodes
        self.lambda_beta = float(lambda_beta)
        self.lambda_arith = float(lambda_arith)
        self.lambda_if = float(lambda_if)
        self.lambda_fix = float(lambda_fix)
        self.terms: list[MeraEvalTerm] = self._enumerate_terms()
        self._terms_set = frozenset(self.terms)

    def _enumerate_terms(self):
        terms = []
        for rule in _ALL_RULES:
            for n in range(self.N_nodes):
                terms.append(MeraEvalTerm(rule, n, 2))
        return terms

    def _lambda_for(self, rule):
        if rule == RULE_R_BETA:
            return self.lambda_beta
        if rule in (RULE_R_ARITH, RULE_R_CMP, RULE_R_SUCC):
            return self.lambda_arith
        if rule == RULE_R_IF:
            return self.lambda_if
        if rule == RULE_R_FIX:
            return self.lambda_fix
        raise MeraEvalTermNotFound(rule)

    # ---- redex-use bookkeeping (addressing only, spec §1.2) -------------

    def _fix_recursion_uses(self, fix_node: int) -> list[int]:
        """Var nodes whose binder is this FIX node — recursion-use sites.

        Read via use_to_binder addressing (spec §1.2); the FIX/use
        correlation itself rides the encoded MERA tree.
        """
        meta = self.meta
        fix_bid = meta.layout.leaf_of(fix_node, "bid")
        uses = []
        for use_bid, binder_bid in meta.use_to_binder.items():
            if binder_bid == fix_bid:
                use_node = (use_bid - SPECIES_LEAF_OFFSET["bid"]) \
                    // LEAVES_PER_NODE
                uses.append(use_node)
        return sorted(uses)

    def _penalty_ops(self, term: MeraEvalTerm):
        """The factored diagonal penalty operator(s) for a term, as a list
        of dict[leaf -> (16,16)]. Empty if the term's kind does not match
        (structural enumeration, spec §1.7)."""
        meta = self.meta
        node = term.node
        lam = self._lambda_for(term.rule_id)
        kids = meta.children_of_node.get(node, [])
        if term.rule_id == RULE_R_BETA:
            if len(kids) < 1:
                return []
            return [beta_penalty_ops(_kind_leaf(meta, node),
                                     _kind_leaf(meta, kids[0]), lam)]
        if term.rule_id in (RULE_R_ARITH, RULE_R_CMP):
            if len(kids) < 2:
                return []
            builder = (arith_penalty_ops if term.rule_id == RULE_R_ARITH
                       else cmp_penalty_ops)
            return [builder(_kind_leaf(meta, node), _value_leaf(meta, node),
                            _kind_leaf(meta, kids[0]),
                            _kind_leaf(meta, kids[1]), lam)]
        if term.rule_id == RULE_R_IF:
            if len(kids) < 1:
                return []
            return [if_penalty_ops(_kind_leaf(meta, node),
                                   _kind_leaf(meta, kids[0]), lam)]
        if term.rule_id == RULE_R_SUCC:
            if len(kids) < 1:
                return []
            return [succ_penalty_ops(_kind_leaf(meta, node),
                                     _kind_leaf(meta, kids[0]), lam)]
        if term.rule_id == RULE_R_FIX:
            ops_list = []
            for use in self._fix_recursion_uses(node):
                ops_list.append(fix_penalty_ops(_kind_leaf(meta, node),
                                                _kind_leaf(meta, use), lam))
            return ops_list
        raise MeraEvalTermNotFound(term)

    def term_energy(self, state: MERA, term: MeraEvalTerm) -> float:
        if term not in self._terms_set:
            raise MeraEvalTermNotFound(term)
        total = 0.0
        for ops in self._penalty_ops(term):
            total += float(mera_window_expectation_factored(state, ops).real)
        return total

    def total_energy(self, state: MERA) -> float:
        return sum(self.term_energy(state, t) for t in self.terms)

    def residuals(self, state: MERA) -> dict:
        return {(t.rule_id, t.node): self.term_energy(state, t)
                for t in self.terms}

    # ---- transition gates for imaginary-time evolution (spec §7.4) ------

    def term_gates(self, state: MERA, term: MeraEvalTerm, dt: float,
                   imaginary: bool = True):
        """Factored transition gates for one redex term, conditioned on
        the redex being present on `state`.

        Returns a list of ((leaf,), gate) single-leaf gates that swing
        amplitude from each differing leaf's unreduced basis value toward
        its reduced value. If the redex is not present (penalty energy
        ~0) no gate is returned — off-redex leaves are never touched
        (spec §7.4). Reduced leaf values are a fixed function of the
        concrete operand values, read via leaf argmax (the program is a
        product MERA during evaluation).
        """
        if term not in self._terms_set:
            raise MeraEvalTermNotFound(term)
        # Only emit gates if this redex is actually present.
        if self.term_energy(state, term) < 1e-9:
            return []
        meta = self.meta
        node = term.node
        lam = self._lambda_for(term.rule_id)
        kids = meta.children_of_node.get(node, [])
        # Each entry: (leaf, unreduced_idx, reduced_idx).
        moves: list[tuple[int, int, int]] = []
        if term.rule_id in (RULE_R_ARITH, RULE_R_CMP):
            lhs, rhs = kids[0], kids[1]
            # Read operator / operand values STABLY from the unreduced
            # configuration: a plain argmax drifts once the reduction
            # rotation passes 50%, which would re-target the gate at a
            # wrong (or out-of-range) result and silently abort it.
            if term.rule_id == RULE_R_ARITH:
                _op_set = (VALUE_PLUS, VALUE_MINUS, VALUE_TIMES)
            else:
                _op_set = (VALUE_LT, VALUE_EQ)
            op_v = _leaf_argmax_in(state, _value_leaf(meta, node), _op_set)
            # Operand value leaves reduce toward slot 0; the unreduced
            # literal is the dominant NON-zero slot.
            _lit_slots = range(1, MERA_LEAF_DIM)
            a_v = _leaf_argmax_in(state, _value_leaf(meta, lhs), _lit_slots)
            b_v = _leaf_argmax_in(state, _value_leaf(meta, rhs), _lit_slots)
            if term.rule_id == RULE_R_ARITH:
                res = arith_result_value_idx(op_v, a_v, b_v)
                res_kind = KIND_INT
            else:
                res = cmp_result_value_idx(op_v, a_v, b_v)
                res_kind = None  # cmp result is a BOOL node
            if res is None:
                return []
            # BIN node -> result literal: kind, value.
            bin_kind = _leaf_argmax(state, _kind_leaf(meta, node))
            if term.rule_id == RULE_R_ARITH:
                moves.append((_kind_leaf(meta, node), bin_kind, KIND_INT))
            else:
                from .mera_encoding import KIND_BOOL
                moves.append((_kind_leaf(meta, node), bin_kind, KIND_BOOL))
            moves.append((_value_leaf(meta, node), op_v, res))
            # operand nodes -> PAD.
            for child in (lhs, rhs):
                ck = _leaf_argmax(state, _kind_leaf(meta, child))
                cv = _leaf_argmax(state, _value_leaf(meta, child))
                moves.append((_kind_leaf(meta, child), ck, KIND_PAD))
                if cv != 0:
                    moves.append((_value_leaf(meta, child), cv, 0))
        elif term.rule_id == RULE_R_SUCC:
            arg = kids[0]
            arg_v = _leaf_argmax(state, _value_leaf(meta, arg))
            # Succ (NatLit n) -> NatLit (n+1): SUCC node becomes NATLIT.
            succ_kind = _leaf_argmax(state, _kind_leaf(meta, node))
            moves.append((_kind_leaf(meta, node), succ_kind, KIND_NATLIT))
            moves.append((_value_leaf(meta, node), 0, arg_v + 1))
            ck = _leaf_argmax(state, _kind_leaf(meta, arg))
            moves.append((_kind_leaf(meta, arg), ck, KIND_PAD))
            if arg_v != 0:
                moves.append((_value_leaf(meta, arg), arg_v, 0))
        elif term.rule_id == RULE_R_IF:
            # if-redex transition handled by Task 13 calibration; the
            # diagonal penalty + the typing constraint suffice for the
            # E3 acceptance, see _if_branch_moves.
            moves = self._if_branch_moves(state, node, kids)
        elif term.rule_id == RULE_R_BETA:
            moves = self._beta_moves(state, node, kids)
        elif term.rule_id == RULE_R_FIX:
            moves, fix_pair_gates = self._fix_moves(state, node, dt, lam)
        gates = []
        for leaf, u, r in moves:
            # The drive is gradual: a single application of the
            # u->r gate drains only a factor (1 - e^{-dt.lam}) of the
            # unreduced weight, so over a step or two the leaf becomes a
            # genuine {u, r} superposition. An argmax-derived `u` flips to
            # `r` once the reduced weight passes 50%, which would halt the
            # rotation half-completed (spec §7.4 / M2: the drive must run
            # until the leaf is FULLY |r>). So `u` here is the dominant
            # NON-r basis index -- the still-unreduced configuration --
            # and the gate is emitted while any unreduced residual remains
            # (|<u|psi>|^2 > eps), not while argmax != r. The gate leaves
            # |r> fixed, so emitting it once the leaf is essentially |r>
            # is a harmless near-identity.
            # NOTE: the `u` carried in the move is argmax-derived and is
            # unreliable once the leaf has rotated past the 50% mark
            # (argmax flips to `r`, making the move look like a no-op).
            # Only `r` is trusted; `u_eff` is recomputed from the leaf's
            # actual weights as the dominant NON-r index.
            w = _leaf_weights(state, leaf)
            w_masked = w.copy()
            w_masked[r] = -1.0
            u_eff = int(np.argmax(w_masked))
            if u_eff == r or w[u_eff] <= 1e-9:
                continue
            g = single_leaf_transition_gate(u_eff, r, dt, lam)
            gates.append(((leaf,), g))
        if term.rule_id == RULE_R_FIX:
            gates.extend(fix_pair_gates)
        return gates

    def _if_branch_moves(self, state, node, kids):
        """If (BoolLit c) t e -> t (if c) or e (if not c). The selected
        branch's root node is promoted into the IF node; the other branch
        and the cond collapse to PAD."""
        if len(kids) < 3:
            return []
        cond, then_b, else_b = kids
        cond_v = _leaf_argmax(state, _value_leaf(meta=self.meta, leaf=None)) \
            if False else _leaf_argmax(state,
                                       self.meta.layout.leaf_of(cond, "value"))
        keep = then_b if cond_v == VALUE_TRUE else else_b
        drop = else_b if cond_v == VALUE_TRUE else then_b
        meta = self.meta
        moves = []
        # Promote the kept branch's 5 leaves into the IF node's 5 leaves.
        for sp in ("kind", "type", "bid", "value", "tobl"):
            if_leaf = meta.layout.leaf_of(node, sp)
            keep_leaf = meta.layout.leaf_of(keep, sp)
            if_cur = _leaf_argmax(state, if_leaf)
            keep_cur = _leaf_argmax(state, keep_leaf)
            moves.append((if_leaf, if_cur, keep_cur))
        # Collapse the cond, the kept branch's old leaves, and the dropped
        # branch to PAD.
        for collapse in (cond, keep, drop):
            for sp in ("kind", "type", "bid", "value", "tobl"):
                leaf = meta.layout.leaf_of(collapse, sp)
                cur = _leaf_argmax(state, leaf)
                tgt = KIND_PAD if sp == "kind" else 0
                moves.append((leaf, cur, tgt))
        return moves

    def _beta_moves(self, state, node, kids):
        """(\\x. body) arg -> body[x:=arg]. For the M2-supported demos the
        body is small; the beta transition promotes the LAM body into the
        APP node and routes the arg's value into the recursion-use Var.

        For E2 / E4 the body is `x + 1` style: the Var-use leaf takes the
        arg's value. We realize beta by: APP node <- body node's leaves,
        the bound Var-use leaves <- the arg node's leaves, LAM + arg + old
        body nodes -> PAD.
        """
        meta = self.meta
        if len(kids) < 2:
            return []
        fn, arg = kids[0], kids[1]
        fn_kids = meta.children_of_node.get(fn, [])
        if not fn_kids:
            return []
        body = fn_kids[0]
        moves = []
        # Promote body's leaves into the APP node.
        for sp in ("kind", "type", "bid", "value", "tobl"):
            app_leaf = meta.layout.leaf_of(node, sp)
            body_leaf = meta.layout.leaf_of(body, sp)
            moves.append((app_leaf, _leaf_argmax(state, app_leaf),
                          _leaf_argmax(state, body_leaf)))
        # Route the arg's value into each recursion-use Var of the LAM.
        fn_bid = meta.layout.leaf_of(fn, "bid")
        arg_kind = _leaf_argmax(state, meta.layout.leaf_of(arg, "kind"))
        arg_val = _leaf_argmax(state, meta.layout.leaf_of(arg, "value"))
        arg_type = _leaf_argmax(state, meta.layout.leaf_of(arg, "type"))
        for use_bid, binder_bid in meta.use_to_binder.items():
            if binder_bid != fn_bid:
                continue
            use_node = (use_bid - SPECIES_LEAF_OFFSET["bid"]) \
                // LEAVES_PER_NODE
            uk = meta.layout.leaf_of(use_node, "kind")
            uv = meta.layout.leaf_of(use_node, "value")
            ut = meta.layout.leaf_of(use_node, "type")
            moves.append((uk, _leaf_argmax(state, uk), arg_kind))
            moves.append((uv, _leaf_argmax(state, uv), arg_val))
            moves.append((ut, _leaf_argmax(state, ut), arg_type))
        # Collapse the LAM, arg, and old body nodes to PAD.
        for collapse in (fn, arg, body):
            for sp in ("kind", "type", "bid", "value", "tobl"):
                leaf = meta.layout.leaf_of(collapse, sp)
                cur = _leaf_argmax(state, leaf)
                tgt = KIND_PAD if sp == "kind" else 0
                moves.append((leaf, cur, tgt))
        return moves

    def _fix_moves(self, state, node, dt, lam):
        """Fix f. body -> body[f := Fix f. body], one tree layer at a time
        (spec §7.2).

        Two coupled effects, both genuine transitions (no classical
        rewrite, spec §1.6):

        1. One-layer unfold: the FIX node's leaves are promoted toward the
           body head — the FIX node becomes the body's head node (for the
           recursive demo, a LAM) so the enclosing APP can then beta-reduce.
        2. Recursion-use coupling: every recursion-use Var node bound to
           this FIX (found by use_to_binder addressing, spec §1.2) receives
           the SAME body head — `f` inside the body is replaced by a fresh
           copy of `Fix f. body`. This is what makes the unfold recursive:
           after the substitution the inner `f` use can unfold again.

        The FIX `kind` leaf and each recursion-use `kind` leaf are coupled
        by a single 256x256 two-leaf `fix_transition_gate` (the §1.1
        binding-as-entanglement anchor); the remaining body-head leaves
        (type/bid/value/tobl) are propagated by factored single-leaf gates,
        exactly mirroring `_beta_moves`' head-promotion pattern.

        Returns (single_leaf_moves, two_leaf_gates). Raises
        MeraEvalBudgetExceeded if an unfold would need to replicate body
        structure into a use site that the encoder reserved no node budget
        for (over-budget unfold — spec §7.2, §8.2: raise, do not truncate).
        """
        meta = self.meta
        kids = meta.children_of_node.get(node, [])
        if not kids:
            return [], []
        body = kids[0]
        body_kind = _leaf_argmax(state, _kind_leaf(meta, body))
        fix_kind = _leaf_argmax(state, _kind_leaf(meta, node))
        uses = self._fix_recursion_uses(node)

        # ---- budget guard (spec §7.2 / §8.2) ----------------------------
        # A one-layer unfold propagates the body HEAD (one node's 5 leaves)
        # into the FIX node and into each recursion-use Var node — that is
        # always representable (a use node owns its own 5 leaves). The
        # over-budget case is structural: the encoder packed the program
        # into `n_nodes` of a fixed `n_nodes_max` budget, and a recursive
        # unfold conceptually appends a body copy. M2 propagates head
        # layers in place rather than growing the tree, but if the program
        # already fills the entire encoder node budget AND the body head
        # is a compound node (so further unfolds would need fresh nodes to
        # represent the replicated subtree beneath a use site), the next
        # unfold has nowhere to materialize — an over-budget configuration.
        # Raise rather than silently truncating structure (spec §8.2).
        body_children = meta.children_of_node.get(body, [])
        n_nodes_max = getattr(meta, "n_nodes_max", None)
        if (uses and body_children and n_nodes_max is not None
                and meta.n_nodes >= n_nodes_max):
            raise MeraEvalBudgetExceeded(
                f"R-Fix unfold at node {node}: program fills the encoder "
                f"node budget ({meta.n_nodes}/{n_nodes_max}) and the body "
                f"head is compound — a further unfold has no node budget "
                f"to materialize the replicated subtree (over-budget "
                f"unfold; spec §7.2, §8.2)")

        moves: list[tuple[int, int, int]] = []
        pair_gates = []

        # (1) Promote body head into the FIX node. The `kind` leaf is
        #     handled jointly with the use coupling below; the rest are
        #     factored single-leaf moves.
        for sp in ("type", "bid", "value", "tobl"):
            fix_leaf = meta.layout.leaf_of(node, sp)
            body_leaf = meta.layout.leaf_of(body, sp)
            moves.append((fix_leaf, _leaf_argmax(state, fix_leaf),
                          _leaf_argmax(state, body_leaf)))
        # Collapse the (now-promoted) body node's leaves to PAD.
        for sp in ("kind", "type", "bid", "value", "tobl"):
            leaf = meta.layout.leaf_of(body, sp)
            cur = _leaf_argmax(state, leaf)
            tgt = KIND_PAD if sp == "kind" else 0
            moves.append((leaf, cur, tgt))

        # (2) Recursion-use coupling. Each recursion-use Var receives the
        #     body head. The kind leaves of the FIX node and the use node
        #     co-rotate through a single 256x256 two-leaf gate.
        if not uses:
            # No recursion use: the FIX `kind` still unfolds toward the
            # body head on its own (single-leaf), mirroring effect (1).
            fk = _kind_leaf(meta, node)
            moves.append((fk, fix_kind, body_kind))
        for use in uses:
            use_kind_leaf = _kind_leaf(meta, use)
            use_kind = _leaf_argmax(state, use_kind_leaf)
            fk = _kind_leaf(meta, node)
            # Joint FIX-kind <-> use-kind unfold: both swing toward the
            # body head together. 256x256 two-leaf gate (spec §7.4).
            g = fix_transition_gate(fix_kind, body_kind,
                                    use_kind, body_kind, dt, lam)
            pair_gates.append(((fk, use_kind_leaf), g))
            # Remaining body-head leaves into the use node (factored).
            for sp in ("type", "bid", "value", "tobl"):
                use_leaf = meta.layout.leaf_of(use, sp)
                body_leaf = meta.layout.leaf_of(body, sp)
                moves.append((use_leaf, _leaf_argmax(state, use_leaf),
                              _leaf_argmax(state, body_leaf)))
        return moves, pair_gates
