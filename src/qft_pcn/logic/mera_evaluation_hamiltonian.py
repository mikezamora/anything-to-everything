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
        # Per-IF-node snapshot of the kept branch's 5 leaf values, captured
        # the first time R-If fires for that node (while the kept branch's
        # leaves are still pristine). See _if_branch_moves for why a live
        # read is unsafe once the collapse drain starts.
        self._if_keep_targets: dict[int, dict[str, int]] = {}
        # Per-APP-node snapshot of the beta body head's 5 leaf values,
        # captured when the staged body subtree first reaches normal form.
        # The body head is the promotion TARGET for the APP node; reading
        # it live is unsafe once the body node's own collapse-to-PAD drain
        # begins (the target would flip toward PAD). See _beta_moves.
        self._beta_body_targets: dict[int, dict[str, int]] = {}

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

    def _collapse_moves(self, state, node) -> list:
        """Moves that collapse a spent node to PAD: the KIND leaf only.

        Every typing- and eval-Hamiltonian penalty on a node is a factored
        product gated by a P[kind=X] projector on that node's kind leaf,
        and the decoder classifies a node as absent purely from its kind
        leaf (KIND_PAD). So once a node's kind leaf is PAD the node is
        structurally absent and contributes ZERO energy regardless of its
        type/bid/value/tobl leaves -- those leaves are dead weight.

        Earlier this method ALSO drained type/bid/value/tobl toward 0
        after the kind leaf was "essentially" PAD (weight > 0.999). That
        drain was the sole cause of a post-convergence <H> bump: while a
        sliver of kind=INT weight remains (up to 1e-3 at the 0.999 gate),
        draining the type leaf away from TYPE_INT makes the node read
        transiently as P[kind=INT].(I-P[type=INT]) -- a tiny `T-Lit-Int`
        energy. With several nodes collapsing in the same step (e.g.
        E3's `if 1<2 then 7 else 0` collapses 4 IntLits at once) those
        transients SUM and break strict monotonicity. The drain serves no
        decode or energy purpose, so it is removed: collapse drives only
        the kind leaf to PAD. The collapse is then exactly energy-monotone
        (spec §7.4 / §13.1)."""
        meta = self.meta
        kind_leaf = meta.layout.leaf_of(node, "kind")
        return [(kind_leaf, _leaf_argmax(state, kind_leaf), KIND_PAD)]

    def _if_cleanup_unfinished(self, state: MERA, term: MeraEvalTerm) -> bool:
        """True if `term` is an R-If redex that has FIRED (the IF node is
        no longer KIND_IF, so its diagonal penalty already reads 0) but
        whose post-reduction cleanup is not finished: the cond subtree or
        the dropped branch's subtree still hold non-PAD nodes.

        Mirrors `_arith_bin_unfinished`: the R-If penalty is gated on the
        IF node being KIND_IF, so once branch selection promotes the IF
        node to its result kind the penalty product reads 0 and a plain
        `penalty < 1e-9` guard would stop R-If's gates -- stranding the
        spent cond/drop subtrees half-collapsed (an ill-typed partial
        node injects typing energy: the post-convergence <H> bump). So
        R-If stays live until those subtrees are fully PAD (spec §7.4:
        drive to FULL reduction)."""
        if term.rule_id != RULE_R_IF:
            return False
        meta = self.meta
        node = term.node
        if node not in self._if_keep_targets:
            return False  # R-If has not fired for this IF node yet.
        snap = self._if_keep_targets[node]
        cond, keep, drop = snap["cond"], snap["keep"], snap["drop"]
        # Any spent node whose kind leaf is not essentially PAD means the
        # cleanup collapse still has work to do.
        spent = list(self._subtree_nodes(cond))
        spent += list(self._subtree_nodes(drop))
        spent.append(keep)
        for cnode in spent:
            w = _leaf_weights(state, meta.layout.leaf_of(cnode, "kind"))
            if w[KIND_PAD] < 0.999999:
                return True
        return False

    def _arith_bin_unfinished(self, state: MERA, term: MeraEvalTerm) -> bool:
        """True if `term` is an arith/cmp redex whose BIN node has begun
        reducing but has NOT yet fully reached its result kind.

        Keeps the redex "live" past the point where the factored penalty
        product reads 0 because the operands have already collapsed
        (spec §7.4: drive to FULL reduction; avoids the staged plateau)."""
        if term.rule_id not in (RULE_R_ARITH, RULE_R_CMP):
            return False
        meta = self.meta
        node = term.node
        kids = meta.children_of_node.get(node, [])
        if len(kids) < 2:
            return False
        from .mera_encoding import KIND_BOOL
        res_kind = KIND_INT if term.rule_id == RULE_R_ARITH else KIND_BOOL
        w_kind = _leaf_weights(state, _kind_leaf(meta, node))
        w_bin = float(w_kind[KIND_BIN]) if KIND_BIN < w_kind.shape[0] else 0.0
        w_res = float(w_kind[res_kind]) if res_kind < w_kind.shape[0] else 0.0
        # If this BIN node is being collapsed toward PAD, a PARENT
        # reduction is consuming it (e.g. R-If collapsing the cond subtree
        # after `1 < 2` was used to select a branch). Keeping the arith/cmp
        # gate "live" here would fight the parent's collapse -- the BIN
        # node's kind leaf is tugged toward KIND_BOOL by R-Cmp and toward
        # KIND_PAD by R-If at once, and <H> oscillates upward. So once the
        # node has appreciable PAD weight, this rule stands down.
        w_pad = float(w_kind[KIND_PAD]) if KIND_PAD < w_kind.shape[0] else 0.0
        if w_pad > 1e-6:
            return False
        # Reduction in progress: some BIN weight has already moved toward
        # the result kind, but the result is not yet fully resolved.
        return w_res > 1e-6 and w_res < 0.999999 and w_bin < 0.999999

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
        #
        # The diagonal penalty is a FACTORED PRODUCT of per-leaf
        # projectors. For an arith/cmp redex the operands are factors:
        # once the reduction starts collapsing the operand nodes toward
        # PAD, an operand factor hits 0 and the whole product reads ~0 —
        # even though the BIN node itself has NOT yet finished rotating to
        # its result. A plain `penalty < 1e-9` guard would then stop the
        # gate mid-reduction, freezing the BIN node in a {PAD,INT,BIN}
        # superposition (the staged-reduction plateau). So for arith/cmp
        # the redex stays "live" while the BIN node still differs from its
        # computed result, regardless of operand collapse (spec §7.4: the
        # drive must run until the leaf is FULLY reduced).
        if self.term_energy(state, term) < 1e-9:
            if (not self._arith_bin_unfinished(state, term)
                    and not self._if_cleanup_unfinished(state, term)):
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
                res_kind = KIND_INT
            else:
                from .mera_encoding import KIND_BOOL
                res_kind = KIND_BOOL
            moves.append((_kind_leaf(meta, node), bin_kind, res_kind))
            moves.append((_value_leaf(meta, node), op_v, res))
            # Operand nodes -> PAD, SEQUENCED after the BIN node itself
            # has fully reached its result. The diagonal penalty is a
            # factored product over {BIN node, lhs, rhs}; collapsing an
            # operand toward PAD drives that operand's projector factor to
            # 0, which zeroes the whole product and would stop the gate
            # before the BIN node finished rotating to the result (the
            # staged plateau: BIN frozen as a {PAD,INT,BIN} mix). So the
            # operand collapse waits until the BIN node's kind AND value
            # leaves are essentially the result — the redex result is then
            # already locked in and collapsing the spent operands cannot
            # abort it (spec §7.4).
            w_bin_kind = _leaf_weights(state, _kind_leaf(meta, node))
            w_bin_val = _leaf_weights(state, _value_leaf(meta, node))
            bin_resolved = (w_bin_kind[res_kind] > 0.999
                            and w_bin_val[res] > 0.999)
            if bin_resolved:
                for child in (lhs, rhs):
                    moves.extend(self._collapse_moves(state, child))
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
        and the cond collapse to PAD.

        Root cause of the historical PAD-at-site-0 decode failure: the
        promote move read its target via a plain argmax of the kept
        branch's leaf, while the collapse loop simultaneously drained that
        same leaf toward PAD. Once the kept leaf rotated past 50% the
        argmax flipped to PAD, so the IF node was driven to PAD instead of
        the branch value -- both the IF node and the kept branch ended
        all-PAD and the decoder hit PAD at site 0.

        Fix: a live read of the kept branch's leaf cannot be trusted once
        the collapse drain starts, and the IF node's own home value (e.g.
        VALUE_NONE on the value leaf, KIND_IF on the kind leaf) ties with
        the kept value in any combined keep+IF read. So the kept branch's
        5 leaf values are SNAPSHOT once -- the first time R-If fires for
        this IF node, while the kept branch is still pristine -- and that
        snapshot is the stable promote target for the rest of the
        evolution. The snapshot is pure leaf-weight addressing (spec
        §1.2); it does not rewrite anything."""
        if len(kids) < 3:
            return []
        meta = self.meta
        # Snapshot, on the first R-If firing for this IF node (cond still a
        # live BOOL, kept branch still pristine): which branch is kept,
        # which is dropped, and the kept branch's 5 leaf values. A later
        # live read is unsafe -- once the cond collapses its value leaf
        # goes to 0 and would flip the keep/drop choice, and once the
        # collapse drain starts the kept leaf values drift toward PAD.
        if node not in self._if_keep_targets:
            cond, then_b, else_b = kids
            # The R-If diagonal penalty P[kind=IF].P[kind=BOOL](cond) is
            # non-zero the instant the cond carries ANY BOOL-kind weight --
            # before R-Cmp has finished resolving `1 < 2` to its truth
            # value. Snapshotting the keep/drop choice then reads a cond
            # `value` leaf still mid-rotation (it may not yet be
            # VALUE_TRUE), which would pick the WRONG branch and promote
            # the wrong literal. So defer the snapshot until the cond is a
            # DECISIVELY resolved BOOL literal: kind essentially KIND_BOOL
            # and the value leaf essentially VALUE_TRUE or VALUE_FALSE.
            from .mera_encoding import KIND_BOOL
            cond_kind_w = _leaf_weights(state,
                                        meta.layout.leaf_of(cond, "kind"))
            cond_val_w = _leaf_weights(state,
                                       meta.layout.leaf_of(cond, "value"))
            if cond_kind_w[KIND_BOOL] < 0.99:
                return []
            if (cond_val_w[VALUE_TRUE] < 0.99
                    and cond_val_w[VALUE_FALSE] < 0.99):
                return []
            cond_v = (VALUE_TRUE if cond_val_w[VALUE_TRUE]
                      >= cond_val_w[VALUE_FALSE] else VALUE_FALSE)
            keep = then_b if cond_v == VALUE_TRUE else else_b
            drop = else_b if cond_v == VALUE_TRUE else then_b
            self._if_keep_targets[node] = {
                "keep": keep, "drop": drop, "cond": cond,
                "leaves": {
                    sp: _leaf_argmax(state, meta.layout.leaf_of(keep, sp))
                    for sp in ("kind", "type", "bid", "value", "tobl")
                },
            }
        snap = self._if_keep_targets[node]
        cond, keep, drop = snap["cond"], snap["keep"], snap["drop"]
        keep_targets = snap["leaves"]
        moves = []
        # Promote the kept branch's 5 leaves into the IF node's 5 leaves,
        # toward the stable snapshot target. All 5 leaves are driven in
        # lockstep: the R-If penalty is gated on the cond still being a
        # live BOOL, so once the cond collapses R-If stops emitting --
        # holding any leaf back would strand the IF node partially
        # promoted at high energy. The promoted IF node is an IntLit whose
        # 5 leaf values are mutually consistent, so lockstep promotion
        # keeps it well-typed.
        for sp in ("kind", "type", "bid", "value", "tobl"):
            if_leaf = meta.layout.leaf_of(node, sp)
            if_cur = _leaf_argmax(state, if_leaf)
            moves.append((if_leaf, if_cur, keep_targets[sp]))
        # Collapse the spent nodes to PAD: the cond's ENTIRE subtree, the
        # dropped branch's ENTIRE subtree, and the kept branch's now-spent
        # root slot (its 5 leaves were promoted into the IF node).
        #
        # Two historical bugs fixed here:
        #  (a) `site N not PAD after AST parse` -- the previous code
        #      collapsed only the cond ROOT node, leaving the cond's
        #      operand subnodes (e.g. the `1` and `2` of `1 < 2`) as
        #      orphaned non-PAD sites the decoder then rejected. R-Cmp's
        #      own operand collapse is gated on the BIN node resolving to
        #      KIND_BOOL, which never happens once R-If drives that same
        #      node to PAD -- so R-If must collapse the whole cond subtree
        #      itself. Same for a multi-node dropped branch.
        #  (b) energy rising mid-reduction -- a concurrent 5-leaf collapse
        #      passes through (kind=X, type drained) configurations that
        #      switch typing penalties on. `_collapse_moves` sequences the
        #      collapse kind-first, which is energy-neutral / monotone.
        collapse_nodes = list(self._subtree_nodes(cond))
        collapse_nodes += list(self._subtree_nodes(drop))
        collapse_nodes.append(keep)
        for cnode in collapse_nodes:
            moves.extend(self._collapse_moves(state, cnode))
        return moves

    def _subtree_nodes(self, root: int) -> list[int]:
        """All node indices in the subtree rooted at `root` (root included),
        via the encoded tree topology (children_of_node — addressing only,
        spec §1.2)."""
        meta = self.meta
        out = []
        stack = [root]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(meta.children_of_node.get(n, []))
        return out

    def _subtree_eval_energy(self, state, root: int) -> float:
        """Total eval-redex penalty for every node strictly inside the
        subtree rooted at `root`. Used to detect whether a beta body still
        contains an unreduced inner redex: a non-zero value means the inner
        (staged) redex has not yet fired."""
        nodes = set(self._subtree_nodes(root))
        total = 0.0
        for t in self.terms:
            if t.node in nodes:
                total += self.term_energy(state, t)
        return total

    def _beta_moves(self, state, node, kids):
        """(\\x. body) arg -> body[x:=arg]. For the M2-supported demos the
        body is small; the beta transition promotes the LAM body into the
        APP node and routes the arg's value into the recursion-use Var.

        For E2 / E4 the body is `x + 1` style: the Var-use leaf takes the
        arg's value. We realize beta by: APP node <- body node's leaves,
        the bound Var-use leaves <- the arg node's leaves, LAM + arg + old
        body nodes -> PAD.

        STAGED REDUCTION (spec §7.4; E2/E4). Beta has two effects with a
        genuine data dependency: (A) routing the arg value into each
        recursion-use Var, which may FORM a new redex inside the body
        (e.g. `x+1` becomes `2+1`); and (C) promoting the body head into
        the APP node and collapsing the spent LAM/arg/body nodes to PAD.
        If (C) ran while (A)'s newly-formed inner redex was still mid-
        reduction, the body-promotion gate would chase the inner redex's
        transient (superposed, argmax-flipping) leaves and inject energy
        into the APP node — a non-monotone <H> rise and a frozen 3-way
        leaf. So (C) is SEQUENCED after the inner redex: the body head is
        promoted only once the body subtree is itself in normal form
        (no active eval redex inside it). This makes staged beta->arith
        reduction step-wise monotone — the inner redex relaxes to 0 first,
        then the outer promotion fires into now-stable leaves.
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

        # ---- Stage (A): route the arg into every recursion-use Var ------
        # This is the substitution x:=arg. It runs FIRST and unconditionally
        # while the beta redex is present; routing the arg value into the
        # body's use sites is what FORMS the staged inner redex.
        fn_bid = meta.layout.leaf_of(fn, "bid")
        arg_kind = _leaf_argmax(state, meta.layout.leaf_of(arg, "kind"))
        arg_val = _leaf_argmax(state, meta.layout.leaf_of(arg, "value"))
        arg_type = _leaf_argmax(state, meta.layout.leaf_of(arg, "type"))
        routing_done = True
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
            # Routing is "past" once no use leaf is still VAR-DOMINANT: a
            # use is either already rewritten to the arg's kind or has
            # since been consumed by the staged inner redex (collapsed to
            # PAD). Use the argmax, not a weight threshold: the use kind
            # leaf's dominant index flips monotonically VAR -> arg-kind ->
            # PAD, whereas a tiny residual VAR *weight* can grow as a
            # FRACTION once the inner redex drains the leaf toward PAD and
            # the leaf is renormalized — which would spuriously re-flag
            # routing as in-flight and stall the outer promotion.
            from .mera_encoding import KIND_VAR as _KV
            if _leaf_argmax(state, uk) == _KV:
                routing_done = False

        # ---- Stage (C): promote the body head into the APP node ---------
        # SEQUENCED: only once (A) is complete AND the body subtree holds
        # no active eval redex (the staged inner redex has relaxed to its
        # normal form). Promoting earlier would chase the inner redex's
        # transient leaves and inject energy into the APP node.
        inner_energy = self._subtree_eval_energy(state, body)
        if not (routing_done and inner_energy < 1e-6):
            return moves

        # Snapshot the body head the FIRST time it is in normal form.
        # The promotion target must be read here, while the body node's
        # leaves are still pristine: Stage (C) also collapses the body
        # node to PAD, so a live re-read of body_leaf would flip the APP
        # node's target toward PAD mid-promotion (the staged plateau).
        if node not in self._beta_body_targets:
            self._beta_body_targets[node] = {
                sp: _leaf_argmax(state, meta.layout.leaf_of(body, sp))
                for sp in ("kind", "type", "bid", "value", "tobl")
            }
        target = self._beta_body_targets[node]

        # Promote the body head into the APP node toward the snapshot.
        app_resolved = True
        for sp in ("kind", "type", "bid", "value", "tobl"):
            app_leaf = meta.layout.leaf_of(node, sp)
            r = target[sp]
            moves.append((app_leaf, _leaf_argmax(state, app_leaf), r))
            if _leaf_weights(state, app_leaf)[r] < 0.999:
                app_resolved = False

        # Collapse the spent LAM, arg and old body nodes to PAD — only
        # once the APP node has fully received the body head. Collapsing
        # the body node earlier would corrupt the very leaves the APP
        # node is still copying from. kind-first (see _collapse_moves).
        if app_resolved:
            for collapse in (fn, arg, body):
                moves.extend(self._collapse_moves(state, collapse))
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
