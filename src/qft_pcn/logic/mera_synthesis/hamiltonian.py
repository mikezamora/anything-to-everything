"""Synthesis Hamiltonian: H_examples + H_target_type + H_size, composed
with M2's H_typing and H_eval (spec §4).

Every term is a factored per-leaf operator measured via
mera_window_expectation_factored — NO dense 16**k operator (principle
§1.3, load-bearing). All operators here are 1-leaf (16, 16) projectors
or their complements.

Composition strategy (option b in the M3 plan): a thin local wrapper
``ComposedMeraSynthesisHamiltonian`` holds the weighted M2
sub-Hamiltonians plus the extra synth terms. This leaves M2's
``compose_mera_hamiltonians`` (currently ``*hamiltonians``, no weights
and no ``extra_terms``) untouched — keeping M2's stable surface intact.
Extending compose_mera_hamiltonians to accept (ham, weight) tuples and
``extra_terms=`` would touch a committed M2 surface and risks breaking
existing callers; the wrapper is local to M3 and exposes the same
``.terms / .term_energy / .residuals / .total_energy`` contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mera_encoder import MeraEncodingMeta
from ..mera_encoding import (
    MERA_LEAF_DIM, SPECIES_LEAF_OFFSET,
)
from ..encoding import (
    KIND_PAD, INT_LIT_OFFSET,
)
from .._mera_window import mera_window_expectation_factored
from ..ast import Ty, IntLit, BoolLit
from .._types import ty_to_tag


def _pad_complement_op() -> np.ndarray:
    """I - |KIND_PAD><KIND_PAD| on a kind leaf."""
    op = np.eye(MERA_LEAF_DIM, dtype=complex)
    op[KIND_PAD, KIND_PAD] = 0.0
    return op


def _basis_complement_op(index: int) -> np.ndarray:
    """I - |index><index| on one leaf."""
    if not 0 <= index < MERA_LEAF_DIM:
        raise ValueError(
            f"basis index {index} out of leaf range [0,{MERA_LEAF_DIM})")
    op = np.eye(MERA_LEAF_DIM, dtype=complex)
    op[index, index] = 0.0
    return op


def _value_index_of(node) -> int:
    """Map a literal output AST node to its `value`-leaf basis index.

    Mirrors the encoder's leaf encoding (see ``_local_kind_type_value``
    in ``_tensors.py``): IntLit n -> n + INT_LIT_OFFSET; BoolLit
    -> VALUE_TRUE / VALUE_FALSE.
    """
    if isinstance(node, IntLit):
        return node.val + INT_LIT_OFFSET
    if isinstance(node, BoolLit):
        # VALUE_FALSE = 0, VALUE_TRUE = 1 in encoding.py.
        return 1 if node.val else 0
    raise TypeError(
        f"unsupported example.output type: {type(node).__name__} "
        f"(only IntLit and BoolLit are supported by X-Example)")


def _type_tag_of(ty: Ty) -> int:
    """Reuse M1's ty_to_tag — the flat tag is what the `type` leaf stores."""
    tag, _nested = ty_to_tag(ty)
    return tag


def _witness_root_node(meta: MeraEncodingMeta, i: int) -> int:
    """Node index of the i-th witness sub-tree's root (the first node in
    meta.witness_node_ranges[i])."""
    ranges = getattr(meta, "witness_node_ranges", [])
    if not ranges or i >= len(ranges):
        raise ValueError(
            f"meta.witness_node_ranges has no entry {i}; "
            f"got {ranges!r}")
    return int(ranges[i][0])


@dataclass
class _FactoredTerm:
    """NamedMeraTerm whose expectation is one factored per-leaf op."""
    name: str
    rule_class: str
    node: int
    leaves: tuple
    weight: float
    leaf_ops: dict          # {absolute_leaf: (16,16) op}

    def expectation(self, state) -> float:
        val = mera_window_expectation_factored(state, self.leaf_ops)
        if abs(np.imag(val)) > 1e-10:
            raise ValueError(
                f"{self.name}: non-Hermitian term, Im={np.imag(val)}")
        return float(np.real(val)) * self.weight


def build_size_terms(meta: MeraEncodingMeta, w_S: float) -> list:
    """One S-Size term per sketch node: w_S * <I - P_PAD> on its kind leaf
    (spec §4.4). Witness nodes (meta.witness_node_ranges) are excluded —
    only the sketch contributes to the Occam penalty."""
    terms: list = []
    witness: set[int] = set()
    for r in getattr(meta, "witness_node_ranges", []):
        witness.update(int(x) for x in r)
    for node in range(meta.n_nodes):
        if node in witness:
            continue
        kind_leaf = meta.layout.leaf_of(node, "kind")
        terms.append(_FactoredTerm(
            name=f"S-Size@node_{node}", rule_class="S-Size", node=node,
            leaves=(kind_leaf,), weight=w_S,
            leaf_ops={kind_leaf: _pad_complement_op()}))
    return terms


def build_target_type_terms(meta: MeraEncodingMeta, target_type: Ty | None,
                            w_Y: float) -> list:
    """One T-Target term pinning node 0's type leaf to target_type
    (spec §4.3). The penalty is w_Y * <I - |tag><tag|>: zero when the
    root's type leaf is exactly the target tag."""
    if target_type is None:
        return []
    tag = _type_tag_of(target_type)
    type_leaf = meta.layout.leaf_of(0, "type")
    return [_FactoredTerm(
        name="T-Target@node_0", rule_class="T-Target", node=0,
        leaves=(type_leaf,), weight=w_Y,
        leaf_ops={type_leaf: _basis_complement_op(tag)})]


def build_example_terms(meta: MeraEncodingMeta, examples: tuple,
                        w_X: float) -> list:
    """One X-Example term per example: a boundary pin on the witness
    root's value leaf, w_X * <I - |output><output|> (spec §4.2)."""
    terms: list = []
    if not examples:
        return terms
    for i, ex in enumerate(examples):
        wit_root = _witness_root_node(meta, i)
        val_leaf = meta.layout.leaf_of(wit_root, "value")
        out_idx = _value_index_of(ex.output)
        terms.append(_FactoredTerm(
            name=f"X-Example@node_{wit_root}", rule_class="X-Example",
            node=wit_root, leaves=(val_leaf,), weight=w_X,
            leaf_ops={val_leaf: _basis_complement_op(out_idx)}))
    return terms


class ComposedMeraSynthesisHamiltonian:
    """Operator-sum of weighted M2 sub-Hamiltonians plus M3 synth terms.

    Each sub-Hamiltonian's contribution to total_energy is scaled by its
    weight; the extra synth terms (already carry per-term ``weight``)
    are summed unweighted. `.terms` is the concatenation: synth terms
    followed by each sub-Hamiltonian's terms (so per-term iteration
    sees every term exactly once). `.term_energy(state, t)` dispatches:
    synth terms call their own .expectation(state); sub-ham terms call
    the owning Hamiltonian's term_energy, scaled by that ham's weight.
    """

    def __init__(self, weighted_sub_hams: list, extra_terms: list):
        self._weighted = list(weighted_sub_hams)   # list of (ham, weight)
        # Cross-check meta agreement.
        if self._weighted:
            ref = self._weighted[0][0].meta
            for h, _w in self._weighted[1:]:
                if (h.meta.n_leaves != ref.n_leaves
                        or h.meta.n_nodes != ref.n_nodes):
                    raise ValueError(
                        "sub-Hamiltonian meta disagrees on n_leaves/n_nodes")
            self.meta = ref
        else:
            self.meta = None
        self._extra = list(extra_terms)
        # Flat .terms list — synth first, then each sub-ham's terms.
        merged: list = list(self._extra)
        self._owner: dict[int, object] = {}
        for h, _w in self._weighted:
            for t in h.terms:
                merged.append(t)
                self._owner[id(t)] = h
        self.terms = merged

    def term_energy(self, state, term) -> float:
        # Synth term?
        if id(term) in {id(t) for t in self._extra}:
            return term.expectation(state)
        owner = self._owner.get(id(term))
        if owner is None:
            raise KeyError(f"term {term!r} not in this Hamiltonian")
        weight = next(w for h, w in self._weighted if h is owner)
        return weight * owner.term_energy(state, term)

    def total_energy(self, state) -> float:
        total = 0.0
        for t in self._extra:
            total += t.expectation(state)
        for h, w in self._weighted:
            total += w * h.total_energy(state)
        return float(total)

    def term_gates(self, state, term, dt: float, imaginary: bool = True):
        """Factored imaginary-time gates for one term (M2's substrate
        contract).

        For synth terms (single-leaf projector P with weight w): emit
        ``exp(-dt * w * P)`` on the leaf -- a diagonal 16x16 gate that
        damps amplitude on the penalized basis element.

        For sub-Hamiltonian terms: delegate to the owning Hamiltonian
        with ``dt`` scaled by the sub-Hamiltonian's weight so the
        evolution sees ``exp(-w*dt*H_sub_term)``.
        """
        # Synth term path.
        if id(term) in {id(t) for t in self._extra}:
            if not imaginary:
                raise NotImplementedError(
                    "synth-term real-time gates not supported")
            # term.leaf_ops is {leaf: (16,16) op}; for the current single-
            # leaf projectors this is exactly one entry, and the op is a
            # projector (idempotent). Build exp(-dt*w*P) per leaf.
            gates = []
            for leaf, op in term.leaf_ops.items():
                # Diagonal projector — eigenvalues in {0,1}; matrix exp
                # via direct exponentiation is fine at dim 16.
                from scipy.linalg import expm
                gate = expm(-dt * term.weight * op)
                gates.append(((leaf,), gate))
            return gates
        owner = self._owner.get(id(term))
        if owner is None:
            raise KeyError(f"term {term!r} not in this Hamiltonian")
        weight = next(w for h, w in self._weighted if h is owner)
        return owner.term_gates(state, term, dt * weight, imaginary)

    def term_affected_leaves(self, term) -> frozenset:
        """Conservative footprint of leaves the term's gates may read/write.

        Used by ``mera_trotter_step``'s redex-presence cache to skip terms
        whose footprint is disjoint from the leaves touched in the previous
        step. Delegates to the owning sub-Hamiltonian for sub-ham terms,
        and uses the synth term's own ``leaves`` tuple for synth terms
        (synth gates only touch the single leaf they pin)."""
        if id(term) in {id(t) for t in self._extra}:
            return frozenset(term.leaves)
        owner = self._owner.get(id(term))
        if owner is None:
            raise KeyError(f"term {term!r} not in this Hamiltonian")
        return owner.term_affected_leaves(term)

    def residuals(self, state) -> dict:
        out: dict = {}
        # Synth residuals: keyed by term name.
        for t in self._extra:
            out[(t.rule_class, t.node)] = t.expectation(state)
        # Sub-ham residuals: keyed by (ham-class, *origkey).
        for h, w in self._weighted:
            for key, val in h.residuals(state).items():
                k = (type(h).__name__,) + (key if isinstance(key, tuple)
                                            else (key,))
                out[k] = w * val
        return out


def compile_mera_synthesis_hamiltonian(meta: MeraEncodingMeta, problem,
                                       weights):
    """Compose H_typing (M2) + H_eval (M2) + H_examples + H_target_type +
    H_size into one operator-sum Hamiltonian exposing
    .terms / .term_energy / .residuals / .total_energy (spec §4.1)."""
    from ..mera_typing_hamiltonian import MeraTypingHamiltonian
    from ..mera_evaluation_hamiltonian import MeraEvalHamiltonian

    h_typing = MeraTypingHamiltonian(meta)
    h_eval = MeraEvalHamiltonian(meta)
    synth_terms = (
        build_example_terms(meta, problem.examples, weights.w_X)
        + build_target_type_terms(meta, problem.target_type, weights.w_Y)
        + build_size_terms(meta, weights.w_S)
    )
    return ComposedMeraSynthesisHamiltonian(
        weighted_sub_hams=[(h_typing, weights.w_T), (h_eval, weights.w_E)],
        extra_terms=synth_terms,
    )
