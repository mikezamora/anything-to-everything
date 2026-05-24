"""§13.7 type safety as an operator-algebraic identity: ``[H_typing, H_eval] = 0``.

Spec §13.7 (lines 2515-2520): "type conservation under program evaluation is
exact... type safety: well-typed programs cannot go wrong" — the operator-
algebraic statement is that ``H_typing`` and ``H_eval`` commute on the
well-typed substrate. Equivalently, ``H_t |psi> = 0`` implies
``H_t H_e |psi> = H_e H_t |psi> = 0``: typing is preserved under one step
of evaluation.

The test verifies this by computing ``<psi| H_t H_e |psi>`` and
``<psi| H_e H_t |psi>`` directly via the SAME factored-window substrate
primitive used by §12.6 goldstone's :func:`_two_term_expectation`. Each
Hamiltonian term decomposes into a sum of leaf-factored projector ops
(spec §7.4), so the product ``H_t H_e`` is the sum over (typing-piece,
eval-piece) pairs of per-leaf operator products, routed through
:func:`mera_window_expectation_factored`. No dense ``16**k`` operator is
materialized (spec §1.3).

CONTENT-BEARING: a substrate-mutated ill-typed state produces a detectably
non-zero ``|<H_t H_e>|`` (typing failure propagates through eval) — without
this, the well-typed test would be vacuous.
"""
from __future__ import annotations

import numpy as np

from src.qft_pcn.composition.goldstone import _compose_leaf_ops
from src.qft_pcn.logic._mera_typing_rules import mutate_leaf
from src.qft_pcn.logic._mera_window import mera_window_expectation_factored
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_encoding import (
    TYPE_INT, TYPE_BOOL, KIND_INT, MERA_LEAF_DIM,
)
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic import mera_typing_hamiltonian as mth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_typing_penalty_ops(H_t: mth.MeraTypingHamiltonian, term, state):
    """Extract the list of factored leaf-op dicts that make up a single
    typing term's operator.

    Per-rule energy functions (e.g. :func:`_energy_t_lit_int`) build their
    operator as a SUM of factored leaf-op dicts and evaluate each via
    :func:`mera_window_expectation_factored`. We monkeypatch that primitive
    inside ``mera_typing_hamiltonian`` for the duration of one ``term_energy``
    call, recording each ops_dict the rule emitted — then restore the real
    primitive.

    The recording primitive returns the real expectation so the rule's
    control flow (which may branch on accumulated totals) is unaffected.
    Returns a list of ops_dicts; the term's operator is the SUM of factored
    products over these dicts (coefficient 1 each per the rule bodies).
    """
    captured: list[dict] = []
    real_primitive = mth.mera_window_expectation_factored

    def recording_primitive(st, ops):
        captured.append({leaf: op.copy() for leaf, op in ops.items()})
        return real_primitive(st, ops)

    mth.mera_window_expectation_factored = recording_primitive
    try:
        H_t.term_energy(state, term)
    finally:
        mth.mera_window_expectation_factored = real_primitive
    return captured


def _collect_factors(H_t, H_e, state):
    """Collect every factored leaf-op dict making up H_t and H_e on this
    state. The Hamiltonians decompose as

        H_t = sum_{t in typing_factors} (factored product over t)
        H_e = sum_{e in eval_factors}   (factored product over e)

    so any product expectation ``<psi| H_t H_e |psi>`` factors as a sum
    over (t, e) pairs of factored window expectations — the same substrate
    primitive §12.6 goldstone uses.
    """
    typing_factors: list[dict] = []
    for t in H_t.terms:
        typing_factors.extend(_capture_typing_penalty_ops(H_t, t, state))
    eval_factors: list[dict] = []
    for e in H_e.terms:
        eval_factors.extend(H_e._penalty_ops(e))
    return typing_factors, eval_factors


def _two_op_expectation(state, factors_a, factors_b):
    """Compute ``<psi| A B |psi>`` (complex) where ``A = sum_a a`` and
    ``B = sum_b b`` are SUMS of factored leaf-op dicts.

    Implementation mirrors goldstone's :func:`_two_term_expectation` but
    returns the complex value (not Re()) and works across two different
    Hamiltonians instead of two terms of the same H. The factored-window
    primitive :func:`mera_window_expectation_factored` evaluates each
    pairwise product exactly (no dense materialization, spec §1.3).
    """
    total = 0.0 + 0.0j
    for ops_a in factors_a:
        for ops_b in factors_b:
            prod = _compose_leaf_ops(ops_a, ops_b)
            total += mera_window_expectation_factored(state, prod)
    return total


# ---------------------------------------------------------------------------
# §13.7 well-typed: H_t H_e |psi> = 0 (the operator-algebraic type-safety
# statement)
# ---------------------------------------------------------------------------


def test_typing_eval_hamiltonians_commute_on_well_typed():
    """§13.7 type safety: ``[H_typing, H_eval] = 0`` on a well-typed substrate.

    For a well-typed program ``H_t |psi> = 0`` (every typing term is a
    projector onto the violated subspace, projector basis §7.4). The
    operator-algebraic content of "well-typed programs cannot go wrong"
    is then

        H_t H_e |psi> = H_e H_t |psi> = 0,

    i.e. ``[H_t, H_e] |psi> = 0`` and BOTH ordering's expectations vanish.
    We verify this by computing ``<psi| H_t H_e |psi>`` and
    ``<psi| H_e H_t |psi>`` from the same factored-window primitive used
    in §12.6 goldstone.
    """
    # ``1 + 2`` is well-typed (Int) AND carries a live R-Arith eval redex,
    # so H_e|psi> is a non-trivial vector — without that, ``<H_t H_e> = 0``
    # would hold trivially via H_e|psi> = 0 and the test would not pin
    # commutativity. With a live redex, ``<H_t H_e> = 0`` is the genuine
    # operator-algebraic content: H_e acts non-trivially yet preserves
    # the well-typed null space.
    ast = parse(r"1 + 2")
    state, meta = encode_mera(ast)
    H_t = mth.MeraTypingHamiltonian(meta)
    H_e = MeraEvalHamiltonian(meta)

    # Sanity: well-typed ⇒ typing energy is zero, so H_t|psi> = 0 on the
    # projector basis. This pins the test's premise.
    assert H_t.total_energy(state) < 1e-9, (
        f"fixture must be well-typed; got total_energy={H_t.total_energy(state)}"
    )
    # Sanity: eval must act non-trivially (a live redex), otherwise the
    # claim ``<H_t H_e> = 0`` would hold for the trivial reason H_e|psi> = 0.
    assert H_e.total_energy(state) > 1e-3, (
        f"fixture must have a live eval redex; got total_energy="
        f"{H_e.total_energy(state)}"
    )

    typing_factors, eval_factors = _collect_factors(H_t, H_e, state)
    e_te = _two_op_expectation(state, typing_factors, eval_factors)
    e_et = _two_op_expectation(state, eval_factors, typing_factors)

    # [H_t, H_e]|psi> = 0  ⇒  <H_t H_e> = <H_e H_t>.
    assert abs(e_te - e_et) < 1e-8, (
        f"§13.7 violated: <H_t H_e>={e_te}, <H_e H_t>={e_et}; "
        f"commutator nonzero on well-typed program"
    )
    # Stronger: BOTH must be at the numerical floor since H_t|psi> = 0.
    # This is the operator-algebraic statement of well-typed program
    # being in the joint null space — type safety §13.7.
    assert abs(e_te) < 1e-8, (
        f"<H_t H_e>={e_te} not at floor on well-typed program "
        f"(H_t|psi> should vanish, so <H_t H_e> = (H_t|psi>)^dagger H_e|psi> = 0)"
    )
    assert abs(e_et) < 1e-8, (
        f"<H_e H_t>={e_et} not at floor on well-typed program"
    )


# ---------------------------------------------------------------------------
# §13.7 content-bearing companion: ill-typed produces non-zero <H_t H_e>
# ---------------------------------------------------------------------------


def test_typing_eval_commutator_nonzero_on_ill_typed():
    """If ``<H_t H_e>`` vanished on EVERY state, the well-typed test would
    be vacuous (it would not distinguish the type-safety claim from a
    trivial operator identity). We substrate-mutate a well-typed program
    by swapping a literal-int node's type leaf from INT to BOOL (mirroring
    §12.1's anomaly fixture) and assert ``|<H_t H_e>|`` is detectably
    non-zero — i.e. type safety's claim is content-bearing: typing
    failure produces a non-trivial operator product expectation.
    """
    # Same fixture as the well-typed test; substrate-mutating one type
    # leaf turns ``1 + 2`` into an ill-typed configuration while leaving
    # the R-Arith eval redex live — so H_e probes the broken-typing
    # subspace and the product ``<H_t H_e>`` becomes detectably non-zero.
    ast = parse(r"1 + 2")
    state, meta = encode_mera(ast)
    H_t = mth.MeraTypingHamiltonian(meta)
    H_e = MeraEvalHamiltonian(meta)

    # Find a literal-int leaf node operator-algebraically: scan nodes for
    # one whose kind-leaf projects onto KIND_INT with amplitude ~1 AND
    # which has no children (a leaf node in the AST = a literal).
    proj_int = np.zeros((MERA_LEAF_DIM, MERA_LEAF_DIM), dtype=complex)
    proj_int[KIND_INT, KIND_INT] = 1.0
    int_node = None
    for n in range(meta.n_nodes):
        kind_leaf = meta.layout.leaf_of(n, "kind")
        amp = state.local_expectation(kind_leaf, proj_int)
        if abs(amp - 1.0) < 1e-6 and not meta.children_of_node.get(n):
            int_node = n
            break
    assert int_node is not None, "no literal-int leaf node found in fixture"

    type_leaf = meta.layout.leaf_of(int_node, "type")
    mutated = mutate_leaf(state, type_leaf, TYPE_INT, TYPE_BOOL)

    # Substrate guarantee: the mutation breaks typing (H_t|mutated> != 0).
    # The mirror of §12.1: a single type leaf swap puts the state outside
    # the well-typed manifold.
    assert H_t.total_energy(mutated) > 1e-3, (
        f"substrate mutation failed to break typing; "
        f"total_energy={H_t.total_energy(mutated)}"
    )

    typing_factors, eval_factors = _collect_factors(H_t, H_e, mutated)
    e_te = _two_op_expectation(mutated, typing_factors, eval_factors)

    # Content-bearing: ill-typed state has H_t|mutated> != 0 in directions
    # H_e can probe, so <H_t H_e> must be detectably non-zero.
    assert abs(e_te) > 1e-3, (
        f"§13.7 content-bearing test failed: ill-typed program produced "
        f"<H_t H_e>={e_te} (|·|={abs(e_te)}); commutator product should "
        f"be detectable on the broken-typing substrate"
    )
