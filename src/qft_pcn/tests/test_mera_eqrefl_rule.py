"""Tests for R-Eq-Refl (I-Task-10 blocker #4; spec §7.1 extended, §8.12).

R-Eq-Refl reduces an `Eq lhs rhs` proposition to `BoolLit(True)` once
the lhs and rhs sub-trees are leaf-for-leaf identical. The reduction is
operator-algebraic:

  * a diagonal projector built as `lam * P[KIND_EQ] * (I - P_equal_pair)`
    per (species, paired-sub-tree-node) — implemented via the
    single-leaf-marginal-product factorisation
    `I - P_equal_pair = I - sum_i P_i(lhs) P_i(rhs)`, so every factored
    op is a dict[leaf -> (16,16)] (no 16^2 dense operator, spec §1.3);
  * a transition gate that fires WHEN the diagonal residual hits 0
    (i.e. lhs and rhs confirmed leaf-equal): the Eq node's kind leaf
    promotes KIND_EQ -> KIND_BOOL, value leaf -> VALUE_TRUE, and the
    operand sub-trees collapse to PAD.

The composite goal — `forall x:Nat. Eq (x + Zero) x` — combines
R-AddZero (blocker #3) + R-Eq-Refl (this rule) + Forall-protected
(blocker #5). R-AddZero promotes (x + Zero) to inherit the Var's
entangled bid; the Eq's lhs and rhs are then leaf-identical (both
carry the Var's bid index, both KIND_VAR); R-Eq-Refl reads 0 and
promotes the Eq node to BoolLit(True). This is the §10.10 induction
theorem path.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.logic.ast import (
    Eq, NatLit, Zero, Var, Forall, TNat, Bin,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_encoding import (
    KIND_EQ, KIND_BOOL, KIND_PAD,
)
from src.qft_pcn.logic.encoding import VALUE_TRUE
from src.qft_pcn.logic.mera_evaluation_hamiltonian import (
    MeraEvalHamiltonian, RULE_R_EQ_REFL, RULE_R_ADD_ZERO,
    _kind_leaf, _value_leaf, _leaf_weights,
)
from src.qft_pcn.logic.mera_evolution_logic import (
    mera_imaginary_evolve_state,
)


# ---- diagonal residual on structural (in)equality ------------------------


def test_eqrefl_zero_on_structural_eq():
    """`Eq Zero Zero`: lhs and rhs are leaf-for-leaf identical, so the
    R-Eq-Refl diagonal residual on the Eq node reads 0."""
    src = Eq(lhs=Zero(), rhs=Zero())
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    refl_terms = [t for t in H.terms if t.rule_id == RULE_R_EQ_REFL]
    # Sum the per-Eq-node R-Eq-Refl energies. The Eq node is the AST
    # root (node 0); other terms enumerate over non-Eq nodes and read
    # 0 trivially (P[KIND_EQ] = 0 on them).
    total = sum(H.term_energy(state, t) for t in refl_terms)
    assert total < 1e-9, (
        f"R-Eq-Refl diagonal non-zero on `Eq Zero Zero`: {total}")


def test_eqrefl_positive_on_structural_neq():
    """`Eq Zero (NatLit 5)`: lhs and rhs differ structurally (kind
    differs: KIND_ZERO vs KIND_NATLIT; value differs), so the
    R-Eq-Refl diagonal residual is strictly positive."""
    src = Eq(lhs=Zero(), rhs=NatLit(val=5))
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    refl_terms = [t for t in H.terms if t.rule_id == RULE_R_EQ_REFL]
    total = sum(H.term_energy(state, t) for t in refl_terms)
    assert total > 1e-6, (
        f"R-Eq-Refl diagonal failed to detect structural inequality: "
        f"{total}")


# ---- closed reduction: Eq Zero Zero -> BoolLit(True) --------------------


def test_eqrefl_reduces_eq_zero_zero_to_true():
    """`Eq Zero Zero` reduces to BoolLit(True) under imaginary-time
    evolution. The Eq node's kind leaf rotates KIND_EQ -> KIND_BOOL
    and the value leaf to VALUE_TRUE."""
    src = Eq(lhs=Zero(), rhs=Zero())
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    # Locate the Eq node (AST root = node 0).
    eq_node = 0
    pre_kind = _leaf_weights(state, _kind_leaf(meta, eq_node))
    assert pre_kind[KIND_EQ] > 0.99, (
        f"pre-evolution Eq node not dominantly KIND_EQ: {pre_kind}")
    _, final = mera_imaginary_evolve_state(
        state, H, dt=0.1, steps=120, chi_layer=16)
    post_kind = _leaf_weights(final, _kind_leaf(meta, eq_node))
    post_val = _leaf_weights(final, _value_leaf(meta, eq_node))
    assert post_kind[KIND_BOOL] > 0.99, (
        f"Eq node did not promote to KIND_BOOL: {post_kind}")
    assert post_val[VALUE_TRUE] > 0.99, (
        f"Eq node value leaf did not promote to VALUE_TRUE: {post_val}")
    refl_residual = sum(
        H.term_energy(final, t) for t in H.terms
        if t.rule_id == RULE_R_EQ_REFL)
    assert refl_residual < 1e-3, (
        f"R-Eq-Refl residual did not fully relax: {refl_residual}")


# ---- Forall-protected: forall x. Eq x x ---------------------------------


def test_eqrefl_forall_x_eq_x():
    """`forall x:Nat. Eq x x`: both sides of the Eq are the SAME
    Forall-bound Var. The two Var occurrences carry the same bid (both
    entangled with the Forall's bid) and the same kind/value/type
    indices by construction, so the R-Eq-Refl diagonal reads 0 from
    the start — no evolution needed."""
    body = Eq(lhs=Var(name="x"), rhs=Var(name="x"))
    src = Forall(param="x", param_ty=TNat(), body=body)
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    refl_total = sum(
        H.term_energy(state, t) for t in H.terms
        if t.rule_id == RULE_R_EQ_REFL)
    assert refl_total < 1e-6, (
        f"R-Eq-Refl on `forall x. Eq x x` non-zero: {refl_total}")


# ---- THE LOAD-BEARING COMPOSITE: forall x. Eq (x + Zero) x ---------------


def test_eqrefl_addzero_composite():
    """`forall x:Nat. Eq (x + Zero) x` — the §10.10 induction theorem
    path. R-AddZero promotes `(x + Zero)` to inherit the Var's
    entangled bid; the Eq's lhs and rhs then carry identical leaves
    (both KIND_VAR, same bid index); R-Eq-Refl reads 0 and promotes
    the Eq node to BoolLit(True). Combined eval residual -> ~0; the
    Eq node's value leaf weight at VALUE_TRUE > 0.99.

    This is THE load-bearing proof-of-system test: it exercises
    blockers #3, #4, and #5 simultaneously and confirms the
    operator-algebraic substrate genuinely realises a universally
    quantified algebraic identity.
    """
    body = Eq(lhs=Bin(op="+", lhs=Var(name="x"), rhs=Zero()),
              rhs=Var(name="x"))
    src = Forall(param="x", param_ty=TNat(), body=body)
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    protected = set(meta.forall_protected_leaves)
    assert protected, (
        "Forall-protected leaves set is empty — #5 must populate it")

    eq_node = 0  # root Forall? No — encode_mera puts root at node 0.
    # The Eq node is the body of the Forall; locate it via children.
    forall_kids = meta.children_of_node.get(0, [])
    assert forall_kids, "Forall encoded with no body child"
    eq_node = forall_kids[0]
    pre_kind = _leaf_weights(state, _kind_leaf(meta, eq_node))
    assert pre_kind[KIND_EQ] > 0.99, (
        f"pre-evolution Eq node not KIND_EQ: {pre_kind}")

    _, final = mera_imaginary_evolve_state(
        state, H, dt=0.1, steps=300, chi_layer=16,
        frozen_leaves=protected)

    # AddZero residual fully relaxed.
    addzero_residual = sum(
        H.term_energy(final, t) for t in H.terms
        if t.rule_id == RULE_R_ADD_ZERO)
    # EqRefl residual fully relaxed.
    eqrefl_residual = sum(
        H.term_energy(final, t) for t in H.terms
        if t.rule_id == RULE_R_EQ_REFL)

    # Eq node promoted to BoolLit(True).
    post_kind = _leaf_weights(final, _kind_leaf(meta, eq_node))
    post_val = _leaf_weights(final, _value_leaf(meta, eq_node))

    # Load-bearing assertions — NEVER weaken (the composite proof IS
    # the proof of the §10.10 induction theorem path).
    assert addzero_residual < 1e-3, (
        f"R-AddZero residual did not relax: {addzero_residual}")
    assert eqrefl_residual < 1e-3, (
        f"R-Eq-Refl residual did not relax: {eqrefl_residual}")
    assert post_kind[KIND_BOOL] > 0.99, (
        f"Eq node did not promote to KIND_BOOL: {post_kind}")
    assert post_val[VALUE_TRUE] > 0.99, (
        f"Eq node value leaf did not promote to VALUE_TRUE: "
        f"{post_val}")
