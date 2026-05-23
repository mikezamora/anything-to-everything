"""Tests for R-AddZero (I-Task-10 blocker #3; spec §7.1 / §8.12).

R-AddZero reduces BIN(+)(x, Zero) (or BIN(+)(x, NatLit(0)) and the
symmetric case) to `x`. The reduction is operator-algebraic:

  * a diagonal projector onto "EXACTLY ONE operand is a Zero / NatLit(0)"
    (an inclusion-exclusion sum of three factored two-leaf projectors —
    never a 16**k operator, spec §1.3),
  * a transition gate that promotes the non-zero operand's root-node 5
    species leaves into the BIN node and collapses the spent sub-trees
    to PAD.

When the non-zero operand is a Forall-protected `Var` (blocker #5), the
BIN node's `bid` leaf inherits the Var's entangled bid index — universal
quantification is preserved through the encoded MERA's entanglement
structure, not through a classical substitution (§1.1).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.logic.ast import (
    Bin, IntLit, NatLit, Zero, Forall, Var, TNat,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import (
    MeraEvalHamiltonian, RULE_R_ADD_ZERO,
)
from src.qft_pcn.logic.mera_evolution_logic import (
    mera_imaginary_evolve_state,
)
from src.qft_pcn.logic.mera_encoding import KIND_BIN
from src.qft_pcn.logic._mera_eval_terms import NATLIT_VALUE_ZERO


# ---- redex-presence projector --------------------------------------------


def test_addzero_penalty_nonzero_on_concrete_natlit_plus_zero():
    """`add (NatLit 5) Zero` carries non-zero R-AddZero energy on the
    BIN(+) node — the diagonal projector recognises the redex."""
    src = Bin(op="+", lhs=NatLit(val=5), rhs=Zero())
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    bin_terms = [t for t in H.terms if t.rule_id == RULE_R_ADD_ZERO]
    energies = [H.term_energy(state, t) for t in bin_terms]
    # At least one R-AddZero term must report appreciable energy on the
    # BIN node — the redex IS present.
    assert any(e > 1e-6 for e in energies), (
        f"R-AddZero penalty is zero everywhere on `NatLit(5) + Zero`; "
        f"energies = {energies}")


def test_addzero_penalty_zero_on_non_redex():
    """`add (NatLit 5) (NatLit 3)` (no zero operand) carries no
    R-AddZero energy — the inclusion-exclusion projector is faithful."""
    src = Bin(op="+", lhs=NatLit(val=5), rhs=NatLit(val=3))
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    bin_terms = [t for t in H.terms if t.rule_id == RULE_R_ADD_ZERO]
    energies = [H.term_energy(state, t) for t in bin_terms]
    for e in energies:
        assert abs(e) < 1e-6, (
            f"R-AddZero penalty spuriously non-zero on a non-redex: "
            f"{e}")


def test_addzero_penalty_zero_when_both_zero():
    """`add Zero Zero` is NOT an R-AddZero redex (the inclusion-
    exclusion subtracts the both-zero cross term)."""
    src = Bin(op="+", lhs=Zero(), rhs=Zero())
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    bin_terms = [t for t in H.terms if t.rule_id == RULE_R_ADD_ZERO]
    energies = [H.term_energy(state, t) for t in bin_terms]
    for e in energies:
        assert abs(e) < 1e-6, (
            f"R-AddZero fires on both-zero configuration "
            f"(should be exactly-one-zero): {e}")


# ---- closed reduction ----------------------------------------------------


def test_addzero_reduces_natlit_5_plus_zero():
    """Imaginary-time evolution drives `NatLit(5) + Zero` toward its
    normal form. The R-AddZero residual drops below 1e-3 and the total
    energy strictly decreases."""
    src = Bin(op="+", lhs=NatLit(val=5), rhs=Zero())
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    e0 = H.total_energy(state)
    assert e0 > 1e-3, f"redex penalty not detected pre-evolution: {e0}"
    traj, final = mera_imaginary_evolve_state(
        state, H, dt=0.1, steps=80, chi_layer=16)
    addzero_residual = sum(
        H.term_energy(final, t) for t in H.terms
        if t.rule_id == RULE_R_ADD_ZERO)
    assert addzero_residual < 1e-3, (
        f"R-AddZero residual did not relax to 0; got {addzero_residual} "
        f"(traj[-3:] = {traj[-3:]})")
    assert traj[-1] < e0, (
        f"no net relaxation: traj[0]={e0} traj[-1]={traj[-1]}")


def test_addzero_reduces_zero_plus_natlit_5():
    """Same as above, with operand order swapped — the rule is
    symmetric (T1 and T2 of the inclusion-exclusion projector)."""
    src = Bin(op="+", lhs=Zero(), rhs=NatLit(val=5))
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    e0 = H.total_energy(state)
    assert e0 > 1e-3
    traj, final = mera_imaginary_evolve_state(
        state, H, dt=0.1, steps=80, chi_layer=16)
    addzero_residual = sum(
        H.term_energy(final, t) for t in H.terms
        if t.rule_id == RULE_R_ADD_ZERO)
    assert addzero_residual < 1e-3, (
        f"R-AddZero residual (rhs=zero case): {addzero_residual}")
    assert traj[-1] < e0


# ---- open case: Forall-protected Var (#5 + #3 integration) --------------


def test_addzero_forall_x_plus_zero_preserves_var_leaves():
    """`forall x:Nat. (x + Zero)` reduces under R-AddZero. Crucially:

      (i)  the Var's 5 species leaves are BITWISE unchanged across the
           full evolution — the Forall-protected-leaves filter is
           respected, universal quantification preserved;
      (ii) the BIN(+) node's `bid` leaf is driven to the SAME basis
           index the Var's `bid` leaf carries — entanglement-preserving
           promotion (§1.1: not classical substitution);
      (iii) the BIN node's `kind` leaf rotates away from KIND_BIN — the
            redex has fired.
    """
    body = Bin(op="+", lhs=Var(name="x"), rhs=Zero())
    src = Forall(param="x", param_ty=TNat(), body=body)
    state, meta = encode_mera(src)

    # Locate the Var node and the BIN node in the encoded layout.
    # children_of_node: root (Forall) -> body (BIN); BIN -> [Var, Zero].
    root = 0  # encode_mera lays the AST root at node 0.
    forall_kids = meta.children_of_node.get(root, [])
    assert forall_kids, "Forall encoded with no body child"
    bin_node = forall_kids[0]
    bin_kids = meta.children_of_node.get(bin_node, [])
    assert len(bin_kids) == 2, (
        f"BIN node has unexpected child count: {bin_kids}")
    # Identify which child is the Var (KIND_VAR) and which is Zero.
    from src.qft_pcn.logic.mera_encoding import KIND_VAR, KIND_ZERO
    from src.qft_pcn.logic.mera_evaluation_hamiltonian import (
        _kind_leaf, _leaf_argmax, _leaf_weights,
    )
    k0 = _leaf_argmax(state, _kind_leaf(meta, bin_kids[0]))
    k1 = _leaf_argmax(state, _kind_leaf(meta, bin_kids[1]))
    if k0 == KIND_VAR:
        var_node, zero_node = bin_kids[0], bin_kids[1]
    else:
        var_node, zero_node = bin_kids[1], bin_kids[0]
    assert _leaf_argmax(state, _kind_leaf(meta, var_node)) == KIND_VAR
    assert _leaf_argmax(state, _kind_leaf(meta, zero_node)) == KIND_ZERO

    # The Forall-protected set covers Forall's bid leaf + every leaf of
    # every Forall-bound Var.
    protected = set(meta.forall_protected_leaves)
    assert protected, (
        "Forall-protected leaves set is empty — #5 should populate it")

    # Snapshot the Var's 5 leaves BEFORE evolution.
    pre = {sp: state.leaves[meta.layout.leaf_of(var_node, sp)].copy()
           for sp in ("kind", "type", "bid", "value", "tobl")}
    var_bid_idx = _leaf_argmax(state, meta.layout.leaf_of(var_node, "bid"))

    H = MeraEvalHamiltonian(meta)
    e0 = H.total_energy(state)

    # Run evolution with the Forall-protected leaves frozen — blocker #6
    # mechanism wired up by #5. The R-AddZero gate must not write to any
    # protected leaf; if it did, the gate would either be dropped by the
    # frozen-leaf filter (preserving (i)) OR — if it bypassed the filter
    # — would corrupt the Var. The bitwise-equality check below catches
    # corruption either way.
    _, final = mera_imaginary_evolve_state(
        state, H, dt=0.1, steps=120, chi_layer=16,
        frozen_leaves=protected)

    # (i) Var leaves bitwise unchanged.
    for sp, before in pre.items():
        after = final.leaves[meta.layout.leaf_of(var_node, sp)]
        np.testing.assert_array_equal(
            before, after,
            err_msg=f"Var.{sp} leaf was modified despite Forall protection")

    # (ii) BIN node's bid leaf is driven to the Var's bid index. The
    # promotion is the entanglement-preserving step: the BIN node now
    # carries the same bid the Var carries, which is itself entangled
    # (through the encoded MERA tree) with the Forall's bid leaf.
    bin_bid_leaf = meta.layout.leaf_of(bin_node, "bid")
    bin_bid_w = _leaf_weights(final, bin_bid_leaf)
    # The Var's bid index should now dominate the BIN node's bid leaf.
    assert bin_bid_w[var_bid_idx] > 0.99, (
        f"BIN node's bid leaf did not inherit the Var's bid index "
        f"{var_bid_idx}: weights = {bin_bid_w}")

    # (iii) BIN(+) has been replaced: its kind leaf rotated away from
    # KIND_BIN toward KIND_VAR (the promoted non-zero operand's kind).
    bin_kind_leaf = meta.layout.leaf_of(bin_node, "kind")
    bin_kind_w = _leaf_weights(final, bin_kind_leaf)
    assert bin_kind_w[KIND_BIN] < 0.5, (
        f"BIN node still dominantly KIND_BIN: weights = {bin_kind_w}")
    assert bin_kind_w[KIND_VAR] > 0.5, (
        f"BIN node's kind leaf did not promote to KIND_VAR: "
        f"weights = {bin_kind_w}")

    # Net relaxation (only the eval residual matters; the typing
    # residual is not composed into H here).
    assert H.total_energy(final) < e0, (
        f"no net relaxation under R-AddZero+Forall protection: "
        f"e0={e0}, final={H.total_energy(final)}")
