"""I-Task-10 blocker #5: Forall-protected leaves under imaginary-time
evolution.

Universal quantification in the tensor-network substrate is genuine
inertia (§1.1 binding-as-entanglement): the Forall binder's own bid
leaf and every bound Var use's 5 species leaves are frozen across the
synthesis driver's three-phase relaxation. The bound Var's `value`
species stays at VALUE_NONE — it is never pinned to a concrete integer
by relaxation. This is what makes the universal quantifier universal:
no classical iteration over values, ever.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.logic.ast import (
    parse, Forall, Var, Eq, TNat,
)
from src.qft_pcn.logic.encoding import VALUE_NONE
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_encoding import KIND_FORALL, KIND_VAR
from src.qft_pcn.logic.mera_evolution_logic import (
    mera_imaginary_evolve_state,
)
from src.qft_pcn.logic.mera_typing_hamiltonian import MeraTypingHamiltonian
from src.qft_pcn.logic.mera_evaluation_hamiltonian import (
    MeraEvalHamiltonian,
)
from src.qft_pcn.logic.mera_compose import compose_mera_hamiltonians


_FORALL_AST = Forall(param="n", param_ty=TNat(),
                     body=Eq(lhs=Var(name="n"), rhs=Var(name="n")))


def _find_sites(meta, kind):
    """Return ordered list of site indices with the given kind, by reading
    species_of_leaf / node_of_leaf — the encoder's authoritative layout."""
    # Walk node_of_leaf to enumerate node indices in order; the binder_leaves
    # map keys are exactly the Forall/Lam/Fix sites.
    return sorted(meta.binder_leaves.keys()) if kind == KIND_FORALL else []


def test_encoder_populates_forall_protected_leaves():
    """A Forall + bound Var uses produces a non-empty
    forall_protected_leaves set: the Forall's own bid leaf, and all 5
    species leaves of every Var use bound to that Forall."""
    state, meta = encode_mera(_FORALL_AST)

    # Locate the Forall site (the only binder in this AST).
    forall_sites = list(meta.binder_leaves.keys())
    assert len(forall_sites) == 1, (
        f"expected exactly one Forall binder site, got {forall_sites}")
    forall_site = forall_sites[0]

    # Locate the bound Var sites: every entry in use_to_binder whose value
    # equals the Forall's bid leaf.
    forall_bid_leaf = meta.layout.leaf_of(forall_site, "bid")
    bound_var_bid_leaves = [
        use_leaf for use_leaf, binder_leaf in meta.use_to_binder.items()
        if binder_leaf == forall_bid_leaf
    ]
    # The body is Eq n n -> two Var uses.
    assert len(bound_var_bid_leaves) == 2, (
        f"expected 2 bound Var uses, got {bound_var_bid_leaves}")

    # Forall's own bid leaf must be protected.
    assert forall_bid_leaf in meta.forall_protected_leaves

    # Each bound Var's 5 species leaves must all be protected.
    # var_node = bid_leaf // 5 (LEAVES_PER_NODE), since bid offset is 2;
    # safer: use layout.leaf_of by recovering the node index via node_of_leaf.
    for bid_leaf in bound_var_bid_leaves:
        var_node = meta.layout.node_of_leaf[bid_leaf]
        for species in ("kind", "type", "bid", "value", "tobl"):
            sp_leaf = meta.layout.leaf_of(var_node, species)
            assert sp_leaf in meta.forall_protected_leaves, (
                f"bound Var node {var_node} species {species} (leaf "
                f"{sp_leaf}) not in forall_protected_leaves")

    # Total = 1 (Forall bid) + 2 * 5 (bound Var species) = 11.
    assert len(meta.forall_protected_leaves) == 11, (
        f"unexpected protected-leaf count: {meta.forall_protected_leaves}")


def test_evolution_preserves_forall_bound_var_leaves():
    """Under imaginary-time evolution with frozen_leaves= forwarded from
    meta.forall_protected_leaves, every protected leaf is bitwise
    unchanged after the full evolution — no accumulated drift."""
    state, meta = encode_mera(_FORALL_AST)
    H = compose_mera_hamiltonians(MeraTypingHamiltonian(meta),
                                  MeraEvalHamiltonian(meta))

    protected = set(meta.forall_protected_leaves)
    assert protected, "expected non-empty protected set for Forall AST"
    snaps = {leaf: state.leaves[leaf].copy() for leaf in protected}

    _, final = mera_imaginary_evolve_state(
        state, H, dt=0.05, steps=40, chi_layer=16,
        frozen_leaves=protected)

    for leaf, snap in snaps.items():
        assert np.array_equal(final.leaves[leaf], snap), (
            f"protected leaf {leaf} drifted under evolution")


def test_relaxation_does_not_pin_forall_var_value():
    """After relaxation, every bound Var's `value` leaf remains at
    VALUE_NONE — universal quantification means the value is never
    pinned to a concrete integer by classical iteration."""
    state, meta = encode_mera(_FORALL_AST)
    H = compose_mera_hamiltonians(MeraTypingHamiltonian(meta),
                                  MeraEvalHamiltonian(meta))

    # Bound Var nodes: those whose bid-leaf maps to the Forall's bid leaf
    # via use_to_binder.
    forall_site = next(iter(meta.binder_leaves.keys()))
    forall_bid_leaf = meta.layout.leaf_of(forall_site, "bid")
    bound_var_nodes = [
        meta.layout.node_of_leaf[use_leaf]
        for use_leaf, b_leaf in meta.use_to_binder.items()
        if b_leaf == forall_bid_leaf
    ]
    assert bound_var_nodes, "no bound Var nodes found"

    _, final = mera_imaginary_evolve_state(
        state, H, dt=0.05, steps=40, chi_layer=16,
        frozen_leaves=set(meta.forall_protected_leaves))

    for var_node in bound_var_nodes:
        val_leaf = meta.layout.leaf_of(var_node, "value")
        vec = final.leaves[val_leaf][0, :, 0]
        # Read the value-species index whose amplitude is dominant; with
        # the leaf bitwise frozen this is a one-hot at VALUE_NONE.
        dom = int(np.argmax(np.abs(vec) ** 2))
        assert dom == VALUE_NONE, (
            f"bound Var node {var_node} value-leaf pinned to index {dom}, "
            f"expected VALUE_NONE={VALUE_NONE}")
        # And it remains essentially a basis vector (not a smeared mix).
        assert abs(abs(vec[VALUE_NONE]) - 1.0) < 1e-9, (
            f"bound Var node {var_node} value-leaf no longer unit-one-hot "
            f"at VALUE_NONE: |vec[VALUE_NONE]| = {abs(vec[VALUE_NONE])}")


def test_default_empty_when_no_forall():
    """Encodings without any Forall produce empty forall_protected_leaves,
    preserving prior behavior bitwise (frozen_leaves default at the
    evolution layer remains None / empty)."""
    state, meta = encode_mera(parse(r"2 + 3"))
    assert meta.forall_protected_leaves == set()
