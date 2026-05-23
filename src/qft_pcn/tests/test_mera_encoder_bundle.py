"""Bundle encoder rebase invariants.

`_encode_bundle` concatenates per-child serializations and rebases each
child's local ``VarRef.binder_site`` indices into the unified pre-order.
A historical bug (fixed in this commit) double-rebased witness-children
(``ci >= 1``) when ``sketch_is_structural`` was True: the first
concatenation pass added ``running`` (== ``child_offsets[ci]`` at that
moment), and the structural-sketch post-pass added ``child_offsets[ci]``
a second time. P4's witness-augmented bundle ended with ``Var x`` at
``binder_site=18`` while ``meta.n_nodes=17``, producing a downstream
``state.leaves[leaf]`` IndexError under the typing Hamiltonian's
factored expectation.

This test pins the encoder-side invariant: after ``_encode_bundle``,
every ``VarRef.binder_site`` (resolvable via ``meta.use_to_binder``)
must fall inside ``[0, meta.n_nodes)``.
"""
from __future__ import annotations

from src.qft_pcn.logic.ast import (
    Lam, Var, Bin, If, IntLit, TInt, HoleVar,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_encoding import (
    SPECIES_LEAF_OFFSET, LEAVES_PER_NODE,
)
from src.qft_pcn.logic.mera_synthesis.encode_ext import _witness_augmented_ast
from src.qft_pcn.logic.mera_synthesis.problem import IOExample


def _p4_witness_augmented_ast():
    """Build P4's witness-augmented sketch+examples bundle.

    Mirrors the construction in
    ``test_typing_hamiltonian_p4_program_does_not_raise``.
    """
    cand_correct = If(
        cond=Bin(op="<", lhs=Var(name="x"), rhs=IntLit(val=5)),
        then_b=Var(name="x"),
        else_b=Bin(op="+", lhs=Var(name="x"), rhs=IntLit(val=1)),
    )
    cand_wrong_thr = If(
        cond=Bin(op="<", lhs=Var(name="x"), rhs=IntLit(val=3)),
        then_b=Var(name="x"),
        else_b=Bin(op="+", lhs=Var(name="x"), rhs=IntLit(val=1)),
    )
    cand_succ_only = Bin(
        op="+", lhs=Var(name="x"), rhs=IntLit(val=1),
    )
    hole = HoleVar(
        candidates=(cand_correct, cand_wrong_thr, cand_succ_only),
    )
    sketch = Lam(param="x", param_ty=TInt(), body=hole)
    examples = (
        IOExample(inputs=(IntLit(val=2),), output=IntLit(val=2)),
        IOExample(inputs=(IntLit(val=7),), output=IntLit(val=8)),
    )
    return _witness_augmented_ast(sketch, examples)


def test_encode_bundle_witness_binder_sites_in_range():
    """Every use->binder resolution must land inside the leaf array.

    Regression for the ``_encode_bundle`` double-rebase bug. Before the
    fix, P4's witness Var x landed at ``binder_node=18`` with
    ``meta.n_nodes=17``. With the fix, all binder_node values must be
    strictly less than ``meta.n_nodes``.
    """
    aug = _p4_witness_augmented_ast()
    state, meta = encode_mera(aug, n_nodes_max=32, chi_layer=16)

    assert len(meta.use_to_binder) > 0, (
        "P4 witness bundle should contain at least one Var use; "
        "test fixture or encoder regressed."
    )
    bid_offset = SPECIES_LEAF_OFFSET["bid"]
    for use_bid, binder_bid in meta.use_to_binder.items():
        binder_node = (binder_bid - bid_offset) // LEAVES_PER_NODE
        assert 0 <= binder_node < meta.n_nodes, (
            f"binder_node {binder_node} (from binder_bid={binder_bid}) "
            f"out of range [0, {meta.n_nodes}); use_bid={use_bid}. "
            "The _encode_bundle double-rebase bug has regressed."
        )
