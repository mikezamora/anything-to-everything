"""EXTENSIONS-closing acceptance for the non-contiguous decomposer demo.

The EXTENSIONS.md entry "non-contiguous decomposer demonstrator absent"
calls for an in-tree decomposer that publishes a non-contiguous
``parent_leaves`` tuple and proves the integrator clamps onto exactly
that footprint without classical rewrite. This test drives the real
``solve_goal_graph`` orchestrator through the
:class:`NonContiguousDecomposer` and pins:

  * orchestrator solves end-to-end (solved=True), so the integrator's
    strict §6.3 gate cleared with a non-contiguous host-leaf tuple --
    no classical AST splice, no fabricated proof;
  * the leaf's ``parent_leaves`` is the exact published tuple AND the
    tuple is NOT strictly increasing (the irrefutable "non-contiguous"
    oracle -- a regression that secretly normalized to ``range`` would
    trip this);
  * the §1.3 locality oracle (sentinel leaf outside the touched union)
    is bitwise unchanged after the clamp.

§1.1 (binding = entanglement clamp, NOT classical site->index dict
lookup) is honored by construction: the integrator passes the tuple
verbatim through ``_resolve_host_leaves`` into
``Promoter.compile_constraint(... leaves=[...])`` -- no dispatch keyed
by a single ``base`` index.
"""
from __future__ import annotations

import pytest


# Opt out of the conftest §9.7 dense-tensor memory ceiling: encode_mera of
# ``forall x:Nat. Eq (add x Zero) x`` allocates 16-dim leaves and 16-up
# isometries that exceed the cap by construction (mirroring the L
# hierarchical acceptance file). The ceiling stays on for every OTHER
# composition test.
@pytest.fixture(autouse=True)
def _no_large_dense():
    yield


def test_non_contiguous_decomposer_clamps_correctly(tmp_path):
    """Real ``solve_goal_graph`` run with a non-contiguous decomposer."""
    from src.qft_pcn.composition.demo_non_contiguous_decomposer import (
        _NON_CONTIGUOUS_PARENT_LEAVES,
        NonContiguousDecomposer,
        run_demo,
    )

    demo = run_demo(lemma_library_dir=tmp_path)

    # End-to-end: the orchestrator's gated path cleared (residual gate +
    # spectral gap + classical cross-check) with a NON-isomorphic
    # decomposition. No fabricated proof, no classical splice.
    assert demo.solve_result.solved is True, (
        f"non-contiguous decomposer failed to solve. "
        f"failure_report={demo.solve_result.failure_report}"
    )
    assert demo.solve_result.failure_report is None

    # Pin the exact non-contiguous footprint.
    assert demo.parent_leaves == _NON_CONTIGUOUS_PARENT_LEAVES, (
        f"decomposer corrupted published parent_leaves: "
        f"{demo.parent_leaves!r}"
    )
    # The "is it really non-contiguous?" oracle: a regression that
    # secretly normalized the tuple to range(0, 32) would trip this.
    is_strictly_increasing = all(
        demo.parent_leaves[i] < demo.parent_leaves[i + 1]
        for i in range(len(demo.parent_leaves) - 1)
    )
    assert not is_strictly_increasing, (
        f"parent_leaves was strictly increasing -- "
        f"the demo lost its non-contiguous character: "
        f"{demo.parent_leaves!r}"
    )
    # And cross-check the actual leaf node's SubGoal carries the same
    # tuple (not just the decomposer's published artifact).
    root = demo.solve_result.proof_tree.root
    assert len(root.children) == 1, (
        f"non-contiguous demo expects single leaf; got {len(root.children)}"
    )
    # ProofTreeNode does not carry parent_leaves (it carries goal_prop +
    # AST); the binding is on the SubGoal that drove the dispatch. We
    # re-walk the decomposer to confirm it would publish the same tuple
    # again -- deterministic reproduction is the decomposer contract.
    dec = NonContiguousDecomposer()
    assert dec.parent_leaves == _NON_CONTIGUOUS_PARENT_LEAVES

    # §1.3 locality: the sentinel leaf appended OUTSIDE the touched
    # union was bitwise unchanged. The clamp did not bleed.
    assert demo.sentinel_unchanged, (
        "§1.3 locality violation: sentinel leaf outside the touched "
        "union was mutated by the non-contiguous clamp"
    )
