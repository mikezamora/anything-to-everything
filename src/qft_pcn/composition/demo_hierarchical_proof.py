"""§10.11 hierarchical proof demo runner.

Spec §10.11 mandates a runnable ``composition/demo_hierarchical_proof.py``
module that drives the hierarchical proof composition demo end-to-end.
The load-bearing acceptance file is
``composition/tests/test_hierarchical_proof_demo.py`` -- it carries the
real-substrate plumbing (``encode_mera`` + ``mera_imaginary_evolve_state``
+ ``solve_goal_graph``) plus the §1.1 / §1.3 / §1.5 oracles.

This module is the spec-mandated runnable demo path: a thin wrapper that
imports the SAME substrate runner + decomposer from the acceptance test
(no duplication) and exposes a single ``run_demo()`` entry point. The
``__main__`` block prints a concise narrative report so the hierarchical
composition can be exercised standalone via::

    python -m src.qft_pcn.composition.demo_hierarchical_proof

The composite proven is ``forall x:Nat. Eq (add x Zero) x`` (§10.10),
decomposed into a 2-level tree (root T -> [L1, L2] -> [A1, A2]); each
leaf runs the real M2 substrate path under frozen-leaves protection.
"""
from __future__ import annotations

from src.qft_pcn.composition.dispatcher import ThreadPoolBackend
from src.qft_pcn.composition.lemma_library import LemmaLibrary
from src.qft_pcn.composition.orchestrator import (
    SolveResult,
    solve_goal_graph,
)
from src.qft_pcn.composition.tests.test_hierarchical_proof_demo import (
    _HierarchicalDecomposer,
    _ast_theorem,
    _runner,
)
from src.qft_pcn.logic.mera_encoder import encode_mera


def run_demo(lemma_library_dir=None) -> SolveResult:
    """Drive the §10.11 hierarchical proof composition end-to-end.

    Encodes the §10.10 composite, builds a fresh ``LemmaLibrary`` (in
    ``lemma_library_dir`` if provided, else a transient temp dir), and
    runs the orchestrator with the 2-level ``_HierarchicalDecomposer``
    + real-substrate ``_runner`` imported from the acceptance test.
    """
    import tempfile
    from pathlib import Path

    if lemma_library_dir is None:
        lemma_library_dir = Path(tempfile.mkdtemp(
            prefix="demo_hierarchical_proof_"
        ))

    pstate, pmeta = encode_mera(_ast_theorem())
    lemma_lib = LemmaLibrary(lemma_library_dir)

    return solve_goal_graph(
        {"theorem": "demo_hierarchical_T", "level": 0, "ast_id": "T"},
        root_prop="theorem_T",
        decomposer=_HierarchicalDecomposer(pmeta.n_leaves),
        backend=ThreadPoolBackend(max_workers=1),
        lemma_library=lemma_lib,
        runner=_runner,
        timeout_s=180.0,
        parent_state=pstate,
        parent_meta=pmeta,
    )


if __name__ == "__main__":
    print("§10.11 hierarchical proof demo (composition/demo_hierarchical_proof.py)")
    print("  composite: forall x:Nat. Eq (add x Zero) x  (§10.10)")
    print("  decomposition: T -> [L1, L2] -> [A1, A2]  (2-level hierarchy)")
    print("  substrate: encode_mera + mera_imaginary_evolve_state + solve_goal_graph")
    print("  running ...")
    res = run_demo()
    print(f"  solved          = {res.solved}")
    print(f"  proof_tree.root = {res.proof_tree.root.goal_prop!r}"
          if res.proof_tree is not None else "  proof_tree      = None")
    if res.proof_tree is not None:
        child_props = sorted(c.goal_prop for c in res.proof_tree.root.children)
        print(f"  sub-lemmas      = {child_props}")
    print(f"  failure_report  = {res.failure_report}")
    # The lemma library is owned by run_demo; we surface count via
    # re-loading goal_ids from the result's proof_tree leaves where present.
    print("§10.11 demo: end")
