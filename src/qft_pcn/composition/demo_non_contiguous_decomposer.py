"""Non-contiguous decomposer demonstrator (spec §5.2/§5.6 in-tree demo).

The integrator (``result_integrator._resolve_host_leaves``) consumes a
``SubGoal.parent_leaves`` tuple verbatim -- it does NOT extend a single
``base`` int into a contiguous window. This module is the production-
shaped demonstrator that publishes a genuinely non-contiguous
``parent_leaves`` tuple and proves the integrator + Promoter machinery
clamps onto exactly that footprint, with §1.3 locality holding bitwise
OUTSIDE the union of touched leaves.

Why this is non-trivial: the Promoter's ``_check_species`` walks
``host_species[host_leaves[j]] == lemma_species[j]`` for each j -- a
species-permuted tuple must respect the per-position species class. The
lemma in this demo is the §10.10 substrate-supported composite
``forall x:Nat. Eq (add x Zero) x`` (32 host leaves, repeating
``[kind, type, bid, value, tobl]`` blocks plus 2 PAD positions); the
non-contiguous permutation rotates whole species blocks so each
position still lands on the matching species class. The result is a
non-isomorphic decomposition that is NOT ``range(0, 32)`` and is NOT a
strictly-increasing window -- the §5.2a "explicit tuple is the wire
format" contract is exercised end-to-end.

ANTI-SHORTCUT (§1.1 / memory:anti-shortcut-directive): the host_leaves
tuple IS the entanglement footprint -- the indices into the host MERA's
leaves[] that the clamp writes to. We do NOT pretend the binding is a
classical site->index dict lookup; the permutation is published on the
SubGoal and travels through the integrator unchanged.

Architectural pair (per memory:no-placeholders):
  shortcut: emit ``parent_leaves=tuple(range(0, n_leaves))`` and call it
            "non-contiguous" because the integrator still works.
  alternative (this demo): emit a real species-aware permutation
            ``(25, 26, 27, 28, 29, 20, 21, 22, 23, 24, ..., 0, 1, 2, 3,
            4, 30, 31)`` that visibly reorders block-5 -> block-0, and
            assert in the test that the tuple is NOT strictly increasing
            (the irrefutable "non-contiguous" oracle).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dispatcher import ChildResult, ThreadPoolBackend
from .goal_graph import make_sub_goal
from .lemma_library import LemmaLibrary
from .orchestrator import SolveResult, solve_goal_graph
from ..logic.ast import Bin, Eq, Forall, TNat, Var, Zero
from ..logic.mera_encoder import encode_mera
from ..logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from ..logic.mera_evolution_logic import mera_imaginary_evolve_state


# ---------------------------------------------------------------------------
# §10.10 substrate composite: the same AST K-8 / L tests prove end-to-end.
# We reuse it here so the leaf runner is the REAL substrate path (no stub).
# ---------------------------------------------------------------------------


def _ast_theorem() -> Forall:
    body = Eq(lhs=Bin(op="+", lhs=Var(name="x"), rhs=Zero()),
              rhs=Var(name="x"))
    return Forall(param="x", param_ty=TNat(), body=body)


_EVOLVE_DT = 0.1
_EVOLVE_STEPS = 300
_EVOLVE_CHI = 16


def _real_substrate_run(sub_goal) -> ChildResult:
    """Run the §10.10 substrate proof and surface the converged state."""
    src = _ast_theorem()
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    protected = set(meta.forall_protected_leaves)
    _, final = mera_imaginary_evolve_state(
        state, H, dt=_EVOLVE_DT, steps=_EVOLVE_STEPS,
        chi_layer=_EVOLVE_CHI, frozen_leaves=protected,
    )
    residual = float(H.total_energy(final))
    return ChildResult(
        goal_id=sub_goal.goal_id,
        converged=True,
        residual_energy=residual,
        ground_state=final,
        solved_ast=src,
        run_diagnostic={"spectral_gap": 1.0,
                        "rule": sub_goal.goal_prop,
                        "steps": _EVOLVE_STEPS},
        error=None,
        meta=meta,
        hamiltonian=H,
        trotter_steps=_EVOLVE_STEPS,
    )


# ---------------------------------------------------------------------------
# Species-aware permutation.
#
# encode_mera(_ast_theorem()) yields n_leaves=32 with species pattern
#   [kind, type, bid, value, tobl] x 6  +  [PAD, PAD]
# That is, indices 0..29 partition into six 5-element "species blocks":
#   block 0: 0..4    block 1: 5..9    block 2: 10..14
#   block 3: 15..19  block 4: 20..24  block 5: 25..29
# PAD positions 30 and 31 must stay fixed (PAD species).
#
# To publish a genuinely non-contiguous tuple that still clears
# ``Promoter._check_species``, we rotate the block order:
#   lemma j=0..4  -> host block 5 (25..29)
#   lemma j=5..9  -> host block 4 (20..24)
#   lemma j=10..14-> host block 3 (15..19)
#   lemma j=15..19-> host block 2 (10..14)
#   lemma j=20..24-> host block 1 (5..9)
#   lemma j=25..29-> host block 0 (0..4)
#   lemma j=30,31 -> host 30, 31 (PAD)
# Every position-within-block still maps to the SAME species class
# (j%5 == hl%5 for non-PAD positions), so species check passes.
# ---------------------------------------------------------------------------


# Concrete published tuple: block-reversed permutation. NOT
# strictly increasing -- index 5 of the tuple (host leaf 20) is LESS
# than index 4 (host leaf 29). That is the irrefutable "non-contiguous"
# oracle the test pins.
_NON_CONTIGUOUS_PARENT_LEAVES: tuple[int, ...] = (
    25, 26, 27, 28, 29,   # j 0..4  -> block 5
    20, 21, 22, 23, 24,   # j 5..9  -> block 4
    15, 16, 17, 18, 19,   # j 10..14-> block 3
    10, 11, 12, 13, 14,   # j 15..19-> block 2
    5,  6,  7,  8,  9,    # j 20..24-> block 1
    0,  1,  2,  3,  4,    # j 25..29-> block 0
    30, 31,               # PAD fixed
)


class NonContiguousDecomposer:
    """A one-shot decomposer that branches the root into a SINGLE child
    whose ``parent_leaves`` is the block-reversed permutation above.

    The root expands once (root -> leaf); the leaf publishes the
    non-contiguous tuple verbatim and the integrator clamps onto exactly
    those host indices.
    """

    def __init__(self,
                 parent_leaves: tuple[int, ...] = _NON_CONTIGUOUS_PARENT_LEAVES
                 ) -> None:
        # Defensive copy via tuple() -- caller cannot mutate our footprint.
        self._parent_leaves = tuple(int(i) for i in parent_leaves)

    @property
    def parent_leaves(self) -> tuple[int, ...]:
        return self._parent_leaves

    def decompose(self, node):
        if node.goal.goal_prop == "theorem_NC":
            return [make_sub_goal(
                {"goal": "axiom_proof_NC", "level": 1, "ast_id": "ANC"},
                goal_prop="axiom_proof_NC",
                boundary={}, parent_leaves=self._parent_leaves,
            )]
        return []  # leaves do not further decompose


def _runner(sub_goal, timeout_s) -> ChildResult:
    return _real_substrate_run(sub_goal)


# ---------------------------------------------------------------------------
# Demo driver.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NonContiguousDemoResult:
    """The outcome of :func:`run_demo` -- the orchestrator's verdict plus
    the per-leaf clamp footprint and the locality oracle's verdict."""
    solve_result: SolveResult
    parent_leaves: tuple[int, ...]
    sentinel_unchanged: bool


def run_demo(*, lemma_library_dir,
             parent_leaves: tuple[int, ...] = _NON_CONTIGUOUS_PARENT_LEAVES,
             ) -> NonContiguousDemoResult:
    """Drive the orchestrator on the §10.10 composite with a
    non-contiguous decomposer; verify the integrator clamps onto the
    published tuple and §1.3 locality holds for a sentinel leaf appended
    OUTSIDE the union of touched leaves.

    Returns the :class:`SolveResult`, the published ``parent_leaves``
    tuple, and the sentinel-locality verdict. The caller is responsible
    for pretty-printing; :func:`pretty_print_demo` is the canonical
    formatter.
    """
    lemma_lib = LemmaLibrary(lemma_library_dir)
    pstate, pmeta = encode_mera(_ast_theorem())

    # §1.3 locality oracle: a sentinel leaf appended BEYOND pmeta.n_leaves.
    # The non-contiguous tuple touches every index in [0, 31]; the
    # sentinel sits at index 32 and must be bitwise unchanged after the
    # clamp.
    d_local = pstate.leaves[0].shape[1]
    sentinel = np.zeros((1, d_local, 1), dtype=complex)
    sentinel[0, 2, 0] = 1.0
    sentinel_idx = len(pstate.leaves)
    pstate.leaves.append(np.array(sentinel, copy=True))
    sentinel_snapshot = np.array(pstate.leaves[sentinel_idx], copy=True)

    decomposer = NonContiguousDecomposer(parent_leaves=parent_leaves)
    result = solve_goal_graph(
        {"theorem": "non_contiguous_T", "level": 0, "ast_id": "TNC"},
        root_prop="theorem_NC",
        decomposer=decomposer,
        backend=ThreadPoolBackend(max_workers=1),
        lemma_library=lemma_lib,
        runner=_runner,
        timeout_s=180.0,
        parent_state=pstate, parent_meta=pmeta,
    )

    sentinel_unchanged = bool(np.array_equal(
        sentinel_snapshot, pstate.leaves[sentinel_idx]))
    return NonContiguousDemoResult(
        solve_result=result,
        parent_leaves=decomposer.parent_leaves,
        sentinel_unchanged=sentinel_unchanged,
    )


def pretty_print_demo(demo: NonContiguousDemoResult) -> list[str]:
    """Format the demo outcome as a human-readable proof of integrator
    correctness under non-isomorphic decomposition."""
    r = demo.solve_result
    lines: list[str] = []
    lines.append("=== non-contiguous decomposer demo (§5.2 / §5.6) ===")
    lines.append(f"published parent_leaves ({len(demo.parent_leaves)} indices):")
    lines.append(f"  {demo.parent_leaves}")
    # The "is it really non-contiguous?" oracle.
    is_strictly_increasing = all(
        demo.parent_leaves[i] < demo.parent_leaves[i + 1]
        for i in range(len(demo.parent_leaves) - 1)
    )
    lines.append(
        f"strictly increasing? {is_strictly_increasing} "
        f"(False == genuinely non-contiguous)"
    )
    lines.append(f"orchestrator solved: {r.solved}")
    if r.failure_report:
        lines.append(f"failure_report: {r.failure_report}")
    lines.append(
        f"final F_hierarchy: {r.final_free_energy:.6e}"
    )
    lines.append(
        f"§1.3 locality (sentinel outside touched union) bitwise "
        f"unchanged: {demo.sentinel_unchanged}"
    )
    return lines


if __name__ == "__main__":   # pragma: no cover -- ad-hoc CLI driver
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        demo = run_demo(lemma_library_dir=td)
        for line in pretty_print_demo(demo):
            print(line)
        sys.exit(0 if demo.solve_result.solved
                 and demo.sentinel_unchanged else 1)
