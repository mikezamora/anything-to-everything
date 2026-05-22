# Hierarchical Proof Composition Demo — Implementation Plan (sub-project L)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Every
> step is TDD: write the failing test, run it, see it fail, implement, run it, see it pass.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the hierarchical proof composition demo — the second publishable milestone
(`QFT_PCN_ARCHITECTURE.md` §10.11). The demo takes a mathematical theorem, builds a goal
graph, dispatches a sub-QPCN per proof-tree node, integrates results via the lemma
machinery, and surfaces a *verified* proof tree, ending with an honest `solved X of 4`
tally.

**Architecture:** Pure orchestration over M1/M2/M3 (MERA logic substrate + synthesis) and
sub-projects I (lemma library), J (wake-sleep), K (cross-level message passing). L builds
no new substrate. A proof-tree node is verified iff its sub-QPCN ground state has residual
energy `≤ ε_verify = 1e-8` (architecture §13.2, correctness by construction).

**Tech Stack:** Python 3.11, numpy 1.26, pytest. Built on `src/qft_pcn/logic/` (M1–M3) and
`src/qft_pcn/composition/` (I, J, K).

**Spec:** `docs/superpowers/specs/2026-05-22-hierarchical-proof-demo-design.md` —
authoritative. When this plan and the spec disagree, the spec wins.

**Test runner:** `.venv/bin/python -m pytest <path> -v`.

---

## Driving principles (from spec §1 — non-negotiable; restate VERBATIM in every subagent prompt)

Subagents default to the easiest path. Here the easiest path is fabricated success.
Refuse it. For each principle, the shortcut and the required alternative are spelled out.

1. **No time/effort estimates.** SHORTCUT: writing "this step takes ~X". ALTERNATIVE: state
   sequencing and dependencies only; the demo *measures and reports* runtime, never predicts.
2. **Binding is genuine entanglement, never a classical lookup.** SHORTCUT: thread shared
   variables between sub-QPCN runs as a Python `dict`. ALTERNATIVE: shared variables across
   proof-tree nodes are bond entanglement on the MERA tree (M1 §5) and `H_coupling` energy
   (architecture §13.3.1); K passes goal propositions and ground-state sub-MPSes, not bindings.
3. **`optimize='greedy'` on every einsum.** SHORTCUT: omit it / leave default. ALTERNATIVE:
   every `np.einsum` and contraction L writes carries `optimize='greedy'`; §6.4 lint test
   enforces it.
4. **A proof is VERIFIED only at zero residual energy.** SHORTCUT: mark a node proved
   because the decoded AST "looks right" or a classical type-checker passed. ALTERNATIVE:
   a node is verified iff `residual_energy ≤ 1e-8` AND `compose_residual ≤ 1e-8` AND the
   classical check passes AND all children are verified. The classical check is necessary,
   never sufficient. `verify_proof_tree` has no "close enough" branch.
5. **Honest reporting.** SHORTCUT: loosen `ε_verify`, stub a sub-QPCN to return a hand-built
   ground state, swallow a non-convergence, skip a node, or inflate the `solved X of 4`
   count. ALTERNATIVE: the tally counts only fully verified trees; `solved 1 of 4` is a
   publishable result; a fabricated `solved 4 of 4` is research misconduct. §6.3 test
   catches an inconsistent tally.
6. **No dependency reimplementation, no stubs.** SHORTCUT: a missing/incompatible M1–K API
   becomes a quick local stub. ALTERNATIVE: a thin adapter mapping the shipped API to the
   spec §7 surface, OR a fix in the dependency with its tests still green, OR stop-and-ask.

---

## Pre-existing worktree state

Unrelated modified files exist (`QFT_PCN_ARCHITECTURE.md`, `src/qft_pcn/logic/mera_encoder.py`,
`lib/`, `src/qft_pcn/logic/_mera_holes.py`, `src/qft_pcn/tests/test_mera_holes.py`). **Leave
them alone.** Stage only the files each task names.

`src/qft_pcn/composition/` does not yet exist as a populated package — sub-projects I, J, K
create it in parallel. This plan assumes their modules are present when L is executed. If
they are not, **Task 0 stops and asks** rather than stubbing them.

---

## Task 0: Dependency reconnaissance and adapters

**Files:**
- Create: `src/qft_pcn/composition/__init__.py` (if absent)
- Create: `src/qft_pcn/composition/_deps.py` (the adapter layer, spec §7.6)
- Test: `tests/test_l_deps.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_l_deps.py`:

```python
"""Sub-project L depends on M1-M3 + I + K. This test pins the import surface
L is written against (spec section 7). If a dependency is missing or its API
differs, this test fails LOUDLY -- L does not degrade to a stub."""
from __future__ import annotations
import pytest


def test_m1_encoder_present():
    from src.qft_pcn.logic.mera_encoder import encode_mera, MeraEncodingMeta
    assert callable(encode_mera)


def test_m1_extended_ast_present():
    from src.qft_pcn.logic.ast import Forall, Fix, Eq, Zero, Succ, Nil, Cons
    assert all(callable(c) for c in (Forall, Fix, Eq, Zero, Succ, Nil, Cons))


def test_m3_relax_to_ground_present():
    from src.qft_pcn.composition._deps import relax_to_ground
    assert callable(relax_to_ground)


def test_i_lemma_library_present():
    from src.qft_pcn.composition._deps import LemmaLibrary, use_lemma
    assert LemmaLibrary is not None and callable(use_lemma)


def test_k_goal_graph_present():
    from src.qft_pcn.composition._deps import GoalGraph, Dispatcher, ResultIntegrator
    assert GoalGraph is not None
    assert Dispatcher is not None
    assert ResultIntegrator is not None
```

- [ ] **Step 2: Run the test, verify it fails**

```
.venv/bin/python -m pytest tests/test_l_deps.py -v
```

- [ ] **Step 3: Inspect the shipped dependency APIs**

Inspect the actually-shipped modules under `src/qft_pcn/composition/` (from I/J/K) and the
M2/M3 Hamiltonian/relaxation modules under `src/qft_pcn/logic/`. Record the real signatures.

**STOP-AND-ASK GATE:** if M1, M3, I, or K is absent, or its API cannot be mapped to spec
§7's surface by a thin adapter, **stop and report which dependency is missing**. Do not
create a stub that fakes a sub-QPCN run. A faked relaxation that returns a hand-built
ground state is a §1.5 violation — the whole demo's correctness rests on real sub-QPCN
convergence.

- [ ] **Step 4: Implement the adapter layer**

Create `src/qft_pcn/composition/_deps.py` re-exporting / thin-adapting the dependency APIs
to spec §7's surface: `relax_to_ground`, `compile_node_hamiltonian` (M2),
`LemmaLibrary`, `use_lemma` (I), `GoalGraph`, `Dispatcher`, `ResultIntegrator`, `Revision`
(K), and M1's `encode_mera`/`decode_mera`. Each adapter is a *thin* mapping — no behavior,
no fake. If the shipped API matches §7 exactly, the adapter is a plain re-export.

- [ ] **Step 5: Run the test, verify it passes**

```
.venv/bin/python -m pytest tests/test_l_deps.py -v
```

- [ ] **Step 6: Commit**

```
git add src/qft_pcn/composition/__init__.py src/qft_pcn/composition/_deps.py tests/test_l_deps.py
git commit -m "$(cat <<'EOF'
feat(composition/_deps): pin sub-project L dependency import surface

L composes M1-M3 + I + K. _deps.py thin-adapts the shipped APIs to the
spec section 7 surface; the test fails loudly on a missing dependency
rather than letting L degrade to a stub.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: The proof tree and the verification predicate

**Files:**
- Create: `src/qft_pcn/composition/proof_tree.py`
- Test: `tests/test_proof_tree.py` (create)

This task implements spec §4 — the load-bearing verification logic. It depends on nothing
but the dependency types from Task 0, so it is fully implementable now.

- [ ] **Step 1: Write the failing test**

Create `tests/test_proof_tree.py` — the spec §6.2 suite, complete:

```python
"""Verification-predicate tests (spec section 6.2). verify_proof_tree must
provably reject every dishonest tree: missing energy, above-threshold energy,
classical-check-only, bad composition residual, unverified child."""
from __future__ import annotations
from src.qft_pcn.composition.proof_tree import (
    ProofNode, ProofTree, verify_proof_tree, EPS_VERIFY,
)


def _leaf(node_id, energy, classical=True, status="solved"):
    return ProofNode(
        node_id=node_id, level=0, goal=f"axiom::{node_id}", children=(),
        state_ref=f"state::{node_id}", residual_energy=energy,
        compose_residual=None, classical_check=classical, status=status,
    )


def _internal(node_id, energy, compose, children, classical=True, status="solved"):
    return ProofNode(
        node_id=node_id, level=1, goal=f"lemma::{node_id}", children=tuple(children),
        state_ref=f"state::{node_id}", residual_energy=energy,
        compose_residual=compose, classical_check=classical, status=status,
    )


def _tree(*nodes, root):
    return ProofTree(theorem_id="t", root_id=root,
                     nodes={n.node_id: n for n in nodes})


def test_eps_verify_is_1e_minus_8():
    assert EPS_VERIFY == 1e-8


def test_verify_accepts_all_zero():
    a = _leaf("a", 1e-12)
    b = _leaf("b", 0.0)
    r = _internal("root", 1e-12, compose=1e-12, children=["a", "b"])
    rep = verify_proof_tree(_tree(a, b, r, root="root"))
    assert rep.verdict is True


def test_verify_rejects_missing_energy():
    a = _leaf("a", None)
    rep = verify_proof_tree(_tree(a, root="a"))
    assert rep.verdict is False


def test_verify_rejects_above_threshold():
    a = _leaf("a", 1e-6)
    rep = verify_proof_tree(_tree(a, root="a"))
    assert rep.verdict is False


def test_verify_rejects_classical_only():
    # classical check True but energy missing -> still rejected
    a = _leaf("a", None, classical=True)
    rep = verify_proof_tree(_tree(a, root="a"))
    assert rep.verdict is False


def test_verify_rejects_failed_classical_check():
    a = _leaf("a", 1e-12, classical=False)
    rep = verify_proof_tree(_tree(a, root="a"))
    assert rep.verdict is False


def test_verify_rejects_bad_compose():
    a = _leaf("a", 1e-12)
    r = _internal("root", 1e-12, compose=0.05, children=["a"])
    rep = verify_proof_tree(_tree(a, r, root="root"))
    assert rep.verdict is False


def test_verify_rejects_internal_missing_compose():
    a = _leaf("a", 1e-12)
    r = _internal("root", 1e-12, compose=None, children=["a"])
    rep = verify_proof_tree(_tree(a, r, root="root"))
    assert rep.verdict is False


def test_verify_rejects_unverified_child():
    a = _leaf("a", 1e-6, status="unverified")          # child fails
    r = _internal("root", 1e-12, compose=1e-12, children=["a"])
    rep = verify_proof_tree(_tree(a, r, root="root"))
    assert rep.verdict is False


def test_report_lists_offending_node():
    a = _leaf("a", 1e-6)
    rep = verify_proof_tree(_tree(a, root="a"))
    bad = [row for row in rep.per_node if row.verdict is False]
    assert any(row.node_id == "a" for row in bad)


def test_verify_does_not_raise_on_dangling_child():
    # root references a child that is not in nodes -> verdict False, no exception
    r = _internal("root", 1e-12, compose=1e-12, children=["ghost"])
    rep = verify_proof_tree(_tree(r, root="root"))
    assert rep.verdict is False
```

- [ ] **Step 2: Run the test, verify it fails**

```
.venv/bin/python -m pytest tests/test_proof_tree.py -v
```

- [ ] **Step 3: Implement `proof_tree.py`**

Create `src/qft_pcn/composition/proof_tree.py`:

```python
"""Proof tree and the verification predicate for sub-project L (spec section 4).

A proof-tree node is VERIFIED iff its sub-QPCN ground state has residual energy
<= EPS_VERIFY (architecture section 13.2, correctness by construction). There is
no other notion of 'proved' in this demo. The classical type-check is a necessary
cross-check, never sufficient (spec section 1.4)."""
from __future__ import annotations
from dataclasses import dataclass, field

EPS_VERIFY: float = 1e-8


@dataclass(frozen=True)
class ProofNode:
    node_id: str
    level: int
    goal: object                       # K's GoalSpec; opaque to L
    children: tuple[str, ...]
    state_ref: object | None           # M3 ground-state sub-MPS reference
    residual_energy: float | None
    compose_residual: float | None     # None for leaves
    classical_check: bool | None
    status: str                        # pending | solved | unverified | unencodable


@dataclass(frozen=True)
class NodeVerdictRow:
    node_id: str
    level: int
    residual_energy: float | None
    compose_residual: float | None
    classical_check: bool | None
    verdict: bool
    reason: str


@dataclass
class VerificationReport:
    verdict: bool
    per_node: list[NodeVerdictRow] = field(default_factory=list)


@dataclass
class ProofTree:
    theorem_id: str
    root_id: str
    nodes: dict[str, ProofNode]

    def is_verified(self) -> bool:
        return verify_proof_tree(self).verdict


def _node_ok(node: ProofNode, nodes: dict[str, ProofNode]) -> tuple[bool, str]:
    """The five spec section 4.2 conditions for a single node. Recursion over
    children is handled by the caller's post-order walk."""
    if node.status != "solved":
        return False, f"status is {node.status!r}, not 'solved'"
    if node.residual_energy is None:
        return False, "residual_energy is None (sub-QPCN never produced a ground state)"
    if node.residual_energy > EPS_VERIFY:
        return False, f"residual_energy {node.residual_energy:.3e} > EPS_VERIFY {EPS_VERIFY:.0e}"
    if node.classical_check is not True:
        return False, "classical cross-check did not pass"
    if node.children:
        if node.compose_residual is None:
            return False, "internal node has no compose_residual"
        if node.compose_residual > EPS_VERIFY:
            return False, (f"compose_residual {node.compose_residual:.3e} "
                           f"> EPS_VERIFY {EPS_VERIFY:.0e} (shared-variable structure clashes)")
    for cid in node.children:
        if cid not in nodes:
            return False, f"child {cid!r} not present in tree"
    return True, "ok"


def verify_proof_tree(tree: ProofTree) -> VerificationReport:
    """A ProofTree is verified iff EVERY node meets all five spec section 4.2
    conditions, including (condition 5) every child being itself verified.
    Never raises; a malformed tree yields verdict False with the node named."""
    rows: list[NodeVerdictRow] = []
    per_node_verdict: dict[str, bool] = {}

    # Post-order: a node's verdict depends on its children's verdicts.
    visiting: set[str] = set()

    def walk(node_id: str) -> bool:
        if node_id in per_node_verdict:
            return per_node_verdict[node_id]
        if node_id in visiting:                       # cycle: not a DAG -> reject
            per_node_verdict[node_id] = False
            return False
        node = tree.nodes.get(node_id)
        if node is None:
            return False
        visiting.add(node_id)
        ok, reason = _node_ok(node, tree.nodes)
        children_ok = True
        for cid in node.children:
            if not walk(cid):
                children_ok = False
        visiting.discard(node_id)
        verdict = ok and children_ok
        if ok and not children_ok:
            reason = "an unverified child invalidates this node"
        rows.append(NodeVerdictRow(
            node_id=node.node_id, level=node.level,
            residual_energy=node.residual_energy,
            compose_residual=node.compose_residual,
            classical_check=node.classical_check,
            verdict=verdict, reason=reason,
        ))
        per_node_verdict[node_id] = verdict
        return verdict

    root_ok = walk(tree.root_id)
    # Any node not reachable from root is still reported but does not gate the verdict;
    # an honest tree has all nodes reachable. Reachability is asserted by the demo.
    overall = root_ok and all(r.verdict for r in rows)
    rows.sort(key=lambda r: (r.level, r.node_id))
    return VerificationReport(verdict=overall, per_node=rows)
```

- [ ] **Step 4: Run the test, verify it passes**

```
.venv/bin/python -m pytest tests/test_proof_tree.py -v
```

- [ ] **Step 5: Commit**

```
git add src/qft_pcn/composition/proof_tree.py tests/test_proof_tree.py
git commit -m "$(cat <<'EOF'
feat(composition/proof_tree): proof tree + zero-residual verification predicate

verify_proof_tree implements spec section 4.2: a node is verified iff
residual energy <= 1e-8 AND compose residual <= 1e-8 AND the classical
cross-check passes AND every child is verified. No 'close enough' branch
-- the classical check is necessary, never sufficient (architecture 13.2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Theorem 1 — frozen list-induction definition

**Files:**
- Create: `src/qft_pcn/composition/theorems/__init__.py`
- Create: `src/qft_pcn/composition/theorems/theorem1_list_induction.py`
- Test: `tests/test_theorem1_list_induction.py` (create)

This task encodes spec §3.3 theorem 1: the axioms, the goal proposition, and the fixed
three-level decomposition. No sub-QPCN runs yet — this task only builds the *definition*
that the demo will dispatch.

- [ ] **Step 1: Write the failing test**

Create `tests/test_theorem1_list_induction.py`:

```python
"""Theorem 1 definition: list-induction (spec section 3.3). Tests the frozen
decomposition is well-formed -- a DAG, 3 levels, axioms at level 0, theorem at
the root -- BEFORE any sub-QPCN runs."""
from __future__ import annotations
from src.qft_pcn.composition.theorems.theorem1_list_induction import (
    THEOREM, build_decomposition,
)


def test_theorem_has_goal_and_axioms():
    assert "length" in THEOREM.goal_text
    # length, ++, + each contribute >= 2 defining equations
    assert len(THEOREM.axioms) >= 6


def test_decomposition_has_three_levels():
    dec = build_decomposition()
    levels = {n.level for n in dec.nodes}
    assert levels == {0, 1, 2}


def test_level0_nodes_are_axioms():
    dec = build_decomposition()
    level0 = [n for n in dec.nodes if n.level == 0]
    assert len(level0) >= 6
    assert all(n.children == () for n in level0)


def test_level1_has_base_case_and_inductive_step():
    dec = build_decomposition()
    level1_ids = {n.node_id for n in dec.nodes if n.level == 1}
    assert "base_case" in level1_ids
    assert "inductive_step" in level1_ids


def test_root_is_the_theorem_at_level_2():
    dec = build_decomposition()
    root = dec.root()
    assert root.level == 2
    assert set(root.children) == {"base_case", "inductive_step"}


def test_decomposition_is_a_dag():
    dec = build_decomposition()
    assert dec.is_acyclic()


def test_fallback_decomposition_exists():
    # spec section 5.1 step 6: a fixed alternative decomposition, no LLM
    dec = build_decomposition(variant="fallback")
    assert dec.root().level >= 2
```

- [ ] **Step 2: Run the test, verify it fails**

```
.venv/bin/python -m pytest tests/test_theorem1_list_induction.py -v
```

- [ ] **Step 3: Implement the theorem module**

Create `src/qft_pcn/composition/theorems/__init__.py`:

```python
"""Frozen, version-pinned target-theorem definitions for the sub-project L demo
(spec section 3.3)."""
```

Create `src/qft_pcn/composition/theorems/theorem1_list_induction.py`:

```python
"""Theorem 1 (FIRM TARGET, spec section 3.3): list-induction.

  forall xs ys : List Nat. length (xs ++ ys) = length xs + length ys

Decomposition (architecture section 10.10 acceptance): axioms -> lemmas ->
inductive step -> theorem, in 3 levels (level 0 axioms, level 1 base case +
inductive step, level 2 theorem). Each node becomes a sub-QPCN run; this module
only DEFINES the decomposition -- the demo dispatches it."""
from __future__ import annotations
from dataclasses import dataclass, field

# AST builders from M1's extended calculus.
from src.qft_pcn.logic.ast import (
    Forall, Eq, Var, Zero, Succ, Nil, Cons,
)


@dataclass(frozen=True)
class Axiom:
    name: str
    equation: object                   # an M1 Eq AST node


@dataclass(frozen=True)
class Theorem:
    theorem_id: str
    goal_text: str
    goal_ast: object                   # the M1 AST of the goal proposition
    axioms: tuple[Axiom, ...]


@dataclass(frozen=True)
class DecompNode:
    node_id: str
    level: int
    goal_text: str
    goal_ast: object
    children: tuple[str, ...]
    inductive_hypothesis: str | None = None   # node_id clamped as a lemma, or None


@dataclass
class Decomposition:
    nodes: list[DecompNode] = field(default_factory=list)
    root_id: str = "theorem"

    def root(self) -> DecompNode:
        return next(n for n in self.nodes if n.node_id == self.root_id)

    def by_id(self, node_id: str) -> DecompNode:
        return next(n for n in self.nodes if n.node_id == node_id)

    def is_acyclic(self) -> bool:
        index = {n.node_id: n for n in self.nodes}
        colour: dict[str, int] = {}            # 0 white, 1 grey, 2 black

        def dfs(nid: str) -> bool:
            colour[nid] = 1
            for c in index[nid].children:
                if c not in index:
                    continue
                if colour.get(c, 0) == 1:
                    return False
                if colour.get(c, 0) == 0 and not dfs(c):
                    return False
            colour[nid] = 2
            return True

        return all(dfs(n.node_id) for n in self.nodes if colour.get(n.node_id, 0) == 0)


# --- the goal proposition -----------------------------------------------------
# forall xs ys. length (xs ++ ys) = length xs + length ys
# Encoded with M1 AST nodes; App/named operators 'length', 'append', 'plus' are
# free constants whose meaning is fixed by the level-0 axioms below.
_GOAL_TEXT = "forall xs ys : List Nat. length (xs ++ ys) = length xs + length ys"


def _goal_ast() -> object:
    # length(append(xs,ys)) = plus(length(xs), length(ys)), under two Foralls.
    from src.qft_pcn.logic.ast import App, Var as V
    lhs = App(fn=V(name="length"),
              arg=App(fn=App(fn=V(name="append"), arg=V(name="xs")),
                      arg=V(name="ys")))
    rhs = App(fn=App(fn=V(name="plus"),
                     arg=App(fn=V(name="length"), arg=V(name="xs"))),
              arg=App(fn=V(name="length"), arg=V(name="ys")))
    from src.qft_pcn.logic.ast import TList, TNat
    body = Eq(lhs=lhs, rhs=rhs)
    inner = Forall(param="ys", param_ty=TList(elem=TNat()), body=body)
    return Forall(param="xs", param_ty=TList(elem=TNat()), body=inner)


def _axioms() -> tuple[Axiom, ...]:
    from src.qft_pcn.logic.ast import App, Var as V
    # length Nil = 0 ; length (Cons x xs) = Succ (length xs)
    ax_len_nil = Axiom("length_nil",
        Eq(lhs=App(fn=V(name="length"), arg=Nil()), rhs=Zero()))
    ax_len_cons = Axiom("length_cons",
        Eq(lhs=App(fn=V(name="length"),
                   arg=Cons(head=V(name="x"), tail=V(name="xs"))),
           rhs=Succ(arg=App(fn=V(name="length"), arg=V(name="xs")))))
    # Nil ++ ys = ys ; (Cons x xs) ++ ys = Cons x (xs ++ ys)
    ax_app_nil = Axiom("append_nil",
        Eq(lhs=App(fn=App(fn=V(name="append"), arg=Nil()), arg=V(name="ys")),
           rhs=V(name="ys")))
    ax_app_cons = Axiom("append_cons",
        Eq(lhs=App(fn=App(fn=V(name="append"),
                          arg=Cons(head=V(name="x"), tail=V(name="xs"))),
                   arg=V(name="ys")),
           rhs=Cons(head=V(name="x"),
                    tail=App(fn=App(fn=V(name="append"), arg=V(name="xs")),
                             arg=V(name="ys")))))
    # 0 + n = n ; Succ m + n = Succ (m + n)
    ax_plus_zero = Axiom("plus_zero",
        Eq(lhs=App(fn=App(fn=V(name="plus"), arg=Zero()), arg=V(name="n")),
           rhs=V(name="n")))
    ax_plus_succ = Axiom("plus_succ",
        Eq(lhs=App(fn=App(fn=V(name="plus"), arg=Succ(arg=V(name="m"))),
                   arg=V(name="n")),
           rhs=Succ(arg=App(fn=App(fn=V(name="plus"), arg=V(name="m")),
                            arg=V(name="n")))))
    return (ax_len_nil, ax_len_cons, ax_app_nil, ax_app_cons,
            ax_plus_zero, ax_plus_succ)


THEOREM = Theorem(
    theorem_id="theorem1_list_induction",
    goal_text=_GOAL_TEXT,
    goal_ast=_goal_ast(),
    axioms=_axioms(),
)


def build_decomposition(variant: str = "primary") -> Decomposition:
    """The frozen 3-level decomposition. `variant='fallback'` is the fixed
    alternative used by K's Revision hook if the primary node fails -- no LLM
    (spec section 3.2, 5.1 step 6). The fallback proves the same theorem with
    the base/step lemmas merged into a single combined-induction node."""
    axiom_nodes = [
        DecompNode(node_id=f"axiom::{ax.name}", level=0,
                   goal_text=ax.name, goal_ast=ax.equation, children=())
        for ax in THEOREM.axioms
    ]
    axiom_ids = tuple(n.node_id for n in axiom_nodes)

    if variant == "primary":
        base = DecompNode(
            node_id="base_case", level=1,
            goal_text="length (Nil ++ ys) = length Nil + length ys",
            goal_ast=_goal_ast(),          # base instance; demo specialises xs:=Nil
            children=axiom_ids)
        step = DecompNode(
            node_id="inductive_step", level=1,
            goal_text=("assuming IH, length ((Cons x xs) ++ ys) "
                       "= length (Cons x xs) + length ys"),
            goal_ast=_goal_ast(),
            children=axiom_ids,
            inductive_hypothesis="base_case")   # IH clamped as a lemma via I
        theorem = DecompNode(
            node_id="theorem", level=2,
            goal_text=_GOAL_TEXT, goal_ast=THEOREM.goal_ast,
            children=("base_case", "inductive_step"))
        return Decomposition(nodes=[*axiom_nodes, base, step, theorem],
                             root_id="theorem")

    if variant == "fallback":
        combined = DecompNode(
            node_id="combined_induction", level=1,
            goal_text="base and inductive step proved jointly",
            goal_ast=_goal_ast(), children=axiom_ids)
        theorem = DecompNode(
            node_id="theorem", level=2,
            goal_text=_GOAL_TEXT, goal_ast=THEOREM.goal_ast,
            children=("combined_induction",))
        return Decomposition(nodes=[*axiom_nodes, combined, theorem],
                             root_id="theorem")

    raise ValueError(f"unknown decomposition variant {variant!r}")
```

> If M1's `ast.py` does not expose `App`, `TList`, `TNat` with these constructors, this is
> a §7.6 reconciliation: adjust the AST builders to the shipped M1 surface in `_deps.py` or
> here — do **not** invent AST nodes. Stop-and-ask if the extended calculus genuinely
> cannot express named operator constants.

- [ ] **Step 4: Run the test, verify it passes**

```
.venv/bin/python -m pytest tests/test_theorem1_list_induction.py -v
```

- [ ] **Step 5: Commit**

```
git add src/qft_pcn/composition/theorems/__init__.py src/qft_pcn/composition/theorems/theorem1_list_induction.py tests/test_theorem1_list_induction.py
git commit -m "$(cat <<'EOF'
feat(composition/theorems): frozen theorem-1 list-induction decomposition

theorem1_list_induction defines the firm-target theorem (spec 3.3): the
goal forall xs ys. length(xs++ys)=length xs+length ys, its six defining
axioms, and the fixed 3-level decomposition (axioms -> base/step lemmas
-> theorem) plus an LLM-free fallback variant for K's revision hook.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Theorems 2-4 — frozen stretch definitions

**Files:**
- Create: `src/qft_pcn/composition/theorems/theorem2_binomial.py`
- Create: `src/qft_pcn/composition/theorems/theorem3_prime_order_cyclic.py`
- Create: `src/qft_pcn/composition/theorems/theorem4_rank_nullity.py`
- Test: `tests/test_stretch_theorems.py` (create)

Theorems 2-4 are stretch (spec §3.3). They ship with hand-authored decompositions and
carry **no obligation to verify**. This task only builds the definitions; the demo will
attempt them and honestly report.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stretch_theorems.py`:

```python
"""Theorems 2-4 are STRETCH targets (spec 3.3). This test only checks the
definitions are well-formed decompositions -- it does NOT assert they verify."""
from __future__ import annotations
import pytest
from src.qft_pcn.composition.theorems import (
    theorem2_binomial as t2,
    theorem3_prime_order_cyclic as t3,
    theorem4_rank_nullity as t4,
)


@pytest.mark.parametrize("mod", [t2, t3, t4])
def test_stretch_theorem_has_decomposition(mod):
    dec = mod.build_decomposition()
    assert dec.is_acyclic()
    assert dec.root().level >= 2
    assert any(n.level == 0 for n in dec.nodes)


@pytest.mark.parametrize("mod", [t2, t3, t4])
def test_stretch_theorem_has_goal_text(mod):
    assert isinstance(mod.THEOREM.goal_text, str) and mod.THEOREM.goal_text
```

- [ ] **Step 2: Run the test, verify it fails**

```
.venv/bin/python -m pytest tests/test_stretch_theorems.py -v
```

- [ ] **Step 3: Implement the three stretch-theorem modules**

Each reuses `theorem1_list_induction`'s `Theorem`, `Axiom`, `DecompNode`, `Decomposition`
dataclasses (import them; do not redefine). Each provides `THEOREM` and
`build_decomposition()` with the spec §3.3 decomposition:

- `theorem2_binomial.py` — `(a+b)^2 = a^2 + 2ab + b^2`. Level 0: commutative-ring axioms
  (assoc, comm, distrib of `+` and `*`). Level 1: the distributive expansion
  `(a+b)*(a+b) = a*a + a*b + b*a + b*b`. Level 2: the regrouping to `a^2 + 2ab + b^2`.
- `theorem3_prime_order_cyclic.py` — every group of prime order is cyclic. Level 0: group
  axioms. Level 1: Lagrange's theorem (supplied as a single registered library lemma node,
  `goal_text` documenting it is assumed-from-library, `children=()`). Level 2: the
  order-of-a-non-identity-element argument.
- `theorem4_rank_nullity.py` — `rank(T) + nullity(T) = dim(domain)`. Level 0: vector-space
  and dimension axioms. Level 1: the basis-extension lemma. Level 2: the
  direct-sum / quotient argument.

Each `build_decomposition` builds a valid DAG with the §3.3 structure. The AST `goal_ast`
fields use M1's extended calculus where it reaches; where the extended calculus genuinely
cannot express the proposition (likely for theorems 3 and 4), set `goal_ast=None` and a
`goal_text` documenting the gap — the demo's encodability gate (§5.1 step 3) then records
`unencodable` honestly. **Do not invent AST nodes to force encodability.**

- [ ] **Step 4: Run the test, verify it passes**

```
.venv/bin/python -m pytest tests/test_stretch_theorems.py -v
```

- [ ] **Step 5: Commit**

```
git add src/qft_pcn/composition/theorems/theorem2_binomial.py src/qft_pcn/composition/theorems/theorem3_prime_order_cyclic.py src/qft_pcn/composition/theorems/theorem4_rank_nullity.py tests/test_stretch_theorems.py
git commit -m "$(cat <<'EOF'
feat(composition/theorems): frozen stretch theorems 2-4

theorem2 (binomial identity from ring axioms), theorem3 (prime-order
group is cyclic), theorem4 (rank-nullity). Hand-authored decompositions
only; no obligation to verify (spec 3.3). goal_ast is None where the M1
extended calculus genuinely cannot express the proposition, so the demo's
encodability gate reports 'unencodable' honestly instead of faking it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: The demo pipeline — encodability gate, dispatch, verify, tally

**Files:**
- Create: `src/qft_pcn/composition/demo_hierarchical_proof.py`
- Modify: `src/qft_pcn/composition/__init__.py` (export `run_demo`)
- Create/Modify: `tests/conftest.py` (memory ceiling fixture, from M1 §12.10)
- Test: `tests/test_hierarchical_proof_demo.py` (create)

This task wires Tasks 0-3 into the runnable demo (spec §5).

- [ ] **Step 1: Write the failing test**

Create `tests/test_hierarchical_proof_demo.py` — the spec §6.1, §6.3, §6.4, §6.5 suite:

```python
"""End-to-end acceptance for the hierarchical proof composition demo
(spec section 6). Theorem 1 MUST verify; theorems 2-4 are attempted with an
honest tally."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.composition.demo_hierarchical_proof import run_demo
from src.qft_pcn.composition.proof_tree import verify_proof_tree


# --- 6.1 theorem 1 end-to-end (REQUIRED -- the milestone) ---------------------
def test_theorem1_list_induction_verified():
    report = run_demo(theorem_ids=(1,), rng_seed=20260522)
    t1 = report.per_theorem["theorem1_list_induction"]
    tree = t1.proof_tree
    # 3 levels of decomposition
    assert {n.level for n in tree.nodes.values()} == {0, 1, 2}
    # every node a sub-QPCN ground state at residual <= 1e-8
    for node in tree.nodes.values():
        assert node.residual_energy is not None
        assert node.residual_energy <= 1e-8
        assert node.classical_check is True
    # every internal node's composition residual <= 1e-8
    for node in tree.nodes.values():
        if node.children:
            assert node.compose_residual is not None
            assert node.compose_residual <= 1e-8
    # the verification predicate agrees
    assert verify_proof_tree(tree).verdict is True
    # chi_max <= 32 per sub-QPCN
    assert all(c <= 32 for c in t1.chi_used_per_node.values())
    # single-machine runtime: generous CI ceiling, not a tight prediction
    assert report.total_runtime_s < 300.0


# --- 6.3 honest tally (REQUIRED that the reporting is honest) -----------------
def test_theorems_2_3_4_honest_tally():
    report = run_demo(theorem_ids=(1, 2, 3, 4), rng_seed=20260522)
    # the solved count EQUALS the number of theorems with a verified tree
    verified = sum(
        1 for t in report.per_theorem.values()
        if t.proof_tree is not None and verify_proof_tree(t.proof_tree).verdict
    )
    assert report.solved_count == verified
    assert report.target_count == 4
    # a theorem reported VERIFIED must actually be verified
    for t in report.per_theorem.values():
        if t.verdict_label == "VERIFIED":
            assert verify_proof_tree(t.proof_tree).verdict is True
        else:
            assert (t.proof_tree is None
                    or verify_proof_tree(t.proof_tree).verdict is False)


# --- 6.4 no-shortcut structural markers (REQUIRED) ----------------------------
def test_shared_var_is_entanglement():
    report = run_demo(theorem_ids=(1,), rng_seed=20260522)
    t1 = report.per_theorem["theorem1_list_induction"]
    # the level-2 composition state: positive entropy across the shared-ys cut
    assert t1.shared_var_entropy is not None
    assert t1.shared_var_entropy > 1e-6        # strictly positive => entanglement


def test_einsum_greedy():
    # every einsum in L's own source carries optimize='greedy'
    import pathlib, re
    root = pathlib.Path("src/qft_pcn/composition")
    files = [root / "demo_hierarchical_proof.py", root / "proof_tree.py"]
    for f in files:
        src = f.read_text()
        for m in re.finditer(r"\.einsum\(", src):
            tail = src[m.start():m.start() + 400]
            assert "optimize=" in tail and "greedy" in tail, f"non-greedy einsum in {f}"


# --- 6.5 reproducibility (REQUIRED) -------------------------------------------
def test_theorem1_reproducible():
    r1 = run_demo(theorem_ids=(1,), rng_seed=20260522)
    r2 = run_demo(theorem_ids=(1,), rng_seed=20260522)
    t1a = r1.per_theorem["theorem1_list_induction"]
    t1b = r2.per_theorem["theorem1_list_induction"]
    # same structure, same verdict; energies may differ in the last digits
    assert set(t1a.proof_tree.nodes) == set(t1b.proof_tree.nodes)
    assert (verify_proof_tree(t1a.proof_tree).verdict
            == verify_proof_tree(t1b.proof_tree).verdict)
```

The `test_no_dense_blowup` marker (spec §6.4) is enforced by the `conftest.py` memory
ceiling, which fails any test that materializes a tensor larger than `16**2`.

- [ ] **Step 2: Run the test, verify it fails**

```
.venv/bin/python -m pytest tests/test_hierarchical_proof_demo.py -v
```

- [ ] **Step 3: Add the memory-ceiling `conftest.py`**

If `tests/conftest.py` does not exist, create it with the M1 §12.10 memory-ceiling fixture
(autouse, asserts no numpy array larger than `16**2 = 256` elements per axis / total
`16**2` for a dense window operator is allocated by L's own code). If it exists, confirm
the ceiling is present and reuse it. The ceiling guards spec §6.4 `test_no_dense_blowup`.

- [ ] **Step 4: Implement the demo**

Create `src/qft_pcn/composition/demo_hierarchical_proof.py`. It implements spec §5: the
per-theorem pipeline (load, build goal graph, encodability gate, bottom-up dispatch,
record, revision on failure, verify) and the §5.3 honest tally. Complete implementation:

```python
"""Hierarchical proof composition demo -- sub-project L (architecture 10.11).

Takes a mathematical theorem, builds a goal graph, dispatches one sub-QPCN per
proof-tree node (M3 relax_to_ground, chi_max=32), integrates results via the
lemma machinery (I), and surfaces a VERIFIED proof tree -- verified meaning
every node's sub-QPCN ground state has residual energy <= 1e-8 (architecture
13.2). Ends with an honest 'solved X of 4' tally.

Run:  python -m src.qft_pcn.composition.demo_hierarchical_proof
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field

from src.qft_pcn.composition.proof_tree import (
    ProofNode, ProofTree, verify_proof_tree, EPS_VERIFY,
)
from src.qft_pcn.composition._deps import (
    encode_mera, compile_node_hamiltonian, relax_to_ground,
    LemmaLibrary, use_lemma, GoalGraph, Dispatcher, ResultIntegrator, Revision,
    shared_cut_entropy,
)
from src.qft_pcn.composition.theorems import (
    theorem1_list_induction as t1mod,
    theorem2_binomial as t2mod,
    theorem3_prime_order_cyclic as t3mod,
    theorem4_rank_nullity as t4mod,
)

CHI_MAX = 32
_THEOREM_MODULES = {1: t1mod, 2: t2mod, 3: t3mod, 4: t4mod}


@dataclass
class TheoremResult:
    theorem_id: str
    proof_tree: ProofTree | None
    verdict_label: str                 # VERIFIED | NOT VERIFIED | UNENCODABLE | NOT ATTEMPTED
    chi_used_per_node: dict[str, int] = field(default_factory=dict)
    shared_var_entropy: float | None = None
    failed_node: str | None = None


@dataclass
class DemoReport:
    per_theorem: dict[str, TheoremResult]
    solved_count: int
    target_count: int
    total_runtime_s: float

    def render(self) -> str:
        lines = ["=== Hierarchical Proof Composition Demo -- results ==="]
        for tr in self.per_theorem.values():
            lines.append(f"{tr.theorem_id:34s}: {tr.verdict_label}")
            if tr.proof_tree is not None:
                rep = verify_proof_tree(tr.proof_tree)
                for row in rep.per_node:
                    e = ("    " if row.residual_energy is None
                         else f"{row.residual_energy:.2e}")
                    lines.append(f"    L{row.level} {row.node_id:24s} "
                                 f"residual={e} verdict={row.verdict}")
        lines.append("-" * 55)
        lines.append(f"solved {self.solved_count} of {self.target_count} target theorems")
        lines.append(f"total measured runtime: {self.total_runtime_s:.1f} s on this machine")
        return "\n".join(lines)


def _solve_node(dnode, library: LemmaLibrary, solved: dict[str, ProofNode],
                rng_seed: int) -> ProofNode:
    """Run one sub-QPCN for one decomposition node and build its ProofNode.

    The node's ground state is found by imaginary-time relaxation (M3). The
    node is 'solved' only if relax_to_ground reports residual_energy <= 1e-8.
    A non-convergence is recorded as 'unverified' -- never swallowed, never
    faked (spec 1.4, 1.5)."""
    # encode the node's goal on the M1 substrate
    state0, meta = encode_mera(dnode.goal_ast, n_nodes_max=32, chi_layer=16)
    # compile the node's Hamiltonian (M2): typing + evaluation + induction terms
    H = compile_node_hamiltonian(dnode, meta)
    # clamp already-solved children as lemmas via I (spec 4.3, architecture 10.8)
    for cid in dnode.children:
        child = solved.get(cid)
        if child is not None and child.status == "solved":
            H = use_lemma(H, library, child.state_ref, meta)
    if dnode.inductive_hypothesis is not None:
        ih = solved.get(dnode.inductive_hypothesis)
        if ih is not None and ih.status == "solved":
            H = use_lemma(H, library, ih.state_ref, meta)
    # the sub-QPCN run
    result = relax_to_ground(H, chi_max=CHI_MAX, rng_seed=rng_seed)
    energy = float(result.residual_energy)
    solved_ok = energy <= EPS_VERIFY
    classical = False
    if solved_ok:
        # I's registration type-check -- the necessary cross-check (spec 1.4)
        classical = library.register(result.state, dnode.goal_text,
                                     metadata={"node_id": dnode.node_id})
    return ProofNode(
        node_id=dnode.node_id, level=dnode.level, goal=dnode.goal_text,
        children=tuple(dnode.children),
        state_ref=result.state if solved_ok else None,
        residual_energy=energy,
        compose_residual=None,             # filled by the integrator for internal nodes
        classical_check=classical if solved_ok else False,
        status="solved" if (solved_ok and classical) else "unverified",
    )


def _run_theorem(theorem_id: int, rng_seed: int) -> TheoremResult:
    mod = _THEOREM_MODULES[theorem_id]
    tid = mod.THEOREM.theorem_id
    decomposition = mod.build_decomposition()

    # --- encodability gate (spec 5.1 step 3) ---
    for dnode in decomposition.nodes:
        if dnode.goal_ast is None:
            return TheoremResult(tid, None, "UNENCODABLE", failed_node=dnode.node_id)
        try:
            encode_mera(dnode.goal_ast, n_nodes_max=32, chi_layer=16)
        except Exception:                  # M1 encoding exception family
            return TheoremResult(tid, None, "UNENCODABLE", failed_node=dnode.node_id)

    # --- build the goal graph (K) ---
    graph = GoalGraph.from_decomposition(decomposition)
    dispatcher = Dispatcher(graph)
    integrator = ResultIntegrator()
    library = LemmaLibrary()

    # --- bottom-up dispatch (spec 5.1 step 4): level 0, then 1, then 2 ---
    solved: dict[str, ProofNode] = {}
    chi_used: dict[str, int] = {}
    levels = sorted({n.level for n in decomposition.nodes})
    for level in levels:
        for dnode in [n for n in decomposition.nodes if n.level == level]:
            pn = _solve_node(dnode, library, solved, rng_seed)
            # revision on failure (spec 5.1 step 6): one fixed fallback, no LLM
            if pn.status != "solved":
                alt = Revision.fallback_for(dnode, mod)
                if alt is not None:
                    pn = _solve_node(alt, library, solved, rng_seed)
            # composition residual for internal nodes (architecture 13.3.1)
            if dnode.children:
                child_states = [solved[c].state_ref for c in dnode.children
                                if c in solved and solved[c].state_ref is not None]
                comp = integrator.compose_residual(pn.state_ref, child_states)
                pn = ProofNode(
                    node_id=pn.node_id, level=pn.level, goal=pn.goal,
                    children=pn.children, state_ref=pn.state_ref,
                    residual_energy=pn.residual_energy,
                    compose_residual=(None if comp is None else float(comp)),
                    classical_check=pn.classical_check, status=pn.status)
            solved[pn.node_id] = pn
            chi_used[pn.node_id] = CHI_MAX

    tree = ProofTree(theorem_id=tid, root_id=decomposition.root_id, nodes=solved)
    verdict = verify_proof_tree(tree).verdict
    label = "VERIFIED" if verdict else "NOT VERIFIED"

    # shared-variable entanglement marker (spec 6.4)
    root = solved.get(decomposition.root_id)
    entropy = None
    if root is not None and root.state_ref is not None and root.children:
        child_states = [solved[c].state_ref for c in root.children
                        if c in solved and solved[c].state_ref is not None]
        entropy = shared_cut_entropy(root.state_ref, child_states)

    return TheoremResult(tid, tree, label, chi_used_per_node=chi_used,
                         shared_var_entropy=entropy)


def run_demo(theorem_ids=(1, 2, 3, 4), rng_seed: int = 20260522) -> DemoReport:
    """Attempt each target theorem in order; return an honest tally."""
    t_start = time.perf_counter()
    per_theorem: dict[str, TheoremResult] = {}
    for tid in theorem_ids:
        per_theorem[_THEOREM_MODULES[tid].THEOREM.theorem_id] = _run_theorem(tid, rng_seed)
    solved = sum(
        1 for tr in per_theorem.values()
        if tr.proof_tree is not None and verify_proof_tree(tr.proof_tree).verdict
    )
    return DemoReport(
        per_theorem=per_theorem, solved_count=solved,
        target_count=len(theorem_ids),
        total_runtime_s=time.perf_counter() - t_start,
    )


def main() -> None:
    report = run_demo()
    print(report.render())


if __name__ == "__main__":
    main()
```

> `compile_node_hamiltonian`, `shared_cut_entropy`, `Revision.fallback_for`,
> `GoalGraph.from_decomposition`, `ResultIntegrator.compose_residual`,
> `LemmaLibrary.register` returning a bool, and `relax_to_ground`'s `RelaxResult` shape are
> the §7 contract. Where the shipped M2/I/K APIs differ, adapt them in `_deps.py` (§7.6) —
> a thin mapping, never a fake. If `compile_node_hamiltonian` (the M2 induction-as-
> ground-state compiler) is genuinely absent, **stop and ask** — the demo cannot run a
> real sub-QPCN without it, and a fake Hamiltonian violates §1.5.

Then add to `src/qft_pcn/composition/__init__.py`:

```python
from src.qft_pcn.composition.demo_hierarchical_proof import run_demo, DemoReport

__all__ = ["run_demo", "DemoReport"]
```

- [ ] **Step 5: Run the test, verify it passes**

```
.venv/bin/python -m pytest tests/test_hierarchical_proof_demo.py -v
```

If theorem 1 does not verify, this is a genuine failure to diagnose with
superpowers:systematic-debugging — **not** an invitation to loosen `ε_verify` or fake a
ground state (§1.5). If theorems 2-4 do not verify, that is expected and acceptable; the
test only requires the *tally* to be honest, not that they pass.

- [ ] **Step 6: Commit**

```
git add src/qft_pcn/composition/demo_hierarchical_proof.py src/qft_pcn/composition/__init__.py tests/conftest.py tests/test_hierarchical_proof_demo.py
git commit -m "$(cat <<'EOF'
feat(composition/demo_hierarchical_proof): runnable hierarchical proof demo

run_demo builds a goal graph per theorem, dispatches one sub-QPCN per
proof-tree node (M3 relax_to_ground, chi_max=32), clamps solved children
as lemmas via I, integrates via K, and verifies the proof tree by the
zero-residual-energy criterion (architecture 13.2). Prints an honest
'solved X of 4' tally with measured runtime.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Full-suite verification and the runnable demo

**Files:** none new — verification only.

- [ ] **Step 1: Run the complete L test suite**

```
.venv/bin/python -m pytest tests/test_l_deps.py tests/test_proof_tree.py tests/test_theorem1_list_induction.py tests/test_stretch_theorems.py tests/test_hierarchical_proof_demo.py -v
```

All must pass. Theorem 1 verified; the honest-tally test consistent.

- [ ] **Step 2: Confirm no dependency regressed**

```
.venv/bin/python -m pytest src/qft_pcn/logic src/qft_pcn/composition src/qft_pcn/qft -v
```

L modified no M1-K source; their suites must still be green. A red dependency suite is a
stop-and-investigate, not a skip.

- [ ] **Step 3: Run the demo end-to-end**

```
.venv/bin/python -m src.qft_pcn.composition.demo_hierarchical_proof
```

Confirm it prints the proof tree(s) and an honest `solved X of 4` line with a measured
runtime. The printed tally must equal the verified-tree count.

- [ ] **Step 4: Verification-before-completion**

Use superpowers:verification-before-completion. Do not claim L complete until the §10
acceptance criteria are each backed by fresh pytest output pasted into the completion
report. The completion report states the true `solved X of 4` — if it is `solved 1 of 4`,
report `solved 1 of 4`. A report claiming more than the test output shows is a §1.5
violation.

- [ ] **Step 5: Final commit (if any verification-driven fixes were made)**

```
git add -A -- src/qft_pcn/composition tests
git commit -m "$(cat <<'EOF'
test(composition): full sub-project L acceptance suite green

Theorem 1 (list-induction) verified end-to-end by 3-level decomposition,
every node a sub-QPCN ground state at residual <= 1e-8, chi_max=32 on a
single machine. Theorems 2-4 attempted with an honest tally.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Acceptance criteria (from spec §10)

L is complete when, with fresh pytest output as evidence:

1. `test_theorem1_list_induction_verified` passes — theorem 1 verified, 3-level
   decomposition, every node a sub-QPCN ground state at `residual ≤ 1e-8`, `χ_max ≤ 32`,
   single machine.
2. All verification-predicate tests (`test_proof_tree.py`) pass.
3. `test_theorems_2_3_4_honest_tally` passes — the `solved X of 4` count is consistent
   with the per-theorem verdicts.
4. `test_shared_var_is_entanglement`, `test_einsum_greedy`, and the `conftest.py` dense
   ceiling all pass.
5. `test_theorem1_reproducible` passes.
6. `python -m src.qft_pcn.composition.demo_hierarchical_proof` runs end-to-end and prints
   the proof tree(s) and an honest tally with measured runtime.
7. M1/M2/M3/I/J/K suites still pass.
8. Every completion claim is backed by fresh pytest output.

**Honest expectation:** theorem 1 is the firm milestone and must verify. Theorems 2-4 are
stretch; `solved 1 of 4` is an acceptable, publishable outcome. `solved 4 of 4` is
acceptable only if every sub-QPCN ground state genuinely reached `residual ≤ 1e-8`.
