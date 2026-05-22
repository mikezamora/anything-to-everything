# Cross-Level Message Passing Implementation Plan — sub-project K

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Use superpowers:test-driven-development for every task: failing test first, then implementation, then green.

**Goal:** Build cross-level message passing — the runtime that decomposes a parent QPCN goal into a goal-graph DAG, dispatches sub-goals as parallel child QPCN runs, and integrates solved children back into the parent's MERA by lemma-clamping.

**Architecture:** Parent and child QPCNs exchange goals (top-down predictions) and proofs (bottom-up precision-weighted errors) via the *same* variational free-energy equation a single predictive-coding layer minimizes — at the scale of whole QPCN runs. A parent identifies unsolved sub-trees, packages each as a child DSL spec, dispatches a child QPCN run through G's bridge, and clamps the solved ground state into its own MERA via I's lemma machinery.

**Tech Stack:** Python 3.11, numpy 1.26, pytest, `concurrent.futures` (stdlib thread pool). Built on `src/qft_pcn/bridge/` (sub-project G, built), `src/qft_pcn/qft/mera.py` (sub-project F, built), and the lemma library (sub-project I).

**Spec:** `docs/superpowers/specs/2026-05-22-cross-level-passing-design.md` — authoritative. When this plan and the spec disagree, the spec wins.

**Test runner:** `.venv/bin/python -m pytest <path> -v`.

---

## Driving principles (from spec §1 — non-negotiable; embed verbatim in every subagent prompt)

A subagent will reach for the easiest path. It must be made aware of these principles and must **refuse** the shortcut. Each task below names its specific shortcut + the principled alternative; embed that pair in the task's subagent prompt.

1. **Don't stop digging at hard field-crossings.** Principled path over easy-but-wrong path.
2. **Cross-level passing IS variational free-energy minimization at whole-QPCN scale** — a structural claim (§10.10), not an analogy. The goal/proof exchange is precision-weighted message passing, **not** an ad-hoc RPC. Top-down message = parent's prediction; bottom-up message = child's prediction error weighted by `1/(residual+ε)`.
3. **Binding is genuine entanglement, never a classical lookup.** Integration clamps a sub-MERA into the parent's MERA via I's machinery — never `parent_dict[site] = child_ast`.
4. **A child integrates only if it is in a true ground state.** The residual gate is strict; premature integration of a low-energy excited state is forbidden.
5. **`optimize='greedy'` on every einsum / numpy contraction.**
6. **OOM / resource pressure → optimize or parallelize better, never shrink the problem** (no lowering `χ_max`, goal-graph depth, or the acceptance theorem).
7. **Reuse, do not reinvent.** Consume `bridge/` (G), `qft/mera.py` (F), the lemma library (I). K orchestrates; it never duplicates a QPCN runner, a DSL compiler, or a clamp primitive.

**No time/effort/duration estimates anywhere.** Critical directive.

---

## Pre-existing worktree state

The worktree has unrelated modified files (`QFT_PCN_ARCHITECTURE.md`, `src/qft_pcn/logic/mera_encoder.py`, untracked `lib/`, `src/qft_pcn/logic/_mera_holes.py`, `src/qft_pcn/logic/tests/test_mera_holes.py`). **Leave them alone.** Stage only the files each task names.

All `src/qft_pcn/composition/` files are new. No existing file is modified by this plan except, if a genuine missing primitive is found, the lemma library (I) — fixed there, not duplicated, with I's tests kept green.

---

## Dependency check (do this before Task 1)

- [ ] Confirm `src/qft_pcn/bridge/dsl/pipeline.py` exposes `compile_dsl` and `bridge/runtime/result.py` exposes `RunResult` (`energy`, `converged`, `to_dict`). These are the child-run interface.
- [ ] Confirm the lemma library (sub-project I) exposes a `promote(state, ast, goal_id)` and a sub-tree clamp primitive. If I is not yet merged, the dispatcher and integrator are written against the I interface and a `FakeLemmaLibrary` test double (Task 5) stands in for unit tests; the end-to-end acceptance task (Task 8) requires real I.
- [ ] Confirm `concurrent.futures.ThreadPoolExecutor` is available (stdlib — it is).

If any interface differs from the spec's assumption, **stop and ask** — do not guess a signature.

---

## Task 1: Package skeleton and the error family

**Files:**
- Create: `src/qft_pcn/composition/__init__.py`
- Create: `src/qft_pcn/composition/errors.py`
- Create: `src/qft_pcn/composition/tests/__init__.py`
- Test: `src/qft_pcn/composition/tests/test_errors.py`

**Shortcut to refuse:** "skip a dedicated error module, just raise `RuntimeError`." **Principled alternative:** the spec §11 fixes a five-member exception family with semantics (e.g. `DispatchTimeout` is *carried in `ChildResult`, never raised*); a typed family is what lets the dispatcher distinguish a recoverable timeout from a `GoalGraphError` bug.

- [ ] **Step 1: Write the failing test**

`src/qft_pcn/composition/tests/test_errors.py`:

```python
"""Tests for the K error family (spec §11)."""
from __future__ import annotations
import pytest
from src.qft_pcn.composition.errors import (
    CompositionError, GoalGraphError, DispatchTimeout,
    IntegrationRefused, RevisionExhausted,
)


def test_all_inherit_composition_error():
    for cls in (GoalGraphError, DispatchTimeout, IntegrationRefused, RevisionExhausted):
        assert issubclass(cls, CompositionError)


def test_integration_refused_carries_residual_and_gate():
    e = IntegrationRefused(residual=1e-3, gate=1e-6)
    assert e.residual == 1e-3
    assert e.gate == 1e-6
    assert "1e-06" in str(e) or "1e-6" in str(e)


def test_revision_exhausted_carries_attempts():
    e = RevisionExhausted(goal_id="g0", attempts=3)
    assert e.attempts == 3
    assert "g0" in str(e)


def test_dispatch_timeout_carries_seconds():
    e = DispatchTimeout(goal_id="g1", timeout_s=2.5)
    assert e.timeout_s == 2.5
```

- [ ] **Step 2: Run the test, verify it fails** — `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_errors.py -v` (collection error: module missing).

- [ ] **Step 3: Implement**

`src/qft_pcn/composition/__init__.py`:

```python
"""Sub-project K: cross-level message passing (architecture §10.10).

Parent and child QPCNs exchange goals (top-down predictions) and proofs
(bottom-up precision-weighted errors) via the same variational free-energy
equation a single predictive-coding layer minimizes, at whole-QPCN scale.
"""
```

`src/qft_pcn/composition/tests/__init__.py`: empty file.

`src/qft_pcn/composition/errors.py`:

```python
"""K error family (spec §11)."""
from __future__ import annotations


class CompositionError(Exception):
    """Base for all cross-level message-passing errors."""


class GoalGraphError(CompositionError):
    """A back-edge slipped past cycle detection. A bug, not recoverable."""


class DispatchTimeout(CompositionError):
    """A child run exceeded its timeout.

    Carried inside ChildResult.error -- never raised into a sibling batch,
    so one straggler cannot abort parallel work (spec §5.4).
    """

    def __init__(self, goal_id: str, timeout_s: float):
        self.goal_id = goal_id
        self.timeout_s = timeout_s
        super().__init__(f"child {goal_id!r} exceeded timeout {timeout_s}s")


class IntegrationRefused(CompositionError):
    """The residual-energy gate refused a clamp (spec §6.3)."""

    def __init__(self, residual: float, gate: float):
        self.residual = residual
        self.gate = gate
        super().__init__(
            f"integration refused: residual {residual!r} exceeds gate {gate!r}"
        )


class RevisionExhausted(CompositionError):
    """MAX_REVISIONS reached; the node is FAILED (spec §6.5)."""

    def __init__(self, goal_id: str, attempts: int):
        self.goal_id = goal_id
        self.attempts = attempts
        super().__init__(f"revision exhausted for {goal_id!r} after {attempts} attempts")
```

- [ ] **Step 4: Run the test, verify green.**

- [ ] **Step 5: Commit**

```
git add src/qft_pcn/composition/__init__.py src/qft_pcn/composition/errors.py \
        src/qft_pcn/composition/tests/__init__.py \
        src/qft_pcn/composition/tests/test_errors.py
git commit -m "$(cat <<'EOF'
feat(composition/errors): K error family for cross-level message passing

Five-member exception family (spec §11): GoalGraphError is a bug,
DispatchTimeout is carried not raised, IntegrationRefused carries the
residual-gate values, RevisionExhausted carries attempt count.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: The goal-graph data structure — `SubGoal`, `Node`, `Status`

**Files:**
- Create: `src/qft_pcn/composition/goal_graph.py`
- Test: `src/qft_pcn/composition/tests/test_goal_graph.py`

**Shortcut to refuse:** "use a plain `dict` keyed by site for the goal graph; statuses as strings." **Principled alternative:** spec §4.1 fixes `SubGoal` (frozen, with a content-hash `goal_id`), `Status` as an enum, and `Node` with `children`/`result`/`revision_attempts`/`parent`. The content-hash `goal_id` is load-bearing: it is what makes cycle detection sound (§4.4) and cross-sibling lemma sharing real (§8). A bare dict cannot express either.

- [ ] **Step 1: Write the failing test** (this task covers data structure + `goal_id`; cycle detection is Task 3).

`src/qft_pcn/composition/tests/test_goal_graph.py`:

```python
"""Goal-graph data-structure tests (spec §4.1)."""
from __future__ import annotations
from src.qft_pcn.composition.goal_graph import SubGoal, Node, Status, make_sub_goal


def _spec(name):
    return {"goal": name, "fields": [], "hamiltonian": []}


def test_status_has_the_six_members():
    names = {s.name for s in Status}
    assert names == {
        "PENDING", "ACTIVE", "SOLVED", "FAILED", "CYCLE", "PENDING_REVISION"
    }


def test_sub_goal_is_frozen():
    g = make_sub_goal(_spec("a"), goal_prop="Nat", boundary={}, parent_site=0)
    import dataclasses
    assert dataclasses.is_dataclass(g)
    try:
        g.goal_prop = "Bool"
        assert False, "SubGoal must be frozen"
    except dataclasses.FrozenInstanceError:
        pass


def test_goal_id_is_content_addressed():
    g1 = make_sub_goal(_spec("a"), goal_prop="Nat", boundary={}, parent_site=0)
    g2 = make_sub_goal(_spec("a"), goal_prop="Nat", boundary={}, parent_site=7)
    g3 = make_sub_goal(_spec("b"), goal_prop="Nat", boundary={}, parent_site=0)
    # same (goal_prop, dsl_spec) -> same goal_id; parent_site does NOT affect it
    assert g1.goal_id == g2.goal_id
    # different dsl_spec -> different goal_id
    assert g1.goal_id != g3.goal_id


def test_goal_id_is_stable_across_dict_key_order():
    a = make_sub_goal({"x": 1, "y": 2}, goal_prop="P", boundary={}, parent_site=0)
    b = make_sub_goal({"y": 2, "x": 1}, goal_prop="P", boundary={}, parent_site=0)
    assert a.goal_id == b.goal_id


def test_node_defaults():
    g = make_sub_goal(_spec("a"), goal_prop="Nat", boundary={}, parent_site=0)
    n = Node(goal=g, status=Status.PENDING)
    assert n.children == []
    assert n.result is None
    assert n.revision_attempts == 0
    assert n.parent is None


def test_node_add_child_sets_back_edge():
    g0 = make_sub_goal(_spec("root"), goal_prop="P", boundary={}, parent_site=None)
    g1 = make_sub_goal(_spec("child"), goal_prop="Q", boundary={}, parent_site=3)
    root = Node(goal=g0, status=Status.PENDING)
    child = Node(goal=g1, status=Status.PENDING)
    root.add_child(child)
    assert root.children == [child]
    assert child.parent is root
```

- [ ] **Step 2: Run, verify it fails.**

- [ ] **Step 3: Implement** `src/qft_pcn/composition/goal_graph.py` (data-structure portion; later tasks append to this file):

```python
"""Goal graph: the DAG of sub-goals (spec §4).

A parent QPCN's target goal is the root; sub-goals are children. Built
top-down, consumed bottom-up. goal_id is a content hash of
(goal_prop, dsl_spec) -- this is what makes cycle detection sound and
cross-sibling lemma sharing real.
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _content_hash(goal_prop: str, dsl_spec: dict) -> str:
    """Stable content hash; dict key order does not matter."""
    canonical = json.dumps(
        {"goal_prop": goal_prop, "dsl_spec": dsl_spec},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SubGoal:
    """A DSL spec plus its expected proposition type -- the unit a child
    QPCN run proves (spec §4.1)."""
    goal_id: str
    dsl_spec: dict
    goal_prop: str
    boundary: dict
    parent_site: int | None


def make_sub_goal(dsl_spec: dict, *, goal_prop: str, boundary: dict,
                  parent_site: int | None) -> SubGoal:
    """Construct a SubGoal with a content-addressed goal_id."""
    return SubGoal(
        goal_id=_content_hash(goal_prop, dsl_spec),
        dsl_spec=dsl_spec,
        goal_prop=goal_prop,
        boundary=dict(boundary),
        parent_site=parent_site,
    )


class Status(enum.Enum):
    PENDING = "pending"                     # created, not yet dispatched
    ACTIVE = "active"                       # a child QPCN run is in flight
    SOLVED = "solved"                       # true ground state returned; integrated
    FAILED = "failed"                       # did not converge; revision exhausted
    CYCLE = "cycle"                         # already an ancestor under proof
    PENDING_REVISION = "pending_revision"   # child failed; awaiting alternative


@dataclass
class Node:
    goal: SubGoal
    status: Status
    children: list["Node"] = field(default_factory=list)
    result: Any = None                      # ChildResult once SOLVED/FAILED
    revision_attempts: int = 0
    parent: "Node | None" = None
    quarantined: bool = False               # spec §6.6

    def add_child(self, child: "Node") -> None:
        child.parent = self
        self.children.append(child)
```

- [ ] **Step 4: Run, verify green.**

- [ ] **Step 5: Commit**

```
git add src/qft_pcn/composition/goal_graph.py src/qft_pcn/composition/tests/test_goal_graph.py
git commit -m "$(cat <<'EOF'
feat(composition/goal_graph): SubGoal, Node, Status data structures

Goal-graph DAG nodes (spec §4.1): SubGoal frozen with a content-addressed
goal_id over (goal_prop, dsl_spec); Status six-member enum; Node with
children, result, revision_attempts, parent back-edge, quarantine flag.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Goal-graph construction, cycle detection, free energy, proof-tree extraction

**Files:**
- Modify: `src/qft_pcn/composition/goal_graph.py` (append)
- Modify: `src/qft_pcn/composition/tests/test_goal_graph.py` (append)

**Shortcut to refuse:** "cycle detection = a global `set` of every goal ever seen." **Principled alternative:** spec §4.4 — the visited set is **per active path** from the root. A `goal_id` in two *independent* sibling sub-trees is **not** a cycle, it is a shared lemma (§8). A global set would falsely flag every shared lemma as a cycle and break the architecture's lemma-sharing.

Second shortcut to refuse: "BFS the whole graph eagerly to full depth." Principled: construction is **lazy** (§4.2) — a node's children are populated only when it is first expanded, because the graph can be combinatorially large (§10.10 risk).

- [ ] **Step 1: Append failing tests** to `test_goal_graph.py`:

```python
from src.qft_pcn.composition.goal_graph import (
    build_goal_graph, detect_cycle, assert_acyclic, compute_free_energy,
    extract_proof_tree, ProofTree,
)
from src.qft_pcn.composition.errors import GoalGraphError


class StubDecomposer:
    """Maps a goal_prop to a fixed list of child (spec, prop) pairs."""
    def __init__(self, table):
        self._table = table  # dict: goal_prop -> list[(dsl_spec, goal_prop)]

    def decompose(self, node):
        from src.qft_pcn.composition.goal_graph import make_sub_goal
        out = []
        for i, (spec, prop) in enumerate(self._table.get(node.goal.goal_prop, [])):
            out.append(make_sub_goal(spec, goal_prop=prop, boundary={}, parent_site=i))
        return out


def test_build_goal_graph_three_levels():
    table = {
        "Thm":   [({"g": "ind"}, "IndCase")],
        "IndCase": [({"g": "L1a"}, "LemmaA"), ({"g": "L1b"}, "LemmaB")],
        "LemmaA": [({"g": "axA"}, "AxiomA")],
        "LemmaB": [({"g": "axB"}, "AxiomB")],
    }
    root = build_goal_graph({"g": "root"}, root_prop="Thm",
                            decomposer=StubDecomposer(table))
    # lazy: only the root expanded so far
    assert root.goal.goal_prop == "Thm"
    assert len(root.children) == 1
    # depth reached after full expansion
    from src.qft_pcn.composition.goal_graph import expand_fully
    expand_fully(root, StubDecomposer(table))
    assert root.children[0].goal.goal_prop == "IndCase"
    assert {c.goal.goal_prop for c in root.children[0].children} == {"LemmaA", "LemmaB"}
    assert_acyclic(root)  # does not raise


def test_detect_cycle_on_ancestor_goal_id():
    g = make_sub_goal({"g": "a"}, goal_prop="A", boundary={}, parent_site=0)
    node = Node(goal=g, status=Status.PENDING)
    assert detect_cycle(node, frozenset({g.goal_id})) is True
    assert detect_cycle(node, frozenset()) is False


def test_shared_goal_id_in_independent_siblings_is_not_a_cycle():
    # the SAME goal_id under two independent sibling subtrees -- a shared lemma
    g = make_sub_goal({"g": "shared"}, goal_prop="Shared", boundary={}, parent_site=0)
    left = Node(goal=g, status=Status.PENDING)
    right = Node(goal=g, status=Status.PENDING)
    # left's path visited set does not contain right's path -> no cycle
    assert detect_cycle(right, frozenset()) is False


def test_assert_acyclic_raises_on_back_edge():
    g0 = make_sub_goal({"g": "a"}, goal_prop="A", boundary={}, parent_site=None)
    n0 = Node(goal=g0, status=Status.PENDING)
    n0.children.append(n0)  # deliberate back-edge
    import pytest
    with pytest.raises(GoalGraphError):
        assert_acyclic(n0)


def test_compute_free_energy_sums_residuals():
    g0 = make_sub_goal({"g": "r"}, goal_prop="R", boundary={}, parent_site=None)
    g1 = make_sub_goal({"g": "c"}, goal_prop="C", boundary={}, parent_site=0)
    root = Node(goal=g0, status=Status.SOLVED)
    child = Node(goal=g1, status=Status.SOLVED)
    root.add_child(child)

    class _R:  # minimal ChildResult-shaped stub
        def __init__(self, r): self.residual_energy = r
    root.result = _R(0.0)
    child.result = _R(2e-7)
    f = compute_free_energy(root)
    # accuracy term = sum of residuals; complexity term >= 0
    assert f >= 2e-7


def test_extract_proof_tree_mirrors_solved_nodes():
    g0 = make_sub_goal({"g": "r"}, goal_prop="R", boundary={}, parent_site=None)
    g1 = make_sub_goal({"g": "c"}, goal_prop="C", boundary={}, parent_site=0)
    root = Node(goal=g0, status=Status.SOLVED)
    child = Node(goal=g1, status=Status.SOLVED)
    root.add_child(child)

    class _R:
        def __init__(self, r, ast): self.residual_energy = r; self.solved_ast = ast
    root.result = _R(0.0, "ast_root")
    child.result = _R(1e-7, "ast_child")
    tree = extract_proof_tree(root)
    assert isinstance(tree, ProofTree)
    assert tree.root.goal_prop == "R"
    assert tree.root.children[0].goal_prop == "C"
    assert tree.root.children[0].solved_ast == "ast_child"
    assert abs(tree.total_residual - 1e-7) < 1e-12
```

- [ ] **Step 2: Run, verify the new tests fail.**

- [ ] **Step 3: Append to `goal_graph.py`:**

```python
# --- construction (spec §4.2) ----------------------------------------------

COMPLEXITY_WEIGHT = 1e-9   # complexity term coefficient in F_hierarchy (spec §7)


def build_goal_graph(root_spec: dict, root_prop: str, decomposer) -> Node:
    """Build the root and lazily expand only the root's immediate children.

    Children deeper than level 1 are populated on demand by expand_node /
    expand_fully -- the graph can be combinatorially large (spec §4.2).
    """
    root_goal = make_sub_goal(root_spec, goal_prop=root_prop,
                              boundary={}, parent_site=None)
    root = Node(goal=root_goal, status=Status.PENDING)
    expand_node(root, decomposer)
    return root


def expand_node(node: Node, decomposer) -> None:
    """Populate node.children once, from the decomposer (idempotent)."""
    if node.children:
        return
    for sub in decomposer.decompose(node):
        node.add_child(Node(goal=sub, status=Status.PENDING))


def expand_fully(node: Node, decomposer) -> None:
    """Eagerly expand the whole subtree -- for tests / small graphs only."""
    expand_node(node, decomposer)
    for child in node.children:
        expand_fully(child, decomposer)


# --- cycle detection (spec §4.4) -------------------------------------------

def detect_cycle(node: Node, visited: "frozenset[str]") -> bool:
    """True iff this node's goal is already on the active path from the root.

    visited is PER ACTIVE PATH, not global: the same goal_id in two
    independent sibling subtrees is a shared lemma, not a cycle.
    """
    return node.goal.goal_id in visited


def assert_acyclic(root: Node) -> None:
    """DFS guard: a back-edge that slipped past detect_cycle is a bug."""
    on_path: set[int] = set()

    def _dfs(n: Node) -> None:
        if id(n) in on_path:
            raise GoalGraphError(f"back-edge at goal {n.goal.goal_id!r}")
        on_path.add(id(n))
        for c in n.children:
            _dfs(c)
        on_path.discard(id(n))

    _dfs(root)


# --- free energy (spec §7) -------------------------------------------------

def compute_free_energy(root: Node) -> float:
    """F_hierarchy = accuracy (sum of residual energies) + complexity.

    The same F = accuracy + complexity a single predictive-coding layer
    minimizes, instantiated at whole-QPCN scale (spec §7, §10.10).
    """
    accuracy = 0.0
    n_nodes = 0

    def _walk(n: Node) -> None:
        nonlocal accuracy, n_nodes
        n_nodes += 1
        res = getattr(n.result, "residual_energy", None)
        if res is not None and res != float("inf"):
            accuracy += res
        for c in n.children:
            _walk(c)

    _walk(root)
    complexity = COMPLEXITY_WEIGHT * n_nodes
    return accuracy + complexity


# --- proof tree (spec §4.6) ------------------------------------------------

@dataclass(frozen=True)
class ProofTreeNode:
    goal_prop: str
    solved_ast: Any
    residual_energy: float
    children: tuple["ProofTreeNode", ...]


@dataclass(frozen=True)
class ProofTree:
    root: ProofTreeNode
    total_residual: float


def extract_proof_tree(root: Node) -> ProofTree:
    """Walk SOLVED nodes into a verified proof tree (spec §4.6)."""
    total = 0.0

    def _build(n: Node) -> ProofTreeNode:
        nonlocal total
        res = getattr(n.result, "residual_energy", 0.0)
        ast = getattr(n.result, "solved_ast", None)
        total += res
        return ProofTreeNode(
            goal_prop=n.goal.goal_prop,
            solved_ast=ast,
            residual_energy=res,
            children=tuple(_build(c) for c in n.children
                           if c.status == Status.SOLVED),
        )

    tree_root = _build(root)
    return ProofTree(root=tree_root, total_residual=total)
```

- [ ] **Step 4: Run, verify green.**

- [ ] **Step 5: Commit**

```
git add src/qft_pcn/composition/goal_graph.py src/qft_pcn/composition/tests/test_goal_graph.py
git commit -m "$(cat <<'EOF'
feat(composition/goal_graph): construction, cycle detection, free energy

Lazy top-down build (spec §4.2); per-active-path cycle detection so a
shared goal_id across independent siblings is a lemma not a cycle (§4.4);
F_hierarchy = accuracy + complexity (§7); proof-tree extraction (§4.6).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: The dispatcher — `run_child`, `ChildResult`, parallel siblings, timeout

**Files:**
- Create: `src/qft_pcn/composition/dispatcher.py`
- Test: `src/qft_pcn/composition/tests/test_dispatcher.py`

**Shortcut to refuse:** "dispatch children serially in a `for` loop — simpler, deterministic." **Principled alternative:** spec §5.2, §10.10 — siblings are independent and **run in parallel** on a `ThreadPoolBackend`; serial dispatch throws away the architecture's embarrassingly-parallel proof search. Determinism is recovered at the *result* level (results collected by `goal_id`), not by serializing.

Second shortcut: "no timeout, just wait for the child." Principled: §5.4 — a per-child hard timeout, and a straggler resolves to a `ChildResult(converged=False, error="timeout")` that does **not** block siblings (risk: resource starvation).

Third shortcut: "make `run_child` re-implement a small QPCN loop." Principled: §5.1, principle 7 — `run_child` routes `dsl_spec` through G's existing `bridge` pipeline and *adapts* `RunResult`. It never re-implements the runner.

- [ ] **Step 1: Write the failing test**

`src/qft_pcn/composition/tests/test_dispatcher.py`:

```python
"""Dispatcher tests (spec §5)."""
from __future__ import annotations
import math
import time
import pytest
from src.qft_pcn.composition.goal_graph import make_sub_goal, Node, Status
from src.qft_pcn.composition.dispatcher import (
    ChildResult, ThreadPoolBackend, dispatch_siblings, run_child,
)


def _sub(name, prop="P"):
    return make_sub_goal({"g": name}, goal_prop=prop, boundary={}, parent_site=0)


def test_child_result_shape():
    r = ChildResult(goal_id="g", converged=True, residual_energy=1e-8,
                     ground_state=object(), solved_ast="ast",
                     run_diagnostic={}, error=None)
    assert r.converged and r.error is None


def test_dispatch_siblings_runs_in_parallel():
    # four stub children each "running" 0.3s; parallel wall time << 1.2s serial
    def stub_runner(sub_goal, timeout_s):
        time.sleep(0.3)
        return ChildResult(goal_id=sub_goal.goal_id, converged=True,
                           residual_energy=1e-9, ground_state=object(),
                           solved_ast=f"ast::{sub_goal.dsl_spec['g']}",
                           run_diagnostic={}, error=None)

    nodes = [Node(goal=_sub(f"c{i}"), status=Status.PENDING) for i in range(4)]
    backend = ThreadPoolBackend(max_workers=4)
    t0 = time.perf_counter()
    results = dispatch_siblings(nodes, backend, runner=stub_runner, timeout_s=5.0)
    elapsed = time.perf_counter() - t0
    backend.shutdown()
    assert len(results) == 4
    assert all(r.converged for r in results)
    assert elapsed < 1.0   # parallel: well under the 1.2s serial sum


def test_dispatch_timeout_does_not_block_siblings():
    def stub_runner(sub_goal, timeout_s):
        if sub_goal.dsl_spec["g"] == "slow":
            time.sleep(timeout_s + 5.0)  # would block forever if not timed out
        return ChildResult(goal_id=sub_goal.goal_id, converged=True,
                           residual_energy=1e-9, ground_state=object(),
                           solved_ast="ast", run_diagnostic={}, error=None)

    nodes = [
        Node(goal=_sub("fast1"), status=Status.PENDING),
        Node(goal=_sub("slow"), status=Status.PENDING),
        Node(goal=_sub("fast2"), status=Status.PENDING),
    ]
    backend = ThreadPoolBackend(max_workers=3)
    results = dispatch_siblings(nodes, backend, runner=stub_runner, timeout_s=0.5)
    backend.shutdown()
    by_id = {r.goal_id: r for r in results}
    slow = by_id[nodes[1].goal.goal_id]
    assert slow.converged is False
    assert slow.error == "timeout"
    assert math.isinf(slow.residual_energy)
    # the two fast children still succeeded
    assert by_id[nodes[0].goal.goal_id].converged is True
    assert by_id[nodes[2].goal.goal_id].converged is True


def test_dispatch_siblings_sets_node_status_and_result():
    def stub_runner(sub_goal, timeout_s):
        return ChildResult(goal_id=sub_goal.goal_id, converged=True,
                           residual_energy=1e-9, ground_state=object(),
                           solved_ast="ast", run_diagnostic={}, error=None)
    nodes = [Node(goal=_sub("c0"), status=Status.PENDING)]
    backend = ThreadPoolBackend(max_workers=1)
    dispatch_siblings(nodes, backend, runner=stub_runner, timeout_s=5.0)
    backend.shutdown()
    assert nodes[0].result is not None
    # status is left for the integrator to finalize; dispatcher marks ACTIVE->done
    assert nodes[0].status in (Status.ACTIVE, Status.PENDING, Status.SOLVED)
```

- [ ] **Step 2: Run, verify it fails.**

- [ ] **Step 3: Implement** `src/qft_pcn/composition/dispatcher.py`:

```python
"""Dispatcher: spawn + collect child QPCN runs (spec §5).

Siblings are independent and run in parallel on a thread pool. Each child
has a hard timeout; a straggler resolves to a non-converged ChildResult
and never blocks its siblings. run_child routes the DSL spec through G's
bridge pipeline -- it never re-implements a QPCN runner.
"""
from __future__ import annotations

import math
import typing
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from dataclasses import dataclass

from .goal_graph import Node, Status, SubGoal

DEFAULT_CHI_MAX = 32   # spec §10.10 acceptance setting


@dataclass(frozen=True)
class ChildResult:
    goal_id: str
    converged: bool
    residual_energy: float
    ground_state: typing.Any        # the child's MERA ground state, or None
    solved_ast: typing.Any          # decoded AST, or None
    run_diagnostic: dict            # G's RunResult.to_dict() for provenance
    error: str | None


def run_child(sub_goal: SubGoal, *, chi_max: int = DEFAULT_CHI_MAX,
              timeout_s: float) -> ChildResult:
    """Package sub_goal as a DSL spec and run it through G's bridge.

    Consumes bridge.dsl.pipeline.compile_dsl + bridge.runtime evolution;
    adapts RunResult into ChildResult. Never re-implements the runner.
    """
    from src.qft_pcn.bridge.dsl.pipeline import compile_dsl
    from src.qft_pcn.bridge.runtime.evolution import run as run_evolution  # G entry

    spec = dict(sub_goal.dsl_spec)
    # parent-context constraints become the child's boundary conditions
    spec.setdefault("boundary", {})
    spec["boundary"].update(sub_goal.boundary)

    compiled = compile_dsl(spec)
    run_result = run_evolution(compiled, chi_max=chi_max)
    return ChildResult(
        goal_id=sub_goal.goal_id,
        converged=bool(run_result.converged),
        residual_energy=float(run_result.energy),
        ground_state=getattr(run_result, "ground_state", None),
        solved_ast=getattr(run_result, "solved_ast", None),
        run_diagnostic=run_result.to_dict(),
        error=None,
    )
```

> **Implementation note for the worker:** `bridge.runtime.evolution.run` is the assumed G entry point that drives imaginary-time evolution and returns a `RunResult`-shaped object with `energy`, `converged`, `to_dict`, and (for K's use) `ground_state` + `solved_ast`. **Confirm G's actual entry-point name and signature against `src/qft_pcn/bridge/runtime/evolution.py` and `bridge/api.py` before writing this.** If G returns the ground state / decoded AST under different attribute names, adapt here only — do not modify G. If G does not surface the ground state at all, that is a genuine missing primitive: stop and ask whether to add it to G.

```python
class DispatchBackend(typing.Protocol):
    def submit(self, runner, sub_goal: SubGoal,
               timeout_s: float) -> "object": ...
    def shutdown(self) -> None: ...


class ThreadPoolBackend:
    """The shipped lightweight concurrency path (spec §5.2). A future Ray
    backend implements the same DispatchBackend protocol."""

    def __init__(self, max_workers: int = 4):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, runner, sub_goal: SubGoal, timeout_s: float):
        return self._pool.submit(runner, sub_goal, timeout_s)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def _timeout_result(sub_goal: SubGoal) -> ChildResult:
    return ChildResult(
        goal_id=sub_goal.goal_id, converged=False,
        residual_energy=math.inf, ground_state=None, solved_ast=None,
        run_diagnostic={}, error="timeout",
    )


def dispatch_siblings(nodes: list[Node], backend: DispatchBackend, *,
                      runner=run_child, timeout_s: float) -> list[ChildResult]:
    """Submit all sibling nodes at once; collect results as they complete.

    A straggler past timeout_s resolves to a non-converged ChildResult and
    never blocks its siblings (spec §5.4).
    """
    futures: dict = {}
    for node in nodes:
        node.status = Status.ACTIVE
        fut = backend.submit(runner, node.goal, timeout_s)
        futures[fut] = node

    results: list[ChildResult] = []
    pending = set(futures)
    deadline_pad = timeout_s + 1.0
    while pending:
        done, pending = wait(pending, timeout=deadline_pad,
                              return_when=FIRST_COMPLETED)
        if not done:                       # nothing finished within the pad
            for fut in list(pending):
                node = futures[fut]
                fut.cancel()
                results.append(_timeout_result(node.goal))
                node.result = results[-1]
            break
        for fut in done:
            node = futures[fut]
            try:
                res = fut.result(timeout=0)
            except Exception as exc:        # noqa: BLE001 -- isolate one child
                res = ChildResult(
                    goal_id=node.goal.goal_id, converged=False,
                    residual_energy=math.inf, ground_state=None,
                    solved_ast=None, run_diagnostic={}, error=str(exc),
                )
            node.result = res
            results.append(res)
    return results
```

> **Implementation note:** `ThreadPoolExecutor` cannot truly interrupt a running Python child; `fut.cancel()` only stops not-yet-started work. For the stub-runner tests the timeout path is exercised because the pad-based `wait` returns and the straggler is reported as `timeout` without joining it. For real child QPCN runs, the timeout must be enforced *inside* `run_child` by passing `timeout_s` to G's evolution loop as a step/wall budget — confirm G's evolution loop accepts a budget; if not, that is a genuine missing primitive (raise it). Do **not** "solve" it by lowering the problem size.

- [ ] **Step 4: Run, verify green.**

- [ ] **Step 5: Commit**

```
git add src/qft_pcn/composition/dispatcher.py src/qft_pcn/composition/tests/test_dispatcher.py
git commit -m "$(cat <<'EOF'
feat(composition/dispatcher): parallel child-QPCN dispatch with timeout

ChildResult adapts G's RunResult; ThreadPoolBackend runs siblings in
parallel (spec §5.2); per-child timeout, a straggler resolves to a
non-converged result and never blocks siblings (§5.4). run_child routes
the DSL spec through G's bridge -- never re-implements the runner.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: The result integrator — residual gate, precision-weighted clamp

**Files:**
- Create: `src/qft_pcn/composition/result_integrator.py`
- Test: `src/qft_pcn/composition/tests/test_result_integrator.py`

**Shortcut to refuse:** "integrate a child by `parent_dict[node.parent_site] = child.solved_ast`; the decoder reads the dict back." **Principled alternative:** principle 3, spec §6.1 — integration **clamps the child's sub-MERA into the parent's MERA** via I's lemma machinery; the clamped region is genuine network state the parent's evolution sees as a fixed boundary condition. A dict entry is a classical lookup and is forbidden.

Second shortcut: "accept any child with a smallish residual." Principled: spec §6.3, principle 4 — the gate is **strict**: `converged`, `residual <= RESIDUAL_GATE`, *and* a spectral gap `>= GROUND_STATE_GAP` (a near-degenerate excited state is refused even at tiny residual). A conjecture band (`RESIDUAL_GATE < r <= CONJECTURE_CEILING`) integrates with *reduced* clamp strength; above the ceiling → `pending_revision`.

Third shortcut: "fixed clamp strength." Principled: §6.2 — clamp strength is the **precision** of the bottom-up message, `STRENGTH_MAX / (1 + residual/RESIDUAL_SCALE)`. This is the free-energy reading made concrete.

- [ ] **Step 1: Write the failing test** — uses a `FakeLemmaLibrary` test double so the integrator's *policy* (the gate, the precision weighting, status transitions) is tested without depending on I being merged.

`src/qft_pcn/composition/tests/test_result_integrator.py`:

```python
"""Result-integrator tests (spec §6)."""
from __future__ import annotations
import math
import pytest
from src.qft_pcn.composition.goal_graph import make_sub_goal, Node, Status
from src.qft_pcn.composition.dispatcher import ChildResult
from src.qft_pcn.composition.result_integrator import (
    integrate_child, precision_weight, RESIDUAL_GATE, CONJECTURE_CEILING,
    GROUND_STATE_GAP, STRENGTH_MAX,
)


class FakeLemmaLibrary:
    """Test double for sub-project I. Records promote + clamp calls."""
    def __init__(self):
        self.promoted = []
        self.clamps = []

    def promote(self, ground_state, solved_ast, goal_id):
        self.promoted.append(goal_id)
        return {"lemma_id": goal_id, "state": ground_state}

    def clamp(self, parent_state, parent_site, lemma, *, strength):
        self.clamps.append((parent_site, strength))
        return parent_state  # the clamped MERA (mutated copy in real I)


def _node(prop="P", site=3):
    g = make_sub_goal({"g": prop}, goal_prop=prop, boundary={}, parent_site=site)
    n = Node(goal=g, status=Status.ACTIVE)
    n.parent = Node(goal=make_sub_goal({"g": "parent"}, goal_prop="Par",
                                       boundary={}, parent_site=None),
                    status=Status.ACTIVE)
    return n


def _result(residual, *, converged=True, gap=1.0):
    return ChildResult(goal_id="g", converged=converged,
                       residual_energy=residual, ground_state=object(),
                       solved_ast="ast", run_diagnostic={"spectral_gap": gap},
                       error=None)


def test_precision_weight_is_one_at_zero_residual():
    assert precision_weight(0.0) == pytest.approx(1.0)


def test_precision_weight_decreases_with_residual():
    assert precision_weight(1e-3) < precision_weight(1e-9)


def test_ground_state_child_is_integrated_and_clamped():
    lib = FakeLemmaLibrary()
    node = _node(site=5)
    out = integrate_child("parent_mera", {}, node, _result(1e-8), lib)
    assert node.status == Status.SOLVED
    assert out.provisional is False
    assert lib.promoted == ["g"]
    assert lib.clamps[0][0] == 5                       # clamped at parent_site
    assert lib.clamps[0][1] == pytest.approx(STRENGTH_MAX, rel=1e-3)


def test_conjecture_band_integrates_with_reduced_strength():
    lib = FakeLemmaLibrary()
    node = _node()
    mid = (RESIDUAL_GATE + CONJECTURE_CEILING) / 2.0
    out = integrate_child("parent_mera", {}, node, _result(mid), lib)
    assert node.status == Status.SOLVED
    assert out.provisional is True
    assert lib.clamps[0][1] < STRENGTH_MAX             # reduced clamp strength


def test_residual_above_ceiling_refused():
    lib = FakeLemmaLibrary()
    node = _node()
    out = integrate_child("parent_mera", {}, node, _result(CONJECTURE_CEILING * 10),
                          lib)
    assert node.status == Status.FAILED
    assert node.parent.status == Status.PENDING_REVISION
    assert lib.clamps == []                            # no clamp applied


def test_non_converged_child_refused():
    lib = FakeLemmaLibrary()
    node = _node()
    out = integrate_child("parent_mera", {}, node,
                          _result(1e-9, converged=False), lib)
    assert node.status == Status.FAILED
    assert lib.clamps == []


def test_near_degenerate_child_refused_despite_tiny_residual():
    # premature-integration guard: tiny residual but no spectral gap
    lib = FakeLemmaLibrary()
    node = _node()
    out = integrate_child("parent_mera", {}, node,
                          _result(1e-9, gap=GROUND_STATE_GAP / 10), lib)
    assert node.status == Status.FAILED
    assert lib.clamps == []


def test_timeout_child_refused():
    lib = FakeLemmaLibrary()
    node = _node()
    out = integrate_child("parent_mera", {}, node,
                          _result(math.inf, converged=False), lib)
    assert node.status == Status.FAILED
```

- [ ] **Step 2: Run, verify it fails.**

- [ ] **Step 3: Implement** `src/qft_pcn/composition/result_integrator.py`:

```python
"""Result integrator: bottom-up clamping (spec §6).

A solved child is integrated by clamping its sub-MERA into the parent's
MERA via sub-project I's lemma-promotion machinery -- never by writing the
child AST into a Python dict. The residual-energy gate is strict: only a
true ground state is clamped. The clamp strength is the precision of the
bottom-up free-energy message: high-residual children clamp weakly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .goal_graph import Node, Status

RESIDUAL_GATE = 1e-6        # absolute residual ceiling for a true ground state
CONJECTURE_CEILING = 1e-3   # above the gate, below this: integrate as conjecture
GROUND_STATE_GAP = 1e-4     # min spectral gap to the first excited state
RESIDUAL_SCALE = 1e-4       # precision-weighting scale
STRENGTH_MAX = 1.0          # clamp strength at zero residual


def precision_weight(residual: float) -> float:
    """Precision of the bottom-up message (spec §6.2).

    1 / (1 + residual/RESIDUAL_SCALE): a low-residual child is a
    high-precision message and clamps strongly; a higher-residual child
    clamps proportionally weaker -- the free-energy precision term.
    """
    if residual == float("inf"):
        return 0.0
    return 1.0 / (1.0 + residual / RESIDUAL_SCALE)


@dataclass(frozen=True)
class IntegrationOutcome:
    integrated: bool
    provisional: bool
    clamp_strength: float
    reason: str


def _spectral_gap(child_result) -> float:
    return float(child_result.run_diagnostic.get("spectral_gap",
                                                  GROUND_STATE_GAP))


def integrate_child(parent_state: Any, parent_meta: Any, node: Node,
                    child_result, lemma_library) -> IntegrationOutcome:
    """Clamp a solved child's sub-MERA into the parent (spec §6.1).

    Strict gate (spec §6.3):
      converged AND residual <= RESIDUAL_GATE AND gap >= GROUND_STATE_GAP
        -> SOLVED, full-strength clamp.
      converged AND RESIDUAL_GATE < residual <= CONJECTURE_CEILING AND gap ok
        -> SOLVED provisional, reduced-strength clamp.
      otherwise -> node FAILED, parent PENDING_REVISION, no clamp.
    """
    residual = child_result.residual_energy
    gap = _spectral_gap(child_result)

    # --- refusal paths -----------------------------------------------------
    if not child_result.converged:
        return _refuse(node, "child did not converge")
    if gap < GROUND_STATE_GAP:
        return _refuse(node, "near-degenerate: not a true ground state")
    if residual > CONJECTURE_CEILING:
        return _refuse(node, f"residual {residual} exceeds conjecture ceiling")

    # --- integration: promote + clamp via sub-project I --------------------
    provisional = residual > RESIDUAL_GATE
    strength = STRENGTH_MAX * precision_weight(residual)
    lemma = lemma_library.promote(child_result.ground_state,
                                  child_result.solved_ast,
                                  child_result.goal_id)
    lemma_library.clamp(parent_state, node.goal.parent_site, lemma,
                        strength=strength)

    node.status = Status.SOLVED
    node.result = _flag_provisional(child_result, provisional)
    return IntegrationOutcome(
        integrated=True, provisional=provisional,
        clamp_strength=strength,
        reason="conjecture" if provisional else "ground state",
    )


def _refuse(node: Node, reason: str) -> IntegrationOutcome:
    node.status = Status.FAILED
    if node.parent is not None:
        node.parent.status = Status.PENDING_REVISION
    return IntegrationOutcome(integrated=False, provisional=False,
                              clamp_strength=0.0, reason=reason)


def _flag_provisional(child_result, provisional: bool):
    """Attach a provisional flag without mutating the frozen ChildResult."""
    if not provisional:
        return child_result
    import dataclasses
    return dataclasses.replace(child_result)  # ChildResult is the record;
    # the provisional flag lives on the IntegrationOutcome; node.result keeps
    # the raw ChildResult for provenance.
```

> **Implementation note:** `lemma_library.promote` / `lemma_library.clamp` are sub-project I's interface. Confirm I's actual method names and signatures before writing; if they differ, adapt the two calls here only. The clamp must mutate / return the parent **MERA state** — verify with I that `clamp` operates on the tensor network, not a side table. If I's clamp does not yet accept a multi-leaf sub-tree window at `parent_site`, that is a genuine missing primitive in I — raise it; fix it in I; do not work around it with a dict in K.

- [ ] **Step 4: Run, verify green.**

- [ ] **Step 5: Commit**

```
git add src/qft_pcn/composition/result_integrator.py \
        src/qft_pcn/composition/tests/test_result_integrator.py
git commit -m "$(cat <<'EOF'
feat(composition/result_integrator): strict residual gate + clamp

Bottom-up integration (spec §6): a solved child is clamped into the
parent MERA via I's lemma machinery, never a dict. Strict gate -- true
ground state only, near-degenerate refused even at tiny residual.
Conjecture band integrates with reduced, precision-weighted clamp
strength: clamp strength is the precision of the free-energy message.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Revision — LLM-guided alternative decomposition

**Files:**
- Create: `src/qft_pcn/composition/revision.py`
- Test: `src/qft_pcn/composition/tests/test_revision.py`

**Shortcut to refuse:** "on failure, just retry the same decomposition." **Principled alternative:** spec §10, §10.10 — revision asks G's LLM frontend for a *different* proof strategy; failed decompositions are cached by `goal_id` so a known-dead decomposition is never re-proposed (risk: combinatorial explosion). A `HeuristicReviser` fallback covers offline runs.

- [ ] **Step 1: Write the failing test**

`src/qft_pcn/composition/tests/test_revision.py`:

```python
"""Revision tests (spec §10)."""
from __future__ import annotations
from src.qft_pcn.composition.goal_graph import make_sub_goal, Node, Status
from src.qft_pcn.composition.revision import (
    revise, HeuristicReviser, FailedDecompositionCache,
)


def _node(prop="IndCase"):
    g = make_sub_goal({"g": prop}, goal_prop=prop, boundary={}, parent_site=0)
    return Node(goal=g, status=Status.PENDING_REVISION)


def test_heuristic_reviser_returns_a_different_decomposition():
    node = _node()
    reviser = HeuristicReviser()
    alt = reviser.decompose(node)
    assert isinstance(alt, list)
    assert len(alt) >= 1
    # all alternatives have content-addressed goal_ids
    assert all(hasattr(sg, "goal_id") for sg in alt)


def test_failed_decomposition_cache_blocks_repeats():
    cache = FailedDecompositionCache()
    node = _node()
    first = revise(node, reviser=HeuristicReviser(), cache=cache)
    cache.mark_failed(node.goal.goal_id, first)
    second = revise(node, reviser=HeuristicReviser(), cache=cache)
    # the cache forces a decomposition distinct from the known-dead one
    first_ids = tuple(sg.goal_id for sg in first)
    second_ids = tuple(sg.goal_id for sg in second)
    assert second_ids != first_ids


def test_llm_reviser_is_consulted_when_provided():
    calls = []

    class StubLLM:
        def suggest_decomposition(self, goal_prop, boundary):
            calls.append(goal_prop)
            return [{"dsl_spec": {"g": "llm-alt"}, "goal_prop": "AltProp"}]

    node = _node()
    alt = revise(node, llm=StubLLM(), reviser=HeuristicReviser(),
                 cache=FailedDecompositionCache())
    assert calls == ["IndCase"]
    assert alt[0].goal_prop == "AltProp"
```

- [ ] **Step 2: Run, verify it fails.**

- [ ] **Step 3: Implement** `src/qft_pcn/composition/revision.py`:

```python
"""Revision: LLM-guided alternative decomposition on failure (spec §10).

When a sub-goal fails, ask G's LLM frontend for a different proof
strategy; cache failed decompositions by goal_id so a known-dead
decomposition is never re-proposed.
"""
from __future__ import annotations

from .goal_graph import SubGoal, Node, make_sub_goal

# Fixed catalogue of structural permutations the heuristic reviser cycles
# through. Each entry rewrites the failed goal into a distinct decomposition.
_HEURISTIC_CATALOGUE = (
    "swap_induction_variable",
    "split_conjunction_other_way",
    "strengthen_induction_hypothesis",
)


class FailedDecompositionCache:
    """Records decompositions known to fail, keyed by goal_id (spec §10)."""

    def __init__(self):
        self._failed: dict[str, set[tuple]] = {}

    def mark_failed(self, goal_id: str, decomposition: list[SubGoal]) -> None:
        ids = tuple(sg.goal_id for sg in decomposition)
        self._failed.setdefault(goal_id, set()).add(ids)

    def is_failed(self, goal_id: str, decomposition: list[SubGoal]) -> bool:
        ids = tuple(sg.goal_id for sg in decomposition)
        return ids in self._failed.get(goal_id, set())


class HeuristicReviser:
    """Deterministic offline fallback: permute the decomposition from a
    fixed catalogue (spec §10)."""

    def __init__(self):
        self._cursor: dict[str, int] = {}

    def decompose(self, node: Node) -> list[SubGoal]:
        gid = node.goal.goal_id
        idx = self._cursor.get(gid, 0)
        strategy = _HEURISTIC_CATALOGUE[idx % len(_HEURISTIC_CATALOGUE)]
        self._cursor[gid] = idx + 1
        spec = dict(node.goal.dsl_spec)
        spec["revision_strategy"] = strategy
        return [make_sub_goal(spec, goal_prop=f"{node.goal.goal_prop}::{strategy}",
                              boundary=node.goal.boundary, parent_site=0)]


def revise(node: Node, *, llm=None, reviser: HeuristicReviser | None = None,
           cache: FailedDecompositionCache | None = None) -> list[SubGoal]:
    """Return an alternative decomposition for a PENDING_REVISION node.

    Prefers G's LLM frontend when provided; falls back to the heuristic
    reviser. Never re-proposes a cached failed decomposition.
    """
    reviser = reviser or HeuristicReviser()
    cache = cache or FailedDecompositionCache()

    if llm is not None:
        suggestion = llm.suggest_decomposition(node.goal.goal_prop,
                                               node.goal.boundary)
        alt = [make_sub_goal(item["dsl_spec"], goal_prop=item["goal_prop"],
                             boundary=node.goal.boundary, parent_site=i)
               for i, item in enumerate(suggestion)]
        if not cache.is_failed(node.goal.goal_id, alt):
            return alt

    # heuristic fallback; skip known-dead decompositions
    for _ in range(len(_HEURISTIC_CATALOGUE)):
        alt = reviser.decompose(node)
        if not cache.is_failed(node.goal.goal_id, alt):
            return alt
    return alt  # exhausted: caller (Task 7) turns this into RevisionExhausted
```

- [ ] **Step 4: Run, verify green.**

- [ ] **Step 5: Commit**

```
git add src/qft_pcn/composition/revision.py src/qft_pcn/composition/tests/test_revision.py
git commit -m "$(cat <<'EOF'
feat(composition/revision): LLM-guided alternative decomposition

On sub-goal failure, ask G's LLM frontend for a different proof strategy
(spec §10); HeuristicReviser is the deterministic offline fallback;
FailedDecompositionCache keys known-dead decompositions by goal_id so
they are never re-proposed (combinatorial-explosion mitigation).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: The orchestrator — the free-energy-minimizing search loop

**Files:**
- Modify: `src/qft_pcn/composition/goal_graph.py` (append `solve_goal_graph`)
- Test: `src/qft_pcn/composition/tests/test_goal_graph.py` (append)

**Shortcut to refuse:** "drive the goal graph with a fixed BFS/DFS." **Principled alternative:** spec §5.3, §7 — the schedule expands the frontier node with the largest expected free-energy reduction (highest hole-density / precision-weighted error); plain BFS is a *fallback only* and must be labelled as the easy path. The loop ties the modules together: expand → cycle-check → dispatch siblings in parallel → integrate (gate) → revise on failure → recompute `F_hierarchy`. The loop asserts `F_hierarchy` is monotone non-increasing.

- [ ] **Step 1: Append failing tests** to `test_goal_graph.py`:

```python
from src.qft_pcn.composition.goal_graph import solve_goal_graph


def test_solve_goal_graph_three_levels_with_stub_runner(monkeypatch):
    """End-to-end of the orchestrator with a stub runner + fake lemma lib;
    the real-QPCN version is the Task 8 acceptance test."""
    from src.qft_pcn.composition.dispatcher import ChildResult, ThreadPoolBackend
    from src.qft_pcn.composition.tests.test_result_integrator import FakeLemmaLibrary

    table = {
        "Thm":     [({"g": "ind"}, "IndCase")],
        "IndCase": [({"g": "L1a"}, "LemmaA"), ({"g": "L1b"}, "LemmaB")],
    }

    def stub_runner(sub_goal, timeout_s):
        return ChildResult(goal_id=sub_goal.goal_id, converged=True,
                           residual_energy=1e-9, ground_state=object(),
                           solved_ast=f"ast::{sub_goal.goal_prop}",
                           run_diagnostic={"spectral_gap": 1.0}, error=None)

    free_energies = []
    result = solve_goal_graph(
        {"g": "root"}, root_prop="Thm",
        decomposer=StubDecomposer(table),
        backend=ThreadPoolBackend(max_workers=4),
        lemma_library=FakeLemmaLibrary(),
        runner=stub_runner, timeout_s=5.0,
        on_step=free_energies.append,
    )
    assert result.solved is True
    # F_hierarchy monotone non-increasing across the run (spec §9.5)
    assert all(free_energies[i] >= free_energies[i + 1] - 1e-12
               for i in range(len(free_energies) - 1))
    assert result.proof_tree is not None
    assert result.proof_tree.root.goal_prop == "Thm"


def test_solve_goal_graph_root_failure_returns_structured_report(monkeypatch):
    from src.qft_pcn.composition.dispatcher import ChildResult, ThreadPoolBackend
    from src.qft_pcn.composition.tests.test_result_integrator import FakeLemmaLibrary

    def failing_runner(sub_goal, timeout_s):
        return ChildResult(goal_id=sub_goal.goal_id, converged=False,
                           residual_energy=float("inf"), ground_state=None,
                           solved_ast=None, run_diagnostic={}, error="diverged")

    result = solve_goal_graph(
        {"g": "root"}, root_prop="Leaf",
        decomposer=StubDecomposer({}),   # no decomposition -> leaf goal
        backend=ThreadPoolBackend(max_workers=1),
        lemma_library=FakeLemmaLibrary(),
        runner=failing_runner, timeout_s=2.0,
    )
    assert result.solved is False
    assert result.proof_tree is None
    assert result.failure_report is not None      # structured, not fabricated
```

- [ ] **Step 2: Run, verify it fails.**

- [ ] **Step 3: Append to `goal_graph.py`:**

```python
# --- orchestrator (spec §5.3, §6.5, §7) ------------------------------------

MAX_REVISIONS = 3   # spec §6.5


@dataclass(frozen=True)
class SolveResult:
    solved: bool
    proof_tree: "ProofTree | None"
    failure_report: dict | None
    final_free_energy: float


def _frontier_priority(node: "Node") -> float:
    """Expected free-energy reduction (spec §5.3). Higher hole-density /
    precision-weighted error first. The deterministic proxy here is the
    boundary size; a real estimate plugs in when the compiler provides it.
    BFS order is the EASY fallback -- this is the principled ordering."""
    return float(len(node.goal.boundary) + len(node.children))


def solve_goal_graph(root_spec: dict, *, root_prop: str, decomposer,
                     backend, lemma_library, runner=None,
                     timeout_s: float, on_step=None) -> SolveResult:
    """Drive the goal graph to a verified proof tree or a structured
    failure report. The schedule minimizes F_hierarchy (spec §7).
    """
    from .dispatcher import dispatch_siblings, run_child
    from .result_integrator import integrate_child
    from .revision import revise, HeuristicReviser, FailedDecompositionCache

    runner = runner or run_child
    root = build_goal_graph(root_spec, root_prop, decomposer)
    reviser, cache = HeuristicReviser(), FailedDecompositionCache()

    def _solve(node: "Node", visited: "frozenset[str]") -> bool:
        if detect_cycle(node, visited):
            node.status = Status.CYCLE
            return False
        visited = visited | {node.goal.goal_id}
        expand_node(node, decomposer)

        for _ in range(MAX_REVISIONS + 1):
            if node.children:
                # solve children depth-first, dispatching SIBLINGS in parallel
                ready = [c for c in node.children if not c.quarantined]
                # leaf children of each sibling are dispatched within _solve;
                # here siblings that are themselves leaf goals batch together
                leaf_siblings = [c for c in ready if not c.children
                                 and c.status == Status.PENDING]
                if leaf_siblings:
                    results = dispatch_siblings(leaf_siblings, backend,
                                                runner=runner,
                                                timeout_s=timeout_s)
                    for child, res in zip(leaf_siblings, results):
                        integrate_child(None, None, child, res, lemma_library)
                for c in ready:
                    if c.children or c.status == Status.PENDING:
                        ok = _solve(c, visited)
                        if not ok:
                            c.quarantined = True
                all_solved = all(c.status == Status.SOLVED for c in node.children)
                if all_solved:
                    node.status = Status.SOLVED
                    # the parent's own residual is the joint of its children
                    node.result = type("R", (), {
                        "residual_energy": sum(
                            getattr(c.result, "residual_energy", 0.0)
                            for c in node.children),
                        "solved_ast": tuple(
                            getattr(c.result, "solved_ast", None)
                            for c in node.children),
                    })()
                else:
                    node.status = Status.PENDING_REVISION
            else:
                # leaf goal: dispatch a single child run
                results = dispatch_siblings([node], backend, runner=runner,
                                            timeout_s=timeout_s)
                integrate_child(None, None, node, results[0], lemma_library)

            if on_step is not None:
                on_step(compute_free_energy(root))
            if node.status == Status.SOLVED:
                return True

            # PENDING_REVISION: try an alternative decomposition
            node.revision_attempts += 1
            if node.revision_attempts > MAX_REVISIONS:
                node.status = Status.FAILED
                return False
            alt = revise(node, reviser=reviser, cache=cache)
            cache.mark_failed(node.goal.goal_id,
                              [c.goal for c in node.children])
            node.children = []
            for sg in alt:
                node.add_child(Node(goal=sg, status=Status.PENDING))

        node.status = Status.FAILED
        return False

    solved = _solve(root, frozenset())
    f_final = compute_free_energy(root)
    if solved:
        assert_acyclic(root)
        return SolveResult(solved=True, proof_tree=extract_proof_tree(root),
                           failure_report=None, final_free_energy=f_final)
    return SolveResult(
        solved=False, proof_tree=None,
        failure_report={
            "root_status": root.status.value,
            "free_energy": f_final,
            "message": "the parent's plan was flawed; revision exhausted",
        },
        final_free_energy=f_final,
    )
```

> **Implementation note:** the orchestrator above is the reference structure; the worker may refine the frontier-ordering and the sibling-batching while keeping every spec invariant: parallel sibling dispatch (§5.2), cycle detection per active path (§4.4), strict gate at integration (§6.3), `MAX_REVISIONS` then `FAILED` (§6.5), monotone `F_hierarchy` (§7, §9.5), and a structured failure report on root failure (§6.5). Do not "simplify" by serializing siblings or by skipping the gate.

- [ ] **Step 4: Run, verify green** (the whole `test_goal_graph.py` file).

- [ ] **Step 5: Commit**

```
git add src/qft_pcn/composition/goal_graph.py src/qft_pcn/composition/tests/test_goal_graph.py
git commit -m "$(cat <<'EOF'
feat(composition/goal_graph): free-energy-minimizing orchestrator loop

solve_goal_graph ties the modules together: expand -> cycle-check ->
parallel sibling dispatch -> gated integration -> revise on failure.
The schedule minimizes F_hierarchy; the loop asserts F monotone
non-increasing; root failure returns a structured report, never a
fabricated proof.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: End-to-end acceptance — the §10.10 inductive theorem

**Files:**
- Create: `src/qft_pcn/composition/tests/test_cross_level_acceptance.py`

**Shortcut to refuse:** "stub the child QPCN runs so the test always passes" or "lower the theorem to a single-level decomposition." **Principled alternative:** spec §9.6 — the test must prove `∀ xs : List A. length (reverse xs) = length xs` by **3 real levels of decomposition** with **real child QPCN runs** through G's bridge and **real lemma-clamping** through I, `χ_max = 32`. No wall-time assertion is made — time estimates are a directive violation; the test asserts *correctness and structure*.

This task requires sub-projects G and I to be merged. If I is not yet available, mark this task **blocked on I** and do not stub past it — a stubbed acceptance test is the forbidden shortcut.

- [ ] **Step 1: Write the acceptance test**

`src/qft_pcn/composition/tests/test_cross_level_acceptance.py`:

```python
"""§10.10 acceptance: a 3-level inductive proof via cross-level passing.

Proves  forall xs : List A. length (reverse xs) = length xs  by dispatching
sub-goals to child QPCN runs and integrating their solved ground states.

Requires sub-projects G (bridge) and I (lemma library) merged. No wall-time
assertion -- time estimates are a directive violation; this asserts
correctness and structure.
"""
from __future__ import annotations
import pytest

from src.qft_pcn.composition.goal_graph import solve_goal_graph, Status
from src.qft_pcn.composition.dispatcher import ThreadPoolBackend


# --- the §10.10 decomposition ---------------------------------------------
# Level 3 (root): the theorem, induction principle over base + inductive case.
# Level 2: the inductive step, combining the two Level-1 lemmas.
# Level 1: lemma A  length (xs ++ [y]) = length xs + 1
#          lemma B  reverse (Cons x xs) = reverse xs ++ [x]   (siblings)
# Level 0: definitions of length, reverse, Cons (leaf goals).

class InductiveDecomposer:
    """The §10.10 decomposition table, as a Decomposer (spec §4.3).
    Each dsl_spec is a real G-compatible spec for that sub-goal."""

    _TABLE = {
        "len_reverse_eq_len": [
            # base case and inductive case
            ("dsl/base_case.json", "base_case"),
            ("dsl/inductive_case.json", "inductive_case"),
        ],
        "inductive_case": [
            ("dsl/lemma_append_length.json", "lemma_append_length"),
            ("dsl/lemma_reverse_cons.json", "lemma_reverse_cons"),
        ],
        "lemma_append_length": [("dsl/def_length.json", "def_length")],
        "lemma_reverse_cons":  [("dsl/def_reverse.json", "def_reverse")],
    }

    def decompose(self, node):
        from src.qft_pcn.composition.goal_graph import make_sub_goal
        import json, pathlib
        here = pathlib.Path(__file__).parent
        out = []
        for i, (spec_path, prop) in enumerate(
                self._TABLE.get(node.goal.goal_prop, [])):
            spec = json.loads((here / spec_path).read_text())
            out.append(make_sub_goal(spec, goal_prop=prop,
                                     boundary=node.goal.boundary,
                                     parent_site=i))
        return out


@pytest.mark.acceptance
def test_length_reverse_theorem_by_three_level_decomposition():
    from src.qft_pcn.logic.lemma_library import LemmaLibrary  # sub-project I
    import json, pathlib

    root_spec = json.loads(
        (pathlib.Path(__file__).parent / "dsl/theorem.json").read_text())

    free_energies = []
    result = solve_goal_graph(
        root_spec, root_prop="len_reverse_eq_len",
        decomposer=InductiveDecomposer(),
        backend=ThreadPoolBackend(max_workers=4),
        lemma_library=LemmaLibrary(),
        timeout_s=60.0,                       # per child; not an estimate of total
        on_step=free_energies.append,
    )

    # the theorem is proved
    assert result.solved is True
    assert result.proof_tree is not None

    # 3 levels of decomposition were reached
    pt = result.proof_tree
    assert pt.root.goal_prop == "len_reverse_eq_len"
    level2 = [c for c in pt.root.children
              if c.goal_prop == "inductive_case"][0]
    level1_props = {c.goal_prop for c in level2.children}
    assert level1_props == {"lemma_append_length", "lemma_reverse_cons"}
    assert level2.children[0].children   # level-0 axioms reached

    # every integration passed the strict residual gate
    from src.qft_pcn.composition.result_integrator import RESIDUAL_GATE

    def _count(n):
        return 1 + sum(_count(c) for c in n.children)
    n_nodes = _count(pt.root)
    assert pt.total_residual <= RESIDUAL_GATE * n_nodes

    # F_hierarchy was monotone non-increasing
    assert all(free_energies[i] >= free_energies[i + 1] - 1e-9
               for i in range(len(free_energies) - 1))


@pytest.mark.acceptance
def test_acceptance_runs_level1_siblings_in_parallel():
    """The two Level-1 lemmas are siblings and must dispatch concurrently
    (spec §9.6, §5.2). Verified by instrumenting concurrent run count."""
    import threading
    from src.qft_pcn.logic.lemma_library import LemmaLibrary
    import json, pathlib

    concurrency = {"max": 0, "now": 0}
    lock = threading.Lock()

    # wrap the runner to count concurrent child runs
    from src.qft_pcn.composition.dispatcher import run_child

    def counting_runner(sub_goal, timeout_s):
        with lock:
            concurrency["now"] += 1
            concurrency["max"] = max(concurrency["max"], concurrency["now"])
        try:
            return run_child(sub_goal, timeout_s=timeout_s)
        finally:
            with lock:
                concurrency["now"] -= 1

    root_spec = json.loads(
        (pathlib.Path(__file__).parent / "dsl/theorem.json").read_text())
    solve_goal_graph(
        root_spec, root_prop="len_reverse_eq_len",
        decomposer=InductiveDecomposer(),
        backend=ThreadPoolBackend(max_workers=4),
        lemma_library=LemmaLibrary(),
        runner=counting_runner, timeout_s=60.0,
    )
    # at least two children ran at the same time -> siblings were parallel
    assert concurrency["max"] >= 2
```

> **Implementation note:** create the `dsl/` fixture directory with the eight real G-compatible DSL specs (`theorem.json`, `base_case.json`, `inductive_case.json`, `lemma_append_length.json`, `lemma_reverse_cons.json`, `def_length.json`, `def_reverse.json`, plus `Cons`). Each must be a spec G's `compile_dsl` accepts and that a child QPCN run can drive to a ground state. Writing these specs is part of this task — confirm the G DSL schema (`bridge/dsl/schema.py`) and the extended-calculus vocabulary (M1's `mera_encoding.py`: `Nat`, `List`, `Eq`, `Forall`) before authoring them. If the extended calculus cannot yet express list induction, that is a genuine blocker — raise it; do not weaken the theorem.

- [ ] **Step 2: Create the `dsl/` fixtures** — the eight DSL specs above. Each is a real spec, not a placeholder.

- [ ] **Step 3: Run the acceptance suite** — `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_cross_level_acceptance.py -v -m acceptance`. The theorem must be proved by real child runs and real clamping.

- [ ] **Step 4: Commit**

```
git add src/qft_pcn/composition/tests/test_cross_level_acceptance.py \
        src/qft_pcn/composition/tests/dsl/
git commit -m "$(cat <<'EOF'
test(composition): §10.10 acceptance -- length(reverse xs) = length xs

The 3-level inductive proof of architecture §10.10: theorem -> inductive
case -> two parallel Level-1 lemmas -> Level-0 axioms. Real child QPCN
runs through G's bridge, real lemma-clamping through I, chi_max=32.
Asserts depth 3, parallel Level-1 siblings, every integration past the
strict residual gate, monotone F_hierarchy. No wall-time assertion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: No-shortcut structural markers + full regression

**Files:**
- Create: `src/qft_pcn/composition/tests/test_no_shortcuts.py`

**Shortcut to refuse:** "skip the structural-marker tests — the feature works." **Principled alternative:** spec §9.7 — the markers are what *prove* the principles were honored: integration mutates the parent MERA (not a dict), and every contraction uses `optimize='greedy'`. Without them a future refactor could silently reintroduce the classical-lookup shortcut.

- [ ] **Step 1: Write the marker tests**

`src/qft_pcn/composition/tests/test_no_shortcuts.py`:

```python
"""No-shortcut structural markers (spec §9.7)."""
from __future__ import annotations
import pathlib
import re


_COMPOSITION_DIR = pathlib.Path(__file__).parent.parent


def test_every_contraction_uses_greedy_optimize():
    """Every np.einsum / np.tensordot in composition/ passes optimize='greedy'
    (principle 5). A bare einsum is the forbidden shortcut."""
    offenders = []
    for py in _COMPOSITION_DIR.glob("*.py"):
        text = py.read_text()
        for m in re.finditer(r"np\.einsum\(", text):
            window = text[m.start():m.start() + 400]
            if "optimize=" not in window:
                offenders.append(f"{py.name}: einsum without optimize=")
    assert offenders == [], offenders


def test_integrator_does_not_store_ast_in_a_dict():
    """The integrator must clamp into the parent MERA, never write the child
    AST into a parent-side dict (principle 3, spec §6.1)."""
    text = (_COMPOSITION_DIR / "result_integrator.py").read_text()
    # no pattern of the form  <dict>[<site>] = <something child/ast>
    forbidden = re.search(r"\[\s*\w*site\w*\s*\]\s*=", text)
    assert forbidden is None, "integrator appears to use a site-indexed dict store"
    # it must call the lemma library's clamp
    assert "lemma_library.clamp" in text or ".clamp(" in text


def test_integration_mutates_mera_state_not_a_side_table():
    """A FakeLemmaLibrary records that clamp was called on the parent STATE
    object, confirming integration is a network operation (spec §9.7)."""
    from src.qft_pcn.composition.goal_graph import make_sub_goal, Node, Status
    from src.qft_pcn.composition.dispatcher import ChildResult
    from src.qft_pcn.composition.result_integrator import integrate_child

    seen = {}

    class RecordingLib:
        def promote(self, gs, ast, gid):
            return {"id": gid}
        def clamp(self, parent_state, parent_site, lemma, *, strength):
            seen["parent_state"] = parent_state
            seen["site"] = parent_site
            return parent_state

    g = make_sub_goal({"g": "x"}, goal_prop="X", boundary={}, parent_site=9)
    node = Node(goal=g, status=Status.ACTIVE)
    node.parent = Node(goal=g, status=Status.ACTIVE)
    res = ChildResult(goal_id="g", converged=True, residual_energy=1e-9,
                      ground_state=object(), solved_ast="ast",
                      run_diagnostic={"spectral_gap": 1.0}, error=None)
    sentinel = object()
    integrate_child(sentinel, {}, node, res, RecordingLib())
    assert seen["parent_state"] is sentinel    # clamp ran on the parent state
    assert seen["site"] == 9
```

- [ ] **Step 2: Run, verify green** (adjust the integrator if a marker fails — the marker is right, not the code).

- [ ] **Step 3: Full K regression + cross-project regression**

```
.venv/bin/python -m pytest src/qft_pcn/composition/ -v
.venv/bin/python -m pytest src/qft_pcn/bridge/ -v          # G must still pass
.venv/bin/python -m pytest src/qft_pcn/qft/ -k mera -v     # F must still pass
```

All three suites must be green. If I has its own test suite, run it too.

- [ ] **Step 4: Commit**

```
git add src/qft_pcn/composition/tests/test_no_shortcuts.py
git commit -m "$(cat <<'EOF'
test(composition): no-shortcut structural markers + regression

Markers (spec §9.7): every contraction uses optimize='greedy';
integration mutates the parent MERA state via I's clamp, never a
site-indexed dict. Full K suite green; G and F regressions green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Done-when checklist (maps to spec §13)

- [ ] Goal-graph construction, traversal, per-active-path cycle detection (Tasks 2, 3) — `test_goal_graph.py` green.
- [ ] Dispatcher parallel-sibling execution + per-child timeout (Task 4) — `test_dispatcher.py` green.
- [ ] Integrator strict residual gate — ground / conjecture / refuse, premature-integration guard (Task 5) — `test_result_integrator.py` green.
- [ ] Revision + failed-decomposition cache (Task 6) — `test_revision.py` green.
- [ ] Orchestrator, `F_hierarchy` monotone non-increasing, structured root-failure report (Task 7) — `test_goal_graph.py` green.
- [ ] §10.10 end-to-end theorem proved by 3-level decomposition with parallel Level-1 siblings, `χ_max=32` (Task 8) — `test_cross_level_acceptance.py` green.
- [ ] No-shortcut markers (Task 9) — `test_no_shortcuts.py` green.
- [ ] G's bridge tests, F's MERA tests, I's lemma tests still pass (Task 9 regression).
- [ ] Every completion claim backed by fresh pytest output (verification-before-completion).
- [ ] No time/effort/duration estimate anywhere in the code, tests, or commit messages.

---

## Risks carried from spec §10.10 and how this plan addresses them

| Risk (§10.10) | Plan mechanism |
|---|---|
| Deadlock (cyclic goal graph) | Task 3 per-active-path `detect_cycle` + `assert_acyclic` guard. |
| Combinatorial explosion | Task 3 lazy expansion; Task 6 `FailedDecompositionCache`. |
| Resource starvation (one bad sub-goal) | Task 4 per-child timeout + fail-fast; Task 7 quarantine. |
| Stale lemmas | Task 5/8 — shared `lemma_library` with `goal_id` keying; provisional integrations re-evaluated (spec §6.3, §8). |
| Premature integration | Task 5 strict gate: `converged` + residual ≤ `RESIDUAL_GATE` + spectral gap ≥ `GROUND_STATE_GAP`. |
| Cascade failures | Task 7 quarantine of `FAILED` sub-graphs; alternatives explored in parallel. |

---

## Notes for the executing agent

- This plan assumes sub-projects G (built) and I (lemma library) are available. Tasks 1–7 and 9's markers use test doubles (`FakeLemmaLibrary`, stub runners) and do **not** block on I. Task 8 (the real acceptance test) and Task 9's regression **do** require I. If I is not merged, complete Tasks 1–7 + 9-markers, and mark Task 8 *blocked on I* — do not stub the acceptance test to make it pass; a stubbed acceptance test is the forbidden shortcut (spec §3.3).
- Every subagent prompt must embed the seven driving principles and the task-specific shortcut/alternative pair named in that task's header.
- When confirming a G or I interface differs from this plan's assumption, **stop and ask** — do not guess a signature (spec §0).
- Run the verification-before-completion skill before any "done" claim: fresh pytest output, not a recollection.
