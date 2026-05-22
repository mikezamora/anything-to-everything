# MERA-Native Typing + Evaluation Hamiltonians Implementation Plan — Part 1 of 2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Use superpowers:test-driven-development for every task and superpowers:systematic-debugging when a test fails unexpectedly.

**Goal:** Build the MERA-native typing and evaluation Hamiltonians (sub-project M2): retarget B and C onto the M1 MERA species-leaf substrate, add the extended-calculus typing/evaluation rules, and add MERA imaginary-time evolution.

**Architecture:** Every Hamiltonian term is a *factored operator* — a `dict[leaf_index -> (16,16) matrix]`, identity on unlisted leaves — evaluated by M1's `mera_window_expectation_factored`. A per-node term spans the 5 leaves of one AST node; a two-node term spans the leaves of two (or more) nodes, with the MERA tree carrying the correlation. No `16**k` dense operator is ever formed. Reduction is ground-state finding: imaginary-time evolution under `H_typing + H_eval` drives the MERA toward the normal form.

**Tech Stack:** Python 3.11, numpy, pytest. Built on M1 (`src/qft_pcn/logic/{mera_encoder,mera_decoder,_mera_window,mera_encoding,_mera_layout}.py`) and F's `src/qft_pcn/qft/mera.py`.

**Spec:** `docs/superpowers/specs/2026-05-22-mera-hamiltonians-design.md` — authoritative. When this plan and the spec disagree, the spec wins.

**Test runner:** `.venv/bin/python -m pytest <path> -v`.

**Part 1** covers: the `MeraEncodingMeta.children_of_node` extension, the shared per-rule factored-operator helpers, `MeraTypingHamiltonian` with the STLC rules, and the extended-calculus typing rules.
**Part 2** (`2026-05-22-mera-hamiltonians-part2.md`) covers: `MeraEvalHamiltonian`, the transition gates, `compose_mera_hamiltonians`, `mera_evolution_logic.py`, the `Fix` recursion demo, and the full acceptance + cross-substrate suite.

---

## Driving principles (from spec §1 — non-negotiable; embed in every subagent prompt)

1. **No time/effort/duration estimates anywhere.** Critical standing directive.
2. **Binding is genuine entanglement, never a classical lookup.** A binder/use typing rule reads the binder node's leaf and the use node's leaf through the encoded MERA; the tree carries the correlation. `meta.use_to_binder` is read for *addressing* (which leaves), never to look up a classical type. **Shortcut (forbidden):** "look up `meta.use_to_binder`, read the binder's classical `param_ty` from meta, compare." **Alternative (required):** a two-node factored operator on the use node's and binder node's `type` leaves, evaluated against the real tree-entangled state.
3. **No dense operator at scale.** A per-node term is 5 leaves (`16**5 ≈ 1e6`, square `1e12`); a two-node term is 10. **Shortcut (forbidden):** build a `16**5 × 16**5` dense per-node operator. **Alternative (required):** the factored `dict[leaf -> (16,16)]` form, always.
4. **OOM means optimize, not shrink.** Never reduce node budget / leaf dim / `chi_layer` to dodge an OOM.
5. **`optimize='greedy'` on every einsum.**
6. **Constraint-based evaluation only.** No explicit beta-reduction rewriter, no classical interpreter, no Lindblad path. **Shortcut (forbidden):** pattern-match `App(Lam, arg)`, substitute, write back. **Alternative (required):** Hermitian penalty + transition gate; the spectrum does the work.
7. **The Hamiltonian is structural.** Built from `MeraEncodingMeta` only. No Python AST walk during construction or evaluation. **Shortcut (forbidden):** a per-AST Hamiltonian populated by walking the program. **Alternative (required):** structural rules + layout addressing.
8. **Reuse M1 and F; do not reinvent or modify the MPS logic stack.**

---

## Pre-existing worktree state

Many unrelated modified files exist (tauri-app/, QFT_PCN_ARCHITECTURE.md, lib/, docs/ viz files). Leave them alone. Stage only the files each task names.

`src/qft_pcn/logic/` already contains M1's deliverables (`mera_encoder.py`, `mera_decoder.py`, `_mera_window.py`, `mera_encoding.py`, `_mera_layout.py`, `_mera_leaves.py`, `_mera_holes.py`) and the MPS logic stack (`typing_hamiltonian.py`, `evaluation_hamiltonian.py`, `compose.py`, `factored_evolution.py`, `_factored_expectation.py`, `_eval_terms.py`). M2 does NOT modify the MPS stack.

Before starting: read `src/qft_pcn/logic/_mera_window.py`, `mera_encoder.py`, `_mera_layout.py`, and `src/qft_pcn/qft/mera.py`'s `apply_local_gate` / `apply_two_site_gate` / `local_expectation` signatures. Read `typing_hamiltonian.py` and `evaluation_hamiltonian.py` — M2 mirrors their `.terms`/`.term_energy`/`.residuals`/`.total_energy` shape.

---

## Task 1: `MeraEncodingMeta.children_of_node` extension

The M2 term enumerator needs, per parent AST node, the indices of its child nodes (spec §5.6). Add an additive field to `MeraEncodingMeta` and populate it in `encode_mera`.

**Files:**
- Modify: `src/qft_pcn/logic/mera_encoder.py`
- Test: `src/qft_pcn/tests/test_mera_children.py` (create)

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_children.py`:

```python
"""Tests for MeraEncodingMeta.children_of_node (spec §5.6)."""
from __future__ import annotations
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


def test_children_of_node_exists():
    _, meta = encode_mera(parse(r"\x:Int. x"))
    assert hasattr(meta, "children_of_node")
    assert isinstance(meta.children_of_node, dict)


def test_lam_has_one_child():
    # \x:Int. x  -> node 0 is Lam, node 1 is Var(x); Var is Lam's child.
    _, meta = encode_mera(parse(r"\x:Int. x"))
    assert meta.children_of_node.get(0) == [1]
    assert meta.children_of_node.get(1, []) == []


def test_app_has_two_children():
    # (\x:Int. x + 1)(2): root App has fn (the Lam) and arg (IntLit 2).
    _, meta = encode_mera(parse(r"(\x:Int. x + 1)(2)"))
    kids = meta.children_of_node.get(0)
    assert kids is not None and len(kids) == 2


def test_bin_has_two_children():
    # node for `x + 1`: lhs Var, rhs IntLit.
    _, meta = encode_mera(parse(r"\x:Int. x + 1"))
    # find the Bin node: it has exactly two children both reachable.
    bin_nodes = [n for n, ks in meta.children_of_node.items()
                 if len(ks) == 2]
    assert len(bin_nodes) >= 1


def test_children_indices_are_contiguous_preorder():
    # children always have index > parent (pre-order layout).
    _, meta = encode_mera(parse(r"(\x:Int. (\y:Int. x + y)(3))(4)"))
    for parent, kids in meta.children_of_node.items():
        for k in kids:
            assert k > parent
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_children.py -v`
Expected: `AttributeError` — `MeraEncodingMeta` has no `children_of_node`.

- [ ] **Step 3: Add and populate `children_of_node`**

In `src/qft_pcn/logic/mera_encoder.py`, add the field to the `MeraEncodingMeta` dataclass (after `use_to_binder`):

```python
    children_of_node: dict[int, list[int]] = field(default_factory=dict)
```

The pre-order serialization (`serialize_preorder`) lays a node's children
contiguously after it; the structure is recoverable from each
`NodeOccupancy.ast_path`. In `encode_mera`, after `sites` and `n_nodes` are
known and before assembling `meta`, compute:

```python
    # children_of_node: parent AST node index -> child node indices.
    # A node c is a child of node p iff c's ast_path == p's ast_path + (j,)
    # for some j (one level deeper, directly under p). This is layout
    # metadata from the pre-order walk (spec §5.6 (b)) — NOT an AST walk.
    children_of_node: dict[int, list[int]] = {}
    path_to_node: dict[tuple, int] = {}
    for node_idx in range(n_nodes):
        path_to_node[sites[node_idx].ast_path] = node_idx
    for node_idx in range(n_nodes):
        children_of_node[node_idx] = []
    for node_idx in range(n_nodes):
        path = sites[node_idx].ast_path
        if len(path) >= 1:
            parent_path = path[:-1]
            parent = path_to_node.get(parent_path)
            if parent is not None and parent != node_idx:
                children_of_node[parent].append(node_idx)
    for node_idx in children_of_node:
        children_of_node[node_idx].sort()
```

Pass `children_of_node=children_of_node` into the `MeraEncodingMeta(...)`
constructor call.

**Note:** confirm `NodeOccupancy.ast_path` is a tuple keyed from the root
(read `_serialize.py`). If `ast_path` is not present or not a path tuple, use
the alternative: `serialize_preorder` already knows parent/child structure
during its walk — extend it to also return a `children` map, or expose the
child indices on each `NodeOccupancy`. The tests above are the oracle; adjust
to the actual `_serialize.py` API but do NOT walk the AST a second time.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_children.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run M1's encoder tests for no regression**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_encoder_concrete.py src/qft_pcn/tests/test_mera_roundtrip.py -v --timeout=60 2>&1 | tail -8`
Expected: all still pass (the new field is additive).

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/mera_encoder.py src/qft_pcn/tests/test_mera_children.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_encoder): MeraEncodingMeta.children_of_node

Additive metadata field: parent AST node index -> child node indices,
computed from the pre-order ast_path map (not a second AST walk). M2's
typing/eval term enumerators need parent->child structure to build the
two-node obligation and redex terms.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Shared per-rule factored-operator helpers

The 16-dim per-leaf projector helpers and the leaf-addressing helpers, shared by every typing and evaluation rule.

**Files:**
- Create: `src/qft_pcn/logic/_mera_typing_rules.py`
- Test: `src/qft_pcn/tests/test_mera_rule_helpers.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_rule_helpers.py`:

```python
"""Tests for the shared MERA Hamiltonian rule helpers (spec §5.2)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic._mera_typing_rules import (
    leaf_proj, leaf_proj_one_minus, leaf_proj_set, node_leaf,
)
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


def test_leaf_proj_is_16x16_one_hot():
    p = leaf_proj(5)
    assert p.shape == (16, 16)
    assert p[5, 5] == 1.0
    assert np.count_nonzero(p) == 1


def test_leaf_proj_one_minus_is_complement():
    p = leaf_proj(3)
    q = leaf_proj_one_minus(3)
    assert np.allclose(p + q, np.eye(16))


def test_leaf_proj_set_sums_basis_projectors():
    p = leaf_proj_set([1, 2, 4])
    assert p[1, 1] == 1.0 and p[2, 2] == 1.0 and p[4, 4] == 1.0
    assert np.count_nonzero(p) == 3


def test_node_leaf_addresses_via_layout():
    _, meta = encode_mera(parse(r"\x:Int. x"))
    # node 0, species "kind" -> leaf 0; node 1, "bid" -> leaf 7.
    assert node_leaf(meta, 0, "kind") == 0
    assert node_leaf(meta, 1, "bid") == 7
    assert node_leaf(meta, 1, "type") == 6
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_rule_helpers.py -v`
Expected: ImportError on `_mera_typing_rules`.

- [ ] **Step 3: Implement `_mera_typing_rules.py`**

Create `src/qft_pcn/logic/_mera_typing_rules.py`:

```python
"""Shared factored-operator helpers for the MERA Hamiltonians (spec §5.2).

Every Hamiltonian term is a dict[leaf_index -> (16,16) matrix], identity
on unlisted leaves, evaluated by mera_window_expectation_factored. These
helpers build the 16-dim per-leaf projectors and address leaves through
the M1 layout. No 16**k dense operator is ever formed (spec §1.3).
"""
from __future__ import annotations

import numpy as np

from .mera_encoding import MERA_LEAF_DIM


def leaf_proj(idx: int) -> np.ndarray:
    """|idx><idx| on a 16-dim leaf."""
    if not 0 <= idx < MERA_LEAF_DIM:
        raise ValueError(f"leaf basis index {idx} out of [0, {MERA_LEAF_DIM})")
    p = np.zeros((MERA_LEAF_DIM, MERA_LEAF_DIM), dtype=complex)
    p[idx, idx] = 1.0
    return p


def leaf_proj_one_minus(idx: int) -> np.ndarray:
    """I - |idx><idx|."""
    return np.eye(MERA_LEAF_DIM, dtype=complex) - leaf_proj(idx)


def leaf_proj_set(idxs) -> np.ndarray:
    """Sum of |i><i| for i in idxs."""
    out = np.zeros((MERA_LEAF_DIM, MERA_LEAF_DIM), dtype=complex)
    for i in idxs:
        if not 0 <= i < MERA_LEAF_DIM:
            raise ValueError(f"leaf basis index {i} out of range")
        out[i, i] = 1.0
    return out


def leaf_identity() -> np.ndarray:
    return np.eye(MERA_LEAF_DIM, dtype=complex)


def node_leaf(meta, node: int, species: str) -> int:
    """Absolute leaf index of `species` of AST `node`, via the M1 layout."""
    return meta.layout.leaf_of(node, species)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_rule_helpers.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/_mera_typing_rules.py src/qft_pcn/tests/test_mera_rule_helpers.py
git commit -m "$(cat <<'EOF'
feat(logic/_mera_typing_rules): shared factored-operator rule helpers

16-dim per-leaf projector builders (leaf_proj, leaf_proj_one_minus,
leaf_proj_set) and layout-addressing (node_leaf). The building blocks of
every M2 typing/eval rule; no 16**k dense operator anywhere.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `MeraTypingHamiltonian` skeleton + term enumeration

The class shell, `MeraTypingTerm`, the 16-rule term enumeration, and the `.term_energy`/`.total_energy`/`.residuals` dispatch — with every rule energy function stubbed to return 0.0 for now (filled in Tasks 4–6).

**Files:**
- Create: `src/qft_pcn/logic/mera_typing_hamiltonian.py`
- Test: `src/qft_pcn/tests/test_mera_typing_skeleton.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_typing_skeleton.py`:

```python
"""Tests for the MeraTypingHamiltonian skeleton (spec §5.1, §5.3)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import (
    MeraTypingHamiltonian, MeraTypingTerm, MeraTermNotFound,
)


def test_constructs_from_meta_only():
    _, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    assert isinstance(H.terms, list)
    assert len(H.terms) > 0


def test_terms_are_mera_typing_terms():
    _, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    for t in H.terms:
        assert isinstance(t, MeraTypingTerm)
        assert isinstance(t.rule_id, str)
        assert 0 <= t.node < meta.n_nodes
        assert t.arity in (1, 2)


def test_all_sixteen_rule_names_present():
    _, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    names = {t.rule_id for t in H.terms}
    expected = {
        "T-Lit-Int", "T-Lit-Bool", "T-Bin-Arith", "T-Bin-Cmp",
        "T-Var", "T-Abs", "T-App-Arrow", "T-Obligation",
        "T-Zero", "T-Succ", "T-NatLit", "T-Nil", "T-Cons",
        "T-Eq", "T-Forall", "T-Fix",
    }
    assert expected <= names


def test_term_energy_unknown_term_raises():
    _, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    bogus = MeraTypingTerm(rule_id="T-Nonsense", node=0, arity=1)
    with pytest.raises(MeraTermNotFound):
        state, _ = encode_mera(parse(r"\x:Int. x"))
        H.term_energy(state, bogus)


def test_total_energy_returns_float():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    e = H.total_energy(state)
    assert isinstance(e, float)


def test_residuals_keyed_by_rule_and_node():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    r = H.residuals(state)
    for key, val in r.items():
        assert isinstance(key, tuple) and len(key) == 2
        assert isinstance(key[0], str) and isinstance(key[1], int)
        assert val >= -1e-12


def test_structural_no_ast_attributes():
    _, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    for attr in dir(H):
        if attr.startswith("_"):
            continue
        low = attr.lower()
        for bad in ("ast", "tree", "node_obj", "walk"):
            assert bad not in low, f"H.{attr} suggests AST dependency"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_typing_skeleton.py -v`
Expected: ImportError on `mera_typing_hamiltonian`.

- [ ] **Step 3: Implement the skeleton**

Create `src/qft_pcn/logic/mera_typing_hamiltonian.py`:

```python
"""MERA-native typing Hamiltonian (sub-project M2; spec §5, §6).

H_typing = sum over rules, over AST nodes, of H_rule(node), each a
factored projector operator on the 5-leaf node windows of the M1 MERA
layout. <psi|H_typing|psi> = 0 exactly when the encoded AST is
well-typed; per-term energies localize violations (consumed by M3).

Per spec §1.7 the Hamiltonian is STRUCTURAL: built from MeraEncodingMeta
only. No AST is walked here. Every term is evaluated via
mera_window_expectation_factored — the dense 16**k operator is NEVER
materialized (spec §1.3).
"""
from __future__ import annotations

from dataclasses import dataclass

from src.qft_pcn.qft.mera import MERA
from .mera_encoder import MeraEncodingMeta
from .mera_encoding import (
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN,
    KIND_ZERO, KIND_SUCC, KIND_NATLIT, KIND_NIL, KIND_CONS,
    KIND_EQ, KIND_FORALL, KIND_FIX,
)
from . import _mera_typing_rules as R


# ---- Rule names ---------------------------------------------------------

RULE_T_LIT_INT = "T-Lit-Int"
RULE_T_LIT_BOOL = "T-Lit-Bool"
RULE_T_BIN_ARITH = "T-Bin-Arith"
RULE_T_BIN_CMP = "T-Bin-Cmp"
RULE_T_VAR = "T-Var"
RULE_T_ABS = "T-Abs"
RULE_T_APP_ARROW = "T-App-Arrow"
RULE_T_OBLIGATION = "T-Obligation"
RULE_T_ZERO = "T-Zero"
RULE_T_SUCC = "T-Succ"
RULE_T_NATLIT = "T-NatLit"
RULE_T_NIL = "T-Nil"
RULE_T_CONS = "T-Cons"
RULE_T_EQ = "T-Eq"
RULE_T_FORALL = "T-Forall"
RULE_T_FIX = "T-Fix"

# Per-node (arity 1) rules.
_ONE_NODE_RULES = (
    RULE_T_LIT_INT, RULE_T_LIT_BOOL, RULE_T_BIN_ARITH, RULE_T_BIN_CMP,
    RULE_T_ZERO, RULE_T_NATLIT, RULE_T_NIL,
)
# Two-node (arity 2) rules; anchored at the primary node.
_TWO_NODE_RULES = (
    RULE_T_VAR, RULE_T_ABS, RULE_T_APP_ARROW, RULE_T_OBLIGATION,
    RULE_T_SUCC, RULE_T_CONS, RULE_T_EQ, RULE_T_FORALL, RULE_T_FIX,
)


# ---- Exceptions ---------------------------------------------------------


class MeraTypingHamiltonianError(Exception):
    """Base error for MeraTypingHamiltonian operations."""


class MeraTermNotFound(MeraTypingHamiltonianError):
    def __init__(self, term):
        super().__init__(f"term {term} not in this Hamiltonian's term list")


# ---- MeraTypingTerm -----------------------------------------------------


@dataclass(frozen=True)
class MeraTypingTerm:
    """A typing-rule term, addressable by (rule_id, node, arity)."""
    rule_id: str
    node: int
    arity: int


# ---- The Hamiltonian class ---------------------------------------------


class MeraTypingHamiltonian:
    """Structural typing Hamiltonian over a MERA leaf layout (spec §5)."""

    def __init__(self, meta: MeraEncodingMeta):
        self.meta = meta
        self.N_nodes = meta.n_nodes
        self.terms: list[MeraTypingTerm] = self._enumerate_terms()
        self._terms_set = frozenset(self.terms)

    def _enumerate_terms(self) -> list[MeraTypingTerm]:
        """Deterministic enumeration: every rule at every AST node.

        A rule's energy function is zero when the node's kind does not
        match the rule (e.g. T-Lit-Int at a Lam node), so enumerating
        every rule at every node is correct and keeps the Hamiltonian
        structural (spec §1.7).
        """
        terms: list[MeraTypingTerm] = []
        for rule in _ONE_NODE_RULES:
            for n in range(self.N_nodes):
                terms.append(MeraTypingTerm(rule, n, 1))
        for rule in _TWO_NODE_RULES:
            for n in range(self.N_nodes):
                terms.append(MeraTypingTerm(rule, n, 2))
        return terms

    def term_energy(self, state: MERA, term: MeraTypingTerm) -> float:
        if term not in self._terms_set:
            raise MeraTermNotFound(term)
        fn = _RULE_DISPATCH.get(term.rule_id)
        if fn is None:
            raise MeraTermNotFound(term)
        return float(fn(state, self.meta, term.node))

    def total_energy(self, state: MERA) -> float:
        return sum(self.term_energy(state, t) for t in self.terms)

    def residuals(self, state: MERA) -> dict:
        return {(t.rule_id, t.node): self.term_energy(state, t)
                for t in self.terms}


# ---- Per-rule energy functions (filled in Tasks 4-6) --------------------
# Each returns <psi|H_rule(node)|psi> as a real float. Stubs for now.


def _stub(state, meta, node) -> float:
    return 0.0


_RULE_DISPATCH: dict = {
    RULE_T_LIT_INT: _stub,
    RULE_T_LIT_BOOL: _stub,
    RULE_T_BIN_ARITH: _stub,
    RULE_T_BIN_CMP: _stub,
    RULE_T_VAR: _stub,
    RULE_T_ABS: _stub,
    RULE_T_APP_ARROW: _stub,
    RULE_T_OBLIGATION: _stub,
    RULE_T_ZERO: _stub,
    RULE_T_SUCC: _stub,
    RULE_T_NATLIT: _stub,
    RULE_T_NIL: _stub,
    RULE_T_CONS: _stub,
    RULE_T_EQ: _stub,
    RULE_T_FORALL: _stub,
    RULE_T_FIX: _stub,
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_typing_skeleton.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/mera_typing_hamiltonian.py src/qft_pcn/tests/test_mera_typing_skeleton.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_typing_hamiltonian): structural skeleton + term enumeration

MeraTypingHamiltonian / MeraTypingTerm with the 16-rule term enumeration
and the .term_energy / .total_energy / .residuals dispatch. Rule energy
functions are stubbed (filled in subsequent tasks). Built from
MeraEncodingMeta only — structural per spec §1.7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: STLC per-node typing rules (T-Lit-Int, T-Lit-Bool, T-Bin-Arith, T-Bin-Cmp)

Fill in the four per-node STLC rules. Each is a factored operator on one node's leaves; the content is copied from B §3.1-§3.3.

**Files:**
- Modify: `src/qft_pcn/logic/_mera_typing_rules.py`
- Modify: `src/qft_pcn/logic/mera_typing_hamiltonian.py`
- Test: `src/qft_pcn/tests/test_mera_typing_pernode.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_typing_pernode.py`:

```python
"""Tests for the STLC per-node typing rules (spec §5.3, B §3.1-§3.3)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import (
    MeraTypingHamiltonian, MeraTypingTerm,
)
from src.qft_pcn.logic._mera_typing_rules import mutate_leaf
from src.qft_pcn.logic.mera_encoding import TYPE_INT, TYPE_BOOL


def _term_energy(src, rule, node):
    state, meta = encode_mera(parse(src))
    H = MeraTypingHamiltonian(meta)
    return H.term_energy(state, MeraTypingTerm(rule, node, 1)), state, meta


def test_lit_int_zero_on_well_typed():
    # \x:Int. 3  -> node 1 is IntLit 3, correctly typed Int.
    e, _, _ = _term_energy(r"\x:Int. 3", "T-Lit-Int", 1)
    assert abs(e) < 1e-9


def test_lit_int_fires_on_mistyped_intlit():
    state, meta = encode_mera(parse(r"\x:Int. 3"))
    # Surgically retag node 1's `type` leaf Int -> Bool.
    type_leaf = meta.layout.leaf_of(1, "type")
    state = mutate_leaf(state, type_leaf, TYPE_INT, TYPE_BOOL)
    H = MeraTypingHamiltonian(meta)
    e = H.term_energy(state, MeraTypingTerm("T-Lit-Int", 1, 1))
    assert e > 0.99


def test_lit_bool_zero_on_well_typed():
    e, _, _ = _term_energy(r"\x:Int. true", "T-Lit-Bool", 1)
    assert abs(e) < 1e-9


def test_lit_bool_fires_on_mistyped_boollit():
    state, meta = encode_mera(parse(r"\x:Int. true"))
    type_leaf = meta.layout.leaf_of(1, "type")
    state = mutate_leaf(state, type_leaf, TYPE_BOOL, TYPE_INT)
    H = MeraTypingHamiltonian(meta)
    e = H.term_energy(state, MeraTypingTerm("T-Lit-Bool", 1, 1))
    assert e > 0.99


def test_bin_arith_zero_on_well_typed():
    # \x:Int. 1 + 2: the Bin node is correctly typed Int.
    state, meta = encode_mera(parse(r"\x:Int. 1 + 2"))
    H = MeraTypingHamiltonian(meta)
    total = sum(H.term_energy(state, MeraTypingTerm("T-Bin-Arith", n, 1))
                for n in range(meta.n_nodes))
    assert abs(total) < 1e-9


def test_bin_arith_fires_on_mistyped_bin():
    state, meta = encode_mera(parse(r"\x:Int. 1 + 2"))
    # Find the Bin node (kind == KIND_BIN); retag its type Int -> Bool.
    from src.qft_pcn.logic.mera_encoding import KIND_BIN
    bin_node = None
    for n in range(meta.n_nodes):
        kind_leaf = meta.layout.leaf_of(n, "kind")
        # measure: the encoder produced a product state, argmax is exact.
        proj = np.zeros((16, 16), dtype=complex)
        # cheap: read via local_expectation against KIND_BIN projector.
        proj[KIND_BIN, KIND_BIN] = 1.0
        if np.real(state.local_expectation(kind_leaf, proj)) > 0.5:
            bin_node = n
            break
    assert bin_node is not None
    type_leaf = meta.layout.leaf_of(bin_node, "type")
    state = mutate_leaf(state, type_leaf, TYPE_INT, TYPE_BOOL)
    H = MeraTypingHamiltonian(meta)
    e = H.term_energy(state, MeraTypingTerm("T-Bin-Arith", bin_node, 1))
    assert e > 0.99
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_typing_pernode.py -v`
Expected: ImportError on `mutate_leaf`, then assertion failures (stubs return 0).

- [ ] **Step 3: Add `mutate_leaf` to `_mera_typing_rules.py`**

Append to `src/qft_pcn/logic/_mera_typing_rules.py`:

```python
def mutate_leaf(state, leaf: int, old_idx: int, new_idx: int):
    """Test helper: swap basis slices `old_idx` and `new_idx` of one leaf
    of a MERA state, returning a mutated copy. Used by the per-rule
    isolation tests to surgically inject a typing violation.

    Applies a 16x16 permutation gate (a swap of two basis vectors) via
    MERA.apply_local_gate — the gate is local, so the rest of the tree
    is untouched.
    """
    import numpy as np
    g = np.eye(MERA_LEAF_DIM, dtype=complex)
    g[old_idx, old_idx] = 0.0
    g[new_idx, new_idx] = 0.0
    g[new_idx, old_idx] = 1.0
    g[old_idx, new_idx] = 1.0
    out = state.copy()
    out.apply_local_gate(leaf, g)
    return out
```

- [ ] **Step 4: Implement the four per-node rules in `mera_typing_hamiltonian.py`**

Replace the four `_stub` entries for the per-node STLC rules with real
functions. Add, after the `_stub` definition (importing the type/value
constants at the top of the module — extend the existing import from
`mera_encoding`):

```python
# Extend the mera_encoding import at the top of the file with:
#   TYPE_INT, TYPE_BOOL, VALUE_PLUS, VALUE_MINUS, VALUE_TIMES,
#   VALUE_LT, VALUE_EQ
# (these come from encoding.py; mera_encoding re-exports the base set —
#  confirm the names and import from whichever module defines them.)


def _window(meta, node, leaf_ops_by_species):
    """Build a factored leaf_ops dict for one node from a
    {species_name: (16,16) op} mapping."""
    return {meta.layout.leaf_of(node, sp): op
            for sp, op in leaf_ops_by_species.items()}


def _energy_t_lit_int(state, meta, node) -> float:
    """B §3.1: P[kind=INT] . (I - P[type=INT])."""
    from ._mera_window import mera_window_expectation_factored
    ops = _window(meta, node, {
        "kind": R.leaf_proj(KIND_INT),
        "type": R.leaf_proj_one_minus(TYPE_INT),
    })
    return float(mera_window_expectation_factored(state, ops).real)


def _energy_t_lit_bool(state, meta, node) -> float:
    """B §3.2."""
    from ._mera_window import mera_window_expectation_factored
    ops = _window(meta, node, {
        "kind": R.leaf_proj(KIND_BOOL),
        "type": R.leaf_proj_one_minus(TYPE_BOOL),
    })
    return float(mera_window_expectation_factored(state, ops).real)


def _energy_t_bin_arith(state, meta, node) -> float:
    """B §3.3a: P[kind=BIN] . P[value in {+,-,*}] . (I - P[type=INT])."""
    from ._mera_window import mera_window_expectation_factored
    ops = _window(meta, node, {
        "kind":  R.leaf_proj(KIND_BIN),
        "value": R.leaf_proj_set([VALUE_PLUS, VALUE_MINUS, VALUE_TIMES]),
        "type":  R.leaf_proj_one_minus(TYPE_INT),
    })
    return float(mera_window_expectation_factored(state, ops).real)


def _energy_t_bin_cmp(state, meta, node) -> float:
    """B §3.3b."""
    from ._mera_window import mera_window_expectation_factored
    ops = _window(meta, node, {
        "kind":  R.leaf_proj(KIND_BIN),
        "value": R.leaf_proj_set([VALUE_LT, VALUE_EQ]),
        "type":  R.leaf_proj_one_minus(TYPE_BOOL),
    })
    return float(mera_window_expectation_factored(state, ops).real)
```

Then update `_RULE_DISPATCH` so `RULE_T_LIT_INT -> _energy_t_lit_int`,
`RULE_T_LIT_BOOL -> _energy_t_lit_bool`, `RULE_T_BIN_ARITH ->
_energy_t_bin_arith`, `RULE_T_BIN_CMP -> _energy_t_bin_cmp`. The others stay
`_stub` until Tasks 5-6.

**Note on the factored product of three projectors:** the factored window
expectation applies the per-leaf operators to a copy of the state and takes
the inner product (M1 `_mera_window.py`). Because each `leaf_proj` /
`leaf_proj_one_minus` is a projector and they act on *distinct leaves*, the
product `⟨ψ| (⊗ P_leaf) |ψ⟩` is exactly what `mera_window_expectation_factored`
computes. Confirm by reading the M1 docstring — if the helper does not handle
a per-leaf non-projector, all M2 operators here are projectors so the path is
sound.

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_typing_pernode.py -v`
Expected: 6 passed. If `test_bin_arith_fires` fails, debug with
systematic-debugging — likely the `value` constant names or the
`mera_window_expectation_factored` product semantics.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/_mera_typing_rules.py src/qft_pcn/logic/mera_typing_hamiltonian.py src/qft_pcn/tests/test_mera_typing_pernode.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_typing): STLC per-node typing rules on the MERA layout

T-Lit-Int, T-Lit-Bool, T-Bin-Arith, T-Bin-Cmp as factored operators on
one node's 5-leaf window, evaluated via mera_window_expectation_factored.
Rule content copied from B §3.1-§3.3; only the substrate changes. Adds a
mutate_leaf test helper for per-rule isolation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: STLC two-node typing rules (T-Var, T-Abs, T-App-Arrow, T-Obligation)

The binder/use and parent/child two-node rules. T-Var realizes binding-as-entanglement on the tree (spec §1.2, §5.4).

**Files:**
- Modify: `src/qft_pcn/logic/mera_typing_hamiltonian.py`
- Test: `src/qft_pcn/tests/test_mera_typing_twonode.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_typing_twonode.py`:

```python
"""Tests for the STLC two-node typing rules (spec §5.4, §5.5)."""
from __future__ import annotations
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import (
    MeraTypingHamiltonian, MeraTypingTerm,
)
from src.qft_pcn.logic._mera_typing_rules import mutate_leaf
from src.qft_pcn.logic.mera_encoding import TYPE_INT, TYPE_BOOL


def _all_rule_energy(state, meta, rule):
    H = MeraTypingHamiltonian(meta)
    return sum(H.term_energy(state, MeraTypingTerm(rule, n, 2))
               for n in range(meta.n_nodes))


def test_t_var_zero_on_well_typed():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    assert abs(_all_rule_energy(state, meta, "T-Var")) < 1e-9


def test_t_var_fires_when_var_type_mismatches_binder():
    # \x:Int. x  -- retag the Var's type Int->Bool; binder param stays Int.
    state, meta = encode_mera(parse(r"\x:Int. x"))
    var_type_leaf = meta.layout.leaf_of(1, "type")   # node 1 is Var
    state = mutate_leaf(state, var_type_leaf, TYPE_INT, TYPE_BOOL)
    assert _all_rule_energy(state, meta, "T-Var") > 0.99


def test_t_abs_zero_on_well_typed():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    assert abs(_all_rule_energy(state, meta, "T-Abs")) < 1e-9


def test_t_app_arrow_zero_on_well_typed():
    state, meta = encode_mera(parse(r"(\x:Int. x)(1)"))
    assert abs(_all_rule_energy(state, meta, "T-App-Arrow")) < 1e-9


def test_t_obligation_zero_on_well_typed():
    state, meta = encode_mera(parse(r"\x:Int. x + 1"))
    assert abs(_all_rule_energy(state, meta, "T-Obligation")) < 1e-9


def test_t_obligation_fires_on_bad_operand():
    # \x:Int. x + true : the BIN's rhs (true) should be Int.
    state, meta = encode_mera(parse(r"\x:Int. x + true"))
    assert _all_rule_energy(state, meta, "T-Obligation") > 0.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_typing_twonode.py -v`
Expected: assertion failures on the `fires` tests (stubs return 0).

- [ ] **Step 3: Implement the four two-node rules**

In `mera_typing_hamiltonian.py`, add the rule functions. Key points:

- **T-Var** (spec §5.4): for node `j`, if `j` is a use site, `meta.use_to_binder`
  maps `j`'s bid leaf `5j+2` to the binder's bid leaf `5i+2`; from that recover
  the binder node `i = (binder_bid_leaf) // 5`. The energy sums over type tags
  `t`: a factored window over leaves `{5j+0: P[kind=VAR], 5i+1: P[type=t],
  5j+1: I-P[type=t]}`. If `j` is not a use site (`5j+2 not in use_to_binder`),
  return 0.0. Read `meta` for *addressing only* (spec §1.2).

```python
def _energy_t_var(state, meta, node) -> float:
    """Spec §5.4: VAR node's type must equal its binder's parameter type.
    Two-node factored term over the use node and its binder node. meta is
    read only to address leaves (spec §1.2); the binder/use correlation
    is carried by the encoded MERA tree.
    """
    from ._mera_window import mera_window_expectation_factored
    from .mera_encoding import MERA_TYPE_CUTOFF
    use_bid = meta.layout.leaf_of(node, "bid")
    binder_bid = meta.use_to_binder.get(use_bid)
    if binder_bid is None:
        return 0.0
    binder_node = binder_bid // 5     # 5 leaves per node, bid is offset 2
    use_kind = meta.layout.leaf_of(node, "kind")
    use_type = meta.layout.leaf_of(node, "type")
    binder_param_type = meta.layout.leaf_of(binder_node, "type")
    total = 0.0
    for t in range(MERA_TYPE_CUTOFF):
        ops = {
            use_kind:          R.leaf_proj(KIND_VAR),
            binder_param_type: R.leaf_proj(t),
            use_type:          R.leaf_proj_one_minus(t),
        }
        total += float(mera_window_expectation_factored(state, ops).real)
    return total
```

  **Caveat — the binder's parameter type leaf.** For `Lam`, the binder's
  `type` leaf holds the LAM's arrow type, not its parameter type. The
  parameter type is `src(arrow)`. Two acceptable realizations: (a) sum over
  arrow tags `a` with `src(a) = s`, projecting the binder `type` leaf onto `a`
  and the use `type` leaf onto `I - P[s]`; (b) the encoder writes the binder's
  parameter type into the binder node's `value` leaf (it does this for
  `Forall`/`Fix` per spec §6.7-§6.8 — extend it to `Lam` too for uniformity).
  **Pick (a)** to avoid a further encoder change: build an `_ARROW_SRC` table
  (copy from `typing_hamiltonian.py` lines ~143-148, retargeted to
  `mera_encoding` tag names) and sum over arrows. For a non-arrow binder
  (`Forall`/`Fix`), the parameter type IS the `type`/`value` leaf directly —
  branch on the binder node's kind (read via a cheap `local_expectation`
  against the kind projectors, or via `meta` — the binder's kind is structural
  layout info, acceptable to read). Document the chosen path in the docstring.

- **T-Abs** (spec §5.4): for a LAM node `i` with body node `b`
  (`b = meta.children_of_node[i][0]`), penalize "LAM type is arrow `a`, body
  type ≠ `dst(a)`" summed over `a`, plus "LAM type is arrow `a`, LAM param
  type ≠ `src(a)`." Use an `_ARROW_SRC` / `_ARROW_DST` table.

- **T-App-Arrow** (B §3.7): for an APP node `i` with fn node
  `f = meta.children_of_node[i][0]`, penalize "APP type is `y`, fn type not in
  the legal-arrows-for-`y` set" — copy the `_APP_FN_ALLOWED_BY_DST` table from
  `typing_hamiltonian.py` lines ~150-159, retargeted to `mera_encoding` names.
  Factored window: `{5i+0: P[kind=APP], 5i+1: P[type=y], 5f+1: I -
  P_set(allowed)}`, summed over `y`.

- **T-Obligation** (spec §5.5): for a parent node `i` whose kind imposes a
  type on a child, for each `(child, expected_type)` derived from the parent's
  kind and `value` leaf, penalize "parent kind = K, child type ≠ expected."
  Enumerate the parent-kind → child-obligation table (LAM body → dst(LAM
  type); APP arg → src(fn type); IF cond → Bool, IF branches → IF type; BIN
  operands → Int). For obligations whose expected type depends on the parent's
  own `type`/`value` leaf, sum over the tag: e.g. IF then-branch obligation is
  `Σ_t P[IF.type=t] · P[then.type≠t]`. Children come from
  `meta.children_of_node[i]` in pre-order (fn before arg, cond/then/else,
  lhs/rhs). Return 0.0 for nodes with no obligation-imposing kind.

For all four: build the factored `dict[leaf -> (16,16)]` and call
`mera_window_expectation_factored`. Update `_RULE_DISPATCH`.

**Confirm the binder-node recovery `binder_bid // 5`.** The `bid` leaf of node
`m` is `5m + 2`, so `binder_node = (binder_bid - 2) // 5`. Use
`(binder_bid - SPECIES_LEAF_OFFSET["bid"]) // LEAVES_PER_NODE` from
`mera_encoding` to be exact — do not hard-code `// 5`. The skeleton above is
indicative; use the offset constants.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_typing_twonode.py -v`
Expected: 6 passed. Debug failures with systematic-debugging — the most likely
fault is the arrow-table tag names or the binder-node index recovery.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/mera_typing_hamiltonian.py src/qft_pcn/tests/test_mera_typing_twonode.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_typing): STLC two-node typing rules on the MERA tree

T-Var, T-Abs, T-App-Arrow, T-Obligation as two-node factored operators.
T-Var reads the binder and use nodes' type leaves through the encoded
MERA — binding-as-entanglement on the tree (spec §1.2), no classical
lookup. Obligations are genuine parent/child two-node terms (spec §5.5),
not one-site tobl reads.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Extended-calculus typing rules (T-Zero, T-Succ, T-NatLit, T-Nil, T-Cons, T-Eq, T-Forall, T-Fix)

The eight calculus-extension rules from spec §6.

**Files:**
- Modify: `src/qft_pcn/logic/mera_typing_hamiltonian.py`
- Test: `src/qft_pcn/tests/test_mera_typing_extended.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_typing_extended.py`:

```python
"""Tests for the extended-calculus typing rules (spec §6)."""
from __future__ import annotations
from src.qft_pcn.logic.ast import (
    parse, Succ, Zero, Cons, NatLit, Nil, Lam, Eq, Var, Forall, Fix, TNat,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import (
    MeraTypingHamiltonian, MeraTypingTerm,
)


def _typing_energy(ast):
    state, meta = encode_mera(ast)
    return MeraTypingHamiltonian(meta).total_energy(state)


def test_succ_succ_zero_well_typed():
    assert abs(_typing_energy(Succ(arg=Succ(arg=Zero())))) < 1e-9


def test_cons_list_well_typed():
    ast = Cons(head=NatLit(val=1),
               tail=Cons(head=NatLit(val=2), tail=Nil()))
    assert abs(_typing_energy(ast)) < 1e-9


def test_eq_reflexivity_well_typed():
    ast = Lam(param="x", param_ty=TNat(),
              body=Eq(lhs=Var(name="x"), rhs=Var(name="x")))
    assert abs(_typing_energy(ast)) < 1e-9


def test_forall_well_typed():
    ast = Forall(param="n", param_ty=TNat(),
                 body=Eq(lhs=Var(name="n"), rhs=Var(name="n")))
    assert abs(_typing_energy(ast)) < 1e-9


def test_succ_of_bool_ill_typed():
    # Succ true : the arg is not a Nat -> T-Succ fires.
    from src.qft_pcn.logic.ast import BoolLit
    state, meta = encode_mera(Succ(arg=BoolLit(val=True)))
    H = MeraTypingHamiltonian(meta)
    res = H.residuals(state)
    fired = [k for k, v in res.items() if v > 0.5]
    assert any(k[0] == "T-Succ" for k in fired)


def test_eq_type_mismatch_ill_typed():
    # Eq (NatLit 1) (BoolLit false): lhs Nat, rhs Bool -> T-Eq fires.
    from src.qft_pcn.logic.ast import BoolLit
    state, meta = encode_mera(Eq(lhs=NatLit(val=1), rhs=BoolLit(val=False)))
    H = MeraTypingHamiltonian(meta)
    res = H.residuals(state)
    assert any(k[0] == "T-Eq" and v > 0.5 for k, v in res.items())
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_typing_extended.py -v`
Expected: assertion failures (stubs return 0); possibly the well-typed tests
already pass trivially (energy 0) — the `ill_typed` tests are the real oracle.

- [ ] **Step 3: Implement the eight extended rules**

In `mera_typing_hamiltonian.py`, implement per spec §6.1-§6.8. All are
factored windows; the only subtlety is the multi-node sets (T-Cons couples
three nodes, T-Eq three, etc. — a single `mera_window_expectation_factored`
call with leaves from all three). Use `meta.children_of_node` for child node
indices.

- `T-Zero` (§6.1): `{5i+0: P[kind=ZERO], 5i+1: I-P[type=NAT]}`.
- `T-Succ` (§6.2): sum of two windows — result type and arg type. Arg node is
  `meta.children_of_node[i][0]`.
- `T-NatLit` (§6.3): `{5i+0: P[kind=NATLIT], 5i+1: I-P[type=NAT]}`.
- `T-Nil` (§6.4): `{5i+0: P[kind=NIL], 5i+1: I-P[type=LIST]}`.
- `T-Cons` (§6.5): three contributions (result List, tail List, head : τ
  summed over τ reading the CONS `value` leaf). Head/tail nodes are
  `children_of_node[i][0]` and `[1]`.
- `T-Eq` (§6.6): result Prop, plus `Σ_τ P[lhs.type=τ] · P[rhs.type≠τ]`.
  lhs/rhs are `children_of_node[i][0]`/`[1]`.
- `T-Forall` (§6.7): result Prop + body Prop. Body is `children_of_node[i][0]`.
- `T-Fix` (§6.8): result : τ and body : τ, τ from the FIX `value` leaf,
  summed over τ. Body is `children_of_node[i][0]`.

Use `KIND_ZERO..KIND_FIX` and `TYPE_NAT..TYPE_PROP` from `mera_encoding`.
Update `_RULE_DISPATCH` for all eight.

**Encoder-contract check.** §6.4/§6.7/§6.8 assume the encoder writes the list
element type into the `Nil`/`Cons` node's `value` leaf and the parameter type
into the `Forall`/`Fix` node's `value` leaf. Verify M1's `_mera_leaves.py` /
`encode_mera` actually does this. If it does NOT (the MPS encoder may not have
needed these for the extended nodes), this is a genuine missing piece — extend
`_mera_leaves.py`'s `node_leaf_vectors` to write the element/parameter type
tag into the `value` leaf for `Nil`/`Cons`/`Forall`/`Fix`, with a focused
test, BEFORE finishing this task. If the AST does not carry the element type
on `Nil` (homogeneous-list inference), then T-Nil/T-Cons's element-type checks
degrade to permissive — document the chosen behavior and keep the structural
checks (result is a `List`). **If this is ambiguous, stop and ask** (spec §0).

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_typing_extended.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run the full M2-typing suite + M1 regression**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/ -k "mera" --timeout=120 -q 2>&1 | tail -10`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/mera_typing_hamiltonian.py src/qft_pcn/tests/test_mera_typing_extended.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_typing): extended-calculus typing rules

T-Zero, T-Succ, T-NatLit, T-Nil, T-Cons, T-Eq, T-Forall, T-Fix as
factored operators per spec §6. Multi-node rules (Cons, Eq) couple
three AST nodes in a single factored window expectation. Well-typed
Nat/List/Eq/Forall programs give <H_typing> ~ 0; ill-typed fire the
attributable rule.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Well-typed / ill-typed acceptance suite for `MeraTypingHamiltonian`

Spec §9.1, §9.2 — the P1-P10 well-typed and IT1-IT5 ill-typed acceptance, plus the structural check (§9.6).

**Files:**
- Create: `src/qft_pcn/tests/test_mera_typing_hamiltonian.py`

- [ ] **Step 1: Write the acceptance tests**

Create `src/qft_pcn/tests/test_mera_typing_hamiltonian.py` covering:

```python
"""Acceptance suite for MeraTypingHamiltonian (spec §9.1, §9.2, §9.6)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import (
    parse, Succ, Zero, Cons, NatLit, Nil, Lam, Eq, Var, Forall, Fix,
    TNat, TArrow, BoolLit,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import (
    MeraTypingHamiltonian, MeraTypingTerm,
)
from src.qft_pcn.logic._mera_typing_rules import mutate_leaf
from src.qft_pcn.logic.mera_encoding import TYPE_INT, TYPE_BOOL

_WELL_TYPED = {
    "P1": parse(r"\x:Int. x"),
    "P2": parse(r"(\x:Int. x + 1)(2)"),
    "P3": parse(r"\f:Int->Int. \x:Int. f (f x)"),
    "P4": parse(r"if 1 < 2 then ((\x:Bool. x)(true)) else false"),
    "P5": parse(r"(\x:Int. (\y:Int. x + y)(3))(4)"),
    "P6": Succ(arg=Succ(arg=Zero())),
    "P7": Cons(head=NatLit(val=1),
               tail=Cons(head=NatLit(val=2), tail=Nil())),
    "P8": Lam(param="x", param_ty=TNat(),
              body=Eq(lhs=Var(name="x"), rhs=Var(name="x"))),
    "P9": Forall(param="n", param_ty=TNat(),
                 body=Eq(lhs=Var(name="n"), rhs=Var(name="n"))),
    "P10": Fix(param="f", param_ty=TArrow(src=TNat(), dst=TNat()),
               body=Lam(param="x", param_ty=TNat(), body=Var(name="x"))),
}


@pytest.mark.parametrize("name", list(_WELL_TYPED))
def test_well_typed_zero_energy(name):
    state, meta = encode_mera(_WELL_TYPED[name])
    H = MeraTypingHamiltonian(meta)
    e = H.total_energy(state)
    assert abs(e) < 1e-9, f"{name}: <H_typing> = {e}; {H.residuals(state)}"


def test_IT1_obligation_violation():
    state, meta = encode_mera(parse(r"\x:Int. x + true"))
    H = MeraTypingHamiltonian(meta)
    assert H.total_energy(state) > 0.5
    big = [k for k, v in H.residuals(state).items() if v > 0.5]
    assert len(big) == 1 and big[0][0] == "T-Obligation"


def test_IT2_succ_violation():
    state, meta = encode_mera(Succ(arg=BoolLit(val=True)))
    H = MeraTypingHamiltonian(meta)
    assert H.total_energy(state) > 0.5
    assert any(k[0] == "T-Succ"
               for k, v in H.residuals(state).items() if v > 0.5)


def test_IT4_eq_violation():
    state, meta = encode_mera(Eq(lhs=NatLit(val=1), rhs=BoolLit(val=False)))
    H = MeraTypingHamiltonian(meta)
    assert any(k[0] == "T-Eq"
               for k, v in H.residuals(state).items() if v > 0.5)


def test_IT5_surgical_lit_violation():
    state, meta = encode_mera(parse(r"\x:Int. 3"))
    type_leaf = meta.layout.leaf_of(1, "type")
    state = mutate_leaf(state, type_leaf, TYPE_INT, TYPE_BOOL)
    H = MeraTypingHamiltonian(meta)
    big = [k for k, v in H.residuals(state).items() if v > 0.5]
    assert ("T-Lit-Int", 1) in big


def test_residuals_sum_to_total():
    state, meta = encode_mera(parse(r"\x:Int. x + true"))
    H = MeraTypingHamiltonian(meta)
    assert abs(H.total_energy(state)
               - sum(H.residuals(state).values())) < 1e-9


def test_structural_same_instance_two_programs():
    # Same node count, two unrelated programs.
    s1, m1 = encode_mera(parse(r"\x:Int. x"))
    s2, m2 = encode_mera(parse(r"\y:Bool. y"))
    # m1 and m2 have the same n_nodes; one Hamiltonian per meta is fine,
    # but verify construction touches no AST.
    H1 = MeraTypingHamiltonian(m1)
    assert abs(H1.total_energy(s1)) < 1e-9
```

(Add an IT3 Cons test consistent with the §6.5 element-type semantics chosen
in Task 6; if `Nil` carries no element type, drop IT3 or make it a structural
violation. Keep the suite consistent with the Task 6 decision.)

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_typing_hamiltonian.py -v --timeout=120`
Expected: all pass. Debug failures per-program with systematic-debugging.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_mera_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
test(mera-typing): P1-P10 well-typed + IT1-IT5 ill-typed acceptance

Ten well-typed STLC + extended-calculus programs give <H_typing> ~ 0;
ill-typed programs give > 0.5 with exactly one attributable per-rule
residual. Structural-Hamiltonian and residual-decomposition checks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

**End of Part 1.** Part 2 (`2026-05-22-mera-hamiltonians-part2.md`) covers `MeraEvalHamiltonian` (redex penalties + transition gates), `compose_mera_hamiltonians`, `mera_evolution_logic.py`, the `Fix` recursion demo, the reduction acceptance suite (E1-E5), the cross-substrate anchor, and final verification.
