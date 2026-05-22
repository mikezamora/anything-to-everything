# MERA-Native Logic Encoder Implementation Plan — Part 1 of 2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the MERA-native logic encoder — an extended-calculus AST encoded onto a single MERA tree with one species-register per 16-dimensional leaf, node-major.

**Architecture:** An N-node AST becomes 5N species-leaves (kind/type/bid/value/tobl per node), padded to a power of two, on one MERA tree. Concrete programs encode to a product MERA via `MERA.from_product`; hole-bearing programs get genuine tree entanglement. Decoder measures leaves, groups by node, rebuilds the AST.

**Tech Stack:** Python 3.11, numpy 1.26, pytest. Built on `src/qft_pcn/qft/mera.py` (sub-project F) and the existing `src/qft_pcn/logic/` MPS encoder front-half (`_resolve.py`, `_serialize.py`, `_types.py`).

**Spec:** `docs/superpowers/specs/2026-05-22-mera-native-encoder-design.md` — authoritative. When this plan and the spec disagree, the spec wins.

**Test runner:** `.venv/bin/python -m pytest <path> -v`.

---

## Driving principles (from spec §1 — non-negotiable; embed in every subagent prompt)

1. **Don't stop digging at hard field-crossings.** Principled path over easy-but-wrong path.
2. **Binding is genuine entanglement, never a classical lookup.** Concrete program → product MERA (a definite correlation IS a product state — correct, not a shortcut). Hole-bearing program → genuine tree entanglement carried by disentanglers/isometries.
3. **No large dense tensor, ever.** Every leaf is 16-dimensional by construction. No `16**k` dense operator for k>2.
4. **OOM means optimize, not shrink.**
5. **`optimize='greedy'` on every einsum.**
6. **Reuse F's MERA substrate (`qft/mera.py`) and the MPS encoder front-half. Do not reinvent or modify them.**

---

## Pre-existing worktree state

Many unrelated modified files exist (tauri-app/, docker-compose.yml, root *.md). Leave them alone. Stage only files each task names.

`src/qft_pcn/logic/ast.py` already contains `Rec` (sub-project F, behind an `EXPERIMENTAL_REC` flag), `HoleVar`, `TypeHole`. Reconcile `Rec` with this plan's `Fix` in Task 1.

---

## Task 1: Extended-calculus AST nodes

**Files:**
- Modify: `src/qft_pcn/logic/ast.py`
- Test: `src/qft_pcn/tests/test_mera_ast.py` (create)

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_ast.py`:

```python
"""Tests for the extended-calculus AST nodes (spec §3.1, §6.3)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import (
    Node, Var, Zero, Succ, NatLit, Nil, Cons, Eq, Forall, Fix,
    Ty, TInt, TNat, TList, TEq, TProp,
)


def test_zero_is_a_node():
    assert isinstance(Zero(), Node)


def test_succ_holds_one_child():
    s = Succ(arg=Zero())
    assert isinstance(s.arg, Zero)


def test_natlit_holds_value():
    assert NatLit(val=3).val == 3


def test_natlit_rejects_negative():
    with pytest.raises(ValueError, match="NatLit value must be >= 0"):
        NatLit(val=-1)


def test_nil_is_a_node():
    assert isinstance(Nil(), Node)


def test_cons_holds_head_and_tail():
    c = Cons(head=NatLit(val=1), tail=Nil())
    assert c.head.val == 1
    assert isinstance(c.tail, Nil)


def test_eq_holds_two_sides():
    e = Eq(lhs=Var(name="x"), rhs=Var(name="x"))
    assert e.lhs.name == "x" and e.rhs.name == "x"


def test_forall_is_a_binder():
    f = Forall(param="n", param_ty=TNat(), body=Var(name="n"))
    assert f.param == "n"
    assert isinstance(f.param_ty, TNat)
    assert isinstance(f.body, Var)


def test_fix_is_a_binder():
    fx = Fix(param="f", param_ty=TArrow_or_skip(), body=Var(name="f"))
    assert fx.param == "f"


def TArrow_or_skip():
    from src.qft_pcn.logic.ast import TArrow
    return TArrow(src=TNat(), dst=TNat())


def test_extended_types_construct():
    assert isinstance(TNat(), Ty)
    assert isinstance(TList(elem=TNat()), Ty)
    assert isinstance(TEq(lhs_ty=TNat()), Ty)
    assert isinstance(TProp(), Ty)


def test_extended_types_equal_by_structure():
    assert TNat() == TNat()
    assert TList(elem=TNat()) == TList(elem=TNat())
    assert TList(elem=TNat()) != TList(elem=TInt())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_ast.py -v`
Expected: ImportError — `Zero`, `Succ`, etc. not defined.

- [ ] **Step 3: Add the extended-calculus nodes to `ast.py`**

In `src/qft_pcn/logic/ast.py`, after the existing `TArrow` type and before `Node`'s subclasses end, add the extended types (frozen dataclasses, alongside `TInt`/`TBool`):

```python
@dataclass(frozen=True)
class TNat(Ty):
    """Peano naturals."""


@dataclass(frozen=True)
class TList(Ty):
    elem: Ty


@dataclass(frozen=True)
class TEq(Ty):
    """The type of a propositional-equality proof; lhs_ty is the type of
    both sides of the equation."""
    lhs_ty: Ty


@dataclass(frozen=True)
class TProp(Ty):
    """The sort of propositions (the type of Forall / Eq results)."""
```

After the existing expression nodes (`Bin`, `HoleVar`, ...), add the extended-calculus expression nodes:

```python
@dataclass
class Zero(Node):
    """The Peano zero."""


@dataclass
class Succ(Node):
    arg: Node


@dataclass
class NatLit(Node):
    """A Peano-natural literal; sugar for Succ^val(Zero)."""
    val: int

    def __post_init__(self) -> None:
        if self.val < 0:
            raise ValueError(f"NatLit value must be >= 0, got {self.val}")


@dataclass
class Nil(Node):
    """The empty list."""


@dataclass
class Cons(Node):
    head: Node
    tail: Node


@dataclass
class Eq(Node):
    """A propositional equality lhs = rhs (a proposition, not a bool)."""
    lhs: Node
    rhs: Node


@dataclass
class Forall(Node):
    """Schematic universal: forall param:param_ty. body. A binder."""
    param: str
    param_ty: Ty
    body: Node


@dataclass
class Fix(Node):
    """Recursion: fix param:param_ty. body, where body may reference param.
    A binder. This is the canonical recursion node for the extended
    calculus; the pre-existing `Rec` node (sub-project F, EXPERIMENTAL_REC
    flag) is its forerunner — if `Rec` is unused elsewhere, prefer `Fix`;
    if `Rec` is referenced, alias `Fix = Rec` to avoid a second node type.
    """
    param: str
    param_ty: Ty
    body: Node
```

Reconcile with `Rec`: run `grep -rn "Rec\b" src/qft_pcn/` — if `Rec` is only defined and not used, leave it; `Fix` is the node this plan uses. If `Rec` is used by F's tests, keep both; `Fix` is still the extended-calculus binder.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_ast.py -v`
Expected: 11 passed.

- [ ] **Step 5: Run the existing ast tests to confirm no regression**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_ast.py -v 2>&1 | tail -5`
Expected: all still pass.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/ast.py src/qft_pcn/tests/test_mera_ast.py
git commit -m "$(cat <<'EOF'
feat(logic/ast): extended-calculus nodes (Nat, List, Eq, Forall, Fix)

Adds Zero, Succ, NatLit, Nil, Cons, Eq, Forall, Fix expression nodes and
TNat, TList, TEq, TProp types for the inductive calculus the §10.8-10.11
composition layer needs. Fix is the canonical recursion binder.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: MERA encoding constants

**Files:**
- Create: `src/qft_pcn/logic/mera_encoding.py`
- Test: `src/qft_pcn/tests/test_mera_encoding.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_encoding.py`:

```python
"""Tests for MERA encoding constants (spec §4, §8.1)."""
from __future__ import annotations
from src.qft_pcn.logic.mera_encoding import (
    MERA_LEAF_DIM, MERA_KIND_CUTOFF, MERA_TYPE_CUTOFF,
    SPECIES_ORDER, SPECIES_LEAF_OFFSET, LEAVES_PER_NODE,
    KIND_ZERO, KIND_SUCC, KIND_NATLIT, KIND_NIL, KIND_CONS,
    KIND_EQ, KIND_FORALL, KIND_FIX,
    TYPE_NAT, TYPE_LIST, TYPE_EQ, TYPE_PROP,
)


def test_leaf_dim_is_16():
    assert MERA_LEAF_DIM == 16


def test_kind_and_type_cutoffs_fit_in_leaf():
    assert MERA_KIND_CUTOFF == 16
    assert MERA_TYPE_CUTOFF == 16


def test_species_order_is_canonical():
    assert SPECIES_ORDER == ("kind", "type", "bid", "value", "tobl")
    assert LEAVES_PER_NODE == 5


def test_species_leaf_offset():
    assert SPECIES_LEAF_OFFSET == {
        "kind": 0, "type": 1, "bid": 2, "value": 3, "tobl": 4,
    }


def test_extended_kind_indices_are_8_through_15():
    indices = [KIND_ZERO, KIND_SUCC, KIND_NATLIT, KIND_NIL,
               KIND_CONS, KIND_EQ, KIND_FORALL, KIND_FIX]
    assert indices == [8, 9, 10, 11, 12, 13, 14, 15]


def test_extended_type_indices_within_16():
    for t in (TYPE_NAT, TYPE_LIST, TYPE_EQ, TYPE_PROP):
        assert 0 <= t < 16


def test_existing_kind_constants_unchanged():
    # mera_encoding re-exports the base kinds unchanged from encoding.py
    from src.qft_pcn.logic.encoding import KIND_PAD, KIND_VAR, KIND_BIN
    from src.qft_pcn.logic.mera_encoding import (
        KIND_PAD as M_PAD, KIND_VAR as M_VAR, KIND_BIN as M_BIN,
    )
    assert (M_PAD, M_VAR, M_BIN) == (KIND_PAD, KIND_VAR, KIND_BIN)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_encoding.py -v`
Expected: ImportError on `mera_encoding`.

- [ ] **Step 3: Implement `mera_encoding.py`**

Create `src/qft_pcn/logic/mera_encoding.py`:

```python
"""Constants for the MERA-native logic encoder (spec §4, §8.1).

Extends — does NOT modify — the MPS-side constants in encoding.py.
encoding.py keeps KIND_CUTOFF = 8 etc. for the MPS stack; this module
adds MERA_*_CUTOFF = 16 and the extended-calculus basis indices.
"""
from __future__ import annotations

from .encoding import (
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN,
    TYPE_NONE, TYPE_INT, TYPE_BOOL,
    TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI, TYPE_ARR_BB, TYPE_ARR_NESTED,
)

# Every MERA leaf is 16-dimensional (the max species cutoff, value's 16).
MERA_LEAF_DIM = 16
MERA_KIND_CUTOFF = 16
MERA_TYPE_CUTOFF = 16

# Node-major species-leaf layout: 5 leaves per AST node.
SPECIES_ORDER = ("kind", "type", "bid", "value", "tobl")
LEAVES_PER_NODE = 5
SPECIES_LEAF_OFFSET = {name: i for i, name in enumerate(SPECIES_ORDER)}

# Extended-calculus kind indices (base kinds 0-7 from encoding.py).
KIND_ZERO = 8
KIND_SUCC = 9
KIND_NATLIT = 10
KIND_NIL = 11
KIND_CONS = 12
KIND_EQ = 13
KIND_FORALL = 14
KIND_FIX = 15

# Extended-calculus type tags (base tags 0-7 from encoding.py).
TYPE_NAT = 8
TYPE_LIST = 9
TYPE_EQ = 10
TYPE_PROP = 11
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_encoding.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/mera_encoding.py src/qft_pcn/tests/test_mera_encoding.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_encoding): constants for the MERA-native encoder

MERA_LEAF_DIM=16, node-major SPECIES_ORDER, extended-calculus kind
indices (8-15) and type tags. Extends encoding.py without modifying it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Leaf-layout computation

**Files:**
- Create: `src/qft_pcn/logic/_mera_layout.py`
- Test: `src/qft_pcn/tests/test_mera_layout.py`

This task computes the deterministic map: list of `NodeOccupancy` descriptors → leaf count, layer count, per-leaf species/node tables. It does NOT build tensors.

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_layout.py`:

```python
"""Tests for the node-major species-leaf layout (spec §4.3)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic._serialize import serialize_preorder
from src.qft_pcn.logic._mera_layout import compute_layout, MeraLayout


def _layout(src, n_max=32):
    sites = serialize_preorder(parse(src), N=n_max)
    n_nodes = sum(1 for s in sites if s.kind != 0)  # non-PAD count
    return compute_layout(n_nodes)


def test_layout_leaf_count_is_5n_padded_to_power_of_2():
    lay = compute_layout(n_nodes=2)        # 2 nodes -> 10 leaves -> pad to 16
    assert lay.n_leaves == 16
    assert lay.L == 4                       # log2(16)


def test_layout_pads_up():
    lay = compute_layout(n_nodes=7)        # 35 leaves -> pad to 64
    assert lay.n_leaves == 64
    assert lay.L == 6


def test_species_of_leaf_is_node_major():
    lay = compute_layout(n_nodes=2)
    # node 0: leaves 0-4, node 1: leaves 5-9, rest PAD
    assert lay.species_of_leaf[:10] == [
        "kind", "type", "bid", "value", "tobl",
        "kind", "type", "bid", "value", "tobl",
    ]
    assert all(s == "PAD" for s in lay.species_of_leaf[10:])


def test_node_of_leaf():
    lay = compute_layout(n_nodes=2)
    assert lay.node_of_leaf[:10] == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert all(n == -1 for n in lay.node_of_leaf[10:])


def test_species_leaf_helpers():
    lay = compute_layout(n_nodes=3)
    # the bid leaf of node 2 is 5*2 + 2 = 12
    assert lay.leaf_of(node=2, species="bid") == 12
    assert lay.leaf_of(node=0, species="kind") == 0


def test_one_node_pads_to_8():
    lay = compute_layout(n_nodes=1)        # 5 leaves -> pad to 8
    assert lay.n_leaves == 8
    assert lay.L == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_layout.py -v`
Expected: ImportError on `_mera_layout`.

- [ ] **Step 3: Implement `_mera_layout.py`**

Create `src/qft_pcn/logic/_mera_layout.py`:

```python
"""Node-major species-leaf layout for the MERA-native encoder (spec §4.3).

An N-node AST -> 5N species-leaves, node i at leaves [5i, 5i+5), padded
with PAD-leaves to the next power of two. Pure computation; no tensors.
"""
from __future__ import annotations

from dataclasses import dataclass

from .mera_encoding import SPECIES_ORDER, LEAVES_PER_NODE, SPECIES_LEAF_OFFSET


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@dataclass
class MeraLayout:
    n_nodes: int
    n_leaves: int                  # 5*n_nodes padded to a power of two
    L: int                         # log2(n_leaves)
    species_of_leaf: list[str]     # length n_leaves; species name or "PAD"
    node_of_leaf: list[int]        # length n_leaves; AST node index or -1

    def leaf_of(self, node: int, species: str) -> int:
        """Absolute leaf index of `species` of AST `node`."""
        return LEAVES_PER_NODE * node + SPECIES_LEAF_OFFSET[species]


def compute_layout(n_nodes: int) -> MeraLayout:
    """Build the MeraLayout for an n_nodes-node AST."""
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
    raw_leaves = LEAVES_PER_NODE * n_nodes
    n_leaves = _next_power_of_two(raw_leaves)
    L = n_leaves.bit_length() - 1   # log2 for an exact power of two
    species_of_leaf: list[str] = []
    node_of_leaf: list[int] = []
    for leaf in range(n_leaves):
        if leaf < raw_leaves:
            node = leaf // LEAVES_PER_NODE
            species = SPECIES_ORDER[leaf % LEAVES_PER_NODE]
            species_of_leaf.append(species)
            node_of_leaf.append(node)
        else:
            species_of_leaf.append("PAD")
            node_of_leaf.append(-1)
    return MeraLayout(
        n_nodes=n_nodes, n_leaves=n_leaves, L=L,
        species_of_leaf=species_of_leaf, node_of_leaf=node_of_leaf,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_layout.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/_mera_layout.py src/qft_pcn/tests/test_mera_layout.py
git commit -m "$(cat <<'EOF'
feat(logic/_mera_layout): node-major species-leaf layout

Maps an N-node AST to 5N species-leaves padded to a power of two.
Pure computation: leaf count, layer count, per-leaf species/node tables.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Per-leaf basis vectors for concrete nodes

**Files:**
- Create: `src/qft_pcn/logic/_mera_leaves.py`
- Test: `src/qft_pcn/tests/test_mera_leaves.py`

Turns one `NodeOccupancy` into its five 16-dimensional one-hot leaf vectors. Concrete (hole-free) only; holes are Part 2.

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_leaves.py`:

```python
"""Tests for per-leaf basis-vector construction (spec §6.2 step 4)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic._serialize import serialize_preorder
from src.qft_pcn.logic._types import compute_site_types
from src.qft_pcn.logic._mera_leaves import node_leaf_vectors
from src.qft_pcn.logic.mera_encoding import (
    MERA_LEAF_DIM, KIND_VAR, KIND_LAM,
)
from src.qft_pcn.logic.encoding import KIND_PAD


def _occ_and_types(src):
    ast = parse(src)
    sites = serialize_preorder(ast, N=8)
    types = compute_site_types(ast, sites)
    return sites, types


def test_returns_five_vectors_each_16dim():
    sites, types = _occ_and_types(r"\x:Int. x")
    vecs = node_leaf_vectors(sites[0], types[0])
    assert len(vecs) == 5
    for v in vecs:
        assert v.shape == (MERA_LEAF_DIM,)
        assert np.isclose(np.linalg.norm(v), 1.0)


def test_kind_leaf_is_one_hot_at_kind_index():
    sites, types = _occ_and_types(r"\x:Int. x")
    # site 0 is the Lam.
    vecs = node_leaf_vectors(sites[0], types[0])
    kind_vec = vecs[0]
    assert np.argmax(np.abs(kind_vec)) == KIND_LAM


def test_pad_node_is_all_pad_basis():
    sites, types = _occ_and_types(r"\x:Int. x")
    # site 2+ are PAD (only 2 AST nodes).
    vecs = node_leaf_vectors(sites[2], types[2])
    assert np.argmax(np.abs(vecs[0])) == KIND_PAD


def test_var_leaf_records_bid():
    sites, types = _occ_and_types(r"\x:Int. x")
    # site 1 is Var(x).
    vecs = node_leaf_vectors(sites[1], types[1])
    assert np.argmax(np.abs(vecs[0])) == KIND_VAR
    # bid leaf (index 2) is one-hot at the var's bid value (non-PAD).
    bid_vec = vecs[2]
    assert np.isclose(np.linalg.norm(bid_vec), 1.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_leaves.py -v`
Expected: ImportError on `_mera_leaves`.

- [ ] **Step 3: Implement `_mera_leaves.py`**

The MPS encoder already computes per-site `(kind, type, bid, value)` indices in `src/qft_pcn/logic/_tensors.py` via `_local_kind_type_value` and `_local_bid_for_kind`, and the tobl tag via `_typing_extension`. Reuse those exact index computations — do not re-derive them. Read `_tensors.py` to find the helpers; the node's five species indices are exactly what the MPS encoder writes into its factored site tensor.

Create `src/qft_pcn/logic/_mera_leaves.py`:

```python
"""Per-leaf basis vectors for one AST node (spec §6.2 step 4).

Reuses the MPS encoder's per-species index computation so the MERA and
MPS encoders agree on what each species register holds. Concrete
(hole-free) nodes only; HoleVar superpositions are handled in Part 2.
"""
from __future__ import annotations

import numpy as np

from ._serialize import NodeOccupancy
from ._tensors import _local_kind_type_value, _local_bid_for_kind
from ._typing_extension import tobl_tag_for_site   # see note below
from .mera_encoding import MERA_LEAF_DIM


def _one_hot(index: int) -> np.ndarray:
    v = np.zeros(MERA_LEAF_DIM, dtype=complex)
    v[index] = 1.0
    return v


def node_leaf_vectors(occ: NodeOccupancy, type_tag: int) -> list[np.ndarray]:
    """The five 16-dim one-hot leaf vectors for one concrete AST node.

    Order: [kind, type, bid, value, tobl].
    """
    kind_idx, type_idx, value_idx = _local_kind_type_value(occ, type_tag)
    bid_idx = _local_bid_for_kind(occ.kind, occ)
    tobl_idx = tobl_tag_for_site(occ)
    for name, idx in (("kind", kind_idx), ("type", type_idx),
                      ("bid", bid_idx), ("value", value_idx),
                      ("tobl", tobl_idx)):
        if not 0 <= idx < MERA_LEAF_DIM:
            raise ValueError(
                f"{name} index {idx} out of leaf range [0, {MERA_LEAF_DIM})")
    return [_one_hot(kind_idx), _one_hot(type_idx), _one_hot(bid_idx),
            _one_hot(value_idx), _one_hot(tobl_idx)]
```

**Note on `tobl_tag_for_site`:** the tobl ("type obligation") tag is computed by sub-project B's encoder extension. Read `src/qft_pcn/logic/_typing_extension.py` to find the exact function that yields a site's tobl index; if it is named differently, use the actual name. If tobl is computed inline in `encoder.py` rather than exposed as a function, extract a small pure helper `tobl_tag_for_site(occ) -> int` into `_typing_extension.py` and import it. The tobl index must match what the MPS encoder writes (cross-substrate anchor, §9.7).

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_leaves.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/_mera_leaves.py src/qft_pcn/tests/test_mera_leaves.py
git commit -m "$(cat <<'EOF'
feat(logic/_mera_leaves): per-leaf basis vectors for concrete AST nodes

One AST node -> five 16-dim one-hot leaf vectors (kind/type/bid/value/
tobl), reusing the MPS encoder's per-species index computation so both
substrates agree on register contents.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `encode_mera` for concrete programs + `MeraEncodingMeta`

**Files:**
- Create: `src/qft_pcn/logic/mera_encoder.py`
- Test: `src/qft_pcn/tests/test_mera_encoder_concrete.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_encoder_concrete.py`:

```python
"""Tests for encode_mera on concrete (hole-free) programs (spec §6)."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera, MeraEncodingMeta
from src.qft_pcn.qft.mera import MERA
from src.qft_pcn.logic.encoding import (
    EncodingTooLarge, IllScopedVar, IntLiteralOutOfRange,
)


def test_encode_returns_mera_and_meta():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    assert isinstance(state, MERA)
    assert isinstance(meta, MeraEncodingMeta)


def test_encoded_state_is_unit_norm():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    assert np.isclose(state.norm_sq(), 1.0, atol=1e-10)


def test_meta_leaf_count_is_5n_padded():
    state, meta = encode_mera(parse(r"\x:Int. x"))   # 2 nodes
    assert meta.n_nodes == 2
    assert meta.n_leaves == 16        # 10 -> pad 16
    assert state.N == 16


def test_meta_records_binder_and_use_leaves():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    # The Lam (node 0) is a binder; Var (node 1) uses it.
    assert 0 in meta.binder_leaves
    # use_to_binder maps the Var's bid leaf to the Lam's bid leaf.
    assert len(meta.use_to_binder) >= 1


def test_p5_unit_norm():
    state, meta = encode_mera(parse(r"(\x:Int. (\y:Int. x + y)(3))(4)"))
    assert np.isclose(state.norm_sq(), 1.0, atol=1e-10)


def test_encoding_too_large_raises():
    with pytest.raises(EncodingTooLarge):
        encode_mera(parse(r"\x:Int. x + x + x"), n_nodes_max=3)


def test_ill_scoped_var_raises():
    with pytest.raises(IllScopedVar):
        encode_mera(parse("undefined_name"))


def test_int_literal_out_of_range_raises():
    with pytest.raises(IntLiteralOutOfRange):
        encode_mera(parse(r"\x:Int. x + 99"))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_encoder_concrete.py -v`
Expected: ImportError on `mera_encoder`.

- [ ] **Step 3: Implement `mera_encoder.py` (concrete path)**

Create `src/qft_pcn/logic/mera_encoder.py`:

```python
"""MERA-native logic encoder (spec §6).

Concrete (hole-free) programs encode to a product MERA via
MERA.from_product. Hole-bearing programs (Part 2) start from the product
MERA and apply entangling gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ast import Node, Lam, Var, Forall, Fix, HoleVar
from ._serialize import serialize_preorder, NodeOccupancy
from ._types import compute_site_types
from ._mera_layout import compute_layout, MeraLayout
from ._mera_leaves import node_leaf_vectors
from .mera_encoding import (
    MERA_LEAF_DIM, KIND_PAD, SPECIES_LEAF_OFFSET, LEAVES_PER_NODE,
)
from .encoding import KIND_PAD as _ENC_KIND_PAD
from src.qft_pcn.qft.mera import MERA


@dataclass
class MeraEncodingMeta:
    n_nodes: int
    n_leaves: int
    L: int
    leaf_dim: int
    species_of_leaf: list[str]
    node_of_leaf: list[int]
    site_to_ast_path: dict[int, tuple[int, ...]]
    binder_leaves: dict[int, int]      # binder AST node -> its bid leaf
    use_to_binder: dict[int, int]      # use's bid leaf -> binder's bid leaf
    layout: MeraLayout = field(repr=False, default=None)


def _pad_leaf_vector() -> np.ndarray:
    v = np.zeros(MERA_LEAF_DIM, dtype=complex)
    v[KIND_PAD] = 1.0    # PAD basis state, index 0
    return v


def _has_holes(ast: Node) -> bool:
    """True if the AST contains a HoleVar (Part-2 path)."""
    found = [False]

    def _walk(n: Node) -> None:
        if isinstance(n, HoleVar):
            found[0] = True
            return
        for attr in ("body", "fn", "arg", "lhs", "rhs", "cond",
                      "then_b", "else_b", "head", "tail"):
            child = getattr(n, attr, None)
            if isinstance(child, Node):
                _walk(child)
    _walk(ast)
    return found[0]


def encode_mera(ast: Node, n_nodes_max: int = 32,
                chi_layer: int = 16) -> tuple[MERA, MeraEncodingMeta]:
    """Encode an AST into a unit-norm MERA (spec §6).

    Concrete path only in this task; if the AST has holes, raise
    NotImplementedError (Part 2 supplies the hole path).
    """
    if _has_holes(ast):
        raise NotImplementedError(
            "hole-bearing encoding is implemented in Part 2 (Task 10)")

    # Front-half: reuse the MPS encoder pipeline.
    sites = serialize_preorder(ast, N=n_nodes_max)
    type_tags = compute_site_types(ast, sites)
    n_nodes = sum(1 for occ in sites if occ.kind != _ENC_KIND_PAD)

    layout = compute_layout(n_nodes)

    # Build the 5N + pad leaf vectors, node-major.
    leaf_vectors: list[np.ndarray] = []
    for node_idx in range(n_nodes):
        five = node_leaf_vectors(sites[node_idx], type_tags[node_idx])
        leaf_vectors.extend(five)
    while len(leaf_vectors) < layout.n_leaves:
        leaf_vectors.append(_pad_leaf_vector())

    # Concrete program -> product MERA.
    state = MERA.from_product(leaf_vectors, chi_layer=chi_layer)
    state.normalize()

    # Binder / use bookkeeping for the meta.
    binder_leaves: dict[int, int] = {}
    use_to_binder: dict[int, int] = {}
    for node_idx in range(n_nodes):
        occ = sites[node_idx]
        if isinstance(getattr(occ, "node", None), (Lam, Forall, Fix)) \
                or occ.kind in _binder_kinds():
            binder_leaves[node_idx] = layout.leaf_of(node_idx, "bid")
        if occ.var_ref is not None:
            use_leaf = layout.leaf_of(node_idx, "bid")
            binder_node = occ.var_ref.binder_site
            use_to_binder[use_leaf] = layout.leaf_of(binder_node, "bid")

    meta = MeraEncodingMeta(
        n_nodes=n_nodes, n_leaves=layout.n_leaves, L=layout.L,
        leaf_dim=MERA_LEAF_DIM,
        species_of_leaf=layout.species_of_leaf,
        node_of_leaf=layout.node_of_leaf,
        site_to_ast_path={k: occ.ast_path for k, occ in enumerate(sites)
                          if occ.kind != _ENC_KIND_PAD},
        binder_leaves=binder_leaves,
        use_to_binder=use_to_binder,
        layout=layout,
    )
    return state, meta


def _binder_kinds() -> set[int]:
    """Kind indices that introduce a binder (Lam, Forall, Fix)."""
    from .mera_encoding import KIND_LAM, KIND_FORALL, KIND_FIX
    return {KIND_LAM, KIND_FORALL, KIND_FIX}
```

**Note:** the `NodeOccupancy` fields (`kind`, `var_ref`, `ast_path`, `binder_ref`) are defined in `_serialize.py` — read it to confirm exact field names. If `NodeOccupancy` does not expose enough to identify binders, use `occ.kind in _binder_kinds()` as the test in the code above does; drop the `getattr(occ, "node", ...)` branch if `NodeOccupancy` has no `node` attribute. Adjust to the actual `_serialize.py` API; the test (`test_meta_records_binder_and_use_leaves`) is the oracle.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_encoder_concrete.py -v`
Expected: 8 passed.

- [ ] **Step 5: Run the full logic suite for regressions**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/ -k "mera or logic_ast" --timeout=30 -q 2>&1 | tail -8`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/mera_encoder.py src/qft_pcn/tests/test_mera_encoder_concrete.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_encoder): encode_mera for concrete programs

Concrete (hole-free) ASTs encode to a product MERA: front-half reuses
the MPS encoder pipeline (serialize, types), then builds 5N+pad
node-major species-leaf vectors and calls MERA.from_product.
MeraEncodingMeta records binder/use bid-leaf bookkeeping. Hole-bearing
ASTs raise NotImplementedError pending Part 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `decode_mera` — leaf measurement + structural parse

**Files:**
- Create: `src/qft_pcn/logic/mera_decoder.py`
- Test: `src/qft_pcn/tests/test_mera_decoder.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_decoder.py`:

```python
"""Tests for decode_mera (spec §7)."""
from __future__ import annotations
from src.qft_pcn.logic.ast import parse, Lam, Var, App, IntLit, TInt
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_decoder import decode_mera, DecodeResult


def test_decode_identity_lambda():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    res = decode_mera(state, meta)
    assert isinstance(res, DecodeResult)
    assert res.residual_norm < 1e-10
    assert isinstance(res.ast, Lam)
    assert isinstance(res.ast.body, Var)
    assert res.ast.body.name == res.ast.param


def test_decode_intlit():
    state, meta = encode_mera(parse(r"\x:Int. 3"))
    res = decode_mera(state, meta)
    assert isinstance(res.ast.body, IntLit)
    assert res.ast.body.val == 3


def test_decode_application():
    state, meta = encode_mera(parse(r"\f:Int->Int. \x:Int. f x"))
    res = decode_mera(state, meta)
    lam_f = res.ast
    assert isinstance(lam_f, Lam)
    app = lam_f.body.body
    assert isinstance(app, App)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_decoder.py -v`
Expected: ImportError on `mera_decoder`.

- [ ] **Step 3: Implement `mera_decoder.py`**

The MPS decoder (`src/qft_pcn/logic/decoder.py`) already implements the structural parse: given per-site `(kind, type, bid, value)` tuples it rebuilds the AST and wires `Var`→`Lam` via a binder stack. Reuse that parse. The only MERA-specific part is measuring leaves.

Create `src/qft_pcn/logic/mera_decoder.py`:

```python
"""MERA -> AST decoder (spec §7).

Measures each leaf (argmax of its per-leaf marginal), groups leaves into
nodes (5 per node, node-major), recovers per-node (kind,type,bid,value,
tobl), then reuses the MPS decoder's structural parse to rebuild the AST.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ast import Node
from .mera_encoder import MeraEncodingMeta
from .mera_encoding import MERA_LEAF_DIM, LEAVES_PER_NODE
from src.qft_pcn.qft.mera import MERA


@dataclass
class DecodeResult:
    ast: Node
    residual_norm: float


def _leaf_marginal(state: MERA, leaf: int) -> np.ndarray:
    """The (MERA_LEAF_DIM,) probability vector for one leaf.

    Computed via MERA.local_expectation against each basis projector.
    The operator is 16x16 — trivially cheap (spec §9.5 note).
    """
    p = np.empty(MERA_LEAF_DIM, dtype=float)
    for b in range(MERA_LEAF_DIM):
        proj = np.zeros((MERA_LEAF_DIM, MERA_LEAF_DIM), dtype=complex)
        proj[b, b] = 1.0
        p[b] = float(np.real(state.local_expectation(leaf, proj)))
    total = p.sum()
    if total > 1e-15:
        p = p / total
    return p


def decode_mera(state: MERA, meta: MeraEncodingMeta) -> DecodeResult:
    """Deterministic argmax decode of a (concrete-program) MERA state."""
    # Measure every non-PAD leaf; group into per-node 5-tuples.
    per_node: list[tuple[int, int, int, int, int]] = []
    residual = 0.0
    for node_idx in range(meta.n_nodes):
        idxs = []
        for offset in range(LEAVES_PER_NODE):
            leaf = LEAVES_PER_NODE * node_idx + offset
            p = _leaf_marginal(state, leaf)
            b = int(np.argmax(p))
            idxs.append(b)
            residual = max(residual, 1.0 - float(p[b]))
        per_node.append(tuple(idxs))   # (kind, type, bid, value, tobl)

    ast = _structural_parse(per_node, meta)
    return DecodeResult(ast=ast, residual_norm=residual)


def _structural_parse(per_node, meta: MeraEncodingMeta) -> Node:
    """Rebuild the AST from per-node (kind,type,bid,value,tobl) tuples.

    Delegates to the MPS decoder's parse machinery. The MPS decoder
    consumes per-SITE tuples; here each node supplies the same
    information from its 5 leaves. Import and reuse the MPS decoder's
    internal parse helper (`decoder._parse_*` — read decoder.py for the
    exact name) rather than re-implementing the Var/Lam binder-stack
    wiring.
    """
    from . import decoder as mps_decoder
    # mps_decoder exposes a structural parse over (kind,type,bid,value)
    # tuples. The MERA tuples additionally carry tobl at index 4; the
    # parse ignores tobl (it is a typing-obligation tag, not structural).
    # Adapt the per_node 5-tuples to whatever shape the MPS parse expects;
    # read decoder.py to find the exact entry point. If the MPS parse is
    # not factored out as a reusable function, extract it into a small
    # `parse_kind_stream(tuples) -> Node` helper in decoder.py and import
    # it here. The structural-parse logic must NOT be duplicated.
    return mps_decoder.parse_kind_stream(
        [(k, t, b, v) for (k, t, b, v, _o) in per_node])
```

**Note:** the MPS `decoder.py` may not currently expose `parse_kind_stream`. If it does not, extract the existing per-site structural-parse loop from `decode()` into a pure function `parse_kind_stream(tuples: list[tuple[int,int,int,int]]) -> Node` and have both `decode()` and `decode_mera()` call it. Do not copy-paste the parse logic. The round-trip tests (Task 7) are the oracle.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_decoder.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/mera_decoder.py src/qft_pcn/tests/test_mera_decoder.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_decoder): MERA -> AST decoder for concrete states

Measures each leaf via 16x16 basis projectors, groups leaves into
per-node 5-tuples, reuses the MPS decoder's structural parse to rebuild
the AST. Var->Lam wiring uses the shared parse helper, not a copy.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Concrete round-trip acceptance (spec §9.1, §9.7)

**Files:**
- Create: `src/qft_pcn/tests/test_mera_roundtrip.py`

- [ ] **Step 1: Write the tests**

Create `src/qft_pcn/tests/test_mera_roundtrip.py`:

```python
"""MERA encoder round-trip acceptance (spec §9.1, §9.7)."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_decoder import decode_mera
from src.qft_pcn.logic.decoder import ast_alpha_eq

# Extended-calculus programs use the parser; if the parser does not yet
# accept Succ/Cons/Eq syntax, build those ASTs directly (see P6-P8).
from src.qft_pcn.logic.ast import (
    Succ, Zero, Cons, NatLit, Nil, Lam, Eq, Var, TNat,
)

_STLC = {
    "P1": r"\x:Int. x",
    "P2": r"(\x:Int. x + 1)(2)",
    "P3": r"\f:Int->Int. \x:Int. f (f x)",
    "P4": r"if 1 < 2 then ((\x:Bool. x)(true)) else false",
    "P5": r"(\x:Int. (\y:Int. x + y)(3))(4)",
}


@pytest.mark.parametrize("name,src", list(_STLC.items()))
def test_roundtrip_stlc(name, src):
    expected = parse(src)
    state, meta = encode_mera(expected)
    assert np.isclose(state.norm_sq(), 1.0, atol=1e-10), f"{name} not unit norm"
    res = decode_mera(state, meta)
    assert res.residual_norm < 1e-10, f"{name} residual {res.residual_norm}"
    assert ast_alpha_eq(res.ast, expected), (
        f"{name}: {res.ast!r} != {expected!r}")


def test_roundtrip_p6_nat():
    expected = Succ(arg=Succ(arg=Zero()))
    state, meta = encode_mera(expected)
    res = decode_mera(state, meta)
    assert ast_alpha_eq(res.ast, expected)


def test_roundtrip_p7_list():
    expected = Cons(head=NatLit(val=1), tail=Cons(head=NatLit(val=2),
                                                  tail=Nil()))
    state, meta = encode_mera(expected)
    res = decode_mera(state, meta)
    assert ast_alpha_eq(res.ast, expected)


def test_roundtrip_p8_eq():
    expected = Lam(param="x", param_ty=TNat(),
                   body=Eq(lhs=Var(name="x"), rhs=Var(name="x")))
    state, meta = encode_mera(expected)
    res = decode_mera(state, meta)
    assert ast_alpha_eq(res.ast, expected)


def test_cross_substrate_anchor_p1_to_p5():
    """MERA and MPS encoders decode to the same AST (spec §9.7)."""
    from src.qft_pcn.logic.encoder import encode as encode_mps
    from src.qft_pcn.logic.decoder import decode as decode_mps
    for name, src in _STLC.items():
        p = parse(src)
        mera_ast = decode_mera(*encode_mera(p)).ast
        mps_state, mps_meta = encode_mps(p, N=32)
        mps_ast = decode_mps(mps_state, mps_meta).ast
        assert ast_alpha_eq(mera_ast, mps_ast), f"{name} substrate mismatch"
```

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_roundtrip.py -v --timeout=60`

If any fail, debug per-program: inspect `decode_mera(...).ast` vs `parse(src)`. The likely fault sites are `_mera_leaves.py` (wrong species index) or `mera_decoder.py` (parse adaptation). Use systematic-debugging — root-cause before fixing.

Expected: 8 round-trip tests + 1 cross-substrate test pass.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_mera_roundtrip.py
git commit -m "$(cat <<'EOF'
test(mera): concrete round-trip acceptance P1-P8 + cross-substrate anchor

Eight programs (5 STLC + 3 extended-calculus) round-trip through
encode_mera -> decode_mera to alpha-equivalence. The cross-substrate
test confirms the MERA and MPS encoders agree on P1-P5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

**End of Part 1.** Part 2 covers the hole-bearing encoding path (genuine tree entanglement), `sample_mera`, the `mera_window_expectation` helper, and the remaining acceptance suite (layout, alpha-renaming, structural marker, PAD vacuum, error paths, k-leaf cross-check) plus public re-exports and final verification. See `2026-05-22-mera-native-encoder-part2.md`.
