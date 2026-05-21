# AST↔MPS Encoder Implementation Plan — Part 2 of 3 (Encoder)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Continues from Part 1** (`2026-05-20-ast-mps-encoder-part1.md`). Assumes Tasks 1–7 are complete: pytest installed, `MPS.inner` added, AST data classes + parser + pretty exist, `logic/encoding.py` constants and exception types exist.

**Driving principle reminder** — see Part 1's preamble. The single most important one to keep front-of-mind during this part: **the `bid` register's bond is genuine entanglement carried by a live-binder channel structure. Do not collapse it to a classical lookup.** The bond-dimension test (Part 3, Task 25) will fail loudly if you do.

This part covers Tasks 8–17 (lexical resolution → pre-order serialization → live-binder bookkeeping → type computation → factored tensor construction → top-level `encode()` → first end-to-end smoke test).

---

## Task 8: Lexical scope resolution (`_resolve_binders`)

**Files:**
- Create: `src/qft_pcn/logic/_resolve.py`.
- Create: `src/qft_pcn/tests/test_logic_resolve.py`.

For each `Var(name)` in the AST, find its binding `Lam` and produce an annotated structure. Also identify the lexical depth at each Var. Raises `IllScopedVar` for unbound names.

- [ ] **Step 1: Write the failing tests**

Create `src/qft_pcn/tests/test_logic_resolve.py`:

```python
"""Tests for logic/_resolve.py (lexical binder resolution)."""

from __future__ import annotations

import pytest

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic._resolve import resolve_binders, ResolvedRef
from src.qft_pcn.logic.encoding import IllScopedVar, TooManyBinders


def _refs(ast):
    out = []
    resolve_binders(ast, on_var=lambda var, ref: out.append((var.name, ref)))
    return out


def test_resolve_identity_lambda():
    ast = parse(r"\x:Int. x")
    refs = _refs(ast)
    # One Var, depth-from-innermost = 0 (its binder is the only one).
    assert len(refs) == 1
    name, ref = refs[0]
    assert name == "x"
    assert isinstance(ref, ResolvedRef)
    assert ref.depth_from_innermost == 0


def test_resolve_outer_binder_used_inside_inner_scope():
    ast = parse(r"\x:Int. \y:Int. x")
    refs = _refs(ast)
    # The Var(x) is inside an inner Lam(y), so x is one step out -> depth 1.
    assert len(refs) == 1
    name, ref = refs[0]
    assert name == "x"
    assert ref.depth_from_innermost == 1


def test_resolve_shadowing():
    ast = parse(r"\x:Int. \x:Int. x")
    # Inner x shadows outer.
    refs = _refs(ast)
    assert len(refs) == 1
    _, ref = refs[0]
    assert ref.depth_from_innermost == 0


def test_resolve_unbound_raises():
    ast = parse("x")
    with pytest.raises(IllScopedVar, match="Var\\('x'\\)"):
        resolve_binders(ast, on_var=lambda v, r: None)


def test_resolve_too_many_binders_raises():
    src = (r"\a:Int. \b:Int. \c:Int. \d:Int. \e:Int. \f:Int. \g:Int. "
           r"\h:Int. a")
    ast = parse(src)
    # 8 nested lambdas, but BID_CUTOFF - 1 = 7 allowed.
    with pytest.raises(TooManyBinders):
        resolve_binders(ast, on_var=lambda v, r: None)
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_resolve.py -v`
Expected: ImportError on `logic._resolve`.

- [ ] **Step 3: Implement resolver**

Create `src/qft_pcn/logic/_resolve.py`:

```python
"""Lexical scope resolution: every Var is paired with its binder.

Walks the AST tracking a stack of in-scope binders. For each Var encountered,
computes:
  - the binding Lam node (via reference)
  - the lexical depth of the binder from the use site (0 = innermost)

Raises IllScopedVar for unbound references and TooManyBinders if scope
nesting exceeds MAX_BINDER_DEPTH.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .ast import Node, Var, Lam, App, IntLit, BoolLit, If, Bin
from .encoding import (IllScopedVar, TooManyBinders, UnsupportedNode,
                       MAX_BINDER_DEPTH)


@dataclass
class ResolvedRef:
    """Annotation attached to a Var by the resolver."""
    binder: Lam                       # the binding Lam node
    depth_from_innermost: int         # 0 = use is directly inside its binder


def resolve_binders(
    root: Node,
    on_var: Callable[[Var, ResolvedRef], None],
) -> None:
    """Walk root pre-order. Call on_var(var, ref) for each Var encountered.

    The walk also maintains a stack of in-scope binders; visiting a Lam
    pushes it before descending into the body and pops on the way out.
    """
    stack: list[Lam] = []

    def _walk(node: Node) -> None:
        if isinstance(node, Var):
            # Find the innermost binder with matching name.
            for offset, lam in enumerate(reversed(stack)):
                if lam.param == node.name:
                    on_var(node, ResolvedRef(binder=lam,
                                             depth_from_innermost=offset))
                    return
            raise IllScopedVar(name=node.name)
        if isinstance(node, Lam):
            if len(stack) >= MAX_BINDER_DEPTH:
                raise TooManyBinders(depth=len(stack) + 1,
                                     cutoff=MAX_BINDER_DEPTH + 1)
            stack.append(node)
            _walk(node.body)
            stack.pop()
            return
        if isinstance(node, App):
            _walk(node.fn); _walk(node.arg); return
        if isinstance(node, If):
            _walk(node.cond); _walk(node.then_b); _walk(node.else_b); return
        if isinstance(node, Bin):
            _walk(node.lhs); _walk(node.rhs); return
        if isinstance(node, (IntLit, BoolLit)):
            return
        raise UnsupportedNode(node_type=type(node).__name__)

    _walk(root)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_resolve.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/_resolve.py src/qft_pcn/tests/test_logic_resolve.py
git commit -m "$(cat <<'EOF'
feat(logic/_resolve): lexical scope resolution for Var nodes

Walks the AST pre-order, maintaining an explicit binder stack; resolves
each Var to its innermost matching binder and reports the depth-from-
innermost. Raises IllScopedVar for unbound names and TooManyBinders for
scope nesting > 7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Pre-order serialization to NodeOccupancy list

**Files:**
- Create: `src/qft_pcn/logic/_serialize.py`.
- Create: `src/qft_pcn/tests/test_logic_serialize.py`.

Walk the AST in canonical pre-order; produce a list of `NodeOccupancy` items of length N, padded with PAD. This task does **not** yet write tensor amplitudes — it produces the per-site descriptor list that subsequent tasks turn into MPS tensors.

- [ ] **Step 1: Write the failing tests**

Create `src/qft_pcn/tests/test_logic_serialize.py`:

```python
"""Tests for logic/_serialize.py."""

from __future__ import annotations

import pytest

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic._serialize import (
    NodeOccupancy, serialize_preorder, BinderRef, VarRef, count_nodes,
)
from src.qft_pcn.logic.encoding import (
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN, EncodingTooLarge,
)


def test_serialize_var():
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    assert len(sites) == 4
    assert sites[0].kind == KIND_LAM
    assert isinstance(sites[0].binder_ref, BinderRef)
    assert sites[1].kind == KIND_VAR
    assert isinstance(sites[1].var_ref, VarRef)
    # Var refers to the Lam at site 0.
    assert sites[1].var_ref.binder_site == 0
    # Tail is PAD.
    assert sites[2].kind == KIND_PAD
    assert sites[3].kind == KIND_PAD


def test_serialize_application():
    ast = parse("f x")
    # Var x is unbound here — to test serialization of pure structure we wrap:
    ast = parse(r"\f:Int->Int. \x:Int. f x")
    sites = serialize_preorder(ast, N=8)
    # \f. \x. App(Var f, Var x)
    assert sites[0].kind == KIND_LAM      # outer
    assert sites[1].kind == KIND_LAM      # inner
    assert sites[2].kind == KIND_APP
    assert sites[3].kind == KIND_VAR      # f
    assert sites[4].kind == KIND_VAR      # x
    assert sites[5].kind == KIND_PAD


def test_serialize_p5_layout():
    ast = parse(r"(\x:Int. (\y:Int. x + y)(3))(4)")
    sites = serialize_preorder(ast, N=32)
    # APP, LAM x, APP, LAM y, BIN +, VAR x, VAR y, INT 3, INT 4
    expected_kinds = [
        KIND_APP, KIND_LAM, KIND_APP, KIND_LAM, KIND_BIN,
        KIND_VAR, KIND_VAR, KIND_INT, KIND_INT,
    ]
    for i, k in enumerate(expected_kinds):
        assert sites[i].kind == k, f"site {i}: expected {k}, got {sites[i].kind}"
    # Tail must be PAD.
    for i in range(len(expected_kinds), 32):
        assert sites[i].kind == KIND_PAD


def test_serialize_too_large_raises():
    ast = parse(r"\x:Int. x + x + x + x + x")
    # Bin(+, Bin(+, ..., x), x) — at least 11 nodes (Lam + Bin*4 + Var*5).
    with pytest.raises(EncodingTooLarge):
        serialize_preorder(ast, N=5)


def test_count_nodes():
    assert count_nodes(parse(r"\x:Int. x")) == 2
    assert count_nodes(parse(r"(\x:Int. x + 1)(2)")) == 6
    # App, Lam, Bin, Var, IntLit, IntLit -> 6


def test_var_ref_includes_depth():
    ast = parse(r"\x:Int. \y:Int. x")
    sites = serialize_preorder(ast, N=8)
    # The Var(x) is at site 2. It refers to the outer Lam at site 0 and
    # the depth-from-innermost at the use site is 1 (one Lam, y, between).
    assert sites[2].kind == KIND_VAR
    assert sites[2].var_ref.binder_site == 0
    assert sites[2].var_ref.depth_from_innermost == 1
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_serialize.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement serializer**

Create `src/qft_pcn/logic/_serialize.py`:

```python
"""Pre-order serialization of AST to a list of NodeOccupancy descriptors.

This produces the per-site descriptor list. It does NOT yet produce MPS
tensors — that's the encoder's job, using these descriptors as input.

Walks the AST in canonical order:
  Lam: visit self, then body
  App: visit self, then fn, then arg
  If: visit self, then cond, then then_b, then else_b
  Bin: visit self, then lhs, then rhs
  Var, IntLit, BoolLit: leaves (visit self only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .ast import Node, Var, Lam, App, IntLit, BoolLit, If, Bin, Ty
from .encoding import (
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN, BID_NONE,
    EncodingTooLarge, UnsupportedNode,
)
from ._resolve import resolve_binders


@dataclass
class BinderRef:
    """Attached to LAM sites; identifies the binder this Lam introduces."""
    lam_node: Lam
    lexical_depth: int   # 0 = outermost binder in the program


@dataclass
class VarRef:
    """Attached to VAR sites; identifies the resolved binder."""
    binder_site: int                 # site index of the binding LAM
    depth_from_innermost: int        # at the use site


@dataclass
class NodeOccupancy:
    """Per-site descriptor produced by the serializer.

    The descriptor records *what* the site holds; the encoder turns these
    into actual MPS tensors. binder_ref / var_ref are non-None only for
    KIND_LAM / KIND_VAR sites respectively.
    """
    kind: int
    ty: Optional[Ty] = None        # the type of the expression at this site
    int_val: Optional[int] = None  # for KIND_INT
    bool_val: Optional[bool] = None  # for KIND_BOOL
    bin_op: Optional[str] = None   # for KIND_BIN
    binder_ref: Optional[BinderRef] = None
    var_ref: Optional[VarRef] = None
    ast_path: tuple[int, ...] = ()  # path from root, for diagnostics


def count_nodes(root: Node) -> int:
    """Count AST nodes (used by callers to pre-check N)."""
    if isinstance(root, (Var, IntLit, BoolLit)):
        return 1
    if isinstance(root, Lam):
        return 1 + count_nodes(root.body)
    if isinstance(root, App):
        return 1 + count_nodes(root.fn) + count_nodes(root.arg)
    if isinstance(root, If):
        return (1 + count_nodes(root.cond) + count_nodes(root.then_b)
                + count_nodes(root.else_b))
    if isinstance(root, Bin):
        return 1 + count_nodes(root.lhs) + count_nodes(root.rhs)
    raise UnsupportedNode(node_type=type(root).__name__)


def serialize_preorder(root: Node, N: int) -> list[NodeOccupancy]:
    """Pre-order walk of root, producing N descriptors (PAD-padded).

    Resolves every Var first (via resolve_binders) so VarRef can carry the
    binder's site index. Raises EncodingTooLarge if the AST has more than N
    non-PAD nodes.
    """
    n_nodes = count_nodes(root)
    if n_nodes > N:
        raise EncodingTooLarge(n_nodes=n_nodes, N=N)

    # First pass: resolve binders to ResolvedRef objects. We need to map
    # each Lam node to its site index, which we only learn during the
    # second-pass walk. So we collect Lam->ResolvedRef during the first
    # pass (by node identity), then look up in the second pass.
    var_refs: dict[int, "ResolvedRef"] = {}   # id(Var node) -> ResolvedRef
    def _capture(var: Var, ref) -> None:
        var_refs[id(var)] = ref
    resolve_binders(root, on_var=_capture)

    # Second pass: pre-order traversal, recording site indices for Lams as
    # we go so VarRef can point back to them.
    sites: list[NodeOccupancy] = []
    lam_to_site: dict[int, int] = {}      # id(Lam) -> site index
    binder_depth_stack: list[int] = []    # tracks lexical depth on the walk

    def _emit(node: Node, ast_path: tuple[int, ...]) -> None:
        idx = len(sites)
        if isinstance(node, Var):
            ref = var_refs[id(node)]
            sites.append(NodeOccupancy(
                kind=KIND_VAR,
                var_ref=VarRef(
                    binder_site=lam_to_site[id(ref.binder)],
                    depth_from_innermost=ref.depth_from_innermost,
                ),
                ast_path=ast_path,
            ))
            return
        if isinstance(node, Lam):
            depth = len(binder_depth_stack)
            lam_to_site[id(node)] = idx
            sites.append(NodeOccupancy(
                kind=KIND_LAM,
                binder_ref=BinderRef(lam_node=node, lexical_depth=depth),
                ast_path=ast_path,
            ))
            binder_depth_stack.append(depth)
            _emit(node.body, ast_path + (0,))
            binder_depth_stack.pop()
            return
        if isinstance(node, App):
            sites.append(NodeOccupancy(kind=KIND_APP, ast_path=ast_path))
            _emit(node.fn, ast_path + (0,))
            _emit(node.arg, ast_path + (1,))
            return
        if isinstance(node, If):
            sites.append(NodeOccupancy(kind=KIND_IF, ast_path=ast_path))
            _emit(node.cond, ast_path + (0,))
            _emit(node.then_b, ast_path + (1,))
            _emit(node.else_b, ast_path + (2,))
            return
        if isinstance(node, Bin):
            sites.append(NodeOccupancy(kind=KIND_BIN, bin_op=node.op,
                                       ast_path=ast_path))
            _emit(node.lhs, ast_path + (0,))
            _emit(node.rhs, ast_path + (1,))
            return
        if isinstance(node, IntLit):
            sites.append(NodeOccupancy(kind=KIND_INT, int_val=node.val,
                                       ast_path=ast_path))
            return
        if isinstance(node, BoolLit):
            sites.append(NodeOccupancy(kind=KIND_BOOL, bool_val=node.val,
                                       ast_path=ast_path))
            return
        raise UnsupportedNode(node_type=type(node).__name__)

    _emit(root, ())

    # Pad to N with PAD descriptors.
    while len(sites) < N:
        sites.append(NodeOccupancy(kind=KIND_PAD))

    return sites
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_serialize.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/_serialize.py src/qft_pcn/tests/test_logic_serialize.py
git commit -m "$(cat <<'EOF'
feat(logic/_serialize): pre-order AST -> NodeOccupancy descriptors

Two-pass walk: resolve_binders captures Var refs by object identity; the
emit pass assigns site indices to Lams so VarRef can point to its binder
by absolute site. PAD descriptors pad the tail. EncodingTooLarge raised
if non-PAD count exceeds N.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Type computation (bottom-up)

**Files:**
- Create: `src/qft_pcn/logic/_types.py`.
- Create: `src/qft_pcn/tests/test_logic_types.py`.

Compute the type of every site's expression. Maps to one of `TYPE_NONE..TYPE_ARR_BB`, or `TYPE_ARR_NESTED` with full Ty stored in the meta side table.

- [ ] **Step 1: Write the failing tests**

Create `src/qft_pcn/tests/test_logic_types.py`:

```python
from __future__ import annotations

from src.qft_pcn.logic.ast import parse, TInt, TBool, TArrow
from src.qft_pcn.logic._serialize import serialize_preorder
from src.qft_pcn.logic._types import compute_site_types, ty_to_tag
from src.qft_pcn.logic.encoding import (
    TYPE_INT, TYPE_BOOL, TYPE_ARR_II, TYPE_ARR_BB, TYPE_ARR_NESTED, TYPE_NONE,
    KIND_VAR,
)


def test_ty_to_tag_simple():
    assert ty_to_tag(TInt())[0] == TYPE_INT
    assert ty_to_tag(TBool())[0] == TYPE_BOOL
    assert ty_to_tag(TArrow(src=TInt(), dst=TInt()))[0] == TYPE_ARR_II
    assert ty_to_tag(TArrow(src=TBool(), dst=TBool()))[0] == TYPE_ARR_BB


def test_ty_to_tag_nested():
    # (Int -> Int) -> Int
    t = TArrow(src=TArrow(src=TInt(), dst=TInt()), dst=TInt())
    tag, nested = ty_to_tag(t)
    assert tag == TYPE_ARR_NESTED
    assert nested == t


def test_compute_p1_types():
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    # site 0: Lam : Int -> Int
    assert types[0] == TYPE_ARR_II
    # site 1: Var(x) : Int
    assert types[1] == TYPE_INT
    # PAD sites have TYPE_NONE
    assert types[2] == TYPE_NONE
    assert types[3] == TYPE_NONE


def test_compute_p2_types():
    ast = parse(r"(\x:Int. x + 1)(2)")
    sites = serialize_preorder(ast, N=8)
    types = compute_site_types(ast, sites)
    # 0: App : Int
    assert types[0] == TYPE_INT
    # 1: Lam : Int -> Int
    assert types[1] == TYPE_ARR_II
    # 2: Bin + : Int
    assert types[2] == TYPE_INT
    # 3: Var(x) : Int
    assert types[3] == TYPE_INT
    # 4: IntLit 1 : Int
    assert types[4] == TYPE_INT
    # 5: IntLit 2 : Int
    assert types[5] == TYPE_INT


def test_compute_compare_op_returns_bool():
    ast = parse(r"\x:Int. x < 5")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    # 0: Lam Int -> Bool
    from src.qft_pcn.logic.encoding import TYPE_ARR_IB
    assert types[0] == TYPE_ARR_IB
    # 1: Bin < : Bool
    assert types[1] == TYPE_BOOL
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_types.py -v`
Expected: ImportError on `logic._types`.

- [ ] **Step 3: Implement**

Create `src/qft_pcn/logic/_types.py`:

```python
"""Bottom-up type computation for each site of the serialized AST.

Maps a Ty to one of the eight flat tags in encoding.py, with TYPE_ARR_NESTED
as the overflow slot (the full Ty is then stored in EncodingMeta's
nested_type_index side table).

This module does NOT do type-checking — sub-project B owns that. Here we
trust the input AST is well-typed and compute the type of each subexpression.
"""

from __future__ import annotations

from typing import Optional

from .ast import (Node, Var, Lam, App, IntLit, BoolLit, If, Bin,
                  Ty, TInt, TBool, TArrow)
from ._serialize import NodeOccupancy
from .encoding import (
    KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL, KIND_IF, KIND_BIN,
    KIND_PAD,
    TYPE_NONE, TYPE_INT, TYPE_BOOL,
    TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI, TYPE_ARR_BB, TYPE_ARR_NESTED,
)


_FLAT_ARROW_TAGS = {
    (TYPE_INT, TYPE_INT): TYPE_ARR_II,
    (TYPE_INT, TYPE_BOOL): TYPE_ARR_IB,
    (TYPE_BOOL, TYPE_INT): TYPE_ARR_BI,
    (TYPE_BOOL, TYPE_BOOL): TYPE_ARR_BB,
}


def ty_to_tag(ty: Ty) -> tuple[int, Optional[Ty]]:
    """Map a Ty to a flat type tag.

    Returns (tag, nested_full_ty). nested_full_ty is non-None only when the
    tag is TYPE_ARR_NESTED — in which case the caller is responsible for
    storing the full Ty in the meta side table indexed by site.
    """
    if isinstance(ty, TInt):
        return (TYPE_INT, None)
    if isinstance(ty, TBool):
        return (TYPE_BOOL, None)
    if isinstance(ty, TArrow):
        src_tag, _ = ty_to_tag(ty.src)
        dst_tag, _ = ty_to_tag(ty.dst)
        if src_tag in (TYPE_INT, TYPE_BOOL) and dst_tag in (TYPE_INT, TYPE_BOOL):
            return (_FLAT_ARROW_TAGS[(src_tag, dst_tag)], None)
        return (TYPE_ARR_NESTED, ty)
    raise TypeError(f"unknown Ty: {ty!r}")


def _compute_ast_type(node: Node, env: list[tuple[str, Ty]]) -> Ty:
    if isinstance(node, IntLit):
        return TInt()
    if isinstance(node, BoolLit):
        return TBool()
    if isinstance(node, Var):
        for name, t in reversed(env):
            if name == node.name:
                return t
        raise KeyError(f"unbound {node.name}")
    if isinstance(node, Lam):
        body_ty = _compute_ast_type(node.body,
                                    env + [(node.param, node.param_ty)])
        return TArrow(src=node.param_ty, dst=body_ty)
    if isinstance(node, App):
        fn_ty = _compute_ast_type(node.fn, env)
        if isinstance(fn_ty, TArrow):
            return fn_ty.dst
        # Ill-typed input: fall back to TInt so encoder doesn't crash.
        # Sub-project B will reject this at the Hamiltonian level.
        return TInt()
    if isinstance(node, If):
        return _compute_ast_type(node.then_b, env)
    if isinstance(node, Bin):
        if node.op in ("+", "-", "*"):
            return TInt()
        if node.op in ("<", "=="):
            return TBool()
        raise ValueError(f"unknown op {node.op}")
    raise TypeError(f"unknown Node: {type(node).__name__}")


def compute_site_types(root: Node,
                       sites: list[NodeOccupancy]
                       ) -> list[int]:
    """For each site, compute the flat type tag.

    Returns a list of length N. The full Ty for any TYPE_ARR_NESTED entry
    is stored on the corresponding NodeOccupancy.ty attribute so the
    encoder can populate meta.nested_type_index.
    """
    # First, walk the AST once and gather a (ast_path -> Ty) map.
    path_to_ty: dict[tuple[int, ...], Ty] = {}

    def _walk(node: Node, env: list[tuple[str, Ty]], path: tuple[int, ...]):
        ty = _compute_ast_type(node, env)
        path_to_ty[path] = ty
        if isinstance(node, Lam):
            _walk(node.body, env + [(node.param, node.param_ty)],
                  path + (0,))
        elif isinstance(node, App):
            _walk(node.fn, env, path + (0,))
            _walk(node.arg, env, path + (1,))
        elif isinstance(node, If):
            _walk(node.cond, env, path + (0,))
            _walk(node.then_b, env, path + (1,))
            _walk(node.else_b, env, path + (2,))
        elif isinstance(node, Bin):
            _walk(node.lhs, env, path + (0,))
            _walk(node.rhs, env, path + (1,))

    _walk(root, [], ())

    # Now produce flat tags per site, in serialization order.
    tags: list[int] = []
    for occ in sites:
        if occ.kind == KIND_PAD:
            tags.append(TYPE_NONE)
            occ.ty = None
            continue
        ty = path_to_ty.get(occ.ast_path)
        if ty is None:
            tags.append(TYPE_NONE)
            occ.ty = None
            continue
        tag, nested = ty_to_tag(ty)
        tags.append(tag)
        # Keep the full Ty on the occupancy when nested or for the encoder
        # to use as needed.
        occ.ty = nested if nested is not None else ty
    return tags
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_types.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/_types.py src/qft_pcn/tests/test_logic_types.py
git commit -m "$(cat <<'EOF'
feat(logic/_types): site-level type computation and flat-tag mapping

Bottom-up computation of each subexpression's Ty, then mapped to one of the
eight flat type tags (TYPE_INT, TYPE_BOOL, TYPE_ARR_II..TYPE_ARR_BB) or the
TYPE_ARR_NESTED overflow slot for higher-order arrow types.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Live-binder bookkeeping per bond

**Files:**
- Create: `src/qft_pcn/logic/_channels.py`.
- Create: `src/qft_pcn/tests/test_logic_channels.py`.

For each bond (between site `i` and `i+1`), compute the ordered list of `BinderHandle` objects that are *live* across it — declared at some `lam_site <= i` with at least one var use at some `var_site >= i+1`. This is the channel structure of spec §5.4.

- [ ] **Step 1: Write the failing tests**

Create `src/qft_pcn/tests/test_logic_channels.py`:

```python
from __future__ import annotations

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic._serialize import serialize_preorder
from src.qft_pcn.logic._channels import compute_live_binders
from src.qft_pcn.logic.encoding import BinderHandle


def test_live_binders_p1():
    """\\x:Int. x — bond 0 (between Lam@0 and Var@1) has Lam_x live."""
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    live = compute_live_binders(sites)
    # N-1 bonds for length-N MPS.
    assert len(live) == 3
    # Bond 0 (between sites 0 and 1): Lam_x is live.
    assert len(live[0]) == 1
    assert live[0][0].lam_site == 0
    # Bond 1 (between sites 1 and 2): after Var consumed Lam_x, nothing live.
    assert len(live[1]) == 0
    assert len(live[2]) == 0


def test_live_binders_p5():
    """(\\x. (\\y. x + y)(3))(4) — nested binders, compaction after each use."""
    ast = parse(r"(\x:Int. (\y:Int. x + y)(3))(4)")
    sites = serialize_preorder(ast, N=32)
    live = compute_live_binders(sites)
    # Layout: APP@0, LAM_x@1, APP@2, LAM_y@3, BIN+@4, VAR_x@5, VAR_y@6,
    #         INT@7, INT@8, PAD@9..31
    # Bond 0 (between APP@0 and LAM_x@1): no binders live yet.
    assert len(live[0]) == 0
    # Bond 1 (between LAM_x@1 and APP@2): Lam_x live.
    assert len(live[1]) == 1
    assert live[1][0].lam_site == 1
    # Bond 2 (between APP@2 and LAM_y@3): Lam_x still live.
    assert len(live[2]) == 1
    # Bond 3 (between LAM_y@3 and BIN@4): both Lam_x and Lam_y live.
    assert len(live[3]) == 2
    assert {b.lam_site for b in live[3]} == {1, 3}
    # Bond 4 (between BIN@4 and VAR_x@5): both still live (Var_x is on the
    # right side of bond 4, but Var_y is also on the right of bond 4 — both
    # are still needed across this bond).
    assert len(live[4]) == 2
    # Bond 5 (between VAR_x@5 and VAR_y@6): after the use at site 5 consumed
    # the LAST occurrence of Lam_x, x leaves; only Lam_y remains.
    assert len(live[5]) == 1
    assert live[5][0].lam_site == 3
    # Bond 6 (between VAR_y@6 and INT@7): Lam_y also consumed; nothing.
    assert len(live[6]) == 0
    # Tail bonds all empty.
    for i in range(7, 31):
        assert len(live[i]) == 0


def test_live_binders_ordering_is_declaration_order():
    """Channel order follows declaration order."""
    ast = parse(r"\x:Int. \y:Int. \z:Int. x + y + z")
    sites = serialize_preorder(ast, N=12)
    live = compute_live_binders(sites)
    # At the bond between LAM_z and the first BIN site, all three are live.
    # Lam_x@0, Lam_y@1, Lam_z@2.
    # Layout: LAM_x@0, LAM_y@1, LAM_z@2, BIN+@3, BIN+@4, VAR_x@5, VAR_y@6,
    #         VAR_z@7, ...
    # Bond 2 between LAM_z@2 and BIN@3:
    handles = live[2]
    assert len(handles) == 3
    # Ordered by lam_site (declaration order).
    assert [h.lam_site for h in handles] == [0, 1, 2]
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_channels.py -v`
Expected: ImportError on `_channels`.

- [ ] **Step 3: Implement**

Create `src/qft_pcn/logic/_channels.py`:

```python
"""Live-binder bookkeeping per bond.

A binder is *live* at bond i (between sites i and i+1) iff:
  - its lam_site <= i
  - it has at least one var use at site >= i+1

The encoder uses this to size the bid-register bond and to assign channels
in declaration order.
"""

from __future__ import annotations

from .encoding import BinderHandle, KIND_LAM, KIND_VAR
from ._serialize import NodeOccupancy


def compute_live_binders(
    sites: list[NodeOccupancy],
) -> list[list[BinderHandle]]:
    """Return, for each of the N-1 bonds, the ordered list of live binders.

    Output[i] = ordered (by declaration / lam_site) list of BinderHandles
    that are live at bond i (between sites i and i+1).
    """
    N = len(sites)
    # First pass: collect Lam metadata and the rightmost var use per binder.
    binders: list[BinderHandle] = []   # in declaration order
    binder_lam_site: dict[int, int] = {}  # lam_site -> position in binders
    last_use_site: dict[int, int] = {}   # lam_site -> last var-use site

    for k, occ in enumerate(sites):
        if occ.kind == KIND_LAM:
            depth = occ.binder_ref.lexical_depth if occ.binder_ref else 0
            handle = BinderHandle(lam_site=k, depth_at_lam=depth)
            binder_lam_site[k] = len(binders)
            binders.append(handle)
            last_use_site[k] = k       # if never used, last_use = lam_site
        elif occ.kind == KIND_VAR and occ.var_ref is not None:
            ls = occ.var_ref.binder_site
            last_use_site[ls] = max(last_use_site.get(ls, ls), k)

    # Second pass: compute liveness per bond.
    # A binder is live across bond i iff lam_site <= i AND last_use_site > i.
    live: list[list[BinderHandle]] = []
    for i in range(N - 1):
        crossing: list[BinderHandle] = []
        for h in binders:
            if h.lam_site <= i and last_use_site[h.lam_site] > i:
                crossing.append(h)
        live.append(crossing)
    return live
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_channels.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/_channels.py src/qft_pcn/tests/test_logic_channels.py
git commit -m "$(cat <<'EOF'
feat(logic/_channels): per-bond live-binder bookkeeping

Computes, for each MPS bond, the ordered list of BinderHandles whose
binder was declared on or before the bond and whose last var use is
after the bond. Channels are ordered by declaration (lam_site).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Site tensor construction — the principled bid bond

**Files:**
- Create: `src/qft_pcn/logic/_tensors.py`.
- Create: `src/qft_pcn/tests/test_logic_tensors.py`.

This is the **architectural soul** of the project. Build per-site MPS tensors of shape `(chi_left, 8192, chi_right)`. The bid register's bond carries `|L_i| + 1` channels (no_info + one per live binder).

**Read spec §5.4 and §5.6 before writing this.** If a future subagent is tempted to "just put the binder id at the var site as a local state and skip the channel structure" — that is the §1.1 shortcut. The bond-dimension test (Part 3, Task 25) will fail.

- [ ] **Step 1: Write the failing tests**

Create `src/qft_pcn/tests/test_logic_tensors.py`:

```python
from __future__ import annotations

import numpy as np

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic._serialize import serialize_preorder
from src.qft_pcn.logic._types import compute_site_types
from src.qft_pcn.logic._channels import compute_live_binders
from src.qft_pcn.logic._tensors import build_site_tensors
from src.qft_pcn.logic.encoding import (
    D_LOCAL, KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF,
    KIND_LAM, KIND_VAR, BID_0, BID_NONE,
)


def _basis_index(kind_idx: int, type_idx: int, bid_idx: int,
                 value_idx: int) -> int:
    """Linear index of (kind, type, bid, value) in the local basis.

    Per embed_op convention in fock.py: leftmost species changes slowest.
    """
    return (((kind_idx * TYPE_CUTOFF + type_idx) * BID_CUTOFF + bid_idx)
            * VALUE_CUTOFF + value_idx)


def test_site_tensor_shapes():
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    # All tensors are rank-3.
    assert len(tensors) == 4
    for t in tensors:
        assert t.ndim == 3
        assert t.shape[1] == D_LOCAL


def test_site_tensor_boundary_bonds_are_dim_1():
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    # Left boundary of site 0 and right boundary of site N-1 are dim 1.
    assert tensors[0].shape[0] == 1
    assert tensors[-1].shape[2] == 1


def test_p1_lam_site_has_growing_right_bond():
    """For \\x.x, bond 0 between LAM and VAR carries Lam_x as a channel.

    Bond dim = len(live[0]) + 1 = 1 + 1 = 2.
    """
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    # LAM site: left bond = 1 (boundary), right bond = 2.
    assert tensors[0].shape == (1, D_LOCAL, 2)
    # VAR site: left bond = 2, right bond = 1 (no live binders after).
    assert tensors[1].shape == (2, D_LOCAL, 1)
    # PAD sites: dim 1 on both sides.
    assert tensors[2].shape == (1, D_LOCAL, 1)
    assert tensors[3].shape == (1, D_LOCAL, 1)


def test_p1_lam_site_nonzero_amplitude_at_correct_basis_state():
    """The LAM site tensor at \\x.x should have amplitude 1 at the basis
    state (KIND_LAM, TYPE_ARR_II, BID_0, VALUE_NONE) on the "no-info-in,
    channel-x-out" bond direction.
    """
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    from src.qft_pcn.logic.encoding import KIND_LAM, TYPE_ARR_II, BID_0, VALUE_NONE
    idx = _basis_index(KIND_LAM, TYPE_ARR_II, BID_0, VALUE_NONE)
    T = tensors[0]  # shape (1, 8192, 2)
    # The bond goes from "no_info" channel (index 0) in to "Lam_x" channel
    # (index 1) out.
    assert abs(T[0, idx, 1] - 1.0) < 1e-12, (
        "LAM site tensor should activate the Lam_x channel on the right bond"
    )
    # All other entries at this local basis state should be zero.
    assert abs(T[0, idx, 0]) < 1e-12  # no-info-out: zero
    # Other local basis states should be zero too.
    other_idx = _basis_index(0, 0, 0, 0)  # PAD/none/none/none
    assert abs(T[0, other_idx, 0]) < 1e-12
    assert abs(T[0, other_idx, 1]) < 1e-12


def test_p1_var_site_reads_channel():
    """The VAR site tensor at \\x.x should read the Lam_x channel from its
    left bond and emit no_info on its right bond, with local bid = BID_0
    (Var(x)'s binder is innermost from its perspective)."""
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    from src.qft_pcn.logic.encoding import KIND_VAR, TYPE_INT, BID_0, VALUE_NONE
    idx = _basis_index(KIND_VAR, TYPE_INT, BID_0, VALUE_NONE)
    T = tensors[1]   # shape (2, 8192, 1)
    # Reading channel 1 (Lam_x) -> no_info out (index 0).
    assert abs(T[1, idx, 0] - 1.0) < 1e-12


def test_pad_site_is_vacuum_amplitude():
    """PAD sites have amplitude 1 at (KIND_PAD, TYPE_NONE, BID_NONE,
    VALUE_NONE) with no_info channels through, and 0 elsewhere."""
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    pad_idx = _basis_index(0, 0, 0, 0)
    for k in (2, 3):
        T = tensors[k]
        assert T.shape == (1, D_LOCAL, 1)
        assert abs(T[0, pad_idx, 0] - 1.0) < 1e-12
        # Some other index must be zero.
        from src.qft_pcn.logic.encoding import KIND_VAR
        other = _basis_index(KIND_VAR, 0, 0, 0)
        assert abs(T[0, other, 0]) < 1e-12


def test_bond_dim_at_least_n_live_plus_one():
    """Spec §7.4 test (structural marker): for every bond, the total bond
    dim must be at least |L_i| + 1. This proves the principled channel
    construction was used; if a subagent collapsed bid to a classical
    field, the bond would be 1 even with binders crossing.
    """
    ast = parse(r"(\x:Int. (\y:Int. x + y)(3))(4)")
    sites = serialize_preorder(ast, N=32)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    for i in range(len(live)):
        # Right bond of site i has dimension = tensors[i].shape[2].
        n_live = len(live[i])
        assert tensors[i].shape[2] >= n_live + 1, (
            f"bond {i} right-bond dim {tensors[i].shape[2]} < "
            f"|L_i|+1 = {n_live + 1}; spec §1.1 violation"
        )
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_tensors.py -v`
Expected: ImportError on `_tensors`.

- [ ] **Step 3: Implement the tensor builder**

Create `src/qft_pcn/logic/_tensors.py`:

```python
"""Site-tensor construction for the AST <-> MPS encoder.

The per-site MPS tensor has shape (chi_left, d_local, chi_right) with
d_local = KIND_CUTOFF * TYPE_CUTOFF * BID_CUTOFF * VALUE_CUTOFF = 8192.

This is the architectural-soul module: the bid register's bond carries
the live-binder channels. Bond dim on bid = |L_i| + 1 (one channel per
live binder + a "no_info" channel). The kind/type/value registers are
product on every bond (bond dim 1 on those registers).

The full bond dim equals the product over registers of register-bond-dims.
Since kind/type/value contribute 1 each, the full bond dim equals the
bid bond dim.

Basis ordering inside the local 8192-dim register follows embed_op in
fock.py: leftmost species (kind) changes slowest. The flat index of
(s_kind, s_type, s_bid, s_value) is:

    s_kind * (T * B * V) + s_type * (B * V) + s_bid * V + s_value

where T = TYPE_CUTOFF, B = BID_CUTOFF, V = VALUE_CUTOFF.

DO NOT short-circuit the channel mechanism with a classical bid lookup at
Var sites — that is the spec §1.1 shortcut, and the bond-dim test in
test_logic_tensors.py will fail with a message naming the violation.
"""

from __future__ import annotations

import numpy as np

from .encoding import (
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN, KIND_CUTOFF,
    TYPE_NONE, TYPE_INT, TYPE_BOOL, TYPE_CUTOFF,
    BID_NONE, BID_0, BID_CUTOFF,
    VALUE_NONE, VALUE_FALSE, VALUE_TRUE,
    BIN_VALUE_FROM_OP, INT_LIT_OFFSET, INT_LIT_MIN, INT_LIT_MAX,
    VALUE_CUTOFF, D_LOCAL,
    BinderHandle, IntLiteralOutOfRange,
)
from ._serialize import NodeOccupancy


def _basis_index(kind_idx: int, type_idx: int, bid_idx: int,
                 value_idx: int) -> int:
    """Linear index of (kind, type, bid, value) in the local 8192-dim basis."""
    return (((kind_idx * TYPE_CUTOFF + type_idx) * BID_CUTOFF + bid_idx)
            * VALUE_CUTOFF + value_idx)


def _local_kind_type_value(occ: NodeOccupancy, type_tag: int
                           ) -> tuple[int, int, int]:
    """Return (kind_idx, type_idx, value_idx) for this site.

    The bid_idx is computed separately because it depends on the bond
    structure (it's read from / written to a channel).
    """
    kind = occ.kind
    if kind == KIND_PAD:
        return (KIND_PAD, TYPE_NONE, VALUE_NONE)
    if kind == KIND_INT:
        if occ.int_val is None:
            raise ValueError("KIND_INT site missing int_val")
        if not INT_LIT_MIN <= occ.int_val <= INT_LIT_MAX:
            raise IntLiteralOutOfRange(n=occ.int_val)
        return (KIND_INT, type_tag, occ.int_val + INT_LIT_OFFSET)
    if kind == KIND_BOOL:
        return (KIND_BOOL, type_tag,
                VALUE_TRUE if occ.bool_val else VALUE_FALSE)
    if kind == KIND_BIN:
        return (KIND_BIN, type_tag, BIN_VALUE_FROM_OP[occ.bin_op])
    # VAR / LAM / APP / IF: value is NONE.
    return (kind, type_tag, VALUE_NONE)


def _local_bid_for_kind(kind: int, occ: NodeOccupancy) -> int:
    """Return the local bid register value for this site.

    LAM: BID_0 (the binder is innermost from its own perspective).
    VAR: BID_k where k = depth_from_innermost at the use site.
    Everything else: BID_NONE.
    """
    if kind == KIND_LAM:
        return BID_0   # "I am introducing a new binder; it is my innermost"
    if kind == KIND_VAR:
        depth = occ.var_ref.depth_from_innermost
        # BID_0 .. BID_6 occupy indices 1..7. BID_k = k + 1.
        return BID_0 + depth
    return BID_NONE


def _bid_bond_tensor_at_site(
    site_idx: int,
    occ: NodeOccupancy,
    local_bid_value: int,
    left_live: list[BinderHandle],
    right_live: list[BinderHandle],
) -> np.ndarray:
    """Construct the bid-register-only sub-tensor for one site.

    Shape: (|left_live| + 1, BID_CUTOFF, |right_live| + 1).

    Channel ordering at each bond:
      - index 0: "no_info" channel
      - index 1: first live binder
      - index 2: second live binder
      - ...

    Tensor entries (all complex, real-valued):

      For a PAD or non-binder, non-var site (just passing channels through):
        T[ch, BID_NONE, ch'] = 1  iff ch and ch' refer to the same binder
                                  or both are no_info
        T[*, BID_NONE, *] = 0    elsewhere
        T[*, BID_!= NONE, *] = 0  (site has no local bid value)

      For a LAM site introducing binder b which is at right channel c_b:
        - The new binder enters via the "no_info" left channel and goes
          out on its own channel.
        - Pre-existing binders pass through (channel preserved).
        T[ch, BID_0, ch']
            = 1 if ch == no_info and ch' == c_b (new binder created)
            = 1 if ch != no_info and ch' is the same binder (passthrough)
        all other T entries = 0

      For a VAR site referring to binder b at left channel c_b:
        - If this is the binder's LAST use, the channel is removed from
          the right bond (right_live drops it).
        - If it's not the last use, the channel passes through.
        T[c_b, BID_local, ch']
            = 1 if ch' is the same binder b passed through, OR
            = 1 if ch' = no_info and b is no longer in right_live
        Other channels pass through normally:
        T[ch, BID_NONE, ch']
            = 1 if ch and ch' refer to the same binder, where ch != c_b
            = 1 if ch == no_info and ch' == no_info
    """
    L_in = len(left_live)
    L_out = len(right_live)
    T = np.zeros((L_in + 1, BID_CUTOFF, L_out + 1), dtype=complex)

    # Pre-compute the mapping from binder handle to channel index on each
    # side. Channel 0 is no_info, 1..L is binders in order.
    left_ch = {bh: i + 1 for i, bh in enumerate(left_live)}
    right_ch = {bh: i + 1 for i, bh in enumerate(right_live)}
    NO_INFO_IN = 0
    NO_INFO_OUT = 0

    # All sites have "no_info passthrough" on the no_info channel by default.
    if occ.kind != KIND_VAR:
        T[NO_INFO_IN, BID_NONE, NO_INFO_OUT] = 1.0

    # Common channel passthrough for binders that survive across this bond.
    # For LAM sites, the new binder goes from no_info_in to its channel_out
    # under BID_0 — but other binders still pass through under BID_NONE.
    # For VAR sites, the referenced binder's channel may be consumed.
    consumed_binder: BinderHandle | None = None
    if occ.kind == KIND_VAR:
        # Identify the binder being referenced. We can recover it from the
        # left_live channels by matching against the var_ref's binder_site.
        target_lam_site = occ.var_ref.binder_site
        for bh in left_live:
            if bh.lam_site == target_lam_site:
                ref_handle = bh
                break
        else:
            raise RuntimeError(
                f"VAR site {site_idx} references binder at lam_site="
                f"{target_lam_site} but it is not in left_live; "
                f"this is an encoder bug — channel bookkeeping is wrong."
            )
        # Is this the binder's last use? It's consumed iff the binder is
        # not in right_live.
        if ref_handle not in right_ch:
            consumed_binder = ref_handle
        c_in = left_ch[ref_handle]
        # Read the channel: BID_local is the local bid value at this site.
        if consumed_binder is not None:
            # The channel is dropped at this bond — emit to no_info_out.
            T[c_in, local_bid_value, NO_INFO_OUT] = 1.0
        else:
            # Pass the channel through.
            c_out = right_ch[ref_handle]
            T[c_in, local_bid_value, c_out] = 1.0

        # Other channels passthrough as identity under BID_NONE
        # (these binders are not being used at this site).
        for bh in left_live:
            if bh == ref_handle:
                continue
            c_in_other = left_ch[bh]
            # If still live on right, passthrough.
            if bh in right_ch:
                c_out_other = right_ch[bh]
                T[c_in_other, BID_NONE, c_out_other] = 1.0
            # Otherwise the binder is being dropped at this bond despite
            # not being referenced here — that shouldn't happen if our
            # liveness analysis is correct (last_use_site logic).
            else:
                # Defensive: drop to no_info_out under BID_NONE.
                T[c_in_other, BID_NONE, NO_INFO_OUT] = 1.0
        # The no_info channel doesn't passthrough at a VAR site to no_info
        # (we already handled that above via the `occ.kind != KIND_VAR`
        # guard skipping it). We need to allow no_info -> no_info under
        # BID_NONE so that the encoder still treats no_info as a valid
        # boundary state. But at VAR sites, no_info-in shouldn't carry
        # any amplitude in the constructed product state. Set to 0.
        # (i.e., leave it as initialized.)
        return T

    if occ.kind == KIND_LAM:
        # The new binder is the rightmost entry of right_live (declaration
        # order). Find its channel.
        new_binder = None
        for bh in right_live:
            if bh.lam_site == site_idx:
                new_binder = bh
                break
        if new_binder is None:
            raise RuntimeError(
                f"LAM site {site_idx}: no matching binder in right_live; "
                f"this is an encoder bug — channels misaligned"
            )
        c_new_out = right_ch[new_binder]
        # The new binder enters from no_info_in with the LAM's local bid (BID_0).
        T[NO_INFO_IN, local_bid_value, c_new_out] = 1.0
        # Existing binders pass through under BID_NONE.
        for bh in left_live:
            c_in = left_ch[bh]
            if bh in right_ch:
                c_out = right_ch[bh]
                T[c_in, BID_NONE, c_out] = 1.0
            else:
                # Shouldn't happen at a LAM site (LAM doesn't drop binders).
                T[c_in, BID_NONE, NO_INFO_OUT] = 1.0
        # Remove the no_info passthrough we added at the top, since at a
        # LAM site the no_info channel is being consumed by the new binder
        # creation — it does NOT also passthrough.
        T[NO_INFO_IN, BID_NONE, NO_INFO_OUT] = 0.0
        return T

    # PAD / APP / IF / INT / BOOL / BIN: pure passthrough.
    # All left channels go to the matching right channel under BID_NONE.
    for bh in left_live:
        c_in = left_ch[bh]
        if bh in right_ch:
            c_out = right_ch[bh]
            T[c_in, BID_NONE, c_out] = 1.0
        else:
            # Binder dropping at a non-VAR site shouldn't happen.
            T[c_in, BID_NONE, NO_INFO_OUT] = 1.0
    return T


def _combine_factored_site(
    kind_idx: int, type_idx: int, value_idx: int,
    bid_tensor: np.ndarray
) -> np.ndarray:
    """Combine the (kind, type, value) deterministic indices and the
    (chi_l, BID, chi_r) bid sub-tensor into the full
    (chi_l, D_LOCAL, chi_r) site tensor.

    The kind/type/value registers are *product*: one basis index has full
    amplitude. So the output tensor is nonzero only on the slice
    flat_basis_index = base + bid_idx * VALUE_CUTOFF for varying bid_idx.

    Specifically, T[ch_in, base + b * V, ch_out] = bid_tensor[ch_in, b, ch_out]
    for each bid index b in [0, BID_CUTOFF).
    """
    chi_l, B, chi_r = bid_tensor.shape
    assert B == BID_CUTOFF
    out = np.zeros((chi_l, D_LOCAL, chi_r), dtype=complex)
    # Compute the flat index for (kind_idx, type_idx, *, value_idx) for each
    # possible bid index.
    base_no_bid = kind_idx * (TYPE_CUTOFF * BID_CUTOFF * VALUE_CUTOFF) \
                  + type_idx * (BID_CUTOFF * VALUE_CUTOFF)
    for b in range(BID_CUTOFF):
        flat = base_no_bid + b * VALUE_CUTOFF + value_idx
        out[:, flat, :] = bid_tensor[:, b, :]
    return out


def build_site_tensors(
    sites: list[NodeOccupancy],
    type_tags: list[int],
    live_binders_per_bond: list[list[BinderHandle]],
) -> list[np.ndarray]:
    """Top-level builder. Produces the list of N rank-3 site tensors.

    Bond i has dimension |live_binders_per_bond[i]| + 1 (one channel per
    live binder + no_info). The full site tensor has shape
    (chi_left, D_LOCAL, chi_right) where chi values come from adjacent
    bond live-binder counts. Boundary bonds (left of site 0, right of
    site N-1) are dimension 1.

    Returns a list of complex numpy arrays.
    """
    N = len(sites)
    # Boundary-aware live lists: prepend [] for left boundary, append []
    # for right boundary so live_at_bond[i] = live across bond between
    # sites i-1 and i.
    bounded_live = [[]] + list(live_binders_per_bond) + [[]]
    # bounded_live has length N + 1. left_live[k] = bounded_live[k];
    # right_live[k] = bounded_live[k + 1].

    tensors: list[np.ndarray] = []
    for k, occ in enumerate(sites):
        left_live = bounded_live[k]
        right_live = bounded_live[k + 1]
        type_tag = type_tags[k]
        kind_idx, type_idx, value_idx = _local_kind_type_value(occ, type_tag)
        local_bid = _local_bid_for_kind(occ.kind, occ)
        bid_T = _bid_bond_tensor_at_site(
            site_idx=k, occ=occ, local_bid_value=local_bid,
            left_live=left_live, right_live=right_live,
        )
        site_T = _combine_factored_site(
            kind_idx=kind_idx, type_idx=type_idx, value_idx=value_idx,
            bid_tensor=bid_T,
        )
        tensors.append(site_T)
    return tensors
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_tensors.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/_tensors.py src/qft_pcn/tests/test_logic_tensors.py
git commit -m "$(cat <<'EOF'
feat(logic/_tensors): site-tensor construction with live-binder channels

The architectural soul of sub-project A. Per spec §5.4, the bid register's
bond carries a 'no_info' channel plus one channel per live binder. LAM
sites inject a new channel at their own BID_0 local value; VAR sites read
from their binder's channel and may drop it from the right bond if this
is the binder's last use. Other sites are pure passthrough.

Bond dim on the bid register equals |L_i| + 1; kind/type/value are
product (bond dim 1 on those registers).

Tests verify the bond dimension is at least |L_i| + 1 across every bond
of a representative program (P5), which is the structural marker that
the channel mechanism was used — the spec §1.1 shortcut would collapse
the bond to dim 1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Top-level `encode()` and EncodingMeta wiring

**Files:**
- Create: `src/qft_pcn/logic/encoder.py`.
- Create: `src/qft_pcn/tests/test_logic_encoder_smoke.py`.

Wire it all together: resolve → serialize → compute types → compute live binders → build tensors → wrap as MPS → normalize → return with EncodingMeta. The detailed round-trip tests come in Part 3.

- [ ] **Step 1: Write the failing tests (smoke level)**

Create `src/qft_pcn/tests/test_logic_encoder_smoke.py`:

```python
from __future__ import annotations

import math

import numpy as np
import pytest

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.encoder import encode
from src.qft_pcn.logic.encoding import (
    SPECIES, EncodingTooLarge, TooManyBinders, IntLiteralOutOfRange,
    IllScopedVar, BinderHandle,
)
from src.qft_pcn.qft.mps import MPS


def test_encode_p1_returns_mps_and_meta():
    state, meta = encode(parse(r"\x:Int. x"), N=8, chi_max=16)
    assert isinstance(state, MPS)
    assert state.N == 8
    assert meta.N == 8
    assert meta.chi_max == 16
    assert meta.species == list(SPECIES)


def test_encode_produces_unit_norm():
    state, meta = encode(parse(r"\x:Int. x"), N=8)
    assert abs(state.norm_sq() - 1.0) < 1e-10


def test_encode_p5_unit_norm():
    state, meta = encode(parse(r"(\x:Int. (\y:Int. x + y)(3))(4)"),
                          N=32, chi_max=16)
    assert abs(state.norm_sq() - 1.0) < 1e-10


def test_encode_records_live_binders():
    state, meta = encode(parse(r"\x:Int. \y:Int. x"), N=8)
    # Two binders, three bonds with binders live.
    # Layout: LAM_x@0, LAM_y@1, VAR_x@2
    # Bond 0 (between LAM_x and LAM_y): [Lam_x]
    # Bond 1 (between LAM_y and VAR_x): [Lam_x, Lam_y]
    # Bond 2 (between VAR_x and PAD): [] (both gone)
    # Bonds 3..N-2: []
    assert len(meta.live_binders_per_bond) == 7  # N - 1
    assert len(meta.live_binders_per_bond[0]) == 1
    assert meta.live_binders_per_bond[0][0].lam_site == 0
    assert len(meta.live_binders_per_bond[1]) == 2
    assert len(meta.live_binders_per_bond[2]) == 0


def test_encode_too_large_raises():
    with pytest.raises(EncodingTooLarge):
        encode(parse(r"\x:Int. x + x + x"), N=3)


def test_encode_too_many_binders_raises():
    src = (r"\a:Int. \b:Int. \c:Int. \d:Int. \e:Int. \f:Int. \g:Int. "
           r"\h:Int. a")
    with pytest.raises(TooManyBinders):
        encode(parse(src), N=32)


def test_encode_int_literal_out_of_range_raises():
    with pytest.raises(IntLiteralOutOfRange):
        encode(parse(r"\x:Int. x + 100"), N=8)


def test_encode_ill_scoped_var_raises():
    with pytest.raises(IllScopedVar):
        encode(parse("x"), N=4)


def test_encode_pad_sites_have_zero_amplitude_for_non_pad():
    """Quick PAD test: site 3 of \\x.x encoding must have no amplitude for
    KIND_VAR (or anything other than the PAD basis state)."""
    state, meta = encode(parse(r"\x:Int. x"), N=4)
    # Build a projector onto the non-PAD subspace of the kind register, then
    # embed it via the qft/fock embed_op machinery, and check the
    # expectation is ~0 at site 3.
    from src.qft_pcn.qft.fock import embed_op
    from src.qft_pcn.logic.encoding import KIND_PAD, KIND_CUTOFF
    import numpy as np
    p = np.eye(KIND_CUTOFF, dtype=complex)
    p[KIND_PAD, KIND_PAD] = 0.0  # project AWAY from PAD
    p_local = embed_op(p, 0, (8, 8, 8, 16))
    val = state.local_expectation(3, p_local)
    assert abs(val) < 1e-10
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_encoder_smoke.py -v`
Expected: ImportError on `logic.encoder`.

- [ ] **Step 3: Implement encode()**

Create `src/qft_pcn/logic/encoder.py`:

```python
"""Top-level encoder: AST -> (MPS, EncodingMeta).

Pipeline (each step is its own module so it can be tested in isolation):

  1. resolve_binders     (_resolve.py)       — annotate every Var with its binder
  2. serialize_preorder  (_serialize.py)     — produce NodeOccupancy list
  3. compute_site_types  (_types.py)         — bottom-up type computation
  4. compute_live_binders (_channels.py)     — per-bond live binders
  5. build_site_tensors  (_tensors.py)       — per-site rank-3 tensors
  6. wrap into MPS and normalize             — qft/mps.py
"""

from __future__ import annotations

from typing import Optional

from .ast import Node, Ty
from .encoding import (
    SPECIES, EncodingMeta, BinderHandle, KIND_CUTOFF, TYPE_CUTOFF,
    BID_CUTOFF, VALUE_CUTOFF,
)
from ._serialize import serialize_preorder
from ._types import compute_site_types
from ._channels import compute_live_binders
from ._tensors import build_site_tensors
from src.qft_pcn.qft.mps import MPS


def encode(ast: Node, N: int = 32, chi_max: int = 16
           ) -> tuple[MPS, EncodingMeta]:
    """Encode an AST into a unit-norm MPS of length N.

    See docs/superpowers/specs/2026-05-20-ast-mps-encoder-design.md, §5.

    Raises:
        EncodingTooLarge, TooManyBinders, IntLiteralOutOfRange,
        IllScopedVar, UnsupportedNode — see logic/encoding.py.
    """
    # Pipeline.
    sites = serialize_preorder(ast, N=N)
    type_tags = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, type_tags, live)

    # Wrap as MPS and normalize.
    state = MPS(tensors=tensors)
    state.normalize()

    # Build the nested_type_index side table from sites whose tag is
    # TYPE_ARR_NESTED. (NodeOccupancy.ty was set by _types.compute_site_types
    # to the full Ty for nested arrows.)
    from .encoding import TYPE_ARR_NESTED
    nested: dict[int, Ty] = {}
    for k, occ in enumerate(sites):
        if type_tags[k] == TYPE_ARR_NESTED and occ.ty is not None:
            nested[k] = occ.ty

    # Site -> AST path map (for debugging).
    site_to_path: dict[int, tuple[int, ...]] = {}
    for k, occ in enumerate(sites):
        site_to_path[k] = occ.ast_path

    meta = EncodingMeta(
        N=N,
        chi_max=chi_max,
        field_dims={
            "kind": KIND_CUTOFF, "type": TYPE_CUTOFF,
            "bid": BID_CUTOFF, "value": VALUE_CUTOFF,
        },
        species=list(SPECIES),
        nested_type_index=nested,
        site_to_ast_path=site_to_path,
        live_binders_per_bond=live,
    )
    return state, meta
```

- [ ] **Step 4: Run smoke tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_encoder_smoke.py -v`
Expected: 9 passed.

- [ ] **Step 5: Run all logic tests to confirm no regressions**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/ -v 2>&1 | tail -20`
Expected: all logic tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/encoder.py src/qft_pcn/tests/test_logic_encoder_smoke.py
git commit -m "$(cat <<'EOF'
feat(logic/encoder): top-level encode() wiring the pipeline

Pipeline: resolve -> serialize -> compute_types -> compute_live_binders ->
build_site_tensors -> MPS -> normalize. Returns (state, EncodingMeta).

Smoke tests cover all five error paths (EncodingTooLarge, TooManyBinders,
IntLiteralOutOfRange, IllScopedVar, plus the PAD vacuum property).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

**End of Part 2.** Continue with Part 3 (decoder, sample, holes, gate-based cross-check, round-trip tests, all acceptance tests, final verification). See `2026-05-20-ast-mps-encoder-part3.md`.
