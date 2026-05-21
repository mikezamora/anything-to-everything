# Typing Hamiltonian Implementation Plan — Part 1 of 3 (Encoder Extension)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement sub-project B from `docs/superpowers/specs/2026-05-21-typing-hamiltonian-design.md`: the Hamiltonian-based STLC type checker. Sub-project A's encoder is extended to write a 5th field species (`tobl`) at each site and a `param_ty` tag on each bid bond channel. Sub-project B then builds `H_typing` whose ⟨ψ|H|ψ⟩ equals zero iff the encoded AST is well-typed.

**Architecture:** Five field species per site (`kind / type / bid / value / tobl`). The bid bond extension carries each live binder's `param_ty` as a real bond DOF: `(1 + 8|L|)`-dimensional bid bond. Hamiltonian terms are stored in **factored** form (per-species small matrices) and evaluated via custom factored-expectation helpers; the dense `65536 × 65536` operators are never materialized.

**Tech Stack:** Same as A — Python 3.11, numpy, scipy, pytest. Built on `src/qft_pcn/logic/` from sub-project A and the existing `src/qft_pcn/qft/` machinery.

**Driving principles (from spec §1, non-negotiable):**

1. **One H_typing per N-site lattice.** No AST data in the Hamiltonian's construction.
2. **No classical type-checker shortcuts.** Every rule = real Hermitian operator on the MPS.
3. **Binder→use coupling realized as bond DOF.** `param_ty` is a real quantum bond degree of freedom on the bid channels (not metadata).
4. **Non-adjacent obligations realized as a per-site `tobl` register.** The encoder writes the parent-imposed obligation at each child's root site at encode time.
5. **Per-term residuals expose (rule_id, site).** Sub-project D consumes them; the API must be granular.
6. **Reuse the QFT machinery.** No new MPS classes; only new logic-layer code.

**Reference:** The spec at `docs/superpowers/specs/2026-05-21-typing-hamiltonian-design.md` is the authoritative contract. When in doubt, re-read it. If something here disagrees with the spec, the spec wins.

**Test command throughout:** Run pytest via the project venv: `.venv/bin/python -m pytest <path> -v`.

**This document is Part 1 of 3.**
- **Part 1** (this document): Tasks 1–9 — encoder extension (constants, NodeOccupancy.tobl, obligation walk, channel param_ty in bond tensors, EncodingMeta extension, encoder default chi_max bump, smoke tests).
- **Part 2**: Tasks 10–18 — TypingHamiltonian skeleton, factored-expectation helpers, per-rule term builders, term_energy, total_energy, residuals.
- **Part 3**: Tasks 19–28 — acceptance tests (5 WT programs, 5 IT programs, isolation tests, classical agreement, performance budget, final verification).

---

## Task 1: Add `tobl` constants and `TOBL_*` basis to `encoding.py`

**Files:**
- Modify: `src/qft_pcn/logic/encoding.py` (add `TOBL_*` constants, extend `SPECIES`).
- Modify: `src/qft_pcn/tests/test_logic_encoding.py` (add tests).

The new species mirrors `type` 1:1 in its basis. Cutoff = 8. Local d grows 8×.

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_logic_encoding.py`:

```python
from src.qft_pcn.logic.encoding import (
    TOBL_NONE, TOBL_INT, TOBL_BOOL,
    TOBL_ARR_II, TOBL_ARR_IB, TOBL_ARR_BI, TOBL_ARR_BB,
    TOBL_ARR_NESTED, TOBL_CUTOFF,
)


def test_tobl_basis_mirrors_type():
    """Spec §4.2: tobl basis is 1:1 with type basis."""
    from src.qft_pcn.logic.encoding import (
        TYPE_NONE, TYPE_INT, TYPE_BOOL,
        TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI, TYPE_ARR_BB,
        TYPE_ARR_NESTED, TYPE_CUTOFF,
    )
    assert TOBL_NONE == TYPE_NONE == 0
    assert TOBL_INT == TYPE_INT == 1
    assert TOBL_BOOL == TYPE_BOOL == 2
    assert TOBL_ARR_II == TYPE_ARR_II == 3
    assert TOBL_ARR_IB == TYPE_ARR_IB == 4
    assert TOBL_ARR_BI == TYPE_ARR_BI == 5
    assert TOBL_ARR_BB == TYPE_ARR_BB == 6
    assert TOBL_ARR_NESTED == TYPE_ARR_NESTED == 7
    assert TOBL_CUTOFF == TYPE_CUTOFF == 8


def test_species_list_has_tobl_as_5th():
    """Spec §4.1: SPECIES is 5 species, with tobl appended last."""
    from src.qft_pcn.logic.encoding import SPECIES, D_LOCAL
    names = [s.name for s in SPECIES]
    assert names == ["kind", "type", "bid", "value", "tobl"]
    assert SPECIES[-1].cutoff == 8
    # d_local = 8 * 8 * 8 * 16 * 8 = 65536.
    assert D_LOCAL == 65536


def test_species_dims_tuple():
    from src.qft_pcn.logic.encoding import SPECIES_DIMS, SPECIES_NAMES
    assert SPECIES_NAMES == ("kind", "type", "bid", "value", "tobl")
    assert SPECIES_DIMS == (8, 8, 8, 16, 8)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_encoding.py -k "tobl or species_list_has_tobl" -v`
Expected: `ImportError` on `TOBL_*` names.

- [ ] **Step 3: Update `encoding.py`**

Edit `src/qft_pcn/logic/encoding.py`. Add the tobl constants right after the type-register constants:

```python
# ---- type-obligation register --------------------------------------------
# Mirrors the type register's basis exactly. Cutoff = 8.

TOBL_NONE = 0       # no parent-imposed type expectation
TOBL_INT = 1
TOBL_BOOL = 2
TOBL_ARR_II = 3
TOBL_ARR_IB = 4
TOBL_ARR_BI = 5
TOBL_ARR_BB = 6
TOBL_ARR_NESTED = 7
TOBL_CUTOFF = 8

assert TOBL_CUTOFF == TYPE_CUTOFF, (
    "tobl basis must mirror type basis exactly"
)
```

Then update `SPECIES_NAMES`, `SPECIES_DIMS`, `D_LOCAL`, and `SPECIES`:

```python
SPECIES_NAMES: tuple[str, ...] = ("kind", "type", "bid", "value", "tobl")
SPECIES_DIMS: tuple[int, ...] = (KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF,
                                 VALUE_CUTOFF, TOBL_CUTOFF)
D_LOCAL: int = math.prod(SPECIES_DIMS)


SPECIES: tuple[FieldSpecies, ...] = (
    FieldSpecies(name="kind",  cutoff=KIND_CUTOFF,  bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="type",  cutoff=TYPE_CUTOFF,  bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="bid",   cutoff=BID_CUTOFF,   bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="value", cutoff=VALUE_CUTOFF, bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="tobl",  cutoff=TOBL_CUTOFF,  bare_mass=0.0, kinetic=0.0),
)
```

- [ ] **Step 4: Run the new tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_encoding.py -v 2>&1 | tail -20`
Expected: the three new tests pass.

The pre-existing tests for `D_LOCAL`, `SPECIES_DIMS`, `SPECIES_NAMES` will now FAIL because they expected the 4-species values. **Don't fix them yet** — they'll be fixed in Task 9 (when we update A's existing tests for the new 5-species default).

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/encoding.py src/qft_pcn/tests/test_logic_encoding.py
git commit -m "$(cat <<'EOF'
feat(logic/encoding): add tobl (type-obligation) as 5th field species

Sub-project B's typing Hamiltonian needs a per-site register to carry the
parent-imposed type expectation. The tobl basis mirrors type 1:1 (cutoff 8,
TOBL_NONE through TOBL_ARR_NESTED). Local d grows 8x to 65536; per-site
operators are kept factored (never materialized dense).

This is the encoder extension's foundation; subsequent commits write tobl
at encode time and use it in the typing-Hamiltonian terms.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Extend `EncodingMeta` with `tobl_per_site`, `nested_tobl_index`, `channel_param_ty_per_bond`

**Files:**
- Modify: `src/qft_pcn/logic/encoding.py` (extend `EncodingMeta` dataclass).
- Modify: `src/qft_pcn/tests/test_logic_encoding.py` (add tests).

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_logic_encoding.py`:

```python
def test_encoding_meta_has_new_fields():
    """Spec §5.3: EncodingMeta gains tobl_per_site, nested_tobl_index,
    channel_param_ty_per_bond."""
    from src.qft_pcn.logic.encoding import (
        EncodingMeta, SPECIES, BinderHandle,
    )
    meta = EncodingMeta(
        N=32,
        chi_max=32,
        field_dims={"kind": 8, "type": 8, "bid": 8, "value": 16, "tobl": 8},
        species=list(SPECIES),
        nested_type_index={},
        site_to_ast_path={},
        live_binders_per_bond=[],
        tobl_per_site=[0] * 32,
        nested_tobl_index={},
        channel_param_ty_per_bond=[],
    )
    assert meta.tobl_per_site == [0] * 32
    assert meta.nested_tobl_index == {}
    assert meta.channel_param_ty_per_bond == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_encoding.py::test_encoding_meta_has_new_fields -v`
Expected: `TypeError: __init__() got an unexpected keyword argument 'tobl_per_site'`.

- [ ] **Step 3: Extend `EncodingMeta`**

Edit `src/qft_pcn/logic/encoding.py`. Replace the existing `EncodingMeta` dataclass with:

```python
@dataclass
class EncodingMeta:
    """Side data produced by the encoder, consumed by decoder and downstream
    sub-projects B/C/D/E.

    The fields tobl_per_site, nested_tobl_index, and channel_param_ty_per_bond
    are populated by the typing-aware encoder extension (sub-project B,
    spec §5). They mirror their type-register counterparts:

      tobl_per_site[k]:        the flat tobl tag at site k. Default TOBL_NONE.
      nested_tobl_index[k]:    full Ty for sites where tobl_per_site[k] = TOBL_ARR_NESTED.
      channel_param_ty_per_bond[b][c]: flat param_ty tag of the binder
                                       occupying channel c (0-indexed) of bond b.
                                       Parallel to live_binders_per_bond.
    """
    N: int
    chi_max: int
    field_dims: dict[str, int]
    species: list[FieldSpecies]
    nested_type_index: dict[int, Ty]
    site_to_ast_path: dict[int, tuple[int, ...]]
    live_binders_per_bond: list[list[BinderHandle]]
    tobl_per_site: list[int]
    nested_tobl_index: dict[int, Ty]
    channel_param_ty_per_bond: list[list[int]]
```

- [ ] **Step 4: Update `encoder.py` to populate the new fields with defaults**

Edit `src/qft_pcn/logic/encoder.py`. Find the `EncodingMeta(...)` construction near the bottom and add the three new fields with safe defaults (we'll populate them properly in later tasks):

```python
    meta = EncodingMeta(
        N=N,
        chi_max=chi_max,
        field_dims={
            "kind": KIND_CUTOFF, "type": TYPE_CUTOFF,
            "bid": BID_CUTOFF, "value": VALUE_CUTOFF, "tobl": TOBL_CUTOFF,
        },
        species=list(SPECIES),
        nested_type_index=nested,
        site_to_ast_path=site_to_path,
        live_binders_per_bond=live,
        tobl_per_site=[0] * N,                 # populated in Task 4
        nested_tobl_index={},                   # populated in Task 4
        channel_param_ty_per_bond=[],           # populated in Task 6
    )
```

Also add the import for `TOBL_CUTOFF` at the top of `encoder.py`:

```python
from .encoding import (
    SPECIES, EncodingMeta, KIND_CUTOFF, TYPE_CUTOFF,
    BID_CUTOFF, VALUE_CUTOFF, TYPE_ARR_NESTED,
    TOBL_CUTOFF,                  # new
)
```

- [ ] **Step 5: Run the test**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_encoding.py::test_encoding_meta_has_new_fields -v`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/encoding.py src/qft_pcn/logic/encoder.py src/qft_pcn/tests/test_logic_encoding.py
git commit -m "$(cat <<'EOF'
feat(logic/encoding): extend EncodingMeta with tobl + channel_param_ty fields

Adds three fields populated by the typing-aware encoder extension:
  - tobl_per_site: parent-imposed type obligation at each site
  - nested_tobl_index: full Ty for TOBL_ARR_NESTED sites
  - channel_param_ty_per_bond: per-bond list of binder param_ty tags

For now they're empty/zero defaults; subsequent commits wire them up.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `tobl_tag` to `NodeOccupancy` and the obligation computation helper

**Files:**
- Modify: `src/qft_pcn/logic/_serialize.py` (add `tobl_tag` field to `NodeOccupancy`).
- Create: `src/qft_pcn/logic/_typing_extension.py` (compute_tobl_tags).
- Create: `src/qft_pcn/tests/test_logic_typing_extension.py`.

The encoder walks the AST in pre-order. For each child, the parent knows what type it expects (per spec §5.1). We need a separate top-down walk that produces a `list[int]` of length `n_nodes` (the tobl tag per emitted site, before PAD).

- [ ] **Step 1: Add `tobl_tag` to `NodeOccupancy`**

Edit `src/qft_pcn/logic/_serialize.py`. Update the `NodeOccupancy` dataclass:

```python
@dataclass
class NodeOccupancy:
    """Per-site descriptor produced by the serializer.

    The descriptor records *what* the site holds; the encoder turns these
    into actual MPS tensors. binder_ref / var_ref are non-None only for
    KIND_LAM / KIND_VAR sites respectively.

    tobl_tag is the flat tag of the parent-imposed type obligation at this
    site (TOBL_NONE if no obligation, e.g. for the root site and for
    leaf-only sites with no parent context). Written by compute_tobl_tags
    (see _typing_extension.py).

    nested_tobl_ty stores the full Ty when tobl_tag is TOBL_ARR_NESTED.
    """
    kind: int
    ty: Optional[Ty] = None
    int_val: Optional[int] = None
    bool_val: Optional[bool] = None
    bin_op: Optional[str] = None
    binder_ref: Optional[BinderRef] = None
    var_ref: Optional[VarRef] = None
    ast_path: tuple[int, ...] = ()
    tobl_tag: int = 0                       # TOBL_NONE
    nested_tobl_ty: Optional[Ty] = None     # set when tobl_tag == TOBL_ARR_NESTED
```

- [ ] **Step 2: Write the failing test for `compute_tobl_tags`**

Create `src/qft_pcn/tests/test_logic_typing_extension.py`:

```python
"""Tests for logic/_typing_extension.py (obligation computation)."""

from __future__ import annotations

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic._serialize import serialize_preorder
from src.qft_pcn.logic._typing_extension import compute_tobl_tags
from src.qft_pcn.logic.encoding import (
    TOBL_NONE, TOBL_INT, TOBL_BOOL, TOBL_ARR_II, TOBL_ARR_BB,
)


def test_root_has_no_obligation():
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    tags = compute_tobl_tags(ast, sites)
    # Root site (LAM): no obligation.
    assert tags[0] == TOBL_NONE
    # Var inside Lam: LAM expects body type = dst(LAM.type) = T_INT.
    assert tags[1] == TOBL_INT
    # PAD sites: no obligation.
    assert tags[2] == TOBL_NONE
    assert tags[3] == TOBL_NONE


def test_app_arg_has_fn_src_obligation():
    """(\\x:Int. x)(1): the arg '1' has obligation = fn.src = T_INT."""
    ast = parse(r"(\x:Int. x)(1)")
    sites = serialize_preorder(ast, N=8)
    tags = compute_tobl_tags(ast, sites)
    # Layout: APP@0, LAM@1, VAR@2, INT@3 (the arg).
    assert tags[0] == TOBL_NONE         # APP root: no parent
    # fn (LAM at site 1) has NO obligation from APP — APP doesn't impose a
    # single type, only an arrow-dst constraint (handled by T-App-Arrow).
    assert tags[1] == TOBL_NONE
    # Body of LAM (VAR at site 2): obligation = dst(LAM.type) = T_INT.
    assert tags[2] == TOBL_INT
    # Arg (INT at site 3): obligation = src(fn.type) = T_INT.
    assert tags[3] == TOBL_INT


def test_if_branches_have_type_obligation():
    """if true then 1 else 2: cond is Bool-obligated, branches are
    obligated to the if's overall type (Int)."""
    ast = parse(r"if true then 1 else 2")
    sites = serialize_preorder(ast, N=8)
    tags = compute_tobl_tags(ast, sites)
    # Layout: IF@0, BoolLit@1 (cond), IntLit@2 (then), IntLit@3 (else).
    assert tags[0] == TOBL_NONE   # IF root
    assert tags[1] == TOBL_BOOL   # cond -> Bool
    assert tags[2] == TOBL_INT    # then -> Int (= if's type)
    assert tags[3] == TOBL_INT    # else -> Int


def test_bin_operands_have_int_obligation():
    """1 + 2: both operands obligated to T_INT regardless of arith vs cmp."""
    ast = parse(r"1 + 2")
    sites = serialize_preorder(ast, N=4)
    tags = compute_tobl_tags(ast, sites)
    # Layout: BIN+@0, IntLit@1, IntLit@2.
    assert tags[0] == TOBL_NONE
    assert tags[1] == TOBL_INT   # lhs
    assert tags[2] == TOBL_INT   # rhs


def test_bin_cmp_operands_still_int():
    """x < 5: operands are Int (BIN-Cmp's operands are still Int)."""
    ast = parse(r"\x:Int. x < 5")
    sites = serialize_preorder(ast, N=8)
    tags = compute_tobl_tags(ast, sites)
    # Layout: LAM@0, BIN<@1, VAR_x@2, IntLit@3.
    assert tags[0] == TOBL_NONE
    assert tags[1] == TOBL_BOOL  # LAM's body obligation = dst(Int->Bool) = Bool
    assert tags[2] == TOBL_INT   # BIN's lhs
    assert tags[3] == TOBL_INT   # BIN's rhs


def test_nested_lambda_obligations():
    """\\x:Int. \\y:Int. x + y:
       outer LAM has body obligation = dst(Int -> Int->Int) = T_ARR_II.
       inner LAM has body obligation = dst(Int->Int) = T_INT.
       BIN's operands obligation = T_INT each.
    """
    ast = parse(r"\x:Int. \y:Int. x + y")
    sites = serialize_preorder(ast, N=8)
    tags = compute_tobl_tags(ast, sites)
    # Layout: LAM_x@0, LAM_y@1, BIN+@2, VAR_x@3, VAR_y@4.
    assert tags[0] == TOBL_NONE
    assert tags[1] == TOBL_ARR_II   # outer LAM expects body type = Int->Int
    assert tags[2] == TOBL_INT      # inner LAM expects body type = Int (sum)
    assert tags[3] == TOBL_INT      # BIN's lhs
    assert tags[4] == TOBL_INT      # BIN's rhs


def test_pad_sites_have_none_obligation():
    """Sites past the AST are PAD with TOBL_NONE."""
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=8)
    tags = compute_tobl_tags(ast, sites)
    for k in range(2, 8):
        assert tags[k] == TOBL_NONE
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_extension.py -v`
Expected: `ImportError` on `_typing_extension`.

- [ ] **Step 4: Implement `compute_tobl_tags`**

Create `src/qft_pcn/logic/_typing_extension.py`:

```python
"""Encoder extension for sub-project B (typing Hamiltonian).

Computes, for each site of a pre-order-serialized AST, the parent-imposed
type obligation (`tobl_tag`). The Hamiltonian's T-Obligation rule then
projects, site-locally, onto the subspace where `type(k)` != `tobl(k)` when
`tobl(k)` is not TOBL_NONE.

See docs/superpowers/specs/2026-05-21-typing-hamiltonian-design.md, §5.1.

The obligation rule table:
  parent     | child position    | tobl(child_root)
  -----------|-------------------|-------------------------------
  (root)     | -                 | TOBL_NONE
  LAM        | body              | dst(LAM.type) as flat tag
  APP        | fn                | TOBL_NONE
  APP        | arg               | src(fn.type) as flat tag
  IF         | cond              | TOBL_BOOL
  IF         | then              | type(IF)
  IF         | else              | type(IF)
  BIN        | lhs               | TOBL_INT
  BIN        | rhs               | TOBL_INT
"""

from __future__ import annotations

from typing import Optional

from .ast import Node, Var, Lam, App, IntLit, BoolLit, If, Bin, Ty, TArrow
from ._serialize import NodeOccupancy
from ._types import ty_to_tag, _compute_ast_type
from .encoding import (
    KIND_PAD,
    TOBL_NONE, TOBL_INT, TOBL_BOOL, TOBL_ARR_NESTED,
    TYPE_ARR_NESTED,
)


def compute_tobl_tags(root: Node,
                      sites: list[NodeOccupancy]
                      ) -> list[int]:
    """For each site, compute the flat tobl tag.

    Returns a list of length N (= len(sites)). Sites with no parent
    obligation (root, PAD sites) get TOBL_NONE. The full Ty for a TOBL_ARR_NESTED
    obligation is stored on the corresponding NodeOccupancy.nested_tobl_ty
    attribute.
    """
    N = len(sites)
    tags: list[int] = [TOBL_NONE] * N

    # Walk the AST top-down with the parent-imposed obligation and the
    # current ast_path. The pre-order layout means we can index into `sites`
    # by ast_path (matching what _serialize emitted), provided the AST
    # walks in canonical order. Easier: re-walk in pre-order and track
    # site_idx as we go.

    site_iter = [0]   # mutable cursor over sites

    def _emit(node: Node, obligation: int,
              obligation_ty: Optional[Ty], env: list[tuple[str, Ty]]) -> None:
        idx = site_iter[0]
        tags[idx] = obligation
        sites[idx].tobl_tag = obligation
        if obligation == TOBL_ARR_NESTED and obligation_ty is not None:
            sites[idx].nested_tobl_ty = obligation_ty
        site_iter[0] += 1

        if isinstance(node, (Var, IntLit, BoolLit)):
            return

        if isinstance(node, Lam):
            # body's obligation = dst(LAM.type) = dst(TArrow(param_ty, body_type))
            #                   = body_type
            body_ty = _compute_ast_type(node.body,
                                        env + [(node.param, node.param_ty)])
            body_tag, body_nested = ty_to_tag(body_ty)
            _emit(node.body, body_tag,
                  body_nested if body_tag == TOBL_ARR_NESTED else None,
                  env + [(node.param, node.param_ty)])
            return

        if isinstance(node, App):
            # fn: no obligation imposed by App's local type (only an
            # arrow-dst constraint, handled by T-App-Arrow).
            _emit(node.fn, TOBL_NONE, None, env)
            # arg: obligation = src(fn.type).
            fn_ty = _compute_ast_type(node.fn, env)
            if isinstance(fn_ty, TArrow):
                src_tag, src_nested = ty_to_tag(fn_ty.src)
                _emit(node.arg, src_tag,
                      src_nested if src_tag == TOBL_ARR_NESTED else None, env)
            else:
                # ill-typed input: fn is not an arrow. We can't propagate
                # an obligation for arg in this case. Use TOBL_NONE; the
                # T-App-Arrow rule will fire at the parent APP regardless.
                _emit(node.arg, TOBL_NONE, None, env)
            return

        if isinstance(node, If):
            # cond: Bool.
            _emit(node.cond, TOBL_BOOL, None, env)
            # then/else: same as IF's overall type, which equals type(then).
            then_ty = _compute_ast_type(node.then_b, env)
            then_tag, then_nested = ty_to_tag(then_ty)
            _emit(node.then_b, then_tag,
                  then_nested if then_tag == TOBL_ARR_NESTED else None, env)
            _emit(node.else_b, then_tag,
                  then_nested if then_tag == TOBL_ARR_NESTED else None, env)
            return

        if isinstance(node, Bin):
            # Both operands: TOBL_INT (true for both arith and cmp ops).
            _emit(node.lhs, TOBL_INT, None, env)
            _emit(node.rhs, TOBL_INT, None, env)
            return

        raise TypeError(f"unsupported AST node in tobl walk: "
                        f"{type(node).__name__}")

    _emit(root, TOBL_NONE, None, [])
    # Remaining (PAD) sites: TOBL_NONE (already initialized).
    return tags
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_extension.py -v`
Expected: 7 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/_serialize.py src/qft_pcn/logic/_typing_extension.py src/qft_pcn/tests/test_logic_typing_extension.py
git commit -m "$(cat <<'EOF'
feat(logic/_typing_extension): per-site parent-obligation computation

Top-down pre-order walk that computes the type obligation each child
inherits from its parent, per spec §5.1. NodeOccupancy gains tobl_tag and
nested_tobl_ty fields. Coverage: Lam body, App fn (none) and arg, If cond
(Bool), If then/else (= type(if)), Bin operands (always Int).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire `compute_tobl_tags` into the encoder

**Files:**
- Modify: `src/qft_pcn/logic/encoder.py` (call `compute_tobl_tags`, populate `tobl_per_site` and `nested_tobl_index`).
- Modify: `src/qft_pcn/tests/test_logic_encoder_smoke.py` (add a tobl-bookkeeping test if it exists; otherwise add to the encoder smoke test file).

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_logic_encoder_smoke.py` (or create it if missing):

```python
def test_encoder_populates_tobl_per_site():
    """The encoder calls compute_tobl_tags and stores results in
    EncodingMeta.tobl_per_site."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.encoding import TOBL_NONE, TOBL_INT

    state, meta = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    assert len(meta.tobl_per_site) == 8
    # Site 0 = LAM (root, no obligation).
    assert meta.tobl_per_site[0] == TOBL_NONE
    # Site 1 = Var(x), body of Lam — obligation = Int.
    assert meta.tobl_per_site[1] == TOBL_INT
    # PAD sites.
    for k in range(2, 8):
        assert meta.tobl_per_site[k] == TOBL_NONE
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_encoder_smoke.py::test_encoder_populates_tobl_per_site -v`
Expected: AssertionError (tobl_per_site is still `[0] * N`).

- [ ] **Step 3: Wire `compute_tobl_tags` into `encoder.encode()`**

Edit `src/qft_pcn/logic/encoder.py`. Add the import:

```python
from ._typing_extension import compute_tobl_tags
```

After `live = compute_live_binders(sites)` and before `tensors = build_site_tensors(...)`, add:

```python
    tobl_tags = compute_tobl_tags(ast, sites)
```

Update the `EncodingMeta` construction to populate `tobl_per_site` and `nested_tobl_index`:

```python
    nested_tobl: dict[int, Ty] = {}
    for k, occ in enumerate(sites):
        if occ.tobl_tag == TYPE_ARR_NESTED and occ.nested_tobl_ty is not None:
            nested_tobl[k] = occ.nested_tobl_ty

    meta = EncodingMeta(
        # ...
        tobl_per_site=tobl_tags,
        nested_tobl_index=nested_tobl,
        # ...
    )
```

(Add `TOBL_ARR_NESTED` to the import list if needed.)

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_encoder_smoke.py::test_encoder_populates_tobl_per_site -v`
Expected: pass.

- [ ] **Step 5: Verify A's full test suite still passes** *(we haven't changed tensor shapes yet; only metadata is added)*

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_encoder_smoke.py src/qft_pcn/tests/test_logic_roundtrip.py -v 2>&1 | tail -20`

Note: D_LOCAL has grown 8x, so any test that compares the actual MPS bond/site shapes will FAIL. The `test_logic_encoding.py::test_species_*` tests already need updating. The tensor tests in `test_logic_tensors.py::test_site_tensor_shapes` etc. will also fail because they expect D_LOCAL=8192. We'll fix all of these in Task 9.

For now, run only the smoke test specific to this task:

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_encoder_smoke.py::test_encoder_populates_tobl_per_site src/qft_pcn/tests/test_logic_typing_extension.py -v`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/encoder.py src/qft_pcn/tests/test_logic_encoder_smoke.py
git commit -m "$(cat <<'EOF'
feat(logic/encoder): write tobl tags into EncodingMeta

The encoder now invokes compute_tobl_tags after serialization and stores
the per-site obligation tags in meta.tobl_per_site. TOBL_ARR_NESTED sites
get their full Ty stored in nested_tobl_index. The tobl values are not yet
written into MPS tensors (that's Task 7); only metadata is populated here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update `_basis_index` and tensor construction for 5-species local Hilbert space

**Files:**
- Modify: `src/qft_pcn/logic/_tensors.py` (extend `_basis_index` to take `tobl_idx`; update `_combine_factored_site`).
- Modify: `src/qft_pcn/tests/test_logic_tensors.py` (update existing tests to pass `tobl_idx`, add new tests for the 5-species basis).

The local basis stride is now `kind > type > bid > value > tobl` (leftmost slowest). Per spec §4.4:

```
flat_index(s_kind, s_type, s_bid, s_value, s_tobl)
  = s_kind * (T * B * V * O)
  + s_type * (B * V * O)
  + s_bid  * (V * O)
  + s_value * O
  + s_tobl
```

with `T=8, B=8, V=16, O=8`. So `D_LOCAL = 65536`.

- [ ] **Step 1: Write the failing test**

Replace the existing `_basis_index` test (if it exists in `test_logic_tensors.py`) with the 5-species version. Or add a new test:

```python
def test_basis_index_5_species():
    """Per spec §4.4: flat index of (s_kind, s_type, s_bid, s_value, s_tobl)
    with leftmost slowest, rightmost (tobl) fastest."""
    from src.qft_pcn.logic._tensors import _basis_index
    from src.qft_pcn.logic.encoding import (
        TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF,
    )
    # (0, 0, 0, 0, 0) -> 0
    assert _basis_index(0, 0, 0, 0, 0) == 0
    # (0, 0, 0, 0, 1) -> 1
    assert _basis_index(0, 0, 0, 0, 1) == 1
    # (0, 0, 0, 1, 0) -> TOBL_CUTOFF = 8
    assert _basis_index(0, 0, 0, 1, 0) == TOBL_CUTOFF
    # (0, 0, 1, 0, 0) -> VALUE_CUTOFF * TOBL_CUTOFF = 128
    assert _basis_index(0, 0, 1, 0, 0) == VALUE_CUTOFF * TOBL_CUTOFF
    # (0, 1, 0, 0, 0) -> BID_CUTOFF * VALUE_CUTOFF * TOBL_CUTOFF = 1024
    assert _basis_index(0, 1, 0, 0, 0) == (BID_CUTOFF * VALUE_CUTOFF
                                            * TOBL_CUTOFF)
    # (1, 0, 0, 0, 0) -> TYPE_CUTOFF * BID_CUTOFF * VALUE_CUTOFF * TOBL_CUTOFF
    assert _basis_index(1, 0, 0, 0, 0) == (TYPE_CUTOFF * BID_CUTOFF
                                            * VALUE_CUTOFF * TOBL_CUTOFF)


def test_site_tensor_d_local_is_65536():
    """After 5-species switch, site tensors have shape (chi_l, 65536, chi_r)."""
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic._serialize import serialize_preorder
    from src.qft_pcn.logic._types import compute_site_types
    from src.qft_pcn.logic._channels import compute_live_binders
    from src.qft_pcn.logic._tensors import build_site_tensors
    from src.qft_pcn.logic._typing_extension import compute_tobl_tags
    from src.qft_pcn.logic.encoding import D_LOCAL

    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    compute_tobl_tags(ast, sites)
    tensors = build_site_tensors(sites, types, live)
    for t in tensors:
        assert t.shape[1] == D_LOCAL == 65536
```

- [ ] **Step 2: Run the test to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_tensors.py::test_basis_index_5_species src/qft_pcn/tests/test_logic_tensors.py::test_site_tensor_d_local_is_65536 -v`
Expected: `TypeError` because `_basis_index` takes 4 args, not 5; the tensor test will also fail.

- [ ] **Step 3: Update `_basis_index` and add `_local_tobl` helper**

Edit `src/qft_pcn/logic/_tensors.py`. Replace the existing `_basis_index`:

```python
def _basis_index(kind_idx: int, type_idx: int, bid_idx: int,
                 value_idx: int, tobl_idx: int) -> int:
    """Linear index of (kind, type, bid, value, tobl) in the local
    65536-dim basis. Leftmost species changes slowest.
    """
    return ((((kind_idx * TYPE_CUTOFF + type_idx) * BID_CUTOFF + bid_idx)
             * VALUE_CUTOFF + value_idx) * TOBL_CUTOFF + tobl_idx)
```

Add the imports for `TOBL_CUTOFF, TOBL_NONE` at the top of `_tensors.py`:

```python
from .encoding import (
    # ... existing ...
    TOBL_CUTOFF, TOBL_NONE,
)
```

- [ ] **Step 4: Update `_combine_factored_site` to take a tobl index**

Replace `_combine_factored_site` in `_tensors.py`:

```python
def _combine_factored_site(
    kind_idx: int, type_idx: int, value_idx: int, tobl_idx: int,
    bid_tensor: np.ndarray
) -> np.ndarray:
    """Combine the (kind, type, value, tobl) deterministic indices and the
    (chi_l, BID, chi_r) bid sub-tensor into the full
    (chi_l, D_LOCAL, chi_r) site tensor.

    The kind/type/value/tobl registers are *product*: one basis index has
    full amplitude. So the output tensor is nonzero only on the slice
    flat_basis_index = base + bid_idx * (VALUE_CUTOFF * TOBL_CUTOFF) + ...
    for varying bid_idx.

    Specifically, for each bid index b in [0, BID_CUTOFF):
        flat = ((kind_idx * TYPE_CUTOFF + type_idx) * BID_CUTOFF + b)
                * VALUE_CUTOFF * TOBL_CUTOFF
              + value_idx * TOBL_CUTOFF
              + tobl_idx
        out[:, flat, :] = bid_tensor[:, b, :]
    """
    chi_l, B, chi_r = bid_tensor.shape
    assert B == BID_CUTOFF
    out = np.zeros((chi_l, D_LOCAL, chi_r), dtype=complex)
    inner = value_idx * TOBL_CUTOFF + tobl_idx
    bid_stride = VALUE_CUTOFF * TOBL_CUTOFF
    base_no_bid = (kind_idx * TYPE_CUTOFF + type_idx) * BID_CUTOFF * bid_stride
    for b in range(BID_CUTOFF):
        flat = base_no_bid + b * bid_stride + inner
        out[:, flat, :] = bid_tensor[:, b, :]
    return out
```

- [ ] **Step 5: Update `build_site_tensors` to pass `tobl_idx`**

Edit the `build_site_tensors` function. Where it currently calls `_combine_factored_site(kind_idx, type_idx, value_idx, bid_tensor)`, change to pass `occ.tobl_tag`:

```python
def build_site_tensors(
    sites: list[NodeOccupancy],
    type_tags: list[int],
    live_binders_per_bond: list[list[BinderHandle]],
) -> list[np.ndarray]:
    """Top-level builder. ..."""
    N = len(sites)
    bounded_live = [[]] + list(live_binders_per_bond) + [[]]

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
        tobl_idx = occ.tobl_tag       # default 0 = TOBL_NONE
        site_T = _combine_factored_site(
            kind_idx=kind_idx, type_idx=type_idx, value_idx=value_idx,
            tobl_idx=tobl_idx, bid_tensor=bid_T,
        )
        tensors.append(site_T)
    return tensors
```

- [ ] **Step 6: Run the new tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_tensors.py::test_basis_index_5_species src/qft_pcn/tests/test_logic_tensors.py::test_site_tensor_d_local_is_65536 -v`
Expected: pass.

- [ ] **Step 7: Verify the build still produces a valid normalized MPS**

Run: `.venv/bin/python -c "
from src.qft_pcn.logic.encoder import encode
from src.qft_pcn.logic.ast import parse
state, meta = encode(parse(r'\x:Int. x'), N=8, chi_max=32)
print('norm_sq:', state.norm_sq())
print('bond dims:', state.bond_dimensions())
print('local d:', state.d)
print('tobl_per_site:', meta.tobl_per_site)
"`

Expected: `norm_sq ≈ 1.0`, `local d = 65536`, tobl_per_site shows correct tags.

- [ ] **Step 8: Commit**

```bash
git add src/qft_pcn/logic/_tensors.py src/qft_pcn/tests/test_logic_tensors.py
git commit -m "$(cat <<'EOF'
feat(logic/_tensors): 5-species local Hilbert space with tobl as innermost

_basis_index now takes (kind, type, bid, value, tobl) with tobl as the
fastest-varying species (stride 1, span 8). _combine_factored_site embeds
the chosen tobl_idx into the flat 65536-dim index alongside the other
deterministic register indices. build_site_tensors reads occ.tobl_tag
(populated by compute_tobl_tags) and passes it through.

The MPS now has D_LOCAL=65536 per site, 8x A's previous D_LOCAL.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Extend the bid bond with per-channel `param_ty` tag (real bond DOF)

**Files:**
- Modify: `src/qft_pcn/logic/_tensors.py` (_bid_bond_tensor_at_site uses 1 + 8|L| channels).
- Modify: `src/qft_pcn/logic/_channels.py` (compute_channel_param_ty_per_bond).
- Modify: `src/qft_pcn/tests/test_logic_tensors.py` and `test_logic_channels.py`.

This is the **architecturally important** task in Part 1. The bid bond's basis grows from `{|no_info⟩} ∪ {|c_i⟩ : i in [1, L]}` (dim `1 + L`) to `{|no_info⟩} ∪ {|c_i, t⟩ : i in [1, L], t in [TYPE_CUTOFF]}` (dim `1 + 8L`).

The channel basis indices, after extension:
- `0`: `|no_info⟩`
- `1 + 8(i-1) + t` for `i ∈ [1, L]` and `t ∈ [0, TYPE_CUTOFF)`: `|c_i, t⟩`

For each binder `c_i`, the encoder writes a definite `t = ty_to_tag(c_i.param_ty)` — only that slot is populated; the other 7 are zero.

- [ ] **Step 1: Write the failing tests**

Append to `src/qft_pcn/tests/test_logic_channels.py`:

```python
def test_compute_channel_param_ty_per_bond_p1():
    """For \\x:Int. x, bond 0 has one channel with param_ty = TYPE_INT."""
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic._serialize import serialize_preorder
    from src.qft_pcn.logic._channels import (
        compute_live_binders, compute_channel_param_ty_per_bond,
    )
    from src.qft_pcn.logic.encoding import TYPE_INT

    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    live = compute_live_binders(sites)
    pt = compute_channel_param_ty_per_bond(sites, live)
    assert len(pt) == 3   # N-1 bonds
    # Bond 0: one channel (Lam_x), param_ty = Int.
    assert pt[0] == [TYPE_INT]
    # Bonds 1, 2: no channels (after Var consumed x).
    assert pt[1] == []
    assert pt[2] == []


def test_compute_channel_param_ty_per_bond_p3():
    """\\f:Int->Int. \\x:Int. f x — two binders with distinct param_ty."""
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic._serialize import serialize_preorder
    from src.qft_pcn.logic._channels import (
        compute_live_binders, compute_channel_param_ty_per_bond,
    )
    from src.qft_pcn.logic.encoding import TYPE_INT, TYPE_ARR_II

    ast = parse(r"\f:Int->Int. \x:Int. f x")
    sites = serialize_preorder(ast, N=16)
    live = compute_live_binders(sites)
    pt = compute_channel_param_ty_per_bond(sites, live)
    # Layout: LAM_f@0, LAM_x@1, APP@2, VAR_f@3, VAR_x@4, PAD...
    # Bond 0: between LAM_f@0 and LAM_x@1 — Lam_f live, param_ty = Int->Int.
    assert pt[0] == [TYPE_ARR_II]
    # Bond 1: between LAM_x@1 and APP@2 — both Lam_f and Lam_x live.
    assert pt[1] == [TYPE_ARR_II, TYPE_INT]
    # Bond 2: between APP@2 and VAR_f@3 — both still live.
    assert pt[2] == [TYPE_ARR_II, TYPE_INT]
    # Bond 3: between VAR_f@3 and VAR_x@4 — Lam_f consumed at site 3;
    #         only Lam_x remains.
    assert pt[3] == [TYPE_INT]
    # Bond 4: between VAR_x@4 and PAD@5 — Lam_x consumed; nothing.
    assert pt[4] == []
```

Append to `src/qft_pcn/tests/test_logic_tensors.py`:

```python
def test_bid_bond_extended_dim_is_one_plus_eight_L():
    """Spec §5.2: bid bond dim = 1 + 8 * |L_i|. For \\x:Int. x with
    1 live binder, bond 0 dim = 1 + 8 = 9. Tensor shape (1, 65536, 9)."""
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic._serialize import serialize_preorder
    from src.qft_pcn.logic._types import compute_site_types
    from src.qft_pcn.logic._channels import compute_live_binders
    from src.qft_pcn.logic._tensors import build_site_tensors
    from src.qft_pcn.logic._typing_extension import compute_tobl_tags

    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    compute_tobl_tags(ast, sites)
    tensors = build_site_tensors(sites, types, live)
    # LAM site 0: left bond dim 1, right bond dim 1 + 8*1 = 9.
    assert tensors[0].shape == (1, 65536, 9)
    # VAR site 1: left bond dim 9, right bond dim 1.
    assert tensors[1].shape == (9, 65536, 1)
    # PAD sites: (1, 65536, 1).
    assert tensors[2].shape == (1, 65536, 1)
    assert tensors[3].shape == (1, 65536, 1)


def test_bid_bond_only_one_slot_per_channel_populated():
    """For each binder channel, exactly one of the 8 param_ty slots is
    populated (the one matching the actual param_ty). The other 7 are 0."""
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.encoding import (
        KIND_LAM, KIND_VAR, KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF,
        VALUE_CUTOFF, TOBL_CUTOFF, TYPE_INT, BID_0,
    )
    import numpy as np

    state, meta = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    T = state.tensors[0]   # LAM site. Shape (1, 65536, 9).
    # The flat basis index of (kind=LAM, type=ARR_II, bid=BID_0, value=NONE,
    # tobl=NONE) is well-defined; we need to find the nonzero right-bond
    # slot for that local-basis-index.
    # Local basis index:
    kind, type_, bid_, value_, tobl_ = KIND_LAM, 3, BID_0, 0, 0
    # _basis_index in 5-species form:
    flat = ((((kind * TYPE_CUTOFF + type_) * BID_CUTOFF + bid_)
             * VALUE_CUTOFF + value_) * TOBL_CUTOFF + tobl_)
    # The right bond has 9 slots: index 0 = no_info, 1..8 = (channel_0, t)
    # for t in 0..7. The encoder wrote channel_0 (Lam_x) with param_ty = T_INT,
    # so slot 1 + 8*0 + T_INT = 1 + 0 + 1 = 2 is populated.
    populated_slot = 1 + 8 * 0 + TYPE_INT
    # Check: T[0, flat, populated_slot] != 0; T[0, flat, other slot] = 0
    # for the other channel-orbit slots.
    # Normalization rescales amplitudes; we just check the structural pattern.
    nonzero = [j for j in range(9) if abs(T[0, flat, j]) > 1e-12]
    # Two nonzero slots: index 0 (no_info passthrough — but actually
    # passthrough is on a DIFFERENT local basis state since the LAM site's
    # local kind is LAM, not PAD. Re-examine per encoder logic.).
    # The LAM site has local bid = BID_0; the bond-out is the new binder's
    # slot. So we expect populated_slot to be nonzero AND no_info (index 0)
    # to be EITHER zero or have a passthrough amplitude.
    # Per spec, the LAM also has a "no_info to no_info under BID_0" entry
    # (so subsequent LAMs can still source from no_info). That means
    # T[0, flat, 0] is also nonzero.
    assert populated_slot in nonzero, (
        f"slot {populated_slot} (Lam_x with param_ty=T_INT) should be populated"
    )
    # Slots in the same channel-orbit but with different param_ty must be 0:
    for t_other in range(TYPE_CUTOFF):
        if t_other == TYPE_INT:
            continue
        wrong_slot = 1 + 8 * 0 + t_other
        assert abs(T[0, flat, wrong_slot]) < 1e-12, (
            f"slot {wrong_slot} (Lam_x with param_ty=t_other={t_other}) "
            f"must be 0; got {T[0, flat, wrong_slot]}"
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_channels.py::test_compute_channel_param_ty_per_bond_p1 src/qft_pcn/tests/test_logic_tensors.py::test_bid_bond_extended_dim_is_one_plus_eight_L -v`
Expected: `ImportError` on `compute_channel_param_ty_per_bond`, and shape mismatch on the tensor test.

- [ ] **Step 3: Add `compute_channel_param_ty_per_bond` to `_channels.py`**

Append to `src/qft_pcn/logic/_channels.py`:

```python
from ._types import ty_to_tag


def compute_channel_param_ty_per_bond(
    sites: list[NodeOccupancy],
    live_binders_per_bond: list[list[BinderHandle]],
) -> list[list[int]]:
    """For each bond, return a list (parallel to live_binders_per_bond[i])
    of flat param_ty tags, one per live binder on the bond.

    The binder's param_ty is read from the LAM node at sites[binder.lam_site].
    """
    out: list[list[int]] = []
    for handles in live_binders_per_bond:
        bond_pts: list[int] = []
        for h in handles:
            occ = sites[h.lam_site]
            assert occ.binder_ref is not None, (
                f"site {h.lam_site} marked as LAM but no binder_ref"
            )
            lam = occ.binder_ref.lam_node
            tag, _nested = ty_to_tag(lam.param_ty)
            bond_pts.append(tag)
        out.append(bond_pts)
    return out
```

- [ ] **Step 4: Extend `_bid_bond_tensor_at_site` in `_tensors.py` to use the new basis**

Edit `src/qft_pcn/logic/_tensors.py`. Replace the entire `_bid_bond_tensor_at_site` function with the extended version. The key changes:
- Channel-out indices go from `1..L+1` (no_info, c_1, ..., c_L) to `0, 1 + 8(i-1) + t` for `i ∈ [1, L]` and `t ∈ [TYPE_CUTOFF]`.
- For each binder on a bond, ONLY the slot with its actual param_ty is written nonzero.

```python
def _bid_channel_slot(
    channel_index_1_based: int,   # 1-based (1..L)
    param_ty_tag: int,            # 0..TOBL_CUTOFF-1
) -> int:
    """Bond bid-register slot index for a binder's (channel, param_ty) pair.

    Channel indexing:
      slot 0          : no_info
      slot 1 + 8*(i-1) + t : channel i (1-based) with param_ty tag t.
    """
    assert 1 <= channel_index_1_based, (
        f"channel index must be >= 1, got {channel_index_1_based}"
    )
    assert 0 <= param_ty_tag < TOBL_CUTOFF, (
        f"param_ty_tag out of range: {param_ty_tag}"
    )
    return 1 + 8 * (channel_index_1_based - 1) + param_ty_tag


def _bid_bond_dim(L: int) -> int:
    """Bond dim on the bid register = 1 + 8*L (one no_info slot + 8 slots per
    live binder)."""
    return 1 + 8 * L
```

Then update the body of `_bid_bond_tensor_at_site` to use these helpers. Replace the entire function:

```python
def _bid_bond_tensor_at_site(
    site_idx: int,
    occ: NodeOccupancy,
    local_bid_value: int,
    left_live: list[BinderHandle],
    right_live: list[BinderHandle],
    left_param_ty: list[int],     # per-channel param_ty for left bond
    right_param_ty: list[int],    # per-channel param_ty for right bond
) -> np.ndarray:
    """Construct the bid-register-only sub-tensor for one site.

    Shape: (1 + 8*L_in, BID_CUTOFF, 1 + 8*L_out).

    Per spec §5.2 of B (and §5.4 of A), each binder channel carries an
    extra param_ty tag. The bond's bid-register basis is:
        index 0:        |no_info⟩
        index 1 + 8(i-1) + t : |channel_i, param_ty=t⟩ for i=1..L, t=0..7

    For each binder b on a bond, ONLY the slot with t = b.param_ty is nonzero;
    the other 7 slots in b's orbit are zero.
    """
    L_in = len(left_live)
    L_out = len(right_live)
    bond_dim_in = _bid_bond_dim(L_in)
    bond_dim_out = _bid_bond_dim(L_out)
    T = np.zeros((bond_dim_in, BID_CUTOFF, bond_dim_out), dtype=complex)

    NO_INFO_IN = 0
    NO_INFO_OUT = 0

    # Map binder handle -> (channel_index_1_based, param_ty_tag).
    left_ch_map: dict[BinderHandle, tuple[int, int]] = {}
    for i, bh in enumerate(left_live):
        left_ch_map[bh] = (i + 1, left_param_ty[i])
    right_ch_map: dict[BinderHandle, tuple[int, int]] = {}
    for i, bh in enumerate(right_live):
        right_ch_map[bh] = (i + 1, right_param_ty[i])

    def L_slot(bh: BinderHandle) -> int:
        c, t = left_ch_map[bh]
        return _bid_channel_slot(c, t)

    def R_slot(bh: BinderHandle) -> int:
        c, t = right_ch_map[bh]
        return _bid_channel_slot(c, t)

    # --- Non-binder, non-var, non-PAD passthrough sites (APP / IF / BIN / INT / BOOL) ---
    if occ.kind not in (KIND_VAR, KIND_LAM):
        # local bid = BID_NONE. no_info passes through; each live binder
        # passes through under its assigned slot.
        T[NO_INFO_IN, BID_NONE, NO_INFO_OUT] = 1.0
        for bh in left_live:
            if bh in right_ch_map:
                T[L_slot(bh), BID_NONE, R_slot(bh)] = 1.0
            else:
                # Shouldn't happen — non-Var sites don't drop binders.
                T[L_slot(bh), BID_NONE, NO_INFO_OUT] = 1.0
        return T

    if occ.kind == KIND_LAM:
        # The new binder is the rightmost entry in right_live (declaration order).
        new_binder = None
        for bh in right_live:
            if bh.lam_site == site_idx:
                new_binder = bh
                break
        if new_binder is None:
            raise RuntimeError(
                f"LAM site {site_idx}: no matching binder in right_live; "
                f"encoder bug — channels misaligned"
            )
        c_new_out_slot = R_slot(new_binder)
        # At a LAM the local bid is BID_0 (its own innermost-binder index).
        # Existing binders pass through under BID_0; the new binder is created
        # by routing no_info_in -> c_new_out_slot.
        T[NO_INFO_IN, local_bid_value, NO_INFO_OUT] = 1.0   # passthrough no_info
        T[NO_INFO_IN, local_bid_value, c_new_out_slot] = 1.0  # create new binder
        for bh in left_live:
            if bh in right_ch_map:
                T[L_slot(bh), local_bid_value, R_slot(bh)] = 1.0
            else:
                T[L_slot(bh), local_bid_value, NO_INFO_OUT] = 1.0
        return T

    # --- VAR site ---
    # The referenced binder's channel is consumed (if last use) or passed through.
    assert occ.var_ref is not None
    cands = (occ.var_ref.candidates
             if occ.var_ref.candidates
             else [(occ.var_ref.binder_site,
                    occ.var_ref.depth_from_innermost)])
    amp = 1.0 / np.sqrt(len(cands))

    from .encoding import MAX_BINDER_DEPTH, TooManyBinders, BID_0
    ref_handles_used: set[BinderHandle] = set()
    for cand_lam_site, cand_depth in cands:
        ref_handle = None
        for bh in left_live:
            if bh.lam_site == cand_lam_site:
                ref_handle = bh
                break
        if ref_handle is None:
            raise RuntimeError(
                f"VAR site {site_idx}: candidate binder at lam_site="
                f"{cand_lam_site} not in left_live (bookkeeping bug)"
            )
        ref_handles_used.add(ref_handle)
        if cand_depth >= MAX_BINDER_DEPTH:
            raise TooManyBinders(depth=cand_depth + 1,
                                 cutoff=MAX_BINDER_DEPTH)
        local_bid = BID_0 + cand_depth
        c_in = L_slot(ref_handle)
        if ref_handle not in right_ch_map:
            T[c_in, local_bid, NO_INFO_OUT] = amp
        else:
            T[c_in, local_bid, R_slot(ref_handle)] = amp

    # Pass-through for binders NOT referenced by this use, located on the
    # primary candidate's bid slice.
    primary_depth = cands[0][1]
    primary_local_bid = BID_0 + primary_depth
    T[NO_INFO_IN, primary_local_bid, NO_INFO_OUT] = 1.0
    for bh in left_live:
        if bh in ref_handles_used:
            continue
        c_in_other = L_slot(bh)
        if bh in right_ch_map:
            T[c_in_other, primary_local_bid, R_slot(bh)] = 1.0
        else:
            T[c_in_other, primary_local_bid, NO_INFO_OUT] = 1.0
    return T
```

- [ ] **Step 5: Update `build_site_tensors` to compute and pass `*_param_ty`**

Replace `build_site_tensors` in `_tensors.py`:

```python
def build_site_tensors(
    sites: list[NodeOccupancy],
    type_tags: list[int],
    live_binders_per_bond: list[list[BinderHandle]],
) -> list[np.ndarray]:
    """Top-level builder. Produces the list of N rank-3 site tensors.

    Bond i has dimension 1 + 8 * |live_binders_per_bond[i]| (one no_info
    slot plus 8 slots per live binder, encoding the binder's param_ty as
    a real bond DOF, per B's spec §5.2).
    """
    from ._channels import compute_channel_param_ty_per_bond
    pt_per_bond = compute_channel_param_ty_per_bond(sites, live_binders_per_bond)

    N = len(sites)
    bounded_live = [[]] + list(live_binders_per_bond) + [[]]
    bounded_pt = [[]] + pt_per_bond + [[]]

    tensors: list[np.ndarray] = []
    for k, occ in enumerate(sites):
        left_live = bounded_live[k]
        right_live = bounded_live[k + 1]
        left_pt = bounded_pt[k]
        right_pt = bounded_pt[k + 1]
        type_tag = type_tags[k]
        kind_idx, type_idx, value_idx = _local_kind_type_value(occ, type_tag)
        local_bid = _local_bid_for_kind(occ.kind, occ)
        bid_T = _bid_bond_tensor_at_site(
            site_idx=k, occ=occ, local_bid_value=local_bid,
            left_live=left_live, right_live=right_live,
            left_param_ty=left_pt, right_param_ty=right_pt,
        )
        tobl_idx = occ.tobl_tag
        site_T = _combine_factored_site(
            kind_idx=kind_idx, type_idx=type_idx, value_idx=value_idx,
            tobl_idx=tobl_idx, bid_tensor=bid_T,
        )
        tensors.append(site_T)
    return tensors
```

- [ ] **Step 6: Update `encoder.py` to populate `channel_param_ty_per_bond` in EncodingMeta**

Edit `src/qft_pcn/logic/encoder.py`. After computing `live = compute_live_binders(sites)`, add:

```python
    from ._channels import compute_channel_param_ty_per_bond
    channel_pt = compute_channel_param_ty_per_bond(sites, live)
```

Then in the `EncodingMeta(...)` construction:

```python
        channel_param_ty_per_bond=channel_pt,
```

- [ ] **Step 7: Run the new tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_channels.py::test_compute_channel_param_ty_per_bond_p1 src/qft_pcn/tests/test_logic_channels.py::test_compute_channel_param_ty_per_bond_p3 src/qft_pcn/tests/test_logic_tensors.py::test_bid_bond_extended_dim_is_one_plus_eight_L src/qft_pcn/tests/test_logic_tensors.py::test_bid_bond_only_one_slot_per_channel_populated -v`
Expected: 4 tests pass.

- [ ] **Step 8: Smoke check round-trip with encoder default chi_max=32**

Run: `.venv/bin/python -c "
from src.qft_pcn.logic.encoder import encode
from src.qft_pcn.logic.decoder import decode, ast_alpha_eq
from src.qft_pcn.logic.ast import parse, pretty

for src in [r'\x:Int. x', r'(\x:Int. x + 1)(2)', r'\f:Int->Int. \x:Int. f (f x)']:
    p = parse(src)
    state, meta = encode(p, N=32, chi_max=32)
    print(src)
    print('  bond dims:', state.bond_dimensions())
    print('  tobl_per_site (first 8):', meta.tobl_per_site[:8])
    print('  channel_param_ty (first 4):', meta.channel_param_ty_per_bond[:4])
    res = decode(state, meta)
    ok = ast_alpha_eq(res.ast, p)
    print('  round-trip OK?', ok)
    assert ok
"`

Expected: all three programs round-trip successfully.

- [ ] **Step 9: Commit**

```bash
git add src/qft_pcn/logic/_tensors.py src/qft_pcn/logic/_channels.py src/qft_pcn/logic/encoder.py src/qft_pcn/tests/test_logic_tensors.py src/qft_pcn/tests/test_logic_channels.py
git commit -m "$(cat <<'EOF'
feat(logic/_tensors): extend bid bond with per-channel param_ty (real DOF)

Per spec §5.2 of B: each live-binder channel on the bid bond is replaced
by an 8-slot orbit indexed by param_ty (one slot per TYPE_CUTOFF flat tag).
Bond dim grows from (1 + |L|) to (1 + 8|L|). For each binder, only the
slot matching its actual param_ty is populated; the other 7 are zero.

compute_channel_param_ty_per_bond produces per-bond param_ty tag lists
(parallel to live_binders_per_bond). The encoder populates the new
EncodingMeta.channel_param_ty_per_bond field for downstream sub-projects
(D's diagnostics + B's typing-Hamiltonian).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Bump encoder default `chi_max` from 16 to 32

**Files:**
- Modify: `src/qft_pcn/logic/encoder.py` (default chi_max=32).

The bid bond extension multiplies bond dim by ~8 per live binder. A's default chi_max=16 only supported |L|=1; B's default must accommodate |L|=2..3 typical for the acceptance tests.

- [ ] **Step 1: Edit `encoder.py`**

Edit `src/qft_pcn/logic/encoder.py`. Change the `encode` signature:

```python
def encode(ast: Node, N: int = 32, chi_max: int = 32
           ) -> tuple[MPS, EncodingMeta]:
```

- [ ] **Step 2: Verify the smoke check still passes**

Run: `.venv/bin/python -c "
from src.qft_pcn.logic.encoder import encode
from src.qft_pcn.logic.ast import parse
state, meta = encode(parse(r'(\x:Int. (\y:Int. x + y)(3))(4)'), N=32)
print('bond dims:', state.bond_dimensions())
print('norm:', state.norm_sq())
"`

Expected: norm ≈ 1.0; bond dims include slots up to 1 + 8*2 = 17 (max two live binders).

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/logic/encoder.py
git commit -m "$(cat <<'EOF'
feat(logic/encoder): bump default chi_max to 32 for the extended bid bond

The bid bond extension (per-channel param_ty as bond DOF) makes A's old
chi_max=16 default inadequate for any program with >= 1 live binder. New
default chi_max=32 handles up to 3 simultaneously live binders comfortably;
deeper nesting still raises TooManyBinders at MAX_BINDER_DEPTH=7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Update A's existing tests for the 5-species encoder

**Files:**
- Modify: many test files in `src/qft_pcn/tests/test_logic_*.py`.

After Tasks 1–7, A's old tests that expected D_LOCAL=8192, 4-species SPECIES, chi_max=16, etc. will fail. We update them now so the full suite is green before moving to Part 2.

- [ ] **Step 1: List the breakages**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_*.py 2>&1 | grep -E "^FAILED" | head -50`

Expected output identifies tests that hardcoded D_LOCAL=8192, SPECIES_NAMES tuple, SPECIES_DIMS tuple, or chi_max=16.

- [ ] **Step 2: Fix `test_logic_encoding.py`**

Edit `src/qft_pcn/tests/test_logic_encoding.py`. Find and update:

```python
def test_species_metadata():
    # was: assert SPECIES_NAMES == ("kind", "type", "bid", "value")
    #      assert SPECIES_DIMS == (8, 8, 8, 16)
    #      assert D_LOCAL == 8 * 8 * 8 * 16
    assert SPECIES_NAMES == ("kind", "type", "bid", "value", "tobl")
    assert SPECIES_DIMS == (8, 8, 8, 16, 8)
    assert D_LOCAL == 8 * 8 * 8 * 16 * 8


def test_species_objects_are_field_species():
    # was: assert len(SPECIES) == 4
    assert len(SPECIES) == 5
    # ... rest unchanged
```

Update `test_encoding_meta_fields` to include the new fields in the constructor call.

- [ ] **Step 3: Fix `test_logic_tensors.py`**

The tests `test_site_tensor_shapes`, `test_p1_lam_site_has_growing_right_bond`, `test_p1_lam_site_nonzero_amplitude_at_correct_basis_state`, `test_p1_var_site_reads_channel`, `test_pad_site_is_vacuum_amplitude`, `test_bond_dim_at_least_n_live_plus_one` use the old 4-species basis. Update them:

```python
# Old _basis_index call:
#   _basis_index(kind, type_, bid, value)
# becomes:
#   _basis_index(kind, type_, bid, value, tobl)
```

For shapes:
- LAM-site tensor shape: was `(1, 8192, 2)`. Now `(1, 65536, 9)` (with param_ty extension).
- VAR-site shape: was `(2, 8192, 1)`. Now `(9, 65536, 1)`.
- PAD shape: was `(1, 8192, 1)`. Now `(1, 65536, 1)`.

For amplitudes: the spec test `test_p1_lam_site_nonzero_amplitude_at_correct_basis_state` checked `T[0, idx, 1]`. With the extension, the correct right-bond slot for `(c=1, t=T_INT)` is `1 + 8*0 + 1 = 2`. Adjust the index:

```python
def test_p1_lam_site_nonzero_amplitude_at_correct_basis_state():
    # ... (build state) ...
    from src.qft_pcn.logic.encoding import (
        KIND_LAM, TYPE_ARR_II, BID_0, VALUE_NONE, TOBL_NONE, TYPE_INT,
    )
    idx = _basis_index(KIND_LAM, TYPE_ARR_II, BID_0, VALUE_NONE, TOBL_NONE)
    T = tensors[0]
    # The right-bond slot for channel 1 (Lam_x) with param_ty=T_INT is
    # 1 + 8*(1-1) + T_INT = 1 + 0 + 1 = 2.
    correct_slot = 1 + 8 * 0 + TYPE_INT
    assert abs(T[0, idx, correct_slot] - 1.0) < 1e-12
```

Update `test_p1_var_site_reads_channel` similarly:

```python
def test_p1_var_site_reads_channel():
    # The VAR site's left bond has slot 2 for the binder (Lam_x with T_INT).
    # We read FROM slot 2 (not slot 1 anymore).
    from src.qft_pcn.logic.encoding import (
        KIND_VAR, TYPE_INT, BID_0, VALUE_NONE, TOBL_INT,
    )
    # NOTE: tobl at VAR site is the inherited obligation. For the body of
    # \\x:Int. x, the body has obligation TOBL_INT (it's the LAM body).
    idx = _basis_index(KIND_VAR, TYPE_INT, BID_0, VALUE_NONE, TOBL_INT)
    T = tensors[1]
    # Left bond slot 2 (channel 1, T_INT) -> right bond slot 0 (no_info).
    assert abs(T[2, idx, 0] - 1.0) < 1e-12
```

Update `test_pad_site_is_vacuum_amplitude`:

```python
def test_pad_site_is_vacuum_amplitude():
    # Pad sites: (kind=PAD, type=NONE, bid=NONE, value=NONE, tobl=NONE)
    pad_idx = _basis_index(0, 0, 0, 0, 0)
    # ... rest unchanged ...
```

Update `test_bond_dim_at_least_n_live_plus_one`. Replace `n_live + 1` with `1 + 8 * n_live`:

```python
def test_bond_dim_at_least_n_live_plus_one():
    """For every bond, total bond dim >= 1 + 8 * |L_i| (the extended bid
    bond dim per spec B §5.2)."""
    ast = parse(r"(\x:Int. (\y:Int. x + y)(3))(4)")
    sites = serialize_preorder(ast, N=32)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    compute_tobl_tags(ast, sites)
    tensors = build_site_tensors(sites, types, live)
    for i in range(len(live)):
        n_live = len(live[i])
        expected_min = 1 + 8 * n_live
        assert tensors[i].shape[2] >= expected_min, (
            f"bond {i} right-bond dim {tensors[i].shape[2]} < "
            f"1 + 8*{n_live} = {expected_min}"
        )
```

- [ ] **Step 4: Fix `test_logic_decoder.py` / `test_logic_roundtrip.py`**

The decoder iterates over the local basis with `_decompose_basis_index` — that function in `src/qft_pcn/logic/decoder.py` needs to be updated for the 5-species basis. Edit `decoder.py`:

```python
def _decompose_basis_index(flat: int) -> tuple[int, int, int, int, int]:
    """Inverse of (k, t, b, v, o) -> flat index (leftmost slowest)."""
    o = flat % TOBL_CUTOFF
    flat //= TOBL_CUTOFF
    v = flat % VALUE_CUTOFF
    flat //= VALUE_CUTOFF
    b = flat % BID_CUTOFF
    flat //= BID_CUTOFF
    t = flat % TYPE_CUTOFF
    k = flat // TYPE_CUTOFF
    return (k, t, b, v, o)
```

Add the import at the top of `decoder.py`:

```python
from .encoding import (
    # ... existing ...
    TOBL_CUTOFF,
)
```

Update `_argmax_site_basis`:

```python
def _argmax_site_basis(state: MPS, site: int
                       ) -> tuple[int, int, int, int, int, float]:
    p = _site_marginal(state, site)
    flat = int(np.argmax(p))
    p_max = float(p[flat])
    k, t, b, v, o = _decompose_basis_index(flat)
    return (k, t, b, v, o, 1.0 - p_max)
```

Update the caller in `decode()`:

```python
    for k in range(meta.N):
        ki, ti, bi, vi, oi, residual = _argmax_site_basis(state, k)
        decoded_sites.append((ki, ti, bi, vi, oi))
        residual_acc = max(residual_acc, residual)
    # ...
    # In _parse_one: also unpack 5 indices.
    ki, ti, bi, vi, oi = decoded_sites[site_idx]
```

Update the PAD check at the end of decode():

```python
    while pos[0] < meta.N:
        ki, _, _, _, _ = decoded_sites[pos[0]]
        if ki != KIND_PAD:
            raise DecodeError(...)
        pos[0] += 1
```

- [ ] **Step 5: Fix `_gate_construction.py`**

Open `src/qft_pcn/logic/_gate_construction.py`. It builds states by applying gates on the 4-species local basis. Update it to use 5-species `_basis_index` (passing `tobl_idx=0` everywhere, since the gate-based path doesn't use tobl). This is a SHIM: we keep the gate-based construction working with `tobl=NONE` everywhere, which means it produces states that LACK the tobl obligations the analytic encoder writes. So the cross-check test in A will fail when comparing analytic vs. gate-based on a program where any site has tobl != NONE.

To handle this cleanly, **update the gate-based construction to ALSO write tobl values**. Read the file:

```bash
.venv/bin/python -c "import src.qft_pcn.logic._gate_construction; help(src.qft_pcn.logic._gate_construction)" | head -50
```

Then edit it to call `compute_tobl_tags` and apply the tobl-write gate at each site. The gate is a single-site gate that maps `|tobl=0⟩` to `|tobl=tag⟩`. With the local basis embedded, this is `embed_op(swap_0_with_tag, species_index=4, species_dims=(8,8,8,16,8))`.

This is a focused update; the gate-based cross-check is a sanity test and we keep it working.

- [ ] **Step 6: Update `test_logic_acceptance.py` and other test files**

Run the full logic suite and fix any remaining shape/chi_max assertions:

```bash
.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_*.py 2>&1 | grep -E "^FAILED" | head -30
```

Patch each failing test to use the new defaults. Common patterns:
- Hardcoded chi_max=16 in test args: leave the explicit value as-is (it's a valid choice) but skip if it would exceed capacity for the bigger bond dim. If the test passed chi_max=16 to a small program with 0-1 live binders, it still works (bond dim ≤ 9).
- Hardcoded `D_LOCAL == 8192`: change to `D_LOCAL == 65536`.
- Hardcoded SPECIES count == 4: change to 5.

- [ ] **Step 7: Run the full logic test suite**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_*.py -v 2>&1 | tail -30`

Expected: all tests pass. If anything fails with a test-side hardcoded constant, fix it.

- [ ] **Step 8: Run A's gate-construction cross-check specifically**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_gate_construction.py -v`

Expected: pass. If it fails, the gate-based path needs the tobl-write gates applied.

- [ ] **Step 9: Commit**

```bash
git add src/qft_pcn/tests/test_logic_encoding.py src/qft_pcn/tests/test_logic_tensors.py src/qft_pcn/tests/test_logic_decoder.py src/qft_pcn/tests/test_logic_acceptance.py src/qft_pcn/logic/decoder.py src/qft_pcn/logic/_gate_construction.py
git commit -m "$(cat <<'EOF'
fix(logic): update existing tests + decoder/gate-construction for 5-species basis

A's existing tests assumed D_LOCAL=8192, 4 species, chi_max=16. After B's
encoder extension these values change:
  D_LOCAL = 65536 (5 species: kind, type, bid, value, tobl)
  default chi_max = 32 (bid bond dim grows 8x per live binder)

Updates:
  - _basis_index and _decompose_basis_index now take 5 indices
  - decoder unpacks 5-tuples from the local basis
  - gate-construction shim applies a tobl-write gate after each site write
  - hard-coded constants in test_logic_*.py updated

All 100+ pre-existing tests in test_logic_*.py pass after these changes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: End-to-end encoder-extension smoke test

**Files:**
- Modify: `src/qft_pcn/tests/test_logic_typing_extension.py` (add an end-to-end test).

A single comprehensive test that exercises the entire encoder pipeline with B's extensions, verifying the `EncodingMeta` exposes all the new fields correctly populated.

- [ ] **Step 1: Write the test**

Append to `src/qft_pcn/tests/test_logic_typing_extension.py`:

```python
def test_end_to_end_encoder_extension_p2():
    """Spec §5: for (\\x:Int. x + 1)(2), the encoder writes correct
    tobl_per_site and channel_param_ty_per_bond.

    Layout: APP@0, LAM@1, BIN+@2, VAR_x@3, IntLit1@4, IntLit2@5, PAD...
    """
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.encoding import (
        TOBL_NONE, TOBL_INT, TYPE_INT,
    )

    state, meta = encode(parse(r"(\x:Int. x + 1)(2)"), N=8, chi_max=32)

    # tobl_per_site:
    #   site 0 (APP root): NONE.
    #   site 1 (LAM, fn of APP): NONE (APP doesn't impose).
    #   site 2 (BIN+, body of LAM, parent expects T_INT): TOBL_INT.
    #   site 3 (Var_x, BIN's lhs): TOBL_INT.
    #   site 4 (IntLit 1, BIN's rhs): TOBL_INT.
    #   site 5 (IntLit 2, APP's arg, parent expects fn.src = T_INT): TOBL_INT.
    #   site 6, 7 (PAD): NONE.
    assert meta.tobl_per_site == [
        TOBL_NONE, TOBL_NONE, TOBL_INT, TOBL_INT,
        TOBL_INT,  TOBL_INT,  TOBL_NONE, TOBL_NONE,
    ]

    # channel_param_ty_per_bond:
    # Bonds: 7 bonds (N-1).
    # Layout above; live binders per bond:
    #   bond 0 (APP@0 - LAM@1): no binders yet.
    #   bond 1 (LAM@1 - BIN@2): Lam_x live, param_ty = Int.
    #   bond 2 (BIN@2 - VAR@3): Lam_x live, param_ty = Int.
    #   bond 3 (VAR@3 - INT@4): Lam_x consumed at VAR; no binders.
    #   bond 4..6: no binders.
    expected_pt = [[], [TYPE_INT], [TYPE_INT], [], [], [], []]
    assert meta.channel_param_ty_per_bond == expected_pt

    # MPS norm = 1.
    assert abs(state.norm_sq() - 1.0) < 1e-10


def test_end_to_end_encoder_extension_p3():
    """For \\f:Int->Int. \\x:Int. f (f x), the two binders have different
    param_ty (T_ARR_II and T_INT) on the bonds where both are live."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.encoding import (
        TOBL_NONE, TOBL_INT, TOBL_ARR_II, TYPE_INT, TYPE_ARR_II,
    )

    state, meta = encode(parse(r"\f:Int->Int. \x:Int. f (f x)"), N=16,
                         chi_max=32)
    # Layout: LAM_f@0, LAM_x@1, APP@2, VAR_f@3, APP@4, VAR_f@5, VAR_x@6,
    #         PAD...
    # tobl:
    #   LAM_f root: NONE.
    #   LAM_x (body of LAM_f, parent expects dst(Int->Int->Int) = Int->Int):
    #     TOBL_ARR_II.
    #   APP@2 (body of LAM_x, parent expects Int): TOBL_INT.
    #   VAR_f@3 (fn of APP@2): NONE (APP doesn't impose).
    #   APP@4 (arg of APP@2, parent expects fn.src = Int): TOBL_INT.
    #   VAR_f@5 (fn of APP@4): NONE.
    #   VAR_x@6 (arg of APP@4, parent expects Int): TOBL_INT.
    assert meta.tobl_per_site[:7] == [
        TOBL_NONE, TOBL_ARR_II, TOBL_INT, TOBL_NONE,
        TOBL_INT, TOBL_NONE, TOBL_INT,
    ]
    # channel_param_ty_per_bond on the bond between LAM_x and APP@2:
    # both binders live (Lam_f with Int->Int, Lam_x with Int).
    assert meta.channel_param_ty_per_bond[1] == [TYPE_ARR_II, TYPE_INT]
```

- [ ] **Step 2: Run**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_extension.py -v`
Expected: 9 tests pass total (7 from Task 3 + 2 here).

- [ ] **Step 3: Run the entire logic test suite to confirm Part 1 is clean**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_*.py -v 2>&1 | tail -10`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/qft_pcn/tests/test_logic_typing_extension.py
git commit -m "$(cat <<'EOF'
test(logic/_typing_extension): end-to-end encoder-extension smoke tests

Two integration tests verifying that EncodingMeta after encode() exposes
correct tobl_per_site and channel_param_ty_per_bond for (a) a chained App+
Bin program (P2 shape) and (b) a nested-binder program with distinct
param_ty per binder (P3 shape).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

**End of Part 1.** Continue with Part 2 (`2026-05-21-typing-hamiltonian-part2.md`): TypingHamiltonian skeleton, factored-expectation helpers, per-rule term builders, term_energy, total_energy, residuals.

After Part 1 is done, the encoder produces 5-species MPS with:
- `meta.tobl_per_site[k]`: the obligation at site k.
- `meta.channel_param_ty_per_bond[b]`: the param_ty of each live binder on bond b (parallel to `live_binders_per_bond[b]`).
- Bid bond carries param_ty as a real bond DOF (1 + 8|L| dim).
- Tobl is a 5th species register at every site.

Part 2 builds the typing Hamiltonian on top of this encoded state.
