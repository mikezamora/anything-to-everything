# AST↔MPS Encoder Implementation Plan — Part 1 of 3 (Setup, AST, Encoding constants)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement sub-project A from `docs/superpowers/specs/2026-05-20-ast-mps-encoder-design.md`: a faithful AST ↔ MPS encoder/decoder where variable binding is realized as genuine entanglement on the `bid` bond.

**Architecture:** Four field species (`kind/type/bid/value`) per site of a fixed-N MPS. The encoder constructs `bid` bond tensors with a "live-binder channel" structure (§5.4 of spec). The decoder uses standard MPS conditional sampling for hole-bearing inputs and deterministic argmax for product inputs.

**Tech Stack:** Python 3.11, numpy 1.26, scipy 1.16, pytest. Built on existing `src/qft_pcn/qft/` machinery (MPS, Fock, Hamiltonian).

**Driving principles (from spec §1, non-negotiable):**

1. **Binder↔use is genuine bond entanglement, never a copied integer field.** If you find yourself writing `bid_value_at_var = binder_id_from_lookup_table`, stop. That is the forbidden shortcut.
2. **PAD is a vacuum state** — the Hamiltonian is a structural object on a fixed-N lattice.
3. **Reuse `src/qft_pcn/qft/` machinery as-is** — do not rebuild MPS/Fock.
4. **Encoder is total on well-scoped ASTs**, raises on out-of-bounds. No lossy fallbacks.
5. **Decoder uses true MPS conditional sampling**, not per-site independent marginals.
6. **Cross-checks (analytic vs gate construction) must agree** to fidelity > 1 - 1e-8.

**Reference:** The spec at `docs/superpowers/specs/2026-05-20-ast-mps-encoder-design.md` is the authoritative contract. When in doubt, re-read it. If something here disagrees with the spec, the spec wins.

**Test command throughout:** Run pytest via the project venv: `.venv/bin/python -m pytest <path> -v`.

**This document is Part 1 of 3.** It covers tasks 1–7 (setup, AST, encoding constants, MPS.inner). Part 2 covers tasks 8–17 (the encoder itself). Part 3 covers tasks 18–30 (decoder, sample, hole support, gate construction, all tests, final verification).

---

## Task 1: Install pytest in the project venv

**Files:**
- Modify: `pyproject.toml` (add pytest to optional `[project.optional-dependencies]` or equivalent dev group).

The existing venv has numpy and scipy but no pytest. Add it via uv if the workspace allows; otherwise pip-install directly into the venv.

- [ ] **Step 1: Check current state**

Run: `.venv/bin/python -c "import pytest" 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'pytest'`.

- [ ] **Step 2: Install pytest directly into the venv**

Run: `.venv/bin/python -m pip install pytest pytest-timeout`
Expected: install summary ending in "Successfully installed pytest-<ver> pytest-timeout-<ver>".

(Rationale for direct pip: `uv sync` is known to fail on this checkout because `modelscope` has filesystem issues with hardlinking. Direct pip avoids the full workspace resync.)

- [ ] **Step 3: Verify**

Run: `.venv/bin/python -m pytest --version`
Expected: `pytest <version>`.

- [ ] **Step 4: Run the existing test suite to establish baseline**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/ -v --ignore=src/qft_pcn/tests/test_quantum.py 2>&1 | tail -30`

(Ignoring test_quantum.py because qiskit isn't installed and isn't needed for this sub-project.)

Expected: All tests pass (the `test_qft.py`, `test_multifield.py`, `test_qft_pcn.py` ones — 21 tests).

- [ ] **Step 5: Do NOT commit pyproject.toml changes**

We won't modify `pyproject.toml` for this — the install is local. Just verify `git status` shows no new tracked changes from this task.

Run: `git status --short pyproject.toml`
Expected: no output (pyproject.toml unchanged).

---

## Task 2: Add MPS.inner to qft/mps.py

**Files:**
- Modify: `src/qft_pcn/qft/mps.py` (add `inner` method).
- Modify: `src/qft_pcn/tests/test_qft.py` (add unit test).

`MPS.inner(other) -> complex` computes ⟨self|other⟩ by sweeping environment tensors. The cross-check test in sub-project A needs this.

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_qft.py`:

```python
def test_mps_inner_product_self_equals_norm_sq():
    """<psi|psi> == ||psi||^2."""
    psi = MPS.number_states([1, 0, 2, 1], d=4)
    inner_self = psi.inner(psi)
    norm_sq = psi.norm_sq()
    assert abs(inner_self - norm_sq) < 1e-12


def test_mps_inner_product_orthogonal_number_states():
    """<n_a | n_b> = 0 for distinct occupation patterns."""
    a = MPS.number_states([1, 0, 0, 0], d=3)
    b = MPS.number_states([0, 1, 0, 0], d=3)
    assert abs(a.inner(b)) < 1e-12
    assert abs(b.inner(a)) < 1e-12


def test_mps_inner_product_after_normalization():
    """After normalize(), <psi|psi> == 1."""
    psi = MPS.number_states([2, 1, 0, 1], d=3).normalize()
    assert abs(psi.inner(psi) - 1.0) < 1e-10
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_qft.py::test_mps_inner_product_self_equals_norm_sq -v`
Expected: FAIL with `AttributeError: 'MPS' object has no attribute 'inner'`.

- [ ] **Step 3: Implement MPS.inner**

In `src/qft_pcn/qft/mps.py`, add this method to the `MPS` class right after `norm_sq`:

```python
def inner(self, other: "MPS") -> complex:
    """<self | other> contracted by sweeping environment tensors.

    Self is the bra (conjugated), other is the ket. Both MPSes must have
    the same length and per-site dimension.
    """
    if other.N != self.N:
        raise ValueError(f"MPS length mismatch: {self.N} vs {other.N}")
    env = np.ones((1, 1), dtype=complex)
    for k in range(self.N):
        bra = self.tensors[k].conj()    # (chi_l_self, d, chi_r_self)
        ket = other.tensors[k]          # (chi_l_other, d, chi_r_other)
        if bra.shape[1] != ket.shape[1]:
            raise ValueError(f"site {k} dim mismatch: "
                             f"{bra.shape[1]} vs {ket.shape[1]}")
        # env has shape (chi_l_self, chi_l_other) -> (chi_r_self, chi_r_other)
        env = np.einsum('ij,isk,jsl->kl', env, bra, ket)
    return complex(env[0, 0])
```

- [ ] **Step 4: Run all three new tests to verify they pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_qft.py -k "test_mps_inner" -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full qft test file to confirm no regressions**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_qft.py -v 2>&1 | tail -10`
Expected: all tests pass (17 = 14 original + 3 new).

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/qft/mps.py src/qft_pcn/tests/test_qft.py
git commit -m "$(cat <<'EOF'
feat(qft/mps): add MPS.inner for bra-ket overlap

Implements <self|other> by sweeping environment tensors. Needed by
the AST<->MPS encoder cross-check test (analytic vs gate-based
construction fidelity).

Three unit tests: self-inner equals norm_sq, orthogonal number states
give zero, normalized states give 1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Create the logic package skeleton

**Files:**
- Create: `src/qft_pcn/logic/__init__.py` (empty for now; will be populated in Task 30).

- [ ] **Step 1: Create the directory and empty init**

Run: `mkdir -p src/qft_pcn/logic && touch src/qft_pcn/logic/__init__.py`

Verify: `ls src/qft_pcn/logic/`
Expected: `__init__.py`.

- [ ] **Step 2: Stage the empty init**

Run: `git add src/qft_pcn/logic/__init__.py`

Expected: file staged but no commit yet — we'll commit once there's something to commit alongside.

---

## Task 4: AST node and type data classes (`logic/ast.py`, part 1: data)

**Files:**
- Create: `src/qft_pcn/logic/ast.py`.

This task introduces the AST data classes only (no parser, no pretty-printer — those come in Tasks 5 and 6). Per spec §3.

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_logic_ast.py`:

```python
"""Tests for the AST data classes in logic/ast.py."""

from __future__ import annotations

import pytest

from src.qft_pcn.logic.ast import (
    Node, Var, Lam, App, IntLit, BoolLit, If, Bin,
    Ty, TInt, TBool, TArrow,
)


def test_var_construction():
    v = Var(name="x")
    assert v.name == "x"


def test_lam_construction_holds_param_and_body():
    lam = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    assert lam.param == "x"
    assert isinstance(lam.param_ty, TInt)
    assert isinstance(lam.body, Var)
    assert lam.body.name == "x"


def test_app_holds_fn_and_arg():
    app = App(fn=Var(name="f"), arg=IntLit(val=3))
    assert isinstance(app.fn, Var)
    assert app.arg.val == 3


def test_intlit_and_boollit():
    assert IntLit(val=5).val == 5
    assert BoolLit(val=True).val is True
    assert BoolLit(val=False).val is False


def test_if_holds_three_branches():
    e = If(cond=BoolLit(val=True), then_b=IntLit(val=1), else_b=IntLit(val=0))
    assert e.cond.val is True
    assert e.then_b.val == 1
    assert e.else_b.val == 0


def test_bin_holds_op_and_operands():
    b = Bin(op="+", lhs=IntLit(val=2), rhs=IntLit(val=3))
    assert b.op == "+"
    assert b.lhs.val == 2 and b.rhs.val == 3


def test_bin_op_must_be_supported():
    # Out-of-grammar op should be a runtime error from __post_init__.
    with pytest.raises(ValueError, match="op must be one of"):
        Bin(op="**", lhs=IntLit(val=1), rhs=IntLit(val=1))


def test_arrow_type_construction():
    t = TArrow(src=TInt(), dst=TBool())
    assert isinstance(t.src, TInt)
    assert isinstance(t.dst, TBool)


def test_ty_equality_by_structure():
    assert TInt() == TInt()
    assert TBool() == TBool()
    assert TArrow(src=TInt(), dst=TInt()) == TArrow(src=TInt(), dst=TInt())
    assert TArrow(src=TInt(), dst=TInt()) != TArrow(src=TBool(), dst=TInt())


def test_node_equality_by_structure():
    # Structural equality on nodes (modulo names, which DO matter for raw AST).
    assert Var(name="x") == Var(name="x")
    assert Var(name="x") != Var(name="y")
    assert IntLit(val=3) == IntLit(val=3)
    assert IntLit(val=3) != IntLit(val=4)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_ast.py -v`
Expected: collection error — module `src.qft_pcn.logic.ast` does not exist.

- [ ] **Step 3: Implement the AST data classes**

Create `src/qft_pcn/logic/ast.py`:

```python
"""AST data classes for the STLC + ints + bools + if + arith/compare surface.

See docs/superpowers/specs/2026-05-20-ast-mps-encoder-design.md, §3.

The classes here are pure data; the parser and pretty-printer live in
ast.py too (added in later tasks). Encoder/decoder are separate modules.
"""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_BIN_OPS: tuple[str, ...] = ("+", "-", "*", "<", "==")


# ---- types ----------------------------------------------------------------


@dataclass(frozen=True)
class Ty:
    """Base class. Use TInt, TBool, TArrow."""


@dataclass(frozen=True)
class TInt(Ty):
    pass


@dataclass(frozen=True)
class TBool(Ty):
    pass


@dataclass(frozen=True)
class TArrow(Ty):
    src: Ty
    dst: Ty


# ---- expression nodes -----------------------------------------------------


@dataclass
class Node:
    """Base class for AST nodes."""


@dataclass
class Var(Node):
    name: str


@dataclass
class Lam(Node):
    param: str
    param_ty: Ty
    body: Node


@dataclass
class App(Node):
    fn: Node
    arg: Node


@dataclass
class IntLit(Node):
    val: int


@dataclass
class BoolLit(Node):
    val: bool


@dataclass
class If(Node):
    cond: Node
    then_b: Node
    else_b: Node


@dataclass
class Bin(Node):
    op: str
    lhs: Node
    rhs: Node

    def __post_init__(self) -> None:
        if self.op not in SUPPORTED_BIN_OPS:
            raise ValueError(
                f"Bin op must be one of {SUPPORTED_BIN_OPS}, got {self.op!r}"
            )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_ast.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/__init__.py src/qft_pcn/logic/ast.py src/qft_pcn/tests/test_logic_ast.py
git commit -m "$(cat <<'EOF'
feat(logic): AST and Ty data classes for the STLC surface

Covers Var, Lam, App, IntLit, BoolLit, If, Bin and types TInt, TBool,
TArrow. Bin ops are restricted to +, -, *, <, == (spec §3). Parser and
pretty-printer come in subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: AST parser

**Files:**
- Modify: `src/qft_pcn/logic/ast.py` (add tokenizer + parser).
- Modify: `src/qft_pcn/tests/test_logic_ast.py` (add parser tests).

A recursive-descent parser for the surface syntax in spec §3. Concrete syntax used by the test suite:

```
λx:T. body                  written  \x:T. body
T1 -> T2                    arrow type
e1 e2                       application (left-assoc, higher precedence than ops)
e1 + e2                     binary ops (left-assoc)
if e1 then e2 else e3       conditional
123                         int literal
true / false                bool literals
x                           variable
( e )                       parenthesized
```

- [ ] **Step 1: Write the failing tests**

Append to `src/qft_pcn/tests/test_logic_ast.py`:

```python
from src.qft_pcn.logic.ast import parse


def test_parse_var():
    assert parse("x") == Var(name="x")


def test_parse_intlit():
    assert parse("42") == IntLit(val=42)
    assert parse("0") == IntLit(val=0)


def test_parse_bool():
    assert parse("true") == BoolLit(val=True)
    assert parse("false") == BoolLit(val=False)


def test_parse_lambda_identity():
    expected = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    assert parse(r"\x:Int. x") == expected


def test_parse_application_left_assoc():
    # f x y = (f x) y
    expected = App(fn=App(fn=Var("f"), arg=Var("x")), arg=Var("y"))
    assert parse("f x y") == expected


def test_parse_addition():
    expected = Bin(op="+", lhs=IntLit(val=2), rhs=IntLit(val=3))
    assert parse("2 + 3") == expected


def test_parse_if():
    expected = If(
        cond=Bin(op="<", lhs=IntLit(val=1), rhs=IntLit(val=2)),
        then_b=IntLit(val=10),
        else_b=IntLit(val=20),
    )
    assert parse("if 1 < 2 then 10 else 20") == expected


def test_parse_arrow_type():
    # \\f:Int->Int. f
    expected = Lam(
        param="f",
        param_ty=TArrow(src=TInt(), dst=TInt()),
        body=Var("f"),
    )
    assert parse(r"\f:Int->Int. f") == expected


def test_parse_arrow_right_assoc():
    # Int -> Int -> Bool  ==  Int -> (Int -> Bool)
    expected = Lam(
        param="g",
        param_ty=TArrow(src=TInt(), dst=TArrow(src=TInt(), dst=TBool())),
        body=Var("g"),
    )
    assert parse(r"\g:Int->Int->Bool. g") == expected


def test_parse_p5_complex_program():
    src = r"(\x:Int. (\y:Int. x + y)(3))(4)"
    p = parse(src)
    # outer App
    assert isinstance(p, App)
    assert isinstance(p.arg, IntLit) and p.arg.val == 4
    # outer Lam
    assert isinstance(p.fn, Lam)
    assert p.fn.param == "x"
    assert isinstance(p.fn.param_ty, TInt)
    # inner App
    inner_app = p.fn.body
    assert isinstance(inner_app, App)
    assert isinstance(inner_app.arg, IntLit) and inner_app.arg.val == 3
    # inner Lam
    assert isinstance(inner_app.fn, Lam)
    assert inner_app.fn.param == "y"
    # x + y
    body = inner_app.fn.body
    assert isinstance(body, Bin) and body.op == "+"
    assert body.lhs == Var(name="x") and body.rhs == Var(name="y")


def test_parse_rejects_unknown_op():
    with pytest.raises(ValueError, match="unexpected token|unexpected character"):
        parse("2 ** 3")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_ast.py -k "parse" -v`
Expected: ImportError on `parse`.

- [ ] **Step 3: Implement the parser**

Append to `src/qft_pcn/logic/ast.py`:

```python
# ---- tokenizer + parser ---------------------------------------------------

import re
from typing import Iterator


_TOKEN_RE = re.compile(
    r"\s+"
    r"|(?P<lambda>\\)"
    r"|(?P<arrow>->)"
    r"|(?P<lt>(?<![\w<>=])<(?![=]))"     # < but not <=
    r"|(?P<eq>==)"
    r"|(?P<colon>:)"
    r"|(?P<dot>\.)"
    r"|(?P<lp>\()"
    r"|(?P<rp>\))"
    r"|(?P<plus>\+)"
    r"|(?P<minus>-)"
    r"|(?P<times>\*)"
    r"|(?P<int>\d+)"
    r"|(?P<ident>[A-Za-z_][A-Za-z_0-9]*)"
)


class _Tok:
    __slots__ = ("kind", "value")

    def __init__(self, kind: str, value: str):
        self.kind, self.value = kind, value

    def __repr__(self) -> str:
        return f"Tok({self.kind!r}, {self.value!r})"


def _tokenize(src: str) -> list[_Tok]:
    pos = 0
    out: list[_Tok] = []
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise ValueError(
                f"unexpected character at position {pos}: {src[pos]!r}"
            )
        kind = m.lastgroup
        if kind is None:
            # whitespace match (no named group)
            pos = m.end()
            continue
        value = m.group(kind)
        if value == "true" or value == "false":
            kind = "bool"
        elif value in ("if", "then", "else", "Int", "Bool"):
            kind = "keyword"
        out.append(_Tok(kind, value))
        pos = m.end()
    out.append(_Tok("eof", ""))
    return out


class _Parser:
    def __init__(self, toks: list[_Tok]):
        self.toks = toks
        self.i = 0

    def peek(self) -> _Tok:
        return self.toks[self.i]

    def eat(self, kind: str, value: str | None = None) -> _Tok:
        t = self.peek()
        if t.kind != kind or (value is not None and t.value != value):
            expected = f"{kind}={value!r}" if value is not None else kind
            raise ValueError(
                f"unexpected token {t.kind}={t.value!r}; expected {expected}"
            )
        self.i += 1
        return t

    # type ::= atom_ty ("->" type)?     (right-assoc)
    # atom_ty ::= "Int" | "Bool" | "(" type ")"
    def parse_ty(self) -> Ty:
        left = self.parse_ty_atom()
        if self.peek().kind == "arrow":
            self.eat("arrow")
            right = self.parse_ty()
            return TArrow(src=left, dst=right)
        return left

    def parse_ty_atom(self) -> Ty:
        t = self.peek()
        if t.kind == "keyword" and t.value == "Int":
            self.eat("keyword", "Int")
            return TInt()
        if t.kind == "keyword" and t.value == "Bool":
            self.eat("keyword", "Bool")
            return TBool()
        if t.kind == "lp":
            self.eat("lp")
            ty = self.parse_ty()
            self.eat("rp")
            return ty
        raise ValueError(f"expected type, got {t.kind}={t.value!r}")

    # Expression grammar (precedences low to high):
    #   expr     ::= if_expr | cmp_expr
    #   if_expr  ::= "if" expr "then" expr "else" expr
    #   cmp_expr ::= add_expr (("<"|"==") add_expr)*    (left-assoc)
    #   add_expr ::= mul_expr (("+"|"-") mul_expr)*     (left-assoc)
    #   mul_expr ::= app_expr ("*" app_expr)*           (left-assoc)
    #   app_expr ::= atom atom*                         (left-assoc app)
    #   atom     ::= var | int | bool | lambda | "(" expr ")"
    #   lambda   ::= "\" ident ":" ty "." expr
    def parse_expr(self) -> Node:
        t = self.peek()
        if t.kind == "keyword" and t.value == "if":
            return self.parse_if()
        return self.parse_cmp()

    def parse_if(self) -> Node:
        self.eat("keyword", "if")
        c = self.parse_expr()
        self.eat("keyword", "then")
        a = self.parse_expr()
        self.eat("keyword", "else")
        b = self.parse_expr()
        return If(cond=c, then_b=a, else_b=b)

    def parse_cmp(self) -> Node:
        left = self.parse_add()
        while self.peek().kind in ("lt", "eq"):
            op = "<" if self.peek().kind == "lt" else "=="
            self.i += 1
            right = self.parse_add()
            left = Bin(op=op, lhs=left, rhs=right)
        return left

    def parse_add(self) -> Node:
        left = self.parse_mul()
        while self.peek().kind in ("plus", "minus"):
            op = "+" if self.peek().kind == "plus" else "-"
            self.i += 1
            right = self.parse_mul()
            left = Bin(op=op, lhs=left, rhs=right)
        return left

    def parse_mul(self) -> Node:
        left = self.parse_app()
        while self.peek().kind == "times":
            self.i += 1
            right = self.parse_app()
            left = Bin(op="*", lhs=left, rhs=right)
        return left

    def parse_app(self) -> Node:
        head = self.parse_atom()
        while self.peek().kind in ("ident", "int", "bool", "lp"):
            arg = self.parse_atom()
            head = App(fn=head, arg=arg)
        return head

    def parse_atom(self) -> Node:
        t = self.peek()
        if t.kind == "ident":
            self.i += 1
            return Var(name=t.value)
        if t.kind == "int":
            self.i += 1
            return IntLit(val=int(t.value))
        if t.kind == "bool":
            self.i += 1
            return BoolLit(val=(t.value == "true"))
        if t.kind == "lambda":
            return self.parse_lambda()
        if t.kind == "lp":
            self.eat("lp")
            e = self.parse_expr()
            self.eat("rp")
            return e
        raise ValueError(
            f"unexpected token at expression position: {t.kind}={t.value!r}"
        )

    def parse_lambda(self) -> Node:
        self.eat("lambda")
        name_tok = self.eat("ident")
        self.eat("colon")
        ty = self.parse_ty()
        self.eat("dot")
        body = self.parse_expr()
        return Lam(param=name_tok.value, param_ty=ty, body=body)


def parse(src: str) -> Node:
    """Parse the surface syntax (spec §3) into an AST."""
    toks = _tokenize(src)
    p = _Parser(toks)
    expr = p.parse_expr()
    if p.peek().kind != "eof":
        raise ValueError(
            f"unexpected trailing token: {p.peek().kind}={p.peek().value!r}"
        )
    return expr
```

- [ ] **Step 4: Run parser tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_ast.py -k "parse" -v`
Expected: all 11 parser tests pass.

- [ ] **Step 5: Run the full ast test file**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_ast.py -v 2>&1 | tail -10`
Expected: 21 tests passed (10 from Task 4 + 11 here).

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/ast.py src/qft_pcn/tests/test_logic_ast.py
git commit -m "$(cat <<'EOF'
feat(logic/ast): recursive-descent parser for the surface syntax

Parses identifiers, int and bool literals, lambdas with type annotations,
application (left-assoc), binary ops (+, -, *, <, ==), if/then/else,
and arrow types (right-assoc). Spec §3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: AST pretty-printer

**Files:**
- Modify: `src/qft_pcn/logic/ast.py` (add `pretty`).
- Modify: `src/qft_pcn/tests/test_logic_ast.py` (add pretty round-trip tests).

- [ ] **Step 1: Write the failing tests**

Append to `src/qft_pcn/tests/test_logic_ast.py`:

```python
from src.qft_pcn.logic.ast import pretty


def test_pretty_roundtrip_simple():
    src = r"\x:Int. x"
    assert parse(pretty(parse(src))) == parse(src)


def test_pretty_roundtrip_p2():
    src = r"(\x:Int. x + 1)(2)"
    assert parse(pretty(parse(src))) == parse(src)


def test_pretty_roundtrip_p3():
    src = r"\f:Int->Int. \x:Int. f (f x)"
    assert parse(pretty(parse(src))) == parse(src)


def test_pretty_roundtrip_p4():
    src = r"if 1 < 2 then ((\x:Bool. x)(true)) else false"
    assert parse(pretty(parse(src))) == parse(src)


def test_pretty_roundtrip_p5():
    src = r"(\x:Int. (\y:Int. x + y)(3))(4)"
    assert parse(pretty(parse(src))) == parse(src)
```

- [ ] **Step 2: Verify they fail**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_ast.py -k "pretty" -v`
Expected: ImportError on `pretty`.

- [ ] **Step 3: Implement pretty()**

Append to `src/qft_pcn/logic/ast.py`:

```python
# ---- pretty printer -------------------------------------------------------


def _pretty_ty(t: Ty) -> str:
    if isinstance(t, TInt):
        return "Int"
    if isinstance(t, TBool):
        return "Bool"
    if isinstance(t, TArrow):
        # Right-associative, so don't paren the right side.
        left = _pretty_ty(t.src)
        if isinstance(t.src, TArrow):
            left = f"({left})"
        return f"{left}->{_pretty_ty(t.dst)}"
    raise TypeError(f"unknown Ty: {t!r}")


# Precedences: atom > app > mul > add > cmp > if  (higher binds tighter)
_PREC_ATOM = 100
_PREC_APP = 60
_PREC_MUL = 50
_PREC_ADD = 40
_PREC_CMP = 30
_PREC_IF = 10
_PREC_LAM = 5


def _pretty_node(node: Node, ctx_prec: int) -> str:
    if isinstance(node, Var):
        return node.name
    if isinstance(node, IntLit):
        return str(node.val)
    if isinstance(node, BoolLit):
        return "true" if node.val else "false"
    if isinstance(node, Lam):
        s = f"\\{node.param}:{_pretty_ty(node.param_ty)}. " \
            f"{_pretty_node(node.body, _PREC_LAM)}"
        if ctx_prec > _PREC_LAM:
            s = f"({s})"
        return s
    if isinstance(node, App):
        s = f"{_pretty_node(node.fn, _PREC_APP)} " \
            f"{_pretty_node(node.arg, _PREC_APP + 1)}"
        if ctx_prec > _PREC_APP:
            s = f"({s})"
        return s
    if isinstance(node, If):
        s = (f"if {_pretty_node(node.cond, _PREC_IF)} "
             f"then {_pretty_node(node.then_b, _PREC_IF)} "
             f"else {_pretty_node(node.else_b, _PREC_IF)}")
        if ctx_prec > _PREC_IF:
            s = f"({s})"
        return s
    if isinstance(node, Bin):
        op = node.op
        if op in ("+", "-"):
            inner_prec = _PREC_ADD
        elif op == "*":
            inner_prec = _PREC_MUL
        else:  # < or ==
            inner_prec = _PREC_CMP
        s = (f"{_pretty_node(node.lhs, inner_prec)} {op} "
             f"{_pretty_node(node.rhs, inner_prec + 1)}")
        if ctx_prec > inner_prec:
            s = f"({s})"
        return s
    raise TypeError(f"unknown Node: {node!r}")


def pretty(node: Node) -> str:
    """Render an AST back to surface syntax that re-parses to the same AST."""
    return _pretty_node(node, _PREC_LAM)
```

- [ ] **Step 4: Run pretty tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_ast.py -k "pretty" -v`
Expected: 5 passed.

- [ ] **Step 5: Run all ast tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_ast.py -v 2>&1 | tail -10`
Expected: 26 passed.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/ast.py src/qft_pcn/tests/test_logic_ast.py
git commit -m "$(cat <<'EOF'
feat(logic/ast): pretty-printer with precedence-aware parentheses

Round-trips all five §7.1 demo programs through parse->pretty->parse to
structural equality. Used by tests and debugging output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Encoding constants, EncodingMeta, BinderHandle, exceptions

**Files:**
- Create: `src/qft_pcn/logic/encoding.py`.
- Create: `src/qft_pcn/tests/test_logic_encoding.py`.

This wires up the field-species constants from spec §4.

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_logic_encoding.py`:

```python
"""Tests for logic/encoding.py constants and metadata classes."""

from __future__ import annotations

import pytest

from src.qft_pcn.logic.encoding import (
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL, KIND_IF,
    KIND_BIN, KIND_CUTOFF,
    TYPE_NONE, TYPE_INT, TYPE_BOOL, TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI,
    TYPE_ARR_BB, TYPE_ARR_NESTED, TYPE_CUTOFF,
    BID_NONE, BID_0, BID_1, BID_2, BID_3, BID_4, BID_5, BID_6, BID_CUTOFF,
    VALUE_NONE, VALUE_FALSE, VALUE_TRUE,
    VALUE_PLUS, VALUE_MINUS, VALUE_TIMES, VALUE_LT, VALUE_EQ, VALUE_CUTOFF,
    INT_LIT_OFFSET, INT_LIT_MIN, INT_LIT_MAX,
    SPECIES, SPECIES_NAMES, SPECIES_DIMS, D_LOCAL,
    BinderHandle, EncodingMeta,
    EncodingError, EncodingTooLarge, TooManyBinders,
    IntLiteralOutOfRange, IllScopedVar, UnsupportedNode, DecodeError,
)


def test_kind_cutoff_and_basis():
    assert KIND_PAD == 0
    assert KIND_VAR == 1
    assert KIND_LAM == 2
    assert KIND_APP == 3
    assert KIND_INT == 4
    assert KIND_BOOL == 5
    assert KIND_IF == 6
    assert KIND_BIN == 7
    assert KIND_CUTOFF == 8


def test_type_cutoff_and_basis():
    assert TYPE_NONE == 0
    assert TYPE_INT == 1
    assert TYPE_BOOL == 2
    assert TYPE_ARR_II == 3
    assert TYPE_ARR_IB == 4
    assert TYPE_ARR_BI == 5
    assert TYPE_ARR_BB == 6
    assert TYPE_ARR_NESTED == 7
    assert TYPE_CUTOFF == 8


def test_bid_cutoff_and_basis():
    assert BID_NONE == 0
    assert BID_0 == 1
    assert BID_1 == 2
    assert BID_2 == 3
    assert BID_3 == 4
    assert BID_4 == 5
    assert BID_5 == 6
    assert BID_6 == 7
    assert BID_CUTOFF == 8


def test_value_cutoff_and_basis():
    assert VALUE_NONE == 0
    # int 0..15 are at value indices 7..15 via INT_LIT_OFFSET=7
    assert VALUE_FALSE == 0
    assert VALUE_TRUE == 1
    assert VALUE_PLUS == 2
    assert VALUE_MINUS == 3
    assert VALUE_TIMES == 4
    assert VALUE_LT == 5
    assert VALUE_EQ == 6
    # 7..15 used for int literals (offset 7 mapping [-7, 8] -> [0, 15])
    assert INT_LIT_OFFSET == 7
    assert INT_LIT_MIN == -7
    assert INT_LIT_MAX == 8
    assert VALUE_CUTOFF == 16


def test_species_metadata():
    assert SPECIES_NAMES == ("kind", "type", "bid", "value")
    assert SPECIES_DIMS == (8, 8, 8, 16)
    assert D_LOCAL == 8 * 8 * 8 * 16


def test_species_objects_are_field_species():
    from src.qft_pcn.qft.hamiltonian import FieldSpecies
    assert len(SPECIES) == 4
    for s in SPECIES:
        assert isinstance(s, FieldSpecies)
    assert [s.name for s in SPECIES] == list(SPECIES_NAMES)
    assert [s.cutoff for s in SPECIES] == list(SPECIES_DIMS)


def test_binder_handle_frozen_and_hashable():
    bh = BinderHandle(lam_site=3, depth_at_lam=2)
    assert bh.lam_site == 3
    assert bh.depth_at_lam == 2
    # Frozen -> hashable.
    s = {bh}
    assert bh in s


def test_encoding_meta_fields():
    meta = EncodingMeta(
        N=32,
        chi_max=16,
        field_dims={"kind": 8, "type": 8, "bid": 8, "value": 16},
        species=list(SPECIES),
        nested_type_index={},
        site_to_ast_path={},
        live_binders_per_bond=[],
    )
    assert meta.N == 32
    assert meta.chi_max == 16


def test_exception_types_inherit_from_encoding_error():
    for cls in (EncodingTooLarge, TooManyBinders, IntLiteralOutOfRange,
                IllScopedVar, UnsupportedNode, DecodeError):
        assert issubclass(cls, EncodingError)


def test_too_large_error_message():
    with pytest.raises(EncodingTooLarge, match="AST has 50 nodes but N=32"):
        raise EncodingTooLarge(n_nodes=50, N=32)


def test_too_many_binders_message():
    with pytest.raises(TooManyBinders, match="scope nesting 8 exceeds bid cutoff 8"):
        raise TooManyBinders(depth=8, cutoff=8)


def test_int_literal_out_of_range_message():
    with pytest.raises(IntLiteralOutOfRange, match=r"IntLit\(9\) outside"):
        raise IntLiteralOutOfRange(n=9)


def test_ill_scoped_var_message():
    with pytest.raises(IllScopedVar, match="Var\\('x'\\) not in lexical scope"):
        raise IllScopedVar(name="x")
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_encoding.py -v`
Expected: ImportError on `logic.encoding`.

- [ ] **Step 3: Implement encoding.py**

Create `src/qft_pcn/logic/encoding.py`:

```python
"""Field-species constants, EncodingMeta, BinderHandle, and exceptions for the
AST <-> MPS encoder.

See docs/superpowers/specs/2026-05-20-ast-mps-encoder-design.md, §4, §5.1, §9.

Per-site local Hilbert space is the tensor product of four registers
(species): kind (8) x type (8) x bid (8) x value (16) = 8192. Basis ordering
within each species follows `embed_op` in qft/fock.py (leftmost species
changes slowest).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.qft_pcn.qft.hamiltonian import FieldSpecies


# ---- kind register --------------------------------------------------------

KIND_PAD = 0
KIND_VAR = 1
KIND_LAM = 2
KIND_APP = 3
KIND_INT = 4
KIND_BOOL = 5
KIND_IF = 6
KIND_BIN = 7
KIND_CUTOFF = 8

KIND_NAMES = ("PAD", "VAR", "LAM", "APP", "INT", "BOOL", "IF", "BIN")


# ---- type register --------------------------------------------------------

TYPE_NONE = 0
TYPE_INT = 1
TYPE_BOOL = 2
TYPE_ARR_II = 3      # Int -> Int
TYPE_ARR_IB = 4      # Int -> Bool
TYPE_ARR_BI = 5      # Bool -> Int
TYPE_ARR_BB = 6      # Bool -> Bool
TYPE_ARR_NESTED = 7  # any higher-order arrow; full Ty stored in meta side table
TYPE_CUTOFF = 8


# ---- binder-id register ---------------------------------------------------

BID_NONE = 0
BID_0 = 1
BID_1 = 2
BID_2 = 3
BID_3 = 4
BID_4 = 5
BID_5 = 6
BID_6 = 7
BID_CUTOFF = 8

# Maximum nested binders supported (BID_0..BID_6 = 7 binders).
MAX_BINDER_DEPTH = BID_CUTOFF - 1


# ---- value register -------------------------------------------------------
# Overloaded by kind. See spec §4.4.

VALUE_NONE = 0      # default for PAD / VAR / LAM / APP / IF
VALUE_FALSE = 0     # for kind == BOOL
VALUE_TRUE = 1      # for kind == BOOL
VALUE_PLUS = 2      # for kind == BIN
VALUE_MINUS = 3
VALUE_TIMES = 4
VALUE_LT = 5
VALUE_EQ = 6
# Indices 7..15 used for int literals via offset.

INT_LIT_OFFSET = 7
INT_LIT_MIN = -7
INT_LIT_MAX = 8     # inclusive bounds [-7, 8]
VALUE_CUTOFF = 16

# Reverse maps for decoder.
BIN_OP_FROM_VALUE = {
    VALUE_PLUS: "+",
    VALUE_MINUS: "-",
    VALUE_TIMES: "*",
    VALUE_LT: "<",
    VALUE_EQ: "==",
}
BIN_VALUE_FROM_OP = {v: k for k, v in BIN_OP_FROM_VALUE.items()}


# ---- species metadata -----------------------------------------------------

SPECIES_NAMES: tuple[str, ...] = ("kind", "type", "bid", "value")
SPECIES_DIMS: tuple[int, ...] = (KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF,
                                 VALUE_CUTOFF)
D_LOCAL: int = 1
for _d in SPECIES_DIMS:
    D_LOCAL *= _d


# FieldSpecies objects ready to drop into HamiltonianConfig (sub-projects
# B/C/E). bare_mass=0, kinetic=0 because the encoder does not require any
# dynamics on these registers — sub-projects B/C will set their own.
SPECIES: tuple[FieldSpecies, ...] = (
    FieldSpecies(name="kind",  cutoff=KIND_CUTOFF,  bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="type",  cutoff=TYPE_CUTOFF,  bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="bid",   cutoff=BID_CUTOFF,   bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="value", cutoff=VALUE_CUTOFF, bare_mass=0.0, kinetic=0.0),
)


# ---- binder handle --------------------------------------------------------


@dataclass(frozen=True)
class BinderHandle:
    """Uniquely identifies a binder for cross-bond bookkeeping.

    lam_site is the absolute site index of the binder's Lam node.
    depth_at_lam is the lexical depth at which it was introduced (0 = outermost).
    """
    lam_site: int
    depth_at_lam: int


# ---- encoding metadata ----------------------------------------------------


@dataclass
class EncodingMeta:
    """Side data produced by the encoder, consumed by decoder and downstream
    sub-projects B/C/D/E.
    """
    N: int
    chi_max: int
    field_dims: dict[str, int]
    species: list[FieldSpecies]
    nested_type_index: dict[int, "object"]   # site -> Ty (kept as object to
                                             # avoid circular import; encoder
                                             # writes proper Ty values)
    site_to_ast_path: dict[int, tuple[int, ...]]
    live_binders_per_bond: list[list[BinderHandle]]


# ---- exception hierarchy --------------------------------------------------


class EncodingError(Exception):
    """Base class for encoder/decoder errors."""


class EncodingTooLarge(EncodingError):
    def __init__(self, n_nodes: int, N: int):
        self.n_nodes, self.N_limit = n_nodes, N
        super().__init__(f"AST has {n_nodes} nodes but N={N}")


class TooManyBinders(EncodingError):
    def __init__(self, depth: int, cutoff: int):
        self.depth, self.cutoff = depth, cutoff
        super().__init__(
            f"scope nesting {depth} exceeds bid cutoff {cutoff}"
        )


class IntLiteralOutOfRange(EncodingError):
    def __init__(self, n: int):
        self.n = n
        super().__init__(
            f"IntLit({n}) outside [{INT_LIT_MIN}, {INT_LIT_MAX}]"
        )


class IllScopedVar(EncodingError):
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Var({name!r}) not in lexical scope")


class UnsupportedNode(EncodingError):
    def __init__(self, node_type: str):
        self.node_type = node_type
        super().__init__(f"node type {node_type} not in supported grammar")


class DecodeError(EncodingError):
    pass
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_encoding.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/encoding.py src/qft_pcn/tests/test_logic_encoding.py
git commit -m "$(cat <<'EOF'
feat(logic/encoding): field-species constants, EncodingMeta, exceptions

Per spec §4 and §9: kind (8) / type (8) / bid (8) / value (16) basis
constants, INT_LIT_OFFSET=7 mapping [-7, 8] to indices [7, 15], SPECIES
list ready for HamiltonianConfig, BinderHandle and EncodingMeta
dataclasses, six exception types under EncodingError.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

**End of Part 1.** Continue with Part 2 (encoder construction) and Part 3 (decoder + tests). See `2026-05-20-ast-mps-encoder-part2.md` and `2026-05-20-ast-mps-encoder-part3.md`.
