"""AST data classes for the STLC + ints + bools + if + arith/compare surface.

See docs/superpowers/specs/2026-05-20-ast-mps-encoder-design.md, §3.

The classes here are pure data; the parser and pretty-printer live in
ast.py too (added in later tasks). Encoder/decoder are separate modules.
"""

from __future__ import annotations

import re
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


# ---- tokenizer + parser ---------------------------------------------------


_TOKEN_RE = re.compile(
    r"\s+"
    r"|(?P<lambda>\\)"
    r"|(?P<arrow>->)"
    r"|(?P<lt><(?!=))"                   # < but not <=
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
        # Negative int literal: unary minus applied to a literal int only
        # (not general unary minus). Recognized only at expression atom
        # position with the next token being an `int`.
        if t.kind == "minus" and self.toks[self.i + 1].kind == "int":
            self.i += 1  # consume '-'
            int_tok = self.eat("int")
            return IntLit(val=-int(int_tok.value))
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


# ---- holes (for sub-project E) -------------------------------------------


@dataclass
class HoleVar(Node):
    """A variable-position hole with a candidate binder name list.

    candidates: binder names admissible at this position (empty = any in-scope).
    target_type: optional expected type for this hole (None = inferred).
    name: optional diagnostic label.

    Backward-compat: existing code constructs HoleVar(candidates=[...]) with
    a list of strings. Sub-project E adds the target_type / name fields.
    """
    candidates: list[str]
    target_type: "Ty | None" = None
    name: str = ""


@dataclass(frozen=True)
class TypeHole(Ty):
    """A type-position hole: this type is unknown but must come from one of
    the candidates. Encoded by sub-project E's encoder extension as an
    equal-amplitude superposition over the candidate tags on the `type`
    register at the hole site.

    Candidates must be flat: TInt, TBool, or single-level TArrow (no nested
    arrows). The encoder works in the flat-tag basis only.
    """
    candidates: tuple
    name: str = ""

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("TypeHole must have at least one candidate")
        for c in self.candidates:
            if isinstance(c, TArrow):
                if isinstance(c.src, TArrow) or isinstance(c.dst, TArrow):
                    raise ValueError(
                        f"TypeHole candidates may not include nested arrow "
                        f"types: {c!r}"
                    )
            elif not isinstance(c, (TInt, TBool)):
                raise ValueError(
                    f"TypeHole candidates must be Ty "
                    f"(TInt/TBool/flat TArrow); got {type(c).__name__}"
                )


def substitute_hole(ast: Node, hole: HoleVar, replacement: Node) -> Node:
    """Return a new AST with the given hole instance replaced.

    Identity-based: replaces only the specific HoleVar object passed in.
    """
    if ast is hole:
        return replacement
    if isinstance(ast, Var) or isinstance(ast, IntLit) \
            or isinstance(ast, BoolLit) or isinstance(ast, HoleVar):
        return ast
    if isinstance(ast, Lam):
        return Lam(
            param=ast.param, param_ty=ast.param_ty,
            body=substitute_hole(ast.body, hole, replacement),
        )
    if isinstance(ast, App):
        return App(
            fn=substitute_hole(ast.fn, hole, replacement),
            arg=substitute_hole(ast.arg, hole, replacement),
        )
    if isinstance(ast, If):
        return If(
            cond=substitute_hole(ast.cond, hole, replacement),
            then_b=substitute_hole(ast.then_b, hole, replacement),
            else_b=substitute_hole(ast.else_b, hole, replacement),
        )
    if isinstance(ast, Bin):
        return Bin(
            op=ast.op,
            lhs=substitute_hole(ast.lhs, hole, replacement),
            rhs=substitute_hole(ast.rhs, hole, replacement),
        )
    return ast


# ---- experimental: recursive fixed-point ----------------------------------
#
# Added by sub-project F (spec §7.4). Minimal node for the MERA bond-dim
# scaling acceptance test; full surface-syntax integration, typing rules,
# and evaluation rules are deferred to a future sub-project owning
# recursion.

EXPERIMENTAL_REC = True


@dataclass
class Rec(Node):
    """Fixed-point: rec f. body, where f is a self-reference inside body.

    EXPERIMENTAL — used by the MERA recursive-Fibonacci bond-dim scaling
    test (spec §10.5). Surface-syntax parsing, typing rules, and
    round-trippable encoder support are deferred.
    """
    name: str
    name_ty: Ty
    body: Node
