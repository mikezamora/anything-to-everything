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
