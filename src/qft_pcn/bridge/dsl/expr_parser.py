"""Recursive-descent parser for the constraint-expression sub-DSL (spec §4.1).

Hand-written, never calls eval/exec/compile/literal_eval. Returns an Expr AST.
"""

from __future__ import annotations

import re

from .expr_ast import Lit, Ident, Call, Unary, Binary, Expr
from ..errors import BadTermError


_TOKEN_RE = re.compile(
    r"(?P<ws>\s+)"
    r"|(?P<le><=)"
    r"|(?P<ge>>=)"
    r"|(?P<eq>==)"
    r"|(?P<ne>!=)"
    r"|(?P<lt><)"
    r"|(?P<gt>>)"
    r"|(?P<lp>\()"
    r"|(?P<rp>\))"
    r"|(?P<comma>,)"
    r"|(?P<plus>\+)"
    r"|(?P<minus>-)"
    r"|(?P<times>\*)"
    r"|(?P<int>\d+)"
    r"|(?P<str>'[^']*')"
    r"|(?P<ident>[A-Za-z_][A-Za-z_0-9]*)"
)


_KEYWORDS = {"and", "or", "not", "true", "false"}


class _Tok:
    __slots__ = ("kind", "value", "pos")
    def __init__(self, kind: str, value: str, pos: int):
        self.kind, self.value, self.pos = kind, value, pos


def _tokenize(src: str) -> list[_Tok]:
    pos = 0
    out: list[_Tok] = []
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise BadTermError(
                message=f"unexpected character at position {pos}: {src[pos]!r}",
                details={"pos": pos, "char": src[pos]},
            )
        kind = m.lastgroup
        value = m.group(kind)
        end = m.end()
        if kind == "ws":
            pos = end
            continue
        if kind == "ident" and value in _KEYWORDS:
            kind = value
        out.append(_Tok(kind, value, pos))
        pos = end
    out.append(_Tok("eof", "", len(src)))
    return out


class _Parser:
    _CMP_KINDS = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=",
                  "gt": ">", "ge": ">="}

    def __init__(self, toks: list[_Tok], src: str):
        self.toks = toks
        self.i = 0
        self.src = src

    def peek(self) -> _Tok:
        return self.toks[self.i]

    def eat(self, kind: str) -> _Tok:
        t = self.peek()
        if t.kind != kind:
            raise BadTermError(
                message=f"unexpected token {t.kind}={t.value!r} at position "
                        f"{t.pos}; expected {kind}",
                details={"pos": t.pos, "expected": kind, "found": t.kind},
            )
        self.i += 1
        return t

    def parse_or(self) -> Expr:
        left = self.parse_and()
        while self.peek().kind == "or":
            self.i += 1
            right = self.parse_and()
            left = Binary(op="or", lhs=left, rhs=right)
        return left

    def parse_and(self) -> Expr:
        left = self.parse_not()
        while self.peek().kind == "and":
            self.i += 1
            right = self.parse_not()
            left = Binary(op="and", lhs=left, rhs=right)
        return left

    def parse_not(self) -> Expr:
        if self.peek().kind == "not":
            self.i += 1
            return Unary(op="not", operand=self.parse_not())
        return self.parse_cmp()

    def parse_cmp(self) -> Expr:
        left = self.parse_add()
        t = self.peek()
        if t.kind in self._CMP_KINDS:
            op = self._CMP_KINDS[t.kind]
            self.i += 1
            right = self.parse_add()
            if self.peek().kind in self._CMP_KINDS:
                bad = self.peek()
                raise BadTermError(
                    message=f"comparison is non-associative; unexpected "
                            f"{bad.value!r} at position {bad.pos}",
                    details={"pos": bad.pos, "found": bad.value},
                )
            return Binary(op=op, lhs=left, rhs=right)
        return left

    def parse_add(self) -> Expr:
        left = self.parse_mul()
        while self.peek().kind in ("plus", "minus"):
            op = "+" if self.peek().kind == "plus" else "-"
            self.i += 1
            right = self.parse_mul()
            left = Binary(op=op, lhs=left, rhs=right)
        return left

    def parse_mul(self) -> Expr:
        left = self.parse_unary()
        while self.peek().kind == "times":
            self.i += 1
            right = self.parse_unary()
            left = Binary(op="*", lhs=left, rhs=right)
        return left

    def parse_unary(self) -> Expr:
        if self.peek().kind == "minus":
            self.i += 1
            return Unary(op="-", operand=self.parse_unary())
        return self.parse_call()

    def parse_call(self) -> Expr:
        atom = self.parse_atom()
        if isinstance(atom, Ident) and self.peek().kind == "lp":
            self.i += 1
            args: list[Expr] = []
            if self.peek().kind != "rp":
                args.append(self.parse_or())
                while self.peek().kind == "comma":
                    self.i += 1
                    args.append(self.parse_or())
            self.eat("rp")
            return Call(fn=atom.name, args=args)
        return atom

    def parse_atom(self) -> Expr:
        t = self.peek()
        if t.kind == "int":
            self.i += 1
            return Lit(value=int(t.value))
        if t.kind == "str":
            self.i += 1
            return Lit(value=t.value[1:-1])
        if t.kind == "true":
            self.i += 1
            return Lit(value=True)
        if t.kind == "false":
            self.i += 1
            return Lit(value=False)
        if t.kind == "ident":
            self.i += 1
            return Ident(name=t.value)
        if t.kind == "lp":
            self.i += 1
            e = self.parse_or()
            self.eat("rp")
            return e
        raise BadTermError(
            message=f"unexpected token {t.kind}={t.value!r} at position {t.pos}",
            details={"pos": t.pos, "found": t.kind},
        )


def parse_expr(src: str) -> Expr:
    """Parse a constraint expression (spec §4.1) into an Expr AST."""
    if not src.strip():
        raise BadTermError(message="empty expression", details={"pos": 0})
    toks = _tokenize(src)
    p = _Parser(toks, src)
    expr = p.parse_or()
    if p.peek().kind != "eof":
        t = p.peek()
        raise BadTermError(
            message=f"unexpected trailing token {t.kind}={t.value!r} "
                    f"at position {t.pos}",
            details={"pos": t.pos, "found": t.kind},
        )
    return expr
