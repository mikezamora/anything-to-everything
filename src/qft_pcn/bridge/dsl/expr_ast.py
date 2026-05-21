"""AST for the constraint-expression DSL (spec §4.1, §4.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Any


@dataclass(frozen=True)
class Lit:
    value: Any           # int | bool | str


@dataclass(frozen=True)
class Ident:
    name: str


@dataclass(frozen=True)
class Call:
    fn: str
    args: tuple = field(default_factory=tuple)

    def __init__(self, fn, args):
        object.__setattr__(self, "fn", fn)
        object.__setattr__(self, "args", tuple(args))


@dataclass(frozen=True)
class Unary:
    op: str               # "not" | "-"
    operand: "Expr"


@dataclass(frozen=True)
class Binary:
    op: str
    lhs: "Expr"
    rhs: "Expr"


Expr = Union[Lit, Ident, Call, Unary, Binary]
