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
