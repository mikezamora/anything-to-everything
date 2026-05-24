"""§2.7 — Pythonic-restricted predicate parser.

Compiles a predicate string into a validated AST whose evaluation is safe
to delegate to a small interpreter. The grammar is a strict subset of
Python expressions: see spec §2.7 for the closed allowlist of node types
and helper names.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass


HELPER_NAMES = frozenset({"type", "value", "binder", "kind", "int", "str", "bool", "len"})
BOUND_NAMES_LOCAL = frozenset({"expr"})
BOUND_NAMES_TWO_SITE = frozenset({"arg1", "arg2", "expr"})


ALLOWED_NODES: frozenset = frozenset({
    ast.Expression, ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult,
    ast.Name, ast.Constant, ast.Load,
    ast.Call, ast.Subscript, ast.Tuple, ast.List, ast.Index,
})


class PredicateRejected(ValueError):
    """Raised when a predicate uses banned syntax or names."""


@dataclass(frozen=True)
class ParsedPredicate:
    tree: ast.Expression
    free_names: frozenset[str]


def parse_predicate(term: str, *, two_site: bool = False,
                    user_field_names: frozenset[str] = frozenset()) -> ParsedPredicate:
    """Parse `term` as a §2.7-allowed predicate. Returns the AST + free names."""
    try:
        tree = ast.parse(term, mode="eval")
    except SyntaxError as exc:
        raise PredicateRejected(f"syntax error: {exc.msg}") from exc

    bound = (BOUND_NAMES_TWO_SITE if two_site else BOUND_NAMES_LOCAL) | user_field_names
    visitor = _Visitor(allowed_callees=HELPER_NAMES, bound_names=bound)
    visitor.visit(tree)
    return ParsedPredicate(tree=tree, free_names=frozenset(visitor.free_names))


class _Visitor(ast.NodeVisitor):
    def __init__(self, allowed_callees: frozenset[str], bound_names: frozenset[str]):
        self.allowed_callees = allowed_callees
        self.bound_names = bound_names
        self.free_names: set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        if type(node) not in ALLOWED_NODES:
            raise PredicateRejected(f"banned AST node: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise PredicateRejected(
                "calls must target a bare identifier from the helper allowlist"
            )
        if node.func.id not in self.allowed_callees:
            raise PredicateRejected(
                f"call to non-allowlisted helper '{node.func.id}'"
            )
        for arg in node.args:
            self.visit(arg)
        if node.keywords:
            raise PredicateRejected("keyword arguments are not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        raise PredicateRejected("attribute access is not allowed")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.allowed_callees or node.id in self.bound_names:
            return
        self.free_names.add(node.id)
