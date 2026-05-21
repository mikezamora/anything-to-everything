"""Compile constraint-expression ASTs into Hamiltonian terms.

Walks Expr trees (from bridge/dsl/expr_parser.py) and emits LocalTerm /
TwoSiteTerm objects via the stub factories in compiler_stub.py.

When sub-project B/C land, the relevant branches dispatch to their named
factory functions instead of the stubs.
"""

from __future__ import annotations

from .expr_ast import Expr, Lit, Ident, Call, Unary, Binary
from .term import LocalTerm, TwoSiteTerm, FieldSpec
from .vocab import resolve_basis
from .compiler_stub import (
    term_field_equals_constant,
    term_field_equality_two_site,
)
from ..errors import TermUnsupportedError, TypeError as BridgeTypeError


_FIELD_ACCESSORS = {"kind", "type", "bid", "value", "n", "phi", "pi"}


def _expect_arg(name: str, kind: str) -> None:
    if name in ("arg1", "self"):
        return
    if name == "arg2" and kind == "two_site":
        return
    if name == "arg2" and kind == "local":
        raise BridgeTypeError(
            message="local constraint cannot reference arg2",
            details={"identifier": "arg2", "kind": kind},
        )
    # any other identifier is fine here; it could be the site-bound default


def _resolve_lit_to_basis(field_name: str, lit: Expr,
                          fields: list[FieldSpec]) -> int:
    names = [f.name for f in fields]
    if field_name not in names:
        raise BridgeTypeError(
            message=f"unknown field {field_name!r}",
            details={"field": field_name, "known": names},
        )
    if isinstance(lit, Lit):
        return resolve_basis(field_name, lit.value)
    if isinstance(lit, Ident):
        return resolve_basis(field_name, lit.name)
    raise TermUnsupportedError(
        message="rhs of == must be a literal or named basis constant",
        details={"got": type(lit).__name__},
    )


def compile_constraint(expr: Expr, *, kind: str,
                       site: int | None = None,
                       sites: tuple[int, int] | None = None,
                       fields: list[FieldSpec],
                       weight: float) -> list:
    """Compile one constraint into a list of LocalTerm | TwoSiteTerm."""
    if kind == "local":
        return _compile_local(expr, site=site, fields=fields, weight=weight)
    if kind == "two_site":
        return _compile_two_site(expr, sites=sites, fields=fields,
                                 weight=weight)
    raise BridgeTypeError(
        message=f"unknown constraint kind {kind!r}",
        details={"kind": kind},
    )


def _compile_local(expr: Expr, *, site: int, fields: list[FieldSpec],
                   weight: float) -> list[LocalTerm]:
    if isinstance(expr, Lit) and expr.value is True:
        return []
    if isinstance(expr, Binary) and expr.op == "and":
        return (_compile_local(expr.lhs, site=site, fields=fields, weight=weight)
                + _compile_local(expr.rhs, site=site, fields=fields, weight=weight))
    if isinstance(expr, Binary) and expr.op == "==":
        field_name = _accessor_field_name(expr.lhs, kind="local",
                                          declared=[f.name for f in fields])
        basis = _resolve_lit_to_basis(field_name, expr.rhs, fields)
        return [term_field_equals_constant(fields, site, field_name, basis,
                                            weight)]
    raise TermUnsupportedError(
        message=f"unsupported local-constraint shape: {type(expr).__name__}",
        details={"expr": repr(expr)},
    )


def _compile_two_site(expr: Expr, *, sites: tuple[int, int],
                      fields: list[FieldSpec], weight: float):
    if isinstance(expr, Lit) and expr.value is True:
        return []
    if isinstance(expr, Binary) and expr.op == "and":
        return (_compile_two_site(expr.lhs, sites=sites, fields=fields,
                                  weight=weight)
                + _compile_two_site(expr.rhs, sites=sites, fields=fields,
                                    weight=weight))
    if isinstance(expr, Binary) and expr.op == "==":
        # accessor(arg1) == accessor(arg2): two-site equality on that field
        if (isinstance(expr.lhs, Call) and isinstance(expr.rhs, Call)
                and expr.lhs.fn == expr.rhs.fn
                and len(expr.lhs.args) == 1 and len(expr.rhs.args) == 1
                and isinstance(expr.lhs.args[0], Ident)
                and isinstance(expr.rhs.args[0], Ident)
                and {expr.lhs.args[0].name, expr.rhs.args[0].name}
                    == {"arg1", "arg2"}):
            field_name = _accessor_to_field(expr.lhs.fn)
            return [term_field_equality_two_site(fields, sites, field_name,
                                                  weight)]
        # accessor(argN) == LIT: attach as a local term at that site
        if isinstance(expr.lhs, Call) and len(expr.lhs.args) == 1 \
                and isinstance(expr.lhs.args[0], Ident):
            arg_ident = expr.lhs.args[0].name
            _expect_arg(arg_ident, kind="two_site")
            target_site = sites[0] if arg_ident in ("arg1", "self") else sites[1]
            field_name = _accessor_to_field(expr.lhs.fn)
            basis = _resolve_lit_to_basis(field_name, expr.rhs, fields)
            return [term_field_equals_constant(fields, target_site,
                                                field_name, basis, weight)]
    raise TermUnsupportedError(
        message=f"unsupported two_site-constraint shape: {type(expr).__name__}",
        details={"expr": repr(expr)},
    )


def _accessor_field_name(lhs: Expr, kind: str, declared: list[str]) -> str:
    """Recognize `<field>` (Ident), `<field>(self)`, `<field>(arg1)` and
    return the field name. The Ident form must reference a declared field."""
    if isinstance(lhs, Ident):
        if lhs.name not in declared:
            raise BridgeTypeError(
                message=f"unknown field {lhs.name!r}",
                details={"field": lhs.name, "known": declared},
            )
        return lhs.name
    if isinstance(lhs, Call) and len(lhs.args) == 1 \
            and isinstance(lhs.args[0], Ident):
        _expect_arg(lhs.args[0].name, kind=kind)
        return _accessor_to_field(lhs.fn)
    raise TermUnsupportedError(
        message="lhs of == must be a field name or field accessor call",
        details={"got": type(lhs).__name__},
    )


def _accessor_to_field(fn: str) -> str:
    if fn in _FIELD_ACCESSORS:
        return fn
    raise TermUnsupportedError(
        message=f"unsupported accessor {fn!r}",
        details={"fn": fn, "known": sorted(_FIELD_ACCESSORS)},
    )
