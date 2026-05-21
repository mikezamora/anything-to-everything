"""Top-level DSL pipeline: parse/validate/compile.

validate_dsl(spec) -> {"valid": bool, "errors": [<BridgeError-as-dict>]}
compile_dsl(spec)  -> CompiledDsl, ready for the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import parse_and_validate, validate_schema, _apply_defaults
from .cross_validate import cross_validate
from .expr_parser import parse_expr
from .compiler import compile_constraint
from .term import LocalTerm, TwoSiteTerm, FieldSpec
from .vocab import resolve_basis
from ..errors import (
    BridgeError, BadSchemaError, BadTermError,
)


@dataclass
class CompiledDsl:
    fields: list[FieldSpec]
    sites: int
    terms: list                              # list[LocalTerm | TwoSiteTerm]
    boundary: dict[int, dict[str, int]]
    observables: list[dict[str, Any]]
    search: dict[str, Any]


def _normalize_input(spec: Any) -> dict[str, Any]:
    if isinstance(spec, str):
        return parse_and_validate(spec)
    if not isinstance(spec, dict):
        raise BadSchemaError(
            message="DSL must be a JSON object or string",
            details={"got": type(spec).__name__},
        )
    validate_schema(spec)
    return _apply_defaults(spec)


def validate_dsl(spec: Any) -> dict[str, Any]:
    """Run phases 1-4 and return a structured report. Never raises."""
    errors: list[dict[str, Any]] = []
    try:
        dsl = _normalize_input(spec)
    except BridgeError as e:
        errors.append(e.to_dict())
        return {"valid": False, "errors": errors}
    try:
        cross_validate(dsl)
    except BridgeError as e:
        errors.append(e.to_dict())
        return {"valid": False, "errors": errors}
    for i, c in enumerate(dsl["constraints"]):
        try:
            parse_expr(c["term"])
        except BadTermError as e:
            e.details["constraint_index"] = i
            errors.append(e.to_dict())
    return {"valid": not errors, "errors": errors}


def compile_dsl(spec: Any) -> CompiledDsl:
    """Validate phases 1-3, parse + compile every constraint, resolve
    boundary basis names. Raises on any failure."""
    dsl = _normalize_input(spec)
    cross_validate(dsl)
    fields = [FieldSpec(name=f["name"], cutoff=f["cutoff"])
              for f in dsl["fields"]]
    terms: list = []
    for i, c in enumerate(dsl["constraints"]):
        try:
            expr = parse_expr(c["term"])
        except BadTermError as e:
            e.details["constraint_index"] = i
            raise
        weight = float(c.get("weight", 1.0))
        if c["kind"] == "local":
            ts = compile_constraint(expr, kind="local", site=c["site"],
                                    fields=fields, weight=weight)
        else:
            ts = compile_constraint(expr, kind="two_site",
                                    sites=tuple(c["sites"]),
                                    fields=fields, weight=weight)
        terms.extend(ts)
    boundary: dict[int, dict[str, int]] = {}
    for site_key, fmap in (dsl.get("boundary") or {}).items():
        site = int(site_key)
        boundary[site] = {fname: resolve_basis(fname, val)
                          for fname, val in fmap.items()}
    return CompiledDsl(
        fields=fields, sites=dsl["sites"], terms=terms, boundary=boundary,
        observables=list(dsl["observables"]), search=dict(dsl["search"]),
    )
