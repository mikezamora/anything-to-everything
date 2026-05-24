"""JSON Schema for the bridge DSL plus phase-1/2 validation.

See spec §3 and §3.8 phases 1-2 in
docs/superpowers/specs/2026-05-21-llm-bridge-design.md.
"""

from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator

from ..errors import BadJsonError, BadSchemaError


_DEFAULT_SEARCH = {
    "method": "imag_time",
    "runtime": "mps",
    "steps": 50,
    "chi_max": 32,
    "dt": 0.05,
}


SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QPCN Bridge DSL",
    "type": "object",
    "required": ["version", "fields", "sites", "constraints", "observables"],
    "additionalProperties": False,
    "properties": {
        "version": {"const": "1"},
        "fields": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object",
                "required": ["name", "cutoff"],
                "additionalProperties": False,
                "properties": {
                    "name":   {"type": "string",
                               "pattern": "^[a-z][a-z0-9_]{0,31}$"},
                    "cutoff": {"type": "integer", "minimum": 2, "maximum": 32},
                },
            },
        },
        "sites": {"type": "integer", "minimum": 1, "maximum": 256},
        "constraints": {
            "type": "array",
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "required": ["kind", "site", "term"],
                        "additionalProperties": False,
                        "properties": {
                            "kind":   {"const": "local"},
                            "site":   {"type": "integer", "minimum": 0},
                            "term":   {"type": "string", "minLength": 1},
                            "weight": {"type": "number", "minimum": 0,
                                       "default": 1.0},
                        },
                    },
                    {
                        "type": "object",
                        "required": ["kind", "sites", "term"],
                        "additionalProperties": False,
                        "properties": {
                            "kind":  {"const": "two_site"},
                            "sites": {"type": "array",
                                      "items": {"type": "integer", "minimum": 0},
                                      "minItems": 2, "maxItems": 2,
                                      "uniqueItems": True},
                            "term":  {"type": "string", "minLength": 1},
                            "weight":{"type": "number", "minimum": 0,
                                      "default": 1.0},
                        },
                    },
                    {
                        "type": "object",
                        "required": ["kind", "root"],
                        "additionalProperties": False,
                        "properties": {
                            "kind":   {"const": "well_typed_subtree"},
                            "root":   {"type": "integer", "minimum": 0},
                            "weight": {"type": "number", "minimum": 0, "default": 1.0},
                        },
                    },
                    {
                        "type": "object",
                        "required": ["kind", "input", "output"],
                        "additionalProperties": False,
                        "properties": {
                            "kind":   {"const": "example"},
                            "input":  {"type": "string"},
                            "output": {"type": "string"},
                            "weight": {"type": "number", "minimum": 0, "default": 1.0},
                        },
                    },
                    {
                        "type": "object",
                        "required": ["kind", "primitives"],
                        "additionalProperties": False,
                        "properties": {
                            "kind":       {"const": "vocabulary"},
                            "primitives": {"type": "array", "items": {"type": "string"},
                                           "minItems": 1, "uniqueItems": True},
                            "weight":     {"type": "number", "minimum": 0, "default": 1.0},
                        },
                    },
                    {
                        "type": "object",
                        "required": ["kind", "lemma_id", "sites"],
                        "additionalProperties": False,
                        "properties": {
                            "kind":     {"const": "use_lemma"},
                            "lemma_id": {"type": "string", "minLength": 1},
                            "sites":    {"type": "array", "items": {"type": "integer", "minimum": 0},
                                         "minItems": 1, "uniqueItems": True},
                            "weight":   {"type": "number", "minimum": 0, "default": 1.0},
                        },
                    },
                ]
            },
        },
        "boundary": {
            "type": "object",
            "patternProperties": {
                "^[0-9]+$": {
                    "type": "object",
                    "minProperties": 1,
                    "additionalProperties": {
                        "type": ["string", "number", "integer", "boolean"]
                    },
                },
            },
            "additionalProperties": False,
            "default": {},
        },
        "observables": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object",
                "required": ["site", "field", "op"],
                "additionalProperties": False,
                "properties": {
                    "site":  {"type": "integer", "minimum": 0},
                    "field": {"type": "string"},
                    "op":    {"enum": ["n", "phi", "pi", "a", "adag",
                                       "identity", "argmax"]},
                },
            },
        },
        "search": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "method":  {"enum": ["imag_time"]},
                "runtime": {"enum": ["mps", "mera"], "default": "mps"},
                "steps":   {"type": "integer", "minimum": 1, "maximum": 1000,
                            "default": 50},
                "chi_max": {"type": "integer", "minimum": 2, "maximum": 64,
                            "default": 32},
                "dt":      {"type": "number", "exclusiveMinimum": 0,
                            "maximum": 0.5, "default": 0.05},
            },
            "default": _DEFAULT_SEARCH,
        },
    },
}


_VALIDATOR = Draft202012Validator(SCHEMA)


def validate_schema(dsl: dict[str, Any]) -> None:
    """Phase 2 of the validation pipeline: JSON Schema check.

    Raises BadSchemaError on any failure; on success returns None.
    """
    try:
        errs = sorted(_VALIDATOR.iter_errors(dsl), key=lambda e: list(e.path))
    except Exception as exc:
        raise BadSchemaError(message=f"schema validator internal: {exc}") from exc
    if errs:
        e = errs[0]
        pointer = "/" + "/".join(str(p) for p in e.absolute_path)
        raise BadSchemaError(
            message=f"{pointer or '/'}: {e.message}",
            details={"pointer": pointer or "/", "constraint": e.validator,
                     "value": e.instance if _json_safe(e.instance) else repr(e.instance)},
        )


def _json_safe(v: Any) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def _apply_defaults(dsl: dict[str, Any]) -> dict[str, Any]:
    """Fill in defaults for `boundary`, `search`, and `weight`. Non-mutating."""
    out = dict(dsl)
    out.setdefault("boundary", {})
    search = dict(out.get("search") or {})
    for k, v in _DEFAULT_SEARCH.items():
        search.setdefault(k, v)
    out["search"] = search
    out["constraints"] = [
        {**c, "weight": c.get("weight", 1.0)} for c in out.get("constraints", [])
    ]
    return out


def parse_and_validate(raw: str) -> dict[str, Any]:
    """Phase 1 (parse) + phase 2 (schema) + default-fill.

    Raises BadJsonError on parse failure, BadSchemaError on schema failure.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BadJsonError(
            message=f"JSON parse error at line {exc.lineno} col {exc.colno}: "
                    f"{exc.msg}",
            details={"line": exc.lineno, "col": exc.colno},
        ) from exc
    if not isinstance(data, dict):
        raise BadSchemaError(
            message="top-level DSL must be a JSON object",
            details={"pointer": "/", "got": type(data).__name__},
        )
    validate_schema(data)
    return _apply_defaults(data)
